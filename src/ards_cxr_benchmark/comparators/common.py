from __future__ import annotations

import gzip
import hashlib
import io
import json
import math
import os
import subprocess
from collections.abc import Iterable, Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from ..config import ensure_dir, ensure_parent_dir
from ..modeling import TASK_LABEL_COLUMNS, assert_no_subject_overlap

SOURCE_DATASET = "mimic_cxr"
COMPARATOR_TARGETS = {"strict", "sensitive", "both"}
COMPARISON_ROLES = {
    "external_comparator",
    "legacy_teacher",
    "compatibility_mirror",
    "trained_silver_baseline",
    "silver_derived_control",
    "mismatched_target_exploratory",
}
CANONICAL_PREDICTION_COLUMNS = [
    "case_id",
    "subject_id",
    "study_id",
    "split",
    "source_dataset",
    "model_name",
    "model_family",
    "comparison_role",
    "intended_target",
    "model_source_repository",
    "model_source_commit",
    "model_artifact_version",
    "text_scope",
    "prediction_score",
    "prediction_label",
    "raw_predicted_class",
    "threshold",
    "run_id",
]
REFERENCE_COLUMNS = [
    "case_id",
    "subject_id",
    "study_id",
    "split",
    "source_dataset",
    "strict_bilateral_opacity_label",
    "sensitive_bilateral_opacity_label",
    "silver_label_source",
    "manual_review_priority",
    "qa_flags",
]
PACKET_TEXT_COLUMNS = ["report_text", "target_text_impression_findings"]
REPORT_TEXT_COLUMNS = [*PACKET_TEXT_COLUMNS, "target_text_impression_fallback"]
UNSAFE_EXACT_COLUMNS = {
    "report_text",
    "primary_target_text",
    "target_text_full_report",
    "target_text_impression_findings",
    "target_text_impression_fallback",
    "findings_text",
    "impression_text",
    "entity_text",
    "input_file",
    "report_path",
}
METRIC_COLUMNS = [
    "task",
    "model_name",
    "model_family",
    "comparison_role",
    "text_scope",
    "split",
    "stratum",
    "stratum_value",
    "n",
    "positives",
    "prevalence",
    "prediction_positive_rate",
    "threshold",
    "accuracy",
    "sensitivity",
    "specificity",
    "ppv",
    "npv",
    "f1",
    "brier",
    "roc_auc",
    "average_precision",
]


def normalize_mimic_id(series: pd.Series, *, column: str, table: str) -> pd.Series:
    if series.isna().any():
        raise ValueError(f"{table}.{column} contains missing values")
    clean = series.astype(str).str.strip()
    if (clean == "").any():
        raise ValueError(f"{table}.{column} contains blank values")
    numeric = pd.to_numeric(clean, errors="coerce")
    if numeric.isna().any() or (~np.isfinite(numeric.astype(float))).any():
        raise ValueError(f"{table}.{column} contains non-numeric values")
    values = numeric.astype(float)
    if ((values % 1) != 0).any():
        raise ValueError(f"{table}.{column} contains non-integer values")
    if (values <= 0).any():
        raise ValueError(f"{table}.{column} must contain positive MIMIC identifiers")
    return values.astype("int64")


def canonical_case_id(subject_id: int | str, study_id: int | str) -> str:
    subject = int(subject_id)
    study = int(study_id)
    if subject <= 0 or study <= 0:
        raise ValueError("MIMIC subject_id and study_id must be positive")
    return f"mimic_{subject}_{study}"


def canonical_case_ids(subject_id: pd.Series, study_id: pd.Series) -> pd.Series:
    return "mimic_" + subject_id.astype(str) + "_" + study_id.astype(str)


def build_comparator_source(
    reports: pd.DataFrame,
    model_extract: pd.DataFrame,
    *,
    expected_rows: int | None = None,
) -> pd.DataFrame:
    report_required = {"subject_id", "study_id", *REPORT_TEXT_COLUMNS}
    extract_required = {
        "subject_id",
        "study_id",
        "split",
        *TASK_LABEL_COLUMNS.values(),
        "silver_label_source",
        "manual_review_priority",
        "qa_flags",
    }
    _require_columns(reports, report_required, "reports")
    _require_columns(model_extract, extract_required, "model_extract")

    left = reports[list(report_required)].copy()
    right = model_extract[list(extract_required)].copy()
    for table_name, frame in (("reports", left), ("model_extract", right)):
        frame["subject_id"] = normalize_mimic_id(
            frame["subject_id"], column="subject_id", table=table_name
        )
        frame["study_id"] = normalize_mimic_id(
            frame["study_id"], column="study_id", table=table_name
        )
        duplicates = frame.duplicated(["subject_id", "study_id"], keep=False)
        if duplicates.any():
            raise ValueError(f"{table_name} has {int(duplicates.sum())} duplicate study rows")

    joined = left.merge(
        right,
        on=["subject_id", "study_id"],
        how="outer",
        validate="one_to_one",
        indicator=True,
    )
    counts = joined["_merge"].value_counts().to_dict()
    if counts.get("left_only", 0) or counts.get("right_only", 0):
        raise ValueError(
            "Comparator source inputs do not have identical study keys: "
            f"left_only={counts.get('left_only', 0)}, right_only={counts.get('right_only', 0)}"
        )
    joined = joined.drop(columns="_merge")
    if expected_rows is not None and len(joined) != expected_rows:
        raise ValueError(f"Expected {expected_rows:,} comparator rows, found {len(joined):,}")
    for column in ("report_text", "target_text_impression_fallback"):
        missing = joined[column].isna() | joined[column].astype(str).str.strip().eq("")
        if missing.any():
            raise ValueError(f"Comparator source has {int(missing.sum())} missing {column} rows")
    impression_findings_missing = joined["target_text_impression_findings"].isna() | (
        joined["target_text_impression_findings"].astype(str).str.strip().eq("")
    )
    joined["impression_findings_fallback_used"] = impression_findings_missing
    joined.loc[impression_findings_missing, "target_text_impression_findings"] = joined.loc[
        impression_findings_missing, "target_text_impression_fallback"
    ]
    joined = joined.drop(columns="target_text_impression_fallback")
    if not set(joined["split"].dropna().unique()).issubset({"train", "validation", "test"}):
        raise ValueError("Comparator source contains invalid split values")
    assert_no_subject_overlap(joined)
    joined["case_id"] = canonical_case_ids(joined["subject_id"], joined["study_id"])
    joined["source_dataset"] = SOURCE_DATASET
    joined["qa_flags_present"] = joined["qa_flags"].map(qa_flags_present)
    return joined.sort_values(["case_id"], kind="stable").reset_index(drop=True)


def write_comparator_input_packet(
    source: pd.DataFrame,
    *,
    packet_path: Path,
    manifest_path: Path,
    summary_path: Path,
) -> dict[str, Any]:
    _require_columns(
        source,
        {"case_id", "subject_id", "study_id", "split", *PACKET_TEXT_COLUMNS},
        "comparator_source",
    )
    if len({packet_path.resolve(), manifest_path.resolve(), summary_path.resolve()}) != 3:
        raise ValueError("Comparator packet, manifest, and summary paths must be distinct")
    ordered = source.sort_values("case_id", kind="stable").reset_index(drop=True)
    ensure_parent_dir(packet_path)
    ensure_parent_dir(manifest_path)
    ensure_parent_dir(summary_path)
    packet_temp = packet_path.with_name(f".{packet_path.name}.partial")
    manifest_temp = manifest_path.with_name(f".{manifest_path.name}.partial")
    summary_temp = summary_path.with_name(f".{summary_path.name}.partial")

    manifest_rows: list[dict[str, Any]] = []
    try:
        with packet_temp.open("wb") as raw:
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
                with io.TextIOWrapper(compressed, encoding="utf-8", newline="\n") as text_out:
                    for input_order, row in ordered.iterrows():
                        full_report = str(row["report_text"])
                        impression_findings = str(row["target_text_impression_findings"])
                        payload = {
                            "case_id": str(row["case_id"]),
                            "subject_id": int(row["subject_id"]),
                            "study_id": int(row["study_id"]),
                            "split": str(row["split"]),
                            "report_text": full_report,
                            "target_text_impression_findings": impression_findings,
                        }
                        text_out.write(
                            json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n"
                        )
                        manifest_rows.append(
                            {
                                "case_id": payload["case_id"],
                                "subject_id": payload["subject_id"],
                                "study_id": payload["study_id"],
                                "split": payload["split"],
                                "source_dataset": SOURCE_DATASET,
                                "input_order": int(input_order),
                                "full_report_chars": len(full_report),
                                "impression_findings_chars": len(impression_findings),
                                "impression_findings_fallback_used": bool(
                                    row.get("impression_findings_fallback_used", False)
                                ),
                                "full_report_sha256": text_sha256(full_report),
                                "impression_findings_sha256": text_sha256(impression_findings),
                            }
                        )
        manifest = pd.DataFrame(manifest_rows)
        assert_no_text_leakage(manifest)
        manifest.to_parquet(manifest_temp, index=False)
        summary = {
            "created_at_utc": datetime.now(UTC).isoformat(),
            "rows": int(len(ordered)),
            "unique_cases": int(ordered["case_id"].nunique()),
            "split_counts": {
                str(key): int(value) for key, value in ordered["split"].value_counts().items()
            },
            "packet_sha256": sha256_path(packet_temp),
            "packet_bytes": int(packet_temp.stat().st_size),
            "impression_findings_fallback_rows": int(
                manifest["impression_findings_fallback_used"].sum()
            ),
            "manifest_columns": manifest.columns.tolist(),
            "contains_report_text": True,
            "restricted": True,
        }
        summary_temp.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        os.replace(packet_temp, packet_path)
        os.replace(manifest_temp, manifest_path)
        os.replace(summary_temp, summary_path)
        return summary
    finally:
        for path in (packet_temp, manifest_temp, summary_temp):
            path.unlink(missing_ok=True)


def iter_comparator_packet(path: Path) -> Iterator[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Comparator packet line {line_number} is not an object")
            yield value


def normalize_clamp_predictions(
    predictions: pd.DataFrame,
    source: pd.DataFrame,
    *,
    run_id: str,
    source_commit: str,
) -> pd.DataFrame:
    _require_columns(
        predictions,
        {"subject_id", "study_id", "prediction_score", "prediction_label"},
        "clamp_predictions",
    )
    base = predictions.copy()
    base["subject_id"] = normalize_mimic_id(
        base["subject_id"], column="subject_id", table="clamp_predictions"
    )
    base["study_id"] = normalize_mimic_id(
        base["study_id"], column="study_id", table="clamp_predictions"
    )
    base = base.drop(columns=["case_id", "split", "source_dataset"], errors="ignore")
    metadata = source[["subject_id", "study_id", "case_id", "split", "source_dataset"]]
    merged = base.merge(metadata, on=["subject_id", "study_id"], how="inner", validate="one_to_one")
    if len(merged) != len(source) or len(base) != len(source):
        raise ValueError(
            "CLAMP predictions must match the complete comparator source: "
            f"predictions={len(base)}, joined={len(merged)}, source={len(source)}"
        )
    merged["model_name"] = "clamp_legacy"
    merged["model_family"] = "rule_based_nlp"
    merged["comparison_role"] = "legacy_teacher"
    merged["intended_target"] = "both"
    merged["model_source_repository"] = "reblocke/ards-nlp:legacy-ards-phenotype-spec-v1"
    merged["model_source_commit"] = source_commit
    merged["model_artifact_version"] = "legacy_clamp_project"
    merged["text_scope"] = "full_report"
    merged["raw_predicted_class"] = merged["prediction_label"].astype("Int64").astype(str)
    merged["threshold"] = 0.5
    merged["run_id"] = run_id
    return validate_canonical_predictions(merged)


def normalize_clamp_python_predictions(
    predictions: pd.DataFrame,
    source: pd.DataFrame,
    *,
    run_id: str,
    source_commit: str,
) -> pd.DataFrame:
    """Normalize the exact Python CLAMP mirror without retaining source-text hashes."""

    _require_columns(
        predictions,
        {
            "clamp_doc_id",
            "prediction_status",
            "prediction_label",
            "clamp_ards_entity_count",
        },
        "clamp_python_predictions",
    )
    _require_columns(
        source,
        {"case_id", "subject_id", "study_id", "split", "source_dataset"},
        "comparator_reference",
    )
    base = predictions[
        [
            "clamp_doc_id",
            "prediction_status",
            "prediction_label",
            "clamp_ards_entity_count",
        ]
    ].copy()
    base["clamp_doc_id"] = _required_strings(
        base["clamp_doc_id"], "clamp_python_predictions.clamp_doc_id"
    )
    study_text = base["clamp_doc_id"].str.extract(r"^s([1-9]\d*)$", expand=False)
    if study_text.isna().any():
        examples = base.loc[study_text.isna(), "clamp_doc_id"].head(5).tolist()
        raise ValueError(
            f"Python CLAMP clamp_doc_id must use s{{positive_study_id}}; examples={examples}"
        )
    base["study_id"] = normalize_mimic_id(
        study_text, column="study_id", table="clamp_python_predictions"
    )
    if base["study_id"].duplicated(keep=False).any():
        raise ValueError("Python CLAMP predictions contain duplicate study IDs")
    base["prediction_status"] = _required_strings(
        base["prediction_status"], "clamp_python_predictions.prediction_status"
    ).str.casefold()
    if set(base["prediction_status"].unique()) != {"evaluable"}:
        raise ValueError("Python CLAMP compatibility predictions must all be evaluable")
    base["prediction_label"] = pd.to_numeric(base["prediction_label"], errors="coerce")
    if base["prediction_label"].isna().any() or not set(base["prediction_label"].unique()).issubset(
        {0, 1}
    ):
        raise ValueError("Python CLAMP prediction_label must contain only 0 and 1")
    base["prediction_label"] = base["prediction_label"].astype(int)
    counts = pd.to_numeric(base["clamp_ards_entity_count"], errors="coerce")
    if counts.isna().any() or (~np.isfinite(counts.astype(float))).any():
        raise ValueError("Python CLAMP entity counts must be finite integers")
    if ((counts.astype(float) % 1) != 0).any() or (counts < 0).any():
        raise ValueError("Python CLAMP entity counts must be nonnegative integers")
    base["clamp_ards_entity_count"] = counts.astype("int64")
    expected_label = base["clamp_ards_entity_count"].gt(0).astype(int)
    if not base["prediction_label"].equals(expected_label):
        raise ValueError("Python CLAMP labels do not agree with entity counts")

    metadata = source[["case_id", "subject_id", "study_id", "split", "source_dataset"]].copy()
    metadata["subject_id"] = normalize_mimic_id(
        metadata["subject_id"], column="subject_id", table="comparator_reference"
    )
    metadata["study_id"] = normalize_mimic_id(
        metadata["study_id"], column="study_id", table="comparator_reference"
    )
    metadata["case_id"] = _required_strings(metadata["case_id"], "comparator_reference.case_id")
    expected_case_ids = canonical_case_ids(metadata["subject_id"], metadata["study_id"])
    if not metadata["case_id"].equals(expected_case_ids):
        raise ValueError("Comparator reference case_id does not match MIMIC identifiers")
    if metadata["study_id"].duplicated(keep=False).any():
        raise ValueError("Comparator reference contains duplicate study IDs")

    merged = base.merge(metadata, on="study_id", how="outer", validate="one_to_one", indicator=True)
    merge_counts = merged["_merge"].value_counts().to_dict()
    if merge_counts.get("left_only", 0) or merge_counts.get("right_only", 0):
        raise ValueError(
            "Python CLAMP predictions must match the complete comparator reference: "
            f"prediction_only={merge_counts.get('left_only', 0)}, "
            f"reference_only={merge_counts.get('right_only', 0)}"
        )
    merged = merged.drop(columns=["_merge", "clamp_doc_id", "prediction_status"])
    merged["prediction_score"] = merged["prediction_label"].astype(float)
    merged["model_name"] = "clamp_python_compatibility"
    merged["model_family"] = "rule_based_nlp"
    merged["comparison_role"] = "compatibility_mirror"
    merged["intended_target"] = "both"
    merged["model_source_repository"] = "reblocke/ards-nlp"
    merged["model_source_commit"] = source_commit
    merged["model_artifact_version"] = "python_clamp_exact_parity_v1"
    merged["text_scope"] = "full_report"
    merged["raw_predicted_class"] = merged["prediction_label"].astype(str)
    merged["threshold"] = 0.5
    merged["run_id"] = run_id
    return validate_canonical_predictions(merged)


def normalize_silver_baseline_predictions(
    predictions: pd.DataFrame,
    source: pd.DataFrame,
    *,
    run_id: str,
    source_commit: str,
) -> pd.DataFrame:
    _require_columns(
        predictions,
        {
            "subject_id",
            "study_id",
            "split",
            "task",
            "model",
            "prediction_score",
            "prediction_label",
        },
        "silver_baselines",
    )
    base = predictions.copy()
    base["subject_id"] = normalize_mimic_id(
        base["subject_id"], column="subject_id", table="silver_baselines"
    )
    base["study_id"] = normalize_mimic_id(
        base["study_id"], column="study_id", table="silver_baselines"
    )
    base["split"] = _required_strings(base["split"], "silver_baselines.split")
    metadata = source[["subject_id", "study_id", "case_id", "split", "source_dataset"]].rename(
        columns={"split": "reference_split"}
    )
    base = base.merge(
        metadata,
        on=["subject_id", "study_id"],
        how="left",
        validate="many_to_one",
        indicator=True,
    )
    unmatched = base["_merge"] != "both"
    if unmatched.any():
        raise ValueError(
            f"Silver baseline predictions contain {int(unmatched.sum())} rows without a "
            "comparator reference"
        )
    split_mismatch = base["split"] != base["reference_split"]
    if split_mismatch.any():
        raise ValueError(
            f"Silver baseline predictions contain {int(split_mismatch.sum())} split mismatches"
        )
    base = base.drop(columns=["reference_split", "_merge"])
    base["model_name"] = "silver_" + base["task"].astype(str) + "_" + base["model"].astype(str)
    base["model_family"] = base["model"].map(
        lambda value: (
            "tfidf_logistic_regression" if value == "tfidf_logreg" else "silver_derived_control"
        )
    )
    base["comparison_role"] = base["model"].map(
        lambda value: (
            "trained_silver_baseline" if value == "tfidf_logreg" else "silver_derived_control"
        )
    )
    base["intended_target"] = base["task"].astype(str)
    base["model_source_repository"] = "reblocke/ards-nlp"
    base["model_source_commit"] = source_commit
    base["model_artifact_version"] = "silver_baseline_v1"
    base["text_scope"] = "full_report"
    base["raw_predicted_class"] = base["prediction_label"].astype("Int64").astype(str)
    base["threshold"] = 0.5
    base["run_id"] = run_id
    return validate_canonical_predictions(base)


def validate_canonical_predictions(df: pd.DataFrame) -> pd.DataFrame:
    assert_no_text_leakage(df)
    _require_columns(df, set(CANONICAL_PREDICTION_COLUMNS), "canonical_predictions")
    out = df.copy()
    out["subject_id"] = normalize_mimic_id(
        out["subject_id"], column="subject_id", table="canonical_predictions"
    )
    out["study_id"] = normalize_mimic_id(
        out["study_id"], column="study_id", table="canonical_predictions"
    )
    expected_case_id = canonical_case_ids(out["subject_id"], out["study_id"])
    out["case_id"] = _required_strings(out["case_id"], "case_id")
    if not out["case_id"].equals(expected_case_id):
        raise ValueError("case_id does not match canonical MIMIC subject/study identifiers")
    for column in [
        "split",
        "source_dataset",
        "model_name",
        "model_family",
        "comparison_role",
        "intended_target",
        "model_source_repository",
        "model_source_commit",
        "model_artifact_version",
        "text_scope",
        "run_id",
    ]:
        out[column] = _required_strings(out[column], column)
    if not set(out["split"].unique()).issubset({"train", "validation", "test"}):
        raise ValueError("Canonical predictions contain invalid split values")
    if not set(out["comparison_role"].unique()).issubset(COMPARISON_ROLES):
        raise ValueError("Canonical predictions contain invalid comparison_role values")
    if not set(out["intended_target"].unique()).issubset(COMPARATOR_TARGETS):
        raise ValueError("Canonical predictions contain invalid intended_target values")
    out["prediction_score"] = pd.to_numeric(out["prediction_score"], errors="coerce")
    if out["prediction_score"].isna().any() or (~out["prediction_score"].between(0, 1)).any():
        raise ValueError("prediction_score must be non-null and in [0, 1]")
    out["prediction_label"] = pd.to_numeric(out["prediction_label"], errors="coerce")
    if out["prediction_label"].isna().any() or not set(out["prediction_label"].unique()).issubset(
        {0, 1}
    ):
        raise ValueError("prediction_label must contain only 0 and 1")
    out["prediction_label"] = out["prediction_label"].astype(int)
    out["threshold"] = pd.to_numeric(out["threshold"], errors="coerce")
    invalid_threshold = out["threshold"].notna() & ~out["threshold"].between(0, 1)
    if invalid_threshold.any():
        raise ValueError("threshold must be missing or in [0, 1]")
    duplicates = out.duplicated(["case_id", "model_name"], keep=False)
    if duplicates.any():
        raise ValueError(
            f"Canonical predictions contain {int(duplicates.sum())} duplicate case/model rows"
        )
    ordered = [*CANONICAL_PREDICTION_COLUMNS]
    ordered.extend(column for column in out.columns if column not in ordered)
    return out[ordered].sort_values(["model_name", "case_id"], kind="stable").reset_index(drop=True)


def benchmark_comparator_predictions(
    predictions: pd.DataFrame,
    reference: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pred = validate_canonical_predictions(predictions)
    reference_owned_columns = {
        *TASK_LABEL_COLUMNS.values(),
        "silver_label_source",
        "manual_review_priority",
        "qa_flags",
        "qa_flags_present",
    }
    pred = pred.drop(
        columns=[column for column in reference_owned_columns if column in pred.columns]
    )
    _require_columns(reference, set(REFERENCE_COLUMNS), "reference")
    ref = reference[REFERENCE_COLUMNS].copy()
    ref["subject_id"] = normalize_mimic_id(
        ref["subject_id"], column="subject_id", table="reference"
    )
    ref["study_id"] = normalize_mimic_id(ref["study_id"], column="study_id", table="reference")
    ref["case_id"] = _required_strings(ref["case_id"], "reference.case_id")
    expected_case_id = canonical_case_ids(ref["subject_id"], ref["study_id"])
    if not np.array_equal(ref["case_id"].to_numpy(), expected_case_id.to_numpy()):
        raise ValueError("Reference case_id does not match canonical MIMIC identifiers")
    ref["split"] = _required_strings(ref["split"], "reference.split")
    if not set(ref["split"].unique()).issubset({"train", "validation", "test"}):
        raise ValueError("Reference contains invalid split values")
    ref["source_dataset"] = _required_strings(ref["source_dataset"], "reference.source_dataset")
    if ref.duplicated("case_id", keep=False).any():
        raise ValueError("Reference contains duplicate case_id rows")
    ref["qa_flags_present"] = ref["qa_flags"].map(qa_flags_present)
    joined = pred.merge(
        ref,
        on=["case_id", "subject_id", "study_id"],
        how="left",
        validate="many_to_one",
        suffixes=("", "_reference"),
        indicator=True,
    )
    unmatched = joined["_merge"] != "both"
    if unmatched.any():
        raise ValueError(
            f"{int(unmatched.sum())} comparator prediction rows did not join the MIMIC reference"
        )
    joined = joined.drop(columns="_merge")
    split_mismatch = joined["split"] != joined["split_reference"]
    if split_mismatch.any():
        raise ValueError(f"{int(split_mismatch.sum())} prediction rows have split mismatches")
    source_mismatch = joined["source_dataset"] != joined["source_dataset_reference"]
    if source_mismatch.any():
        raise ValueError(
            f"{int(source_mismatch.sum())} prediction rows have source_dataset mismatches"
        )
    joined = joined[joined["split"].isin(["validation", "test"])].copy()

    overall_rows: list[dict[str, Any]] = []
    strata_rows: list[dict[str, Any]] = []
    for _model_name, model_df in joined.groupby("model_name", sort=True, observed=True):
        intended_target = model_df["intended_target"].iloc[0]
        for task, label_column in TASK_LABEL_COLUMNS.items():
            if intended_target not in {"both", task}:
                continue
            for split, split_df in model_df.groupby("split", sort=True, observed=True):
                eligible = split_df[split_df[label_column].notna()].copy()
                if eligible.empty:
                    continue
                overall_rows.append(
                    _metric_row(
                        eligible,
                        task=task,
                        label_column=label_column,
                        split=str(split),
                        stratum="overall",
                        stratum_value="all",
                    )
                )
                for column in [
                    "silver_label_source",
                    "manual_review_priority",
                    "qa_flags_present",
                ]:
                    for value, group in eligible.groupby(column, sort=True, dropna=False):
                        strata_rows.append(
                            _metric_row(
                                group,
                                task=task,
                                label_column=label_column,
                                split=str(split),
                                stratum=column,
                                stratum_value=value,
                            )
                        )
    if not overall_rows:
        raise ValueError("No comparator metric rows could be produced")
    metrics = pd.DataFrame(overall_rows).reindex(columns=METRIC_COLUMNS)
    strata = pd.DataFrame(strata_rows).reindex(columns=METRIC_COLUMNS)
    coverage = make_prediction_catalog(pred, reference_cases=set(ref["case_id"]))
    return metrics, strata, coverage


def make_prediction_catalog(
    predictions: pd.DataFrame,
    *,
    reference_cases: set[str] | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model_name, group in predictions.groupby("model_name", sort=True, observed=True):
        case_ids = set(group["case_id"])
        rows.append(
            {
                "model_name": model_name,
                "model_family": group["model_family"].iloc[0],
                "comparison_role": group["comparison_role"].iloc[0],
                "intended_target": group["intended_target"].iloc[0],
                "text_scope": group["text_scope"].iloc[0],
                "prediction_rows": int(len(group)),
                "unique_cases": int(len(case_ids)),
                "validation_rows": int((group["split"] == "validation").sum()),
                "test_rows": int((group["split"] == "test").sum()),
                "mean_prediction_score": float(group["prediction_score"].mean()),
                "prediction_positive_rate": float(group["prediction_label"].mean()),
                "cases_without_reference": (
                    0 if reference_cases is None else int(len(case_ids - reference_cases))
                ),
            }
        )
    return pd.DataFrame(rows)


def write_combined_benchmark_outputs(
    *,
    metrics: pd.DataFrame,
    strata: pd.DataFrame,
    catalog: pd.DataFrame,
    statuses: list[dict[str, Any]],
    out_dir: Path,
) -> None:
    ensure_dir(out_dir)
    exploratory_role = "mismatched_target_exploratory"
    compatibility_role = "compatibility_mirror"
    validate_compatibility_mirror_metrics(metrics)
    validate_compatibility_mirror_metrics(strata)
    primary_metrics = metrics.loc[
        ~metrics["comparison_role"].isin({exploratory_role, compatibility_role})
    ].copy()
    exploratory_metrics = metrics.loc[metrics["comparison_role"] == exploratory_role].copy()
    compatibility_metrics = metrics.loc[metrics["comparison_role"] == compatibility_role].copy()
    primary_strata = strata.loc[
        ~strata["comparison_role"].isin({exploratory_role, compatibility_role})
    ].copy()
    exploratory_strata = strata.loc[strata["comparison_role"] == exploratory_role].copy()
    compatibility_strata = strata.loc[strata["comparison_role"] == compatibility_role].copy()
    primary_metrics.to_csv(out_dir / "silver_metrics.csv", index=False)
    primary_strata.to_csv(out_dir / "silver_strata.csv", index=False)
    compatibility_metrics.to_csv(out_dir / "compatibility_mirror_metrics.csv", index=False)
    compatibility_strata.to_csv(out_dir / "compatibility_mirror_strata.csv", index=False)
    exploratory_metrics.to_csv(out_dir / "mismatched_target_exploratory_metrics.csv", index=False)
    exploratory_strata.to_csv(out_dir / "mismatched_target_exploratory_strata.csv", index=False)
    catalog.to_csv(out_dir / "prediction_catalog.csv", index=False)
    (out_dir / "comparator_status.json").write_text(
        json.dumps(statuses, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# MIMIC-CXR Comparator Bakeoff",
        "",
        (
            "These are comparisons with automated silver labels for engineering diagnostics. "
            "They are not clinical accuracy estimates."
        ),
        "",
        "## Comparator Status",
        "",
    ]
    for status in statuses:
        lines.append(f"- {status.get('name', 'unknown')}: {status.get('status', 'unknown')}")
    lines.extend(["", "## Available Models", ""])
    for _, row in catalog.iterrows():
        lines.append(
            f"- {row['model_name']}: {int(row['prediction_rows']):,} predictions; "
            f"role={row['comparison_role']}"
        )
    lines.extend(["", "## Holdout Metrics", ""])
    for _, row in primary_metrics.iterrows():
        lines.append(
            f"- {row['model_name']} / {row['task']} / {row['split']}: "
            f"n={int(row['n']):,}, AUROC={_format_metric(row['roc_auc'])}, "
            f"F1={_format_metric(row['f1'])}"
        )
    if not compatibility_metrics.empty:
        lines.extend(["", "## Compatibility Mirrors", ""])
        lines.append(
            "These rows verify software equivalence with legacy CLAMP and are not independent "
            "validation evidence."
        )
        for _, row in compatibility_metrics.iterrows():
            lines.append(
                f"- {row['model_name']} / {row['task']} / {row['split']}: "
                f"n={int(row['n']):,}, AUROC={_format_metric(row['roc_auc'])}, "
                f"F1={_format_metric(row['f1'])}"
            )
    if not exploratory_metrics.empty:
        lines.extend(["", "## Mismatched-Target Exploratory Metrics", ""])
        for _, row in exploratory_metrics.iterrows():
            lines.append(
                f"- {row['model_name']} / {row['task']} / {row['split']}: "
                f"n={int(row['n']):,}, AUROC={_format_metric(row['roc_auc'])}, "
                f"F1={_format_metric(row['f1'])}"
            )
    (out_dir / "comparison_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def validate_compatibility_mirror_metrics(frame: pd.DataFrame) -> None:
    """Require the Python compatibility row to exactly reproduce legacy CLAMP metrics."""

    mirror = frame.loc[frame["comparison_role"] == "compatibility_mirror"].copy()
    if mirror.empty:
        return
    if set(mirror["model_name"].unique()) != {"clamp_python_compatibility"}:
        raise ValueError("Unexpected compatibility-mirror model set")
    legacy = frame.loc[frame["model_name"] == "clamp_legacy"].copy()
    if legacy.empty:
        raise ValueError("Python CLAMP compatibility metrics require legacy CLAMP metrics")
    comparison_columns = [
        column for column in frame.columns if column not in {"model_name", "comparison_role"}
    ]
    sort_columns = [
        column
        for column in ("task", "split", "stratum", "stratum_value")
        if column in comparison_columns
    ]
    left = (
        legacy[comparison_columns].sort_values(sort_columns, kind="stable").reset_index(drop=True)
    )
    right = (
        mirror[comparison_columns].sort_values(sort_columns, kind="stable").reset_index(drop=True)
    )
    try:
        pd.testing.assert_frame_equal(left, right, check_exact=True)
    except AssertionError as exc:
        raise ValueError("Python CLAMP compatibility metrics differ from legacy CLAMP") from exc


def write_comparator_status(
    *,
    name: str,
    status: str,
    reason: str,
    out_path: Path,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    ensure_parent_dir(out_path)
    payload = {
        "name": name,
        "status": status,
        "reason": reason,
        "checked_at_utc": datetime.now(UTC).isoformat(),
        "details": details or {},
    }
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def load_status_files(paths: Iterable[Path]) -> list[dict[str, Any]]:
    statuses: list[dict[str, Any]] = []
    for path in paths:
        if not path.exists():
            continue
        value = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(value, dict):
            statuses.append(value)
    return statuses


def assert_no_text_leakage(df: pd.DataFrame) -> None:
    unsafe: list[str] = []
    for column in df.columns:
        lowered = str(column).lower()
        if lowered in UNSAFE_EXACT_COLUMNS:
            unsafe.append(str(column))
        elif lowered != "text_scope" and lowered.endswith("_text"):
            unsafe.append(str(column))
        elif lowered.endswith("_path") or lowered.endswith("_file"):
            unsafe.append(str(column))
    if unsafe:
        raise ValueError(f"Output contains text- or path-bearing columns: {sorted(unsafe)}")


def qa_flags_present(value: object) -> bool:
    if value is None:
        return False
    if isinstance(value, np.ndarray):
        return any(qa_flags_present(item) for item in value.flat)
    if isinstance(value, (list, tuple, set, frozenset)):
        return any(qa_flags_present(item) for item in value)
    try:
        if pd.isna(value):
            return False
    except (TypeError, ValueError):
        pass
    return str(value).strip().lower() not in {"", "none", "[]", "nan", "<na>"}


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def repository_state(root: Path) -> dict[str, Any]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    return {"commit": commit, "dirty": dirty}


def _metric_row(
    df: pd.DataFrame,
    *,
    task: str,
    label_column: str,
    split: str,
    stratum: str,
    stratum_value: object,
) -> dict[str, Any]:
    y_true = df[label_column].astype(int)
    y_score = df["prediction_score"].astype(float)
    y_pred = df["prediction_label"].astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if tn + fp else None
    npv = tn / (tn + fn) if tn + fn else None
    has_two_classes = y_true.nunique() == 2
    threshold_values = df["threshold"].dropna().unique()
    threshold = float(threshold_values[0]) if len(threshold_values) == 1 else None
    return {
        "task": task,
        "model_name": df["model_name"].iloc[0],
        "model_family": df["model_family"].iloc[0],
        "comparison_role": df["comparison_role"].iloc[0],
        "text_scope": df["text_scope"].iloc[0],
        "split": split,
        "stratum": stratum,
        "stratum_value": str(stratum_value),
        "n": int(len(df)),
        "positives": int(y_true.sum()),
        "prevalence": float(y_true.mean()),
        "prediction_positive_rate": float(y_pred.mean()),
        "threshold": threshold,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "sensitivity": float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity": _safe_float(specificity),
        "ppv": float(precision_score(y_true, y_pred, zero_division=0)),
        "npv": _safe_float(npv),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "brier": float(brier_score_loss(y_true, y_score)),
        "roc_auc": float(roc_auc_score(y_true, y_score)) if has_two_classes else None,
        "average_precision": (
            float(average_precision_score(y_true, y_score)) if has_two_classes else None
        ),
    }


def _required_strings(series: pd.Series, column: str) -> pd.Series:
    if series.isna().any():
        raise ValueError(f"{column} contains missing values")
    clean = series.astype(str).str.strip()
    if (clean == "").any():
        raise ValueError(f"{column} contains blank values")
    return clean


def _require_columns(df: pd.DataFrame, required: set[str], table: str) -> None:
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{table} is missing required columns: {missing}")


def _safe_float(value: object) -> float | None:
    if value is None:
        return None
    result = float(value)
    return None if math.isnan(result) else result


def _format_metric(value: object) -> str:
    if value is None or pd.isna(value):
        return "NA"
    return f"{float(value):.3f}"
