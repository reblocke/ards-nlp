from __future__ import annotations

import ast
import json
from datetime import UTC, datetime
from math import isclose
from pathlib import Path
from typing import Any

import pandas as pd

from ..config import ensure_dir, ensure_parent_dir
from .common import (
    assert_no_text_leakage,
    repository_state,
    sha256_path,
    validate_canonical_predictions,
    write_comparator_status,
)
from .config import AmaralComparatorConfig
from .external import verify_external_repository

AMARAL_PRIMARY_MODEL = "amaral_xgboost_bilateral_infiltrates"
AMARAL_SENSITIVITY_MODEL = "amaral_xgboost_bilateral_infiltrates_raw_text_direct"
PREPROCESSING_KEYS = {
    "section_order_mimic_iii",
    "exclusion_set",
    "targeted_stemming",
    "complex_stopwords",
    "simple_stopwords",
    "useless_statements",
    "dictation",
}


def verify_amaral_resources(config: AmaralComparatorConfig) -> dict[str, Any]:
    source = verify_external_repository(config.source)
    resources = {
        "model": config.model_path,
        "vectorizer": config.vectorizer_path,
        "segmentation_notebook": config.segmentation_notebook,
        "segmentation_module": config.segmentation_module,
        "tokenizer_module": config.tokenizer_module,
        "license": config.license_path,
    }
    observed: dict[str, dict[str, Any]] = {}
    for name, path in resources.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing Amaral resource {name}: {path}")
        digest = sha256_path(path)
        expected = config.expected_sha256.get(name)
        if not expected:
            raise ValueError(f"No expected SHA-256 configured for Amaral resource {name}")
        if digest != expected:
            raise ValueError(
                f"Amaral resource checksum mismatch for {name}: expected {expected}, found {digest}"
            )
        observed[name] = {"sha256": digest, "bytes": int(path.stat().st_size)}
    return {
        "name": config.source.name,
        "repository": source,
        "resources": observed,
        "pickle_loading_boundary": "isolated_subprocess_only",
    }


def extract_published_preprocessing_config(notebook_path: Path) -> dict[str, Any]:
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    values: dict[str, Any] = {}
    section_orders: dict[str, Any] = {}
    for cell in notebook.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        source = cell.get("source", [])
        code = "".join(source) if isinstance(source, list) else str(source)
        try:
            tree = ast.parse(code)
        except SyntaxError:
            continue
        for node in tree.body:
            if not isinstance(node, ast.Assign) or len(node.targets) != 1:
                continue
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id in {
                "exclusion_set",
                "targeted_stemming",
                "complex_stopwords",
                "simple_stopwords",
                "useless_statements",
                "dictation",
            }:
                values[target.id] = ast.literal_eval(node.value)
            elif (
                isinstance(target, ast.Subscript)
                and isinstance(target.value, ast.Name)
                and target.value.id == "section_order"
            ):
                key = ast.literal_eval(target.slice)
                section_orders[str(key)] = ast.literal_eval(node.value)
    if "mimic_iii" in section_orders:
        values["section_order_mimic_iii"] = section_orders["mimic_iii"]
    missing = sorted(PREPROCESSING_KEYS - set(values))
    if missing:
        raise ValueError(f"Could not extract Amaral preprocessing values: {missing}")
    serializable = {
        key: sorted(value) if isinstance(value, set) else value for key, value in values.items()
    }
    serializable["source_notebook_sha256"] = sha256_path(notebook_path)
    serializable["preprocessing_contract"] = "published_mimic_preprocessing"
    return serializable


def write_published_preprocessing_config(config: AmaralComparatorConfig) -> dict[str, Any]:
    payload = extract_published_preprocessing_config(config.segmentation_notebook)
    ensure_parent_dir(config.preprocessing_config)
    config.preprocessing_config.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return payload


def build_amaral_predictions(
    *,
    runner_outputs: list[tuple[str, Path]],
    manifest: pd.DataFrame,
    config: AmaralComparatorConfig,
    run_id: str,
    repository_commit: str,
) -> pd.DataFrame:
    required_manifest = {"case_id", "subject_id", "study_id", "split", "source_dataset"}
    missing_manifest = sorted(required_manifest - set(manifest.columns))
    if missing_manifest:
        raise ValueError(f"Amaral input manifest is missing columns: {missing_manifest}")
    frames: list[pd.DataFrame] = []
    for mode, path in runner_outputs:
        raw = pd.read_json(path, lines=True)
        required_raw = {
            "case_id",
            "prediction_score",
            "raw_predicted_class",
            "retained_statement_count",
            "preprocessing_sha256",
        }
        missing_raw = sorted(required_raw - set(raw.columns))
        if missing_raw:
            raise ValueError(f"Amaral runner output is missing columns: {missing_raw}")
        if raw.duplicated("case_id", keep=False).any():
            raise ValueError(f"Amaral runner output has duplicate case IDs for mode {mode}")
        merged = manifest[list(required_manifest)].merge(
            raw,
            on="case_id",
            how="inner",
            validate="one_to_one",
        )
        if len(merged) != len(manifest) or len(raw) != len(manifest):
            raise ValueError(
                f"Amaral {mode} row-set mismatch: manifest={len(manifest)}, "
                f"runner={len(raw)}, joined={len(merged)}"
            )
        is_primary = mode == "published_mimic_preprocessing"
        merged["model_name"] = AMARAL_PRIMARY_MODEL if is_primary else AMARAL_SENSITIVITY_MODEL
        merged["model_family"] = "xgboost_count_vectorizer"
        merged["comparison_role"] = "external_comparator"
        merged["intended_target"] = "both"
        merged["model_source_repository"] = "amarallab/ARDS_diagnosis"
        merged["model_source_commit"] = repository_commit
        merged["model_artifact_version"] = config.expected_sha256["model"][:12]
        merged["text_scope"] = (
            "full_report_with_published_segmentation" if is_primary else "full_report_raw_direct"
        )
        merged["prediction_label"] = (
            merged["prediction_score"].astype(float) >= config.threshold
        ).astype(int)
        merged["threshold"] = config.threshold
        merged["run_id"] = run_id
        merged["model_sha256"] = config.expected_sha256["model"]
        merged["vectorizer_sha256"] = config.expected_sha256["vectorizer"]
        merged["preprocessing_mode"] = mode
        frames.append(merged)
    predictions = validate_canonical_predictions(pd.concat(frames, ignore_index=True))
    assert_no_text_leakage(predictions)
    return predictions


def validate_amaral_anchor_predictions(
    predictions: pd.DataFrame,
    expected_path: Path,
    *,
    score_tolerance: float = 1e-8,
) -> dict[str, Any]:
    payload = json.loads(expected_path.read_text(encoding="utf-8"))
    rows = payload.get("rows") if isinstance(payload, dict) else None
    if not isinstance(rows, list) or not rows:
        raise ValueError("Amaral anchor expectation must contain a non-empty rows list")
    expected = pd.DataFrame(rows)
    keys = ["case_id", "model_name"]
    exact_columns = [
        "prediction_label",
        "raw_predicted_class",
        "retained_statement_count",
        "preprocessing_sha256",
    ]
    required = {*keys, "prediction_score", *exact_columns}
    missing = sorted(required - set(expected.columns))
    if missing:
        raise ValueError(f"Amaral anchor expectation is missing columns: {missing}")
    observed = predictions[list(required)].copy()
    joined = expected.merge(
        observed,
        on=keys,
        how="outer",
        validate="one_to_one",
        indicator=True,
        suffixes=("_expected", "_observed"),
    )
    if len(joined) != len(expected) or not joined["_merge"].eq("both").all():
        raise ValueError("Amaral synthetic anchor row set does not match its pinned expectation")
    for column in exact_columns:
        if not joined[f"{column}_expected"].eq(joined[f"{column}_observed"]).all():
            raise ValueError(f"Amaral synthetic anchor parity failed for {column}")
    score_matches = [
        isclose(float(expected_score), float(observed_score), abs_tol=score_tolerance, rel_tol=0)
        for expected_score, observed_score in zip(
            joined["prediction_score_expected"],
            joined["prediction_score_observed"],
            strict=True,
        )
    ]
    if not all(score_matches):
        raise ValueError("Amaral synthetic anchor parity failed for prediction_score")
    return {
        "status": "passed",
        "rows": int(len(expected)),
        "expectation_sha256": sha256_path(expected_path),
        "score_tolerance": score_tolerance,
    }


def write_amaral_run_outputs(
    *,
    predictions: pd.DataFrame,
    config: AmaralComparatorConfig,
    resource_verification: dict[str, Any],
    run_id: str,
    command: list[str],
    run_kind: str,
) -> dict[str, Any]:
    if run_kind not in {"full", "smoke", "partial"}:
        raise ValueError(f"Unknown Amaral run kind: {run_kind}")
    ensure_parent_dir(config.prediction_output)
    predictions.to_parquet(config.prediction_output, index=False)
    ensure_dir(config.artifact_dir)
    state = repository_state(Path.cwd())
    models = sorted(predictions["model_name"].unique().tolist())
    rows_by_model = {
        str(key): int(value)
        for key, value in predictions["model_name"].value_counts(sort=False).items()
    }
    summary = {
        "name": config.source.name,
        "run_id": run_id,
        "run_at_utc": datetime.now(UTC).isoformat(),
        "prediction_rows": int(len(predictions)),
        "unique_cases": int(predictions["case_id"].nunique()),
        "models": models,
        "prediction_rows_by_model": rows_by_model,
        "positive_rate_by_model": {
            str(key): float(value)
            for key, value in predictions.groupby("model_name", observed=True)["prediction_label"]
            .mean()
            .items()
        },
        "score_mean_by_model": {
            str(key): float(value)
            for key, value in predictions.groupby("model_name", observed=True)["prediction_score"]
            .mean()
            .items()
        },
        "resource_verification": resource_verification,
        "ards_nlp_repository": state,
        "command": command,
        "contains_report_text": False,
        "run_kind": run_kind,
    }
    (config.artifact_dir / "run_manifest.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    qa = {
        "prediction_rows": summary["prediction_rows"],
        "unique_cases": summary["unique_cases"],
        "duplicate_case_model_rows": int(predictions.duplicated(["case_id", "model_name"]).sum()),
        "missing_scores": int(predictions["prediction_score"].isna().sum()),
        "scores_outside_0_1": int((~predictions["prediction_score"].between(0, 1)).sum()),
        "contains_report_text": False,
    }
    (config.artifact_dir / "output_qa.json").write_text(
        json.dumps(qa, indent=2) + "\n", encoding="utf-8"
    )
    status_name = AMARAL_PRIMARY_MODEL if AMARAL_PRIMARY_MODEL in models else models[0]
    reasons = {
        "full": "full MIMIC prediction modes completed",
        "smoke": "synthetic Amaral anchor modes completed",
        "partial": "explicit noncanonical Amaral mode completed",
    }
    write_comparator_status(
        name=status_name,
        status="available",
        reason=reasons[run_kind],
        out_path=config.artifact_dir / "status.json",
        details={
            "prediction_rows": int(len(predictions)),
            "prediction_rows_by_model": rows_by_model,
            "models": models,
            "run_id": run_id,
            "run_kind": run_kind,
        },
    )
    return summary
