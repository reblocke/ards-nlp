from __future__ import annotations

import gzip
import json
from pathlib import Path

import pandas as pd
import pytest

from ards_cxr_benchmark.comparators.common import (
    CANONICAL_PREDICTION_COLUMNS,
    benchmark_comparator_predictions,
    build_comparator_source,
    canonical_case_id,
    normalize_clamp_predictions,
    normalize_clamp_python_predictions,
    normalize_silver_baseline_predictions,
    validate_canonical_predictions,
    validate_compatibility_mirror_metrics,
    write_combined_benchmark_outputs,
    write_comparator_input_packet,
)


def test_build_source_and_packet_are_exact_and_text_safe(tmp_path: Path) -> None:
    reports, extract = _source_frames()
    source = build_comparator_source(reports, extract, expected_rows=4)

    assert source["case_id"].tolist() == [
        "mimic_1_10",
        "mimic_2_20",
        "mimic_3_30",
        "mimic_4_40",
    ]
    packet = tmp_path / "packet.jsonl.gz"
    manifest = tmp_path / "manifest.parquet"
    summary = tmp_path / "summary.json"
    first = write_comparator_input_packet(
        source,
        packet_path=packet,
        manifest_path=manifest,
        summary_path=summary,
    )
    first_bytes = packet.read_bytes()
    second = write_comparator_input_packet(
        source,
        packet_path=packet,
        manifest_path=manifest,
        summary_path=summary,
    )

    assert first_bytes == packet.read_bytes()
    assert first["packet_sha256"] == second["packet_sha256"]
    manifest_df = pd.read_parquet(manifest)
    assert "report_text" not in manifest_df.columns
    assert "target_text_impression_findings" not in manifest_df.columns
    with gzip.open(packet, "rt", encoding="utf-8") as handle:
        row = json.loads(next(handle))
    assert row["report_text"] == "full 1"
    assert row["target_text_impression_findings"] == "focused 1"


def test_build_source_uses_explicit_impression_findings_fallback(tmp_path: Path) -> None:
    reports, extract = _source_frames()
    reports.loc[0, "target_text_impression_findings"] = "  "
    reports.loc[0, "target_text_impression_fallback"] = "fallback 1"

    source = build_comparator_source(reports, extract)

    assert source.loc[0, "target_text_impression_findings"] == "fallback 1"
    assert bool(source.loc[0, "impression_findings_fallback_used"])
    packet = tmp_path / "packet.jsonl.gz"
    manifest = tmp_path / "manifest.parquet"
    summary = tmp_path / "summary.json"
    result = write_comparator_input_packet(
        source,
        packet_path=packet,
        manifest_path=manifest,
        summary_path=summary,
    )
    manifest_df = pd.read_parquet(manifest)
    assert manifest_df["impression_findings_fallback_used"].tolist() == [True, False, False, False]
    assert result["impression_findings_fallback_rows"] == 1


def test_build_source_rejects_nonidentical_keys() -> None:
    reports, extract = _source_frames()
    extract.loc[0, "study_id"] = 999

    with pytest.raises(ValueError, match="identical study keys"):
        build_comparator_source(reports, extract)


def test_canonical_prediction_validation_rejects_case_mismatch_and_text() -> None:
    predictions = _canonical_predictions()
    predictions.loc[0, "case_id"] = "mimic_1_999"
    with pytest.raises(ValueError, match="case_id does not match"):
        validate_canonical_predictions(predictions)

    predictions = _canonical_predictions()
    predictions["report_text"] = "restricted"
    with pytest.raises(ValueError, match="text- or path-bearing"):
        validate_canonical_predictions(predictions)


def test_benchmark_uses_holdout_splits_native_labels_and_intended_target() -> None:
    reports, extract = _source_frames()
    source = build_comparator_source(reports, extract)
    predictions = _canonical_predictions()
    strict_only = predictions.copy()
    strict_only["model_name"] = "strict_model"
    strict_only["intended_target"] = "strict"
    combined = pd.concat([predictions, strict_only], ignore_index=True)

    metrics, strata, catalog = benchmark_comparator_predictions(combined, source)

    assert set(metrics["split"]) == {"validation", "test"}
    assert not ((metrics["model_name"] == "strict_model") & (metrics["task"] == "sensitive")).any()
    val = metrics[
        (metrics["model_name"] == "example_model")
        & (metrics["task"] == "strict")
        & (metrics["split"] == "validation")
    ].iloc[0]
    assert val["n"] == 2
    assert val["accuracy"] == 1.0
    assert val["npv"] == 1.0
    assert set(strata["stratum"]) == {
        "silver_label_source",
        "manual_review_priority",
        "qa_flags_present",
    }
    assert set(catalog["model_name"]) == {"example_model", "strict_model"}


def test_benchmark_stratifies_all_models_by_authoritative_reference_fields() -> None:
    reports, extract = _source_frames()
    source = build_comparator_source(reports, extract)
    external = _canonical_predictions()
    baseline = external.copy()
    baseline["model_name"] = "baseline_model"
    baseline["comparison_role"] = "trained_silver_baseline"
    baseline["silver_label_source"] = "stale_source"
    baseline["manual_review_priority"] = "stale_priority"
    baseline["qa_flags_present"] = False
    combined = pd.concat([external, baseline], ignore_index=True)

    _, strata, _ = benchmark_comparator_predictions(combined, source)

    validation_strict = strata[(strata["task"] == "strict") & (strata["split"] == "validation")]
    for model_name in ["example_model", "baseline_model"]:
        model_strata = validation_strict[validation_strict["model_name"] == model_name]
        source_values = set(
            model_strata.loc[model_strata["stratum"] == "silver_label_source", "stratum_value"]
        )
        priority_values = set(
            model_strata.loc[model_strata["stratum"] == "manual_review_priority", "stratum_value"]
        )
        qa_values = set(
            model_strata.loc[model_strata["stratum"] == "qa_flags_present", "stratum_value"]
        )
        assert source_values == {"regex", "radgraph"}
        assert priority_values == {"high"}
        assert qa_values == {"False", "True"}


def test_existing_prediction_normalizers_assign_roles() -> None:
    reports, extract = _source_frames()
    source = build_comparator_source(reports, extract)
    clamp = pd.DataFrame(
        {
            "subject_id": [1, 2, 3, 4],
            "study_id": [10, 20, 30, 40],
            "source_dataset": ["legacy"] * 4,
            "prediction_score": [0.0, 1.0, 0.0, 1.0],
            "prediction_label": [0, 1, 0, 1],
        }
    )
    normalized_clamp = normalize_clamp_predictions(
        clamp, source, run_id="run", source_commit="a" * 40
    )
    assert set(normalized_clamp["comparison_role"]) == {"legacy_teacher"}
    assert set(normalized_clamp["source_dataset"]) == {"mimic_cxr"}

    python_clamp = pd.DataFrame(
        {
            "clamp_doc_id": ["s10", "s20", "s30", "s40"],
            "prediction_status": ["evaluable"] * 4,
            "prediction_label": [0, 1, 0, 1],
            "clamp_ards_entity_count": [0, 1, 0, 2],
            "source_text_sha256": ["restricted-hash"] * 4,
        }
    )
    normalized_python = normalize_clamp_python_predictions(
        python_clamp, source, run_id="run", source_commit="a" * 40
    )
    assert set(normalized_python["comparison_role"]) == {"compatibility_mirror"}
    assert set(normalized_python["model_name"]) == {"clamp_python_compatibility"}
    assert "source_text_sha256" not in normalized_python.columns
    assert normalized_python["case_id"].tolist() == [
        "mimic_1_10",
        "mimic_2_20",
        "mimic_3_30",
        "mimic_4_40",
    ]

    baseline = pd.DataFrame(
        {
            "subject_id": [2, 2],
            "study_id": [20, 20],
            "split": ["validation", "validation"],
            "task": ["strict", "strict"],
            "model": ["tfidf_logreg", "structured_logreg"],
            "prediction_score": [0.8, 1.0],
            "prediction_label": [1, 1],
        }
    )
    normalized_baseline = normalize_silver_baseline_predictions(
        baseline, source, run_id="run", source_commit="a" * 40
    )
    assert set(normalized_baseline["comparison_role"]) == {
        "trained_silver_baseline",
        "silver_derived_control",
    }


def test_silver_baseline_normalizer_rejects_split_mismatch() -> None:
    reports, extract = _source_frames()
    source = build_comparator_source(reports, extract)
    baseline = pd.DataFrame(
        {
            "subject_id": [2],
            "study_id": [20],
            "split": ["test"],
            "task": ["strict"],
            "model": ["tfidf_logreg"],
            "prediction_score": [0.8],
            "prediction_label": [1],
        }
    )

    with pytest.raises(ValueError, match="split mismatches"):
        normalize_silver_baseline_predictions(
            baseline, source, run_id="run", source_commit="a" * 40
        )


def test_silver_baseline_normalizer_rejects_unmatched_ids() -> None:
    reports, extract = _source_frames()
    source = build_comparator_source(reports, extract)
    baseline = pd.DataFrame(
        {
            "subject_id": [999],
            "study_id": [9990],
            "split": ["test"],
            "task": ["strict"],
            "model": ["tfidf_logreg"],
            "prediction_score": [0.8],
            "prediction_label": [1],
        }
    )

    with pytest.raises(ValueError, match="without a comparator reference"):
        normalize_silver_baseline_predictions(
            baseline, source, run_id="run", source_commit="a" * 40
        )


@pytest.mark.parametrize("invalid_id", [None, "", "10", "s0", "sbad"])
def test_python_clamp_normalizer_rejects_malformed_ids(invalid_id: object) -> None:
    reports, extract = _source_frames()
    source = build_comparator_source(reports, extract)
    predictions = _python_clamp_predictions()
    predictions.loc[0, "clamp_doc_id"] = invalid_id

    with pytest.raises(ValueError, match=r"(missing|blank|s\{positive_study_id\})"):
        normalize_clamp_python_predictions(
            predictions, source, run_id="run", source_commit="a" * 40
        )


def test_python_clamp_normalizer_rejects_missing_and_duplicate_studies() -> None:
    reports, extract = _source_frames()
    source = build_comparator_source(reports, extract)
    predictions = _python_clamp_predictions()

    with pytest.raises(ValueError, match="complete comparator reference"):
        normalize_clamp_python_predictions(
            predictions.iloc[:-1], source, run_id="run", source_commit="a" * 40
        )

    duplicate = predictions.copy()
    duplicate.loc[1, "clamp_doc_id"] = "s10"
    with pytest.raises(ValueError, match="duplicate study IDs"):
        normalize_clamp_python_predictions(duplicate, source, run_id="run", source_commit="a" * 40)


def test_python_clamp_normalizer_rejects_label_count_disagreement() -> None:
    reports, extract = _source_frames()
    source = build_comparator_source(reports, extract)
    predictions = _python_clamp_predictions()
    predictions.loc[0, "prediction_label"] = 1

    with pytest.raises(ValueError, match="labels do not agree"):
        normalize_clamp_python_predictions(
            predictions, source, run_id="run", source_commit="a" * 40
        )


def test_benchmark_rejects_incomplete_reference_join() -> None:
    reports, extract = _source_frames()
    source = build_comparator_source(reports, extract).iloc[:-1].copy()

    with pytest.raises(ValueError, match="did not join"):
        benchmark_comparator_predictions(_canonical_predictions(), source)


def test_mismatched_target_metrics_are_separate_from_primary_ranking(tmp_path: Path) -> None:
    reports, extract = _source_frames()
    source = build_comparator_source(reports, extract)
    primary = _canonical_predictions()
    exploratory = primary.copy()
    exploratory["model_name"] = "afshar_text_svc_full_ards"
    exploratory["comparison_role"] = "mismatched_target_exploratory"
    metrics, strata, catalog = benchmark_comparator_predictions(
        pd.concat([primary, exploratory], ignore_index=True), source
    )

    write_combined_benchmark_outputs(
        metrics=metrics,
        strata=strata,
        catalog=catalog,
        statuses=[],
        out_dir=tmp_path,
    )

    primary_metrics = pd.read_csv(tmp_path / "silver_metrics.csv")
    exploratory_metrics = pd.read_csv(tmp_path / "mismatched_target_exploratory_metrics.csv")
    assert set(primary_metrics["model_name"]) == {"example_model"}
    assert set(exploratory_metrics["model_name"]) == {"afshar_text_svc_full_ards"}


def test_compatibility_metrics_are_separate_and_must_match_legacy(tmp_path: Path) -> None:
    reports, extract = _source_frames()
    source = build_comparator_source(reports, extract)
    legacy = _canonical_predictions()
    legacy["model_name"] = "clamp_legacy"
    legacy["model_family"] = "rule_based_nlp"
    legacy["comparison_role"] = "legacy_teacher"
    mirror = legacy.copy()
    mirror["model_name"] = "clamp_python_compatibility"
    mirror["comparison_role"] = "compatibility_mirror"
    metrics, strata, catalog = benchmark_comparator_predictions(
        pd.concat([legacy, mirror], ignore_index=True), source
    )

    validate_compatibility_mirror_metrics(metrics)
    write_combined_benchmark_outputs(
        metrics=metrics,
        strata=strata,
        catalog=catalog,
        statuses=[],
        out_dir=tmp_path,
    )

    assert (
        pd.read_csv(tmp_path / "compatibility_mirror_metrics.csv")["model_name"]
        .eq("clamp_python_compatibility")
        .all()
    )
    assert "clamp_python_compatibility" not in set(
        pd.read_csv(tmp_path / "silver_metrics.csv")["model_name"]
    )

    changed = metrics.copy()
    mask = changed["comparison_role"] == "compatibility_mirror"
    changed.loc[mask, "accuracy"] = changed.loc[mask, "accuracy"] - 0.1
    with pytest.raises(ValueError, match="differ from legacy CLAMP"):
        validate_compatibility_mirror_metrics(changed)


def test_canonical_case_id() -> None:
    assert canonical_case_id("1", 10.0) == "mimic_1_10"
    with pytest.raises(ValueError):
        canonical_case_id(0, 10)


def _source_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    reports = pd.DataFrame(
        {
            "subject_id": [1, 2, 3, 4],
            "study_id": [10, 20, 30, 40],
            "report_text": [f"full {index}" for index in range(1, 5)],
            "target_text_impression_findings": [f"focused {index}" for index in range(1, 5)],
            "target_text_impression_fallback": [f"focused {index}" for index in range(1, 5)],
        }
    )
    extract = pd.DataFrame(
        {
            "subject_id": [1, 2, 3, 4],
            "study_id": [10, 20, 30, 40],
            "split": ["train", "validation", "validation", "test"],
            "strict_bilateral_opacity_label": [0, 1, 0, 1],
            "sensitive_bilateral_opacity_label": [0, 1, 1, 1],
            "silver_label_source": ["negative", "regex", "radgraph", "regex"],
            "manual_review_priority": ["low", "high", "high", "medium"],
            "qa_flags": [[], [], ["conflict"], []],
        }
    )
    return reports, extract


def _canonical_predictions() -> pd.DataFrame:
    rows = []
    for subject_id, study_id, split, score, label in [
        (1, 10, "train", 0.1, 0),
        (2, 20, "validation", 0.8, 1),
        (3, 30, "validation", 0.4, 0),
        (4, 40, "test", 0.9, 1),
    ]:
        rows.append(
            {
                "case_id": canonical_case_id(subject_id, study_id),
                "subject_id": subject_id,
                "study_id": study_id,
                "split": split,
                "source_dataset": "mimic_cxr",
                "model_name": "example_model",
                "model_family": "example",
                "comparison_role": "external_comparator",
                "intended_target": "both",
                "model_source_repository": "example/repo",
                "model_source_commit": "a" * 40,
                "model_artifact_version": "v1",
                "text_scope": "full_report",
                "prediction_score": score,
                "prediction_label": label,
                "raw_predicted_class": str(label),
                "threshold": 0.5,
                "run_id": "run",
            }
        )
    return pd.DataFrame(rows).reindex(columns=CANONICAL_PREDICTION_COLUMNS)


def _python_clamp_predictions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "clamp_doc_id": ["s10", "s20", "s30", "s40"],
            "prediction_status": ["evaluable"] * 4,
            "prediction_label": [0, 1, 0, 1],
            "clamp_ards_entity_count": [0, 1, 0, 2],
        }
    )
