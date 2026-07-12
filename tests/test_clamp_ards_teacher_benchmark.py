from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ards_cxr_benchmark.clamp_ards_benchmark import (
    BENCHMARK_INTERPRETATION,
    benchmark_clamp_against_silver,
    join_predictions_to_model_extract,
    model_extract_sql,
    write_clamp_teacher_benchmark_outputs,
)


def test_join_predictions_to_model_extract_normalizes_numeric_ids() -> None:
    predictions = pd.DataFrame(
        {
            "subject_id": ["1"],
            "study_id": ["10"],
            "model_name": ["clamp_legacy"],
            "prediction_score": [1.0],
        }
    )
    extract = pd.DataFrame(
        {
            "subject_id": [1],
            "study_id": [10],
            "strict_bilateral_opacity_label": [1],
            "sensitive_bilateral_opacity_label": [1],
        }
    )

    joined = join_predictions_to_model_extract(predictions, extract)

    assert len(joined) == 1


def test_join_predictions_to_model_extract_normalizes_decimal_integer_ids() -> None:
    predictions = pd.DataFrame(
        {
            "subject_id": [1.0],
            "study_id": [10.0],
            "model_name": ["clamp_legacy"],
            "prediction_score": [1.0],
        }
    )
    extract = pd.DataFrame(
        {
            "subject_id": ["1"],
            "study_id": ["10"],
            "strict_bilateral_opacity_label": [1],
            "sensitive_bilateral_opacity_label": [1],
        }
    )

    joined = join_predictions_to_model_extract(predictions, extract)

    assert len(joined) == 1


@pytest.mark.parametrize(
    ("table_name", "column_name", "bad_value", "message"),
    [
        ("predictions", "subject_id", None, "predictions.subject_id contains missing values"),
        ("predictions", "study_id", "", "predictions.study_id contains blank values"),
        ("predictions", "subject_id", "abc", "predictions.subject_id contains non-numeric values"),
        ("predictions", "study_id", 10.5, "predictions.study_id contains non-integer values"),
        ("model_extract", "subject_id", None, "model_extract.subject_id contains missing values"),
        ("model_extract", "study_id", " ", "model_extract.study_id contains blank values"),
        (
            "model_extract",
            "subject_id",
            "abc",
            "model_extract.subject_id contains non-numeric values",
        ),
        ("model_extract", "study_id", 20.5, "model_extract.study_id contains non-integer values"),
    ],
)
def test_join_predictions_to_model_extract_rejects_invalid_join_ids(
    table_name: str,
    column_name: str,
    bad_value: object,
    message: str,
) -> None:
    predictions = pd.DataFrame(
        {
            "subject_id": [1],
            "study_id": [10],
            "model_name": ["clamp_legacy"],
            "prediction_score": [1.0],
        }
    )
    extract = pd.DataFrame(
        {
            "subject_id": [1],
            "study_id": [10],
            "strict_bilateral_opacity_label": [1],
            "sensitive_bilateral_opacity_label": [1],
        }
    )
    predictions = predictions.astype({"subject_id": "object", "study_id": "object"})
    extract = extract.astype({"subject_id": "object", "study_id": "object"})
    if table_name == "predictions":
        predictions.loc[0, column_name] = bad_value
    else:
        extract.loc[0, column_name] = bad_value

    with pytest.raises(ValueError, match=message):
        join_predictions_to_model_extract(predictions, extract)


def test_benchmark_computes_expected_metrics_and_strata(tmp_path: Path) -> None:
    predictions = pd.DataFrame(
        {
            "subject_id": [1, 2, 3, 4],
            "study_id": [10, 20, 30, 40],
            "model_name": ["clamp_legacy"] * 4,
            "prediction_score": [1.0, 0.0, 1.0, 0.0],
            "prediction_label": [1, 0, 1, 0],
        }
    )
    extract = pd.DataFrame(
        {
            "subject_id": [1, 2, 3, 4],
            "study_id": [10, 20, 30, 40],
            "strict_bilateral_opacity_label": [1, 0, 0, 0],
            "sensitive_bilateral_opacity_label": [1, 0, 1, 0],
            "silver_label_source": ["regex", "regex", "radgraph", "radgraph"],
            "manual_review_priority": ["high", "low", "high", "low"],
            "qa_flags": ["none", "none", "conflict", "none"],
        }
    )

    metrics, strata, summary = benchmark_clamp_against_silver(
        predictions=predictions,
        model_extract=extract,
    )

    strict = metrics.set_index("task").loc["strict"]
    sensitive = metrics.set_index("task").loc["sensitive"]
    assert strict["n"] == 4
    assert strict["accuracy"] == pytest.approx(0.75)
    assert strict["precision"] == pytest.approx(0.5)
    assert strict["recall"] == pytest.approx(1.0)
    assert strict["specificity"] == pytest.approx(2 / 3)
    assert strict["clamp_positive_rate"] == pytest.approx(0.5)
    assert sensitive["accuracy"] == pytest.approx(1.0)
    assert {"silver_label_source", "manual_review_priority", "qa_flags_present"}.issubset(
        set(strata["stratum"])
    )
    assert summary["joined_rows"] == 4

    write_clamp_teacher_benchmark_outputs(
        metrics=metrics,
        strata=strata,
        summary=summary,
        out_dir=tmp_path / "benchmark",
    )
    summary_md = (tmp_path / "benchmark" / "clamp_vs_silver_summary.md").read_text(encoding="utf-8")
    assert BENCHMARK_INTERPRETATION in summary_md


def test_join_normalizes_repeated_bigquery_qa_flags() -> None:
    predictions = pd.DataFrame(
        {
            "subject_id": [1, 2],
            "study_id": [10, 20],
            "model_name": ["clamp_legacy", "clamp_legacy"],
            "prediction_score": [0.0, 1.0],
        }
    )
    extract = pd.DataFrame(
        {
            "subject_id": [1, 2],
            "study_id": [10, 20],
            "strict_bilateral_opacity_label": [0, 1],
            "sensitive_bilateral_opacity_label": [0, 1],
            "qa_flags": [np.array([], dtype=object), np.array(["conflict"])],
        }
    )

    joined = join_predictions_to_model_extract(predictions, extract)

    assert joined["qa_flags_present"].tolist() == [False, True]


def test_benchmark_fails_when_no_predictions_join() -> None:
    predictions = pd.DataFrame(
        {
            "subject_id": [1],
            "study_id": [10],
            "model_name": ["clamp_legacy"],
            "prediction_score": [1.0],
            "prediction_label": [1],
        }
    )
    extract = pd.DataFrame(
        {
            "subject_id": [2],
            "study_id": [20],
            "strict_bilateral_opacity_label": [1],
            "sensitive_bilateral_opacity_label": [1],
        }
    )

    with pytest.raises(ValueError, match="No CLAMP prediction rows joined"):
        benchmark_clamp_against_silver(predictions=predictions, model_extract=extract)


def test_model_extract_sql_validates_source_table() -> None:
    sql = model_extract_sql("mimic-hypercapnia.ards_mimic_cxr_benchmark.model_development_extract")

    assert "FROM `mimic-hypercapnia.ards_mimic_cxr_benchmark.model_development_extract`" in sql


@pytest.mark.parametrize("source_table", ["", "  ", "project.dataset.`table`"])
def test_model_extract_sql_rejects_unsafe_source_table(source_table: str) -> None:
    with pytest.raises(ValueError, match="Unsafe BigQuery table reference"):
        model_extract_sql(source_table)
