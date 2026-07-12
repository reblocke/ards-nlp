from __future__ import annotations

import pandas as pd
import pytest

from ards_cxr_benchmark.probabilistic_benchmark import (
    normalize_prediction_dataframe,
    normalize_reference_dataframe,
    run_probabilistic_benchmark,
)


def test_normalize_predictions_from_silver_predictions() -> None:
    df = pd.DataFrame(
        {
            "subject_id": [1],
            "study_id": [10],
            "task": ["strict"],
            "model": ["tfidf_logreg"],
            "prediction_score": [0.7],
        }
    )

    normalized = normalize_prediction_dataframe(df)

    assert normalized.loc[0, "case_id"] == "1_10"
    assert normalized.loc[0, "model_name"] == "strict__tfidf_logreg"


@pytest.mark.parametrize("column", ["case_id", "model_name"])
def test_normalize_prediction_dataframe_rejects_missing_direct_identifiers(column: str) -> None:
    df = pd.DataFrame(
        {
            "case_id": ["case_a"],
            "model_name": ["model_a"],
            "prediction_score": [0.7],
        }
    )
    df.loc[0, column] = None

    with pytest.raises(ValueError, match=rf"{column} must be non-null"):
        normalize_prediction_dataframe(df)


@pytest.mark.parametrize("column", ["case_id", "model_name"])
def test_normalize_prediction_dataframe_rejects_blank_direct_identifiers(column: str) -> None:
    df = pd.DataFrame(
        {
            "case_id": ["case_a"],
            "model_name": ["model_a"],
            "prediction_score": [0.7],
        }
    )
    df.loc[0, column] = " "

    with pytest.raises(ValueError, match=rf"{column} must be non-blank"):
        normalize_prediction_dataframe(df)


@pytest.mark.parametrize("column", ["subject_id", "study_id", "model", "task"])
def test_normalize_prediction_dataframe_rejects_missing_derived_identifiers(column: str) -> None:
    df = pd.DataFrame(
        {
            "subject_id": [1],
            "study_id": [10],
            "task": ["strict"],
            "model": ["tfidf_logreg"],
            "prediction_score": [0.7],
        }
    )
    df.loc[0, column] = None

    with pytest.raises(ValueError, match=rf"{column} must be non-null"):
        normalize_prediction_dataframe(df)


def test_normalize_prediction_dataframe_rejects_text_leakage() -> None:
    df = pd.DataFrame(
        {
            "case_id": ["case_a"],
            "model_name": ["model"],
            "prediction_score": [0.7],
            "report_text": ["leaked"],
        }
    )

    with pytest.raises(ValueError, match="Unsafe text/path"):
        normalize_prediction_dataframe(df)


def test_normalize_reference_dataframe_rejects_missing_case_id() -> None:
    df = pd.DataFrame({"case_id": [None], "mean_report_probability": [0.8]})

    with pytest.raises(ValueError, match="case_id must be non-null"):
        normalize_reference_dataframe(df)


def test_normalize_reference_dataframe_rejects_blank_case_id() -> None:
    df = pd.DataFrame({"case_id": [" "], "mean_report_probability": [0.8]})

    with pytest.raises(ValueError, match="case_id must be non-blank"):
        normalize_reference_dataframe(df)


def test_normalize_reference_dataframe_rejects_malformed_nonblank_probability() -> None:
    df = pd.DataFrame({"case_id": ["case_a"], "mean_report_probability": ["oops"]})

    with pytest.raises(ValueError, match="non-numeric probability"):
        normalize_reference_dataframe(df)


def test_normalize_reference_dataframe_accepts_blank_missing_and_numeric_strings() -> None:
    df = pd.DataFrame(
        {
            "case_id": ["case_a", "case_b"],
            "mean_report_probability": ["0.7", " "],
            "mean_image_probability": [None, "0.5"],
        }
    )

    normalized = normalize_reference_dataframe(df)

    assert normalized.loc[0, "mean_report_probability"] == pytest.approx(0.7)
    assert pd.isna(normalized.loc[1, "mean_report_probability"])
    assert normalized.loc[1, "mean_image_probability"] == pytest.approx(0.5)


def test_probabilistic_benchmark_outputs_report_and_image_metrics() -> None:
    result = run_probabilistic_benchmark(_reference(), _predictions(), thresholds=[0.5], n_bins=2)

    assert result.join_audit["joined_rows"] == 4
    assert set(result.metrics["target_type"]) == {"report", "image"}
    assert set(result.threshold_metrics["target_type"]) == {"report", "image"}
    assert set(result.calibration["target_type"]) == {"report", "image"}


def test_probabilistic_benchmark_degrades_to_report_only_when_image_target_absent() -> None:
    reference = _reference().drop(columns=["mean_image_probability"])

    result = run_probabilistic_benchmark(reference, _predictions(), thresholds=[0.5])

    assert set(result.metrics["target_type"]) == {"report"}
    assert result.join_audit["targets_available"] == ["report"]


def test_probabilistic_benchmark_rejects_zero_overlapping_case_ids() -> None:
    predictions = pd.DataFrame(
        {
            "case_id": ["case_c"],
            "model_name": ["model_a"],
            "prediction_score": [0.7],
        }
    )

    with pytest.raises(ValueError, match="No prediction rows matched reference rows"):
        run_probabilistic_benchmark(_reference(), predictions)


def test_probabilistic_benchmark_rejects_join_with_no_evaluable_targets() -> None:
    reference = pd.DataFrame(
        {
            "case_id": ["case_a", "case_b"],
            "mean_report_probability": [pd.NA, 0.8],
        }
    )
    predictions = pd.DataFrame(
        {
            "case_id": ["case_a"],
            "model_name": ["model_a"],
            "prediction_score": [0.7],
        }
    )

    with pytest.raises(ValueError, match="No benchmark metric rows"):
        run_probabilistic_benchmark(reference, predictions)


def _reference() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "case_id": ["case_a", "case_b"],
            "mean_report_probability": [0.8, 0.2],
            "mean_image_probability": [0.7, 0.1],
        }
    )


def _predictions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "case_id": ["case_a", "case_b", "case_a", "case_b"],
            "model_name": ["model_a", "model_a", "model_b", "model_b"],
            "prediction_score": [0.9, 0.1, 0.6, 0.4],
        }
    )
