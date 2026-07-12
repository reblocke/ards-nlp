from __future__ import annotations

import pandas as pd
import pytest

from ards_cxr_benchmark.annotation_reference import (
    build_case_reference_standard,
    clean_rater_ratings,
    cohen_kappa,
    rater_pairwise_agreement,
    validate_rating_dataframe,
)


def test_probabilistic_rating_validation_accepts_long_schema_and_strips_text() -> None:
    df = _ratings()
    df["report_text"] = "sensitive report text"
    df["findings"] = "sensitive findings text"
    df["impression"] = "sensitive impression text"
    df["image_path"] = "/not/for/output.png"

    result = validate_rating_dataframe(df)
    cleaned = clean_rater_ratings(df)

    assert result.passed is True
    assert result.n_completed_rows == len(df)
    assert "report_text" not in cleaned.columns
    assert "findings" not in cleaned.columns
    assert "impression" not in cleaned.columns
    assert "image_path" not in cleaned.columns
    assert cleaned["probability"].between(0, 1).all()


def test_probabilistic_rating_validation_rejects_missing_columns() -> None:
    result = validate_rating_dataframe(pd.DataFrame({"case_id": ["case_a"]}))

    assert result.passed is False
    assert result.issues[0].check == "required_columns"


def test_probabilistic_rating_validation_rejects_invalid_task_probability_and_duplicate() -> None:
    df = _ratings()
    df.loc[0, "review_task"] = "combined"
    df.loc[1, "probability_0_100"] = 101
    df = pd.concat([df, df.iloc[[2]]], ignore_index=True)

    result = validate_rating_dataframe(df)

    assert result.passed is False
    assert {
        "review_task",
        "probability_range",
        "duplicate_case_rater_task",
    }.issubset({issue.check for issue in result.issues})


def test_probabilistic_rating_validation_rejects_normalized_duplicate_keys() -> None:
    df = pd.DataFrame(
        {
            "case_id": ["case_a", " case_a "],
            "rater_id": ["R01", " R01 "],
            "review_task": ["report_only", " report_only "],
            "probability_0_100": [80, 60],
        }
    )

    result = validate_rating_dataframe(df)

    assert result.passed is False
    assert "duplicate_case_rater_task" in {issue.check for issue in result.issues}


def test_case_reference_standard_computes_counts_means_ranges_and_secondary_labels() -> None:
    reference = build_case_reference_standard(_ratings())

    case_a = reference[reference["case_id"] == "case_a"].iloc[0]
    assert case_a["n_report_raters"] == 2
    assert case_a["n_image_raters"] == 2
    assert case_a["mean_report_probability"] == pytest.approx(0.7)
    assert case_a["sd_report_probability"] == pytest.approx(0.1)
    assert case_a["range_image_probability"] == pytest.approx(0.2)
    assert bool(case_a["report_label_ge_050"]) is True
    assert bool(case_a["image_label_ge_067"]) is True


def test_pairwise_rater_agreement_computes_mae_rmse_and_kappa() -> None:
    pairwise = rater_pairwise_agreement(_ratings())

    report_pair = pairwise[
        (pairwise["review_task"] == "report_only")
        & (pairwise["rater_a"] == "R01")
        & (pairwise["rater_b"] == "R02")
    ].iloc[0]
    assert report_pair["n_cases"] == 2
    assert report_pair["mean_absolute_difference"] == pytest.approx(0.15)
    assert report_pair["root_mean_squared_difference"] == pytest.approx((0.025) ** 0.5)
    assert report_pair["binary_agreement_ge_050"] == pytest.approx(1.0)
    assert cohen_kappa(pd.Series([True, False]), pd.Series([True, False])) == pytest.approx(1.0)


def _ratings() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "case_id": ["case_a", "case_a", "case_a", "case_a", "case_b", "case_b"],
            "rater_id": ["R01", "R02", "R01", "R02", "R01", "R02"],
            "review_task": [
                "report_only",
                "report_only",
                "image_only",
                "image_only",
                "report_only",
                "report_only",
            ],
            "probability_0_100": [80, 60, 70, 90, 20, 30],
        }
    )
