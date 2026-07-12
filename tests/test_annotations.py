from __future__ import annotations

import pandas as pd
import pytest

from ards_cxr_benchmark.annotations import (
    text_stripped_review_labels,
    validate_review_dataframe,
)


def test_validate_review_dataframe_accepts_completed_rows_and_strips_text() -> None:
    df = pd.DataFrame(
        {
            "subject_id": [1],
            "study_id": [10],
            "human_report_probability_0_100": [85],
            "reviewer_id": ["reviewer_a"],
            "reviewer_notes": ["fits target construct"],
            "report_text": ["Sensitive report text"],
            "impression_text": ["Sensitive impression text"],
            "silver_label_source": ["regex_strict"],
        }
    )

    result = validate_review_dataframe(df)
    labels = text_stripped_review_labels(df)

    assert result.passed is True
    assert result.n_completed_rows == 1
    assert "report_text" not in labels.columns
    assert "impression_text" not in labels.columns
    assert labels.loc[0, "human_report_probability_0_100"] == 85.0
    assert bool(labels.loc[0, "human_review_completed"]) is True


def test_validate_review_dataframe_rejects_missing_required_columns() -> None:
    result = validate_review_dataframe(pd.DataFrame({"subject_id": [1]}))

    assert result.passed is False
    assert result.issues[0].check == "required_columns"


def test_validate_review_dataframe_rejects_invalid_probability() -> None:
    df = _valid_review_df()
    df.loc[0, "human_report_probability_0_100"] = 101

    result = validate_review_dataframe(df)

    assert result.passed is False
    assert {issue.check for issue in result.issues} == {"probability_range"}


def test_validate_review_dataframe_rejects_blank_reviewer_for_completed_row() -> None:
    df = _valid_review_df()
    df.loc[0, "reviewer_id"] = " "

    result = validate_review_dataframe(df)

    assert result.passed is False
    assert {issue.check for issue in result.issues} == {"reviewer_id"}


def test_validate_review_dataframe_rejects_duplicate_reviewer_study_rows() -> None:
    df = pd.concat([_valid_review_df(), _valid_review_df()], ignore_index=True)

    result = validate_review_dataframe(df)

    assert result.passed is False
    assert {issue.check for issue in result.issues} == {"duplicate_reviewer_study"}


def test_text_stripped_review_labels_raises_for_invalid_input() -> None:
    df = _valid_review_df()
    df.loc[0, "human_report_probability_0_100"] = -1

    with pytest.raises(ValueError, match="invalid probability"):
        text_stripped_review_labels(df)


def _valid_review_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "subject_id": [1],
            "study_id": [10],
            "human_report_probability_0_100": [75],
            "reviewer_id": ["reviewer_a"],
            "reviewer_notes": ["reviewed"],
        }
    )
