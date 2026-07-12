from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .config import ensure_parent_dir

REQUIRED_REVIEW_COLUMNS = [
    "subject_id",
    "study_id",
    "human_report_probability_0_100",
    "reviewer_id",
    "reviewer_notes",
]
TEXT_REVIEW_COLUMNS = {
    "report_text",
    "findings_text",
    "impression_text",
    "target_text_full_report",
    "target_text_impression_findings",
    "target_text_impression_fallback",
    "primary_target_text",
    "radgraph_opacity_relation_examples",
}


@dataclass(frozen=True)
class AnnotationValidationIssue:
    check: str
    severity: str
    message: str


@dataclass(frozen=True)
class AnnotationValidationResult:
    passed: bool
    n_rows: int
    n_completed_rows: int
    issues: list[AnnotationValidationIssue]


def validate_review_dataframe(df: pd.DataFrame) -> AnnotationValidationResult:
    issues: list[AnnotationValidationIssue] = []
    missing_columns = [column for column in REQUIRED_REVIEW_COLUMNS if column not in df.columns]
    if missing_columns:
        issues.append(
            _issue("required_columns", "error", f"Missing required columns: {missing_columns}")
        )
        return AnnotationValidationResult(
            passed=False,
            n_rows=int(len(df)),
            n_completed_rows=0,
            issues=issues,
        )

    if df["subject_id"].isna().any() or df["study_id"].isna().any():
        issues.append(_issue("ids_present", "error", "subject_id and study_id must be non-null"))

    probabilities = pd.to_numeric(df["human_report_probability_0_100"], errors="coerce")
    completed_mask = _completed_review_mask(df, probabilities)
    completed = df[completed_mask].copy()
    completed_probabilities = probabilities[completed_mask]

    invalid_probability = completed_probabilities.isna() | ~completed_probabilities.between(0, 100)
    if invalid_probability.any():
        issues.append(
            _issue(
                "probability_range",
                "error",
                f"{int(invalid_probability.sum())} completed rows have invalid probability values",
            )
        )

    blank_reviewer = completed["reviewer_id"].map(_is_blank)
    if blank_reviewer.any():
        issues.append(
            _issue(
                "reviewer_id",
                "error",
                f"{int(blank_reviewer.sum())} completed rows have blank reviewer_id",
            )
        )

    duplicate_mask = completed.duplicated(["subject_id", "study_id", "reviewer_id"], keep=False)
    if duplicate_mask.any():
        issues.append(
            _issue(
                "duplicate_reviewer_study",
                "error",
                f"{int(duplicate_mask.sum())} completed rows duplicate subject/study/reviewer",
            )
        )

    return AnnotationValidationResult(
        passed=not any(issue.severity == "error" for issue in issues),
        n_rows=int(len(df)),
        n_completed_rows=int(completed_mask.sum()),
        issues=issues,
    )


def text_stripped_review_labels(df: pd.DataFrame) -> pd.DataFrame:
    result = validate_review_dataframe(df)
    if not result.passed:
        raise ValueError("; ".join(issue.message for issue in result.issues))

    probabilities = pd.to_numeric(df["human_report_probability_0_100"], errors="coerce")
    completed = df[_completed_review_mask(df, probabilities)].copy()
    drop_columns = [
        column
        for column in completed.columns
        if column in TEXT_REVIEW_COLUMNS or "text" in column.lower()
    ]
    completed = completed.drop(columns=drop_columns, errors="ignore")
    completed["human_report_probability_0_100"] = probabilities.loc[completed.index].astype(float)
    completed["human_review_completed"] = True
    return completed.reset_index(drop=True)


def write_annotation_validation(result: AnnotationValidationResult, *, out_json: Path) -> None:
    ensure_parent_dir(out_json)
    out_json.write_text(
        json.dumps(
            {
                "passed": result.passed,
                "n_rows": result.n_rows,
                "n_completed_rows": result.n_completed_rows,
                "issues": [asdict(issue) for issue in result.issues],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def _completed_review_mask(df: pd.DataFrame, probabilities: pd.Series) -> pd.Series:
    reviewer_present = ~df["reviewer_id"].map(_is_blank)
    notes_present = ~df["reviewer_notes"].map(_is_blank)
    return probabilities.notna() | reviewer_present | notes_present


def _is_blank(value: Any) -> bool:
    return value is None or pd.isna(value) or str(value).strip() == ""


def _issue(check: str, severity: str, message: str) -> AnnotationValidationIssue:
    return AnnotationValidationIssue(check=check, severity=severity, message=message)
