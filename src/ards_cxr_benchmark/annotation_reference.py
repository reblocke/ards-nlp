from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import ensure_parent_dir

RATING_KEY_COLUMNS = ["case_id", "rater_id", "review_task"]
REQUIRED_RATING_COLUMNS = [*RATING_KEY_COLUMNS, "probability_0_100"]
VALID_REVIEW_TASKS = {"image_only", "report_only"}
COMPLETED_STATUSES = {"", "complete", "completed"}
KNOWN_REVIEW_STATUSES = COMPLETED_STATUSES | {"incomplete", "skipped"}
OPTIONAL_CASE_IDENTIFIER_COLUMNS = [
    "source_dataset",
    "subject_id",
    "study_id",
    "accession_id",
    "encounter_id",
    "annotation_phase",
]
UNSAFE_EXACT_COLUMNS = {
    "findings",
    "impression",
    "report_text",
    "findings_text",
    "impression_text",
    "primary_target_text",
    "target_text_full_report",
    "target_text_impression_findings",
    "target_text_impression_fallback",
    "reviewer_notes",
    "image_path",
    "image_file",
    "image_uri",
    "image_url",
    "dicom_path",
    "pdf_path",
}


@dataclass(frozen=True)
class RatingValidationIssue:
    check: str
    severity: str
    message: str


@dataclass(frozen=True)
class RatingValidationResult:
    passed: bool
    n_rows: int
    n_completed_rows: int
    issues: list[RatingValidationIssue]

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "n_rows": self.n_rows,
            "n_completed_rows": self.n_completed_rows,
            "issues": [asdict(issue) for issue in self.issues],
        }


def validate_rating_dataframe(df: pd.DataFrame) -> RatingValidationResult:
    issues: list[RatingValidationIssue] = []
    missing_columns = [column for column in REQUIRED_RATING_COLUMNS if column not in df.columns]
    if missing_columns:
        issues.append(
            _issue("required_columns", "error", f"Missing required columns: {missing_columns}")
        )
        return RatingValidationResult(False, int(len(df)), 0, issues)

    status = _normalized_review_status(df)
    invalid_status = ~status.isin(KNOWN_REVIEW_STATUSES)
    if invalid_status.any():
        values = sorted(status[invalid_status].dropna().unique().tolist())
        issues.append(_issue("review_status", "error", f"Invalid review_status values: {values}"))

    completed_mask = status.isin(COMPLETED_STATUSES)
    completed = _normalize_rating_key_columns(df[completed_mask].copy())
    probabilities = pd.to_numeric(completed["probability_0_100"], errors="coerce")

    if completed.empty:
        issues.append(_issue("completed_rows", "warning", "No completed rating rows found"))

    for column in RATING_KEY_COLUMNS:
        blank = completed[column].map(_is_blank)
        if blank.any():
            issues.append(
                _issue(column, "error", f"{int(blank.sum())} completed rows have blank {column}")
            )

    task_values = completed.loc[~completed["review_task"].map(_is_blank), "review_task"]
    invalid_task = ~task_values.isin(VALID_REVIEW_TASKS)
    if invalid_task.any():
        values = sorted(task_values[invalid_task].dropna().unique().tolist())
        issues.append(_issue("review_task", "error", f"Invalid review_task values: {values}"))

    invalid_probability = probabilities.isna() | ~probabilities.between(0, 100)
    if invalid_probability.any():
        issues.append(
            _issue(
                "probability_range",
                "error",
                f"{int(invalid_probability.sum())} completed rows have invalid probability values",
            )
        )

    duplicate_mask = completed.duplicated(RATING_KEY_COLUMNS, keep=False)
    if duplicate_mask.any():
        issues.append(
            _issue(
                "duplicate_case_rater_task",
                "error",
                f"{int(duplicate_mask.sum())} completed rows duplicate case/rater/task",
            )
        )

    return RatingValidationResult(
        passed=not any(issue.severity == "error" for issue in issues),
        n_rows=int(len(df)),
        n_completed_rows=int(completed_mask.sum()),
        issues=issues,
    )


def clean_rater_ratings(df: pd.DataFrame) -> pd.DataFrame:
    result = validate_rating_dataframe(df)
    if not result.passed:
        raise ValueError("; ".join(issue.message for issue in result.issues))

    status = _normalized_review_status(df)
    completed = _normalize_rating_key_columns(df[status.isin(COMPLETED_STATUSES)].copy())
    completed["probability_0_100"] = pd.to_numeric(
        completed["probability_0_100"], errors="raise"
    ).astype(float)
    completed["probability"] = completed["probability_0_100"] / 100.0
    return strip_unsafe_columns(completed).reset_index(drop=True)


def strip_unsafe_columns(df: pd.DataFrame) -> pd.DataFrame:
    drop_columns = [column for column in df.columns if is_unsafe_column(column)]
    return df.drop(columns=drop_columns, errors="ignore")


def is_unsafe_column(column: str) -> bool:
    lowered = column.lower()
    if lowered in UNSAFE_EXACT_COLUMNS:
        return True
    if "text" in lowered:
        return True
    if lowered.endswith("_path") or lowered.endswith("_file") or lowered.endswith("_uri"):
        return True
    return False


def build_case_reference_standard(ratings: pd.DataFrame) -> pd.DataFrame:
    cleaned = ratings.copy() if "probability" in ratings.columns else clean_rater_ratings(ratings)
    rows: list[dict[str, Any]] = []
    for case_id, case_df in cleaned.groupby("case_id", sort=True, observed=True):
        row: dict[str, Any] = {"case_id": case_id}
        for column in OPTIONAL_CASE_IDENTIFIER_COLUMNS:
            if column in case_df.columns:
                row[column] = _first_non_null(case_df[column])
        for task, prefix in (("image_only", "image"), ("report_only", "report")):
            values = case_df.loc[case_df["review_task"] == task, "probability"].astype(float)
            row.update(_case_task_stats(values, prefix))
        _add_secondary_binary_columns(row, "image")
        _add_secondary_binary_columns(row, "report")
        rows.append(row)
    return pd.DataFrame(rows)


def rater_pairwise_agreement(ratings: pd.DataFrame) -> pd.DataFrame:
    cleaned = ratings.copy() if "probability" in ratings.columns else clean_rater_ratings(ratings)
    rows: list[dict[str, Any]] = []
    for task, task_df in cleaned.groupby("review_task", sort=True, observed=True):
        pivot = task_df.pivot(index="case_id", columns="rater_id", values="probability")
        for rater_a, rater_b in combinations(sorted(pivot.columns), 2):
            paired = pivot[[rater_a, rater_b]].dropna()
            if paired.empty:
                continue
            diff = paired[rater_a] - paired[rater_b]
            row = {
                "review_task": task,
                "rater_a": rater_a,
                "rater_b": rater_b,
                "n_cases": int(len(paired)),
                "mean_absolute_difference": float(diff.abs().mean()),
                "root_mean_squared_difference": float(np.sqrt(np.mean(np.square(diff)))),
                "mean_signed_difference": float(diff.mean()),
                "pearson_correlation": _corr_or_none(paired[rater_a], paired[rater_b], "pearson"),
                "spearman_correlation": _corr_or_none(paired[rater_a], paired[rater_b], "spearman"),
            }
            for threshold in (0.50, 0.67):
                suffix = f"{int(threshold * 100):03d}"
                a_binary = paired[rater_a] >= threshold
                b_binary = paired[rater_b] >= threshold
                row[f"binary_agreement_ge_{suffix}"] = float((a_binary == b_binary).mean())
                row[f"cohen_kappa_ge_{suffix}"] = cohen_kappa(a_binary, b_binary)
            rows.append(row)
    return pd.DataFrame(rows)


def rater_level_summary(ratings: pd.DataFrame) -> pd.DataFrame:
    cleaned = ratings.copy() if "probability" in ratings.columns else clean_rater_ratings(ratings)
    grouped = cleaned.groupby(["review_task", "rater_id"], sort=True, observed=True)["probability"]
    return (
        grouped.agg(
            n_ratings="size",
            mean_probability="mean",
            sd_probability=lambda values: float(np.std(values, ddof=0)),
            min_probability="min",
            max_probability="max",
        )
        .reset_index()
        .sort_values(["review_task", "rater_id"])
    )


def leave_one_rater_out_agreement(ratings: pd.DataFrame) -> pd.DataFrame:
    cleaned = ratings.copy() if "probability" in ratings.columns else clean_rater_ratings(ratings)
    rows: list[dict[str, Any]] = []
    for task, task_df in cleaned.groupby("review_task", sort=True, observed=True):
        pivot = task_df.pivot(index="case_id", columns="rater_id", values="probability")
        for rater in sorted(pivot.columns):
            own = pivot[rater]
            others = pivot.drop(columns=[rater])
            other_mean = others.mean(axis=1, skipna=True)
            paired = pd.DataFrame({"own": own, "other_mean": other_mean}).dropna()
            if paired.empty:
                continue
            diff = paired["own"] - paired["other_mean"]
            rows.append(
                {
                    "review_task": task,
                    "rater_id": rater,
                    "n_cases": int(len(paired)),
                    "leave_one_out_mae": float(diff.abs().mean()),
                    "leave_one_out_squared_error": float(np.square(diff).mean()),
                }
            )
    return pd.DataFrame(rows)


def annotation_agreement_summary(
    ratings: pd.DataFrame, reference: pd.DataFrame | None = None
) -> pd.DataFrame:
    cleaned = ratings.copy() if "probability" in ratings.columns else clean_rater_ratings(ratings)
    ref = reference if reference is not None else build_case_reference_standard(cleaned)
    rows: list[dict[str, Any]] = []
    for task, prefix in (("image_only", "image"), ("report_only", "report")):
        task_df = cleaned[cleaned["review_task"] == task]
        if task_df.empty:
            continue
        range_column = f"range_{prefix}_probability"
        sd_column = f"sd_{prefix}_probability"
        rows.append(
            {
                "review_task": task,
                "n_ratings": int(len(task_df)),
                "n_cases": int(task_df["case_id"].nunique()),
                "n_raters": int(task_df["rater_id"].nunique()),
                "mean_probability": float(task_df["probability"].mean()),
                "sd_probability": float(np.std(task_df["probability"], ddof=0)),
                "mean_case_sd": _safe_mean(ref.get(sd_column)),
                "mean_case_range": _safe_mean(ref.get(range_column)),
                "proportion_case_range_ge_025": _safe_proportion(ref.get(range_column), 0.25),
                "proportion_case_range_ge_050": _safe_proportion(ref.get(range_column), 0.50),
            }
        )
    return pd.DataFrame(rows)


def case_level_disagreement_flags(reference: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, row in reference.iterrows():
        output = {"case_id": row["case_id"]}
        for prefix in ("image", "report"):
            range_value = row.get(f"range_{prefix}_probability")
            output[f"{prefix}_range_probability"] = range_value
            output[f"{prefix}_range_ge_025"] = _ge_or_false(range_value, 0.25)
            output[f"{prefix}_range_ge_050"] = _ge_or_false(range_value, 0.50)
        output["any_range_ge_025"] = bool(
            output["image_range_ge_025"] or output["report_range_ge_025"]
        )
        output["any_range_ge_050"] = bool(
            output["image_range_ge_050"] or output["report_range_ge_050"]
        )
        rows.append(output)
    return pd.DataFrame(rows)


def render_annotation_summary_markdown(
    *,
    validation: RatingValidationResult,
    agreement_summary: pd.DataFrame,
    pairwise_agreement: pd.DataFrame,
    leave_one_out: pd.DataFrame,
) -> str:
    lines = [
        "# Probabilistic annotation summary",
        "",
        f"- Validation passed: {validation.passed}",
        f"- Input rows: {validation.n_rows:,}",
        f"- Completed rows: {validation.n_completed_rows:,}",
        f"- Validation issues: {len(validation.issues):,}",
        "",
        "## Agreement by task",
        "",
        _table_or_placeholder(agreement_summary),
        "",
        "## Pairwise rater agreement",
        "",
        _table_or_placeholder(pairwise_agreement),
        "",
        "## Leave-one-rater-out agreement",
        "",
        _table_or_placeholder(leave_one_out),
        "",
    ]
    return "\n".join(lines)


def write_rating_validation(result: RatingValidationResult, *, out_json: Path) -> None:
    ensure_parent_dir(out_json)
    out_json.write_text(json.dumps(result.to_dict(), indent=2) + "\n", encoding="utf-8")


def cohen_kappa(a: pd.Series | np.ndarray, b: pd.Series | np.ndarray) -> float | None:
    a_series = pd.Series(a).astype(bool)
    b_series = pd.Series(b).astype(bool)
    if len(a_series) == 0:
        return None
    observed = float((a_series == b_series).mean())
    p_a = float(a_series.mean())
    p_b = float(b_series.mean())
    expected = p_a * p_b + (1 - p_a) * (1 - p_b)
    if expected == 1:
        return 1.0 if observed == 1 else None
    return float((observed - expected) / (1 - expected))


def _case_task_stats(values: pd.Series, prefix: str) -> dict[str, Any]:
    n = int(values.notna().sum())
    output: dict[str, Any] = {f"n_{prefix}_raters": n}
    if n == 0:
        for statistic in ("mean", "sd", "min", "max", "range"):
            output[f"{statistic}_{prefix}_probability"] = pd.NA
        return output

    clean_values = values.dropna().astype(float)
    min_value = float(clean_values.min())
    max_value = float(clean_values.max())
    output.update(
        {
            f"mean_{prefix}_probability": float(clean_values.mean()),
            f"sd_{prefix}_probability": float(np.std(clean_values, ddof=0)),
            f"min_{prefix}_probability": min_value,
            f"max_{prefix}_probability": max_value,
            f"range_{prefix}_probability": float(max_value - min_value),
        }
    )
    return output


def _add_secondary_binary_columns(row: dict[str, Any], prefix: str) -> None:
    mean_value = row.get(f"mean_{prefix}_probability")
    if _is_missing(mean_value):
        row[f"{prefix}_label_ge_050"] = pd.NA
        row[f"{prefix}_label_ge_067"] = pd.NA
        row[f"{prefix}_uncertain_033_067"] = pd.NA
        return

    probability = float(mean_value)
    row[f"{prefix}_label_ge_050"] = probability >= 0.50
    row[f"{prefix}_label_ge_067"] = probability >= 0.67
    row[f"{prefix}_uncertain_033_067"] = 0.33 <= probability <= 0.67


def _normalized_review_status(df: pd.DataFrame) -> pd.Series:
    if "review_status" not in df.columns:
        return pd.Series([""] * len(df), index=df.index)
    return df["review_status"].fillna("").astype(str).str.strip().str.lower()


def _normalize_rating_key_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for column in RATING_KEY_COLUMNS:
        out[column] = out[column].map(_normalize_rating_key_value)
    return out


def _normalize_rating_key_value(value: Any) -> Any:
    if _is_blank(value):
        return pd.NA
    return str(value).strip()


def _is_blank(value: Any) -> bool:
    return value is None or pd.isna(value) or str(value).strip() == ""


def _is_missing(value: Any) -> bool:
    return value is None or value is pd.NA or pd.isna(value)


def _first_non_null(values: pd.Series) -> Any:
    non_null = values.dropna()
    if non_null.empty:
        return pd.NA
    return non_null.iloc[0]


def _corr_or_none(a: pd.Series, b: pd.Series, method: str) -> float | None:
    if len(a) < 2 or a.nunique(dropna=True) < 2 or b.nunique(dropna=True) < 2:
        return None
    value = a.corr(b, method=method)
    return None if pd.isna(value) else float(value)


def _safe_mean(values: pd.Series | None) -> float | None:
    if values is None:
        return None
    clean = pd.to_numeric(values, errors="coerce").dropna()
    return None if clean.empty else float(clean.mean())


def _safe_proportion(values: pd.Series | None, threshold: float) -> float | None:
    if values is None:
        return None
    clean = pd.to_numeric(values, errors="coerce").dropna()
    return None if clean.empty else float((clean >= threshold).mean())


def _ge_or_false(value: Any, threshold: float) -> bool:
    return False if _is_missing(value) else bool(float(value) >= threshold)


def _table_or_placeholder(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    return "```text\n" + df.to_string(index=False) + "\n```"


def _issue(check: str, severity: str, message: str) -> RatingValidationIssue:
    return RatingValidationIssue(check=check, severity=severity, message=message)
