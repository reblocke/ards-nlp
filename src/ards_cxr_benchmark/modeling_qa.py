from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .config import ensure_parent_dir
from .modeling import TASK_LABEL_COLUMNS

EXPECTED_MODELS = ["silver_score_rule", "tfidf_logreg", "structured_logreg"]
EXPECTED_EVAL_SPLITS = ["validation", "test"]
REQUIRED_PREDICTION_COLUMNS = [
    "subject_id",
    "study_id",
    "split",
    "task",
    "model",
    "true_label",
    "prediction_score",
    "prediction_label",
]
REQUIRED_METRIC_COLUMNS = ["task", "model", "split", "n", "roc_auc", "average_precision"]
TEXT_LEAK_COLUMNS = {
    "report_text",
    "findings_text",
    "impression_text",
    "target_text_full_report",
    "target_text_impression_findings",
    "target_text_impression_fallback",
    "primary_target_text",
}


@dataclass(frozen=True)
class ModelingQAIssue:
    check: str
    severity: str
    message: str


@dataclass(frozen=True)
class ModelingQAResult:
    passed: bool
    summary: dict[str, Any]
    issues: list[ModelingQAIssue]


def evaluate_modeling_outputs(
    extract_df: pd.DataFrame,
    predictions_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    *,
    expected_models: list[str] | None = None,
    expected_eval_splits: list[str] | None = None,
) -> ModelingQAResult:
    models = expected_models or EXPECTED_MODELS
    eval_splits = expected_eval_splits or EXPECTED_EVAL_SPLITS
    issues: list[ModelingQAIssue] = []

    split_counts = _value_counts(extract_df, "split")
    for split in ["train", *eval_splits]:
        if split_counts.get(split, 0) == 0:
            issues.append(_issue("split_counts", "error", f"No rows found for split: {split}"))

    subject_overlap = _subject_overlap_count(extract_df)
    if subject_overlap:
        issues.append(
            _issue(
                "subject_overlap",
                "error",
                f"{subject_overlap} subjects appear in more than one split",
            )
        )

    missing_text = _missing_text_count(extract_df)
    if missing_text:
        issues.append(
            _issue("missing_text", "error", f"{missing_text} rows have missing primary_target_text")
        )

    missing_prediction_columns = [
        column for column in REQUIRED_PREDICTION_COLUMNS if column not in predictions_df.columns
    ]
    if missing_prediction_columns:
        issues.append(
            _issue(
                "prediction_schema",
                "error",
                f"Missing prediction columns: {missing_prediction_columns}",
            )
        )

    text_columns = [
        column
        for column in predictions_df.columns
        if column in TEXT_LEAK_COLUMNS or "text" in column.lower()
    ]
    if text_columns:
        issues.append(
            _issue(
                "prediction_text_leakage",
                "error",
                f"Prediction output has text columns: {text_columns}",
            )
        )

    missing_metric_columns = [
        column for column in REQUIRED_METRIC_COLUMNS if column not in metrics_df.columns
    ]
    if missing_metric_columns:
        issues.append(
            _issue("metric_schema", "error", f"Missing metric columns: {missing_metric_columns}")
        )

    eligible_rows: dict[str, int] = {}
    expected_prediction_rows = 0
    missing_metric_rows: list[str] = []
    missing_prediction_rows: list[str] = []
    for task, label_column in TASK_LABEL_COLUMNS.items():
        eligible = extract_df[extract_df[label_column].notna()].copy()
        eligible_rows[task] = int(len(eligible))
        if eligible.empty:
            issues.append(_issue("eligible_rows", "error", f"No eligible rows for task: {task}"))
            continue

        for split in eval_splits:
            eval_count = int((eligible["split"] == split).sum())
            expected_prediction_rows += eval_count * len(models)
            for model in models:
                if not _has_metric_row(metrics_df, task=task, split=split, model=model):
                    missing_metric_rows.append(f"{task}/{model}/{split}")
                if not missing_prediction_columns:
                    actual_predictions = len(
                        predictions_df[
                            (predictions_df["task"] == task)
                            & (predictions_df["model"] == model)
                            & (predictions_df["split"] == split)
                        ]
                    )
                    if actual_predictions != eval_count:
                        missing_prediction_rows.append(
                            f"{task}/{model}/{split}: expected {eval_count}, "
                            f"got {actual_predictions}"
                        )

    if missing_metric_rows:
        issues.append(_issue("metric_rows", "error", f"Missing metric rows: {missing_metric_rows}"))
    if missing_prediction_rows:
        issues.append(
            _issue(
                "prediction_rows", "error", f"Prediction row mismatches: {missing_prediction_rows}"
            )
        )

    summary: dict[str, Any] = {
        "n_extract_rows": int(len(extract_df)),
        "split_counts": split_counts,
        "subjects_with_split_overlap": int(subject_overlap),
        "missing_primary_target_text_rows": int(missing_text),
        "eligible_rows": eligible_rows,
        "n_prediction_rows": int(len(predictions_df)),
        "expected_prediction_rows": int(expected_prediction_rows),
        "n_metric_rows": int(len(metrics_df)),
        "expected_metric_rows": int(len(TASK_LABEL_COLUMNS) * len(models) * len(eval_splits)),
        "prediction_text_columns": text_columns,
    }
    return ModelingQAResult(
        passed=not any(issue.severity == "error" for issue in issues),
        summary=summary,
        issues=issues,
    )


def write_modeling_qa(result: ModelingQAResult, *, out_json: Path, out_md: Path) -> None:
    ensure_parent_dir(out_json)
    ensure_parent_dir(out_md)
    out_json.write_text(
        json.dumps(
            {
                "passed": result.passed,
                "summary": result.summary,
                "issues": [asdict(issue) for issue in result.issues],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    out_md.write_text(render_modeling_qa_markdown(result), encoding="utf-8")


def render_modeling_qa_markdown(result: ModelingQAResult) -> str:
    lines = [
        "# Modeling QA Summary",
        "",
        f"- Passed: `{result.passed}`",
        f"- Extract rows: {result.summary['n_extract_rows']:,}",
        f"- Split counts: `{json.dumps(result.summary['split_counts'], sort_keys=True)}`",
        f"- Subjects with split overlap: {result.summary['subjects_with_split_overlap']:,}",
        "- Missing primary target text rows: "
        f"{result.summary['missing_primary_target_text_rows']:,}",
        f"- Prediction rows: {result.summary['n_prediction_rows']:,}",
        f"- Expected prediction rows: {result.summary['expected_prediction_rows']:,}",
        f"- Metric rows: {result.summary['n_metric_rows']:,}",
        f"- Expected metric rows: {result.summary['expected_metric_rows']:,}",
        "",
        "## Eligible Rows",
        "",
    ]
    for task, n_rows in sorted(result.summary["eligible_rows"].items()):
        lines.append(f"- {task}: {n_rows:,}")
    lines.extend(["", "## Issues", ""])
    if not result.issues:
        lines.append("- None")
    else:
        for issue in result.issues:
            lines.append(f"- `{issue.severity}` `{issue.check}`: {issue.message}")
    lines.append("")
    return "\n".join(lines)


def _issue(check: str, severity: str, message: str) -> ModelingQAIssue:
    return ModelingQAIssue(check=check, severity=severity, message=message)


def _value_counts(df: pd.DataFrame, column: str) -> dict[str, int]:
    if column not in df.columns:
        return {}
    return {str(key): int(value) for key, value in df[column].value_counts().to_dict().items()}


def _subject_overlap_count(df: pd.DataFrame) -> int:
    if "subject_id" not in df.columns or "split" not in df.columns:
        return 0
    return int(df.groupby("subject_id", observed=True)["split"].nunique().gt(1).sum())


def _missing_text_count(df: pd.DataFrame) -> int:
    if "primary_target_text" not in df.columns:
        return len(df)
    return int(df["primary_target_text"].isna().sum())


def _has_metric_row(metrics_df: pd.DataFrame, *, task: str, split: str, model: str) -> bool:
    if any(column not in metrics_df.columns for column in ("task", "split", "model")):
        return False
    matched = metrics_df[
        (metrics_df["task"] == task)
        & (metrics_df["split"] == split)
        & (metrics_df["model"] == model)
    ]
    return len(matched) == 1
