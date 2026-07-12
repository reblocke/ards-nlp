from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import ensure_dir
from .modeling import TASK_LABEL_COLUMNS, binary_metrics

BENCHMARK_INTERPRETATION = (
    "These are CLAMP-vs-automated-silver-label diagnostics. They are not clinical accuracy "
    "estimates and not human gold-standard performance."
)
METRIC_COLUMNS = [
    "task",
    "model_name",
    "stratum",
    "stratum_value",
    "n",
    "positives",
    "prevalence",
    "clamp_positive_rate",
    "threshold",
    "accuracy",
    "precision",
    "recall",
    "specificity",
    "f1",
    "brier",
    "roc_auc",
    "average_precision",
]
MODEL_EXTRACT_COLUMNS = [
    "subject_id",
    "study_id",
    "strict_bilateral_opacity_label",
    "sensitive_bilateral_opacity_label",
    "silver_label_source",
    "manual_review_priority",
    "qa_flags",
]


def benchmark_clamp_against_silver(
    *,
    predictions: pd.DataFrame,
    model_extract: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    joined = join_predictions_to_model_extract(predictions, model_extract)
    if joined.empty:
        raise ValueError("No CLAMP prediction rows joined to model_development_extract")

    metric_rows: list[dict[str, Any]] = []
    strata_rows: list[dict[str, Any]] = []
    for task, label_column in TASK_LABEL_COLUMNS.items():
        if label_column not in joined.columns:
            continue
        task_df = joined[joined[label_column].notna() & joined["prediction_score"].notna()].copy()
        if task_df.empty:
            continue
        overall = _metric_row(task_df, task=task, label_column=label_column, stratum="overall")
        metric_rows.append(overall)
        for stratum, stratum_df in _iter_strata(task_df):
            strata_rows.append(
                _metric_row(
                    stratum_df,
                    task=task,
                    label_column=label_column,
                    stratum=stratum[0],
                    stratum_value=stratum[1],
                )
            )
    if not metric_rows:
        raise ValueError("No CLAMP-vs-silver metric rows could be produced")

    metrics = pd.DataFrame(metric_rows).reindex(columns=METRIC_COLUMNS)
    strata = pd.DataFrame(strata_rows).reindex(columns=METRIC_COLUMNS)
    summary = {
        "interpretation": BENCHMARK_INTERPRETATION,
        "prediction_rows": int(len(predictions)),
        "evaluable_prediction_rows": int(predictions["prediction_score"].notna().sum()),
        "model_extract_rows": int(len(model_extract)),
        "joined_rows": int(len(joined)),
        "metric_rows": int(len(metrics)),
        "strata_rows": int(len(strata)),
        "tasks": sorted(metrics["task"].dropna().unique().tolist()),
    }
    return metrics, strata, summary


def join_predictions_to_model_extract(
    predictions: pd.DataFrame,
    model_extract: pd.DataFrame,
) -> pd.DataFrame:
    required_prediction = {"subject_id", "study_id", "prediction_score", "model_name"}
    missing_prediction = sorted(required_prediction - set(predictions.columns))
    if missing_prediction:
        raise ValueError(f"Missing prediction columns: {missing_prediction}")
    required_extract = {"subject_id", "study_id", *TASK_LABEL_COLUMNS.values()}
    missing_extract = sorted(required_extract - set(model_extract.columns))
    if missing_extract:
        raise ValueError(f"Missing model extract columns: {missing_extract}")

    pred = predictions.copy()
    extract = model_extract.copy()
    pred["subject_id_join"] = _join_id(
        pred["subject_id"], table_name="predictions", column_name="subject_id"
    )
    pred["study_id_join"] = _join_id(
        pred["study_id"], table_name="predictions", column_name="study_id"
    )
    extract["subject_id_join"] = _join_id(
        extract["subject_id"], table_name="model_extract", column_name="subject_id"
    )
    extract["study_id_join"] = _join_id(
        extract["study_id"], table_name="model_extract", column_name="study_id"
    )
    joined = pred.merge(
        extract,
        on=["subject_id_join", "study_id_join"],
        how="inner",
        suffixes=("", "_silver"),
    )
    if "qa_flags" in joined.columns:
        joined["qa_flags_present"] = joined["qa_flags"].map(_qa_flags_present)
    return joined


def write_clamp_teacher_benchmark_outputs(
    *,
    metrics: pd.DataFrame,
    strata: pd.DataFrame,
    summary: dict[str, Any],
    out_dir: Path,
) -> None:
    ensure_dir(out_dir)
    metrics.to_csv(out_dir / "clamp_vs_silver_metrics.csv", index=False)
    (out_dir / "clamp_vs_silver_metrics.json").write_text(
        json.dumps(json.loads(metrics.to_json(orient="records")), indent=2) + "\n",
        encoding="utf-8",
    )
    strata.to_csv(out_dir / "clamp_vs_silver_strata.csv", index=False)
    (out_dir / "clamp_vs_silver_summary.md").write_text(
        render_clamp_teacher_benchmark_summary(summary, metrics),
        encoding="utf-8",
    )


def render_clamp_teacher_benchmark_summary(
    summary: dict[str, Any],
    metrics: pd.DataFrame,
) -> str:
    lines = [
        "# ARDS CLAMP vs Silver Benchmark",
        "",
        BENCHMARK_INTERPRETATION,
        "",
        f"- Prediction rows: {summary['prediction_rows']:,}",
        f"- Evaluable prediction rows: {summary['evaluable_prediction_rows']:,}",
        f"- Model extract rows: {summary['model_extract_rows']:,}",
        f"- Joined rows: {summary['joined_rows']:,}",
        "",
        "## Overall Metrics",
        "",
    ]
    for _, row in metrics.iterrows():
        lines.extend(
            [
                f"### {row['task']}",
                "",
                f"- n: {int(row['n']):,}",
                f"- Silver prevalence: {_fmt(row['prevalence'])}",
                f"- CLAMP positive rate: {_fmt(row['clamp_positive_rate'])}",
                f"- Accuracy: {_fmt(row['accuracy'])}",
                f"- Precision: {_fmt(row['precision'])}",
                f"- Recall: {_fmt(row['recall'])}",
                f"- Specificity: {_fmt(row['specificity'])}",
                f"- F1: {_fmt(row['f1'])}",
                "",
            ]
        )
    return "\n".join(lines)


def model_extract_sql(source_table: str) -> str:
    source_table = normalize_source_table(source_table)
    columns = ",\n  ".join(f"`{column}`" for column in MODEL_EXTRACT_COLUMNS)
    return f"SELECT\n  {columns}\nFROM `{source_table}`"


def normalize_source_table(source_table: str | None) -> str:
    normalized = str(source_table or "").strip()
    if "`" in normalized or not normalized:
        raise ValueError(f"Unsafe BigQuery table reference: {normalized!r}")
    return normalized


def _metric_row(
    df: pd.DataFrame,
    *,
    task: str,
    label_column: str,
    stratum: str,
    stratum_value: object = "all",
) -> dict[str, Any]:
    metrics = binary_metrics(df[label_column].astype(int), df["prediction_score"].astype(float))
    return {
        "task": task,
        "model_name": "clamp_legacy",
        "stratum": stratum,
        "stratum_value": str(stratum_value),
        **metrics,
        "clamp_positive_rate": float(df["prediction_label"].astype(int).mean()),
    }


def _iter_strata(df: pd.DataFrame):
    for column in ["silver_label_source", "manual_review_priority", "qa_flags_present"]:
        if column not in df.columns:
            continue
        for value, group in df.groupby(column, dropna=False, sort=True):
            if group.empty:
                continue
            yield (column, value), group


def _join_id(series: pd.Series, *, table_name: str, column_name: str) -> pd.Series:
    if series.isna().any():
        raise ValueError(f"{table_name}.{column_name} contains missing values")
    clean = series.astype(str).str.strip()
    if (clean == "").any():
        raise ValueError(f"{table_name}.{column_name} contains blank values")
    numeric = pd.to_numeric(clean, errors="coerce")
    if numeric.isna().any():
        raise ValueError(f"{table_name}.{column_name} contains non-numeric values")
    numeric_float = numeric.astype(float)
    if ((numeric_float % 1) != 0).any():
        raise ValueError(f"{table_name}.{column_name} contains non-integer values")
    return numeric_float.astype("int64").astype(str)


def _qa_flags_present(value: object) -> bool:
    if value is None:
        return False
    if isinstance(value, np.ndarray):
        return any(_qa_flags_present(item) for item in value.flat)
    if isinstance(value, (list, tuple, set, frozenset)):
        return any(_qa_flags_present(item) for item in value)
    if pd.isna(value):
        return False
    clean = str(value).strip().lower()
    return clean not in {"", "none", "[]"}


def _fmt(value: object) -> str:
    if value is None or pd.isna(value):
        return "NA"
    return f"{float(value):.3f}"
