from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
import pandas as pd

from .config import ensure_dir
from .probabilistic_metrics import (
    DEFAULT_THRESHOLD_GRID,
    brier_soft,
    calibration_bins,
    calibration_slope_intercept,
    expected_threshold_metrics,
    log_loss_soft,
    mae_soft,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

TARGET_COLUMNS = {
    "report": "mean_report_probability",
    "image": "mean_image_probability",
}
REQUIRED_PREDICTION_COLUMNS = ["case_id", "model_name", "prediction_score"]
TIMING_FIELDS = [
    "intubation_time",
    "cxr_time",
    "report_available_time",
    "prediction_time",
    "alert_time",
]


@dataclass(frozen=True)
class ProbabilisticBenchmarkResult:
    join_audit: dict[str, Any]
    metrics: pd.DataFrame
    threshold_metrics: pd.DataFrame
    calibration: pd.DataFrame
    workflow_timing: pd.DataFrame
    alert_burden_by_week: pd.DataFrame


def normalize_prediction_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    assert_no_unsafe_columns(df)
    out = df.copy()
    if "case_id" not in out.columns and {"subject_id", "study_id"}.issubset(out.columns):
        subject_id = _normalize_required_identifier(out["subject_id"], "subject_id")
        study_id = _normalize_required_identifier(out["study_id"], "study_id")
        out["case_id"] = subject_id + "_" + study_id
    if "model_name" not in out.columns and "model" in out.columns:
        model = _normalize_required_identifier(out["model"], "model")
        if "task" in out.columns:
            task = _normalize_required_identifier(out["task"], "task")
            out["model_name"] = task + "__" + model
        else:
            out["model_name"] = model

    missing = [column for column in REQUIRED_PREDICTION_COLUMNS if column not in out.columns]
    if missing:
        raise ValueError(f"Missing required prediction columns: {missing}")

    out["case_id"] = _normalize_required_identifier(out["case_id"], "case_id")
    out["model_name"] = _normalize_required_identifier(out["model_name"], "model_name")
    out["prediction_score"] = pd.to_numeric(out["prediction_score"], errors="coerce")
    if out["prediction_score"].isna().any():
        raise ValueError("prediction_score must be non-null")
    if (~out["prediction_score"].between(0, 1)).any():
        raise ValueError("prediction_score values must be in [0, 1]")

    duplicate_mask = out.duplicated(["case_id", "model_name"], keep=False)
    if duplicate_mask.any():
        raise ValueError(
            f"{int(duplicate_mask.sum())} prediction rows duplicate case_id/model_name"
        )
    return drop_unsafe_columns(out).reset_index(drop=True)


def normalize_reference_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    assert_no_unsafe_columns(df)
    if "case_id" not in df.columns:
        raise ValueError("Reference table must contain case_id")
    if not any(column in df.columns for column in TARGET_COLUMNS.values()):
        raise ValueError("Reference table must include report or image mean probability")

    out = df.copy()
    out["case_id"] = _normalize_required_identifier(out["case_id"], "case_id")
    if out.duplicated(["case_id"], keep=False).any():
        raise ValueError("Reference table has duplicate case_id rows")

    for column in TARGET_COLUMNS.values():
        if column in out.columns:
            out[column] = _parse_probability_column(out[column], column)
    return drop_unsafe_columns(out).reset_index(drop=True)


def run_probabilistic_benchmark(
    reference: pd.DataFrame,
    predictions: pd.DataFrame,
    *,
    thresholds: list[float] | tuple[float, ...] = DEFAULT_THRESHOLD_GRID,
    n_bins: int = 10,
    calibration_strategy: str = "quantile",
) -> ProbabilisticBenchmarkResult:
    reference_df = normalize_reference_dataframe(reference)
    prediction_df = normalize_prediction_dataframe(predictions)
    joined = prediction_df.merge(reference_df, on="case_id", how="inner")
    join_audit = make_join_audit(reference_df, prediction_df, joined)
    targets = available_targets(reference_df)
    if not targets:
        raise ValueError("No non-null report or image targets are available")
    if joined.empty:
        raise ValueError("No prediction rows matched reference rows by case_id")

    metric_rows: list[dict[str, Any]] = []
    threshold_rows: list[dict[str, Any]] = []
    calibration_rows: list[pd.DataFrame] = []
    for model_name, model_df in joined.groupby("model_name", sort=True, observed=True):
        for target_type in targets:
            target_column = TARGET_COLUMNS[target_type]
            target_df = model_df[model_df[target_column].notna()].copy()
            if target_df.empty:
                continue
            slope = calibration_slope_intercept(
                target_df[target_column],
                target_df["prediction_score"],
            )
            metric_rows.append(
                {
                    "model_name": model_name,
                    "target_type": target_type,
                    "n": int(len(target_df)),
                    "mean_target_probability": float(target_df[target_column].mean()),
                    "mean_prediction_score": float(target_df["prediction_score"].mean()),
                    "brier_score": brier_soft(
                        target_df[target_column], target_df["prediction_score"]
                    ),
                    "mean_absolute_error": mae_soft(
                        target_df[target_column], target_df["prediction_score"]
                    ),
                    "soft_log_loss": log_loss_soft(
                        target_df[target_column], target_df["prediction_score"]
                    ),
                    "calibration_intercept": slope["intercept"],
                    "calibration_slope": slope["slope"],
                    "calibration_r_squared": slope["r_squared"],
                }
            )
            for threshold in thresholds:
                threshold_rows.append(
                    {
                        "model_name": model_name,
                        "target_type": target_type,
                        **expected_threshold_metrics(
                            target_df[target_column],
                            target_df["prediction_score"],
                            threshold=threshold,
                        ),
                    }
                )
            bins = calibration_bins(
                target_df[target_column],
                target_df["prediction_score"],
                n_bins=n_bins,
                strategy=calibration_strategy,
            )
            bins.insert(0, "target_type", target_type)
            bins.insert(0, "model_name", model_name)
            calibration_rows.append(bins)

    if not metric_rows:
        raise ValueError("No benchmark metric rows could be produced for available targets")

    calibration = (
        pd.concat(calibration_rows, ignore_index=True)
        if calibration_rows
        else pd.DataFrame(
            columns=[
                "model_name",
                "target_type",
                "bin_id",
                "n",
                "score_min",
                "score_max",
                "mean_prediction_score",
                "mean_target_probability",
                "absolute_calibration_error",
            ]
        )
    )
    return ProbabilisticBenchmarkResult(
        join_audit=join_audit,
        metrics=pd.DataFrame(metric_rows),
        threshold_metrics=pd.DataFrame(threshold_rows),
        calibration=calibration,
        workflow_timing=workflow_timing_summary(joined),
        alert_burden_by_week=alert_burden_by_week(joined, thresholds=thresholds),
    )


def write_probabilistic_benchmark_outputs(
    result: ProbabilisticBenchmarkResult,
    *,
    out_dir: Path,
) -> None:
    ensure_dir(out_dir)
    result.metrics.to_csv(out_dir / "probabilistic_metrics.csv", index=False)
    (out_dir / "probabilistic_metrics.json").write_text(
        json.dumps(json.loads(result.metrics.to_json(orient="records")), indent=2) + "\n",
        encoding="utf-8",
    )
    result.threshold_metrics.to_csv(out_dir / "expected_threshold_metrics.csv", index=False)
    (out_dir / "benchmark_join_audit.json").write_text(
        json.dumps(result.join_audit, indent=2) + "\n",
        encoding="utf-8",
    )

    for target_type in sorted(result.calibration["target_type"].dropna().unique().tolist()):
        target_bins = result.calibration[result.calibration["target_type"] == target_type]
        target_bins.to_csv(out_dir / f"calibration_bins_{target_type}.csv", index=False)
        plot_calibration_curve(
            target_bins,
            out_png=out_dir / f"calibration_curve_{target_type}.png",
            title=f"Calibration vs {target_type} reference",
        )

    if not result.workflow_timing.empty:
        result.workflow_timing.to_csv(out_dir / "workflow_timing_summary.csv", index=False)
    if not result.alert_burden_by_week.empty:
        result.alert_burden_by_week.to_csv(out_dir / "alert_burden_by_week.csv", index=False)
    (out_dir / "benchmark_summary.md").write_text(
        render_benchmark_summary(result),
        encoding="utf-8",
    )


def make_join_audit(
    reference: pd.DataFrame, predictions: pd.DataFrame, joined: pd.DataFrame
) -> dict[str, Any]:
    reference_cases = set(reference["case_id"])
    prediction_cases = set(predictions["case_id"])
    return {
        "reference_rows": int(len(reference)),
        "prediction_rows": int(len(predictions)),
        "joined_rows": int(len(joined)),
        "reference_cases": int(len(reference_cases)),
        "prediction_cases": int(len(prediction_cases)),
        "cases_missing_predictions": int(len(reference_cases - prediction_cases)),
        "prediction_cases_without_reference": int(len(prediction_cases - reference_cases)),
        "models": sorted(predictions["model_name"].dropna().unique().tolist()),
        "targets_available": available_targets(reference),
    }


def available_targets(reference: pd.DataFrame) -> list[str]:
    targets: list[str] = []
    for target_type, column in TARGET_COLUMNS.items():
        if column in reference.columns and reference[column].notna().any():
            targets.append(target_type)
    return targets


def _parse_probability_column(series: pd.Series, column: str) -> pd.Series:
    blank_mask = series.map(_is_blank_value)
    parsed = pd.to_numeric(series.mask(blank_mask), errors="coerce")
    malformed = ~blank_mask & parsed.isna()
    if malformed.any():
        examples = series[malformed].astype(str).head(5).tolist()
        raise ValueError(f"{column} contains non-numeric probability values: {examples}")
    invalid = parsed.notna() & ~parsed.between(0, 1)
    if invalid.any():
        raise ValueError(f"{column} values must be in [0, 1]")
    return parsed


def _is_blank_value(value: Any) -> bool:
    return value is None or pd.isna(value) or str(value).strip() == ""


def _normalize_required_identifier(series: pd.Series, column: str) -> pd.Series:
    missing = series.isna()
    if missing.any():
        raise ValueError(f"{column} must be non-null")
    normalized = series.astype(str).str.strip()
    blank = normalized == ""
    if blank.any():
        raise ValueError(f"{column} must be non-blank")
    return normalized


def assert_no_unsafe_columns(df: pd.DataFrame) -> None:
    unsafe = unsafe_output_columns(df)
    if unsafe:
        raise ValueError(f"Unsafe text/path columns are not allowed: {unsafe}")


def drop_unsafe_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=unsafe_output_columns(df), errors="ignore")


def unsafe_output_columns(df: pd.DataFrame) -> list[str]:
    unsafe: list[str] = []
    for column in df.columns:
        lowered = column.lower()
        if "text" in lowered:
            unsafe.append(column)
        elif lowered.endswith("_path") or lowered.endswith("_file") or lowered.endswith("_uri"):
            unsafe.append(column)
        elif lowered in {"report_text", "findings", "impression", "image_url"}:
            unsafe.append(column)
    return unsafe


def workflow_timing_summary(joined: pd.DataFrame) -> pd.DataFrame:
    available = [field for field in TIMING_FIELDS if field in joined.columns]
    if len(available) < 2:
        return pd.DataFrame()

    times = joined[available].apply(pd.to_datetime, errors="coerce")
    pairs = [
        ("cxr_to_report_hours", "cxr_time", "report_available_time"),
        ("report_to_prediction_hours", "report_available_time", "prediction_time"),
        ("intubation_to_cxr_hours", "intubation_time", "cxr_time"),
        ("intubation_to_alert_hours", "intubation_time", "alert_time"),
    ]
    rows: list[dict[str, Any]] = []
    for metric, start_column, end_column in pairs:
        if start_column not in times.columns or end_column not in times.columns:
            continue
        hours = (times[end_column] - times[start_column]).dt.total_seconds() / 3600
        hours = hours.dropna()
        if hours.empty:
            continue
        rows.append(
            {
                "metric": metric,
                "n": int(len(hours)),
                "mean_hours": float(hours.mean()),
                "median_hours": float(hours.median()),
                "p25_hours": float(hours.quantile(0.25)),
                "p75_hours": float(hours.quantile(0.75)),
            }
        )
    return pd.DataFrame(rows)


def alert_burden_by_week(
    joined: pd.DataFrame,
    *,
    thresholds: list[float] | tuple[float, ...],
) -> pd.DataFrame:
    date_column = next(
        (column for column in ("prediction_time", "alert_time", "cxr_time") if column in joined),
        None,
    )
    if date_column is None:
        return pd.DataFrame()

    dates = pd.to_datetime(joined[date_column], errors="coerce")
    frame = joined.copy()
    frame["week_start"] = dates.dt.to_period("W").dt.start_time
    frame = frame[frame["week_start"].notna()]
    if frame.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for model_name, model_df in frame.groupby("model_name", sort=True, observed=True):
        for threshold in thresholds:
            flagged = model_df["prediction_score"] >= threshold
            for week_start, week_df in model_df.assign(flagged=flagged).groupby(
                "week_start", sort=True, observed=True
            ):
                alerts = int(week_df["flagged"].sum())
                evaluated = int(len(week_df))
                rows.append(
                    {
                        "model_name": model_name,
                        "threshold": float(threshold),
                        "week_start": week_start.date().isoformat(),
                        "evaluated_cxrs": evaluated,
                        "alerts": alerts,
                        "alert_burden": alerts / evaluated if evaluated else None,
                        "alerts_per_100_cxrs": 100 * alerts / evaluated if evaluated else None,
                    }
                )
    return pd.DataFrame(rows)


def plot_calibration_curve(calibration: pd.DataFrame, *, out_png: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    for model_name, model_df in calibration.groupby("model_name", sort=True, observed=True):
        ax.plot(
            model_df["mean_prediction_score"],
            model_df["mean_target_probability"],
            marker="o",
            label=str(model_name),
        )
    ax.plot([0, 1], [0, 1], linestyle="--", color="black", linewidth=1, label="Perfect")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Mean prediction score")
    ax.set_ylabel("Mean target probability")
    ax.set_title(title)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def render_benchmark_summary(result: ProbabilisticBenchmarkResult) -> str:
    metric_table = (
        "No metric rows."
        if result.metrics.empty
        else result.metrics.to_string(index=False, max_cols=20)
    )
    lines = [
        "# Probabilistic benchmark summary",
        "",
        "These metrics compare model probabilities with continuous human reference targets.",
        "Report-only probability is the primary report-NLP target; image-only probability is the",
        "end-to-end clinical/image target when available.",
        "",
        "## Join audit",
        "",
        "```json",
        json.dumps(result.join_audit, indent=2),
        "```",
        "",
        "## Metrics",
        "",
        "```text",
        metric_table,
        "```",
        "",
    ]
    return "\n".join(lines)
