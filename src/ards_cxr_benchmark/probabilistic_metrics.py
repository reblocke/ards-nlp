from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from typing import Any

import numpy as np
import pandas as pd

DEFAULT_THRESHOLD_GRID = (0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90)


def brier_soft(
    y_prob: Iterable[float],
    y_score: Iterable[float],
    sample_weight: Iterable[float] | None = None,
) -> float:
    frame = _metric_frame(y_prob, y_score, sample_weight)
    error = np.square(frame["y_score"] - frame["y_prob"])
    return _weighted_mean(error, frame.get("sample_weight"))


def mae_soft(
    y_prob: Iterable[float],
    y_score: Iterable[float],
    sample_weight: Iterable[float] | None = None,
) -> float:
    frame = _metric_frame(y_prob, y_score, sample_weight)
    error = (frame["y_score"] - frame["y_prob"]).abs()
    return _weighted_mean(error, frame.get("sample_weight"))


def log_loss_soft(
    y_prob: Iterable[float],
    y_score: Iterable[float],
    sample_weight: Iterable[float] | None = None,
    eps: float = 1e-6,
) -> float:
    frame = _metric_frame(y_prob, y_score, sample_weight)
    score = frame["y_score"].clip(eps, 1 - eps)
    loss = -(frame["y_prob"] * np.log(score) + (1 - frame["y_prob"]) * np.log(1 - score))
    return _weighted_mean(loss, frame.get("sample_weight"))


def calibration_bins(
    y_prob: Iterable[float],
    y_score: Iterable[float],
    *,
    n_bins: int = 10,
    strategy: str = "quantile",
) -> pd.DataFrame:
    if n_bins <= 0:
        raise ValueError("n_bins must be positive")
    if strategy not in {"quantile", "uniform"}:
        raise ValueError("strategy must be 'quantile' or 'uniform'")

    frame = _metric_frame(y_prob, y_score)
    if strategy == "quantile":
        frame["bin_id"] = _quantile_bin_ids(frame["y_score"], n_bins)
    else:
        frame["bin_id"] = _uniform_bin_ids(frame["y_score"], n_bins)

    rows: list[dict[str, Any]] = []
    for raw_bin_id, group in frame.groupby("bin_id", sort=True, observed=True):
        mean_prediction = float(group["y_score"].mean())
        mean_target = float(group["y_prob"].mean())
        rows.append(
            {
                "bin_id": int(raw_bin_id) + 1,
                "n": int(len(group)),
                "score_min": float(group["y_score"].min()),
                "score_max": float(group["y_score"].max()),
                "mean_prediction_score": mean_prediction,
                "mean_target_probability": mean_target,
                "absolute_calibration_error": abs(mean_prediction - mean_target),
            }
        )
    return pd.DataFrame(rows)


def calibration_slope_intercept(
    y_prob: Iterable[float],
    y_score: Iterable[float],
    *,
    eps: float = 1e-6,
) -> dict[str, float | int | None]:
    frame = _metric_frame(y_prob, y_score)
    if len(frame) < 2 or frame["y_score"].nunique() < 2:
        return {"n": int(len(frame)), "intercept": None, "slope": None, "r_squared": None}

    x = _logit(frame["y_score"].clip(eps, 1 - eps).to_numpy())
    y = _logit(frame["y_prob"].clip(eps, 1 - eps).to_numpy())
    design = np.column_stack([np.ones(len(x)), x])
    intercept, slope = np.linalg.lstsq(design, y, rcond=None)[0]
    predicted = intercept + slope * x
    total_ss = float(np.square(y - y.mean()).sum())
    residual_ss = float(np.square(y - predicted).sum())
    r_squared = None if total_ss == 0 else 1 - residual_ss / total_ss
    return {
        "n": int(len(frame)),
        "intercept": float(intercept),
        "slope": float(slope),
        "r_squared": None if r_squared is None else float(r_squared),
    }


def expected_confusion_counts(
    y_prob: Iterable[float],
    y_score: Iterable[float],
    *,
    threshold: float,
) -> dict[str, float]:
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be in [0, 1]")
    frame = _metric_frame(y_prob, y_score)
    predicted_positive = (frame["y_score"] >= threshold).astype(float)
    target = frame["y_prob"].astype(float)
    tp = float((predicted_positive * target).sum())
    fp = float((predicted_positive * (1 - target)).sum())
    fn = float(((1 - predicted_positive) * target).sum())
    tn = float(((1 - predicted_positive) * (1 - target)).sum())
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def expected_threshold_metrics(
    y_prob: Iterable[float],
    y_score: Iterable[float],
    *,
    threshold: float,
) -> dict[str, float | None]:
    counts = expected_confusion_counts(y_prob, y_score, threshold=threshold)
    tp = counts["tp"]
    fp = counts["fp"]
    fn = counts["fn"]
    tn = counts["tn"]
    n = tp + fp + fn + tn
    return {
        "threshold": float(threshold),
        "n": float(n),
        **counts,
        "sensitivity": _safe_ratio(tp, tp + fn),
        "specificity": _safe_ratio(tn, tn + fp),
        "ppv": _safe_ratio(tp, tp + fp),
        "npv": _safe_ratio(tn, tn + fn),
        "alert_burden": _safe_ratio(tp + fp, n),
    }


def threshold_grid_metrics(
    y_prob: Iterable[float],
    y_score: Iterable[float],
    *,
    thresholds: Sequence[float] = DEFAULT_THRESHOLD_GRID,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            expected_threshold_metrics(y_prob, y_score, threshold=threshold)
            for threshold in thresholds
        ]
    )


def bootstrap_metric_ci(
    df: pd.DataFrame,
    metric_fn: Callable[[pd.DataFrame], float],
    *,
    cluster_col: str | None = None,
    n_boot: int = 2000,
    seed: int = 20260628,
) -> dict[str, float | int]:
    if n_boot <= 0:
        raise ValueError("n_boot must be positive")
    if df.empty:
        raise ValueError("Cannot bootstrap an empty DataFrame")
    if cluster_col is not None and cluster_col not in df.columns:
        raise KeyError(f"cluster_col {cluster_col!r} is not present")

    rng = np.random.default_rng(seed)
    values: list[float] = []
    if cluster_col is None:
        row_positions = np.arange(len(df))
        for _ in range(n_boot):
            sampled_positions = rng.choice(row_positions, size=len(row_positions), replace=True)
            values.append(float(metric_fn(df.iloc[sampled_positions].reset_index(drop=True))))
    else:
        clusters = np.array(sorted(df[cluster_col].dropna().unique().tolist()))
        if len(clusters) == 0:
            raise ValueError("cluster_col has no non-null clusters")
        grouped = {cluster: group for cluster, group in df.groupby(cluster_col, sort=False)}
        for _ in range(n_boot):
            sampled_clusters = rng.choice(clusters, size=len(clusters), replace=True)
            sample = pd.concat(
                [grouped[cluster] for cluster in sampled_clusters],
                ignore_index=True,
            )
            values.append(float(metric_fn(sample)))

    estimates = np.array(values, dtype=float)
    return {
        "estimate": float(metric_fn(df)),
        "ci_lower": float(np.percentile(estimates, 2.5)),
        "ci_upper": float(np.percentile(estimates, 97.5)),
        "n_boot": int(n_boot),
        "seed": int(seed),
    }


def _metric_frame(
    y_prob: Iterable[float],
    y_score: Iterable[float],
    sample_weight: Iterable[float] | None = None,
) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "y_prob": pd.to_numeric(pd.Series(list(y_prob)), errors="coerce"),
            "y_score": pd.to_numeric(pd.Series(list(y_score)), errors="coerce"),
        }
    )
    if sample_weight is not None:
        frame["sample_weight"] = pd.to_numeric(pd.Series(list(sample_weight)), errors="coerce")
    frame = frame.dropna()
    if frame.empty:
        raise ValueError("No complete y_prob/y_score rows are available")
    for column in ("y_prob", "y_score"):
        invalid = ~frame[column].between(0, 1)
        if invalid.any():
            raise ValueError(f"{column} values must be in [0, 1]")
    if "sample_weight" in frame.columns and (frame["sample_weight"] < 0).any():
        raise ValueError("sample_weight values must be non-negative")
    return frame


def _weighted_mean(values: pd.Series, weights: pd.Series | None = None) -> float:
    if weights is None:
        return float(values.mean())
    total_weight = float(weights.sum())
    if total_weight == 0:
        raise ValueError("sample_weight sum must be positive")
    return float((values * weights).sum() / total_weight)


def _quantile_bin_ids(scores: pd.Series, n_bins: int) -> pd.Series:
    if len(scores) == 1 or scores.nunique() == 1:
        return pd.Series([0] * len(scores), index=scores.index, dtype=int)
    bins = min(n_bins, len(scores), scores.nunique())
    try:
        result = pd.qcut(scores, q=bins, labels=False, duplicates="drop")
    except ValueError:
        return pd.Series([0] * len(scores), index=scores.index, dtype=int)
    return result.astype(int)


def _uniform_bin_ids(scores: pd.Series, n_bins: int) -> pd.Series:
    clipped = scores.clip(0, 1)
    result = pd.cut(
        clipped,
        bins=np.linspace(0, 1, n_bins + 1),
        include_lowest=True,
        labels=False,
    )
    return result.astype(int)


def _logit(values: np.ndarray) -> np.ndarray:
    return np.log(values / (1 - values))


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    return None if denominator == 0 else float(numerator / denominator)
