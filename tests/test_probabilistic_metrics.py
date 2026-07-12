from __future__ import annotations

import pandas as pd
import pytest

from ards_cxr_benchmark.probabilistic_metrics import (
    bootstrap_metric_ci,
    brier_soft,
    calibration_bins,
    expected_confusion_counts,
    expected_threshold_metrics,
    log_loss_soft,
    mae_soft,
)


def test_soft_brier_mae_and_log_loss_match_hand_calculation() -> None:
    y = [0.0, 0.5, 1.0]
    score = [0.2, 0.5, 0.8]

    assert brier_soft(y, score) == pytest.approx((0.04 + 0 + 0.04) / 3)
    assert mae_soft(y, score) == pytest.approx((0.2 + 0 + 0.2) / 3)
    assert log_loss_soft([1.0], [0.8]) == pytest.approx(0.2231435513)


def test_expected_confusion_counts_and_threshold_metrics_match_soft_labels() -> None:
    y = [0.2, 0.8]
    score = [0.7, 0.4]

    counts = expected_confusion_counts(y, score, threshold=0.5)
    metrics = expected_threshold_metrics(y, score, threshold=0.5)

    assert counts == pytest.approx({"tp": 0.2, "fp": 0.8, "fn": 0.8, "tn": 0.2})
    assert metrics["sensitivity"] == pytest.approx(0.2)
    assert metrics["specificity"] == pytest.approx(0.2)
    assert metrics["alert_burden"] == pytest.approx(0.5)


def test_calibration_bins_handles_duplicate_scores_and_small_n() -> None:
    bins = calibration_bins([0.1, 0.2, 0.8], [0.5, 0.5, 0.5], n_bins=10)

    assert len(bins) == 1
    assert bins.loc[0, "n"] == 3
    assert bins.loc[0, "mean_prediction_score"] == pytest.approx(0.5)


def test_bootstrap_metric_ci_is_deterministic_with_fixed_seed() -> None:
    df = pd.DataFrame(
        {
            "cluster": ["a", "a", "b", "b"],
            "value": [0.0, 1.0, 2.0, 3.0],
        }
    )

    def metric(sample: pd.DataFrame) -> float:
        return float(sample["value"].mean())

    first = bootstrap_metric_ci(df, metric, cluster_col="cluster", n_boot=25, seed=123)
    second = bootstrap_metric_ci(df, metric, cluster_col="cluster", n_boot=25, seed=123)

    assert first == second
    assert first["estimate"] == pytest.approx(1.5)
