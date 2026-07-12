from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from ards_cxr_benchmark.comparator_snapshot import (
    render_comparator_snapshot,
    write_comparator_snapshot,
)


def test_snapshot_is_aggregate_and_labels_knox_clamp(tmp_path: Path) -> None:
    catalog = _catalog()
    metrics = _metrics()
    statuses = [
        {"name": "clamp_legacy", "status": "available", "reason": "predictions present"},
        {
            "name": "uw_hanso_bilateral_infiltrates",
            "status": "blocked_missing_model_artifacts",
            "reason": "weights unavailable",
        },
    ]

    rendered = render_comparator_snapshot(catalog, metrics, statuses)

    assert "Dan Knox legacy CLAMP" in rendered
    assert "blocked_missing_model_artifacts" in rendered
    assert "report_text" not in rendered
    assert "/Users/" not in rendered


def test_snapshot_writer_requires_core_models(tmp_path: Path) -> None:
    catalog_path = tmp_path / "catalog.csv"
    metrics_path = tmp_path / "metrics.csv"
    status_path = tmp_path / "status.json"
    _catalog().iloc[:-1].to_csv(catalog_path, index=False)
    _metrics().to_csv(metrics_path, index=False)
    status_path.write_text("[]\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Missing required comparator models"):
        write_comparator_snapshot(
            catalog_path=catalog_path,
            metrics_path=metrics_path,
            status_path=status_path,
            output_path=tmp_path / "snapshot.md",
        )


def test_snapshot_allows_additional_available_comparators() -> None:
    catalog = pd.concat(
        [
            _catalog(),
            pd.DataFrame(
                [
                    {
                        "model_name": "uw_hanso_bilateral_infiltrates",
                        "comparison_role": "external_comparator",
                        "prediction_rows": 227835,
                        "unique_cases": 227835,
                        "validation_rows": 19754,
                        "test_rows": 20702,
                        "cases_without_reference": 0,
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    rendered = render_comparator_snapshot(catalog, _metrics(), [])

    assert "Available prediction models/controls: **11**" in rendered
    assert "UW HANSO" in rendered


def _catalog() -> pd.DataFrame:
    models = [
        ("clamp_legacy", "legacy_teacher"),
        ("clamp_python_compatibility", "compatibility_mirror"),
        ("amaral_xgboost_bilateral_infiltrates", "external_comparator"),
        ("amaral_xgboost_bilateral_infiltrates_raw_text_direct", "external_comparator"),
        ("silver_sensitive_silver_score_rule", "silver_derived_control"),
        ("silver_sensitive_structured_logreg", "silver_derived_control"),
        ("silver_sensitive_tfidf_logreg", "trained_silver_baseline"),
        ("silver_strict_silver_score_rule", "silver_derived_control"),
        ("silver_strict_structured_logreg", "silver_derived_control"),
        ("silver_strict_tfidf_logreg", "trained_silver_baseline"),
    ]
    return pd.DataFrame(
        [
            {
                "model_name": name,
                "comparison_role": role,
                "prediction_rows": 227835,
                "unique_cases": 227835,
                "validation_rows": 19754,
                "test_rows": 20702,
                "cases_without_reference": 0,
            }
            for name, role in models
        ]
    )


def _metrics() -> pd.DataFrame:
    rows = []
    for _, model in _catalog().iterrows():
        rows.append(
            {
                "model_name": model["model_name"],
                "comparison_role": model["comparison_role"],
                "task": "strict",
                "split": "test",
                "n": 20702,
                "prevalence": 0.188,
                "roc_auc": 0.85,
                "average_precision": 0.7,
                "f1": 0.7,
                "sensitivity": 0.75,
                "specificity": 0.9,
                "brier": 0.1,
            }
        )
    return pd.DataFrame(rows)
