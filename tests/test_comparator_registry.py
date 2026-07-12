from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from ards_cxr_benchmark.comparators.registry import (
    select_default_prediction_artifacts,
    validate_prediction_status,
)


def test_default_registry_rejects_blocked_stale_predictions(tmp_path: Path) -> None:
    derived = tmp_path / "data/derived/comparators"
    artifacts = tmp_path / "artifacts/comparators"
    derived.mkdir(parents=True)
    for name in [
        "clamp_legacy_predictions.parquet",
        "clamp_python_compatibility_predictions.parquet",
        "silver_baseline_predictions.parquet",
        "amaral_bilateral_infiltrates_predictions.parquet",
        "uw_hanso_predictions.parquet",
    ]:
        (derived / name).touch()
    _write_status(
        artifacts / "amaral/status.json",
        status="available",
        run_id="amaral-run",
    )
    _write_status(
        artifacts / "uw_hanso/status.json",
        status="blocked_missing_model_artifacts",
        run_id=None,
    )

    with pytest.raises(ValueError, match="Blocked comparator uw_hanso"):
        select_default_prediction_artifacts(tmp_path)


def test_external_prediction_status_requires_matching_run_id() -> None:
    predictions = pd.DataFrame(
        {
            "model_name": ["model"],
            "run_id": ["prediction-run"],
        }
    )
    status = {
        "status": "available",
        "details": {"run_id": "different-run", "models": ["model"]},
    }

    with pytest.raises(ValueError, match="does not match its status run_id"):
        validate_prediction_status(predictions, status, artifact_name="model")


def _write_status(path: Path, *, status: str, run_id: str | None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    details = {} if run_id is None else {"run_id": run_id}
    path.write_text(json.dumps({"status": status, "details": details}), encoding="utf-8")
