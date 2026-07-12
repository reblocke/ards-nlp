from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

AVAILABLE_EXTERNAL_STATUSES = {
    "available",
    "available_exploratory_mismatched_target",
}


@dataclass(frozen=True)
class PredictionArtifactSpec:
    name: str
    prediction_path: Path
    required: bool
    status_path: Path | None = None
    allowed_statuses: frozenset[str] = frozenset()


@dataclass(frozen=True)
class SelectedPredictionArtifact:
    spec: PredictionArtifactSpec
    status: dict[str, Any] | None


def default_prediction_artifacts(root: Path) -> list[PredictionArtifactSpec]:
    derived = root / "data/derived/comparators"
    artifacts = root / "artifacts/comparators"
    return [
        PredictionArtifactSpec(
            name="clamp_legacy",
            prediction_path=derived / "clamp_legacy_predictions.parquet",
            required=True,
        ),
        PredictionArtifactSpec(
            name="clamp_python_compatibility",
            prediction_path=derived / "clamp_python_compatibility_predictions.parquet",
            required=True,
        ),
        PredictionArtifactSpec(
            name="silver_baselines",
            prediction_path=derived / "silver_baseline_predictions.parquet",
            required=True,
        ),
        PredictionArtifactSpec(
            name="amaral",
            prediction_path=derived / "amaral_bilateral_infiltrates_predictions.parquet",
            required=True,
            status_path=artifacts / "amaral/status.json",
            allowed_statuses=frozenset({"available"}),
        ),
        PredictionArtifactSpec(
            name="uw_hanso",
            prediction_path=derived / "uw_hanso_predictions.parquet",
            required=False,
            status_path=artifacts / "uw_hanso/status.json",
            allowed_statuses=frozenset({"available"}),
        ),
        PredictionArtifactSpec(
            name="afshar",
            prediction_path=derived / "afshar_text_svc_predictions.parquet",
            required=False,
            status_path=artifacts / "afshar/status.json",
            allowed_statuses=frozenset({"available_exploratory_mismatched_target"}),
        ),
    ]


def select_default_prediction_artifacts(root: Path) -> list[SelectedPredictionArtifact]:
    selected: list[SelectedPredictionArtifact] = []
    for spec in default_prediction_artifacts(root):
        prediction_exists = spec.prediction_path.is_file()
        status = _read_status(spec.status_path)
        status_value = str(status.get("status", "")) if status else ""

        if spec.status_path is not None and prediction_exists:
            if status is None:
                raise ValueError(
                    f"Prediction artifact {spec.prediction_path} has no current status file; "
                    "remove the stale prediction or rerun the comparator"
                )
            if status_value not in spec.allowed_statuses:
                raise ValueError(
                    f"Blocked comparator {spec.name} still has prediction artifact "
                    f"{spec.prediction_path} (status={status_value!r}); remove it or complete "
                    "a current successful run"
                )

        if spec.required and not prediction_exists:
            raise FileNotFoundError(
                f"Required comparator prediction artifact is missing: {spec.prediction_path}"
            )
        if (
            spec.status_path is not None
            and status_value in spec.allowed_statuses
            and not prediction_exists
        ):
            raise FileNotFoundError(
                f"Comparator {spec.name} is marked {status_value!r} but its prediction artifact "
                f"is missing: {spec.prediction_path}"
            )
        if prediction_exists:
            selected.append(SelectedPredictionArtifact(spec=spec, status=status))
    return selected


def validate_prediction_status(
    predictions: pd.DataFrame,
    status: dict[str, Any],
    *,
    artifact_name: str,
    check_model_set: bool = True,
) -> None:
    status_value = str(status.get("status", ""))
    if status_value not in AVAILABLE_EXTERNAL_STATUSES:
        raise ValueError(
            f"External prediction artifact {artifact_name} has non-available status "
            f"{status_value!r}"
        )
    if "run_id" not in predictions.columns:
        raise ValueError(f"External prediction artifact {artifact_name} has no run_id column")
    run_ids = predictions["run_id"].dropna().astype(str).str.strip().unique().tolist()
    expected_run_id = str(status.get("details", {}).get("run_id", "")).strip()
    if len(run_ids) != 1 or not expected_run_id or run_ids[0] != expected_run_id:
        raise ValueError(
            f"External prediction artifact {artifact_name} does not match its status run_id: "
            f"predictions={run_ids}, status={expected_run_id!r}"
        )
    expected_models = status.get("details", {}).get("models")
    if check_model_set and expected_models is not None:
        observed_models = set(predictions["model_name"].astype(str))
        if observed_models != {str(value) for value in expected_models}:
            raise ValueError(
                f"External prediction artifact {artifact_name} model set does not match its "
                f"status: predictions={sorted(observed_models)}, status={sorted(expected_models)}"
            )


def validate_external_status_coverage(
    predictions: pd.DataFrame,
    statuses: list[dict[str, Any]],
) -> None:
    external_roles = {"external_comparator", "mismatched_target_exploratory"}
    external = predictions[predictions["comparison_role"].isin(external_roles)]
    for model_name, group in external.groupby("model_name", sort=True, observed=True):
        matches = [
            status
            for status in statuses
            if str(status.get("name", "")) == str(model_name)
            or str(model_name)
            in {str(value) for value in status.get("details", {}).get("models", [])}
        ]
        if len(matches) != 1:
            raise ValueError(
                f"External model {model_name} must have exactly one matching status; "
                f"found {len(matches)}"
            )
        validate_prediction_status(
            group,
            matches[0],
            artifact_name=str(model_name),
            check_model_set=False,
        )


def _read_status(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.is_file():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Comparator status must be a JSON object: {path}")
    return value
