from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from ..config import ensure_dir, ensure_parent_dir
from .common import (
    assert_no_text_leakage,
    sha256_path,
    validate_canonical_predictions,
    write_comparator_status,
)
from .config import UWHansoComparatorConfig
from .external import verify_external_repository

HANSO_CLASSES = ["none", "present", "unilateral", "bilateral"]
HANSO_SCOPES = {
    "impression_findings": (
        "uw_hanso_impression_findings",
        "target_text_impression_findings",
    ),
    "full_report": ("uw_hanso_full_report", "report_text"),
}


def verify_uw_hanso_resources(config: UWHansoComparatorConfig) -> dict[str, Any]:
    source = verify_external_repository(
        config.source,
        allowed_untracked_paths=(config.parameters_path, config.state_dict_path),
    )
    license_path = config.source.external_repo_dir / "LICENSE"
    if not license_path.is_file():
        raise FileNotFoundError(f"UW HANSO license is missing: {license_path}")
    missing = [
        str(path) for path in (config.parameters_path, config.state_dict_path) if not path.is_file()
    ]
    missing_checksums = [
        name
        for name in ("parameters", "state_dict")
        if not _is_sha256(config.expected_sha256.get(name, ""))
    ]
    terms_ok = config.terms_of_use.lower() not in {"", "unknown", "verify", "pending"}
    if missing or missing_checksums or not terms_ok:
        reasons: list[str] = []
        if missing:
            reasons.append("trained model artifacts are missing")
        if missing_checksums:
            reasons.append(
                "expected checksums are not configured for " + ", ".join(missing_checksums)
            )
        if not terms_ok:
            reasons.append("terms of use are not documented")
        return {
            "name": config.source.name,
            "status": "blocked_missing_model_artifacts",
            "reason": "; ".join(reasons),
            "source": source,
            "missing_paths": missing,
            "missing_checksum_keys": missing_checksums,
            "terms_of_use": config.terms_of_use,
            "license_sha256": sha256_path(license_path),
        }
    parameters_sha256 = sha256_path(config.parameters_path)
    state_dict_sha256 = sha256_path(config.state_dict_path)
    for name, actual in (
        ("parameters", parameters_sha256),
        ("state_dict", state_dict_sha256),
    ):
        expected = config.expected_sha256[name]
        if actual != expected:
            raise ValueError(
                f"UW HANSO {name} checksum mismatch: expected {expected}, found {actual}"
            )
    return {
        "name": config.source.name,
        "status": "resources_present_requires_synthetic_smoke",
        "reason": "weights and terms are present; container smoke test is still required",
        "source": source,
        "parameters_sha256": parameters_sha256,
        "state_dict_sha256": state_dict_sha256,
        "terms_of_use": config.terms_of_use,
        "license_sha256": sha256_path(license_path),
    }


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(char in "0123456789abcdef" for char in value.lower())


def write_uw_hanso_verification(config: UWHansoComparatorConfig, result: dict[str, Any]) -> None:
    ensure_dir(config.artifact_dir)
    (config.artifact_dir / "resource_verification.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    write_comparator_status(
        name=config.source.name,
        status=str(result["status"]),
        reason=str(result["reason"]),
        out_path=config.artifact_dir / "status.json",
        details={
            key: value for key, value in result.items() if key not in {"name", "status", "reason"}
        },
    )


def run_hanso_batches(
    records: list[dict[str, Any]],
    *,
    batch_size: int,
    text_key: str,
    probability_fn: Callable[[list[str]], Iterable[dict[str, Any]]],
) -> list[tuple[str, dict[str, Any]]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    outputs: list[tuple[str, dict[str, Any]]] = []
    for start in range(0, len(records), batch_size):
        batch = records[start : start + batch_size]
        texts = [str(record[text_key]) for record in batch]
        probabilities = list(probability_fn(texts))
        if len(probabilities) != len(batch):
            raise ValueError(
                f"UW HANSO batch output mismatch: inputs={len(batch)}, outputs={len(probabilities)}"
            )
        outputs.extend(
            (str(record["case_id"]), probability)
            for record, probability in zip(batch, probabilities, strict=True)
        )
    return outputs


def build_uw_hanso_predictions(
    *,
    runner_outputs: list[tuple[str, Path]],
    manifest: pd.DataFrame,
    config: UWHansoComparatorConfig,
    run_id: str,
    parameters_sha256: str,
    state_dict_sha256: str,
) -> pd.DataFrame:
    required_manifest = {"case_id", "subject_id", "study_id", "split", "source_dataset"}
    missing_manifest = sorted(required_manifest - set(manifest.columns))
    if missing_manifest:
        raise ValueError(f"UW HANSO manifest is missing columns: {missing_manifest}")
    frames: list[pd.DataFrame] = []
    for scope, path in runner_outputs:
        if scope not in HANSO_SCOPES:
            raise ValueError(f"Unknown UW HANSO scope: {scope}")
        raw = pd.read_json(path, lines=True)
        probability_columns = [f"prob_infiltrates_{label}" for label in HANSO_CLASSES]
        required = {"case_id", "raw_predicted_infiltrates_class", *probability_columns}
        missing = sorted(required - set(raw.columns))
        if missing:
            raise ValueError(f"UW HANSO runner output is missing columns: {missing}")
        for column in probability_columns:
            raw[column] = pd.to_numeric(raw[column], errors="coerce")
            if raw[column].isna().any() or (~raw[column].between(0, 1)).any():
                raise ValueError(f"UW HANSO {column} must be in [0, 1]")
        sums = raw[probability_columns].sum(axis=1)
        if not sums.between(0.999, 1.001).all():
            raise ValueError("UW HANSO infiltrate probabilities do not sum to one")
        merged = manifest[list(required_manifest)].merge(
            raw, on="case_id", how="inner", validate="one_to_one"
        )
        if len(merged) != len(manifest) or len(raw) != len(manifest):
            raise ValueError(
                f"UW HANSO {scope} row-set mismatch: manifest={len(manifest)}, "
                f"runner={len(raw)}, joined={len(merged)}"
            )
        model_name, _text_column = HANSO_SCOPES[scope]
        merged["model_name"] = model_name
        merged["model_family"] = "hierarchical_attention_network"
        merged["comparison_role"] = "external_comparator"
        merged["intended_target"] = "both"
        merged["model_source_repository"] = "uw-bionlp/ards"
        merged["model_source_commit"] = config.source.commit
        merged["model_artifact_version"] = parameters_sha256[:12]
        merged["text_scope"] = scope
        merged["prediction_score"] = merged["prob_infiltrates_bilateral"]
        merged["prediction_label"] = (
            merged["raw_predicted_infiltrates_class"] == "bilateral"
        ).astype(int)
        merged["raw_predicted_class"] = merged["raw_predicted_infiltrates_class"]
        merged["threshold"] = None
        merged["run_id"] = run_id
        merged["parameters_sha256"] = parameters_sha256
        merged["state_dict_sha256"] = state_dict_sha256
        frames.append(merged)
    predictions = validate_canonical_predictions(pd.concat(frames, ignore_index=True))
    assert_no_text_leakage(predictions)
    return predictions


def write_uw_hanso_predictions(
    predictions: pd.DataFrame,
    *,
    config: UWHansoComparatorConfig,
    run_id: str,
    run_kind: str = "full",
) -> None:
    if run_kind not in {"full", "smoke"}:
        raise ValueError(f"Unknown UW HANSO run kind: {run_kind}")
    prediction_output = (
        config.prediction_output if run_kind == "full" else config.smoke_prediction_output
    )
    ensure_parent_dir(prediction_output)
    predictions.to_parquet(prediction_output, index=False)
    ensure_dir(config.artifact_dir)
    summary = {
        "run_id": run_id,
        "run_at_utc": datetime.now(UTC).isoformat(),
        "prediction_rows": int(len(predictions)),
        "unique_cases": int(predictions["case_id"].nunique()),
        "models": sorted(predictions["model_name"].unique().tolist()),
        "contains_report_text": False,
        "run_kind": run_kind,
    }
    manifest_name = "run_manifest.json" if run_kind == "full" else "synthetic_smoke_manifest.json"
    (config.artifact_dir / manifest_name).write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    write_comparator_status(
        name=config.source.name,
        status="available" if run_kind == "full" else "synthetic_smoke_passed",
        reason=(
            "full MIMIC predictions completed in the isolated legacy runtime"
            if run_kind == "full"
            else "synthetic inference completed with checksum-verified model artifacts"
        ),
        out_path=config.artifact_dir
        / ("status.json" if run_kind == "full" else "synthetic_smoke_status.json"),
        details=summary,
    )
