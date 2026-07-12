from __future__ import annotations

import json
import pickletools
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from ..config import ensure_dir
from .common import (
    assert_no_text_leakage,
    sha256_path,
    validate_canonical_predictions,
    write_comparator_status,
)
from .config import AfsharComparatorConfig
from .external import verify_external_repository


def inspect_pickle_artifact(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    operations = list(pickletools.genops(data))
    protocols = sorted(
        {int(argument) for opcode, argument, _position in operations if opcode.name == "PROTO"}
    )
    globals_found = {
        str(argument)
        for opcode, argument, _position in operations
        if opcode.name in {"GLOBAL", "INST"}
    }
    for index, (opcode, _argument, _position) in enumerate(operations):
        if opcode.name != "STACK_GLOBAL":
            continue
        preceding = [
            str(argument)
            for previous, argument, _ in operations[max(0, index - 8) : index]
            if previous.name in {"SHORT_BINUNICODE", "BINUNICODE", "UNICODE"}
        ]
        if len(preceding) >= 2:
            globals_found.add(f"{preceding[-2]} {preceding[-1]}")
    sklearn_versions: set[str] = set()
    for index, (_opcode, argument, _position) in enumerate(operations):
        if str(argument) != "_sklearn_version":
            continue
        for next_opcode, next_argument, _ in operations[index + 1 : index + 8]:
            if next_opcode.name in {"SHORT_BINUNICODE", "BINUNICODE", "UNICODE"}:
                sklearn_versions.add(str(next_argument))
                break
    return {
        "artifact": path.name,
        "sha256": sha256_path(path),
        "bytes": int(path.stat().st_size),
        "pickle_protocols": protocols,
        "referenced_modules_classes": sorted(globals_found),
        "embedded_sklearn_versions": sorted(sklearn_versions),
        "loaded": False,
    }


def inspect_afshar_resources(config: AfsharComparatorConfig) -> dict[str, Any]:
    source = verify_external_repository(config.source)
    for path in (config.model_path, config.vectorizer_path):
        if not path.is_file():
            raise FileNotFoundError(f"Missing Afshar pickle artifact: {path}")
    artifacts = [
        inspect_pickle_artifact(config.model_path),
        inspect_pickle_artifact(config.vectorizer_path),
    ]
    for name, artifact in zip(["model", "vectorizer"], artifacts, strict=True):
        expected = config.expected_sha256.get(name)
        if not expected:
            raise ValueError(f"No expected SHA-256 configured for Afshar {name}")
        if artifact["sha256"] != expected:
            raise ValueError(
                f"Afshar {name} checksum mismatch: expected {expected}, found {artifact['sha256']}"
            )
    inventory = {
        "name": config.source.name,
        "source": source,
        "license_status": config.source.license,
        "permission_status": config.permission_status,
        "verified_target": config.verified_target,
        "artifacts": artifacts,
        "static_inspection_only": True,
    }
    return inventory


def afshar_container_command(
    *,
    config: AfsharComparatorConfig,
    packet: Path,
    output: Path,
    runner: Path,
) -> list[str]:
    model_container_path = "/artifacts/model.sav"
    vectorizer_container_path = "/artifacts/vectorizer.sav"
    return [
        "docker",
        "run",
        "--rm",
        "--platform",
        "linux/amd64",
        "-v",
        f"{config.source.external_repo_dir}:/external:ro",
        "-v",
        f"{config.model_path}:{model_container_path}:ro",
        "-v",
        f"{config.vectorizer_path}:{vectorizer_container_path}:ro",
        "-v",
        f"{packet}:/input/packet.jsonl.gz:ro",
        "-v",
        f"{runner}:/runner.py:ro",
        "-v",
        f"{output.parent}:/output",
        config.container_image,
        "python",
        "/runner.py",
        "--model",
        model_container_path,
        "--vectorizer",
        vectorizer_container_path,
        "--expected-model-sha256",
        config.expected_sha256["model"],
        "--expected-vectorizer-sha256",
        config.expected_sha256["vectorizer"],
        "--packet",
        "/input/packet.jsonl.gz",
        "--output",
        f"/output/{output.name}",
        "--batch-size",
        str(config.batch_size),
    ]


def write_afshar_audit(config: AfsharComparatorConfig, inventory: dict[str, Any]) -> None:
    ensure_dir(config.artifact_dir)
    (config.artifact_dir / "pickle_inventory.json").write_text(
        json.dumps(inventory, indent=2) + "\n", encoding="utf-8"
    )
    anchor_rows = [
        {"anchor_id": "clear_negative", "report_text": "The lungs are clear."},
        {
            "anchor_id": "bilateral_infiltrates",
            "report_text": "There are bilateral pulmonary infiltrates.",
        },
        {"anchor_id": "unilateral_opacity", "report_text": "Focal left basilar opacity."},
        {
            "anchor_id": "negated_bilateral",
            "report_text": "No bilateral pulmonary infiltrates are present.",
        },
        {
            "anchor_id": "bilateral_effusions",
            "report_text": "Small bilateral pleural effusions without airspace opacity.",
        },
        {"anchor_id": "pulmonary_edema", "report_text": "Moderate pulmonary edema."},
        {
            "anchor_id": "ambiguous_bibasilar",
            "report_text": "Mild bibasilar opacities may reflect atelectasis.",
        },
    ]
    (config.artifact_dir / "anchor_set.json").write_text(
        json.dumps(anchor_rows, indent=2) + "\n", encoding="utf-8"
    )
    permission_cleared = config.permission_status.lower() in {
        "cleared",
        "approved",
        "internal_use_cleared",
    }
    status = (
        "requires_isolated_synthetic_smoke" if permission_cleared else "blocked_license_permission"
    )
    reason = (
        "permission documented; isolated legacy-environment smoke remains required"
        if permission_cleared
        else "no upstream license or explicit internal-use permission is documented"
    )
    write_comparator_status(
        name=config.source.name,
        status=status,
        reason=reason,
        out_path=config.artifact_dir / "status.json",
        details={
            "verified_target": config.verified_target,
            "comparison_role": "mismatched_target_exploratory",
            "static_inspection_only": True,
        },
    )


def build_afshar_predictions(
    *,
    runner_output: Path,
    manifest: pd.DataFrame,
    config: AfsharComparatorConfig,
    run_id: str,
) -> pd.DataFrame:
    raw = pd.read_json(runner_output, lines=True)
    required = {"case_id", "prediction_score", "raw_predicted_class"}
    missing = sorted(required - set(raw.columns))
    if missing:
        raise ValueError(f"Afshar runner output is missing columns: {missing}")
    metadata_columns = ["case_id", "subject_id", "study_id", "split", "source_dataset"]
    missing_manifest = sorted(set(metadata_columns) - set(manifest.columns))
    if missing_manifest:
        raise ValueError(f"Afshar manifest is missing columns: {missing_manifest}")
    merged = manifest[metadata_columns].merge(raw, on="case_id", how="inner", validate="one_to_one")
    if len(raw) != len(manifest) or len(merged) != len(manifest):
        raise ValueError(
            f"Afshar row-set mismatch: manifest={len(manifest)}, "
            f"runner={len(raw)}, joined={len(merged)}"
        )
    merged["model_name"] = "afshar_text_svc_full_ards"
    merged["model_family"] = "tfidf_svc"
    merged["comparison_role"] = "mismatched_target_exploratory"
    merged["intended_target"] = "both"
    merged["model_source_repository"] = "AfsharJoyceInfoLab/ARDS_Classifier"
    merged["model_source_commit"] = config.source.commit
    merged["model_artifact_version"] = config.expected_sha256["model"][:12]
    merged["text_scope"] = "full_report"
    merged["prediction_label"] = (
        merged["prediction_score"].astype(float) >= config.threshold
    ).astype(int)
    merged["threshold"] = config.threshold
    merged["run_id"] = run_id
    merged["model_sha256"] = config.expected_sha256["model"]
    merged["vectorizer_sha256"] = config.expected_sha256["vectorizer"]
    merged["verified_target"] = config.verified_target
    predictions = validate_canonical_predictions(merged)
    assert_no_text_leakage(predictions)
    return predictions


def write_afshar_predictions(
    predictions: pd.DataFrame,
    *,
    config: AfsharComparatorConfig,
    run_id: str,
) -> None:
    config.prediction_output.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(config.prediction_output, index=False)
    ensure_dir(config.artifact_dir)
    summary = {
        "run_id": run_id,
        "run_at_utc": datetime.now(UTC).isoformat(),
        "prediction_rows": int(len(predictions)),
        "unique_cases": int(predictions["case_id"].nunique()),
        "models": sorted(predictions["model_name"].unique().tolist()),
        "verified_target": config.verified_target,
        "comparison_role": "mismatched_target_exploratory",
        "contains_report_text": False,
    }
    (config.artifact_dir / "run_manifest.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    write_comparator_status(
        name="afshar_text_svc_full_ards",
        status="available_exploratory_mismatched_target",
        reason="full MIMIC predictions completed after permission and anchor gates",
        out_path=config.artifact_dir / "status.json",
        details=summary,
    )
