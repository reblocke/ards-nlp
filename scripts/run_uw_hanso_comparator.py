from __future__ import annotations

import argparse
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from ards_cxr_benchmark.comparators.config import load_uw_hanso_config
from ards_cxr_benchmark.comparators.uw_hanso import (
    HANSO_SCOPES,
    build_uw_hanso_predictions,
    verify_uw_hanso_resources,
    write_uw_hanso_predictions,
)
from ards_cxr_benchmark.paths import get_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the gated UW HANSO comparator")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--smoke-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_uw_hanso_config(args.config)
    verification = verify_uw_hanso_resources(config)
    if verification["status"] != "resources_present_requires_synthetic_smoke":
        raise RuntimeError(
            f"UW HANSO execution is blocked: {verification['status']}: {verification['reason']}"
        )
    if shutil.which("docker") is None:
        raise RuntimeError("Docker is required for the UW HANSO legacy runtime")
    if not config.smoke_input_packet.is_file() or not config.smoke_input_manifest.is_file():
        raise FileNotFoundError(
            "Synthetic comparator packet is missing; run make comparator-source-smoke"
        )

    smoke_run_id = datetime.now(UTC).strftime("uw-hanso-smoke-%Y%m%dT%H%M%SZ")
    smoke_predictions = _run_scopes(
        config=config,
        verification=verification,
        packet=config.smoke_input_packet,
        manifest_path=config.smoke_input_manifest,
        output_dir=config.artifact_dir / "runner/smoke",
        run_id=smoke_run_id,
    )
    write_uw_hanso_predictions(
        smoke_predictions,
        config=config,
        run_id=smoke_run_id,
        run_kind="smoke",
    )
    if args.smoke_only:
        print(f"Wrote {len(smoke_predictions):,} synthetic UW HANSO prediction rows")
        return

    if not config.input_packet.is_file() or not config.input_manifest.is_file():
        raise FileNotFoundError("Comparator input packet is missing; run make comparator-source")
    run_id = datetime.now(UTC).strftime("uw-hanso-%Y%m%dT%H%M%SZ")
    predictions = _run_scopes(
        config=config,
        verification=verification,
        packet=config.input_packet,
        manifest_path=config.input_manifest,
        output_dir=config.artifact_dir / "runner/full",
        run_id=run_id,
    )
    write_uw_hanso_predictions(predictions, config=config, run_id=run_id)
    print(f"Wrote {len(predictions):,} UW HANSO canonical prediction rows")


def _run_scopes(
    *,
    config,
    verification: dict,
    packet: Path,
    manifest_path: Path,
    output_dir: Path,
    run_id: str,
) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    runner = get_paths().root / "scripts/external_runners/uw_hanso_runner.py"
    outputs: list[tuple[str, Path]] = []
    for scope in HANSO_SCOPES:
        output = output_dir / f"{scope}.jsonl"
        command = [
            "docker",
            "run",
            "--rm",
            "--platform",
            "linux/amd64",
            "-v",
            f"{config.source.external_repo_dir}:/external:ro",
            "-v",
            f"{packet}:/input/packet.jsonl.gz:ro",
            "-v",
            f"{runner}:/runner.py:ro",
            "-v",
            f"{output_dir}:/output",
            config.container_image,
            "python",
            "/runner.py",
            "--external-repo",
            "/external",
            "--packet",
            "/input/packet.jsonl.gz",
            "--output",
            f"/output/{scope}.jsonl",
            "--scope",
            scope,
            "--batch-size",
            str(config.batch_size),
        ]
        subprocess.run(command, check=True)
        outputs.append((scope, output))
    return build_uw_hanso_predictions(
        runner_outputs=outputs,
        manifest=pd.read_parquet(manifest_path),
        config=config,
        run_id=run_id,
        parameters_sha256=verification["parameters_sha256"],
        state_dict_sha256=verification["state_dict_sha256"],
    )


if __name__ == "__main__":
    main()
