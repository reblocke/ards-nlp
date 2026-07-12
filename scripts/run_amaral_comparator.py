from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from ards_cxr_benchmark.comparators.amaral import (
    build_amaral_predictions,
    validate_amaral_anchor_predictions,
    verify_amaral_resources,
    write_amaral_run_outputs,
    write_published_preprocessing_config,
)
from ards_cxr_benchmark.comparators.config import load_amaral_config
from ards_cxr_benchmark.paths import get_paths

MODES = ["published_mimic_preprocessing", "raw_text_direct"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the isolated Amaral comparator")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--mode", choices=[*MODES, "all"], default="all")
    parser.add_argument("--input-packet", type=Path)
    parser.add_argument("--input-manifest", type=Path)
    parser.add_argument("--runner-output-dir", type=Path)
    parser.add_argument("--prediction-output", type=Path)
    parser.add_argument("--artifact-dir", type=Path)
    parser.add_argument("--anchor-expected", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_config = load_amaral_config(args.config)
    if args.mode != "all":
        _validate_partial_mode_paths(args, base_config)
    config = base_config
    config = replace(
        config,
        input_packet=(args.input_packet.resolve() if args.input_packet else config.input_packet),
        input_manifest=(
            args.input_manifest.resolve() if args.input_manifest else config.input_manifest
        ),
        runner_output_dir=(
            args.runner_output_dir.resolve() if args.runner_output_dir else config.runner_output_dir
        ),
        prediction_output=(
            args.prediction_output.resolve() if args.prediction_output else config.prediction_output
        ),
        artifact_dir=args.artifact_dir.resolve() if args.artifact_dir else config.artifact_dir,
    )
    resources = verify_amaral_resources(config)
    write_published_preprocessing_config(config)
    if not config.input_packet.is_file() or not config.input_manifest.is_file():
        raise FileNotFoundError("Comparator input packet is missing; run make comparator-source")
    if not (config.nltk_data_dir / "tokenizers").exists():
        raise FileNotFoundError("Amaral NLTK data is missing; run make comparator-amaral-runtime")

    modes = MODES if args.mode == "all" else [args.mode]
    run_id = datetime.now(UTC).strftime("amaral-%Y%m%dT%H%M%SZ")
    runner = get_paths().root / "scripts/external_runners/amaral_runner.py"
    config.runner_output_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["NLTK_DATA"] = str(config.nltk_data_dir)
    outputs: list[tuple[str, Path]] = []
    for mode in modes:
        output = config.runner_output_dir / f"{mode}.jsonl"
        command = [
            "uv",
            "run",
            "--project",
            str(config.environment_dir),
            "python",
            str(runner),
            "--external-repo",
            str(config.source.external_repo_dir),
            "--model",
            str(config.model_path),
            "--vectorizer",
            str(config.vectorizer_path),
            "--expected-model-sha256",
            config.expected_sha256["model"],
            "--expected-vectorizer-sha256",
            config.expected_sha256["vectorizer"],
            "--packet",
            str(config.input_packet),
            "--preprocessing-config",
            str(config.preprocessing_config),
            "--output",
            str(output),
            "--mode",
            mode,
            "--batch-size",
            str(config.batch_size),
            "--nltk-data",
            str(config.nltk_data_dir),
        ]
        subprocess.run(command, check=True, env=env)
        outputs.append((mode, output))
    manifest = pd.read_parquet(config.input_manifest)
    predictions = build_amaral_predictions(
        runner_outputs=outputs,
        manifest=manifest,
        config=config,
        run_id=run_id,
        repository_commit=config.source.commit,
    )
    if args.anchor_expected:
        config.artifact_dir.mkdir(parents=True, exist_ok=True)
        anchor_result = validate_amaral_anchor_predictions(
            predictions,
            args.anchor_expected.resolve(),
        )
        (config.artifact_dir / "anchor_parity.json").write_text(
            json.dumps(anchor_result, indent=2) + "\n",
            encoding="utf-8",
        )
    write_amaral_run_outputs(
        predictions=predictions,
        config=config,
        resource_verification=resources,
        run_id=run_id,
        command=[sys.executable, *sys.argv],
        run_kind=(
            "smoke"
            if args.anchor_expected
            else "full"
            if args.mode == "all"
            and not any(
                [
                    args.input_packet,
                    args.input_manifest,
                    args.runner_output_dir,
                    args.prediction_output,
                    args.artifact_dir,
                ]
            )
            else "partial"
        ),
    )
    print(
        f"Wrote {len(predictions):,} canonical Amaral prediction rows to {config.prediction_output}"
    )


def _validate_partial_mode_paths(args: argparse.Namespace, config) -> None:
    required = {
        "--runner-output-dir": (args.runner_output_dir, config.runner_output_dir),
        "--prediction-output": (args.prediction_output, config.prediction_output),
        "--artifact-dir": (args.artifact_dir, config.artifact_dir),
    }
    missing = [name for name, (value, _default) in required.items() if value is None]
    if missing:
        raise ValueError(
            "Single-mode Amaral runs require explicit noncanonical output paths: "
            + ", ".join(missing)
        )
    canonical = [
        name
        for name, (value, default) in required.items()
        if value is not None and value.resolve() == default.resolve()
    ]
    if canonical:
        raise ValueError(
            "Single-mode Amaral runs cannot use canonical full-run paths: " + ", ".join(canonical)
        )


if __name__ == "__main__":
    main()
