from __future__ import annotations

import argparse
import gzip
import json
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from ards_cxr_benchmark.comparators.afshar import (
    afshar_container_command,
    build_afshar_predictions,
    inspect_afshar_resources,
    write_afshar_predictions,
)
from ards_cxr_benchmark.comparators.config import load_afshar_config
from ards_cxr_benchmark.paths import get_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the gated Afshar text-SVC comparator")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--smoke-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_afshar_config(args.config)
    permission_ok = config.permission_status.lower() in {
        "cleared",
        "approved",
        "internal_use_cleared",
    }
    if not permission_ok:
        raise PermissionError(
            "Afshar comparator execution is blocked until permission_status is explicitly cleared"
        )
    inspect_afshar_resources(config)
    if shutil.which("docker") is None:
        raise RuntimeError("Docker is required for the Afshar scikit-learn 0.19 runtime")
    output_dir = config.artifact_dir / "runner"
    output_dir.mkdir(parents=True, exist_ok=True)
    runner = get_paths().root / "scripts/external_runners/afshar_runner.py"
    smoke_summary = _run_anchor_smoke(config=config, output_dir=output_dir, runner=runner)
    if args.smoke_only:
        print(json.dumps(smoke_summary, indent=2))
        return
    if config.anchor_review_status.lower() != "passed":
        raise PermissionError(
            "Afshar MIMIC execution is blocked until anchor_review_status is set to passed"
        )
    if not config.input_packet.is_file() or not config.input_manifest.is_file():
        raise FileNotFoundError("Comparator input packet is missing; run make comparator-source")

    runner_output = output_dir / "full_report.jsonl"
    _run_container(
        config=config,
        packet=config.input_packet,
        output=runner_output,
        runner=runner,
    )
    run_id = datetime.now(UTC).strftime("afshar-%Y%m%dT%H%M%SZ")
    predictions = build_afshar_predictions(
        runner_output=runner_output,
        manifest=pd.read_parquet(config.input_manifest),
        config=config,
        run_id=run_id,
    )
    write_afshar_predictions(predictions, config=config, run_id=run_id)
    print(f"Wrote {len(predictions):,} exploratory Afshar prediction rows")


def _run_anchor_smoke(*, config, output_dir: Path, runner: Path) -> dict[str, object]:
    anchor_path = config.artifact_dir / "anchor_set.json"
    if not anchor_path.is_file():
        raise FileNotFoundError("Afshar anchor set is missing; run make comparator-afshar-verify")
    anchors = json.loads(anchor_path.read_text(encoding="utf-8"))
    if not isinstance(anchors, list) or not anchors:
        raise ValueError("Afshar anchor set must be a non-empty JSON list")
    packet = output_dir / "anchor_input.jsonl.gz"
    with gzip.open(packet, "wt", encoding="utf-8", newline="\n") as handle:
        for anchor in anchors:
            handle.write(
                json.dumps(
                    {
                        "case_id": str(anchor["anchor_id"]),
                        "report_text": str(anchor["report_text"]),
                    },
                    separators=(",", ":"),
                )
                + "\n"
            )
    output = output_dir / "anchor_predictions.jsonl"
    _run_container(config=config, packet=packet, output=output, runner=runner)
    predictions = pd.read_json(output, lines=True)
    expected_ids = {str(anchor["anchor_id"]) for anchor in anchors}
    observed_ids = set(predictions.get("case_id", pd.Series(dtype=str)).astype(str))
    if observed_ids != expected_ids or len(predictions) != len(anchors):
        raise ValueError(
            "Afshar anchor output row set does not match the configured synthetic anchor set"
        )
    scores = pd.to_numeric(predictions["prediction_score"], errors="coerce")
    if scores.isna().any() or (~scores.between(0, 1)).any():
        raise ValueError("Afshar anchor prediction scores must be in [0, 1]")
    summary = {
        "status": "synthetic_smoke_requires_human_review",
        "anchor_rows": int(len(predictions)),
        "model_sha256": config.expected_sha256["model"],
        "vectorizer_sha256": config.expected_sha256["vectorizer"],
        "prediction_output": str(output),
    }
    (config.artifact_dir / "synthetic_smoke_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def _run_container(*, config, packet: Path, output: Path, runner: Path) -> None:
    subprocess.run(
        afshar_container_command(config=config, packet=packet, output=output, runner=runner),
        check=True,
    )


if __name__ == "__main__":
    main()
