from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from ards_cxr_benchmark.comparators.common import (
    normalize_clamp_predictions,
    normalize_silver_baseline_predictions,
    repository_state,
)
from ards_cxr_benchmark.config import ensure_parent_dir
from ards_cxr_benchmark.paths import get_paths


def parse_args() -> argparse.Namespace:
    root = get_paths().root
    parser = argparse.ArgumentParser(
        description="Normalize existing CLAMP and silver baselines to the comparator contract"
    )
    parser.add_argument(
        "--reference",
        type=Path,
        default=root / "data/derived/comparators/mimic_cxr_silver_reference.parquet",
    )
    parser.add_argument(
        "--clamp-input",
        type=Path,
        default=root / "data/derived/clamp_ards/clamp_legacy_predictions.parquet",
    )
    parser.add_argument(
        "--baseline-input",
        type=Path,
        default=root / "data/derived/modeling/silver_baseline_predictions.parquet",
    )
    parser.add_argument(
        "--clamp-output",
        type=Path,
        default=root / "data/derived/comparators/clamp_legacy_predictions.parquet",
    )
    parser.add_argument(
        "--baseline-output",
        type=Path,
        default=root / "data/derived/comparators/silver_baseline_predictions.parquet",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = pd.read_parquet(args.reference)
    state = repository_state(get_paths().root)
    run_id = datetime.now(UTC).strftime("existing-%Y%m%dT%H%M%SZ")
    clamp = normalize_clamp_predictions(
        pd.read_parquet(args.clamp_input),
        source,
        run_id=run_id,
        source_commit=state["commit"],
    )
    baselines = normalize_silver_baseline_predictions(
        pd.read_parquet(args.baseline_input),
        source,
        run_id=run_id,
        source_commit=state["commit"],
    )
    for frame, path in ((clamp, args.clamp_output), (baselines, args.baseline_output)):
        ensure_parent_dir(path)
        frame.to_parquet(path, index=False)
    print(f"Wrote {len(clamp):,} CLAMP and {len(baselines):,} silver-baseline rows")


if __name__ == "__main__":
    main()
