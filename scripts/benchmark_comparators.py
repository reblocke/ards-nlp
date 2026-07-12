from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from ards_cxr_benchmark.comparators.common import (
    benchmark_comparator_predictions,
    load_status_files,
    write_combined_benchmark_outputs,
)
from ards_cxr_benchmark.comparators.registry import (
    select_default_prediction_artifacts,
    validate_external_status_coverage,
    validate_prediction_status,
)
from ards_cxr_benchmark.paths import get_paths


def parse_args() -> argparse.Namespace:
    root = get_paths().root
    parser = argparse.ArgumentParser(
        description="Benchmark canonical comparator predictions against MIMIC silver labels"
    )
    parser.add_argument(
        "--reference",
        type=Path,
        default=root / "data/derived/comparators/mimic_cxr_silver_reference.parquet",
    )
    parser.add_argument("--predictions", type=Path, action="append")
    parser.add_argument("--status", type=Path, action="append")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=root / "artifacts/comparators/combined",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = get_paths().root
    selected = None if args.predictions else select_default_prediction_artifacts(root)
    prediction_paths = args.predictions or [item.spec.prediction_path for item in selected or []]
    frames = [pd.read_parquet(path) for path in prediction_paths]
    if selected is not None:
        for frame, item in zip(frames, selected, strict=True):
            if item.status is not None:
                validate_prediction_status(
                    frame,
                    item.status,
                    artifact_name=item.spec.name,
                )
    predictions = pd.concat(frames, ignore_index=True)
    reference = pd.read_parquet(args.reference)
    metrics, strata, catalog = benchmark_comparator_predictions(predictions, reference)
    status_paths = args.status or sorted((root / "artifacts/comparators").glob("*/status.json"))
    statuses = load_status_files(status_paths)
    validate_external_status_coverage(predictions, statuses)
    available_names = set(catalog["model_name"])
    for model_name in sorted(available_names):
        if not any(status.get("name") == model_name for status in statuses):
            statuses.append(
                {
                    "name": model_name,
                    "status": "available",
                    "reason": "canonical predictions present",
                    "details": {},
                }
            )
    statuses = sorted(statuses, key=lambda value: str(value.get("name", "")))
    write_combined_benchmark_outputs(
        metrics=metrics,
        strata=strata,
        catalog=catalog,
        statuses=statuses,
        out_dir=args.out_dir,
    )
    print(
        f"Benchmarked {catalog['model_name'].nunique()} model(s); "
        f"wrote {len(metrics)} overall metric row(s) to {args.out_dir}"
    )


if __name__ == "__main__":
    main()
