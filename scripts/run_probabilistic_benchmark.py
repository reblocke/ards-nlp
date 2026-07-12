from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from ards_cxr_benchmark.paths import get_paths
from ards_cxr_benchmark.probabilistic_benchmark import (
    run_probabilistic_benchmark,
    write_probabilistic_benchmark_outputs,
)
from ards_cxr_benchmark.probabilistic_metrics import DEFAULT_THRESHOLD_GRID


def default_reference() -> Path:
    return get_paths().root / "tests" / "fixtures" / "probabilistic_case_reference.csv"


def default_predictions() -> Path:
    return get_paths().root / "tests" / "fixtures" / "probabilistic_predictions.csv"


def default_output_dir(reference_path: Path, prediction_path: Path) -> Path:
    output = get_paths().root / "artifacts" / "benchmark"
    if (
        reference_path.resolve() == default_reference().resolve()
        and prediction_path.resolve() == default_predictions().resolve()
    ):
        output = output / "synthetic"
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark model probabilities against continuous human references"
    )
    parser.add_argument("--reference", type=Path, default=default_reference())
    parser.add_argument("--predictions", type=Path, default=default_predictions())
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--n-bins", type=int, default=10)
    parser.add_argument(
        "--thresholds",
        nargs="+",
        type=float,
        default=list(DEFAULT_THRESHOLD_GRID),
        help="Threshold grid for expected soft-count operating metrics",
    )
    parser.add_argument(
        "--cds-threshold",
        type=float,
        help="Optional additional CDS operating threshold to include in the grid",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    thresholds = sorted(
        set(args.thresholds + ([] if args.cds_threshold is None else [args.cds_threshold]))
    )
    out_dir = args.out_dir or default_output_dir(args.reference, args.predictions)
    result = run_probabilistic_benchmark(
        reference=read_table(args.reference),
        predictions=read_table(args.predictions),
        thresholds=thresholds,
        n_bins=args.n_bins,
    )
    write_probabilistic_benchmark_outputs(result, out_dir=out_dir)
    print(f"Wrote probabilistic benchmark artifacts to {out_dir}")


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() in {".csv", ".txt"}:
        return pd.read_csv(path)
    raise ValueError(f"Unsupported input format for {path}")


if __name__ == "__main__":
    main()
