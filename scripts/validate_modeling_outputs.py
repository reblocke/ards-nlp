from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from ards_cxr_benchmark.modeling_qa import evaluate_modeling_outputs, write_modeling_qa
from ards_cxr_benchmark.paths import get_paths


def default_extract_path() -> Path:
    return get_paths().root / "data" / "derived" / "modeling" / "model_development_extract.parquet"


def default_predictions_path() -> Path:
    return (
        get_paths().root / "data" / "derived" / "modeling" / "silver_baseline_predictions.parquet"
    )


def default_metrics_path() -> Path:
    return get_paths().root / "artifacts" / "modeling" / "silver_baseline_metrics.csv"


def default_out_json() -> Path:
    return get_paths().root / "artifacts" / "modeling" / "modeling_qa_summary.json"


def default_out_md() -> Path:
    return get_paths().root / "artifacts" / "modeling" / "modeling_qa_summary.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate local silver-label modeling artifacts")
    parser.add_argument("--extract", type=Path, default=default_extract_path())
    parser.add_argument("--predictions", type=Path, default=default_predictions_path())
    parser.add_argument("--metrics", type=Path, default=default_metrics_path())
    parser.add_argument("--out-json", type=Path, default=default_out_json())
    parser.add_argument("--out-md", type=Path, default=default_out_md())
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    extract = pd.read_parquet(args.extract)
    predictions = pd.read_parquet(args.predictions)
    metrics = pd.read_csv(args.metrics)
    result = evaluate_modeling_outputs(extract, predictions, metrics)
    write_modeling_qa(result, out_json=args.out_json, out_md=args.out_md)
    print(f"Wrote modeling QA to {args.out_md}")
    if not result.passed:
        raise SystemExit("Modeling QA failed; see summary for details")


if __name__ == "__main__":
    main()
