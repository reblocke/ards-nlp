from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from ards_cxr_benchmark.bq import query_to_dataframe
from ards_cxr_benchmark.clamp_ards_benchmark import (
    benchmark_clamp_against_silver,
    model_extract_sql,
    write_clamp_teacher_benchmark_outputs,
)
from ards_cxr_benchmark.config import default_config_path, load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark ARDS CLAMP teacher predictions against silver labels"
    )
    parser.add_argument("--config", type=Path, default=default_config_path())
    parser.add_argument("--prediction-input", type=Path)
    parser.add_argument("--model-extract", type=Path)
    parser.add_argument(
        "--source-table",
        help=(
            "BigQuery model extract table. Defaults to "
            "<project>.<dataset>.model_development_extract."
        ),
    )
    parser.add_argument("--out-dir", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    prediction_input = args.prediction_input or config.clamp_ards.teacher_prediction_output
    predictions = pd.read_parquet(prediction_input)
    if args.model_extract:
        model_extract = _read_table(args.model_extract)
    else:
        source_table = args.source_table or f"{config.bq.dataset_ref}.model_development_extract"
        model_extract = query_to_dataframe(model_extract_sql(source_table), config)
    metrics, strata, summary = benchmark_clamp_against_silver(
        predictions=predictions,
        model_extract=model_extract,
    )
    out_dir = args.out_dir or config.clamp_ards.teacher_benchmark_dir
    write_clamp_teacher_benchmark_outputs(
        metrics=metrics,
        strata=strata,
        summary=summary,
        out_dir=out_dir,
    )
    print(f"Wrote ARDS CLAMP teacher benchmark metrics to {out_dir}")


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.name.lower().endswith((".csv", ".csv.gz", ".txt", ".txt.gz")):
        return pd.read_csv(path)
    raise ValueError(f"Unsupported model extract format: {path}")


if __name__ == "__main__":
    main()
