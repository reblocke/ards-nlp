from __future__ import annotations

import argparse
from pathlib import Path

from ards_cxr_benchmark.bq import load_and_render_sql, query_to_dataframe
from ards_cxr_benchmark.config import (
    default_config_path,
    ensure_parent_dir,
    load_config,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a BigQuery table or SQL result to CSV")
    parser.add_argument("--config", type=Path, default=default_config_path())
    parser.add_argument("--table", help="Table name without project.dataset")
    parser.add_argument("--sql", type=Path, help="SQL file to render and export")
    parser.add_argument("--out-csv", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if bool(args.table) == bool(args.sql):
        raise ValueError("Pass exactly one of --table or --sql")

    if args.sql:
        sql = load_and_render_sql(args.sql, config)
    else:
        sql = f"SELECT * FROM `{config.bq.dataset_ref}.{args.table}`"

    df = query_to_dataframe(sql, config)
    ensure_parent_dir(args.out_csv)
    df.to_csv(args.out_csv, index=False)
    print(f"Wrote {len(df):,} rows to {args.out_csv}")


if __name__ == "__main__":
    main()
