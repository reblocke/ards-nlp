from __future__ import annotations

import argparse
from pathlib import Path

from ards_cxr_benchmark.bq import execute_sql, load_and_render_sql
from ards_cxr_benchmark.config import default_config_path, load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render and run a benchmark BigQuery SQL file")
    parser.add_argument("--config", type=Path, default=default_config_path())
    parser.add_argument("--sql", required=True, type=Path)
    parser.add_argument("--render-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--param",
        action="append",
        default=[],
        help="Extra SQL template parameter as key=value; may be repeated",
    )
    return parser.parse_args()


def parse_extra_params(items: list[str]) -> dict[str, str]:
    params: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Expected key=value for --param, got: {item}")
        key, value = item.split("=", 1)
        params[key] = value
    return params


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    sql = load_and_render_sql(args.sql, config, extra=parse_extra_params(args.param))

    if args.render_only:
        print(sql)
        return

    result = execute_sql(sql, config, dry_run=args.dry_run)
    if args.dry_run:
        print(f"Dry run complete: {result.total_bytes_processed:,} bytes would be processed")
    else:
        print(f"Completed SQL file: {args.sql}")


if __name__ == "__main__":
    main()
