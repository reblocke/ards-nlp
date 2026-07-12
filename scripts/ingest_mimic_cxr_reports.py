from __future__ import annotations

import argparse
from pathlib import Path

from ards_cxr_benchmark.config import (
    default_config_path,
    ensure_parent_dir,
    load_config,
    require_existing_path,
)
from ards_cxr_benchmark.text_sections import build_report_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Parse MIMIC-CXR report .txt files to Parquet")
    parser.add_argument("--config", type=Path, default=default_config_path())
    parser.add_argument("--report-root", type=Path)
    parser.add_argument("--out-parquet", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    report_root = args.report_root or config.paths.report_root
    out_parquet = args.out_parquet or config.paths.report_parquet

    require_existing_path(report_root, "MIMIC-CXR report root")
    reports = build_report_rows(report_root, primary_scope=config.primary_label_text_scope)
    ensure_parent_dir(out_parquet)
    reports.to_parquet(out_parquet, index=False)

    print(f"Wrote {len(reports):,} report rows to {out_parquet}")


if __name__ == "__main__":
    main()
