from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd

from ards_cxr_benchmark.comparators.common import (
    REFERENCE_COLUMNS,
    assert_no_text_leakage,
    build_comparator_source,
    write_comparator_input_packet,
)
from ards_cxr_benchmark.config import ensure_parent_dir
from ards_cxr_benchmark.paths import get_paths

EXPECTED_MIMIC_ROWS = 227_835


def parse_args() -> argparse.Namespace:
    root = get_paths().root
    parser = argparse.ArgumentParser(
        description="Build the restricted common MIMIC comparator packet and text-free reference"
    )
    parser.add_argument(
        "--reports",
        type=Path,
        default=root / "data/processed/mimic_cxr_reports.parquet",
    )
    parser.add_argument(
        "--model-extract",
        type=Path,
        default=root / "data/derived/modeling/model_development_extract.parquet",
    )
    parser.add_argument(
        "--packet",
        type=Path,
        default=root / "artifacts/restricted/comparators/mimic_cxr_comparator_input.jsonl.gz",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=root / "artifacts/restricted/comparators/mimic_cxr_comparator_manifest.parquet",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=root / "artifacts/restricted/comparators/mimic_cxr_comparator_summary.json",
    )
    parser.add_argument(
        "--reference-out",
        type=Path,
        default=root / "data/derived/comparators/mimic_cxr_silver_reference.parquet",
    )
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive")
    reports = pd.read_parquet(args.reports)
    model_extract = pd.read_parquet(args.model_extract)
    source = build_comparator_source(
        reports,
        model_extract,
        expected_rows=EXPECTED_MIMIC_ROWS,
    )
    if args.limit is not None:
        source = source.head(args.limit).copy()
    summary = write_comparator_input_packet(
        source,
        packet_path=args.packet,
        manifest_path=args.manifest,
        summary_path=args.summary,
    )
    reference = source[REFERENCE_COLUMNS].copy()
    assert_no_text_leakage(reference)
    ensure_parent_dir(args.reference_out)
    temp = args.reference_out.with_name(f".{args.reference_out.name}.partial")
    try:
        reference.to_parquet(temp, index=False)
        os.replace(temp, args.reference_out)
    finally:
        temp.unlink(missing_ok=True)
    print(
        json.dumps(
            {
                "rows": summary["rows"],
                "packet": str(args.packet),
                "manifest": str(args.manifest),
                "reference": str(args.reference_out),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
