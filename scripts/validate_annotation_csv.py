from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from ards_cxr_benchmark.annotations import (
    text_stripped_review_labels,
    validate_review_dataframe,
    write_annotation_validation,
)
from ards_cxr_benchmark.config import ensure_parent_dir
from ards_cxr_benchmark.paths import get_paths


def default_out_parquet() -> Path:
    return get_paths().root / "data" / "derived" / "annotations" / "reviewed_labels.parquet"


def default_out_json() -> Path:
    return get_paths().root / "data" / "derived" / "annotations" / "reviewed_labels_validation.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate and export completed review labels")
    parser.add_argument("--input-csv", required=True, type=Path)
    parser.add_argument("--out-parquet", type=Path, default=default_out_parquet())
    parser.add_argument("--out-json", type=Path, default=default_out_json())
    parser.add_argument(
        "--require-completed",
        action="store_true",
        help="Fail when no completed review rows are present",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input_csv)
    result = validate_review_dataframe(df)
    if args.require_completed and result.n_completed_rows == 0:
        raise SystemExit("No completed review rows found")

    write_annotation_validation(result, out_json=args.out_json)
    if not result.passed:
        raise SystemExit(f"Annotation validation failed; see {args.out_json}")

    reviewed = text_stripped_review_labels(df)
    ensure_parent_dir(args.out_parquet)
    reviewed.to_parquet(args.out_parquet, index=False)
    print(f"Wrote {len(reviewed):,} reviewed rows to {args.out_parquet}")
    print(f"Wrote validation summary to {args.out_json}")


if __name__ == "__main__":
    main()
