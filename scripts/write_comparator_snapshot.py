from __future__ import annotations

import argparse
from pathlib import Path

from ards_cxr_benchmark.comparator_snapshot import write_comparator_snapshot
from ards_cxr_benchmark.paths import get_paths


def parse_args() -> argparse.Namespace:
    root = get_paths().root
    parser = argparse.ArgumentParser(description="Write a tracked aggregate comparator snapshot")
    parser.add_argument(
        "--catalog",
        type=Path,
        default=root / "artifacts/comparators/combined/prediction_catalog.csv",
    )
    parser.add_argument(
        "--metrics",
        type=Path,
        default=root / "artifacts/comparators/combined/silver_metrics.csv",
    )
    parser.add_argument(
        "--compatibility-metrics",
        type=Path,
        default=root / "artifacts/comparators/combined/compatibility_mirror_metrics.csv",
    )
    parser.add_argument(
        "--status",
        type=Path,
        default=root / "artifacts/comparators/combined/comparator_status.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "docs/COMPARATOR_SNAPSHOT_V1.md",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    write_comparator_snapshot(
        catalog_path=args.catalog,
        metrics_path=args.metrics,
        compatibility_metrics_path=args.compatibility_metrics,
        status_path=args.status,
        output_path=args.output,
    )
    print(f"Wrote aggregate comparator snapshot: {args.output}")


if __name__ == "__main__":
    main()
