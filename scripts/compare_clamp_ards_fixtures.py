from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ards_cxr_benchmark.clamp_ards.fixture_parity import compare_clamp_ards_fixture
from ards_cxr_benchmark.clamp_ards.fixtures import FixturePendingError


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Strictly compare the Python ARDS mirror with CLAMP-generated fixtures"
    )
    parser.add_argument(
        "--fixture-root",
        type=Path,
        default=Path("tests/fixtures/clamp_ards_parity"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/restricted/clamp_ards/python/fixture_parity"),
    )
    parser.add_argument(
        "--project-dir",
        type=Path,
        help="Load user-supplied local CLAMP resources instead of the tracked project",
    )
    parser.add_argument(
        "--allow-pending",
        action="store_true",
        help="Exit successfully only for a valid, pristine awaiting_legacy_runs scaffold",
    )
    parser.add_argument("--progress", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        result = compare_clamp_ards_fixture(
            fixture_root=args.fixture_root,
            output_dir=args.output_dir,
            allow_pending=args.allow_pending,
            show_progress=args.progress,
            project_dir=args.project_dir,
        )
    except FixturePendingError as exc:
        print(f"CLAMP fixture is incomplete: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"Invalid CLAMP fixture or parity invocation: {exc}", file=sys.stderr)
        return 2

    print(json.dumps(result.summary, indent=2, sort_keys=True))
    if result.pending:
        print("CLAMP fixture scaffold is valid and awaiting two legacy runs")
        return 0
    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
