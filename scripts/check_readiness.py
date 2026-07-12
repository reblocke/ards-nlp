from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from ards_cxr_benchmark.paths import get_paths
from ards_cxr_benchmark.readiness import (
    collect_readiness,
    format_readiness,
    readiness_has_blockers,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check repository readiness by collaborator use case"
    )
    parser.add_argument(
        "--use-case",
        choices=["all", "annotation", "comparators", "build"],
        default="all",
    )
    parser.add_argument("--annotation-config", type=Path)
    parser.add_argument("--planning-config", type=Path)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--json", action="store_true", dest="as_json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checks = collect_readiness(
        get_paths().root,
        use_case=args.use_case,
        annotation_config=args.annotation_config,
        planning_config=args.planning_config,
    )
    if args.as_json:
        print(json.dumps([asdict(check) for check in checks], indent=2))
    else:
        print(format_readiness(checks))
    if args.strict and readiness_has_blockers(checks):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
