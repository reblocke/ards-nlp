from __future__ import annotations

import argparse
from pathlib import Path

from ards_cxr_benchmark.annotation_planning import (
    default_annotation_planning_config_path,
    load_annotation_planning_config,
)
from ards_cxr_benchmark.annotation_planning_render import render_annotation_planning


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render annotation reliability, validation, and workload scenarios"
    )
    parser.add_argument("--config", type=Path, default=default_annotation_planning_config_path())
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_annotation_planning_config(args.config)
    report = render_annotation_planning(config, config_path=args.config)
    print(f"Rendered annotation planning report: {report}")


if __name__ == "__main__":
    main()
