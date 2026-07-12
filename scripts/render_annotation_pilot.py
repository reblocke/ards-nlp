from __future__ import annotations

import argparse
from pathlib import Path

from ards_cxr_benchmark.pilot_annotation_agreement import (
    default_annotation_pilot_config_path,
    load_annotation_pilot_config,
)
from ards_cxr_benchmark.pilot_annotation_render import render_annotation_pilot


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render the REDCap pilot agreement report to its configured output directory"
    )
    parser.add_argument("--config", type=Path, default=default_annotation_pilot_config_path())
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_annotation_pilot_config(args.config)
    report_path = render_annotation_pilot(config, config_path=args.config)
    print(f"Rendered annotation pilot report: {report_path}")


if __name__ == "__main__":
    main()
