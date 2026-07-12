from __future__ import annotations

import argparse
from pathlib import Path

from ards_cxr_benchmark.clamp_ards_output_packet import prepare_clamp_txt_output_packet
from ards_cxr_benchmark.config import default_config_path, load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a restricted TXT-only packet from returned ARDS CLAMP outputs"
    )
    parser.add_argument("--config", type=Path, default=default_config_path())
    parser.add_argument("--source-archive", type=Path, required=True)
    parser.add_argument("--output-archive", type=Path)
    parser.add_argument("--input-manifest", type=Path)
    parser.add_argument("--summary-output", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    summary = prepare_clamp_txt_output_packet(
        source_archive=args.source_archive,
        output_archive=args.output_archive or config.clamp_ards.output_archive,
        input_manifest_path=args.input_manifest or config.clamp_ards.input_manifest,
        summary_output=args.summary_output or config.clamp_ards.output_packet_summary,
        overwrite=args.overwrite,
        show_progress=True,
    )
    print(
        "Built CLAMP TXT-only output packet: "
        f"{summary['output_txt_members']:,} document(s), "
        f"{summary['output_archive_bytes']:,} byte(s)"
    )


if __name__ == "__main__":
    main()
