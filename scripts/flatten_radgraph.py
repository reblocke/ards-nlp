from __future__ import annotations

import argparse
from pathlib import Path

from ards_cxr_benchmark.config import default_config_path, load_config, require_existing_path
from ards_cxr_benchmark.radgraph import write_radgraph_parquet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Flatten RadGraph JSON to staging Parquet files")
    parser.add_argument("--config", type=Path, default=default_config_path())
    parser.add_argument("--radgraph-json", type=Path)
    parser.add_argument("--out-dir", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    radgraph_json = args.radgraph_json or config.paths.radgraph_json
    out_dir = args.out_dir or config.paths.radgraph_parquet_dir

    require_existing_path(radgraph_json, "RadGraph MIMIC-CXR JSON")
    reports_path, entities_path, relations_path = write_radgraph_parquet(radgraph_json, out_dir)

    print(f"Wrote RadGraph reports to {reports_path}")
    print(f"Wrote RadGraph entities to {entities_path}")
    print(f"Wrote RadGraph relations to {relations_path}")


if __name__ == "__main__":
    main()
