from __future__ import annotations

import argparse
import json
from pathlib import Path

from ards_cxr_benchmark.comparators.config import load_amaral_config
from ards_cxr_benchmark.comparators.external import fetch_pinned_repository


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch the pinned Amaral comparator source")
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    config = load_amaral_config(parse_args().config)
    result = fetch_pinned_repository(config.source)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
