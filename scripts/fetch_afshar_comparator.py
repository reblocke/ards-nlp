from __future__ import annotations

import argparse
import json
from pathlib import Path

from ards_cxr_benchmark.comparators.config import load_afshar_config
from ards_cxr_benchmark.comparators.external import fetch_pinned_repository


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch the pinned Afshar comparator source")
    parser.add_argument("--config", type=Path, required=True)
    config = load_afshar_config(parser.parse_args().config)
    print(json.dumps(fetch_pinned_repository(config.source), indent=2))


if __name__ == "__main__":
    main()
