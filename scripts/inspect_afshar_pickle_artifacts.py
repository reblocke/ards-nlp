from __future__ import annotations

import argparse
import json
from pathlib import Path

from ards_cxr_benchmark.comparators.afshar import inspect_afshar_resources, write_afshar_audit
from ards_cxr_benchmark.comparators.config import load_afshar_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Statically inspect Afshar pickle artifacts")
    parser.add_argument("--config", type=Path, required=True)
    config = load_afshar_config(parser.parse_args().config)
    inventory = inspect_afshar_resources(config)
    write_afshar_audit(config, inventory)
    print(json.dumps(inventory, indent=2))


if __name__ == "__main__":
    main()
