from __future__ import annotations

import argparse
import json
from pathlib import Path

from ards_cxr_benchmark.comparators.config import load_uw_hanso_config
from ards_cxr_benchmark.comparators.uw_hanso import (
    verify_uw_hanso_resources,
    write_uw_hanso_verification,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify UW HANSO source and model gates")
    parser.add_argument("--config", type=Path, required=True)
    config = load_uw_hanso_config(parser.parse_args().config)
    result = verify_uw_hanso_resources(config)
    write_uw_hanso_verification(config, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
