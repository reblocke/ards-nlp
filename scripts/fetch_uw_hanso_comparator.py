from __future__ import annotations

import argparse
import json
from pathlib import Path

from ards_cxr_benchmark.comparators.config import load_uw_hanso_config
from ards_cxr_benchmark.comparators.external import fetch_pinned_repository


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch the pinned UW HANSO source")
    parser.add_argument("--config", type=Path, required=True)
    config = load_uw_hanso_config(parser.parse_args().config)
    print(
        json.dumps(
            fetch_pinned_repository(
                config.source,
                allowed_untracked_paths=(config.parameters_path, config.state_dict_path),
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
