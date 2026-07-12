from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

from ards_cxr_benchmark.comparators.config import load_amaral_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create the isolated Amaral runtime")
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    config = load_amaral_config(parse_args().config)
    subprocess.run(["uv", "sync", "--project", str(config.environment_dir)], check=True)
    config.nltk_data_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["NLTK_DATA"] = str(config.nltk_data_dir)
    subprocess.run(
        [
            "uv",
            "run",
            "--project",
            str(config.environment_dir),
            "python",
            "-m",
            "nltk.downloader",
            "-d",
            str(config.nltk_data_dir),
            "punkt",
            "punkt_tab",
        ],
        check=True,
        env=env,
    )
    print(f"Prepared isolated Amaral runtime at {config.environment_dir}")


if __name__ == "__main__":
    main()
