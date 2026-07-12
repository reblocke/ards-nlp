from __future__ import annotations

import argparse
import json
from pathlib import Path

from ards_cxr_benchmark.comparators.amaral import (
    verify_amaral_resources,
    write_published_preprocessing_config,
)
from ards_cxr_benchmark.comparators.common import write_comparator_status
from ards_cxr_benchmark.comparators.config import load_amaral_config
from ards_cxr_benchmark.config import ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify pinned Amaral model resources")
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    config = load_amaral_config(parse_args().config)
    result = verify_amaral_resources(config)
    preprocessing = write_published_preprocessing_config(config)
    result["preprocessing_config_sha256"] = preprocessing["source_notebook_sha256"]
    ensure_dir(config.artifact_dir)
    (config.artifact_dir / "resource_verification.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    write_comparator_status(
        name=config.source.name,
        status="resources_verified",
        reason="pinned source and model checksums verified",
        out_path=config.artifact_dir / "status.json",
        details={"commit": config.source.commit},
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
