from __future__ import annotations

import argparse
from pathlib import Path

from ards_cxr_benchmark.clamp_ards_inputs import sync_clamp_ards_project
from ards_cxr_benchmark.config import (
    default_config_path,
    load_config,
    validate_clamp_ards_operational_paths,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Copy the tracked ARDS CLAMP project into a local CLAMP workspace"
    )
    parser.add_argument("--config", type=Path, default=default_config_path())
    parser.add_argument("--clamp-project-root", type=Path)
    parser.add_argument("--project-source-dir", type=Path)
    parser.add_argument("--project-live-dir", type=Path)
    parser.add_argument(
        "--runtime-project-dir",
        help="Path that the staged ARDS project will have on the CLAMP machine.",
    )
    parser.add_argument("--artifact-dir", type=Path)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    source_dir = args.project_source_dir or config.clamp_ards.project_source_dir
    if args.project_live_dir is not None:
        live_dir = args.project_live_dir
    elif args.clamp_project_root is not None:
        live_dir = args.clamp_project_root / config.clamp_ards.project_name
    else:
        live_dir = config.clamp_ards.project_live_dir
    artifact_dir = args.artifact_dir or config.clamp_ards.restricted_artifact_dir
    summary_path = args.summary or artifact_dir / "project_sync_summary.json"
    runtime_project_dir = args.runtime_project_dir or config.clamp_ards.runtime_project_dir
    validate_clamp_ards_operational_paths(project_live_dir=live_dir)
    summary = sync_clamp_ards_project(
        source_dir=source_dir,
        live_dir=live_dir,
        runtime_project_dir=runtime_project_dir,
        artifact_dir=artifact_dir,
        summary_path=summary_path,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
    )
    print(
        "Synced ARDS CLAMP project "
        f"({summary['rendered_file_count']} rendered, "
        f"{summary['unchanged_file_count']} unchanged) to {live_dir}"
    )


if __name__ == "__main__":
    main()
