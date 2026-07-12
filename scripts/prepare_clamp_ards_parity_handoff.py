from __future__ import annotations

import argparse
import json
from pathlib import Path

from ards_cxr_benchmark.clamp_ards.legacy_runs import prepare_legacy_clamp_parity_handoff


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the deterministic manual-Windows CLAMP synthetic parity handoff"
    )
    parser.add_argument(
        "--fixture-root",
        type=Path,
        default=Path("artifacts/restricted/clamp_ards/python/pending_fixture"),
    )
    parser.add_argument(
        "--project-source-dir",
        type=Path,
        default=Path("data/external/clamp_ards_project"),
    )
    parser.add_argument(
        "--destination-dir",
        type=Path,
        default=Path("artifacts/restricted/clamp_ards/parity_handoff"),
    )
    parser.add_argument("--output-archive", type=Path)
    parser.add_argument(
        "--collector-script",
        type=Path,
        default=Path("scripts/windows/collect_clamp_ards_parity_provenance.ps1"),
    )
    parser.add_argument(
        "--resource-manifest",
        type=Path,
        default=Path("config/clamp_ards_resource_manifest.json"),
    )
    parser.add_argument("--project-commit")
    parser.add_argument(
        "--windows-project-dir",
        default=r"C:\ClampWin_1.6.6\workspace\ARDS",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = prepare_legacy_clamp_parity_handoff(
        fixture_root=args.fixture_root,
        project_source_dir=args.project_source_dir,
        destination_dir=args.destination_dir,
        output_archive=args.output_archive,
        collector_script=args.collector_script,
        resource_manifest=args.resource_manifest,
        project_commit=args.project_commit,
        expected_windows_project_dir=args.windows_project_dir,
        overwrite=args.overwrite,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
