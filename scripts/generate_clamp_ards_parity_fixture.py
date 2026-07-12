from __future__ import annotations

import argparse
import json
from pathlib import Path

from ards_cxr_benchmark.clamp_ards.fixtures import generate_fixture, validate_fixture


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate or validate the synthetic ARDS CLAMP golden-corpus scaffold"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser(
        "generate",
        help="Generate deterministic synthetic inputs without expected annotations",
    )
    generate.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/restricted/clamp_ards/python/pending_fixture"),
    )
    generate.add_argument(
        "--project-dir",
        type=Path,
        default=Path("data/external/clamp_ards_project"),
    )
    generate.add_argument(
        "--resource-manifest",
        type=Path,
        default=Path("config/clamp_ards_resource_manifest.json"),
    )
    generate.add_argument(
        "--force",
        action="store_true",
        help="Replace only an existing scaffold carrying generated-fixture provenance",
    )

    validate = subparsers.add_parser(
        "validate",
        help="Validate fixture inventory, hashes, lifecycle, and review gates",
    )
    validate.add_argument(
        "--fixture",
        type=Path,
        default=Path("artifacts/restricted/clamp_ards/python/pending_fixture"),
    )
    validate.add_argument(
        "--allow-pending",
        action="store_true",
        help="Allow an awaiting_legacy_runs scaffold; strict validation rejects it",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "generate":
        result = generate_fixture(
            args.output,
            project_dir=args.project_dir,
            resource_manifest_path=args.resource_manifest,
            force=args.force,
        )
    else:
        result = validate_fixture(args.fixture, allow_pending=args.allow_pending)
    print(json.dumps(result.as_dict(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
