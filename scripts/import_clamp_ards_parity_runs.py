from __future__ import annotations

import argparse
import json
from pathlib import Path

from ards_cxr_benchmark.clamp_ards.legacy_runs import import_legacy_clamp_parity_runs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Strictly import two returned legacy CLAMP synthetic parity runs"
    )
    parser.add_argument(
        "--fixture-root",
        type=Path,
        default=Path("artifacts/restricted/clamp_ards/python/pending_fixture"),
    )
    parser.add_argument("--run-1", type=Path, required=True)
    parser.add_argument("--run-2", type=Path, required=True)
    parser.add_argument("--run-1-provenance", type=Path, required=True)
    parser.add_argument("--run-2-provenance", type=Path, required=True)
    parser.add_argument(
        "--run-1-sha256s",
        type=Path,
        help="Returned checksum manifest; defaults beside run-1 provenance.",
    )
    parser.add_argument(
        "--run-2-sha256s",
        type=Path,
        help="Returned checksum manifest; defaults beside run-2 provenance.",
    )
    parser.add_argument(
        "--candidate-output-dir",
        type=Path,
        default=Path("artifacts/restricted/clamp_ards/parity_fixture_candidate"),
    )
    parser.add_argument(
        "--resource-manifest",
        type=Path,
        default=Path("config/clamp_ards_resource_manifest.json"),
    )
    parser.add_argument(
        "--finalize",
        action="store_true",
        help="Install the candidate into the tracked fixture after explicit reviews.",
    )
    parser.add_argument("--phi-reviewer")
    parser.add_argument("--phi-reviewed-at")
    parser.add_argument("--redistribution-authority")
    parser.add_argument("--redistribution-evidence")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = import_legacy_clamp_parity_runs(
        fixture_root=args.fixture_root,
        run_1_source=args.run_1,
        run_2_source=args.run_2,
        run_1_provenance=args.run_1_provenance,
        run_2_provenance=args.run_2_provenance,
        run_1_sha256s=args.run_1_sha256s,
        run_2_sha256s=args.run_2_sha256s,
        candidate_output_dir=args.candidate_output_dir,
        resource_manifest=args.resource_manifest,
        finalize=args.finalize,
        phi_reviewer=args.phi_reviewer,
        phi_reviewed_at=args.phi_reviewed_at,
        redistribution_authority=args.redistribution_authority,
        redistribution_evidence=args.redistribution_evidence,
    )
    print(json.dumps(summary.as_dict(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
