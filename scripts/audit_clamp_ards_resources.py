from __future__ import annotations

import argparse
import json
from pathlib import Path

from ards_cxr_benchmark.clamp_ards.governance import audit_clamp_resources


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate the file-by-file ARDS CLAMP redistribution ledger"
    )
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--public-release",
        action="store_true",
        help="Fail while an excluded or unresolved resource remains tracked",
    )
    parser.add_argument("--summary-output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = audit_clamp_resources(args.root, public_release=args.public_release)
    payload = result.as_dict()
    if args.summary_output:
        args.summary_output.parent.mkdir(parents=True, exist_ok=True)
        args.summary_output.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    for finding in result.findings:
        print(f"[{finding.check}] {finding.path}: {finding.detail}")
    mode = "public-release" if args.public_release else "private-review"
    print(
        f"CLAMP resource audit ({mode}): files={result.file_count}; "
        f"unresolved={result.unresolved_count}; tracked={result.tracked_resource_count}; "
        f"public_release_blocked={str(result.public_release_blocked).lower()}"
    )
    if result.findings:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
