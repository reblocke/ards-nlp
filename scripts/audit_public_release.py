from __future__ import annotations

from ards_cxr_benchmark.paths import get_paths
from ards_cxr_benchmark.release_audit import audit_repository


def main() -> None:
    findings = audit_repository(get_paths().root)
    if findings:
        for finding in findings:
            print(f"[{finding.check}] {finding.path}: {finding.detail}")
        raise SystemExit(f"Public release audit failed with {len(findings)} finding(s)")
    print("Public release audit passed")


if __name__ == "__main__":
    main()
