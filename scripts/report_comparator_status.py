from __future__ import annotations

import json

from ards_cxr_benchmark.comparators.common import load_status_files
from ards_cxr_benchmark.paths import get_paths

EXPECTED = {
    "amaral_xgboost_bilateral_infiltrates",
    "uw_hanso_bilateral_infiltrates",
    "afshar_text_svc_full_ards",
}


def main() -> None:
    root = get_paths().root
    statuses = load_status_files(sorted((root / "artifacts/comparators").glob("*/status.json")))
    observed = {str(status.get("name")) for status in statuses}
    for name in sorted(EXPECTED - observed):
        statuses.append(
            {
                "name": name,
                "status": "not_run",
                "reason": "resource verification has not been run",
                "details": {},
            }
        )
    print(json.dumps(sorted(statuses, key=lambda value: str(value.get("name", ""))), indent=2))


if __name__ == "__main__":
    main()
