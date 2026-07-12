from __future__ import annotations

from ards_cxr_benchmark.annotation_report_privacy import validate_smoke_reports
from ards_cxr_benchmark.paths import get_paths


def main() -> None:
    validate_smoke_reports(get_paths().root)
    print("Synthetic annotation report privacy audit passed")


if __name__ == "__main__":
    main()
