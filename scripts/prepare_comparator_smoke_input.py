from __future__ import annotations

import pandas as pd

from ards_cxr_benchmark.comparators.common import (
    REFERENCE_COLUMNS,
    build_comparator_source,
    write_comparator_input_packet,
)
from ards_cxr_benchmark.paths import get_paths


def main() -> None:
    root = get_paths().root
    reports = pd.DataFrame(
        {
            "subject_id": list(range(1, 8)),
            "study_id": list(range(101, 108)),
            "report_text": [
                "FINDINGS: The lungs are clear. IMPRESSION: No acute cardiopulmonary process.",
                "FINDINGS: Bilateral diffuse pulmonary opacities. IMPRESSION: Pulmonary edema.",
                "FINDINGS: Focal right lower lobe opacity. IMPRESSION: Right basilar pneumonia.",
                "FINDINGS: No bilateral infiltrates. IMPRESSION: No acute pulmonary disease.",
                "FINDINGS: Small bilateral pleural effusions without airspace opacity.",
                "FINDINGS: Patchy bibasilar opacities may reflect atelectasis.",
                "FINDINGS: Widespread bilateral airspace consolidation concerning for ARDS.",
            ],
            "target_text_impression_findings": [
                "The lungs are clear. No acute cardiopulmonary process.",
                "Bilateral diffuse pulmonary opacities. Pulmonary edema.",
                "Focal right lower lobe opacity. Right basilar pneumonia.",
                "No bilateral infiltrates. No acute pulmonary disease.",
                "Small bilateral pleural effusions without airspace opacity.",
                "Patchy bibasilar opacities may reflect atelectasis.",
                "Widespread bilateral airspace consolidation concerning for ARDS.",
            ],
            "target_text_impression_fallback": [
                "The lungs are clear. No acute cardiopulmonary process.",
                "Bilateral diffuse pulmonary opacities. Pulmonary edema.",
                "Focal right lower lobe opacity. Right basilar pneumonia.",
                "No bilateral infiltrates. No acute pulmonary disease.",
                "Small bilateral pleural effusions without airspace opacity.",
                "Patchy bibasilar opacities may reflect atelectasis.",
                "Widespread bilateral airspace consolidation concerning for ARDS.",
            ],
        }
    )
    model_extract = pd.DataFrame(
        {
            "subject_id": list(range(1, 8)),
            "study_id": list(range(101, 108)),
            "split": ["train", "validation", "validation", "test", "test", "test", "test"],
            "strict_bilateral_opacity_label": [0, 1, 0, 0, 0, 0, 1],
            "sensitive_bilateral_opacity_label": [0, 1, 0, 0, 0, 1, 1],
            "silver_label_source": ["synthetic"] * 7,
            "manual_review_priority": ["synthetic"] * 7,
            "qa_flags": [[] for _ in range(7)],
        }
    )
    source = build_comparator_source(reports, model_extract, expected_rows=7)
    restricted = root / "artifacts/restricted/comparators/smoke"
    write_comparator_input_packet(
        source,
        packet_path=restricted / "input.jsonl.gz",
        manifest_path=restricted / "manifest.parquet",
        summary_path=restricted / "summary.json",
    )
    reference = root / "data/derived/comparators/smoke/reference.parquet"
    reference.parent.mkdir(parents=True, exist_ok=True)
    source[REFERENCE_COLUMNS].to_parquet(reference, index=False)
    print(f"Wrote seven synthetic comparator cases under {restricted}")


if __name__ == "__main__":
    main()
