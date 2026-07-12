from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from ards_cxr_benchmark.text_sections import (
    build_report_rows,
    build_target_texts,
    extract_last_paragraph,
    extract_section,
    normalize_report_text,
    parse_subject_study_from_path,
)


def test_parse_subject_study_from_mimic_path() -> None:
    path = Path("files/p10/p10000032/s50414267.txt")
    assert parse_subject_study_from_path(path) == (10000032, 50414267)


def test_parse_subject_study_rejects_unmapped_path() -> None:
    with pytest.raises(ValueError, match="Could not parse"):
        parse_subject_study_from_path(Path("bad/report.txt"))


def test_extract_sections_and_last_paragraph() -> None:
    text = normalize_report_text(
        textwrap.dedent(
            """
        FINAL REPORT

        FINDINGS:
        Low lung volumes. Bibasilar opacities are present.

        IMPRESSION:
        Bilateral airspace opacities.
        """
        )
    )

    assert extract_section(text, "FINDINGS") == "Low lung volumes. Bibasilar opacities are present."
    assert extract_section(text, "IMPRESSION") == "Bilateral airspace opacities."
    assert extract_last_paragraph(text) == "IMPRESSION:\nBilateral airspace opacities."


def test_build_target_texts_uses_full_report_primary_scope() -> None:
    target = build_target_texts(
        "full report",
        findings_text="findings",
        impression_text="impression",
        last_paragraph_text="last",
        primary_scope="full_report",
    )
    assert target.primary_target_text == "full report"
    assert target.target_text_impression_findings == "impression\nfindings"
    assert target.target_text_impression_fallback == "impression"


def test_build_report_rows_from_directory(tmp_path: Path) -> None:
    report_dir = tmp_path / "files" / "p10" / "p10000032"
    report_dir.mkdir(parents=True)
    report_path = report_dir / "s50414267.txt"
    report_path.write_text(
        "FINAL REPORT\n\nFINDINGS:\nClear lungs.\n\nIMPRESSION:\nNo acute process.\n",
        encoding="utf-8",
    )

    rows = build_report_rows(tmp_path)

    assert rows.shape[0] == 1
    assert rows.loc[0, "subject_id"] == 10000032
    assert rows.loc[0, "study_id"] == 50414267
    assert rows.loc[0, "primary_target_text"] == rows.loc[0, "report_text"]
