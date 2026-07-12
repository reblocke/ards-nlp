from __future__ import annotations

from pathlib import Path

import pytest

from ards_cxr_benchmark.annotation_report_privacy import REPORTS, validate_smoke_reports


def test_smoke_report_validator_accepts_aggregate_reports(tmp_path: Path) -> None:
    for relative in REPORTS:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("<html>aggregate synthetic report</html>", encoding="utf-8")

    validate_smoke_reports(tmp_path)


def test_smoke_report_validator_rejects_identifiers(tmp_path: Path) -> None:
    fixture = tmp_path / "tests/fixtures/redcap_annotation/rater_01.csv"
    fixture.parent.mkdir(parents=True)
    fixture.write_text(
        "id,id2,id_accession\nR01-008,SYN-008,000008\n",
        encoding="utf-8",
    )
    for relative in REPORTS:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("<html>aggregate synthetic report</html>", encoding="utf-8")
    (tmp_path / REPORTS[0]).write_text("<html>case 000008</html>", encoding="utf-8")

    with pytest.raises(ValueError, match="forbidden"):
        validate_smoke_reports(tmp_path)


@pytest.mark.parametrize(
    "machine_path",
    [
        "/home/runner/work/ards-nlp/report.csv",
        "/workspace/ards-nlp/report.csv",
        "/data/ards-nlp/report.csv",
        "D:\\research\\ards-nlp\\report.csv",
    ],
)
def test_smoke_report_validator_rejects_machine_paths(tmp_path: Path, machine_path: str) -> None:
    for relative in REPORTS:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("<html>aggregate synthetic report</html>", encoding="utf-8")
    (tmp_path / REPORTS[0]).write_text(f"<html>{machine_path}</html>", encoding="utf-8")

    with pytest.raises(ValueError, match="absolute_machine_path"):
        validate_smoke_reports(tmp_path)


def test_smoke_report_validator_allows_web_urls(tmp_path: Path) -> None:
    for relative in REPORTS:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            '<html><a href="https://example.org/home/project/report">source</a></html>',
            encoding="utf-8",
        )

    validate_smoke_reports(tmp_path)
