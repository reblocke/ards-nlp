from __future__ import annotations

import json
from pathlib import Path

from ards_cxr_benchmark.readiness import collect_readiness, readiness_has_blockers
from ards_cxr_benchmark.release_audit import (
    audit_historical_paths,
    audit_local_markdown_links,
    audit_machine_paths,
    audit_tracked_paths,
)


def test_annotation_readiness_reports_missing_real_config(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        "ards_cxr_benchmark.readiness.shutil.which", lambda command: f"/bin/{command}"
    )
    monkeypatch.setattr(
        "ards_cxr_benchmark.readiness._quarto_check",
        lambda: _check("annotation", "Quarto", "ready", "1.8.26"),
    )

    checks = collect_readiness(tmp_path, use_case="annotation")

    assert readiness_has_blockers(checks)
    assert any(check.check == "real pilot config" and check.status == "blocked" for check in checks)


def test_comparator_readiness_requires_expected_local_inputs(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        "ards_cxr_benchmark.readiness.shutil.which", lambda command: f"/bin/{command}"
    )
    required = [
        tmp_path / "config/config.yaml",
        tmp_path / "data/processed/mimic_cxr_reports.parquet",
        tmp_path / "data/derived/modeling/model_development_extract.parquet",
        tmp_path / "data/derived/clamp_ards/clamp_legacy_predictions.parquet",
        tmp_path / "data/derived/clamp_ards/clamp_legacy_entities.parquet",
        tmp_path / "data/derived/modeling/silver_baseline_predictions.parquet",
    ]
    for path in required:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()

    checks = collect_readiness(tmp_path, use_case="comparators")

    assert not readiness_has_blockers(checks)


def test_release_audit_rejects_restricted_paths_and_binary_outputs() -> None:
    findings = audit_tracked_paths(
        [
            "README.md",
            "config/config.yaml",
            "data/derived/predictions.parquet",
            "docs/internal-presentation.pptx",
        ]
    )

    assert {finding.check for finding in findings} == {"restricted_path", "forbidden_extension"}
    assert {finding.path for finding in findings if finding.check == "forbidden_extension"} == {
        "data/derived/predictions.parquet",
        "docs/internal-presentation.pptx",
    }


def test_release_audit_allows_only_known_synthetic_historical_data() -> None:
    findings = audit_historical_paths(
        [
            "data/raw/example.csv",
            "data/raw/.gitkeep",
            "data/raw/real_report.txt",
            "docs/internal-presentation.pptx",
        ]
    )

    assert {finding.path for finding in findings} == {
        "data/raw/real_report.txt",
        "docs/internal-presentation.pptx",
    }


def test_release_audit_detects_machine_paths(tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "mac.md").write_text("Source: /Users/example/private/file.pdf\n", encoding="utf-8")
    (docs / "linux.md").write_text("Source: /home/alice/project/file.csv\n", encoding="utf-8")
    (docs / "windows.md").write_text("Source: D:\\research\\project\\file.csv\n", encoding="utf-8")

    findings = audit_machine_paths(
        tmp_path,
        ["docs/mac.md", "docs/linux.md", "docs/windows.md"],
    )

    assert len(findings) == 3
    assert {finding.check for finding in findings} == {"machine_path"}


def test_release_audit_allows_documented_clamp_runtime_path(tmp_path: Path) -> None:
    doc = tmp_path / "docs" / "clamp.md"
    doc.parent.mkdir()
    doc.write_text("Runtime: C:/ClampWin_1.6.6/workspace/ARDS\n", encoding="utf-8")

    assert audit_machine_paths(tmp_path, ["docs/clamp.md"]) == []


def test_release_audit_detects_machine_paths_in_json_csv_tsv_and_txt(tmp_path: Path) -> None:
    files = {
        "fixture/provenance.json": json.dumps(
            {"source": r"C:\Users\Jane Doe\private fixture\run_1"}
        ),
        "fixture/manifest.csv": 'source\n"/Users/Jane Doe/private fixture/input.txt"\n',
        "fixture/entities.tsv": "source\tstatus\n/home/jane/private fixture/output\tready\n",
        "fixture/notes.txt": r"Returned from \\research-host\private share\run_1",
    }
    for relative, content in files.items():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    findings = audit_machine_paths(tmp_path, list(files))

    assert {finding.path for finding in findings} == set(files)
    assert {finding.check for finding in findings} == {"machine_path"}


def test_release_audit_allows_relative_and_resource_text_paths(tmp_path: Path) -> None:
    files = {
        "fixture/provenance.json": json.dumps(
            {"resource": "Components/Assertion classifier/defaultNegexDict.txt"}
        ),
        "fixture/manifest.csv": "input_path\ninput/dict_case_01_lower.txt\n",
        "resources/tokenizer.txt": r"DELIMETER=/DEL\d+DEL\/DEL\d+",
        "resources/config.conf": "<params><dictFile>defaultNegexDict.txt</dictFile></params>",
    }
    for relative, content in files.items():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    assert audit_machine_paths(tmp_path, list(files)) == []


def test_release_audit_detects_broken_local_markdown_links(tmp_path: Path) -> None:
    readme = tmp_path / "README.md"
    readme.write_text("See [missing](docs/missing.md).\n", encoding="utf-8")

    findings = audit_local_markdown_links(tmp_path, ["README.md"])

    assert len(findings) == 1
    assert findings[0].check == "broken_link"


def _check(use_case: str, check: str, status: str, detail: str):
    from ards_cxr_benchmark.readiness import ReadinessCheck

    return ReadinessCheck(use_case, check, status, detail)
