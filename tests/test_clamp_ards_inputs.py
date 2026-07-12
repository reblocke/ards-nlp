from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from ards_cxr_benchmark.clamp_ards_inputs import (
    export_clamp_ards_inputs,
    invalid_windows_filename_stem_reason,
)


def test_valid_rows_write_exact_text_and_manifest_has_no_report_text(tmp_path: Path) -> None:
    df = pd.DataFrame(
        {
            "doc_id": ["s1", "s2"],
            "report": ["Line 1\nLine 2 <BR>", "No edema. Bilateral opacities."],
            "subject_id": [1, 2],
            "study_id": [10, 20],
        }
    )

    result = _export(df, tmp_path, id_col="doc_id", text_col="report")

    assert (tmp_path / "clamp" / "Input" / "s1.txt").read_text(encoding="utf-8") == (
        "Line 1\nLine 2 <BR>"
    )
    assert result.summary["written_files"] == 2
    assert "Do not transfer the full BigQuery/model-development dataset" in result.handoff_markdown
    assert "report" not in result.manifest.columns
    assert len(pd.read_csv(tmp_path / "artifacts" / "input_manifest.csv")) == 2


def test_missing_text_and_unsafe_ids_are_skipped_and_counted(tmp_path: Path) -> None:
    df = pd.DataFrame(
        {
            "doc_id": ["ok", "bad/name", "missing"],
            "report": ["text", "text", ""],
        }
    )

    result = _export(df, tmp_path, id_col="doc_id", text_col="report")

    assert result.summary["written_files"] == 1
    assert result.summary["unsafe_doc_id_rows"] == 1
    assert result.summary["missing_text_rows"] == 1
    assert not (tmp_path / "clamp" / "Input" / "bad/name.txt").exists()


def test_duplicate_included_doc_ids_fail_before_writing_text(tmp_path: Path) -> None:
    df = pd.DataFrame({"doc_id": ["dup", "dup"], "report": ["one", "two"]})

    with pytest.raises(ValueError, match="Duplicate clamp_doc_id"):
        _export(df, tmp_path, id_col="doc_id", text_col="report")

    assert not (tmp_path / "clamp" / "Input" / "dup.txt").exists()


def test_dry_run_writes_restricted_manifest_but_no_text_files(tmp_path: Path) -> None:
    df = pd.DataFrame({"doc_id": ["s1"], "report": ["text"]})

    result = _export(df, tmp_path, id_col="doc_id", text_col="report", dry_run=True)

    assert result.summary["written_files"] == 1
    assert (tmp_path / "artifacts" / "input_manifest.csv").exists()
    assert not (tmp_path / "clamp" / "Input" / "s1.txt").exists()


def test_existing_files_require_overwrite_or_clear(tmp_path: Path) -> None:
    df = pd.DataFrame({"doc_id": ["s1"], "report": ["text"]})
    existing = tmp_path / "clamp" / "Input" / "s1.txt"
    existing.parent.mkdir(parents=True)
    existing.write_text("old", encoding="utf-8")

    with pytest.raises(FileExistsError):
        _export(df, tmp_path, id_col="doc_id", text_col="report")

    _export(df, tmp_path, id_col="doc_id", text_col="report", overwrite=True)
    assert existing.read_text(encoding="utf-8") == "text"


def test_clear_existing_inputs_can_archive_stale_text_files(tmp_path: Path) -> None:
    df = pd.DataFrame({"doc_id": ["s1"], "report": ["text"]})
    stale = tmp_path / "clamp" / "Input" / "old.txt"
    stale.parent.mkdir(parents=True)
    stale.write_text("old", encoding="utf-8")

    _export(
        df,
        tmp_path,
        id_col="doc_id",
        text_col="report",
        clear_existing_inputs=True,
        archive_cleared_files=True,
    )

    assert not stale.exists()
    assert list((tmp_path / "clamp" / "Input" / "Archive").glob("*/*.txt"))


def test_template_doc_id_rejects_missing_study_id(tmp_path: Path) -> None:
    df = pd.DataFrame({"study_id": [None], "report": ["text"]})

    with pytest.raises(ValueError, match="No CLAMP input files"):
        _export(df, tmp_path, text_col="report", doc_id_template="s{study_id}", dry_run=True)

    manifest = pd.read_csv(tmp_path / "artifacts" / "input_manifest.csv")
    assert manifest.loc[0, "export_status"] == "skipped"
    assert manifest.loc[0, "skip_reason"] == "unsafe_doc_id:missing"


@pytest.mark.parametrize("value", ["CON", "a/b", "bad:", "trail.", " lead", ".."])
def test_windows_filename_reasons_reject_unsafe_values(value: str) -> None:
    assert invalid_windows_filename_stem_reason(value) is not None


def _export(
    df: pd.DataFrame,
    tmp_path: Path,
    *,
    id_col: str | None = None,
    text_col: str,
    doc_id_template: str | None = None,
    overwrite: bool = False,
    dry_run: bool = False,
    clear_existing_inputs: bool = False,
    archive_cleared_files: bool = False,
):
    return export_clamp_ards_inputs(
        df,
        input_dir=tmp_path / "clamp" / "Input",
        output_dir=tmp_path / "clamp" / "Output",
        artifact_dir=tmp_path / "artifacts",
        manifest_path=tmp_path / "artifacts" / "input_manifest.csv",
        summary_path=tmp_path / "artifacts" / "input_summary.json",
        handoff_path=tmp_path / "artifacts" / "NEXT_STEP_RUN_CLAMP.md",
        text_col=text_col,
        source_type="synthetic",
        source_name="fixture",
        command="export-test",
        project_live_dir=tmp_path / "clamp",
        id_col=id_col,
        doc_id_template=doc_id_template,
        overwrite=overwrite,
        clear_existing_inputs=clear_existing_inputs,
        clear_existing_outputs=False,
        archive_cleared_files=archive_cleared_files,
        dry_run=dry_run,
    )
