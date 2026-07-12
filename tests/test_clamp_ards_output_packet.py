from __future__ import annotations

import json
import subprocess
import zipfile
from pathlib import Path

import pandas as pd
import pytest

from ards_cxr_benchmark.clamp_ards_output_packet import prepare_clamp_txt_output_packet

HEADER = b"Start\tEnd\tSemantic\tCUI\tAssertion\tEntity\n"
POSITIVE = HEADER + b"1\t8\tARDS\tnull\tpresent\topacity\n"


def test_prepare_packet_copies_only_manifest_matched_txt_bytes(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path, ["s1", "s2"])
    source = tmp_path / "combined.zip"
    with zipfile.ZipFile(source, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("Output/s1.txt", POSITIVE)
        archive.writestr("Output/s1.xmi", b"<xmi />")
        archive.writestr("Output/s2.txt", HEADER)
        archive.writestr("Output/s2.xmi", b"<xmi />")

    destination = tmp_path / "restricted" / "txt-only.zip"
    summary_path = tmp_path / "restricted" / "summary.json"
    summary = prepare_clamp_txt_output_packet(
        source_archive=source,
        output_archive=destination,
        input_manifest_path=manifest,
        summary_output=summary_path,
    )

    with zipfile.ZipFile(destination) as archive:
        names = [info.filename for info in archive.infolist() if not info.is_dir()]
        assert names == ["Output/s1.txt", "Output/s2.txt"]
        assert archive.read("Output/s1.txt") == POSITIVE
        assert archive.read("Output/s2.txt") == HEADER
    assert summary["source_txt_members"] == 2
    assert summary["source_xmi_members_excluded"] == 2
    assert summary["output_txt_members"] == 2
    assert len(summary["source_archive_sha256"]) == 64
    assert len(summary["output_archive_sha256"]) == 64
    assert json.loads(summary_path.read_text(encoding="utf-8")) == summary


def test_prepare_packet_rejects_manifest_mismatch(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path, ["s1", "s2"])
    source = tmp_path / "combined.zip"
    with zipfile.ZipFile(source, "w") as archive:
        archive.writestr("Output/s1.txt", HEADER)
        archive.writestr("Output/unexpected.txt", HEADER)

    with pytest.raises(ValueError, match="missing=1, unexpected=1"):
        prepare_clamp_txt_output_packet(
            source_archive=source,
            output_archive=tmp_path / "packet.zip",
            input_manifest_path=manifest,
            summary_output=tmp_path / "summary.json",
        )


def test_prepare_packet_rejects_duplicate_normalized_document_ids(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path, ["s1"])
    source = tmp_path / "combined.zip"
    with zipfile.ZipFile(source, "w") as archive:
        archive.writestr("Output/s1.txt", HEADER)
        archive.writestr("duplicate/s1.txt", HEADER)

    with pytest.raises(ValueError, match="Duplicate CLAMP TXT output"):
        prepare_clamp_txt_output_packet(
            source_archive=source,
            output_archive=tmp_path / "packet.zip",
            input_manifest_path=manifest,
            summary_output=tmp_path / "summary.json",
        )


def test_prepare_packet_rejects_summary_collision_with_source(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path, ["s1"])
    source = _write_source_archive(tmp_path)
    original = source.read_bytes()
    destination = tmp_path / "packet.zip"

    with pytest.raises(ValueError, match="paths must be distinct"):
        prepare_clamp_txt_output_packet(
            source_archive=source,
            output_archive=destination,
            input_manifest_path=manifest,
            summary_output=source,
        )

    assert source.read_bytes() == original
    assert zipfile.is_zipfile(source)
    assert not destination.exists()


def test_prepare_packet_rejects_summary_collision_with_output(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path, ["s1"])
    source = _write_source_archive(tmp_path)
    destination = tmp_path / "packet.zip"

    with pytest.raises(ValueError, match="paths must be distinct"):
        prepare_clamp_txt_output_packet(
            source_archive=source,
            output_archive=destination,
            input_manifest_path=manifest,
            summary_output=destination,
        )

    assert zipfile.is_zipfile(source)
    assert not destination.exists()


@pytest.mark.parametrize("destination_kind", ["archive", "summary"])
def test_prepare_packet_rejects_destination_collision_with_manifest(
    tmp_path: Path, destination_kind: str
) -> None:
    manifest = _write_manifest(tmp_path, ["s1"])
    original = manifest.read_bytes()
    source = _write_source_archive(tmp_path)
    output = manifest if destination_kind == "archive" else tmp_path / "packet.zip"
    summary = manifest if destination_kind == "summary" else tmp_path / "summary.json"

    with pytest.raises(ValueError, match="paths must be distinct"):
        prepare_clamp_txt_output_packet(
            source_archive=source,
            output_archive=output,
            input_manifest_path=manifest,
            summary_output=summary,
            overwrite=True,
        )

    assert manifest.read_bytes() == original
    assert not (tmp_path / "packet.zip").exists()
    assert not (tmp_path / "summary.json").exists()


def test_prepare_packet_rejects_symlink_path_alias(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path, ["s1"])
    source = _write_source_archive(tmp_path)
    source_alias = tmp_path / "source-alias.zip"
    source_alias.symlink_to(source)

    with pytest.raises(ValueError, match="paths must be distinct"):
        prepare_clamp_txt_output_packet(
            source_archive=source,
            output_archive=tmp_path / "packet.zip",
            input_manifest_path=manifest,
            summary_output=source_alias,
        )

    assert zipfile.is_zipfile(source)


def test_combined_archive_is_defensively_ignored() -> None:
    result = subprocess.run(
        ["git", "check-ignore", "--quiet", "ARDS CLAMP Output.zip"],
        check=False,
    )
    assert result.returncode == 0


def _write_manifest(tmp_path: Path, doc_ids: list[str]) -> Path:
    path = tmp_path / "input_manifest.csv"
    pd.DataFrame({"clamp_doc_id": doc_ids}).to_csv(path, index=False)
    return path


def _write_source_archive(tmp_path: Path) -> Path:
    source = tmp_path / "source.zip"
    with zipfile.ZipFile(source, "w") as archive:
        archive.writestr("Output/s1.txt", HEADER)
    return source
