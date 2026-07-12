from __future__ import annotations

import zipfile
from pathlib import Path

import pandas as pd
import pytest

from ards_cxr_benchmark.clamp_ards_output_archive import (
    parse_clamp_ards_output_archive,
    parse_clamp_txt_payload,
)
from ards_cxr_benchmark.clamp_ards_outputs import parse_clamp_ards_outputs

HEADER = "Start\tEnd\tSemantic\tCUI\tAssertion\tEntity\n"
POSITIVE = HEADER + "1\t8\tARDS\tnull\tpresent\topacity\n"


def test_archive_parser_writes_positive_and_header_only_negative(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path, ["s1", "s2"])
    archive = _write_archive(tmp_path, {"s1": POSITIVE, "s2": HEADER})
    outputs = _outputs(tmp_path)

    summary = parse_clamp_ards_output_archive(
        input_manifest_path=manifest,
        output_archive=archive,
        **outputs,
    )

    entities = pd.read_parquet(outputs["entity_output"])
    predictions = pd.read_parquet(outputs["prediction_output"]).set_index("clamp_doc_id")
    probabilistic = pd.read_parquet(outputs["probabilistic_prediction_output"])
    audit = pd.read_csv(outputs["audit_output"]).set_index("clamp_doc_id")
    assert len(entities) == 1
    assert entities.iloc[0]["entity_text"] == "opacity"
    assert "attribute" in entities.columns
    assert pd.isna(entities.iloc[0]["attribute"])
    assert predictions.loc["s1", "prediction_label"] == 1
    assert predictions.loc["s2", "prediction_label"] == 0
    assert predictions.loc["s2", "clamp_parse_status"] == "parsed_empty"
    assert audit.loc["s2", "parse_status"] == "parsed_empty"
    assert len(probabilistic) == 2
    assert not any("text" in column.lower() for column in predictions.columns)
    assert summary["entity_rows"] == 1
    assert summary["documents_with_ards_entity"] == 1
    assert summary["documents_without_ards_entity"] == 1
    assert summary["semantic_tag_counts"] == {"ARDS": 1}
    assert summary["assertion_counts"] == {"present": 1}


def test_archive_and_directory_parsers_normalize_equivalent_outputs(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path, ["s1", "s2"])
    archive = _write_archive(tmp_path, {"s1": POSITIVE, "s2": HEADER})
    archive_outputs = _outputs(tmp_path)
    parse_clamp_ards_output_archive(
        input_manifest_path=manifest,
        output_archive=archive,
        **archive_outputs,
    )

    directory = tmp_path / "directory"
    directory.mkdir()
    (directory / "s1.txt").write_text(POSITIVE, encoding="utf-8")
    (directory / "s2.txt").write_text(HEADER, encoding="utf-8")
    directory_result = parse_clamp_ards_outputs(
        input_manifest_path=manifest,
        output_dir=directory,
    )

    archive_entities = pd.read_parquet(archive_outputs["entity_output"])
    archive_predictions = pd.read_parquet(archive_outputs["prediction_output"])
    entity_columns = [
        "clamp_doc_id",
        "start",
        "end",
        "semantic_tag",
        "assertion",
        "cui",
        "attribute",
        "entity_text",
    ]
    pd.testing.assert_frame_equal(
        archive_entities[entity_columns].fillna("<missing>").reset_index(drop=True),
        directory_result.entities[entity_columns].fillna("<missing>").reset_index(drop=True),
        check_dtype=False,
    )
    prediction_columns = [
        "clamp_doc_id",
        "prediction_score",
        "prediction_label",
        "clamp_ards_entity_count",
    ]
    pd.testing.assert_frame_equal(
        archive_predictions[prediction_columns].reset_index(drop=True),
        directory_result.predictions[prediction_columns].reset_index(drop=True),
        check_dtype=False,
    )


def test_archive_parse_failure_preserves_existing_outputs(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path, ["s1"])
    archive = _write_archive(tmp_path, {"s1": "Start\tEnd\tSemantic\n1\t8\tARDS\n"})
    outputs = _outputs(tmp_path)
    outputs["prediction_output"].parent.mkdir(parents=True)
    outputs["prediction_output"].write_text("existing", encoding="utf-8")

    with pytest.raises(ValueError, match="parse failures"):
        parse_clamp_ards_output_archive(
            input_manifest_path=manifest,
            output_archive=archive,
            **outputs,
        )

    assert outputs["prediction_output"].read_text(encoding="utf-8") == "existing"
    assert not list(tmp_path.rglob("*.partial"))


def test_archive_parser_requires_exact_manifest_match(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path, ["s1", "s2"])
    archive = _write_archive(tmp_path, {"s1": HEADER, "unexpected": HEADER})

    with pytest.raises(ValueError, match="missing=1, unexpected=1"):
        parse_clamp_ards_output_archive(
            input_manifest_path=manifest,
            output_archive=archive,
            **_outputs(tmp_path),
        )


def test_archive_parser_rejects_output_collision_with_source_archive(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path, ["s1"])
    archive = _write_archive(tmp_path, {"s1": HEADER})
    original = archive.read_bytes()
    outputs = _outputs(tmp_path)
    outputs["entity_output"] = archive

    with pytest.raises(ValueError, match="paths must be distinct"):
        parse_clamp_ards_output_archive(
            input_manifest_path=manifest,
            output_archive=archive,
            **outputs,
        )

    assert archive.read_bytes() == original
    assert zipfile.is_zipfile(archive)
    assert not list(tmp_path.rglob("*.partial"))


def test_archive_parser_rejects_manifest_collision_with_source_archive(tmp_path: Path) -> None:
    archive = _write_archive(tmp_path, {"s1": HEADER})
    original = archive.read_bytes()

    with pytest.raises(ValueError, match="paths must be distinct"):
        parse_clamp_ards_output_archive(
            input_manifest_path=archive,
            output_archive=archive,
            **_outputs(tmp_path),
        )

    assert archive.read_bytes() == original
    assert zipfile.is_zipfile(archive)
    assert not list(tmp_path.rglob("*.partial"))


def test_archive_parser_rejects_duplicate_output_destinations(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path, ["s1"])
    archive = _write_archive(tmp_path, {"s1": HEADER})
    outputs = _outputs(tmp_path)
    outputs["audit_output"] = outputs["prediction_output"]

    with pytest.raises(ValueError, match="paths must be distinct"):
        parse_clamp_ards_output_archive(
            input_manifest_path=manifest,
            output_archive=archive,
            **outputs,
        )

    assert zipfile.is_zipfile(archive)
    assert not outputs["prediction_output"].exists()
    assert not list(tmp_path.rglob("*.partial"))


def test_archive_parser_rejects_summary_markdown_collision(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path, ["s1"])
    archive = _write_archive(tmp_path, {"s1": HEADER})
    outputs = _outputs(tmp_path)
    outputs["entity_output"] = outputs["summary_output"].with_suffix(".md")

    with pytest.raises(ValueError, match="paths must be distinct"):
        parse_clamp_ards_output_archive(
            input_manifest_path=manifest,
            output_archive=archive,
            **outputs,
        )

    assert zipfile.is_zipfile(archive)
    assert not list(tmp_path.rglob("*.partial"))


def test_payload_parser_rejects_invalid_offsets() -> None:
    parsed = parse_clamp_txt_payload(
        (HEADER + "not-an-offset\t8\tARDS\tnull\tpresent\topacity\n").encode()
    )

    assert parsed.parse_status == "parse_error"
    assert parsed.parse_error == "invalid_offsets:row_2"


def test_payload_parser_preserves_attribute_when_exported() -> None:
    payload = (
        b"Start\tEnd\tSemantic\tCUI\tAssertion\tAttribute\tEntity\n"
        b"1\t8\tARDS\tnull\tpresent\tlegacy-value\topacity\n"
    )

    parsed = parse_clamp_txt_payload(payload)

    assert parsed.parse_status == "parsed"
    assert parsed.entities[0]["attribute"] == "legacy-value"


def _write_archive(tmp_path: Path, documents: dict[str, str]) -> Path:
    path = tmp_path / "outputs.zip"
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for doc_id, payload in documents.items():
            archive.writestr(f"Output/{doc_id}.txt", payload)
    return path


def _write_manifest(tmp_path: Path, doc_ids: list[str]) -> Path:
    path = tmp_path / "input_manifest.csv"
    pd.DataFrame(
        {
            "clamp_doc_id": doc_ids,
            "input_file": [f"/input/{doc_id}.txt" for doc_id in doc_ids],
            "source_dataset": ["mimic_cxr"] * len(doc_ids),
            "subject_id": [str(100 + index) for index in range(len(doc_ids))],
            "study_id": [str(200 + index) for index in range(len(doc_ids))],
            "accession_id": [""] * len(doc_ids),
            "encounter_id": [""] * len(doc_ids),
            "annotation_phase": ["stage1"] * len(doc_ids),
        }
    ).to_csv(path, index=False)
    return path


def _outputs(tmp_path: Path) -> dict[str, Path]:
    return {
        "entity_output": tmp_path / "derived" / "entities.parquet",
        "prediction_output": tmp_path / "derived" / "predictions.parquet",
        "probabilistic_prediction_output": tmp_path / "derived" / "probabilistic.parquet",
        "audit_output": tmp_path / "artifacts" / "audit.csv",
        "summary_output": tmp_path / "artifacts" / "summary.json",
    }
