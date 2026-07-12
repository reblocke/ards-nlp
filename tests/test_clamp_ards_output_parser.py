from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd
import pytest

from ards_cxr_benchmark.clamp_ards_outputs import (
    map_output_files_by_doc_id,
    parse_clamp_ards_outputs,
    parse_clamp_output_file,
    write_clamp_teacher_outputs,
)

FIXTURE_DIR = Path("tests/fixtures/clamp_ards_outputs")


def test_parser_builds_entities_predictions_and_audit(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path, ["s1", "s2", "missing", "bad"])
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    for name in ["s1.tsv", "s2.tsv", "bad.txt", "unexpected.tsv"]:
        shutil.copy(FIXTURE_DIR / name, output_dir / name)

    result = parse_clamp_ards_outputs(input_manifest_path=manifest, output_dir=output_dir)

    s1_prediction = result.predictions.set_index("clamp_doc_id").loc["s1"]
    s2_prediction = result.predictions.set_index("clamp_doc_id").loc["s2"]
    missing_prediction = result.predictions.set_index("clamp_doc_id").loc["missing"]
    bad_prediction = result.predictions.set_index("clamp_doc_id").loc["bad"]
    assert s1_prediction["prediction_label"] == 1
    assert s1_prediction["clamp_ards_entity_count"] == 1
    assert s2_prediction["prediction_label"] == 0
    assert s2_prediction["clamp_ards_entity_count"] == 0
    assert pd.isna(missing_prediction["prediction_label"])
    assert missing_prediction["clamp_parse_status"] == "missing_output"
    assert pd.isna(bad_prediction["prediction_label"])
    assert bad_prediction["clamp_parse_status"] == "parse_error"
    assert "entity_text" in result.entities.columns
    assert "attribute" in result.entities.columns
    assert result.entities["attribute"].isna().all()
    assert result.entities["entity_text_sha256"].notna().any()
    assert result.summary["unexpected_output_files"] == 1
    assert result.summary["missing_output_files"] == 1
    assert result.summary["parse_success_files"] == 2
    assert result.summary["parse_error_files"] == 1
    assert set(result.probabilistic_predictions["case_id"]) == {"s1", "s2"}


def test_write_outputs_keeps_prediction_tables_text_free(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path, ["s1", "s2"])
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    shutil.copy(FIXTURE_DIR / "s1.tsv", output_dir / "s1.tsv")
    shutil.copy(FIXTURE_DIR / "s2.tsv", output_dir / "s2.tsv")
    result = parse_clamp_ards_outputs(input_manifest_path=manifest, output_dir=output_dir)

    write_clamp_teacher_outputs(
        result,
        entity_output=tmp_path / "derived" / "entities.parquet",
        prediction_output=tmp_path / "derived" / "predictions.parquet",
        probabilistic_prediction_output=tmp_path / "derived" / "prob.parquet",
        audit_output=tmp_path / "artifacts" / "audit.csv",
        summary_output=tmp_path / "artifacts" / "summary.json",
    )

    predictions = pd.read_parquet(tmp_path / "derived" / "predictions.parquet")
    assert "entity_text" not in predictions.columns
    assert "entity_text_sha256" not in predictions.columns
    assert (tmp_path / "artifacts" / "summary.md").exists()


def test_parser_fails_when_no_outputs_match_manifest(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path, ["s1"])
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    shutil.copy(FIXTURE_DIR / "unexpected.tsv", output_dir / "unexpected.tsv")

    with pytest.raises(ValueError, match="No CLAMP output files matched"):
        parse_clamp_ards_outputs(input_manifest_path=manifest, output_dir=output_dir)


def test_parser_matches_outputs_with_original_txt_suffix(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path, ["s1"])
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    shutil.copy(FIXTURE_DIR / "s1.tsv", output_dir / "s1.txt.tsv")

    result = parse_clamp_ards_outputs(input_manifest_path=manifest, output_dir=output_dir)

    prediction = result.predictions.set_index("clamp_doc_id").loc["s1"]
    assert prediction["prediction_label"] == 1
    assert prediction["clamp_ards_entity_count"] == 1
    audit = result.audit.set_index("clamp_doc_id").loc["s1"]
    assert str(audit["expected_output_file"]).endswith("s1.txt.tsv")


def test_output_doc_id_strips_original_txt_suffix_for_xmi_variants(tmp_path: Path) -> None:
    xmi = tmp_path / "s1.txt.xmi"
    xmi_gz = tmp_path / "s1.txt.xmi.gz"
    xmi.write_text("<xmi />", encoding="utf-8")
    xmi_gz.write_text("<xmi />", encoding="utf-8")

    output_map, duplicate_doc_ids = map_output_files_by_doc_id([xmi, xmi_gz])

    assert set(output_map) == {"s1"}
    assert duplicate_doc_ids == {"s1"}
    assert parse_clamp_output_file(xmi).parse_error == "unsupported_output_format:.xmi"
    assert parse_clamp_output_file(xmi_gz).parse_error == "unsupported_output_format:.xmi.gz"


def test_duplicate_normalized_output_doc_ids_are_not_parsed_as_separate_docs(
    tmp_path: Path,
) -> None:
    manifest = _write_manifest(tmp_path, ["s1", "s2"])
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    shutil.copy(FIXTURE_DIR / "s1.tsv", output_dir / "s1.tsv")
    shutil.copy(FIXTURE_DIR / "s1.tsv", output_dir / "s1.txt.tsv")
    shutil.copy(FIXTURE_DIR / "s2.tsv", output_dir / "s2.tsv")

    result = parse_clamp_ards_outputs(input_manifest_path=manifest, output_dir=output_dir)

    s1_prediction = result.predictions.set_index("clamp_doc_id").loc["s1"]
    s2_prediction = result.predictions.set_index("clamp_doc_id").loc["s2"]
    assert pd.isna(s1_prediction["prediction_label"])
    assert s1_prediction["clamp_parse_status"] == "duplicate_output"
    assert s1_prediction["clamp_parse_error"] == "multiple_output_files_for_doc_id"
    assert s2_prediction["prediction_label"] == 0
    assert result.summary["duplicate_output_doc_ids"] == 1
    assert result.summary["duplicate_output_files"] == 2


def test_parser_fails_when_all_matched_files_have_no_recognized_fields(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path, ["bad"])
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    shutil.copy(FIXTURE_DIR / "bad.txt", output_dir / "bad.txt")

    with pytest.raises(ValueError, match="recognized CLAMP fields"):
        parse_clamp_ards_outputs(input_manifest_path=manifest, output_dir=output_dir)


def test_xmi_output_is_explicit_parse_failure(tmp_path: Path) -> None:
    output = tmp_path / "s1.xmi"
    output.write_text("<xmi />", encoding="utf-8")

    parsed = parse_clamp_output_file(output)

    assert parsed.parse_status == "parse_error"
    assert "unsupported_output_format" in parsed.parse_error


def test_non_empty_output_without_semantic_field_is_parse_error(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path, ["no_semantic", "s1"])
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    shutil.copy(FIXTURE_DIR / "no_semantic.tsv", output_dir / "no_semantic.tsv")
    shutil.copy(FIXTURE_DIR / "s1.tsv", output_dir / "s1.tsv")

    result = parse_clamp_ards_outputs(input_manifest_path=manifest, output_dir=output_dir)

    prediction = result.predictions.set_index("clamp_doc_id").loc["no_semantic"]
    assert pd.isna(prediction["prediction_label"])
    assert prediction["clamp_parse_status"] == "parse_error"
    assert prediction["clamp_parse_error"] == "missing_required_field:semantic_tag"


def test_blank_output_can_be_parsed_empty_negative(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path, ["blank"])
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    shutil.copy(FIXTURE_DIR / "blank.tsv", output_dir / "blank.tsv")

    result = parse_clamp_ards_outputs(input_manifest_path=manifest, output_dir=output_dir)

    prediction = result.predictions.set_index("clamp_doc_id").loc["blank"]
    assert prediction["prediction_label"] == 0
    assert prediction["clamp_parse_status"] == "parsed_empty"


def _write_manifest(tmp_path: Path, doc_ids: list[str]) -> Path:
    manifest = pd.DataFrame(
        {
            "clamp_doc_id": doc_ids,
            "input_file": [f"/input/{doc_id}.txt" for doc_id in doc_ids],
            "source_dataset": ["mimic_cxr"] * len(doc_ids),
            "subject_id": list(range(100, 100 + len(doc_ids))),
            "study_id": list(range(200, 200 + len(doc_ids))),
            "accession_id": [""] * len(doc_ids),
            "encounter_id": [""] * len(doc_ids),
            "annotation_phase": ["stage1"] * len(doc_ids),
        }
    )
    path = tmp_path / "input_manifest.csv"
    manifest.to_csv(path, index=False)
    return path
