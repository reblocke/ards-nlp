from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import pytest
import yaml

from ards_cxr_benchmark.clamp_ards.fixtures import (
    EXPECTED_CASE_COUNT,
    EXPECTED_CATEGORY_COUNTS,
    EXPECTED_ENTITY_FIELDS,
    FixturePendingError,
    _validate_expected_tsv,
    _validate_intermediate_json,
    build_fixture_cases,
    find_phi_sentinels,
    generate_fixture,
    validate_fixture,
    write_sha256s,
)
from ards_cxr_benchmark.clamp_ards.resources import (
    default_project_dir,
    default_resource_manifest_path,
    load_clamp_resources,
)


def test_generated_fixture_has_frozen_coverage_and_explicit_pending_state(
    tmp_path: Path,
) -> None:
    root = tmp_path / "fixture"

    generated = generate_fixture(
        root,
        project_dir=default_project_dir(),
        resource_manifest_path=default_resource_manifest_path(),
    )

    assert generated.case_count == EXPECTED_CASE_COUNT == 463
    assert generated.category_counts == EXPECTED_CATEGORY_COUNTS
    assert generated.lifecycle == "awaiting_legacy_runs"
    assert generated.pending
    assert {path.name for path in (root / "clamp_expected").iterdir()} == {
        "PENDING",
        "README.md",
    }
    assert {path.name for path in (root / "intermediate_expected").iterdir()} == {
        "PENDING",
        "README.md",
    }
    with pytest.raises(FixturePendingError, match="awaiting two legacy CLAMP runs"):
        validate_fixture(root)
    assert validate_fixture(root, allow_pending=True) == generated


def test_case_matrix_exhausts_dictionary_assertion_tokenizer_and_ruta_resources() -> None:
    resources = load_clamp_resources(
        default_project_dir(),
        manifest_path=default_resource_manifest_path(),
    )
    cases = build_fixture_cases(resources)

    dictionary_cases = [case for case in cases if case.primary_category == "dictionary_case"]
    assert len(dictionary_cases) == 23 * 4
    assert Counter(dict(case.parameters)["case_form"] for case in dictionary_cases) == {
        "lower": 23,
        "title": 23,
        "upper": 23,
        "mixed": 23,
    }
    assert Counter(case.resource_index for case in dictionary_cases) == {
        index: 4 for index in range(1, 24)
    }

    assertion_cases = [case for case in cases if case.resource_kind == "assertion_cue"]
    expected_assertion_occurrences = {
        index: 2 if cue.category == "pseNegPhrases" else 1
        for index, cue in enumerate(resources.assertion_cues, start=1)
    }
    assert (
        Counter(case.resource_index for case in assertion_cases) == expected_assertion_occurrences
    )

    delimiter_cases = [case for case in cases if case.primary_category == "tokenizer_delimiter"]
    assert {
        case.payload.decode("utf-8")[len("bilateral") : -len("opacities")]
        for case in delimiter_cases
    } == resources.delimiters
    no_split_cases = [case for case in cases if case.primary_category == "tokenizer_no_split"]
    assert len(no_split_cases) == 2 * len(resources.no_split_strings)
    assert all(
        any(value in case.payload.decode("utf-8") for value in resources.no_split_strings)
        for case in no_split_cases
    )

    gap_cases = [case for case in cases if case.primary_category == "ruta_gap"]
    assert {
        (dict(case.parameters)["direction"], int(dict(case.parameters)["gap"]))
        for case in gap_cases
    } == {
        (direction, gap)
        for direction in ("location_morphology", "morphology_location")
        for gap in (0, 1, 5, 6)
    }

    sentence_cases = {
        dict(case.parameters)["scenario"]: case.payload
        for case in cases
        if case.primary_category == "sentence_input"
    }
    assert set(sentence_cases) == {
        "lf_boundary",
        "crlf_boundary",
        "repeated_lf",
        "repeated_crlf",
        "surrounding_whitespace",
        "period_boundary",
        "abbreviation_lower",
        "abbreviation_upper",
        "split_pattern",
        "section_header",
        "exactly_500_tokens",
        "more_than_500_tokens",
        "empty",
        "whitespace_only",
        "unicode_bmp",
        "unicode_supplementary",
        "split_pattern_without_terminal_periods",
        "section_header_inline",
    }
    assert sentence_cases["split_pattern_without_terminal_periods"] == (
        b"1) ARDS 2) pulmonary edema"
    )
    assert sentence_cases["section_header_inline"] == b"IMPRESSION: ARDS."


def test_fixture_generation_is_byte_deterministic(tmp_path: Path) -> None:
    left = tmp_path / "left"
    right = tmp_path / "right"
    kwargs = {
        "project_dir": default_project_dir(),
        "resource_manifest_path": default_resource_manifest_path(),
    }

    generate_fixture(left, **kwargs)
    generate_fixture(right, **kwargs)

    left_files = {
        path.relative_to(left).as_posix(): path.read_bytes()
        for path in left.rglob("*")
        if path.is_file()
    }
    right_files = {
        path.relative_to(right).as_posix(): path.read_bytes()
        for path in right.rglob("*")
        if path.is_file()
    }
    assert left_files == right_files


def test_forced_fixture_generation_rejects_unmarked_existing_directory(
    tmp_path: Path,
) -> None:
    output = tmp_path / "unrelated"
    output.mkdir()
    sentinel = output / "keep.txt"
    sentinel.write_text("do not delete", encoding="utf-8")

    with pytest.raises(ValueError, match="generated fixture provenance"):
        generate_fixture(
            output,
            project_dir=default_project_dir(),
            resource_manifest_path=default_resource_manifest_path(),
            force=True,
        )

    assert sentinel.read_text(encoding="utf-8") == "do not delete"


def test_forced_fixture_generation_rejects_project_overlap(tmp_path: Path) -> None:
    project = tmp_path / "project"
    output = project / "generated"

    with pytest.raises(ValueError, match="must not overlap the CLAMP project"):
        generate_fixture(
            output,
            project_dir=project,
            resource_manifest_path=default_resource_manifest_path(),
            force=True,
        )

    assert not output.exists()


def test_forced_fixture_generation_rejects_symlink_output(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.mkdir()
    output = tmp_path / "fixture-link"
    output.symlink_to(target, target_is_directory=True)

    with pytest.raises(ValueError, match="must not be a symlink"):
        generate_fixture(
            output,
            project_dir=default_project_dir(),
            resource_manifest_path=default_resource_manifest_path(),
            force=True,
        )

    assert target.is_dir()


def test_forced_fixture_generation_validates_resources_before_replacement(
    tmp_path: Path,
) -> None:
    output = tmp_path / "fixture"
    generate_fixture(
        output,
        project_dir=default_project_dir(),
        resource_manifest_path=default_resource_manifest_path(),
    )
    original_provenance = (output / "provenance.json").read_bytes()

    with pytest.raises(FileNotFoundError):
        generate_fixture(
            output,
            project_dir=default_project_dir(),
            resource_manifest_path=tmp_path / "missing-resource-manifest.json",
            force=True,
        )

    assert (output / "provenance.json").read_bytes() == original_provenance
    assert (output / "input").is_dir()


def test_fixture_validator_rejects_tampering_and_unexpected_files(tmp_path: Path) -> None:
    root = tmp_path / "fixture"
    generate_fixture(
        root,
        project_dir=default_project_dir(),
        resource_manifest_path=default_resource_manifest_path(),
    )
    first_input = next((root / "input").iterdir())
    first_input.write_bytes(first_input.read_bytes() + b"x")

    with pytest.raises(ValueError, match="frozen case matrix"):
        validate_fixture(root, allow_pending=True)

    generate_fixture(
        root,
        project_dir=default_project_dir(),
        resource_manifest_path=default_resource_manifest_path(),
        force=True,
    )
    (root / "unexpected.txt").write_text("synthetic", encoding="utf-8")
    with pytest.raises(ValueError, match="unexpected=.*unexpected.txt"):
        validate_fixture(root, allow_pending=True)


def test_fixture_validator_rejects_unsafe_case_ids_before_hash_checks(tmp_path: Path) -> None:
    root = tmp_path / "fixture"
    generate_fixture(
        root,
        project_dir=default_project_dir(),
        resource_manifest_path=default_resource_manifest_path(),
    )
    cases_path = root / "cases.yaml"
    payload = yaml.safe_load(cases_path.read_text(encoding="utf-8"))
    payload["cases"][0]["case_id"] = "../unsafe"
    cases_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="Unsafe fixture case_id"):
        validate_fixture(root, allow_pending=True)


def test_fixture_validator_rejects_rehashed_coverage_matrix_substitution(
    tmp_path: Path,
) -> None:
    root = tmp_path / "fixture"
    generate_fixture(
        root,
        project_dir=default_project_dir(),
        resource_manifest_path=default_resource_manifest_path(),
    )
    cases_path = root / "cases.yaml"
    payload = yaml.safe_load(cases_path.read_text(encoding="utf-8"))
    payload["cases"][1]["resource_ref"]["index"] = 999
    cases_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    write_sha256s(root)

    with pytest.raises(ValueError, match="frozen coverage matrix"):
        validate_fixture(root, allow_pending=True)


@pytest.mark.parametrize(
    ("text", "sentinel"),
    [
        ("[**Known lastname**]", "mimic_deidentification_marker"),
        ("MRN 123", "record_identifier_label"),
        ("person@example.org", "email_address"),
        ("303-555-0199", "phone_number"),
        ("study_id value", "record_identifier_label"),
        ("identifier 123456", "long_numeric_identifier"),
    ],
)
def test_phi_sentinel_screen_rejects_disallowed_patterns(text: str, sentinel: str) -> None:
    assert sentinel in find_phi_sentinels(text)


def test_expected_tsv_validator_checks_utf16_text_order_and_duplicate_occurrence(
    tmp_path: Path,
) -> None:
    source_text = "😀 ARDS"
    path = tmp_path / "one.tsv"
    rows = [
        _normalized_entity_row(raw_order=0, duplicate_occurrence=0),
        _normalized_entity_row(raw_order=1, duplicate_occurrence=1),
    ]
    _write_normalized_tsv(path, rows)

    entities = _validate_expected_tsv(path, "one", source_text=source_text)

    assert len(entities) == 2
    assert entities[0] == entities[1]

    rows[1]["duplicate_occurrence"] = "0"
    _write_normalized_tsv(path, rows)
    with pytest.raises(ValueError, match="duplicate_occurrence"):
        _validate_expected_tsv(path, "one", source_text=source_text)

    rows = [_normalized_entity_row(raw_order=0, duplicate_occurrence=0)]
    rows[0]["start"] = "1"
    _write_normalized_tsv(path, rows)
    with pytest.raises(ValueError, match="Invalid UTF-16 entity span"):
        _validate_expected_tsv(path, "one", source_text=source_text)

    rows[0]["start"] = "3"
    rows[0]["entity_text"] = "not-ARDS"
    _write_normalized_tsv(path, rows)
    with pytest.raises(ValueError, match="covered text differs"):
        _validate_expected_tsv(path, "one", source_text=source_text)

    rows[0]["entity_text"] = "ARDS"
    rows[0]["start"] = "3.0"
    _write_normalized_tsv(path, rows)
    with pytest.raises(ValueError, match="Invalid normalized entity start"):
        _validate_expected_tsv(path, "one", source_text=source_text)

    path.write_text(
        "\t".join((*EXPECTED_ENTITY_FIELDS, "unexpected")) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Unexpected normalized entity schema"):
        _validate_expected_tsv(path, "one", source_text=source_text)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("top_level_schema", "top-level schema"),
        ("row_schema", "row schema"),
        ("integer_type", "Invalid intermediate start"),
        ("numbering", "contiguous zero-based order"),
        ("covered_text", "covered text differs"),
        ("surrogate_boundary", "Invalid UTF-16 final_entities span"),
        ("nullable_type", "Invalid intermediate cui"),
    ],
)
def test_intermediate_validator_rejects_schema_type_offset_and_text_drift(
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    source_text = "😀 ARDS"
    payload = _valid_intermediate_payload(source_text)
    if mutation == "top_level_schema":
        payload["unexpected"] = True
    elif mutation == "row_schema":
        payload["tokens"][0]["unexpected"] = True
    elif mutation == "integer_type":
        payload["tokens"][0]["start"] = True
    elif mutation == "numbering":
        payload["tokens"][1]["token_number"] = 3
    elif mutation == "covered_text":
        payload["sentences"][0]["covered_text"] = "not source text"
    elif mutation == "surrogate_boundary":
        payload["final_entities"][0]["start"] = 1
    elif mutation == "nullable_type":
        payload["final_entities"][0]["cui"] = 42
    else:  # pragma: no cover - exhaustive parameter contract
        raise AssertionError(mutation)
    path = tmp_path / "one.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        _validate_intermediate_json(
            path,
            "one",
            {"input_sha256": hashlib.sha256(source_text.encode()).hexdigest()},
            source_text=source_text,
        )


def test_complete_lifecycle_requires_two_runs_reviews_and_expected_inventory(
    tmp_path: Path,
) -> None:
    root = tmp_path / "fixture"
    generate_fixture(
        root,
        project_dir=default_project_dir(),
        resource_manifest_path=default_resource_manifest_path(),
    )
    for expected_dir in (root / "clamp_expected", root / "intermediate_expected"):
        (expected_dir / "PENDING").unlink()

    with (root / "manifest.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    header = "\t".join(EXPECTED_ENTITY_FIELDS) + "\n"
    for row in rows:
        case_id = row["case_id"]
        (root / "clamp_expected" / f"{case_id}.tsv").write_text(
            header,
            encoding="utf-8",
            newline="\n",
        )
        intermediate = {
            "schema_version": 1,
            "case_id": case_id,
            "source_text_sha256": row["input_sha256"],
            "offset_coordinate_system": "utf16_code_units",
            "interval_convention": "half_open",
            "legacy_output_sha256": {
                "run_1": {"txt": "1" * 64, "xmi": "2" * 64},
                "run_2": {"txt": "3" * 64, "xmi": "4" * 64},
            },
            "sentences": [],
            "tokens": [],
            "final_entities": [],
        }
        (root / "intermediate_expected" / f"{case_id}.json").write_text(
            json.dumps(intermediate, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )

    provenance_path = root / "provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance["lifecycle"] = "complete"
    provenance["legacy_runtime"] = {
        "clamp_version": "1.6.6",
        "clamp_build": "test-build",
        "operating_system": "test-windows",
        "java_version": "test-java",
        "locale": "en-US",
        "timezone": "UTC",
        "pipeline_export_settings": "test-settings",
    }
    provenance["runs"] = [
        {
            "run_number": run_number,
            "started_at": "2026-01-01T00:00:00Z",
            "completed_at": "2026-01-01T00:01:00Z",
            "output_manifest_sha256": str(run_number) * 64,
        }
        for run_number in (1, 2)
    ]
    provenance["determinism"] = {
        "status": "passed",
        "required_run_count": 2,
        "raw_order_required": True,
        "txt_row_order_stable": True,
        "xmi_entity_order_stable": True,
        "exact_sentence_annotations": True,
        "exact_token_annotations": True,
        "exact_entity_multisets": True,
        "txt_order_difference_documents": 0,
        "xmi_order_difference_documents": 0,
    }
    resource_manifest = json.loads(default_resource_manifest_path().read_text(encoding="utf-8"))
    resource_hashes = {
        path: digest
        for path, digest in resource_manifest["files"].items()
        if path.startswith("Components/")
    }
    import_runs = {
        f"run_{run_number}": {
            "run_label": f"run_{run_number}",
            "started_at_utc": "2026-01-01T00:00:00Z",
            "finished_at_utc": "2026-01-01T00:01:00Z",
            "recorded_at_utc": "2026-01-01T00:02:00Z",
            "output_file_count": EXPECTED_CASE_COUNT * 2,
            "output_manifest_sha256": str(run_number) * 64,
        }
        for run_number in (1, 2)
    }
    provenance["legacy_import"] = {
        "generated_only_from_returned_legacy_clamp": True,
        "raw_xmi_committed": False,
        "runtime_details": {
            "clamp": {"version": "1.6.6", "build": "test-build"},
            "windows": {"caption": "test-windows"},
            "java": {"version_output": "test-java"},
            "project_commit": resource_manifest["project_commit"],
            "project_files_sha256": resource_manifest["files"],
            "resources_sha256": resource_hashes,
            "export_settings": "test-settings",
            "offset_convention": "half-open UTF-16 code units",
            "null_convention": "literal null",
            "manual_commands": ["test batch run"],
        },
        "fixture_counts": {
            "cases": EXPECTED_CASE_COUNT,
            "sentences": 0,
            "tokens": 0,
            "final_entities": 0,
        },
        **import_runs,
    }
    provenance["reviews"] = {
        "phi": {
            "automated_screen": "passed",
            "manual_review": "approved",
            "reviewer": "test-reviewer",
            "reviewed_at": "2026-01-02T00:00:00Z",
        },
        "redistribution": {
            "status": "approved",
            "authority": "test-authority",
            "evidence": "test-evidence",
        },
    }
    provenance_path.write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    write_sha256s(root)

    result = validate_fixture(root)

    assert result.lifecycle == "complete"
    assert not result.pending

    disagreement_path = root / "clamp_expected/dict_case_04_lower.tsv"
    disagreement_path.write_text(
        header + "dict_case_04_lower\t0\t4\tARDS\tpresent\t\\N\t\\N\tards\t0\t0\n",
        encoding="utf-8",
        newline="\n",
    )
    write_sha256s(root)
    with pytest.raises(ValueError, match="TSV/intermediate final-entity disagreement"):
        validate_fixture(root)

    disagreement_path.write_text(header, encoding="utf-8", newline="\n")
    provenance["legacy_import"]["fixture_counts"]["tokens"] = 1
    provenance_path.write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    write_sha256s(root)
    with pytest.raises(ValueError, match=r"legacy_import\.fixture_counts"):
        validate_fixture(root)

    provenance["legacy_import"]["fixture_counts"]["tokens"] = 0
    provenance.pop("legacy_import")
    provenance_path.write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    write_sha256s(root)
    with pytest.raises(ValueError, match="importer-generated legacy_import"):
        validate_fixture(root)


def _normalized_entity_row(
    *,
    raw_order: int,
    duplicate_occurrence: int,
) -> dict[str, str]:
    return {
        "clamp_doc_id": "one",
        "start": "3",
        "end": "7",
        "semantic_tag": "ARDS",
        "assertion": "present",
        "cui": r"\N",
        "attribute": r"\N",
        "entity_text": "ARDS",
        "raw_order": str(raw_order),
        "duplicate_occurrence": str(duplicate_occurrence),
    }


def _write_normalized_tsv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=EXPECTED_ENTITY_FIELDS,
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def _valid_intermediate_payload(source_text: str) -> dict[str, object]:
    return {
        "schema_version": 1,
        "case_id": "one",
        "source_text_sha256": hashlib.sha256(source_text.encode()).hexdigest(),
        "offset_coordinate_system": "utf16_code_units",
        "interval_convention": "half_open",
        "legacy_output_sha256": {
            "run_1": {"txt": "1" * 64, "xmi": "2" * 64},
            "run_2": {"txt": "3" * 64, "xmi": "4" * 64},
        },
        "sentences": [
            {
                "start": 0,
                "end": 7,
                "sentence_number": 0,
                "covered_text": source_text,
            }
        ],
        "tokens": [
            {"start": 0, "end": 2, "token_number": 0, "covered_text": "😀"},
            {"start": 3, "end": 7, "token_number": 1, "covered_text": "ARDS"},
        ],
        "final_entities": [
            {
                "start": 3,
                "end": 7,
                "semantic_tag": "ARDS",
                "assertion": "present",
                "cui": None,
                "attribute": None,
                "covered_text": "ARDS",
                "raw_order": 0,
            }
        ],
    }
