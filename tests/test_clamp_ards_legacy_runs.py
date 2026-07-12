from __future__ import annotations

import csv
import hashlib
import html
import json
import shutil
import zipfile
from dataclasses import replace
from pathlib import Path

import pytest

from ards_cxr_benchmark.clamp_ards.fixtures import (
    EXPECTED_CASE_COUNT,
    EXPECTED_ENTITY_FIELDS,
    validate_fixture,
)
from ards_cxr_benchmark.clamp_ards.legacy_runs import (
    FixtureCase,
    LegacyDocument,
    LegacyImportSummary,
    LegacyRun,
    OutputPayload,
    RunInventory,
    _load_provenance,
    _parse_legacy_document,
    _sanitize_value,
    _validate_exact_repeat,
    _validate_inventory,
    discover_legacy_output,
    import_legacy_clamp_parity_runs,
    load_fixture_cases,
    prepare_legacy_clamp_parity_handoff,
)
from ards_cxr_benchmark.clamp_ards.resources import (
    default_project_dir,
    default_resource_manifest_path,
)

PROJECT_COMMIT = "9f8c92fbbeb44645a1066be3510d4ab993995c1e"


@pytest.fixture(scope="module")
def returned_runs(
    tmp_path_factory: pytest.TempPathFactory,
    pending_clamp_fixture: Path,
) -> dict[str, Path]:
    root = tmp_path_factory.mktemp("legacy_runs")
    cases = load_fixture_cases(pending_clamp_fixture)
    target = next(case for case in cases if case.case_id == "dict_boundary_08")
    first = target.text.index("ARDS")
    second = target.text.index("ARDS", first + 1)
    entities = {
        target.case_id: [
            (first, first + 4, "ARDS"),
            (second, second + 4, "ARDS"),
        ]
    }
    result: dict[str, Path] = {}
    for run_number in (1, 2):
        label = f"run_{run_number}"
        run_dir = root / label
        output = run_dir / "Output"
        output.mkdir(parents=True)
        for case in cases:
            case_entities = entities.get(case.case_id, [])
            (output / f"{case.case_id}.txt").write_bytes(_txt_payload(case.text, case_entities))
            (output / f"{case.case_id}.xmi").write_bytes(_xmi_payload(case.text, case_entities))
        provenance = root / f"{label}_provenance.json"
        _write_provenance(provenance, label=label, run_dir=run_dir)
        result[label] = run_dir
        result[f"{label}_provenance"] = provenance
    result["fixture_root"] = pending_clamp_fixture
    return result


def test_handoff_is_deterministic_and_contains_only_synthetic_inputs(
    tmp_path: Path,
    pending_clamp_fixture: Path,
) -> None:
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    first_zip = tmp_path / "first.zip"
    second_zip = tmp_path / "second.zip"
    manifest_path = default_resource_manifest_path()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    project = tmp_path / "project"
    for relative in manifest["files"]:
        source = default_project_dir() / relative
        destination = project / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    kwargs = {
        "fixture_root": pending_clamp_fixture,
        "project_source_dir": project,
        "collector_script": Path("scripts/windows/collect_clamp_ards_parity_provenance.ps1"),
        "resource_manifest": manifest_path,
    }

    first = prepare_legacy_clamp_parity_handoff(
        **kwargs,
        destination_dir=first_dir,
        output_archive=first_zip,
    )
    second = prepare_legacy_clamp_parity_handoff(
        **kwargs,
        destination_dir=second_dir,
        output_archive=second_zip,
    )

    assert first["case_count"] == EXPECTED_CASE_COUNT
    assert first["project_commit"] == PROJECT_COMMIT
    assert first["restricted_oracle_included"] is False
    assert first["output_archive_sha256"] == second["output_archive_sha256"]
    assert first_zip.read_bytes() == second_zip.read_bytes()
    assert len(list((first_dir / "ARDS/Data/Input").glob("*.txt"))) == EXPECTED_CASE_COUNT
    assert not any(
        "oracle" in name.lower() or "mimic" in name.lower() for name in _zip_names(first_zip)
    )


def test_import_writes_review_candidate_without_mutating_pending_fixture(
    tmp_path: Path,
    returned_runs: dict[str, Path],
) -> None:
    fixture = tmp_path / "fixture"
    shutil.copytree(returned_runs["fixture_root"], fixture)
    original_sha = hashlib.sha256((fixture / "provenance.json").read_bytes()).hexdigest()
    run_2_zip = tmp_path / "run_2.zip"
    _zip_directory(returned_runs["run_2"], run_2_zip, prefix="returned/run_2")

    summary = import_legacy_clamp_parity_runs(
        fixture_root=fixture,
        run_1_source=returned_runs["run_1"],
        run_2_source=run_2_zip,
        run_1_provenance=returned_runs["run_1_provenance"],
        run_2_provenance=returned_runs["run_2_provenance"],
        candidate_output_dir=tmp_path / "candidate",
    )

    assert summary.case_count == EXPECTED_CASE_COUNT
    assert summary.entity_count == 2
    assert summary.raw_row_order_stable
    assert not summary.finalized
    assert hashlib.sha256((fixture / "provenance.json").read_bytes()).hexdigest() == original_sha
    assert validate_fixture(fixture, allow_pending=True).pending
    candidate = tmp_path / "candidate"
    with (candidate / "clamp_expected/dict_boundary_08.tsv").open(
        encoding="utf-8", newline=""
    ) as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        assert tuple(reader.fieldnames or ()) == EXPECTED_ENTITY_FIELDS
        assert [row["raw_order"] for row in reader] == ["0", "1"]
    proposed = json.loads((candidate / "proposed_provenance.json").read_text(encoding="utf-8"))
    assert proposed["lifecycle"] == "awaiting_reviews"
    assert proposed["reviews"]["phi"]["manual_review"] == "pending"
    assert proposed["reviews"]["redistribution"]["status"] == "pending"
    assert not list(candidate.rglob("*.xmi"))


@pytest.mark.parametrize(
    "protected_name",
    [
        "fixture_root",
        "run_1",
        "run_2",
        "run_1_provenance",
        "run_2_provenance",
        "run_1_sha256s",
        "run_2_sha256s",
        "resource_manifest",
    ],
)
def test_import_rejects_candidate_equal_to_every_protected_input(
    returned_runs: dict[str, Path],
    protected_name: str,
) -> None:
    protected = _protected_import_paths(returned_runs)[protected_name]
    original = _path_tree_sha256(protected)

    with pytest.raises(ValueError, match="must not overlap protected source path"):
        _import_returned_runs(returned_runs, candidate_output_dir=protected)

    assert _path_tree_sha256(protected) == original


@pytest.mark.parametrize("relationship", ["inside", "ancestor"])
def test_import_rejects_candidate_descendant_or_ancestor_of_returned_run(
    returned_runs: dict[str, Path],
    relationship: str,
) -> None:
    run_1 = returned_runs["run_1"]
    original = _path_tree_sha256(run_1)
    candidate = run_1 / "generated" if relationship == "inside" else run_1.parent

    with pytest.raises(ValueError, match="must not overlap protected source path"):
        _import_returned_runs(returned_runs, candidate_output_dir=candidate)

    assert _path_tree_sha256(run_1) == original


def test_import_requires_returned_sha256_manifests(
    tmp_path: Path,
    returned_runs: dict[str, Path],
) -> None:
    fixture = tmp_path / "fixture"
    shutil.copytree(returned_runs["fixture_root"], fixture)
    corrupt = tmp_path / "corrupt.SHA256SUMS"
    original = returned_runs["run_1_provenance"].with_suffix(".SHA256SUMS")
    first_line, *remaining = original.read_text(encoding="utf-8").splitlines()
    corrupt.write_text(
        "\n".join([f"{'0' * 64}{first_line[64:]}", *remaining]) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="SHA256SUMS differs from provenance"):
        import_legacy_clamp_parity_runs(
            fixture_root=fixture,
            run_1_source=returned_runs["run_1"],
            run_2_source=returned_runs["run_2"],
            run_1_provenance=returned_runs["run_1_provenance"],
            run_2_provenance=returned_runs["run_2_provenance"],
            run_1_sha256s=corrupt,
            candidate_output_dir=tmp_path / "candidate",
        )

    assert not (tmp_path / "candidate").exists()


def test_finalize_requires_reviews_and_produces_a_strict_complete_fixture(
    tmp_path: Path,
    returned_runs: dict[str, Path],
) -> None:
    fixture = tmp_path / "fixture"
    shutil.copytree(returned_runs["fixture_root"], fixture)

    with pytest.raises(ValueError, match="phi_reviewer"):
        import_legacy_clamp_parity_runs(
            fixture_root=fixture,
            run_1_source=returned_runs["run_1"],
            run_2_source=returned_runs["run_2"],
            run_1_provenance=returned_runs["run_1_provenance"],
            run_2_provenance=returned_runs["run_2_provenance"],
            candidate_output_dir=tmp_path / "candidate_pending",
            finalize=True,
        )
    assert validate_fixture(fixture, allow_pending=True).pending

    summary = import_legacy_clamp_parity_runs(
        fixture_root=fixture,
        run_1_source=returned_runs["run_1"],
        run_2_source=returned_runs["run_2"],
        run_1_provenance=returned_runs["run_1_provenance"],
        run_2_provenance=returned_runs["run_2_provenance"],
        candidate_output_dir=tmp_path / "candidate_approved",
        finalize=True,
        phi_reviewer="Synthetic fixture reviewer",
        phi_reviewed_at="2026-07-11T12:00:00Z",
        redistribution_authority="Maintainer legal review",
        redistribution_evidence="issue-5 review record 2026-07-11",
    )

    assert summary.finalized
    validation = validate_fixture(fixture)
    assert not validation.pending
    provenance = json.loads((fixture / "provenance.json").read_text(encoding="utf-8"))
    assert provenance["lifecycle"] == "complete"
    assert provenance["determinism"]["raw_order_required"] is True
    assert provenance["reviews"]["phi"]["manual_review"] == "approved"
    assert provenance["reviews"]["redistribution"]["status"] == "approved"
    assert not list(fixture.rglob("*.xmi"))


def test_import_records_unstable_run_order_without_rejecting_exact_multisets(
    tmp_path: Path,
    returned_runs: dict[str, Path],
) -> None:
    fixture = tmp_path / "fixture"
    shutil.copytree(returned_runs["fixture_root"], fixture)
    changed_run = tmp_path / "run_2_changed"
    shutil.copytree(returned_runs["run_2"], changed_run)
    text = (fixture / "input/dict_boundary_08.txt").read_text(encoding="utf-8")
    first = text.index("ARDS")
    second = text.index("ARDS", first + 1)
    reversed_entities = [(second, second + 4, "ARDS"), (first, first + 4, "ARDS")]
    (changed_run / "Output/dict_boundary_08.txt").write_bytes(_txt_payload(text, reversed_entities))
    (changed_run / "Output/dict_boundary_08.xmi").write_bytes(_xmi_payload(text, reversed_entities))
    provenance = tmp_path / "run_2_changed_provenance.json"
    _write_provenance(provenance, label="run_2", run_dir=changed_run)

    summary = import_legacy_clamp_parity_runs(
        fixture_root=fixture,
        run_1_source=returned_runs["run_1"],
        run_2_source=changed_run,
        run_1_provenance=returned_runs["run_1_provenance"],
        run_2_provenance=provenance,
        candidate_output_dir=tmp_path / "candidate",
    )

    assert not summary.raw_row_order_stable
    assert not summary.output_order_required
    assert not summary.xmi_entity_order_stable
    assert summary.txt_order_difference_documents == 1
    assert summary.xmi_order_difference_documents == 1
    assert validate_fixture(fixture, allow_pending=True).pending
    proposed = json.loads(
        (tmp_path / "candidate/proposed_provenance.json").read_text(encoding="utf-8")
    )
    assert proposed["determinism"]["raw_order_required"] is False
    assert proposed["determinism"]["xmi_entity_order_stable"] is False


@pytest.mark.parametrize(
    ("txt_ids", "xmi_ids", "expected", "detail"),
    [
        (set(), {"one"}, {"one"}, "missing_txt"),
        ({"one"}, set(), {"one"}, "missing_xmi"),
        ({"one", "extra"}, {"one"}, {"one"}, "extra_txt"),
        ({"one"}, {"one", "extra"}, {"one"}, "extra_xmi"),
    ],
)
def test_inventory_rejects_missing_and_extra_cases(
    txt_ids: set[str],
    xmi_ids: set[str],
    expected: set[str],
    detail: str,
) -> None:
    inventory = RunInventory(
        source_name="run",
        txt={case_id: OutputPayload(f"{case_id}.txt", b"") for case_id in txt_ids},
        xmi={case_id: OutputPayload(f"{case_id}.xmi", b"") for case_id in xmi_ids},
    )

    with pytest.raises(ValueError, match=detail):
        _validate_inventory(inventory, expected, label="run_1")


def test_inventory_discovery_rejects_duplicate_case_outputs(tmp_path: Path) -> None:
    for directory in (tmp_path / "a", tmp_path / "b"):
        directory.mkdir()
        (directory / "one.txt").write_text("duplicate", encoding="utf-8")

    with pytest.raises(ValueError, match="Duplicate TXT output"):
        discover_legacy_output(tmp_path)


@pytest.mark.parametrize(
    "local_value",
    [
        r'Run from "C:\Users\Alice Smith\CLAMP project" then export',
        "Run from /Users/Alice Smith/CLAMP project then export",
        r"Read from \\workstation\Alice Smith\CLAMP project",
    ],
)
def test_provenance_sanitizer_redacts_entire_local_path_value(local_value: str) -> None:
    assert _sanitize_value(local_value) == "<redacted-local-path-value>"


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("clamp.version", "2.0", "CLAMP version must be 1.6.6"),
        ("started_at_utc", "not-a-time", "valid ISO-8601"),
        ("finished_at_utc", "2026-07-11T12:20:00", "UTC offset"),
    ],
)
def test_provenance_rejects_wrong_runtime_or_invalid_timestamps(
    tmp_path: Path,
    returned_runs: dict[str, Path],
    field: str,
    value: str,
    message: str,
) -> None:
    payload = json.loads(returned_runs["run_1_provenance"].read_text(encoding="utf-8"))
    if field == "clamp.version":
        payload["clamp"]["version"] = value
    else:
        payload[field] = value
    path = tmp_path / "invalid_provenance.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        _load_provenance(path, expected_label="run_1")


def test_provenance_rejects_machine_local_path_in_manual_steps(
    tmp_path: Path,
    returned_runs: dict[str, Path],
) -> None:
    payload = json.loads(returned_runs["run_1_provenance"].read_text(encoding="utf-8"))
    payload["manual_commands"] = [r"Open C:\Users\Alice Smith\ARDS and run the pipeline"]
    path = tmp_path / "unsafe_commands.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="without machine-local paths"):
        _load_provenance(path, expected_label="run_1")


def test_document_import_rejects_txt_parse_error() -> None:
    case = _fixture_case("ARDS")
    inventory = _one_case_inventory(
        case,
        txt=b"not\ta\tCLAMP\theader\n",
        xmi=_xmi_payload(case.text, []),
    )

    with pytest.raises(ValueError, match="TXT parse failure"):
        _parse_legacy_document(case, inventory, run_label="run_1")


def test_document_import_rejects_xmi_parse_error() -> None:
    case = _fixture_case("ARDS")
    inventory = _one_case_inventory(
        case,
        txt=_txt_payload(case.text, []),
        xmi=b"<not-valid-xmi>",
    )

    with pytest.raises(ValueError, match="XMI parse failure"):
        _parse_legacy_document(case, inventory, run_label="run_1")


def test_document_import_rejects_sofa_source_mismatch() -> None:
    case = _fixture_case("ARDS")
    inventory = _one_case_inventory(
        case,
        txt=_txt_payload(case.text, []),
        xmi=_xmi_payload("OTHER", []),
    )

    with pytest.raises(ValueError, match="Sofa text differs"):
        _parse_legacy_document(case, inventory, run_label="run_1")


@pytest.mark.parametrize(("start", "end"), [(1, 2), (0, 99)])
def test_document_import_rejects_invalid_utf16_boundaries(start: int, end: int) -> None:
    case = _fixture_case("😀 ARDS")
    inventory = _one_case_inventory(
        case,
        txt=_txt_payload(case.text, []),
        xmi=_xmi_payload(case.text, [(start, end, "ARDS")]),
    )

    with pytest.raises(ValueError, match="invalid XMI entity offset"):
        _parse_legacy_document(case, inventory, run_label="run_1")


def test_document_import_rejects_txt_xmi_final_entity_field_disagreement() -> None:
    text = "ARDS ARDS"
    case = _fixture_case(text)
    xmi_entities = [(0, 4, "ARDS"), (5, 9, "ARDS")]
    txt_entities = list(xmi_entities)
    txt_entities[0] = (0, 4, "Morphology")
    inventory = _one_case_inventory(
        case,
        txt=_txt_payload(text, txt_entities),
        xmi=_xmi_payload(text, xmi_entities),
    )

    with pytest.raises(ValueError, match="TXT/XMI final-entity disagreement"):
        _parse_legacy_document(case, inventory, run_label="run_1")


def test_document_import_accepts_and_preserves_distinct_txt_xmi_orders() -> None:
    text = "ARDS ARDS"
    case = _fixture_case(text)
    xmi_entities = [(0, 4, "ARDS"), (5, 9, "ARDS")]
    txt_entities = list(reversed(xmi_entities))
    inventory = _one_case_inventory(
        case,
        txt=_txt_payload(text, txt_entities),
        xmi=_xmi_payload(text, xmi_entities),
    )

    document = _parse_legacy_document(case, inventory, run_label="run_1")

    assert [row[0] for row in document.entities] == [0, 5]
    assert [row[0] for row in document.txt_rows] == [5, 0]


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("sentences", ((0, 3, 0, "ARD"),)),
        ("tokens", ((0, 3, 0, "ARD"),)),
        ("entities", ()),
    ],
)
def test_repeat_comparison_rejects_sentence_token_and_entity_drift(
    field: str,
    replacement: tuple[object, ...],
) -> None:
    document = LegacyDocument(
        case_id="one",
        source_text_sha256="0" * 64,
        txt_sha256="1" * 64,
        xmi_sha256="2" * 64,
        sentences=((0, 4, 0, "ARDS"),),
        tokens=((0, 4, 0, "ARDS"),),
        entities=((0, 4, "ARDS", "present", None, None, "ARDS"),),
        txt_rows=((0, 4, "ARDS", "present", None, "ARDS"),),
    )
    changed = replace(document, **{field: replacement})
    run_1 = LegacyRun("run_1", {"one": document}, {})
    run_2 = LegacyRun("run_2", {"one": changed}, {})

    with pytest.raises(ValueError, match="not exact repeats"):
        _validate_exact_repeat(run_1, run_2, {"one"})


def _protected_import_paths(returned_runs: dict[str, Path]) -> dict[str, Path]:
    return {
        **returned_runs,
        "run_1_sha256s": returned_runs["run_1_provenance"].with_suffix(".SHA256SUMS"),
        "run_2_sha256s": returned_runs["run_2_provenance"].with_suffix(".SHA256SUMS"),
        "resource_manifest": default_resource_manifest_path().resolve(),
    }


def _import_returned_runs(
    returned_runs: dict[str, Path],
    *,
    candidate_output_dir: Path,
) -> LegacyImportSummary:
    return import_legacy_clamp_parity_runs(
        fixture_root=returned_runs["fixture_root"],
        run_1_source=returned_runs["run_1"],
        run_2_source=returned_runs["run_2"],
        run_1_provenance=returned_runs["run_1_provenance"],
        run_2_provenance=returned_runs["run_2_provenance"],
        candidate_output_dir=candidate_output_dir,
        resource_manifest=default_resource_manifest_path(),
    )


def _path_tree_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    if path.is_file():
        digest.update(path.read_bytes())
        return digest.hexdigest()
    for member in sorted(item for item in path.rglob("*") if item.is_file()):
        digest.update(member.relative_to(path).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(member.read_bytes())
    return digest.hexdigest()


def _txt_payload(text: str, entities: list[tuple[int, int, str]]) -> bytes:
    rows = ["Start\tEnd\tSemantic\tCUI\tAssertion\tEntity"]
    rows.extend(
        f"{start}\t{end}\t{semantic}\tnull\tpresent\t{text[start:end]}"
        for start, end, semantic in entities
    )
    return ("\n".join(rows) + "\n").encode()


def _xmi_payload(text: str, entities: list[tuple[int, int, str]]) -> bytes:
    sofa = _xml_attribute(text)
    rows = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<xmi:XMI xmlns:xmi="http://www.omg.org/XMI" '
        'xmlns:cas="http:///uima/cas.ecore" '
        'xmlns:clamp="http:///edu/uth/clamp/nlp/typesystem.ecore">',
        f'<cas:Sofa xmi:id="1" sofaNum="1" sofaID="_InitialView" sofaString="{sofa}"/>',
    ]
    for index, (start, end, semantic) in enumerate(entities, start=2):
        rows.append(
            f'<clamp:ClampNameEntityUIMA xmi:id="{index}" sofa="1" '
            f'begin="{start}" end="{end}" semanticTag="{semantic}" assertion="present"/>'
        )
    rows.append("</xmi:XMI>")
    return ("\n".join(rows) + "\n").encode()


def _xml_attribute(value: str) -> str:
    return (
        html.escape(value, quote=True)
        .replace("\r", "&#13;")
        .replace("\n", "&#10;")
        .replace("\t", "&#9;")
    )


def _write_provenance(path: Path, *, label: str, run_dir: Path) -> None:
    records = []
    for output in sorted((run_dir / "Output").iterdir()):
        payload = output.read_bytes()
        records.append(
            {
                "relative_path": f"Output/{output.name}",
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    resource_manifest = json.loads(default_resource_manifest_path().read_text(encoding="utf-8"))
    project_hashes = resource_manifest["files"]
    resource_hashes = {
        name: digest for name, digest in project_hashes.items() if name.startswith("Components/")
    }
    run_number = int(label.rsplit("_", maxsplit=1)[1])
    start_minute = (run_number - 1) * 30
    payload = {
        "schema_version": 1,
        "run_label": label,
        "recorded_at_utc": f"2026-07-11T12:{start_minute + 25:02d}:00Z",
        "started_at_utc": f"2026-07-11T12:{start_minute:02d}:00Z",
        "finished_at_utc": f"2026-07-11T12:{start_minute + 20:02d}:00Z",
        "clamp": {"version": "1.6.6", "build": "licensed-build-test"},
        "windows": {
            "caption": "Windows 11 Pro",
            "version": "10.0.26100",
            "build_number": "26100",
            "architecture": "64-bit",
            "locale": "en-US",
            "timezone": "Mountain Standard Time",
            "timezone_utc_offset": "-06:00:00",
        },
        "java": {"version_output": 'java version "1.8.0-test"'},
        "project": {
            "commit": PROJECT_COMMIT,
            "files_sha256": project_hashes,
        },
        "resources_sha256": resource_hashes,
        "export_settings": "TXT and XMI, UTF-8",
        "offset_convention": "half-open UTF-16 code units",
        "null_convention": "literal null in TXT",
        "manual_commands": ["CLAMP GUI batch run; copy output; collect provenance"],
        "output_files": records,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    path.with_suffix(".SHA256SUMS").write_text(
        "".join(f"{record['sha256']}  {record['relative_path']}\n" for record in records),
        encoding="utf-8",
        newline="\n",
    )


def _fixture_case(text: str) -> FixtureCase:
    payload = text.encode()
    return FixtureCase(
        case_id="one",
        category="test",
        input_file="input/one.txt",
        payload=payload,
        sha256=hashlib.sha256(payload).hexdigest(),
    )


def _one_case_inventory(case: FixtureCase, *, txt: bytes, xmi: bytes) -> RunInventory:
    return RunInventory(
        source_name="run",
        txt={case.case_id: OutputPayload(f"Output/{case.case_id}.txt", txt)},
        xmi={case.case_id: OutputPayload(f"Output/{case.case_id}.xmi", xmi)},
    )


def _zip_directory(root: Path, destination: Path, *, prefix: str) -> None:
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(item for item in root.rglob("*") if item.is_file()):
            archive.write(path, f"{prefix}/{path.relative_to(root).as_posix()}")


def _zip_names(path: Path) -> list[str]:
    with zipfile.ZipFile(path) as archive:
        return archive.namelist()
