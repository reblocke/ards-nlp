from __future__ import annotations

import hashlib
import importlib.resources
import json
import shutil
import subprocess
import sys
from pathlib import Path, PureWindowsPath

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from hypothesis import given
from hypothesis import strategies as st

from ards_cxr_benchmark.clamp_ards import (
    load_legacy_mirror,
    run_clamp_ards_batch,
    run_legacy_ards_clamp_mirror,
)
from ards_cxr_benchmark.clamp_ards import resources as clamp_resource_module
from ards_cxr_benchmark.clamp_ards.parity import compare_clamp_ards_outputs
from ards_cxr_benchmark.clamp_ards.resources import (
    ABBREVIATION_RESOURCE,
    ASSERTION_RESOURCE,
    PACKAGED_RESOURCE_MANIFEST,
    PROJECT_DIR_ENV,
    TOKEN_RULE_RESOURCE,
    _manifest_key,
    default_project_dir,
    default_resource_manifest_path,
    load_clamp_resources,
)
from ards_cxr_benchmark.clamp_ards.stemming import PorterCompatibilityStemmer
from ards_cxr_benchmark.clamp_ards.tokenization import Utf16OffsetMap
from ards_cxr_benchmark.clamp_ards.xmi import parse_clamp_xmi


def test_synthetic_external_resources_load_authorized_phenotype() -> None:
    resources = load_clamp_resources(
        default_project_dir(),
        manifest_path=default_resource_manifest_path(),
    )

    assert len(resources.dictionary) == 23
    assert len(resources.assertion_cues) == 240
    assert set(resources.resource_sha256) == {
        ABBREVIATION_RESOURCE,
        ASSERTION_RESOURCE,
        TOKEN_RULE_RESOURCE,
    }
    assert resources.phenotype_spec_version == "1.0.0"
    assert len(resources.phenotype_spec_sha256) == 64


@pytest.mark.licensed_clamp
def test_separately_licensed_resources_match_production_fingerprints() -> None:
    resources = load_clamp_resources(
        default_project_dir(),
        manifest_path=Path("config/clamp_ards_resource_manifest.json"),
    )

    assert len(resources.resource_sha256) == 3


def test_authorized_phenotype_contains_all_23_terms_in_frozen_order() -> None:
    resources = load_clamp_resources(
        default_project_dir(),
        manifest_path=default_resource_manifest_path(),
    )

    assert [entry.term for entry in resources.dictionary] == [
        "congestive heart failure",
        "left and right-sided infiltrates",
        "right and left-sided infiltrates",
        "ARDS",
        "pulmonary edema",
        "infiltrates",
        "opacities",
        "airspace",
        "aspiration",
        "consolidation",
        "pneumonia",
        "bilateral",
        "biapical",
        "bibasal",
        "bibasilar",
        "widespread",
        "diffuse",
        "perihilar",
        "parahilar",
        "multifocal",
        "multi-focal",
        "extensive",
        "throughout",
    ]


def test_packaged_resource_manifest_matches_governance_source() -> None:
    packaged = (
        importlib.resources.files("ards_cxr_benchmark.clamp_ards")
        .joinpath(PACKAGED_RESOURCE_MANIFEST)
        .read_bytes()
    )

    assert packaged == Path("config/clamp_ards_resource_manifest.json").read_bytes()


def test_packaged_resource_manifest_is_v3_external_contract() -> None:
    payload, _ = clamp_resource_module._load_resource_manifest(None)

    assert payload["manifest_version"] == 3
    assert set(payload["runtime_required_files"]) == {
        ABBREVIATION_RESOURCE,
        ASSERTION_RESOURCE,
        TOKEN_RULE_RESOURCE,
    }
    assert payload["phenotype_spec"]["attribution"] == "Dan Knox"


def test_default_resource_loading_rejects_tampered_required_file(tmp_path: Path) -> None:
    manifest_path = default_resource_manifest_path()
    manifest = json.loads(manifest_path.read_text())
    project = tmp_path / "project"
    for relative in manifest["runtime_required_files"]:
        source = default_project_dir() / relative
        destination = project / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    token_rules = project / TOKEN_RULE_RESOURCE
    payload = token_rules.read_bytes()
    token_rules.write_bytes(payload[:-1] + b" ")

    with pytest.raises(ValueError, match="hashes differ from frozen manifest"):
        load_clamp_resources(project, manifest_path=manifest_path)


def test_missing_external_resources_fail_with_precise_inventory(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="defaultNegexDict.txt") as error:
        load_clamp_resources(tmp_path, manifest_path=default_resource_manifest_path())

    assert "defaultAbbrs.txt" in str(error.value)
    assert "defaultTokenRule.txt" in str(error.value)


def test_manifest_v2_remains_readable_with_external_resource_boundary(tmp_path: Path) -> None:
    manifest = json.loads(default_resource_manifest_path().read_text(encoding="utf-8"))
    manifest["manifest_version"] = 2
    manifest.pop("phenotype_spec")
    manifest.pop("resource_contract")
    manifest_path = tmp_path / "manifest-v2.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    resources = load_clamp_resources(default_project_dir(), manifest_path=manifest_path)

    assert len(resources.resource_sha256) == 3
    assert resources.phenotype_spec_version == "1.0.0"


def test_packaged_resource_manifest_missing_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(clamp_resource_module.importlib.resources, "files", lambda _name: tmp_path)

    with pytest.raises(FileNotFoundError, match="Packaged frozen CLAMP resource manifest"):
        clamp_resource_module._load_resource_manifest(None)


def test_explicit_malformed_resource_manifest_fails_closed(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text("not-json", encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid CLAMP resource manifest JSON"):
        clamp_resource_module._load_resource_manifest(manifest)


@pytest.mark.licensed_clamp
@pytest.mark.skipif(sys.platform != "win32", reason="PowerShell collector runs only on Windows")
def test_windows_provenance_collector_records_frozen_project_and_outputs(
    tmp_path: Path,
) -> None:
    pwsh = shutil.which("pwsh")
    java = shutil.which("java")
    if pwsh is None or java is None:
        pytest.skip("PowerShell 7 and Java are required for the collector integration smoke")

    root = Path(__file__).resolve().parents[1]
    project_dir = default_project_dir()
    collector = root / "scripts/windows/collect_clamp_ards_parity_provenance.ps1"
    manifest = json.loads(
        (root / "config/clamp_ards_resource_manifest.json").read_text(encoding="utf-8")
    )
    output_dir = tmp_path / "run_1" / "Output"
    output_dir.mkdir(parents=True)
    output_payloads = {
        "a.txt": b"Start\tEnd\tSemantic\tCUI\tAssertion\tEntity\n",
        "b.xmi": b'<?xml version="1.0" encoding="UTF-8"?><xmi:XMI/>\n',
    }
    for name, payload in output_payloads.items():
        (output_dir / name).write_bytes(payload)
    (output_dir / "ignored.json").write_text("{}\n", encoding="utf-8")

    provenance_path = tmp_path / "returned" / "run_1_provenance.json"
    manual_command = "Automated synthetic collector integration smoke"
    process = subprocess.run(
        [
            pwsh,
            "-NoLogo",
            "-NoProfile",
            "-NonInteractive",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(collector),
            "-RunLabel",
            "run_1",
            "-ClampVersion",
            "1.6.6",
            "-ClampBuild",
            "windows-ci-smoke",
            "-ProjectCommit",
            manifest["project_commit"],
            "-ProjectDir",
            str(project_dir),
            "-OutputDir",
            str(output_dir),
            "-ProvenanceOutput",
            str(provenance_path),
            "-ExportSettings",
            "TXT and XMI; UTF-8",
            "-OffsetConvention",
            "half-open UTF-16 code units",
            "-NullConvention",
            "literal null in TXT",
            "-ManualCommand",
            manual_command,
            "-StartedAtUtc",
            "2026-01-01T00:00:00Z",
            "-FinishedAtUtc",
            "2026-01-01T00:01:00Z",
        ],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )

    provenance_bytes = provenance_path.read_bytes()
    assert not provenance_bytes.startswith(b"\xef\xbb\xbf")
    assert provenance_bytes.endswith(b"\n")
    provenance = json.loads(provenance_bytes)
    assert provenance["schema_version"] == 1
    assert provenance["run_label"] == "run_1"
    assert provenance["started_at_utc"] == "2026-01-01T00:00:00Z"
    assert provenance["finished_at_utc"] == "2026-01-01T00:01:00Z"
    assert provenance["clamp"] == {"version": "1.6.6", "build": "windows-ci-smoke"}
    assert provenance["project"] == {
        "commit": manifest["project_commit"],
        "files_sha256": manifest["files"],
    }
    assert provenance["resources_sha256"] == {
        path: digest for path, digest in manifest["files"].items() if path.startswith("Components/")
    }
    assert provenance["export_settings"] == "TXT and XMI; UTF-8"
    assert provenance["offset_convention"] == "half-open UTF-16 code units"
    assert provenance["null_convention"] == "literal null in TXT"
    assert provenance["manual_commands"] == [manual_command]
    assert provenance["java"]["version_output"]
    assert all(provenance["windows"].values())

    expected_records = [
        {
            "relative_path": name,
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
        for name, payload in output_payloads.items()
    ]
    assert provenance["output_files"] == expected_records
    assert "Recorded 2 output files" in process.stdout

    checksum_path = provenance_path.with_suffix(".SHA256SUMS")
    checksum_bytes = checksum_path.read_bytes()
    assert not checksum_bytes.startswith(b"\xef\xbb\xbf")
    assert checksum_bytes == "".join(
        f"{record['sha256']}  {record['relative_path']}\n" for record in expected_records
    ).encode("utf-8")


def test_resource_manifest_keys_are_posix_normalized_for_windows_paths() -> None:
    root = PureWindowsPath(r"C:\repo\clamp_ARDS")
    resource = root / "Components" / "Tokenizer" / "defaultTokenRule.txt"

    assert _manifest_key(resource, root) == "Components/Tokenizer/defaultTokenRule.txt"


def test_python_runtime_accepts_required_only_user_supplied_resources(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path = default_resource_manifest_path()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    project = tmp_path / "local_ards_project"
    for relative in manifest["runtime_required_files"]:
        source = default_project_dir() / relative
        destination = project / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)

    resources = load_clamp_resources(project, manifest_path=manifest_path)
    monkeypatch.setenv(PROJECT_DIR_ENV, str(project))

    assert len(resources.resource_sha256) == 3
    assert resources.split_patterns == ()
    assert resources.section_headers == ()
    assert default_project_dir() == project.resolve()


def test_git_attributes_pin_clamp_resource_and_fixture_byte_contracts() -> None:
    text_resources = [
        "config/clamp_ards_resource_manifest.json",
        "src/ards_cxr_benchmark/clamp_ards/data/clamp_ards_resource_manifest.json",
        "src/ards_cxr_benchmark/clamp_ards/data/legacy_ards_phenotype_spec.json",
        "tests/fixtures/clamp_ards_external_resources/manifest.json",
        "tests/fixtures/clamp_ards_external_resources/Components/Tokenizer/DF_Clamp_tokenizer/defaultTokenRule.txt",
        "tests/fixtures/clamp_ards_parity/cases.yaml",
        "tests/fixtures/clamp_ards_parity/manifest.csv",
        "tests/fixtures/clamp_ards_parity/provenance.json",
        "tests/fixtures/clamp_ards_parity/SHA256SUMS",
        "tests/fixtures/clamp_ards_parity/clamp_expected/README.md",
    ]
    exact_input = "tests/fixtures/clamp_ards_parity/input/sentence_input_02.txt"

    text_attributes = [
        subprocess.run(
            ["git", "check-attr", "text", "eol", "--", resource],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        for resource in text_resources
    ]
    input_attributes = subprocess.run(
        ["git", "check-attr", "text", "eol", "--", exact_input],
        check=True,
        capture_output=True,
        text=True,
    ).stdout

    assert all("text: set" in attributes for attributes in text_attributes)
    assert all("eol: lf" in attributes for attributes in text_attributes)
    assert "text: unset" in input_attributes


def test_utf16_offset_map_counts_supplementary_characters_as_two_code_units() -> None:
    offsets = Utf16OffsetMap.from_text("😀🫁 ARDS")

    assert offsets.span(0, 2) == (0, 4)
    assert offsets.span(3, 7) == (5, 9)
    assert offsets.python_span(0, 4) == (0, 2)
    assert offsets.python_span(5, 9) == (3, 7)
    assert Utf16OffsetMap.from_text("ARDS").span(0, 4) == (0, 4)


def test_utf16_offset_map_rejects_surrogate_interior_and_out_of_range() -> None:
    offsets = Utf16OffsetMap.from_text("😀 ARDS")

    with pytest.raises(ValueError, match="inside a surrogate pair"):
        offsets.python_index(1)
    with pytest.raises(ValueError, match="outside"):
        offsets.python_index(8)


def test_public_entity_offsets_use_clamp_utf16_coordinates() -> None:
    text = "😀 ARDS"

    entity = run_legacy_ards_clamp_mirror(text)[0]
    internal = load_legacy_mirror().trace(text).final_entities[0]

    assert (internal.start, internal.end) == (2, 6)
    assert (entity.start, entity.end) == (3, 7)
    assert entity.covered_text(text) == "ARDS"


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("opacities", "opac"),
        ("opacity", "opac"),
        ("consolidated", "consolid"),
        ("consolidations", "consolid"),
        ("bilaterally", "bilater"),
        ("bilateral", "bilater"),
        ("multifocality", "multifoc"),
    ],
)
def test_stemmer_reproduces_observed_clamp_inflections(source: str, expected: str) -> None:
    assert PorterCompatibilityStemmer().stem(source) == expected


@given(st.text(alphabet=st.characters(blacklist_categories=("Cs",)), max_size=200))
def test_tokenizer_offsets_are_ordered_nonempty_and_cover_exact_text(text: str) -> None:
    tokenizer = load_legacy_mirror().tokenizer
    tokens = tokenizer.tokenize(text)

    assert all(0 <= token.start < token.end <= len(text) for token in tokens)
    assert all(left.end <= right.start for left, right in zip(tokens, tokens[1:], strict=False))
    assert all(token.covered_text(text) == text[token.start : token.end] for token in tokens)


def test_tokenizer_preserves_observed_clamp_alphanumeric_asymmetry() -> None:
    text = "2week O2 POD1 CABGx3 C7 multi-focal"
    tokenizer = load_legacy_mirror().tokenizer

    assert [token.covered_text(text) for token in tokenizer.tokenize(text)] == [
        "2",
        "week",
        "O2",
        "POD1",
        "CABGx3",
        "C7",
        "multi",
        "-",
        "focal",
    ]


@pytest.mark.parametrize(
    ("text", "expected_spans"),
    [
        ("bilateral infiltrates", [(10, 21, "infiltrates")]),
        ("infiltrates bilaterally", [(12, 23, "bilaterally")]),
        ("diffuse airspace opacities", [(8, 16, "airspace"), (17, 26, "opacities")]),
        ("pulmonary edema", [(0, 15, "pulmonary edema")]),
        ("multi-focal pneumonia", []),
        ("No pulmonary edema.", []),
        ("No change in pulmonary edema.", []),
        ("no ptx. worsened pulmonary edema.", []),
        ("IMPRESSION: No change in pulmonary edema.", [(25, 40, "pulmonary edema")]),
        ("There is no change in pulmonary edema.", [(22, 37, "pulmonary edema")]),
        (
            "No acute chest abnormality with no change in improving pulmonary edema.",
            [(55, 70, "pulmonary edema")],
        ),
        ("not necessarily improvement of pulmonary edema", [(31, 46, "pulmonary edema")]),
        ("widespread ARDS with aspiration", [(11, 15, "ARDS")]),
        (
            "diffuse bilateral opacities consistent with pneumonia",
            [(18, 27, "opacities")],
        ),
    ],
)
def test_safe_synthetic_compatibility_cases(
    text: str,
    expected_spans: list[tuple[int, int, str]],
) -> None:
    entities = run_legacy_ards_clamp_mirror(text)

    assert [(entity.start, entity.end, entity.covered_text(text)) for entity in entities] == (
        expected_spans
    )


def test_batch_runner_writes_entities_and_text_free_predictions(tmp_path: Path) -> None:
    input_path = tmp_path / "input.jsonl"
    input_path.write_text(
        "\n".join(
            [
                json.dumps({"doc": "one", "text": "bilateral infiltrates"}),
                json.dumps({"doc": "two", "text": "No pulmonary edema."}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    entity_output = tmp_path / "entities.parquet"
    prediction_output = tmp_path / "predictions.parquet"

    summary = run_clamp_ards_batch(
        input_path=input_path,
        entity_output=entity_output,
        prediction_output=prediction_output,
        id_column="doc",
        text_column="text",
        show_progress=False,
    )

    entities = pq.read_table(entity_output).to_pylist()
    predictions = pq.read_table(prediction_output)
    assert summary.document_count == 2
    assert summary.positive_document_count == 1
    assert summary.entity_count == 1
    assert summary.implementation_version == "0.3.0"
    assert len(summary.source_input_sha256) == 64
    assert len(summary.resource_sha256) == 3
    assert summary.phenotype_spec_version == "1.0.0"
    assert len(summary.phenotype_spec_sha256) == 64
    assert entities[0]["entity_text"] == "infiltrates"
    assert "entity_text" not in predictions.column_names
    assert "report_text" not in predictions.column_names
    assert [row["prediction_label"] for row in predictions.to_pylist()] == [1, 0]


def test_batch_runner_serializes_utf16_entity_offsets(tmp_path: Path) -> None:
    input_path = tmp_path / "input.jsonl"
    input_path.write_text(
        json.dumps({"doc": "emoji", "text": "😀 ARDS"}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    entity_output = tmp_path / "entities.parquet"
    prediction_output = tmp_path / "predictions.parquet"

    summary = run_clamp_ards_batch(
        input_path=input_path,
        entity_output=entity_output,
        prediction_output=prediction_output,
        id_column="doc",
        text_column="text",
        show_progress=False,
    )

    entity = pq.read_table(entity_output).to_pylist()[0]
    assert (entity["start"], entity["end"], entity["entity_text"]) == (3, 7, "ARDS")
    assert pq.read_schema(entity_output).metadata[b"offset_coordinate_system"] == (
        b"utf16_code_units"
    )
    assert summary.offset_coordinate_system == "utf16_code_units"


def test_batch_runner_rejects_duplicate_document_ids_without_replacing_outputs(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "input.csv"
    input_path.write_text("doc,text\none,ARDS\none,ARDS\n", encoding="utf-8")
    entity_output = tmp_path / "entities.parquet"
    prediction_output = tmp_path / "predictions.parquet"
    entity_output.write_text("existing", encoding="utf-8")

    with pytest.raises(ValueError, match="Duplicate input document ID"):
        run_clamp_ards_batch(
            input_path=input_path,
            entity_output=entity_output,
            prediction_output=prediction_output,
            id_column="doc",
            text_column="text",
            show_progress=False,
        )

    assert entity_output.read_text(encoding="utf-8") == "existing"
    assert not prediction_output.exists()
    assert not list(tmp_path.glob("*.partial"))


def test_python_clamp_cli_requires_isolated_outputs_for_custom_run(tmp_path: Path) -> None:
    input_path = tmp_path / "input.jsonl"
    input_path.write_text(
        json.dumps({"study_id": 1, "report_text": "ARDS"}) + "\n",
        encoding="utf-8",
    )

    process = subprocess.run(
        [
            sys.executable,
            "scripts/run_python_clamp_ards.py",
            "--config",
            "config/config.example.yaml",
            "--input",
            str(input_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert process.returncode != 0
    assert "require explicit noncanonical output paths" in process.stderr


def test_python_clamp_cli_writes_custom_run_only_to_explicit_outputs(tmp_path: Path) -> None:
    input_path = tmp_path / "input.jsonl"
    input_path.write_text(
        json.dumps({"doc": "one", "text": "ARDS"}) + "\n",
        encoding="utf-8",
    )
    entity_output = tmp_path / "custom" / "entities.parquet"
    prediction_output = tmp_path / "custom" / "predictions.parquet"
    summary_output = tmp_path / "custom" / "summary.json"

    process = subprocess.run(
        [
            sys.executable,
            "scripts/run_python_clamp_ards.py",
            "--config",
            "config/config.example.yaml",
            "--input",
            str(input_path),
            "--id-col",
            "doc",
            "--text-col",
            "text",
            "--id-prefix",
            "",
            "--entity-output",
            str(entity_output),
            "--prediction-output",
            str(prediction_output),
            "--summary-output",
            str(summary_output),
            "--no-progress",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert process.returncode == 0, process.stderr
    assert pq.read_table(entity_output).num_rows == 1
    assert pq.read_table(prediction_output).num_rows == 1
    assert json.loads(summary_output.read_text(encoding="utf-8"))["document_count"] == 1


def test_python_clamp_cli_rejects_custom_run_at_canonical_paths(tmp_path: Path) -> None:
    input_path = tmp_path / "input.jsonl"
    input_path.write_text(
        json.dumps({"study_id": 1, "report_text": "ARDS"}) + "\n",
        encoding="utf-8",
    )
    entity_output = tmp_path / "canonical" / "entities.parquet"
    prediction_output = tmp_path / "canonical" / "predictions.parquet"
    summary_output = tmp_path / "canonical" / "summary.json"
    config = tmp_path / "config.yaml"
    config.write_text(
        "clamp_ards:\n"
        f"  python_entity_output: {entity_output}\n"
        f"  python_prediction_output: {prediction_output}\n"
        f"  python_summary_output: {summary_output}\n",
        encoding="utf-8",
    )

    process = subprocess.run(
        [
            sys.executable,
            "scripts/run_python_clamp_ards.py",
            "--config",
            str(config),
            "--input",
            str(input_path),
            "--entity-output",
            str(entity_output),
            "--prediction-output",
            str(prediction_output),
            "--summary-output",
            str(summary_output),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert process.returncode != 0
    assert "must not overwrite configured full-corpus outputs" in process.stderr
    assert not entity_output.exists()
    assert not prediction_output.exists()
    assert not summary_output.exists()


def test_strict_parity_comparator_passes_exact_multisets(tmp_path: Path) -> None:
    paths = _write_parity_tables(tmp_path, actual_text="infiltrates")

    result = compare_clamp_ards_outputs(**paths)

    assert result.passed
    assert result.summary["missing_entities"] == 0
    assert result.summary["document_label_mismatches"] == 0


def test_strict_parity_comparator_fails_entity_difference(tmp_path: Path) -> None:
    paths = _write_parity_tables(tmp_path, actual_text="opacities")

    result = compare_clamp_ards_outputs(**paths)

    assert not result.passed
    assert result.summary["missing_entities"] == 1
    assert result.summary["unexpected_entities"] == 1
    assert result.summary["field_mismatches"] == 1
    assert result.summary["document_label_mismatches"] == 0
    assert all("infiltrates" not in json.dumps(row) for row in result.mismatches)


def test_parity_cli_returns_nonzero_for_required_mismatch(tmp_path: Path) -> None:
    paths = _write_parity_tables(tmp_path, actual_text="opacities")
    summary_output = tmp_path / "summary.json"
    mismatch_output = tmp_path / "mismatches.csv"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/compare_clamp_python_parity.py",
            "--config",
            "config/config.example.yaml",
            "--expected-entities",
            str(paths["expected_entities"]),
            "--expected-predictions",
            str(paths["expected_predictions"]),
            "--actual-entities",
            str(paths["actual_entities"]),
            "--actual-predictions",
            str(paths["actual_predictions"]),
            "--summary-output",
            str(summary_output),
            "--mismatch-output",
            str(mismatch_output),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert json.loads(summary_output.read_text(encoding="utf-8"))["passed"] is False
    assert "infiltrates" not in mismatch_output.read_text(encoding="utf-8")


def test_parity_comparator_preserves_duplicate_multiplicity(tmp_path: Path) -> None:
    paths = _write_parity_tables(tmp_path, actual_text="infiltrates")
    expected = pq.read_table(paths["expected_entities"])
    pq.write_table(pa.concat_tables([expected, expected]), paths["expected_entities"])
    expected_predictions = pq.read_table(paths["expected_predictions"]).to_pylist()
    expected_predictions[0]["clamp_ards_entity_count"] = 2
    pq.write_table(
        pa.Table.from_pylist(
            expected_predictions, schema=pq.read_schema(paths["expected_predictions"])
        ),
        paths["expected_predictions"],
    )

    result = compare_clamp_ards_outputs(**paths)

    assert not result.passed
    assert result.summary["missing_entities"] == 1
    assert result.summary["multiplicity_mismatches"] == 1


@pytest.mark.parametrize("side", ["expected", "actual"])
def test_parity_rejects_entity_documents_without_prediction_rows(
    tmp_path: Path,
    side: str,
) -> None:
    paths = _write_parity_tables(tmp_path, actual_text="infiltrates")
    path = paths[f"{side}_entities"]
    table = pq.read_table(path)
    rows = table.to_pylist()
    rows[0]["clamp_doc_id"] = "orphan"
    pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), path)

    with pytest.raises(ValueError, match=rf"{side.capitalize()} entity table contains"):
        compare_clamp_ards_outputs(**paths)


@pytest.mark.parametrize("side", ["expected", "actual"])
@pytest.mark.parametrize("value", [None, " "])
def test_parity_rejects_null_or_blank_entity_document_ids(
    tmp_path: Path,
    side: str,
    value: str | None,
) -> None:
    paths = _write_parity_tables(tmp_path, actual_text="infiltrates")
    path = paths[f"{side}_entities"]
    table = pq.read_table(path)
    rows = table.to_pylist()
    rows[0]["clamp_doc_id"] = value
    pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), path)

    with pytest.raises(ValueError, match=r"(Null|Blank) clamp_doc_id in entity table"):
        compare_clamp_ards_outputs(**paths)


@pytest.mark.parametrize("side", ["expected", "actual"])
@pytest.mark.parametrize("value", [None, " "])
def test_parity_rejects_null_or_blank_prediction_document_ids(
    tmp_path: Path,
    side: str,
    value: str | None,
) -> None:
    paths = _write_parity_tables(tmp_path, actual_text="infiltrates")
    path = paths[f"{side}_predictions"]
    table = pq.read_table(path)
    rows = table.to_pylist()
    rows[0]["clamp_doc_id"] = value
    pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), path)

    with pytest.raises(ValueError, match=r"(Null|Blank) clamp_doc_id in prediction table"):
        compare_clamp_ards_outputs(**paths)


@pytest.mark.parametrize("side", ["expected", "actual"])
@pytest.mark.parametrize("table_kind", ["entities", "predictions"])
def test_parity_rejects_nan_document_ids(
    tmp_path: Path,
    side: str,
    table_kind: str,
) -> None:
    paths = _write_parity_tables(tmp_path, actual_text="infiltrates")
    path = paths[f"{side}_{table_kind}"]
    table = pq.read_table(path)
    index = table.column_names.index("clamp_doc_id")
    table = table.set_column(index, "clamp_doc_id", pa.array([float("nan")]))
    pq.write_table(table, path)

    table_name = "entity" if table_kind == "entities" else "prediction"
    with pytest.raises(ValueError, match=rf"Null clamp_doc_id in {table_name} table"):
        compare_clamp_ards_outputs(**paths)


@pytest.mark.parametrize("side", ["expected", "actual"])
def test_parity_rejects_prediction_entity_count_inconsistency(
    tmp_path: Path,
    side: str,
) -> None:
    paths = _write_parity_tables(tmp_path, actual_text="infiltrates")
    _write_prediction_table(
        paths[f"{side}_predictions"],
        status_columns={"prediction_status": "evaluable"},
        label=1,
        count=2,
    )

    with pytest.raises(ValueError, match=rf"{side.capitalize()} evaluable prediction/entity"):
        compare_clamp_ards_outputs(**paths)


@pytest.mark.parametrize("side", ["expected", "actual"])
def test_parity_rejects_prediction_label_count_inconsistency(
    tmp_path: Path,
    side: str,
) -> None:
    paths = _write_parity_tables(tmp_path, actual_text="infiltrates")
    _write_prediction_table(
        paths[f"{side}_predictions"],
        status_columns={"prediction_status": "evaluable"},
        label=0,
        count=1,
    )

    with pytest.raises(ValueError, match=rf"{side.capitalize()} evaluable prediction label"):
        compare_clamp_ards_outputs(**paths)


def test_parity_maps_teacher_parse_status_to_evaluable(tmp_path: Path) -> None:
    paths = _write_parity_tables(tmp_path, actual_text="infiltrates")
    _write_prediction_table(
        paths["expected_predictions"],
        status_columns={"clamp_parse_status": "parsed"},
        label=1,
        count=1,
    )

    result = compare_clamp_ards_outputs(**paths)

    assert result.passed


@pytest.mark.parametrize("parse_status", ["missing_output", "duplicate_output", "parse_error"])
def test_parity_compares_non_evaluable_null_teacher_rows(
    tmp_path: Path,
    parse_status: str,
) -> None:
    paths = _write_parity_tables(tmp_path, actual_text="infiltrates")
    _write_prediction_table(
        paths["expected_predictions"],
        status_columns={"clamp_parse_status": parse_status},
        label=None,
        count=None,
    )
    expected_schema = pq.read_schema(paths["expected_entities"])
    pq.write_table(pa.Table.from_pylist([], schema=expected_schema), paths["expected_entities"])

    result = compare_clamp_ards_outputs(**paths)

    assert not result.passed
    assert result.summary["document_status_mismatches"] == 1
    assert result.summary["document_label_mismatches"] == 1
    assert result.summary["document_count_mismatches"] == 1


def test_parity_stale_non_evaluable_values_cannot_pass_status_parity(tmp_path: Path) -> None:
    paths = _write_parity_tables(tmp_path, actual_text="infiltrates")
    _write_prediction_table(
        paths["expected_predictions"],
        status_columns={"clamp_parse_status": "parse_error"},
        label=1,
        count=1,
    )

    with pytest.raises(ValueError, match="non-evaluable prediction must have no entities"):
        compare_clamp_ards_outputs(**paths)


@pytest.mark.parametrize(
    ("status_columns", "error"),
    [
        (
            {"prediction_status": "evaluable", "clamp_parse_status": "parse_error"},
            "Conflicting prediction status columns",
        ),
        ({"clamp_parse_status": "unknown"}, "Unknown clamp_parse_status"),
        ({"clamp_parse_status": None}, "Null/blank clamp_parse_status"),
        ({}, "must contain prediction_status or clamp_parse_status"),
    ],
)
def test_parity_rejects_invalid_status_contracts(
    tmp_path: Path,
    status_columns: dict[str, str | None],
    error: str,
) -> None:
    paths = _write_parity_tables(tmp_path, actual_text="infiltrates")
    _write_prediction_table(
        paths["expected_predictions"],
        status_columns=status_columns,
        label=1,
        count=1,
    )

    with pytest.raises(ValueError, match=error):
        compare_clamp_ards_outputs(**paths)


def test_parity_canonicalizes_same_span_entities_before_field_comparison(
    tmp_path: Path,
) -> None:
    paths = _write_parity_tables(tmp_path, actual_text="infiltrates")
    expected_table = pq.read_table(paths["expected_entities"])
    first = expected_table.to_pylist()[0]
    first["cui"] = "C1"
    second = {**first, "cui": "C2"}
    pq.write_table(
        pa.Table.from_pylist([first, second], schema=expected_table.schema),
        paths["expected_entities"],
    )
    pq.write_table(
        pa.Table.from_pylist([second, first], schema=expected_table.schema),
        paths["actual_entities"],
    )
    for prediction_path in (paths["expected_predictions"], paths["actual_predictions"]):
        _write_prediction_table(
            prediction_path,
            status_columns={"prediction_status": "evaluable"},
            label=1,
            count=2,
        )

    result = compare_clamp_ards_outputs(**paths)

    assert result.passed
    assert result.summary["field_mismatches"] == 0
    assert result.summary["output_order_differences"] == 1


def test_xmi_parser_preserves_offsets_nulls_and_duplicate_entities() -> None:
    payload = b"""<?xml version='1.0' encoding='UTF-8'?>
<xmi:XMI xmlns:xmi="http://www.omg.org/XMI" xmlns:cas="http:///uima/cas.ecore"
 xmlns:textspan="http:///org/apache/ctakes/typesystem/type/textspan.ecore"
 xmlns:syntax="http:///org/apache/ctakes/typesystem/type/syntax.ecore"
 xmlns:clamp="http:///edu/uth/clamp/nlp/typesystem.ecore">
 <cas:Sofa xmi:id="1" sofaNum="1" sofaID="_InitialView" sofaString="ARDS"/>
 <textspan:Sentence xmi:id="2" sofa="1" begin="0" end="4" sentenceNumber="0"/>
 <syntax:BaseToken xmi:id="3" sofa="1" begin="0" end="4" tokenNumber="0"/>
 <clamp:ClampNameEntityUIMA xmi:id="4" sofa="1" begin="0" end="4"
  semanticTag="ARDS" assertion="present"/>
 <clamp:ClampNameEntityUIMA xmi:id="5" sofa="1" begin="0" end="4"
  semanticTag="ARDS" assertion="present"/>
</xmi:XMI>"""

    document = parse_clamp_xmi(payload)

    assert document.text == "ARDS"
    assert len(document.sentences) == 1
    assert len(document.tokens) == 1
    assert len(document.entities) == 2
    assert document.entities[0].cui is None
    assert document.entities[0].attribute is None


def test_xmi_characterization_uses_utf16_coordinates_for_python_trace() -> None:
    text = "😀 ARDS"
    payload = f"""<?xml version='1.0' encoding='UTF-8'?>
<xmi:XMI xmlns:xmi="http://www.omg.org/XMI" xmlns:cas="http:///uima/cas.ecore"
 xmlns:textspan="http:///org/apache/ctakes/typesystem/type/textspan.ecore"
 xmlns:syntax="http:///org/apache/ctakes/typesystem/type/syntax.ecore"
 xmlns:clamp="http:///edu/uth/clamp/nlp/typesystem.ecore">
 <cas:Sofa xmi:id="1" sofaNum="1" sofaID="_InitialView" sofaString="{text}"/>
 <textspan:Sentence xmi:id="2" sofa="1" begin="0" end="7" sentenceNumber="0"/>
 <syntax:BaseToken xmi:id="3" sofa="1" begin="0" end="2" tokenNumber="0"/>
 <syntax:BaseToken xmi:id="4" sofa="1" begin="3" end="7" tokenNumber="1"/>
 <clamp:ClampNameEntityUIMA xmi:id="5" sofa="1" begin="3" end="7"
  semanticTag="ARDS" assertion="present"/>
</xmi:XMI>""".encode()

    document = parse_clamp_xmi(payload)
    trace = load_legacy_mirror().trace(text)
    offsets = Utf16OffsetMap.from_text(text)

    assert [offsets.span(item.start, item.end) for item in trace.sentences] == [
        (item.start, item.end) for item in document.sentences
    ]
    assert [offsets.span(item.start, item.end) for item in trace.tokens] == [
        (item.start, item.end) for item in document.tokens
    ]
    internal_entity = trace.final_entities[0]
    assert offsets.span(internal_entity.start, internal_entity.end) == (
        document.entities[0].start,
        document.entities[0].end,
    )
    assert document.sentences[0].covered_text(text) == text
    assert [item.covered_text(text) for item in document.tokens] == ["😀", "ARDS"]
    assert document.entities[0].covered_text(text) == "ARDS"


def test_xmi_parser_rejects_offsets_inside_surrogate_pairs() -> None:
    payload = """<?xml version='1.0' encoding='UTF-8'?>
<xmi:XMI xmlns:xmi="http://www.omg.org/XMI" xmlns:cas="http:///uima/cas.ecore"
 xmlns:syntax="http:///org/apache/ctakes/typesystem/type/syntax.ecore">
 <cas:Sofa xmi:id="1" sofaNum="1" sofaID="_InitialView" sofaString="😀 ARDS"/>
 <syntax:BaseToken xmi:id="2" sofa="1" begin="1" end="2" tokenNumber="0"/>
</xmi:XMI>""".encode()

    with pytest.raises(ValueError, match="inside a surrogate pair"):
        parse_clamp_xmi(payload)


def _write_prediction_table(
    path: Path,
    *,
    status_columns: dict[str, str | None],
    label: int | None,
    count: int | None,
) -> None:
    fields = [("clamp_doc_id", pa.string())]
    fields.extend((column, pa.string()) for column in status_columns)
    fields.extend(
        [
            ("prediction_label", pa.int8()),
            ("clamp_ards_entity_count", pa.int64()),
        ]
    )
    row = {
        "clamp_doc_id": "s1",
        **status_columns,
        "prediction_label": label,
        "clamp_ards_entity_count": count,
    }
    pq.write_table(pa.Table.from_pylist([row], schema=pa.schema(fields)), path)


def _write_parity_tables(
    root: Path,
    *,
    actual_text: str,
    actual_label: int = 1,
) -> dict[str, Path]:
    entity_schema = pa.schema(
        [
            ("clamp_doc_id", pa.string()),
            ("start", pa.int64()),
            ("end", pa.int64()),
            ("entity_text", pa.string()),
            ("semantic_tag", pa.string()),
            ("assertion", pa.string()),
            ("cui", pa.string()),
            ("attribute", pa.string()),
        ]
    )
    prediction_schema = pa.schema(
        [
            ("clamp_doc_id", pa.string()),
            ("prediction_status", pa.string()),
            ("prediction_label", pa.int8()),
            ("clamp_ards_entity_count", pa.int64()),
        ]
    )
    expected_entities = root / "expected_entities.parquet"
    actual_entities = root / "actual_entities.parquet"
    expected_predictions = root / "expected_predictions.parquet"
    actual_predictions = root / "actual_predictions.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "clamp_doc_id": "s1",
                    "start": 10,
                    "end": 21,
                    "entity_text": "infiltrates",
                    "semantic_tag": "ARDS",
                    "assertion": "present",
                    "cui": None,
                    "attribute": None,
                }
            ],
            schema=entity_schema,
        ),
        expected_entities,
    )
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "clamp_doc_id": "s1",
                    "start": 10,
                    "end": 21,
                    "entity_text": actual_text,
                    "semantic_tag": "ARDS",
                    "assertion": "present",
                    "cui": None,
                    "attribute": None,
                }
            ],
            schema=entity_schema,
        ),
        actual_entities,
    )
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "clamp_doc_id": "s1",
                    "prediction_status": "evaluable",
                    "prediction_label": 1,
                    "clamp_ards_entity_count": 1,
                }
            ],
            schema=prediction_schema,
        ),
        expected_predictions,
    )
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "clamp_doc_id": "s1",
                    "prediction_status": "evaluable",
                    "prediction_label": actual_label,
                    "clamp_ards_entity_count": actual_label,
                }
            ],
            schema=prediction_schema,
        ),
        actual_predictions,
    )
    return {
        "expected_entities": expected_entities,
        "expected_predictions": expected_predictions,
        "actual_entities": actual_entities,
        "actual_predictions": actual_predictions,
    }
