from __future__ import annotations

import csv
import json
import shutil
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from ards_cxr_benchmark.clamp_ards.batch import iter_fixture_documents
from ards_cxr_benchmark.clamp_ards.fixture_parity import (
    INTERMEDIATE_MISMATCH_SCHEMA,
    compare_clamp_ards_fixture,
)
from ards_cxr_benchmark.clamp_ards.fixtures import (
    EXPECTED_CASE_COUNT,
    EXPECTED_ENTITY_FIELDS,
    validate_fixture,
    write_sha256s,
)
from ards_cxr_benchmark.clamp_ards.pipeline import load_legacy_mirror
from ards_cxr_benchmark.clamp_ards.resources import default_resource_manifest_path
from ards_cxr_benchmark.clamp_ards.tokenization import Utf16OffsetMap


def test_pending_fixture_cli_is_allowed_only_explicitly(
    tmp_path: Path,
    pending_clamp_fixture: Path,
) -> None:
    common = [
        sys.executable,
        "scripts/compare_clamp_ards_fixtures.py",
        "--fixture-root",
        str(pending_clamp_fixture),
        "--output-dir",
        str(tmp_path / "outputs"),
    ]

    strict = subprocess.run(common, check=False, capture_output=True, text=True)
    allowed = subprocess.run(
        [*common, "--allow-pending"], check=False, capture_output=True, text=True
    )

    assert strict.returncode == 2
    assert allowed.returncode == 0
    assert '"status": "pending"' in allowed.stdout


def test_allow_pending_rejects_partial_expected_output(
    tmp_path: Path,
    pending_clamp_fixture: Path,
) -> None:
    fixture = tmp_path / "fixture"
    shutil.copytree(pending_clamp_fixture, fixture)
    (fixture / "clamp_expected" / "partial.tsv").write_text("partial\n", encoding="utf-8")

    process = subprocess.run(
        [
            sys.executable,
            "scripts/compare_clamp_ards_fixtures.py",
            "--fixture-root",
            str(fixture),
            "--output-dir",
            str(tmp_path / "outputs"),
            "--allow-pending",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert process.returncode == 2
    assert "Invalid CLAMP fixture" in process.stderr


def test_fixture_iterator_preserves_exact_crlf_bytes(pending_clamp_fixture: Path) -> None:
    documents = dict(iter_fixture_documents(pending_clamp_fixture))
    expected = (
        (pending_clamp_fixture / "input" / "sentence_input_02.txt").read_bytes().decode("utf-8")
    )

    assert "\r\n" in expected
    assert documents["sentence_input_02"] == expected
    assert (
        documents["sentence_input_02"].encode("utf-8")
        == (pending_clamp_fixture / "input" / "sentence_input_02.txt").read_bytes()
    )


def test_complete_fixture_exact_parity_writes_empty_stable_ledgers(
    complete_fixture: Path,
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "outputs"

    result = compare_clamp_ards_fixture(
        fixture_root=complete_fixture,
        output_dir=output_dir,
    )

    assert result.passed
    assert result.summary["exact_sentence_documents"] == EXPECTED_CASE_COUNT
    assert result.summary["exact_token_documents"] == EXPECTED_CASE_COUNT
    assert result.summary["exact_final_entity_documents"] == EXPECTED_CASE_COUNT
    intermediate = output_dir / "intermediate_mismatches.parquet"
    assert pq.read_schema(intermediate) == INTERMEDIATE_MISMATCH_SCHEMA
    assert pq.read_table(intermediate).num_rows == 0
    assert pq.read_table(output_dir / "entity_mismatches.parquet").num_rows == 0
    assert pq.read_table(output_dir / "document_mismatches.parquet").num_rows == 0
    assert pq.read_table(output_dir / "order_mismatches.parquet").num_rows == 0
    markdown = (output_dir / "parity_summary.md").read_text(encoding="utf-8")
    assert "Strict status: **PASSED**" in markdown
    assert "dict_case_01_lower" not in markdown


@pytest.mark.parametrize(
    ("mutation", "expected_field"),
    [
        ("sentence", "exact_sentence_documents"),
        ("token", "exact_token_documents"),
        ("final", "exact_final_entity_documents"),
    ],
)
def test_intermediate_mismatch_fails_complete_fixture(
    complete_fixture: Path,
    tmp_path: Path,
    mutation: str,
    expected_field: str,
) -> None:
    fixture = _copy_complete_fixture(complete_fixture, tmp_path)
    _mutate_intermediate(fixture, mutation)

    result = compare_clamp_ards_fixture(
        fixture_root=fixture,
        output_dir=tmp_path / "outputs",
    )

    assert not result.passed
    assert result.summary[expected_field] == EXPECTED_CASE_COUNT - 1
    assert result.summary["intermediate_mismatch_positions"] > 0


@pytest.mark.parametrize(
    ("mutation", "summary_field"),
    [
        ("field", "field_mismatches"),
        ("multiplicity", "multiplicity_mismatches"),
        ("document", "document_label_mismatches"),
        ("order", "output_order_differences"),
    ],
)
def test_final_entity_and_document_mismatch_fails_complete_fixture(
    complete_fixture: Path,
    tmp_path: Path,
    mutation: str,
    summary_field: str,
) -> None:
    fixture = _copy_complete_fixture(complete_fixture, tmp_path)
    _mutate_expected_tsv(fixture, mutation)

    result = compare_clamp_ards_fixture(
        fixture_root=fixture,
        output_dir=tmp_path / "outputs",
    )

    assert not result.passed
    assert result.summary[summary_field] > 0


def test_unstable_legacy_txt_order_is_reported_but_not_required(
    complete_fixture: Path,
    tmp_path: Path,
) -> None:
    fixture = _copy_complete_fixture(complete_fixture, tmp_path)
    _mutate_expected_tsv(fixture, "order")
    _set_order_contract(fixture, txt_stable=False, xmi_stable=True)

    result = compare_clamp_ards_fixture(
        fixture_root=fixture,
        output_dir=tmp_path / "outputs",
    )

    assert result.passed
    assert result.summary["require_order"] is False
    assert result.summary["output_order_differences"] > 0


def test_unstable_legacy_xmi_order_uses_exact_final_entity_multiset(
    complete_fixture: Path,
    tmp_path: Path,
) -> None:
    fixture = _copy_complete_fixture(complete_fixture, tmp_path)
    _reverse_intermediate_final_order(fixture)
    _set_order_contract(fixture, txt_stable=True, xmi_stable=False)

    result = compare_clamp_ards_fixture(
        fixture_root=fixture,
        output_dir=tmp_path / "outputs",
    )

    assert result.passed
    assert result.summary["fixture_xmi_order_required"] is False
    assert result.summary["exact_final_entity_documents"] == EXPECTED_CASE_COUNT


def test_complete_fixture_cli_returns_one_for_valid_mismatch(
    complete_fixture: Path,
    tmp_path: Path,
) -> None:
    fixture = _copy_complete_fixture(complete_fixture, tmp_path)
    _mutate_expected_tsv(fixture, "field")

    process = subprocess.run(
        [
            sys.executable,
            "scripts/compare_clamp_ards_fixtures.py",
            "--fixture-root",
            str(fixture),
            "--output-dir",
            str(tmp_path / "outputs"),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert process.returncode == 1
    assert '"status": "failed"' in process.stdout


def test_fixture_make_target_returns_nonzero_for_valid_mismatch(
    complete_fixture: Path,
    tmp_path: Path,
) -> None:
    fixture = _copy_complete_fixture(complete_fixture, tmp_path)
    _mutate_expected_tsv(fixture, "field")

    process = subprocess.run(
        [
            "make",
            "clamp-ards-parity-fixtures",
            f"CLAMP_ARDS_FIXTURE_ROOT={fixture}",
            f"CLAMP_ARDS_FIXTURE_ARTIFACT_DIR={tmp_path / 'outputs'}",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert process.returncode != 0
    assert '"status": "failed"' in process.stdout


@pytest.fixture(scope="session")
def complete_fixture(
    tmp_path_factory: pytest.TempPathFactory,
    pending_clamp_fixture: Path,
) -> Path:
    root = tmp_path_factory.mktemp("complete-clamp-fixture") / "fixture"
    shutil.copytree(pending_clamp_fixture, root)
    _populate_python_expected_fixture(root)
    validate_fixture(root)
    return root


def _populate_python_expected_fixture(root: Path) -> None:
    for pending in (
        root / "clamp_expected" / "PENDING",
        root / "intermediate_expected" / "PENDING",
    ):
        pending.unlink()
    mirror = load_legacy_mirror()
    with (root / "manifest.csv").open("r", encoding="utf-8", newline="") as handle:
        manifest = list(csv.DictReader(handle))
    for row in manifest:
        case_id = row["case_id"]
        text = (root / row["input_path"]).read_bytes().decode("utf-8")
        trace = mirror.trace(text)
        offsets = Utf16OffsetMap.from_text(text)
        occurrences: dict[tuple[object, ...], int] = {}
        with (root / "clamp_expected" / f"{case_id}.tsv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=EXPECTED_ENTITY_FIELDS,
                delimiter="\t",
                lineterminator="\n",
            )
            writer.writeheader()
            for raw_order, entity in enumerate(trace.final_entities):
                start, end = offsets.span(entity.start, entity.end)
                entity_text = entity.covered_text(text)
                base = (
                    start,
                    end,
                    entity_text,
                    entity.semantic_tag,
                    entity.assertion,
                    entity.cui,
                    entity.attribute,
                )
                occurrence = occurrences.get(base, 0)
                occurrences[base] = occurrence + 1
                writer.writerow(
                    {
                        "clamp_doc_id": case_id,
                        "start": start,
                        "end": end,
                        "semantic_tag": entity.semantic_tag,
                        "assertion": entity.assertion,
                        "cui": _null(entity.cui),
                        "attribute": _null(entity.attribute),
                        "entity_text": entity_text,
                        "raw_order": raw_order,
                        "duplicate_occurrence": occurrence,
                    }
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
            "sentences": [
                {
                    "start": offsets.span(item.start, item.end)[0],
                    "end": offsets.span(item.start, item.end)[1],
                    "sentence_number": item.sentence_number,
                    "covered_text": item.covered_text(text),
                }
                for item in trace.sentences
            ],
            "tokens": [
                {
                    "start": offsets.span(item.start, item.end)[0],
                    "end": offsets.span(item.start, item.end)[1],
                    "token_number": item.token_number,
                    "covered_text": item.covered_text(text),
                }
                for item in trace.tokens
            ],
            "final_entities": [
                {
                    "start": offsets.span(item.start, item.end)[0],
                    "end": offsets.span(item.start, item.end)[1],
                    "semantic_tag": item.semantic_tag,
                    "assertion": item.assertion,
                    "cui": item.cui,
                    "attribute": item.attribute,
                    "covered_text": item.covered_text(text),
                    "raw_order": raw_order,
                }
                for raw_order, item in enumerate(trace.final_entities)
            ],
        }
        (root / "intermediate_expected" / f"{case_id}.json").write_text(
            json.dumps(intermediate, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
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
            "started_at": f"2026-01-0{run_number}T00:00:00Z",
            "completed_at": f"2026-01-0{run_number}T00:01:00Z",
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
            "cases": len(manifest),
            "sentences": sum(
                len(
                    json.loads(
                        (root / "intermediate_expected" / f"{row['case_id']}.json").read_text(
                            encoding="utf-8"
                        )
                    )["sentences"]
                )
                for row in manifest
            ),
            "tokens": sum(
                len(
                    json.loads(
                        (root / "intermediate_expected" / f"{row['case_id']}.json").read_text(
                            encoding="utf-8"
                        )
                    )["tokens"]
                )
                for row in manifest
            ),
            "final_entities": sum(
                len(
                    json.loads(
                        (root / "intermediate_expected" / f"{row['case_id']}.json").read_text(
                            encoding="utf-8"
                        )
                    )["final_entities"]
                )
                for row in manifest
            ),
        },
        **{
            f"run_{run_number}": {
                "run_label": f"run_{run_number}",
                "started_at_utc": f"2026-01-0{run_number}T00:00:00Z",
                "finished_at_utc": f"2026-01-0{run_number}T00:01:00Z",
                "recorded_at_utc": f"2026-01-0{run_number}T00:02:00Z",
                "output_file_count": len(manifest) * 2,
                "output_manifest_sha256": str(run_number) * 64,
            }
            for run_number in (1, 2)
        },
    }
    provenance["reviews"] = {
        "phi": {
            "automated_screen": "passed",
            "manual_review": "approved",
            "reviewer": "fixture-test",
            "reviewed_at": "2026-01-03T00:00:00Z",
        },
        "redistribution": {
            "status": "approved",
            "authority": "fixture-test",
            "evidence": "synthetic fixture test",
        },
    }
    provenance_path.write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_sha256s(root)


def _copy_complete_fixture(source: Path, tmp_path: Path) -> Path:
    destination = tmp_path / "fixture"
    shutil.copytree(source, destination)
    return destination


def _mutate_intermediate(root: Path, stage: str) -> None:
    for path in sorted((root / "intermediate_expected").glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        key = {"sentence": "sentences", "token": "tokens", "final": "final_entities"}[stage]
        if payload[key]:
            if stage in {"sentence", "token"}:
                payload[key].pop()
            else:
                entity = payload[key][0]
                replacement = "absent" if entity["assertion"] != "absent" else "present"
                _replace_expected_tsv_entity_assertion(
                    root,
                    case_id=payload["case_id"],
                    entity=entity,
                    replacement=replacement,
                )
                entity["assertion"] = replacement
            path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            _sync_fixture_counts(root)
            write_sha256s(root)
            return
    raise AssertionError(f"No fixture record available for {stage}")


def _reverse_intermediate_final_order(root: Path) -> None:
    for path in sorted((root / "intermediate_expected").glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if len(payload["final_entities"]) < 2:
            continue
        payload["final_entities"].reverse()
        for raw_order, row in enumerate(payload["final_entities"]):
            row["raw_order"] = raw_order
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        write_sha256s(root)
        return
    raise AssertionError("No intermediate final-entity sequence available for order mutation")


def _set_order_contract(root: Path, *, txt_stable: bool, xmi_stable: bool) -> None:
    path = root / "provenance.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    determinism = payload["determinism"]
    determinism["raw_order_required"] = txt_stable
    determinism["txt_row_order_stable"] = txt_stable
    determinism["txt_order_difference_documents"] = 0 if txt_stable else 1
    determinism["xmi_entity_order_stable"] = xmi_stable
    determinism["xmi_order_difference_documents"] = 0 if xmi_stable else 1
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_sha256s(root)


def _mutate_expected_tsv(root: Path, mutation: str) -> None:
    mutators: dict[str, Callable[[list[dict[str, str]]], list[dict[str, str]]]] = {
        "field": _field_mutation,
        "multiplicity": _multiplicity_mutation,
        "document": lambda rows: [],
        "order": _order_mutation,
    }
    for path in sorted((root / "clamp_expected").glob("*.tsv")):
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle, delimiter="\t"))
        minimum = 2 if mutation == "order" else 1
        if len(rows) < minimum:
            continue
        changed = mutators[mutation](rows)
        _renumber_expected_rows(changed)
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=EXPECTED_ENTITY_FIELDS,
                delimiter="\t",
                lineterminator="\n",
            )
            writer.writeheader()
            writer.writerows(changed)
        if mutation != "order":
            _replace_intermediate_final_entities_from_tsv(
                root,
                case_id=path.stem,
                rows=changed,
            )
        _sync_fixture_counts(root)
        write_sha256s(root)
        return
    raise AssertionError(f"No expected TSV available for {mutation}")


def _field_mutation(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    rows[0]["assertion"] = "absent"
    return rows


def _multiplicity_mutation(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [*rows, dict(rows[0])]


def _order_mutation(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return list(reversed(rows))


def _renumber_expected_rows(rows: list[dict[str, str]]) -> None:
    occurrences: dict[tuple[str, ...], int] = {}
    for position, row in enumerate(rows):
        base = tuple(
            row[field]
            for field in (
                "start",
                "end",
                "entity_text",
                "semantic_tag",
                "assertion",
                "cui",
                "attribute",
            )
        )
        occurrence = occurrences.get(base, 0)
        occurrences[base] = occurrence + 1
        row["raw_order"] = str(position)
        row["duplicate_occurrence"] = str(occurrence)


def _replace_intermediate_final_entities_from_tsv(
    root: Path,
    *,
    case_id: str,
    rows: list[dict[str, str]],
) -> None:
    path = root / "intermediate_expected" / f"{case_id}.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["final_entities"] = [
        {
            "start": int(row["start"]),
            "end": int(row["end"]),
            "semantic_tag": row["semantic_tag"],
            "assertion": row["assertion"],
            "cui": _json_null(row["cui"]),
            "attribute": _json_null(row["attribute"]),
            "covered_text": row["entity_text"],
            "raw_order": raw_order,
        }
        for raw_order, row in enumerate(rows)
    ]
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _replace_expected_tsv_entity_assertion(
    root: Path,
    *,
    case_id: str,
    entity: dict[str, object],
    replacement: str,
) -> None:
    path = root / "clamp_expected" / f"{case_id}.tsv"
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    target = (
        str(entity["start"]),
        str(entity["end"]),
        str(entity["semantic_tag"]),
        str(entity["assertion"]),
        _null(entity["cui"] if isinstance(entity["cui"], str) else None),
        _null(entity["attribute"] if isinstance(entity["attribute"], str) else None),
        str(entity["covered_text"]),
    )
    for row in rows:
        candidate = tuple(
            row[field]
            for field in (
                "start",
                "end",
                "semantic_tag",
                "assertion",
                "cui",
                "attribute",
                "entity_text",
            )
        )
        if candidate == target:
            row["assertion"] = replacement
            _renumber_expected_rows(rows)
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=EXPECTED_ENTITY_FIELDS,
                    delimiter="\t",
                    lineterminator="\n",
                )
                writer.writeheader()
                writer.writerows(rows)
            return
    raise AssertionError(f"No matching expected TSV entity for {case_id}")


def _sync_fixture_counts(root: Path) -> None:
    counts = {
        "cases": 0,
        "sentences": 0,
        "tokens": 0,
        "final_entities": 0,
    }
    for path in (root / "intermediate_expected").glob("*.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        counts["cases"] += 1
        counts["sentences"] += len(payload["sentences"])
        counts["tokens"] += len(payload["tokens"])
        counts["final_entities"] += len(payload["final_entities"])
    provenance_path = root / "provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance["legacy_import"]["fixture_counts"] = counts
    provenance_path.write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _json_null(value: str) -> str | None:
    return None if value == r"\N" else value


def _null(value: str | None) -> str:
    return r"\N" if value is None else value
