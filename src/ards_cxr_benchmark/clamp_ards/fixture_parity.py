from __future__ import annotations

import csv
import hashlib
import json
import os
from dataclasses import dataclass, replace
from itertools import zip_longest
from pathlib import Path
from typing import Any
from uuid import uuid4

import pyarrow as pa
import pyarrow.parquet as pq

from .batch import ENTITY_SCHEMA, PREDICTION_SCHEMA, run_clamp_ards_batch
from .fixtures import EXPECTED_ENTITY_FIELDS, FixtureValidation, validate_fixture
from .parity import ParityResult, compare_clamp_ards_outputs, write_parity_result
from .pipeline import load_legacy_mirror
from .tokenization import Utf16OffsetMap

INTERMEDIATE_MISMATCH_SCHEMA = pa.schema(
    [
        pa.field("mismatch_type", pa.string(), nullable=False),
        pa.field("clamp_doc_id", pa.string(), nullable=False),
        pa.field("stage", pa.string(), nullable=False),
        pa.field("position", pa.int64(), nullable=False),
        pa.field("expected_count", pa.int64(), nullable=False),
        pa.field("actual_count", pa.int64(), nullable=False),
        pa.field("expected_hash", pa.string(), nullable=False),
        pa.field("actual_hash", pa.string(), nullable=False),
    ]
)


@dataclass(frozen=True)
class FixtureParityResult:
    validation: FixtureValidation
    summary: dict[str, Any]
    parity: ParityResult | None
    intermediate_mismatches: tuple[dict[str, Any], ...]

    @property
    def pending(self) -> bool:
        return self.validation.pending

    @property
    def passed(self) -> bool:
        return self.summary["status"] == "passed"


def compare_clamp_ards_fixture(
    *,
    fixture_root: Path,
    output_dir: Path,
    allow_pending: bool = False,
    show_progress: bool = False,
    project_dir: Path | None = None,
) -> FixtureParityResult:
    """Validate and, when complete, strictly compare the CLAMP-generated fixture."""

    fixture_root = fixture_root.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    if output_dir.is_relative_to(fixture_root):
        raise ValueError("Fixture parity outputs must remain outside the tracked fixture root")
    validation = validate_fixture(fixture_root, allow_pending=allow_pending)
    if validation.pending:
        return FixtureParityResult(
            validation=validation,
            summary={
                "status": "pending",
                "passed": False,
                "fixture_version": validation.fixture_version,
                "fixture_lifecycle": validation.lifecycle,
                "fixture_case_count": validation.case_count,
            },
            parity=None,
            intermediate_mismatches=(),
        )

    manifest_rows = _read_manifest(fixture_root)
    order_contract = _read_order_contract(fixture_root)
    output_order_required = order_contract["raw_order_required"]
    xmi_order_required = order_contract["xmi_entity_order_stable"]
    output_dir.mkdir(parents=True, exist_ok=True)
    expected_entities = output_dir / "expected_entities.parquet"
    expected_predictions = output_dir / "expected_predictions.parquet"
    actual_entities = output_dir / "actual_entities.parquet"
    actual_predictions = output_dir / "actual_predictions.parquet"
    _write_expected_parquets(
        fixture_root,
        manifest_rows,
        entity_output=expected_entities,
        prediction_output=expected_predictions,
    )
    run_clamp_ards_batch(
        input_path=None,
        fixture_root=fixture_root,
        entity_output=actual_entities,
        prediction_output=actual_predictions,
        summary_output=output_dir / "python_batch_summary.json",
        id_column="case_id",
        text_column="report_text",
        id_prefix="",
        project_dir=project_dir,
        show_progress=show_progress,
    )
    parity = compare_clamp_ards_outputs(
        expected_entities=expected_entities,
        expected_predictions=expected_predictions,
        actual_entities=actual_entities,
        actual_predictions=actual_predictions,
        require_order=output_order_required,
    )
    intermediate_summary, intermediate_mismatches = _compare_intermediate_annotations(
        fixture_root,
        manifest_rows,
        project_dir=project_dir,
        xmi_order_required=xmi_order_required,
    )
    passed = parity.passed and not intermediate_mismatches
    summary = {
        **parity.summary,
        **intermediate_summary,
        "passed": passed,
        "status": "passed" if passed else "failed",
        "fixture_version": validation.fixture_version,
        "fixture_lifecycle": validation.lifecycle,
        "fixture_case_count": validation.case_count,
        "fixture_output_order_required": output_order_required,
        "fixture_xmi_order_required": xmi_order_required,
    }
    combined = replace(parity, summary=summary)
    write_parity_result(
        combined,
        summary_output=output_dir / "parity_summary.json",
        markdown_output=output_dir / "parity_summary.md",
        entity_mismatch_output=output_dir / "entity_mismatches.parquet",
        document_mismatch_output=output_dir / "document_mismatches.parquet",
        order_mismatch_output=output_dir / "order_mismatches.parquet",
    )
    _atomic_write_parquet(
        output_dir / "intermediate_mismatches.parquet",
        intermediate_mismatches,
        INTERMEDIATE_MISMATCH_SCHEMA,
    )
    _atomic_write_text(
        output_dir / "parity_summary.md",
        render_fixture_parity_summary(summary),
    )
    return FixtureParityResult(
        validation=validation,
        summary=summary,
        parity=combined,
        intermediate_mismatches=tuple(intermediate_mismatches),
    )


def render_fixture_parity_summary(summary: dict[str, Any]) -> str:
    """Render aggregate-only fixture parity evidence."""

    rows = [
        ("Fixture cases", summary["fixture_case_count"]),
        ("Exact entity-multiset documents", summary["exact_entity_document_count"]),
        ("Expected entities", summary["expected_entity_count"]),
        ("Actual entities", summary["actual_entity_count"]),
        ("Entity field mismatches", summary["field_mismatches"]),
        ("Entity multiplicity mismatches", summary["multiplicity_mismatches"]),
        ("Document count mismatches", summary["document_count_mismatches"]),
        ("Document label mismatches", summary["document_label_mismatches"]),
        ("Entity order differences", summary["output_order_differences"]),
        ("Exact sentence documents", summary["exact_sentence_documents"]),
        ("Expected sentences", summary["expected_sentence_count"]),
        ("Actual sentences", summary["actual_sentence_count"]),
        ("Exact token documents", summary["exact_token_documents"]),
        ("Expected tokens", summary["expected_token_count"]),
        ("Actual tokens", summary["actual_token_count"]),
        ("Exact intermediate final-entity documents", summary["exact_final_entity_documents"]),
        ("Intermediate mismatch positions", summary["intermediate_mismatch_positions"]),
    ]
    status = str(summary["status"]).upper()
    lines = [
        "# ARDS CLAMP Synthetic Fixture Parity",
        "",
        f"Strict status: **{status}**",
        "",
        (
            "Entity output order is required because the two legacy TXT runs were stable."
            if summary["fixture_output_order_required"]
            else "Entity output order is reported but not required because the two legacy TXT "
            "runs were not stable."
        ),
        (
            "XMI final-entity order is required because the two legacy XMI runs were stable."
            if summary["fixture_xmi_order_required"]
            else "XMI final-entity order is canonicalized because the two legacy XMI runs were "
            "not stable."
        ),
        "",
        "| Comparison | Count |",
        "|---|---:|",
        *[f"| {label} | {int(value):,} |" for label, value in rows],
        "",
    ]
    return "\n".join(lines)


def _read_manifest(fixture_root: Path) -> list[dict[str, str]]:
    with (fixture_root / "manifest.csv").open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"case_id", "input_path", "input_sha256"}
        missing = sorted(required - set(reader.fieldnames or ()))
        if missing:
            raise ValueError(f"Fixture manifest is missing fields: {missing}")
        return list(reader)


def _read_order_contract(fixture_root: Path) -> dict[str, bool]:
    payload = json.loads((fixture_root / "provenance.json").read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("determinism"), dict):
        raise ValueError("Complete fixture lacks determinism provenance")
    determinism = payload["determinism"]
    result = {
        field: determinism.get(field) for field in ("raw_order_required", "xmi_entity_order_stable")
    }
    if any(value not in {True, False} for value in result.values()):
        raise ValueError("Complete fixture has an unresolved order contract")
    return {field: bool(value) for field, value in result.items()}


def _write_expected_parquets(
    fixture_root: Path,
    manifest_rows: list[dict[str, str]],
    *,
    entity_output: Path,
    prediction_output: Path,
) -> None:
    entity_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    for manifest_row in manifest_rows:
        case_id = manifest_row["case_id"]
        rows = _read_expected_tsv(fixture_root / "clamp_expected" / f"{case_id}.tsv", case_id)
        entity_rows.extend(rows)
        ards_count = sum(str(row["semantic_tag"]).casefold() == "ards" for row in rows)
        prediction_rows.append(
            {
                "clamp_doc_id": case_id,
                "prediction_status": "evaluable",
                "prediction_label": int(ards_count > 0),
                "clamp_ards_entity_count": ards_count,
                "source_text_sha256": manifest_row["input_sha256"],
            }
        )
    _atomic_write_parquet(entity_output, entity_rows, ENTITY_SCHEMA)
    _atomic_write_parquet(prediction_output, prediction_rows, PREDICTION_SCHEMA)


def _read_expected_tsv(path: Path, case_id: str) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if tuple(reader.fieldnames or ()) != EXPECTED_ENTITY_FIELDS:
            raise ValueError(f"Unexpected expected-entity schema: {path}")
        records = list(reader)
    result: list[dict[str, Any]] = []
    occurrences: dict[tuple[Any, ...], int] = {}
    for position, row in enumerate(records):
        if row["clamp_doc_id"] != case_id:
            raise ValueError(f"Expected entity contains the wrong document ID: {path}")
        start = _nonnegative_int(row["start"], field="start", path=path)
        end = _nonnegative_int(row["end"], field="end", path=path)
        if end < start:
            raise ValueError(f"Expected entity has end before start: {path}")
        raw_order = _nonnegative_int(row["raw_order"], field="raw_order", path=path)
        if raw_order != position:
            raise ValueError(f"Expected entity raw_order differs from file order: {path}")
        semantic_tag = row["semantic_tag"]
        assertion = row["assertion"]
        entity_text = row["entity_text"]
        if not semantic_tag or not assertion or not entity_text:
            raise ValueError(f"Expected entity has a blank required field: {path}")
        cui = _nullable(row["cui"])
        attribute = _nullable(row["attribute"])
        base = (start, end, entity_text, semantic_tag, assertion, cui, attribute)
        occurrence = occurrences.get(base, 0)
        occurrences[base] = occurrence + 1
        declared_occurrence = _nonnegative_int(
            row["duplicate_occurrence"],
            field="duplicate_occurrence",
            path=path,
        )
        if declared_occurrence != occurrence:
            raise ValueError(f"Expected entity duplicate_occurrence is inconsistent: {path}")
        result.append(
            {
                "clamp_doc_id": case_id,
                "start": start,
                "end": end,
                "entity_text": entity_text,
                "entity_text_sha256": hashlib.sha256(entity_text.encode("utf-8")).hexdigest(),
                "semantic_tag": semantic_tag,
                "assertion": assertion,
                "cui": cui,
                "attribute": attribute,
                "duplicate_occurrence": occurrence,
            }
        )
    return result


def _compare_intermediate_annotations(
    fixture_root: Path,
    manifest_rows: list[dict[str, str]],
    *,
    project_dir: Path | None,
    xmi_order_required: bool,
) -> tuple[dict[str, int], list[dict[str, Any]]]:
    mirror = (
        load_legacy_mirror(str(project_dir.expanduser().resolve()))
        if project_dir is not None
        else load_legacy_mirror()
    )
    mismatches: list[dict[str, Any]] = []
    counts = {
        "exact_sentence_documents": 0,
        "expected_sentence_count": 0,
        "actual_sentence_count": 0,
        "exact_token_documents": 0,
        "expected_token_count": 0,
        "actual_token_count": 0,
        "exact_final_entity_documents": 0,
        "expected_final_entity_count": 0,
        "actual_final_entity_count": 0,
    }
    for manifest_row in manifest_rows:
        case_id = manifest_row["case_id"]
        text = (
            (fixture_root / manifest_row["input_path"])
            .read_bytes()
            .decode("utf-8", errors="strict")
        )
        expected = json.loads(
            (fixture_root / "intermediate_expected" / f"{case_id}.json").read_text(encoding="utf-8")
        )
        if not isinstance(expected, dict):
            raise ValueError(f"Intermediate expected output must be a JSON object: {case_id}")
        trace = mirror.trace(text)
        offsets = Utf16OffsetMap.from_text(text)
        actual = {
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
                    "raw_order": row_order,
                }
                for row_order, item in enumerate(trace.final_entities)
            ],
        }
        for stage, summary_stem in (
            ("sentences", "sentence"),
            ("tokens", "token"),
            ("final_entities", "final_entity"),
        ):
            expected_rows = _normalized_intermediate_rows(expected, stage, case_id)
            actual_rows = actual[stage]
            if stage == "final_entities" and not xmi_order_required:
                expected_rows = _canonical_entity_multiset(expected_rows)
                actual_rows = _canonical_entity_multiset(actual_rows)
            counts[f"expected_{summary_stem}_count"] += len(expected_rows)
            counts[f"actual_{summary_stem}_count"] += len(actual_rows)
            if expected_rows == actual_rows:
                counts[f"exact_{summary_stem}_documents"] += 1
            else:
                mismatches.extend(_sequence_mismatches(case_id, stage, expected_rows, actual_rows))
    counts["intermediate_mismatch_positions"] = len(mismatches)
    return counts, mismatches


def _canonical_entity_multiset(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    without_order = [
        {field: value for field, value in row.items() if field != "raw_order"} for row in rows
    ]
    return sorted(
        without_order,
        key=lambda row: json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
    )


def _normalized_intermediate_rows(
    payload: dict[str, Any],
    stage: str,
    case_id: str,
) -> list[dict[str, Any]]:
    rows = payload.get(stage)
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise ValueError(f"Intermediate {stage} must be a list of objects: {case_id}")
    required = {
        "sentences": {"start", "end", "sentence_number", "covered_text"},
        "tokens": {"start", "end", "token_number", "covered_text"},
        "final_entities": {
            "start",
            "end",
            "semantic_tag",
            "assertion",
            "cui",
            "attribute",
            "covered_text",
            "raw_order",
        },
    }[stage]
    result: list[dict[str, Any]] = []
    for row in rows:
        if set(row) != required:
            raise ValueError(f"Intermediate {stage} has unexpected fields: {case_id}")
        result.append({field: row[field] for field in sorted(required)})
    return result


def _sequence_mismatches(
    case_id: str,
    stage: str,
    expected: list[dict[str, Any]],
    actual: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for position, (expected_row, actual_row) in enumerate(
        zip_longest(expected, actual, fillvalue=None)
    ):
        if expected_row == actual_row:
            continue
        rows.append(
            {
                "mismatch_type": "intermediate_sequence_mismatch",
                "clamp_doc_id": case_id,
                "stage": stage,
                "position": position,
                "expected_count": len(expected),
                "actual_count": len(actual),
                "expected_hash": _record_hash(expected_row),
                "actual_hash": _record_hash(actual_row),
            }
        )
    return rows


def _record_hash(record: dict[str, Any] | None) -> str:
    if record is None:
        return ""
    payload = json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _nonnegative_int(value: str, *, field: str, path: Path) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid expected entity {field}: {path}") from exc
    if result < 0:
        raise ValueError(f"Negative expected entity {field}: {path}")
    return result


def _nullable(value: str) -> str | None:
    clean = value.strip()
    return None if not clean or clean.casefold() in {"null", r"\n"} else clean


def _atomic_write_parquet(path: Path, rows: list[dict[str, Any]], schema: pa.Schema) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.partial")
    try:
        pq.write_table(pa.Table.from_pylist(rows, schema=schema), temporary, compression="zstd")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.partial")
    try:
        temporary.write_text(content, encoding="utf-8", newline="\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
