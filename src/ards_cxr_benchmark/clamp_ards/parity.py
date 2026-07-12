from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import zip_longest
from pathlib import Path
from typing import Any
from uuid import uuid4

import pyarrow as pa
import pyarrow.parquet as pq

BASE_ENTITY_FIELDS = (
    "start",
    "end",
    "entity_text",
    "semantic_tag",
    "assertion",
    "cui",
    "attribute",
)
ENTITY_FIELDS = (*BASE_ENTITY_FIELDS, "duplicate_occurrence")
REQUIRED_MISMATCH_FIELDS = (
    "missing_documents",
    "unexpected_documents",
    "missing_entities",
    "unexpected_entities",
    "field_mismatches",
    "multiplicity_mismatches",
    "document_label_mismatches",
    "document_count_mismatches",
    "document_status_mismatches",
)

ENTITY_MISMATCH_SCHEMA = pa.schema(
    [
        pa.field("mismatch_type", pa.string(), nullable=False),
        pa.field("clamp_doc_id", pa.string(), nullable=False),
        pa.field("start", pa.int64()),
        pa.field("end", pa.int64()),
        pa.field("occurrence", pa.int64()),
        pa.field("count", pa.int64()),
        pa.field("entity_text_sha256", pa.string()),
        pa.field("expected_hash", pa.string()),
        pa.field("actual_hash", pa.string()),
    ]
)
DOCUMENT_MISMATCH_SCHEMA = pa.schema(
    [
        pa.field("mismatch_type", pa.string(), nullable=False),
        pa.field("clamp_doc_id", pa.string(), nullable=False),
        pa.field("field", pa.string(), nullable=False),
        pa.field("expected", pa.string(), nullable=False),
        pa.field("actual", pa.string(), nullable=False),
    ]
)
ORDER_MISMATCH_SCHEMA = pa.schema(
    [
        pa.field("mismatch_type", pa.string(), nullable=False),
        pa.field("clamp_doc_id", pa.string(), nullable=False),
        pa.field("position", pa.int64(), nullable=False),
        pa.field("expected_hash", pa.string(), nullable=False),
        pa.field("actual_hash", pa.string(), nullable=False),
    ]
)


@dataclass(frozen=True)
class ParityResult:
    summary: dict[str, Any]
    entity_mismatches: tuple[dict[str, Any], ...]
    document_mismatches: tuple[dict[str, Any], ...]
    order_mismatches: tuple[dict[str, Any], ...]

    @property
    def mismatches(self) -> tuple[dict[str, Any], ...]:
        """Return the legacy combined mismatch ledger."""

        return (*self.entity_mismatches, *self.document_mismatches, *self.order_mismatches)

    @property
    def passed(self) -> bool:
        return bool(self.summary["passed"])


def compare_clamp_ards_outputs(
    *,
    expected_entities: Path,
    expected_predictions: Path,
    actual_entities: Path,
    actual_predictions: Path,
    require_order: bool = False,
) -> ParityResult:
    """Compare exact entity multisets and document labels without emitting source text."""

    expected_entity_rows = _read_entities(expected_entities)
    actual_entity_rows = _read_entities(actual_entities)
    expected_prediction_rows = _read_predictions(expected_predictions)
    actual_prediction_rows = _read_predictions(actual_predictions)
    _validate_parity_input(
        expected_entity_rows,
        expected_prediction_rows,
        side="expected",
    )
    _validate_parity_input(
        actual_entity_rows,
        actual_prediction_rows,
        side="actual",
    )
    expected_docs = set(expected_prediction_rows)
    actual_docs = set(actual_prediction_rows)
    entity_mismatches: list[dict[str, Any]] = []
    document_mismatches: list[dict[str, Any]] = []
    order_mismatches: list[dict[str, Any]] = []
    counts = {name: 0 for name in REQUIRED_MISMATCH_FIELDS}

    for doc_id in sorted(expected_docs - actual_docs):
        counts["missing_documents"] += 1
        document_mismatches.append(
            _document_mismatch(
                "missing_document",
                doc_id,
                field="document_membership",
                expected="present",
                actual="missing",
            )
        )
    for doc_id in sorted(actual_docs - expected_docs):
        counts["unexpected_documents"] += 1
        document_mismatches.append(
            _document_mismatch(
                "unexpected_document",
                doc_id,
                field="document_membership",
                expected="missing",
                actual="present",
            )
        )

    output_order_differences = 0
    for doc_id in sorted(expected_docs & actual_docs):
        expected_prediction = expected_prediction_rows[doc_id]
        actual_prediction = actual_prediction_rows[doc_id]
        if expected_prediction[0] != actual_prediction[0]:
            counts["document_status_mismatches"] += 1
            document_mismatches.append(
                _document_mismatch(
                    "document_status_mismatch",
                    doc_id,
                    field="prediction_status",
                    expected=expected_prediction[0],
                    actual=actual_prediction[0],
                )
            )
        if expected_prediction[1] != actual_prediction[1]:
            counts["document_label_mismatches"] += 1
            document_mismatches.append(
                _document_mismatch(
                    "document_label_mismatch",
                    doc_id,
                    field="prediction_label",
                    expected=expected_prediction[1],
                    actual=actual_prediction[1],
                )
            )
        if expected_prediction[2] != actual_prediction[2]:
            counts["document_count_mismatches"] += 1
            document_mismatches.append(
                _document_mismatch(
                    "document_count_mismatch",
                    doc_id,
                    field="clamp_ards_entity_count",
                    expected=expected_prediction[2],
                    actual=actual_prediction[2],
                )
            )

        expected_sequence = expected_entity_rows.get(doc_id, [])
        actual_sequence = actual_entity_rows.get(doc_id, [])
        expected_counter = Counter(expected_sequence)
        actual_counter = Counter(actual_sequence)
        missing = expected_counter - actual_counter
        unexpected = actual_counter - expected_counter
        counts["missing_entities"] += sum(missing.values())
        counts["unexpected_entities"] += sum(unexpected.values())
        for entity, amount in sorted(missing.items(), key=lambda item: _entity_sort_key(item[0])):
            entity_mismatches.append(
                _entity_mismatch("missing_entity", doc_id, entity, count=amount)
            )
        for entity, amount in sorted(
            unexpected.items(), key=lambda item: _entity_sort_key(item[0])
        ):
            entity_mismatches.append(
                _entity_mismatch("unexpected_entity", doc_id, entity, count=amount)
            )

        expected_base_counter = Counter(entity[:-1] for entity in expected_sequence)
        actual_base_counter = Counter(entity[:-1] for entity in actual_sequence)
        common_keys = set(expected_base_counter) & set(actual_base_counter)
        multiplicity = sum(
            1
            for entity in common_keys
            if expected_base_counter[entity] != actual_base_counter[entity]
        )
        counts["multiplicity_mismatches"] += multiplicity
        if multiplicity:
            entity_mismatches.append(_mismatch("multiplicity_mismatch", doc_id, count=multiplicity))

        expected_by_anchor = _entities_by_anchor(expected_sequence)
        actual_by_anchor = _entities_by_anchor(actual_sequence)
        for anchor in sorted(set(expected_by_anchor) & set(actual_by_anchor)):
            for occurrence, (expected, actual) in enumerate(
                zip(
                    expected_by_anchor[anchor],
                    actual_by_anchor[anchor],
                    strict=False,
                )
            ):
                if expected != actual:
                    counts["field_mismatches"] += 1
                    entity_mismatches.append(
                        _mismatch(
                            "field_mismatch",
                            doc_id,
                            start=anchor[0],
                            end=anchor[1],
                            occurrence=occurrence,
                            expected_hash=_entity_hash(expected),
                            actual_hash=_entity_hash(actual),
                        )
                    )
        if expected_counter == actual_counter and expected_sequence != actual_sequence:
            output_order_differences += 1
            order_mismatches.extend(
                _order_mismatch_rows(doc_id, expected_sequence, actual_sequence)
            )

    expected_entity_count = sum(len(rows) for rows in expected_entity_rows.values())
    actual_entity_count = sum(len(rows) for rows in actual_entity_rows.values())
    exact_document_count = sum(
        Counter(expected_entity_rows.get(doc_id, [])) == Counter(actual_entity_rows.get(doc_id, []))
        for doc_id in expected_docs & actual_docs
    )
    required_mismatch_count = sum(counts.values())
    if require_order:
        required_mismatch_count += output_order_differences
    summary: dict[str, Any] = {
        **counts,
        "passed": required_mismatch_count == 0,
        "expected_entities_sha256": _file_sha256(expected_entities),
        "expected_predictions_sha256": _file_sha256(expected_predictions),
        "actual_entities_sha256": _file_sha256(actual_entities),
        "actual_predictions_sha256": _file_sha256(actual_predictions),
        "expected_document_count": len(expected_docs),
        "actual_document_count": len(actual_docs),
        "expected_entity_count": expected_entity_count,
        "actual_entity_count": actual_entity_count,
        "exact_entity_document_count": exact_document_count,
        "output_order_differences": output_order_differences,
        "output_order_mismatch_positions": len(order_mismatches),
        "require_order": require_order,
        "comparison_fields": list(ENTITY_FIELDS),
    }
    return ParityResult(
        summary=summary,
        entity_mismatches=tuple(entity_mismatches),
        document_mismatches=tuple(document_mismatches),
        order_mismatches=tuple(order_mismatches),
    )


def write_parity_result(
    result: ParityResult,
    *,
    summary_output: Path,
    mismatch_output: Path | None = None,
    entity_mismatch_output: Path | None = None,
    document_mismatch_output: Path | None = None,
    order_mismatch_output: Path | None = None,
    markdown_output: Path | None = None,
) -> None:
    """Atomically write aggregate summaries and text-free mismatch ledgers."""

    output_prefix = _summary_output_prefix(summary_output)
    entity_mismatch_output = entity_mismatch_output or summary_output.with_name(
        f"{output_prefix}_entity_mismatches.parquet"
    )
    document_mismatch_output = document_mismatch_output or summary_output.with_name(
        f"{output_prefix}_document_mismatches.parquet"
    )
    order_mismatch_output = order_mismatch_output or summary_output.with_name(
        f"{output_prefix}_order_mismatches.parquet"
    )
    markdown_output = markdown_output or summary_output.with_suffix(".md")
    destinations = {
        "summary": summary_output,
        "markdown": markdown_output,
        "entities": entity_mismatch_output,
        "documents": document_mismatch_output,
        "order": order_mismatch_output,
    }
    if mismatch_output is not None:
        destinations["legacy"] = mismatch_output
    _require_distinct_output_paths(destinations)
    temporary = {name: _temporary_path(path) for name, path in destinations.items()}
    try:
        for path in destinations.values():
            path.parent.mkdir(parents=True, exist_ok=True)
        temporary["summary"].write_text(
            json.dumps(result.summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary["markdown"].write_text(
            render_parity_summary(result.summary),
            encoding="utf-8",
        )
        _write_mismatch_parquet(
            temporary["entities"], result.entity_mismatches, ENTITY_MISMATCH_SCHEMA
        )
        _write_mismatch_parquet(
            temporary["documents"], result.document_mismatches, DOCUMENT_MISMATCH_SCHEMA
        )
        _write_mismatch_parquet(temporary["order"], result.order_mismatches, ORDER_MISMATCH_SCHEMA)
        if "legacy" in temporary:
            _write_legacy_mismatch_csv(temporary["legacy"], result.mismatches)
        for name, destination in destinations.items():
            os.replace(temporary[name], destination)
    finally:
        for path in temporary.values():
            path.unlink(missing_ok=True)


def render_parity_summary(summary: dict[str, Any]) -> str:
    """Render an identifier-free Markdown summary."""

    status = "PASS" if summary["passed"] else "FAIL"
    rows = [
        ("Expected documents", summary["expected_document_count"]),
        ("Actual documents", summary["actual_document_count"]),
        ("Exact entity-multiset documents", summary["exact_entity_document_count"]),
        ("Expected entities", summary["expected_entity_count"]),
        ("Actual entities", summary["actual_entity_count"]),
        ("Missing documents", summary["missing_documents"]),
        ("Unexpected documents", summary["unexpected_documents"]),
        ("Missing entities", summary["missing_entities"]),
        ("Unexpected entities", summary["unexpected_entities"]),
        ("Field mismatches", summary["field_mismatches"]),
        ("Multiplicity mismatches", summary["multiplicity_mismatches"]),
        ("Document status mismatches", summary["document_status_mismatches"]),
        ("Document count mismatches", summary["document_count_mismatches"]),
        ("Document label mismatches", summary["document_label_mismatches"]),
        ("Output-order differences", summary["output_order_differences"]),
        ("Output-order mismatch positions", summary["output_order_mismatch_positions"]),
    ]
    lines = [
        "# ARDS CLAMP Python Parity Summary",
        "",
        f"Strict status: **{status}**",
        "",
        f"Output order required: **{'yes' if summary['require_order'] else 'no'}**",
        "",
        "| Comparison | Count |",
        "|---|---:|",
        *[f"| {label} | {int(value):,} |" for label, value in rows],
        "",
    ]
    return "\n".join(lines)


def _read_entities(path: Path) -> dict[str, list[tuple[Any, ...]]]:
    table = pq.read_table(path)
    available = set(table.column_names)
    required = {"clamp_doc_id", *BASE_ENTITY_FIELDS}
    missing = sorted(required - available)
    if missing:
        raise ValueError(f"Entity table is missing column(s) {missing}: {path}")
    rows: dict[str, list[tuple[Any, ...]]] = defaultdict(list)
    occurrences: dict[tuple[str, tuple[Any, ...]], int] = defaultdict(int)
    for row in table.to_pylist():
        doc_id = _validated_doc_id(row["clamp_doc_id"], path=path, table="entity")
        entity = tuple(row.get(field) for field in BASE_ENTITY_FIELDS)
        occurrence_key = (doc_id, entity)
        occurrence = occurrences[occurrence_key]
        occurrences[occurrence_key] += 1
        rows[doc_id].append((*entity, occurrence))
    return dict(rows)


def _read_predictions(path: Path) -> dict[str, tuple[str, int | None, int | None]]:
    table = pq.read_table(path)
    available = set(table.column_names)
    required = {"clamp_doc_id", "prediction_label", "clamp_ards_entity_count"}
    missing = sorted(required - available)
    if missing:
        raise ValueError(f"Prediction table is missing column(s) {missing}: {path}")
    status_columns = available & {"prediction_status", "clamp_parse_status"}
    if not status_columns:
        raise ValueError(
            f"Prediction table must contain prediction_status or clamp_parse_status: {path}"
        )
    result: dict[str, tuple[str, int | None, int | None]] = {}
    for row in table.to_pylist():
        doc_id = _validated_doc_id(row["clamp_doc_id"], path=path, table="prediction")
        if doc_id in result:
            raise ValueError(f"Duplicate prediction document ID in {path}: {doc_id}")
        status = _normalized_prediction_status(row, available, path=path, doc_id=doc_id)
        label = _optional_int(row["prediction_label"], "prediction_label", path, doc_id)
        count = _optional_int(
            row["clamp_ards_entity_count"],
            "clamp_ards_entity_count",
            path,
            doc_id,
        )
        if label is not None and label not in {0, 1}:
            raise ValueError(f"Invalid prediction_label for {doc_id} in {path}: {label}")
        if count is not None and count < 0:
            raise ValueError(f"Negative clamp_ards_entity_count for {doc_id} in {path}: {count}")
        if status == "evaluable" and (label is None or count is None):
            raise ValueError(
                f"Evaluable prediction row has null label/count for {doc_id} in {path}"
            )
        result[doc_id] = (status, label, count)
    return result


def _entities_by_anchor(
    entities: list[tuple[Any, ...]],
) -> dict[tuple[int, int], list[tuple[Any, ...]]]:
    grouped: dict[tuple[int, int], list[tuple[Any, ...]]] = defaultdict(list)
    for entity in entities:
        grouped[(int(entity[0]), int(entity[1]))].append(entity)
    return {anchor: sorted(values, key=_entity_sort_key) for anchor, values in grouped.items()}


def _validate_parity_input(
    entities: dict[str, list[tuple[Any, ...]]],
    predictions: dict[str, tuple[str, int | None, int | None]],
    *,
    side: str,
) -> None:
    orphan_ids = sorted(set(entities) - set(predictions))
    if orphan_ids:
        raise ValueError(
            f"{side.capitalize()} entity table contains {len(orphan_ids)} document ID(s) "
            "without prediction rows; "
            f"examples={orphan_ids[:10]}"
        )
    for doc_id, (status, label, count) in predictions.items():
        entity_rows = entities.get(doc_id, [])
        entity_count = sum(str(entity[3]).casefold() == "ards" for entity in entity_rows)
        if status == "evaluable":
            if count != entity_count:
                raise ValueError(
                    f"{side.capitalize()} evaluable prediction/entity count mismatch for "
                    f"{doc_id}: prediction={count}, entities={entity_count}"
                )
            expected_label = int(entity_count > 0)
            if label != expected_label:
                raise ValueError(
                    f"{side.capitalize()} evaluable prediction label/count mismatch for "
                    f"{doc_id}: label={label}, entity_count={entity_count}"
                )
        elif entity_rows or label is not None or count is not None:
            raise ValueError(
                f"{side.capitalize()} non-evaluable prediction must have no entities and null "
                f"label/count for {doc_id}: entities={len(entity_rows)}, label={label}, "
                f"count={count}"
            )


def _validated_doc_id(value: Any, *, path: Path, table: str) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        raise ValueError(f"Null clamp_doc_id in {table} table: {path}")
    doc_id = str(value).strip()
    if not doc_id:
        raise ValueError(f"Blank clamp_doc_id in {table} table: {path}")
    return doc_id


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalized_prediction_status(
    row: dict[str, Any],
    available: set[str],
    *,
    path: Path,
    doc_id: str,
) -> str:
    statuses: list[str] = []
    if "prediction_status" in available:
        value = _required_status(row["prediction_status"], "prediction_status", path, doc_id)
        if value not in {"evaluable", "non_evaluable"}:
            raise ValueError(f"Unknown prediction_status for {doc_id} in {path}: {value}")
        statuses.append(value)
    if "clamp_parse_status" in available:
        value = _required_status(row["clamp_parse_status"], "clamp_parse_status", path, doc_id)
        mapping = {
            "parsed": "evaluable",
            "parsed_empty": "evaluable",
            "missing_output": "non_evaluable",
            "duplicate_output": "non_evaluable",
            "parse_error": "non_evaluable",
        }
        if value not in mapping:
            raise ValueError(f"Unknown clamp_parse_status for {doc_id} in {path}: {value}")
        statuses.append(mapping[value])
    if len(set(statuses)) != 1:
        raise ValueError(
            f"Conflicting prediction status columns for {doc_id} in {path}: {statuses}"
        )
    return statuses[0]


def _required_status(value: Any, column: str, path: Path, doc_id: str) -> str:
    if value is None or not str(value).strip():
        raise ValueError(f"Null/blank {column} for {doc_id} in {path}")
    return str(value).strip().casefold()


def _optional_int(value: Any, column: str, path: Path, doc_id: str) -> int | None:
    if value is None:
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"Invalid {column} for {doc_id} in {path}: {value!r}") from exc
    if isinstance(value, float) and not value.is_integer():
        raise ValueError(f"Invalid {column} for {doc_id} in {path}: {value!r}")
    return parsed


def _entity_mismatch(
    mismatch_type: str,
    doc_id: str,
    entity: tuple[Any, ...],
    *,
    count: int,
) -> dict[str, Any]:
    return _mismatch(
        mismatch_type,
        doc_id,
        start=entity[0],
        end=entity[1],
        count=count,
        entity_text_sha256=hashlib.sha256(str(entity[2]).encode("utf-8")).hexdigest(),
        expected_hash=_entity_hash(entity) if mismatch_type == "missing_entity" else None,
        actual_hash=_entity_hash(entity) if mismatch_type == "unexpected_entity" else None,
    )


def _document_mismatch(
    mismatch_type: str,
    doc_id: str,
    *,
    field: str,
    expected: Any,
    actual: Any,
) -> dict[str, Any]:
    return _mismatch(
        mismatch_type,
        doc_id,
        field=field,
        expected=_mismatch_value(expected),
        actual=_mismatch_value(actual),
    )


def _order_mismatch_rows(
    doc_id: str,
    expected: list[tuple[Any, ...]],
    actual: list[tuple[Any, ...]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for position, (expected_entity, actual_entity) in enumerate(
        zip_longest(expected, actual, fillvalue=None)
    ):
        if expected_entity == actual_entity:
            continue
        rows.append(
            _mismatch(
                "entity_order_mismatch",
                doc_id,
                position=position,
                expected_hash=("" if expected_entity is None else _entity_hash(expected_entity)),
                actual_hash="" if actual_entity is None else _entity_hash(actual_entity),
            )
        )
    return rows


def _mismatch(mismatch_type: str, doc_id: str, **values: Any) -> dict[str, Any]:
    return {"mismatch_type": mismatch_type, "clamp_doc_id": doc_id, **values}


def _entity_hash(entity: tuple[Any, ...]) -> str:
    payload = json.dumps(entity, ensure_ascii=False, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _entity_sort_key(entity: tuple[Any, ...]) -> tuple[str, ...]:
    return tuple("" if value is None else str(value) for value in entity)


def _mismatch_value(value: Any) -> str:
    return "null" if value is None else str(value)


def _summary_output_prefix(path: Path) -> str:
    return path.stem.removesuffix("_summary")


def _write_mismatch_parquet(
    path: Path,
    rows: tuple[dict[str, Any], ...],
    schema: pa.Schema,
) -> None:
    pq.write_table(pa.Table.from_pylist(list(rows), schema=schema), path, compression="zstd")


def _write_legacy_mismatch_csv(
    path: Path,
    rows: tuple[dict[str, Any], ...],
) -> None:
    fieldnames = [
        "mismatch_type",
        "clamp_doc_id",
        "field",
        "start",
        "end",
        "occurrence",
        "position",
        "count",
        "entity_text_sha256",
        "expected",
        "actual",
        "expected_hash",
        "actual_hash",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _temporary_path(destination: Path) -> Path:
    return destination.with_name(f".{destination.name}.{uuid4().hex}.partial")


def _require_distinct_output_paths(destinations: dict[str, Path]) -> None:
    names_by_path: dict[Path, list[str]] = defaultdict(list)
    for name, path in destinations.items():
        names_by_path[path.resolve()].append(name)
    collisions = [(path, names) for path, names in names_by_path.items() if len(names) > 1]
    if collisions:
        details = "; ".join(f"{','.join(names)} -> {path}" for path, names in collisions)
        raise ValueError(f"Parity output paths must be distinct: {details}")
