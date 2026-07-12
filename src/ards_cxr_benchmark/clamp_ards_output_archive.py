from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import zipfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any
from uuid import uuid4

import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

from .clamp_ards_outputs import (
    AUDIT_COLUMNS,
    FIELD_ALIASES,
    IDENTIFIER_COLUMNS,
    MODEL_NAME,
    load_input_manifest,
    render_clamp_teacher_summary,
    validate_distinct_clamp_paths,
)
from .config import ensure_parent_dir

REQUIRED_ARCHIVE_FIELDS = {
    "start",
    "end",
    "semantic_tag",
    "cui",
    "assertion",
    "entity_text",
}

ENTITY_SCHEMA = pa.schema(
    [
        pa.field("clamp_doc_id", pa.string(), nullable=False),
        *[pa.field(column, pa.string()) for column in IDENTIFIER_COLUMNS],
        pa.field("source_output_file", pa.string(), nullable=False),
        pa.field("output_file_exists", pa.bool_(), nullable=False),
        pa.field("parse_status", pa.string(), nullable=False),
        pa.field("parse_error", pa.string(), nullable=False),
        pa.field("start", pa.int64()),
        pa.field("end", pa.int64()),
        pa.field("semantic_tag", pa.string()),
        pa.field("assertion", pa.string()),
        pa.field("cui", pa.string()),
        pa.field("attribute", pa.string()),
        pa.field("entity_text", pa.string()),
        pa.field("entity_text_sha256", pa.string()),
    ]
)
PREDICTION_SCHEMA = pa.schema(
    [
        pa.field("clamp_doc_id", pa.string(), nullable=False),
        *[pa.field(column, pa.string()) for column in IDENTIFIER_COLUMNS],
        pa.field("model_name", pa.string(), nullable=False),
        pa.field("prediction_score", pa.float64()),
        pa.field("prediction_label", pa.int64()),
        pa.field("clamp_ards_entity_count", pa.int64(), nullable=False),
        pa.field("clamp_output_available", pa.bool_(), nullable=False),
        pa.field("clamp_parse_status", pa.string(), nullable=False),
        pa.field("clamp_parse_error", pa.string(), nullable=False),
    ]
)
PROBABILISTIC_SCHEMA = pa.schema(
    [
        pa.field("case_id", pa.string(), nullable=False),
        pa.field("model_name", pa.string(), nullable=False),
        pa.field("prediction_score", pa.float64(), nullable=False),
        pa.field("prediction_label", pa.int64(), nullable=False),
        *[pa.field(column, pa.string()) for column in IDENTIFIER_COLUMNS],
    ]
)


@dataclass(frozen=True)
class ParsedClampTxt:
    entities: list[dict[str, Any]]
    parse_status: str
    parse_error: str


def parse_clamp_ards_output_archive(
    *,
    input_manifest_path: Path,
    output_archive: Path,
    entity_output: Path,
    prediction_output: Path,
    probabilistic_prediction_output: Path,
    audit_output: Path,
    summary_output: Path,
    batch_size: int = 5000,
    show_progress: bool = False,
) -> dict[str, Any]:
    """Strictly parse a complete CLAMP TXT archive into atomic teacher artifacts."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    input_manifest_path = input_manifest_path.resolve()
    output_archive = output_archive.resolve()
    destinations = {
        "entities": entity_output.resolve(),
        "predictions": prediction_output.resolve(),
        "probabilistic": probabilistic_prediction_output.resolve(),
        "audit": audit_output.resolve(),
        "summary": summary_output.resolve(),
        "summary_markdown": summary_output.resolve().with_suffix(".md"),
    }
    validate_distinct_clamp_paths(
        input_manifest_path=input_manifest_path,
        output_archive=output_archive,
        **destinations,
    )
    if not output_archive.is_file() or not zipfile.is_zipfile(output_archive):
        raise ValueError(f"CLAMP output archive is not a readable ZIP: {output_archive}")

    manifest = load_input_manifest(input_manifest_path)
    manifest_records = manifest.to_dict(orient="records")
    manifest_ids = set(manifest["clamp_doc_id"])
    temporary = {name: _temporary_path(path) for name, path in destinations.items()}

    try:
        with zipfile.ZipFile(output_archive) as archive:
            members, ignored_members = _discover_txt_members(archive)
            member_ids = set(members)
            missing = sorted(manifest_ids - member_ids)
            unexpected = sorted(member_ids - manifest_ids)
            if missing or unexpected:
                raise ValueError(
                    "CLAMP TXT archive does not exactly match input_manifest.csv: "
                    f"missing={len(missing)}, unexpected={len(unexpected)}"
                )

            counters = Counter()
            semantic_tags = Counter()
            assertions = Counter()
            parse_error_examples: list[str] = []
            entity_batch: list[dict[str, Any]] = []
            prediction_batch: list[dict[str, Any]] = []
            probabilistic_batch: list[dict[str, Any]] = []

            with (
                pq.ParquetWriter(
                    temporary["entities"], ENTITY_SCHEMA, compression="zstd"
                ) as entity_writer,
                pq.ParquetWriter(
                    temporary["predictions"], PREDICTION_SCHEMA, compression="zstd"
                ) as prediction_writer,
                pq.ParquetWriter(
                    temporary["probabilistic"], PROBABILISTIC_SCHEMA, compression="zstd"
                ) as probabilistic_writer,
                temporary["audit"].open("w", encoding="utf-8", newline="") as audit_handle,
            ):
                audit_writer = csv.DictWriter(audit_handle, fieldnames=AUDIT_COLUMNS)
                audit_writer.writeheader()

                for manifest_row in tqdm(
                    manifest_records,
                    desc="Parsing CLAMP TXT outputs",
                    unit="file",
                    disable=not show_progress,
                ):
                    doc_id = str(manifest_row["clamp_doc_id"])
                    member_name = members[doc_id]
                    parsed = parse_clamp_txt_payload(archive.read(member_name))
                    if parsed.parse_status == "parse_error":
                        counters["parse_errors"] += 1
                        if len(parse_error_examples) < 20:
                            parse_error_examples.append(f"{doc_id}:{parsed.parse_error}")
                    else:
                        counters["parse_success"] += 1
                        counters[parsed.parse_status] += 1

                    metadata = {
                        column: _clean_manifest_value(manifest_row.get(column, ""))
                        for column in IDENTIFIER_COLUMNS
                    }
                    ards_count = sum(
                        str(entity["semantic_tag"]).upper() == "ARDS" for entity in parsed.entities
                    )
                    prediction_label = None
                    if parsed.parse_status in {"parsed", "parsed_empty"}:
                        prediction_label = int(ards_count > 0)
                    prediction_score = None if prediction_label is None else float(prediction_label)

                    for entity in parsed.entities:
                        semantic_tags[str(entity["semantic_tag"])] += 1
                        assertions[str(entity["assertion"])] += 1
                        entity_batch.append(
                            {
                                "clamp_doc_id": doc_id,
                                **metadata,
                                "source_output_file": member_name,
                                "output_file_exists": True,
                                "parse_status": parsed.parse_status,
                                "parse_error": parsed.parse_error,
                                **entity,
                            }
                        )
                    counters["entity_rows"] += len(parsed.entities)
                    counters["positive_documents"] += int(prediction_label == 1)
                    counters["negative_documents"] += int(prediction_label == 0)

                    prediction = {
                        "clamp_doc_id": doc_id,
                        **metadata,
                        "model_name": MODEL_NAME,
                        "prediction_score": prediction_score,
                        "prediction_label": prediction_label,
                        "clamp_ards_entity_count": int(ards_count),
                        "clamp_output_available": True,
                        "clamp_parse_status": parsed.parse_status,
                        "clamp_parse_error": parsed.parse_error,
                    }
                    prediction_batch.append(prediction)
                    if prediction_label is not None:
                        probabilistic_batch.append(
                            {
                                "case_id": doc_id,
                                "model_name": MODEL_NAME,
                                "prediction_score": float(prediction_score),
                                "prediction_label": int(prediction_label),
                                **metadata,
                            }
                        )
                    audit_writer.writerow(
                        {
                            "clamp_doc_id": doc_id,
                            "input_file": manifest_row.get("input_file", ""),
                            "expected_output_file": member_name,
                            "output_file_exists": True,
                            "parse_status": parsed.parse_status,
                            "parse_error": parsed.parse_error,
                            "entity_count": len(parsed.entities),
                            "ards_entity_count": ards_count,
                            "prediction_label": prediction_label,
                        }
                    )

                    if len(prediction_batch) >= batch_size:
                        _write_parquet_batch(entity_writer, entity_batch, ENTITY_SCHEMA)
                        _write_parquet_batch(prediction_writer, prediction_batch, PREDICTION_SCHEMA)
                        _write_parquet_batch(
                            probabilistic_writer,
                            probabilistic_batch,
                            PROBABILISTIC_SCHEMA,
                        )
                        entity_batch.clear()
                        prediction_batch.clear()
                        probabilistic_batch.clear()

                _write_parquet_batch(entity_writer, entity_batch, ENTITY_SCHEMA)
                _write_parquet_batch(prediction_writer, prediction_batch, PREDICTION_SCHEMA)
                _write_parquet_batch(
                    probabilistic_writer,
                    probabilistic_batch,
                    PROBABILISTIC_SCHEMA,
                )

        if counters["parse_errors"]:
            raise ValueError(
                "CLAMP TXT archive contains parse failures; no outputs were replaced: "
                f"count={counters['parse_errors']}, examples={parse_error_examples}"
            )

        prediction_rows = len(manifest_records)
        summary = {
            "source_kind": "txt_zip_archive",
            "source_archive_name": output_archive.name,
            "source_archive_sha256": _file_sha256(output_archive),
            "expected_input_files": prediction_rows,
            "observed_output_files": len(members),
            "matched_output_files": prediction_rows,
            "missing_output_files": 0,
            "unexpected_output_files": 0,
            "duplicate_output_doc_ids": 0,
            "duplicate_output_files": 0,
            "ignored_non_txt_archive_members": ignored_members,
            "parse_success_files": int(counters["parse_success"]),
            "parse_error_files": 0,
            "parsed_entity_files": int(counters["parsed"]),
            "parsed_empty_files": int(counters["parsed_empty"]),
            "entity_rows": int(counters["entity_rows"]),
            "documents_with_ards_entity": int(counters["positive_documents"]),
            "documents_without_ards_entity": int(counters["negative_documents"]),
            "evaluable_prediction_rows": prediction_rows,
            "non_evaluable_prediction_rows": 0,
            "semantic_tag_counts": dict(sorted(semantic_tags.items())),
            "assertion_counts": dict(sorted(assertions.items())),
            "warnings": [],
            "unexpected_output_file_examples": [],
            "duplicate_doc_id_examples": [],
        }
        temporary["summary"].write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary["summary_markdown"].write_text(
            render_clamp_teacher_summary(summary),
            encoding="utf-8",
        )
        for name, destination in destinations.items():
            os.replace(temporary[name], destination)
        return summary
    finally:
        for path in temporary.values():
            path.unlink(missing_ok=True)


def parse_clamp_txt_payload(payload: bytes) -> ParsedClampTxt:
    """Parse one strict CLAMP tab-delimited output without retaining full source text."""

    try:
        text = payload.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        return ParsedClampTxt([], "parse_error", f"decode_error:{error.start}")

    reader = csv.DictReader(io.StringIO(text), delimiter="\t")
    if reader.fieldnames is None:
        return ParsedClampTxt([], "parse_error", "missing_header")
    mapping = _recognized_mapping(reader.fieldnames)
    missing_fields = sorted(REQUIRED_ARCHIVE_FIELDS - set(mapping))
    if missing_fields:
        return ParsedClampTxt(
            [],
            "parse_error",
            f"missing_required_fields:{','.join(missing_fields)}",
        )

    entities: list[dict[str, Any]] = []
    for row_number, row in enumerate(reader, start=2):
        if None in row:
            return ParsedClampTxt([], "parse_error", f"extra_fields:row_{row_number}")
        try:
            start = int(str(row[mapping["start"]]).strip())
            end = int(str(row[mapping["end"]]).strip())
        except (TypeError, ValueError):
            return ParsedClampTxt([], "parse_error", f"invalid_offsets:row_{row_number}")
        if start < 0 or end < start:
            return ParsedClampTxt([], "parse_error", f"invalid_offsets:row_{row_number}")

        semantic_tag = _required_cell(row[mapping["semantic_tag"]])
        entity_text = _required_cell(row[mapping["entity_text"]])
        if semantic_tag is None:
            return ParsedClampTxt([], "parse_error", f"blank_semantic_tag:row_{row_number}")
        if entity_text is None:
            return ParsedClampTxt([], "parse_error", f"blank_entity_text:row_{row_number}")
        entities.append(
            {
                "start": start,
                "end": end,
                "semantic_tag": semantic_tag,
                "assertion": _optional_cell(row[mapping["assertion"]]),
                "cui": _optional_cell(row[mapping["cui"]]),
                "attribute": (
                    _optional_cell(row[mapping["attribute"]]) if "attribute" in mapping else None
                ),
                "entity_text": entity_text,
                "entity_text_sha256": hashlib.sha256(entity_text.encode("utf-8")).hexdigest(),
            }
        )

    return ParsedClampTxt(
        entities,
        "parsed" if entities else "parsed_empty",
        "",
    )


def _discover_txt_members(archive: zipfile.ZipFile) -> tuple[dict[str, str], int]:
    members: dict[str, str] = {}
    ignored = 0
    for info in archive.infolist():
        if info.is_dir():
            continue
        _validate_member_name(info.filename)
        if not info.filename.lower().endswith(".txt"):
            ignored += 1
            continue
        doc_id = _member_doc_id(info.filename)
        if doc_id in members:
            raise ValueError(f"Duplicate CLAMP TXT output for clamp_doc_id={doc_id}")
        members[doc_id] = info.filename
    if not members:
        raise ValueError("CLAMP output archive contains no TXT members")
    return members, ignored


def _recognized_mapping(fieldnames: list[str]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for fieldname in fieldnames:
        normalized = "".join(
            character for character in fieldname.strip().lower() if character.isalnum()
        )
        for target, aliases in FIELD_ALIASES.items():
            if target not in mapping and normalized in aliases:
                mapping[target] = fieldname
    return mapping


def _required_cell(value: object) -> str | None:
    clean = str(value or "").strip()
    return clean or None


def _optional_cell(value: object) -> str | None:
    clean = _required_cell(value)
    return None if clean is None or clean.lower() == "null" else clean


def _clean_manifest_value(value: object) -> str:
    return "" if value is None else str(value)


def _write_parquet_batch(
    writer: pq.ParquetWriter,
    rows: list[dict[str, Any]],
    schema: pa.Schema,
) -> None:
    if rows:
        writer.write_table(pa.Table.from_pylist(rows, schema=schema))


def _member_doc_id(member_name: str) -> str:
    name = PurePosixPath(member_name).name
    doc_id = name[:-4] if name.lower().endswith(".txt") else name
    if doc_id.lower().endswith(".txt"):
        doc_id = doc_id[:-4]
    doc_id = doc_id.strip()
    if not doc_id:
        raise ValueError(f"CLAMP output member has a blank document ID: {member_name}")
    return doc_id


def _validate_member_name(member_name: str) -> None:
    path = PurePosixPath(member_name)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Unsafe path in CLAMP output archive: {member_name}")


def _temporary_path(destination: Path) -> Path:
    ensure_parent_dir(destination)
    return destination.with_name(f".{destination.name}.{uuid4().hex}.partial")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
