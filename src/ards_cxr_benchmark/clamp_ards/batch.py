from __future__ import annotations

import csv
import gzip
import hashlib
import json
import os
import platform
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

from ards_cxr_benchmark import __version__

from .fixtures import validate_fixture
from .pipeline import load_legacy_mirror

ENTITY_SCHEMA = pa.schema(
    [
        pa.field("clamp_doc_id", pa.string(), nullable=False),
        pa.field("start", pa.int64(), nullable=False),
        pa.field("end", pa.int64(), nullable=False),
        pa.field("entity_text", pa.string(), nullable=False),
        pa.field("entity_text_sha256", pa.string(), nullable=False),
        pa.field("semantic_tag", pa.string(), nullable=False),
        pa.field("assertion", pa.string(), nullable=False),
        pa.field("cui", pa.string()),
        pa.field("attribute", pa.string()),
        pa.field("duplicate_occurrence", pa.int64(), nullable=False),
    ],
    metadata={b"offset_coordinate_system": b"utf16_code_units"},
)

PREDICTION_SCHEMA = pa.schema(
    [
        pa.field("clamp_doc_id", pa.string(), nullable=False),
        pa.field("prediction_status", pa.string(), nullable=False),
        pa.field("prediction_label", pa.int8(), nullable=False),
        pa.field("clamp_ards_entity_count", pa.int64(), nullable=False),
        pa.field("source_text_sha256", pa.string(), nullable=False),
    ]
)


@dataclass(frozen=True)
class BatchSummary:
    document_count: int
    positive_document_count: int
    entity_count: int
    entity_output: str
    prediction_output: str
    source_input: str
    source_input_sha256: str
    implementation_version: str
    python_version: str
    platform: str
    clamp_project_dir: str
    resource_sha256: dict[str, str]
    phenotype_spec_version: str
    phenotype_spec_sha256: str
    offset_coordinate_system: str

    def as_dict(self) -> dict[str, object]:
        return {
            "document_count": self.document_count,
            "positive_document_count": self.positive_document_count,
            "entity_count": self.entity_count,
            "entity_output": self.entity_output,
            "prediction_output": self.prediction_output,
            "source_input": self.source_input,
            "source_input_sha256": self.source_input_sha256,
            "implementation_version": self.implementation_version,
            "python_version": self.python_version,
            "platform": self.platform,
            "clamp_project_dir": self.clamp_project_dir,
            "resource_sha256": self.resource_sha256,
            "phenotype_spec_version": self.phenotype_spec_version,
            "phenotype_spec_sha256": self.phenotype_spec_sha256,
            "offset_coordinate_system": self.offset_coordinate_system,
        }


def iter_input_documents(
    input_path: Path,
    *,
    id_column: str,
    text_column: str,
    id_prefix: str = "",
) -> Iterator[tuple[str, str]]:
    """Yield immutable document IDs and text from CSV, Parquet, or JSON Lines."""

    suffixes = input_path.suffixes
    if input_path.suffix == ".parquet":
        parquet = pq.ParquetFile(input_path)
        available = set(parquet.schema_arrow.names)
        _require_columns(available, id_column, text_column, input_path)
        for batch in parquet.iter_batches(columns=[id_column, text_column], batch_size=4096):
            for row in batch.to_pylist():
                yield _validated_document(row[id_column], row[text_column], id_prefix)
        return

    opener = gzip.open if suffixes and suffixes[-1] == ".gz" else open
    if ".jsonl" in suffixes or input_path.suffix == ".jsonl":
        with opener(input_path, "rt", encoding="utf-8", newline="") as handle:
            for line_number, line in enumerate(handle, start=1):
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON on line {line_number}: {input_path}") from exc
                if not isinstance(row, dict):
                    raise ValueError(f"JSON line {line_number} is not an object: {input_path}")
                _require_columns(set(row), id_column, text_column, input_path)
                yield _validated_document(row[id_column], row[text_column], id_prefix)
        return

    if ".csv" in suffixes or input_path.suffix == ".csv":
        with opener(input_path, "rt", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            _require_columns(set(reader.fieldnames or []), id_column, text_column, input_path)
            for row in reader:
                yield _validated_document(row[id_column], row[text_column], id_prefix)
        return

    raise ValueError(f"Unsupported input format (expected CSV, Parquet, or JSONL): {input_path}")


def iter_fixture_documents(
    fixture_root: Path,
    *,
    project_dir: Path | None = None,
    resource_manifest: Path | None = None,
) -> Iterator[tuple[str, str]]:
    """Yield validated fixture inputs in frozen manifest order without newline conversion."""

    fixture_root = fixture_root.expanduser().resolve()
    validate_fixture(
        fixture_root,
        allow_pending=True,
        project_dir=project_dir,
        resource_manifest_path=resource_manifest,
    )
    manifest_path = fixture_root / "manifest.csv"
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"case_id", "input_path", "input_sha256"}
        missing = sorted(required - set(reader.fieldnames or ()))
        if missing:
            raise ValueError(f"Fixture manifest is missing column(s) {missing}: {manifest_path}")
        for row in reader:
            case_id = str(row["case_id"])
            input_path = fixture_root / str(row["input_path"])
            payload = input_path.read_bytes()
            digest = hashlib.sha256(payload).hexdigest()
            if digest != row["input_sha256"]:
                raise ValueError(f"Fixture input SHA-256 mismatch for case {case_id}")
            yield case_id, payload.decode("utf-8", errors="strict")


def run_clamp_ards_batch(
    *,
    input_path: Path | None = None,
    fixture_root: Path | None = None,
    entity_output: Path,
    prediction_output: Path,
    id_column: str,
    text_column: str,
    id_prefix: str = "",
    summary_output: Path | None = None,
    project_dir: Path | None = None,
    resource_manifest: Path | None = None,
    batch_size: int = 5000,
    limit: int | None = None,
    show_progress: bool = True,
) -> BatchSummary:
    """Run the compatibility pipeline without retaining the input corpus in memory."""

    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    if limit is not None and limit < 1:
        raise ValueError("limit must be at least 1")
    if (input_path is None) == (fixture_root is None):
        raise ValueError("Provide exactly one of input_path or fixture_root")
    source_path = input_path if input_path is not None else fixture_root
    if source_path is None:  # Narrow the type after the exclusive-or validation.
        raise AssertionError("unreachable")
    _require_distinct_paths(source_path, entity_output, prediction_output, summary_output)
    if fixture_root is not None:
        _require_outputs_outside_fixture(
            fixture_root,
            entity_output,
            prediction_output,
            summary_output,
        )
    mirror = load_legacy_mirror(
        str(project_dir.resolve()) if project_dir else None,
        str(resource_manifest.resolve()) if resource_manifest else None,
    )
    entity_partial = _partial_path(entity_output)
    prediction_partial = _partial_path(prediction_output)
    entity_output.parent.mkdir(parents=True, exist_ok=True)
    prediction_output.parent.mkdir(parents=True, exist_ok=True)
    entity_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []
    seen: set[str] = set()
    document_count = positive_count = entity_count = 0

    entity_writer = pq.ParquetWriter(entity_partial, ENTITY_SCHEMA, compression="zstd")
    prediction_writer = pq.ParquetWriter(
        prediction_partial,
        PREDICTION_SCHEMA,
        compression="zstd",
    )
    try:
        if fixture_root is not None:
            documents = iter_fixture_documents(
                fixture_root,
                project_dir=project_dir,
                resource_manifest=resource_manifest,
            )
        else:
            if input_path is None:
                raise AssertionError("unreachable")
            documents = iter_input_documents(
                input_path,
                id_column=id_column,
                text_column=text_column,
                id_prefix=id_prefix,
            )
        progress = tqdm(documents, desc="Python CLAMP ARDS", disable=not show_progress)
        for doc_id, text in progress:
            if limit is not None and document_count >= limit:
                break
            if doc_id in seen:
                raise ValueError(f"Duplicate input document ID: {doc_id}")
            seen.add(doc_id)
            entities = mirror.run(text, doc_id=doc_id)
            occurrences: dict[tuple[object, ...], int] = {}
            for entity in entities:
                entity_text = entity.covered_text(text)
                key = (
                    entity.start,
                    entity.end,
                    entity_text,
                    entity.semantic_tag,
                    entity.assertion,
                    entity.cui,
                    entity.attribute,
                )
                occurrence = occurrences.get(key, 0)
                occurrences[key] = occurrence + 1
                entity_rows.append(
                    {
                        "clamp_doc_id": doc_id,
                        "start": entity.start,
                        "end": entity.end,
                        "entity_text": entity_text,
                        "entity_text_sha256": _text_sha256(entity_text),
                        "semantic_tag": entity.semantic_tag,
                        "assertion": entity.assertion,
                        "cui": entity.cui,
                        "attribute": entity.attribute,
                        "duplicate_occurrence": occurrence,
                    }
                )
            count = len(entities)
            prediction_rows.append(
                {
                    "clamp_doc_id": doc_id,
                    "prediction_status": "evaluable",
                    "prediction_label": int(count > 0),
                    "clamp_ards_entity_count": count,
                    "source_text_sha256": _text_sha256(text),
                }
            )
            document_count += 1
            positive_count += int(count > 0)
            entity_count += count
            if len(prediction_rows) >= batch_size:
                _write_rows(entity_writer, ENTITY_SCHEMA, entity_rows)
                _write_rows(prediction_writer, PREDICTION_SCHEMA, prediction_rows)
                entity_rows.clear()
                prediction_rows.clear()
        _write_rows(entity_writer, ENTITY_SCHEMA, entity_rows)
        _write_rows(prediction_writer, PREDICTION_SCHEMA, prediction_rows)
    except BaseException:
        entity_writer.close()
        prediction_writer.close()
        entity_partial.unlink(missing_ok=True)
        prediction_partial.unlink(missing_ok=True)
        raise
    else:
        entity_writer.close()
        prediction_writer.close()
        os.replace(entity_partial, entity_output)
        os.replace(prediction_partial, prediction_output)

    summary = BatchSummary(
        document_count=document_count,
        positive_document_count=positive_count,
        entity_count=entity_count,
        entity_output=str(entity_output),
        prediction_output=str(prediction_output),
        source_input=str(source_path),
        source_input_sha256=(
            _fixture_input_tree_sha256(fixture_root)
            if fixture_root is not None
            else _file_sha256(source_path)
        ),
        implementation_version=__version__,
        python_version=sys.version.split()[0],
        platform=platform.platform(),
        clamp_project_dir=str(mirror.resources.project_dir),
        resource_sha256=mirror.resources.resource_sha256,
        phenotype_spec_version=mirror.resources.phenotype_spec_version,
        phenotype_spec_sha256=mirror.resources.phenotype_spec_sha256,
        offset_coordinate_system="utf16_code_units",
    )
    if summary_output is not None:
        summary_output.parent.mkdir(parents=True, exist_ok=True)
        summary_output.write_text(
            json.dumps(summary.as_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return summary


def _validated_document(raw_id: object, raw_text: object, prefix: str) -> tuple[str, str]:
    if raw_id is None or str(raw_id) == "":
        raise ValueError("Input document ID is blank")
    if not isinstance(raw_text, str):
        raise ValueError(f"Input text for {raw_id!r} must be a string")
    return f"{prefix}{raw_id}", raw_text


def _require_columns(
    available: set[str],
    id_column: str,
    text_column: str,
    path: Path,
) -> None:
    missing = sorted({id_column, text_column} - available)
    if missing:
        raise ValueError(f"Missing input column(s) {missing}: {path}")


def _write_rows(writer: pq.ParquetWriter, schema: pa.Schema, rows: list[dict[str, object]]) -> None:
    if rows:
        writer.write_table(pa.Table.from_pylist(rows, schema=schema))


def _partial_path(path: Path) -> Path:
    return path.with_name(f".{path.name}.{uuid4().hex}.partial")


def _text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_distinct_paths(input_path: Path, *outputs: Path | None) -> None:
    paths = [input_path.resolve(), *(path.resolve() for path in outputs if path is not None)]
    if len(paths) != len(set(paths)):
        raise ValueError("Input and output paths must be distinct")


def _require_outputs_outside_fixture(fixture_root: Path, *outputs: Path | None) -> None:
    root = fixture_root.expanduser().resolve()
    nested = [path for path in outputs if path is not None and path.resolve().is_relative_to(root)]
    if nested:
        raise ValueError("Generated Python CLAMP outputs must remain outside the fixture root")


def _fixture_input_tree_sha256(fixture_root: Path) -> str:
    rows: list[str] = []
    with (fixture_root / "manifest.csv").open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(f"{row['case_id']}\t{row['input_sha256']}")
    return hashlib.sha256(("\n".join(rows) + "\n").encode("utf-8")).hexdigest()
