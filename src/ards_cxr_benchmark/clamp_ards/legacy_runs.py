from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
import zipfile
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path, PurePosixPath
from typing import Any
from uuid import uuid4

from ards_cxr_benchmark.clamp_ards_output_archive import parse_clamp_txt_payload

from .fixtures import EXPECTED_ENTITY_FIELDS, validate_fixture, write_sha256s
from .resources import default_resource_manifest_path
from .xmi import ClampXmiDocument, parse_clamp_xmi

FIXED_ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)
MAX_OUTPUT_MEMBER_BYTES = 50 * 1024 * 1024
MAX_RUN_PAYLOAD_BYTES = 512 * 1024 * 1024
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_LOCAL_PATH_VALUE_RE = re.compile(
    r"(?i)(?:^|[\s\"'(`])(?:[A-Z]:[\\/]|\\\\[^\\/\s]+[\\/]|/(?:Users|home|Volumes|tmp)/)"
)
_EXPECTED_CLAMP_VERSION = "1.6.6"


@dataclass(frozen=True)
class FixtureCase:
    case_id: str
    category: str
    input_file: str
    payload: bytes
    sha256: str

    @property
    def text(self) -> str:
        return self.payload.decode("utf-8")


@dataclass(frozen=True)
class OutputPayload:
    relative_path: str
    payload: bytes

    @property
    def sha256(self) -> str:
        return _bytes_sha256(self.payload)


@dataclass(frozen=True)
class RunInventory:
    source_name: str
    txt: Mapping[str, OutputPayload]
    xmi: Mapping[str, OutputPayload]


@dataclass(frozen=True)
class LegacyDocument:
    case_id: str
    source_text_sha256: str
    txt_sha256: str
    xmi_sha256: str
    sentences: tuple[tuple[int, int, int, str], ...]
    tokens: tuple[tuple[int, int, int, str], ...]
    entities: tuple[tuple[int, int, str, str, str | None, str | None, str], ...]
    txt_rows: tuple[tuple[int, int, str, str, str | None, str], ...]


@dataclass(frozen=True)
class LegacyRun:
    label: str
    documents: Mapping[str, LegacyDocument]
    provenance: Mapping[str, Any]


@dataclass(frozen=True)
class LegacyImportSummary:
    case_count: int
    sentence_count: int
    token_count: int
    entity_count: int
    raw_row_order_stable: bool
    output_order_required: bool
    xmi_entity_order_stable: bool
    txt_order_difference_documents: int
    xmi_order_difference_documents: int
    fixture_root: str
    candidate_output_dir: str
    finalized: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "case_count": self.case_count,
            "sentence_count": self.sentence_count,
            "token_count": self.token_count,
            "entity_count": self.entity_count,
            "raw_row_order_stable": self.raw_row_order_stable,
            "output_order_required": self.output_order_required,
            "xmi_entity_order_stable": self.xmi_entity_order_stable,
            "txt_order_difference_documents": self.txt_order_difference_documents,
            "xmi_order_difference_documents": self.xmi_order_difference_documents,
            "fixture_root": self.fixture_root,
            "candidate_output_dir": self.candidate_output_dir,
            "finalized": self.finalized,
        }


@dataclass(frozen=True)
class LegacyRepeatability:
    """Semantic repeatability plus separately observed exporter-order behavior."""

    txt_order_difference_documents: int
    xmi_order_difference_documents: int

    @property
    def txt_row_order_stable(self) -> bool:
        return self.txt_order_difference_documents == 0

    @property
    def xmi_entity_order_stable(self) -> bool:
        return self.xmi_order_difference_documents == 0


def prepare_legacy_clamp_parity_handoff(
    *,
    fixture_root: Path,
    project_source_dir: Path,
    destination_dir: Path,
    collector_script: Path,
    output_archive: Path | None = None,
    resource_manifest: Path | None = None,
    project_commit: str | None = None,
    expected_windows_project_dir: str = r"C:\ClampWin_1.6.6\workspace\ARDS",
    overwrite: bool = False,
) -> dict[str, Any]:
    """Build an ignored, deterministic handoff without touching the restricted oracle."""

    fixture_root = fixture_root.resolve()
    project_source_dir = project_source_dir.resolve()
    destination_dir = destination_dir.resolve()
    collector_script = collector_script.resolve()
    output_archive = (output_archive or destination_dir.with_suffix(".zip")).resolve()
    _require_existing_directory(fixture_root, "Fixture root")
    validate_fixture(fixture_root, allow_pending=True)
    _require_existing_directory(project_source_dir, "Frozen CLAMP project")
    if not collector_script.is_file():
        raise FileNotFoundError(f"Windows provenance collector not found: {collector_script}")
    if output_archive.is_relative_to(destination_dir):
        raise ValueError("Handoff ZIP must not be inside the handoff directory")
    for protected in (fixture_root, project_source_dir, collector_script):
        _require_disjoint_generated_path(destination_dir, protected)
        _require_disjoint_generated_path(output_archive, protected)
    if (destination_dir.exists() or output_archive.exists()) and not overwrite:
        raise FileExistsError(
            "Handoff destination already exists; pass overwrite=True to replace generated output"
        )

    cases = load_fixture_cases(fixture_root)
    manifest_path = fixture_root / "manifest.csv"
    cases_path = fixture_root / "cases.yaml"
    if not cases_path.is_file():
        raise FileNotFoundError(f"Synthetic case declaration not found: {cases_path}")
    manifest_payload = manifest_path.read_bytes()
    cases_payload = cases_path.read_bytes()
    frozen_contract = _load_fixture_frozen_contract(
        fixture_root,
        resource_manifest=resource_manifest,
    )
    frozen = _validate_frozen_project(
        project_source_dir,
        resource_manifest=Path(str(frozen_contract["manifest_path"])),
        project_commit=project_commit,
    )
    project_commit = str(frozen["project_commit"])
    if project_commit != frozen_contract["project_commit"]:
        raise ValueError(
            "Frozen project commit differs between the fixture and resource manifest: "
            f"{frozen_contract['project_commit']} != {project_commit}"
        )

    temporary_dir = destination_dir.with_name(f".{destination_dir.name}.{uuid4().hex}.partial")
    temporary_archive = output_archive.with_name(f".{output_archive.name}.{uuid4().hex}.partial")
    backup_dir = destination_dir.with_name(f".{destination_dir.name}.{uuid4().hex}.backup")
    backup_archive = output_archive.with_name(f".{output_archive.name}.{uuid4().hex}.backup")
    temporary_dir.mkdir(parents=True)
    try:
        project_destination = temporary_dir / "ARDS"
        _copy_frozen_project(project_source_dir, project_destination)
        input_dir = project_destination / "Data" / "Input"
        output_dir = project_destination / "Data" / "Output"
        input_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / ".keep").write_bytes(b"")
        for case in cases:
            (input_dir / PurePosixPath(case.input_file).name).write_bytes(case.payload)

        (temporary_dir / "manifest.csv").write_bytes(manifest_payload)
        (temporary_dir / "cases.yaml").write_bytes(cases_payload)
        shutil.copyfile(collector_script, temporary_dir / collector_script.name)
        (temporary_dir / "RUNBOOK.md").write_text(
            _render_handoff_runbook(
                case_count=len(cases),
                project_commit=project_commit,
                expected_windows_project_dir=expected_windows_project_dir,
                collector_name=collector_script.name,
            ),
            encoding="utf-8",
            newline="\n",
        )
        handoff_provenance = {
            "schema_version": 1,
            "source_kind": "synthetic_non_phi_candidate",
            "expected_case_count": len(cases),
            "expected_windows_project_dir": expected_windows_project_dir,
            "project_commit": project_commit,
            "project_files_sha256": frozen["files"],
            "resource_manifest_sha256": frozen_contract["manifest_sha256"],
            "fixture_manifest_sha256": _bytes_sha256(manifest_payload),
            "fixture_cases_sha256": _bytes_sha256(cases_payload),
            "collector_sha256": _file_sha256(collector_script),
            "restricted_oracle_included": False,
        }
        (temporary_dir / "handoff_provenance.json").write_text(
            json.dumps(handoff_provenance, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        _write_sha256sums(temporary_dir)
        _write_deterministic_zip(temporary_dir, temporary_archive)
        archive_sha256 = _file_sha256(temporary_archive)

        _replace_generated_path(
            temporary_dir,
            destination_dir,
            backup=backup_dir,
            overwrite=overwrite,
        )
        try:
            _replace_generated_path(
                temporary_archive,
                output_archive,
                backup=backup_archive,
                overwrite=overwrite,
            )
        except BaseException:
            if backup_dir.exists():
                _restore_backup(destination_dir, backup_dir)
            else:
                shutil.rmtree(destination_dir, ignore_errors=True)
            raise
        shutil.rmtree(backup_dir, ignore_errors=True)
        backup_archive.unlink(missing_ok=True)
    finally:
        shutil.rmtree(temporary_dir, ignore_errors=True)
        temporary_archive.unlink(missing_ok=True)
        shutil.rmtree(backup_dir, ignore_errors=True)
        backup_archive.unlink(missing_ok=True)

    return {
        "case_count": len(cases),
        "project_commit": project_commit,
        "project_file_count": len(frozen["files"]),
        "destination_dir": str(destination_dir),
        "output_archive": str(output_archive),
        "output_archive_sha256": archive_sha256,
        "restricted_oracle_included": False,
    }


def import_legacy_clamp_parity_runs(
    *,
    fixture_root: Path,
    run_1_source: Path,
    run_2_source: Path,
    run_1_provenance: Path,
    run_2_provenance: Path,
    run_1_sha256s: Path | None = None,
    run_2_sha256s: Path | None = None,
    candidate_output_dir: Path,
    resource_manifest: Path | None = None,
    finalize: bool = False,
    phi_reviewer: str | None = None,
    phi_reviewed_at: str | None = None,
    redistribution_authority: str | None = None,
    redistribution_evidence: str | None = None,
) -> LegacyImportSummary:
    """Validate two legacy runs and transactionally install genuine expected fixtures."""

    fixture_root = fixture_root.resolve()
    run_sources = (run_1_source.resolve(), run_2_source.resolve())
    provenance_paths = (run_1_provenance.resolve(), run_2_provenance.resolve())
    checksum_paths = (
        (run_1_sha256s or provenance_paths[0].with_suffix(".SHA256SUMS")).resolve(),
        (run_2_sha256s or provenance_paths[1].with_suffix(".SHA256SUMS")).resolve(),
    )
    resource_manifest_path = (resource_manifest or default_resource_manifest_path()).resolve()
    candidate_output_dir = candidate_output_dir.resolve()
    protected_inputs = (
        fixture_root,
        *run_sources,
        *provenance_paths,
        *checksum_paths,
        resource_manifest_path,
    )
    for protected in protected_inputs:
        _require_disjoint_generated_path(candidate_output_dir, protected)
    validate_fixture(fixture_root, allow_pending=True)
    frozen_contract = _load_fixture_frozen_contract(
        fixture_root,
        resource_manifest=resource_manifest_path,
    )
    cases = load_fixture_cases(fixture_root)
    expected_ids = {case.case_id for case in cases}
    inventories = (
        discover_legacy_output(run_sources[0]),
        discover_legacy_output(run_sources[1]),
    )
    provenance_payloads = tuple(
        _load_provenance(path, expected_label=label)
        for path, label in zip(provenance_paths, ("run_1", "run_2"), strict=True)
    )
    for label, provenance, checksum_path in zip(
        ("run_1", "run_2"), provenance_payloads, checksum_paths, strict=True
    ):
        _validate_returned_checksum_manifest(
            checksum_path,
            provenance,
            label=label,
        )
    runs: list[LegacyRun] = []
    for label, inventory, provenance in zip(
        ("run_1", "run_2"), inventories, provenance_payloads, strict=True
    ):
        _validate_returned_project_contract(provenance, frozen_contract, label=label)
        _validate_inventory(inventory, expected_ids, label=label)
        _validate_provenance_output_manifest(provenance, inventory, label=label)
        documents = {
            case.case_id: _parse_legacy_document(case, inventory, run_label=label) for case in cases
        }
        runs.append(LegacyRun(label=label, documents=documents, provenance=provenance))

    _validate_runtime_repeatability(runs[0].provenance, runs[1].provenance)
    _validate_run_chronology(runs[0].provenance, runs[1].provenance)
    repeatability = _validate_exact_repeat(runs[0], runs[1], expected_ids)
    summary = LegacyImportSummary(
        case_count=len(cases),
        sentence_count=sum(len(item.sentences) for item in runs[0].documents.values()),
        token_count=sum(len(item.tokens) for item in runs[0].documents.values()),
        entity_count=sum(len(item.entities) for item in runs[0].documents.values()),
        raw_row_order_stable=repeatability.txt_row_order_stable,
        output_order_required=repeatability.txt_row_order_stable,
        xmi_entity_order_stable=repeatability.xmi_entity_order_stable,
        txt_order_difference_documents=repeatability.txt_order_difference_documents,
        xmi_order_difference_documents=repeatability.xmi_order_difference_documents,
        fixture_root=str(fixture_root),
        candidate_output_dir=str(candidate_output_dir),
        finalized=finalize,
    )
    candidate = _write_candidate_fixture_outputs(
        candidate_output_dir,
        fixture_root,
        cases=cases,
        run_1=runs[0],
        run_2=runs[1],
        summary=summary,
    )
    if finalize:
        reviews = {
            "phi_reviewer": _required_review_value(phi_reviewer, "phi_reviewer"),
            "phi_reviewed_at": _required_review_timestamp(phi_reviewed_at),
            "redistribution_authority": _required_review_value(
                redistribution_authority, "redistribution_authority"
            ),
            "redistribution_evidence": _required_review_value(
                redistribution_evidence, "redistribution_evidence"
            ),
        }
        _finalize_expected_fixture(
            fixture_root,
            candidate=candidate,
            run_1=runs[0],
            run_2=runs[1],
            summary=summary,
            reviews=reviews,
        )
    return summary


def load_fixture_cases(fixture_root: Path) -> tuple[FixtureCase, ...]:
    fixture_root = fixture_root.resolve()
    manifest_path = fixture_root / "manifest.csv"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Synthetic fixture manifest not found: {manifest_path}")
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Fixture manifest has no header: {manifest_path}")
        case_column = _manifest_column(reader.fieldnames, "clamp_doc_id", "case_id")
        input_column = _manifest_column(reader.fieldnames, "input_path", "input_file")
        category_column = _manifest_column(
            reader.fieldnames, "primary_category", "category", required=False
        )
        byte_column = _manifest_column(
            reader.fieldnames, "byte_count", "input_bytes", "bytes", required=False
        )
        sha_column = _manifest_column(
            reader.fieldnames,
            "sha256",
            "text_sha256",
            "input_sha256",
            "source_text_sha256",
        )
        records = list(reader)
    if not records:
        raise ValueError("Fixture manifest contains no cases")

    cases: list[FixtureCase] = []
    seen_ids: set[str] = set()
    seen_inputs: set[str] = set()
    for row_number, row in enumerate(records, start=2):
        case_id = str(row.get(case_column, "")).strip()
        if not case_id:
            raise ValueError(f"Blank fixture case ID on manifest row {row_number}")
        if case_id in seen_ids:
            raise ValueError(f"Duplicate fixture case ID: {case_id}")
        seen_ids.add(case_id)
        input_name = str(row.get(input_column, "")).strip().replace("\\", "/")
        relative = PurePosixPath(input_name)
        if not input_name or relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Unsafe fixture input path on row {row_number}: {input_name}")
        if input_name in seen_inputs:
            raise ValueError(f"Duplicate fixture input path: {input_name}")
        seen_inputs.add(input_name)
        input_basename = relative.name
        if not input_basename.lower().endswith(".txt"):
            raise ValueError(f"Fixture input filename must end in .txt: {input_name}")
        if _output_case_id(input_basename, suffix=".txt") != case_id:
            raise ValueError(
                f"Fixture input filename stem must equal case ID {case_id}: {input_name}"
            )
        candidates = (fixture_root / relative, fixture_root / "input" / relative.name)
        input_path = next((path for path in candidates if path.is_file()), None)
        if input_path is None:
            raise FileNotFoundError(f"Fixture input is missing for case {case_id}: {input_name}")
        payload = input_path.read_bytes()
        try:
            payload.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(f"Fixture input is not strict UTF-8: {input_name}") from exc
        expected_sha = str(row.get(sha_column, "")).strip().lower()
        if not _SHA256_RE.fullmatch(expected_sha):
            raise ValueError(f"Invalid fixture SHA-256 on row {row_number}: {expected_sha}")
        actual_sha = _bytes_sha256(payload)
        if expected_sha != actual_sha:
            raise ValueError(f"Fixture input SHA-256 mismatch for case {case_id}")
        if byte_column:
            try:
                expected_bytes = int(str(row.get(byte_column, "")).strip())
            except ValueError as exc:
                raise ValueError(f"Invalid fixture byte count on row {row_number}") from exc
            if expected_bytes != len(payload):
                raise ValueError(f"Fixture input byte count mismatch for case {case_id}")
        category = str(row.get(category_column, "")).strip() if category_column else ""
        cases.append(
            FixtureCase(
                case_id=case_id,
                category=category,
                input_file=input_name,
                payload=payload,
                sha256=actual_sha,
            )
        )
    return tuple(cases)


def discover_legacy_output(source: Path) -> RunInventory:
    source = source.resolve()
    if source.is_dir():
        entries = _read_directory_output(source)
    elif source.is_file() and zipfile.is_zipfile(source):
        entries = _read_zip_output(source)
    else:
        raise ValueError(f"Legacy CLAMP run must be a directory or readable ZIP: {source}")
    txt: dict[str, OutputPayload] = {}
    xmi: dict[str, OutputPayload] = {}
    for entry in entries:
        lower = entry.relative_path.lower()
        if lower.endswith(".txt"):
            destination = txt
            suffix = ".txt"
        elif lower.endswith(".xmi"):
            destination = xmi
            suffix = ".xmi"
        else:
            continue
        case_id = _output_case_id(PurePosixPath(entry.relative_path).name, suffix=suffix)
        if case_id in destination:
            raise ValueError(
                f"Duplicate {suffix[1:].upper()} output for case {case_id}: {source.name}"
            )
        destination[case_id] = entry
    return RunInventory(source_name=source.name, txt=txt, xmi=xmi)


def _parse_legacy_document(
    case: FixtureCase,
    inventory: RunInventory,
    *,
    run_label: str,
) -> LegacyDocument:
    text = case.text
    txt_payload = inventory.txt[case.case_id]
    xmi_payload = inventory.xmi[case.case_id]
    parsed_txt = parse_clamp_txt_payload(txt_payload.payload)
    if parsed_txt.parse_status == "parse_error":
        raise ValueError(
            f"{run_label} TXT parse failure for {case.case_id}: {parsed_txt.parse_error}"
        )
    try:
        parsed_xmi = parse_clamp_xmi(xmi_payload.payload)
    except ValueError as exc:
        if "UTF-16 offset" in str(exc):
            raise ValueError(
                f"{run_label} invalid XMI entity offset for {case.case_id}: {exc}"
            ) from exc
        raise ValueError(f"{run_label} XMI parse failure for {case.case_id}: {exc}") from exc
    except Exception as exc:
        raise ValueError(f"{run_label} XMI parse failure for {case.case_id}: {exc}") from exc
    if parsed_xmi.text != text:
        raise ValueError(f"{run_label} XMI Sofa text differs from input for {case.case_id}")

    sentences = _normalized_spans(text, parsed_xmi, kind="sentence", run_label=run_label)
    tokens = _normalized_spans(text, parsed_xmi, kind="token", run_label=run_label)
    entities = _normalized_xmi_entities(text, parsed_xmi, run_label=run_label, case_id=case.case_id)
    txt_rows = _normalized_txt_entities(
        text,
        parsed_txt.entities,
        run_label=run_label,
        case_id=case.case_id,
    )
    shared_xmi = tuple(
        (start, end, semantic, assertion, cui, entity_text)
        for start, end, semantic, assertion, cui, _attribute, entity_text in entities
    )
    if Counter(txt_rows) != Counter(shared_xmi):
        raise ValueError(
            f"{run_label} TXT/XMI final-entity disagreement for {case.case_id}: "
            "entity fields or multiplicity differ"
        )
    return LegacyDocument(
        case_id=case.case_id,
        source_text_sha256=case.sha256,
        txt_sha256=txt_payload.sha256,
        xmi_sha256=xmi_payload.sha256,
        sentences=sentences,
        tokens=tokens,
        entities=entities,
        txt_rows=txt_rows,
    )


def _normalized_spans(
    text: str,
    document: ClampXmiDocument,
    *,
    kind: str,
    run_label: str,
) -> tuple[tuple[int, int, int, str], ...]:
    spans = document.sentences if kind == "sentence" else document.tokens
    numbers = [item.sentence_number if kind == "sentence" else item.token_number for item in spans]
    if len(numbers) != len(set(numbers)):
        raise ValueError(f"{run_label} XMI contains duplicate {kind} numbers")
    normalized: list[tuple[int, int, int, str]] = []
    for item, number in zip(spans, numbers, strict=True):
        covered = _utf16_slice(text, item.start, item.end)
        normalized.append((item.start, item.end, number, covered))
    return tuple(normalized)


def _normalized_xmi_entities(
    text: str,
    document: ClampXmiDocument,
    *,
    run_label: str,
    case_id: str,
) -> tuple[tuple[int, int, str, str, str | None, str | None, str], ...]:
    result = []
    for item in document.entities:
        try:
            covered = _utf16_slice(text, item.start, item.end)
        except ValueError as exc:
            raise ValueError(f"{run_label} invalid XMI entity offset for {case_id}: {exc}") from exc
        result.append(
            (
                item.start,
                item.end,
                item.semantic_tag,
                item.assertion,
                item.cui,
                item.attribute,
                covered,
            )
        )
    return tuple(result)


def _normalized_txt_entities(
    text: str,
    entities: Iterable[Mapping[str, Any]],
    *,
    run_label: str,
    case_id: str,
) -> tuple[tuple[int, int, str, str, str | None, str], ...]:
    result = []
    for item in entities:
        start = int(item["start"])
        end = int(item["end"])
        try:
            covered = _utf16_slice(text, start, end)
        except ValueError as exc:
            raise ValueError(f"{run_label} invalid TXT entity offset for {case_id}: {exc}") from exc
        entity_text = str(item["entity_text"])
        if covered != entity_text:
            raise ValueError(
                f"{run_label} TXT covered text differs from input for {case_id} at [{start}, {end})"
            )
        result.append(
            (
                start,
                end,
                str(item["semantic_tag"]),
                str(item["assertion"] or "present"),
                _nullable(item.get("cui")),
                entity_text,
            )
        )
    return tuple(result)


def _validate_inventory(inventory: RunInventory, expected_ids: set[str], *, label: str) -> None:
    txt_ids = set(inventory.txt)
    xmi_ids = set(inventory.xmi)
    missing_txt = sorted(expected_ids - txt_ids)
    missing_xmi = sorted(expected_ids - xmi_ids)
    extra_txt = sorted(txt_ids - expected_ids)
    extra_xmi = sorted(xmi_ids - expected_ids)
    if missing_txt or missing_xmi or extra_txt or extra_xmi:
        raise ValueError(
            f"{label} output inventory mismatch: missing_txt={missing_txt[:5]}, "
            f"missing_xmi={missing_xmi[:5]}, extra_txt={extra_txt[:5]}, "
            f"extra_xmi={extra_xmi[:5]}"
        )


def _validate_exact_repeat(
    run_1: LegacyRun,
    run_2: LegacyRun,
    expected_ids: set[str],
) -> LegacyRepeatability:
    mismatches: list[str] = []
    txt_order_differences = 0
    xmi_order_differences = 0
    for case_id in sorted(expected_ids):
        first = run_1.documents[case_id]
        second = run_2.documents[case_id]
        if first.sentences != second.sentences:
            mismatches.append(f"{case_id}:sentences")
        if first.tokens != second.tokens:
            mismatches.append(f"{case_id}:tokens")
        if Counter(first.entities) != Counter(second.entities):
            mismatches.append(f"{case_id}:entity_multiset")
        if Counter(first.txt_rows) != Counter(second.txt_rows):
            mismatches.append(f"{case_id}:txt_entity_multiset")
        xmi_order_differences += int(first.entities != second.entities)
        txt_order_differences += int(first.txt_rows != second.txt_rows)
    if mismatches:
        raise ValueError("Legacy CLAMP runs are not exact repeats: " + ", ".join(mismatches[:20]))
    return LegacyRepeatability(
        txt_order_difference_documents=txt_order_differences,
        xmi_order_difference_documents=xmi_order_differences,
    )


def _write_candidate_fixture_outputs(
    candidate_output_dir: Path,
    fixture_root: Path,
    *,
    cases: tuple[FixtureCase, ...],
    run_1: LegacyRun,
    run_2: LegacyRun,
    summary: LegacyImportSummary,
) -> Path:
    temporary = candidate_output_dir.with_name(
        f".{candidate_output_dir.name}.{uuid4().hex}.partial"
    )
    backup = candidate_output_dir.with_name(f".{candidate_output_dir.name}.{uuid4().hex}.backup")
    temporary.mkdir(parents=True)
    try:
        clamp_expected = temporary / "clamp_expected"
        intermediate_expected = temporary / "intermediate_expected"
        clamp_expected.mkdir(parents=True)
        intermediate_expected.mkdir(parents=True)
        for name in ("clamp_expected", "intermediate_expected"):
            source_readme = fixture_root / name / "README.md"
            if source_readme.is_file():
                shutil.copyfile(source_readme, temporary / name / "README.md")
        for case in cases:
            document = run_1.documents[case.case_id]
            _write_expected_tsv(clamp_expected / f"{case.case_id}.tsv", document)
            _write_intermediate_json(
                intermediate_expected / f"{case.case_id}.json",
                document,
                repeated=run_2.documents[case.case_id],
            )
        previous = _read_optional_json(fixture_root / "provenance.json")
        proposed = _complete_fixture_provenance(
            previous,
            run_1=run_1.provenance,
            run_2=run_2.provenance,
            summary=summary,
            reviews=None,
        )
        proposed["lifecycle"] = "awaiting_reviews"
        (temporary / "proposed_provenance.json").write_text(
            json.dumps(proposed, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        (temporary / "import_summary.json").write_text(
            json.dumps(summary.as_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        (temporary / "REVIEW_REQUIRED.md").write_text(
            "# Review required before fixture finalization\n\n"
            "Both legacy runs passed the strict technical import. The tracked fixture has not "
            "been modified. Before finalization, record: (1) the manual non-PHI reviewer and "
            "review timestamp, and (2) the redistribution authority and documentary evidence "
            "covering these normalized legacy outputs. Re-run the importer with `--finalize` "
            "and all four review metadata options. Resource redistribution remains a separate "
            "repository release gate.\n",
            encoding="utf-8",
            newline="\n",
        )
        write_sha256s(temporary)
        _replace_generated_path(
            temporary,
            candidate_output_dir,
            backup=backup,
            overwrite=True,
        )
        shutil.rmtree(backup, ignore_errors=True)
    finally:
        shutil.rmtree(temporary, ignore_errors=True)
        shutil.rmtree(backup, ignore_errors=True)
    return candidate_output_dir


def _finalize_expected_fixture(
    fixture_root: Path,
    *,
    candidate: Path,
    run_1: LegacyRun,
    run_2: LegacyRun,
    summary: LegacyImportSummary,
    reviews: Mapping[str, str],
) -> None:
    parent = fixture_root.parent
    temporary = parent / f".{fixture_root.name}.{uuid4().hex}.partial"
    backup = parent / f".{fixture_root.name}.{uuid4().hex}.backup"
    shutil.copytree(fixture_root, temporary)
    try:
        for name in ("clamp_expected", "intermediate_expected"):
            shutil.rmtree(temporary / name, ignore_errors=True)
            shutil.copytree(candidate / name, temporary / name)
        previous = _read_optional_json(temporary / "provenance.json")
        provenance = _complete_fixture_provenance(
            previous,
            run_1=run_1.provenance,
            run_2=run_2.provenance,
            summary=summary,
            reviews=reviews,
        )
        (temporary / "provenance.json").write_text(
            json.dumps(provenance, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        write_sha256s(temporary)
        validate_fixture(temporary, allow_pending=False)
        _replace_generated_path(temporary, fixture_root, backup=backup, overwrite=True)
        shutil.rmtree(backup, ignore_errors=True)
    finally:
        shutil.rmtree(temporary, ignore_errors=True)
        shutil.rmtree(backup, ignore_errors=True)


def _write_expected_tsv(path: Path, document: LegacyDocument) -> None:
    occurrences: dict[tuple[Any, ...], int] = {}
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=EXPECTED_ENTITY_FIELDS,
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        for raw_order, item in enumerate(_entities_in_txt_order(document)):
            start, end, semantic, assertion, cui, attribute, entity_text = item
            base = (start, end, entity_text, semantic, assertion, cui, attribute)
            duplicate_occurrence = occurrences.get(base, 0)
            occurrences[base] = duplicate_occurrence + 1
            writer.writerow(
                {
                    "clamp_doc_id": document.case_id,
                    "start": start,
                    "end": end,
                    "semantic_tag": semantic,
                    "assertion": assertion,
                    "cui": _serialize_nullable(cui),
                    "attribute": _serialize_nullable(attribute),
                    "entity_text": entity_text,
                    "raw_order": raw_order,
                    "duplicate_occurrence": duplicate_occurrence,
                }
            )


def _entities_in_txt_order(
    document: LegacyDocument,
) -> tuple[tuple[int, int, str, str, str | None, str | None, str], ...]:
    """Join XMI-only attributes onto the authoritative tabular exporter order."""

    by_shared_fields: dict[
        tuple[int, int, str, str, str | None, str],
        list[tuple[int, int, str, str, str | None, str | None, str]],
    ] = {}
    for entity in document.entities:
        start, end, semantic, assertion, cui, _attribute, covered = entity
        key = (start, end, semantic, assertion, cui, covered)
        by_shared_fields.setdefault(key, []).append(entity)

    result = []
    for row in document.txt_rows:
        candidates = by_shared_fields.get(row)
        if not candidates:
            raise ValueError(
                "TXT/XMI entity alignment failed after validated multiset match: "
                f"{document.case_id}"
            )
        result.append(candidates.pop(0))
    if any(candidates for candidates in by_shared_fields.values()):
        raise ValueError(f"TXT/XMI entity alignment left unmatched XMI rows: {document.case_id}")
    return tuple(result)


def _write_intermediate_json(
    path: Path,
    document: LegacyDocument,
    *,
    repeated: LegacyDocument,
) -> None:
    payload = {
        "schema_version": 1,
        "case_id": document.case_id,
        "source_text_sha256": document.source_text_sha256,
        "offset_coordinate_system": "utf16_code_units",
        "interval_convention": "half_open",
        "legacy_output_sha256": {
            "run_1": {"txt": document.txt_sha256, "xmi": document.xmi_sha256},
            "run_2": {"txt": repeated.txt_sha256, "xmi": repeated.xmi_sha256},
        },
        "sentences": [
            {
                "start": start,
                "end": end,
                "sentence_number": number,
                "covered_text": covered,
            }
            for start, end, number, covered in document.sentences
        ],
        "tokens": [
            {
                "start": start,
                "end": end,
                "token_number": number,
                "covered_text": covered,
            }
            for start, end, number, covered in document.tokens
        ],
        "final_entities": [
            {
                "start": start,
                "end": end,
                "semantic_tag": semantic,
                "assertion": assertion,
                "cui": cui,
                "attribute": attribute,
                "covered_text": covered,
                "raw_order": raw_order,
            }
            for raw_order, (start, end, semantic, assertion, cui, attribute, covered) in enumerate(
                document.entities
            )
        ],
    }
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _complete_fixture_provenance(
    previous: Mapping[str, Any],
    *,
    run_1: Mapping[str, Any],
    run_2: Mapping[str, Any],
    summary: LegacyImportSummary,
    reviews: Mapping[str, str] | None,
) -> dict[str, Any]:
    payload = json.loads(json.dumps(previous))
    first_runtime = _sanitized_runtime(run_1)
    windows = _require_mapping(run_1, "windows")
    clamp = _require_mapping(run_1, "clamp")
    java = _require_mapping(run_1, "java")
    payload["lifecycle"] = "complete"
    payload["legacy_runtime"] = {
        "clamp_version": str(clamp["version"]),
        "clamp_build": str(clamp["build"]),
        "operating_system": " | ".join(
            str(windows.get(field, ""))
            for field in ("caption", "version", "build_number", "architecture")
        ),
        "java_version": str(java.get("version_output", "")),
        "locale": str(windows.get("locale", "")),
        "timezone": (
            f"{windows.get('timezone', '')} (UTC offset {windows.get('timezone_utc_offset', '')})"
        ),
        "pipeline_export_settings": str(_sanitize_value(run_1["export_settings"])),
    }
    first_record = _sanitized_run_record(run_1)
    second_record = _sanitized_run_record(run_2)
    payload["runs"] = [
        {
            "run_number": 1,
            "started_at": first_record["started_at_utc"],
            "completed_at": first_record["finished_at_utc"],
            "output_manifest_sha256": first_record["output_manifest_sha256"],
        },
        {
            "run_number": 2,
            "started_at": second_record["started_at_utc"],
            "completed_at": second_record["finished_at_utc"],
            "output_manifest_sha256": second_record["output_manifest_sha256"],
        },
    ]
    payload["determinism"] = {
        "status": "passed",
        "required_run_count": 2,
        "raw_order_required": summary.output_order_required,
        "txt_row_order_stable": summary.raw_row_order_stable,
        "exact_sentence_annotations": True,
        "exact_token_annotations": True,
        "exact_entity_multisets": True,
        "xmi_entity_order_stable": summary.xmi_entity_order_stable,
        "txt_order_difference_documents": summary.txt_order_difference_documents,
        "xmi_order_difference_documents": summary.xmi_order_difference_documents,
    }
    payload["legacy_import"] = {
        "generated_only_from_returned_legacy_clamp": True,
        "raw_xmi_committed": False,
        "runtime_details": first_runtime,
        "fixture_counts": {
            "cases": summary.case_count,
            "sentences": summary.sentence_count,
            "tokens": summary.token_count,
            "final_entities": summary.entity_count,
        },
        "run_1": first_record,
        "run_2": second_record,
    }
    existing_reviews = payload.get("reviews")
    if not isinstance(existing_reviews, dict):
        existing_reviews = {}
    phi = existing_reviews.get("phi")
    if not isinstance(phi, dict):
        phi = {"automated_screen": "passed"}
    redistribution = existing_reviews.get("redistribution")
    if not isinstance(redistribution, dict):
        redistribution = {}
    if reviews is None:
        phi.update({"manual_review": "pending", "reviewer": "VERIFY", "reviewed_at": "VERIFY"})
        redistribution.update({"status": "pending", "authority": "VERIFY", "evidence": "VERIFY"})
    else:
        phi.update(
            {
                "manual_review": "approved",
                "reviewer": reviews["phi_reviewer"],
                "reviewed_at": reviews["phi_reviewed_at"],
            }
        )
        redistribution.update(
            {
                "status": "approved",
                "authority": reviews["redistribution_authority"],
                "evidence": reviews["redistribution_evidence"],
            }
        )
    payload["reviews"] = {"phi": phi, "redistribution": redistribution}
    return _sanitize_value(payload)


def _sanitized_runtime(payload: Mapping[str, Any]) -> dict[str, Any]:
    project = _require_mapping(payload, "project")
    project_files = _relative_hash_mapping(
        _require_mapping(project, "files_sha256"),
        field="project.files_sha256",
    )
    resources = _require_mapping(payload, "resources_sha256")
    clean_resources = _relative_hash_mapping(resources, field="resources_sha256")
    return {
        "clamp": _scalar_mapping(_require_mapping(payload, "clamp")),
        "windows": _scalar_mapping(_require_mapping(payload, "windows")),
        "java": _scalar_mapping(_require_mapping(payload, "java")),
        "project_commit": str(project["commit"]),
        "project_files_sha256": project_files,
        "resources_sha256": clean_resources,
        "export_settings": _sanitize_value(payload["export_settings"]),
        "offset_convention": _sanitize_value(payload["offset_convention"]),
        "null_convention": _sanitize_value(payload["null_convention"]),
        "manual_commands": _sanitize_value(payload["manual_commands"]),
    }


def _sanitized_run_record(payload: Mapping[str, Any]) -> dict[str, Any]:
    records = _validated_output_manifest_records(payload)
    digest = hashlib.sha256()
    for record in records:
        relative = str(record["relative_path"])
        sha256 = str(record["sha256"])
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256.encode("ascii"))
        digest.update(b"\n")
    return {
        "run_label": str(payload["run_label"]),
        "started_at_utc": str(payload["started_at_utc"]),
        "finished_at_utc": str(payload["finished_at_utc"]),
        "recorded_at_utc": str(payload["recorded_at_utc"]),
        "output_file_count": len(records),
        "output_manifest_sha256": digest.hexdigest(),
    }


def _load_provenance(path: Path, *, expected_label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Unreadable {expected_label} provenance JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{expected_label} provenance must be a JSON object")
    required = {
        "schema_version",
        "run_label",
        "recorded_at_utc",
        "started_at_utc",
        "finished_at_utc",
        "clamp",
        "windows",
        "java",
        "project",
        "resources_sha256",
        "export_settings",
        "offset_convention",
        "null_convention",
        "manual_commands",
        "output_files",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"{expected_label} provenance is missing fields: {missing}")
    if payload["schema_version"] != 1:
        raise ValueError(f"{expected_label} provenance has an unsupported schema_version")
    if payload["run_label"] != expected_label:
        raise ValueError(
            f"Provenance run_label must be {expected_label}, found {payload['run_label']}"
        )
    times = {
        field: _parse_utc_timestamp(payload[field], field=f"{expected_label}.{field}")
        for field in ("recorded_at_utc", "started_at_utc", "finished_at_utc")
    }
    if not times["started_at_utc"] < times["finished_at_utc"] <= times["recorded_at_utc"]:
        raise ValueError(
            f"{expected_label} provenance timestamps must satisfy started < finished <= recorded"
        )
    clamp = _require_mapping(payload, "clamp")
    version = _required_provenance_text(
        clamp.get("version"), field=f"{expected_label}.clamp.version"
    )
    _required_provenance_text(clamp.get("build"), field=f"{expected_label}.clamp.build")
    if version != _EXPECTED_CLAMP_VERSION:
        raise ValueError(
            f"{expected_label} CLAMP version must be {_EXPECTED_CLAMP_VERSION}, found {version!r}"
        )
    project = _require_mapping(payload, "project")
    _required_provenance_text(project.get("commit"), field=f"{expected_label}.project.commit")
    windows = _require_mapping(payload, "windows")
    for field in (
        "caption",
        "version",
        "build_number",
        "architecture",
        "locale",
        "timezone",
        "timezone_utc_offset",
    ):
        _required_provenance_text(windows.get(field), field=f"{expected_label}.windows.{field}")
    java = _require_mapping(payload, "java")
    _required_provenance_text(
        java.get("version_output"), field=f"{expected_label}.java.version_output"
    )
    for field in ("export_settings", "offset_convention", "null_convention"):
        _required_provenance_text(payload[field], field=f"{expected_label}.{field}")
    commands = payload["manual_commands"]
    if (
        not isinstance(commands, list)
        or not commands
        or any(not _is_nonplaceholder_text(command) for command in commands)
    ):
        raise ValueError(f"{expected_label} provenance must record manual_commands")
    if any(_LOCAL_PATH_VALUE_RE.search(str(command)) for command in commands):
        raise ValueError(
            f"{expected_label} manual_commands must describe the exact actions without "
            "machine-local paths"
        )
    _relative_hash_mapping(_require_mapping(payload, "resources_sha256"), field="resources_sha256")
    _validated_output_manifest_records(payload)
    return payload


def _validate_provenance_output_manifest(
    provenance: Mapping[str, Any], inventory: RunInventory, *, label: str
) -> None:
    records = _validated_output_manifest_records(provenance)
    declared: dict[tuple[str, str], dict[str, Any]] = {}
    for record in records:
        relative = str(record["relative_path"])
        suffix = PurePosixPath(relative).suffix.lower()
        if suffix not in {".txt", ".xmi"}:
            raise ValueError(f"{label} provenance manifest contains a non-output file: {relative}")
        key = (_output_case_id(PurePosixPath(relative).name, suffix=suffix), suffix)
        if key in declared:
            raise ValueError(f"{label} provenance contains duplicate output case/type: {key}")
        declared[key] = record
    actual_entries = [*inventory.txt.values(), *inventory.xmi.values()]
    actual: dict[tuple[str, str], OutputPayload] = {}
    for item in actual_entries:
        suffix = PurePosixPath(item.relative_path).suffix.lower()
        key = (_output_case_id(PurePosixPath(item.relative_path).name, suffix=suffix), suffix)
        actual[key] = item
    if set(declared) != set(actual):
        raise ValueError(
            f"{label} provenance output manifest does not match returned run files: "
            f"missing={sorted(set(actual) - set(declared))[:5]}, "
            f"extra={sorted(set(declared) - set(actual))[:5]}"
        )
    for key, item in actual.items():
        record = declared[key]
        if int(record["bytes"]) != len(item.payload) or record["sha256"] != item.sha256:
            raise ValueError(f"{label} output checksum mismatch: {key[0]}{key[1]}")


def _validated_output_manifest_records(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw = payload.get("output_files")
    if not isinstance(raw, list) or not raw:
        raise ValueError("Provenance output_files must be a non-empty array")
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in raw:
        if not isinstance(item, dict):
            raise ValueError("Provenance output_files entries must be objects")
        relative = str(item.get("relative_path", "")).replace("\\", "/")
        path = PurePosixPath(relative)
        if not relative or path.is_absolute() or ".." in path.parts:
            raise ValueError(f"Unsafe provenance output path: {relative}")
        if relative in seen:
            raise ValueError(f"Duplicate provenance output path: {relative}")
        seen.add(relative)
        sha256 = str(item.get("sha256", "")).lower()
        if not _SHA256_RE.fullmatch(sha256):
            raise ValueError(f"Invalid provenance output SHA-256: {relative}")
        try:
            byte_count = int(item.get("bytes"))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid provenance output byte count: {relative}") from exc
        if byte_count < 0:
            raise ValueError(f"Negative provenance output byte count: {relative}")
        result.append({"relative_path": relative, "bytes": byte_count, "sha256": sha256})
    return sorted(result, key=lambda item: item["relative_path"])


def _validate_returned_checksum_manifest(
    path: Path,
    provenance: Mapping[str, Any],
    *,
    label: str,
) -> None:
    try:
        lines = path.read_text(encoding="utf-8-sig").splitlines()
    except OSError as exc:
        raise ValueError(f"Missing or unreadable {label} SHA256SUMS manifest: {path}") from exc
    declared: dict[str, str] = {}
    for line_number, line in enumerate(lines, start=1):
        if not line:
            continue
        match = re.fullmatch(r"([0-9A-Fa-f]{64})  (.+)", line)
        if match is None:
            raise ValueError(f"Malformed {label} SHA256SUMS row {line_number}: {path}")
        digest, raw_relative = match.groups()
        relative = raw_relative.replace("\\", "/")
        member = PurePosixPath(relative)
        if not relative or member.is_absolute() or ".." in member.parts:
            raise ValueError(f"Unsafe {label} SHA256SUMS path: {relative}")
        if relative in declared:
            raise ValueError(f"Duplicate {label} SHA256SUMS path: {relative}")
        declared[relative] = digest.lower()
    expected = {
        str(record["relative_path"]): str(record["sha256"])
        for record in _validated_output_manifest_records(provenance)
    }
    if declared != expected:
        missing = sorted(set(expected) - set(declared))
        extra = sorted(set(declared) - set(expected))
        changed = sorted(
            relative
            for relative in set(expected) & set(declared)
            if expected[relative] != declared[relative]
        )
        raise ValueError(
            f"{label} SHA256SUMS differs from provenance: missing={missing[:5]}, "
            f"extra={extra[:5]}, changed={changed[:5]}"
        )


def _validate_runtime_repeatability(first: Mapping[str, Any], second: Mapping[str, Any]) -> None:
    fields = (
        "clamp",
        "windows",
        "java",
        "resources_sha256",
        "export_settings",
        "offset_convention",
        "null_convention",
        "manual_commands",
    )
    differences = [field for field in fields if first[field] != second[field]]
    first_project = _require_mapping(first, "project")
    second_project = _require_mapping(second, "project")
    for field in ("commit", "files_sha256"):
        if first_project.get(field) != second_project.get(field):
            differences.append(f"project.{field}")
    if differences:
        raise ValueError(
            "Legacy CLAMP runtime provenance differs between runs: " + ", ".join(differences)
        )


def _validate_run_chronology(first: Mapping[str, Any], second: Mapping[str, Any]) -> None:
    first_finished = _parse_utc_timestamp(first["finished_at_utc"], field="run_1.finished_at_utc")
    second_started = _parse_utc_timestamp(second["started_at_utc"], field="run_2.started_at_utc")
    if first_finished > second_started:
        raise ValueError("Legacy CLAMP run_2 must start after run_1 has finished")


def _load_fixture_frozen_contract(
    fixture_root: Path,
    *,
    resource_manifest: Path | None,
) -> dict[str, Any]:
    manifest_path = (resource_manifest or default_resource_manifest_path()).resolve()
    try:
        manifest_bytes = manifest_path.read_bytes()
        manifest = json.loads(manifest_bytes)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Unreadable frozen CLAMP resource manifest: {manifest_path}") from exc
    if not isinstance(manifest, dict) or not isinstance(manifest.get("files"), dict):
        raise ValueError("Frozen CLAMP resource manifest must contain a files mapping")
    files = _relative_hash_mapping(manifest["files"], field="resource_manifest.files")
    project_commit = str(manifest.get("project_commit", "")).strip()
    if not project_commit:
        raise ValueError("Frozen CLAMP resource manifest lacks project_commit")

    fixture_provenance = _read_optional_json(fixture_root / "provenance.json")
    generator = fixture_provenance.get("generator")
    if not isinstance(generator, dict):
        raise ValueError("Fixture provenance lacks generator metadata")
    fixture_commit = str(generator.get("reference_project_commit", "")).strip()
    if fixture_commit != project_commit:
        raise ValueError(
            "Fixture reference project commit differs from the frozen resource manifest: "
            f"{fixture_commit} != {project_commit}"
        )
    manifest_sha256 = _bytes_sha256(manifest_bytes)
    compatible_manifest_hashes = manifest.get("compatible_manifest_sha256", [])
    if not isinstance(compatible_manifest_hashes, list):
        raise ValueError("Frozen CLAMP resource manifest compatibility hashes are invalid")
    accepted_manifest_hashes = {
        manifest_sha256,
        *(str(value) for value in compatible_manifest_hashes),
    }
    fixture_manifest_sha256 = str(generator.get("resource_manifest_sha256", ""))
    if fixture_manifest_sha256 not in accepted_manifest_hashes:
        raise ValueError("Fixture provenance resource-manifest SHA-256 is stale")
    resources = {key: value for key, value in files.items() if key.startswith("Components/")}
    if not resources:
        raise ValueError("Frozen resource manifest contains no Components resources")
    return {
        "manifest_path": str(manifest_path),
        "manifest_sha256": fixture_manifest_sha256,
        "project_commit": project_commit,
        "project_files_sha256": files,
        "resources_sha256": resources,
    }


def _validate_returned_project_contract(
    provenance: Mapping[str, Any],
    frozen: Mapping[str, Any],
    *,
    label: str,
) -> None:
    project = _require_mapping(provenance, "project")
    if str(project.get("commit", "")) != frozen["project_commit"]:
        raise ValueError(f"{label} project commit differs from the frozen fixture contract")
    project_hashes = _relative_hash_mapping(
        _require_mapping(project, "files_sha256"),
        field=f"{label}.project.files_sha256",
    )
    if project_hashes != frozen["project_files_sha256"]:
        raise ValueError(f"{label} project file hashes differ from the frozen handoff contract")
    resource_hashes = _relative_hash_mapping(
        _require_mapping(provenance, "resources_sha256"),
        field=f"{label}.resources_sha256",
    )
    if resource_hashes != frozen["resources_sha256"]:
        raise ValueError(f"{label} resource hashes differ from the frozen handoff contract")


def _validate_frozen_project(
    project_dir: Path,
    *,
    resource_manifest: Path | None,
    project_commit: str | None,
) -> dict[str, Any]:
    actual = {
        path.relative_to(project_dir).as_posix(): _file_sha256(path)
        for path in _iter_project_files(project_dir)
    }
    if resource_manifest is None:
        if project_commit is None:
            project_commit = _git_head(project_dir)
        return {"project_commit": project_commit, "files": actual}
    try:
        manifest = json.loads(resource_manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Unreadable frozen project manifest: {resource_manifest}") from exc
    expected = manifest.get("files")
    if not isinstance(expected, dict):
        raise ValueError("Frozen project manifest must contain a files mapping")
    expected_hashes = {str(key): str(value).lower() for key, value in expected.items()}
    if actual != expected_hashes:
        missing = sorted(set(expected_hashes) - set(actual))
        extra = sorted(set(actual) - set(expected_hashes))
        changed = sorted(
            key for key in set(actual) & set(expected_hashes) if actual[key] != expected_hashes[key]
        )
        raise ValueError(
            "Frozen CLAMP project differs from its manifest: "
            f"missing={missing[:5]}, extra={extra[:5]}, changed={changed[:5]}"
        )
    frozen_commit = str(manifest.get("project_commit") or project_commit or "").strip()
    if not frozen_commit:
        raise ValueError("Frozen project commit is missing")
    if project_commit is not None and project_commit != frozen_commit:
        raise ValueError(
            f"Requested project commit {project_commit} differs from manifest {frozen_commit}"
        )
    return {"project_commit": frozen_commit, "files": actual}


def _copy_frozen_project(source: Path, destination: Path) -> None:
    for path in _iter_project_files(source):
        relative = path.relative_to(source)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.read_bytes())


def _iter_project_files(project_dir: Path) -> Iterable[Path]:
    for path in sorted(project_dir.rglob("*")):
        if not path.is_file() or path.is_symlink() or path.name == ".DS_Store":
            continue
        relative = path.relative_to(project_dir)
        lowered = tuple(part.lower() for part in relative.parts)
        if "archive" in lowered:
            continue
        adjacent = set(zip(lowered, lowered[1:], strict=False))
        if adjacent & {("data", "input"), ("data", "output")}:
            continue
        yield path


def _read_directory_output(root: Path) -> list[OutputPayload]:
    result: list[OutputPayload] = []
    total_bytes = 0
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"Symlinks are not allowed in returned CLAMP output: {path}")
        if not path.is_file() or path.suffix.lower() not in {".txt", ".xmi"}:
            continue
        if path.stat().st_size > MAX_OUTPUT_MEMBER_BYTES:
            raise ValueError(f"Returned CLAMP output member is unexpectedly large: {path}")
        total_bytes += path.stat().st_size
        if total_bytes > MAX_RUN_PAYLOAD_BYTES:
            raise ValueError("Returned CLAMP output exceeds the synthetic-run size limit")
        result.append(
            OutputPayload(
                relative_path=path.relative_to(root).as_posix(),
                payload=path.read_bytes(),
            )
        )
    return result


def _read_zip_output(path: Path) -> list[OutputPayload]:
    result: list[OutputPayload] = []
    total_bytes = 0
    with zipfile.ZipFile(path) as archive:
        for info in sorted(archive.infolist(), key=lambda item: item.filename):
            if info.is_dir():
                continue
            relative = info.filename.replace("\\", "/")
            member = PurePosixPath(relative)
            if member.is_absolute() or ".." in member.parts:
                raise ValueError(f"Unsafe path in returned CLAMP ZIP: {info.filename}")
            if info.flag_bits & 0x1:
                raise ValueError(f"Encrypted returned CLAMP ZIP member: {info.filename}")
            if member.suffix.lower() not in {".txt", ".xmi"}:
                continue
            if info.file_size > MAX_OUTPUT_MEMBER_BYTES:
                raise ValueError(
                    f"Returned CLAMP ZIP member is unexpectedly large: {info.filename}"
                )
            total_bytes += info.file_size
            if total_bytes > MAX_RUN_PAYLOAD_BYTES:
                raise ValueError("Returned CLAMP ZIP exceeds the synthetic-run size limit")
            result.append(OutputPayload(relative_path=relative, payload=archive.read(info)))
    return result


def _utf16_slice(text: str, start: int, end: int) -> str:
    if start < 0 or end < start:
        raise ValueError(f"invalid half-open span [{start}, {end})")
    boundaries: dict[int, int] = {0: 0}
    offset = 0
    for index, character in enumerate(text, start=1):
        offset += 2 if ord(character) > 0xFFFF else 1
        boundaries[offset] = index
    if start not in boundaries or end not in boundaries:
        raise ValueError(
            f"UTF-16 span [{start}, {end}) does not align to source boundaries 0..{offset}"
        )
    return text[boundaries[start] : boundaries[end]]


def _write_sha256sums(root: Path) -> None:
    destination = root / "SHA256SUMS"
    rows = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path == destination:
            continue
        rows.append(f"{_file_sha256(path)}  {path.relative_to(root).as_posix()}")
    destination.write_text("\n".join(rows) + "\n", encoding="utf-8", newline="\n")


def _write_deterministic_zip(root: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(
        destination,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
    ) as archive:
        for path in sorted(item for item in root.rglob("*") if item.is_file()):
            info = zipfile.ZipInfo(path.relative_to(root).as_posix(), date_time=FIXED_ZIP_TIMESTAMP)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, path.read_bytes(), compresslevel=9)


def _render_handoff_runbook(
    *,
    case_count: int,
    project_commit: str,
    expected_windows_project_dir: str,
    collector_name: str,
) -> str:
    return f"""# ARDS CLAMP synthetic parity handoff

This packet contains {case_count:,} synthetic input files and no restricted MIMIC corpus.

1. Confirm the licensed CLAMP application reports its exact version and build.
2. Place `ARDS` at `{expected_windows_project_dir}`. Do not edit the project or inputs.
3. Confirm the frozen project commit recorded by this packet is `{project_commit}`.
4. Clear `ARDS\\Data\\Output`, record the start time, and run the ARDS pipeline once.
5. Copy the clean output directory to `Returned\\run_1` without renaming files.
6. Run `{collector_name}` for `run_1`, entering the exact CLAMP build, export settings,
   offset/null conventions, timestamps, and path-free manual action description. Do not include a
   Windows username or machine-local path in `ManualCommand`.
7. Clear `ARDS\\Data\\Output` again and repeat steps 4-6 into `Returned\\run_2`.
8. Return each run directory (or a ZIP of it), both provenance JSON files, and the two
   generated SHA256SUMS files. Do not mix files from the two runs.

Example collector invocation for the first run (replace every quoted placeholder with the exact
value observed for that run):

```powershell
powershell -ExecutionPolicy Bypass -File .\\{collector_name} `
  -RunLabel "run_1" `
  -ClampVersion "1.6.6" `
  -ClampBuild "<application build>" `
  -ProjectCommit "{project_commit}" `
  -ProjectDir "{expected_windows_project_dir}" `
  -OutputDir "C:\\Returned\\run_1" `
  -ProvenanceOutput "C:\\Returned\\run_1_provenance.json" `
  -ExportSettings "<exact TXT and XMI export settings>" `
  -OffsetConvention "half-open UTF-16 code units" `
  -NullConvention "<exact observed TXT/XMI null convention>" `
  -ManualCommand "<exact CLAMP GUI actions or command>" `
  -StartedAtUtc "<ISO-8601 UTC>" `
  -FinishedAtUtc "<ISO-8601 UTC>"
```

The import is intentionally strict: missing, extra, duplicate, unparsable, source-mismatched,
offset-invalid, or semantically run-different outputs are rejected without changing the fixture.
TXT/XMI row ordering is measured separately and becomes required only when the repeated runs prove
it stable.
"""


def _replace_generated_path(
    source: Path,
    destination: Path,
    *,
    backup: Path,
    overwrite: bool,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    moved_existing = False
    try:
        if destination.exists():
            if not overwrite:
                raise FileExistsError(f"Generated destination already exists: {destination}")
            os.replace(destination, backup)
            moved_existing = True
        os.replace(source, destination)
    except BaseException:
        if moved_existing and backup.exists() and not destination.exists():
            os.replace(backup, destination)
        raise


def _restore_backup(destination: Path, backup: Path) -> None:
    if backup.exists():
        if destination.is_dir():
            shutil.rmtree(destination)
        else:
            destination.unlink(missing_ok=True)
        os.replace(backup, destination)


def _manifest_column(fieldnames: list[str], *candidates: str, required: bool = True) -> str | None:
    normalized = {field.strip().lower(): field for field in fieldnames}
    for candidate in candidates:
        if candidate in normalized:
            return normalized[candidate]
    if required:
        raise ValueError(f"Fixture manifest is missing one of columns: {list(candidates)}")
    return None


def _output_case_id(name: str, *, suffix: str) -> str:
    case_id = name[: -len(suffix)]
    if case_id.lower().endswith(".txt"):
        case_id = case_id[:-4]
    case_id = case_id.strip()
    if not case_id:
        raise ValueError(f"Returned CLAMP output has a blank case ID: {name}")
    return case_id


def _relative_hash_mapping(mapping: Mapping[str, Any], *, field: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for raw_key, raw_value in mapping.items():
        key = str(raw_key).replace("\\", "/")
        path = PurePosixPath(key)
        value = str(raw_value).lower()
        if not key or path.is_absolute() or ".." in path.parts:
            raise ValueError(f"Unsafe relative path in {field}: {key}")
        if not _SHA256_RE.fullmatch(value):
            raise ValueError(f"Invalid SHA-256 in {field}: {key}")
        result[key] = value
    if not result:
        raise ValueError(f"{field} must not be empty")
    return dict(sorted(result.items()))


def _require_mapping(payload: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"Provenance field {key} must be an object")
    return value


def _scalar_mapping(payload: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in payload.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            result[str(key)] = _sanitize_value(value)
    return dict(sorted(result.items()))


def _parse_utc_timestamp(value: Any, *, field: str) -> datetime:
    text = str(value).strip()
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{field} must be a valid ISO-8601 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != UTC.utcoffset(parsed):
        raise ValueError(f"{field} must include a UTC offset")
    return parsed


def _is_nonplaceholder_text(value: Any) -> bool:
    text = str(value or "").strip()
    return (
        bool(text)
        and not re.search(r"<[^>]+>", text)
        and text.upper()
        not in {
            "VERIFY",
            "TODO",
            "PENDING",
            "UNKNOWN",
            "NOASSERTION",
            "<APPLICATION BUILD>",
        }
    )


def _required_provenance_text(value: Any, *, field: str) -> str:
    text = str(value or "").strip()
    if not _is_nonplaceholder_text(text):
        raise ValueError(f"Provenance field {field} must be nonblank and non-placeholder")
    return text


def _sanitize_value(value: Any) -> Any:
    if isinstance(value, str):
        if _LOCAL_PATH_VALUE_RE.search(value):
            return "<redacted-local-path-value>"
        return value
    if isinstance(value, list):
        return [_sanitize_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _sanitize_value(item) for key, item in value.items()}
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    return str(value)


def _serialize_nullable(value: str | None) -> str:
    return r"\N" if value is None else value


def _required_review_value(value: str | None, field: str) -> str:
    clean = str(value or "").strip()
    if not clean or clean.upper() in {"VERIFY", "TODO", "PENDING"}:
        raise ValueError(f"--finalize requires non-placeholder {field}")
    if _LOCAL_PATH_VALUE_RE.search(clean):
        raise ValueError(f"--finalize {field} must not contain a machine-local path")
    sanitized = str(_sanitize_value(clean))
    if "<redacted-path>" in sanitized:
        raise ValueError(f"{field} must be a Git-safe reference, not an absolute local path")
    return sanitized


def _required_review_timestamp(value: str | None) -> str:
    clean = _required_review_value(value, "phi_reviewed_at")
    try:
        date.fromisoformat(clean)
    except ValueError:
        _parse_utc_timestamp(clean, field="phi_reviewed_at")
    return clean


def _nullable(value: Any) -> str | None:
    if value is None:
        return None
    clean = str(value).strip()
    return None if not clean or clean.lower() == "null" else clean


def _read_optional_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Existing fixture provenance is invalid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError("Existing fixture provenance must be a JSON object")
    return payload


def _require_existing_directory(path: Path, label: str) -> None:
    if not path.is_dir():
        raise FileNotFoundError(f"{label} not found: {path}")


def _require_disjoint_generated_path(generated: Path, protected: Path) -> None:
    if (
        generated == protected
        or generated.is_relative_to(protected)
        or protected.is_relative_to(generated)
    ):
        raise ValueError(
            f"Generated path must not overlap protected source path: {generated} and {protected}"
        )


def _git_head(path: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=path,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError("project_commit is required outside a Git checkout") from exc
    return result.stdout.strip()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bytes_sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()
