from __future__ import annotations

import json
import re
import shutil
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from string import Formatter
from typing import Any

import pandas as pd

from .config import ensure_dir, ensure_parent_dir

WINDOWS_INVALID_FILENAME_CHARS_RE = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
WINDOWS_RESERVED_FILENAME_STEMS = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{idx}" for idx in range(1, 10)),
    *(f"LPT{idx}" for idx in range(1, 10)),
}
WORD_RE = re.compile(r"\b\w+\b")
SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+|\n+")
TEXT_CHARS_FLAG = 16_000
SENTENCE_WORDS_FLAG = 500
OUTPUT_EXTENSIONS = (".txt", ".xdi", ".xmi")
INPUT_EXTENSIONS = (".txt",)
OPTIONAL_IDENTIFIER_COLUMNS = [
    "source_dataset",
    "subject_id",
    "study_id",
    "accession_id",
    "encounter_id",
    "annotation_phase",
]
MANIFEST_COLUMNS = [
    "clamp_doc_id",
    "input_file",
    "source_dataset",
    "subject_id",
    "study_id",
    "accession_id",
    "encounter_id",
    "annotation_phase",
    "source_row_index",
    "text_chars",
    "text_words",
    "max_sentence_like_words",
    "chars_gt_16000",
    "sentence_words_gt_500",
    "export_status",
    "skip_reason",
]


@dataclass(frozen=True)
class ClampInputExportResult:
    manifest: pd.DataFrame
    summary: dict[str, Any]
    handoff_markdown: str


@dataclass(frozen=True)
class ClampProjectSyncOperation:
    destination: Path
    payload: bytes


def invalid_windows_filename_stem_reason(value: object) -> str | None:
    if value is None or pd.isna(value):
        return "missing"
    name = str(value)
    if not name:
        return "empty"
    if name != name.strip():
        return "leading_or_trailing_whitespace"
    if name in {".", ".."}:
        return "dot_path_component"
    if name.endswith("."):
        return "trailing_dot"
    if WINDOWS_INVALID_FILENAME_CHARS_RE.search(name):
        return "windows_invalid_character"
    reserved_token = name.split(".", 1)[0].upper()
    if reserved_token in WINDOWS_RESERVED_FILENAME_STEMS:
        return "windows_reserved_device_name"
    return None


def text_length_metrics(text: str) -> dict[str, int]:
    segments = [segment for segment in SENTENCE_BOUNDARY_RE.split(text) if segment.strip()]
    if text.strip() and not segments:
        segments = [text]
    sentence_counts = [len(WORD_RE.findall(segment)) for segment in segments]
    return {
        "text_chars": len(text),
        "text_words": len(WORD_RE.findall(text)),
        "max_sentence_like_words": max(sentence_counts, default=0),
    }


def export_clamp_ards_inputs(
    df: pd.DataFrame,
    *,
    input_dir: Path,
    output_dir: Path,
    artifact_dir: Path,
    manifest_path: Path,
    summary_path: Path,
    handoff_path: Path,
    text_col: str,
    source_type: str,
    source_name: str,
    command: str,
    project_live_dir: Path,
    id_col: str | None = None,
    doc_id_template: str | None = None,
    overwrite: bool = False,
    clear_existing_inputs: bool = False,
    clear_existing_outputs: bool = False,
    archive_cleared_files: bool = False,
    dry_run: bool = False,
    run_timestamp: str | None = None,
) -> ClampInputExportResult:
    if text_col not in df.columns:
        raise ValueError(f"Missing text column: {text_col}")
    if id_col is not None and id_col not in df.columns:
        raise ValueError(f"Missing id column: {id_col}")

    run_timestamp = run_timestamp or datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    input_dir = input_dir.resolve()
    output_dir = output_dir.resolve()
    artifact_dir = artifact_dir.resolve()
    manifest_path = manifest_path.resolve()
    summary_path = summary_path.resolve()
    handoff_path = handoff_path.resolve()
    project_live_dir = project_live_dir.resolve()

    manifest, text_by_doc_id = _build_manifest_and_texts(
        df,
        input_dir=input_dir,
        text_col=text_col,
        id_col=id_col,
        doc_id_template=doc_id_template,
    )
    duplicate_doc_ids = _candidate_duplicate_doc_ids(manifest)
    if duplicate_doc_ids:
        manifest.loc[
            manifest["clamp_doc_id"].isin(duplicate_doc_ids)
            & (manifest["export_status"] == "candidate"),
            ["export_status", "skip_reason"],
        ] = ["skipped", "duplicate_doc_id"]

    summary = summarize_manifest(
        manifest,
        source_type=source_type,
        input_dir=input_dir,
        output_dir=output_dir,
    )
    handoff = render_handoff_markdown(
        project_live_dir=project_live_dir,
        input_dir=input_dir,
        output_dir=output_dir,
        manifest_path=manifest_path,
        run_timestamp=run_timestamp,
        source_name=source_name,
        command=command,
        expected_input_file_count=summary["written_files"],
    )

    ensure_dir(artifact_dir)
    write_manifest_and_summaries(
        manifest,
        summary,
        handoff,
        manifest_path=manifest_path,
        summary_path=summary_path,
        handoff_path=handoff_path,
    )

    if duplicate_doc_ids:
        raise ValueError(f"Duplicate clamp_doc_id values: {duplicate_doc_ids}")
    if summary["written_files"] == 0:
        raise ValueError("No CLAMP input files would be written")

    if dry_run:
        return ClampInputExportResult(manifest=manifest, summary=summary, handoff_markdown=handoff)

    _prepare_clamp_dirs(
        input_dir=input_dir,
        output_dir=output_dir,
        clear_existing_inputs=clear_existing_inputs,
        clear_existing_outputs=clear_existing_outputs,
        archive_cleared_files=archive_cleared_files,
    )
    existing = [
        Path(path)
        for path in manifest.loc[manifest["export_status"] == "candidate", "input_file"].tolist()
        if Path(path).exists()
    ]
    if existing and not overwrite:
        raise FileExistsError(
            "CLAMP input files already exist. Re-run with --overwrite or "
            f"--clear-existing-inputs: {existing[:5]}"
        )

    for doc_id, text in text_by_doc_id.items():
        path = _safe_input_file(input_dir, doc_id)
        ensure_parent_dir(path)
        path.write_text(text, encoding="utf-8", newline="")

    manifest.loc[manifest["export_status"] == "candidate", "export_status"] = "written"
    summary = summarize_manifest(
        manifest,
        source_type=source_type,
        input_dir=input_dir,
        output_dir=output_dir,
    )
    handoff = render_handoff_markdown(
        project_live_dir=project_live_dir,
        input_dir=input_dir,
        output_dir=output_dir,
        manifest_path=manifest_path,
        run_timestamp=run_timestamp,
        source_name=source_name,
        command=command,
        expected_input_file_count=summary["written_files"],
    )
    write_manifest_and_summaries(
        manifest,
        summary,
        handoff,
        manifest_path=manifest_path,
        summary_path=summary_path,
        handoff_path=handoff_path,
    )
    return ClampInputExportResult(manifest=manifest, summary=summary, handoff_markdown=handoff)


def summarize_manifest(
    manifest: pd.DataFrame,
    *,
    source_type: str,
    input_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    written_like = manifest["export_status"].isin(["candidate", "written"])
    summary = {
        "source_type": source_type,
        "source_rows": int(len(manifest)),
        "written_files": int(written_like.sum()),
        "skipped_rows": int((manifest["export_status"] == "skipped").sum()),
        "missing_text_rows": int((manifest["skip_reason"] == "missing_text").sum()),
        "duplicate_doc_id_rows": int((manifest["skip_reason"] == "duplicate_doc_id").sum()),
        "unsafe_doc_id_rows": int(
            manifest["skip_reason"].fillna("").str.startswith("unsafe_doc_id:").sum()
        ),
        "chars_gt_16000": int(manifest["chars_gt_16000"].fillna(False).sum()),
        "sentence_words_gt_500": int(manifest["sentence_words_gt_500"].fillna(False).sum()),
        "max_text_chars": _max_int(manifest["text_chars"]),
        "max_text_words": _max_int(manifest["text_words"]),
        "max_sentence_like_words": _max_int(manifest["max_sentence_like_words"]),
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
    }
    return summary


def write_manifest_and_summaries(
    manifest: pd.DataFrame,
    summary: dict[str, Any],
    handoff_markdown: str,
    *,
    manifest_path: Path,
    summary_path: Path,
    handoff_path: Path,
) -> None:
    ensure_parent_dir(manifest_path)
    ensure_parent_dir(summary_path)
    ensure_parent_dir(handoff_path)
    manifest.reindex(columns=MANIFEST_COLUMNS).to_csv(manifest_path, index=False)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary_path.with_suffix(".md").write_text(render_summary_markdown(summary), encoding="utf-8")
    handoff_path.write_text(handoff_markdown, encoding="utf-8")


def render_summary_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# ARDS CLAMP input summary",
        "",
        f"- Source type: {summary['source_type']}",
        f"- Source rows: {summary['source_rows']:,}",
        f"- Expected/written files: {summary['written_files']:,}",
        f"- Skipped rows: {summary['skipped_rows']:,}",
        f"- Missing text rows: {summary['missing_text_rows']:,}",
        f"- Duplicate doc-id rows: {summary['duplicate_doc_id_rows']:,}",
        f"- Unsafe doc-id rows: {summary['unsafe_doc_id_rows']:,}",
        f"- Texts >16,000 chars: {summary['chars_gt_16000']:,}",
        f"- Sentence-like segments >500 words: {summary['sentence_words_gt_500']:,}",
        f"- Input directory: `{summary['input_dir']}`",
        f"- Output directory: `{summary['output_dir']}`",
        "",
    ]
    return "\n".join(lines)


def render_handoff_markdown(
    *,
    project_live_dir: Path,
    input_dir: Path,
    output_dir: Path,
    manifest_path: Path,
    run_timestamp: str,
    source_name: str,
    command: str,
    expected_input_file_count: int,
) -> str:
    lines = [
        "# Next Step: Run ARDS CLAMP",
        "",
        "Stage 1 has prepared CLAMP input files only. Run CLAMP manually on the CLAMP",
        "machine, then return the CLAMP output files to this MIMIC-enabled machine for",
        "future Stage 2 parsing and merging.",
        "",
        f"- Run timestamp: `{run_timestamp}`",
        f"- Live CLAMP project path: `{project_live_dir}`",
        f"- CLAMP input directory: `{input_dir}`",
        f"- Expected input file count: {expected_input_file_count:,}",
        f"- CLAMP output directory: `{output_dir}`",
        f"- Input manifest: `{manifest_path}`",
        f"- Source: `{source_name}`",
        "",
        "## Command Used",
        "",
        "```bash",
        command,
        "```",
        "",
        "## Transfer-First Workflow",
        "",
        "1. Transfer the ARDS CLAMP project and the `.txt` input files to the CLAMP machine.",
        "2. Do not transfer the full BigQuery/model-development dataset to the CLAMP machine.",
        "3. Run CLAMP on the CLAMP machine and write outputs to the output directory above.",
        "4. Transfer the CLAMP output files back to this machine.",
        "5. Merge returned outputs here using `input_manifest.csv` and `clamp_doc_id`.",
        "",
        "The `.txt` input files and CLAMP outputs contain report-derived restricted content.",
        "",
    ]
    return "\n".join(lines)


def clear_or_archive_files(
    dir_path: Path,
    *,
    extensions: tuple[str, ...],
    archive_cleared_files: bool,
    archive_timestamp: str | None = None,
) -> dict[str, Any]:
    dir_path.mkdir(parents=True, exist_ok=True)
    archive_dir = None
    removed = 0
    if archive_cleared_files:
        archive_timestamp = archive_timestamp or datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        archive_dir = dir_path / "Archive" / archive_timestamp
        archive_dir.mkdir(parents=True, exist_ok=True)

    for child in sorted(dir_path.iterdir()):
        if not child.is_file() or child.suffix.lower() not in extensions:
            continue
        if archive_dir is None:
            child.unlink()
        else:
            shutil.move(str(child), str(_unique_target_path(archive_dir / child.name)))
        removed += 1
    return {
        "removed_files": removed,
        "archive_dir": None if archive_dir is None else str(archive_dir),
    }


def sync_clamp_ards_project(
    *,
    source_dir: Path,
    live_dir: Path,
    runtime_project_dir: str | Path,
    artifact_dir: Path,
    summary_path: Path,
    overwrite: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    if not source_dir.exists():
        raise FileNotFoundError(f"CLAMP source project not found: {source_dir}")
    source_dir = source_dir.resolve()
    live_dir = live_dir.resolve()
    summary_path = summary_path.resolve()

    rendered_files: list[str] = []
    unchanged_files: list[str] = []
    skipped_files: list[str] = []
    conflicts: list[str] = []
    planned_operations: list[ClampProjectSyncOperation] = []
    for source_file in _iter_clamp_project_files(source_dir):
        rel_path = source_file.relative_to(source_dir)
        destination = live_dir / rel_path
        payload = source_file.read_bytes()
        if source_file.suffix.lower() in {".xml", ".conf", ".pipeline", ".project"}:
            payload = _normalize_clamp_descriptor_paths(
                source_file.read_text(encoding="utf-8"),
                runtime_project_dir=runtime_project_dir,
            ).encode("utf-8")
        if dry_run:
            rendered_files.append(str(destination))
            continue
        if destination.exists() and destination.read_bytes() != payload and not overwrite:
            conflicts.append(str(destination))
            continue
        if destination.exists() and destination.read_bytes() == payload:
            unchanged_files.append(str(destination))
        else:
            planned_operations.append(
                ClampProjectSyncOperation(destination=destination, payload=payload)
            )

    if conflicts:
        raise FileExistsError(
            "Destination files differ and --overwrite was not set: " + ", ".join(conflicts[:5])
        )

    for operation in planned_operations:
        operation.destination.parent.mkdir(parents=True, exist_ok=True)
        operation.destination.write_bytes(operation.payload)
        rendered_files.append(str(operation.destination))

    ensure_dir(artifact_dir)
    summary = {
        "source_dir": str(source_dir),
        "live_dir": str(live_dir),
        "runtime_project_dir": str(runtime_project_dir),
        "overwrite": overwrite,
        "dry_run": dry_run,
        "rendered_file_count": len(rendered_files),
        "unchanged_file_count": len(unchanged_files),
        "skipped_file_count": len(skipped_files),
        "rendered_files": rendered_files,
        "unchanged_files": unchanged_files,
        "skipped_files": skipped_files,
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def _build_manifest_and_texts(
    df: pd.DataFrame,
    *,
    input_dir: Path,
    text_col: str,
    id_col: str | None,
    doc_id_template: str | None,
) -> tuple[pd.DataFrame, dict[str, str]]:
    records: list[dict[str, Any]] = []
    text_by_doc_id: dict[str, str] = {}
    for source_row_index, row in df.reset_index(drop=True).iterrows():
        record = {column: None for column in MANIFEST_COLUMNS}
        record["source_row_index"] = int(source_row_index)
        for column in OPTIONAL_IDENTIFIER_COLUMNS:
            if column in df.columns:
                record[column] = row.get(column)

        raw_text = row.get(text_col)
        if raw_text is None or pd.isna(raw_text) or str(raw_text) == "":
            record["export_status"] = "skipped"
            record["skip_reason"] = "missing_text"
            records.append(record)
            continue
        report_text = str(raw_text)
        try:
            doc_id = _render_doc_id(row, id_col=id_col, doc_id_template=doc_id_template)
        except KeyError as exc:
            raise ValueError(f"Cannot render clamp_doc_id; missing column {exc}") from exc
        reason = invalid_windows_filename_stem_reason(doc_id)
        record["clamp_doc_id"] = None if doc_id is None else str(doc_id)
        if reason is not None:
            record["export_status"] = "skipped"
            record["skip_reason"] = f"unsafe_doc_id:{reason}"
            records.append(record)
            continue
        doc_id = str(doc_id)
        input_file = _safe_input_file(input_dir, doc_id)
        metrics = text_length_metrics(report_text)
        record.update(
            {
                "clamp_doc_id": doc_id,
                "input_file": str(input_file),
                "text_chars": metrics["text_chars"],
                "text_words": metrics["text_words"],
                "max_sentence_like_words": metrics["max_sentence_like_words"],
                "chars_gt_16000": metrics["text_chars"] > TEXT_CHARS_FLAG,
                "sentence_words_gt_500": metrics["max_sentence_like_words"] > SENTENCE_WORDS_FLAG,
                "export_status": "candidate",
                "skip_reason": "",
            }
        )
        records.append(record)
        text_by_doc_id[doc_id] = report_text
    return pd.DataFrame(records).reindex(columns=MANIFEST_COLUMNS), text_by_doc_id


def _render_doc_id(row: pd.Series, *, id_col: str | None, doc_id_template: str | None) -> object:
    if doc_id_template:
        values: dict[str, str] = {}
        for _, field_name, _, _ in Formatter().parse(doc_id_template):
            if not field_name:
                continue
            value = row.get(field_name)
            if value is None or pd.isna(value) or str(value).strip() == "":
                return None
            values[field_name] = str(value).strip()
        return doc_id_template.format_map(values)
    if id_col is None:
        raise ValueError("id_col is required when doc_id_template is not provided")
    return row.get(id_col)


def _max_int(values: pd.Series) -> int:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    return 0 if clean.empty else int(clean.max())


def _candidate_duplicate_doc_ids(manifest: pd.DataFrame) -> list[str]:
    candidates = manifest[manifest["export_status"] == "candidate"]
    duplicate_mask = candidates.duplicated(["clamp_doc_id"], keep=False)
    if not duplicate_mask.any():
        return []
    return sorted(candidates.loc[duplicate_mask, "clamp_doc_id"].astype(str).unique().tolist())


def _safe_input_file(input_dir: Path, doc_id: str) -> Path:
    path = (input_dir / f"{doc_id}.txt").resolve()
    if not path.is_relative_to(input_dir.resolve()):
        raise ValueError(f"Input path escapes configured input directory: {path}")
    return path


def _prepare_clamp_dirs(
    *,
    input_dir: Path,
    output_dir: Path,
    clear_existing_inputs: bool,
    clear_existing_outputs: bool,
    archive_cleared_files: bool,
) -> None:
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    if clear_existing_inputs:
        clear_or_archive_files(
            input_dir,
            extensions=INPUT_EXTENSIONS,
            archive_cleared_files=archive_cleared_files,
            archive_timestamp=timestamp,
        )
    if clear_existing_outputs:
        clear_or_archive_files(
            output_dir,
            extensions=OUTPUT_EXTENSIONS,
            archive_cleared_files=archive_cleared_files,
            archive_timestamp=timestamp,
        )


def _iter_clamp_project_files(source_dir: Path) -> Iterable[Path]:
    for path in sorted(source_dir.rglob("*")):
        if not path.is_file() or path.name == ".DS_Store":
            continue
        parts = path.relative_to(source_dir).parts
        if "Archive" in parts:
            continue
        if _is_data_input_or_output(parts):
            continue
        yield path


def _is_data_input_or_output(parts: tuple[str, ...]) -> bool:
    lowered = [part.lower() for part in parts]
    for idx, part in enumerate(lowered[:-1]):
        if part == "data" and lowered[idx + 1] in {"input", "output"}:
            return True
    return False


def _normalize_clamp_descriptor_paths(text: str, *, runtime_project_dir: str | Path) -> str:
    posix_live, windows_live = _clamp_runtime_path_variants(runtime_project_dir)
    return (
        text.replace("C:/ClampWin_1.6.6/workspace/ARDS", posix_live)
        .replace(r"C:\ClampWin_1.6.6\workspace\ARDS", windows_live)
        .replace("{{CLAMP_PROJECT_LIVE_DIR_POSIX}}", posix_live)
        .replace("{{CLAMP_PROJECT_LIVE_DIR_WINDOWS}}", windows_live)
    )


def _clamp_runtime_path_variants(runtime_project_dir: str | Path) -> tuple[str, str]:
    raw = str(runtime_project_dir).strip()
    posix = raw.replace("\\", "/")
    if re.fullmatch(r"[A-Za-z]:/.*", posix):
        windows = posix.replace("/", "\\")
    else:
        windows = raw
    return posix, windows


def _unique_target_path(path: Path) -> Path:
    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    for idx in range(1, 10_000):
        candidate = path.with_name(f"{stem}_{idx}{suffix}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Could not find unique archive path for {path}")
