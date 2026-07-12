from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .config import ensure_parent_dir

IDENTIFIER_COLUMNS = [
    "source_dataset",
    "subject_id",
    "study_id",
    "accession_id",
    "encounter_id",
    "annotation_phase",
]
ENTITY_COLUMNS = [
    "clamp_doc_id",
    *IDENTIFIER_COLUMNS,
    "source_output_file",
    "output_file_exists",
    "parse_status",
    "parse_error",
    "start",
    "end",
    "semantic_tag",
    "assertion",
    "cui",
    "attribute",
    "entity_text",
    "entity_text_sha256",
]
PREDICTION_COLUMNS = [
    "clamp_doc_id",
    *IDENTIFIER_COLUMNS,
    "model_name",
    "prediction_score",
    "prediction_label",
    "clamp_ards_entity_count",
    "clamp_output_available",
    "clamp_parse_status",
    "clamp_parse_error",
]
PROBABILISTIC_PREDICTION_COLUMNS = [
    "case_id",
    "model_name",
    "prediction_score",
    "prediction_label",
    *IDENTIFIER_COLUMNS,
]
AUDIT_COLUMNS = [
    "clamp_doc_id",
    "input_file",
    "expected_output_file",
    "output_file_exists",
    "parse_status",
    "parse_error",
    "entity_count",
    "ards_entity_count",
    "prediction_label",
]
SUPPORTED_OUTPUT_SUFFIXES = {".csv", ".tsv", ".txt"}
UNSUPPORTED_OUTPUT_SUFFIXES = {".xmi", ".xmi.gz", ".xdi"}
MODEL_NAME = "clamp_legacy"

FIELD_ALIASES = {
    "start": {"start", "begin", "offsetstart", "spanstart"},
    "end": {"end", "stop", "offsetend", "spanend"},
    "semantic_tag": {"semantic", "semantictag", "semantic_tag", "type", "tag"},
    "assertion": {"assertion", "assertionstatus", "certainty", "negation"},
    "cui": {"cui", "conceptid", "concept_id"},
    "attribute": {"attribute", "attributes", "feature", "features"},
    "entity_text": {"entity", "entitytext", "entity_text", "coveredtext", "text", "mention"},
}


def validate_distinct_clamp_paths(**paths: Path) -> None:
    """Reject direct or symlink-aliased CLAMP input/output path collisions."""

    names_by_path: dict[Path, list[str]] = {}
    for name, path in paths.items():
        resolved = Path(path).expanduser().resolve()
        names_by_path.setdefault(resolved, []).append(name)
    collisions = [(path, names) for path, names in names_by_path.items() if len(names) > 1]
    if collisions:
        details = "; ".join(f"{', '.join(sorted(names))} -> {path}" for path, names in collisions)
        raise ValueError(f"CLAMP input and output paths must be distinct: {details}")


@dataclass(frozen=True)
class ClampOutputParseResult:
    entities: pd.DataFrame
    predictions: pd.DataFrame
    probabilistic_predictions: pd.DataFrame
    audit: pd.DataFrame
    summary: dict[str, Any]


@dataclass(frozen=True)
class ParsedOutputFile:
    entities: pd.DataFrame
    parse_status: str
    parse_error: str
    recognized_fields: list[str]


def parse_clamp_ards_outputs(
    *,
    input_manifest_path: Path,
    output_dir: Path,
) -> ClampOutputParseResult:
    manifest = load_input_manifest(input_manifest_path)
    output_files = discover_clamp_output_files(output_dir)
    if not output_files:
        raise ValueError(f"No CLAMP output files found in {output_dir}")

    output_map, duplicate_doc_ids = map_output_files_by_doc_id(output_files)
    manifest_doc_ids = set(manifest["clamp_doc_id"])
    matched_doc_ids = sorted(manifest_doc_ids & set(output_map))
    unexpected_files = [
        str(path)
        for doc_id, paths in output_map.items()
        if doc_id not in manifest_doc_ids
        for path in paths
    ]
    if not matched_doc_ids:
        raise ValueError("No CLAMP output files matched clamp_doc_id values in the input manifest")

    entity_frames: list[pd.DataFrame] = []
    audit_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    recognized_fields_seen: set[str] = set()
    parse_success_files = 0
    parse_error_files = 0

    for _, manifest_row in manifest.iterrows():
        doc_id = str(manifest_row["clamp_doc_id"])
        output_paths = output_map.get(doc_id, [])
        metadata = _manifest_metadata(manifest_row)
        expected_output_file = str(output_paths[0]) if output_paths else ""
        output_file_exists = bool(output_paths)
        parse_status = "missing_output"
        parse_error = ""
        parsed_entities = _empty_normalized_entities()

        if doc_id in duplicate_doc_ids:
            parse_status = "duplicate_output"
            parse_error = "multiple_output_files_for_doc_id"
            parse_error_files += 1
        elif output_paths:
            parsed = parse_clamp_output_file(output_paths[0])
            parse_status = parsed.parse_status
            parse_error = parsed.parse_error
            parsed_entities = parsed.entities
            recognized_fields_seen.update(parsed.recognized_fields)
            if parse_status in {"parsed", "parsed_empty"}:
                parse_success_files += 1
            else:
                parse_error_files += 1

        entity_count = int(len(parsed_entities))
        ards_entity_count = count_ards_entities(parsed_entities)
        prediction_label = _prediction_label(parse_status, ards_entity_count)
        prediction_score = None if prediction_label is None else float(prediction_label)

        if entity_count:
            entity_frame = parsed_entities.copy()
            for column, value in metadata.items():
                entity_frame[column] = value
            entity_frame["clamp_doc_id"] = doc_id
            entity_frame["source_output_file"] = expected_output_file
            entity_frame["output_file_exists"] = output_file_exists
            entity_frame["parse_status"] = parse_status
            entity_frame["parse_error"] = parse_error
            entity_frames.append(entity_frame.reindex(columns=ENTITY_COLUMNS))

        audit_rows.append(
            {
                "clamp_doc_id": doc_id,
                "input_file": manifest_row.get("input_file", ""),
                "expected_output_file": expected_output_file,
                "output_file_exists": output_file_exists,
                "parse_status": parse_status,
                "parse_error": parse_error,
                "entity_count": entity_count,
                "ards_entity_count": ards_entity_count,
                "prediction_label": prediction_label,
            }
        )
        prediction_rows.append(
            {
                "clamp_doc_id": doc_id,
                **metadata,
                "model_name": MODEL_NAME,
                "prediction_score": prediction_score,
                "prediction_label": prediction_label,
                "clamp_ards_entity_count": ards_entity_count,
                "clamp_output_available": output_file_exists,
                "clamp_parse_status": parse_status,
                "clamp_parse_error": parse_error,
            }
        )

    if not recognized_fields_seen and parse_success_files == 0:
        raise ValueError("Parser could not identify any recognized CLAMP fields")
    if parse_success_files == 0:
        raise ValueError("All matched CLAMP output files failed parsing")

    entities = (
        pd.concat(entity_frames, ignore_index=True).reindex(columns=ENTITY_COLUMNS)
        if entity_frames
        else pd.DataFrame(columns=ENTITY_COLUMNS)
    )
    predictions = pd.DataFrame(prediction_rows).reindex(columns=PREDICTION_COLUMNS)
    audit = pd.DataFrame(audit_rows).reindex(columns=AUDIT_COLUMNS)
    probabilistic_predictions = make_probabilistic_predictions(predictions)
    summary = summarize_clamp_outputs(
        manifest=manifest,
        output_files=output_files,
        output_map=output_map,
        duplicate_doc_ids=duplicate_doc_ids,
        unexpected_files=unexpected_files,
        audit=audit,
        predictions=predictions,
        parse_success_files=parse_success_files,
        parse_error_files=parse_error_files,
    )
    return ClampOutputParseResult(
        entities=entities,
        predictions=predictions,
        probabilistic_predictions=probabilistic_predictions,
        audit=audit,
        summary=summary,
    )


def load_input_manifest(path: Path) -> pd.DataFrame:
    manifest = pd.read_csv(path, dtype=str).fillna("")
    if "clamp_doc_id" not in manifest.columns:
        raise ValueError(f"Input manifest is missing clamp_doc_id: {path}")
    manifest["clamp_doc_id"] = manifest["clamp_doc_id"].astype(str).str.strip()
    if (manifest["clamp_doc_id"] == "").any():
        raise ValueError("Input manifest contains blank clamp_doc_id values")
    if manifest.duplicated(["clamp_doc_id"], keep=False).any():
        raise ValueError("Input manifest contains duplicate clamp_doc_id values")
    return manifest


def discover_clamp_output_files(output_dir: Path) -> list[Path]:
    if not output_dir.exists():
        return []
    files: list[Path] = []
    for path in sorted(output_dir.rglob("*")):
        if not path.is_file() or "Archive" in path.parts:
            continue
        if _output_suffix(path) in SUPPORTED_OUTPUT_SUFFIXES | UNSUPPORTED_OUTPUT_SUFFIXES:
            files.append(path)
    return files


def map_output_files_by_doc_id(output_files: list[Path]) -> tuple[dict[str, list[Path]], set[str]]:
    output_map: dict[str, list[Path]] = {}
    for path in output_files:
        output_map.setdefault(_output_doc_id(path), []).append(path)
    duplicate_doc_ids = {doc_id for doc_id, paths in output_map.items() if len(paths) > 1}
    return output_map, duplicate_doc_ids


def parse_clamp_output_file(path: Path) -> ParsedOutputFile:
    suffix = _output_suffix(path)
    if suffix in UNSUPPORTED_OUTPUT_SUFFIXES:
        return ParsedOutputFile(
            entities=_empty_normalized_entities(),
            parse_status="parse_error",
            parse_error=f"unsupported_output_format:{suffix}",
            recognized_fields=[],
        )
    if suffix not in SUPPORTED_OUTPUT_SUFFIXES:
        return ParsedOutputFile(
            entities=_empty_normalized_entities(),
            parse_status="parse_error",
            parse_error=f"unsupported_output_format:{suffix}",
            recognized_fields=[],
        )
    if path.stat().st_size == 0 or not path.read_text(encoding="utf-8", errors="ignore").strip():
        return ParsedOutputFile(
            entities=_empty_normalized_entities(),
            parse_status="parsed_empty",
            parse_error="",
            recognized_fields=[],
        )
    try:
        raw = _read_tabular_output(path)
    except Exception as exc:
        return ParsedOutputFile(
            entities=_empty_normalized_entities(),
            parse_status="parse_error",
            parse_error=f"read_error:{exc}",
            recognized_fields=[],
        )
    mapping = _recognized_column_mapping(raw.columns)
    if not mapping:
        return ParsedOutputFile(
            entities=_empty_normalized_entities(),
            parse_status="parse_error",
            parse_error="no_recognized_fields",
            recognized_fields=[],
        )
    if "semantic_tag" not in mapping:
        return ParsedOutputFile(
            entities=_empty_normalized_entities(),
            parse_status="parse_error",
            parse_error="missing_required_field:semantic_tag",
            recognized_fields=sorted(mapping),
        )
    normalized = pd.DataFrame()
    for target in [
        "start",
        "end",
        "semantic_tag",
        "assertion",
        "cui",
        "attribute",
        "entity_text",
    ]:
        if target in mapping:
            normalized[target] = raw[mapping[target]]
        else:
            normalized[target] = None
    normalized["start"] = pd.to_numeric(normalized["start"], errors="coerce").astype("Int64")
    normalized["end"] = pd.to_numeric(normalized["end"], errors="coerce").astype("Int64")
    for column in ["semantic_tag", "assertion", "cui", "attribute", "entity_text"]:
        normalized[column] = normalized[column].map(_clean_optional_string)
    normalized["semantic_tag"] = normalized["semantic_tag"].map(_normalize_semantic_tag)
    normalized["entity_text_sha256"] = normalized["entity_text"].map(_sha256_or_none)
    return ParsedOutputFile(
        entities=normalized.reindex(
            columns=[
                "start",
                "end",
                "semantic_tag",
                "assertion",
                "cui",
                "attribute",
                "entity_text",
                "entity_text_sha256",
            ]
        ),
        parse_status="parsed" if len(normalized) else "parsed_empty",
        parse_error="",
        recognized_fields=sorted(mapping),
    )


def count_ards_entities(entities: pd.DataFrame) -> int:
    if "semantic_tag" not in entities.columns or entities.empty:
        return 0
    return int((entities["semantic_tag"].fillna("").astype(str).str.upper() == "ARDS").sum())


def make_probabilistic_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    eligible = predictions[predictions["prediction_score"].notna()].copy()
    eligible["case_id"] = eligible["clamp_doc_id"].astype(str)
    return eligible.reindex(columns=PROBABILISTIC_PREDICTION_COLUMNS).reset_index(drop=True)


def summarize_clamp_outputs(
    *,
    manifest: pd.DataFrame,
    output_files: list[Path],
    output_map: dict[str, list[Path]],
    duplicate_doc_ids: set[str],
    unexpected_files: list[str],
    audit: pd.DataFrame,
    predictions: pd.DataFrame,
    parse_success_files: int,
    parse_error_files: int,
) -> dict[str, Any]:
    expected = len(manifest)
    observed = len(output_files)
    matched = int(audit["output_file_exists"].sum())
    missing = int((~audit["output_file_exists"]).sum())
    positive = int((predictions["prediction_label"] == 1).sum())
    negative = int((predictions["prediction_label"] == 0).sum())
    evaluable = int(predictions["prediction_label"].notna().sum())
    warnings: list[str] = []
    if missing:
        warnings.append("some_input_files_have_no_clamp_output")
    if unexpected_files:
        warnings.append("unexpected_output_files_without_manifest_rows")
    if duplicate_doc_ids:
        warnings.append("duplicate_output_files_for_doc_ids")
    if evaluable and (positive == 0 or negative == 0):
        warnings.append("all_evaluable_predictions_have_one_class")
    return {
        "expected_input_files": int(expected),
        "observed_output_files": int(observed),
        "matched_output_files": int(matched),
        "missing_output_files": int(missing),
        "unexpected_output_files": int(len(unexpected_files)),
        "duplicate_output_doc_ids": int(len(duplicate_doc_ids)),
        "duplicate_output_files": int(
            sum(len(paths) for doc_id, paths in output_map.items() if doc_id in duplicate_doc_ids)
        ),
        "parse_success_files": int(parse_success_files),
        "parse_error_files": int(parse_error_files),
        "documents_with_ards_entity": positive,
        "documents_without_ards_entity": negative,
        "evaluable_prediction_rows": evaluable,
        "non_evaluable_prediction_rows": int(len(predictions) - evaluable),
        "warnings": warnings,
        "unexpected_output_file_examples": unexpected_files[:20],
        "duplicate_doc_id_examples": sorted(duplicate_doc_ids)[:20],
    }


def write_clamp_teacher_outputs(
    result: ClampOutputParseResult,
    *,
    entity_output: Path,
    prediction_output: Path,
    probabilistic_prediction_output: Path,
    audit_output: Path,
    summary_output: Path,
) -> None:
    ensure_parent_dir(entity_output)
    ensure_parent_dir(prediction_output)
    ensure_parent_dir(probabilistic_prediction_output)
    ensure_parent_dir(audit_output)
    ensure_parent_dir(summary_output)
    result.entities.to_parquet(entity_output, index=False)
    result.predictions.to_parquet(prediction_output, index=False)
    result.probabilistic_predictions.to_parquet(probabilistic_prediction_output, index=False)
    result.audit.to_csv(audit_output, index=False)
    summary_output.write_text(
        json.dumps(result.summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary_output.with_suffix(".md").write_text(
        render_clamp_teacher_summary(result.summary),
        encoding="utf-8",
    )


def render_clamp_teacher_summary(summary: dict[str, Any]) -> str:
    lines = [
        "# ARDS CLAMP Teacher Output Summary",
        "",
        "These are legacy CLAMP teacher/baseline outputs, not human gold-standard labels.",
        "",
        f"- Expected input files: {summary['expected_input_files']:,}",
        f"- Observed output files: {summary['observed_output_files']:,}",
        f"- Matched output files: {summary['matched_output_files']:,}",
        f"- Missing output files: {summary['missing_output_files']:,}",
        f"- Unexpected output files: {summary['unexpected_output_files']:,}",
        f"- Parse success files: {summary['parse_success_files']:,}",
        f"- Parse error files: {summary['parse_error_files']:,}",
        f"- Documents with ARDS entity: {summary['documents_with_ards_entity']:,}",
        f"- Documents without ARDS entity: {summary['documents_without_ards_entity']:,}",
        f"- Evaluable prediction rows: {summary['evaluable_prediction_rows']:,}",
        f"- Non-evaluable prediction rows: {summary['non_evaluable_prediction_rows']:,}",
        "",
    ]
    if "entity_rows" in summary:
        lines.insert(-1, f"- Entity rows: {summary['entity_rows']:,}")
    if summary.get("source_archive_name"):
        lines.extend(
            [
                "## Source",
                "",
                f"- Source kind: {summary.get('source_kind', 'archive')}",
                f"- Archive: {summary['source_archive_name']}",
                f"- Archive SHA-256: `{summary['source_archive_sha256']}`",
                "",
            ]
        )
    warnings = summary.get("warnings", [])
    if warnings:
        lines.extend(["## Warnings", "", *[f"- {warning}" for warning in warnings], ""])
    return "\n".join(lines)


def _read_tabular_output(path: Path) -> pd.DataFrame:
    suffix = _output_suffix(path)
    sep = "\t" if suffix == ".tsv" else "," if suffix == ".csv" else None
    df = pd.read_csv(path, sep=sep, engine="python", dtype=str)
    if len(df.columns) == 1 and suffix == ".txt":
        df = pd.read_csv(path, sep="\t", engine="python", dtype=str)
    return df.fillna("")


def _recognized_column_mapping(columns: pd.Index) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for column in columns:
        normalized = _normalize_column_name(str(column))
        for target, aliases in FIELD_ALIASES.items():
            if target not in mapping and normalized in aliases:
                mapping[target] = str(column)
    return mapping


def _normalize_column_name(value: str) -> str:
    return "".join(char for char in value.strip().lower() if char.isalnum())


def _normalize_semantic_tag(value: object) -> str | None:
    clean = _clean_optional_string(value)
    return None if clean is None else clean.strip()


def _clean_optional_string(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    clean = str(value).strip()
    return None if clean == "" else clean


def _sha256_or_none(value: object) -> str | None:
    clean = _clean_optional_string(value)
    if clean is None:
        return None
    return hashlib.sha256(clean.encode("utf-8")).hexdigest()


def _prediction_label(parse_status: str, ards_entity_count: int) -> int | None:
    if parse_status not in {"parsed", "parsed_empty"}:
        return None
    return int(ards_entity_count > 0)


def _manifest_metadata(row: pd.Series) -> dict[str, Any]:
    return {column: row.get(column, "") for column in IDENTIFIER_COLUMNS}


def _empty_normalized_entities() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "start",
            "end",
            "semantic_tag",
            "assertion",
            "cui",
            "attribute",
            "entity_text",
            "entity_text_sha256",
        ]
    )


def _output_doc_id(path: Path) -> str:
    name = path.name
    suffix = _output_suffix(path)
    stem = name[: -len(suffix)] if suffix else path.stem
    if stem.lower().endswith(".txt"):
        stem = stem[:-4]
    return stem


def _output_suffix(path: Path) -> str:
    name = path.name.lower()
    if name.endswith(".xmi.gz"):
        return ".xmi.gz"
    return path.suffix.lower()
