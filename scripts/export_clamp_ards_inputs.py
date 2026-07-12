from __future__ import annotations

import argparse
import re
import shlex
import sys
from pathlib import Path

import pandas as pd

from ards_cxr_benchmark.bq import query_to_dataframe
from ards_cxr_benchmark.clamp_ards_inputs import export_clamp_ards_inputs
from ards_cxr_benchmark.config import (
    default_config_path,
    load_config,
    validate_clamp_ards_operational_paths,
)

DEFAULT_BQ_TEXT_COLUMN = "primary_target_text"
DEFAULT_BQ_DOC_ID_TEMPLATE = "s{study_id}"
BQ_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
DEFAULT_BQ_COLUMNS = [
    "subject_id",
    "study_id",
    DEFAULT_BQ_TEXT_COLUMN,
    "manual_review_priority",
    "silver_label_source",
    "qa_flags",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export CXR reports as one CLAMP-ready .txt file per report"
    )
    parser.add_argument("--config", type=Path, default=default_config_path())
    parser.add_argument("--source-file", type=Path, help="Local CSV or Parquet source")
    parser.add_argument(
        "--source-table",
        help="BigQuery source table. Defaults to <project>.<dataset>.model_development_extract.",
    )
    parser.add_argument("--id-col", help="Local source ID column for CLAMP document IDs")
    parser.add_argument("--text-col", default=DEFAULT_BQ_TEXT_COLUMN)
    parser.add_argument("--doc-id-template")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--input-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--project-live-dir", type=Path)
    parser.add_argument("--artifact-dir", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--handoff", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--clear-existing-inputs", action="store_true")
    parser.add_argument("--clear-existing-outputs", action="store_true")
    parser.add_argument("--archive-cleared-files", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive when provided")
    config = load_config(args.config)
    artifact_dir = args.artifact_dir or config.clamp_ards.restricted_artifact_dir
    manifest = args.manifest or config.clamp_ards.input_manifest
    summary = args.summary or config.clamp_ards.input_summary
    handoff = args.handoff or config.clamp_ards.handoff_markdown
    input_dir = args.input_dir or config.clamp_ards.input_dir
    output_dir = args.output_dir or config.clamp_ards.output_dir
    project_live_dir = args.project_live_dir or config.clamp_ards.project_live_dir
    validate_clamp_ards_operational_paths(
        project_live_dir=project_live_dir,
        input_dir=input_dir,
        output_dir=output_dir,
    )

    if args.source_file:
        df = read_table(args.source_file)
        if args.limit is not None:
            df = df.head(args.limit).copy()
        if not args.id_col and not args.doc_id_template:
            raise ValueError("Local source mode requires --id-col or --doc-id-template")
        source_type = "local_file"
        source_name = str(args.source_file)
        id_col = args.id_col
        doc_id_template = args.doc_id_template
    else:
        source_table = normalize_source_table(
            args.source_table, default=f"{config.bq.dataset_ref}.model_development_extract"
        )
        df = query_to_dataframe(
            build_bigquery_source_sql(
                source_table=source_table,
                text_col=args.text_col,
                limit=args.limit,
            ),
            config,
        )
        source_type = "bigquery"
        source_name = source_table
        id_col = args.id_col
        doc_id_template = args.doc_id_template or DEFAULT_BQ_DOC_ID_TEMPLATE

    result = export_clamp_ards_inputs(
        df,
        input_dir=input_dir,
        output_dir=output_dir,
        artifact_dir=artifact_dir,
        manifest_path=manifest,
        summary_path=summary,
        handoff_path=handoff,
        text_col=args.text_col,
        source_type=source_type,
        source_name=source_name,
        command=" ".join(shlex.quote(part) for part in sys.argv),
        project_live_dir=project_live_dir,
        id_col=id_col,
        doc_id_template=doc_id_template,
        overwrite=args.overwrite,
        clear_existing_inputs=args.clear_existing_inputs,
        clear_existing_outputs=args.clear_existing_outputs,
        archive_cleared_files=args.archive_cleared_files,
        dry_run=args.dry_run,
    )
    print(f"Prepared {result.summary['written_files']:,} CLAMP input file(s); manifest: {manifest}")


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.name.lower().endswith((".csv", ".csv.gz", ".txt", ".txt.gz")):
        return pd.read_csv(path)
    raise ValueError(f"Unsupported source-file format: {path}")


def normalize_source_table(source_table: str | None, *, default: str) -> str:
    normalized = str(source_table or default).strip()
    if "`" in normalized or not normalized:
        raise ValueError(f"Unsafe BigQuery table reference: {normalized!r}")
    return normalized


def build_bigquery_source_sql(*, source_table: str, text_col: str, limit: int | None) -> str:
    columns = list(dict.fromkeys([*DEFAULT_BQ_COLUMNS, text_col]))
    select_items = [
        "'mimic_cxr' AS source_dataset",
        *[_quote_identifier(column) for column in columns],
    ]
    select_list = ",\n  ".join(select_items)
    limit_clause = "" if limit is None else f"\nLIMIT {int(limit)}"
    return f"SELECT\n  {select_list}\nFROM `{source_table}`\nORDER BY `study_id`{limit_clause}"


def _quote_identifier(identifier: str) -> str:
    if not BQ_IDENTIFIER_RE.fullmatch(identifier):
        raise ValueError(f"Unsafe BigQuery column identifier: {identifier!r}")
    return f"`{identifier}`"


if __name__ == "__main__":
    main()
