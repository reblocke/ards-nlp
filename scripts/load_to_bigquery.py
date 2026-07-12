from __future__ import annotations

import argparse
from pathlib import Path

from ards_cxr_benchmark.bq import ensure_dataset_exists
from ards_cxr_benchmark.config import default_config_path, load_config, require_existing_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Load a local CSV or Parquet file to BigQuery")
    parser.add_argument("--config", type=Path, default=default_config_path())
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument(
        "--table", required=True, help="Destination table name, without project.dataset"
    )
    parser.add_argument("--source-format", choices=["CSV", "PARQUET"], default="PARQUET")
    parser.add_argument("--skip-leading-rows", type=int, default=1)
    parser.add_argument("--replace", action="store_true")
    return parser.parse_args()


def main() -> None:
    from google.cloud import bigquery

    args = parse_args()
    config = load_config(args.config)
    require_existing_path(args.source, "BigQuery load source file")

    client = bigquery.Client(project=config.bq.project_id)
    ensure_dataset_exists(config, client=client)
    table_id = f"{config.bq.dataset_ref}.{args.table}"
    job_config = bigquery.LoadJobConfig(
        source_format=getattr(bigquery.SourceFormat, args.source_format),
        write_disposition=(
            bigquery.WriteDisposition.WRITE_TRUNCATE
            if args.replace
            else bigquery.WriteDisposition.WRITE_EMPTY
        ),
    )
    if args.source_format == "CSV":
        job_config.skip_leading_rows = args.skip_leading_rows
        job_config.autodetect = True

    with args.source.open("rb") as file:
        load_job = client.load_table_from_file(
            file,
            table_id,
            job_config=job_config,
            location=config.bq.location,
        )

    load_job.result()
    table = client.get_table(table_id)
    print(f"Loaded {table.num_rows:,} rows to {table_id}")


if __name__ == "__main__":
    main()
