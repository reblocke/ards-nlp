from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from .config import PipelineConfig

_TEMPLATE_PATTERN = re.compile(r"{{\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*}}")


def render_sql_template(sql: str, params: dict[str, str]) -> str:
    missing: set[str] = set()

    def replace(match: re.Match[str]) -> str:
        key = match.group(1)
        if key not in params:
            missing.add(key)
            return match.group(0)
        return params[key]

    rendered = _TEMPLATE_PATTERN.sub(replace, sql)
    if missing:
        raise KeyError(f"Missing SQL template parameters: {sorted(missing)}")
    return rendered


def load_and_render_sql(
    path: Path, config: PipelineConfig, extra: dict[str, str] | None = None
) -> str:
    params = config.sql_parameters()
    if extra:
        params.update(extra)
    return render_sql_template(path.read_text(encoding="utf-8"), params)


def make_bigquery_client(project_id: str | None = None) -> Any:
    from google.cloud import bigquery

    return bigquery.Client(project=project_id)


def ensure_dataset_exists(config: PipelineConfig, client: Any | None = None) -> Any:
    from google.cloud import bigquery

    client = client or make_bigquery_client(config.bq.project_id)
    dataset = bigquery.Dataset(config.bq.dataset_ref)
    dataset.location = config.bq.location
    return client.create_dataset(dataset, exists_ok=True)


def execute_sql(
    sql: str,
    config: PipelineConfig,
    *,
    dry_run: bool = False,
    use_query_cache: bool = True,
) -> Any:
    from google.cloud import bigquery

    client = make_bigquery_client(config.bq.project_id)
    job_config = bigquery.QueryJobConfig(
        dry_run=dry_run,
        use_query_cache=use_query_cache,
    )
    query_job = client.query(sql, job_config=job_config, location=config.bq.location)
    if dry_run:
        return query_job
    return query_job.result()


def query_to_dataframe(sql: str, config: PipelineConfig) -> Any:
    client = make_bigquery_client(config.bq.project_id)
    return client.query(sql, location=config.bq.location).to_dataframe()
