from __future__ import annotations

import argparse
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from ards_cxr_benchmark import PIPELINE_VERSION
from ards_cxr_benchmark.bq import query_to_dataframe
from ards_cxr_benchmark.config import default_config_path, ensure_parent_dir, load_config
from ards_cxr_benchmark.modeling import SPLIT_SALT
from ards_cxr_benchmark.paths import get_paths

SNAPSHOT_TABLES = [
    "silver_reference_candidates",
    "model_development_extract",
    "model_development_splits",
    "manual_review_sample",
]


def default_metrics_path() -> Path:
    return get_paths().root / "artifacts" / "modeling" / "silver_baseline_metrics.csv"


def default_qa_json_path() -> Path:
    return get_paths().root / "artifacts" / "modeling" / "modeling_qa_summary.json"


def default_out_json_path() -> Path:
    return get_paths().root / "artifacts" / "modeling" / "modeling_snapshot_v1.json"


def default_out_md_path() -> Path:
    return get_paths().root / "docs" / "MODELING_SNAPSHOT_V1.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write aggregate v1 modeling snapshot")
    parser.add_argument("--config", type=Path, default=default_config_path())
    parser.add_argument("--metrics-csv", type=Path, default=default_metrics_path())
    parser.add_argument("--modeling-qa-json", type=Path, default=default_qa_json_path())
    parser.add_argument("--out-json", type=Path, default=default_out_json_path())
    parser.add_argument("--out-md", type=Path, default=default_out_md_path())
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    metrics = pd.read_csv(args.metrics_csv)
    modeling_qa = json.loads(args.modeling_qa_json.read_text(encoding="utf-8"))

    snapshot = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "pipeline_version": PIPELINE_VERSION,
        "repo": git_snapshot(),
        "bigquery": {
            "dataset": config.bq.dataset_ref,
            "location": config.bq.location,
            "tables": table_metadata(config),
            "row_counts": row_counts(config),
            "eligible_rows": eligible_rows(config),
            "split_counts": split_counts(config),
        },
        "split_salt": SPLIT_SALT,
        "artifact_paths": {
            "metrics_csv": "artifacts/modeling/silver_baseline_metrics.csv",
            "modeling_qa_json": "artifacts/modeling/modeling_qa_summary.json",
            "predictions_parquet": "data/derived/modeling/silver_baseline_predictions.parquet",
            "snapshot_json": "artifacts/modeling/modeling_snapshot_v1.json",
        },
        "modeling_qa": modeling_qa,
        "test_metrics": test_metrics(metrics),
    }

    ensure_parent_dir(args.out_json)
    ensure_parent_dir(args.out_md)
    args.out_json.write_text(json.dumps(snapshot, indent=2, default=str) + "\n", encoding="utf-8")
    args.out_md.write_text(render_snapshot_markdown(snapshot), encoding="utf-8")
    print(f"Wrote snapshot JSON to {args.out_json}")
    print(f"Wrote tracked snapshot markdown to {args.out_md}")


def git_snapshot() -> dict[str, Any]:
    return {
        "commit": _git(["rev-parse", "HEAD"]),
        "short_commit": _git(["rev-parse", "--short", "HEAD"]),
        "tracked_dirty": bool(_git(["status", "--short", "--untracked-files=no"])),
    }


def table_metadata(config: Any) -> list[dict[str, Any]]:
    table_list = ", ".join(f"'{table}'" for table in SNAPSHOT_TABLES)
    sql = f"""
    SELECT
      table_id AS table_name,
      row_count,
      size_bytes,
      TIMESTAMP_MILLIS(creation_time) AS created_at,
      TIMESTAMP_MILLIS(last_modified_time) AS last_modified_at
    FROM `{config.bq.dataset_ref}.__TABLES__`
    WHERE table_id IN ({table_list})
    ORDER BY table_name
    """
    return query_to_dataframe(sql, config).to_dict(orient="records")


def row_counts(config: Any) -> list[dict[str, Any]]:
    selects = [
        f"SELECT '{table}' AS table_name, COUNT(*) AS n_rows FROM `{config.bq.dataset_ref}.{table}`"
        for table in SNAPSHOT_TABLES
    ]
    return query_to_dataframe("\nUNION ALL\n".join(selects), config).to_dict(orient="records")


def eligible_rows(config: Any) -> dict[str, int]:
    sql = f"""
    SELECT
      COUNTIF(strict_bilateral_opacity_label IS NOT NULL) AS strict_eligible_rows,
      COUNTIF(sensitive_bilateral_opacity_label IS NOT NULL) AS sensitive_eligible_rows
    FROM `{config.bq.dataset_ref}.model_development_extract`
    """
    row = query_to_dataframe(sql, config).to_dict(orient="records")[0]
    return {key: int(value) for key, value in row.items()}


def split_counts(config: Any) -> list[dict[str, Any]]:
    sql = f"""
    SELECT
      split,
      COUNT(*) AS n_rows,
      COUNT(DISTINCT subject_id) AS n_subjects
    FROM `{config.bq.dataset_ref}.model_development_splits`
    GROUP BY split
    ORDER BY split
    """
    return query_to_dataframe(sql, config).to_dict(orient="records")


def test_metrics(metrics: pd.DataFrame) -> list[dict[str, Any]]:
    columns = [
        "task",
        "model",
        "split",
        "n",
        "prevalence",
        "roc_auc",
        "average_precision",
        "f1",
        "recall",
        "specificity",
    ]
    return metrics[metrics["split"] == "test"][columns].to_dict(orient="records")


def render_snapshot_markdown(snapshot: dict[str, Any]) -> str:
    repo = snapshot["repo"]
    bq = snapshot["bigquery"]
    qa = snapshot["modeling_qa"]
    lines = [
        "# Modeling Snapshot V1",
        "",
        "Aggregate, non-sensitive snapshot for the silver-label modeling baseline.",
        "",
        f"- Generated UTC: `{snapshot['generated_at_utc']}`",
        f"- Commit: `{repo['short_commit']}`",
        f"- Tracked dirty at snapshot time: `{repo['tracked_dirty']}`",
        f"- BigQuery dataset: `{bq['dataset']}`",
        f"- Split salt: `{snapshot['split_salt']}`",
        f"- Modeling QA passed: `{qa['passed']}`",
        "",
        "## Row Counts",
        "",
        _markdown_table(bq["row_counts"], ["table_name", "n_rows"]),
        "",
        "## Eligible Rows",
        "",
        _markdown_table([bq["eligible_rows"]], list(bq["eligible_rows"].keys())),
        "",
        "## Split Counts",
        "",
        _markdown_table(bq["split_counts"], ["split", "n_rows", "n_subjects"]),
        "",
        "## Test Metrics",
        "",
        _markdown_table(
            snapshot["test_metrics"],
            [
                "task",
                "model",
                "n",
                "prevalence",
                "roc_auc",
                "average_precision",
                "f1",
                "recall",
                "specificity",
            ],
        ),
        "",
        "## Artifact Policy",
        "",
        "Generated model caches, fitted models, predictions, and JSON snapshots remain ignored.",
        "This tracked snapshot intentionally contains only aggregate metadata and metrics.",
        "",
    ]
    return "\n".join(lines)


def _markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    if not rows:
        return "_No rows._"
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_format_value(row.get(column)) for column in columns) + " |")
    return "\n".join(lines)


def _format_value(value: Any) -> str:
    if value is None or pd.isna(value):
        return "NULL"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _git(args: list[str]) -> str:
    result = subprocess.run(
        ["git", *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


if __name__ == "__main__":
    main()
