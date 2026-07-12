from __future__ import annotations

import argparse
import json
from pathlib import Path

from ards_cxr_benchmark.bq import query_to_dataframe
from ards_cxr_benchmark.config import default_config_path, ensure_parent_dir, load_config
from ards_cxr_benchmark.schemas import SILVER_REFERENCE_REQUIRED_COLUMNS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run BigQuery QA checks for benchmark tables")
    parser.add_argument("--config", type=Path, default=default_config_path())
    parser.add_argument("--out-json", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    out_json = args.out_json or (config.paths.qa_dir / "reference_table_qa.json")

    columns_sql = f"""
    SELECT column_name
    FROM `{config.bq.dataset_ref}.INFORMATION_SCHEMA.COLUMNS`
    WHERE table_name = 'silver_reference_candidates'
    ORDER BY ordinal_position
    """
    row_count_sql = f"""
    SELECT
      COUNT(*) AS n_rows,
      COUNTIF(report_text IS NOT NULL) AS n_with_report_text,
      COUNTIF(has_mimic_cxr_jpg_labels) AS n_with_mimic_cxr_jpg_labels,
      COUNTIF(silver_bilateral_opacity_score IS NOT NULL) AS n_with_silver_score,
      COUNTIF(strict_bilateral_opacity_label IS NOT NULL) AS n_strict_nonnull,
      COUNTIF(sensitive_bilateral_opacity_label IS NOT NULL) AS n_sensitive_nonnull
    FROM `{config.bq.dataset_ref}.silver_reference_candidates`
    """

    columns = query_to_dataframe(columns_sql, config)["column_name"].tolist()
    missing = [column for column in SILVER_REFERENCE_REQUIRED_COLUMNS if column not in set(columns)]
    row_counts = query_to_dataframe(row_count_sql, config).to_dict(orient="records")[0]
    result = {
        "table": f"{config.bq.dataset_ref}.silver_reference_candidates",
        "missing_required_columns": missing,
        "row_counts": row_counts,
    }

    ensure_parent_dir(out_json)
    out_json.write_text(json.dumps(result, indent=2, default=str) + "\n", encoding="utf-8")
    print(f"Wrote QA summary to {out_json}")
    if missing:
        raise SystemExit(f"Missing required columns: {missing}")


if __name__ == "__main__":
    main()
