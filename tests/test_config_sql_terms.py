from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from ards_cxr_benchmark.bq import ensure_dataset_exists, render_sql_template
from ards_cxr_benchmark.config import load_config
from ards_cxr_benchmark.label_terms import load_label_terms, missing_required_term_groups
from ards_cxr_benchmark.schemas import SILVER_REFERENCE_REQUIRED_COLUMNS


class FakeBigQueryClient:
    def __init__(self) -> None:
        self.created: list[tuple[Any, bool]] = []

    def create_dataset(self, dataset: Any, *, exists_ok: bool = False) -> Any:
        self.created.append((dataset, exists_ok))
        return dataset


def test_load_example_config() -> None:
    config = load_config(Path("config/config.example.yaml"))

    assert config.name == "ards-cxr-mimic-benchmark"
    assert config.bq.dataset == "ards_mimic_cxr_benchmark"
    assert config.primary_label_text_scope == "full_report"
    assert config.paths.report_root.is_absolute()
    assert config.clamp_ards.output_archive.name == "ARDS_CLAMP_Output_txt_only.zip"
    assert config.clamp_ards.output_packet_summary.is_absolute()


def test_label_terms_config_has_required_groups() -> None:
    terms = load_label_terms(Path("config/label_terms.yaml"))

    assert missing_required_term_groups(terms) == []
    assert "opacity" in terms["opacity_observation_terms"]
    assert "bibasilar" in terms["bilateral_anatomy_terms"]


def test_render_sql_template() -> None:
    sql = "SELECT * FROM `{{ project_id }}.{{ bq_dataset }}.table`"
    rendered = render_sql_template(
        sql,
        {"project_id": "proj", "bq_dataset": "dataset"},
    )

    assert rendered == "SELECT * FROM `proj.dataset.table`"


def test_render_sql_template_rejects_missing_parameter() -> None:
    with pytest.raises(KeyError, match="Missing SQL template parameters"):
        render_sql_template("SELECT '{{ missing }}'", {})


def test_core_sql_files_render_with_example_config() -> None:
    config = load_config(Path("config/config.example.yaml"))
    for sql_path in sorted(Path("sql").glob("*.sql")):
        rendered = render_sql_template(
            sql_path.read_text(encoding="utf-8"), config.sql_parameters()
        )
        assert "{{" not in rendered
        assert rendered.strip()


def test_silver_sql_and_schema_include_jpg_join_provenance() -> None:
    config = load_config(Path("config/config.example.yaml"))
    sql = render_sql_template(
        Path("sql/60_build_silver_reference_candidates.sql").read_text(encoding="utf-8"),
        config.sql_parameters(),
    )

    assert "has_mimic_cxr_jpg_labels" in SILVER_REFERENCE_REQUIRED_COLUMNS
    assert "has_mimic_cxr_jpg_labels" in sql
    assert "has_mimic_cxr_jpg_labels\n      AND COALESCE(chexpert_lung_opacity, 0) = 0" in sql


def test_model_split_sql_renders_subject_level_hash_split() -> None:
    config = load_config(Path("config/config.example.yaml"))
    sql = render_sql_template(
        Path("sql/81_build_model_development_splits.sql").read_text(encoding="utf-8"),
        config.sql_parameters(),
    )

    assert "model_development_splits" in sql
    assert "model_development_extract" in sql
    assert "SHA256(CONCAT(CAST(subject_id AS STRING), '-mimic-bilat-opacity-v1-split'))" in sql
    assert "WHEN split_bucket < 7000 THEN 'train'" in sql
    assert "WHEN split_bucket < 8500 THEN 'validation'" in sql


def test_reviewed_subset_comparison_sql_renders_without_gold_outputs() -> None:
    config = load_config(Path("config/config.example.yaml"))
    sql = render_sql_template(
        Path("sql/95_compare_reviewed_subset_template.sql").read_text(encoding="utf-8"),
        config.sql_parameters(),
    )

    assert "manual_review_completed_labels" in sql
    assert "reviewed_subset_silver_comparison" in sql
    assert "gold_" not in sql


def test_ensure_dataset_exists_uses_configured_dataset_and_exists_ok() -> None:
    config = load_config(Path("config/config.example.yaml"))
    client = FakeBigQueryClient()

    dataset = ensure_dataset_exists(config, client=client)

    assert len(client.created) == 1
    created_dataset, exists_ok = client.created[0]
    assert dataset is created_dataset
    assert exists_ok is True
    assert created_dataset.project == config.bq.project_id
    assert created_dataset.dataset_id == config.bq.dataset
    assert created_dataset.location == config.bq.location
