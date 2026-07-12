from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from ards_cxr_benchmark.pilot_annotation_agreement import (
    AnnotationPilotConfig,
    PilotColumnConfig,
    PilotOutputConfig,
    PilotProjectConfig,
    bootstrap_alignment_metrics,
    build_case_disagreement_review,
    build_case_summary,
    build_task_long_table,
    compute_image_report_alignment,
    compute_interrater_agreement,
    compute_pairwise_agreement,
    load_annotation_pilot_config,
    load_redcap_rater_exports,
    run_annotation_pilot,
    validate_annotation_pilot_output_paths,
)

ROOT = Path(__file__).resolve().parents[1]
FIXTURE_CONFIG = ROOT / "tests" / "fixtures" / "redcap_annotation" / "config.yaml"


def test_example_and_synthetic_configs_load_with_explicit_rater_files() -> None:
    example = load_annotation_pilot_config(ROOT / "config" / "annotation_pilot.example.yaml")
    synthetic = load_annotation_pilot_config(FIXTURE_CONFIG)

    assert example.rater_ids == ("R01", "R02", "R03")
    assert synthetic.project.expected_raters == 3
    assert synthetic.columns.case_id == "id_accession"


def test_config_rejects_reused_input_file(tmp_path: Path) -> None:
    config_data = yaml.safe_load(FIXTURE_CONFIG.read_text(encoding="utf-8"))
    config_data["inputs"]["R02"] = config_data["inputs"]["R01"]
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config_data), encoding="utf-8")

    with pytest.raises(ValueError, match="different REDCap export"):
        load_annotation_pilot_config(config_path)


def test_loader_preserves_string_ids_and_drops_report_text() -> None:
    wide = load_redcap_rater_exports(load_annotation_pilot_config(FIXTURE_CONFIG))

    assert wide.loc[0, "case_id"] == "000001"
    assert "interpretation_text" not in wide.columns
    assert "assessment_report" not in wide.columns
    assert set(wide["rater_id"]) == {"R01", "R02", "R03"}


def test_loader_rejects_missing_required_source_column(tmp_path: Path) -> None:
    config = _temp_config(tmp_path)
    frame = _source_frame().drop(columns="assessment_image")
    frame.to_csv(config.inputs["R01"], index=False)

    with pytest.raises(ValueError, match="missing columns"):
        load_redcap_rater_exports(config)


def test_loader_rejects_duplicate_case_within_rater(tmp_path: Path) -> None:
    config = _temp_config(tmp_path)
    frame = _source_frame()
    pd.concat([frame, frame.iloc[[0]]], ignore_index=True).to_csv(config.inputs["R01"], index=False)

    with pytest.raises(ValueError, match="duplicate case rows"):
        load_redcap_rater_exports(config)


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        ("adjudication_image_complete", "7", "invalid values"),
        ("assessment_report", "101", "invalid completed values"),
        ("assessment_image", "not-a-number", "invalid completed values"),
    ],
)
def test_loader_rejects_invalid_completion_or_completed_rating(
    tmp_path: Path, column: str, value: str, message: str
) -> None:
    config = _temp_config(tmp_path)
    frame = _source_frame()
    frame[column] = frame[column].astype(object)
    frame.loc[0, column] = value
    frame.to_csv(config.inputs["R01"], index=False)

    with pytest.raises(ValueError, match=message):
        load_redcap_rater_exports(config)


def test_incomplete_task_is_excluded_without_removing_other_task() -> None:
    config = load_annotation_pilot_config(FIXTURE_CONFIG)
    wide = load_redcap_rater_exports(config)
    long, exclusions = build_task_long_table(
        wide,
        complete_status_value=2,
        expected_rater_ids=config.rater_ids,
    )

    r03_case7 = long[(long["case_id"] == "000007") & (long["rater_id"] == "R03")]
    assert r03_case7["task"].tolist() == ["report"]
    excluded = exclusions[
        (exclusions["case_id"] == "000007")
        & (exclusions["rater_id"] == "R03")
        & (exclusions["task"] == "image")
    ]
    assert excluded.iloc[0]["exclusion_reason"] == "instrument_incomplete"


def test_case_summary_uses_sample_sd_and_correct_counts() -> None:
    long = _long_ratings()
    summary = build_case_summary(long)

    case_a = summary[summary["case_id"] == "case_a"].iloc[0]
    assert case_a["n_image_raters"] == 3
    assert case_a["image_mean_0_100"] == pytest.approx(20.0)
    assert case_a["image_sd_0_100"] == pytest.approx(10.0)
    assert case_a["image_range_0_100"] == pytest.approx(20.0)
    assert case_a["report_minus_image_mean_0_100"] == pytest.approx(5.0)


def test_case_summary_retains_source_case_with_no_completed_ratings() -> None:
    summary = build_case_summary(_long_ratings(), all_case_ids=["case_a", "case_b", "case_c"])

    case_c = summary[summary["case_id"] == "case_c"].iloc[0]
    assert case_c["n_image_raters"] == 0
    assert case_c["n_report_raters"] == 0
    assert pd.isna(case_c["image_mean_0_100"])


def test_pairwise_metrics_match_hand_calculation() -> None:
    pairwise = compute_pairwise_agreement(_long_ratings(), ("R01", "R02", "R03"))
    row = pairwise[
        (pairwise["task"] == "image")
        & (pairwise["rater_a"] == "R01")
        & (pairwise["rater_b"] == "R02")
    ].iloc[0]

    assert row["n_paired_cases"] == 2
    assert row["mean_signed_difference_a_minus_b_0_100"] == pytest.approx(-10.0)
    assert row["mean_absolute_difference_0_100"] == pytest.approx(10.0)
    assert row["root_mean_squared_difference_0_100"] == pytest.approx(10.0)
    assert row["pearson_correlation_association"] == pytest.approx(1.0)


def test_identical_rater_profiles_produce_unit_icc() -> None:
    rows = []
    for case_id, value in (("a", 10), ("b", 50), ("c", 90), ("d", 30)):
        for rater_id in ("R01", "R02", "R03"):
            rows.extend(
                [
                    _long_row(case_id, rater_id, "image", value),
                    _long_row(case_id, rater_id, "report", value),
                ]
            )
    long = pd.DataFrame(rows)
    summary = compute_interrater_agreement(long, build_case_summary(long), ("R01", "R02", "R03"))

    assert summary["icc_status"].isin({"estimated", "estimated_ci_undefined"}).all()
    assert summary["icc_2_1"].to_numpy() == pytest.approx([1.0, 1.0])
    assert summary["icc_2_k"].to_numpy() == pytest.approx([1.0, 1.0])


def test_image_report_alignment_and_bootstrap_are_correct_and_deterministic() -> None:
    case_summary = pd.DataFrame(
        {
            "case_id": ["a", "b", "c"],
            "n_image_raters": [3, 3, 3],
            "n_report_raters": [3, 3, 3],
            "image_mean_0_100": [10.0, 20.0, 30.0],
            "report_mean_0_100": [20.0, 10.0, 50.0],
        }
    )
    first, paired = compute_image_report_alignment(
        case_summary, expected_raters=3, n_boot=50, seed=123
    )
    second, _ = compute_image_report_alignment(case_summary, expected_raters=3, n_boot=50, seed=123)

    pd.testing.assert_frame_equal(first, second)
    assert _metric(first, "mean_signed_difference") == pytest.approx(20 / 3)
    assert _metric(first, "mean_absolute_difference") == pytest.approx(40 / 3)
    assert _metric(first, "root_mean_squared_difference") == pytest.approx(np.sqrt(200))
    bootstrap = bootstrap_alignment_metrics(paired, n_boot=50, seed=123)
    assert bootstrap == bootstrap_alignment_metrics(paired, n_boot=50, seed=123)


def test_constant_series_returns_undefined_correlation() -> None:
    case_summary = pd.DataFrame(
        {
            "case_id": ["a", "b", "c"],
            "n_image_raters": [3, 3, 3],
            "n_report_raters": [3, 3, 3],
            "image_mean_0_100": [50.0, 50.0, 50.0],
            "report_mean_0_100": [10.0, 20.0, 30.0],
        }
    )
    summary, _ = compute_image_report_alignment(case_summary, expected_raters=3, n_boot=10, seed=1)

    pearson = summary[summary["metric"] == "pearson_correlation"].iloc[0]
    assert pd.isna(pearson["estimate"])
    assert pearson["status"] == "undefined"


def test_disagreement_review_contains_no_text_columns() -> None:
    long = _long_ratings()
    review = build_case_disagreement_review(long, build_case_summary(long), limit=2)

    assert set(review["flag_type"]) == {
        "largest_image_interrater_range",
        "largest_report_interrater_range",
        "largest_absolute_report_image_difference",
    }
    assert not any("text" in column.lower() for column in review.columns)


def test_full_run_writes_text_free_outputs_and_relative_manifest(tmp_path: Path) -> None:
    config = _temp_config(tmp_path, source_text="RESTRICTED_SENTINEL_REPORT_TEXT")
    result = run_annotation_pilot(config)

    expected = {
        "ratings_long.parquet",
        "case_summary.parquet",
        "input_qa_summary.csv",
        "interrater_agreement_summary.csv",
        "image_report_alignment_summary.csv",
        "case_disagreement_review.csv",
        "output_manifest.csv",
    }
    generated = {
        path.name
        for directory in (config.outputs.derived_dir, config.outputs.artifact_dir)
        for path in directory.iterdir()
    }
    assert expected.issubset(generated)
    assert (
        result.output_manifest["relative_path"].map(lambda value: not value.startswith("/")).all()
    )
    for path in config.outputs.artifact_dir.glob("*.csv"):
        assert "RESTRICTED_SENTINEL_REPORT_TEXT" not in path.read_text(encoding="utf-8")
    assert (
        "interpretation_text"
        not in pd.read_parquet(config.outputs.derived_dir / "ratings_long.parquet").columns
    )


@pytest.mark.parametrize(
    ("field", "invalid_path", "expected_root"),
    [
        ("report_dir", "docs/reports", "reports"),
        ("artifact_dir", "docs/artifacts", "artifacts"),
        ("derived_dir", "derived", "data/derived"),
    ],
)
def test_output_validation_rejects_wrong_repository_root(
    tmp_path: Path,
    field: str,
    invalid_path: str,
    expected_root: str,
) -> None:
    config = _temp_config(tmp_path)
    outputs = {
        "report_dir": config.outputs.report_dir,
        "artifact_dir": config.outputs.artifact_dir,
        "derived_dir": config.outputs.derived_dir,
    }
    outputs[field] = tmp_path / invalid_path
    invalid = AnnotationPilotConfig(
        project=config.project,
        inputs=config.inputs,
        columns=config.columns,
        outputs=PilotOutputConfig(**outputs),
        repo_root=config.repo_root,
    )

    with pytest.raises(ValueError, match=expected_root):
        validate_annotation_pilot_output_paths(invalid)


def test_output_validation_rejects_symlink_escape(tmp_path: Path) -> None:
    config = _temp_config(tmp_path)
    outside = tmp_path.parent / f"{tmp_path.name}-outside"
    outside.mkdir()
    config.outputs.artifact_dir.parent.mkdir(parents=True, exist_ok=True)
    config.outputs.artifact_dir.symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="outputs.artifact_dir"):
        validate_annotation_pilot_output_paths(config)


def _temp_config(tmp_path: Path, *, source_text: str = "") -> AnnotationPilotConfig:
    inputs = {rater_id: tmp_path / f"{rater_id}.csv" for rater_id in ("R01", "R02", "R03")}
    for offset, (_rater_id, path) in enumerate(inputs.items()):
        frame = _source_frame(offset=offset, source_text=source_text)
        frame.to_csv(path, index=False)
    return AnnotationPilotConfig(
        project=PilotProjectConfig(
            expected_raters=3,
            complete_status_value=2,
            seed=123,
            bootstrap_replicates=20,
        ),
        inputs=inputs,
        columns=PilotColumnConfig(
            case_id="id_accession",
            record_id="id",
            secondary_record_id="id2",
            report_text="interpretation_text",
            raw_data_complete="raw_data_complete",
            report_rating="assessment_report",
            report_complete="adjudication_interpretation_complete",
            image_rating="assessment_image",
            image_complete="adjudication_image_complete",
        ),
        outputs=PilotOutputConfig(
            report_dir=tmp_path / "reports" / "annotation_pilot",
            artifact_dir=tmp_path / "artifacts" / "annotations" / "pilot",
            derived_dir=tmp_path / "data" / "derived" / "annotations" / "pilot",
        ),
        repo_root=tmp_path,
    )


def _source_frame(offset: int = 0, source_text: str = "") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": [f"record-{offset}-1", f"record-{offset}-2", f"record-{offset}-3"],
            "id2": ["secondary-1", "secondary-2", "secondary-3"],
            "id_accession": ["000001", "000002", "000003"],
            "interpretation_text": [source_text] * 3,
            "raw_data_complete": ["2", "2", "2"],
            "assessment_report": [10 + offset, 50 + offset, 80 + offset],
            "adjudication_interpretation_complete": ["2", "2", "2"],
            "assessment_image": [5 + offset, 45 + offset, 85 + offset],
            "adjudication_image_complete": ["2", "2", "2"],
        }
    )


def _long_ratings() -> pd.DataFrame:
    rows = []
    values = {
        "case_a": {
            "R01": (10, 15),
            "R02": (20, 25),
            "R03": (30, 35),
        },
        "case_b": {
            "R01": (40, 35),
            "R02": (50, 55),
            "R03": (60, 65),
        },
    }
    for case_id, raters in values.items():
        for rater_id, (image, report) in raters.items():
            rows.extend(
                [
                    _long_row(case_id, rater_id, "image", image),
                    _long_row(case_id, rater_id, "report", report),
                ]
            )
    return pd.DataFrame(rows)


def _long_row(case_id: str, rater_id: str, task: str, rating: float) -> dict[str, object]:
    return {
        "case_id": case_id,
        "rater_id": rater_id,
        "task": task,
        "rating_0_100": float(rating),
        "rating_probability": float(rating) / 100.0,
        "instrument_complete": True,
    }


def _metric(summary: pd.DataFrame, metric: str) -> float:
    return float(summary.loc[summary["metric"] == metric, "estimate"].iloc[0])
