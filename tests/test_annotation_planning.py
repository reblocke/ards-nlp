from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
import pytest

from ards_cxr_benchmark.annotation_planning import (
    AnnotationPlanningConfig,
    PlanningAssumptions,
    PlanningOutputConfig,
    assert_aggregate_planning_output,
    required_binomial_denominator,
    required_raters_for_reliability,
    run_annotation_planning,
    wilson_half_width,
)


def test_required_binomial_denominator_meets_requested_wilson_precision() -> None:
    denominator = required_binomial_denominator(0.8, 0.05)

    assert wilson_half_width(0.8, denominator) <= 0.05
    assert wilson_half_width(0.8, denominator - 1) > 0.05


def test_required_raters_uses_spearman_brown_relationship() -> None:
    assert required_raters_for_reliability(0.5, 0.8) == 4
    assert required_raters_for_reliability(0.85, 0.8) == 1
    assert required_raters_for_reliability(math.nan, 0.8) is None
    assert required_raters_for_reliability(1.1, 0.8) is None


def test_planner_writes_aggregate_synthetic_scenarios(tmp_path: Path) -> None:
    artifact_input = tmp_path / "artifacts" / "annotations" / "pilot"
    artifact_input.mkdir(parents=True)
    _write_pilot_inputs(artifact_input)
    config = _config(tmp_path, artifact_input)

    result = run_annotation_planning(config)

    assert result.analysis_mode == "synthetic"
    assert set(result.pilot_summary["task"]) == {"image", "report"}
    assert set(result.workload_scenarios["task"]) == {"image", "report", "both"}
    assert set(result.workload_scenarios["review_design"]) == {
        "full_three_rater_review",
        "single_review_plus_overlap",
        "double_review_plus_disagreement_triggered_third",
    }
    assert result.workload_scenarios["reviewer_hours"].isna().all()
    assert (config.outputs.artifact_dir / "planning_summary.json").is_file()
    assert (config.outputs.artifact_dir / "output_manifest.csv").is_file()
    for frame in (
        result.pilot_summary,
        result.reliability_requirements,
        result.validation_precision_grid,
        result.workload_scenarios,
    ):
        assert_aggregate_planning_output(frame)


def test_planner_rejects_row_level_output_columns() -> None:
    with pytest.raises(ValueError, match="row-level"):
        assert_aggregate_planning_output(pd.DataFrame({"case_id": ["restricted"]}))


def test_planner_requires_three_rater_pilot(tmp_path: Path) -> None:
    artifact_input = tmp_path / "artifacts" / "annotations" / "pilot"
    artifact_input.mkdir(parents=True)
    _write_pilot_inputs(artifact_input, expected_raters=2)

    with pytest.raises(ValueError, match="three-rater"):
        run_annotation_planning(_config(tmp_path, artifact_input))


def test_real_planner_rejects_smoke_artifacts(tmp_path: Path) -> None:
    artifact_input = tmp_path / "artifacts" / "annotations" / "pilot" / "smoke"
    artifact_input.mkdir(parents=True)
    _write_pilot_inputs(artifact_input)
    base = _config(tmp_path, artifact_input)
    config = AnnotationPlanningConfig(
        analysis_mode="real",
        confidence_level=base.confidence_level,
        pilot_artifact_dir=base.pilot_artifact_dir,
        assumptions=base.assumptions,
        outputs=base.outputs,
        repo_root=base.repo_root,
    )

    with pytest.raises(ValueError, match="cannot consume smoke"):
        run_annotation_planning(config)


def _config(root: Path, artifact_input: Path) -> AnnotationPlanningConfig:
    return AnnotationPlanningConfig(
        analysis_mode="synthetic",
        confidence_level=0.95,
        pilot_artifact_dir=artifact_input,
        assumptions=PlanningAssumptions(
            prevalence_grid=(0.1, 0.2, 0.3),
            expected_performance_grid=(0.8, 0.9),
            ci_half_width_grid=(0.05, 0.1),
            reliability_targets=(0.8, 0.9),
            overlap_fraction=0.2,
            disagreement_threshold_points=25,
            retraining_case_counts=(250, 500, 1000),
            image_minutes_per_rating=None,
            report_minutes_per_rating=None,
        ),
        outputs=PlanningOutputConfig(
            report_dir=root / "reports" / "annotation_planning" / "smoke",
            artifact_dir=root / "artifacts" / "annotations" / "planning" / "smoke",
        ),
        repo_root=root,
    )


def _write_pilot_inputs(path: Path, expected_raters: int = 3) -> None:
    pd.DataFrame(
        {
            "task": ["image", "report"],
            "expected_raters": [expected_raters, expected_raters],
            "n_cases_with_any_completed_rating": [50, 50],
            "n_all_rater_complete_cases": [42, 45],
            "proportion_case_range_ge_25": [0.2, 0.1],
            "proportion_case_range_ge_50": [0.05, 0.02],
            "icc_2_1": [0.6, 0.7],
            "icc_status": ["estimated", "estimated"],
        }
    ).to_csv(path / "interrater_agreement_summary.csv", index=False)
    pd.DataFrame(
        {
            "check": [
                "configured_raters",
                "unique_cases",
                "completed_image_ratings",
                "completed_report_ratings",
            ],
            "status": ["pass"] * 4,
            "value": [expected_raters, 50, 140, 145],
        }
    ).to_csv(path / "input_qa_summary.csv", index=False)
    pd.DataFrame(
        {
            "analysis": ["case_mean_report_vs_image"],
            "metric": ["mean_absolute_difference"],
            "estimate": [8.0],
            "n_cases": [45],
        }
    ).to_csv(path / "image_report_alignment_summary.csv", index=False)
