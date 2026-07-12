from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from statistics import NormalDist
from typing import Any

import pandas as pd
import yaml

from .config import ensure_dir
from .paths import get_paths

TASKS = ("image", "report")
VALID_ANALYSIS_MODES = {"real", "synthetic"}
FORBIDDEN_OUTPUT_COLUMNS = {
    "case_id",
    "rater_id",
    "accession_id",
    "accession_number",
    "report_text",
    "interpretation_text",
}


@dataclass(frozen=True)
class PlanningAssumptions:
    prevalence_grid: tuple[float, ...]
    expected_performance_grid: tuple[float, ...]
    ci_half_width_grid: tuple[float, ...]
    reliability_targets: tuple[float, ...]
    overlap_fraction: float
    disagreement_threshold_points: int
    retraining_case_counts: tuple[int, ...]
    image_minutes_per_rating: float | None
    report_minutes_per_rating: float | None


@dataclass(frozen=True)
class PlanningOutputConfig:
    report_dir: Path
    artifact_dir: Path


@dataclass(frozen=True)
class AnnotationPlanningConfig:
    analysis_mode: str
    confidence_level: float
    pilot_artifact_dir: Path
    assumptions: PlanningAssumptions
    outputs: PlanningOutputConfig
    repo_root: Path


@dataclass(frozen=True)
class AnnotationPlanningResult:
    pilot_summary: pd.DataFrame
    reliability_requirements: pd.DataFrame
    validation_precision_grid: pd.DataFrame
    workload_scenarios: pd.DataFrame
    image_report_alignment_summary: pd.DataFrame
    output_manifest: pd.DataFrame
    analysis_mode: str


def default_annotation_planning_config_path() -> Path:
    raw = os.environ.get("ANNOTATION_PLANNING_CONFIG", "config/annotation_planning.yaml")
    path = Path(raw)
    return path if path.is_absolute() else (get_paths().root / path).resolve()


def load_annotation_planning_config(
    config_path: Path | None = None,
) -> AnnotationPlanningConfig:
    path = (config_path or default_annotation_planning_config_path()).resolve()
    if not path.is_file():
        raise FileNotFoundError(
            "Annotation planning config not found. Create config/annotation_planning.yaml from "
            "the example or set ANNOTATION_PLANNING_CONFIG."
        )
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Annotation planning config must be a YAML mapping")

    root = get_paths().root
    project = _mapping(raw, "project")
    inputs = _mapping(raw, "inputs")
    assumptions_raw = _mapping(raw, "assumptions")
    outputs_raw = _mapping(raw, "outputs")

    analysis_mode = str(project.get("analysis_mode", "real")).strip().lower()
    if analysis_mode not in VALID_ANALYSIS_MODES:
        raise ValueError("project.analysis_mode must be real or synthetic")
    confidence_level = float(project.get("confidence_level", 0.95))
    if not 0.5 < confidence_level < 1:
        raise ValueError("project.confidence_level must be between 0.5 and 1")

    assumptions = PlanningAssumptions(
        prevalence_grid=_float_tuple(assumptions_raw, "prevalence_grid", lower=0, upper=1),
        expected_performance_grid=_float_tuple(
            assumptions_raw, "expected_performance_grid", lower=0, upper=1
        ),
        ci_half_width_grid=_float_tuple(assumptions_raw, "ci_half_width_grid", lower=0, upper=0.5),
        reliability_targets=_float_tuple(assumptions_raw, "reliability_targets", lower=0, upper=1),
        overlap_fraction=float(assumptions_raw.get("overlap_fraction", 0.2)),
        disagreement_threshold_points=int(assumptions_raw.get("disagreement_threshold_points", 25)),
        retraining_case_counts=_int_tuple(assumptions_raw, "retraining_case_counts"),
        image_minutes_per_rating=_optional_positive_float(
            assumptions_raw.get("image_minutes_per_rating"),
            "assumptions.image_minutes_per_rating",
        ),
        report_minutes_per_rating=_optional_positive_float(
            assumptions_raw.get("report_minutes_per_rating"),
            "assumptions.report_minutes_per_rating",
        ),
    )
    if not 0 <= assumptions.overlap_fraction <= 1:
        raise ValueError("assumptions.overlap_fraction must be in [0, 1]")
    if assumptions.disagreement_threshold_points not in {25, 50}:
        raise ValueError("assumptions.disagreement_threshold_points must be 25 or 50")

    config = AnnotationPlanningConfig(
        analysis_mode=analysis_mode,
        confidence_level=confidence_level,
        pilot_artifact_dir=_resolve_path(inputs.get("pilot_artifact_dir"), root),
        assumptions=assumptions,
        outputs=PlanningOutputConfig(
            report_dir=_resolve_path(outputs_raw.get("report_dir"), root),
            artifact_dir=_resolve_path(outputs_raw.get("artifact_dir"), root),
        ),
        repo_root=root,
    )
    validate_annotation_planning_output_paths(config)
    if config.analysis_mode == "real" and "smoke" in config.pilot_artifact_dir.parts:
        raise ValueError("Real annotation planning cannot consume smoke pilot artifacts")
    return config


def run_annotation_planning(config: AnnotationPlanningConfig) -> AnnotationPlanningResult:
    validate_annotation_planning_output_paths(config)
    if config.analysis_mode == "real" and "smoke" in config.pilot_artifact_dir.parts:
        raise ValueError("Real annotation planning cannot consume smoke pilot artifacts")
    interrater, input_qa, alignment = load_pilot_planning_inputs(config.pilot_artifact_dir)
    pilot_summary = build_pilot_planning_summary(
        interrater,
        input_qa,
        disagreement_threshold_points=config.assumptions.disagreement_threshold_points,
    )
    reliability = build_reliability_requirements(
        pilot_summary,
        targets=config.assumptions.reliability_targets,
    )
    precision = build_validation_precision_grid(
        pilot_summary,
        prevalence_grid=config.assumptions.prevalence_grid,
        expected_performance_grid=config.assumptions.expected_performance_grid,
        ci_half_width_grid=config.assumptions.ci_half_width_grid,
        confidence_level=config.confidence_level,
    )
    workload = build_workload_scenarios(
        pilot_summary,
        precision,
        expected_raters=int(pilot_summary["expected_raters"].iloc[0]),
        assumptions=config.assumptions,
    )
    for frame in (pilot_summary, reliability, precision, workload, alignment):
        assert_aggregate_planning_output(frame)

    ensure_dir(config.outputs.artifact_dir)
    output_frames = {
        "pilot_planning_summary": pilot_summary,
        "rater_reliability_requirements": reliability,
        "validation_precision_grid": precision,
        "workload_scenarios": workload,
        "image_report_alignment_summary": alignment,
    }
    manifest_rows: list[dict[str, Any]] = []
    for name, frame in output_frames.items():
        path = config.outputs.artifact_dir / f"{name}.csv"
        frame.to_csv(path, index=False)
        manifest_rows.append(
            {
                "artifact_name": name,
                "relative_path": _relative_path(path, config.repo_root),
                "row_count": int(len(frame)),
                "aggregate_only": True,
            }
        )

    summary_path = config.outputs.artifact_dir / "planning_summary.json"
    summary = {
        "analysis_mode": config.analysis_mode,
        "confidence_level": config.confidence_level,
        "pilot_artifact_dir": _relative_path(config.pilot_artifact_dir, config.repo_root),
        "validation_scenarios": int(len(precision)),
        "workload_scenarios": int(len(workload)),
        "timing_available": bool(
            config.assumptions.image_minutes_per_rating is not None
            and config.assumptions.report_minutes_per_rating is not None
        ),
        "interpretation": (
            "illustrative synthetic output"
            if config.analysis_mode == "synthetic"
            else "planning estimates from configured real pilot aggregates"
        ),
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    manifest_rows.append(
        {
            "artifact_name": "planning_summary",
            "relative_path": _relative_path(summary_path, config.repo_root),
            "row_count": None,
            "aggregate_only": True,
        }
    )
    report_path = config.outputs.report_dir / "ards_annotation_design_scenarios.html"
    manifest_rows.append(
        {
            "artifact_name": "rendered_report",
            "relative_path": _relative_path(report_path, config.repo_root),
            "row_count": None,
            "aggregate_only": True,
        }
    )
    manifest = pd.DataFrame(manifest_rows)
    manifest_path = config.outputs.artifact_dir / "output_manifest.csv"
    manifest.loc[len(manifest)] = {
        "artifact_name": "output_manifest",
        "relative_path": _relative_path(manifest_path, config.repo_root),
        "row_count": len(manifest) + 1,
        "aggregate_only": True,
    }
    manifest.to_csv(manifest_path, index=False)

    return AnnotationPlanningResult(
        pilot_summary=pilot_summary,
        reliability_requirements=reliability,
        validation_precision_grid=precision,
        workload_scenarios=workload,
        image_report_alignment_summary=alignment,
        output_manifest=manifest,
        analysis_mode=config.analysis_mode,
    )


def load_pilot_planning_inputs(
    artifact_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    paths = {
        "interrater": artifact_dir / "interrater_agreement_summary.csv",
        "input_qa": artifact_dir / "input_qa_summary.csv",
        "alignment": artifact_dir / "image_report_alignment_summary.csv",
    }
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Annotation pilot aggregate outputs are missing: "
            + ", ".join(missing)
            + ". Run the annotation pilot first."
        )
    interrater = pd.read_csv(paths["interrater"])
    input_qa = pd.read_csv(paths["input_qa"])
    alignment = pd.read_csv(paths["alignment"])
    _require_columns(
        interrater,
        {
            "task",
            "expected_raters",
            "n_cases_with_any_completed_rating",
            "n_all_rater_complete_cases",
            "proportion_case_range_ge_25",
            "proportion_case_range_ge_50",
            "icc_2_1",
            "icc_status",
        },
        "interrater agreement summary",
    )
    _require_columns(input_qa, {"check", "value"}, "input QA summary")
    if set(interrater["task"].astype(str)) != set(TASKS):
        raise ValueError("Interrater summary must contain exactly image and report tasks")
    return interrater, input_qa, alignment


def build_pilot_planning_summary(
    interrater: pd.DataFrame,
    input_qa: pd.DataFrame,
    *,
    disagreement_threshold_points: int,
) -> pd.DataFrame:
    qa = input_qa.assign(check=input_qa["check"].astype(str)).set_index("check")["value"]
    expected_raters = _qa_int(qa, "configured_raters")
    if expected_raters != 3:
        raise ValueError("The v1 annotation planner requires the configured three-rater pilot")
    unique_cases = _qa_int(qa, "unique_cases")
    if unique_cases <= 0:
        raise ValueError("Pilot unique case count must be positive")

    disagreement_column = f"proportion_case_range_ge_{disagreement_threshold_points}"
    rows: list[dict[str, Any]] = []
    for task in TASKS:
        row = interrater.loc[interrater["task"].astype(str) == task].iloc[0]
        completed_ratings = _qa_int(qa, f"completed_{task}_ratings")
        rating_completion_rate = completed_ratings / (unique_cases * expected_raters)
        all_rater_complete_cases = int(row["n_all_rater_complete_cases"])
        case_completion_rate = all_rater_complete_cases / unique_cases
        disagreement_rate = pd.to_numeric(
            pd.Series([row[disagreement_column]]), errors="coerce"
        ).iloc[0]
        if pd.isna(disagreement_rate):
            raise ValueError(f"Pilot {task} disagreement rate is missing")
        if not 0 <= rating_completion_rate <= 1:
            raise ValueError(f"Pilot {task} rating completion rate must be in [0, 1]")
        if not 0 <= case_completion_rate <= 1:
            raise ValueError(f"Pilot {task} all-rater case completion rate must be in [0, 1]")
        if not 0 <= disagreement_rate <= 1:
            raise ValueError(f"Pilot {task} disagreement rate must be in [0, 1]")
        rows.append(
            {
                "task": task,
                "expected_raters": expected_raters,
                "unique_pilot_cases": unique_cases,
                "completed_ratings": completed_ratings,
                "rating_completion_rate": float(rating_completion_rate),
                "all_rater_complete_cases": all_rater_complete_cases,
                "case_completion_rate": float(case_completion_rate),
                "disagreement_threshold_points": disagreement_threshold_points,
                "disagreement_rate": float(disagreement_rate),
                "icc_2_1": pd.to_numeric(pd.Series([row["icc_2_1"]]), errors="coerce").iloc[0],
                "icc_status": str(row["icc_status"]),
            }
        )
    return pd.DataFrame(rows)


def build_reliability_requirements(
    pilot_summary: pd.DataFrame,
    *,
    targets: tuple[float, ...],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, pilot in pilot_summary.iterrows():
        icc = float(pilot["icc_2_1"]) if pd.notna(pilot["icc_2_1"]) else math.nan
        for target in targets:
            rows.append(
                {
                    "task": pilot["task"],
                    "single_rater_icc_2_1": icc,
                    "target_reliability": target,
                    "required_raters": required_raters_for_reliability(icc, target),
                    "status": (
                        "estimated" if math.isfinite(icc) and 0 < icc <= 1 else "unavailable"
                    ),
                }
            )
    return pd.DataFrame(rows)


def build_validation_precision_grid(
    pilot_summary: pd.DataFrame,
    *,
    prevalence_grid: tuple[float, ...],
    expected_performance_grid: tuple[float, ...],
    ci_half_width_grid: tuple[float, ...],
    confidence_level: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, pilot in pilot_summary.iterrows():
        completion_rate = float(pilot["case_completion_rate"])
        if not 0 < completion_rate <= 1:
            raise ValueError(f"Pilot {pilot['task']} completion rate must be in (0, 1]")
        for prevalence in prevalence_grid:
            for performance in expected_performance_grid:
                for half_width in ci_half_width_grid:
                    positive_n = required_binomial_denominator(
                        performance,
                        half_width,
                        confidence_level=confidence_level,
                    )
                    negative_n = positive_n
                    complete_cases = max(
                        math.ceil(positive_n / prevalence),
                        math.ceil(negative_n / (1 - prevalence)),
                    )
                    invited_cases = math.ceil(complete_cases / completion_rate)
                    scenario_id = (
                        f"prev_{prevalence:.2f}_perf_{performance:.2f}_hw_{half_width:.2f}"
                    )
                    rows.append(
                        {
                            "task": pilot["task"],
                            "scenario_id": scenario_id,
                            "confidence_level": confidence_level,
                            "assumed_prevalence": prevalence,
                            "expected_sensitivity": performance,
                            "expected_specificity": performance,
                            "target_ci_half_width": half_width,
                            "required_positive_cases": positive_n,
                            "required_negative_cases": negative_n,
                            "required_complete_cases": complete_cases,
                            "completion_rate_used": completion_rate,
                            "planned_invited_cases": invited_cases,
                            "method": "expected_wilson_interval",
                            "endpoint_scope": "secondary_binary_validation",
                        }
                    )
    return pd.DataFrame(rows)


def build_workload_scenarios(
    pilot_summary: pd.DataFrame,
    validation_grid: pd.DataFrame,
    *,
    expected_raters: int,
    assumptions: PlanningAssumptions,
) -> pd.DataFrame:
    pilot_by_task = pilot_summary.set_index("task")
    rows: list[dict[str, Any]] = []
    for _, validation in validation_grid.iterrows():
        task = str(validation["task"])
        rows.extend(
            _workload_rows(
                task=task,
                planning_family="validation",
                scenario_id=str(validation["scenario_id"]),
                planned_complete_cases=int(validation["required_complete_cases"]),
                planned_invited_cases=int(validation["planned_invited_cases"]),
                expected_raters=expected_raters,
                completion_rate=float(pilot_by_task.loc[task, "case_completion_rate"]),
                disagreement_rate=float(pilot_by_task.loc[task, "disagreement_rate"]),
                assumptions=assumptions,
            )
        )
    for task in TASKS:
        pilot = pilot_by_task.loc[task]
        for case_count in assumptions.retraining_case_counts:
            invited = math.ceil(case_count / float(pilot["case_completion_rate"]))
            rows.extend(
                _workload_rows(
                    task=task,
                    planning_family="retraining_workload_example",
                    scenario_id=f"retraining_{case_count}",
                    planned_complete_cases=case_count,
                    planned_invited_cases=invited,
                    expected_raters=expected_raters,
                    completion_rate=float(pilot["case_completion_rate"]),
                    disagreement_rate=float(pilot["disagreement_rate"]),
                    assumptions=assumptions,
                )
            )
    task_rows = pd.DataFrame(rows)
    combined_rows: list[dict[str, Any]] = []
    group_columns = ["planning_family", "scenario_id", "review_design"]
    for keys, group in task_rows.groupby(group_columns, sort=True):
        if set(group["task"]) != set(TASKS):
            continue
        hours = group["reviewer_hours"].sum(min_count=len(TASKS))
        combined_rows.append(
            {
                "task": "both",
                "planning_family": keys[0],
                "scenario_id": keys[1],
                "review_design": keys[2],
                "planned_complete_cases": int(group["planned_complete_cases"].max()),
                "planned_invited_cases": int(group["planned_invited_cases"].max()),
                "completion_rate_used": float(group["completion_rate_used"].min()),
                "disagreement_rate_used": None,
                "overlap_fraction_used": float(group["overlap_fraction_used"].iloc[0]),
                "disagreement_threshold_points_used": int(
                    group["disagreement_threshold_points_used"].iloc[0]
                ),
                "person_ratings": int(group["person_ratings"].sum()),
                "reviewer_hours": None if pd.isna(hours) else float(hours),
                "interpretation": group["interpretation"].iloc[0],
            }
        )
    return pd.concat([task_rows, pd.DataFrame(combined_rows)], ignore_index=True).sort_values(
        ["planning_family", "scenario_id", "task", "review_design"], kind="stable"
    )


def required_binomial_denominator(
    expected_probability: float,
    target_half_width: float,
    *,
    confidence_level: float = 0.95,
) -> int:
    if not 0.5 < confidence_level < 1:
        raise ValueError("confidence_level must be between 0.5 and 1")
    if not 0 < expected_probability < 1:
        raise ValueError("expected_probability must be in (0, 1)")
    if not 0 < target_half_width < 0.5:
        raise ValueError("target_half_width must be in (0, 0.5)")
    for denominator in range(2, 1_000_001):
        if (
            wilson_half_width(
                expected_probability,
                denominator,
                confidence_level=confidence_level,
            )
            <= target_half_width
        ):
            return denominator
    raise ValueError("Required denominator exceeds planner limit")


def wilson_half_width(
    expected_probability: float,
    denominator: int,
    *,
    confidence_level: float = 0.95,
) -> float:
    if denominator <= 0:
        raise ValueError("denominator must be positive")
    if not 0.5 < confidence_level < 1:
        raise ValueError("confidence_level must be between 0.5 and 1")
    z = NormalDist().inv_cdf(0.5 + confidence_level / 2)
    z2 = z * z
    return (
        z
        / (1 + z2 / denominator)
        * math.sqrt(
            expected_probability * (1 - expected_probability) / denominator
            + z2 / (4 * denominator * denominator)
        )
    )


def required_raters_for_reliability(single_rater_icc: float, target: float) -> int | None:
    if not 0 < target < 1:
        raise ValueError("target reliability must be in (0, 1)")
    if not math.isfinite(single_rater_icc) or not 0 < single_rater_icc <= 1:
        return None
    if single_rater_icc >= target:
        return 1
    required = target * (1 - single_rater_icc) / (single_rater_icc * (1 - target))
    return max(1, math.ceil(required - 1e-12))


def assert_aggregate_planning_output(frame: pd.DataFrame) -> None:
    unsafe = []
    for column in frame.columns:
        lowered = str(column).lower()
        if lowered in FORBIDDEN_OUTPUT_COLUMNS:
            unsafe.append(str(column))
        elif lowered.endswith(("_text", "_path", "_file")):
            unsafe.append(str(column))
    if unsafe:
        raise ValueError(f"Planning output contains row-level or path-bearing columns: {unsafe}")


def validate_annotation_planning_output_paths(config: AnnotationPlanningConfig) -> None:
    requirements = (
        ("outputs.report_dir", config.outputs.report_dir, Path("reports")),
        ("outputs.artifact_dir", config.outputs.artifact_dir, Path("artifacts")),
    )
    root = config.repo_root.resolve()
    for name, path, relative_root in requirements:
        allowed = (root / relative_root).resolve()
        candidate = path.resolve()
        if not allowed.is_relative_to(root) or not candidate.is_relative_to(allowed):
            raise ValueError(f"{name} must remain under {relative_root.as_posix()}")


def _workload_rows(
    *,
    task: str,
    planning_family: str,
    scenario_id: str,
    planned_complete_cases: int,
    planned_invited_cases: int,
    expected_raters: int,
    completion_rate: float,
    disagreement_rate: float,
    assumptions: PlanningAssumptions,
) -> list[dict[str, Any]]:
    overlap_cases = math.ceil(planned_invited_cases * assumptions.overlap_fraction)
    disagreement_cases = math.ceil(planned_invited_cases * disagreement_rate)
    designs = {
        "full_three_rater_review": expected_raters * planned_invited_cases,
        "single_review_plus_overlap": (
            planned_invited_cases + (expected_raters - 1) * overlap_cases
        ),
        "double_review_plus_disagreement_triggered_third": (
            2 * planned_invited_cases + disagreement_cases
        ),
    }
    minutes = (
        assumptions.image_minutes_per_rating
        if task == "image"
        else assumptions.report_minutes_per_rating
    )
    interpretation = (
        "validation precision planning"
        if planning_family == "validation"
        else "illustrative workload only; not a sufficient training sample claim"
    )
    return [
        {
            "task": task,
            "planning_family": planning_family,
            "scenario_id": scenario_id,
            "review_design": design,
            "planned_complete_cases": planned_complete_cases,
            "planned_invited_cases": planned_invited_cases,
            "completion_rate_used": completion_rate,
            "disagreement_rate_used": disagreement_rate,
            "overlap_fraction_used": assumptions.overlap_fraction,
            "disagreement_threshold_points_used": assumptions.disagreement_threshold_points,
            "person_ratings": int(person_ratings),
            "reviewer_hours": None if minutes is None else person_ratings * minutes / 60,
            "interpretation": interpretation,
        }
        for design, person_ratings in designs.items()
    ]


def _mapping(raw: dict[str, Any], key: str) -> dict[str, Any]:
    value = raw.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"Annotation planning config requires a {key} mapping")
    return value


def _float_tuple(raw: dict[str, Any], key: str, *, lower: float, upper: float) -> tuple[float, ...]:
    values = raw.get(key)
    if not isinstance(values, list) or not values:
        raise ValueError(f"assumptions.{key} must be a non-empty list")
    parsed = tuple(float(value) for value in values)
    if any(not lower < value < upper for value in parsed):
        raise ValueError(f"assumptions.{key} values must be between {lower} and {upper}")
    return parsed


def _int_tuple(raw: dict[str, Any], key: str) -> tuple[int, ...]:
    values = raw.get(key)
    if not isinstance(values, list) or not values:
        raise ValueError(f"assumptions.{key} must be a non-empty list")
    parsed = tuple(int(value) for value in values)
    if any(value <= 0 for value in parsed):
        raise ValueError(f"assumptions.{key} values must be positive")
    return parsed


def _optional_positive_float(value: Any, name: str) -> float | None:
    if value in {None, ""}:
        return None
    parsed = float(value)
    if parsed <= 0:
        raise ValueError(f"{name} must be positive when configured")
    return parsed


def _resolve_path(value: Any, root: Path) -> Path:
    if value is None or not str(value).strip():
        raise ValueError("Annotation planning paths must be nonblank")
    path = Path(str(value))
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _qa_int(qa: pd.Series, key: str) -> int:
    if key not in qa.index:
        raise ValueError(f"Input QA summary is missing {key}")
    try:
        return int(float(qa.loc[key]))
    except (TypeError, ValueError) as error:
        raise ValueError(f"Input QA value for {key} is not numeric") from error


def _require_columns(frame: pd.DataFrame, required: set[str], name: str) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing columns: {missing}")


def _relative_path(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as error:
        raise ValueError("Annotation planning paths must remain inside the repository") from error
