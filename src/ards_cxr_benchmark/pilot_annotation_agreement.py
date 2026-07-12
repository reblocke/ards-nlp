from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
import pingouin as pg
import yaml
from scipy import stats

from .config import ensure_dir, ensure_parent_dir
from .paths import get_paths

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

VALID_COMPLETION_VALUES = {0, 1, 2}
TASKS = ("image", "report")
BOOTSTRAP_METRICS = (
    "mean_signed_difference",
    "mean_absolute_difference",
    "root_mean_squared_difference",
    "pearson_correlation",
    "spearman_correlation",
)


@dataclass(frozen=True)
class PilotProjectConfig:
    expected_raters: int
    complete_status_value: int
    seed: int
    bootstrap_replicates: int


@dataclass(frozen=True)
class PilotColumnConfig:
    case_id: str
    report_rating: str
    report_complete: str
    image_rating: str
    image_complete: str
    record_id: str | None = None
    secondary_record_id: str | None = None
    report_text: str | None = None
    raw_data_complete: str | None = None


@dataclass(frozen=True)
class PilotOutputConfig:
    report_dir: Path
    artifact_dir: Path
    derived_dir: Path


@dataclass(frozen=True)
class AnnotationPilotConfig:
    project: PilotProjectConfig
    inputs: dict[str, Path]
    columns: PilotColumnConfig
    outputs: PilotOutputConfig
    repo_root: Path

    @property
    def rater_ids(self) -> tuple[str, ...]:
        return tuple(self.inputs)


@dataclass(frozen=True)
class PilotAnalysisResult:
    input_qa_summary: pd.DataFrame
    rater_descriptive_summary: pd.DataFrame
    interrater_agreement_summary: pd.DataFrame
    pairwise_rater_agreement: pd.DataFrame
    image_report_alignment_summary: pd.DataFrame
    within_rater_image_report_alignment: pd.DataFrame
    case_coverage_by_rater: pd.DataFrame
    disagreement_review_summary: pd.DataFrame
    output_manifest: pd.DataFrame
    figure_paths: dict[str, Path]


def default_annotation_pilot_config_path() -> Path:
    raw = os.environ.get("ANNOTATION_PILOT_CONFIG", "config/annotation_pilot.yaml")
    path = Path(raw)
    return path if path.is_absolute() else (get_paths().root / path).resolve()


def load_annotation_pilot_config(config_path: Path | None = None) -> AnnotationPilotConfig:
    path = (config_path or default_annotation_pilot_config_path()).resolve()
    if not path.exists():
        raise FileNotFoundError(
            "Annotation pilot config not found. Create config/annotation_pilot.yaml from the "
            "example or set ANNOTATION_PILOT_CONFIG."
        )
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Annotation pilot config must be a YAML mapping")

    root = get_paths().root
    project_raw = _required_mapping(raw, "project")
    inputs_raw = _required_mapping(raw, "inputs")
    columns_raw = _required_mapping(raw, "columns")
    outputs_raw = _required_mapping(raw, "outputs")

    project = PilotProjectConfig(
        expected_raters=int(project_raw.get("expected_raters", 3)),
        complete_status_value=int(project_raw.get("complete_status_value", 2)),
        seed=int(project_raw.get("seed", 20260710)),
        bootstrap_replicates=int(project_raw.get("bootstrap_replicates", 2000)),
    )
    if project.expected_raters < 2:
        raise ValueError("project.expected_raters must be at least 2")
    if project.complete_status_value != 2:
        raise ValueError("project.complete_status_value must be 2 for completed REDCap instruments")
    if project.bootstrap_replicates <= 0:
        raise ValueError("project.bootstrap_replicates must be positive")

    inputs: dict[str, Path] = {}
    for raw_rater_id, raw_path in inputs_raw.items():
        rater_id = str(raw_rater_id).strip()
        if not rater_id:
            raise ValueError("Configured rater IDs must be nonblank")
        if rater_id in inputs:
            raise ValueError(f"Duplicate normalized rater ID: {rater_id}")
        inputs[rater_id] = _resolve_repo_path(raw_path, root)
    if len(inputs) != project.expected_raters:
        raise ValueError(
            "Configured input count must equal project.expected_raters: "
            f"expected {project.expected_raters}, found {len(inputs)}"
        )
    if len({path.resolve() for path in inputs.values()}) != len(inputs):
        raise ValueError("Each configured rater must use a different REDCap export file")

    columns = PilotColumnConfig(
        case_id=_required_name(columns_raw, "case_id"),
        report_rating=_required_name(columns_raw, "report_rating"),
        report_complete=_required_name(columns_raw, "report_complete"),
        image_rating=_required_name(columns_raw, "image_rating"),
        image_complete=_required_name(columns_raw, "image_complete"),
        record_id=_optional_name(columns_raw, "record_id", "id"),
        secondary_record_id=_optional_name(columns_raw, "secondary_record_id", "id2"),
        report_text=_optional_name(columns_raw, "report_text", "interpretation_text"),
        raw_data_complete=_optional_name(columns_raw, "raw_data_complete", "raw_data_complete"),
    )
    required_names = [
        columns.case_id,
        columns.report_rating,
        columns.report_complete,
        columns.image_rating,
        columns.image_complete,
    ]
    if len(set(required_names)) != len(required_names):
        raise ValueError("Required annotation column mappings must be distinct")

    outputs = PilotOutputConfig(
        report_dir=_resolve_repo_path(outputs_raw.get("report_dir"), root),
        artifact_dir=_resolve_repo_path(outputs_raw.get("artifact_dir"), root),
        derived_dir=_resolve_repo_path(outputs_raw.get("derived_dir"), root),
    )
    return AnnotationPilotConfig(
        project=project,
        inputs=inputs,
        columns=columns,
        outputs=outputs,
        repo_root=root,
    )


def load_redcap_rater_exports(config: AnnotationPilotConfig) -> pd.DataFrame:
    missing_raters = [rater_id for rater_id, path in config.inputs.items() if not path.is_file()]
    if missing_raters:
        raise FileNotFoundError(
            "Configured REDCap export files are missing for raters: "
            + ", ".join(sorted(missing_raters))
        )

    frames: list[pd.DataFrame] = []
    for rater_id, path in config.inputs.items():
        source = pd.read_csv(path, dtype=str)
        required = [
            config.columns.case_id,
            config.columns.report_rating,
            config.columns.report_complete,
            config.columns.image_rating,
            config.columns.image_complete,
        ]
        missing = [column for column in required if column not in source.columns]
        if missing:
            raise ValueError(f"REDCap export for {rater_id} is missing columns: {missing}")
        if source.empty:
            raise ValueError(f"REDCap export for {rater_id} has no rows")

        case_id = _normalize_identifier(source[config.columns.case_id])
        blank_case_id = case_id.isna()
        if blank_case_id.any():
            raise ValueError(
                f"REDCap export for {rater_id} has {int(blank_case_id.sum())} blank case IDs"
            )
        duplicate_case = case_id.duplicated(keep=False)
        if duplicate_case.any():
            raise ValueError(
                f"REDCap export for {rater_id} has {int(duplicate_case.sum())} duplicate case rows"
            )

        image_complete = _parse_completion(
            source[config.columns.image_complete],
            rater_id=rater_id,
            column_name=config.columns.image_complete,
        )
        report_complete = _parse_completion(
            source[config.columns.report_complete],
            rater_id=rater_id,
            column_name=config.columns.report_complete,
        )
        image_rating = _parse_rating(
            source[config.columns.image_rating],
            image_complete,
            complete_value=config.project.complete_status_value,
            rater_id=rater_id,
            column_name=config.columns.image_rating,
        )
        report_rating = _parse_rating(
            source[config.columns.report_rating],
            report_complete,
            complete_value=config.project.complete_status_value,
            rater_id=rater_id,
            column_name=config.columns.report_rating,
        )

        standardized = pd.DataFrame(
            {
                "case_id": case_id,
                "rater_id": rater_id,
                "record_id": _optional_identifier(source, config.columns.record_id),
                "secondary_record_id": _optional_identifier(
                    source, config.columns.secondary_record_id
                ),
                "raw_data_complete": _optional_text(source, config.columns.raw_data_complete),
                "image_rating_0_100": image_rating,
                "report_rating_0_100": report_rating,
                "image_complete": image_complete,
                "report_complete": report_complete,
            }
        )
        frames.append(standardized)
        del source

    wide = pd.concat(frames, ignore_index=True)
    validate_annotation_pilot(wide, config)
    return wide.sort_values(["case_id", "rater_id"]).reset_index(drop=True)


def validate_annotation_pilot(wide: pd.DataFrame, config: AnnotationPilotConfig) -> None:
    required = {
        "case_id",
        "rater_id",
        "image_rating_0_100",
        "report_rating_0_100",
        "image_complete",
        "report_complete",
    }
    missing = sorted(required.difference(wide.columns))
    if missing:
        raise ValueError(f"Normalized pilot table is missing columns: {missing}")
    if wide["case_id"].isna().any() or wide["rater_id"].isna().any():
        raise ValueError("Normalized pilot table contains missing case or rater IDs")
    if wide.duplicated(["case_id", "rater_id"]).any():
        raise ValueError("Normalized pilot table contains duplicate case/rater rows")
    if set(wide["rater_id"].unique()) != set(config.rater_ids):
        raise ValueError("Normalized pilot table does not contain every configured rater")

    for task in TASKS:
        status = wide[f"{task}_complete"]
        invalid_status = status.notna() & ~status.isin(VALID_COMPLETION_VALUES)
        if invalid_status.any():
            raise ValueError(f"Normalized pilot table contains invalid {task} completion values")
        completed = status.eq(config.project.complete_status_value).fillna(False)
        rating = wide[f"{task}_rating_0_100"]
        invalid_rating = completed & (rating.isna() | ~rating.between(0, 100))
        if invalid_rating.any():
            raise ValueError(
                f"Normalized pilot table has {int(invalid_rating.sum())} invalid completed "
                f"{task} ratings"
            )
        if not completed.any():
            raise ValueError(f"No completed {task} ratings are available")


def build_task_long_table(
    wide: pd.DataFrame,
    *,
    complete_status_value: int,
    expected_rater_ids: tuple[str, ...],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[pd.DataFrame] = []
    exclusions: list[dict[str, Any]] = []
    for task in TASKS:
        status_column = f"{task}_complete"
        rating_column = f"{task}_rating_0_100"
        eligible = wide[status_column].eq(complete_status_value).fillna(False)
        task_rows = wide.loc[eligible, ["case_id", "rater_id", rating_column]].rename(
            columns={rating_column: "rating_0_100"}
        )
        task_rows["task"] = task
        task_rows["rating_probability"] = task_rows["rating_0_100"] / 100.0
        task_rows["instrument_complete"] = True
        rows.append(task_rows)

        for row in wide.loc[~eligible, ["case_id", "rater_id", status_column]].itertuples(
            index=False
        ):
            status = row[2]
            if pd.isna(status):
                reason = "missing_completion_status"
                status_value: Any = pd.NA
            else:
                status_value = int(status)
                reason = "instrument_incomplete" if status_value == 0 else "instrument_unverified"
            exclusions.append(
                {
                    "case_id": row[0],
                    "rater_id": row[1],
                    "task": task,
                    "completion_status": status_value,
                    "exclusion_reason": reason,
                }
            )

    all_cases = sorted(wide["case_id"].unique().tolist())
    present = set(zip(wide["case_id"], wide["rater_id"], strict=False))
    for case_id in all_cases:
        for rater_id in expected_rater_ids:
            if (case_id, rater_id) not in present:
                exclusions.append(
                    {
                        "case_id": case_id,
                        "rater_id": rater_id,
                        "task": "both",
                        "completion_status": pd.NA,
                        "exclusion_reason": "case_absent_from_rater_export",
                    }
                )

    long = pd.concat(rows, ignore_index=True)[
        [
            "case_id",
            "rater_id",
            "task",
            "rating_0_100",
            "rating_probability",
            "instrument_complete",
        ]
    ]
    exclusion_frame = pd.DataFrame(
        exclusions,
        columns=[
            "case_id",
            "rater_id",
            "task",
            "completion_status",
            "exclusion_reason",
        ],
    )
    return (
        long.sort_values(["task", "case_id", "rater_id"]).reset_index(drop=True),
        exclusion_frame.sort_values(["case_id", "rater_id", "task"]).reset_index(drop=True),
    )


def build_case_summary(
    long: pd.DataFrame, *, all_case_ids: list[str] | tuple[str, ...] | None = None
) -> pd.DataFrame:
    source_case_ids = long["case_id"].unique().tolist() if all_case_ids is None else all_case_ids
    case_ids = pd.DataFrame({"case_id": sorted(set(source_case_ids))})
    result = case_ids
    for task in TASKS:
        task_frame = long[long["task"] == task]
        grouped = task_frame.groupby("case_id", sort=True)["rating_0_100"]
        summary = grouped.agg(
            **{
                f"n_{task}_raters": "size",
                f"{task}_mean_0_100": "mean",
                f"{task}_sd_0_100": lambda values: values.std(ddof=1),
                f"{task}_min_0_100": "min",
                f"{task}_max_0_100": "max",
            }
        ).reset_index()
        summary[f"{task}_range_0_100"] = summary[f"{task}_max_0_100"] - summary[f"{task}_min_0_100"]
        summary[f"{task}_mean_probability"] = summary[f"{task}_mean_0_100"] / 100.0
        result = result.merge(summary, on="case_id", how="left", validate="one_to_one")

    for task in TASKS:
        result[f"n_{task}_raters"] = result[f"n_{task}_raters"].fillna(0).astype(int)
    result["report_minus_image_mean_0_100"] = (
        result["report_mean_0_100"] - result["image_mean_0_100"]
    )
    result["absolute_report_image_difference_0_100"] = result["report_minus_image_mean_0_100"].abs()
    return result.sort_values("case_id").reset_index(drop=True)


def rater_descriptive_summary(long: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (task, rater_id), group in long.groupby(["task", "rater_id"], sort=True):
        values = group["rating_0_100"].astype(float)
        rows.append(
            {
                "task": task,
                "rater_id": rater_id,
                "n_complete": int(len(values)),
                "mean_0_100": float(values.mean()),
                "sd_0_100": _sample_sd(values),
                "median_0_100": float(values.median()),
                "p25_0_100": float(values.quantile(0.25)),
                "p75_0_100": float(values.quantile(0.75)),
                "minimum_0_100": float(values.min()),
                "maximum_0_100": float(values.max()),
            }
        )
    return pd.DataFrame(rows)


def compute_pairwise_agreement(
    long: pd.DataFrame, expected_rater_ids: tuple[str, ...]
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for task in TASKS:
        pivot = long[long["task"] == task].pivot(
            index="case_id", columns="rater_id", values="rating_0_100"
        )
        for rater_a, rater_b in combinations(expected_rater_ids, 2):
            paired = pivot.reindex(columns=[rater_a, rater_b]).dropna()
            difference = paired[rater_a] - paired[rater_b]
            rows.append(
                {
                    "task": task,
                    "rater_a": rater_a,
                    "rater_b": rater_b,
                    "n_paired_cases": int(len(paired)),
                    "mean_signed_difference_a_minus_b_0_100": _safe_mean(difference),
                    "mean_absolute_difference_0_100": _safe_mean(difference.abs()),
                    "root_mean_squared_difference_0_100": _safe_rmse(difference),
                    "pearson_correlation_association": _safe_correlation(
                        paired.get(rater_a), paired.get(rater_b), method="pearson"
                    ),
                    "spearman_correlation_association": _safe_correlation(
                        paired.get(rater_a), paired.get(rater_b), method="spearman"
                    ),
                }
            )
    return pd.DataFrame(rows)


def compute_interrater_agreement(
    long: pd.DataFrame,
    case_summary: pd.DataFrame,
    expected_rater_ids: tuple[str, ...],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    expected_count = len(expected_rater_ids)
    for task in TASKS:
        complete_case_ids = _complete_case_ids(long, task, expected_rater_ids)
        task_cases = case_summary[case_summary[f"n_{task}_raters"] > 0]
        spread_cases = task_cases[task_cases[f"n_{task}_raters"] >= 2]
        icc = _compute_icc(long, task, complete_case_ids, expected_rater_ids)
        rows.append(
            {
                "task": task,
                "expected_raters": expected_count,
                "n_cases_with_any_completed_rating": int(len(task_cases)),
                "n_cases_with_two_or_more_raters": int(len(spread_cases)),
                "n_all_rater_complete_cases": int(len(complete_case_ids)),
                "n_cases_missing_one_or_more_raters": int(
                    len(case_summary) - len(complete_case_ids)
                ),
                "median_case_sd_0_100": _safe_median(spread_cases[f"{task}_sd_0_100"]),
                "median_case_range_0_100": _safe_median(spread_cases[f"{task}_range_0_100"]),
                "proportion_case_range_ge_25": _safe_proportion(
                    spread_cases[f"{task}_range_0_100"], 25.0
                ),
                "proportion_case_range_ge_50": _safe_proportion(
                    spread_cases[f"{task}_range_0_100"], 50.0
                ),
                **icc,
            }
        )
    return pd.DataFrame(rows)


def compute_image_report_alignment(
    case_summary: pd.DataFrame,
    *,
    expected_raters: int,
    n_boot: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    paired = case_summary[
        (case_summary["n_image_raters"] == expected_raters)
        & (case_summary["n_report_raters"] == expected_raters)
    ][["case_id", "image_mean_0_100", "report_mean_0_100"]].dropna()
    estimates = _alignment_estimates(paired)
    bootstrap = bootstrap_alignment_metrics(paired, n_boot=n_boot, seed=seed)
    n_excluded = int(len(case_summary) - len(paired))
    rows: list[dict[str, Any]] = []
    metric_units = {
        "mean_signed_difference": "percentage_points",
        "sd_paired_difference": "percentage_points",
        "mean_absolute_difference": "percentage_points",
        "root_mean_squared_difference": "percentage_points",
        "pearson_correlation": "correlation",
        "spearman_correlation": "correlation",
        "bland_altman_lower_limit": "percentage_points",
        "bland_altman_upper_limit": "percentage_points",
    }
    for metric, unit in metric_units.items():
        ci = bootstrap.get(metric, {})
        estimate = estimates.get(metric)
        rows.append(
            {
                "analysis": "case_mean_report_vs_image",
                "metric": metric,
                "estimate": estimate,
                "ci_lower": ci.get("ci_lower"),
                "ci_upper": ci.get("ci_upper"),
                "n_cases": int(len(paired)),
                "n_cases_excluded": n_excluded,
                "n_bootstrap_valid": ci.get("n_bootstrap_valid", 0),
                "unit": unit,
                "status": (
                    "estimated" if estimate is not None and not pd.isna(estimate) else "undefined"
                ),
            }
        )
    return pd.DataFrame(rows), paired.reset_index(drop=True)


def bootstrap_alignment_metrics(
    paired: pd.DataFrame, *, n_boot: int, seed: int
) -> dict[str, dict[str, float | int | None]]:
    if n_boot <= 0:
        raise ValueError("n_boot must be positive")
    output: dict[str, dict[str, float | int | None]] = {}
    if paired.empty:
        return {
            metric: {"ci_lower": None, "ci_upper": None, "n_bootstrap_valid": 0}
            for metric in BOOTSTRAP_METRICS
        }

    rng = np.random.default_rng(seed)
    positions = np.arange(len(paired))
    values: dict[str, list[float]] = {metric: [] for metric in BOOTSTRAP_METRICS}
    for _ in range(n_boot):
        sample_positions = rng.choice(positions, size=len(positions), replace=True)
        sample = paired.iloc[sample_positions]
        estimates = _alignment_estimates(sample)
        for metric in BOOTSTRAP_METRICS:
            value = estimates.get(metric)
            if value is not None and np.isfinite(value):
                values[metric].append(float(value))

    for metric, estimates in values.items():
        if estimates:
            output[metric] = {
                "ci_lower": float(np.percentile(estimates, 2.5)),
                "ci_upper": float(np.percentile(estimates, 97.5)),
                "n_bootstrap_valid": int(len(estimates)),
            }
        else:
            output[metric] = {
                "ci_lower": None,
                "ci_upper": None,
                "n_bootstrap_valid": 0,
            }
    return output


def compute_within_rater_alignment(
    long: pd.DataFrame, expected_rater_ids: tuple[str, ...]
) -> pd.DataFrame:
    pivot = long.pivot_table(
        index=["case_id", "rater_id"], columns="task", values="rating_0_100", aggfunc="first"
    ).reset_index()
    rows: list[dict[str, Any]] = []
    for rater_id in expected_rater_ids:
        rater = pivot[pivot["rater_id"] == rater_id]
        if not set(TASKS).issubset(rater.columns):
            paired = pd.DataFrame(columns=list(TASKS))
        else:
            paired = rater.dropna(subset=list(TASKS))
        difference = paired.get("report", pd.Series(dtype=float)) - paired.get(
            "image", pd.Series(dtype=float)
        )
        rows.append(
            {
                "rater_id": rater_id,
                "n_paired_cases": int(len(paired)),
                "mean_signed_report_minus_image_0_100": _safe_mean(difference),
                "mean_absolute_difference_0_100": _safe_mean(difference.abs()),
                "root_mean_squared_difference_0_100": _safe_rmse(difference),
                "pearson_correlation_association": _safe_correlation(
                    paired.get("image"), paired.get("report"), method="pearson"
                ),
                "spearman_correlation_association": _safe_correlation(
                    paired.get("image"), paired.get("report"), method="spearman"
                ),
            }
        )
    return pd.DataFrame(rows)


def build_case_disagreement_review(
    long: pd.DataFrame, case_summary: pd.DataFrame, *, limit: int = 10
) -> pd.DataFrame:
    compact: dict[tuple[str, str], str] = {}
    for (case_id, task), group in long.groupby(["case_id", "task"], sort=True):
        ratings = group.sort_values("rater_id")[["rater_id", "rating_0_100"]]
        compact[(case_id, task)] = ";".join(
            f"{row.rater_id}={float(row.rating_0_100):g}" for row in ratings.itertuples()
        )

    categories = (
        ("largest_image_interrater_range", "image_range_0_100"),
        ("largest_report_interrater_range", "report_range_0_100"),
        (
            "largest_absolute_report_image_difference",
            "absolute_report_image_difference_0_100",
        ),
    )
    rows: list[dict[str, Any]] = []
    for flag_type, sort_column in categories:
        ranked = case_summary.dropna(subset=[sort_column]).sort_values(
            [sort_column, "case_id"], ascending=[False, True]
        )
        for rank, row in enumerate(ranked.head(limit).itertuples(index=False), start=1):
            case_id = str(row.case_id)
            rows.append(
                {
                    "case_id": case_id,
                    "flag_type": flag_type,
                    "flag_rank": rank,
                    "image_ratings_compact": compact.get((case_id, "image"), ""),
                    "report_ratings_compact": compact.get((case_id, "report"), ""),
                    "image_mean_0_100": row.image_mean_0_100,
                    "report_mean_0_100": row.report_mean_0_100,
                    "image_range_0_100": row.image_range_0_100,
                    "report_range_0_100": row.report_range_0_100,
                    "report_minus_image_mean_0_100": row.report_minus_image_mean_0_100,
                }
            )
    return pd.DataFrame(rows)


def build_case_coverage_by_rater(
    wide: pd.DataFrame, long: pd.DataFrame, expected_rater_ids: tuple[str, ...]
) -> pd.DataFrame:
    all_case_count = int(wide["case_id"].nunique())
    rows: list[dict[str, Any]] = []
    for rater_id in expected_rater_ids:
        rater_wide = wide[wide["rater_id"] == rater_id]
        rater_long = long[long["rater_id"] == rater_id]
        row: dict[str, Any] = {
            "rater_id": rater_id,
            "source_case_rows": int(len(rater_wide)),
            "cases_absent_from_export": all_case_count - int(len(rater_wide)),
        }
        for task in TASKS:
            status = rater_wide[f"{task}_complete"]
            row[f"{task}_complete_ratings"] = int((rater_long["task"] == task).sum())
            row[f"{task}_incomplete_status_0"] = int(status.eq(0).sum())
            row[f"{task}_unverified_status_1"] = int(status.eq(1).sum())
            row[f"{task}_missing_completion_status"] = int(status.isna().sum())
        rows.append(row)
    return pd.DataFrame(rows)


def build_input_qa_summary(
    wide: pd.DataFrame,
    long: pd.DataFrame,
    case_summary: pd.DataFrame,
    config: AnnotationPilotConfig,
) -> pd.DataFrame:
    all_rater_cases = wide.groupby("case_id")["rater_id"].nunique()
    both_complete = (case_summary["n_image_raters"] == config.project.expected_raters) & (
        case_summary["n_report_raters"] == config.project.expected_raters
    )
    rows: list[dict[str, Any]] = [
        _qa_row("configured_raters", "pass", len(config.rater_ids)),
        _qa_row("source_case_rater_rows", "pass", len(wide)),
        _qa_row("unique_cases", "pass", wide["case_id"].nunique()),
        _qa_row(
            "cases_present_for_all_raters",
            "pass",
            int((all_rater_cases == config.project.expected_raters).sum()),
        ),
        _qa_row(
            "cases_missing_one_or_more_rater_exports",
            "warn" if (all_rater_cases < config.project.expected_raters).any() else "pass",
            int((all_rater_cases < config.project.expected_raters).sum()),
        ),
        _qa_row("completed_image_ratings", "pass", int((long["task"] == "image").sum())),
        _qa_row("completed_report_ratings", "pass", int((long["task"] == "report").sum())),
        _qa_row(
            "all_rater_complete_image_cases",
            "pass",
            int((case_summary["n_image_raters"] == config.project.expected_raters).sum()),
        ),
        _qa_row(
            "all_rater_complete_report_cases",
            "pass",
            int((case_summary["n_report_raters"] == config.project.expected_raters).sum()),
        ),
        _qa_row("all_rater_complete_both_task_cases", "pass", int(both_complete.sum())),
    ]
    for column in ("record_id", "secondary_record_id"):
        rows.append(_identifier_qa_row(wide, column))
    return pd.DataFrame(rows)


def plot_ratings_by_case(long: pd.DataFrame, *, task: str, output_path: Path) -> None:
    task_frame = long[long["task"] == task]
    means = task_frame.groupby("case_id")["rating_0_100"].mean().sort_values()
    position = {case_id: index + 1 for index, case_id in enumerate(means.index)}
    figure, axis = plt.subplots(figsize=(10, 5))
    for rater_id, group in task_frame.groupby("rater_id", sort=True):
        axis.scatter(
            group["case_id"].map(position),
            group["rating_0_100"],
            label=rater_id,
        )
    axis.set_xlabel(f"Cases ordered by mean {task} rating")
    axis.set_ylabel("Rating (0-100 percentage points)")
    axis.set_ylim(0, 100)
    axis.set_xlim(0.5, len(position) + 0.5)
    axis.set_title(f"{task.capitalize()} ratings by case")
    axis.legend(title="Rater")
    figure.tight_layout()
    ensure_parent_dir(output_path)
    figure.savefig(output_path)
    plt.close(figure)


def plot_image_report_scatter(
    paired: pd.DataFrame, alignment_summary: pd.DataFrame, *, output_path: Path
) -> None:
    figure, axis = plt.subplots(figsize=(6, 6))
    axis.scatter(paired["image_mean_0_100"], paired["report_mean_0_100"])
    axis.plot([0, 100], [0, 100], label="Identity")
    axis.set_xlim(0, 100)
    axis.set_ylim(0, 100)
    axis.set_xlabel("Mean image rating (0-100)")
    axis.set_ylabel("Mean report rating (0-100)")
    axis.set_title("Case-mean report versus image ratings")
    mean_difference = _summary_estimate(alignment_summary, "mean_signed_difference")
    mae = _summary_estimate(alignment_summary, "mean_absolute_difference")
    axis.text(
        0.03,
        0.97,
        f"n = {len(paired)}\nMean difference = {_format_number(mean_difference)}\n"
        f"MAE = {_format_number(mae)}",
        transform=axis.transAxes,
        va="top",
    )
    axis.legend()
    figure.tight_layout()
    ensure_parent_dir(output_path)
    figure.savefig(output_path)
    plt.close(figure)


def plot_bland_altman(
    paired: pd.DataFrame, alignment_summary: pd.DataFrame, *, output_path: Path
) -> None:
    figure, axis = plt.subplots(figsize=(7, 5))
    mean_axis = (paired["image_mean_0_100"] + paired["report_mean_0_100"]) / 2.0
    difference = paired["report_mean_0_100"] - paired["image_mean_0_100"]
    axis.scatter(mean_axis, difference)
    lines = (
        ("Mean difference", "mean_signed_difference"),
        ("Lower limit", "bland_altman_lower_limit"),
        ("Upper limit", "bland_altman_upper_limit"),
    )
    for label, metric in lines:
        value = _summary_estimate(alignment_summary, metric)
        if value is not None and np.isfinite(value):
            axis.axhline(value, label=label)
    axis.set_xlabel("Mean of image and report case means (0-100)")
    axis.set_ylabel("Report minus image case mean (percentage points)")
    axis.set_title("Image-report Bland-Altman plot")
    axis.legend()
    figure.tight_layout()
    ensure_parent_dir(output_path)
    figure.savefig(output_path)
    plt.close(figure)


def run_annotation_pilot(config: AnnotationPilotConfig) -> PilotAnalysisResult:
    validate_annotation_pilot_output_paths(config)
    wide = load_redcap_rater_exports(config)
    long, exclusions = build_task_long_table(
        wide,
        complete_status_value=config.project.complete_status_value,
        expected_rater_ids=config.rater_ids,
    )
    case_summary = build_case_summary(long, all_case_ids=wide["case_id"].unique().tolist())
    descriptive = rater_descriptive_summary(long)
    interrater = compute_interrater_agreement(long, case_summary, config.rater_ids)
    pairwise = compute_pairwise_agreement(long, config.rater_ids)
    alignment, paired = compute_image_report_alignment(
        case_summary,
        expected_raters=config.project.expected_raters,
        n_boot=config.project.bootstrap_replicates,
        seed=config.project.seed,
    )
    within_rater = compute_within_rater_alignment(long, config.rater_ids)
    disagreement = build_case_disagreement_review(long, case_summary)
    coverage = build_case_coverage_by_rater(wide, long, config.rater_ids)
    input_qa = build_input_qa_summary(wide, long, case_summary, config)

    ensure_dir(config.outputs.derived_dir)
    ensure_dir(config.outputs.artifact_dir)
    ensure_dir(config.outputs.report_dir)

    paths: list[tuple[str, str, Path, int | None, bool]] = []
    paths.extend(
        [
            _write_parquet(
                long,
                config.outputs.derived_dir / "ratings_long.parquet",
                "derived",
                restricted=True,
            ),
            _write_parquet(
                case_summary,
                config.outputs.derived_dir / "case_summary.parquet",
                "derived",
                restricted=True,
            ),
            _write_csv(
                input_qa,
                config.outputs.artifact_dir / "input_qa_summary.csv",
                "aggregate",
            ),
            _write_csv(
                descriptive,
                config.outputs.artifact_dir / "rater_descriptive_summary.csv",
                "aggregate",
            ),
            _write_csv(
                interrater,
                config.outputs.artifact_dir / "interrater_agreement_summary.csv",
                "aggregate",
            ),
            _write_csv(
                pairwise,
                config.outputs.artifact_dir / "pairwise_rater_agreement.csv",
                "aggregate",
            ),
            _write_csv(
                alignment,
                config.outputs.artifact_dir / "image_report_alignment_summary.csv",
                "aggregate",
            ),
            _write_csv(
                within_rater,
                config.outputs.artifact_dir / "within_rater_image_report_alignment.csv",
                "aggregate",
            ),
            _write_csv(
                coverage,
                config.outputs.artifact_dir / "case_coverage_by_rater.csv",
                "aggregate",
            ),
            _write_csv(
                exclusions,
                config.outputs.artifact_dir / "excluded_or_invalid_rows.csv",
                "restricted_audit",
                restricted=True,
            ),
            _write_csv(
                disagreement,
                config.outputs.artifact_dir / "case_disagreement_review.csv",
                "restricted_review",
                restricted=True,
            ),
        ]
    )

    figure_paths = {
        "image_ratings_by_case": config.outputs.artifact_dir / "image_ratings_by_case.png",
        "report_ratings_by_case": config.outputs.artifact_dir / "report_ratings_by_case.png",
        "image_report_scatter": config.outputs.artifact_dir / "image_report_scatter.png",
        "image_report_bland_altman": config.outputs.artifact_dir / "image_report_bland_altman.png",
    }
    plot_ratings_by_case(long, task="image", output_path=figure_paths["image_ratings_by_case"])
    plot_ratings_by_case(long, task="report", output_path=figure_paths["report_ratings_by_case"])
    plot_image_report_scatter(paired, alignment, output_path=figure_paths["image_report_scatter"])
    plot_bland_altman(paired, alignment, output_path=figure_paths["image_report_bland_altman"])
    for name, path in figure_paths.items():
        paths.append((name, "figure", path, None, False))

    report_path = config.outputs.report_dir / "ards_annotation_pilot_agreement.html"
    paths.append(("rendered_report", "report", report_path, None, False))
    manifest_path = config.outputs.artifact_dir / "output_manifest.csv"
    manifest_rows = [
        {
            "artifact_name": name,
            "category": category,
            "relative_path": _relative_path(path, config.repo_root),
            "row_count": row_count,
            "restricted_row_level": restricted,
        }
        for name, category, path, row_count, restricted in paths
    ]
    manifest_rows.append(
        {
            "artifact_name": "output_manifest",
            "category": "aggregate",
            "relative_path": _relative_path(manifest_path, config.repo_root),
            "row_count": len(manifest_rows) + 1,
            "restricted_row_level": False,
        }
    )
    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(manifest_path, index=False)

    disagreement_summary = (
        disagreement.groupby("flag_type", sort=True).size().rename("n_cases_listed").reset_index()
    )
    return PilotAnalysisResult(
        input_qa_summary=input_qa,
        rater_descriptive_summary=descriptive,
        interrater_agreement_summary=interrater,
        pairwise_rater_agreement=pairwise,
        image_report_alignment_summary=alignment,
        within_rater_image_report_alignment=within_rater,
        case_coverage_by_rater=coverage,
        disagreement_review_summary=disagreement_summary,
        output_manifest=manifest,
        figure_paths=figure_paths,
    )


def _compute_icc(
    long: pd.DataFrame,
    task: str,
    complete_case_ids: list[str],
    expected_rater_ids: tuple[str, ...],
) -> dict[str, Any]:
    empty = {
        "icc_2_1": None,
        "icc_2_1_ci_lower": None,
        "icc_2_1_ci_upper": None,
        "icc_2_k": None,
        "icc_2_k_ci_lower": None,
        "icc_2_k_ci_upper": None,
    }
    if len(complete_case_ids) < 2 or len(expected_rater_ids) < 2:
        return {**empty, "icc_status": "insufficient_balanced_cases"}
    balanced = long[
        (long["task"] == task)
        & long["case_id"].isin(complete_case_ids)
        & long["rater_id"].isin(expected_rater_ids)
    ][["case_id", "rater_id", "rating_0_100"]]
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            table = pg.intraclass_corr(
                data=balanced,
                targets="case_id",
                raters="rater_id",
                ratings="rating_0_100",
                nan_policy="raise",
            )
        single = _icc_row(table, "ICC2", "ICC(A,1)")
        average = _icc_row(table, "ICC2k", "ICC(A,k)")
        single_ci = _ci_bounds(_icc_ci_value(single))
        average_ci = _ci_bounds(_icc_ci_value(average))
        status = (
            "estimated"
            if all(value is not None for value in (*single_ci, *average_ci))
            else "estimated_ci_undefined"
        )
        return {
            "icc_2_1": float(single["ICC"]),
            "icc_2_1_ci_lower": single_ci[0],
            "icc_2_1_ci_upper": single_ci[1],
            "icc_2_k": float(average["ICC"]),
            "icc_2_k_ci_lower": average_ci[0],
            "icc_2_k_ci_upper": average_ci[1],
            "icc_status": status,
        }
    except (IndexError, KeyError, ValueError, ZeroDivisionError, np.linalg.LinAlgError):
        return {**empty, "icc_status": "undefined_for_observed_values"}


def _alignment_estimates(paired: pd.DataFrame) -> dict[str, float | None]:
    if paired.empty:
        return {
            "mean_signed_difference": None,
            "sd_paired_difference": None,
            "mean_absolute_difference": None,
            "root_mean_squared_difference": None,
            "pearson_correlation": None,
            "spearman_correlation": None,
            "bland_altman_lower_limit": None,
            "bland_altman_upper_limit": None,
        }
    difference = paired["report_mean_0_100"] - paired["image_mean_0_100"]
    mean_difference = float(difference.mean())
    sd_difference = _sample_sd(difference)
    lower = None if sd_difference is None else mean_difference - 1.96 * sd_difference
    upper = None if sd_difference is None else mean_difference + 1.96 * sd_difference
    return {
        "mean_signed_difference": mean_difference,
        "sd_paired_difference": sd_difference,
        "mean_absolute_difference": float(difference.abs().mean()),
        "root_mean_squared_difference": float(np.sqrt(np.mean(np.square(difference)))),
        "pearson_correlation": _safe_correlation(
            paired["image_mean_0_100"], paired["report_mean_0_100"], method="pearson"
        ),
        "spearman_correlation": _safe_correlation(
            paired["image_mean_0_100"], paired["report_mean_0_100"], method="spearman"
        ),
        "bland_altman_lower_limit": lower,
        "bland_altman_upper_limit": upper,
    }


def _complete_case_ids(
    long: pd.DataFrame, task: str, expected_rater_ids: tuple[str, ...]
) -> list[str]:
    pivot = long[long["task"] == task].pivot(
        index="case_id", columns="rater_id", values="rating_0_100"
    )
    expected = pivot.reindex(columns=list(expected_rater_ids))
    return expected.index[expected.notna().all(axis=1)].astype(str).tolist()


def _parse_completion(series: pd.Series, *, rater_id: str, column_name: str) -> pd.Series:
    text = series.astype("string").str.strip()
    numeric = pd.to_numeric(text.where(text.ne("")), errors="coerce")
    nonblank = text.notna() & text.ne("")
    invalid = nonblank & (
        numeric.isna() | ~numeric.isin(VALID_COMPLETION_VALUES) | ((numeric % 1) != 0)
    )
    if invalid.any():
        raise ValueError(
            f"REDCap export for {rater_id} has {int(invalid.sum())} invalid values in {column_name}"
        )
    return numeric.astype("Int64")


def _parse_rating(
    series: pd.Series,
    completion: pd.Series,
    *,
    complete_value: int,
    rater_id: str,
    column_name: str,
) -> pd.Series:
    text = series.astype("string").str.strip()
    numeric = pd.to_numeric(text.where(text.ne("")), errors="coerce")
    completed = completion.eq(complete_value).fillna(False)
    invalid = completed & (numeric.isna() | ~numeric.between(0, 100))
    if invalid.any():
        raise ValueError(
            f"REDCap export for {rater_id} has {int(invalid.sum())} invalid completed values "
            f"in {column_name}"
        )
    return numeric.astype(float)


def _optional_identifier(source: pd.DataFrame, column: str | None) -> pd.Series:
    if column is None or column not in source.columns:
        return pd.Series(pd.NA, index=source.index, dtype="string")
    return _normalize_identifier(source[column])


def _optional_text(source: pd.DataFrame, column: str | None) -> pd.Series:
    if column is None or column not in source.columns:
        return pd.Series(pd.NA, index=source.index, dtype="string")
    text = source[column].astype("string").str.strip()
    return text.mask(text.eq(""), pd.NA)


def _normalize_identifier(series: pd.Series) -> pd.Series:
    text = series.astype("string").str.strip()
    return text.mask(text.eq(""), pd.NA)


def _identifier_qa_row(wide: pd.DataFrame, column: str) -> dict[str, Any]:
    pairs = wide[["case_id", column]].dropna().drop_duplicates()
    if pairs.empty:
        return _qa_row(f"{column}_one_to_one_with_case_id", "not_available", None)
    case_to_id = pairs.groupby("case_id")[column].nunique()
    id_to_case = pairs.groupby(column)["case_id"].nunique()
    passed = bool((case_to_id <= 1).all() and (id_to_case <= 1).all())
    return _qa_row(
        f"{column}_one_to_one_with_case_id",
        "pass" if passed else "warn",
        passed,
    )


def _qa_row(check: str, status: str, value: Any) -> dict[str, Any]:
    return {"check": check, "status": status, "value": value}


def _safe_correlation(a: pd.Series | None, b: pd.Series | None, *, method: str) -> float | None:
    if a is None or b is None:
        return None
    paired = pd.DataFrame({"a": a, "b": b}).dropna()
    if len(paired) < 2 or paired["a"].nunique() < 2 or paired["b"].nunique() < 2:
        return None
    if method == "pearson":
        return float(stats.pearsonr(paired["a"], paired["b"]).statistic)
    if method == "spearman":
        return float(stats.spearmanr(paired["a"], paired["b"]).statistic)
    raise ValueError(f"Unsupported correlation method: {method}")


def _sample_sd(values: pd.Series) -> float | None:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    return None if len(clean) < 2 else float(clean.std(ddof=1))


def _safe_mean(values: pd.Series) -> float | None:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    return None if clean.empty else float(clean.mean())


def _safe_median(values: pd.Series) -> float | None:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    return None if clean.empty else float(clean.median())


def _safe_rmse(values: pd.Series) -> float | None:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    return None if clean.empty else float(np.sqrt(np.mean(np.square(clean))))


def _safe_proportion(values: pd.Series, threshold: float) -> float | None:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    return None if clean.empty else float((clean >= threshold).mean())


def _ci_bounds(value: Any) -> tuple[float | None, float | None]:
    if isinstance(value, (list, tuple, np.ndarray)) and len(value) == 2:
        lower = float(value[0])
        upper = float(value[1])
        return (
            lower if np.isfinite(lower) else None,
            upper if np.isfinite(upper) else None,
        )
    return None, None


def _icc_row(table: pd.DataFrame, *type_names: str) -> pd.Series:
    rows = table[table["Type"].isin(type_names)]
    if rows.empty:
        raise KeyError(f"ICC result did not contain any of {type_names}")
    return rows.iloc[0]


def _icc_ci_value(row: pd.Series) -> Any:
    for column in ("CI95%", "CI95"):
        if column in row.index:
            return row[column]
    raise KeyError("ICC result did not contain a 95% confidence interval column")


def _summary_estimate(summary: pd.DataFrame, metric: str) -> float | None:
    values = summary.loc[summary["metric"] == metric, "estimate"]
    if values.empty or pd.isna(values.iloc[0]):
        return None
    return float(values.iloc[0])


def _format_number(value: float | None) -> str:
    return "NA" if value is None or not np.isfinite(value) else f"{value:.1f}"


def _required_mapping(raw: dict[str, Any], key: str) -> dict[str, Any]:
    value = raw.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"Annotation pilot config requires a {key!r} mapping")
    return value


def _required_name(raw: dict[str, Any], key: str) -> str:
    value = str(raw.get(key, "")).strip()
    if not value:
        raise ValueError(f"columns.{key} must be nonblank")
    return value


def _optional_name(raw: dict[str, Any], key: str, default: str) -> str | None:
    value = raw.get(key, default)
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


def _resolve_repo_path(value: Any, root: Path) -> Path:
    if value is None or str(value).strip() == "":
        raise ValueError("Configured annotation pilot paths must be nonblank")
    path = Path(str(value)).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _write_csv(
    frame: pd.DataFrame,
    path: Path,
    category: str,
    *,
    restricted: bool = False,
) -> tuple[str, str, Path, int, bool]:
    ensure_parent_dir(path)
    frame.to_csv(path, index=False)
    return path.stem, category, path, int(len(frame)), restricted


def _write_parquet(
    frame: pd.DataFrame,
    path: Path,
    category: str,
    *,
    restricted: bool,
) -> tuple[str, str, Path, int, bool]:
    ensure_parent_dir(path)
    frame.to_parquet(path, index=False)
    return path.stem, category, path, int(len(frame)), restricted


def _relative_path(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as error:
        raise ValueError("Annotation pilot outputs must remain inside the repository") from error


def validate_annotation_pilot_output_paths(config: AnnotationPilotConfig) -> None:
    """Require each output category to remain under its ignored repository root."""

    requirements = (
        ("outputs.report_dir", config.outputs.report_dir, Path("reports")),
        ("outputs.artifact_dir", config.outputs.artifact_dir, Path("artifacts")),
        ("outputs.derived_dir", config.outputs.derived_dir, Path("data/derived")),
    )
    repo_root = config.repo_root.resolve()
    for name, path, relative_root in requirements:
        allowed_lexical_root = repo_root / relative_root
        allowed_resolved_root = allowed_lexical_root.resolve()
        if not allowed_resolved_root.is_relative_to(repo_root):
            raise ValueError(
                f"Annotation pilot output root escapes the repository through a symlink: "
                f"{relative_root.as_posix()}"
            )

        candidate = path if path.is_absolute() else repo_root / path
        candidate_lexical = Path(os.path.abspath(candidate))
        candidate_resolved = candidate.resolve()
        if not candidate_lexical.is_relative_to(allowed_lexical_root) or not (
            candidate_resolved.is_relative_to(allowed_resolved_root)
        ):
            raise ValueError(
                f"{name} must remain under {allowed_lexical_root.relative_to(repo_root)}"
            )
