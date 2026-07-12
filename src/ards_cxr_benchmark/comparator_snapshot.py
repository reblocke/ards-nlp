from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

MODEL_DISPLAY_NAMES = {
    "clamp_legacy": "Dan Knox legacy CLAMP",
    "clamp_python_compatibility": "Python CLAMP compatibility mirror",
    "amaral_xgboost_bilateral_infiltrates": "Amaral published preprocessing",
    "amaral_xgboost_bilateral_infiltrates_raw_text_direct": "Amaral raw-text sensitivity",
    "uw_hanso_bilateral_infiltrates": "UW HANSO",
    "afshar_text_svc_full_ards": "Afshar text SVC (full-ARDS target)",
}
INDEPENDENT_ROLES = {"external_comparator", "legacy_teacher", "trained_silver_baseline"}
REQUIRED_MODELS = frozenset(
    {
        "clamp_legacy",
        "clamp_python_compatibility",
        "amaral_xgboost_bilateral_infiltrates",
        "amaral_xgboost_bilateral_infiltrates_raw_text_direct",
        "silver_sensitive_silver_score_rule",
        "silver_sensitive_structured_logreg",
        "silver_sensitive_tfidf_logreg",
        "silver_strict_silver_score_rule",
        "silver_strict_structured_logreg",
        "silver_strict_tfidf_logreg",
    }
)


def load_comparator_snapshot_inputs(
    *,
    catalog_path: Path,
    metrics_path: Path,
    compatibility_metrics_path: Path | None = None,
    status_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    paths = [catalog_path, metrics_path, status_path]
    if compatibility_metrics_path is not None:
        paths.append(compatibility_metrics_path)
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"Comparator snapshot input not found: {path}")
    catalog = pd.read_csv(catalog_path)
    metrics = pd.read_csv(metrics_path)
    if compatibility_metrics_path is not None:
        metrics = pd.concat(
            [metrics, pd.read_csv(compatibility_metrics_path)],
            ignore_index=True,
        )
    statuses = json.loads(status_path.read_text(encoding="utf-8"))
    if not isinstance(statuses, list):
        raise ValueError("Comparator status input must be a JSON list")
    return catalog, metrics, statuses


def render_comparator_snapshot(
    catalog: pd.DataFrame,
    metrics: pd.DataFrame,
    statuses: list[dict[str, Any]],
    *,
    corpus_rows: int = 227_835,
) -> str:
    _require_columns(
        catalog,
        {
            "model_name",
            "comparison_role",
            "prediction_rows",
            "unique_cases",
            "validation_rows",
            "test_rows",
            "cases_without_reference",
        },
        "prediction catalog",
    )
    _require_columns(
        metrics,
        {
            "model_name",
            "comparison_role",
            "task",
            "split",
            "n",
            "prevalence",
            "roc_auc",
            "average_precision",
            "f1",
            "sensitivity",
            "specificity",
            "brier",
        },
        "silver metrics",
    )
    if catalog["model_name"].duplicated().any():
        raise ValueError("Prediction catalog contains duplicate model names")
    observed_models = set(catalog["model_name"].astype(str))
    missing_models = REQUIRED_MODELS - observed_models
    if missing_models:
        raise ValueError(f"Missing required comparator models: {sorted(missing_models)}")
    if int(catalog["cases_without_reference"].sum()) != 0:
        raise ValueError("Comparator catalog contains predictions without reference rows")

    lines = [
        "# MIMIC-CXR Comparator Snapshot V1",
        "",
        "> Aggregate engineering diagnostics against automated silver labels. These are not human",
        "> gold-standard or clinical-accuracy estimates.",
        "",
        "## Corpus",
        "",
        f"- MIMIC-CXR studies in the canonical source: **{corpus_rows:,}**",
        f"- Available prediction models/controls: **{len(catalog)}**",
        "- Metrics are calculated only on fixed subject-level validation and test splits.",
        "- Report text, identifiers, and row-level predictions are not included in this snapshot.",
        "",
        "## Comparator Status",
        "",
        "| Comparator | Status | Reason |",
        "|---|---|---|",
    ]
    for status in sorted(statuses, key=lambda item: str(item.get("name", ""))):
        name = str(status.get("name", "unknown"))
        lines.append(
            f"| {_display_name(name)} | {status.get('status', 'unknown')} | "
            f"{_cell(status.get('reason', ''))} |"
        )

    lines.extend(
        [
            "",
            "## Available Predictions",
            "",
            "| Model | Role | Prediction rows | Unique cases | Validation | Test |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for _, row in catalog.sort_values(["comparison_role", "model_name"]).iterrows():
        lines.append(
            f"| {_display_name(str(row['model_name']))} | {row['comparison_role']} | "
            f"{int(row['prediction_rows']):,} | {int(row['unique_cases']):,} | "
            f"{int(row['validation_rows']):,} | {int(row['test_rows']):,} |"
        )

    independent = metrics[metrics["comparison_role"].isin(INDEPENDENT_ROLES)].copy()
    compatibility = metrics[metrics["comparison_role"] == "compatibility_mirror"].copy()
    controls = metrics[
        ~metrics["comparison_role"].isin({*INDEPENDENT_ROLES, "compatibility_mirror"})
    ].copy()
    lines.extend(_metric_section("Holdout Metrics", independent))
    lines.extend(_metric_section("Compatibility Mirrors", compatibility))
    lines.extend(_metric_section("Silver-Derived Controls", controls))
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- `clamp_legacy` is the returned Dan Knox CLAMP workflow. The Python compatibility",
            "  mirror has exact full-corpus parity and is not independent validation evidence.",
            "- Amaral is an external pretrained comparator; the raw-text variant is a sensitivity",
            "  analysis rather than a separate published model.",
            "- TF-IDF models are trained on silver labels. Structured rules and score controls are",
            "  label-derived checks and must not be interpreted as independent validation.",
            "- UW HANSO and Afshar remain gate-controlled; blocked runs are reported explicitly",
            "  and predictions are never fabricated.",
            "- Final model claims require the physician report-only and image-only reference data.",
            "",
        ]
    )
    return "\n".join(lines)


def write_comparator_snapshot(
    *,
    catalog_path: Path,
    metrics_path: Path,
    compatibility_metrics_path: Path | None = None,
    status_path: Path,
    output_path: Path,
) -> None:
    catalog, metrics, statuses = load_comparator_snapshot_inputs(
        catalog_path=catalog_path,
        metrics_path=metrics_path,
        compatibility_metrics_path=compatibility_metrics_path,
        status_path=status_path,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        render_comparator_snapshot(catalog, metrics, statuses),
        encoding="utf-8",
    )


def _metric_section(title: str, metrics: pd.DataFrame) -> list[str]:
    lines = [
        "",
        f"## {title}",
        "",
        "| Model | Task | Split | n | Prevalence | AUROC | Average precision | F1 | "
        "Sensitivity | Specificity | Brier |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in metrics.sort_values(["model_name", "task", "split"]).iterrows():
        lines.append(
            f"| {_display_name(str(row['model_name']))} | {row['task']} | {row['split']} | "
            f"{int(row['n']):,} | {_metric(row['prevalence'])} | {_metric(row['roc_auc'])} | "
            f"{_metric(row['average_precision'])} | {_metric(row['f1'])} | "
            f"{_metric(row['sensitivity'])} | {_metric(row['specificity'])} | "
            f"{_metric(row['brier'])} |"
        )
    if metrics.empty:
        lines.append("| None | - | - | - | - | - | - | - | - | - | - |")
    return lines


def _display_name(model_name: str) -> str:
    return MODEL_DISPLAY_NAMES.get(model_name, model_name.replace("_", " "))


def _metric(value: Any) -> str:
    if pd.isna(value):
        return "NA"
    return f"{float(value):.3f}"


def _cell(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _require_columns(frame: pd.DataFrame, required: set[str], name: str) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing columns: {missing}")
