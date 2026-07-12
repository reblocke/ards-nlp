from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from ards_cxr_benchmark.bq import query_to_dataframe
from ards_cxr_benchmark.config import (
    default_config_path,
    ensure_dir,
    ensure_parent_dir,
    load_config,
)
from ards_cxr_benchmark.modeling import (
    SPLIT_SALT,
    TASK_LABEL_COLUMNS,
    TEXT_COLUMN,
    add_subject_splits,
    assert_no_subject_overlap,
    available_structured_feature_columns,
    binary_metrics,
    eligible_task_frame,
    fit_structured_baseline,
    fit_text_baseline,
    predict_positive_probability,
    prepare_structured_features,
    resolve_modeling_paths,
)
from ards_cxr_benchmark.modeling_qa import evaluate_modeling_outputs, write_modeling_qa

MODEL_EXTRACT_COLUMNS = [
    "subject_id",
    "study_id",
    TEXT_COLUMN,
    "strict_bilateral_opacity_label",
    "sensitive_bilateral_opacity_label",
    "silver_bilateral_opacity_score",
    "silver_label_source",
    "chexpert_lung_opacity",
    "chexpert_edema",
    "chexpert_consolidation",
    "chexpert_atelectasis",
    "has_mimic_cxr_jpg_labels",
    "negbio_lung_opacity",
    "negbio_edema",
    "negbio_consolidation",
    "negbio_atelectasis",
    "radgraph_strict_bilateral_opacity_present",
    "radgraph_sensitive_bilateral_opacity_present",
    "regex_bilateral_opacity_present",
    "regex_bilateral_opacity_uncertain",
    "regex_bilateral_opacity_negated",
    "bilateral_opacity_any",
    "bilateral_opacity_non_atelectatic",
    "bilateral_edema",
    "bilateral_atelectasis",
    "bilateral_consolidation_or_airspace",
    "bilateral_ambiguous_or_uncertain",
    "manual_review_priority",
    "qa_flags",
]
SPLIT_COLUMNS = ["split", "split_bucket"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train silver-label baseline models")
    parser.add_argument("--config", type=Path, default=default_config_path())
    parser.add_argument("--refresh-cache", action="store_true")
    parser.add_argument("--data-cache", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--prediction-out", type=Path)
    parser.add_argument("--limit", type=int, help="Optional deterministic row limit for smoke runs")
    parser.add_argument("--max-features", type=int, default=20_000)
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=sorted(TASK_LABEL_COLUMNS),
        choices=sorted(TASK_LABEL_COLUMNS),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive when provided")

    paths = resolve_modeling_paths(
        limit=args.limit,
        data_cache=args.data_cache,
        output_dir=args.output_dir,
        prediction_out=args.prediction_out,
    )
    config = load_config(args.config)
    df = load_or_query_extract(
        config=config,
        cache_path=paths.data_cache,
        refresh=args.refresh_cache,
        limit=args.limit,
    )
    df = normalize_extract(df)
    assert_no_subject_overlap(df)

    run_at = datetime.now(UTC).isoformat()
    ensure_dir(paths.output_dir)
    ensure_parent_dir(paths.prediction_out)
    ensure_dir(paths.output_dir / "models")

    metrics: list[dict[str, Any]] = []
    predictions: list[pd.DataFrame] = []
    structured_features = available_structured_feature_columns(df)

    for task in args.tasks:
        task_frame = eligible_task_frame(df, task)
        task_data = task_frame.data
        split_counts = task_data["split"].value_counts().to_dict()
        if any(split_counts.get(split, 0) == 0 for split in ("train", "validation", "test")):
            raise ValueError(f"Task {task} does not have rows in every split: {split_counts}")

        train = task_data[task_data["split"] == "train"].copy()
        y_train = train[task_frame.label_column]
        if y_train.nunique() < 2:
            raise ValueError(f"Task {task} train split has one class and cannot fit baselines")

        for eval_split in ("validation", "test"):
            eval_df = task_data[task_data["split"] == eval_split].copy()
            score = eval_df["silver_bilateral_opacity_score"].fillna(0)
            metrics.append(
                make_metric_row(
                    run_at=run_at,
                    task=task,
                    model_name="silver_score_rule",
                    split=eval_split,
                    n_train=len(train),
                    eligible_rows=len(task_data),
                    metrics=binary_metrics(eval_df[task_frame.label_column], score),
                )
            )
            predictions.append(
                make_prediction_frame(
                    eval_df,
                    task=task,
                    model_name="silver_score_rule",
                    label_column=task_frame.label_column,
                    score=score,
                )
            )

        text_model = fit_text_baseline(
            train[TEXT_COLUMN].fillna(""),
            y_train,
            max_features=args.max_features,
        )
        joblib.dump(text_model, paths.output_dir / "models" / f"{task}_tfidf_logreg.joblib")
        for eval_split in ("validation", "test"):
            eval_df = task_data[task_data["split"] == eval_split].copy()
            score = predict_positive_probability(text_model, eval_df[TEXT_COLUMN].fillna(""))
            metrics.append(
                make_metric_row(
                    run_at=run_at,
                    task=task,
                    model_name="tfidf_logreg",
                    split=eval_split,
                    n_train=len(train),
                    eligible_rows=len(task_data),
                    metrics=binary_metrics(eval_df[task_frame.label_column], score),
                )
            )
            predictions.append(
                make_prediction_frame(
                    eval_df,
                    task=task,
                    model_name="tfidf_logreg",
                    label_column=task_frame.label_column,
                    score=score,
                )
            )

        x_train = prepare_structured_features(train, structured_features)
        structured_model = fit_structured_baseline(x_train, y_train)
        joblib.dump(
            structured_model,
            paths.output_dir / "models" / f"{task}_structured_logreg.joblib",
        )
        for eval_split in ("validation", "test"):
            eval_df = task_data[task_data["split"] == eval_split].copy()
            x_eval = prepare_structured_features(eval_df, structured_features)
            score = predict_positive_probability(structured_model, x_eval)
            metrics.append(
                make_metric_row(
                    run_at=run_at,
                    task=task,
                    model_name="structured_logreg",
                    split=eval_split,
                    n_train=len(train),
                    eligible_rows=len(task_data),
                    metrics=binary_metrics(eval_df[task_frame.label_column], score),
                )
            )
            predictions.append(
                make_prediction_frame(
                    eval_df,
                    task=task,
                    model_name="structured_logreg",
                    label_column=task_frame.label_column,
                    score=score,
                )
            )

    metrics_df = pd.DataFrame(metrics)
    prediction_df = pd.concat(predictions, ignore_index=True)
    strata_df = stratified_prediction_summary(prediction_df)

    metrics_csv = paths.output_dir / "silver_baseline_metrics.csv"
    metrics_json = paths.output_dir / "silver_baseline_metrics.json"
    strata_csv = paths.output_dir / "silver_baseline_strata.csv"
    summary_md = paths.output_dir / "silver_baseline_summary.md"

    metrics_df.to_csv(metrics_csv, index=False)
    metrics_json.write_text(
        json.dumps(json.loads(metrics_df.to_json(orient="records")), indent=2),
        encoding="utf-8",
    )
    strata_df.to_csv(strata_csv, index=False)
    prediction_df.to_parquet(paths.prediction_out, index=False)
    qa_result = evaluate_modeling_outputs(df, prediction_df, metrics_df)
    write_modeling_qa(
        qa_result,
        out_json=paths.output_dir / "modeling_qa_summary.json",
        out_md=paths.output_dir / "modeling_qa_summary.md",
    )
    summary_md.write_text(
        render_summary(
            metrics_df=metrics_df,
            strata_df=strata_df,
            rows=len(df),
            split_counts=df.groupby("split", observed=True).size().to_dict(),
            structured_features=structured_features,
            prediction_out=paths.prediction_out,
        ),
        encoding="utf-8",
    )

    print(f"Wrote metrics to {metrics_csv}")
    print(f"Wrote strata summary to {strata_csv}")
    print(f"Wrote predictions without report text to {paths.prediction_out}")
    print(f"Wrote modeling QA to {paths.output_dir / 'modeling_qa_summary.md'}")
    print(f"Wrote summary to {summary_md}")


def load_or_query_extract(
    *,
    config: Any,
    cache_path: Path,
    refresh: bool,
    limit: int | None,
) -> pd.DataFrame:
    if cache_path.exists() and not refresh:
        return pd.read_parquet(cache_path)

    select_columns = ",\n    ".join(f"e.{column}" for column in MODEL_EXTRACT_COLUMNS)
    limit_sql = ""
    if limit is not None:
        limit_sql = f"""
ORDER BY FARM_FINGERPRINT(CONCAT(CAST(e.subject_id AS STRING), '-', CAST(e.study_id AS STRING)))
LIMIT {int(limit)}
"""
    sql = f"""
SELECT
    {select_columns},
    s.split,
    s.split_bucket
FROM `{config.bq.dataset_ref}.model_development_extract` AS e
INNER JOIN `{config.bq.dataset_ref}.model_development_splits` AS s
  USING (subject_id, study_id)
WHERE e.{TEXT_COLUMN} IS NOT NULL
{limit_sql}
"""
    df = query_to_dataframe(sql, config)
    ensure_parent_dir(cache_path)
    df.to_parquet(cache_path, index=False)
    print(f"Wrote model extract cache with {len(df):,} rows to {cache_path}")
    return df


def normalize_extract(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "split" not in out.columns or "split_bucket" not in out.columns:
        out = add_subject_splits(out, salt=SPLIT_SALT)
    out["qa_flags"] = out.get("qa_flags", pd.Series([None] * len(out))).map(normalize_qa_flags)
    out["qa_flags_present"] = out["qa_flags"].map(lambda value: value != "none")
    for column in ("manual_review_priority", "silver_label_source"):
        if column not in out.columns:
            out[column] = "unknown"
        out[column] = out[column].fillna("unknown")
    return out


def normalize_qa_flags(value: Any) -> str:
    if value is None:
        return "none"
    if isinstance(value, list):
        return "|".join(str(item) for item in value) if value else "none"
    if hasattr(value, "tolist"):
        items = value.tolist()
        return "|".join(str(item) for item in items) if items else "none"
    text = str(value)
    return text if text not in ("", "[]", "None", "<NA>", "nan") else "none"


def make_metric_row(
    *,
    run_at: str,
    task: str,
    model_name: str,
    split: str,
    n_train: int,
    eligible_rows: int,
    metrics: dict[str, Any],
) -> dict[str, Any]:
    return {
        "run_at_utc": run_at,
        "task": task,
        "model": model_name,
        "split": split,
        "n_train": int(n_train),
        "eligible_rows": int(eligible_rows),
        **metrics,
    }


def make_prediction_frame(
    df: pd.DataFrame,
    *,
    task: str,
    model_name: str,
    label_column: str,
    score: pd.Series,
) -> pd.DataFrame:
    out = df[
        [
            "subject_id",
            "study_id",
            "split",
            "silver_label_source",
            "manual_review_priority",
            "qa_flags",
            "qa_flags_present",
        ]
    ].copy()
    out["task"] = task
    out["model"] = model_name
    out["true_label"] = df[label_column].astype(int).to_numpy()
    out["prediction_score"] = score.to_numpy()
    out["prediction_label"] = (out["prediction_score"] >= 0.5).astype(int)
    return out


def stratified_prediction_summary(predictions: pd.DataFrame) -> pd.DataFrame:
    groups = [
        ("silver_label_source", "silver_label_source"),
        ("manual_review_priority", "manual_review_priority"),
        ("qa_flags_present", "qa_flags_present"),
    ]
    rows: list[dict[str, Any]] = []
    for task in sorted(predictions["task"].unique()):
        task_df = predictions[predictions["task"] == task]
        for model_name in sorted(task_df["model"].unique()):
            model_df = task_df[task_df["model"] == model_name]
            for split in sorted(model_df["split"].unique()):
                split_df = model_df[model_df["split"] == split]
                for group_name, column in groups:
                    for group_value, group_df in split_df.groupby(
                        column, dropna=False, observed=True
                    ):
                        rows.append(
                            {
                                "task": task,
                                "model": model_name,
                                "split": split,
                                "group": group_name,
                                "value": str(group_value),
                                "n": int(len(group_df)),
                                "prevalence": float(group_df["true_label"].mean()),
                                "mean_prediction_score": float(group_df["prediction_score"].mean()),
                            }
                        )
    return pd.DataFrame(rows)


def render_summary(
    *,
    metrics_df: pd.DataFrame,
    strata_df: pd.DataFrame,
    rows: int,
    split_counts: dict[str, int],
    structured_features: list[str],
    prediction_out: Path,
) -> str:
    split_counts = {str(key): int(value) for key, value in split_counts.items()}
    best_test = metrics_df[metrics_df["split"] == "test"].sort_values(
        ["task", "average_precision"], ascending=[True, False], na_position="last"
    )
    metric_table = best_test[
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
        ]
    ].to_string(index=False)
    summary_lines = [
        "# Silver-label baseline modeling summary",
        "",
        "These baselines use automated silver labels, not human-adjudicated gold labels.",
        "The structured baseline uses source-derived silver-rule inputs and is a rule diagnostic.",
        "",
        f"- Cached model rows: {rows:,}",
        f"- Split counts: {json.dumps(split_counts, sort_keys=True)}",
        f"- Structured feature count: {len(structured_features)}",
        f"- Prediction artifact: `{prediction_out}`",
        "",
        "## Test metrics",
        "",
        "```text",
        metric_table,
        "```",
        "",
        "## Stratified outputs",
        "",
        f"Stratified prediction summaries written for {len(strata_df):,} task/model/split groups.",
        "",
    ]
    return "\n".join(summary_lines)


if __name__ == "__main__":
    main()
