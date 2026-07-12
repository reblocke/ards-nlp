from __future__ import annotations

import pandas as pd
import pytest

from ards_cxr_benchmark.modeling import (
    TEXT_COLUMN,
    add_subject_splits,
    assert_no_subject_overlap,
    available_structured_feature_columns,
    binary_metrics,
    default_modeling_paths,
    eligible_task_frame,
    fit_text_baseline,
    predict_positive_probability,
    resolve_modeling_paths,
    subject_split_bucket,
)
from ards_cxr_benchmark.modeling_qa import evaluate_modeling_outputs


def test_subject_split_assignment_is_deterministic_and_subject_level() -> None:
    df = pd.DataFrame(
        {
            "subject_id": [100, 100, 200],
            "study_id": [1, 2, 3],
        }
    )

    split_df = add_subject_splits(df)

    assert subject_split_bucket(100) == subject_split_bucket("100")
    assert split_df.loc[0, "split"] == split_df.loc[1, "split"]
    assert split_df["split_bucket"].between(0, 9999).all()


def test_assert_no_subject_overlap_rejects_leakage() -> None:
    leaking = pd.DataFrame(
        {
            "subject_id": [1, 1, 2],
            "split": ["train", "test", "train"],
        }
    )

    with pytest.raises(ValueError, match="Subject leakage"):
        assert_no_subject_overlap(leaking)


def test_eligible_task_frame_drops_null_labels() -> None:
    df = pd.DataFrame(
        {
            "strict_bilateral_opacity_label": [1, 0, None],
            "sensitive_bilateral_opacity_label": [1, 0, 1],
        }
    )

    task_frame = eligible_task_frame(df, "strict")

    assert len(task_frame.data) == 2
    assert task_frame.label_column == "strict_bilateral_opacity_label"
    assert task_frame.data["strict_bilateral_opacity_label"].tolist() == [1, 0]


def test_structured_features_exclude_composite_silver_outputs() -> None:
    df = pd.DataFrame(
        {
            "chexpert_lung_opacity": [1],
            "regex_bilateral_opacity_present": [True],
            "bilateral_opacity_any": [True],
            "bilateral_opacity_non_atelectatic": [True],
        }
    )

    columns = available_structured_feature_columns(df)

    assert "chexpert_lung_opacity" in columns
    assert "regex_bilateral_opacity_present" in columns
    assert "bilateral_opacity_any" not in columns
    assert "bilateral_opacity_non_atelectatic" not in columns


def test_binary_metrics_include_threshold_metrics_and_auc() -> None:
    metrics = binary_metrics([0, 0, 1, 1], [0.1, 0.4, 0.6, 0.9])

    assert metrics["n"] == 4
    assert metrics["positives"] == 2
    assert metrics["accuracy"] == 1.0
    assert metrics["specificity"] == 1.0
    assert metrics["roc_auc"] == 1.0


def test_text_baseline_smoke_fit_and_predict() -> None:
    train = pd.DataFrame(
        {
            TEXT_COLUMN: [
                "Bilateral airspace opacities are present.",
                "Bilateral pulmonary opacity persists.",
                "Clear lungs without focal opacity.",
                "Clear lungs without edema.",
            ],
            "label": [1, 1, 0, 0],
        }
    )

    model = fit_text_baseline(train[TEXT_COLUMN], train["label"], max_features=20)
    scores = predict_positive_probability(model, train[TEXT_COLUMN])

    assert len(scores) == len(train)
    assert scores.between(0, 1).all()


def test_smoke_default_paths_do_not_overlap_full_paths(tmp_path) -> None:
    full = default_modeling_paths(limit=None, root=tmp_path)
    smoke = default_modeling_paths(limit=5000, root=tmp_path)

    assert full.data_cache != smoke.data_cache
    assert full.output_dir != smoke.output_dir
    assert full.prediction_out != smoke.prediction_out
    assert "smoke" in smoke.data_cache.parts
    assert "smoke" in smoke.output_dir.parts


def test_explicit_modeling_paths_override_smoke_defaults(tmp_path) -> None:
    explicit_cache = tmp_path / "custom" / "cache.parquet"

    resolved = resolve_modeling_paths(limit=5000, data_cache=explicit_cache)

    assert resolved.data_cache == explicit_cache
    assert "smoke" in resolved.output_dir.parts
    assert "smoke" in resolved.prediction_out.parts


def test_modeling_qa_passes_for_complete_outputs() -> None:
    extract = _qa_extract()
    predictions = _qa_predictions(extract)
    metrics = _qa_metrics()

    result = evaluate_modeling_outputs(extract, predictions, metrics)

    assert result.passed is True
    assert result.summary["subjects_with_split_overlap"] == 0
    assert result.summary["prediction_text_columns"] == []


def test_modeling_qa_catches_overlap_missing_text_and_prediction_text_leakage() -> None:
    extract = _qa_extract()
    extract.loc[1, "split"] = "test"
    extract.loc[2, TEXT_COLUMN] = None
    predictions = _qa_predictions(_qa_extract())
    predictions["report_text"] = "leaked text"
    metrics = _qa_metrics().iloc[:-1]

    result = evaluate_modeling_outputs(extract, predictions, metrics)

    assert result.passed is False
    checks = {issue.check for issue in result.issues}
    assert "subject_overlap" in checks
    assert "missing_text" in checks
    assert "prediction_text_leakage" in checks
    assert "metric_rows" in checks


def _qa_extract() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "subject_id": [1, 1, 2, 3, 4, 5],
            "study_id": [10, 11, 20, 30, 40, 50],
            "split": ["train", "train", "validation", "test", "validation", "test"],
            TEXT_COLUMN: ["a", "b", "c", "d", "e", "f"],
            "strict_bilateral_opacity_label": [1, 0, 1, 0, None, None],
            "sensitive_bilateral_opacity_label": [1, 0, 1, 0, 1, 0],
        }
    )


def _qa_predictions(extract: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for task, label_column in {
        "strict": "strict_bilateral_opacity_label",
        "sensitive": "sensitive_bilateral_opacity_label",
    }.items():
        eligible = extract[extract[label_column].notna()]
        for _, row in eligible[eligible["split"].isin(["validation", "test"])].iterrows():
            for model in ["silver_score_rule", "tfidf_logreg", "structured_logreg"]:
                rows.append(
                    {
                        "subject_id": row["subject_id"],
                        "study_id": row["study_id"],
                        "split": row["split"],
                        "task": task,
                        "model": model,
                        "true_label": int(row[label_column]),
                        "prediction_score": 0.7,
                        "prediction_label": 1,
                    }
                )
    return pd.DataFrame(rows)


def _qa_metrics() -> pd.DataFrame:
    rows = []
    for task in ["strict", "sensitive"]:
        for split in ["validation", "test"]:
            for model in ["silver_score_rule", "tfidf_logreg", "structured_logreg"]:
                rows.append(
                    {
                        "task": task,
                        "split": split,
                        "model": model,
                        "n": 1,
                        "roc_auc": 1.0,
                        "average_precision": 1.0,
                    }
                )
    return pd.DataFrame(rows)
