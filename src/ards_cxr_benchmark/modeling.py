from __future__ import annotations

import hashlib
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .paths import get_paths

SPLIT_SALT = "mimic-bilat-opacity-v1-split"
SPLIT_MODULUS = 10000
TRAIN_CUTOFF = 7000
VALIDATION_CUTOFF = 8500

TEXT_COLUMN = "primary_target_text"
TASK_LABEL_COLUMNS = {
    "strict": "strict_bilateral_opacity_label",
    "sensitive": "sensitive_bilateral_opacity_label",
}

# Source-derived rule inputs. Do not include silver labels, silver score, label source,
# QA fields, or composite silver-output flags such as bilateral_opacity_any.
STRUCTURED_FEATURE_COLUMNS = [
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
    "bilateral_edema",
    "bilateral_atelectasis",
    "bilateral_consolidation_or_airspace",
    "bilateral_ambiguous_or_uncertain",
]


@dataclass(frozen=True)
class TaskFrame:
    task: str
    label_column: str
    data: pd.DataFrame


@dataclass(frozen=True)
class ModelingRunPaths:
    data_cache: Path
    output_dir: Path
    prediction_out: Path


def default_modeling_paths(*, limit: int | None, root: Path | None = None) -> ModelingRunPaths:
    repo_root = root or get_paths().root
    data_dir = repo_root / "data" / "derived" / "modeling"
    artifact_dir = repo_root / "artifacts" / "modeling"
    if limit is not None:
        data_dir = data_dir / "smoke"
        artifact_dir = artifact_dir / "smoke"
    return ModelingRunPaths(
        data_cache=data_dir / "model_development_extract.parquet",
        output_dir=artifact_dir,
        prediction_out=data_dir / "silver_baseline_predictions.parquet",
    )


def resolve_modeling_paths(
    *,
    limit: int | None,
    data_cache: Path | None = None,
    output_dir: Path | None = None,
    prediction_out: Path | None = None,
) -> ModelingRunPaths:
    defaults = default_modeling_paths(limit=limit)
    return ModelingRunPaths(
        data_cache=data_cache or defaults.data_cache,
        output_dir=output_dir or defaults.output_dir,
        prediction_out=prediction_out or defaults.prediction_out,
    )


def subject_split_bucket(subject_id: int | str, salt: str = SPLIT_SALT) -> int:
    """Return the BigQuery-compatible split bucket for a subject identifier."""

    subject = int(subject_id)
    digest = hashlib.sha256(f"{subject}-{salt}".encode()).hexdigest()
    return int(digest[:15], 16) % SPLIT_MODULUS


def assign_subject_split(subject_id: int | str, salt: str = SPLIT_SALT) -> str:
    bucket = subject_split_bucket(subject_id, salt=salt)
    if bucket < TRAIN_CUTOFF:
        return "train"
    if bucket < VALIDATION_CUTOFF:
        return "validation"
    return "test"


def add_subject_splits(
    df: pd.DataFrame,
    *,
    subject_col: str = "subject_id",
    salt: str = SPLIT_SALT,
) -> pd.DataFrame:
    out = df.copy()
    out["split_bucket"] = out[subject_col].map(
        lambda subject_id: subject_split_bucket(subject_id, salt)
    )
    out["split"] = out["split_bucket"].map(
        lambda bucket: (
            "train"
            if bucket < TRAIN_CUTOFF
            else "validation"
            if bucket < VALIDATION_CUTOFF
            else "test"
        )
    )
    return out


def assert_no_subject_overlap(
    df: pd.DataFrame,
    *,
    subject_col: str = "subject_id",
    split_col: str = "split",
) -> None:
    split_counts = df.groupby(subject_col, observed=True)[split_col].nunique()
    overlapping = split_counts[split_counts > 1]
    if not overlapping.empty:
        examples = ", ".join(str(value) for value in overlapping.index[:5].tolist())
        raise ValueError(
            f"Subject leakage across splits for {len(overlapping)} subjects: {examples}"
        )


def eligible_task_frame(df: pd.DataFrame, task: str) -> TaskFrame:
    if task not in TASK_LABEL_COLUMNS:
        raise KeyError(f"Unknown task {task!r}; expected one of {sorted(TASK_LABEL_COLUMNS)}")

    label_column = TASK_LABEL_COLUMNS[task]
    data = df[df[label_column].notna()].copy()
    data[label_column] = data[label_column].astype(int)
    return TaskFrame(task=task, label_column=label_column, data=data)


def available_structured_feature_columns(df: pd.DataFrame) -> list[str]:
    return [column for column in STRUCTURED_FEATURE_COLUMNS if column in df.columns]


def prepare_structured_features(
    df: pd.DataFrame, columns: Iterable[str] | None = None
) -> pd.DataFrame:
    selected_columns = list(columns or available_structured_feature_columns(df))
    if not selected_columns:
        raise ValueError("No structured feature columns are present in the model extract")

    features = df[selected_columns].copy()
    for column in features.columns:
        if pd.api.types.is_bool_dtype(features[column]):
            features[column] = features[column].astype("Int64")
    return features.apply(pd.to_numeric, errors="coerce")


def fit_text_baseline(
    train_text: Iterable[str],
    y_train: Iterable[int],
    *,
    max_features: int = 20_000,
    ngram_range: tuple[int, int] = (1, 2),
) -> Pipeline:
    model = Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    max_features=max_features,
                    min_df=2,
                    ngram_range=ngram_range,
                    strip_accents="unicode",
                ),
            ),
            (
                "logreg",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=1_000,
                    solver="liblinear",
                ),
            ),
        ]
    )
    return model.fit(list(train_text), list(y_train))


def fit_structured_baseline(x_train: pd.DataFrame, y_train: Iterable[int]) -> Pipeline:
    numeric_columns = list(x_train.columns)
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
                        ("scaler", StandardScaler(with_mean=False)),
                    ]
                ),
                numeric_columns,
            )
        ]
    )
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "logreg",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=1_000,
                    solver="liblinear",
                ),
            ),
        ]
    )
    return model.fit(x_train, list(y_train))


def predict_positive_probability(model: Any, x_eval: Any) -> pd.Series:
    probabilities = model.predict_proba(x_eval)
    return pd.Series(probabilities[:, 1])


def binary_metrics(
    y_true: Iterable[int], y_score: Iterable[float], *, threshold: float = 0.5
) -> dict[str, Any]:
    y_true_series = pd.Series(list(y_true)).astype(int)
    y_score_series = pd.Series(list(y_score)).astype(float)
    y_pred = (y_score_series >= threshold).astype(int)

    labels_present = sorted(y_true_series.unique().tolist())
    tn, fp, fn, tp = confusion_matrix(y_true_series, y_pred, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if tn + fp else None

    metrics: dict[str, Any] = {
        "n": int(len(y_true_series)),
        "positives": int(y_true_series.sum()),
        "prevalence": _safe_float(y_true_series.mean()),
        "threshold": float(threshold),
        "accuracy": _safe_float(accuracy_score(y_true_series, y_pred)),
        "precision": _safe_float(precision_score(y_true_series, y_pred, zero_division=0)),
        "recall": _safe_float(recall_score(y_true_series, y_pred, zero_division=0)),
        "specificity": _safe_float(specificity),
        "f1": _safe_float(f1_score(y_true_series, y_pred, zero_division=0)),
        "brier": _safe_float(brier_score_loss(y_true_series, y_score_series)),
    }
    if len(labels_present) == 2:
        metrics["roc_auc"] = _safe_float(roc_auc_score(y_true_series, y_score_series))
        metrics["average_precision"] = _safe_float(
            average_precision_score(y_true_series, y_score_series)
        )
    else:
        metrics["roc_auc"] = None
        metrics["average_precision"] = None
    return metrics


def _safe_float(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)
