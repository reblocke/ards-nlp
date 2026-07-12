from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from ..paths import get_paths


@dataclass(frozen=True)
class ExternalRepositoryConfig:
    name: str
    repository: str
    commit: str
    license: str
    external_repo_dir: Path


@dataclass(frozen=True)
class AmaralComparatorConfig:
    source: ExternalRepositoryConfig
    model_path: Path
    vectorizer_path: Path
    segmentation_notebook: Path
    segmentation_module: Path
    tokenizer_module: Path
    license_path: Path
    expected_sha256: dict[str, str]
    environment_dir: Path
    nltk_data_dir: Path
    input_packet: Path
    input_manifest: Path
    preprocessing_config: Path
    runner_output_dir: Path
    prediction_output: Path
    artifact_dir: Path
    batch_size: int
    threshold: float


@dataclass(frozen=True)
class UWHansoComparatorConfig:
    source: ExternalRepositoryConfig
    model_dir: Path
    parameters_path: Path
    state_dict_path: Path
    expected_sha256: dict[str, str]
    terms_of_use: str
    container_image: str
    input_packet: Path
    input_manifest: Path
    smoke_input_packet: Path
    smoke_input_manifest: Path
    smoke_prediction_output: Path
    prediction_output: Path
    artifact_dir: Path
    batch_size: int


@dataclass(frozen=True)
class AfsharComparatorConfig:
    source: ExternalRepositoryConfig
    model_path: Path
    vectorizer_path: Path
    permission_status: str
    anchor_review_status: str
    verified_target: str
    expected_sha256: dict[str, str]
    container_image: str
    input_packet: Path
    input_manifest: Path
    prediction_output: Path
    artifact_dir: Path
    threshold: float
    batch_size: int


def load_amaral_config(path: Path) -> AmaralComparatorConfig:
    cfg = _load_yaml(path)
    source = _source_config(cfg)
    root = get_paths().root
    checksums = cfg.get("expected_sha256", {})
    if not isinstance(checksums, dict):
        raise ValueError("expected_sha256 must be a mapping")
    result = AmaralComparatorConfig(
        source=source,
        model_path=_under_repo(source.external_repo_dir, cfg, "model_path"),
        vectorizer_path=_under_repo(source.external_repo_dir, cfg, "vectorizer_path"),
        segmentation_notebook=_under_repo(source.external_repo_dir, cfg, "segmentation_notebook"),
        segmentation_module=_under_repo(source.external_repo_dir, cfg, "segmentation_module"),
        tokenizer_module=_under_repo(source.external_repo_dir, cfg, "tokenizer_module"),
        license_path=_under_repo(source.external_repo_dir, cfg, "license_path"),
        expected_sha256={str(key): str(value) for key, value in checksums.items()},
        environment_dir=_resolve(cfg.get("environment_dir"), root, "environments/amaral"),
        nltk_data_dir=_resolve(cfg.get("nltk_data_dir"), root, "data/external/amaral_nltk_data"),
        input_packet=_resolve(
            cfg.get("input_packet"),
            root,
            "artifacts/restricted/comparators/mimic_cxr_comparator_input.jsonl.gz",
        ),
        input_manifest=_resolve(
            cfg.get("input_manifest"),
            root,
            "artifacts/restricted/comparators/mimic_cxr_comparator_manifest.parquet",
        ),
        preprocessing_config=_resolve(
            cfg.get("preprocessing_config"),
            root,
            "artifacts/restricted/comparators/amaral/published_preprocessing_config.json",
        ),
        runner_output_dir=_resolve(
            cfg.get("runner_output_dir"),
            root,
            "artifacts/restricted/comparators/amaral/runner",
        ),
        prediction_output=_resolve(
            cfg.get("prediction_output"),
            root,
            "data/derived/comparators/amaral_bilateral_infiltrates_predictions.parquet",
        ),
        artifact_dir=_resolve(cfg.get("artifact_dir"), root, "artifacts/comparators/amaral"),
        batch_size=int(cfg.get("batch_size", 4096)),
        threshold=float(cfg.get("threshold", 0.5)),
    )
    if result.batch_size <= 0:
        raise ValueError("Amaral batch_size must be positive")
    if not 0 <= result.threshold <= 1:
        raise ValueError("Amaral threshold must be in [0, 1]")
    return result


def load_uw_hanso_config(path: Path) -> UWHansoComparatorConfig:
    cfg = _load_yaml(path)
    source = _source_config(cfg)
    root = get_paths().root
    checksums = cfg.get("expected_sha256", {})
    if not isinstance(checksums, dict):
        raise ValueError("UW HANSO expected_sha256 must be a mapping")
    model_dir = _resolve(cfg.get("model_dir"), root, str(source.external_repo_dir / "model"))
    result = UWHansoComparatorConfig(
        source=source,
        model_dir=model_dir,
        parameters_path=_resolve(
            cfg.get("parameters_path"), root, str(model_dir / "parameters.pkl")
        ),
        state_dict_path=_resolve(
            cfg.get("state_dict_path"), root, str(model_dir / "state_dict.pt")
        ),
        expected_sha256={str(key): str(value).strip().lower() for key, value in checksums.items()},
        terms_of_use=str(cfg.get("terms_of_use", "unknown")).strip(),
        container_image=str(cfg.get("container_image", "ards-nlp-uw-hanso:legacy")).strip(),
        input_packet=_resolve(
            cfg.get("input_packet"),
            root,
            "artifacts/restricted/comparators/mimic_cxr_comparator_input.jsonl.gz",
        ),
        input_manifest=_resolve(
            cfg.get("input_manifest"),
            root,
            "artifacts/restricted/comparators/mimic_cxr_comparator_manifest.parquet",
        ),
        smoke_input_packet=_resolve(
            cfg.get("smoke_input_packet"),
            root,
            "artifacts/restricted/comparators/smoke/input.jsonl.gz",
        ),
        smoke_input_manifest=_resolve(
            cfg.get("smoke_input_manifest"),
            root,
            "artifacts/restricted/comparators/smoke/manifest.parquet",
        ),
        smoke_prediction_output=_resolve(
            cfg.get("smoke_prediction_output"),
            root,
            "data/derived/comparators/smoke/uw_hanso_predictions.parquet",
        ),
        prediction_output=_resolve(
            cfg.get("prediction_output"),
            root,
            "data/derived/comparators/uw_hanso_predictions.parquet",
        ),
        artifact_dir=_resolve(cfg.get("artifact_dir"), root, "artifacts/comparators/uw_hanso"),
        batch_size=int(cfg.get("batch_size", 8)),
    )
    if result.batch_size <= 0:
        raise ValueError("UW HANSO batch_size must be positive")
    return result


def load_afshar_config(path: Path) -> AfsharComparatorConfig:
    cfg = _load_yaml(path)
    source = _source_config(cfg)
    root = get_paths().root
    threshold = float(cfg.get("threshold", 0.5))
    if not 0 <= threshold <= 1:
        raise ValueError("Afshar threshold must be in [0, 1]")
    checksums = cfg.get("expected_sha256", {})
    if not isinstance(checksums, dict):
        raise ValueError("Afshar expected_sha256 must be a mapping")
    result = AfsharComparatorConfig(
        source=source,
        model_path=_under_repo(source.external_repo_dir, cfg, "model_path"),
        vectorizer_path=_under_repo(source.external_repo_dir, cfg, "vectorizer_path"),
        permission_status=str(cfg.get("permission_status", "unknown")).strip(),
        anchor_review_status=str(cfg.get("anchor_review_status", "not_reviewed")).strip(),
        verified_target=str(cfg.get("verified_target", "full_ards_phenotype")).strip(),
        expected_sha256={str(key): str(value) for key, value in checksums.items()},
        container_image=str(cfg.get("container_image", "ards-nlp-afshar-svc:legacy")).strip(),
        input_packet=_resolve(
            cfg.get("input_packet"),
            root,
            "artifacts/restricted/comparators/mimic_cxr_comparator_input.jsonl.gz",
        ),
        input_manifest=_resolve(
            cfg.get("input_manifest"),
            root,
            "artifacts/restricted/comparators/mimic_cxr_comparator_manifest.parquet",
        ),
        prediction_output=_resolve(
            cfg.get("prediction_output"),
            root,
            "data/derived/comparators/afshar_text_svc_predictions.parquet",
        ),
        artifact_dir=_resolve(cfg.get("artifact_dir"), root, "artifacts/comparators/afshar"),
        threshold=threshold,
        batch_size=int(cfg.get("batch_size", 4096)),
    )
    if result.batch_size <= 0:
        raise ValueError("Afshar batch_size must be positive")
    return result


def _source_config(cfg: dict[str, Any]) -> ExternalRepositoryConfig:
    root = get_paths().root
    required = ["name", "repository", "commit", "license", "external_repo_dir"]
    missing = [key for key in required if not str(cfg.get(key, "")).strip()]
    if missing:
        raise ValueError(f"Missing external comparator config fields: {missing}")
    commit = str(cfg["commit"]).strip()
    if len(commit) != 40 or any(char not in "0123456789abcdef" for char in commit.lower()):
        raise ValueError(f"External comparator commit must be a full SHA: {commit!r}")
    return ExternalRepositoryConfig(
        name=str(cfg["name"]).strip(),
        repository=str(cfg["repository"]).strip(),
        commit=commit,
        license=str(cfg["license"]).strip(),
        external_repo_dir=_resolve(cfg["external_repo_dir"], root, "data/external/missing"),
    )


def _load_yaml(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Comparator config must be a YAML mapping: {path}")
    return raw


def _resolve(value: object, root: Path, default: str) -> Path:
    path = Path(str(value or default)).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _under_repo(external_repo_dir: Path, cfg: dict[str, Any], key: str) -> Path:
    value = str(cfg.get(key, "")).strip()
    if not value:
        raise ValueError(f"Missing comparator config field: {key}")
    path = Path(value)
    if path.is_absolute():
        return path.resolve()
    return (external_repo_dir / path).resolve()
