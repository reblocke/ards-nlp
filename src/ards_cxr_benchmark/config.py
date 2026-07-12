from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from . import PIPELINE_VERSION
from .paths import get_paths

CLAMP_WORKSPACE_PLACEHOLDER = "YOUR_LOCAL_CLAMP_WORKSPACE_ROOT"
CLAMP_RESTRICTED_ARTIFACT_DIR = "artifacts/restricted/clamp_ards"
CLAMP_WORKSPACE_DIR = "artifacts/restricted/clamp_ards/workspace"
CLAMP_RUNTIME_PROJECT_DIR = "C:/ClampWin_1.6.6/workspace/ARDS"


@dataclass(frozen=True)
class BigQueryConfig:
    project_id: str
    dataset: str
    location: str = "US"
    physionet_project: str = "physionet-data"
    source_dataset: str = "mimic_cxr"
    gcs_bucket: str | None = None

    @property
    def dataset_ref(self) -> str:
        return f"{self.project_id}.{self.dataset}"


@dataclass(frozen=True)
class SourcePaths:
    report_root: Path
    report_parquet: Path
    chexpert_csv: Path
    negbio_csv: Path
    split_csv: Path
    metadata_csv: Path
    jpg_parquet_dir: Path
    radgraph_json: Path
    radgraph_parquet_dir: Path
    qa_dir: Path
    sample_dir: Path
    export_dir: Path


@dataclass(frozen=True)
class ClampARDSConfig:
    clamp_project_root: Path
    project_name: str
    project_source_dir: Path
    project_live_dir: Path
    runtime_project_dir: str
    input_dir: Path
    output_dir: Path
    output_archive: Path
    full_output_archive: Path
    output_packet_summary: Path
    restricted_artifact_dir: Path
    input_manifest: Path
    input_summary: Path
    handoff_markdown: Path
    teacher_entity_output: Path
    teacher_prediction_output: Path
    teacher_probabilistic_prediction_output: Path
    teacher_output_audit: Path
    teacher_summary_output: Path
    teacher_benchmark_dir: Path
    python_input: Path
    python_entity_output: Path
    python_prediction_output: Path
    python_summary_output: Path
    python_parity_summary: Path
    python_parity_mismatches: Path


@dataclass(frozen=True)
class PipelineConfig:
    name: str
    seed: int
    pipeline_version: str
    bq: BigQueryConfig
    paths: SourcePaths
    clamp_ards: ClampARDSConfig
    primary_label_text_scope: str = "full_report"

    def sql_parameters(self) -> dict[str, str]:
        return {
            "project_id": self.bq.project_id,
            "bq_dataset": self.bq.dataset,
            "physionet_project": self.bq.physionet_project,
            "source_dataset": self.bq.source_dataset,
            "location": self.bq.location,
            "pipeline_version": self.pipeline_version,
        }


def _nested_get(cfg: dict[str, object], *keys: str, default: object = None) -> object:
    current: object = cfg
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def _resolve_path(value: object, root: Path, default: str) -> Path:
    raw = Path(str(value or default))
    return raw if raw.is_absolute() else (root / raw).resolve()


def load_config(config_path: Path) -> PipelineConfig:
    """Load benchmark config and normalize repo-relative paths."""

    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise ValueError(f"Config must be a YAML mapping: {config_path}")

    p = get_paths()
    seed = int(_nested_get(cfg, "project", "seed", default=0))

    bq = BigQueryConfig(
        project_id=str(_nested_get(cfg, "gcp", "project_id", default="YOUR_GCP_PROJECT")),
        dataset=str(_nested_get(cfg, "gcp", "bq_dataset", default="ards_mimic_cxr_benchmark")),
        location=str(_nested_get(cfg, "gcp", "location", default="US")),
        physionet_project=str(
            _nested_get(cfg, "gcp", "physionet_project", default="physionet-data")
        ),
        source_dataset=str(_nested_get(cfg, "gcp", "source_dataset", default="mimic_cxr")),
        gcs_bucket=(
            None
            if _nested_get(cfg, "gcp", "gcs_bucket", default=None) in (None, "")
            else str(_nested_get(cfg, "gcp", "gcs_bucket"))
        ),
    )

    paths = SourcePaths(
        report_root=_resolve_path(
            _nested_get(cfg, "paths", "report_root"),
            p.root,
            "data/raw/mimic-cxr/files",
        ),
        report_parquet=_resolve_path(
            _nested_get(cfg, "paths", "report_parquet"),
            p.root,
            "data/processed/mimic_cxr_reports.parquet",
        ),
        chexpert_csv=_resolve_path(
            _nested_get(cfg, "paths", "chexpert_csv"),
            p.root,
            "data/raw/mimic-cxr-jpg/mimic-cxr-2.0.0-chexpert.csv.gz",
        ),
        negbio_csv=_resolve_path(
            _nested_get(cfg, "paths", "negbio_csv"),
            p.root,
            "data/raw/mimic-cxr-jpg/mimic-cxr-2.0.0-negbio.csv.gz",
        ),
        split_csv=_resolve_path(
            _nested_get(cfg, "paths", "split_csv"),
            p.root,
            "data/raw/mimic-cxr-jpg/mimic-cxr-2.0.0-split.csv.gz",
        ),
        metadata_csv=_resolve_path(
            _nested_get(cfg, "paths", "metadata_csv"),
            p.root,
            "data/raw/mimic-cxr-jpg/mimic-cxr-2.0.0-metadata.csv.gz",
        ),
        jpg_parquet_dir=_resolve_path(
            _nested_get(cfg, "paths", "jpg_parquet_dir"),
            p.root,
            "data/processed/mimic_cxr_jpg",
        ),
        radgraph_json=_resolve_path(
            _nested_get(cfg, "paths", "radgraph_json"),
            p.root,
            "data/raw/radgraph/MIMIC-CXR_graphs.json",
        ),
        radgraph_parquet_dir=_resolve_path(
            _nested_get(cfg, "paths", "radgraph_parquet_dir"),
            p.root,
            "data/processed/radgraph",
        ),
        qa_dir=_resolve_path(_nested_get(cfg, "paths", "qa_dir"), p.root, "artifacts/qa"),
        sample_dir=_resolve_path(
            _nested_get(cfg, "paths", "sample_dir"), p.root, "artifacts/samples"
        ),
        export_dir=_resolve_path(_nested_get(cfg, "paths", "export_dir"), p.root, "data/derived"),
    )
    clamp_project_root = _resolve_path(
        _nested_get(cfg, "clamp_ards", "clamp_project_root"),
        p.root,
        CLAMP_WORKSPACE_DIR,
    )
    clamp_project_name = str(_nested_get(cfg, "clamp_ards", "project_name", default="ARDS"))
    clamp_project_live_dir = _resolve_path(
        _nested_get(cfg, "clamp_ards", "project_live_dir"),
        p.root,
        str(clamp_project_root / clamp_project_name),
    )
    clamp_artifact_dir = _resolve_path(
        _nested_get(cfg, "clamp_ards", "restricted_artifact_dir"),
        p.root,
        CLAMP_RESTRICTED_ARTIFACT_DIR,
    )
    clamp_ards = ClampARDSConfig(
        clamp_project_root=clamp_project_root,
        project_name=clamp_project_name,
        project_source_dir=_resolve_path(
            _nested_get(cfg, "clamp_ards", "project_source_dir"),
            p.root,
            "data/external/clamp_ards_project",
        ),
        project_live_dir=clamp_project_live_dir,
        runtime_project_dir=str(
            _nested_get(
                cfg,
                "clamp_ards",
                "runtime_project_dir",
                default=CLAMP_RUNTIME_PROJECT_DIR,
            )
        ),
        input_dir=_resolve_path(
            _nested_get(cfg, "clamp_ards", "input_dir"),
            p.root,
            str(clamp_project_live_dir / "Data" / "Input"),
        ),
        output_dir=_resolve_path(
            _nested_get(cfg, "clamp_ards", "output_dir"),
            p.root,
            str(clamp_project_live_dir / "Data" / "Output"),
        ),
        output_archive=_resolve_path(
            _nested_get(cfg, "clamp_ards", "output_archive"),
            p.root,
            str(clamp_artifact_dir / "incoming" / "ARDS_CLAMP_Output_txt_only.zip"),
        ),
        full_output_archive=_resolve_path(
            _nested_get(cfg, "clamp_ards", "full_output_archive"),
            p.root,
            str(clamp_artifact_dir / "oracle" / "ARDS_CLAMP_Output_full.zip"),
        ),
        output_packet_summary=_resolve_path(
            _nested_get(cfg, "clamp_ards", "output_packet_summary"),
            p.root,
            str(clamp_artifact_dir / "clamp_output_packet_summary.json"),
        ),
        restricted_artifact_dir=clamp_artifact_dir,
        input_manifest=_resolve_path(
            _nested_get(cfg, "clamp_ards", "input_manifest"),
            p.root,
            str(clamp_artifact_dir / "input_manifest.csv"),
        ),
        input_summary=_resolve_path(
            _nested_get(cfg, "clamp_ards", "input_summary"),
            p.root,
            str(clamp_artifact_dir / "input_summary.json"),
        ),
        handoff_markdown=_resolve_path(
            _nested_get(cfg, "clamp_ards", "handoff_markdown"),
            p.root,
            str(clamp_artifact_dir / "NEXT_STEP_RUN_CLAMP.md"),
        ),
        teacher_entity_output=_resolve_path(
            _nested_get(cfg, "clamp_ards", "teacher_entity_output"),
            p.root,
            "data/derived/clamp_ards/clamp_legacy_entities.parquet",
        ),
        teacher_prediction_output=_resolve_path(
            _nested_get(cfg, "clamp_ards", "teacher_prediction_output"),
            p.root,
            "data/derived/clamp_ards/clamp_legacy_predictions.parquet",
        ),
        teacher_probabilistic_prediction_output=_resolve_path(
            _nested_get(cfg, "clamp_ards", "teacher_probabilistic_prediction_output"),
            p.root,
            "data/derived/clamp_ards/clamp_legacy_predictions_for_probabilistic_benchmark.parquet",
        ),
        teacher_output_audit=_resolve_path(
            _nested_get(cfg, "clamp_ards", "teacher_output_audit"),
            p.root,
            str(clamp_artifact_dir / "clamp_output_audit.csv"),
        ),
        teacher_summary_output=_resolve_path(
            _nested_get(cfg, "clamp_ards", "teacher_summary_output"),
            p.root,
            str(clamp_artifact_dir / "clamp_teacher_summary.json"),
        ),
        teacher_benchmark_dir=_resolve_path(
            _nested_get(cfg, "clamp_ards", "teacher_benchmark_dir"),
            p.root,
            "artifacts/clamp_ards/teacher_benchmark",
        ),
        python_input=_resolve_path(
            _nested_get(cfg, "clamp_ards", "python_input"),
            p.root,
            "artifacts/restricted/comparators/mimic_cxr_comparator_input.jsonl.gz",
        ),
        python_entity_output=_resolve_path(
            _nested_get(cfg, "clamp_ards", "python_entity_output"),
            p.root,
            "data/derived/clamp_ards/clamp_python_entities.parquet",
        ),
        python_prediction_output=_resolve_path(
            _nested_get(cfg, "clamp_ards", "python_prediction_output"),
            p.root,
            "data/derived/clamp_ards/clamp_python_predictions.parquet",
        ),
        python_summary_output=_resolve_path(
            _nested_get(cfg, "clamp_ards", "python_summary_output"),
            p.root,
            str(clamp_artifact_dir / "python" / "batch_summary.json"),
        ),
        python_parity_summary=_resolve_path(
            _nested_get(cfg, "clamp_ards", "python_parity_summary"),
            p.root,
            str(clamp_artifact_dir / "python" / "parity_summary.json"),
        ),
        python_parity_mismatches=_resolve_path(
            _nested_get(cfg, "clamp_ards", "python_parity_mismatches"),
            p.root,
            str(clamp_artifact_dir / "python" / "parity_mismatches.csv"),
        ),
    )

    return PipelineConfig(
        name=str(_nested_get(cfg, "project", "name", default="ards-cxr-mimic-benchmark")),
        seed=seed,
        pipeline_version=str(
            _nested_get(cfg, "project", "pipeline_version", default=PIPELINE_VERSION)
        ),
        bq=bq,
        paths=paths,
        clamp_ards=clamp_ards,
        primary_label_text_scope=str(
            _nested_get(cfg, "labels", "primary_text_scope", default="full_report")
        ),
    )


def default_config_path() -> Path:
    root = get_paths().root
    preferred = root / "config" / "config.yaml"
    fallback = root / "config" / "config.example.yaml"
    return preferred if preferred.exists() else fallback


def load_default_config() -> PipelineConfig:
    return load_config(default_config_path())


def require_existing_path(path: Path, description: str) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {description}: {path}. Configure this in config/config.yaml."
        )


def validate_clamp_ards_operational_paths(**paths: Path) -> None:
    """Reject placeholder CLAMP paths before sync/export commands can write report text."""

    errors: list[str] = []
    repo_root = get_paths().root.resolve()
    allowed_repo_root = (repo_root / CLAMP_RESTRICTED_ARTIFACT_DIR).resolve()
    for name, path in paths.items():
        raw_path = str(path)
        if not raw_path.strip():
            errors.append(f"{name} is empty")
        if CLAMP_WORKSPACE_PLACEHOLDER in raw_path:
            errors.append(f"{name} still contains {CLAMP_WORKSPACE_PLACEHOLDER}")
        path_obj = Path(path)
        if not path_obj.is_absolute():
            errors.append(f"{name} must be an absolute path")
            continue
        resolved_path = path_obj.resolve()
        is_repo_local = resolved_path == repo_root or resolved_path.is_relative_to(repo_root)
        is_allowed_restricted = resolved_path == allowed_repo_root or resolved_path.is_relative_to(
            allowed_repo_root
        )
        if is_repo_local and not is_allowed_restricted:
            errors.append(
                f"{name} must be outside the repository root or under "
                f"{CLAMP_RESTRICTED_ARTIFACT_DIR}"
            )
    if errors:
        raise ValueError(
            "CLAMP ARDS workspace paths are not configured for operational use: "
            + "; ".join(errors)
            + ". Update ignored config/config.yaml or pass explicit CLI path flags."
        )


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
