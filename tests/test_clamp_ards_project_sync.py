from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

from ards_cxr_benchmark.clamp_ards_inputs import sync_clamp_ards_project
from ards_cxr_benchmark.config import (
    CLAMP_RESTRICTED_ARTIFACT_DIR,
    load_config,
    validate_clamp_ards_operational_paths,
)
from ards_cxr_benchmark.paths import get_paths


def test_sync_clamp_project_copies_and_normalizes_descriptor_paths(tmp_path: Path) -> None:
    source = _source_project(tmp_path)
    live = tmp_path / "workspace" / "ARDS"
    runtime_project_dir = "D:/ClampRuntime/workspace/ARDS"

    summary = sync_clamp_ards_project(
        source_dir=source,
        live_dir=live,
        runtime_project_dir=runtime_project_dir,
        artifact_dir=tmp_path / "artifacts",
        summary_path=tmp_path / "artifacts" / "project_sync_summary.json",
    )

    descriptor = (live / "descriptor" / "defaultEngine.xml").read_text(encoding="utf-8")
    assert "D:/ClampRuntime/workspace/ARDS" in descriptor
    assert live.as_posix() not in descriptor
    assert "C:/ClampWin_1.6.6/workspace/ARDS" not in descriptor
    assert (live / "Components" / "ARDS.pipeline").exists()
    assert summary["rendered_file_count"] >= 2
    assert summary["runtime_project_dir"] == runtime_project_dir


def test_sync_clamp_project_refuses_conflicting_existing_file_without_overwrite(
    tmp_path: Path,
) -> None:
    source = _source_project(tmp_path)
    live = tmp_path / "workspace" / "ARDS"
    target = live / "Components" / "ARDS.pipeline"
    target.parent.mkdir(parents=True)
    target.write_text("manual edit", encoding="utf-8")

    with pytest.raises(FileExistsError):
        sync_clamp_ards_project(
            source_dir=source,
            live_dir=live,
            runtime_project_dir="C:/ClampWin_1.6.6/workspace/ARDS",
            artifact_dir=tmp_path / "artifacts",
            summary_path=tmp_path / "artifacts" / "project_sync_summary.json",
        )


def test_sync_clamp_project_preflights_conflicts_before_writing_any_file(tmp_path: Path) -> None:
    source = tmp_path / "source" / "clamp_ARDS"
    (source / "A").mkdir(parents=True)
    (source / "B").mkdir(parents=True)
    (source / "A" / "new.pipeline").write_text("new", encoding="utf-8")
    (source / "B" / "conflict.pipeline").write_text("source", encoding="utf-8")
    live = tmp_path / "workspace" / "ARDS"
    conflict = live / "B" / "conflict.pipeline"
    conflict.parent.mkdir(parents=True)
    conflict.write_text("manual edit", encoding="utf-8")

    with pytest.raises(FileExistsError):
        sync_clamp_ards_project(
            source_dir=source,
            live_dir=live,
            runtime_project_dir="C:/ClampWin_1.6.6/workspace/ARDS",
            artifact_dir=tmp_path / "artifacts",
            summary_path=tmp_path / "artifacts" / "project_sync_summary.json",
        )

    assert not (live / "A" / "new.pipeline").exists()
    assert conflict.read_text(encoding="utf-8") == "manual edit"


def test_sync_clamp_project_dry_run_writes_no_live_files(tmp_path: Path) -> None:
    source = _source_project(tmp_path)
    live = tmp_path / "workspace" / "ARDS"

    sync_clamp_ards_project(
        source_dir=source,
        live_dir=live,
        runtime_project_dir="C:/ClampWin_1.6.6/workspace/ARDS",
        artifact_dir=tmp_path / "artifacts",
        summary_path=tmp_path / "artifacts" / "project_sync_summary.json",
        dry_run=True,
    )

    assert not live.exists()
    assert (tmp_path / "artifacts" / "project_sync_summary.json").exists()


def test_sync_clamp_project_skips_data_input_and_output(tmp_path: Path) -> None:
    source = _source_project(tmp_path)
    (source / "Data" / "Input").mkdir(parents=True)
    (source / "Data" / "Output").mkdir(parents=True)
    (source / "Data" / "Input" / "phi.txt").write_text("phi", encoding="utf-8")
    (source / "Data" / "Output" / "out.txt").write_text("out", encoding="utf-8")

    sync_clamp_ards_project(
        source_dir=source,
        live_dir=tmp_path / "workspace" / "ARDS",
        runtime_project_dir="C:/ClampWin_1.6.6/workspace/ARDS",
        artifact_dir=tmp_path / "artifacts",
        summary_path=tmp_path / "artifacts" / "project_sync_summary.json",
    )

    assert not (tmp_path / "workspace" / "ARDS" / "Data" / "Input" / "phi.txt").exists()
    assert not (tmp_path / "workspace" / "ARDS" / "Data" / "Output" / "out.txt").exists()


def test_example_config_loads_clamp_ards_defaults() -> None:
    config = load_config(Path("config/config.example.yaml"))

    assert config.clamp_ards.project_name == "ARDS"
    assert config.clamp_ards.project_source_dir.as_posix().endswith(
        "data/external/clamp_ards_project"
    )
    assert config.clamp_ards.runtime_project_dir == "C:/ClampWin_1.6.6/workspace/ARDS"
    assert str(config.clamp_ards.input_dir).endswith(
        "artifacts/restricted/clamp_ards/workspace/ARDS/Data/Input"
    )
    assert config.clamp_ards.input_manifest.name == "input_manifest.csv"
    assert config.clamp_ards.teacher_entity_output.name == "clamp_legacy_entities.parquet"
    assert config.clamp_ards.teacher_benchmark_dir.name == "teacher_benchmark"
    assert config.clamp_ards.full_output_archive.name == "ARDS_CLAMP_Output_full.zip"
    assert config.clamp_ards.python_entity_output.name == "clamp_python_entities.parquet"
    assert config.clamp_ards.python_parity_summary.name == "parity_summary.json"


@pytest.mark.parametrize(
    ("path", "message"),
    [
        (Path("clamp/Input"), "absolute"),
        (get_paths().root / "clamp" / "Input", "outside the repository root"),
        (Path("/tmp/YOUR_LOCAL_CLAMP_WORKSPACE_ROOT/ARDS"), "YOUR_LOCAL_CLAMP_WORKSPACE_ROOT"),
    ],
)
def test_clamp_operational_path_validation_rejects_unsafe_paths(
    path: Path,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        validate_clamp_ards_operational_paths(input_dir=path)


def test_clamp_operational_path_validation_accepts_absolute_path_outside_repo(
    tmp_path: Path,
) -> None:
    validate_clamp_ards_operational_paths(input_dir=tmp_path)


def test_clamp_operational_path_validation_accepts_ignored_repo_restricted_path() -> None:
    restricted_path = get_paths().root / CLAMP_RESTRICTED_ARTIFACT_DIR / "workspace" / "ARDS"

    validate_clamp_ards_operational_paths(project_live_dir=restricted_path)


def test_sync_script_rejects_placeholder_clamp_workspace() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/sync_clamp_ards_project.py",
            "--config",
            "config/config.example.yaml",
            "--project-live-dir",
            "/tmp/YOUR_LOCAL_CLAMP_WORKSPACE_ROOT/ARDS",
            "--dry-run",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "YOUR_LOCAL_CLAMP_WORKSPACE_ROOT" in result.stderr


def test_sync_script_rejects_relative_project_live_dir() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/sync_clamp_ards_project.py",
            "--config",
            "config/config.example.yaml",
            "--project-live-dir",
            "clamp/ARDS",
            "--dry-run",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "absolute path" in result.stderr


def test_export_script_rejects_placeholder_clamp_workspace(tmp_path: Path) -> None:
    source = tmp_path / "source.csv"
    source.write_text("doc_id,report\ns1,text\n", encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            "scripts/export_clamp_ards_inputs.py",
            "--config",
            "config/config.example.yaml",
            "--source-file",
            str(source),
            "--id-col",
            "doc_id",
            "--text-col",
            "report",
            "--project-live-dir",
            "/tmp/YOUR_LOCAL_CLAMP_WORKSPACE_ROOT/ARDS",
            "--input-dir",
            "/tmp/YOUR_LOCAL_CLAMP_WORKSPACE_ROOT/ARDS/Data/Input",
            "--output-dir",
            "/tmp/YOUR_LOCAL_CLAMP_WORKSPACE_ROOT/ARDS/Data/Output",
            "--dry-run",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "YOUR_LOCAL_CLAMP_WORKSPACE_ROOT" in result.stderr


def test_export_script_rejects_relative_input_dir(tmp_path: Path) -> None:
    source = tmp_path / "source.csv"
    source.write_text("doc_id,report\ns1,text\n", encoding="utf-8")
    outside = tmp_path / "clamp" / "ARDS"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/export_clamp_ards_inputs.py",
            "--config",
            "config/config.example.yaml",
            "--source-file",
            str(source),
            "--id-col",
            "doc_id",
            "--text-col",
            "report",
            "--project-live-dir",
            str(outside),
            "--input-dir",
            "clamp/Input",
            "--output-dir",
            str(outside / "Data" / "Output"),
            "--dry-run",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "absolute path" in result.stderr


def test_sync_script_runtime_project_dir_override_normalizes_descriptors(tmp_path: Path) -> None:
    source = _source_project(tmp_path)
    live = tmp_path / "workspace" / "ARDS"
    artifact_dir = tmp_path / "artifacts"
    runtime_project_dir = "E:/ClampRuntime/workspace/ARDS"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/sync_clamp_ards_project.py",
            "--config",
            "config/config.example.yaml",
            "--project-source-dir",
            str(source),
            "--project-live-dir",
            str(live),
            "--runtime-project-dir",
            runtime_project_dir,
            "--artifact-dir",
            str(artifact_dir),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    descriptor = (live / "descriptor" / "defaultEngine.xml").read_text(encoding="utf-8")
    assert runtime_project_dir in descriptor
    assert live.as_posix() not in descriptor


def test_placeholder_clamp_workspace_is_git_ignored() -> None:
    result = subprocess.run(
        ["git", "check-ignore", "YOUR_LOCAL_CLAMP_WORKSPACE_ROOT/example.txt"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0


@pytest.mark.parametrize(
    "script",
    [
        "scripts/sync_clamp_ards_project.py",
        "scripts/export_clamp_ards_inputs.py",
        "scripts/parse_clamp_ards_outputs.py",
        "scripts/benchmark_clamp_ards_teacher.py",
    ],
)
def test_clamp_scripts_render_help(script: str) -> None:
    result = subprocess.run(
        [sys.executable, script, "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "usage:" in result.stdout


def test_bigquery_source_sql_renders_without_credentials() -> None:
    module = _load_export_script()
    sql = module.build_bigquery_source_sql(
        source_table=module.normalize_source_table(
            None,
            default="mimic-hypercapnia.ards_mimic_cxr_benchmark.model_development_extract",
        ),
        text_col="primary_target_text",
        limit=25,
    )

    assert "FROM `mimic-hypercapnia.ards_mimic_cxr_benchmark.model_development_extract`" in sql
    assert "`primary_target_text`" in sql
    assert "LIMIT 25" in sql


def _load_export_script() -> ModuleType:
    path = Path("scripts/export_clamp_ards_inputs.py")
    spec = importlib.util.spec_from_file_location("export_clamp_ards_inputs", path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"Could not load script module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _source_project(tmp_path: Path) -> Path:
    source = tmp_path / "source" / "clamp_ARDS"
    (source / "descriptor").mkdir(parents=True)
    (source / "Components").mkdir(parents=True)
    (source / "descriptor" / "defaultEngine.xml").write_text(
        "<string>C:/ClampWin_1.6.6/workspace/ARDS/descriptor</string>",
        encoding="utf-8",
    )
    (source / "Components" / "ARDS.pipeline").write_text("<Pipeline />", encoding="utf-8")
    return source
