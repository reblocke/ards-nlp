from __future__ import annotations

import subprocess
from pathlib import Path

from ards_cxr_benchmark.annotation_planning import (
    AnnotationPlanningConfig,
    PlanningAssumptions,
    PlanningOutputConfig,
)
from ards_cxr_benchmark.annotation_planning_render import (
    REPORT_FILENAME,
    render_annotation_planning,
)


def test_planning_renderer_places_html_in_configured_report_directory(
    tmp_path: Path,
    monkeypatch,
) -> None:
    notebook = tmp_path / "notebooks" / "02_annotation_design_planner.qmd"
    notebook.parent.mkdir()
    notebook.write_text("---\ntitle: test\n---\n", encoding="utf-8")
    config_path = tmp_path / "config" / "planning.yaml"
    config_path.parent.mkdir()
    config_path.write_text("project: {}\n", encoding="utf-8")
    config = _config(tmp_path)

    def fake_run(command, *, cwd, env, check):
        assert command[-1] == REPORT_FILENAME
        assert env["ANNOTATION_PLANNING_CONFIG"] == str(config_path.resolve())
        assert env["QUARTO_PYTHON"]
        assert check is True
        (Path(cwd) / REPORT_FILENAME).write_text("rendered", encoding="utf-8")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr("ards_cxr_benchmark.annotation_planning_render.subprocess.run", fake_run)

    rendered = render_annotation_planning(
        config,
        config_path=config_path,
        notebook_path=notebook,
    )

    assert rendered == config.outputs.report_dir / REPORT_FILENAME
    assert rendered.read_text(encoding="utf-8") == "rendered"
    assert not (notebook.parent / REPORT_FILENAME).exists()


def _config(root: Path) -> AnnotationPlanningConfig:
    return AnnotationPlanningConfig(
        analysis_mode="synthetic",
        confidence_level=0.95,
        pilot_artifact_dir=root / "artifacts" / "annotations" / "pilot" / "smoke",
        assumptions=PlanningAssumptions(
            prevalence_grid=(0.2,),
            expected_performance_grid=(0.8,),
            ci_half_width_grid=(0.1,),
            reliability_targets=(0.8,),
            overlap_fraction=0.2,
            disagreement_threshold_points=25,
            retraining_case_counts=(250,),
            image_minutes_per_rating=None,
            report_minutes_per_rating=None,
        ),
        outputs=PlanningOutputConfig(
            report_dir=root / "reports" / "custom" / "planning",
            artifact_dir=root / "artifacts" / "custom" / "planning",
        ),
        repo_root=root,
    )
