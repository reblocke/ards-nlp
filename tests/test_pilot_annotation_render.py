from __future__ import annotations

import subprocess
from pathlib import Path

from ards_cxr_benchmark.pilot_annotation_agreement import (
    AnnotationPilotConfig,
    PilotColumnConfig,
    PilotOutputConfig,
    PilotProjectConfig,
)
from ards_cxr_benchmark.pilot_annotation_render import (
    REPORT_FILENAME,
    render_annotation_pilot,
)


def test_renderer_places_html_in_configured_report_directory(
    tmp_path: Path,
    monkeypatch,
) -> None:
    notebook = tmp_path / "notebooks" / "01_redcap_pilot_agreement.qmd"
    notebook.parent.mkdir()
    notebook.write_text("---\ntitle: test\n---\n", encoding="utf-8")
    config_path = tmp_path / "config" / "pilot.yaml"
    config_path.parent.mkdir()
    config_path.write_text("project: {}\n", encoding="utf-8")
    config = _config(tmp_path)

    def fake_run(command, *, cwd, env, check):
        assert command[-1] == REPORT_FILENAME
        assert env["ANNOTATION_PILOT_CONFIG"] == str(config_path.resolve())
        assert env["QUARTO_PYTHON"]
        assert check is True
        (Path(cwd) / REPORT_FILENAME).write_text("rendered", encoding="utf-8")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr("ards_cxr_benchmark.pilot_annotation_render.subprocess.run", fake_run)

    rendered = render_annotation_pilot(
        config,
        config_path=config_path,
        notebook_path=notebook,
    )

    assert rendered == config.outputs.report_dir / REPORT_FILENAME
    assert rendered.read_text(encoding="utf-8") == "rendered"
    assert not (notebook.parent / REPORT_FILENAME).exists()


def _config(root: Path) -> AnnotationPilotConfig:
    return AnnotationPilotConfig(
        project=PilotProjectConfig(3, 2, 1, 10),
        inputs={"R01": root / "r1.csv", "R02": root / "r2.csv", "R03": root / "r3.csv"},
        columns=PilotColumnConfig(
            case_id="case",
            report_rating="report",
            report_complete="report_complete",
            image_rating="image",
            image_complete="image_complete",
        ),
        outputs=PilotOutputConfig(
            report_dir=root / "reports" / "custom" / "pilot",
            artifact_dir=root / "artifacts" / "custom" / "pilot",
            derived_dir=root / "data" / "derived" / "custom" / "pilot",
        ),
        repo_root=root,
    )
