from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from .annotation_planning import (
    AnnotationPlanningConfig,
    validate_annotation_planning_output_paths,
)
from .config import ensure_dir

REPORT_FILENAME = "ards_annotation_design_scenarios.html"
NOTEBOOK_FILENAME = "02_annotation_design_planner.qmd"


def render_annotation_planning(
    config: AnnotationPlanningConfig,
    *,
    config_path: Path,
    notebook_path: Path | None = None,
) -> Path:
    validate_annotation_planning_output_paths(config)
    notebook = (notebook_path or config.repo_root / "notebooks" / NOTEBOOK_FILENAME).resolve()
    if not notebook.is_file():
        raise FileNotFoundError(f"Annotation planning notebook not found: {notebook}")

    rendered = notebook.parent / REPORT_FILENAME
    destination = config.outputs.report_dir / REPORT_FILENAME
    rendered.unlink(missing_ok=True)
    environment = os.environ.copy()
    environment["ANNOTATION_PLANNING_CONFIG"] = str(config_path.resolve())
    environment["QUARTO_PYTHON"] = sys.executable
    try:
        subprocess.run(
            [
                "quarto",
                "render",
                notebook.name,
                "--to",
                "html",
                "--output",
                REPORT_FILENAME,
            ],
            cwd=notebook.parent,
            env=environment,
            check=True,
        )
        if not rendered.is_file():
            raise RuntimeError(f"Quarto did not create the expected report: {rendered}")
        ensure_dir(destination.parent)
        os.replace(rendered, destination)
    finally:
        rendered.unlink(missing_ok=True)
    return destination
