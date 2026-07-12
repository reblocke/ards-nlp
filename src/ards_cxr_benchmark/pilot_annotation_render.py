from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from .config import ensure_dir
from .pilot_annotation_agreement import (
    AnnotationPilotConfig,
    validate_annotation_pilot_output_paths,
)

REPORT_FILENAME = "ards_annotation_pilot_agreement.html"
NOTEBOOK_FILENAME = "01_redcap_pilot_agreement.qmd"


def render_annotation_pilot(
    config: AnnotationPilotConfig,
    *,
    config_path: Path,
    notebook_path: Path | None = None,
) -> Path:
    """Render the pilot notebook and place the report at its configured destination."""

    validate_annotation_pilot_output_paths(config)
    notebook = notebook_path or config.repo_root / "notebooks" / NOTEBOOK_FILENAME
    notebook = notebook.resolve()
    if not notebook.is_file():
        raise FileNotFoundError(f"Annotation pilot notebook not found: {notebook}")

    rendered = notebook.parent / REPORT_FILENAME
    destination = config.outputs.report_dir / REPORT_FILENAME
    rendered.unlink(missing_ok=True)
    environment = os.environ.copy()
    environment["ANNOTATION_PILOT_CONFIG"] = str(config_path.resolve())
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
