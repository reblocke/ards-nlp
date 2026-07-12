from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


def find_repo_root(start: Path | None = None) -> Path:
    """Find the repository root by walking upward for a `pyproject.toml`.

    This avoids hard-coded absolute paths and makes scripts runnable from any CWD.
    """

    start = (start or Path.cwd()).resolve()
    for candidate in [start, *start.parents]:
        if (candidate / "pyproject.toml").exists():
            return candidate
    msg = (
        f"Could not find repo root from {start}. Expected pyproject.toml in an ancestor directory."
    )
    raise FileNotFoundError(msg)


@dataclass(frozen=True)
class ProjectPaths:
    root: Path

    @property
    def config(self) -> Path:
        return self.root / "config"

    @property
    def sql(self) -> Path:
        return self.root / "sql"

    @property
    def data(self) -> Path:
        return self.root / "data"

    @property
    def raw(self) -> Path:
        return self.data / "raw"

    @property
    def external(self) -> Path:
        return self.data / "external"

    @property
    def processed(self) -> Path:
        return self.data / "processed"

    @property
    def derived(self) -> Path:
        return self.data / "derived"

    @property
    def artifacts(self) -> Path:
        return self.root / "artifacts"

    @property
    def qa_artifacts(self) -> Path:
        return self.artifacts / "qa"

    @property
    def sample_artifacts(self) -> Path:
        return self.artifacts / "samples"

    @property
    def reports(self) -> Path:
        return self.root / "reports"


def get_paths(start: Path | None = None) -> ProjectPaths:
    return ProjectPaths(root=find_repo_root(start=start))
