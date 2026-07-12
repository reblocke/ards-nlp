from __future__ import annotations

import os
from pathlib import Path

import pytest

from ards_cxr_benchmark.clamp_ards.fixtures import generate_fixture

ROOT = Path(__file__).resolve().parents[1]
SYNTHETIC_CLAMP_PROJECT = ROOT / "tests/fixtures/clamp_ards_external_resources"
SYNTHETIC_CLAMP_MANIFEST = SYNTHETIC_CLAMP_PROJECT / "manifest.json"
CONFIGURED_CLAMP_PROJECT = os.environ.get("ARDS_CLAMP_PROJECT_DIR")
LICENSED_CLAMP_CONFIGURED = bool(
    CONFIGURED_CLAMP_PROJECT
    and Path(CONFIGURED_CLAMP_PROJECT).expanduser().resolve() != SYNTHETIC_CLAMP_PROJECT.resolve()
)
os.environ.setdefault("ARDS_CLAMP_PROJECT_DIR", str(SYNTHETIC_CLAMP_PROJECT))
os.environ.setdefault(
    "ARDS_CLAMP_RESOURCE_MANIFEST",
    str(ROOT / "config/clamp_ards_resource_manifest.json")
    if LICENSED_CLAMP_CONFIGURED
    else str(SYNTHETIC_CLAMP_MANIFEST),
)


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    if LICENSED_CLAMP_CONFIGURED:
        return
    marker = pytest.mark.skip(
        reason="requires ARDS_CLAMP_PROJECT_DIR with separately licensed CLAMP resources"
    )
    for item in items:
        if "licensed_clamp" in item.keywords:
            item.add_marker(marker)


@pytest.fixture(scope="session")
def pending_clamp_fixture(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Generate the pending synthetic CLAMP contract without a tracked fixture tree."""

    root = tmp_path_factory.mktemp("pending-clamp-fixture") / "fixture"
    generate_fixture(
        root,
        project_dir=SYNTHETIC_CLAMP_PROJECT,
        resource_manifest_path=SYNTHETIC_CLAMP_MANIFEST,
    )
    return root
