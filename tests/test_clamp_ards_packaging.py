from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path


def test_wheel_bundles_manifest_and_enforces_it_outside_checkout(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    build_root = tmp_path / "package-source"
    shutil.copytree(root / "src", build_root / "src")
    for filename in ("LICENSE", "README.md", "pyproject.toml"):
        shutil.copy2(root / filename, build_root / filename)
    wheel_dir = tmp_path / "wheel"
    subprocess.run(
        ["uv", "build", "--wheel", "--out-dir", str(wheel_dir)],
        cwd=build_root,
        check=True,
        capture_output=True,
        text=True,
    )
    wheels = list(wheel_dir.glob("*.whl"))
    assert len(wheels) == 1
    wheel = wheels[0]
    member = "ards_cxr_benchmark/clamp_ards/data/clamp_ards_resource_manifest.json"
    spec_member = "ards_cxr_benchmark/clamp_ards/data/legacy_ards_phenotype_spec.json"
    with zipfile.ZipFile(wheel) as archive:
        assert member in archive.namelist()
        assert spec_member in archive.namelist()

    outside = tmp_path / "outside"
    outside.mkdir()
    project = root / "tests/fixtures/clamp_ards_external_resources"
    manifest_path = project / "manifest.json"
    valid = _run_wheel_resource_load(wheel, project, manifest_path, outside)
    assert valid.returncode == 0, valid.stderr
    assert str(wheel) in valid.stdout
    assert valid.stdout.rstrip().endswith("3")

    manifest = json.loads(manifest_path.read_text())
    tampered = tmp_path / "tampered_project"
    for relative in manifest["runtime_required_files"]:
        source = project / relative
        destination = tampered / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    token_rules = tampered / "Components/Tokenizer/DF_Clamp_tokenizer/defaultTokenRule.txt"
    payload = token_rules.read_bytes()
    token_rules.write_bytes(payload[:-1] + b" ")

    rejected = _run_wheel_resource_load(wheel, tampered, manifest_path, outside)
    assert rejected.returncode != 0
    assert "hashes differ from frozen manifest" in rejected.stderr


def _run_wheel_resource_load(
    wheel: Path,
    project: Path,
    manifest: Path,
    working_dir: Path,
) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    existing = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = str(wheel) if not existing else f"{wheel}{os.pathsep}{existing}"
    return subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from pathlib import Path; "
                "import ards_cxr_benchmark.clamp_ards.resources as resources; "
                "loaded = resources.load_clamp_resources("
                "Path(__import__('sys').argv[1]), "
                "manifest_path=Path(__import__('sys').argv[2])); "
                "print(resources.__file__); print(len(loaded.resource_sha256))"
            ),
            str(project),
            str(manifest),
        ],
        cwd=working_dir,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
