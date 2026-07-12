from __future__ import annotations

import csv
import hashlib
import json
import re
import subprocess
from pathlib import Path

import pytest
import yaml

from ards_cxr_benchmark.clamp_ards.governance import audit_clamp_resources


def test_all_github_actions_are_read_only_and_sha_pinned() -> None:
    root = Path(__file__).resolve().parents[1]
    for path in sorted((root / ".github/workflows").glob("*.yml")):
        workflow = yaml.load(path.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)
        assert workflow["permissions"] == {"contents": "read"}
        for job in workflow["jobs"].values():
            for step in job["steps"]:
                uses = str(step.get("uses", ""))
                if uses:
                    assert re.fullmatch(r"[^@]+@[0-9a-f]{40}", uses), (path, uses)


def test_release_workflow_has_hard_resource_and_explicit_fixture_status_jobs() -> None:
    root = Path(__file__).resolve().parents[1]
    workflow = yaml.load(
        (root / ".github/workflows/clamp-release.yml").read_text(encoding="utf-8"),
        Loader=yaml.BaseLoader,
    )

    assert {"pull_request", "push", "workflow_dispatch"} <= set(workflow["on"])
    assert workflow["permissions"] == {"contents": "read"}
    assert set(workflow["jobs"]) == {"fixture-status", "resource-audit"}

    fixture_steps = workflow["jobs"]["fixture-status"]["steps"]
    fixture_commands = "\n".join(step.get("run", "") for step in fixture_steps)
    assert "make clamp-ards-parity-fixture-prepare" in fixture_commands
    assert "make clamp-ards-parity-fixture-validate" in fixture_commands
    assert "Status: pending" in fixture_commands
    assert all("continue-on-error" not in step for step in fixture_steps)

    resource_steps = workflow["jobs"]["resource-audit"]["steps"]
    checkout = next(
        step for step in resource_steps if str(step.get("uses", "")).startswith("actions/checkout@")
    )
    assert checkout["with"]["fetch-depth"] == "0"
    resource_commands = "\n".join(step.get("run", "") for step in resource_steps)
    assert "make clamp-ards-resources-public-audit" in resource_commands
    assert "make release-audit" in resource_commands
    assert all("continue-on-error" not in step for step in resource_steps)

    for job in workflow["jobs"].values():
        for step in job["steps"]:
            uses = str(step.get("uses", ""))
            if uses:
                assert uses.rsplit("@", maxsplit=1)[1].split()[0].isalnum()


def test_repository_resource_ledger_is_complete_and_public_boundary_is_clear() -> None:
    root = Path(__file__).resolve().parents[1]

    result = audit_clamp_resources(root, public_release=False)

    assert result.valid
    assert result.file_count == 23
    assert result.unresolved_count > 0
    assert not result.public_release_blocked


def test_public_resource_audit_rejects_unresolved_tracked_file(tmp_path: Path) -> None:
    resource = tmp_path / "clamp_ARDS" / "Components" / "resource.txt"
    resource.parent.mkdir(parents=True)
    resource.write_text("frozen", encoding="utf-8")
    digest = hashlib.sha256(resource.read_bytes()).hexdigest()
    config = tmp_path / "config"
    docs = tmp_path / "docs"
    config.mkdir()
    docs.mkdir()
    (config / "clamp_ards_resource_manifest.json").write_text(
        json.dumps(
            {
                "files": {"Components/resource.txt": digest},
                "file_sizes": {"Components/resource.txt": resource.stat().st_size},
                "runtime_required_files": ["Components/resource.txt"],
            }
        ),
        encoding="utf-8",
    )
    with (docs / "CLAMP_ARDS_RESOURCE_LEDGER.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "path",
                "sha256",
                "bytes",
                "origin",
                "author_or_rightsholder",
                "license_spdx",
                "license_evidence",
                "runtime_role",
                "redistribution_status",
                "disposition",
                "verification_needed",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "path": "Components/resource.txt",
                "sha256": digest,
                "bytes": resource.stat().st_size,
                "origin": "test",
                "author_or_rightsholder": "unknown",
                "license_spdx": "NOASSERTION",
                "license_evidence": "none",
                "runtime_role": "required",
                "redistribution_status": "unresolved",
                "disposition": "user-supplied local resource boundary",
                "verification_needed": "permission",
            }
        )

    result = audit_clamp_resources(
        tmp_path,
        public_release=True,
        tracked_files={"clamp_ARDS/Components/resource.txt"},
    )

    assert not result.valid
    assert {finding.check for finding in result.findings} == {
        "unresolved_tracked_resource",
    }

    boundary_only = audit_clamp_resources(
        tmp_path,
        public_release=True,
        tracked_files=set(),
    )
    assert boundary_only.valid
    assert not boundary_only.public_release_blocked


@pytest.mark.parametrize(
    ("runtime_required", "error"),
    [
        (["Components/resource.txt", "Components/resource.txt"], "Duplicate"),
        (["Components/missing.txt"], "absent from manifest files"),
        (["../resource.txt"], "Unsafe"),
        ([r"C:\resource.txt"], "Unsafe"),
    ],
)
def test_runtime_required_files_must_be_unique_safe_manifest_members(
    tmp_path: Path,
    runtime_required: list[str],
    error: str,
) -> None:
    _write_resource_audit_fixture(tmp_path, runtime_required=runtime_required)

    with pytest.raises(ValueError, match=error):
        audit_clamp_resources(tmp_path, public_release=False, tracked_files=set())


@pytest.mark.parametrize(
    ("license_spdx", "license_evidence", "missing_field"),
    [
        ("NOASSERTION", "Documented permission", "license_spdx"),
        ("unknown", "Documented permission", "license_spdx"),
        ("MIT", "pending review", "license_evidence"),
    ],
)
def test_cleared_status_requires_affirmative_license_evidence(
    tmp_path: Path,
    license_spdx: str,
    license_evidence: str,
    missing_field: str,
) -> None:
    _write_resource_audit_fixture(
        tmp_path,
        runtime_required=[],
        redistribution_status="cleared",
        license_spdx=license_spdx,
        license_evidence=license_evidence,
        disposition="retain",
    )

    result = audit_clamp_resources(tmp_path, public_release=False, tracked_files=set())

    assert not result.valid
    assert result.public_release_blocked
    assert [(finding.check, finding.path) for finding in result.findings] == [
        ("cleared_without_affirmative_evidence", "Components/resource.txt")
    ]
    assert missing_field in result.findings[0].detail


def test_public_blocked_tracks_local_resource_boundary_finding(tmp_path: Path) -> None:
    _write_resource_audit_fixture(
        tmp_path,
        runtime_required=[],
        redistribution_status="cleared",
        license_spdx="MIT",
        license_evidence="Repository MIT license",
        disposition="user-supplied local resource boundary",
    )

    result = audit_clamp_resources(
        tmp_path,
        public_release=True,
        tracked_files={"clamp_ARDS/Components/resource.txt"},
    )

    assert not result.valid
    assert result.public_release_blocked
    assert {finding.check for finding in result.findings} == {"excluded_tracked_resource"}


def test_public_blocked_tracks_unreviewed_tracked_resource_finding(tmp_path: Path) -> None:
    _write_resource_audit_fixture(
        tmp_path,
        runtime_required=[],
        redistribution_status="cleared",
        license_spdx="MIT",
        license_evidence="Repository MIT license",
        disposition="retain",
    )

    result = audit_clamp_resources(
        tmp_path,
        public_release=True,
        tracked_files={"clamp_ARDS/Components/unreviewed.txt"},
    )

    assert not result.valid
    assert result.public_release_blocked
    assert {finding.check for finding in result.findings} == {"unreviewed_tracked_resource"}


def test_public_boundary_rejects_unredistributable_resource_left_in_history(
    tmp_path: Path,
) -> None:
    _write_resource_audit_fixture(
        tmp_path,
        runtime_required=[],
        redistribution_status="unresolved",
        disposition="user-supplied local resource boundary",
    )

    result = audit_clamp_resources(
        tmp_path,
        public_release=True,
        tracked_files=set(),
        historical_files={"clamp_ARDS/Components/resource.txt"},
    )

    assert not result.valid
    assert result.public_release_blocked
    assert {finding.check for finding in result.findings} == {"historical_resource_exposure"}


@pytest.mark.parametrize(
    ("redistribution_status", "disposition"),
    [
        ("unresolved", "exclude from public distribution"),
        ("cleared", "exclude because unnecessary"),
    ],
)
def test_public_boundary_finds_renamed_restricted_tracked_blob(
    tmp_path: Path,
    redistribution_status: str,
    disposition: str,
) -> None:
    _write_resource_audit_fixture(
        tmp_path,
        runtime_required=[],
        redistribution_status=redistribution_status,
        license_spdx="MIT" if redistribution_status == "cleared" else "NOASSERTION",
        license_evidence=(
            "Repository MIT license" if redistribution_status == "cleared" else "None found"
        ),
        disposition=disposition,
    )
    renamed = tmp_path / "public" / "renamed-resource.txt"
    renamed.parent.mkdir()
    (tmp_path / "clamp_ARDS/Components/resource.txt").rename(renamed)
    _initialize_git_repository(tmp_path)

    result = audit_clamp_resources(tmp_path, public_release=True)

    matches = [
        finding
        for finding in result.findings
        if finding.check == "tracked_restricted_resource_blob"
    ]
    assert [(finding.path, finding.detail) for finding in matches] == [
        (
            "public/renamed-resource.txt",
            "tracked Git blob matches unresolved or excluded resource Components/resource.txt",
        )
    ]


def test_public_boundary_finds_renamed_restricted_blob_in_reachable_history(
    tmp_path: Path,
) -> None:
    _write_resource_audit_fixture(
        tmp_path,
        runtime_required=[],
        redistribution_status="unresolved",
        disposition="exclude from public distribution",
    )
    renamed = tmp_path / "public" / "renamed-resource.txt"
    renamed.parent.mkdir()
    (tmp_path / "clamp_ARDS/Components/resource.txt").rename(renamed)
    _initialize_git_repository(tmp_path)
    renamed.unlink()
    _git(tmp_path, "add", "--all")
    _git(tmp_path, "commit", "-m", "remove renamed resource")

    result = audit_clamp_resources(tmp_path, public_release=True)

    matches = [
        finding
        for finding in result.findings
        if finding.check == "historical_restricted_resource_blob"
    ]
    assert len(matches) == 1
    assert matches[0].path.startswith("git-blob:")
    assert matches[0].detail == (
        "reachable historical Git blob matches unresolved or excluded resource "
        "Components/resource.txt"
    )


def test_public_clearance_rejects_retained_resource_with_pending_verification(
    tmp_path: Path,
) -> None:
    _write_resource_audit_fixture(
        tmp_path,
        runtime_required=[],
        redistribution_status="cleared",
        license_spdx="MIT",
        license_evidence="Repository MIT license",
        disposition="retain",
        verification_needed="Verify no copied prose remains",
    )

    result = audit_clamp_resources(
        tmp_path,
        public_release=True,
        tracked_files=set(),
        historical_files=set(),
    )

    assert [(finding.check, finding.path) for finding in result.findings] == [
        ("retained_resource_pending_verification", "Components/resource.txt")
    ]


def _write_resource_audit_fixture(
    root: Path,
    *,
    runtime_required: list[str],
    redistribution_status: str = "unresolved",
    license_spdx: str = "NOASSERTION",
    license_evidence: str = "None found",
    disposition: str = "user-supplied local resource boundary",
    verification_needed: str = "None",
) -> None:
    resource = root / "clamp_ARDS" / "Components" / "resource.txt"
    resource.parent.mkdir(parents=True)
    resource.write_text("frozen", encoding="utf-8")
    digest = hashlib.sha256(resource.read_bytes()).hexdigest()
    config = root / "config"
    docs = root / "docs"
    config.mkdir()
    docs.mkdir()
    (config / "clamp_ards_resource_manifest.json").write_text(
        json.dumps(
            {
                "files": {"Components/resource.txt": digest},
                "file_sizes": {"Components/resource.txt": resource.stat().st_size},
                "runtime_required_files": runtime_required,
            }
        ),
        encoding="utf-8",
    )
    with (docs / "CLAMP_ARDS_RESOURCE_LEDGER.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "path",
                "sha256",
                "bytes",
                "origin",
                "author_or_rightsholder",
                "license_spdx",
                "license_evidence",
                "runtime_role",
                "redistribution_status",
                "disposition",
                "verification_needed",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "path": "Components/resource.txt",
                "sha256": digest,
                "bytes": resource.stat().st_size,
                "origin": "test",
                "author_or_rightsholder": "test author",
                "license_spdx": license_spdx,
                "license_evidence": license_evidence,
                "runtime_role": "required",
                "redistribution_status": redistribution_status,
                "disposition": disposition,
                "verification_needed": verification_needed,
            }
        )


def _initialize_git_repository(root: Path) -> None:
    _git(root, "init", "--initial-branch=main")
    _git(root, "config", "user.name", "Fixture Test")
    _git(root, "config", "user.email", "fixture@example.invalid")
    _git(root, "add", "--all")
    _git(root, "commit", "-m", "initial fixture")


def _git(root: Path, *args: str) -> None:
    subprocess.run(
        ["git", *args],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
