from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
GUARD = ROOT / "scripts/check_staged_office.py"


def git(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True, check=True)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    git(tmp_path, "init", "-q")
    git(tmp_path, "config", "user.name", "Test Author")
    git(tmp_path, "config", "user.email", "author@example.invalid")
    git(tmp_path, "config", "commit.gpgsign", "false")
    git(tmp_path, "config", "core.hooksPath", str(tmp_path / "empty-hooks"))
    return tmp_path


def guard(repo: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(GUARD)], cwd=repo, capture_output=True, text=True, check=False
    )


@pytest.mark.parametrize("suffix", [".docx", ".docm", ".pptx", ".pptm", ".xlsx", ".xlsm"])
@pytest.mark.parametrize("uppercase", [False, True])
def test_rejects_staged_office_files_even_when_empty(
    repo: Path, suffix: str, uppercase: bool
) -> None:
    name = "private form" + (suffix.upper() if uppercase else suffix)
    (repo / name).touch()
    git(repo, "add", "--", name)
    result = guard(repo)
    assert result.returncode == 1
    assert name in result.stderr


def test_rejects_force_added_ignored_dated_form(repo: Path) -> None:
    shutil.copyfile(ROOT / ".gitignore", repo / ".gitignore")
    path = "docs/ARDS NLP Cohort Request bl Sep 12.docx"
    (repo / "docs").mkdir()
    (repo / path).write_bytes(b"private")
    git(repo, "check-ignore", "--", path)
    git(repo, "add", "-f", "--", path)
    assert guard(repo).returncode == 1


@pytest.mark.parametrize(
    "name",
    [
        "ARDS NLP Cohort Request.docx",
        "ARDS NLP Cohort Request_new.docx",
        "ARDS NLP Cohort Request bl Sep 9.docx",
        "ARDS NLP Cohort Request bl Sep 12.docx",
        "ARDS NLP Cohort Request future revision.docx",
    ],
)
def test_ignores_all_cohort_request_versions(repo: Path, name: str) -> None:
    shutil.copyfile(ROOT / ".gitignore", repo / ".gitignore")
    git(repo, "check-ignore", "--no-index", "--", f"docs/{name}")


def test_allows_source_changes_and_unstaged_private_files(repo: Path) -> None:
    (repo / "source.py").write_text("value = 1\n")
    (repo / "private.docx").touch()
    git(repo, "add", "source.py")
    assert guard(repo).returncode == 0


def test_allows_removal_of_tracked_office_file(repo: Path) -> None:
    (repo / "private.docx").write_bytes(b"private")
    git(repo, "add", "private.docx")
    git(repo, "commit", "-qm", "Fixture")
    git(repo, "rm", "private.docx")
    assert guard(repo).returncode == 0


def test_rejects_rename_into_office_extension(repo: Path) -> None:
    (repo / "draft.txt").write_text("private")
    git(repo, "add", "draft.txt")
    git(repo, "commit", "-qm", "Fixture")
    git(repo, "mv", "draft.txt", "draft.docx")
    assert guard(repo).returncode == 1


def test_fails_closed_outside_git_repository(tmp_path: Path) -> None:
    assert guard(tmp_path).returncode == 2
