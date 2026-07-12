from __future__ import annotations

import os
import subprocess
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from .config import ExternalRepositoryConfig


def fetch_pinned_repository(
    source: ExternalRepositoryConfig,
    *,
    allowed_untracked_paths: Iterable[Path] = (),
) -> dict[str, Any]:
    destination = source.external_repo_dir
    if destination.exists():
        return verify_external_repository(
            source,
            allowed_untracked_paths=allowed_untracked_paths,
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "clone", "--no-checkout", source.repository, str(destination)],
        check=True,
    )
    subprocess.run(
        ["git", "checkout", "--detach", source.commit],
        cwd=destination,
        check=True,
    )
    return verify_external_repository(
        source,
        allowed_untracked_paths=allowed_untracked_paths,
    )


def verify_external_repository(
    source: ExternalRepositoryConfig,
    *,
    allowed_untracked_paths: Iterable[Path] = (),
) -> dict[str, Any]:
    root = source.external_repo_dir
    if not (root / ".git").exists():
        raise FileNotFoundError(f"External repository is missing: {root}")
    head = _git(root, "rev-parse", "HEAD")
    if head != source.commit:
        raise ValueError(
            f"External repository HEAD mismatch: expected {source.commit}, found {head}"
        )
    allowed = _allowed_untracked_relative_paths(root, allowed_untracked_paths)
    dirty = [
        (status, path)
        for status, path in _git_status_entries(root)
        if status != "??" or path not in allowed
    ]
    if dirty:
        raise ValueError(f"External repository has local modifications: {root}: {dirty}")
    remote = _git(root, "remote", "get-url", "origin")
    expected_repo = source.repository.removesuffix(".git")
    observed_repo = remote.removesuffix(".git")
    if expected_repo != observed_repo:
        raise ValueError(
            f"External repository remote mismatch: expected {source.repository}, found {remote}"
        )
    return {
        "repository": source.repository,
        "commit": head,
        "license": source.license,
        "external_repo_dir": str(root),
        "clean": True,
    }


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _allowed_untracked_relative_paths(root: Path, paths: Iterable[Path]) -> set[str]:
    root = root.resolve()
    allowed: set[str] = set()
    for path in paths:
        try:
            relative = path.resolve().relative_to(root)
        except ValueError:
            continue
        allowed.add(relative.as_posix())
    return allowed


def _git_status_entries(root: Path) -> list[tuple[str, str]]:
    output = subprocess.run(
        ["git", "status", "--porcelain=v1", "-z", "--untracked-files=all"],
        cwd=root,
        check=True,
        capture_output=True,
    ).stdout
    records = output.split(b"\0")
    entries: list[tuple[str, str]] = []
    index = 0
    while index < len(records):
        record = records[index]
        if not record:
            index += 1
            continue
        status = record[:2].decode("ascii")
        entries.append((status, Path(os.fsdecode(record[3:])).as_posix()))
        if "R" in status or "C" in status:
            index += 1
        index += 1
    return entries
