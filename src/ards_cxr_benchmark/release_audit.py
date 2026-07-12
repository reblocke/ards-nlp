from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote

IGNORED_TRACKED_ROOTS = (
    "artifacts/",
    "data/raw/",
    "data/processed/",
    "data/derived/",
    "data/external/",
    "reports/",
)
ALLOWED_TRACKED_OUTPUT_FILES = {
    "artifacts/.gitkeep",
    "data/raw/.gitkeep",
    "data/processed/.gitkeep",
    "data/derived/.gitkeep",
    "data/external/.gitkeep",
    "reports/.gitkeep",
}
ALLOWED_HISTORICAL_OUTPUT_FILES = {
    *ALLOWED_TRACKED_OUTPUT_FILES,
    "data/raw/example.csv",
}
FORBIDDEN_EXACT_PATHS = {"config/config.yaml"}
FORBIDDEN_OUTPUT_SUFFIXES = {
    ".docm",
    ".docx",
    ".dta",
    ".feather",
    ".gz",
    ".joblib",
    ".parquet",
    ".pdf",
    ".pickle",
    ".pkl",
    ".pptm",
    ".pptx",
    ".rds",
    ".sav",
    ".xlsm",
    ".xlsx",
    ".xmi",
    ".zip",
}
ALLOWED_BINARY_ARTIFACT_PATHS: set[str] = set()
ALLOWED_HISTORICAL_BINARY_ARTIFACT_PATHS: set[str] = set()
LOCAL_LINK_PATTERN = re.compile(r"\[[^\]]*\]\(([^)]+)\)")
URL_PATTERN = re.compile(r"(?:https?|ftp)://\S+")
POSIX_ABSOLUTE_PATH_PATTERN = re.compile(
    r"(?<![:/\\\w~])/(?:[^/\\\s`\"'()<>\[\]]+/)+[^/\\\s`\"'()<>\[\]]*"
)
WINDOWS_ABSOLUTE_PATH_PATTERN = re.compile(
    r"(?i)(?<![A-Za-z0-9])(?:[A-Z]:[\\/]+)(?:[^\\/\s`\"'()<>\[\]]+[\\/]+)*"
    r"[^\\/\s`\"'()<>\[\]]+"
)
UNC_ABSOLUTE_PATH_PATTERN = re.compile(
    r"(?i)(?<![A-Za-z0-9\\])\\{2,}[A-Za-z0-9][A-Za-z0-9.-]*"
    r"\\{1,2}[A-Za-z0-9_$.-]+"
)
ALLOWED_ABSOLUTE_PATHS = ("C:/ClampWin_1.6.6/workspace/ARDS",)
ALLOWED_ABSOLUTE_PATH_PREFIXES = ("/absolute/path/to/", "/path/to/")
MACHINE_PATH_TEXT_SUFFIXES = {
    ".cff",
    ".cfg",
    ".conf",
    ".csv",
    ".ini",
    ".json",
    ".md",
    ".qmd",
    ".toml",
    ".tsv",
    ".txt",
    ".xml",
    ".yaml",
    ".yml",
}


@dataclass(frozen=True)
class AuditFinding:
    check: str
    path: str
    detail: str


def audit_repository(root: Path) -> list[AuditFinding]:
    root = root.resolve()
    tracked = tracked_files(root)
    findings = audit_tracked_paths(tracked)
    findings.extend(audit_historical_paths(historical_files(root)))
    findings.extend(audit_machine_paths(root, tracked))
    findings.extend(audit_local_markdown_links(root, tracked))
    return sorted(findings, key=lambda item: (item.check, item.path, item.detail))


def tracked_files(root: Path) -> list[str]:
    result = subprocess.run(
        ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
        cwd=root,
        check=True,
        capture_output=True,
    )
    return [value.decode("utf-8") for value in result.stdout.split(b"\0") if value]


def historical_files(root: Path) -> list[str]:
    has_head = subprocess.run(
        ["git", "rev-parse", "--verify", "HEAD"],
        cwd=root,
        check=False,
        capture_output=True,
    )
    if has_head.returncode != 0:
        return []
    result = subprocess.run(
        ["git", "log", "--all", "--pretty=format:", "--name-only"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return sorted({line.strip() for line in result.stdout.splitlines() if line.strip()})


def audit_tracked_paths(paths: list[str]) -> list[AuditFinding]:
    findings: list[AuditFinding] = []
    for path in paths:
        if path in FORBIDDEN_EXACT_PATHS:
            findings.append(
                AuditFinding("restricted_path", path, "local config must not be tracked")
            )
        if path.startswith(IGNORED_TRACKED_ROOTS) and path not in ALLOWED_TRACKED_OUTPUT_FILES:
            findings.append(
                AuditFinding("restricted_path", path, "generated or restricted output is tracked")
            )
        if (
            Path(path).suffix.lower() in FORBIDDEN_OUTPUT_SUFFIXES
            and path not in ALLOWED_BINARY_ARTIFACT_PATHS
        ):
            findings.append(
                AuditFinding("forbidden_extension", path, "binary/data artifact is tracked")
            )
    return findings


def audit_historical_paths(paths: list[str]) -> list[AuditFinding]:
    findings: list[AuditFinding] = []
    for path in paths:
        if path in FORBIDDEN_EXACT_PATHS:
            findings.append(
                AuditFinding("historical_restricted_path", path, "local config exists in history")
            )
        if path.startswith(IGNORED_TRACKED_ROOTS) and path not in ALLOWED_HISTORICAL_OUTPUT_FILES:
            findings.append(
                AuditFinding(
                    "historical_restricted_path",
                    path,
                    "generated or restricted output exists in history",
                )
            )
        if (
            Path(path).suffix.lower() in FORBIDDEN_OUTPUT_SUFFIXES
            and path not in ALLOWED_HISTORICAL_BINARY_ARTIFACT_PATHS
        ):
            findings.append(
                AuditFinding(
                    "historical_forbidden_extension",
                    path,
                    "binary/data artifact exists in history",
                )
            )
    return findings


def audit_machine_paths(root: Path, tracked: list[str]) -> list[AuditFinding]:
    findings: list[AuditFinding] = []
    for relative in tracked:
        path = root / relative
        if not path.is_file() or path.suffix.lower() not in MACHINE_PATH_TEXT_SUFFIXES:
            continue
        try:
            content = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        scrubbed = URL_PATTERN.sub("", content)
        for allowed in ALLOWED_ABSOLUTE_PATHS:
            scrubbed = scrubbed.replace(allowed, "")
            scrubbed = scrubbed.replace(allowed.replace("/", "\\"), "")
        for allowed in ALLOWED_ABSOLUTE_PATH_PREFIXES:
            scrubbed = scrubbed.replace(allowed, "")
        has_machine_path = bool(
            "file://" in content
            or POSIX_ABSOLUTE_PATH_PATTERN.search(scrubbed)
            or WINDOWS_ABSOLUTE_PATH_PATTERN.search(scrubbed)
            or UNC_ABSOLUTE_PATH_PATTERN.search(scrubbed)
        )
        if has_machine_path:
            findings.append(
                AuditFinding("machine_path", relative, "tracked text contains a machine-local path")
            )
    return findings


def audit_local_markdown_links(root: Path, tracked: list[str]) -> list[AuditFinding]:
    tracked_set = set(tracked)
    findings: list[AuditFinding] = []
    for relative in tracked:
        if Path(relative).suffix.lower() not in {".md", ".qmd"}:
            continue
        path = root / relative
        content = path.read_text(encoding="utf-8")
        for raw_target in LOCAL_LINK_PATTERN.findall(content):
            target = raw_target.strip().strip("<>").split(" ", maxsplit=1)[0]
            if not target or target.startswith(("#", "http://", "https://", "mailto:")):
                continue
            target = unquote(target).split("#", maxsplit=1)[0]
            resolved = (path.parent / target).resolve()
            try:
                target_relative = resolved.relative_to(root).as_posix()
            except ValueError:
                findings.append(
                    AuditFinding(
                        "broken_link",
                        relative,
                        f"local link escapes repository: {raw_target}",
                    )
                )
                continue
            if target_relative not in tracked_set and not resolved.is_dir():
                findings.append(
                    AuditFinding("broken_link", relative, f"missing tracked target: {raw_target}")
                )
    return findings
