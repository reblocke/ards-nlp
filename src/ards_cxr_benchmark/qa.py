from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from .config import ensure_parent_dir


@dataclass(frozen=True)
class ValidationIssue:
    check: str
    severity: str
    message: str


def missing_columns(columns: list[str] | set[str], required: list[str]) -> list[str]:
    present = set(columns)
    return [column for column in required if column not in present]


def validate_required_columns(
    columns: list[str] | set[str],
    required: list[str],
    *,
    check_name: str,
) -> list[ValidationIssue]:
    missing = missing_columns(columns, required)
    if not missing:
        return []
    return [
        ValidationIssue(
            check=check_name,
            severity="error",
            message=f"Missing required columns: {missing}",
        )
    ]


def write_validation_issues(path: Path, issues: list[ValidationIssue]) -> None:
    ensure_parent_dir(path)
    path.write_text(
        json.dumps([asdict(issue) for issue in issues], indent=2) + "\n",
        encoding="utf-8",
    )
