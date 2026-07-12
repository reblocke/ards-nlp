from __future__ import annotations

import csv
import re
from html.parser import HTMLParser
from pathlib import Path

REPORTS = (
    "reports/annotation_pilot/smoke/ards_annotation_pilot_agreement.html",
    "reports/annotation_planning/smoke/ards_annotation_design_scenarios.html",
)
STATIC_FORBIDDEN = (
    "interpretation_text",
    "rater_01.csv",
    "rater_02.csv",
    "rater_03.csv",
)
IDENTIFIER_COLUMNS = ("id", "id2", "id_accession")
URL_PATTERN = re.compile(r"(?:https?|ftp)://\S+")
POSIX_ABSOLUTE_PATH_PATTERN = re.compile(
    r"(?<![:/\w~])/(?:[^/\s`\"'()<>\[\]]+/)+[^/\s`\"'()<>\[\]]*"
)
WINDOWS_MACHINE_PATH_PATTERN = re.compile(
    r"(?i)(?<![A-Za-z0-9])(?:[A-Z]:[\\/])(?:[^\\/\s<>\"'`]+[\\/])*"
    r"[^\\/\s<>\"'`]+"
)


class _VisibleReportPayloadParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._ignored_depth = 0
        self.parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() in {"script", "style"}:
            self._ignored_depth += 1
        elif self._ignored_depth == 0:
            self.parts.extend(
                value for _, value in attrs if value and not value.lower().startswith("data:")
            )

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() in {"script", "style"} and self._ignored_depth:
            self._ignored_depth -= 1

    def handle_data(self, data: str) -> None:
        if self._ignored_depth == 0:
            self.parts.append(data)


def validate_smoke_reports(root: Path) -> None:
    forbidden = [*STATIC_FORBIDDEN, *smoke_fixture_identifiers(root)]
    for relative in REPORTS:
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"Synthetic annotation report not found: {path}")
        content = path.read_text(encoding="utf-8")
        leaked = [value for value in forbidden if value in content]
        if contains_machine_path(content):
            leaked.append("absolute_machine_path")
        if leaked:
            raise ValueError(f"Synthetic annotation report contains forbidden values: {leaked}")


def contains_machine_path(content: str) -> bool:
    parser = _VisibleReportPayloadParser()
    parser.feed(content)
    visible_payload = URL_PATTERN.sub("", "\n".join(parser.parts))
    return bool(
        "file://" in content
        or POSIX_ABSOLUTE_PATH_PATTERN.search(visible_payload)
        or WINDOWS_MACHINE_PATH_PATTERN.search(visible_payload)
    )


def smoke_fixture_identifiers(root: Path) -> tuple[str, ...]:
    fixture_dir = root / "tests/fixtures/redcap_annotation"
    identifiers: set[str] = set()
    for path in sorted(fixture_dir.glob("rater_*.csv")):
        with path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                for column in IDENTIFIER_COLUMNS:
                    value = str(row.get(column, "")).strip()
                    if value:
                        identifiers.add(value)
    return tuple(sorted(identifiers))
