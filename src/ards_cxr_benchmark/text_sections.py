from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

SECTION_NAMES = [
    "FINAL REPORT",
    "EXAMINATION",
    "INDICATION",
    "TECHNIQUE",
    "COMPARISON",
    "FINDINGS",
    "IMPRESSION",
    "ADDENDUM",
]


@dataclass(frozen=True)
class TargetTexts:
    target_text_full_report: str
    target_text_impression_findings: str
    target_text_impression_fallback: str
    primary_target_text: str


def parse_subject_study_from_path(path: Path) -> tuple[int, int]:
    subject_match = re.search(r"p(?P<subject_id>\d{8})", str(path))
    study_match = re.search(r"s(?P<study_id>\d{8})\.txt$", path.name)

    if subject_match is None or study_match is None:
        raise ValueError(f"Could not parse subject/study from path: {path}")

    return int(subject_match.group("subject_id")), int(study_match.group("study_id"))


def normalize_report_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_section(text: str, section_name: str) -> str | None:
    other_headers = "|".join(re.escape(name) for name in SECTION_NAMES)
    pattern = re.compile(
        rf"(?ims)^\s*{re.escape(section_name)}\s*:?\s*"
        rf"(?P<body>.*?)"
        rf"(?=^\s*(?:{other_headers})\s*:?\s*$|\Z)"
    )
    match = pattern.search(text)
    if match is None:
        return None

    body = match.group("body").strip()
    return body or None


def extract_last_paragraph(text: str) -> str | None:
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    return paragraphs[-1] if paragraphs else None


def build_target_texts(
    report_text: str,
    findings_text: str | None,
    impression_text: str | None,
    last_paragraph_text: str | None,
    *,
    primary_scope: str = "full_report",
) -> TargetTexts:
    impression_findings = "\n".join(
        part for part in [impression_text, findings_text] if part
    ).strip()
    fallback = impression_text or findings_text or last_paragraph_text or report_text

    if primary_scope == "full_report":
        primary = report_text
    elif primary_scope == "impression_findings":
        primary = impression_findings or fallback
    elif primary_scope == "impression_fallback":
        primary = fallback
    else:
        raise ValueError(f"Unknown primary text scope: {primary_scope}")

    return TargetTexts(
        target_text_full_report=report_text,
        target_text_impression_findings=impression_findings,
        target_text_impression_fallback=fallback,
        primary_target_text=primary,
    )


def build_report_rows(report_root: Path, *, primary_scope: str = "full_report") -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for report_path in sorted(report_root.rglob("s*.txt")):
        subject_id, study_id = parse_subject_study_from_path(report_path)
        raw_text = report_path.read_text(encoding="utf-8", errors="replace")
        report_text = normalize_report_text(raw_text)

        findings_text = extract_section(report_text, "FINDINGS")
        impression_text = extract_section(report_text, "IMPRESSION")
        addendum_text = extract_section(report_text, "ADDENDUM")
        last_paragraph_text = extract_last_paragraph(report_text)
        target_texts = build_target_texts(
            report_text,
            findings_text,
            impression_text,
            last_paragraph_text,
            primary_scope=primary_scope,
        )

        rows.append(
            {
                "subject_id": subject_id,
                "study_id": study_id,
                "report_path": str(report_path),
                "report_text": report_text,
                "findings_text": findings_text,
                "impression_text": impression_text,
                "addendum_text": addendum_text,
                "last_paragraph_text": last_paragraph_text,
                "target_text_full_report": target_texts.target_text_full_report,
                "target_text_impression_findings": (target_texts.target_text_impression_findings),
                "target_text_impression_fallback": target_texts.target_text_impression_fallback,
                "primary_target_text": target_texts.primary_target_text,
                "has_findings": findings_text is not None,
                "has_impression": impression_text is not None,
                "has_addendum": addendum_text is not None,
            }
        )

    reports = pd.DataFrame(rows)
    if reports.empty:
        raise ValueError(f"No reports found under {report_root}")

    if reports["study_id"].duplicated().any():
        duplicated = reports.loc[reports["study_id"].duplicated(), "study_id"].head().tolist()
        raise ValueError(f"Duplicate study_id values detected: {duplicated}")

    return reports
