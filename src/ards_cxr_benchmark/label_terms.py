from __future__ import annotations

import re
from pathlib import Path

import yaml

REQUIRED_TERM_GROUPS = [
    "opacity_observation_terms",
    "consolidation_or_airspace_terms",
    "edema_terms",
    "atelectasis_terms",
    "left_anatomy_terms",
    "right_anatomy_terms",
    "bilateral_anatomy_terms",
    "negating_context_terms",
    "uncertain_context_terms",
]


def load_label_terms(path: Path) -> dict[str, list[str]]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Label terms file must be a YAML mapping: {path}")

    terms: dict[str, list[str]] = {}
    for key, value in payload.items():
        if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
            raise ValueError(f"Label term group must be a list of strings: {key}")
        terms[key] = value

    missing = missing_required_term_groups(terms)
    if missing:
        raise ValueError(f"Missing required label term groups: {missing}")

    return terms


def missing_required_term_groups(terms: dict[str, list[str]]) -> list[str]:
    return [key for key in REQUIRED_TERM_GROUPS if not terms.get(key)]


def terms_to_regex(terms: list[str]) -> str:
    escaped = [re.escape(term.lower()).replace(r"\ ", r"\s+") for term in terms]
    escaped.sort(key=len, reverse=True)
    return r"(?:" + "|".join(escaped) + r")"
