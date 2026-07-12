from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

OPACITY_PATTERN = (
    r"(opacit|infiltrat|air ?space|consolidat|edema|oedema|hazy|haze|density|"
    r"densities|disease|ground[- ]glass|interstitial)"
)
BILATERAL_PATTERN = (
    r"(bilateral|bilaterally|diffuse|diffusely|multifocal|both lungs|bibasilar|biapical)"
)
UNCERTAIN_PATTERN = (
    r"(possible|possibly|may represent|could represent|cannot exclude|difficult to exclude|"
    r"questionable|suspected|likely|probably|favor)"
)
NEGATING_PATTERN = r"(no|without|absent|resolved|cleared|clear of|negative for|free of)"


@dataclass(frozen=True)
class RegexFeatures:
    regex_bilateral_opacity_present: bool
    regex_bilateral_opacity_uncertain: bool
    regex_bilateral_opacity_negated: bool
    regex_right_opacity: bool
    regex_left_opacity: bool
    regex_bilateral_edema: bool
    regex_bilateral_atelectasis: bool
    regex_bilateral_consolidation_or_airspace: bool

    def as_dict(self) -> dict[str, bool]:
        return {
            "regex_bilateral_opacity_present": self.regex_bilateral_opacity_present,
            "regex_bilateral_opacity_uncertain": self.regex_bilateral_opacity_uncertain,
            "regex_bilateral_opacity_negated": self.regex_bilateral_opacity_negated,
            "regex_right_opacity": self.regex_right_opacity,
            "regex_left_opacity": self.regex_left_opacity,
            "regex_bilateral_edema": self.regex_bilateral_edema,
            "regex_bilateral_atelectasis": self.regex_bilateral_atelectasis,
            "regex_bilateral_consolidation_or_airspace": (
                self.regex_bilateral_consolidation_or_airspace
            ),
        }


def _contains(text: str, pattern: str) -> bool:
    return re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL) is not None


def detect_regex_features(text: str) -> RegexFeatures:
    text = text.lower()
    bilateral_before_opacity = rf"{BILATERAL_PATTERN}.{{0,120}}{OPACITY_PATTERN}"
    opacity_before_bilateral = rf"{OPACITY_PATTERN}.{{0,120}}{BILATERAL_PATTERN}"
    uncertain_bilateral_opacity = (
        rf"{UNCERTAIN_PATTERN}.{{0,120}}{BILATERAL_PATTERN}.{{0,120}}{OPACITY_PATTERN}"
    )
    bilateral_uncertain_opacity = (
        rf"{BILATERAL_PATTERN}.{{0,120}}{UNCERTAIN_PATTERN}.{{0,120}}{OPACITY_PATTERN}"
    )
    negated_bilateral_opacity = (
        rf"{NEGATING_PATTERN}.{{0,120}}{BILATERAL_PATTERN}.{{0,120}}{OPACITY_PATTERN}"
    )
    bilateral_opacity_resolved = (
        rf"{BILATERAL_PATTERN}.{{0,120}}{OPACITY_PATTERN}.{{0,120}}"
        r"(resolved|cleared|improved to resolution)"
    )

    return RegexFeatures(
        regex_bilateral_opacity_present=_contains(text, bilateral_before_opacity)
        or _contains(text, opacity_before_bilateral),
        regex_bilateral_opacity_uncertain=_contains(text, uncertain_bilateral_opacity)
        or _contains(text, bilateral_uncertain_opacity),
        regex_bilateral_opacity_negated=_contains(text, negated_bilateral_opacity)
        or _contains(text, bilateral_opacity_resolved),
        regex_right_opacity=_contains(text, rf"right.{{0,80}}{OPACITY_PATTERN}"),
        regex_left_opacity=_contains(text, rf"left.{{0,80}}{OPACITY_PATTERN}"),
        regex_bilateral_edema=_contains(
            text,
            r"(bilateral|bibasilar|both lungs|diffuse).{0,120}"
            r"(edema|oedema|vascular congestion|congestive|chf|fluid overload)",
        ),
        regex_bilateral_atelectasis=_contains(
            text, r"(bilateral|bibasilar|both lungs).{0,120}(atelecta|volume loss|low volume)"
        ),
        regex_bilateral_consolidation_or_airspace=_contains(
            text,
            r"(bilateral|bibasilar|both lungs|diffuse|multifocal).{0,120}"
            r"(air ?space|consolidat|pneumonia|infiltrat)",
        ),
    )


def _bool(value: Any) -> bool:
    return bool(value) if value is not None else False


def _label_value(features: dict[str, Any], key: str) -> int | None:
    value = features.get(key)
    if value is None:
        return None
    return int(value)


def construct_silver_labels(features: dict[str, Any]) -> dict[str, Any]:
    broad_keys = [
        "chexpert_lung_opacity",
        "chexpert_edema",
        "chexpert_consolidation",
        "chexpert_atelectasis",
    ]
    broad_label_values = {key: _label_value(features, key) for key in broad_keys}
    has_mimic_cxr_jpg_labels = _bool(features.get("has_mimic_cxr_jpg_labels")) or any(
        key in features for key in broad_keys
    )
    broad_negative = has_mimic_cxr_jpg_labels and all(
        (broad_label_values[key] or 0) == 0 for key in broad_keys
    )

    bilateral_edema = _bool(features.get("radgraph_bilateral_edema")) or _bool(
        features.get("regex_bilateral_edema")
    )
    bilateral_atelectasis = _bool(features.get("radgraph_bilateral_atelectasis")) or _bool(
        features.get("regex_bilateral_atelectasis")
    )
    bilateral_consolidation_or_airspace = _bool(
        features.get("radgraph_bilateral_consolidation_or_airspace")
    ) or _bool(features.get("regex_bilateral_consolidation_or_airspace"))
    bilateral_ambiguous_or_uncertain = _bool(
        features.get("radgraph_bilateral_opacity_uncertain")
    ) or _bool(features.get("regex_bilateral_opacity_uncertain"))

    radgraph_strict = _bool(features.get("radgraph_strict_bilateral_opacity_present"))
    radgraph_sensitive = _bool(features.get("radgraph_sensitive_bilateral_opacity_present"))
    regex_present = _bool(features.get("regex_bilateral_opacity_present"))
    regex_uncertain = _bool(features.get("regex_bilateral_opacity_uncertain"))
    regex_negated = _bool(features.get("regex_bilateral_opacity_negated"))

    strict_positive = (radgraph_strict or regex_present) and not regex_negated
    sensitive_positive = (
        radgraph_sensitive
        or regex_present
        or regex_uncertain
        or bilateral_edema
        or bilateral_atelectasis
        or bilateral_consolidation_or_airspace
    ) and not regex_negated

    no_bilateral_signal = (
        not radgraph_sensitive
        and not regex_present
        and not regex_uncertain
        and not bilateral_edema
        and not bilateral_atelectasis
        and not bilateral_consolidation_or_airspace
    )

    strict_label = 1 if strict_positive else 0 if broad_negative and no_bilateral_signal else None
    sensitive_label = (
        1 if sensitive_positive else 0 if broad_negative and no_bilateral_signal else None
    )

    bilateral_opacity_any = (
        strict_positive or sensitive_positive or bilateral_ambiguous_or_uncertain
    )
    bilateral_opacity_non_atelectatic = (
        radgraph_strict
        or regex_present
        or regex_uncertain
        or bilateral_edema
        or bilateral_consolidation_or_airspace
    ) and not regex_negated

    chexpert_lung_opacity = broad_label_values["chexpert_lung_opacity"]
    if radgraph_strict and regex_present:
        score = 0.95
        source = "radgraph_strict_plus_regex"
    elif radgraph_strict:
        score = 0.90
        source = "radgraph_strict"
    elif regex_present and not regex_negated:
        score = 0.85
        source = "regex_strict"
    elif radgraph_sensitive:
        score = 0.75
        source = "radgraph_sensitive"
    elif regex_uncertain or _bool(features.get("radgraph_bilateral_opacity_uncertain")):
        score = 0.55
        source = "regex_or_radgraph_uncertain"
    elif chexpert_lung_opacity == 1:
        score = 0.35
        source = "chexpert_lung_opacity_only"
    elif chexpert_lung_opacity == -1:
        score = 0.30
        source = "chexpert_lung_opacity_uncertain_only"
    elif has_mimic_cxr_jpg_labels and (chexpert_lung_opacity or 0) == 0:
        score = 0.05
        source = "chexpert_negative"
    else:
        score = None
        source = "unclassified"

    qa_flags: list[str] = []
    if radgraph_strict != regex_present:
        qa_flags.append("radgraph_regex_disagreement")
    if regex_negated and (radgraph_strict or regex_present):
        qa_flags.append("positive_and_negated_conflict")
    if chexpert_lung_opacity == 1 and not radgraph_sensitive and not regex_present:
        qa_flags.append("broad_opacity_without_bilateral_signal")

    if "radgraph_regex_disagreement" in qa_flags or bilateral_ambiguous_or_uncertain:
        manual_review_priority = "high"
    elif "broad_opacity_without_bilateral_signal" in qa_flags:
        manual_review_priority = "medium"
    else:
        manual_review_priority = "low"

    return {
        "bilateral_opacity_any": bilateral_opacity_any,
        "bilateral_opacity_non_atelectatic": bilateral_opacity_non_atelectatic,
        "has_mimic_cxr_jpg_labels": has_mimic_cxr_jpg_labels,
        "bilateral_edema": bilateral_edema,
        "bilateral_atelectasis": bilateral_atelectasis,
        "bilateral_consolidation_or_airspace": bilateral_consolidation_or_airspace,
        "bilateral_ambiguous_or_uncertain": bilateral_ambiguous_or_uncertain,
        "strict_bilateral_opacity_label": strict_label,
        "sensitive_bilateral_opacity_label": sensitive_label,
        "silver_bilateral_opacity_score": score,
        "silver_label_source": source,
        "qa_flags": qa_flags,
        "manual_review_priority": manual_review_priority,
    }
