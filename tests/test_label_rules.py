from __future__ import annotations

from ards_cxr_benchmark.label_rules import construct_silver_labels, detect_regex_features


def test_detect_regex_features_for_explicit_bilateral_opacities() -> None:
    features = detect_regex_features("There are diffuse bilateral airspace opacities.")

    assert features.regex_bilateral_opacity_present is True
    assert features.regex_bilateral_consolidation_or_airspace is True
    assert features.regex_bilateral_opacity_negated is False


def test_construct_silver_labels_for_regex_positive() -> None:
    regex = detect_regex_features("Bilateral airspace opacities are present.")
    labels = construct_silver_labels(regex.as_dict())

    assert labels["strict_bilateral_opacity_label"] == 1
    assert labels["sensitive_bilateral_opacity_label"] == 1
    assert labels["bilateral_opacity_any"] is True
    assert labels["silver_label_source"] == "regex_strict"


def test_construct_silver_labels_keeps_negated_positive_as_review_conflict() -> None:
    regex = detect_regex_features("No bilateral airspace opacities are present.")
    labels = construct_silver_labels(regex.as_dict())

    assert labels["strict_bilateral_opacity_label"] is None
    assert labels["sensitive_bilateral_opacity_label"] is None
    assert "positive_and_negated_conflict" in labels["qa_flags"]
    assert labels["manual_review_priority"] == "high"


def test_chexpert_uncertain_does_not_make_sensitive_positive() -> None:
    labels = construct_silver_labels({"chexpert_lung_opacity": -1})

    assert labels["sensitive_bilateral_opacity_label"] is None
    assert labels["has_mimic_cxr_jpg_labels"] is True
    assert labels["silver_bilateral_opacity_score"] == 0.30
    assert labels["silver_label_source"] == "chexpert_lung_opacity_uncertain_only"


def test_missing_jpg_labels_without_bilateral_signal_stays_unknown() -> None:
    labels = construct_silver_labels({})

    assert labels["has_mimic_cxr_jpg_labels"] is False
    assert labels["strict_bilateral_opacity_label"] is None
    assert labels["sensitive_bilateral_opacity_label"] is None
    assert labels["silver_bilateral_opacity_score"] is None
    assert labels["silver_label_source"] == "unclassified"


def test_broad_negative_without_bilateral_signal_is_negative() -> None:
    labels = construct_silver_labels(
        {
            "has_mimic_cxr_jpg_labels": True,
            "chexpert_lung_opacity": 0,
            "chexpert_edema": 0,
            "chexpert_consolidation": 0,
            "chexpert_atelectasis": 0,
        }
    )

    assert labels["has_mimic_cxr_jpg_labels"] is True
    assert labels["strict_bilateral_opacity_label"] == 0
    assert labels["sensitive_bilateral_opacity_label"] == 0


def test_joined_jpg_row_with_blank_broad_labels_is_negative_score_source() -> None:
    labels = construct_silver_labels({"has_mimic_cxr_jpg_labels": True})

    assert labels["strict_bilateral_opacity_label"] == 0
    assert labels["sensitive_bilateral_opacity_label"] == 0
    assert labels["silver_bilateral_opacity_score"] == 0.05
    assert labels["silver_label_source"] == "chexpert_negative"
