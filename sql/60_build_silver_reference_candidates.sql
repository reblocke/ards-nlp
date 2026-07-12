CREATE OR REPLACE TABLE `{{ project_id }}.{{ bq_dataset }}.silver_reference_candidates` AS
WITH reports AS (
  SELECT *
  FROM `{{ project_id }}.{{ bq_dataset }}.mimic_cxr_report_sections`
),
labels AS (
  SELECT *
  FROM `{{ project_id }}.{{ bq_dataset }}.mimic_cxr_jpg_labels_normalized`
),
rg AS (
  SELECT *
  FROM `{{ project_id }}.{{ bq_dataset }}.radgraph_bilateral_features`
),
rx AS (
  SELECT *
  FROM `{{ project_id }}.{{ bq_dataset }}.regex_report_features`
),
joined AS (
  SELECT
    r.subject_id,
    r.study_id,
    r.report_text,
    r.findings_text,
    r.impression_text,
    r.target_text_full_report,
    r.target_text_impression_findings,
    r.target_text_impression_fallback,
    r.primary_target_text,

    l.chexpert_lung_opacity,
    l.chexpert_edema,
    l.chexpert_consolidation,
    l.chexpert_atelectasis,
    l.chexpert_pleural_effusion,
    l.chexpert_pneumonia,
    l.negbio_lung_opacity,
    l.negbio_edema,
    l.negbio_consolidation,
    l.negbio_atelectasis,
    l.negbio_pleural_effusion,
    l.negbio_pneumonia,
    l.subject_id IS NOT NULL AND l.study_id IS NOT NULL AS has_mimic_cxr_jpg_labels,

    rg.left_opacity_present AS radgraph_left_opacity_present,
    rg.right_opacity_present AS radgraph_right_opacity_present,
    rg.bilateral_anatomy_opacity_present AS radgraph_bilateral_opacity_present,
    rg.bilateral_anatomy_opacity_uncertain AS radgraph_bilateral_opacity_uncertain,
    rg.radgraph_strict_bilateral_opacity_present,
    rg.radgraph_sensitive_bilateral_opacity_present,
    rg.bilateral_edema_present AS radgraph_bilateral_edema,
    rg.bilateral_atelectasis_present AS radgraph_bilateral_atelectasis,
    rg.bilateral_consolidation_or_airspace_present
      AS radgraph_bilateral_consolidation_or_airspace,
    rg.radgraph_opacity_relation_examples,

    rx.regex_bilateral_opacity_present,
    rx.regex_bilateral_opacity_uncertain,
    rx.regex_bilateral_opacity_negated,
    rx.regex_right_opacity,
    rx.regex_left_opacity,
    rx.regex_bilateral_edema,
    rx.regex_bilateral_atelectasis,
    rx.regex_bilateral_consolidation_or_airspace,
    rx.regex_bilateral_opacity_present_impression_findings,
    rx.regex_bilateral_opacity_present_impression_fallback

  FROM reports AS r
  LEFT JOIN labels AS l USING (subject_id, study_id)
  LEFT JOIN rg USING (subject_id, study_id)
  LEFT JOIN rx USING (subject_id, study_id)
),
enriched AS (
  SELECT
    *,

    COALESCE(radgraph_bilateral_edema, FALSE)
      OR COALESCE(regex_bilateral_edema, FALSE)
      AS bilateral_edema,

    COALESCE(radgraph_bilateral_atelectasis, FALSE)
      OR COALESCE(regex_bilateral_atelectasis, FALSE)
      AS bilateral_atelectasis,

    COALESCE(radgraph_bilateral_consolidation_or_airspace, FALSE)
      OR COALESCE(regex_bilateral_consolidation_or_airspace, FALSE)
      AS bilateral_consolidation_or_airspace,

    COALESCE(radgraph_bilateral_opacity_uncertain, FALSE)
      OR COALESCE(regex_bilateral_opacity_uncertain, FALSE)
      AS bilateral_ambiguous_or_uncertain,

    (
      COALESCE(radgraph_strict_bilateral_opacity_present, FALSE)
      OR COALESCE(regex_bilateral_opacity_present, FALSE)
    )
    AND NOT COALESCE(regex_bilateral_opacity_negated, FALSE)
      AS strict_positive_signal,

    (
      COALESCE(radgraph_sensitive_bilateral_opacity_present, FALSE)
      OR COALESCE(regex_bilateral_opacity_present, FALSE)
      OR COALESCE(regex_bilateral_opacity_uncertain, FALSE)
      OR COALESCE(radgraph_bilateral_edema, FALSE)
      OR COALESCE(regex_bilateral_edema, FALSE)
      OR COALESCE(radgraph_bilateral_atelectasis, FALSE)
      OR COALESCE(regex_bilateral_atelectasis, FALSE)
      OR COALESCE(radgraph_bilateral_consolidation_or_airspace, FALSE)
      OR COALESCE(regex_bilateral_consolidation_or_airspace, FALSE)
    )
    AND NOT COALESCE(regex_bilateral_opacity_negated, FALSE)
      AS sensitive_positive_signal,

    has_mimic_cxr_jpg_labels
      AND COALESCE(chexpert_lung_opacity, 0) = 0
      AND COALESCE(chexpert_edema, 0) = 0
      AND COALESCE(chexpert_consolidation, 0) = 0
      AND COALESCE(chexpert_atelectasis, 0) = 0
      AND NOT COALESCE(radgraph_sensitive_bilateral_opacity_present, FALSE)
      AND NOT COALESCE(regex_bilateral_opacity_present, FALSE)
      AND NOT COALESCE(regex_bilateral_opacity_uncertain, FALSE)
      AND NOT COALESCE(regex_bilateral_edema, FALSE)
      AND NOT COALESCE(regex_bilateral_atelectasis, FALSE)
      AND NOT COALESCE(regex_bilateral_consolidation_or_airspace, FALSE)
      AS negative_signal

  FROM joined
),
labeled AS (
  SELECT
    *,

    strict_positive_signal
      OR sensitive_positive_signal
      OR bilateral_ambiguous_or_uncertain
      AS bilateral_opacity_any,

    (
      COALESCE(radgraph_strict_bilateral_opacity_present, FALSE)
      OR COALESCE(regex_bilateral_opacity_present, FALSE)
      OR COALESCE(regex_bilateral_opacity_uncertain, FALSE)
      OR bilateral_edema
      OR bilateral_consolidation_or_airspace
    )
    AND NOT COALESCE(regex_bilateral_opacity_negated, FALSE)
      AS bilateral_opacity_non_atelectatic,

    CASE
      WHEN strict_positive_signal THEN 1
      WHEN negative_signal THEN 0
      ELSE NULL
    END AS strict_bilateral_opacity_label,

    CASE
      WHEN sensitive_positive_signal THEN 1
      WHEN negative_signal THEN 0
      ELSE NULL
    END AS sensitive_bilateral_opacity_label

  FROM enriched
)
SELECT
  * EXCEPT(strict_positive_signal, sensitive_positive_signal, negative_signal),

  CASE
    WHEN COALESCE(radgraph_strict_bilateral_opacity_present, FALSE)
      AND COALESCE(regex_bilateral_opacity_present, FALSE)
      THEN 0.95

    WHEN COALESCE(radgraph_strict_bilateral_opacity_present, FALSE)
      THEN 0.90

    WHEN COALESCE(regex_bilateral_opacity_present, FALSE)
      AND NOT COALESCE(regex_bilateral_opacity_negated, FALSE)
      THEN 0.85

    WHEN COALESCE(radgraph_sensitive_bilateral_opacity_present, FALSE)
      THEN 0.75

    WHEN COALESCE(regex_bilateral_opacity_uncertain, FALSE)
      OR COALESCE(radgraph_bilateral_opacity_uncertain, FALSE)
      THEN 0.55

    WHEN COALESCE(chexpert_lung_opacity, 0) = 1
      THEN 0.35

    WHEN COALESCE(chexpert_lung_opacity, 0) = -1
      THEN 0.30

    WHEN has_mimic_cxr_jpg_labels
      AND COALESCE(chexpert_lung_opacity, 0) = 0
      THEN 0.05

    ELSE NULL
  END AS silver_bilateral_opacity_score,

  CASE
    WHEN COALESCE(radgraph_strict_bilateral_opacity_present, FALSE)
      AND COALESCE(regex_bilateral_opacity_present, FALSE)
      THEN 'radgraph_strict_plus_regex'

    WHEN COALESCE(radgraph_strict_bilateral_opacity_present, FALSE)
      THEN 'radgraph_strict'

    WHEN COALESCE(regex_bilateral_opacity_present, FALSE)
      AND NOT COALESCE(regex_bilateral_opacity_negated, FALSE)
      THEN 'regex_strict'

    WHEN COALESCE(radgraph_sensitive_bilateral_opacity_present, FALSE)
      THEN 'radgraph_sensitive'

    WHEN COALESCE(regex_bilateral_opacity_uncertain, FALSE)
      OR COALESCE(radgraph_bilateral_opacity_uncertain, FALSE)
      THEN 'regex_or_radgraph_uncertain'

    WHEN COALESCE(chexpert_lung_opacity, 0) = 1
      THEN 'chexpert_lung_opacity_only'

    WHEN COALESCE(chexpert_lung_opacity, 0) = -1
      THEN 'chexpert_lung_opacity_uncertain_only'

    WHEN has_mimic_cxr_jpg_labels
      AND COALESCE(chexpert_lung_opacity, 0) = 0
      THEN 'chexpert_negative'

    ELSE 'unclassified'
  END AS silver_label_source,

  ARRAY_CONCAT(
    IF(
      COALESCE(radgraph_strict_bilateral_opacity_present, FALSE)
      != COALESCE(regex_bilateral_opacity_present, FALSE),
      ['radgraph_regex_disagreement'],
      []
    ),
    IF(
      COALESCE(regex_bilateral_opacity_negated, FALSE)
      AND (
        COALESCE(radgraph_strict_bilateral_opacity_present, FALSE)
        OR COALESCE(regex_bilateral_opacity_present, FALSE)
      ),
      ['positive_and_negated_conflict'],
      []
    ),
    IF(
      COALESCE(chexpert_lung_opacity, 0) = 1
      AND NOT COALESCE(radgraph_sensitive_bilateral_opacity_present, FALSE)
      AND NOT COALESCE(regex_bilateral_opacity_present, FALSE),
      ['broad_opacity_without_bilateral_signal'],
      []
    )
  ) AS qa_flags,

  CASE
    WHEN COALESCE(radgraph_strict_bilateral_opacity_present, FALSE)
      != COALESCE(regex_bilateral_opacity_present, FALSE)
      THEN 'high'

    WHEN COALESCE(regex_bilateral_opacity_uncertain, FALSE)
      OR COALESCE(radgraph_bilateral_opacity_uncertain, FALSE)
      THEN 'high'

    WHEN COALESCE(chexpert_lung_opacity, 0) = 1
      AND NOT COALESCE(radgraph_sensitive_bilateral_opacity_present, FALSE)
      AND NOT COALESCE(regex_bilateral_opacity_present, FALSE)
      THEN 'medium'

    ELSE 'low'
  END AS manual_review_priority,

  CURRENT_TIMESTAMP() AS created_at,
  '{{ pipeline_version }}' AS pipeline_version

FROM labeled;
