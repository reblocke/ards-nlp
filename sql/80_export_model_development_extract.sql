CREATE OR REPLACE TABLE `{{ project_id }}.{{ bq_dataset }}.model_development_extract` AS
SELECT
  subject_id,
  study_id,

  report_text,
  target_text_full_report,
  target_text_impression_findings,
  target_text_impression_fallback,
  primary_target_text,

  strict_bilateral_opacity_label,
  sensitive_bilateral_opacity_label,
  silver_bilateral_opacity_score,
  silver_label_source,

  chexpert_lung_opacity,
  chexpert_edema,
  chexpert_consolidation,
  chexpert_atelectasis,
  has_mimic_cxr_jpg_labels,

  negbio_lung_opacity,
  negbio_edema,
  negbio_consolidation,
  negbio_atelectasis,

  radgraph_strict_bilateral_opacity_present,
  radgraph_sensitive_bilateral_opacity_present,
  regex_bilateral_opacity_present,
  regex_bilateral_opacity_uncertain,
  regex_bilateral_opacity_negated,

  bilateral_opacity_any,
  bilateral_opacity_non_atelectatic,
  bilateral_edema,
  bilateral_atelectasis,
  bilateral_consolidation_or_airspace,
  bilateral_ambiguous_or_uncertain,

  manual_review_priority,
  qa_flags,
  pipeline_version

FROM `{{ project_id }}.{{ bq_dataset }}.silver_reference_candidates`
WHERE report_text IS NOT NULL;
