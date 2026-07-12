CREATE OR REPLACE TABLE `{{ project_id }}.{{ bq_dataset }}.radgraph_bilateral_features` AS
SELECT
  subject_id,
  study_id,

  left_opacity_present,
  right_opacity_present,
  bilateral_anatomy_opacity_present,
  bilateral_anatomy_opacity_uncertain,

  bilateral_edema_present,
  bilateral_atelectasis_present,
  bilateral_consolidation_or_airspace_present,

  (
    COALESCE(bilateral_anatomy_opacity_present, FALSE)
    OR (COALESCE(left_opacity_present, FALSE) AND COALESCE(right_opacity_present, FALSE))
  ) AS radgraph_strict_bilateral_opacity_present,

  (
    COALESCE(bilateral_anatomy_opacity_present, FALSE)
    OR COALESCE(bilateral_anatomy_opacity_uncertain, FALSE)
    OR (COALESCE(left_opacity_present, FALSE) AND COALESCE(right_opacity_present, FALSE))
  ) AS radgraph_sensitive_bilateral_opacity_present,

  radgraph_opacity_relation_examples

FROM `{{ project_id }}.{{ bq_dataset }}.radgraph_report_features`;
