CREATE OR REPLACE TABLE `{{ project_id }}.{{ bq_dataset }}.reviewed_subset_silver_comparison` AS
SELECT
  r.subject_id,
  r.study_id,
  r.human_report_probability_0_100,
  r.reviewer_id,
  r.reviewer_notes,
  s.strict_bilateral_opacity_label,
  s.sensitive_bilateral_opacity_label,
  s.silver_bilateral_opacity_score,
  s.silver_label_source,
  s.manual_review_priority,
  s.qa_flags,
  '{{ pipeline_version }}' AS pipeline_version
FROM `{{ project_id }}.{{ bq_dataset }}.manual_review_completed_labels` AS r
INNER JOIN `{{ project_id }}.{{ bq_dataset }}.silver_reference_candidates` AS s
  USING (subject_id, study_id);
