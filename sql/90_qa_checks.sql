CREATE OR REPLACE TABLE `{{ project_id }}.{{ bq_dataset }}.qa_row_counts` AS
SELECT 'mimic_cxr_report_base' AS table_name, COUNT(*) AS n
FROM `{{ project_id }}.{{ bq_dataset }}.mimic_cxr_report_base`
UNION ALL
SELECT 'mimic_cxr_jpg_labels_normalized', COUNT(*) AS n
FROM `{{ project_id }}.{{ bq_dataset }}.mimic_cxr_jpg_labels_normalized`
UNION ALL
SELECT 'stg_radgraph_reports', COUNT(*) AS n
FROM `{{ project_id }}.{{ bq_dataset }}.stg_radgraph_reports`
UNION ALL
SELECT 'silver_reference_candidates', COUNT(*) AS n
FROM `{{ project_id }}.{{ bq_dataset }}.silver_reference_candidates`;

CREATE OR REPLACE TABLE `{{ project_id }}.{{ bq_dataset }}.qa_join_completeness` AS
SELECT
  COUNT(*) AS n_reports,
  COUNTIF(has_mimic_cxr_jpg_labels) AS n_with_mimic_cxr_jpg_labels,
  COUNTIF(chexpert_lung_opacity IS NOT NULL) AS n_with_chexpert_lung_opacity,
  COUNTIF(radgraph_strict_bilateral_opacity_present IS NOT NULL) AS n_with_radgraph,
  COUNTIF(silver_bilateral_opacity_score IS NOT NULL) AS n_with_silver_score
FROM `{{ project_id }}.{{ bq_dataset }}.silver_reference_candidates`;

CREATE OR REPLACE TABLE `{{ project_id }}.{{ bq_dataset }}.qa_label_prevalence` AS
SELECT
  strict_bilateral_opacity_label,
  sensitive_bilateral_opacity_label,
  silver_label_source,
  COUNT(*) AS n,
  AVG(silver_bilateral_opacity_score) AS mean_silver_score
FROM `{{ project_id }}.{{ bq_dataset }}.silver_reference_candidates`
GROUP BY
  strict_bilateral_opacity_label,
  sensitive_bilateral_opacity_label,
  silver_label_source
ORDER BY n DESC;

CREATE OR REPLACE TABLE `{{ project_id }}.{{ bq_dataset }}.qa_conflict_cases` AS
SELECT
  study_id,
  silver_bilateral_opacity_score,
  strict_bilateral_opacity_label,
  sensitive_bilateral_opacity_label,
  silver_label_source,
  qa_flags,
  impression_text,
  findings_text,
  radgraph_opacity_relation_examples
FROM `{{ project_id }}.{{ bq_dataset }}.silver_reference_candidates`
WHERE ARRAY_LENGTH(qa_flags) > 0
ORDER BY FARM_FINGERPRINT(CAST(study_id AS STRING))
LIMIT 200;

CREATE OR REPLACE TABLE `{{ project_id }}.{{ bq_dataset }}.qa_high_priority_review_queue` AS
SELECT
  study_id,
  silver_bilateral_opacity_score,
  strict_bilateral_opacity_label,
  sensitive_bilateral_opacity_label,
  silver_label_source,
  manual_review_priority,
  impression_text,
  findings_text,
  radgraph_opacity_relation_examples,
  qa_flags
FROM `{{ project_id }}.{{ bq_dataset }}.silver_reference_candidates`
WHERE manual_review_priority = 'high'
ORDER BY FARM_FINGERPRINT(CAST(study_id AS STRING))
LIMIT 200;
