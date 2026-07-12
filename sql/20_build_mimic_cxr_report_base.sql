CREATE OR REPLACE TABLE `{{ project_id }}.{{ bq_dataset }}.mimic_cxr_report_base` AS
SELECT
  SAFE_CAST(subject_id AS INT64) AS subject_id,
  SAFE_CAST(study_id AS INT64) AS study_id,
  CAST(report_text AS STRING) AS report_text,
  CAST(report_path AS STRING) AS report_path
FROM `{{ project_id }}.{{ bq_dataset }}.stg_mimic_cxr_reports`
WHERE report_text IS NOT NULL;
