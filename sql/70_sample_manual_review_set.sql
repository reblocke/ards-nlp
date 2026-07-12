CREATE OR REPLACE TABLE `{{ project_id }}.{{ bq_dataset }}.manual_review_sample` AS
WITH candidate AS (
  SELECT
    *,
    CASE
      WHEN silver_bilateral_opacity_score < 0.10 THEN 'p00_10'
      WHEN silver_bilateral_opacity_score < 0.30 THEN 'p10_30'
      WHEN silver_bilateral_opacity_score < 0.50 THEN 'p30_50'
      WHEN silver_bilateral_opacity_score < 0.70 THEN 'p50_70'
      WHEN silver_bilateral_opacity_score < 0.90 THEN 'p70_90'
      ELSE 'p90_100'
    END AS probability_bin
  FROM `{{ project_id }}.{{ bq_dataset }}.silver_reference_candidates`
  WHERE silver_bilateral_opacity_score IS NOT NULL
),
ranked AS (
  SELECT
    *,
    ROW_NUMBER() OVER (
      PARTITION BY probability_bin
      ORDER BY FARM_FINGERPRINT(CONCAT(CAST(study_id AS STRING), '-mimic-bilat-opacity-v1'))
    ) AS random_rank
  FROM candidate
)
SELECT
  subject_id,
  study_id,
  probability_bin,
  silver_bilateral_opacity_score,
  strict_bilateral_opacity_label,
  sensitive_bilateral_opacity_label,
  silver_label_source,
  manual_review_priority,
  has_mimic_cxr_jpg_labels,
  report_text,
  findings_text,
  impression_text,
  radgraph_opacity_relation_examples,
  qa_flags,
  CAST(NULL AS FLOAT64) AS human_report_probability_0_100,
  CAST(NULL AS STRING) AS reviewer_id,
  CAST(NULL AS STRING) AS reviewer_notes
FROM ranked
WHERE random_rank <= 50;
