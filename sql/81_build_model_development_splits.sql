CREATE OR REPLACE TABLE `{{ project_id }}.{{ bq_dataset }}.model_development_splits` AS
WITH subject_splits AS (
  SELECT
    subject_id,
    MOD(
      CAST(
        CONCAT(
          '0x',
          SUBSTR(
            TO_HEX(SHA256(CONCAT(CAST(subject_id AS STRING), '-mimic-bilat-opacity-v1-split'))),
            1,
            15
          )
        ) AS INT64
      ),
      10000
    ) AS split_bucket
  FROM (
    SELECT DISTINCT subject_id
    FROM `{{ project_id }}.{{ bq_dataset }}.model_development_extract`
  )
),
split_labels AS (
  SELECT
    subject_id,
    split_bucket,
    CASE
      WHEN split_bucket < 7000 THEN 'train'
      WHEN split_bucket < 8500 THEN 'validation'
      ELSE 'test'
    END AS split
  FROM subject_splits
)
SELECT
  e.subject_id,
  e.study_id,
  s.split,
  s.split_bucket,
  'mimic-bilat-opacity-v1-split' AS split_salt,
  '{{ pipeline_version }}' AS pipeline_version
FROM `{{ project_id }}.{{ bq_dataset }}.model_development_extract` AS e
INNER JOIN split_labels AS s
  USING (subject_id);
