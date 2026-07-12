CREATE TEMP FUNCTION normalize_report_text(text STRING)
RETURNS STRING
AS (
  TRIM(REGEXP_REPLACE(REGEXP_REPLACE(text, r'\r\n?', '\n'), r'\n{3,}', '\n\n'))
);

CREATE TEMP FUNCTION extract_section(text STRING, section_name STRING)
RETURNS STRING
AS ((
  SELECT NULLIF(
    TRIM(
      REGEXP_EXTRACT(
        text,
        CONCAT(
          r'(?is)(?:^|\n)\s*',
          section_name,
          r'\s*:?\s*(.*?)(?:\n\s*(?:FINAL REPORT|EXAMINATION|INDICATION|TECHNIQUE|COMPARISON|FINDINGS|IMPRESSION|ADDENDUM)\s*:?\s*|\z)'
        )
      )
    ),
    ''
  )
));

CREATE OR REPLACE TABLE `{{ project_id }}.{{ bq_dataset }}.mimic_cxr_report_sections` AS
WITH base AS (
  SELECT
    subject_id,
    study_id,
    report_path,
    normalize_report_text(report_text) AS report_text
  FROM `{{ project_id }}.{{ bq_dataset }}.mimic_cxr_report_base`
),
parsed AS (
  SELECT
    *,
    extract_section(report_text, 'FINDINGS') AS findings_text,
    extract_section(report_text, 'IMPRESSION') AS impression_text,
    extract_section(report_text, 'ADDENDUM') AS addendum_text,
    ARRAY_REVERSE(SPLIT(REGEXP_REPLACE(TRIM(report_text), r'\n\s*\n+', '\n\n'), '\n\n'))[
      SAFE_OFFSET(0)
    ] AS last_paragraph_text
  FROM base
)
SELECT
  subject_id,
  study_id,
  report_path,
  report_text,
  findings_text,
  impression_text,
  addendum_text,
  last_paragraph_text,
  report_text AS target_text_full_report,
  TRIM(CONCAT(COALESCE(impression_text, ''), '\n', COALESCE(findings_text, '')))
    AS target_text_impression_findings,
  COALESCE(impression_text, findings_text, last_paragraph_text, report_text)
    AS target_text_impression_fallback,
  report_text AS primary_target_text,
  findings_text IS NOT NULL AS has_findings,
  impression_text IS NOT NULL AS has_impression,
  addendum_text IS NOT NULL AS has_addendum
FROM parsed;
