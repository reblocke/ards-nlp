CREATE OR REPLACE TABLE `{{ project_id }}.{{ bq_dataset }}.regex_report_features` AS
WITH target AS (
  SELECT
    subject_id,
    study_id,
    LOWER(COALESCE(report_text, '')) AS target_text,
    LOWER(COALESCE(target_text_impression_findings, '')) AS impression_findings_text,
    LOWER(COALESCE(target_text_impression_fallback, '')) AS impression_fallback_text
  FROM `{{ project_id }}.{{ bq_dataset }}.mimic_cxr_report_sections`
)
SELECT
  subject_id,
  study_id,

  REGEXP_CONTAINS(
    target_text,
    r'(bilateral|bilaterally|diffuse|diffusely|multifocal|both lungs|bibasilar|biapical).{0,120}(opacit|infiltrat|air ?space|consolidat|edema|oedema|hazy|haze|density|densities|disease|ground[- ]glass|interstitial)'
  )
  OR REGEXP_CONTAINS(
    target_text,
    r'(opacit|infiltrat|air ?space|consolidat|edema|oedema|hazy|haze|density|densities|disease|ground[- ]glass|interstitial).{0,120}(bilateral|bilaterally|diffuse|diffusely|multifocal|both lungs|bibasilar|biapical)'
  ) AS regex_bilateral_opacity_present,

  REGEXP_CONTAINS(
    target_text,
    r'(possible|possibly|may represent|could represent|cannot exclude|difficult to exclude|questionable|suspected|likely|probably|favor).{0,120}(bilateral|diffuse|multifocal|bibasilar).{0,120}(opacit|infiltrat|air ?space|consolidat|edema|hazy|interstitial)'
  )
  OR REGEXP_CONTAINS(
    target_text,
    r'(bilateral|diffuse|multifocal|bibasilar).{0,120}(possible|possibly|may represent|could represent|cannot exclude|questionable|suspected).{0,120}(opacit|infiltrat|air ?space|consolidat|edema|hazy|interstitial)'
  ) AS regex_bilateral_opacity_uncertain,

  REGEXP_CONTAINS(
    target_text,
    r'(no|without|absent|resolved|cleared|clear of|negative for|free of).{0,120}(bilateral|diffuse|multifocal|bibasilar).{0,120}(opacit|infiltrat|air ?space|consolidat|edema|hazy|interstitial)'
  )
  OR REGEXP_CONTAINS(
    target_text,
    r'(bilateral|diffuse|multifocal|bibasilar).{0,120}(opacit|infiltrat|air ?space|consolidat|edema|hazy|interstitial).{0,120}(resolved|cleared|improved to resolution)'
  ) AS regex_bilateral_opacity_negated,

  REGEXP_CONTAINS(
    target_text,
    r'(right).{0,80}(opacit|infiltrat|air ?space|consolidat|edema|hazy|interstitial)'
  ) AS regex_right_opacity,

  REGEXP_CONTAINS(
    target_text,
    r'(left).{0,80}(opacit|infiltrat|air ?space|consolidat|edema|hazy|interstitial)'
  ) AS regex_left_opacity,

  REGEXP_CONTAINS(
    target_text,
    r'(bilateral|bibasilar|both lungs|diffuse).{0,120}(edema|oedema|vascular congestion|congestive|chf|fluid overload)'
  ) AS regex_bilateral_edema,

  REGEXP_CONTAINS(
    target_text,
    r'(bilateral|bibasilar|both lungs).{0,120}(atelecta|volume loss|low volume)'
  ) AS regex_bilateral_atelectasis,

  REGEXP_CONTAINS(
    target_text,
    r'(bilateral|bibasilar|both lungs|diffuse|multifocal).{0,120}(air ?space|consolidat|pneumonia|infiltrat)'
  ) AS regex_bilateral_consolidation_or_airspace,

  -- Stored for later sensitivity analysis; v1 labels use full report text above.
  REGEXP_CONTAINS(
    impression_findings_text,
    r'(bilateral|bilaterally|diffuse|diffusely|multifocal|both lungs|bibasilar|biapical).{0,120}(opacit|infiltrat|air ?space|consolidat|edema|hazy|interstitial)'
  ) AS regex_bilateral_opacity_present_impression_findings,

  REGEXP_CONTAINS(
    impression_fallback_text,
    r'(bilateral|bilaterally|diffuse|diffusely|multifocal|both lungs|bibasilar|biapical).{0,120}(opacit|infiltrat|air ?space|consolidat|edema|hazy|interstitial)'
  ) AS regex_bilateral_opacity_present_impression_fallback

FROM target;
