CREATE OR REPLACE TABLE `{{ project_id }}.{{ bq_dataset }}.radgraph_report_features` AS
WITH rel AS (
  SELECT
    *,
    REGEXP_CONTAINS(
      source_tokens,
      r'(opacity|opacities|infiltrat|air ?space|consolidat|edema|oedema|hazy|haze|density|densities|disease|ground[- ]glass|interstitial)'
    ) AS is_opacity_observation,

    REGEXP_CONTAINS(source_tokens, r'(air ?space|consolidat|pneumonia|infiltrat)')
      AS is_consolidation_or_airspace,

    REGEXP_CONTAINS(source_tokens, r'(edema|oedema|vascular congestion|congestive|fluid overload|chf)')
      AS is_edema_observation,

    REGEXP_CONTAINS(source_tokens, r'(atelecta|volume loss|low volume)')
      AS is_atelectasis_observation,

    REGEXP_CONTAINS(
      target_tokens,
      r'(left lung|left base|left lung base|left lower|left upper|left mid|left hemithorax|left perihilar|left lower lobe|left upper lobe)'
    ) AS is_left_anatomy,

    REGEXP_CONTAINS(
      target_tokens,
      r'(right lung|right base|right lung base|right lower|right upper|right mid|right hemithorax|right perihilar|right lower lobe|right upper lobe)'
    ) AS is_right_anatomy,

    REGEXP_CONTAINS(
      target_tokens,
      r'(bilateral|bilaterally|both lungs|bibasilar|biapical|diffuse|diffusely|multifocal|widespread|bilateral bases|bilateral lower|bilateral perihilar)'
    ) AS is_bilateral_anatomy
  FROM `{{ project_id }}.{{ bq_dataset }}.radgraph_relation_expanded`
  WHERE relation_type = 'located_at'
)
SELECT
  subject_id,
  study_id,

  COUNT(*) AS n_located_at_relations,

  LOGICAL_OR(is_opacity_observation AND source_label = 'OBS-DP') AS any_opacity_present,
  LOGICAL_OR(is_opacity_observation AND source_label = 'OBS-U') AS any_opacity_uncertain,
  LOGICAL_OR(is_opacity_observation AND source_label = 'OBS-DA') AS any_opacity_absent,

  LOGICAL_OR(is_opacity_observation AND source_label = 'OBS-DP' AND is_left_anatomy)
    AS left_opacity_present,

  LOGICAL_OR(is_opacity_observation AND source_label = 'OBS-DP' AND is_right_anatomy)
    AS right_opacity_present,

  LOGICAL_OR(is_opacity_observation AND source_label = 'OBS-DP' AND is_bilateral_anatomy)
    AS bilateral_anatomy_opacity_present,

  LOGICAL_OR(is_opacity_observation AND source_label = 'OBS-U' AND is_bilateral_anatomy)
    AS bilateral_anatomy_opacity_uncertain,

  LOGICAL_OR(is_edema_observation AND source_label = 'OBS-DP' AND is_bilateral_anatomy)
    AS bilateral_edema_present,

  LOGICAL_OR(is_atelectasis_observation AND source_label = 'OBS-DP' AND is_bilateral_anatomy)
    AS bilateral_atelectasis_present,

  LOGICAL_OR(is_consolidation_or_airspace AND source_label = 'OBS-DP' AND is_bilateral_anatomy)
    AS bilateral_consolidation_or_airspace_present,

  ARRAY_AGG(
    DISTINCT IF(
      is_opacity_observation,
      CONCAT(source_label, ': ', source_tokens, ' -> ', target_tokens),
      NULL
    )
    IGNORE NULLS
    LIMIT 20
  ) AS radgraph_opacity_relation_examples

FROM rel
GROUP BY subject_id, study_id;
