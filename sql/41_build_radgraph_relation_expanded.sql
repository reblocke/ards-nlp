CREATE OR REPLACE TABLE `{{ project_id }}.{{ bq_dataset }}.radgraph_relation_expanded` AS
SELECT
  r.report_uid,
  r.subject_id,
  r.study_id,

  src.entity_id AS source_entity_id,
  LOWER(src.tokens) AS source_tokens,
  src.label AS source_label,

  tgt.entity_id AS target_entity_id,
  LOWER(tgt.tokens) AS target_tokens,
  tgt.label AS target_label,

  r.relation_type
FROM `{{ project_id }}.{{ bq_dataset }}.stg_radgraph_relations` AS r
LEFT JOIN `{{ project_id }}.{{ bq_dataset }}.stg_radgraph_entities` AS src
  ON r.report_uid = src.report_uid
 AND r.source_entity_id = src.entity_id
LEFT JOIN `{{ project_id }}.{{ bq_dataset }}.stg_radgraph_entities` AS tgt
  ON r.report_uid = tgt.report_uid
 AND r.target_entity_id = tgt.entity_id;
