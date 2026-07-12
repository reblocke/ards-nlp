-- RadGraph JSON flattening is done by scripts/flatten_radgraph.py before BigQuery load.
-- This file provides a lightweight staging sanity table for build-chain visibility.

CREATE OR REPLACE TABLE `{{ project_id }}.{{ bq_dataset }}.radgraph_staging_counts` AS
SELECT 'stg_radgraph_reports' AS table_name, COUNT(*) AS n
FROM `{{ project_id }}.{{ bq_dataset }}.stg_radgraph_reports`
UNION ALL
SELECT 'stg_radgraph_entities' AS table_name, COUNT(*) AS n
FROM `{{ project_id }}.{{ bq_dataset }}.stg_radgraph_entities`
UNION ALL
SELECT 'stg_radgraph_relations' AS table_name, COUNT(*) AS n
FROM `{{ project_id }}.{{ bq_dataset }}.stg_radgraph_relations`;
