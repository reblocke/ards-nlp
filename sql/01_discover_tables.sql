SELECT
  table_catalog,
  table_schema,
  table_name,
  table_type
FROM `{{ physionet_project }}.{{ source_dataset }}.INFORMATION_SCHEMA.TABLES`
ORDER BY table_name;
