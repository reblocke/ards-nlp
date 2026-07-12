SELECT
  schema_name
FROM `{{ physionet_project }}.region-us`.INFORMATION_SCHEMA.SCHEMATA
ORDER BY schema_name;
