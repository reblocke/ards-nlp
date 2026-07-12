CREATE SCHEMA IF NOT EXISTS `{{ project_id }}.{{ bq_dataset }}`
OPTIONS (
  location = '{{ location }}',
  description = 'MIMIC-CXR bilateral opacity report-NLP silver benchmark'
);
