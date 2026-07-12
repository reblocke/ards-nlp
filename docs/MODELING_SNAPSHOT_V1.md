# Modeling Snapshot V1

Aggregate, non-sensitive snapshot for the silver-label modeling baseline.

- Generated UTC: `2026-06-26T17:43:04.164035+00:00`
- Commit: `f5d1ae86`
- Tracked dirty at snapshot time: `False`
- BigQuery dataset: configured private working dataset (name omitted from the collaborator snapshot)
- Split salt: `mimic-bilat-opacity-v1-split`
- Modeling QA passed: `True`

## Row Counts

| table_name | n_rows |
| --- | --- |
| model_development_extract | 227835 |
| silver_reference_candidates | 227835 |
| model_development_splits | 227835 |
| manual_review_sample | 206 |

## Eligible Rows

| strict_eligible_rows | sensitive_eligible_rows |
| --- | --- |
| 134814 | 146122 |

## Split Counts

| split | n_rows | n_subjects |
| --- | --- | --- |
| test | 35151 | 9861 |
| train | 159236 | 45841 |
| validation | 33448 | 9677 |

## Test Metrics

| task | model | n | prevalence | roc_auc | average_precision | f1 | recall | specificity |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sensitive | silver_score_rule | 22375 | 0.250771 | 0.86874 | 0.803312 | 0.819979 | 0.694885 | 1 |
| sensitive | tfidf_logreg | 22375 | 0.250771 | 0.994459 | 0.983652 | 0.938124 | 0.963286 | 0.969757 |
| sensitive | structured_logreg | 22375 | 0.250771 | 1 | 1 | 1 | 1 | 1 |
| strict | silver_score_rule | 20702 | 0.188243 | 1 | 1 | 1 | 1 | 1 |
| strict | tfidf_logreg | 20702 | 0.188243 | 0.993994 | 0.97562 | 0.914363 | 0.958943 | 0.967867 |
| strict | structured_logreg | 20702 | 0.188243 | 1 | 1 | 1 | 1 | 1 |

## Artifact Policy

Generated model caches, fitted models, predictions, and JSON snapshots remain ignored.
This tracked snapshot intentionally contains only aggregate metadata and metrics.
