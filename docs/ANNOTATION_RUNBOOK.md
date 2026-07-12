# Annotation validation runbook

This repo currently exports manual-review samples only. Reviewed labels become gold only after a
human-completed review file exists and is validated.

## Input

Use a local completed copy of:

```text
artifacts/samples/manual_review_sample_for_adjudication.csv
```

Required columns:

- `subject_id`
- `study_id`
- `human_report_probability_0_100`
- `reviewer_id`
- `reviewer_notes`

## Validation

Validate and export text-stripped completed rows:

```bash
uv run python scripts/validate_annotation_csv.py \
  --input-csv artifacts/samples/manual_review_sample_for_adjudication.csv \
  --require-completed
```

Default outputs:

```text
data/derived/annotations/reviewed_labels.parquet
data/derived/annotations/reviewed_labels_validation.json
```

The reviewed-label export strips source report text and RadGraph example text, but it remains
reviewer-level derived data and is ignored by git.

## Future BigQuery Comparison

After a reviewed-label table has been loaded to BigQuery as `manual_review_completed_labels`, the
template `sql/95_compare_reviewed_subset_template.sql` can compare reviewed labels with silver
labels. Do not create `gold_*` outputs until completed human review exists.
