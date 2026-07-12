# Probabilistic Evaluation Runbook

This runbook covers completed physician ratings and model evaluation against continuous reference
probabilities. It is additive to the silver-label modeling workflow.

The current Intermountain pilot produces equivalent report/image concepts with a pilot-specific
schema. A direct adapter from pilot outputs to this generic input contract is deferred; do not
rename or manually reinterpret columns without a validated conversion step.

## Targets

- `report_only`: primary report-NLP model-development target.
- `image_only`: end-to-end clinical/image-validity target when image ratings are available.

Binary labels, kappa, and CDS thresholds are secondary summaries. Do not create `gold_*` outputs
until a completed human-reviewed reference standard exists.

## Annotation Input

Use a long rater-assignment table with required columns:

```text
case_id
rater_id
review_task
probability_0_100
```

Valid `review_task` values are:

```text
image_only
report_only
```

Optional local identifiers such as `subject_id`, `study_id`, `accession_id`, `encounter_id`, and
`annotation_phase` may be present in ignored local artifacts. Report text, image paths, full
annotation workbooks, and reviewer free text should not be committed.

## Annotation Evaluation

Run the synthetic smoke example:

```bash
make annotation-eval
```

The synthetic smoke target writes under `data/derived/annotations/synthetic/` and
`artifacts/annotations/synthetic/`.

Run with a real local completed annotation export:

```bash
uv run python scripts/validate_probabilistic_annotations.py \
  --input /path/to/completed_probabilistic_annotations.csv \
  --require-completed
```

Default outputs:

```text
data/derived/annotations/validated_rater_ratings.parquet
data/derived/annotations/case_reference_standard.parquet
artifacts/annotations/annotation_validation_summary.json
artifacts/annotations/annotation_agreement_summary.csv
artifacts/annotations/annotation_agreement_summary.md
artifacts/annotations/rater_pairwise_agreement.csv
artifacts/annotations/case_level_disagreement_flags.csv
```

All outputs are ignored by git.

## Probabilistic Benchmark Input

Predictions use a long table with:

```text
case_id
model_name
prediction_score
```

Existing silver-baseline predictions can also be normalized when they contain `subject_id`,
`study_id`, `task`, `model`, and `prediction_score`; `case_id` is derived from
`subject_id_study_id` and `model_name` from `task__model`.

## Probabilistic Benchmark

Run the synthetic smoke example:

```bash
make benchmark-eval
```

The synthetic smoke target writes under `artifacts/benchmark/synthetic/`.

Run with a real local reference standard and prediction table:

```bash
uv run python scripts/run_probabilistic_benchmark.py \
  --reference data/derived/annotations/case_reference_standard.parquet \
  --predictions /path/to/model_predictions.parquet \
  --cds-threshold 0.67
```

Default outputs:

```text
artifacts/benchmark/probabilistic_metrics.csv
artifacts/benchmark/probabilistic_metrics.json
artifacts/benchmark/expected_threshold_metrics.csv
artifacts/benchmark/calibration_bins_report.csv
artifacts/benchmark/calibration_bins_image.csv
artifacts/benchmark/calibration_curve_report.png
artifacts/benchmark/calibration_curve_image.png
artifacts/benchmark/benchmark_join_audit.json
artifacts/benchmark/benchmark_summary.md
```

Optional timing outputs are written only when timing fields are present.

## Notebook Reports

Renderable notebook reports:

```bash
quarto render notebooks/01_annotation_reference_standard.qmd
quarto render notebooks/02_probabilistic_model_benchmark.qmd
```

The notebooks default to tracked synthetic fixtures. Use environment variables to point them at
real local files:

```bash
ARDS_PROB_ANNOTATIONS=/path/to/annotations.csv \
  quarto render notebooks/01_annotation_reference_standard.qmd

ARDS_PROB_REFERENCE=data/derived/annotations/case_reference_standard.parquet \
ARDS_PROB_PREDICTIONS=/path/to/predictions.parquet \
  quarto render notebooks/02_probabilistic_model_benchmark.qmd
```

Notebook outputs should not include raw report text, image paths, or row-level PHI-bearing
workbooks.
