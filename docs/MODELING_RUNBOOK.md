# Silver-label modeling runbook

This runbook covers exploratory modeling against automated silver labels. These outputs are not
human-adjudicated gold labels.

## Inputs

- BigQuery source table: `model_development_extract`
- BigQuery split table: `model_development_splits`
- Local full-run cache: `data/derived/modeling/model_development_extract.parquet`
- Local smoke-run cache: `data/derived/modeling/smoke/model_development_extract.parquet`

The local caches may contain report text and are ignored by git.

## Commands

Build deterministic subject-level splits:

```bash
make splits
```

Run a limited smoke model pass:

```bash
make modeling-smoke
```

Smoke artifacts are written under:

```text
data/derived/modeling/smoke/
artifacts/modeling/smoke/
```

Run the full silver-label baseline pass:

```bash
make modeling
```

Full artifacts are written under:

```text
data/derived/modeling/
artifacts/modeling/
```

Validate full local modeling artifacts:

```bash
make modeling-qa
```

Generate the aggregate v1 modeling snapshot:

```bash
uv run python scripts/write_modeling_snapshot.py
```

## Outputs

- `silver_baseline_metrics.csv` and `.json`: model metrics by task and eval split.
- `silver_baseline_strata.csv`: prediction summaries by source, priority, and QA flags.
- `silver_baseline_summary.md`: concise local run summary.
- `modeling_qa_summary.md` and `.json`: split, row-count, schema, and text-leakage checks.
- `silver_baseline_predictions.parquet`: predictions without report text columns.

Fitted TF-IDF and structured models are stored under `artifacts/modeling/models/`.

## Interpretation

- `silver_score_rule` is the explicit score/rule benchmark.
- `tfidf_logreg` is the weak-label text baseline.
- `structured_logreg` uses source-derived silver-rule inputs, so treat it as a rule diagnostic rather
  than independent generalization evidence.

All metrics are against automated silver labels. Do not present them as clinical accuracy or gold
standard performance.

## QA Expectations

- No subject should appear in more than one split.
- Full predictions should contain no report-text columns.
- Required metric rows should exist for strict and sensitive tasks across validation and test splits.
- Generated data, model artifacts, predictions, and snapshots remain ignored by git.
