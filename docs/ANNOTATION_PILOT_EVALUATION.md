# Annotation Pilot Evaluation

This workflow evaluates continuous image-only and report-only physician ratings from the
Intermountain REDCap pilot. This cohort is separate from MIMIC. The workflow is limited to
agreement and image-report alignment; it does not evaluate an NLP model.

## Input Contract

V1 requires one REDCap CSV export per rater. Rater identity is assigned only through an explicit
YAML mapping. Do not infer rater identity from row order, timestamps, filenames, or completion
order. A single export whose rating fields were overwritten by successive raters cannot support
inter-rater analysis.

Create the ignored local configuration:

```bash
cp config/annotation_pilot.example.yaml config/annotation_pilot.yaml
```

Update the three input paths. Relative paths resolve from the repository root. The default case key
is `id_accession`; identifiers remain local and do not appear in the rendered report.

Required source fields are the configured case key, image/report rating fields, and their two
completion fields. `id`, `id2`, `raw_data_complete`, and `interpretation_text` are optional for QA.
If report text is present, it is discarded immediately after ingestion and is never exported.

## Completion Rule

A task rating is analyzed only when its completion field equals `2`, its probability is numeric,
and it lies within 0-100 inclusive. Completion codes outside `0`, `1`, and `2` fail validation.
Incomplete or unverified ratings remain visible in local QA counts but do not enter the analyses.

## Run

Run the tracked synthetic smoke packet without REDCap or BigQuery access:

```bash
make annotation-pilot-smoke
```

Run the real local pilot after configuring the three restricted exports:

```bash
make annotation-pilot
```

After the real agreement report is complete, use `make annotation-planning` to translate its
aggregate completion, disagreement, and ICC results into configurable reviewer-workload and
validation-precision scenarios. See `docs/ANNOTATION_PLANNING.md`.

Override the config explicitly when needed:

```bash
ANNOTATION_PILOT_CONFIG=/absolute/path/to/config.yaml make annotation-pilot
```

## Analyses

Image and report ratings are analyzed separately. ICC(2,1) and ICC(2,k) use cases completed by all
configured raters. Pairwise signed difference, MAE, RMSE, Pearson correlation, and Spearman
correlation use each pair's available shared cases. Correlations are descriptive association
measures, not agreement measures.

The primary image-report analysis compares all-rater case means. It reports signed difference,
MAE, RMSE, correlations, Bland-Altman limits, and deterministic percentile bootstrap intervals.
Within-rater image-report comparisons are reported separately.

## Outputs And Privacy

Restricted, text-free row-level outputs are written under:

```text
data/derived/annotations/pilot/
artifacts/annotations/pilot/
```

The rendered report is:

```text
reports/annotation_pilot/ards_annotation_pilot_agreement.html
```

All configured output paths are validated before analysis. Reports must remain under `reports/`,
aggregate/review artifacts under `artifacts/`, and derived tables under `data/derived/`. The render
command places HTML in the configured `outputs.report_dir` and records the same path in the output
manifest.

The report contains aggregate tables, anonymous case positions in figures, and repository-relative
output paths. It does not contain report text, accession numbers, source rows, or absolute paths.
The disagreement review CSV remains restricted because it contains local case IDs and rater-level
ratings.

## Interpretation Boundaries

- Continuous 0-100 ratings are primary.
- This workflow estimates annotation feasibility and measurement reproducibility.
- Image-report differences describe discordance between information sources, not NLP error.
- No `gold_*` output is created by this pilot workflow.
- Annotation time is unavailable unless explicit timestamps, duration fields, or audit-log data are
  supplied.
- The generic probabilistic benchmark can later use Intermountain report-only and image-only case
  means, but a pilot-output-to-benchmark adapter is not implemented in v1.
