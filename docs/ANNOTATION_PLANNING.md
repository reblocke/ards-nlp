# Annotation Planning

This workflow converts aggregate outputs from the three-rater Intermountain REDCap pilot into transparent
annotation-design scenarios. It does not determine a statistically sufficient training sample or
replace a protocol/statistical-analysis-plan decision.

## Minimal Setup

Run the synthetic demonstration without REDCap or MIMIC access:

```bash
uv sync --locked
make annotation-planning-smoke
```

For real pilot results:

```bash
cp config/annotation_pilot.example.yaml config/annotation_pilot.yaml
cp config/annotation_planning.example.yaml config/annotation_planning.yaml
make doctor-annotation
make annotation-planning
```

`make annotation-planning` first refreshes the real agreement report, then reads only its
aggregate, text-free CSV outputs.

## Scenario Contract

The default planning grid uses:

- prevalence assumptions of 0.10, 0.20, and 0.30;
- expected sensitivity/specificity of 0.80 and 0.90;
- 95% Wilson interval half-width targets of 0.05 and 0.10;
- reliability targets of 0.80 and 0.90;
- a 20% three-rater overlap scenario;
- third-rater review at the observed proportion of cases with at least a 25-point spread;
- illustrative retraining workloads of 250, 500, and 1,000 completed cases.

All values are configurable in the ignored `config/annotation_planning.yaml`. Results are shown
separately for image review, report review, and combined person-rating workload.

Required positive and negative validation denominators use expected Wilson intervals. Total case
counts apply the configured prevalence and observed task-specific all-rater case-completion rate.
Individual rating completion is reported separately. The planner reports a grid rather than
selecting one preferred design.

Rater-count estimates apply the Spearman-Brown relationship to the point estimate of ICC(2,1).
They inherit the pilot ICC's uncertainty and should not be interpreted without its confidence
interval and the underlying disagreement summaries.

## Workload Designs

- `full_three_rater_review`: every invited case receives all three ratings.
- `single_review_plus_overlap`: one rating per case plus two additional ratings on the configured
  overlap sample (20% by default).
- `double_review_plus_disagreement_triggered_third`: two ratings per case plus a third rating at
  the observed pilot disagreement rate.

Person-ratings are always reported. Reviewer-hours remain blank unless explicit minutes per image
and report rating are configured. Timing is never inferred from filenames, timestamps, or row
order.

## Outputs And Boundaries

```text
artifacts/annotations/planning/
reports/annotation_planning/ards_annotation_design_scenarios.html
```

Outputs are aggregate and ignored by git. They contain no case IDs, accessions, rater IDs, report
text, or absolute paths. Synthetic outputs are marked as illustrative.

The binary precision grid is secondary to continuous physician probabilities. Retraining rows are
fixed workload examples only; empirical learning curves are required before claiming that a
specific training-set size is sufficient.

These planning scenarios are independent of the MIMIC silver-label comparator workflow.
