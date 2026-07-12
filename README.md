# ARDS CXR Report-NLP Benchmark

[![CI](https://github.com/reblocke/ards-nlp/actions/workflows/ci.yml/badge.svg)](https://github.com/reblocke/ards-nlp/actions/workflows/ci.yml)
[![CLAMP release gates](https://github.com/reblocke/ards-nlp/actions/workflows/clamp-release.yml/badge.svg)](https://github.com/reblocke/ards-nlp/actions/workflows/clamp-release.yml)

This public preview supports two research workflows for the same imaging criterion:

> The report or image indicates current bilateral pulmonary opacities or infiltrates.

1. **Physician annotation:** estimate agreement, image-report alignment, and review workload from
   three separately exported REDCap ratings.
2. **MIMIC-CXR benchmarking:** compare report-NLP methods on 227,835 studies against automated
   strict and sensitive silver references.

There is no MIMIC human gold standard in this repository. Aggregate benchmark results are
engineering diagnostics against automated labels, not clinical accuracy. The code does not
diagnose full clinical ARDS.

## Quick Start

Install [uv](https://docs.astral.sh/uv/) and Python 3.11 or later:

```bash
uv sync --locked
make run
make test
```

For the synthetic annotation reports, also install
[Quarto 1.8.26](https://quarto.org/docs/get-started/), then run:

```bash
make annotation-planning-smoke
```

`make run` prints configuration and available targets. It does not query BigQuery or read
restricted data.

## Workflow Status

| Workflow | State | Primary entry point |
|---|---|---|
| Synthetic physician-annotation planning | Ready | `make annotation-planning-smoke` |
| Three-rater REDCap analysis | Code ready; restricted exports required | `make annotation-planning` |
| Aggregate MIMIC comparator snapshot | Available | [`docs/COMPARATOR_SNAPSHOT_V1.md`](docs/COMPARATOR_SNAPSHOT_V1.md) |
| MIMIC silver-label rebuild | Code ready; PhysioNet and GCP access required | `make build` |
| Python legacy-CLAMP compatibility mirror | Optional; three external resources required | `make clamp-ards-python` |
| Genuine public CLAMP parity fixture | Pending | [`docs/CLAMP_ARDS_GOLDEN_CORPUS_RUNBOOK.md`](docs/CLAMP_ARDS_GOLDEN_CORPUS_RUNBOOK.md) |

## Physician Annotation

The annotation workflow expects three separate REDCap CSV exports mapped explicitly to `R01`,
`R02`, and `R03`. It never infers rater identity from filenames, row order, or timestamps.

Copy the example configuration, enter approved local paths, and run:

```bash
cp config/annotation_pilot.example.yaml config/annotation_pilot.yaml
cp config/annotation_planning.example.yaml config/annotation_planning.yaml
make doctor-annotation
make annotation-planning
```

Report text is discarded before derived tables are written. Outputs remain ignored under
`data/derived/annotations/`, `artifacts/annotations/`, and `reports/`.

See [`docs/ANNOTATION_PILOT_EVALUATION.md`](docs/ANNOTATION_PILOT_EVALUATION.md),
[`docs/ANNOTATION_PLANNING.md`](docs/ANNOTATION_PLANNING.md), and
[`docs/PROBABILISTIC_EVALUATION_RUNBOOK.md`](docs/PROBABILISTIC_EVALUATION_RUNBOOK.md).

## MIMIC Comparator Benchmark

The source cohort contains 227,835 studies with deterministic subject-level train, validation, and
test assignments. Metrics are reported on validation and test only. The strict and sensitive
targets are automated silver labels built from report regex, RadGraph, and MIMIC-CXR-JPG evidence.

Available aggregate comparators include the legacy CLAMP teacher, its Python compatibility mirror,
Amaral preprocessing, TF-IDF logistic regression, and structured controls. UW HANSO remains gated
on trained weights and terms; the Afshar SVC remains permission-gated.

Restricted local execution begins with:

```bash
cp config/config.example.yaml config/config.yaml
make doctor-comparators
make comparators-ready
make comparator-snapshot
```

External clones, model artifacts, report packets, entities, and row-level predictions remain
ignored.

## Optional CLAMP Compatibility Mirror

This repository contains **no CLAMP executable and no exported CLAMP project**. The MIT-licensed
JSON specification at
[`src/ards_cxr_benchmark/clamp_ards/data/legacy_ards_phenotype_spec.json`](src/ards_cxr_benchmark/clamp_ards/data/legacy_ards_phenotype_spec.json)
re-expresses the authorized 23-term dictionary, pipeline choices, and rule semantics. That design
is attributed to Dan Knox.

The mirror requires these three resources, obtained separately under licenses applicable to the
user:

| Required relative path | Purpose |
|---|---|
| `Components/Sentence detector/DF_Clamp_sentence_detector/defaultAbbrs.txt` | Sentence abbreviations |
| `Components/Assertion classifier/DF_NegEx_assertion/defaultNegexDict.txt` | NegEx cues |
| `Components/Tokenizer/DF_Clamp_tokenizer/defaultTokenRule.txt` | Token delimiters |

Place them under the ignored default directory `data/external/clamp_ards_project/`, set
`ARDS_CLAMP_PROJECT_DIR`, or pass `--project-dir`. The packaged manifest validates the exact
production fingerprints. `--resource-manifest` is available for reviewed alternate fixtures.

```bash
export ARDS_CLAMP_PROJECT_DIR=/path/to/licensed/ards-project
uv run python scripts/run_python_clamp_ards.py \
  --config config/config.yaml
```

`run_legacy_ards_clamp_mirror()` and `predict_legacy_ards_label()` remain available as Python APIs.
Batch summaries record hashes for the three external resources plus
`phenotype_spec_version` and `phenotype_spec_sha256`.

### Reproducibility Qualification

Exact parity was observed on the restricted 227,835-document corpus under the fingerprinted
resource set: 227,835 exact entity-multiset documents, 80,908 exact entities, and no document
label, count, status, multiplicity, field, or output-order differences. This is a compatibility
result, not independent model evidence.

Independent reproduction requires separately licensed resources. The public synthetic input
matrix is deterministic and non-clinical, but genuine CLAMP outputs, named PHI review, and
two-run determinism evidence remain pending. CI reports that maturity state explicitly without
claiming a completed public oracle.

See [`docs/CLAMP_ARDS_PYTHON_PORT.md`](docs/CLAMP_ARDS_PYTHON_PORT.md),
[`docs/CLAMP_ARDS_PYTHON_PARITY.md`](docs/CLAMP_ARDS_PYTHON_PARITY.md), and
[`docs/CLAMP_ARDS_RESOURCE_REVIEW.md`](docs/CLAMP_ARDS_RESOURCE_REVIEW.md).

## Data And Security

Never commit MIMIC reports or images, RadGraph JSON, REDCap exports, rendered real annotation
reports, CLAMP inputs or outputs, entity mentions, row-level predictions, local configuration, or
external model artifacts.

Before submitting a change:

```bash
make fmt
make lint
make test
make release-audit
make clamp-ards-resources-public-audit
```

The resource audit checks the current tree and reachable Git history for fingerprints of all 23
files in the excluded legacy export. Office documents and common restricted binary/data formats
also fail the release audit.

Report vulnerabilities through GitHub private vulnerability reporting as described in
[`SECURITY.md`](SECURITY.md).

## Documentation

- [`docs/DATA_MANAGEMENT.md`](docs/DATA_MANAGEMENT.md): restricted-data boundaries.
- [`docs/COMPARATOR_BAKEOFF_RUNBOOK.md`](docs/COMPARATOR_BAKEOFF_RUNBOOK.md): comparator execution.
- [`docs/MODELING_RUNBOOK.md`](docs/MODELING_RUNBOOK.md): silver-label modeling.
- [`docs/CLAMP_ARDS_STAGE1_INPUT_RUNBOOK.md`](docs/CLAMP_ARDS_STAGE1_INPUT_RUNBOOK.md): licensed local CLAMP handoff.
- [`docs/CLAMP_ARDS_STAGE2_OUTPUT_RUNBOOK.md`](docs/CLAMP_ARDS_STAGE2_OUTPUT_RUNBOOK.md): returned-output parsing.
- [`docs/CLAMP_ARDS_GOLDEN_CORPUS_RUNBOOK.md`](docs/CLAMP_ARDS_GOLDEN_CORPUS_RUNBOOK.md): pending public fixture workflow.
- [`docs/THIRD_PARTY_NOTICES.md`](docs/THIRD_PARTY_NOTICES.md): licenses and external boundaries.

## License

Repository code and the authorized re-expressed phenotype specification are available under the
MIT License. External CLAMP resources, MIMIC-CXR, and third-party comparator software retain their
own terms and are not relicensed or redistributed here.
