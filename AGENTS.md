# AGENTS.md

## Purpose
- This repository builds a MIMIC-CXR report-NLP benchmark for bilateral pulmonary opacities.

## Repo Map
- `src/ards_cxr_benchmark/` - importable package code for config, parsing, RadGraph flattening, BigQuery helpers, QA, and label logic.
- `scripts/` - thin orchestration and entrypoints.
- `sql/` - parameterized BigQuery SQL for discovery, derived tables, QA, sampling, and exports.
- `tests/` - pytest coverage for reusable code and pipeline behavior.
- `data/raw/`, `data/external/`, `data/processed/`, `data/derived/` - input and generated datasets.
- `artifacts/`, `reports/`, `docs/`, `notebooks/`, `config/` - outputs, docs, exploration, and config examples.

## Commands and task-specific runbooks
Setup uses `uv sync`. Local code checks are `make lint` and affected pytest tests; `make test` runs the full suite. `make run` checks local config/CLI wiring. No typecheck target is configured.

Consult the relevant runbook before using its workflow; these are separate workflows, not a checklist to run for every edit:
- Source discovery, ingestion, BigQuery, QA, sample/export/splits: `README.md`, `Makefile`, and `docs/DATA_MANAGEMENT.md`. Cloud work needs authorized access and configured project/source paths.
- Silver-label training and artifact QA: `docs/MODELING_RUNBOOK.md`.
- Probabilistic benchmarks and REDCap annotation: `docs/PROBABILISTIC_EVALUATION_RUNBOOK.md`, `docs/ANNOTATION_PILOT_EVALUATION.md`, or `docs/ANNOTATION_PLANNING.md`.
- Licensed CLAMP input/output: `docs/CLAMP_ARDS_STAGE1_INPUT_RUNBOOK.md` and `docs/CLAMP_ARDS_STAGE2_OUTPUT_RUNBOOK.md`.
- Python mirror, fixture evidence, and restricted parity: `docs/CLAMP_ARDS_PYTHON_PARITY.md` and `docs/CLAMP_ARDS_PUBLIC_FIXTURE_ACCEPTANCE.md`.
- External comparator gates and bakeoff: `docs/COMPARATOR_BAKEOFF_RUNBOOK.md` and the named comparator runbook it links.
- Readiness/public hygiene: `make doctor`, `make release-audit`, and `make clamp-ards-resources-audit` as applicable. Full target definitions remain in `Makefile`.

## Project Conventions
- Python 3.11+ is the runtime (`.python-version`, `pyproject.toml`).
- Use `uv` for environment and dependency management (`pyproject.toml`, `uv.lock`).
- Use Ruff only for formatting and linting; do not invent additional formatter or linter steps unless the repo adds them.
- Keep reusable logic in `src/` and keep `scripts/` focused on I/O and orchestration.
- Keep generated outputs in the existing `data/`, `artifacts/`, and `reports/` directories.
- Do not commit raw MIMIC report text, MIMIC-CXR-JPG files, RadGraph JSON, images, or derived report-level text tables.
- MIMIC-wide automated reference labels must use `silver_*` naming; reserve `gold_*` for a manually reviewed MIMIC subset.
- Primary v1 silver-label rule scope is full report text while retaining impression+findings and fallback fields for sensitivity analysis.
- Silver-label modeling must read from `model_development_extract`/`model_development_splits`; do not train baseline scripts directly from raw source files.
- `make modeling-smoke` must write to smoke-specific ignored paths, not overwrite full-run modeling artifacts.
- CLAMP Stage 1 inputs are selected report text and must remain in the configured local CLAMP workspace or ignored restricted artifacts.
- CLAMP Stage 2 outputs are legacy teacher/baseline artifacts, not `gold_*` labels, and must remain ignored.
- CLAMP Stage 2 uses the final tabular TXT exports; parse the compact archive directly and do not
  extract or commit returned entity files.
- Python CLAMP processing uses Python source indices internally; public/exported sentence, token,
  and entity offsets use half-open Java/UIMA UTF-16 code units.
- Frozen CLAMP resource keys use POSIX separators and raw-byte hashes. Never add the exported
  project or default CLAMP resources to Git.
- The Python mirror requires only `defaultAbbrs.txt`, `defaultNegexDict.txt`, and
  `defaultTokenRule.txt`; load them through `ARDS_CLAMP_PROJECT_DIR` or `--project-dir`.
- The authorized 23-term phenotype and rule settings live in the packaged MIT JSON specification,
  attributed to Dan Knox. Do not reconstruct dependencies on exported config or Ruta files.
- The public resource audit is mandatory on every PR and must fail if an excluded resource appears
  in the current tree or reachable history.
- The fixture-status job must remain green but explicitly pending until genuine two-run CLAMP
  output, named PHI review, and independent reproducibility evidence are available.
- Synthetic fixture inputs are authoritative bytes; read them through `manifest.csv` without
  newline conversion. Keep the pending corpus and smoke/parity Parquets under ignored artifacts
  until named reviews permit a finalized fixture commit.
- `--allow-pending` is a CI scaffold allowance only: partial expected output must fail, and a
  completed fixture must always execute strict sentence/token/entity/document/order comparison.
- REDCap pilot inputs require separate explicitly mapped rater exports; never infer rater identity
  from row order, timestamps, or filenames.
- REDCap pilot outputs must exclude report text, and rendered reports must exclude case identifiers
  and absolute paths.
- External comparator source, model artifacts, caches, and report packets remain ignored. Pickles
  load only in pinned isolated subprocesses after checksum verification.
- Comparator predictions use canonical `mimic_<subject_id>_<study_id>` IDs and must contain no
  report text or source paths.
- REDCap pilot output paths must remain under `reports/`, `artifacts/`, and `data/derived/` according
  to output category.
- Annotation planning consumes aggregate pilot outputs only; retraining case counts are workload
  examples until empirical learning curves exist.

## Done Criteria
- An implementation request includes local edits, applicable safe checks, and resolving regressions caused by the change. Complete that scope; preserve licensed, restricted-data, resource-review, and scientific acceptance gates.
- Documentation-only changes need affected-reference checks and `git diff --check`. For code changes, run focused pytest tests and Ruff checks on touched code; broaden for shared contracts or unresolved failures.
- Run `make run` when config loading or CLI wiring changes. `make qa` and other cloud/data targets require both authorized scope and configured access/inputs; credentials alone do not authorize them.
- Keep the mandatory public resource audit on every PR. Synthetic/CI coverage cannot accept licensed fixtures or restricted full-corpus parity; retain all pending reviews and evidence requirements below.
- Report commands actually run and unavailable gates. Leave unknown repo facts as `VERIFY` or `TODO` markers.

## Maintainer Checklist
- `VERIFY: update GCP project, BigQuery dataset, GCS bucket, and source paths in local config/config.yaml before running cloud targets.`
- `VERIFY: confirm actual PhysioNet dataset/table names available to the linked Google account.`
- `TODO: add Chest ImaGenome only after RadGraph/regex v1 is validated.`
- `TODO: run the real REDCap pilot after receiving three separate physician exports.`
- `TODO: complete the genuine CLAMP non-PHI fixture, named PHI review, and two-run determinism
  evidence in the public fixture issue.`
