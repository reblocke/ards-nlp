# AGENTS.md

## Purpose
- This file adds repo-specific guidance on top of `~/.codex/AGENTS.md`.
- Keep `~/.codex/AGENTS.md` behavior-only. Keep this file limited to facts visible in this repo.
- This repository builds a MIMIC-CXR report-NLP benchmark for bilateral pulmonary opacities.

## Repo Map
- `src/ards_cxr_benchmark/` - importable package code for config, parsing, RadGraph flattening, BigQuery helpers, QA, and label logic.
- `scripts/` - thin orchestration and entrypoints.
- `sql/` - parameterized BigQuery SQL for discovery, derived tables, QA, sampling, and exports.
- `tests/` - pytest coverage for reusable code and pipeline behavior.
- `data/raw/`, `data/external/`, `data/processed/`, `data/derived/` - input and generated datasets.
- `artifacts/`, `reports/`, `docs/`, `notebooks/`, `config/` - outputs, docs, exploration, and config examples.

## Commands
- Setup: `uv sync`
- Discover BigQuery source availability: `make discover`
- Create configured BigQuery working dataset: `make init`
- Ingest local MIMIC/RadGraph files to Parquet: `make ingest`
- Build benchmark BigQuery tables: `make build`
- QA benchmark tables: `make qa`
- Generate manual-review sample: `make sample`
- Build model-development extract: `make export`
- Build deterministic model-development splits: `make splits`
- Run a limited silver-label modeling smoke test: `make modeling-smoke`
- Run full silver-label baseline modeling: `make modeling`
- Validate local silver-label modeling artifacts: `make modeling-qa`
- Validate probabilistic image-only/report-only annotation ratings: `make annotation-eval`
- Run probabilistic model benchmark: `make benchmark-eval`
- Evaluate real REDCap pilot agreement: `make annotation-pilot`
- Render the synthetic REDCap pilot report: `make annotation-pilot-smoke`
- Convert real pilot aggregates into annotation-design scenarios: `make annotation-planning`
- Render synthetic annotation-design scenarios: `make annotation-planning-smoke`
- Sync a separately licensed local ARDS CLAMP project to the restricted workspace: `make clamp-ards-sync`
- Export CLAMP-ready CXR report inputs: `make clamp-ards-inputs`
- Build a compact returned CLAMP TXT packet: `make clamp-ards-output-packet`
- Parse returned ARDS CLAMP outputs: `make clamp-ards-parse`
- Benchmark CLAMP teacher outputs against silver labels: `make clamp-ards-teacher-benchmark`
- Run the deterministic Python CLAMP compatibility mirror: `make clamp-ards-python`
- Run the Python mirror on a generated synthetic fixture: `make clamp-ards-python-smoke`
- Strictly compare completed CLAMP-generated fixtures: `make clamp-ards-parity-fixtures`
- Validate the pending or completed fixture inventory: `make clamp-ards-parity-fixture-validate`
- Build the ignored licensed-Windows handoff: `make clamp-ards-parity-fixture-handoff`
- Audit CLAMP resource governance: `make clamp-ards-resources-audit`
- Run restricted full-corpus Python/CLAMP parity: `make clamp-ards-parity-restricted`
- Build the shared restricted comparator source packet: `make comparator-source`
- Normalize existing CLAMP and silver baselines: `make comparator-existing`
- Normalize the exact Python CLAMP compatibility mirror: `make comparator-clamp-python`
- Run the pinned Amaral comparator: `make comparator-amaral`
- Audit UW HANSO resource gates: `make comparator-uw-hanso-verify`
- Run UW HANSO synthetic smoke after acquiring verified resources: `make comparator-uw-hanso-smoke`
- Audit Afshar pickle and permission gates: `make comparator-afshar-inspect`
- Run the permission-gated Afshar synthetic smoke: `make comparator-afshar-smoke`
- Run the combined comparator bakeoff: `make comparator-benchmark`
- Run all currently available comparators and gated audits: `make comparators-ready`
- Write the tracked aggregate comparator snapshot: `make comparator-snapshot`
- Print collaborator workflow readiness: `make doctor`
- Audit public-release path and documentation hygiene: `make release-audit`
- Format: `make fmt`
- Lint: `make lint`
- Typecheck: `VERIFY: no typecheck command is configured.`
- Test: `make test`
- Run local config smoke check: `make run`

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
- The change is the smallest adequate diff for the request.
- Any command you mention was run, or you state clearly that it was not run.
- For code changes, `make fmt`, `make lint`, and `make test` pass.
- Run `make run` when the change touches config loading or CLI wiring.
- Run `make qa` only when configured BigQuery credentials and source tables are available.
- Leave unknown repo facts as explicit `VERIFY` or `TODO` markers instead of inventing policy.

## Maintainer Checklist
- `VERIFY: update GCP project, BigQuery dataset, GCS bucket, and source paths in local config/config.yaml before running cloud targets.`
- `VERIFY: confirm actual PhysioNet dataset/table names available to the linked Google account.`
- `TODO: add Chest ImaGenome only after RadGraph/regex v1 is validated.`
- `TODO: run the real REDCap pilot after receiving three separate physician exports.`
- `TODO: complete the genuine CLAMP non-PHI fixture, named PHI review, and two-run determinism
  evidence in the public fixture issue.`
