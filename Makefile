.DEFAULT_GOAL := help

.PHONY: help
help:
	@echo "Targets:"
	@echo "  uv-sync   Create/update the local env from uv.lock"
	@echo "  fmt       Format code (ruff)"
	@echo "  lint      Lint code (ruff)"
	@echo "  test      Run unit tests (pytest)"
	@echo "  run       Print benchmark configuration summary"
	@echo "  doctor    Print readiness for annotation, comparator, and full-build use cases"
	@echo "  doctor-*  Strict readiness check for one collaborator use case"
	@echo "  release-audit Check tracked-file and documentation sharing hygiene"
	@echo "  discover  List available PhysioNet BigQuery datasets"
	@echo "  init      Create the configured BigQuery working dataset"
	@echo "  ingest    Convert local MIMIC-CXR/JPG/RadGraph files to Parquet"
	@echo "  build     Build derived BigQuery benchmark tables"
	@echo "  qa        Run BigQuery QA checks"
	@echo "  sample    Build and export manual-review sample"
	@echo "  export    Build model-development extract"
	@echo "  splits    Build deterministic subject-level model splits"
	@echo "  modeling-smoke Run a limited silver-label modeling smoke run"
	@echo "  modeling  Run silver-label baseline models"
	@echo "  modeling-qa Validate local silver-label modeling artifacts"
	@echo "  annotation-eval Validate probabilistic annotation ratings"
	@echo "  benchmark-eval Run probabilistic model benchmark"
	@echo "  annotation-pilot Evaluate real REDCap pilot agreement"
	@echo "  annotation-pilot-smoke Render the synthetic REDCap pilot report"
	@echo "  annotation-planning Convert pilot aggregates into planning scenarios"
	@echo "  annotation-planning-smoke Render synthetic annotation planning scenarios"
	@echo "  clamp-ards-sync Sync a local ARDS CLAMP project into the restricted workspace"
	@echo "  clamp-ards-inputs Export CXR reports to CLAMP-ready input files"
	@echo "  clamp-ards-output-packet Build a restricted TXT-only returned-output packet"
	@echo "  clamp-ards-parse Parse returned ARDS CLAMP outputs"
	@echo "  clamp-ards-teacher-benchmark Benchmark CLAMP teacher outputs against silver labels"
	@echo "  clamp-ards-python Run the deterministic Python CLAMP compatibility port"
	@echo "  clamp-ards-python-smoke Run the Python port on tracked synthetic inputs"
	@echo "  clamp-ards-characterize Stream a 5,000-document XMI characterization sample"
	@echo "  clamp-ards-parity-restricted Strictly compare Python output with local CLAMP oracle"
	@echo "  clamp-ards-parity-fixtures Strictly compare completed CLAMP-generated fixtures"
	@echo "  clamp-ards-parity-fixture-* Prepare, validate, hand off, or import golden fixtures"
	@echo "  clamp-ards-resources-* Audit private-review or public-release resource gates"
	@echo "  comparator-source Build the restricted common MIMIC comparator packet"
	@echo "  comparator-existing Normalize CLAMP and internal silver baselines"
	@echo "  comparator-clamp-python Normalize the exact Python CLAMP compatibility mirror"
	@echo "  comparator-amaral-* Fetch, verify, smoke, run, and benchmark Amaral"
	@echo "  comparator-uw-hanso-* Fetch, verify, or run gated UW HANSO"
	@echo "  comparator-afshar-* Fetch, inspect, or run gated Afshar SVC"
	@echo "  comparator-benchmark Build the combined MIMIC silver-label bakeoff"
	@echo "  comparator-snapshot Write the tracked aggregate comparator snapshot"
	@echo "  comparators-ready Run all currently available comparators and gated audits"
	@echo "  clean     Remove caches / local build artifacts"

.PHONY: uv-sync
uv-sync:
	uv sync

.PHONY: fmt
fmt:
	uv run ruff format .

.PHONY: lint
lint:
	uv run ruff check .

.PHONY: test
test:
	uv run pytest -q

.PHONY: run
run:
	uv run python scripts/run_pipeline.py

.PHONY: doctor
doctor:
	uv run python scripts/check_readiness.py

.PHONY: doctor-annotation
doctor-annotation:
	uv run python scripts/check_readiness.py --use-case annotation --strict

.PHONY: doctor-comparators
doctor-comparators:
	uv run python scripts/check_readiness.py --use-case comparators --strict

.PHONY: doctor-build
doctor-build:
	uv run python scripts/check_readiness.py --use-case build --strict

.PHONY: release-audit
release-audit:
	uv run python scripts/audit_public_release.py

.PHONY: discover
discover:
	uv run python scripts/run_bigquery_sql.py --sql sql/00_discover_physionet_datasets.sql

.PHONY: init
init:
	uv run python scripts/run_bigquery_sql.py --sql sql/10_create_working_dataset.sql

.PHONY: ingest
ingest:
	uv run python scripts/ingest_mimic_cxr_reports.py
	uv run python scripts/ingest_mimic_cxr_jpg_labels.py
	uv run python scripts/flatten_radgraph.py

.PHONY: build
build:
	$(MAKE) init
	uv run python scripts/run_bigquery_sql.py --sql sql/20_build_mimic_cxr_report_base.sql
	uv run python scripts/run_bigquery_sql.py --sql sql/21_parse_report_sections.sql
	uv run python scripts/run_bigquery_sql.py --sql sql/30_load_or_join_mimic_cxr_jpg_labels.sql
	uv run python scripts/run_bigquery_sql.py --sql sql/40_flatten_radgraph.sql
	uv run python scripts/run_bigquery_sql.py --sql sql/41_build_radgraph_relation_expanded.sql
	uv run python scripts/run_bigquery_sql.py --sql sql/42_build_radgraph_report_features.sql
	uv run python scripts/run_bigquery_sql.py --sql sql/43_build_radgraph_bilateral_features.sql
	uv run python scripts/run_bigquery_sql.py --sql sql/50_build_regex_report_features.sql
	uv run python scripts/run_bigquery_sql.py --sql sql/60_build_silver_reference_candidates.sql

.PHONY: qa
qa:
	uv run python scripts/run_bigquery_sql.py --sql sql/90_qa_checks.sql
	uv run python scripts/validate_reference_tables.py

.PHONY: sample
sample:
	uv run python scripts/run_bigquery_sql.py --sql sql/70_sample_manual_review_set.sql
	uv run python scripts/export_annotation_csv.py --table manual_review_sample --out-csv artifacts/samples/manual_review_sample.csv

.PHONY: export
export:
	uv run python scripts/run_bigquery_sql.py --sql sql/80_export_model_development_extract.sql

.PHONY: splits
splits:
	uv run python scripts/run_bigquery_sql.py --sql sql/81_build_model_development_splits.sql

.PHONY: modeling-smoke
modeling-smoke:
	$(MAKE) splits
	uv run python scripts/run_silver_baselines.py --refresh-cache --limit 5000 --max-features 5000

.PHONY: modeling
modeling:
	$(MAKE) splits
	uv run python scripts/run_silver_baselines.py --refresh-cache

.PHONY: modeling-qa
modeling-qa:
	uv run python scripts/validate_modeling_outputs.py

.PHONY: annotation-eval
annotation-eval:
	uv run python scripts/validate_probabilistic_annotations.py

.PHONY: benchmark-eval
benchmark-eval:
	uv run python scripts/run_probabilistic_benchmark.py

ANNOTATION_PILOT_CONFIG ?= config/annotation_pilot.yaml
ANNOTATION_PLANNING_CONFIG ?= config/annotation_planning.yaml

.PHONY: annotation-pilot
annotation-pilot:
	uv run python scripts/render_annotation_pilot.py --config "$(ANNOTATION_PILOT_CONFIG)"

.PHONY: annotation-pilot-smoke
annotation-pilot-smoke:
	uv run python scripts/render_annotation_pilot.py \
		--config tests/fixtures/redcap_annotation/config.yaml

.PHONY: annotation-planning
annotation-planning: annotation-pilot
	uv run python scripts/render_annotation_planning.py --config "$(ANNOTATION_PLANNING_CONFIG)"

.PHONY: annotation-planning-smoke
annotation-planning-smoke: annotation-pilot-smoke
	uv run python scripts/render_annotation_planning.py \
		--config tests/fixtures/redcap_annotation/planning_config.yaml

.PHONY: clamp-ards-sync
clamp-ards-sync:
	uv run python scripts/sync_clamp_ards_project.py --config config/config.yaml

.PHONY: clamp-ards-inputs
clamp-ards-inputs:
	uv run python scripts/export_clamp_ards_inputs.py --config config/config.yaml

CLAMP_OUTPUT_SOURCE_ARCHIVE ?= ARDS CLAMP Output.zip

.PHONY: clamp-ards-output-packet
clamp-ards-output-packet:
	uv run python scripts/prepare_clamp_ards_output_packet.py \
		--config config/config.yaml \
		--source-archive "$(CLAMP_OUTPUT_SOURCE_ARCHIVE)"

.PHONY: clamp-ards-parse
clamp-ards-parse:
	uv run python scripts/parse_clamp_ards_outputs.py --config config/config.yaml

.PHONY: clamp-ards-teacher-benchmark
clamp-ards-teacher-benchmark:
	uv run python scripts/benchmark_clamp_ards_teacher.py --config config/config.yaml

.PHONY: clamp-ards-python
clamp-ards-python:
	uv run python scripts/run_python_clamp_ards.py --config config/config.yaml

CLAMP_ARDS_FIXTURE_ROOT ?=
CLAMP_ARDS_PUBLIC_RESOURCE_DIR ?= tests/fixtures/clamp_ards_external_resources
CLAMP_ARDS_PUBLIC_RESOURCE_MANIFEST ?= tests/fixtures/clamp_ards_external_resources/manifest.json
CLAMP_ARDS_PENDING_FIXTURE_ROOT ?= artifacts/restricted/clamp_ards/python/pending_fixture
CLAMP_ARDS_GOLDEN_FIXTURE_ROOT ?= tests/fixtures/clamp_ards_parity
CLAMP_ARDS_PENDING_ROOT = $(if $(CLAMP_ARDS_FIXTURE_ROOT),$(CLAMP_ARDS_FIXTURE_ROOT),$(CLAMP_ARDS_PENDING_FIXTURE_ROOT))
CLAMP_ARDS_STRICT_ROOT = $(if $(CLAMP_ARDS_FIXTURE_ROOT),$(CLAMP_ARDS_FIXTURE_ROOT),$(CLAMP_ARDS_GOLDEN_FIXTURE_ROOT))
CLAMP_ARDS_FIXTURE_ARTIFACT_DIR ?= artifacts/restricted/clamp_ards/python/fixture_parity
CLAMP_ARDS_HANDOFF_DIR ?= artifacts/restricted/clamp_ards/parity_handoff
CLAMP_ARDS_RUN_1 ?=
CLAMP_ARDS_RUN_2 ?=
CLAMP_ARDS_RUN_1_PROVENANCE ?=
CLAMP_ARDS_RUN_2_PROVENANCE ?=
CLAMP_FIXTURE_ALLOW_PENDING ?= 0
CLAMP_FIXTURE_PENDING_FLAG = $(if $(filter 1 true yes,$(CLAMP_FIXTURE_ALLOW_PENDING)),--allow-pending,)

.PHONY: clamp-ards-python-smoke
clamp-ards-python-smoke:
	ARDS_CLAMP_PROJECT_DIR="$(CLAMP_ARDS_PUBLIC_RESOURCE_DIR)" \
	ARDS_CLAMP_RESOURCE_MANIFEST="$(CLAMP_ARDS_PUBLIC_RESOURCE_MANIFEST)" \
	uv run python scripts/run_python_clamp_ards.py \
		--config config/config.example.yaml \
		--fixture-root "$(CLAMP_ARDS_PENDING_ROOT)" \
		--project-dir "$(CLAMP_ARDS_PUBLIC_RESOURCE_DIR)" \
		--resource-manifest "$(CLAMP_ARDS_PUBLIC_RESOURCE_MANIFEST)" \
		--entity-output "$(CLAMP_ARDS_FIXTURE_ARTIFACT_DIR)/smoke_entities.parquet" \
		--prediction-output "$(CLAMP_ARDS_FIXTURE_ARTIFACT_DIR)/smoke_predictions.parquet" \
		--summary-output "$(CLAMP_ARDS_FIXTURE_ARTIFACT_DIR)/smoke_summary.json" \
		--no-progress

.PHONY: clamp-ards-characterize
clamp-ards-characterize:
	uv run python scripts/characterize_clamp_ards_xmi.py --config config/config.yaml

.PHONY: clamp-ards-parity-restricted
clamp-ards-parity-restricted: clamp-ards-python
	uv run python scripts/compare_clamp_python_parity.py --config config/config.yaml --require-order

.PHONY: clamp-ards-parity-fixtures
clamp-ards-parity-fixtures:
	uv run python scripts/compare_clamp_ards_fixtures.py \
		--fixture-root "$(CLAMP_ARDS_STRICT_ROOT)" \
		--output-dir "$(CLAMP_ARDS_FIXTURE_ARTIFACT_DIR)" $(CLAMP_FIXTURE_PENDING_FLAG)

.PHONY: clamp-ards-parity-fixture-prepare
clamp-ards-parity-fixture-prepare:
	uv run python scripts/generate_clamp_ards_parity_fixture.py generate \
		--output "$(CLAMP_ARDS_PENDING_ROOT)" \
		--project-dir "$(CLAMP_ARDS_PUBLIC_RESOURCE_DIR)" \
		--resource-manifest "$(CLAMP_ARDS_PUBLIC_RESOURCE_MANIFEST)"

.PHONY: clamp-ards-parity-fixture-validate
clamp-ards-parity-fixture-validate:
	ARDS_CLAMP_PROJECT_DIR="$(CLAMP_ARDS_PUBLIC_RESOURCE_DIR)" \
	ARDS_CLAMP_RESOURCE_MANIFEST="$(CLAMP_ARDS_PUBLIC_RESOURCE_MANIFEST)" \
	uv run python scripts/generate_clamp_ards_parity_fixture.py validate \
		--fixture "$(CLAMP_ARDS_PENDING_ROOT)" --allow-pending

.PHONY: clamp-ards-parity-fixture-handoff
clamp-ards-parity-fixture-handoff:
	uv run python scripts/prepare_clamp_ards_parity_handoff.py \
		--fixture-root "$(CLAMP_ARDS_PENDING_ROOT)" \
		--destination-dir "$(CLAMP_ARDS_HANDOFF_DIR)"

.PHONY: clamp-ards-parity-fixture-import
clamp-ards-parity-fixture-import:
	@test -n "$(CLAMP_ARDS_RUN_1)" -a -n "$(CLAMP_ARDS_RUN_2)" \
		-a -n "$(CLAMP_ARDS_RUN_1_PROVENANCE)" -a -n "$(CLAMP_ARDS_RUN_2_PROVENANCE)" || \
		(echo "Set CLAMP_ARDS_RUN_1, CLAMP_ARDS_RUN_2, and both provenance paths"; exit 2)
	uv run python scripts/import_clamp_ards_parity_runs.py \
		--fixture-root "$(CLAMP_ARDS_PENDING_ROOT)" \
		--run-1 "$(CLAMP_ARDS_RUN_1)" --run-2 "$(CLAMP_ARDS_RUN_2)" \
		--run-1-provenance "$(CLAMP_ARDS_RUN_1_PROVENANCE)" \
		--run-2-provenance "$(CLAMP_ARDS_RUN_2_PROVENANCE)"

.PHONY: clamp-ards-resources-audit
clamp-ards-resources-audit:
	uv run python scripts/audit_clamp_ards_resources.py

.PHONY: clamp-ards-resources-public-audit
clamp-ards-resources-public-audit:
	uv run python scripts/audit_clamp_ards_resources.py --public-release

AMARAL_CONFIG ?= config/external_comparators/amaral_ards_diagnosis.example.yaml
UW_HANSO_CONFIG ?= config/external_comparators/uw_hanso.example.yaml
AFSHAR_CONFIG ?= config/external_comparators/afshar_text_svc.example.yaml
UW_HANSO_IMAGE ?= ards-nlp-uw-hanso:legacy
AFSHAR_IMAGE ?= ards-nlp-afshar-svc:legacy

.PHONY: comparator-source
comparator-source:
	uv run python scripts/prepare_comparator_source.py

.PHONY: comparator-source-smoke
comparator-source-smoke:
	uv run python scripts/prepare_comparator_smoke_input.py

.PHONY: comparator-existing
comparator-existing: comparator-source
	uv run python scripts/normalize_existing_comparators.py

.PHONY: comparator-clamp-python
comparator-clamp-python: comparator-source clamp-ards-parity-restricted
	uv run python scripts/normalize_clamp_python_comparator.py --config config/config.yaml

.PHONY: comparator-amaral-fetch
comparator-amaral-fetch:
	uv run python scripts/fetch_amaral_comparator.py --config "$(AMARAL_CONFIG)"

.PHONY: comparator-amaral-runtime
comparator-amaral-runtime:
	uv run python scripts/setup_amaral_runtime.py --config "$(AMARAL_CONFIG)"

.PHONY: comparator-amaral-verify
comparator-amaral-verify: comparator-amaral-fetch
	uv run python scripts/verify_amaral_comparator_resources.py --config "$(AMARAL_CONFIG)"

.PHONY: comparator-amaral-smoke
comparator-amaral-smoke: comparator-source-smoke comparator-amaral-runtime comparator-amaral-verify
	uv run python scripts/run_amaral_comparator.py \
		--config "$(AMARAL_CONFIG)" \
		--input-packet artifacts/restricted/comparators/smoke/input.jsonl.gz \
		--input-manifest artifacts/restricted/comparators/smoke/manifest.parquet \
		--runner-output-dir artifacts/restricted/comparators/smoke/amaral/runner \
		--prediction-output data/derived/comparators/smoke/amaral_predictions.parquet \
		--artifact-dir artifacts/comparators/smoke/amaral \
		--anchor-expected tests/fixtures/comparators/amaral_anchor_expected.json

.PHONY: comparator-amaral
comparator-amaral: comparator-source comparator-amaral-smoke
	uv run python scripts/run_amaral_comparator.py --config "$(AMARAL_CONFIG)"

.PHONY: comparator-amaral-benchmark
comparator-amaral-benchmark:
	uv run python scripts/benchmark_comparators.py \
		--predictions data/derived/comparators/amaral_bilateral_infiltrates_predictions.parquet \
		--status artifacts/comparators/amaral/status.json \
		--out-dir artifacts/comparators/amaral/silver

.PHONY: comparator-uw-hanso-fetch
comparator-uw-hanso-fetch:
	uv run python scripts/fetch_uw_hanso_comparator.py --config "$(UW_HANSO_CONFIG)"

.PHONY: comparator-uw-hanso-verify
comparator-uw-hanso-verify: comparator-uw-hanso-fetch
	uv run python scripts/verify_uw_hanso_resources.py --config "$(UW_HANSO_CONFIG)"

.PHONY: comparator-uw-hanso-runtime
comparator-uw-hanso-runtime:
	docker build --platform linux/amd64 -t "$(UW_HANSO_IMAGE)" environments/uw_hanso

.PHONY: comparator-uw-hanso-smoke
comparator-uw-hanso-smoke: comparator-source-smoke comparator-uw-hanso-verify
	uv run python scripts/run_uw_hanso_comparator.py \
		--config "$(UW_HANSO_CONFIG)" \
		--smoke-only

.PHONY: comparator-uw-hanso
comparator-uw-hanso: comparator-source comparator-source-smoke comparator-uw-hanso-verify
	uv run python scripts/run_uw_hanso_comparator.py --config "$(UW_HANSO_CONFIG)"

.PHONY: comparator-uw-hanso-benchmark
comparator-uw-hanso-benchmark:
	uv run python scripts/benchmark_comparators.py \
		--predictions data/derived/comparators/uw_hanso_predictions.parquet \
		--status artifacts/comparators/uw_hanso/status.json \
		--out-dir artifacts/comparators/uw_hanso/silver

.PHONY: comparator-afshar-fetch
comparator-afshar-fetch:
	uv run python scripts/fetch_afshar_comparator.py --config "$(AFSHAR_CONFIG)"

.PHONY: comparator-afshar-inspect
comparator-afshar-inspect: comparator-afshar-fetch
	uv run python scripts/inspect_afshar_pickle_artifacts.py --config "$(AFSHAR_CONFIG)"

.PHONY: comparator-afshar-verify
comparator-afshar-verify: comparator-afshar-inspect

.PHONY: comparator-afshar-runtime
comparator-afshar-runtime:
	docker build --platform linux/amd64 -t "$(AFSHAR_IMAGE)" environments/afshar_svc

.PHONY: comparator-afshar-smoke
comparator-afshar-smoke: comparator-afshar-verify
	uv run python scripts/run_afshar_comparator.py \
		--config "$(AFSHAR_CONFIG)" \
		--smoke-only

.PHONY: comparator-afshar
comparator-afshar: comparator-source comparator-afshar-verify
	uv run python scripts/run_afshar_comparator.py --config "$(AFSHAR_CONFIG)"

.PHONY: comparator-afshar-benchmark
comparator-afshar-benchmark:
	uv run python scripts/benchmark_comparators.py \
		--predictions data/derived/comparators/afshar_text_svc_predictions.parquet \
		--status artifacts/comparators/afshar/status.json \
		--out-dir artifacts/comparators/afshar/silver

.PHONY: comparator-status
comparator-status:
	uv run python scripts/report_comparator_status.py

.PHONY: comparator-benchmark
comparator-benchmark: comparator-existing comparator-clamp-python comparator-amaral
	uv run python scripts/benchmark_comparators.py

.PHONY: comparator-snapshot
comparator-snapshot:
	uv run python scripts/write_comparator_snapshot.py

.PHONY: comparators-ready
comparators-ready: comparator-existing comparator-clamp-python comparator-amaral comparator-uw-hanso-verify comparator-afshar-inspect comparator-benchmark

.PHONY: comparator-clean-inputs
comparator-clean-inputs:
	rm -f artifacts/restricted/comparators/mimic_cxr_comparator_input.jsonl.gz
	rm -f artifacts/restricted/comparators/mimic_cxr_comparator_manifest.parquet

.PHONY: clean
clean:
	@rm -rf .pytest_cache .ruff_cache __pycache__ */__pycache__ src/*/__pycache__
	@rm -rf dist build .venv
