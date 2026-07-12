# ARDS CLAMP Stage 1 Input Runbook

This runbook prepares CXR report text files for a separately licensed local ARDS CLAMP project. It
only creates CLAMP-ready input packets and handoff metadata. It does not run CLAMP, parse CLAMP
output, train a model, or modify the local project.

The intended workflow is transfer-first: prepare inputs on this MIMIC-enabled machine, transfer the
ARDS CLAMP project plus `.txt` input files to the CLAMP machine, run CLAMP there, transfer CLAMP
outputs back, and parse/merge outputs back on this machine in a future Stage 2.

## Configuration

By default, Stage 1 writes the CLAMP workspace under ignored repository-local restricted artifacts:

```yaml
clamp_ards:
  clamp_project_root: artifacts/restricted/clamp_ards/workspace
  project_name: ARDS
  project_source_dir: data/external/clamp_ards_project
  project_live_dir: artifacts/restricted/clamp_ards/workspace/ARDS
  runtime_project_dir: C:/ClampWin_1.6.6/workspace/ARDS
  input_dir: artifacts/restricted/clamp_ards/workspace/ARDS/Data/Input
  output_dir: artifacts/restricted/clamp_ards/workspace/ARDS/Data/Output
  restricted_artifact_dir: artifacts/restricted/clamp_ards
```

Do not commit CLAMP input files, CLAMP output files, or row-level manifests. The default
repo-local workspace is ignored by git. If you override `project_live_dir`, `input_dir`, or
`output_dir`, use absolute paths outside the repository or keep them under
`artifacts/restricted/clamp_ards/`.

Sync/export commands fail if operational paths still contain `YOUR_LOCAL_CLAMP_WORKSPACE_ROOT`.
They also fail for relative paths or arbitrary repo-local paths outside
`artifacts/restricted/clamp_ards/`. This keeps report-text files out of tracked repository paths.

## Sync The CLAMP Project

Place the separately licensed project under ignored `data/external/clamp_ards_project/`, or set
`project_source_dir` to another approved local path. Sync it into the restricted staging workspace
before preparing inputs:

```bash
make clamp-ards-sync
```

The sync copies project files to local `project_live_dir`, skips `Data/Input`, `Data/Output`,
archives, and generated/restricted files, and normalizes descriptor paths to
`runtime_project_dir`, which is the path where the transferred ARDS project should live on the CLAMP
machine. The default runtime path is `C:/ClampWin_1.6.6/workspace/ARDS`. If the CLAMP machine uses a
different workspace path, configure `clamp_ards.runtime_project_dir` or pass `--runtime-project-dir`
before syncing. The sync refuses to overwrite changed staged files unless `--overwrite` is supplied:

```bash
uv run python scripts/sync_clamp_ards_project.py --overwrite
```

Use `--dry-run` to preview the copy set without mutating the live workspace.

## Export Inputs From BigQuery

The default source is:

```text
<project>.<dataset>.model_development_extract
```

The default selected report text column is `primary_target_text`, and the default CLAMP document ID
is `s{study_id}`.

Dry-run a small packet first:

```bash
uv run python scripts/export_clamp_ards_inputs.py --limit 25 --dry-run
```

Write the first 25 inputs after confirming the configured CLAMP input directory is correct:

```bash
uv run python scripts/export_clamp_ards_inputs.py --limit 25 --clear-existing-inputs
```

Run the configured full export:

```bash
make clamp-ards-inputs
```

Default behavior is conservative:

- existing target `.txt` files cause failure unless `--overwrite` is provided;
- existing inputs/outputs are not cleared unless explicitly requested;
- stale files can be moved under `Archive/YYYYMMDD_HHMMSS/` with `--archive-cleared-files`;
- unsafe Windows filenames are skipped and counted in the manifest;
- duplicate CLAMP document IDs fail the export before report text is written.

## Export Inputs From A Local File

Local mode accepts CSV, compressed CSV, text-delimited CSV, or Parquet sources. Provide either an
ID column or a document ID template:

```bash
uv run python scripts/export_clamp_ards_inputs.py \
  --source-file data/derived/local_clamp_export.parquet \
  --id-col clamp_doc_id \
  --text-col primary_target_text \
  --dry-run
```

Template mode can derive IDs from columns:

```bash
uv run python scripts/export_clamp_ards_inputs.py \
  --source-file data/derived/local_clamp_export.csv.gz \
  --doc-id-template 's{study_id}' \
  --text-col primary_target_text
```

## Text Writing Contract

Each CLAMP input file contains exactly the selected report text:

- no headers;
- no metadata;
- no line-break normalization;
- no `<BR>` replacement;
- no chunking;
- no truncation.

The exporter only adds metadata to ignored restricted artifacts, not to the `.txt` report files.

## Transfer To And From The CLAMP Machine

Transfer only the minimum CLAMP packet through an approved MIMIC-compliant route:

- the synced ARDS CLAMP project;
- the generated `.txt` input files;
- `input_manifest.csv` and handoff summaries.

Do not transfer the full BigQuery table, model-development extract, raw MIMIC files, RadGraph JSON,
or generated Parquet caches to the CLAMP machine. After CLAMP runs, transfer only the CLAMP output
directory back to this MIMIC-enabled machine. Future Stage 2 parsing should merge returned outputs
here using `input_manifest.csv` and `clamp_doc_id`.

Copy the staged `ARDS` project so it lands at the configured `runtime_project_dir` on the CLAMP
machine. The staged descriptors are written for that runtime path, not for this Mac's repo-local
`artifacts/restricted/...` path.

## Outputs

Restricted local outputs are written under `artifacts/restricted/clamp_ards/` by default:

```text
input_manifest.csv
input_summary.json
input_summary.md
project_sync_summary.json
NEXT_STEP_RUN_CLAMP.md
```

The manifest excludes raw report text. The CLAMP `.txt` inputs live only in the configured local
CLAMP input directory.

## Manual Acceptance Check

After a 25-report trial export:

```bash
find "$(python - <<'PY'
from ards_cxr_benchmark.config import load_default_config
print(load_default_config().clamp_ards.input_dir)
PY
)" -maxdepth 1 -name '*.txt' | wc -l
```

Confirm the count matches the manifest rows with `export_status == written`. Then transfer the
packet to the CLAMP machine, open the `ARDS` project in CLAMP, run the ARDS pipeline manually, and
return the output directory to this machine. Then continue with
`docs/CLAMP_ARDS_STAGE2_OUTPUT_RUNBOOK.md`.
