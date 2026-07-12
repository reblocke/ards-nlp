# ARDS CLAMP synthetic golden-corpus runbook

This workflow turns the deterministic synthetic candidate into a genuine legacy-CLAMP oracle. It
never uses the restricted MIMIC archive and never commits raw XMI or licensed CLAMP resources.

## Current state

- Synthetic inputs: generated and structurally valid.
- Fixture lifecycle: `awaiting_legacy_runs`.
- Expected annotations: intentionally absent; no span has been hand-authored.
- Licensed Windows runs: pending.
- Manual non-PHI and output-redistribution reviews: pending.
- Local candidate path: `artifacts/restricted/clamp_ards/python/pending_fixture/` (ignored).
- `tests/fixtures/clamp_ards_parity/` remains ignored until finalization is approved.

## Validate and build the handoff

```bash
make clamp-ards-parity-fixture-prepare
make clamp-ards-parity-fixture-validate
make clamp-ards-python-smoke
make clamp-ards-parity-fixture-handoff
shasum -a 256 artifacts/restricted/clamp_ards/parity_handoff.zip
```

The ignored ZIP contains the separately licensed local project, 463 exact-byte inputs, the case
manifest, resource/project checksums, an embedded runbook, and a PowerShell provenance collector.
It does not contain the MIMIC oracle and must not be published. Handoff ZIP timestamps are fixed so
identical local source inputs produce an identical archive.

## Run the exact project twice on licensed Windows CLAMP

Follow `RUNBOOK.md` inside the handoff ZIP. In summary:

1. Record the version and build shown by the licensed CLAMP application.
2. Use the unmodified project commit `9f8c92fbbeb44645a1066be3510d4ab993995c1e`.
3. Run the 463 inputs into an empty output directory and preserve it as `run_1`.
4. Run the included PowerShell collector with exact timestamps, CLAMP actions or command, export
   settings, offset convention, and null convention. Describe the manual actions without a Windows
   username or machine-local path; unsafe local paths are rejected during import.
5. Clear the output directory, repeat without changing the project or inputs, and preserve `run_2`.
6. Return both directories (or separate ZIPs), both provenance JSON files, and their generated
   SHA-256 manifests.

Do not combine the runs. Do not rename output files. Empty or whitespace-only inputs are part of the
contract; a missing output for either remains a failed legacy run rather than an inferred result.

## Strict technical import

Set the four Make variables to ignored returned paths, then build an ignored review candidate:

```bash
make clamp-ards-parity-fixture-import \
  CLAMP_ARDS_RUN_1=artifacts/restricted/clamp_ards/returned/run_1 \
  CLAMP_ARDS_RUN_2=artifacts/restricted/clamp_ards/returned/run_2 \
  CLAMP_ARDS_RUN_1_PROVENANCE=artifacts/restricted/clamp_ards/returned/run_1_provenance.json \
  CLAMP_ARDS_RUN_2_PROVENANCE=artifacts/restricted/clamp_ards/returned/run_2_provenance.json
```

The import fails atomically for an incorrect project/resource hash; missing, extra, or duplicate
TXT/XMI; parse errors; Sofa/input differences; invalid UTF-16 offsets or covered text; TXT/XMI
final-entity field/multiplicity disagreement; or any sentence, token, or final-entity multiset
difference between the two runs. TXT and XMI ordering are compared separately and recorded. Stable
TXT order becomes a strict fixture-output requirement; unstable order remains reported but is not
treated as a semantic mismatch. A successful default import writes only an ignored review
candidate and does not create the public fixture tree.

The importer also requires each collector-generated `.SHA256SUMS` file beside its provenance JSON
and cross-checks it against the embedded output-file manifest. Use `--run-1-sha256s` and
`--run-2-sha256s` only if the returned manifests were stored under different ignored paths.

## Review and finalize

Review the candidate and record documentary evidence before finalization:

- a named reviewer and ISO-8601 date approving the inputs as non-PHI and non-MIMIC-derived;
- the authority and evidence approving redistribution of normalized synthetic CLAMP output;
- the separate file-level resource disposition in `docs/CLAMP_ARDS_RESOURCE_LEDGER.csv`.

Only after those reviews are approved, regenerate the identical scaffold at its public fixture
path and rerun the importer there with `--finalize` and all four non-placeholder review fields:

```bash
uv run python scripts/generate_clamp_ards_parity_fixture.py generate \
  --output tests/fixtures/clamp_ards_parity

uv run python scripts/import_clamp_ards_parity_runs.py \
  --fixture-root tests/fixtures/clamp_ards_parity \
  --run-1 artifacts/restricted/clamp_ards/returned/run_1 \
  --run-2 artifacts/restricted/clamp_ards/returned/run_2 \
  --run-1-provenance artifacts/restricted/clamp_ards/returned/run_1_provenance.json \
  --run-2-provenance artifacts/restricted/clamp_ards/returned/run_2_provenance.json \
  --finalize \
  --phi-reviewer '<reviewer>' \
  --phi-reviewed-at '<ISO-8601 date>' \
  --redistribution-authority '<authority>' \
  --redistribution-evidence '<Git-safe evidence reference>'
```

`--force` can replace only a directory carrying this generator's fixture provenance. It rejects
symlinks, arbitrary directories, paths that contain the repository root or resource manifest, and
paths that overlap the CLAMP project.

Finalization copies normalized per-case TSV and sentence/token/final-entity JSON into the fixture,
records each run's per-case TXT/XMI SHA-256 plus repeat-run/order evidence, regenerates
`SHA256SUMS`, and validates the complete lifecycle. Raw XMI and the second normalized run remain
ignored. The deterministic input-tree hash must match the reviewed ignored scaffold exactly; do not
copy or edit cases by hand.

The repository-root ignore rule for `tests/fixtures/clamp_ards_parity/` is a pre-approval safety
lock. After the finalized tree passes strict validation and both reviews are recorded, remove that
one ignore rule, run `make release-audit`, and stage only the normalized fixture files. Do not use a
broad forced add that could also capture returned XMI or either raw run directory.

## Completion gates

```bash
make clamp-ards-parity-fixtures
make clamp-ards-parity-restricted
make clamp-ards-resources-public-audit
make fmt
make lint
make test
make run
make release-audit
git diff --check
```

Normal CI generates a fresh pending scaffold on Linux and Windows. The `fixture-status` job passes
only after the pending scaffold validates and reports its incomplete maturity explicitly. The
`resource-audit` job is always enforced. A finalized golden-corpus commit is prohibited until every
command above passes from the intended committed tree and the named reviews are complete.
