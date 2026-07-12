# External comparator bakeoff

## Purpose

The comparator layer runs published or legacy report-text models against the same 227,835
MIMIC-CXR studies and fixed subject-level splits. Silver-label results are engineering diagnostics,
not clinical accuracy or human gold-standard performance.

## Data flow

`make comparator-source` joins the local processed report table to the text-free modeling extract.
The join must contain exactly 227,835 unique studies with no missing keys or split overlap. It
writes one gzip JSONL packet containing full-report and impression/findings text, a text-free
manifest, and a text-free silver reference table. When parsed impression/findings is blank, the
focused scope uses the existing impression/findings/last-paragraph/full-report fallback and records
`impression_findings_fallback_used` in the manifest. The current corpus uses this fallback for
11,109 studies.

Restricted packet:

```text
artifacts/restricted/comparators/mimic_cxr_comparator_input.jsonl.gz
```

Canonical predictions:

```text
data/derived/comparators/*_predictions.parquet
```

Aggregate outputs:

```text
artifacts/comparators/combined/
```

The tracked aggregate-only team snapshot is `docs/COMPARATOR_SNAPSHOT_V1.md`. Refresh it after a
verified combined run with `make comparator-snapshot`.

`silver_metrics.csv` and `silver_strata.csv` contain the primary bilateral-opacity comparisons.
Any future full-ARDS or otherwise mismatched-target result is written separately to
`mismatched_target_exploratory_metrics.csv` and
`mismatched_target_exploratory_strata.csv`.

The combined benchmark uses an explicit artifact registry. It fails rather than consuming an old
prediction file when the corresponding external comparator is currently blocked, and it requires
external prediction `run_id` values to match the current available status manifest.

All paths above are ignored. Prediction tables must not contain report text or source paths.

## Ready workflow

Run the currently executable comparators and gated audits in sequence:

```bash
make comparators-ready
make comparator-status
```

This runs the existing CLAMP and internal baseline normalization, the Amaral synthetic smoke and
full MIMIC inference, UW resource verification, Afshar static inspection, and the combined
benchmark. UW and Afshar blocked states are successful audit outcomes; they are not fabricated
prediction results.

Current verified local status:

| Comparator | Status |
|---|---|
| Amaral published preprocessing | 227,835 predictions; available |
| Amaral raw-text sensitivity | 227,835 predictions; available |
| Dan Knox legacy CLAMP teacher | 227,835 normalized predictions; available |
| Python CLAMP compatibility mirror | 227,835 predictions; exact legacy parity; available |
| Internal silver baselines/controls | 253,071 normalized holdout predictions; available |
| UW HANSO | blocked: weights, expected checksums, and terms unavailable |
| Afshar text SVC | blocked: no license or internal-use permission documented |

## Benchmark rules

- Generate external-model predictions for all 227,835 studies.
- Calculate metrics only on `validation` and `test`.
- Evaluate external and legacy models against both strict and sensitive silver labels.
- Evaluate strict/sensitive silver-trained baselines only against their intended task.
- Preserve each model's native label rule; do not tune thresholds on validation or test.
- Require every supplied prediction row to join the canonical reference and preserve its original
  subject-level split assignment.
- Keep silver-derived controls separate from independent external comparators.
- Keep Afshar out of the bilateral-opacity ranking because its documented target is full ARDS.
- Keep `clamp_python_compatibility` in the `compatibility_mirror` role. It must exactly reproduce
  legacy CLAMP metrics and is reported separately rather than counted as independent evidence.

Run the Python compatibility path with:

```bash
make comparator-clamp-python
```

This target uses the Python prediction and parity-summary paths from `config/config.yaml`, reruns
strict restricted parity, verifies the prediction-file SHA-256 against that summary, and then writes
`data/derived/comparators/clamp_python_compatibility_predictions.parquet`. The combined benchmark
writes its metrics to `compatibility_mirror_metrics.csv` and keeps them out of the primary
`silver_metrics.csv` ranking.

Expected holdout label counts:

| Task | Validation | Test |
|---|---:|---:|
| Strict | 19,754 | 20,702 |
| Sensitive | 21,526 | 22,375 |

## Cleanup

The common report packet is reproducible from ignored local Parquet files. After every required
external run finishes, it may be removed with `make comparator-clean-inputs`. Keep text-free
predictions, manifests, checksums, and aggregate metrics.
