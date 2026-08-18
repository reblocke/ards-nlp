# ARDS CLAMP Python Parity Result

## Result

The table below is historical evidence for the `v0.2.0` pre-optimization implementation. Under the
fingerprinted restricted resource set, that implementation exactly reproduced the legacy CLAMP
oracle on the complete 227,835-document MIMIC-CXR corpus. Independent reproduction requires
separately licensed resources; the public synthetic oracle is still pending.

### Indexed-matcher result

On 2026-08-17, the indexed `v0.3.0` working tree based on
`b197d4f14a5880158625994a86bd6d0fb3e2af41` ran the complete restricted corpus twice. Both runs
passed the frozen legacy oracle with `--require-order`, and their ordered outputs were byte-identical:

- 227,835/227,835 exact entity documents and 80,908/80,908 entities;
- zero missing, unexpected, field, multiplicity, count, status, label, or order differences;
- entity Parquet SHA-256
  `d3cac1f1a2704445a76f522102ed44518d148d2dacac4e1f4a9431a8ae38b4f7` in both runs;
- prediction Parquet SHA-256
  `ece6496da355ad23cd1c665ba77776c22c33dec9b6671ff465822677c3f300f4` in both runs.

The input packet SHA-256 remained
`531af96dda7cead0cb5b4ae2c721995ca68de2c0b437cc2452c4bc96bac1e7c5`; resource hashes and the
phenotype hash validated against their frozen manifests. The exact dirty execution tree is bound by
source fingerprint `3078d8b19be51a3dfb77f92d01d52838cdb70590139026738064ba025e27b8ed`:
the fingerprint covers the base tree, scoped binary tracked diff, and execution-relevant untracked
source bytes. The reference-only memory-baseline correction did not alter production behavior, but
this fresh pair recertifies the shared provenance. It is persisted with the restricted batch,
parity, determinism, and ordered-file hashes in ignored
`artifacts/restricted/clamp_ards/python/indexed_validation_memory_baseline/indexed_validation_provenance.json`.

This renews restricted-corpus compatibility acceptance for the indexed implementation. Public
fixture maturity remains separately pending until the genuine licensed synthetic runs and named
reviews are complete.

| Strict comparison | Result |
|---|---:|
| Expected documents | 227,835 |
| Actual documents | 227,835 |
| Exact entity-multiset documents | 227,835 |
| Expected entities | 80,908 |
| Actual entities | 80,908 |
| Missing/unexpected documents | 0 / 0 |
| Missing/unexpected entities | 0 / 0 |
| Field mismatches | 0 |
| Multiplicity mismatches | 0 |
| Output-order differences | 0 |
| Document count/status/label mismatches | 0 / 0 / 0 |
| Expected/actual positive documents | 51,924 / 51,924 |
| Strict status | PASS |

The comparison fields were `start`, `end`, `entity_text`, `semantic_tag`, `assertion`, `cui`,
`attribute`, and derived duplicate occurrence. The generated mismatch ledger contains only its
header and no restricted entity text. Entity offsets use CLAMP/UIMA UTF-16 code units. Both entity
tables had zero document IDs absent from their corresponding prediction tables; same-span field
comparison used canonical entity ordering. The legacy TXT export has no attribute column, so the
normalizer emits an explicit null `attribute`; omission of the normalized field is a schema error.

## Run provenance

- Date: 2026-07-11.
- Implementation version at the restricted run: `v0.2.0`.
- Python: 3.11.11 on arm64 macOS.
- Input packet SHA-256: `531af96dda7cead0cb5b4ae2c721995ca68de2c0b437cc2452c4bc96bac1e7c5`.
- Full TXT/XMI oracle SHA-256:
  `753ad353a2ab043124348194dfdd0c0dd5328860be28c6566c8a5e8ef1292cf8`.
- Resource hashes: `config/clamp_ards_resource_manifest.json`.
- Oracle size/member manifest: `config/clamp_ards_oracle_manifest.json`.
- Restricted generated provenance:
  `artifacts/restricted/clamp_ards/python/batch_summary.json`.
- Restricted strict result:
  `artifacts/restricted/clamp_ards/python/parity_summary.json`.

The full run was streamed from compressed JSONL and wrote about 10 MB of Python Parquet outputs. The
1.7 GB oracle remained compressed; no roughly 10 GB XMI member tree was extracted.

## XMI characterization

The restricted 5,000-document XMI characterization produced:

- 5,000/5,000 identical source strings;
- 597,188/597,188 exact token spans;
- 4,952/5,000 exact complete sentence-span sequences;
- 5,000/5,000 exact complete final-entity sequences.

The 48 learned-detector sentence-sequence differences did not affect final outputs in the sample,
and the complete final-entity parity run proves output invariance of the explicit POS no-op and the
sentence approximation on this corpus.

## Remaining reproducibility work

Restricted-corpus compatibility acceptance is complete for both historical `v0.2.0` and the
fingerprinted indexed `v0.3.0` working tree described above. Public independent reproduction is
still incomplete:

1. A 463-case deterministic non-PHI candidate scaffold now exists, but its expected annotations are
   intentionally absent until it is run twice through the exact licensed CLAMP project with known
   build, OS, Java, export, timestamp, and checksum provenance. Manual PHI and output redistribution
   review must then be recorded before the fixture can be committed as complete.
2. The 23-file export is excluded. The public mirror requires only three separately licensed
   resources and therefore cannot reproduce the restricted result without those exact external
   bytes.

These qualifications do not alter either measured restricted entity/document parity result, but
they keep the independently reproducible public fixture gate pending.

## Comparator registration

The verified Python predictions are normalized as
`clamp_python_compatibility` with
`comparison_role = compatibility_mirror`. The combined MIMIC benchmark requires complete
227,835-study coverage and exact aggregate metric equality with `clamp_legacy`. Compatibility
metrics are reported separately and are not independent clinical or model-performance evidence.
