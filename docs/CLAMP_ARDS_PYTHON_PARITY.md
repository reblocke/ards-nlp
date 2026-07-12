# ARDS CLAMP Python Parity Result

## Result

Under the fingerprinted restricted resource set, the deterministic Python compatibility mirror
exactly reproduced the legacy CLAMP oracle on the complete 227,835-document MIMIC-CXR corpus.
Independent reproduction requires separately licensed resources; the public synthetic oracle is
still pending.

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

Restricted-corpus compatibility acceptance is complete. Public independent reproduction is not:

1. A 463-case deterministic non-PHI candidate scaffold now exists, but its expected annotations are
   intentionally absent until it is run twice through the exact licensed CLAMP project with known
   build, OS, Java, export, timestamp, and checksum provenance. Manual PHI and output redistribution
   review must then be recorded before the fixture can be committed as complete.
2. The 23-file export is excluded. The public mirror requires only three separately licensed
   resources and therefore cannot reproduce the restricted result without those exact external
   bytes.

These qualifications affect independent reproducibility and fixture maturity, not the measured
restricted entity/document parity above.

## Comparator registration

The verified Python predictions are normalized as `clamp_python_compatibility` with
`comparison_role = compatibility_mirror`. The combined MIMIC benchmark requires complete
227,835-study coverage and exact aggregate metric equality with `clamp_legacy`. Compatibility
metrics are reported separately and are not independent clinical or model-performance evidence.
