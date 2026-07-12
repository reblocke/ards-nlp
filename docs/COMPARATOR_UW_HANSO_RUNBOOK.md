# UW BioNLP HANSO comparator

UW HANSO explicitly models infiltrates as `none`, `present`, `unilateral`, or `bilateral`. The
primary score is `P(infiltrates == bilateral)` and the native binary decision is whether bilateral
is the multiclass argmax.

The public source does not contain `model/parameters.pkl` or `model/state_dict.pt`. Until both
files, their checksums, and terms of use are documented, verification writes
`blocked_missing_model_artifacts` and restricted inference does not run.

```bash
make comparator-uw-hanso-fetch
make comparator-uw-hanso-verify
```

After acquisition, copy `uw_hanso.example.yaml` to a local ignored config, record both SHA-256
values and terms of use, complete `UW_HANSO_MODEL_ACQUISITION.md`, and run:

```bash
make comparator-uw-hanso-runtime
make comparator-uw-hanso-verify UW_HANSO_CONFIG=<local-config>
make comparator-uw-hanso-smoke UW_HANSO_CONFIG=<local-config>
make comparator-uw-hanso UW_HANSO_CONFIG=<local-config>
make comparator-uw-hanso-benchmark
```

The full runner performs the checksum-verified synthetic inference before exposing any MIMIC text
to the container. Both model files, expected checksums, terms, Docker, and the synthetic smoke must
succeed before full inference proceeds.

The path-gated `HANSO Runtime` GitHub workflow builds the Linux amd64 image and imports the pinned
upstream `process` module plus the exact spaCy model without requiring weights. Actual inference
remains blocked until the two author-provided model files are available.

The adapter resets each batch, calls the probability API rather than the label-only upstream CLI,
and produces separate impression/findings and full-report variants. No hosted endpoint may receive
report text.
