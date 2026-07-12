# Afshar text-SVC comparator audit

The Afshar repository contains a text SVC and TF-IDF vectorizer but no license file. Its associated
publication describes identification of the full ARDS phenotype from radiology reports, not the
bilateral-opacity imaging criterion.

Default workflow:

```bash
make comparator-afshar-fetch
make comparator-afshar-inspect
make comparator-afshar-verify
```

Static inspection uses `pickletools` and never loads the artifacts. It records protocol, referenced
classes, embedded scikit-learn version, checksums, and a synthetic anchor set. Current artifacts
were created with pickle protocol 3 and scikit-learn 0.19.0-era classes.

MIMIC execution remains blocked unless a local ignored config records explicit permission. Once
permission is documented, build the runtime and generate the no-PHI anchor predictions before
setting `anchor_review_status: passed`:

```bash
make comparator-afshar-runtime
make comparator-afshar-smoke AFSHAR_CONFIG=<local-config>
make comparator-afshar AFSHAR_CONFIG=<local-config>
make comparator-afshar-benchmark
```

Any result is named `afshar_text_svc_full_ards`, marked
`mismatched_target_exploratory`, and excluded from the primary bilateral-opacity ranking.
