# ARDS CLAMP Stage 2 Output Runbook

This runbook recombines returned legacy ARDS CLAMP output files with the Stage 1 input manifest. It
does not run CLAMP and does not create human gold-standard labels.

## Inputs

Required local inputs:

```text
artifacts/restricted/clamp_ards/input_manifest.csv
ARDS CLAMP Output.zip
```

CLAMP emits paired `.txt` and `.xmi` files. For this benchmark, the tab-delimited `.txt` export is
authoritative because it contains the final named-entity fields needed for the teacher label:
`Start`, `End`, `Semantic`, `CUI`, `Assertion`, and `Entity`. XMI is not needed unless results must
be reopened in CLAMP or inspected at the UIMA annotation level.

Build the compact packet without extracting individual files:

```bash
make clamp-ards-output-packet CLAMP_OUTPUT_SOURCE_ARCHIVE="ARDS CLAMP Output.zip"
```

This creates:

```text
artifacts/restricted/clamp_ards/incoming/ARDS_CLAMP_Output_txt_only.zip
artifacts/restricted/clamp_ards/clamp_output_packet_summary.json
```

Packet creation fails on duplicate IDs or any mismatch with `input_manifest.csv`. It records source
and packet checksums and excludes every XMI member. The existing directory parser remains available
through `scripts/parse_clamp_ards_outputs.py --clamp-output-dir` for other tabular exports.

## Parse Returned CLAMP Outputs

```bash
make clamp-ards-parse
```

Outputs:

```text
data/derived/clamp_ards/clamp_legacy_entities.parquet
data/derived/clamp_ards/clamp_legacy_predictions.parquet
data/derived/clamp_ards/clamp_legacy_predictions_for_probabilistic_benchmark.parquet
artifacts/restricted/clamp_ards/clamp_output_audit.csv
artifacts/restricted/clamp_ards/clamp_teacher_summary.json
artifacts/restricted/clamp_ards/clamp_teacher_summary.md
```

The archive parser streams members without extraction and replaces final artifacts only after the
complete archive parses successfully. Header-only TXT files are valid `parsed_empty` negatives. A
document is positive when at least one final entity has `semantic_tag == "ARDS"`. Entity mentions
remain only in the ignored entity table; prediction and audit outputs contain no entity text.
The normalized entity schema always includes `attribute`; legacy TXT exports that omit it receive
an explicit null so exact parity never treats a missing schema field as equivalent to a null value.

## Benchmark Against Silver Labels

```bash
make clamp-ards-teacher-benchmark
```

Outputs:

```text
artifacts/clamp_ards/teacher_benchmark/clamp_vs_silver_metrics.csv
artifacts/clamp_ards/teacher_benchmark/clamp_vs_silver_metrics.json
artifacts/clamp_ards/teacher_benchmark/clamp_vs_silver_strata.csv
artifacts/clamp_ards/teacher_benchmark/clamp_vs_silver_summary.md
```

These are CLAMP-vs-automated-silver-label diagnostics. They are not clinical accuracy estimates and
not human gold-standard performance. The benchmark compares `clamp_legacy` predictions against
`strict_bilateral_opacity_label` and `sensitive_bilateral_opacity_label`, with strata for
`silver_label_source`, `manual_review_priority`, and `qa_flags_present`.

## Acceptance Checks

After parsing:

```bash
python - <<'PY'
import json
from pathlib import Path

summary = json.loads(Path("artifacts/restricted/clamp_ards/clamp_teacher_summary.json").read_text())
print(summary)
PY
```

For the current returned packet, confirm:

- 227,835 matched and evaluable documents;
- 80,908 entity rows;
- 51,924 positive and 175,911 negative documents;
- zero missing, unexpected, duplicate, or failed documents;
- no CLAMP output files, parsed entities, report text, or row-level predictions are staged for git.

The combined TXT/XMI ZIP is now the restricted Python-port characterization oracle at
`artifacts/restricted/clamp_ards/oracle/ARDS_CLAMP_Output_full.zip`. Keep it compressed, ignored,
and local; do not extract its roughly 10 GB expanded member tree. Keep the compact TXT-only archive
and its checksum summary ignored and local as the final-entity parsing source.
