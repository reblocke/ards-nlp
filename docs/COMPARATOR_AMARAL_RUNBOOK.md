# Amaral/Morales bilateral-infiltrates comparator

The primary external comparator uses the pretrained XGBoost bilateral-infiltrates model from
`amarallab/ARDS_diagnosis` at commit `6154ac32e16dd9497a466351582603e1c1095a05`.

## Security and license boundary

The upstream source is GPL-2.0 and remains under ignored `data/external/`. The pretrained model and
vectorizer are pickles and load only inside the dedicated `environments/amaral` subprocess after
commit and SHA-256 verification. The primary process never unpickles them.

The vectorizer references the pinned upstream `custom_functions.tokenizer_better`; the isolated
runtime imports that module from the verified external clone. Published MIMIC segmentation calls
the pinned upstream segmentation functions. Notebook constants are extracted into an ignored
runtime JSON file rather than copied into tracked source.

## Commands

```bash
make comparator-source
make comparator-amaral-fetch
make comparator-amaral-runtime
make comparator-amaral-verify
make comparator-amaral-smoke
make comparator-amaral
make comparator-amaral-benchmark
```

The smoke run uses seven synthetic reports, isolated smoke paths, and a tracked no-PHI golden
output covering scores, labels, retained-statement counts, and preprocessing hashes. The full run
emits two model variants: published preprocessing as primary and raw full text as sensitivity
analysis. Both use the fixed 0.5 threshold and preserve the continuous positive-class probability.

No threshold tuning, retraining, or full Berlin-definition pipeline is included.

`--mode published_mimic_preprocessing` and `--mode raw_text_direct` are debugging interfaces only.
A single-mode run must provide explicit noncanonical runner, prediction, and artifact paths so it
cannot replace the full two-mode output or status.
