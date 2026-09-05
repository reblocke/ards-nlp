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

The isolated runtime pins setuptools 80.10.2 (`<81`) because Hyperopt 0.2.7 requires
`pkg_resources`; lift the pin only after replacing that dependency. This inference path neither
publishes an sdist nor relies on `MANIFEST.in` exclusions, so the packaging-only advisory fixed in
setuptools 83 does not apply to it.

NLTK 3.10.3 remains affected by `GHSA-8mgp-746c-j5xp` because no patched release is available. The
comparator uses only `word_tokenize` with a dedicated, operator-controlled `NLTK_DATA` directory
and never calls the affected model-artifact read/write APIs. Keep the advisory open and reassess
when upstream publishes a fix.

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
