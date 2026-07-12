# ARDS Continuous Subphenotyping

## Citation

- **Title:** Beyond binary: rethinking subphenotyping in ARDS as a continuous spectrum
- **Authors:** Prashant Nasa; Ken Kuljit S. Parhar; Ryuichi Nakayama
- **Year:** 2026
- **Journal:** Intensive Care Medicine
- **DOI:** 10.1007/s00134-026-08531-1
- **Article type:** Editorial/commentary
- **Local wiki note:** `ICM_2026_Nasa_Beyond_Binary_ARDS_Subphenotyping_Continuous_Spectrum.md`
- **Source PDF filename:** `ICM 2026 Nasa Beyond Binary ARDS Subphenotyping Continuous Spectrum.pdf`

The literature wiki and source PDF remain local and are not distributed with this repository.

## Why It Matters Here

This article is useful background for the benchmark's scientific framing. It supports keeping a
clear distinction between binary convenience labels and richer continuous or probabilistic disease
representations when discussing ARDS phenotype work.

For this repository, the most relevant implications are:

- keep MIMIC-wide automated outputs named `silver_*` unless a human-reviewed subset exists;
- treat `silver_bilateral_opacity_score` as a weak ranking/provenance signal, not a clinical
  disease-severity construct;
- report model calibration and threshold behavior alongside binary metrics when using silver labels;
- avoid implying that a binary bilateral-opacity NLP target captures the full ARDS phenotype.

## Caveats

This article is not a label source, not a gold-standard adjudication source, and not part of the
current MIMIC-CXR silver-label construction. It should inform interpretation and future scientific
discussion, not change the current `silver_*` definitions without a separate scoped decision.
