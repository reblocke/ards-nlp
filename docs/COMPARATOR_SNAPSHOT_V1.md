# MIMIC-CXR Comparator Snapshot V1

> Aggregate engineering diagnostics against automated silver labels. These are not human
> gold-standard or clinical-accuracy estimates.

## Corpus

- MIMIC-CXR studies in the canonical source: **227,835**
- Available prediction models/controls: **10**
- Metrics are calculated only on fixed subject-level validation and test splits.
- Report text, identifiers, and row-level predictions are not included in this snapshot.

## Comparator Status

| Comparator | Status | Reason |
|---|---|---|
| Afshar text SVC (full-ARDS target) | blocked_license_permission | no upstream license or explicit internal-use permission is documented |
| Amaral published preprocessing | available | full MIMIC prediction modes completed |
| Amaral raw-text sensitivity | available | canonical predictions present |
| Dan Knox legacy CLAMP | available | canonical predictions present |
| Python CLAMP compatibility mirror | available | canonical predictions present |
| silver sensitive silver score rule | available | canonical predictions present |
| silver sensitive structured logreg | available | canonical predictions present |
| silver sensitive tfidf logreg | available | canonical predictions present |
| silver strict silver score rule | available | canonical predictions present |
| silver strict structured logreg | available | canonical predictions present |
| silver strict tfidf logreg | available | canonical predictions present |
| UW HANSO | blocked_missing_model_artifacts | trained model artifacts are missing; expected checksums are not configured for parameters, state_dict; terms of use are not documented |

## Available Predictions

| Model | Role | Prediction rows | Unique cases | Validation | Test |
|---|---|---:|---:|---:|---:|
| Python CLAMP compatibility mirror | compatibility_mirror | 227,835 | 227,835 | 33,448 | 35,151 |
| Amaral published preprocessing | external_comparator | 227,835 | 227,835 | 33,448 | 35,151 |
| Amaral raw-text sensitivity | external_comparator | 227,835 | 227,835 | 33,448 | 35,151 |
| Dan Knox legacy CLAMP | legacy_teacher | 227,835 | 227,835 | 33,448 | 35,151 |
| silver sensitive silver score rule | silver_derived_control | 43,901 | 43,901 | 21,526 | 22,375 |
| silver sensitive structured logreg | silver_derived_control | 43,901 | 43,901 | 21,526 | 22,375 |
| silver strict silver score rule | silver_derived_control | 40,456 | 40,456 | 19,754 | 20,702 |
| silver strict structured logreg | silver_derived_control | 40,456 | 40,456 | 19,754 | 20,702 |
| silver sensitive tfidf logreg | trained_silver_baseline | 43,901 | 43,901 | 21,526 | 22,375 |
| silver strict tfidf logreg | trained_silver_baseline | 40,456 | 40,456 | 19,754 | 20,702 |

## Holdout Metrics

| Model | Task | Split | n | Prevalence | AUROC | Average precision | F1 | Sensitivity | Specificity | Brier |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Amaral published preprocessing | sensitive | test | 22,375 | 0.251 | 0.813 | 0.684 | 0.595 | 0.487 | 0.950 | 0.132 |
| Amaral published preprocessing | sensitive | validation | 21,526 | 0.254 | 0.806 | 0.679 | 0.586 | 0.473 | 0.952 | 0.137 |
| Amaral published preprocessing | strict | test | 20,702 | 0.188 | 0.881 | 0.728 | 0.666 | 0.607 | 0.950 | 0.090 |
| Amaral published preprocessing | strict | validation | 19,754 | 0.185 | 0.880 | 0.732 | 0.671 | 0.612 | 0.952 | 0.088 |
| Amaral raw-text sensitivity | sensitive | test | 22,375 | 0.251 | 0.799 | 0.660 | 0.562 | 0.451 | 0.949 | 0.139 |
| Amaral raw-text sensitivity | sensitive | validation | 21,526 | 0.254 | 0.793 | 0.653 | 0.552 | 0.438 | 0.949 | 0.143 |
| Amaral raw-text sensitivity | strict | test | 20,702 | 0.188 | 0.856 | 0.687 | 0.623 | 0.552 | 0.949 | 0.098 |
| Amaral raw-text sensitivity | strict | validation | 19,754 | 0.185 | 0.855 | 0.685 | 0.624 | 0.556 | 0.949 | 0.097 |
| Dan Knox legacy CLAMP | sensitive | test | 22,375 | 0.251 | 0.772 | 0.558 | 0.675 | 0.611 | 0.933 | 0.148 |
| Dan Knox legacy CLAMP | sensitive | validation | 21,526 | 0.254 | 0.759 | 0.546 | 0.657 | 0.581 | 0.936 | 0.154 |
| Dan Knox legacy CLAMP | strict | test | 20,702 | 0.188 | 0.851 | 0.603 | 0.748 | 0.770 | 0.933 | 0.098 |
| Dan Knox legacy CLAMP | strict | validation | 19,754 | 0.185 | 0.848 | 0.598 | 0.744 | 0.760 | 0.936 | 0.097 |
| silver sensitive tfidf logreg | sensitive | test | 22,375 | 0.251 | 0.994 | 0.984 | 0.938 | 0.963 | 0.970 | 0.027 |
| silver sensitive tfidf logreg | sensitive | validation | 21,526 | 0.254 | 0.995 | 0.986 | 0.940 | 0.960 | 0.972 | 0.027 |
| silver strict tfidf logreg | strict | test | 20,702 | 0.188 | 0.994 | 0.976 | 0.914 | 0.959 | 0.968 | 0.028 |
| silver strict tfidf logreg | strict | validation | 19,754 | 0.185 | 0.994 | 0.976 | 0.912 | 0.952 | 0.970 | 0.027 |

## Compatibility Mirrors

| Model | Task | Split | n | Prevalence | AUROC | Average precision | F1 | Sensitivity | Specificity | Brier |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Python CLAMP compatibility mirror | sensitive | test | 22,375 | 0.251 | 0.772 | 0.558 | 0.675 | 0.611 | 0.933 | 0.148 |
| Python CLAMP compatibility mirror | sensitive | validation | 21,526 | 0.254 | 0.759 | 0.546 | 0.657 | 0.581 | 0.936 | 0.154 |
| Python CLAMP compatibility mirror | strict | test | 20,702 | 0.188 | 0.851 | 0.603 | 0.748 | 0.770 | 0.933 | 0.098 |
| Python CLAMP compatibility mirror | strict | validation | 19,754 | 0.185 | 0.848 | 0.598 | 0.744 | 0.760 | 0.936 | 0.097 |

## Silver-Derived Controls

| Model | Task | Split | n | Prevalence | AUROC | Average precision | F1 | Sensitivity | Specificity | Brier |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| silver sensitive silver score rule | sensitive | test | 22,375 | 0.251 | 0.869 | 0.803 | 0.820 | 0.695 | 1.000 | 0.069 |
| silver sensitive silver score rule | sensitive | validation | 21,526 | 0.254 | 0.857 | 0.787 | 0.801 | 0.668 | 1.000 | 0.075 |
| silver sensitive structured logreg | sensitive | test | 22,375 | 0.251 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 |
| silver sensitive structured logreg | sensitive | validation | 21,526 | 0.254 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 |
| silver strict silver score rule | strict | test | 20,702 | 0.188 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.005 |
| silver strict silver score rule | strict | validation | 19,754 | 0.185 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.005 |
| silver strict structured logreg | strict | test | 20,702 | 0.188 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 |
| silver strict structured logreg | strict | validation | 19,754 | 0.185 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 |

## Interpretation

- `clamp_legacy` is the returned Dan Knox CLAMP workflow. The Python compatibility
  mirror has exact full-corpus parity and is not independent validation evidence.
- Amaral is an external pretrained comparator; the raw-text variant is a sensitivity
  analysis rather than a separate published model.
- TF-IDF models are trained on silver labels. Structured rules and score controls are
  label-derived checks and must not be interpreted as independent validation.
- UW HANSO and Afshar remain gate-controlled; blocked runs are reported explicitly
  and predictions are never fabricated.
- Final model claims require the physician report-only and image-only reference data.
