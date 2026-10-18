# Caldonazzo 2026: CXR-CT Agreement and ARDS NLP Reference Standards

## Citation and Retrieval

- **Authors:** Jérémy Caldonazzo and colleagues.
- **Title:** Agreement between chest radiograph and computed tomography for the diagnosis of
  acute respiratory distress syndrome: A retrospective observational study.
- **Journal:** Annals of Intensive Care. 2026;16:100172. Published September 30, 2026.
- **DOI:** [10.1016/j.aicoj.2026.100172](https://doi.org/10.1016/j.aicoj.2026.100172).
- **PMID:** [42893020](https://pubmed.ncbi.nlm.nih.gov/42893020/).
- **Full text:** [PMC13675848](https://pmc.ncbi.nlm.nih.gov/articles/PMC13675848/).
- **Registry:** [NCT07400445](https://clinicaltrials.gov/study/NCT07400445).
- **Reviewed:** October 18, 2026. Main full text, both main tables, and the exclusion flow figure.
- **License:** The article identifies CC BY 4.0. This note contains original appraisal and project
  relevance; no article copy or figure is included.
- **Retrieval pointer:** The PMC page links the article PDF and supplementary `mmc1.docx`.
  Any local retrieval packet belongs under ignored `artifacts/literature/`.

## Why This Belongs in the Project

The paper supports separating three questions when interpreting this benchmark: how reliably
clinicians identify bilateral opacities on an image, whether the report conveys that finding, and
whether NLP recovers the report's meaning. A full clinical ARDS diagnosis adds further criteria.

The repository already makes this distinction. Preserve it when writing the manuscript or planning
additional validation. This paper supplies an external rationale for the existing design, not a
new source of training labels or a numerical correction for model performance.

## Evidence Anchors for Future Writing

### What Was Compared

This retrospective study selected 277 intubated, hypoxemic patients at one French medical ICU during
2015-2021. All met the non-imaging Berlin criteria on the CT day. Inclusion required clinically
obtained CT and a CXR within 24 hours before or after it; the closest CXR was used. The flow chart
starts with 336 patients and excludes 3 for missing data, 11 for PEEP below 5, 8 for cardiogenic
pulmonary edema, and 37 for no qualifying CXR. The broader hypoxemic population without CT was not
evaluated.

Two intensivists independently read CXRs: a fellow with five years and a senior with 20 years of
ICU experience. A radiologist with ten years of experience and a resident with two years interpreted
CTs by consensus. Readers were blinded to clinical history and clinical/biological data but told
that the other Berlin criteria were satisfied. This was image interpretation, not report NLP.

The imaging thresholds differed:

- **CT:** ground-glass opacity or consolidation occupying at least 25% of at least one right and
  one left quadrant; opacities fully explained by effusion, atelectasis, or nodules/masses were
  excluded.
- **CXR:** at least interstitial infiltrates, score 1 on the study's 0-3 scale, in at least one
  quadrant on each side. No corresponding 25% extent requirement was specified.

The reference was therefore a defined CT-consensus classification in clinically preselected
patients. It was not an independent biological gold standard for all ARDS or for report semantics.

### Main Results

CT classified 223/277 patients as positive and 54 as negative. Table 2 reports:

| Measure | Fellow CXR reader | Senior CXR reader |
| --- | --- | --- |
| Kappa against CT, 95% CI | 0.29 (0.17-0.42) | 0.44 (0.32-0.55) |
| Sensitivity, 95% CI | 80% (74-85) | 78% (72-84) |
| Specificity, 95% CI | 54% (40-67) | 76% (62-87) |
| Positive predictive value | 88% | 93% |
| Negative predictive value | 39% | 46% |
| Observed accuracy/agreement | 75% | 78% |

The reported CXR-positive totals, CT totals, and rounded metrics are consistent with reconstructed
TP/FN/FP/TN counts of **178/45/25/29** and **175/48/13/41**, respectively. These are reconstructed
counts, not published patient-level data. They imply that 45-48 of 223 CT-positive cases were missed.
Among CXR-positive cases, the reconstructed false-positive proportions are 25/203 (12.3%) and
13/188 (6.9%); the paper's 46% and 24% figures instead describe false-positive rates among the
54 CT-negative patients.

Inter-intensivist kappa on the same CXRs was 0.42 (0.30-0.54). Similar kappa magnitudes suggest a
contribution from reader variation. They do not establish equal sources of error, equivalence
between readers and modalities, or a performance ceiling for NLP.

Requiring alveolar rather than interstitial-or-alveolar CXR infiltrates reduced sensitivity to
35% and 43%, while specificity rose to 96% and 89%. Raising the CT extent threshold to 50% changed
CT positivity to 131/277 and changed the performance estimates. These are changes to the test or
reference definition, not improvements in the underlying imaging.

### Appraisal Limits and Subgroup Cautions

- **Selection and transportability:** everyone analyzed had both tests, but only CT-selected
  patients entered. CT indication and disease spectrum can affect performance; estimates do not
  identify accuracy in all intubated patients or in the project's MIMIC development population.
- **Combined measurement differences:** modality, specialty, individual reader versus consensus,
  and positivity threshold all changed. The design cannot isolate any one contribution. Two
  readers cannot establish a general effect of experience.
- **Reference uncertainty:** no independent CT-reader agreement was measured. CT extent and
  morphology do not alone establish the full clinical syndrome.
- **Timing:** tests could be up to 24 hours apart in either direction, with changing physiology
  or treatment. The main text reports timing-stratified results in Supplemental Table 1; those
  estimates need direct verification before quantitative reuse.
- **Precision and analysis:** specificity rests on 54 CT-negative cases. The study recruited 277
  of the planned 326 patients and registered in February 2026 after the source cohort ended.
  Treat numerous subgroup comparisons as exploratory and focus on intervals rather than achieved
  post hoc power or claims that similar subgroup estimates prove consistency.
- **Focal subgroup:** the article reports sensitivity of 67% and 54% in 45 focal CT-positive
  cases, versus 93% and 90% in 134 diffuse cases. The focal denominator and estimates do not
  reconcile as ordinary binary counts: 54% is unattainable when rounded from an integer numerator
  over 45. Both published focal confidence intervals also suggest a denominator discrepancy.
  Preserve the reported figures as provisional; do not silently replace 45 with another number.
  Verify a correction or obtain clarification before quoting exact focal rates in a manuscript.
- **Four-quadrant CXR subgroup:** selecting patients whose CXR already satisfies the bilateral
  positivity rule makes sensitivity conditional on a positive index test tautological. The
  authors also state all selected patients were CT-positive; that supports a within-subgroup
  positive predictive value observation, not a general claim of perfect screening sensitivity.
- **Clinical utility:** the study did not compare CT-guided versus usual-care strategies. More
  CT-positive labels do not establish that routine CT improves ventilation practice or outcomes.

## Application to the Existing Repository

1. **Keep the targets separate.** The [probabilistic evaluation runbook](../PROBABILISTIC_EVALUATION_RUNBOOK.md)
   names `report_only` as the primary NLP development target and `image_only` as the separate
   imaging-validity target. Neither alone diagnoses clinical ARDS. CT should not silently enter
   the current image-only reference.
2. **Preserve individual judgments and continuous ratings.** The
   [annotation pilot](../ANNOTATION_PILOT_EVALUATION.md) already requires three separate rater
   exports, analyzes image/report agreement separately, and retains disagreement. A mean physician
   probability summarizes judgments; it is not automatically a calibrated biological probability.
3. **Attribute discordance to the relevant comparison.** NLP versus report reviewers measures
   extraction performance. Report versus image reviewers measures information-source discordance.
   NLP versus image reviewers measures their combined pathway. Human-versus-human comparisons
   contextualize disagreement without removing model errors from the prespecified evaluation.
4. **Retain broad report semantics.** The sensitivity loss with an alveolar-only CXR criterion
   motivates checking subtle interstitial and basal findings during error review. It does not
   validate any particular NLP vocabulary change. Preserve the current strict/sensitive
   `silver_*` definitions pending a separate empirical decision.
5. **Use the correct uncertainty unit.** Keep repeated studies from each patient together in
   development/evaluation splits. If the local validation includes repeated accessions per
   patient, verify patient-level resampling for intervals before manuscript claims. Do not infer
   this property from a case-level bootstrap alone.

The pilot currently measures annotation agreement without NLP predictions. Its adapter to the
generic probabilistic benchmark is deferred in the inspected runbooks. Synthetic workflow
support is not evidence that a real local model-versus-human validation has been completed.

## Return to This Before the Manuscript

These are proposed checkpoints, not accepted changes to the pipeline or annotation protocol.

- **Reference-standard checkpoint:** document what each reader could see, reading order,
  blinding, any washout, timing, technical image quality, and how uncertainty and adjudication
  were handled. Verify the actual study procedures before describing them as independent.
- **Evaluation checkpoint:** complete and validate the deferred pilot-to-benchmark adapter,
  freeze the applicable target and predictions, and report report-only performance, image-only
  performance, human agreement, and report-image alignment on explicitly matched denominators.
- **Discordance checkpoint:** review subtle/basal findings, ambiguity, alternative explanations,
  and report omissions in the protected environment. Retain a representative validation sample;
  enriched error-review cases cannot supply unweighted population accuracy estimates.
- **Optional CT checkpoint:** if existing paired CT is available, plan a separate selected-subset
  analysis with documented CT indication, timing, independent ratings, and adjudication. Preserve
  the primary CXR cohort. Report what selection prevents generalizing. No CT acquisition is
  justified solely by this paper or by this repository note.
- **Citation checkpoint:** reconcile the focal subgroup denominator and inspect the supplemental
  timing table before quoting those subgroup estimates.

### Original Wording to Adapt When Writing

Automated extraction of bilateral pulmonary opacities from radiology reports depends on both the
information in the report and the reproducibility of the imaging assessment. Caldonazzo and
colleagues found fair-to-moderate agreement between intensivist-interpreted chest radiographs and
a radiologist-consensus CT reference in a selected cohort of hypoxemic intubated patients. This
motivates evaluating report extraction separately from image-level validity and preserving
individual reviewer judgments. Agreement with report annotations should not be interpreted as
equivalent agreement with an imaging reference or as validation of the complete clinical ARDS
phenotype.

Use this citation in the introduction's measurement rationale, the methods' reference-standard
justification, and the discussion of residual disagreement. Insert the project's actual empirical
results only after the corresponding restricted analyses are complete.
