# Decisions

Use this file to record decisions that are hard to infer from the code alone.

## Template

### YYYY-MM-DD: Decision title

**Context:**

**Decision:**

**Alternatives considered:**

**Consequences:**

### 2026-06-21: MIMIC-CXR bilateral opacity benchmark v1 scope

**Context:** The repository now implements a MIMIC-CXR external development benchmark for report-semantics NLP around current bilateral pulmonary opacities/infiltrates.

**Decision:** V1 builds ingestion helpers, BigQuery derived tables, strict and sensitive silver labels, subtype flags, QA tables, deterministic manual-review exports, and a model-development extract. V1 does not include a full model bake-off.

**Alternatives considered:** Include baseline model training immediately; include full bake-off immediately; require completed MIMIC manual review before v1 acceptance.

**Consequences:** The dataset interface is ready for downstream modeling. Later silver-label baseline modeling is documented separately from a future full model bake-off and human-review adjudication.

### 2026-06-21: Full report is primary silver-rule text scope

**Context:** MIMIC reports contain findings, impression, addenda, and variable templates. The benchmark needs one primary text field for silver label construction while retaining alternate scopes for sensitivity checks.

**Decision:** Use full report text as the primary v1 silver-rule matching surface. Retain `target_text_impression_findings` and `target_text_impression_fallback` in output tables.

**Alternatives considered:** Impression+findings as primary; impression fallback as primary.

**Consequences:** V1 favors recall over section-restricted precision and stores enough fields to quantify scope sensitivity later.

### 2026-06-21: Automated MIMIC-wide labels are silver labels

**Context:** MIMIC-CXR-JPG labels, MIMIC-wide RadGraph annotations, and regex rules are not human-adjudicated for this exact bilateral-opacity construct.

**Decision:** Name automated MIMIC-wide labels and tables as silver reference outputs. Reserve gold naming for future manually reviewed MIMIC subsets.

**Alternatives considered:** Calling the automated benchmark a gold standard.

**Consequences:** Table names and documentation distinguish weak automated development labels from human-reviewed external or local validation references.

### 2026-06-26: Silver-label modeling may proceed before gold adjudication

**Context:** The benchmark now has a BigQuery `model_development_extract` and manual-review packet, but human adjudication is not yet complete.

**Decision:** Add reproducible baseline-modeling infrastructure that trains and evaluates against the existing automated silver labels while human annotation proceeds as a separate validation track. Splits are deterministic at the subject level to prevent patient leakage across train, validation, and test sets.

**Alternatives considered:** Block all modeling until manual review is complete; train baselines directly from raw MIMIC/RadGraph inputs.

**Consequences:** Exploratory modeling, feature debugging, and weak-label stress tests can start now, but outputs remain silver-label benchmarks and cannot be described as gold-standard performance.

### 2026-06-26: Gold annotation path is scaffolded but not materialized

**Context:** The manual-review sample exists locally, but completed human review is not yet available.

**Decision:** Add validation and text-stripped export helpers for future completed review CSVs. Do not create `gold_*` tables or outputs until reviewed labels exist.

**Alternatives considered:** Create empty gold tables now; defer all annotation-ingestion code until review is complete.

**Consequences:** The future reviewed subset has a reproducible ingestion path, while current MIMIC-wide outputs remain clearly silver-only.

### 2026-06-28: Probabilistic human-reference evaluation is additive

**Context:** The benchmark now has automated silver-label modeling, but physician review should preserve uncertainty and support separate report-only and image-only judgments.

**Decision:** Add an additive probabilistic evaluation layer with long-form rater ratings, case-level continuous reference probabilities, and model benchmarking against report-only and image-only targets. Keep report-only probability as the primary report-NLP target and image-only probability as the end-to-end clinical/image target.

**Alternatives considered:** Collapse physician ratings into binary labels only; replace the existing silver-label modeling workflow; use image-only review as the report-NLP target.

**Consequences:** Continuous model calibration and threshold behavior can be evaluated without weakening the existing `silver_*` naming boundary. Binary thresholds and kappa remain secondary summaries.

### 2026-06-29: ARDS CLAMP Stage 1 is an input handoff only

**Context:** The repository now needs a reproducible bridge from the MIMIC-CXR model-development
extract to the legacy `clamp_ARDS` project.

**Decision:** Add a Stage 1 workflow that syncs the tracked CLAMP project into a local CLAMP
workspace and exports one selected report text per `.txt` file. The workflow does not run CLAMP,
parse CLAMP output, train models, or alter the legacy ARDS CLAMP rules.

**Alternatives considered:** Run CLAMP from this repo; parse CLAMP output in the same ticket; copy
report metadata into CLAMP text files; commit a local CLAMP workspace.

**Consequences:** CLAMP execution stays a manual handoff and raw report text remains outside git.
Inputs are prepared on the MIMIC-enabled machine, only the CLAMP project/input packet is transferred
to the CLAMP machine, and returned CLAMP outputs are parsed/merged back on the MIMIC-enabled
machine in a later Stage 2. The repo-local workspace is staging only; descriptor paths are rendered
for the configured CLAMP-machine runtime project path.

### 2026-06-29: ARDS CLAMP Stage 2 creates teacher outputs, not gold labels

**Context:** Returned ARDS CLAMP outputs need to become reproducible comparison data for future
Python-port parity work and weak benchmark diagnostics.

**Decision:** Parse returned CLAMP outputs into ignored entity-level and document-level
`clamp_legacy` teacher artifacts. Compare CLAMP predictions with automated silver labels for
diagnostics only. Do not create `gold_*` outputs or clinical accuracy claims from CLAMP outputs.

**Consequences:** CLAMP entity text and row-level predictions stay local/ignored. Aggregate
CLAMP-vs-silver summaries may be used to guide implementation and parity work, but human-reviewed
reference standards remain the only path to gold labels.

### 2026-07-10: CLAMP tabular TXT is the Stage 2 source

**Context:** The returned CLAMP package contains paired TXT and XMI files for every Stage 1 report.
The TXT files contain final named-entity rows; XMI primarily preserves full UIMA state for CLAMP
visualization and debugging.

**Decision:** Build and parse a compact TXT-only archive without extracting individual files. Use
header-only TXT files as valid negatives and `Semantic == ARDS` rows as the positive teacher
signal. Keep XMI outside the working pipeline.

**Consequences:** Stage 2 avoids roughly 10 GB of expanded XMI while retaining every field required
for the legacy teacher benchmark. TXT packets and entity mentions remain restricted and ignored.

### 2026-07-09: REDCap pilot evaluation is continuous and agreement-focused

**Context:** The approximately 50-case pilot uses three physicians and separate image-only and
report-only 0-100 probability ratings. The current REDCap fields do not contain a rater identifier.

**Decision:** Require one explicitly mapped CSV export per rater, use `id_accession` as the local
case key, and analyze continuous ratings as the primary measurements. Report ICC(2,1), ICC(2,k),
pairwise differences, case-level disagreement, and image-report alignment. Do not add binary kappa,
model benchmarking, calibration, CLAMP, or timing estimates to this pilot notebook.

### 2026-07-10: Annotation planning uses transparent scenario grids

**Context:** Pilot agreement can inform reviewer allocation and validation planning, but it cannot
by itself determine a sufficient model-training sample or annotation duration.

**Decision:** Add an aggregate-only planning report with configurable prevalence, expected binary
performance, Wilson interval width, reliability, overlap, disagreement, and fixed retraining
workload assumptions. Person-ratings are primary workload outputs. Reviewer-hours require explicit
minutes per task, and retraining case counts are examples rather than sufficiency claims.

**Alternatives considered:** Produce one headline sample size; require all assumptions before any
output; infer annotation time; treat pilot ICC as a model-training sample-size calculation.

**Consequences:** Collaborators can compare designs without hiding assumptions. Continuous
probabilistic evaluation remains primary, while binary validation precision is a secondary planning
view.

### 2026-07-12: Public preview uses an external CLAMP resource boundary

**Context:** Code, aggregate metrics, and the authorized custom phenotype design are shareable.
Restricted MIMIC/REDCap artifacts and default CLAMP resources must remain external.

**Decision:** Publish v0.3.0 from a clean root. Exclude the complete exported CLAMP project and
re-express the authorized dictionary and rule semantics as MIT JSON attributed to Dan Knox. Require
only three separately licensed default resources at runtime and retain export hashes for audits.

**Consequences:** Public synthetic workflows run without licensed resources. Exact CLAMP
reproduction and restricted workflows still require each user's approved access and exact external
resource bytes.

**Alternatives considered:** Infer raters from row order or timestamps; use a single overwritten
wide export; make binary labels primary; implement the broader model-evaluation issue in the same
notebook.

**Consequences:** Inter-rater analysis requires separately preserved physician ratings. The pilot
report estimates reproducibility and assessment alignment but does not create a gold standard or
measure NLP accuracy.

### 2026-07-10: External comparators use a common gate-aware contract

**Context:** Published models differ in text scope, dependencies, licenses, artifact availability,
and native decision rules. Directly mixing their outputs would obscure these differences.

**Decision:** Generate canonical predictions for every available comparator using stable MIMIC
case IDs and fixed subject-level splits. Run external code and pickle artifacts only in isolated,
pinned environments. Preserve native labels and continuous scores, evaluate validation/test only,
and record unavailable models as explicit blocked statuses.

**Consequences:** Amaral can run as the primary external bilateral-infiltrates comparator. UW waits
for author-provided weights. Afshar remains a full-ARDS exploratory model outside the primary
ranking and cannot receive MIMIC text without permission. All current comparisons remain
silver-label engineering diagnostics.

Focused-text external variants use the stored impression fallback only when parsed
impression/findings is blank and record that substitution in a text-free manifest field. Models
whose intended target is not bilateral opacity are written to separate exploratory metric tables,
not the primary ranking.

### 2026-07-11: The first Python CLAMP replacement is a deterministic compatibility mirror

**Context:** The 23-term ARDS CLAMP project has a restricted 227,835-document oracle, but its
sentence detector and assertion component expose implementation-specific behavior not fully
defined by the exported resources.

**Decision:** Implement the first port with Python span dataclasses, a configured offset scanner,
classic Porter-compatible stemming, custom directional assertion logic, and imperative Ruta rule
passes. Do not use spaCy, medSpaCy, or another generic clinical NLP engine as the parity authority.
Freeze every tracked component hash and keep the full XMI/TXT oracle compressed and ignored.

**Alternatives considered:** Reimplement with spaCy/medSpaCy defaults; extract the complete XMI
archive; hard-code document-specific exceptions for inconsistent assertion outputs; call aggregate
agreement sufficient for strict acceptance.

**Consequences:** The mirror is auditable and inexpensive while preserving immutable offsets and
duplicates. It achieved exact final-output parity on all 227,835 restricted documents and 80,908
entities; sampled sentence-intermediate differences remain explicitly documented and must be
resolved against the genuine synthetic CLAMP fixture. The strict comparator exits nonzero for every
required mismatch. Public completion also requires that non-PHI CLAMP-generated fixture and
CLAMP-resource redistribution clearance.

### 2026-07-11: MIMIC benchmarking and Intermountain annotation are separate tracks

**Context:** MIMIC provides a complete external engineering corpus with automated labels, while
the three-rater physician pilot uses local Intermountain images and reports to plan human
validation and possible retraining.

**Decision:** Keep MIMIC comparator metrics explicitly silver-only. Register the exact Python CLAMP
port as `compatibility_mirror`, report it separately from independent comparators, and require its
metrics to equal legacy CLAMP. Treat Intermountain annotation as the future human reference track.
Document, but do not yet implement, conversion from the pilot schema to the generic probabilistic
benchmark schema.

**Consequences:** The combined MIMIC catalog contains ten available models/controls without
double-counting the Python mirror as independent evidence. No MIMIC gold standard is implied.
Physician agreement and workload planning can proceed before an internal prediction-evaluation
adapter is built.
