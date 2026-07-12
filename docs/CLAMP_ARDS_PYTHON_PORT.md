# ARDS CLAMP Python Compatibility Port

This workflow is a deterministic Python mirror of the legacy Dan Knox ARDS CLAMP project. It is a
compatibility baseline, not a clinically improved classifier and not a human gold standard.

## Scope and frozen oracle

The implementation loads the authorized custom phenotype from a packaged MIT JSON specification
attributed to Dan Knox. It reads only three separately licensed external files from
`ARDS_CLAMP_PROJECT_DIR`, `--project-dir`, or ignored `data/external/clamp_ards_project/`:
`defaultAbbrs.txt`, `defaultNegexDict.txt`, and `defaultTokenRule.txt`. All 23 original export paths,
sizes, and hashes remain fingerprint-only provenance in `config/clamp_ards_resource_manifest.json`.
The excluded export is not a runtime fallback. Archive size, checksum, member counts, and known
provenance gaps are frozen in `config/clamp_ards_oracle_manifest.json`. The local restricted oracle
is:

```text
artifacts/restricted/clamp_ards/oracle/ARDS_CLAMP_Output_full.zip
SHA-256 753ad353a2ab043124348194dfdd0c0dd5328860be28c6566c8a5e8ef1292cf8
```

The archive contains 227,835 TXT/XMI pairs, is about 1.7 GB compressed, and would require roughly
10 GB when expanded. All tooling streams ZIP members and must not extract it. The compact TXT-only
archive remains the authoritative final-entity source for the restricted parity comparison. XMI is
used only to characterize source text, sentence spans, token spans, and final UIMA annotations.

Both archives and every derived row-level result are ignored and restricted. Frozen resource keys
always use POSIX separators. The batch summary records raw-byte hashes only for the three files it
actually reads, plus the packaged phenotype version and hash.

## Pipeline contract

The Python mirror keeps input text immutable and runs these stages in legacy order:

1. Sentence segmentation that trims whitespace, ends spans at newlines, handles configured
   abbreviation-period cases, and preserves terminal punctuation.
2. Offset-preserving tokenization from the external delimiter configuration. Internal processing
   uses Python code-point indices so source slicing remains exact; public/final offsets are mapped
   to Java/UIMA UTF-16 code units. Punctuation counts as `BaseToken` gaps; `2week` splits while
   `O2`, `POD1`, and `C7` do not.
3. POS compatibility no-op. The excluded legacy POS model hash is retained as provenance, but the
   model is not distributed or loaded because no downstream dictionary or rule references POS.
   Re-running the no-op pipeline over all 227,835 restricted reports produced the exact 80,908
   CLAMP entities and document outputs, which is the empirical output-invariance check.
4. Case-insensitive 23-entry dictionary lookup from the authorized packaged specification with a
   local classic Porter compatibility stemmer.
5. Directional NegEx-like assertion using the external cue dictionary, sentence boundaries,
   conjunction terminators, and the configured 11-token scope.
6. Imperative execution of the authorized bidirectional promotion rules with a five-token maximum,
   followed by removal of unpromoted `Morphology` and `location` annotations.

The final entity span is the second dictionary annotation relabeled to `ARDS`; no combined span is
created. Duplicate annotations remain a multiset and receive a zero-based `duplicate_occurrence`
only when written to Parquet.

Intermediate compatibility is deliberately not claimed before the licensed synthetic runs. The
restricted XMI contains no occurrence of the configured `&apos;s` no-split string, so its standalone
and attached token behavior remains unresolved. The 463-case packet also contains isolated
no-period split-pattern and inline section-header probes. If genuine CLAMP annotations differ, the
tokenizer/segmenter and the runtime-required resource boundary must be updated from that evidence,
then both synthetic and complete restricted parity must be rerun.

`pulmonary edema`, `congestive heart failure`, and literal `ARDS` are direct legacy positives. The
output therefore represents the historical imaging proxy, not a literal ARDS diagnosis.

## Run the Python mirror

The default restricted input is the existing common comparator JSONL packet:

```bash
export ARDS_CLAMP_PROJECT_DIR=/path/to/licensed/ards-project
make clamp-ards-python
```

The runner also accepts CSV, compressed CSV, Parquet, JSONL, and compressed JSONL with configurable
ID/text columns:

```bash
uv run python scripts/run_python_clamp_ards.py \
  --input path/to/input.parquet \
  --id-col clamp_doc_id \
  --text-col report_text \
  --id-prefix '' \
  --entity-output artifacts/restricted/clamp_ards/python/custom/example_entities.parquet \
  --prediction-output artifacts/restricted/clamp_ards/python/custom/example_predictions.parquet \
  --summary-output artifacts/restricted/clamp_ards/python/custom/example_summary.json
```

Custom, fixture, limited, or alternate-project runs require all three explicit output paths and
cannot target the configured full-corpus artifacts.

Outputs are written atomically:

```text
data/derived/clamp_ards/clamp_python_entities.parquet
data/derived/clamp_ards/clamp_python_predictions.parquet
artifacts/restricted/clamp_ards/python/batch_summary.json
```

The entity table contains mention text and is restricted. Its `start` and `end` columns use
half-open UTF-16 code-unit offsets, matching CLAMP/UIMA; the batch summary and Parquet schema record
that convention. The prediction table contains only the document ID, evaluability status, binary
label, entity count, and source-text SHA-256.

## Strict parity

Run the full local comparison with:

```bash
make clamp-ards-parity-restricted
```

The comparator rejects entity document IDs without same-side prediction rows, checks document
membership, exact entity fields, multiset multiplicity, and document entity count/status/label.
Teacher `parsed`/`parsed_empty` states map to `evaluable`; missing, duplicate, and parse-error states
map to `non_evaluable`. Same-span entities are canonically paired for field checks. Because the
complete restricted comparison already demonstrated stable order, the restricted target requires
raw row order as well. Entity, document, and order mismatches are written to separate typed Parquet
ledgers; JSON and Markdown summaries contain aggregates only. The command returns exit status 1 for
any required mismatch and writes only hashes—not entity substrings—to its mismatch ledgers. The
summary also records SHA-256 values for all four compared Parquet files, binding downstream
normalization to the exact prediction bytes that passed parity.

All public sentence, token, and entity spans use half-open UTF-16 code-unit coordinates. Each span
also retains comparison-excluded Python source bounds so `covered_text()` remains exact. Reverse
mapping rejects offsets outside the source and offsets inside a surrogate pair.

The deterministic synthetic scaffold can be checked without claiming legacy parity:

```bash
make clamp-ards-parity-fixture-validate
```

It contains 463 exact-byte inputs covering all 23 dictionary rows in lower/title/upper/mixed case,
all 240 assertion cues, tokenizer delimiters and no-split behavior, Ruta direction/gap/order cases,
and sentence/input edge cases including CRLF, empty, Unicode, and more than 500 tokens. Its expected
directories deliberately contain only `PENDING` and a README; no expected span is hand-authored.

`make clamp-ards-parity-fixtures` is strict: it fails until two outputs from the exact licensed
legacy CLAMP project have been imported and both PHI and redistribution reviews are approved.
Linux and Windows CI generate and validate a temporary pending scaffold using independently
authored public resources. The `fixture-status` job succeeds only after that scaffold validates and
reports its pending maturity explicitly. The public resource audit is always a hard gate. Once the
genuine fixture is complete, strict validation requires exact sentence, token, final-entity,
multiplicity, document-label, and empirically demonstrated-order parity.

The importer requires CLAMP 1.6.6, valid UTC run chronology, identical runtime/project/resource
protocol, exact sentence/token/entity multisets, and complete TXT/XMI inventories. It records TXT
and XMI order stability independently. Stable TXT order becomes a required final-output field;
unstable order is still reported but cannot be treated as a semantic difference. Complete fixture
validation also regenerates the 463-case coverage matrix from the frozen resources and requires the
importer-origin project/resource/run provenance contract.

## Characterization

Restricted XMI characterization established exact token offsets over all 597,188 tokens in the
first 5,000 documents and exact final entities in 5,000/5,000 documents. The sentence approximation
exactly reproduced all sentence spans in 4,952/5,000 sampled documents; the remaining differences
are abbreviation decisions made by CLAMP's learned detector and did not change final entities in
that sample.

On the first 20,000 restricted documents, the deterministic mirror reproduced all 20,000 complete
final-entity sequences and document label/count/status results. Characterization established two
non-obvious compatibility rules without document-ID exceptions: pseudo-negation precedence depends
on lexical context before the cue, and the configured hyphenated `multi-focal` dictionary row is
inert in all 27 corpus occurrences.

Under the fingerprinted restricted resource set, the final full-corpus run passed with
227,835/227,835 exact entity-multiset documents, 80,908/80,908 entities, and zero required or
output-order differences. Independent reproduction requires separately licensed resources. The
generated restricted summary remains local; the identifier-free aggregate result is in
`docs/CLAMP_ARDS_PYTHON_PARITY.md`.

## Governance boundary

Repository code and the authorized re-expressed phenotype specification are MIT licensed. The
three default resources remain separately licensed and external; the other 20 export files remain
fingerprint-only provenance. See `docs/CLAMP_ARDS_RESOURCE_LEDGER.csv`,
`docs/CLAMP_ARDS_RESOURCE_REVIEW.md`, and `docs/THIRD_PARTY_NOTICES.md`.
