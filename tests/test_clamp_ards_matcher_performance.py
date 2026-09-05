from __future__ import annotations

import hashlib
import importlib
import json
import random
import subprocess
import sys
from pathlib import Path
from types import MappingProxyType

import pyarrow.parquet as pq
import pytest

from ards_cxr_benchmark.clamp_ards import pipeline as clamp_pipeline
from ards_cxr_benchmark.clamp_ards.assertion import ClampNegExAssertion, CueSpan
from ards_cxr_benchmark.clamp_ards.batch import (
    ENTITY_SCHEMA,
    PREDICTION_SCHEMA,
    iter_fixture_documents,
    run_clamp_ards_batch,
)
from ards_cxr_benchmark.clamp_ards.dictionary import ClampDictionaryMatcher
from ards_cxr_benchmark.clamp_ards.pipeline import PipelineTrace, load_legacy_mirror
from ards_cxr_benchmark.clamp_ards.resources import AssertionCue, DictionaryEntry
from ards_cxr_benchmark.clamp_ards.span_index import DocumentSpanIndex
from ards_cxr_benchmark.clamp_ards.stemming import PorterCompatibilityStemmer
from ards_cxr_benchmark.clamp_ards.types import EntitySpan, SentenceSpan, TokenSpan


def test_reference_mirror_never_constructs_indexed_matchers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The naive benchmark baseline must not include transient index allocations."""

    resources = load_legacy_mirror().resources

    def fail_if_constructed(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise AssertionError("The naive reference constructed an indexed matcher")

    # Patch the source classes before the reference module imports anything, then
    # patch the pipeline aliases as well. The latter makes a reintroduced
    # ``super().__init__`` fail even if the source classes are later rebound.
    monkeypatch.setattr(ClampDictionaryMatcher, "__init__", fail_if_constructed)
    monkeypatch.setattr(ClampNegExAssertion, "__init__", fail_if_constructed)
    monkeypatch.setattr(clamp_pipeline, "ClampDictionaryMatcher", fail_if_constructed)
    monkeypatch.setattr(clamp_pipeline, "ClampNegExAssertion", fail_if_constructed)

    monkeypatch.syspath_prepend(str(Path("scripts").resolve()))
    reference_module = importlib.import_module("_clamp_ards_matcher_reference")

    mirror = reference_module.ReferenceLegacyARDSClampMirror(resources)
    assert isinstance(mirror.run("No pulmonary edema."), list)


def _reference_dictionary_matches(
    *,
    entries: tuple[DictionaryEntry, ...],
    excluded_terms: frozenset[str],
    stemmer: PorterCompatibilityStemmer,
    tokenizer: object,
    text: str,
    tokens: list[TokenSpan],
    sentences: list[SentenceSpan],
) -> list[EntitySpan]:
    """Pre-index implementation retained only as an exact regression oracle."""

    compiled = []
    for entry in entries:
        if entry.term in excluded_terms:
            continue
        entry_tokens = tokenizer.tokenize(entry.term)
        values = tuple(stemmer.stem(token.covered_text(entry.term)) for token in entry_tokens)
        compiled.append((entry, values))

    token_values = [stemmer.stem(token.covered_text(text)) for token in tokens]
    entities: list[EntitySpan] = []
    for start_index in range(len(tokens)):
        for entry, values in compiled:
            length = len(values)
            if tuple(token_values[start_index : start_index + length]) != values:
                continue
            end_index = start_index + length - 1
            if end_index >= len(tokens):
                continue
            if not any(
                tokens[start_index].start >= sentence.start
                and tokens[end_index].end <= sentence.end
                for sentence in sentences
            ):
                continue
            entities.append(
                EntitySpan(
                    start=tokens[start_index].start,
                    end=tokens[end_index].end,
                    semantic_tag=entry.semantic_tag,
                    dictionary_term=entry.term,
                    dictionary_index=entry.index,
                )
            )
    return sorted(entities, key=lambda entity: (entity.start, entity.end, entity.dictionary_index))


class _ReferenceAssertion:
    """Pre-index assertion implementation; never imported by production code."""

    def __init__(
        self,
        cues: tuple[AssertionCue, ...],
        tokenizer: object,
        *,
        scope_tokens: int,
    ) -> None:
        self._scope_tokens = scope_tokens
        self._compiled = tuple(
            (
                cue,
                tuple(
                    token.covered_text(cue.phrase).casefold()
                    for token in tokenizer.tokenize(cue.phrase)
                ),
            )
            for cue in cues
        )

    def find_cues(self, text: str, tokens: list[TokenSpan]) -> list[CueSpan]:
        values = [token.covered_text(text).casefold() for token in tokens]
        matches: list[CueSpan] = []
        for start_index in range(len(tokens)):
            for cue, cue_values in self._compiled:
                length = len(cue_values)
                if tuple(values[start_index : start_index + length]) != cue_values:
                    continue
                end_index = start_index + length - 1
                if end_index >= len(tokens):
                    continue
                matches.append(
                    CueSpan(
                        start=tokens[start_index].start,
                        end=tokens[end_index].end,
                        phrase=cue.phrase,
                        category=cue.category,
                        index=cue.index,
                    )
                )
        return sorted(matches, key=lambda cue: (cue.start, cue.end, cue.index))

    def classify(
        self,
        text: str,
        sentences: list[SentenceSpan],
        tokens: list[TokenSpan],
        entities: list[EntitySpan],
    ) -> list[EntitySpan]:
        cues = self.find_cues(text, tokens)
        result: list[EntitySpan] = []
        for entity in entities:
            sentence = next(
                (
                    span
                    for span in sentences
                    if entity.start >= span.start and entity.end <= span.end
                ),
                None,
            )
            if sentence is None:
                result.append(entity)
                continue
            sentence_tokens = [
                token
                for token in tokens
                if token.start >= sentence.start and token.end <= sentence.end
            ]
            sentence_cues = [
                cue for cue in cues if cue.start >= sentence.start and cue.end <= sentence.end
            ]
            cue = self._negating_cue(text, entity, sentence_tokens, sentence_cues)
            result.append(
                entity.with_assertion("absent", cue.phrase)
                if cue is not None
                else entity.with_assertion("present")
            )
        return result

    def _negating_cue(
        self,
        text: str,
        entity: EntitySpan,
        tokens: list[TokenSpan],
        cues: list[CueSpan],
    ) -> CueSpan | None:
        conjunctions = [cue for cue in cues if cue.category == "conjunctions"]
        pre = [
            cue
            for cue in cues
            if cue.category == "negPhrases"
            and cue.end <= entity.start
            and self._gap(cue.end, entity.start, tokens) <= self._scope_tokens
            and not self._terminated(cue.end, entity.start, conjunctions)
            and not self._blocked_by_pseudo(text, cue, entity, cues, tokens)
        ]
        if pre:
            return max(pre, key=lambda cue: (cue.end, cue.start, -cue.index))

        post = [
            cue
            for cue in cues
            if cue.category == "postNegPhrases"
            and cue.start >= entity.end
            and self._gap(entity.end, cue.start, tokens) <= self._scope_tokens
            and not self._terminated(entity.end, cue.start, conjunctions)
        ]
        if post:
            return min(post, key=lambda cue: (cue.start, cue.end, cue.index))
        return None

    @classmethod
    def _blocked_by_pseudo(
        cls,
        text: str,
        negation: CueSpan,
        entity: EntitySpan,
        cues: list[CueSpan],
        tokens: list[TokenSpan],
    ) -> bool:
        sentence_start = tokens[0].start if tokens else 0
        for pseudo in cues:
            if pseudo.category != "pseNegPhrases":
                continue
            honored = pseudo.phrase.casefold().startswith("not ") or any(
                char.isalpha() for char in text[sentence_start : pseudo.start]
            )
            overlaps = cls._overlaps(negation, pseudo)
            intervenes = negation.end <= pseudo.start and pseudo.end <= entity.start
            if honored and (overlaps or intervenes):
                return True
        return False

    @staticmethod
    def _gap(left_end: int, right_start: int, tokens: list[TokenSpan]) -> int:
        return sum(token.start >= left_end and token.end <= right_start for token in tokens)

    @staticmethod
    def _terminated(start: int, end: int, conjunctions: list[CueSpan]) -> bool:
        return any(cue.start >= start and cue.end <= end for cue in conjunctions)

    @staticmethod
    def _overlaps(left: CueSpan, right: CueSpan) -> bool:
        return left.start < right.end and right.start < left.end


def _reference_trace(text: str) -> PipelineTrace:
    mirror = load_legacy_mirror()
    tokens = mirror.tokenizer.tokenize(text)
    sentences = mirror.segmenter.segment(text, tokens)
    pos_tokens = mirror.pos.apply(tokens)
    dictionary_entities = _reference_dictionary_matches(
        entries=mirror.resources.dictionary,
        excluded_terms=mirror.resources.excluded_dictionary_terms,
        stemmer=PorterCompatibilityStemmer(),
        tokenizer=mirror.tokenizer,
        text=text,
        tokens=pos_tokens,
        sentences=sentences,
    )
    assertion = _ReferenceAssertion(
        mirror.resources.assertion_cues,
        mirror.tokenizer,
        scope_tokens=mirror.resources.assertion_scope_tokens,
    )
    asserted_entities = assertion.classify(text, sentences, pos_tokens, dictionary_entities)
    final_entities = mirror.ruta.apply(sentences, pos_tokens, asserted_entities)
    return PipelineTrace(
        sentences=tuple(sentences),
        tokens=tuple(tokens),
        dictionary_entities=tuple(dictionary_entities),
        asserted_entities=tuple(asserted_entities),
        final_entities=tuple(final_entities),
    )


def _tokenize_and_segment(text: str) -> tuple[list[TokenSpan], list[SentenceSpan]]:
    mirror = load_legacy_mirror()
    tokens = mirror.tokenizer.tokenize(text)
    return tokens, mirror.segmenter.segment(text, tokens)


def test_span_index_handles_empty_sentences_and_crossing_tokens() -> None:
    tokens = [
        TokenSpan(start=0, end=2, token_number=0),
        TokenSpan(start=2, end=7, token_number=1),
        TokenSpan(start=8, end=10, token_number=2),
    ]
    sentences = [
        SentenceSpan(start=0, end=3, sentence_number=0),
        SentenceSpan(start=3, end=10, sentence_number=1),
        SentenceSpan(start=11, end=11, sentence_number=2),
    ]

    index = DocumentSpanIndex.build(tokens, sentences)

    assert index.token_to_sentence == (0, None, 1)
    assert index.sentence_token_bounds == ((0, 1), (2, 3), (3, 3))
    assert index.sentence_for_token_span(1, 1) is None
    assert index.sentence_for_token_span(2, 2) == 1
    assert index.sentence_for_span(11, 11) == 2


def test_span_index_token_gap_matches_reference_scan_at_all_boundaries() -> None:
    tokens = [
        TokenSpan(start=1, end=3, token_number=0),
        TokenSpan(start=4, end=5, token_number=1),
        TokenSpan(start=8, end=12, token_number=2),
    ]
    sentences = [SentenceSpan(start=0, end=12, sentence_number=0)]
    index = DocumentSpanIndex.build(tokens, sentences)
    boundaries = sorted({0, 1, 3, 4, 5, 8, 12, 13})

    for left_end in boundaries:
        for right_start in boundaries:
            if left_end > right_start:
                assert index.token_gap(left_end, right_start) == -1
                continue
            expected = sum(token.start >= left_end and token.end <= right_start for token in tokens)
            assert index.token_gap(left_end, right_start) == expected


def test_first_token_buckets_preserve_compiled_resource_order() -> None:
    mirror = load_legacy_mirror()
    entries = (
        DictionaryEntry("opacity", "Morphology", 30),
        DictionaryEntry("opacities", "Morphology", 10),
        DictionaryEntry("opacity", "Morphology", 20),
    )
    matcher = ClampDictionaryMatcher(
        entries,
        PorterCompatibilityStemmer(),
        mirror.tokenizer,
    )
    dictionary_buckets = matcher._entries_by_first_token
    assert isinstance(dictionary_buckets, MappingProxyType)
    dictionary_bucket = next(
        bucket for bucket in dictionary_buckets.values() if len(bucket) == len(entries)
    )
    assert [compiled.entry.index for compiled in dictionary_bucket] == [30, 10, 20]

    cues = (
        AssertionCue("no evidence", "negPhrases", 20),
        AssertionCue("no", "negPhrases", 5),
        AssertionCue("no change", "pseNegPhrases", 15),
    )
    assertion = ClampNegExAssertion(cues, mirror.tokenizer)
    cue_buckets = assertion._cues_by_first_token
    assert isinstance(cue_buckets, MappingProxyType)
    cue_bucket = next(bucket for bucket in cue_buckets.values() if len(bucket) == len(cues))
    assert [compiled.cue.index for compiled in cue_bucket] == [20, 5, 15]


def test_dictionary_overlap_final_token_and_cross_sentence_match_reference() -> None:
    mirror = load_legacy_mirror()
    entries = (
        DictionaryEntry("air", "Morphology", 8),
        DictionaryEntry("air space", "Morphology", 3),
        DictionaryEntry("space", "Morphology", 6),
        DictionaryEntry("air", "location", 1),
    )
    matcher = ClampDictionaryMatcher(entries, PorterCompatibilityStemmer(), mirror.tokenizer)
    text = "air space air"
    tokens, sentences = _tokenize_and_segment(text)
    expected = _reference_dictionary_matches(
        entries=entries,
        excluded_terms=frozenset(),
        stemmer=PorterCompatibilityStemmer(),
        tokenizer=mirror.tokenizer,
        text=text,
        tokens=tokens,
        sentences=sentences,
    )
    assert matcher.match(text, tokens, sentences) == expected
    assert [(entity.start, entity.end, entity.dictionary_index) for entity in expected] == [
        (0, 3, 1),
        (0, 3, 8),
        (0, 9, 3),
        (4, 9, 6),
        (10, 13, 1),
        (10, 13, 8),
    ]

    boundary_text = "air\nspace"
    boundary_tokens = mirror.tokenizer.tokenize(boundary_text)
    boundary_sentences = [
        SentenceSpan(start=0, end=3, sentence_number=0),
        SentenceSpan(start=4, end=9, sentence_number=1),
    ]
    actual = matcher.match(boundary_text, boundary_tokens, boundary_sentences)
    reference = _reference_dictionary_matches(
        entries=entries,
        excluded_terms=frozenset(),
        stemmer=PorterCompatibilityStemmer(),
        tokenizer=mirror.tokenizer,
        text=boundary_text,
        tokens=boundary_tokens,
        sentences=boundary_sentences,
    )
    assert actual == reference
    assert all(entity.dictionary_term != "air space" for entity in actual)


def test_all_dictionary_terms_match_reference() -> None:
    mirror = load_legacy_mirror()
    for entry in mirror.resources.dictionary:
        text = entry.term
        tokens, sentences = _tokenize_and_segment(text)
        assert mirror.dictionary.match(text, tokens, sentences) == _reference_dictionary_matches(
            entries=mirror.resources.dictionary,
            excluded_terms=mirror.resources.excluded_dictionary_terms,
            stemmer=PorterCompatibilityStemmer(),
            tokenizer=mirror.tokenizer,
            text=text,
            tokens=tokens,
            sentences=sentences,
        )


def test_all_assertion_cues_match_reference() -> None:
    mirror = load_legacy_mirror()
    reference = _ReferenceAssertion(
        mirror.resources.assertion_cues,
        mirror.tokenizer,
        scope_tokens=mirror.resources.assertion_scope_tokens,
    )
    for cue in mirror.resources.assertion_cues:
        text = cue.phrase
        tokens = mirror.tokenizer.tokenize(text)
        assert mirror.assertion.find_cues(text, tokens) == reference.find_cues(text, tokens)


@pytest.mark.parametrize(
    "text",
    [
        "No pulmonary edema.",
        "pulmonary edema is absent.",
        "No pulmonary edema but bilateral opacities.",
        "There is no change in pulmonary edema.",
        "not necessarily pulmonary edema.",
    ],
)
def test_assertion_scope_pseudo_and_conjunction_match_reference(text: str) -> None:
    mirror = load_legacy_mirror()
    tokens, sentences = _tokenize_and_segment(text)
    entities = mirror.dictionary.match(text, tokens, sentences)
    reference = _ReferenceAssertion(
        mirror.resources.assertion_cues,
        mirror.tokenizer,
        scope_tokens=mirror.resources.assertion_scope_tokens,
    )
    assert mirror.assertion.classify(text, sentences, tokens, entities) == reference.classify(
        text,
        sentences,
        tokens,
        entities,
    )


def test_entity_outside_sentences_remains_unclassified_as_in_reference() -> None:
    mirror = load_legacy_mirror()
    text = "ARDS"
    tokens = mirror.tokenizer.tokenize(text)
    entity = EntitySpan(start=0, end=4, semantic_tag="ARDS", dictionary_index=3)
    reference = _ReferenceAssertion(
        mirror.resources.assertion_cues,
        mirror.tokenizer,
        scope_tokens=mirror.resources.assertion_scope_tokens,
    )
    assert (
        mirror.assertion.classify(text, [], tokens, [entity])
        == reference.classify(
            text,
            [],
            tokens,
            [entity],
        )
        == [entity]
    )


def test_character_contained_entities_reset_present_without_token_alignment() -> None:
    mirror = load_legacy_mirror()
    text = "abc   xyz"
    tokens = [
        TokenSpan(start=0, end=3, token_number=0),
        TokenSpan(start=6, end=9, token_number=1),
    ]
    sentences = [
        SentenceSpan(start=0, end=3, sentence_number=0),
        SentenceSpan(start=3, end=6, sentence_number=1),
        SentenceSpan(start=6, end=9, sentence_number=2),
    ]
    entities = [
        EntitySpan(
            start=1,
            end=2,
            semantic_tag="ARDS",
            assertion="absent",
            assertion_cue="stale",
        ),
        EntitySpan(
            start=4,
            end=5,
            semantic_tag="ARDS",
            assertion="absent",
            assertion_cue="stale",
        ),
    ]
    reference = _ReferenceAssertion(
        mirror.resources.assertion_cues,
        mirror.tokenizer,
        scope_tokens=mirror.resources.assertion_scope_tokens,
    )

    actual = mirror.assertion.classify(text, sentences, tokens, entities)
    expected = reference.classify(text, sentences, tokens, entities)

    assert actual == expected
    assert [(entity.assertion, entity.assertion_cue) for entity in actual] == [
        ("present", None),
        ("present", None),
    ]


@pytest.mark.parametrize(
    "text",
    [
        "",
        "... -- !!!",
        "😀 🫁 ARDS; not necessarily pulmonary edema.",
        "No change in pulmonary edema, but diffuse bilateral opacities remain.",
        " ".join(["neutral"] * 1_000 + ["ARDS"]),
    ],
)
def test_empty_unicode_assertion_and_long_traces_match_reference(text: str) -> None:
    assert load_legacy_mirror().trace(text) == _reference_trace(text)


def test_seeded_random_full_traces_match_reference() -> None:
    randomizer = random.Random(20260811)
    vocabulary = [
        "neutral",
        "ARDS",
        "pulmonary",
        "edema",
        "bilateral",
        "opacities",
        "no",
        "change",
        "not",
        "necessarily",
        "but",
        ".",
        ",",
        "😀",
        "🫁",
        "\n",
    ]
    for _ in range(64):
        text = " ".join(randomizer.choice(vocabulary) for _ in range(randomizer.randrange(0, 80)))
        assert load_legacy_mirror().trace(text) == _reference_trace(text)


def test_generated_463_case_fixture_matches_reference(pending_clamp_fixture: Path) -> None:
    documents = list(iter_fixture_documents(pending_clamp_fixture))
    assert len(documents) == 463
    mirror = load_legacy_mirror()
    for case_id, text in documents:
        assert mirror.trace(text) == _reference_trace(text), case_id


def test_batch_is_deterministic_and_preserves_exact_schemas(tmp_path: Path) -> None:
    input_path = tmp_path / "input.jsonl"
    rows = [
        {"doc": "one", "text": "bilateral infiltrates"},
        {"doc": "two", "text": "No pulmonary edema."},
        {"doc": "three", "text": "😀 ARDS"},
        {"doc": "four", "text": ""},
    ]
    input_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )

    outputs: list[tuple[Path, Path, object]] = []
    for run in (1, 2):
        entity_output = tmp_path / f"entities-{run}.parquet"
        prediction_output = tmp_path / f"predictions-{run}.parquet"
        summary = run_clamp_ards_batch(
            input_path=input_path,
            entity_output=entity_output,
            prediction_output=prediction_output,
            id_column="doc",
            text_column="text",
            batch_size=1,
            show_progress=False,
        )
        outputs.append((entity_output, prediction_output, summary))

    first_entities, first_predictions, first_summary = outputs[0]
    second_entities, second_predictions, second_summary = outputs[1]
    assert pq.read_schema(first_entities) == ENTITY_SCHEMA
    assert pq.read_schema(first_predictions) == PREDICTION_SCHEMA
    assert pq.read_table(first_entities).equals(pq.read_table(second_entities))
    assert pq.read_table(first_predictions).equals(pq.read_table(second_predictions))
    assert (
        hashlib.sha256(first_entities.read_bytes()).hexdigest()
        == hashlib.sha256(second_entities.read_bytes()).hexdigest()
    )
    assert (
        hashlib.sha256(first_predictions.read_bytes()).hexdigest()
        == hashlib.sha256(second_predictions.read_bytes()).hexdigest()
    )
    assert (
        first_summary.document_count,
        first_summary.positive_document_count,
        first_summary.entity_count,
        first_summary.source_input_sha256,
        first_summary.resource_sha256,
        first_summary.offset_coordinate_system,
    ) == (
        second_summary.document_count,
        second_summary.positive_document_count,
        second_summary.entity_count,
        second_summary.source_input_sha256,
        second_summary.resource_sha256,
        second_summary.offset_coordinate_system,
    )


def test_batch_failure_after_flush_preserves_outputs_and_removes_partials(tmp_path: Path) -> None:
    input_path = tmp_path / "input.jsonl"
    input_path.write_text(
        json.dumps({"doc": "duplicate", "text": "ARDS"})
        + "\n"
        + json.dumps({"doc": "duplicate", "text": "pulmonary edema"})
        + "\n",
        encoding="utf-8",
    )
    entity_output = tmp_path / "entities.parquet"
    prediction_output = tmp_path / "predictions.parquet"
    entity_output.write_bytes(b"existing entities")
    prediction_output.write_bytes(b"existing predictions")

    with pytest.raises(ValueError, match="Duplicate input document ID"):
        run_clamp_ards_batch(
            input_path=input_path,
            entity_output=entity_output,
            prediction_output=prediction_output,
            id_column="doc",
            text_column="text",
            batch_size=1,
            show_progress=False,
        )

    assert entity_output.read_bytes() == b"existing entities"
    assert prediction_output.read_bytes() == b"existing predictions"
    assert not list(tmp_path.glob(".*.partial"))


def test_benchmark_cli_writes_machine_readable_parity_and_performance_data(
    tmp_path: Path,
) -> None:
    output = tmp_path / "benchmark.json"
    process = subprocess.run(
        [
            sys.executable,
            "scripts/benchmark_clamp_ards_matcher.py",
            "--documents",
            "44",
            "--tokens-per-document",
            "24",
            "--repeats",
            "2",
            "--warmups",
            "0",
            "--output",
            str(output),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    # Tiny timings may miss a target; the CLI must still persist a complete, honest result.
    assert process.returncode in {0, 2}, process.stderr
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 2
    assert payload["status"] in {"ok", "performance_failure"}
    assert payload["parity"]["passed"] is True
    assert payload["parity"]["documents_checked"] == 44
    assert payload["parity"]["documents_available"] == 44
    assert payload["parity"]["full_corpus"] is True
    assert payload["benchmark"]["repeats"] == 2
    assert payload["benchmark"]["warmups"] == 0
    assert payload["benchmark"]["parity_documents"] == 44
    assert payload["benchmark"]["parity_documents_requested"] is None
    assert payload["benchmark"]["parity_mode"] == "full"
    assert payload["benchmark"]["clock"] == "time.perf_counter"
    assert payload["benchmark"]["dispersion"] == "interquartile_range"
    assert payload["benchmark"]["profile_stage_scope"] == {
        "optimized_only": ["span_index_build"],
        "naive_reference": "Pre-index implementation has no span-index construction",
    }
    assert payload["fixture"]["workload_distribution"] == {
        "dictionary_only": 11,
        "dictionary_plus_cue": 11,
        "filler_only": 22,
    }
    assert set(payload["fixture"]["interaction_cue_categories"]) == {
        "conjunctions",
        "negPhrases",
        "postNegPhrases",
        "pseNegPhrases",
    }
    assert payload["fixture"]["interaction_cue_categories"] == {
        "conjunctions": 2,
        "negPhrases": 3,
        "postNegPhrases": 3,
        "pseNegPhrases": 3,
    }
    assert payload["benchmark"]["reference_base_commit"] == (
        "b197d4f14a5880158625994a86bd6d0fb3e2af41"
    )
    assert payload["comparisons"]["cue_matching"]["target_speedup"] == 3.0
    assert payload["comparisons"]["full_mirror"]["target_speedup"] == 2.0
    assert payload["comparisons"]["full_mirror_memory"]["absolute_allowance_bytes"] == (
        5 * 1024 * 1024
    )

    fixture = payload["fixture"]
    assert fixture["document_count"] == 44
    assert fixture["seed"] == payload["benchmark"]["seed"]
    assert fixture["utf8_bytes"] > 0
    assert len(fixture["sha256"]) == 64
    int(fixture["sha256"], 16)
    dictionary_coverage = fixture["dictionary_term_coverage"]
    assert dictionary_coverage["active_term_count"] == 22
    assert dictionary_coverage["observed_term_count"] == 22
    assert dictionary_coverage["dictionary_bearing_document_count"] == 22
    assert (
        dictionary_coverage["observed_dictionary_indices"]
        == dictionary_coverage["active_dictionary_indices"]
    )
    assert dictionary_coverage["missing_dictionary_indices"] == []
    assert set(dictionary_coverage["document_count_by_dictionary_index"].values()) == {1}
    assert dictionary_coverage["all_active_terms_observed"] is True
    cue_coverage = fixture["assertion_cue_coverage"]
    assert cue_coverage["configured_cue_count"] == 240
    assert cue_coverage["observed_cue_count"] == 11
    assert cue_coverage["inserted_cue_occurrence_count"] == 13
    assert cue_coverage["observed_categories"] == cue_coverage["configured_categories"]
    assert cue_coverage["missing_categories"] == []
    assert cue_coverage["cue_occurrence_count_by_category"] == {
        "conjunctions": 2,
        "negPhrases": 5,
        "postNegPhrases": 3,
        "pseNegPhrases": 3,
    }
    assert cue_coverage["all_configured_cues_observed"] is False
    conjunction_termination = fixture["conjunction_termination"]
    assert conjunction_termination["document_count"] == 2
    assert conjunction_termination["dictionary_entity_count"] >= 2
    assert (
        conjunction_termination["present_entity_count"]
        == conjunction_termination["dictionary_entity_count"]
    )
    assert conjunction_termination["absent_entity_count"] == 0
    assert conjunction_termination["all_dictionary_entities_present"] is True
    token_distribution = fixture["token_distribution"]
    assert set(token_distribution) == {
        "total",
        "minimum",
        "q1",
        "median",
        "mean",
        "q3",
        "maximum",
    }
    assert token_distribution["total"] > 0
    assert (
        token_distribution["minimum"]
        <= token_distribution["q1"]
        <= token_distribution["median"]
        <= token_distribution["q3"]
        <= token_distribution["maximum"]
    )
    assert token_distribution["minimum"] <= token_distribution["mean"]
    assert token_distribution["mean"] <= token_distribution["maximum"]
    assert token_distribution["mean"] == pytest.approx(
        token_distribution["total"] / fixture["document_count"]
    )

    environment = payload["environment"]
    assert {
        "python_version",
        "python_implementation",
        "platform",
        "machine",
        "processor",
        "logical_cpu_count",
        "package_version",
        "dependencies",
        "git_commit",
        "git_dirty",
        "phenotype_spec_version",
        "phenotype_spec_sha256",
        "resource_sha256",
        "source_tree_fingerprint",
    } <= set(environment)
    assert all(
        isinstance(environment[key], str)
        for key in (
            "python_version",
            "python_implementation",
            "platform",
            "machine",
            "processor",
            "package_version",
            "phenotype_spec_version",
        )
    )
    assert environment["logical_cpu_count"] is None or environment["logical_cpu_count"] > 0
    assert isinstance(environment["git_commit"], str)
    assert isinstance(environment["git_dirty"], bool)
    assert set(environment["dependencies"]) == {"pyarrow", "regex", "setuptools"}
    assert all(
        version is None or isinstance(version, str)
        for version in environment["dependencies"].values()
    )
    for digest in (
        environment["phenotype_spec_sha256"],
        *environment["resource_sha256"].values(),
    ):
        assert len(digest) == 64
        int(digest, 16)
    assert set(environment["resource_sha256"]) == {
        "Components/Assertion classifier/DF_NegEx_assertion/defaultNegexDict.txt",
        "Components/Sentence detector/DF_Clamp_sentence_detector/defaultAbbrs.txt",
        "Components/Tokenizer/DF_Clamp_tokenizer/defaultTokenRule.txt",
    }
    source_fingerprint = environment["source_tree_fingerprint"]
    assert len(source_fingerprint["sha256"]) == 64
    int(source_fingerprint["sha256"], 16)
    assert len(source_fingerprint["head_tree"]) == 40
    int(source_fingerprint["head_tree"], 16)
    assert len(source_fingerprint["tracked_diff_sha256"]) == 64
    int(source_fingerprint["tracked_diff_sha256"], 16)
    assert source_fingerprint["tracked_diff_bytes"] >= 0
    assert "scripts/benchmark_clamp_ards_matcher.py" in source_fingerprint["untracked_pathspecs"]
    assert isinstance(source_fingerprint["untracked_files"], list)
    for file in source_fingerprint["untracked_files"]:
        assert set(file) == {"path", "sha256", "bytes"}
        assert len(file["sha256"]) == 64

    expected_stages = {
        "optimized": {
            "tokenization",
            "sentence_segmentation",
            "span_index_build",
            "dictionary_matching",
            "cue_matching",
            "assertion_classification",
            "postprocessing",
            "utf16_conversion",
            "batch_serialization",
            "full_mirror",
        },
        "naive_reference": {
            "dictionary_matching",
            "cue_matching",
            "assertion_classification",
            "full_mirror",
        },
    }
    assert set(payload["timings"]) == set(expected_stages)
    for implementation, stage_names in expected_stages.items():
        timings = payload["timings"][implementation]
        assert set(timings) == stage_names
        for timing in timings.values():
            assert timing["repeats"] == 2
            assert timing["warmups"] == 0
            assert len(timing["seconds"]) == 2
            assert all(seconds > 0 for seconds in timing["seconds"])
            assert timing["median_seconds"] > 0
            assert timing["q1_seconds"] <= timing["median_seconds"]
            assert timing["median_seconds"] <= timing["q3_seconds"]
            assert timing["iqr_seconds"] == pytest.approx(
                timing["q3_seconds"] - timing["q1_seconds"]
            )
            assert timing["documents_per_second"] == pytest.approx(
                fixture["document_count"] / timing["median_seconds"]
            )
            assert timing["tokens_per_second"] == pytest.approx(
                token_distribution["total"] / timing["median_seconds"]
            )
            assert timing["python_peak_memory_bytes"] >= 0

    for implementation in ("optimized", "naive_reference"):
        full_mirror = payload["timings"][implementation]["full_mirror"]
        assert full_mirror["memory_measurement_scope"] == (
            "fresh_mirror_construction_plus_complete_corpus_run"
        )
        assert full_mirror["memory_work_units"] == full_mirror["work_units"]

    memory = payload["comparisons"]["full_mirror_memory"]
    assert memory["relative_allowance"] == 0.10
    assert memory["measurement_scope"] == ("fresh mirror construction plus one complete corpus run")
    assert (
        memory["reference_peak_bytes"]
        == payload["timings"]["naive_reference"]["full_mirror"]["python_peak_memory_bytes"]
    )
    assert (
        memory["optimized_peak_bytes"]
        == payload["timings"]["optimized"]["full_mirror"]["python_peak_memory_bytes"]
    )
    assert memory["allowed_peak_bytes"] == memory["reference_peak_bytes"] + max(
        round(memory["reference_peak_bytes"] * memory["relative_allowance"]),
        memory["absolute_allowance_bytes"],
    )
    assert memory["target_met"] is (memory["optimized_peak_bytes"] <= memory["allowed_peak_bytes"])
    assert payload["status"] == (
        "ok"
        if all(
            payload["comparisons"][name]["target_met"]
            for name in ("cue_matching", "full_mirror", "full_mirror_memory")
        )
        else "performance_failure"
    )
    acceptance = payload["acceptance"]
    performance_gates_met = all(
        payload["comparisons"][name]["target_met"]
        for name in ("cue_matching", "full_mirror", "full_mirror_memory")
    )
    assert acceptance == {
        "eligible": True,
        "parity_gate_met": True,
        "performance_gates_met": performance_gates_met,
        "passed": performance_gates_met,
        "reason": None if performance_gates_met else "performance_targets_not_met",
    }


def test_benchmark_cli_marks_partial_parity_as_diagnostic_only(tmp_path: Path) -> None:
    output = tmp_path / "partial-benchmark.json"
    process = subprocess.run(
        [
            sys.executable,
            "scripts/benchmark_clamp_ards_matcher.py",
            "--documents",
            "8",
            "--tokens-per-document",
            "16",
            "--repeats",
            "1",
            "--warmups",
            "0",
            "--parity-documents",
            "3",
            "--output",
            str(output),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert process.returncode == 0, process.stderr
    assert "diagnostic-only" in process.stderr
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 2
    assert payload["status"] == "diagnostic_only"
    assert payload["benchmark"]["parity_documents"] == 3
    assert payload["benchmark"]["parity_documents_requested"] == 3
    assert payload["benchmark"]["parity_mode"] == "partial_diagnostic"
    assert payload["parity"] == {
        "passed": True,
        "documents_checked": 3,
        "documents_available": 8,
        "full_corpus": False,
        "checks": {
            "dictionary_matches": 3,
            "cue_matches": 3,
            "assertion_results": 3,
            "full_mirror_outputs": 3,
        },
    }
    performance_gates_met = all(
        payload["comparisons"][name]["target_met"]
        for name in ("cue_matching", "full_mirror", "full_mirror_memory")
    )
    assert payload["acceptance"] == {
        "eligible": False,
        "parity_gate_met": False,
        "performance_gates_met": performance_gates_met,
        "passed": False,
        "reason": "partial_parity_check",
    }
