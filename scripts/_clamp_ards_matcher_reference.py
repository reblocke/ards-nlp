"""Naive CLAMP matcher oracle retained outside the production package.

This module intentionally preserves the pre-indexing implementation. It is imported by
tests and benchmark support only; production entry points must always use the packaged
matchers under ``ards_cxr_benchmark.clamp_ards``.
"""

from __future__ import annotations

from dataclasses import dataclass

from ards_cxr_benchmark.clamp_ards.assertion import CueSpan
from ards_cxr_benchmark.clamp_ards.pipeline import LegacyARDSClampMirror, PipelineTrace
from ards_cxr_benchmark.clamp_ards.pos import ClampPosNoOp
from ards_cxr_benchmark.clamp_ards.resources import (
    AssertionCue,
    ClampResources,
    DictionaryEntry,
)
from ards_cxr_benchmark.clamp_ards.ruta import ClampRutaPostprocessor
from ards_cxr_benchmark.clamp_ards.segmentation import ClampSentenceSegmenter
from ards_cxr_benchmark.clamp_ards.stemming import (
    PorterCompatibilityStemmer,
    StemmerProtocol,
)
from ards_cxr_benchmark.clamp_ards.tokenization import ClampTokenizer
from ards_cxr_benchmark.clamp_ards.types import EntitySpan, SentenceSpan, TokenSpan


@dataclass(frozen=True)
class _CompiledEntry:
    entry: DictionaryEntry
    token_values: tuple[str, ...]


class ReferenceClampDictionaryMatcher:
    """Exact pre-indexing dictionary matcher used as a test/benchmark oracle."""

    def __init__(
        self,
        entries: tuple[DictionaryEntry, ...],
        stemmer: StemmerProtocol,
        tokenizer: object,
        *,
        excluded_terms: frozenset[str] = frozenset(),
    ) -> None:
        self._stemmer = stemmer
        self._tokenizer = tokenizer
        self._entries = tuple(
            self._compile(entry) for entry in entries if entry.term not in excluded_terms
        )

    def match(
        self,
        text: str,
        tokens: list[TokenSpan],
        sentences: list[SentenceSpan],
        *,
        span_index: object | None = None,
    ) -> list[EntitySpan]:
        del span_index
        token_values = [self._normalize(token.covered_text(text)) for token in tokens]
        entities: list[EntitySpan] = []
        for start_index in range(len(tokens)):
            for compiled in self._entries:
                length = len(compiled.token_values)
                if tuple(token_values[start_index : start_index + length]) != compiled.token_values:
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
                        semantic_tag=compiled.entry.semantic_tag,
                        dictionary_term=compiled.entry.term,
                        dictionary_index=compiled.entry.index,
                    )
                )
        return sorted(
            entities, key=lambda entity: (entity.start, entity.end, entity.dictionary_index)
        )

    def _compile(self, entry: DictionaryEntry) -> _CompiledEntry:
        tokens = self._tokenizer.tokenize(entry.term)
        values = tuple(self._normalize(token.covered_text(entry.term)) for token in tokens)
        return _CompiledEntry(entry=entry, token_values=values)

    def _normalize(self, value: str) -> str:
        return self._stemmer.stem(value)


@dataclass(frozen=True)
class _CompiledCue:
    cue: AssertionCue
    token_values: tuple[str, ...]


class ReferenceClampNegExAssertion:
    """Exact pre-indexing NegEx matcher/classifier used as a test/benchmark oracle."""

    def __init__(
        self,
        cues: tuple[AssertionCue, ...],
        tokenizer: object,
        *,
        scope_tokens: int = 11,
    ) -> None:
        self._tokenizer = tokenizer
        self._scope_tokens = scope_tokens
        self._compiled = tuple(self._compile(cue) for cue in cues)

    def classify(
        self,
        text: str,
        sentences: list[SentenceSpan],
        tokens: list[TokenSpan],
        entities: list[EntitySpan],
        *,
        span_index: object | None = None,
    ) -> list[EntitySpan]:
        del span_index
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

    def find_cues(self, text: str, tokens: list[TokenSpan]) -> list[CueSpan]:
        values = [token.covered_text(text).casefold() for token in tokens]
        matches: list[CueSpan] = []
        for start_index in range(len(tokens)):
            for compiled in self._compiled:
                length = len(compiled.token_values)
                if tuple(values[start_index : start_index + length]) != compiled.token_values:
                    continue
                end_index = start_index + length - 1
                if end_index >= len(tokens):
                    continue
                matches.append(
                    CueSpan(
                        start=tokens[start_index].start,
                        end=tokens[end_index].end,
                        phrase=compiled.cue.phrase,
                        category=compiled.cue.category,
                        index=compiled.cue.index,
                    )
                )

        # CLAMP's exported NegEx behavior still allows a shorter negation cue (for example
        # ``no``) to fire when it overlaps a longer pseudo-negation such as ``no change``.
        # Retain every cue and let the directional scope rules select only neg/post-neg types.
        return sorted(matches, key=lambda cue: (cue.start, cue.end, cue.index))

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
            # The restricted oracle honors ``not necessarily``/``not only`` consistently. Its
            # ``no change`` family is implementation-dependent: an overlapping ``no`` still fires
            # when the pseudo cue begins the CLAMP sentence, but is suppressed after lexical
            # context such as ``There is`` or ``IMPRESSION:``.
            honored = pseudo.phrase.casefold().startswith("not ") or any(
                char.isalpha() for char in text[sentence_start : pseudo.start]
            )
            overlaps = cls._overlaps(negation, pseudo)
            intervenes = negation.end <= pseudo.start and pseudo.end <= entity.start
            if honored and (overlaps or intervenes):
                return True
        return False

    def _compile(self, cue: AssertionCue) -> _CompiledCue:
        tokens = self._tokenizer.tokenize(cue.phrase)
        values = tuple(token.covered_text(cue.phrase).casefold() for token in tokens)
        return _CompiledCue(cue=cue, token_values=values)

    @staticmethod
    def _gap(left_end: int, right_start: int, tokens: list[TokenSpan]) -> int:
        return sum(token.start >= left_end and token.end <= right_start for token in tokens)

    @staticmethod
    def _terminated(start: int, end: int, conjunctions: list[CueSpan]) -> bool:
        return any(cue.start >= start and cue.end <= end for cue in conjunctions)

    @staticmethod
    def _overlaps(left: CueSpan, right: CueSpan) -> bool:
        return left.start < right.end and right.start < left.end


class ReferenceLegacyARDSClampMirror(LegacyARDSClampMirror):
    """Full mirror with only the dictionary and assertion stages replaced by the oracle."""

    def __init__(self, resources: ClampResources) -> None:
        # Do not call ``LegacyARDSClampMirror.__init__`` here.  Its optimized
        # dictionary and assertion matchers allocate immutable indexes, which would
        # contaminate this module's naive-reference memory baseline before they are
        # replaced below.  Preserve the legacy component construction order instead.
        self.resources = resources
        self.tokenizer = ClampTokenizer(resources.delimiters)
        self.segmenter = ClampSentenceSegmenter(
            resources.abbreviations,
            newline_ends_sentence=resources.newline_ends_sentence,
            break_long_sentences=resources.break_long_sentences,
            max_sentence_tokens=resources.max_sentence_tokens,
        )
        self.pos = ClampPosNoOp()
        stemmer = PorterCompatibilityStemmer()
        self.dictionary = ReferenceClampDictionaryMatcher(
            resources.dictionary,
            stemmer,
            self.tokenizer,
            excluded_terms=resources.excluded_dictionary_terms,
        )
        self.assertion = ReferenceClampNegExAssertion(
            resources.assertion_cues,
            self.tokenizer,
            scope_tokens=resources.assertion_scope_tokens,
        )
        self.ruta = ClampRutaPostprocessor(
            remove_assertions=resources.remove_assertions,
            promotion_rules=resources.promotion_rules,
            max_intervening_tokens=resources.max_intervening_tokens,
            final_semantic_tag=resources.final_semantic_tag,
            remove_unpromoted_semantic_tags=resources.remove_unpromoted_semantic_tags,
        )

    def trace(self, text: str) -> PipelineTrace:
        """Run the exact pre-index pipeline without constructing optimized span indexes."""

        if not isinstance(text, str):
            raise TypeError("CLAMP mirror input must be a string")
        tokens = self.tokenizer.tokenize(text)
        sentences = self.segmenter.segment(text, tokens)
        pos_tokens = self.pos.apply(tokens)
        dictionary_entities = self.dictionary.match(text, pos_tokens, sentences)
        asserted_entities = self.assertion.classify(
            text,
            sentences,
            pos_tokens,
            dictionary_entities,
        )
        final_entities = self.ruta.apply(sentences, pos_tokens, asserted_entities)
        return PipelineTrace(
            sentences=tuple(sentences),
            tokens=tuple(tokens),
            dictionary_entities=tuple(dictionary_entities),
            asserted_entities=tuple(asserted_entities),
            final_entities=tuple(final_entities),
        )
