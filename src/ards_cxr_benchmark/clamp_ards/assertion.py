from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from .resources import AssertionCue
from .span_index import DocumentSpanIndex
from .types import EntitySpan, SentenceSpan, TokenSpan


@dataclass(frozen=True, order=True)
class CueSpan:
    start: int
    end: int
    phrase: str
    category: str
    index: int


@dataclass(frozen=True)
class _CompiledCue:
    cue: AssertionCue
    token_values: tuple[str, ...]


@dataclass(frozen=True)
class _SentenceCues:
    negations: tuple[CueSpan, ...]
    post_negations: tuple[CueSpan, ...]
    pseudo_negations: tuple[CueSpan, ...]
    conjunctions: tuple[CueSpan, ...]


class ClampNegExAssertion:
    """Empirical NegEx compatibility layer using the supplied CLAMP cue inventory."""

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
        buckets: dict[str, list[_CompiledCue]] = {}
        for compiled in self._compiled:
            buckets.setdefault(compiled.token_values[0], []).append(compiled)
        self._cues_by_first_token: Mapping[str, tuple[_CompiledCue, ...]] = MappingProxyType(
            {value: tuple(bucket) for value, bucket in buckets.items()}
        )

    def classify(
        self,
        text: str,
        sentences: list[SentenceSpan],
        tokens: list[TokenSpan],
        entities: list[EntitySpan],
        *,
        span_index: DocumentSpanIndex | None = None,
    ) -> list[EntitySpan]:
        document_spans = span_index or DocumentSpanIndex.build(tokens, sentences)
        cues = self.find_cues(text, tokens)
        cues_by_sentence = self._partition_cues(cues, len(sentences), document_spans)
        result: list[EntitySpan] = []
        for entity in entities:
            sentence_index = document_spans.sentence_for_span(entity.start, entity.end)
            if sentence_index is None:
                result.append(entity)
                continue
            first_token, stop_token = document_spans.sentence_token_bounds[sentence_index]
            sentence_token_start = (
                document_spans.token_starts[first_token] if first_token < stop_token else 0
            )
            cue = self._negating_cue(
                text,
                entity,
                document_spans,
                cues_by_sentence[sentence_index],
                sentence_token_start,
            )
            result.append(
                entity.with_assertion("absent", cue.phrase)
                if cue is not None
                else entity.with_assertion("present")
            )
        return result

    def find_cues(self, text: str, tokens: list[TokenSpan]) -> list[CueSpan]:
        values = [token.covered_text(text).casefold() for token in tokens]
        matches: list[CueSpan] = []
        for start_index, token_value in enumerate(values):
            for compiled in self._cues_by_first_token.get(token_value, ()):
                length = len(compiled.token_values)
                stop_index = start_index + length
                if stop_index > len(tokens) or not self._matches_at(
                    values,
                    start_index,
                    compiled.token_values,
                ):
                    continue
                end_index = stop_index - 1
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
        span_index: DocumentSpanIndex,
        cues: _SentenceCues,
        sentence_start: int,
    ) -> CueSpan | None:
        selected: CueSpan | None = None
        for cue in cues.negations:
            if cue.end > entity.start:
                continue
            if span_index.token_gap(cue.end, entity.start) > self._scope_tokens:
                continue
            if self._terminated(cue.end, entity.start, cues.conjunctions):
                continue
            if self._blocked_by_pseudo(
                text,
                cue,
                entity,
                cues.pseudo_negations,
                sentence_start,
            ):
                continue
            if selected is None or (cue.end, cue.start, -cue.index) > (
                selected.end,
                selected.start,
                -selected.index,
            ):
                selected = cue
        if selected is not None:
            return selected

        for cue in cues.post_negations:
            if cue.start < entity.end:
                continue
            if span_index.token_gap(entity.end, cue.start) > self._scope_tokens:
                continue
            if self._terminated(entity.end, cue.start, cues.conjunctions):
                continue
            if selected is None or (cue.start, cue.end, cue.index) < (
                selected.start,
                selected.end,
                selected.index,
            ):
                selected = cue
        return selected

    @classmethod
    def _blocked_by_pseudo(
        cls,
        text: str,
        negation: CueSpan,
        entity: EntitySpan,
        pseudo_negations: tuple[CueSpan, ...],
        sentence_start: int,
    ) -> bool:
        for pseudo in pseudo_negations:
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
    def _matches_at(values: list[str], start: int, pattern: tuple[str, ...]) -> bool:
        for offset in range(1, len(pattern)):
            if values[start + offset] != pattern[offset]:
                return False
        return True

    @staticmethod
    def _partition_cues(
        cues: list[CueSpan],
        sentence_count: int,
        span_index: DocumentSpanIndex,
    ) -> tuple[_SentenceCues, ...]:
        negations: list[list[CueSpan]] = [[] for _ in range(sentence_count)]
        post_negations: list[list[CueSpan]] = [[] for _ in range(sentence_count)]
        pseudo_negations: list[list[CueSpan]] = [[] for _ in range(sentence_count)]
        conjunctions: list[list[CueSpan]] = [[] for _ in range(sentence_count)]
        category_lists = {
            "negPhrases": negations,
            "postNegPhrases": post_negations,
            "pseNegPhrases": pseudo_negations,
            "conjunctions": conjunctions,
        }
        for cue in cues:
            sentence_index = span_index.sentence_for_span(cue.start, cue.end)
            target = category_lists.get(cue.category)
            if sentence_index is not None and target is not None:
                target[sentence_index].append(cue)
        return tuple(
            _SentenceCues(
                negations=tuple(negations[index]),
                post_negations=tuple(post_negations[index]),
                pseudo_negations=tuple(pseudo_negations[index]),
                conjunctions=tuple(conjunctions[index]),
            )
            for index in range(sentence_count)
        )

    @staticmethod
    def _terminated(start: int, end: int, conjunctions: tuple[CueSpan, ...]) -> bool:
        return any(cue.start >= start and cue.end <= end for cue in conjunctions)

    @staticmethod
    def _overlaps(left: CueSpan, right: CueSpan) -> bool:
        return left.start < right.end and right.start < left.end
