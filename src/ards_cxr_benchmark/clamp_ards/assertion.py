from __future__ import annotations

from dataclasses import dataclass

from .resources import AssertionCue
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
