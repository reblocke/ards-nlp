from __future__ import annotations

from bisect import bisect_left, bisect_right
from dataclasses import dataclass

from .types import SentenceSpan, TokenSpan


@dataclass(frozen=True)
class DocumentSpanIndex:
    """Immutable sentence membership and token-boundary index for one document."""

    token_to_sentence: tuple[int | None, ...]
    sentence_token_bounds: tuple[tuple[int, int], ...]
    sentence_starts: tuple[int, ...]
    sentence_ends: tuple[int, ...]
    token_starts: tuple[int, ...]
    token_ends: tuple[int, ...]

    @classmethod
    def build(
        cls,
        tokens: list[TokenSpan],
        sentences: list[SentenceSpan],
    ) -> DocumentSpanIndex:
        memberships: list[int | None] = [None] * len(tokens)
        first_tokens: list[int | None] = [None] * len(sentences)
        stop_tokens: list[int | None] = [None] * len(sentences)
        sentence_index = 0
        for token_index, token in enumerate(tokens):
            while sentence_index < len(sentences) and sentences[sentence_index].end <= token.start:
                sentence_index += 1
            if sentence_index >= len(sentences):
                break
            sentence = sentences[sentence_index]
            if token.start < sentence.start or token.end > sentence.end:
                continue
            memberships[token_index] = sentence_index
            if first_tokens[sentence_index] is None:
                first_tokens[sentence_index] = token_index
            stop_tokens[sentence_index] = token_index + 1

        token_starts = tuple(token.start for token in tokens)
        bounds = tuple(
            (first, stop)
            if first is not None and stop is not None
            else (bisect_left(token_starts, sentence.start),) * 2
            for sentence, first, stop in zip(
                sentences,
                first_tokens,
                stop_tokens,
                strict=True,
            )
        )
        return cls(
            token_to_sentence=tuple(memberships),
            sentence_token_bounds=bounds,
            sentence_starts=tuple(sentence.start for sentence in sentences),
            sentence_ends=tuple(sentence.end for sentence in sentences),
            token_starts=token_starts,
            token_ends=tuple(token.end for token in tokens),
        )

    def sentence_for_token_span(self, start_index: int, end_index: int) -> int | None:
        return self.sentence_for_span(self.token_starts[start_index], self.token_ends[end_index])

    def sentence_for_span(self, start: int, end: int) -> int | None:
        sentence_index = bisect_left(self.sentence_ends, end)
        if sentence_index >= len(self.sentence_starts):
            return None
        return sentence_index if start >= self.sentence_starts[sentence_index] else None

    def token_gap(self, left_end: int, right_start: int) -> int:
        if left_end > right_start:
            return -1
        first = bisect_left(self.token_starts, left_end)
        stop = bisect_right(self.token_ends, right_start)
        return max(0, stop - first)
