from __future__ import annotations

import re

from .types import SentenceSpan, TokenSpan


class ClampSentenceSegmenter:
    """Empirical compatibility segmenter for the configured CLAMP detector.

    The restricted XMI corpus establishes that sentence spans trim surrounding whitespace,
    always break on configured newlines, include terminal punctuation, and usually break after a
    period followed by whitespace. CLAMP's learned/implementation-specific decisions around
    abbreviations are approximated with its supplied abbreviation lexicon and next-token context.
    """

    def __init__(
        self,
        abbreviations: frozenset[str],
        *,
        newline_ends_sentence: bool,
        break_long_sentences: bool,
        max_sentence_tokens: int,
    ) -> None:
        self._abbreviations = abbreviations
        self._newline_ends_sentence = newline_ends_sentence
        self._break_long_sentences = break_long_sentences
        self._max_sentence_tokens = max_sentence_tokens

    def segment(self, text: str, tokens: list[TokenSpan]) -> list[SentenceSpan]:
        boundaries = self._candidate_boundaries(text)
        raw_spans: list[tuple[int, int]] = []
        start = 0
        for boundary_start, boundary_end, kind in boundaries:
            if kind == "period" and self._suppress_period_boundary(text, start, boundary_start):
                continue
            span = self._trimmed_span(text, start, boundary_start)
            if span is not None:
                raw_spans.append(span)
            start = boundary_end
        final_span = self._trimmed_span(text, start, len(text))
        if final_span is not None:
            raw_spans.append(final_span)

        if self._break_long_sentences:
            raw_spans = self._split_long_spans(raw_spans, tokens)
        return [
            SentenceSpan(start=start, end=end, sentence_number=index)
            for index, (start, end) in enumerate(raw_spans)
        ]

    def _candidate_boundaries(self, text: str) -> list[tuple[int, int, str]]:
        pattern = r"[\r\n]+|(?<=\.)(?=\s)" if self._newline_ends_sentence else r"(?<=\.)(?=\s)"
        boundaries: list[tuple[int, int, str]] = []
        for match in re.finditer(pattern, text):
            value = match.group(0)
            kind = "newline" if "\n" in value or "\r" in value else "period"
            boundaries.append((match.start(), match.end(), kind))
        return boundaries

    def _suppress_period_boundary(self, text: str, segment_start: int, boundary: int) -> bool:
        prefix = text[segment_start:boundary]
        token_match = re.search(r"\S+$", prefix)
        if token_match is None:
            return False
        token = token_match.group(0)
        following = text[boundary:].lstrip(" \t")
        bare = token[:-1] if token.endswith(".") else token
        return bool(
            following
            and (
                (
                    token in self._abbreviations
                    and (following[0].islower() or following.startswith("___"))
                )
                or (bare.islower() and bare in self._abbreviations and following[0].islower())
            )
        )

    @staticmethod
    def _trimmed_span(text: str, start: int, end: int) -> tuple[int, int] | None:
        while start < end and text[start].isspace():
            start += 1
        while end > start and text[end - 1].isspace():
            end -= 1
        return None if start == end else (start, end)

    def _split_long_spans(
        self,
        spans: list[tuple[int, int]],
        tokens: list[TokenSpan],
    ) -> list[tuple[int, int]]:
        result: list[tuple[int, int]] = []
        for start, end in spans:
            sentence_tokens = [
                token for token in tokens if token.start >= start and token.end <= end
            ]
            if len(sentence_tokens) <= self._max_sentence_tokens:
                result.append((start, end))
                continue
            chunk_start = start
            for offset in range(
                self._max_sentence_tokens, len(sentence_tokens), self._max_sentence_tokens
            ):
                chunk_end = sentence_tokens[offset - 1].end
                result.append((chunk_start, chunk_end))
                chunk_start = sentence_tokens[offset].start
            result.append((chunk_start, end))
        return result
