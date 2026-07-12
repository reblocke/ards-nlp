from __future__ import annotations

import re
from bisect import bisect_left
from dataclasses import dataclass

from .types import TokenSpan


@dataclass(frozen=True)
class Utf16OffsetMap:
    """Map Python source boundaries to Java/UIMA UTF-16 code-unit offsets."""

    offsets: tuple[int, ...]

    @classmethod
    def from_text(cls, text: str) -> Utf16OffsetMap:
        offsets = [0]
        current = 0
        for character in text:
            current += 2 if ord(character) > 0xFFFF else 1
            offsets.append(current)
        return cls(tuple(offsets))

    def offset(self, python_index: int) -> int:
        if python_index < 0 or python_index >= len(self.offsets):
            raise ValueError(
                f"Python source index {python_index} is outside 0..{len(self.offsets) - 1}"
            )
        return self.offsets[python_index]

    def span(self, start: int, end: int) -> tuple[int, int]:
        if end < start:
            raise ValueError(f"Invalid Python source span: [{start}, {end})")
        return self.offset(start), self.offset(end)

    def python_index(self, utf16_offset: int) -> int:
        """Return the Python boundary for an exact UTF-16 code-unit boundary."""

        if utf16_offset < 0 or utf16_offset > self.offsets[-1]:
            raise ValueError(f"UTF-16 offset {utf16_offset} is outside 0..{self.offsets[-1]}")
        index = bisect_left(self.offsets, utf16_offset)
        if index >= len(self.offsets) or self.offsets[index] != utf16_offset:
            raise ValueError(f"UTF-16 offset {utf16_offset} falls inside a surrogate pair")
        return index

    def python_span(self, start: int, end: int) -> tuple[int, int]:
        if end < start:
            raise ValueError(f"Invalid UTF-16 span: [{start}, {end})")
        return self.python_index(start), self.python_index(end)


class ClampTokenizer:
    """Offset-preserving scanner matching the configured CLAMP delimiter behavior."""

    def __init__(self, delimiters: frozenset[str]) -> None:
        escaped = re.escape("".join(sorted(delimiters)))
        # CLAMP splits a leading numeric run from following letters (``2week``), while an
        # alphanumeric token beginning with a non-digit remains intact (``O2`` and ``POD1``).
        self._pattern = re.compile(rf"\d+|[^\s\d{escaped}][^\s{escaped}]*|[{escaped}]")

    def tokenize(self, text: str) -> list[TokenSpan]:
        return [
            TokenSpan(start=match.start(), end=match.end(), token_number=index)
            for index, match in enumerate(self._pattern.finditer(text))
        ]


def token_gap(left: TokenSpan | object, right: TokenSpan | object, tokens: list[TokenSpan]) -> int:
    left_end = int(left.end)
    right_start = int(right.start)
    if left_end > right_start:
        return -1
    return sum(token.start >= left_end and token.end <= right_start for token in tokens)
