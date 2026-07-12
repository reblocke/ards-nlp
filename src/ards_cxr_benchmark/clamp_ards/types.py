from __future__ import annotations

from dataclasses import dataclass, field, replace


@dataclass(frozen=True, order=True)
class TextSpan:
    """Half-open character span over the immutable source text."""

    start: int
    end: int
    source_start: int | None = field(default=None, compare=False, repr=False, kw_only=True)
    source_end: int | None = field(default=None, compare=False, repr=False, kw_only=True)

    def __post_init__(self) -> None:
        if self.start < 0 or self.end < self.start:
            raise ValueError(f"Invalid half-open span: [{self.start}, {self.end})")
        if (self.source_start is None) != (self.source_end is None):
            raise ValueError("source_start and source_end must be provided together")
        if self.source_start is not None and (
            self.source_start < 0 or self.source_end is None or self.source_end < self.source_start
        ):
            raise ValueError(f"Invalid source span: [{self.source_start}, {self.source_end})")

    def covered_text(self, text: str) -> str:
        start = self.start if self.source_start is None else self.source_start
        end = self.end if self.source_end is None else self.source_end
        if end > len(text):
            raise ValueError(f"Span end {end} exceeds text length {len(text)}")
        return text[start:end]


@dataclass(frozen=True, order=True)
class SentenceSpan(TextSpan):
    sentence_number: int


@dataclass(frozen=True, order=True)
class TokenSpan(TextSpan):
    token_number: int


@dataclass(frozen=True, order=True)
class EntitySpan(TextSpan):
    semantic_tag: str
    assertion: str = "present"
    cui: str | None = None
    attribute: str | None = None
    dictionary_term: str | None = None
    dictionary_index: int = -1
    assertion_cue: str | None = None

    def with_semantic_tag(self, semantic_tag: str) -> EntitySpan:
        return replace(self, semantic_tag=semantic_tag)

    def with_assertion(self, assertion: str, cue: str | None = None) -> EntitySpan:
        return replace(self, assertion=assertion, assertion_cue=cue)
