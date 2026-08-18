from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from .resources import DictionaryEntry
from .span_index import DocumentSpanIndex
from .stemming import StemmerProtocol
from .types import EntitySpan, SentenceSpan, TokenSpan


@dataclass(frozen=True)
class _CompiledEntry:
    entry: DictionaryEntry
    token_values: tuple[str, ...]


class ClampDictionaryMatcher:
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
        buckets: dict[str, list[_CompiledEntry]] = {}
        for compiled in self._entries:
            buckets.setdefault(compiled.token_values[0], []).append(compiled)
        self._entries_by_first_token: Mapping[str, tuple[_CompiledEntry, ...]] = MappingProxyType(
            {value: tuple(bucket) for value, bucket in buckets.items()}
        )

    def match(
        self,
        text: str,
        tokens: list[TokenSpan],
        sentences: list[SentenceSpan],
        *,
        span_index: DocumentSpanIndex | None = None,
    ) -> list[EntitySpan]:
        token_values = [self._normalize(token.covered_text(text)) for token in tokens]
        document_spans = span_index or DocumentSpanIndex.build(tokens, sentences)
        entities: list[EntitySpan] = []
        for start_index, token_value in enumerate(token_values):
            for compiled in self._entries_by_first_token.get(token_value, ()):
                length = len(compiled.token_values)
                stop_index = start_index + length
                if stop_index > len(tokens) or not self._matches_at(
                    token_values,
                    start_index,
                    compiled.token_values,
                ):
                    continue
                end_index = stop_index - 1
                if document_spans.sentence_for_token_span(start_index, end_index) is None:
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

    @staticmethod
    def _matches_at(values: list[str], start: int, pattern: tuple[str, ...]) -> bool:
        for offset in range(1, len(pattern)):
            if values[start + offset] != pattern[offset]:
                return False
        return True
