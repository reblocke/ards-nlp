from __future__ import annotations

from dataclasses import dataclass

from .resources import DictionaryEntry
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

    def match(
        self,
        text: str,
        tokens: list[TokenSpan],
        sentences: list[SentenceSpan],
    ) -> list[EntitySpan]:
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
