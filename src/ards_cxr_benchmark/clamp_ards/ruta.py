from __future__ import annotations

from .tokenization import token_gap
from .types import EntitySpan, SentenceSpan, TokenSpan


class ClampRutaPostprocessor:
    """Imperative mirror of the authorized phenotype rule order."""

    def __init__(
        self,
        *,
        remove_assertions: frozenset[str],
        promotion_rules: tuple[tuple[str, str], ...],
        max_intervening_tokens: int,
        final_semantic_tag: str,
        remove_unpromoted_semantic_tags: frozenset[str],
    ) -> None:
        self._remove_assertions = remove_assertions
        self._promotion_rules = promotion_rules
        self._max_intervening_tokens = max_intervening_tokens
        self._final_semantic_tag = final_semantic_tag
        self._intermediate_tags = remove_unpromoted_semantic_tags

    def apply(
        self,
        sentences: list[SentenceSpan],
        tokens: list[TokenSpan],
        entities: list[EntitySpan],
    ) -> list[EntitySpan]:
        active = [entity for entity in entities if entity.assertion not in self._remove_assertions]
        for sentence in sentences:
            indices = [
                index
                for index, entity in enumerate(active)
                if entity.start >= sentence.start and entity.end <= sentence.end
            ]
            immediate_matches = [
                self._immediate_pass(
                    active,
                    indices,
                    tokens,
                    first_tag=first_tag,
                    second_tag=second_tag,
                )
                for first_tag, second_tag in self._promotion_rules
            ]
            for (first_tag, second_tag), matches in zip(
                self._promotion_rules,
                immediate_matches,
                strict=True,
            ):
                self._distance_pass(
                    active,
                    indices,
                    tokens,
                    first_tag=first_tag,
                    second_tag=second_tag,
                    immediate_matches=matches,
                )
        return [entity for entity in active if entity.semantic_tag == self._final_semantic_tag]

    def _immediate_pass(
        self,
        entities: list[EntitySpan],
        indices: list[int],
        tokens: list[TokenSpan],
        *,
        first_tag: str,
        second_tag: str,
    ) -> dict[int, int]:
        snapshot = list(entities)
        intermediate = [
            index for index in indices if snapshot[index].semantic_tag in self._intermediate_tags
        ]
        intermediate.sort(key=lambda index: self._sort_key(snapshot[index]))
        clusters: list[list[int]] = []
        for index in intermediate:
            if not clusters or token_gap(snapshot[clusters[-1][-1]], snapshot[index], tokens) != 0:
                clusters.append([index])
            else:
                clusters[-1].append(index)

        matches: dict[int, int] = {}
        for cluster in clusters:
            for offset, first_index in enumerate(cluster):
                if snapshot[first_index].semantic_tag != first_tag:
                    continue
                candidates = [
                    index
                    for index in cluster[offset + 1 :]
                    if snapshot[index].semantic_tag == second_tag
                ]
                if candidates:
                    matches[first_index] = candidates[0]
        for second_index in sorted(set(matches.values())):
            entities[second_index] = entities[second_index].with_semantic_tag(
                self._final_semantic_tag
            )
        return matches

    def _distance_pass(
        self,
        entities: list[EntitySpan],
        indices: list[int],
        tokens: list[TokenSpan],
        *,
        first_tag: str,
        second_tag: str,
        immediate_matches: dict[int, int],
    ) -> None:
        snapshot = list(entities)
        matches: dict[int, int] = {}
        for first_index in indices:
            first = snapshot[first_index]
            if first.semantic_tag != first_tag:
                continue
            candidates = [
                index
                for index in indices
                if snapshot[index].semantic_tag == second_tag
                and snapshot[index].start >= first.end
                and 0 <= token_gap(first, snapshot[index], tokens) <= self._max_intervening_tokens
                and (
                    first_index in immediate_matches
                    or not self._crosses_final_tag(first, snapshot[index], snapshot, indices)
                )
            ]
            if first_index in immediate_matches:
                immediate_target = entities[immediate_matches[first_index]]
                candidates = [
                    index
                    for index in candidates
                    if token_gap(immediate_target, snapshot[index], tokens) == 0
                ]
            if candidates:
                matches[first_index] = min(
                    candidates,
                    key=lambda index: self._sort_key(snapshot[index]),
                )
        for second_index in sorted(set(matches.values())):
            entities[second_index] = entities[second_index].with_semantic_tag(
                self._final_semantic_tag
            )

    def _crosses_final_tag(
        self,
        first: EntitySpan,
        second: EntitySpan,
        entities: list[EntitySpan],
        indices: list[int],
    ) -> bool:
        return any(
            entities[index].semantic_tag == self._final_semantic_tag
            and entities[index].start >= first.end
            and entities[index].end <= second.start
            for index in indices
        )

    @staticmethod
    def _sort_key(entity: EntitySpan) -> tuple[int, int, int]:
        return (entity.start, entity.end, entity.dictionary_index)
