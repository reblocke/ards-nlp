from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path

from .assertion import ClampNegExAssertion
from .dictionary import ClampDictionaryMatcher
from .pos import ClampPosNoOp
from .resources import ClampResources, load_clamp_resources
from .ruta import ClampRutaPostprocessor
from .segmentation import ClampSentenceSegmenter
from .stemming import PorterCompatibilityStemmer
from .tokenization import ClampTokenizer, Utf16OffsetMap
from .types import EntitySpan, SentenceSpan, TokenSpan


@dataclass(frozen=True)
class PipelineTrace:
    sentences: tuple[SentenceSpan, ...]
    tokens: tuple[TokenSpan, ...]
    dictionary_entities: tuple[EntitySpan, ...]
    asserted_entities: tuple[EntitySpan, ...]
    final_entities: tuple[EntitySpan, ...]


class LegacyARDSClampMirror:
    def __init__(self, resources: ClampResources) -> None:
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
        self.dictionary = ClampDictionaryMatcher(
            resources.dictionary,
            stemmer,
            self.tokenizer,
            excluded_terms=resources.excluded_dictionary_terms,
        )
        self.assertion = ClampNegExAssertion(
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

    def run(self, text: str, *, doc_id: str | None = None) -> list[EntitySpan]:
        del doc_id  # Reserved for provenance/debugging; never inserted into source text.
        internal_entities = self.trace(text).final_entities
        offsets = Utf16OffsetMap.from_text(text)
        result: list[EntitySpan] = []
        for entity in internal_entities:
            start, end = offsets.span(entity.start, entity.end)
            result.append(
                replace(
                    entity,
                    start=start,
                    end=end,
                    source_start=entity.start,
                    source_end=entity.end,
                )
            )
        return result

    def trace(self, text: str) -> PipelineTrace:
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


@lru_cache(maxsize=8)
def load_legacy_mirror(
    project_dir: str | None = None,
    resource_manifest: str | None = None,
) -> LegacyARDSClampMirror:
    root = Path(project_dir) if project_dir is not None else None
    manifest = Path(resource_manifest) if resource_manifest is not None else None
    resources = load_clamp_resources(root, manifest_path=manifest)
    return LegacyARDSClampMirror(resources)


def run_legacy_ards_clamp_mirror(
    text: str,
    *,
    doc_id: str | None = None,
) -> list[EntitySpan]:
    return load_legacy_mirror().run(text, doc_id=doc_id)


def predict_legacy_ards_label(text: str) -> int:
    return int(bool(run_legacy_ards_clamp_mirror(text)))
