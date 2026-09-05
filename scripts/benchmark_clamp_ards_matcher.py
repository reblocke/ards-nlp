from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import random
import statistics
import subprocess
import sys
import time
import tracemalloc
from collections.abc import Callable
from dataclasses import dataclass, replace
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from _clamp_ards_matcher_reference import ReferenceLegacyARDSClampMirror
from ards_cxr_benchmark import __version__
from ards_cxr_benchmark.clamp_ards.batch import ENTITY_SCHEMA, PREDICTION_SCHEMA
from ards_cxr_benchmark.clamp_ards.pipeline import LegacyARDSClampMirror
from ards_cxr_benchmark.clamp_ards.resources import (
    AssertionCue,
    ClampResources,
    DictionaryEntry,
    load_clamp_resources,
)
from ards_cxr_benchmark.clamp_ards.span_index import DocumentSpanIndex
from ards_cxr_benchmark.clamp_ards.tokenization import Utf16OffsetMap
from ards_cxr_benchmark.clamp_ards.types import EntitySpan, SentenceSpan, TokenSpan

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROJECT_DIR = REPOSITORY_ROOT / "tests/fixtures/clamp_ards_external_resources"
DEFAULT_RESOURCE_MANIFEST = DEFAULT_PROJECT_DIR / "manifest.json"
DEFAULT_OUTPUT = REPOSITORY_ROOT / "artifacts/benchmark/clamp_ards_matcher/benchmark.json"

DEFAULT_DOCUMENTS = 5_000
DEFAULT_TOKENS_PER_DOCUMENT = 100
DEFAULT_SEED = 20_260_811
DEFAULT_REPEATS = 5
DEFAULT_WARMUPS = 2
REFERENCE_BASE_COMMIT = "b197d4f14a5880158625994a86bd6d0fb3e2af41"
_SOURCE_FINGERPRINT_PATHS = (
    "Makefile",
    "src/ards_cxr_benchmark",
    "scripts/_clamp_ards_matcher_reference.py",
    "scripts/benchmark_clamp_ards_matcher.py",
    "scripts/compare_clamp_python_parity.py",
    "scripts/run_pipeline.py",
    "scripts/run_python_clamp_ards.py",
    "config/clamp_ards_oracle_manifest.json",
    "config/clamp_ards_resource_manifest.json",
)

_CUE_CATEGORIES = (
    "negPhrases",
    "postNegPhrases",
    "pseNegPhrases",
    "conjunctions",
)

_NEUTRAL_WORDS = (
    "amber",
    "atlas",
    "birch",
    "bridge",
    "canyon",
    "cedar",
    "circle",
    "cobalt",
    "delta",
    "ember",
    "falcon",
    "garden",
    "harbor",
    "island",
    "juniper",
    "lantern",
    "maple",
    "meadow",
    "orbit",
    "pebble",
    "quartz",
    "river",
    "silver",
    "summit",
    "timber",
    "valley",
    "willow",
)


@dataclass(frozen=True)
class PreparedDocument:
    doc_id: str
    text: str
    tokens: list[TokenSpan]
    sentences: list[SentenceSpan]
    span_index: DocumentSpanIndex
    dictionary_entities: list[EntitySpan]
    cues: list[Any]
    asserted_entities: list[EntitySpan]
    final_entities: list[EntitySpan]
    exported_entities: list[EntitySpan]


class ReferenceParityError(RuntimeError):
    pass


class _PreparedCueAssertionProxy:
    """Run the real classifier while supplying already-timed cue matches."""

    def __init__(self, target: object) -> None:
        self._target = target
        self._cues: list[Any] = []

    def set_cues(self, cues: list[Any]) -> None:
        self._cues = cues

    def find_cues(self, text: str, tokens: list[TokenSpan]) -> list[Any]:
        del text, tokens
        return self._cues

    def __getattr__(self, name: str) -> Any:
        return getattr(self._target, name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark the exact ARDS CLAMP dictionary and NegEx matchers on deterministic "
            "synthetic text"
        )
    )
    parser.add_argument("--documents", type=int, default=DEFAULT_DOCUMENTS)
    parser.add_argument(
        "--tokens-per-document",
        type=int,
        default=DEFAULT_TOKENS_PER_DOCUMENT,
        help="Approximate generated token count per document",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    parser.add_argument("--warmups", type=int, default=DEFAULT_WARMUPS)
    parser.add_argument(
        "--parity-documents",
        type=int,
        default=None,
        help=(
            "Number of generated documents checked against the naive oracle before timing; "
            "defaults to all documents, while an explicit smaller value is diagnostic-only"
        ),
    )
    parser.add_argument("--project-dir", type=Path, default=DEFAULT_PROJECT_DIR)
    parser.add_argument("--resource-manifest", type=Path, default=DEFAULT_RESOURCE_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    _validate_args(args)
    output = args.output.expanduser().resolve()
    resources = load_clamp_resources(
        args.project_dir.expanduser().resolve(),
        manifest_path=args.resource_manifest.expanduser().resolve(),
    )
    optimized = LegacyARDSClampMirror(resources)
    reference = ReferenceLegacyARDSClampMirror(resources)
    documents = _generate_documents(
        resources,
        document_count=args.documents,
        approximate_tokens=args.tokens_per_document,
        seed=args.seed,
    )
    prepared = _prepare_documents(optimized, documents)
    parity_document_count = (
        len(prepared)
        if args.parity_documents is None
        else min(args.parity_documents, len(prepared))
    )
    full_parity_coverage = parity_document_count == len(prepared)
    fixture = _fixture_metadata(
        prepared,
        args.tokens_per_document,
        args.seed,
        resources,
    )
    environment = _environment_metadata(resources)
    base_payload: dict[str, object] = {
        "schema_version": 2,
        "benchmark": {
            "document_count": args.documents,
            "approximate_tokens_per_document": args.tokens_per_document,
            "seed": args.seed,
            "repeats": args.repeats,
            "warmups": args.warmups,
            "parity_documents": parity_document_count,
            "parity_documents_requested": args.parity_documents,
            "parity_mode": "full" if full_parity_coverage else "partial_diagnostic",
            "clock": "time.perf_counter",
            "dispersion": "interquartile_range",
            "memory_measurement": (
                "tracemalloc peak Python bytes from fresh mirror construction through one "
                "complete corpus run"
            ),
            "profile_stage_scope": {
                "optimized_only": ["span_index_build"],
                "naive_reference": "Pre-index implementation has no span-index construction",
            },
            "command": [sys.executable, *sys.argv],
            "reference_base_commit": REFERENCE_BASE_COMMIT,
        },
        "environment": environment,
        "fixture": fixture,
    }

    try:
        parity = _check_parity(
            optimized,
            reference,
            prepared[:parity_document_count],
        )
    except ReferenceParityError as exc:
        failure = {
            **base_payload,
            "status": "parity_failure",
            "parity": {
                "passed": False,
                "error": str(exc),
                "documents_requested": parity_document_count,
                "documents_available": len(prepared),
                "full_corpus": full_parity_coverage,
            },
            "acceptance": {
                "eligible": full_parity_coverage,
                "parity_gate_met": False,
                "performance_gates_met": None,
                "passed": False,
                "reason": "parity_failure",
            },
        }
        _write_json(output, failure)
        print(f"CLAMP matcher benchmark parity failure: {exc}", file=sys.stderr)
        print(f"Failure details written to {output}", file=sys.stderr)
        return 1

    parity = {
        **parity,
        "documents_available": len(prepared),
        "full_corpus": full_parity_coverage,
    }

    document_count = len(prepared)
    token_count = sum(len(document.tokens) for document in prepared)
    optimized_stages = _optimized_stage_functions(optimized, prepared)
    reference_stages = _reference_stage_functions(reference, prepared)
    optimized_timings = {
        name: _measure(
            function,
            document_count=document_count,
            token_count=token_count,
            repeats=args.repeats,
            warmups=args.warmups,
        )
        for name, function in optimized_stages.items()
    }
    reference_timings = {
        name: _measure(
            function,
            document_count=document_count,
            token_count=token_count,
            repeats=args.repeats,
            warmups=args.warmups,
        )
        for name, function in reference_stages.items()
    }
    optimized_memory = _measure_construction_and_full_run_memory(
        lambda: LegacyARDSClampMirror(resources),
        documents,
    )
    reference_memory = _measure_construction_and_full_run_memory(
        lambda: ReferenceLegacyARDSClampMirror(resources),
        documents,
    )
    optimized_timings["full_mirror"].update(optimized_memory)
    reference_timings["full_mirror"].update(reference_memory)
    comparisons = _comparisons(optimized_timings, reference_timings)
    targets_passed = all(
        bool(comparisons[name]["target_met"])
        for name in ("cue_matching", "full_mirror", "full_mirror_memory")
    )
    acceptance_passed = full_parity_coverage and targets_passed
    status = (
        "diagnostic_only"
        if not full_parity_coverage
        else "ok"
        if targets_passed
        else "performance_failure"
    )
    result = {
        **base_payload,
        "status": status,
        "parity": parity,
        "acceptance": {
            "eligible": full_parity_coverage,
            "parity_gate_met": full_parity_coverage,
            "performance_gates_met": targets_passed,
            "passed": acceptance_passed,
            "reason": (
                None
                if acceptance_passed
                else "partial_parity_check"
                if not full_parity_coverage
                else "performance_targets_not_met"
            ),
        },
        "timings": {
            "optimized": optimized_timings,
            "naive_reference": reference_timings,
        },
        "comparisons": comparisons,
    }
    _write_json(output, result)
    print(_console_summary(result, output))
    if not full_parity_coverage:
        print(
            "CLAMP matcher benchmark is diagnostic-only because parity did not cover the "
            "complete generated corpus",
            file=sys.stderr,
        )
        return 0
    if not targets_passed:
        print(
            "CLAMP matcher benchmark failed one or more hard performance targets",
            file=sys.stderr,
        )
        return 2
    return 0


def _validate_args(args: argparse.Namespace) -> None:
    if args.documents < 1:
        raise ValueError("--documents must be at least 1")
    if args.tokens_per_document < 8:
        raise ValueError("--tokens-per-document must be at least 8")
    if args.repeats < 1:
        raise ValueError("--repeats must be at least 1")
    if args.warmups < 0:
        raise ValueError("--warmups cannot be negative")
    if args.parity_documents is not None and args.parity_documents < 1:
        raise ValueError("--parity-documents must be at least 1")


def _generate_documents(
    resources: ClampResources,
    *,
    document_count: int,
    approximate_tokens: int,
    seed: int,
) -> list[tuple[str, str]]:
    rng = random.Random(seed)
    dictionary = _active_dictionary(resources)
    cues_by_category = _cues_by_category(resources)
    documents: list[tuple[str, str]] = []
    for document_index in range(document_count):
        target = max(8, round(approximate_tokens * rng.uniform(0.85, 1.15)))
        parts = [rng.choice(_NEUTRAL_WORDS) for _ in range(target)]
        workload = document_index % 4
        left_index = min(target - 2, 3)
        right_index = min(target - 1, max(left_index + 1, 10))

        # Exact mix: 50% filler-only, 25% dictionary-only, and 25% dictionary-plus-cue.
        dictionary_ordinal = _dictionary_ordinal(document_index)
        if dictionary and dictionary_ordinal is not None:
            term = dictionary[dictionary_ordinal % len(dictionary)].term
            if document_index % 3 == 1:
                term = term.upper()
            term_index = right_index
            if workload == 3:
                interaction_index = document_index // 4
                interaction_cue = _interaction_cue(cues_by_category, interaction_index)
                category = interaction_cue.category
                cue = interaction_cue.phrase
                if interaction_index % 3 == 1:
                    cue = cue.upper()
                if category == "postNegPhrases":
                    term_index = left_index
                    parts[right_index] = cue
                elif category == "conjunctions":
                    # Exercise actual scope termination: without the conjunction, the leading
                    # negation would mark the following dictionary entity absent.
                    term_index = left_index + 2
                    parts[left_index] = cues_by_category["negPhrases"][0].phrase
                    parts[left_index + 1] = cue
                else:
                    parts[left_index] = cue
            parts[term_index] = term
        if document_index % 31 == 0:
            parts[1] = "orbit😀"

        sentence_width = 18 + document_index % 7
        for boundary in range(sentence_width - 1, len(parts) - 1, sentence_width):
            parts[boundary] = f"{parts[boundary]}."
        documents.append((f"synthetic_{document_index:06d}", " ".join(parts)))
    return documents


def _active_dictionary(resources: ClampResources) -> tuple[DictionaryEntry, ...]:
    return tuple(
        entry
        for entry in resources.dictionary
        if entry.term not in resources.excluded_dictionary_terms
    )


def _dictionary_ordinal(document_index: int) -> int | None:
    """Return an ordinal among dictionary-bearing synthetic documents."""
    workload = document_index % 4
    if workload < 2:
        return None
    return (document_index // 4) * 2 + workload - 2


def _cues_by_category(
    resources: ClampResources,
) -> dict[str, tuple[AssertionCue, ...]]:
    return {
        category: tuple(cue for cue in resources.assertion_cues if cue.category == category)
        for category in _CUE_CATEGORIES
    }


def _interaction_cue(
    cues_by_category: dict[str, tuple[AssertionCue, ...]],
    interaction_index: int,
) -> AssertionCue:
    category = _CUE_CATEGORIES[interaction_index % len(_CUE_CATEGORIES)]
    category_cues = cues_by_category[category]
    return category_cues[(interaction_index // len(_CUE_CATEGORIES)) % len(category_cues)]


def _prepare_documents(
    mirror: LegacyARDSClampMirror,
    documents: list[tuple[str, str]],
) -> list[PreparedDocument]:
    prepared: list[PreparedDocument] = []
    for doc_id, text in documents:
        tokens = mirror.pos.apply(mirror.tokenizer.tokenize(text))
        sentences = mirror.segmenter.segment(text, tokens)
        span_index = DocumentSpanIndex.build(tokens, sentences)
        dictionary_entities = mirror.dictionary.match(
            text,
            tokens,
            sentences,
            span_index=span_index,
        )
        cues = mirror.assertion.find_cues(text, tokens)
        asserted_entities = mirror.assertion.classify(
            text,
            sentences,
            tokens,
            dictionary_entities,
            span_index=span_index,
        )
        final_entities = mirror.ruta.apply(sentences, tokens, asserted_entities)
        exported_entities = _export_entities(text, final_entities)
        prepared.append(
            PreparedDocument(
                doc_id=doc_id,
                text=text,
                tokens=tokens,
                sentences=sentences,
                span_index=span_index,
                dictionary_entities=dictionary_entities,
                cues=cues,
                asserted_entities=asserted_entities,
                final_entities=final_entities,
                exported_entities=exported_entities,
            )
        )
    return prepared


def _export_entities(text: str, entities: list[EntitySpan]) -> list[EntitySpan]:
    offsets = Utf16OffsetMap.from_text(text)
    result: list[EntitySpan] = []
    for entity in entities:
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


def _check_parity(
    optimized: LegacyARDSClampMirror,
    reference: ReferenceLegacyARDSClampMirror,
    prepared: list[PreparedDocument],
) -> dict[str, object]:
    checks = {
        "dictionary_matches": 0,
        "cue_matches": 0,
        "assertion_results": 0,
        "full_mirror_outputs": 0,
    }
    for document in prepared:
        reference_dictionary = reference.dictionary.match(
            document.text,
            document.tokens,
            document.sentences,
        )
        _require_equal(
            document.dictionary_entities,
            reference_dictionary,
            document.doc_id,
            "dictionary matches",
        )
        checks["dictionary_matches"] += 1

        reference_cues = reference.assertion.find_cues(document.text, document.tokens)
        _require_equal(document.cues, reference_cues, document.doc_id, "cue matches")
        checks["cue_matches"] += 1

        reference_assertions = reference.assertion.classify(
            document.text,
            document.sentences,
            document.tokens,
            reference_dictionary,
        )
        _require_equal(
            document.asserted_entities,
            reference_assertions,
            document.doc_id,
            "assertion results",
        )
        checks["assertion_results"] += 1

        optimized_output = optimized.run(document.text, doc_id=document.doc_id)
        reference_output = reference.run(document.text, doc_id=document.doc_id)
        _require_equal(
            optimized_output,
            reference_output,
            document.doc_id,
            "full mirror outputs",
        )
        checks["full_mirror_outputs"] += 1
    return {"passed": True, "documents_checked": len(prepared), "checks": checks}


def _require_equal(
    optimized: object,
    reference: object,
    doc_id: str,
    stage: str,
) -> None:
    if optimized != reference:
        raise ReferenceParityError(f"{stage} differ for {doc_id}")


def _optimized_stage_functions(
    mirror: LegacyARDSClampMirror,
    prepared: list[PreparedDocument],
) -> dict[str, Callable[[], int]]:
    return {
        "tokenization": lambda: _time_tokenization(mirror, prepared),
        "sentence_segmentation": lambda: _time_segmentation(mirror, prepared),
        "span_index_build": lambda: _time_span_index_build(prepared),
        "dictionary_matching": lambda: _time_dictionary(mirror.dictionary, prepared),
        "cue_matching": lambda: _time_cues(mirror.assertion, prepared),
        "assertion_classification": lambda: _time_assertion_classification(
            mirror.assertion, prepared
        ),
        "postprocessing": lambda: _time_postprocessing(mirror, prepared),
        "utf16_conversion": lambda: _time_utf16(prepared),
        "batch_serialization": lambda: _time_serialization(prepared),
        "full_mirror": lambda: _time_full_mirror(mirror, prepared),
    }


def _reference_stage_functions(
    mirror: ReferenceLegacyARDSClampMirror,
    prepared: list[PreparedDocument],
) -> dict[str, Callable[[], int]]:
    return {
        "dictionary_matching": lambda: _time_dictionary(mirror.dictionary, prepared),
        "cue_matching": lambda: _time_cues(mirror.assertion, prepared),
        "assertion_classification": lambda: _time_assertion_classification(
            mirror.assertion, prepared
        ),
        "full_mirror": lambda: _time_full_mirror(mirror, prepared),
    }


def _time_tokenization(mirror: LegacyARDSClampMirror, prepared: list[PreparedDocument]) -> int:
    return sum(len(mirror.tokenizer.tokenize(document.text)) for document in prepared)


def _time_segmentation(mirror: LegacyARDSClampMirror, prepared: list[PreparedDocument]) -> int:
    return sum(
        len(mirror.segmenter.segment(document.text, document.tokens)) for document in prepared
    )


def _time_span_index_build(prepared: list[PreparedDocument]) -> int:
    work_units = 0
    for document in prepared:
        span_index = DocumentSpanIndex.build(document.tokens, document.sentences)
        work_units += len(span_index.token_to_sentence) + len(span_index.sentence_token_bounds)
    return work_units


def _time_dictionary(matcher: object, prepared: list[PreparedDocument]) -> int:
    return sum(
        len(
            matcher.match(
                document.text,
                document.tokens,
                document.sentences,
                span_index=document.span_index,
            )
        )
        for document in prepared
    )


def _time_cues(assertion: object, prepared: list[PreparedDocument]) -> int:
    return sum(len(assertion.find_cues(document.text, document.tokens)) for document in prepared)


def _time_assertion_classification(assertion: object, prepared: list[PreparedDocument]) -> int:
    proxy = _PreparedCueAssertionProxy(assertion)
    classify = type(assertion).classify
    count = 0
    for document in prepared:
        proxy.set_cues(document.cues)
        count += len(
            classify(
                proxy,
                document.text,
                document.sentences,
                document.tokens,
                document.dictionary_entities,
                span_index=document.span_index,
            )
        )
    return count


def _time_postprocessing(mirror: LegacyARDSClampMirror, prepared: list[PreparedDocument]) -> int:
    return sum(
        len(
            mirror.ruta.apply(
                document.sentences,
                document.tokens,
                document.asserted_entities,
            )
        )
        for document in prepared
    )


def _time_utf16(prepared: list[PreparedDocument]) -> int:
    checksum = 0
    for document in prepared:
        offsets = Utf16OffsetMap.from_text(document.text)
        for entity in document.final_entities:
            start, end = offsets.span(entity.start, entity.end)
            checksum += start + end
    return checksum


def _time_serialization(prepared: list[PreparedDocument]) -> int:
    entity_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []
    for document in prepared:
        occurrences: dict[tuple[object, ...], int] = {}
        for entity in document.exported_entities:
            entity_text = entity.covered_text(document.text)
            key = (
                entity.start,
                entity.end,
                entity_text,
                entity.semantic_tag,
                entity.assertion,
                entity.cui,
                entity.attribute,
            )
            occurrence = occurrences.get(key, 0)
            occurrences[key] = occurrence + 1
            entity_rows.append(
                {
                    "clamp_doc_id": document.doc_id,
                    "start": entity.start,
                    "end": entity.end,
                    "entity_text": entity_text,
                    "entity_text_sha256": _text_sha256(entity_text),
                    "semantic_tag": entity.semantic_tag,
                    "assertion": entity.assertion,
                    "cui": entity.cui,
                    "attribute": entity.attribute,
                    "duplicate_occurrence": occurrence,
                }
            )
        count = len(document.exported_entities)
        prediction_rows.append(
            {
                "clamp_doc_id": document.doc_id,
                "prediction_status": "evaluable",
                "prediction_label": int(count > 0),
                "clamp_ards_entity_count": count,
                "source_text_sha256": _text_sha256(document.text),
            }
        )

    entity_table = pa.Table.from_pylist(entity_rows, schema=ENTITY_SCHEMA)
    prediction_table = pa.Table.from_pylist(prediction_rows, schema=PREDICTION_SCHEMA)
    for table in (entity_table, prediction_table):
        sink = pa.BufferOutputStream()
        pq.write_table(table, sink, compression="zstd")
        sink.getvalue()
    return len(entity_rows) + len(prediction_rows)


def _time_full_mirror(mirror: LegacyARDSClampMirror, prepared: list[PreparedDocument]) -> int:
    return sum(len(mirror.run(document.text, doc_id=document.doc_id)) for document in prepared)


def _measure(
    function: Callable[[], int],
    *,
    document_count: int,
    token_count: int,
    repeats: int,
    warmups: int,
) -> dict[str, object]:
    expected: int | None = None
    for _ in range(warmups):
        expected = _stable_work_units(expected, function())

    durations: list[float] = []
    for _ in range(repeats):
        gc.collect()
        started = time.perf_counter()
        work_units = function()
        durations.append(time.perf_counter() - started)
        expected = _stable_work_units(expected, work_units)

    gc.collect()
    tracemalloc.start()
    try:
        memory_work_units = function()
        _, peak_bytes = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    expected = _stable_work_units(expected, memory_work_units)

    ordered = sorted(durations)
    median_seconds = statistics.median(ordered)
    q1_seconds = _percentile(ordered, 0.25)
    q3_seconds = _percentile(ordered, 0.75)
    return {
        "repeats": repeats,
        "warmups": warmups,
        "seconds": durations,
        "median_seconds": median_seconds,
        "q1_seconds": q1_seconds,
        "q3_seconds": q3_seconds,
        "iqr_seconds": q3_seconds - q1_seconds,
        "min_seconds": min(ordered),
        "max_seconds": max(ordered),
        "documents_per_second": document_count / median_seconds,
        "tokens_per_second": token_count / median_seconds,
        "python_peak_memory_bytes": peak_bytes,
        "work_units": expected,
    }


def _measure_construction_and_full_run_memory(
    mirror_factory: Callable[[], LegacyARDSClampMirror],
    documents: list[tuple[str, str]],
) -> dict[str, object]:
    gc.collect()
    tracemalloc.start()
    try:
        mirror = mirror_factory()
        work_units = sum(len(mirror.run(text, doc_id=doc_id)) for doc_id, text in documents)
        _, peak_bytes = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
        if "mirror" in locals():
            del mirror
        gc.collect()
    return {
        "python_peak_memory_bytes": peak_bytes,
        "memory_work_units": work_units,
        "memory_measurement_scope": "fresh_mirror_construction_plus_complete_corpus_run",
    }


def _stable_work_units(expected: int | None, observed: int) -> int:
    if expected is not None and observed != expected:
        raise RuntimeError(
            f"Benchmark result changed between repetitions: expected {expected}, got {observed}"
        )
    return observed


def _percentile(ordered: list[float], fraction: float) -> float:
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _comparisons(
    optimized: dict[str, dict[str, object]],
    reference: dict[str, dict[str, object]],
) -> dict[str, object]:
    targets = {"cue_matching": 3.0, "full_mirror": 2.0}
    result: dict[str, object] = {}
    for stage in (
        "dictionary_matching",
        "cue_matching",
        "assertion_classification",
        "full_mirror",
    ):
        optimized_seconds = float(optimized[stage]["median_seconds"])
        reference_seconds = float(reference[stage]["median_seconds"])
        speedup = reference_seconds / optimized_seconds
        comparison: dict[str, object] = {"reference_over_optimized_speedup": speedup}
        if stage in targets:
            comparison["target_speedup"] = targets[stage]
            comparison["target_met"] = speedup >= targets[stage]
        result[stage] = comparison

    optimized_peak = int(optimized["full_mirror"]["python_peak_memory_bytes"])
    reference_peak = int(reference["full_mirror"]["python_peak_memory_bytes"])
    memory_ratio = optimized_peak / reference_peak if reference_peak else None
    absolute_allowance_bytes = 5 * 1024 * 1024
    allowed_peak = reference_peak + max(
        round(reference_peak * 0.10),
        absolute_allowance_bytes,
    )
    result["full_mirror_memory"] = {
        "optimized_over_reference_ratio": memory_ratio,
        "reference_peak_bytes": reference_peak,
        "optimized_peak_bytes": optimized_peak,
        "relative_allowance": 0.10,
        "absolute_allowance_bytes": absolute_allowance_bytes,
        "allowed_peak_bytes": allowed_peak,
        "target_met": optimized_peak <= allowed_peak,
        "measurement_scope": "fresh mirror construction plus one complete corpus run",
    }
    return result


def _fixture_metadata(
    prepared: list[PreparedDocument],
    approximate_tokens: int,
    seed: int,
    resources: ClampResources,
) -> dict[str, object]:
    token_counts = sorted(len(document.tokens) for document in prepared)
    workload_counts = {
        "filler_only": sum(index % 4 < 2 for index in range(len(prepared))),
        "dictionary_only": sum(index % 4 == 2 for index in range(len(prepared))),
        "dictionary_plus_cue": sum(index % 4 == 3 for index in range(len(prepared))),
    }
    interaction_count = workload_counts["dictionary_plus_cue"]
    cue_category_counts = {
        category: sum(
            interaction_index % len(_CUE_CATEGORIES) == category_index
            for interaction_index in range(interaction_count)
        )
        for category_index, category in enumerate(_CUE_CATEGORIES)
    }
    byte_count = sum(len(document.text.encode("utf-8")) for document in prepared)
    digest = hashlib.sha256()
    for document in prepared:
        payload = document.text.encode("utf-8")
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)

    dictionary = _active_dictionary(resources)
    dictionary_document_counts = {entry.index: 0 for entry in dictionary}
    for document_index in range(len(prepared)):
        dictionary_ordinal = _dictionary_ordinal(document_index)
        if dictionary and dictionary_ordinal is not None:
            entry = dictionary[dictionary_ordinal % len(dictionary)]
            dictionary_document_counts[entry.index] += 1
    active_dictionary_indices = [entry.index for entry in dictionary]
    observed_dictionary_indices = [
        index for index in active_dictionary_indices if dictionary_document_counts[index] > 0
    ]
    missing_dictionary_indices = [
        index for index in active_dictionary_indices if dictionary_document_counts[index] == 0
    ]

    cues_by_category = _cues_by_category(resources)
    cue_document_counts = {cue.index: 0 for cue in resources.assertion_cues}
    for interaction_index in range(interaction_count):
        interaction_cue = _interaction_cue(cues_by_category, interaction_index)
        cue_document_counts[interaction_cue.index] += 1
        if interaction_cue.category == "conjunctions":
            termination_negation = cues_by_category["negPhrases"][0]
            cue_document_counts[termination_negation.index] += 1
    observed_cue_indices_by_category = {
        category: [
            cue.index for cue in cues_by_category[category] if cue_document_counts[cue.index] > 0
        ]
        for category in _CUE_CATEGORIES
    }
    missing_cue_indices_by_category = {
        category: [
            cue.index for cue in cues_by_category[category] if cue_document_counts[cue.index] == 0
        ]
        for category in _CUE_CATEGORIES
    }
    observed_cue_categories = [
        category for category in _CUE_CATEGORIES if observed_cue_indices_by_category[category]
    ]
    missing_cue_categories = [
        category for category in _CUE_CATEGORIES if not observed_cue_indices_by_category[category]
    ]

    conjunction_documents = [
        document
        for index, document in enumerate(prepared)
        if index % 4 == 3 and _CUE_CATEGORIES[(index // 4) % len(_CUE_CATEGORIES)] == "conjunctions"
    ]
    conjunction_asserted_entities = [
        entity for document in conjunction_documents for entity in document.asserted_entities
    ]

    return {
        "description": (
            "Deterministic generated text using neutral filler and configured matcher phrases; "
            "contains no source reports"
        ),
        "document_count": len(prepared),
        "utf8_bytes": byte_count,
        "sha256": digest.hexdigest(),
        "seed": seed,
        "requested_approximate_tokens_per_document": approximate_tokens,
        "workload_distribution": workload_counts,
        "interaction_cue_categories": cue_category_counts,
        "assertion_cue_coverage": {
            "configured_cue_count": len(resources.assertion_cues),
            "observed_cue_count": sum(count > 0 for count in cue_document_counts.values()),
            "inserted_cue_occurrence_count": sum(cue_document_counts.values()),
            "configured_categories": list(_CUE_CATEGORIES),
            "observed_categories": observed_cue_categories,
            "missing_categories": missing_cue_categories,
            "observed_cue_indices_by_category": observed_cue_indices_by_category,
            "missing_cue_indices_by_category": missing_cue_indices_by_category,
            "cue_occurrence_count_by_category": {
                category: sum(cue_document_counts[cue.index] for cue in cues_by_category[category])
                for category in _CUE_CATEGORIES
            },
            "all_configured_cues_observed": not any(missing_cue_indices_by_category.values()),
        },
        "conjunction_termination": {
            "document_count": len(conjunction_documents),
            "dictionary_entity_count": len(conjunction_asserted_entities),
            "present_entity_count": sum(
                entity.assertion == "present" for entity in conjunction_asserted_entities
            ),
            "absent_entity_count": sum(
                entity.assertion == "absent" for entity in conjunction_asserted_entities
            ),
            "all_dictionary_entities_present": all(
                entity.assertion == "present" for entity in conjunction_asserted_entities
            ),
        },
        "dictionary_term_coverage": {
            "active_term_count": len(active_dictionary_indices),
            "observed_term_count": len(observed_dictionary_indices),
            "dictionary_bearing_document_count": sum(dictionary_document_counts.values()),
            "active_dictionary_indices": active_dictionary_indices,
            "observed_dictionary_indices": observed_dictionary_indices,
            "missing_dictionary_indices": missing_dictionary_indices,
            "document_count_by_dictionary_index": {
                str(index): dictionary_document_counts[index] for index in active_dictionary_indices
            },
            "all_active_terms_observed": not missing_dictionary_indices,
        },
        "token_distribution": {
            "total": sum(token_counts),
            "minimum": min(token_counts),
            "q1": _percentile([float(value) for value in token_counts], 0.25),
            "median": statistics.median(token_counts),
            "mean": statistics.fmean(token_counts),
            "q3": _percentile([float(value) for value in token_counts], 0.75),
            "maximum": max(token_counts),
        },
    }


def _environment_metadata(resources: ClampResources) -> dict[str, object]:
    return {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "logical_cpu_count": os.cpu_count(),
        "package_version": __version__,
        "dependencies": {
            package: _installed_version(package) for package in ("pyarrow", "regex", "setuptools")
        },
        "git_commit": _git_output("rev-parse", "HEAD"),
        "git_dirty": bool(_git_output("status", "--porcelain")),
        "source_tree_fingerprint": _source_tree_fingerprint(),
        "phenotype_spec_version": resources.phenotype_spec_version,
        "phenotype_spec_sha256": resources.phenotype_spec_sha256,
        "resource_sha256": resources.resource_sha256,
    }


def _installed_version(package: str) -> str | None:
    try:
        return version(package)
    except PackageNotFoundError:
        return None


def _git_output(*arguments: str) -> str | None:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


def _source_tree_fingerprint() -> dict[str, object]:
    head_tree = _required_git_bytes("rev-parse", "HEAD^{tree}").strip()
    tracked_diff = _required_git_bytes(
        "diff",
        "--binary",
        "--no-ext-diff",
        "--no-textconv",
        "HEAD",
        "--",
        *_SOURCE_FINGERPRINT_PATHS,
    )
    raw_untracked = _required_git_bytes(
        "ls-files",
        "-z",
        "--others",
        "--exclude-standard",
        "--",
        *_SOURCE_FINGERPRINT_PATHS,
    )
    untracked_paths = sorted(
        path.decode("utf-8", errors="surrogateescape")
        for path in raw_untracked.split(b"\0")
        if path
    )

    digest = hashlib.sha256()
    _update_length_prefixed(digest, b"head_tree", head_tree)
    _update_length_prefixed(digest, b"tracked_diff", tracked_diff)
    untracked_files: list[dict[str, object]] = []
    for relative_path in untracked_paths:
        path = REPOSITORY_ROOT / relative_path
        payload = (
            os.readlink(path).encode("utf-8", errors="surrogateescape")
            if path.is_symlink()
            else path.read_bytes()
        )
        _update_length_prefixed(
            digest,
            b"untracked_path",
            relative_path.encode("utf-8", errors="surrogateescape"),
        )
        _update_length_prefixed(digest, b"untracked_payload", payload)
        untracked_files.append(
            {
                "path": relative_path,
                "sha256": hashlib.sha256(payload).hexdigest(),
                "bytes": len(payload),
            }
        )
    return {
        "sha256": digest.hexdigest(),
        "method": (
            "SHA-256 over domain-separated, length-prefixed HEAD tree ID, scoped tracked "
            "git diff --binary HEAD, and sorted execution-relevant untracked path/payload bytes"
        ),
        "head_tree": head_tree.decode("ascii"),
        "tracked_diff_sha256": hashlib.sha256(tracked_diff).hexdigest(),
        "tracked_diff_bytes": len(tracked_diff),
        "untracked_pathspecs": list(_SOURCE_FINGERPRINT_PATHS),
        "untracked_files": untracked_files,
    }


def _required_git_bytes(*arguments: str) -> bytes:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        timeout=10,
    )
    if completed.returncode != 0:
        error = completed.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(
            f"Could not fingerprint source tree with git {' '.join(arguments)}: {error}"
        )
    return completed.stdout


def _update_length_prefixed(digest: Any, domain: bytes, payload: bytes) -> None:
    digest.update(len(domain).to_bytes(8, "big"))
    digest.update(domain)
    digest.update(len(payload).to_bytes(8, "big"))
    digest.update(payload)


def _text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _console_summary(result: dict[str, object], output: Path) -> str:
    comparisons = result["comparisons"]
    if not isinstance(comparisons, dict):
        raise TypeError("Benchmark comparisons are invalid")
    cue = comparisons["cue_matching"]
    full = comparisons["full_mirror"]
    if not isinstance(cue, dict) or not isinstance(full, dict):
        raise TypeError("Benchmark comparison rows are invalid")
    status = result.get("status")
    if not isinstance(status, str):
        raise TypeError("Benchmark status is invalid")
    return (
        "CLAMP matcher benchmark complete: "
        f"cue speedup={float(cue['reference_over_optimized_speedup']):.2f}x, "
        f"full-mirror speedup={float(full['reference_over_optimized_speedup']):.2f}x, "
        f"status={status}; "
        f"JSON written to {output}"
    )


if __name__ == "__main__":
    raise SystemExit(main())
