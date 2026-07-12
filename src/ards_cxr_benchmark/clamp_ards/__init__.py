"""Deterministic Python compatibility port of the legacy ARDS CLAMP pipeline."""

from .batch import run_clamp_ards_batch
from .parity import compare_clamp_ards_outputs
from .pipeline import (
    load_legacy_mirror,
    predict_legacy_ards_label,
    run_legacy_ards_clamp_mirror,
)
from .types import EntitySpan, SentenceSpan, TokenSpan

__all__ = [
    "EntitySpan",
    "SentenceSpan",
    "TokenSpan",
    "compare_clamp_ards_outputs",
    "load_legacy_mirror",
    "predict_legacy_ards_label",
    "run_clamp_ards_batch",
    "run_legacy_ards_clamp_mirror",
]
