from __future__ import annotations

from .types import TokenSpan


class ClampPosNoOp:
    """Explicit compatibility stage for the output-invariant legacy POS component."""

    @staticmethod
    def apply(tokens: list[TokenSpan]) -> list[TokenSpan]:
        return tokens
