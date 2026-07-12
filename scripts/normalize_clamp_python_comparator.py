from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from ards_cxr_benchmark.clamp_ards.parity import REQUIRED_MISMATCH_FIELDS
from ards_cxr_benchmark.comparators.common import (
    normalize_clamp_python_predictions,
    repository_state,
)
from ards_cxr_benchmark.config import default_config_path, ensure_parent_dir, load_config
from ards_cxr_benchmark.paths import get_paths


def parse_args() -> argparse.Namespace:
    root = get_paths().root
    parser = argparse.ArgumentParser(
        description="Normalize exact-parity Python CLAMP predictions to the comparator contract"
    )
    parser.add_argument("--config", type=Path, default=default_config_path())
    parser.add_argument(
        "--reference",
        type=Path,
        default=root / "data/derived/comparators/mimic_cxr_silver_reference.parquet",
    )
    parser.add_argument(
        "--input",
        type=Path,
    )
    parser.add_argument(
        "--entity-input",
        type=Path,
    )
    parser.add_argument(
        "--parity-summary",
        type=Path,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "data/derived/comparators/clamp_python_compatibility_predictions.parquet",
    )
    parser.add_argument("--expected-rows", type=int, default=227_835)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    prediction_input = args.input or config.clamp_ards.python_prediction_output
    entity_input = args.entity_input or config.clamp_ards.python_entity_output
    parity_summary = args.parity_summary or config.clamp_ards.python_parity_summary
    parity = json.loads(parity_summary.read_text(encoding="utf-8"))
    if parity.get("passed") is not True:
        raise ValueError("Python CLAMP predictions require a passing restricted parity summary")
    if parity.get("require_order") is not True:
        raise ValueError(
            "Python CLAMP compatibility normalization requires strict output-order parity"
        )
    if _required_integer(parity, "output_order_differences") != 0:
        raise ValueError(
            "Python CLAMP compatibility normalization requires zero output-order differences"
        )
    mismatch_counts = {
        field: _required_integer(parity, field) for field in REQUIRED_MISMATCH_FIELDS
    }
    nonzero_mismatches = {field: count for field, count in mismatch_counts.items() if count != 0}
    if nonzero_mismatches:
        raise ValueError(
            f"Python CLAMP parity summary contains required mismatches: {nonzero_mismatches}"
        )
    for field in ("expected_document_count", "actual_document_count"):
        if _required_integer(parity, field) != args.expected_rows:
            raise ValueError(
                f"Parity summary {field} must equal {args.expected_rows:,}; "
                f"found {parity.get(field)!r}"
            )
    if _required_integer(parity, "exact_entity_document_count") != args.expected_rows:
        raise ValueError(
            "Parity summary exact_entity_document_count must equal "
            f"{args.expected_rows:,}; found {parity.get('exact_entity_document_count')!r}"
        )
    parity_hashes = {
        field: _required_sha256(parity, field)
        for field in (
            "expected_entities_sha256",
            "expected_predictions_sha256",
            "actual_entities_sha256",
            "actual_predictions_sha256",
        )
    }
    prediction_sha256 = _file_sha256(prediction_input)
    if prediction_sha256 != parity_hashes["actual_predictions_sha256"]:
        raise ValueError(
            "Python CLAMP prediction file does not match the passing parity summary: "
            f"expected_sha256={parity_hashes['actual_predictions_sha256']}, "
            f"actual_sha256={prediction_sha256}"
        )
    entity_sha256 = _file_sha256(entity_input)
    if entity_sha256 != parity_hashes["actual_entities_sha256"]:
        raise ValueError(
            "Python CLAMP entity file does not match the passing parity summary: "
            f"expected_sha256={parity_hashes['actual_entities_sha256']}, "
            f"actual_sha256={entity_sha256}"
        )
    state = repository_state(get_paths().root)
    run_id = datetime.now(UTC).strftime("clamp-python-compatibility-%Y%m%dT%H%M%SZ")
    normalized = normalize_clamp_python_predictions(
        pd.read_parquet(prediction_input),
        pd.read_parquet(args.reference),
        run_id=run_id,
        source_commit=state["commit"],
    )
    if len(normalized) != args.expected_rows:
        raise ValueError(
            f"Expected {args.expected_rows:,} Python CLAMP rows, found {len(normalized):,}"
        )
    ensure_parent_dir(args.output)
    normalized.to_parquet(args.output, index=False)
    print(f"Wrote {len(normalized):,} Python CLAMP compatibility rows to {args.output}")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _required_integer(payload: dict[str, object], field: str) -> int:
    value = payload.get(field)
    if isinstance(value, bool):
        raise ValueError(f"Python CLAMP parity summary has invalid {field}: {value!r}")
    try:
        parsed = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Python CLAMP parity summary is missing or has invalid {field}: {value!r}"
        ) from exc
    if isinstance(value, float) and not value.is_integer():
        raise ValueError(f"Python CLAMP parity summary has invalid {field}: {value!r}")
    return parsed


def _required_sha256(payload: dict[str, object], field: str) -> str:
    value = payload.get(field)
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(
            f"Python CLAMP parity summary is missing or has invalid {field}; "
            "rerun strict restricted parity"
        )
    return value


if __name__ == "__main__":
    main()
