from __future__ import annotations

import argparse
from pathlib import Path

from ards_cxr_benchmark.clamp_ards.parity import (
    compare_clamp_ards_outputs,
    write_parity_result,
)
from ards_cxr_benchmark.config import default_config_path, load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Strict entity-multiset and document-label parity for Python ARDS CLAMP"
    )
    parser.add_argument("--config", type=Path, default=default_config_path())
    parser.add_argument("--expected-entities", type=Path)
    parser.add_argument("--expected-predictions", type=Path)
    parser.add_argument("--actual-entities", type=Path)
    parser.add_argument("--actual-predictions", type=Path)
    parser.add_argument("--summary-output", type=Path)
    parser.add_argument("--markdown-output", type=Path)
    parser.add_argument("--entity-mismatch-output", type=Path)
    parser.add_argument("--document-mismatch-output", type=Path)
    parser.add_argument("--order-mismatch-output", type=Path)
    parser.add_argument(
        "--mismatch-output",
        type=Path,
        help="Optional backward-compatible combined mismatch CSV",
    )
    parser.add_argument(
        "--require-order",
        action="store_true",
        help="Treat entity row-order differences as required parity failures",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    summary_output = args.summary_output or config.clamp_ards.python_parity_summary
    result = compare_clamp_ards_outputs(
        expected_entities=args.expected_entities or config.clamp_ards.teacher_entity_output,
        expected_predictions=(
            args.expected_predictions or config.clamp_ards.teacher_prediction_output
        ),
        actual_entities=args.actual_entities or config.clamp_ards.python_entity_output,
        actual_predictions=args.actual_predictions or config.clamp_ards.python_prediction_output,
        require_order=args.require_order,
    )
    write_parity_result(
        result,
        summary_output=summary_output,
        mismatch_output=args.mismatch_output or config.clamp_ards.python_parity_mismatches,
        entity_mismatch_output=args.entity_mismatch_output,
        document_mismatch_output=args.document_mismatch_output,
        order_mismatch_output=args.order_mismatch_output,
        markdown_output=args.markdown_output,
    )
    summary = result.summary
    print(
        "Python/CLAMP parity: "
        f"{summary['exact_entity_document_count']:,}/"
        f"{summary['expected_document_count']:,} exact entity documents; "
        f"missing entities={summary['missing_entities']:,}; "
        f"unexpected entities={summary['unexpected_entities']:,}; "
        f"count mismatches={summary['document_count_mismatches']:,}; "
        f"label mismatches={summary['document_label_mismatches']:,}; "
        f"order differences={summary['output_order_differences']:,}"
    )
    if not result.passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
