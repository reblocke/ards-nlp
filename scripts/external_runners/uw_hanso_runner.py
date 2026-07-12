from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Legacy UW HANSO probability runner")
    parser.add_argument("--external-repo", type=Path, required=True)
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scope", choices=["impression_findings", "full_report"], required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    os.chdir(args.external_repo)
    sys.path.insert(0, str(args.external_repo))
    from process import DocumentProcessor  # noqa: PLC0415

    processor = DocumentProcessor()
    text_key = (
        "target_text_impression_findings" if args.scope == "impression_findings" else "report_text"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temp = args.output.with_name(f".{args.output.name}.partial")
    try:
        with temp.open("w", encoding="utf-8", newline="\n") as output:
            batch: list[dict[str, Any]] = []
            for record in _iter_packet(args.packet):
                batch.append(record)
                if len(batch) >= args.batch_size:
                    _write_batch(processor, batch, text_key=text_key, output=output)
                    batch = []
            if batch:
                _write_batch(processor, batch, text_key=text_key, output=output)
        os.replace(temp, args.output)
    finally:
        temp.unlink(missing_ok=True)


def _write_batch(
    processor: Any, batch: list[dict[str, Any]], *, text_key: str, output: Any
) -> None:
    texts = [str(record[text_key]) for record in batch]
    probabilities = processor.model.prob(texts)
    if len(probabilities) != len(batch):
        raise ValueError("UW HANSO batch output count does not match its input count")
    for record, probability in zip(batch, probabilities, strict=True):
        doc_labels = probability["doc_labels"]
        infiltrates = doc_labels["infiltrates"]
        extraparenchymal = doc_labels.get("extraparenchymal", {})
        raw_infiltrates = max(infiltrates, key=infiltrates.get)
        raw_extraparenchymal = (
            max(extraparenchymal, key=extraparenchymal.get) if extraparenchymal else None
        )
        row = {
            "case_id": str(record["case_id"]),
            **{f"prob_infiltrates_{key}": float(value) for key, value in infiltrates.items()},
            **{
                f"prob_extraparenchymal_{key}": float(value)
                for key, value in extraparenchymal.items()
            },
            "raw_predicted_infiltrates_class": raw_infiltrates,
            "raw_predicted_extraparenchymal_class": raw_extraparenchymal,
        }
        output.write(json.dumps(row, separators=(",", ":")) + "\n")


def _iter_packet(path: Path):
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


if __name__ == "__main__":
    main()
