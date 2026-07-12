from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import pickle
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Isolated Amaral model inference runner")
    parser.add_argument("--external-repo", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--vectorizer", type=Path, required=True)
    parser.add_argument("--expected-model-sha256", required=True)
    parser.add_argument("--expected-vectorizer-sha256", required=True)
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--preprocessing-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--mode",
        choices=["published_mimic_preprocessing", "raw_text_direct"],
        required=True,
    )
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--nltk-data", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    _verify_sha256(args.model, args.expected_model_sha256)
    _verify_sha256(args.vectorizer, args.expected_vectorizer_sha256)
    os.environ["NLTK_DATA"] = str(args.nltk_data)
    os.chdir(args.external_repo)
    sys.path.insert(0, str(args.external_repo))

    # The vectorizer pickle references this exact pinned upstream module.
    __import__("custom_functions")
    from src.segmentation_tools import (  # noqa: PLC0415
        handle_subsection_titles,
        refine_cleaning,
        remove_dictation,
        remove_easy_sections,
        remove_lines_on_other_organs,
        remove_sections_n_duplicate_lines,
        remove_stopwords,
        stem_indicator_words,
    )

    with args.vectorizer.open("rb") as handle:
        vectorizer = pickle.load(handle)  # noqa: S301
    with args.model.open("rb") as handle:
        model = pickle.load(handle)  # noqa: S301
    classes = [int(value) for value in model.classes_]
    if classes != [0, 1]:
        raise ValueError(f"Unexpected Amaral model classes: {classes}")

    preprocessing = json.loads(args.preprocessing_config.read_text(encoding="utf-8"))
    functions = {
        "remove_easy_sections": remove_easy_sections,
        "handle_subsection_titles": handle_subsection_titles,
        "remove_lines_on_other_organs": remove_lines_on_other_organs,
        "stem_indicator_words": stem_indicator_words,
        "remove_stopwords": remove_stopwords,
        "remove_sections_n_duplicate_lines": remove_sections_n_duplicate_lines,
        "refine_cleaning": refine_cleaning,
        "remove_dictation": remove_dictation,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temp = args.output.with_name(f".{args.output.name}.partial")
    try:
        with temp.open("w", encoding="utf-8", newline="\n") as output:
            batch: list[dict[str, Any]] = []
            for record in _iter_packet(args.packet):
                batch.append(record)
                if len(batch) >= args.batch_size:
                    _predict_batch(
                        batch,
                        output=output,
                        mode=args.mode,
                        preprocessing=preprocessing,
                        functions=functions,
                        vectorizer=vectorizer,
                        model=model,
                    )
                    batch = []
            if batch:
                _predict_batch(
                    batch,
                    output=output,
                    mode=args.mode,
                    preprocessing=preprocessing,
                    functions=functions,
                    vectorizer=vectorizer,
                    model=model,
                )
        os.replace(temp, args.output)
    finally:
        temp.unlink(missing_ok=True)


def _predict_batch(
    records: list[dict[str, Any]],
    *,
    output: Any,
    mode: str,
    preprocessing: dict[str, Any],
    functions: dict[str, Any],
    vectorizer: Any,
    model: Any,
) -> None:
    processed: list[str] = []
    retained_counts: list[int] = []
    for record in records:
        report = str(record["report_text"])
        if mode == "published_mimic_preprocessing":
            text, retained_count = _published_preprocess(
                report,
                preprocessing=preprocessing,
                functions=functions,
            )
        else:
            text, retained_count = _final_vectorizer_text(report), 0
        processed.append(text)
        retained_counts.append(retained_count)
    features = vectorizer.transform(processed).toarray()
    scores = model.predict_proba(features)[:, 1]
    labels = model.predict(features)
    for record, text, retained_count, score, label in zip(
        records, processed, retained_counts, scores, labels, strict=True
    ):
        output.write(
            json.dumps(
                {
                    "case_id": str(record["case_id"]),
                    "prediction_score": float(score),
                    "raw_predicted_class": int(label),
                    "retained_statement_count": int(retained_count),
                    "preprocessing_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                },
                separators=(",", ":"),
            )
            + "\n"
        )


def _published_preprocess(
    report: str,
    *,
    preprocessing: dict[str, Any],
    functions: dict[str, Any],
) -> tuple[str, int]:
    note = functions["remove_easy_sections"](
        report,
        [tuple(value) for value in preprocessing["section_order_mimic_iii"]],
    )
    statements = note.split(".")
    statements = functions["handle_subsection_titles"](statements)
    statements = functions["remove_lines_on_other_organs"](
        statements, set(preprocessing["exclusion_set"])
    )
    statements = functions["stem_indicator_words"](statements, preprocessing["targeted_stemming"])
    statements = functions["remove_stopwords"](
        statements,
        preprocessing["complex_stopwords"],
        preprocessing["simple_stopwords"],
    )
    statements = functions["remove_sections_n_duplicate_lines"](statements)
    statements = functions["refine_cleaning"](statements, preprocessing["useless_statements"])
    statements = functions["remove_dictation"](statements, preprocessing["dictation"])
    return _final_vectorizer_text(str(statements)), len(statements)


def _final_vectorizer_text(value: str) -> str:
    return value.replace("'", "").replace("[", "").replace("]", "").replace(",", "")


def _iter_packet(path: Path) -> Iterator[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _verify_sha256(path: Path, expected: str) -> None:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    observed = digest.hexdigest()
    if observed != expected:
        raise ValueError(
            f"Checksum mismatch for {path.name}: expected {expected}, found {observed}"
        )


if __name__ == "__main__":
    main()
