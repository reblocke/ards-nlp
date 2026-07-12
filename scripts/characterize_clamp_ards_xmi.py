from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

from ards_cxr_benchmark.clamp_ards.batch import iter_input_documents
from ards_cxr_benchmark.clamp_ards.pipeline import load_legacy_mirror
from ards_cxr_benchmark.clamp_ards.tokenization import Utf16OffsetMap
from ards_cxr_benchmark.clamp_ards.xmi import parse_clamp_xmi
from ards_cxr_benchmark.config import default_config_path, load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stream restricted CLAMP XMI and characterize Python span/final-entity parity"
    )
    parser.add_argument("--config", type=Path, default=default_config_path())
    parser.add_argument("--archive", type=Path)
    parser.add_argument(
        "--oracle-manifest",
        type=Path,
        default=Path("config/clamp_ards_oracle_manifest.json"),
    )
    parser.add_argument("--input", type=Path)
    parser.add_argument("--id-col", default="study_id")
    parser.add_argument("--text-col", default="report_text")
    parser.add_argument("--id-prefix", default="s")
    parser.add_argument("--limit", type=int, default=5000)
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path("artifacts/restricted/clamp_ards/python/xmi_characterization.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.limit < 1:
        raise ValueError("limit must be at least 1")
    config = load_config(args.config)
    archive_path = args.archive or config.clamp_ards.full_output_archive
    input_path = args.input or config.clamp_ards.python_input
    oracle_manifest = json.loads(args.oracle_manifest.read_text(encoding="utf-8"))
    if archive_path.name != oracle_manifest["archive_basename"]:
        raise ValueError(f"Unexpected CLAMP oracle basename: {archive_path.name}")
    if archive_path.stat().st_size != oracle_manifest["archive_bytes"]:
        raise ValueError("CLAMP oracle byte size differs from the frozen manifest")
    if _file_sha256(archive_path) != oracle_manifest["archive_sha256"]:
        raise ValueError("CLAMP oracle SHA-256 differs from the frozen manifest")
    mirror = load_legacy_mirror()
    counts = {
        "document_count": 0,
        "source_text_mismatches": 0,
        "exact_sentence_documents": 0,
        "expected_sentence_count": 0,
        "actual_sentence_count": 0,
        "exact_token_documents": 0,
        "expected_token_count": 0,
        "actual_token_count": 0,
        "exact_final_entity_documents": 0,
        "expected_final_entity_count": 0,
        "actual_final_entity_count": 0,
    }
    documents = iter_input_documents(
        input_path,
        id_column=args.id_col,
        text_column=args.text_col,
        id_prefix=args.id_prefix,
    )
    with zipfile.ZipFile(archive_path) as archive:
        txt_count = sum(name.endswith(".txt") for name in archive.namelist())
        xmi_count = sum(name.endswith(".xmi") for name in archive.namelist())
        if txt_count != oracle_manifest["txt_member_count"]:
            raise ValueError("CLAMP oracle TXT member count differs from the frozen manifest")
        if xmi_count != oracle_manifest["xmi_member_count"]:
            raise ValueError("CLAMP oracle XMI member count differs from the frozen manifest")
        for doc_id, source_text in documents:
            if counts["document_count"] >= args.limit:
                break
            xmi = parse_clamp_xmi(archive.read(f"Output/{doc_id}.xmi"))
            trace = mirror.trace(source_text)
            offsets = Utf16OffsetMap.from_text(source_text)
            counts["document_count"] += 1
            counts["source_text_mismatches"] += int(source_text != xmi.text)
            expected_sentences = [(item.start, item.end) for item in xmi.sentences]
            actual_sentences = [offsets.span(item.start, item.end) for item in trace.sentences]
            counts["exact_sentence_documents"] += int(expected_sentences == actual_sentences)
            counts["expected_sentence_count"] += len(expected_sentences)
            counts["actual_sentence_count"] += len(actual_sentences)
            expected_tokens = [(item.start, item.end) for item in xmi.tokens]
            actual_tokens = [offsets.span(item.start, item.end) for item in trace.tokens]
            counts["exact_token_documents"] += int(expected_tokens == actual_tokens)
            counts["expected_token_count"] += len(expected_tokens)
            counts["actual_token_count"] += len(actual_tokens)
            expected_entities = [
                (
                    item.start,
                    item.end,
                    item.semantic_tag,
                    item.assertion,
                    item.cui,
                    item.attribute,
                )
                for item in xmi.entities
            ]
            actual_entities = [
                (
                    *offsets.span(item.start, item.end),
                    item.semantic_tag,
                    item.assertion,
                    item.cui,
                    item.attribute,
                )
                for item in trace.final_entities
            ]
            counts["exact_final_entity_documents"] += int(expected_entities == actual_entities)
            counts["expected_final_entity_count"] += len(expected_entities)
            counts["actual_final_entity_count"] += len(actual_entities)

    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(
        json.dumps(counts, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(counts, indent=2, sort_keys=True))


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
