from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from ards_cxr_benchmark.clamp_ards.parity import (
    DOCUMENT_MISMATCH_SCHEMA,
    ENTITY_MISMATCH_SCHEMA,
    ORDER_MISMATCH_SCHEMA,
    compare_clamp_ards_outputs,
    write_parity_result,
)

ENTITY_SCHEMA = pa.schema(
    [
        ("clamp_doc_id", pa.string()),
        ("start", pa.int64()),
        ("end", pa.int64()),
        ("entity_text", pa.string()),
        ("semantic_tag", pa.string()),
        ("assertion", pa.string()),
        ("cui", pa.string()),
        ("attribute", pa.string()),
    ]
)
PREDICTION_SCHEMA = pa.schema(
    [
        ("clamp_doc_id", pa.string()),
        ("prediction_status", pa.string()),
        ("prediction_label", pa.int8()),
        ("clamp_ards_entity_count", pa.int64()),
    ]
)


def test_parity_writer_emits_empty_typed_parquets_and_markdown(tmp_path: Path) -> None:
    paths = _write_tables(
        tmp_path,
        expected_entities=[_entity("ARDS", 0, 4)],
        actual_entities=[_entity("ARDS", 0, 4)],
    )
    result = compare_clamp_ards_outputs(**paths, require_order=True)
    summary = tmp_path / "parity_summary.json"

    write_parity_result(result, summary_output=summary)

    entity_path = tmp_path / "parity_entity_mismatches.parquet"
    document_path = tmp_path / "parity_document_mismatches.parquet"
    order_path = tmp_path / "parity_order_mismatches.parquet"
    assert result.passed
    assert pq.read_schema(entity_path) == ENTITY_MISMATCH_SCHEMA
    assert pq.read_schema(document_path) == DOCUMENT_MISMATCH_SCHEMA
    assert pq.read_schema(order_path) == ORDER_MISMATCH_SCHEMA
    assert pq.read_table(entity_path).num_rows == 0
    assert pq.read_table(document_path).num_rows == 0
    assert pq.read_table(order_path).num_rows == 0
    payload = json.loads(summary.read_text(encoding="utf-8"))
    assert payload["passed"] is True
    for name, path in paths.items():
        assert payload[f"{name}_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    markdown = summary.with_suffix(".md").read_text(encoding="utf-8")
    assert "Strict status: **PASS**" in markdown
    assert "s1" not in markdown
    assert not list(tmp_path.glob(".*.partial"))


def test_parity_separates_entity_and_document_mismatches_without_text(
    tmp_path: Path,
) -> None:
    paths = _write_tables(
        tmp_path,
        expected_entities=[_entity("infiltrates", 10, 21)],
        actual_entities=[],
        expected_label=1,
        actual_label=0,
    )

    result = compare_clamp_ards_outputs(**paths)

    assert not result.passed
    assert {row["mismatch_type"] for row in result.entity_mismatches} == {
        "missing_entity",
    }
    assert [row["mismatch_type"] for row in result.document_mismatches] == [
        "document_label_mismatch",
        "document_count_mismatch",
    ]
    assert result.order_mismatches == ()
    serialized = json.dumps(result.mismatches)
    assert "infiltrates" not in serialized
    assert "opacities" not in serialized


def test_order_difference_is_diagnostic_unless_required(tmp_path: Path) -> None:
    first = _entity("ARDS", 0, 4)
    second = _entity("edema", 5, 10)
    paths = _write_tables(
        tmp_path,
        expected_entities=[first, second],
        actual_entities=[second, first],
    )

    diagnostic = compare_clamp_ards_outputs(**paths)
    required = compare_clamp_ards_outputs(**paths, require_order=True)

    assert diagnostic.passed
    assert diagnostic.summary["output_order_differences"] == 1
    assert diagnostic.summary["output_order_mismatch_positions"] == 2
    assert len(diagnostic.order_mismatches) == 2
    assert not required.passed
    assert required.summary["require_order"] is True


@pytest.mark.parametrize("table_name", ["expected_entities", "actual_entities"])
def test_parity_requires_explicit_attribute_column(
    tmp_path: Path,
    table_name: str,
) -> None:
    paths = _write_tables(
        tmp_path,
        expected_entities=[_entity("ARDS", 0, 4)],
        actual_entities=[_entity("ARDS", 0, 4)],
    )
    entity_path = paths[table_name]
    table_without_attribute = pq.read_table(entity_path).drop_columns(["attribute"])
    pq.write_table(table_without_attribute, entity_path)

    with pytest.raises(ValueError, match=r"missing column\(s\) \['attribute'\]"):
        compare_clamp_ards_outputs(**paths)


def test_cli_writes_split_outputs_legacy_csv_and_returns_nonzero(tmp_path: Path) -> None:
    paths = _write_tables(
        tmp_path,
        expected_entities=[_entity("infiltrates", 10, 21)],
        actual_entities=[_entity("opacities", 10, 21)],
    )
    summary = tmp_path / "summary.json"
    markdown = tmp_path / "summary.md"
    entity_output = tmp_path / "entity.parquet"
    document_output = tmp_path / "document.parquet"
    order_output = tmp_path / "order.parquet"
    legacy_output = tmp_path / "legacy.csv"

    process = subprocess.run(
        [
            sys.executable,
            "scripts/compare_clamp_python_parity.py",
            "--config",
            "config/config.example.yaml",
            "--expected-entities",
            str(paths["expected_entities"]),
            "--expected-predictions",
            str(paths["expected_predictions"]),
            "--actual-entities",
            str(paths["actual_entities"]),
            "--actual-predictions",
            str(paths["actual_predictions"]),
            "--summary-output",
            str(summary),
            "--markdown-output",
            str(markdown),
            "--entity-mismatch-output",
            str(entity_output),
            "--document-mismatch-output",
            str(document_output),
            "--order-mismatch-output",
            str(order_output),
            "--mismatch-output",
            str(legacy_output),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert process.returncode == 1
    assert all(
        path.is_file()
        for path in (
            summary,
            markdown,
            entity_output,
            document_output,
            order_output,
            legacy_output,
        )
    )
    combined_output = legacy_output.read_text(encoding="utf-8") + markdown.read_text(
        encoding="utf-8"
    )
    assert "infiltrates" not in combined_output
    assert "opacities" not in combined_output
    assert pq.read_table(entity_output).num_rows == 3
    assert pq.read_table(document_output).num_rows == 0
    assert pq.read_table(order_output).num_rows == 0
    assert not list(tmp_path.glob(".*.partial"))


def test_cli_require_order_returns_nonzero_for_order_only_difference(tmp_path: Path) -> None:
    first = _entity("ARDS", 0, 4)
    second = _entity("edema", 5, 10)
    paths = _write_tables(
        tmp_path,
        expected_entities=[first, second],
        actual_entities=[second, first],
    )
    summary = tmp_path / "summary.json"
    order_output = tmp_path / "order.parquet"

    process = subprocess.run(
        [
            sys.executable,
            "scripts/compare_clamp_python_parity.py",
            "--config",
            "config/config.example.yaml",
            "--expected-entities",
            str(paths["expected_entities"]),
            "--expected-predictions",
            str(paths["expected_predictions"]),
            "--actual-entities",
            str(paths["actual_entities"]),
            "--actual-predictions",
            str(paths["actual_predictions"]),
            "--summary-output",
            str(summary),
            "--mismatch-output",
            str(tmp_path / "legacy.csv"),
            "--order-mismatch-output",
            str(order_output),
            "--require-order",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert process.returncode == 1
    written = json.loads(summary.read_text(encoding="utf-8"))
    assert written["require_order"] is True
    assert written["output_order_differences"] == 1
    assert pq.read_table(order_output).num_rows == 2


def _entity(text: str, start: int, end: int) -> dict[str, object]:
    return {
        "clamp_doc_id": "s1",
        "start": start,
        "end": end,
        "entity_text": text,
        "semantic_tag": "ARDS",
        "assertion": "present",
        "cui": None,
        "attribute": None,
    }


def _write_tables(
    root: Path,
    *,
    expected_entities: list[dict[str, object]],
    actual_entities: list[dict[str, object]],
    expected_label: int = 1,
    actual_label: int = 1,
) -> dict[str, Path]:
    expected_entity_path = root / "expected_entities.parquet"
    actual_entity_path = root / "actual_entities.parquet"
    expected_prediction_path = root / "expected_predictions.parquet"
    actual_prediction_path = root / "actual_predictions.parquet"
    pq.write_table(
        pa.Table.from_pylist(expected_entities, schema=ENTITY_SCHEMA), expected_entity_path
    )
    pq.write_table(pa.Table.from_pylist(actual_entities, schema=ENTITY_SCHEMA), actual_entity_path)
    pq.write_table(
        _prediction_table(expected_label, len(expected_entities)), expected_prediction_path
    )
    pq.write_table(_prediction_table(actual_label, len(actual_entities)), actual_prediction_path)
    return {
        "expected_entities": expected_entity_path,
        "expected_predictions": expected_prediction_path,
        "actual_entities": actual_entity_path,
        "actual_predictions": actual_prediction_path,
    }


def _prediction_table(label: int, count: int) -> pa.Table:
    return pa.Table.from_pylist(
        [
            {
                "clamp_doc_id": "s1",
                "prediction_status": "evaluable",
                "prediction_label": label,
                "clamp_ards_entity_count": count,
            }
        ],
        schema=PREDICTION_SCHEMA,
    )
