from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

from ards_cxr_benchmark.clamp_ards.parity import REQUIRED_MISMATCH_FIELDS


def test_normalizer_uses_configured_prediction_and_parity_paths(tmp_path: Path) -> None:
    reference, entities, predictions = _write_inputs(tmp_path)
    summary = _write_parity_summary(tmp_path / "configured_summary.json", entities, predictions)
    config = _write_config(tmp_path / "config.yaml", entities, predictions, summary)
    output = tmp_path / "normalized.parquet"

    process = _run_normalizer(config, reference, output)

    assert process.returncode == 0, process.stderr
    normalized = pd.read_parquet(output)
    assert normalized["case_id"].tolist() == ["mimic_1_10"]
    assert normalized["model_name"].tolist() == ["clamp_python_compatibility"]


def test_normalizer_cli_paths_override_config(tmp_path: Path) -> None:
    reference, entities, predictions = _write_inputs(tmp_path)
    valid_summary = _write_parity_summary(tmp_path / "valid_summary.json", entities, predictions)
    stale_entities = tmp_path / "stale_entities.parquet"
    pd.DataFrame({"not": ["entities"]}).to_parquet(stale_entities, index=False)
    stale_predictions = tmp_path / "stale_predictions.parquet"
    pd.DataFrame({"not": ["predictions"]}).to_parquet(stale_predictions, index=False)
    stale_summary = tmp_path / "stale_summary.json"
    stale_summary.write_text("{}", encoding="utf-8")
    config = _write_config(
        tmp_path / "config.yaml", stale_entities, stale_predictions, stale_summary
    )
    output = tmp_path / "normalized.parquet"

    process = _run_normalizer(
        config,
        reference,
        output,
        "--input",
        str(predictions),
        "--entity-input",
        str(entities),
        "--parity-summary",
        str(valid_summary),
    )

    assert process.returncode == 0, process.stderr
    assert output.is_file()


def test_normalizer_rejects_prediction_bytes_not_covered_by_parity(tmp_path: Path) -> None:
    reference, entities, predictions = _write_inputs(tmp_path)
    summary = _write_parity_summary(tmp_path / "summary.json", entities, predictions)
    changed = pd.read_parquet(predictions)
    changed.loc[0, "prediction_label"] = 0
    changed.loc[0, "clamp_ards_entity_count"] = 0
    changed.to_parquet(predictions, index=False)
    config = _write_config(tmp_path / "config.yaml", entities, predictions, summary)

    process = _run_normalizer(config, reference, tmp_path / "normalized.parquet")

    assert process.returncode != 0
    assert "does not match the passing parity summary" in process.stderr


def test_normalizer_rejects_entity_bytes_not_covered_by_parity(tmp_path: Path) -> None:
    reference, entities, predictions = _write_inputs(tmp_path)
    summary = _write_parity_summary(tmp_path / "summary.json", entities, predictions)
    changed = pd.read_parquet(entities)
    changed.loc[0, "entity_text"] = "changed"
    changed.to_parquet(entities, index=False)
    config = _write_config(tmp_path / "config.yaml", entities, predictions, summary)

    process = _run_normalizer(config, reference, tmp_path / "normalized.parquet")

    assert process.returncode != 0
    assert "entity file does not match the passing parity summary" in process.stderr


def test_normalizer_rejects_non_strict_parity_summary(tmp_path: Path) -> None:
    reference, entities, predictions = _write_inputs(tmp_path)
    summary = _write_parity_summary(tmp_path / "summary.json", entities, predictions)
    payload = json.loads(summary.read_text(encoding="utf-8"))
    payload["require_order"] = False
    summary.write_text(json.dumps(payload), encoding="utf-8")
    config = _write_config(tmp_path / "config.yaml", entities, predictions, summary)

    process = _run_normalizer(config, reference, tmp_path / "normalized.parquet")

    assert process.returncode != 0
    assert "requires strict output-order parity" in process.stderr


def test_normalizer_rejects_order_differences(tmp_path: Path) -> None:
    reference, entities, predictions = _write_inputs(tmp_path)
    summary = _write_parity_summary(tmp_path / "summary.json", entities, predictions)
    payload = json.loads(summary.read_text(encoding="utf-8"))
    payload["output_order_differences"] = 1
    summary.write_text(json.dumps(payload), encoding="utf-8")
    config = _write_config(tmp_path / "config.yaml", entities, predictions, summary)

    process = _run_normalizer(config, reference, tmp_path / "normalized.parquet")

    assert process.returncode != 0
    assert "requires zero output-order differences" in process.stderr


def test_normalizer_rejects_passing_summary_with_required_mismatch(tmp_path: Path) -> None:
    reference, entities, predictions = _write_inputs(tmp_path)
    summary = _write_parity_summary(tmp_path / "summary.json", entities, predictions)
    payload = json.loads(summary.read_text(encoding="utf-8"))
    payload["missing_documents"] = 1
    summary.write_text(json.dumps(payload), encoding="utf-8")
    config = _write_config(tmp_path / "config.yaml", entities, predictions, summary)

    process = _run_normalizer(config, reference, tmp_path / "normalized.parquet")

    assert process.returncode != 0
    assert "contains required mismatches" in process.stderr


def test_normalizer_rejects_legacy_summary_without_strict_contract(tmp_path: Path) -> None:
    reference, entities, predictions = _write_inputs(tmp_path)
    summary = tmp_path / "summary.json"
    summary.write_text(
        json.dumps(
            {
                "passed": True,
                "expected_document_count": 1,
                "actual_document_count": 1,
            }
        ),
        encoding="utf-8",
    )
    config = _write_config(tmp_path / "config.yaml", entities, predictions, summary)

    process = _run_normalizer(config, reference, tmp_path / "normalized.parquet")

    assert process.returncode != 0
    assert "requires strict output-order parity" in process.stderr


def _write_inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    reference = tmp_path / "reference.parquet"
    pd.DataFrame(
        {
            "case_id": ["mimic_1_10"],
            "subject_id": [1],
            "study_id": [10],
            "split": ["validation"],
            "source_dataset": ["mimic-cxr"],
        }
    ).to_parquet(reference, index=False)
    entities = tmp_path / "entities.parquet"
    pd.DataFrame(
        {
            "clamp_doc_id": ["s10"],
            "start": [0],
            "end": [4],
            "entity_text": ["ARDS"],
        }
    ).to_parquet(entities, index=False)
    predictions = tmp_path / "predictions.parquet"
    pd.DataFrame(
        {
            "clamp_doc_id": ["s10"],
            "prediction_status": ["evaluable"],
            "prediction_label": [1],
            "clamp_ards_entity_count": [1],
        }
    ).to_parquet(predictions, index=False)
    return reference, entities, predictions


def _write_parity_summary(path: Path, entities: Path, predictions: Path) -> Path:
    entity_sha256 = hashlib.sha256(entities.read_bytes()).hexdigest()
    prediction_sha256 = hashlib.sha256(predictions.read_bytes()).hexdigest()
    path.write_text(
        json.dumps(
            {
                "passed": True,
                "require_order": True,
                "expected_document_count": 1,
                "actual_document_count": 1,
                "exact_entity_document_count": 1,
                "output_order_differences": 0,
                "expected_entities_sha256": entity_sha256,
                "expected_predictions_sha256": prediction_sha256,
                "actual_entities_sha256": entity_sha256,
                "actual_predictions_sha256": prediction_sha256,
                **{field: 0 for field in REQUIRED_MISMATCH_FIELDS},
            }
        ),
        encoding="utf-8",
    )
    return path


def _write_config(path: Path, entities: Path, predictions: Path, summary: Path) -> Path:
    path.write_text(
        "clamp_ards:\n"
        f"  python_entity_output: {entities}\n"
        f"  python_prediction_output: {predictions}\n"
        f"  python_parity_summary: {summary}\n",
        encoding="utf-8",
    )
    return path


def _run_normalizer(
    config: Path,
    reference: Path,
    output: Path,
    *extra: str,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            "scripts/normalize_clamp_python_comparator.py",
            "--config",
            str(config),
            "--reference",
            str(reference),
            "--output",
            str(output),
            "--expected-rows",
            "1",
            *extra,
        ],
        check=False,
        capture_output=True,
        text=True,
    )
