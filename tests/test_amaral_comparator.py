from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from ards_cxr_benchmark.comparators.amaral import (
    AMARAL_PRIMARY_MODEL,
    AMARAL_SENSITIVITY_MODEL,
    build_amaral_predictions,
    extract_published_preprocessing_config,
    validate_amaral_anchor_predictions,
    write_amaral_run_outputs,
)
from ards_cxr_benchmark.comparators.config import load_amaral_config


def test_example_config_loads_with_pinned_resources() -> None:
    config = load_amaral_config(
        Path("config/external_comparators/amaral_ards_diagnosis.example.yaml")
    )

    assert config.source.commit == "6154ac32e16dd9497a466351582603e1c1095a05"
    assert config.expected_sha256["model"].startswith("a72c2fe1")
    assert config.expected_sha256["vectorizer"].startswith("ddc982d5")


def test_extract_published_preprocessing_config(tmp_path: Path) -> None:
    notebook = {
        "cells": [
            {
                "cell_type": "code",
                "source": [
                    "section_order = {}\n",
                    "section_order['mimic_iii'] = [('finding:', True)]\n",
                    "exclusion_set = {'heart', 'line'}\n",
                    "targeted_stemming = {'opacities': 'opacity'}\n",
                    "complex_stopwords = ['there is ']\n",
                    "simple_stopwords = ['the']\n",
                    "useless_statements = ['', 'clinic']\n",
                    "dictation = 'dictated at hospital'\n",
                ],
            }
        ]
    }
    path = tmp_path / "notebook.ipynb"
    path.write_text(json.dumps(notebook), encoding="utf-8")

    result = extract_published_preprocessing_config(path)

    assert result["section_order_mimic_iii"] == [("finding:", True)]
    assert result["exclusion_set"] == ["heart", "line"]
    assert result["targeted_stemming"] == {"opacities": "opacity"}


def test_build_amaral_predictions_preserves_modes_without_text(tmp_path: Path) -> None:
    config = load_amaral_config(
        Path("config/external_comparators/amaral_ards_diagnosis.example.yaml")
    )
    manifest = pd.DataFrame(
        {
            "case_id": ["mimic_1_10", "mimic_2_20"],
            "subject_id": [1, 2],
            "study_id": [10, 20],
            "split": ["validation", "test"],
            "source_dataset": ["mimic_cxr", "mimic_cxr"],
        }
    )
    outputs: list[tuple[str, Path]] = []
    for mode, scores in [
        ("published_mimic_preprocessing", [0.2, 0.8]),
        ("raw_text_direct", [0.3, 0.7]),
    ]:
        path = tmp_path / f"{mode}.jsonl"
        pd.DataFrame(
            {
                "case_id": manifest["case_id"],
                "prediction_score": scores,
                "raw_predicted_class": [0, 1],
                "retained_statement_count": [1, 2],
                "preprocessing_sha256": ["a" * 64, "b" * 64],
            }
        ).to_json(path, orient="records", lines=True)
        outputs.append((mode, path))

    result = build_amaral_predictions(
        runner_outputs=outputs,
        manifest=manifest,
        config=config,
        run_id="run",
        repository_commit=config.source.commit,
    )

    assert set(result["model_name"]) == {AMARAL_PRIMARY_MODEL, AMARAL_SENSITIVITY_MODEL}
    assert len(result) == 4
    assert "report_text" not in result.columns
    assert result.groupby("model_name")["prediction_label"].sum().to_dict() == {
        AMARAL_PRIMARY_MODEL: 1,
        AMARAL_SENSITIVITY_MODEL: 1,
    }


def test_anchor_parity_rejects_score_drift(tmp_path: Path) -> None:
    expected_path = tmp_path / "expected.json"
    expected_path.write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "case_id": "mimic_1_10",
                        "model_name": AMARAL_PRIMARY_MODEL,
                        "prediction_score": 0.25,
                        "prediction_label": 0,
                        "raw_predicted_class": 0,
                        "retained_statement_count": 1,
                        "preprocessing_sha256": "a" * 64,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    observed = pd.DataFrame(json.loads(expected_path.read_text())["rows"])

    assert validate_amaral_anchor_predictions(observed, expected_path)["status"] == "passed"
    observed.loc[0, "prediction_score"] = 0.5
    with pytest.raises(ValueError, match="prediction_score"):
        validate_amaral_anchor_predictions(observed, expected_path)


def test_single_mode_requires_noncanonical_output_paths() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_amaral_comparator.py",
            "--config",
            "config/external_comparators/amaral_ards_diagnosis.example.yaml",
            "--mode",
            "raw_text_direct",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "require explicit noncanonical output paths" in result.stderr


def test_single_mode_status_uses_actual_model_and_row_count(tmp_path: Path) -> None:
    config = load_amaral_config(
        Path("config/external_comparators/amaral_ards_diagnosis.example.yaml")
    )
    config = replace(
        config,
        prediction_output=tmp_path / "predictions.parquet",
        artifact_dir=tmp_path / "artifacts",
    )
    manifest = pd.DataFrame(
        {
            "case_id": ["mimic_1_10", "mimic_2_20"],
            "subject_id": [1, 2],
            "study_id": [10, 20],
            "split": ["validation", "test"],
            "source_dataset": ["mimic_cxr", "mimic_cxr"],
        }
    )
    output = tmp_path / "raw.jsonl"
    pd.DataFrame(
        {
            "case_id": manifest["case_id"],
            "prediction_score": [0.2, 0.8],
            "raw_predicted_class": [0, 1],
            "retained_statement_count": [0, 0],
            "preprocessing_sha256": ["a" * 64, "b" * 64],
        }
    ).to_json(output, orient="records", lines=True)
    predictions = build_amaral_predictions(
        runner_outputs=[("raw_text_direct", output)],
        manifest=manifest,
        config=config,
        run_id="partial-run",
        repository_commit=config.source.commit,
    )

    write_amaral_run_outputs(
        predictions=predictions,
        config=config,
        resource_verification={},
        run_id="partial-run",
        command=["test"],
        run_kind="partial",
    )

    status = json.loads((config.artifact_dir / "status.json").read_text())
    assert status["name"] == AMARAL_SENSITIVITY_MODEL
    assert status["details"]["prediction_rows"] == 2
    assert status["details"]["prediction_rows_by_model"] == {AMARAL_SENSITIVITY_MODEL: 2}
