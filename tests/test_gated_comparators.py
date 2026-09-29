from __future__ import annotations

import pickle
import subprocess
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC

from ards_cxr_benchmark.comparators.afshar import (
    afshar_container_command,
    inspect_pickle_artifact,
)
from ards_cxr_benchmark.comparators.config import (
    ExternalRepositoryConfig,
    load_afshar_config,
    load_uw_hanso_config,
)
from ards_cxr_benchmark.comparators.external import verify_external_repository
from ards_cxr_benchmark.comparators.uw_hanso import (
    build_uw_hanso_predictions,
)


def test_hanso_probability_mapping_and_sum_validation(tmp_path: Path) -> None:
    config = load_uw_hanso_config(Path("config/external_comparators/uw_hanso.example.yaml"))
    manifest = pd.DataFrame(
        {
            "case_id": ["mimic_1_10"],
            "subject_id": [1],
            "study_id": [10],
            "split": ["test"],
            "source_dataset": ["mimic_cxr"],
        }
    )
    path = tmp_path / "hanso.jsonl"
    pd.DataFrame(
        {
            "case_id": ["mimic_1_10"],
            "prob_infiltrates_none": [0.1],
            "prob_infiltrates_present": [0.1],
            "prob_infiltrates_unilateral": [0.2],
            "prob_infiltrates_bilateral": [0.6],
            "raw_predicted_infiltrates_class": ["bilateral"],
        }
    ).to_json(path, orient="records", lines=True)

    result = build_uw_hanso_predictions(
        runner_outputs=[("full_report", path)],
        manifest=manifest,
        config=config,
        run_id="run",
        parameters_sha256="a" * 64,
        state_dict_sha256="b" * 64,
    )

    assert result.loc[0, "prediction_score"] == pytest.approx(0.6)
    assert result.loc[0, "prediction_label"] == 1
    assert pd.isna(result.loc[0, "threshold"])


def test_afshar_static_inventory_does_not_load_pickle(tmp_path: Path) -> None:
    vectorizer = TfidfVectorizer().fit(["clear lungs", "bilateral opacity"])
    model = SVC().fit(vectorizer.transform(["clear lungs", "opacity"]), [0, 1])
    path = tmp_path / "model.sav"
    path.write_bytes(pickle.dumps(model, protocol=3))

    result = inspect_pickle_artifact(path)

    assert result["pickle_protocols"] == [3]
    assert result["loaded"] is False
    assert any("sklearn" in value for value in result["referenced_modules_classes"])


def test_combined_benchmark_target_builds_required_amaral_predictions() -> None:
    result = subprocess.run(
        ["make", "-n", "comparator-benchmark"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "scripts/run_amaral_comparator.py" in result.stdout
    assert result.stdout.index("scripts/run_amaral_comparator.py") < result.stdout.rindex(
        "scripts/benchmark_comparators.py"
    )


def test_afshar_container_uses_configured_artifact_paths(tmp_path: Path) -> None:
    config = load_afshar_config(Path("config/external_comparators/afshar_text_svc.example.yaml"))
    model_path = tmp_path / "custom-model.sav"
    vectorizer_path = tmp_path / "custom-vectorizer.sav"
    config = replace(config, model_path=model_path, vectorizer_path=vectorizer_path)

    command = afshar_container_command(
        config=config,
        packet=tmp_path / "input.jsonl.gz",
        output=tmp_path / "output.jsonl",
        runner=tmp_path / "runner.py",
    )

    assert f"{model_path}:/artifacts/model.sav:ro" in command
    assert f"{vectorizer_path}:/artifacts/vectorizer.sav:ro" in command
    assert command[command.index("--model") + 1] == "/artifacts/model.sav"
    assert command[command.index("--vectorizer") + 1] == "/artifacts/vectorizer.sav"


def test_external_repository_allows_only_configured_untracked_artifacts(tmp_path: Path) -> None:
    repository = tmp_path / "external"
    repository.mkdir()
    _git(repository, "init")
    _git(repository, "config", "user.email", "test@example.com")
    _git(repository, "config", "user.name", "Test User")
    tracked = repository / "source.py"
    tracked.write_text("VALUE = 1\n", encoding="utf-8")
    _git(repository, "add", "source.py")
    _git(repository, "commit", "-m", "fixture")
    remote = "https://example.com/uw-bionlp/ards.git"
    _git(repository, "remote", "add", "origin", remote)
    source = ExternalRepositoryConfig(
        name="uw",
        repository=remote,
        commit=_git(repository, "rev-parse", "HEAD").stdout.strip(),
        license="BSD-3-Clause",
        external_repo_dir=repository,
    )
    parameters = repository / "model/parameters.pkl"
    state_dict = repository / "model/state_dict.pt"
    parameters.parent.mkdir()
    parameters.write_bytes(b"parameters")
    state_dict.write_bytes(b"state")

    result = verify_external_repository(
        source,
        allowed_untracked_paths=(parameters, state_dict),
    )

    assert result["clean"] is True
    extra = repository / "model/unexpected.bin"
    extra.write_bytes(b"unexpected")
    with pytest.raises(ValueError, match="local modifications"):
        verify_external_repository(
            source,
            allowed_untracked_paths=(parameters, state_dict),
        )
    extra.unlink()
    tracked.write_text("VALUE = 2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="local modifications"):
        verify_external_repository(
            source,
            allowed_untracked_paths=(parameters, state_dict),
        )


def _git(repository: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    )
