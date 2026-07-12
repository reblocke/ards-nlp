from __future__ import annotations

import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from .annotation_planning import load_annotation_planning_config
from .pilot_annotation_agreement import load_annotation_pilot_config


@dataclass(frozen=True)
class ReadinessCheck:
    use_case: str
    check: str
    status: str
    detail: str


def collect_readiness(
    root: Path,
    *,
    use_case: str = "all",
    annotation_config: Path | None = None,
    planning_config: Path | None = None,
) -> list[ReadinessCheck]:
    root = root.resolve()
    requested = {"annotation", "comparators", "build"} if use_case == "all" else {use_case}
    invalid = requested - {"annotation", "comparators", "build"}
    if invalid:
        raise ValueError(f"Unknown readiness use case: {sorted(invalid)}")

    checks: list[ReadinessCheck] = []
    if "annotation" in requested:
        checks.extend(
            _annotation_checks(
                root,
                annotation_config=annotation_config,
                planning_config=planning_config,
            )
        )
    if "comparators" in requested:
        checks.extend(_comparator_checks(root))
    if "build" in requested:
        checks.extend(_build_checks(root))
    return checks


def readiness_has_blockers(checks: list[ReadinessCheck]) -> bool:
    return any(check.status == "blocked" for check in checks)


def format_readiness(checks: list[ReadinessCheck]) -> str:
    widths = {
        "use_case": max(len("USE CASE"), *(len(check.use_case) for check in checks)),
        "check": max(len("CHECK"), *(len(check.check) for check in checks)),
        "status": max(len("STATUS"), *(len(check.status) for check in checks)),
    }
    header = (
        f"{'USE CASE':<{widths['use_case']}}  "
        f"{'CHECK':<{widths['check']}}  "
        f"{'STATUS':<{widths['status']}}  DETAIL"
    )
    rows = [header, "-" * len(header)]
    rows.extend(
        f"{check.use_case:<{widths['use_case']}}  "
        f"{check.check:<{widths['check']}}  "
        f"{check.status:<{widths['status']}}  {check.detail}"
        for check in checks
    )
    return "\n".join(rows)


def _annotation_checks(
    root: Path,
    *,
    annotation_config: Path | None,
    planning_config: Path | None,
) -> list[ReadinessCheck]:
    checks = [
        _python_check("annotation"),
        _command_check("annotation", "uv"),
        _quarto_check(),
    ]
    config_path = (annotation_config or root / "config/annotation_pilot.yaml").resolve()
    if not config_path.is_file():
        checks.append(
            ReadinessCheck(
                "annotation",
                "real pilot config",
                "blocked",
                "create config/annotation_pilot.yaml from the tracked example",
            )
        )
        checks.append(_planning_config_check(root, planning_config))
        return checks

    try:
        config = load_annotation_pilot_config(config_path)
    except (FileNotFoundError, TypeError, ValueError) as error:
        checks.append(ReadinessCheck("annotation", "real pilot config", "blocked", str(error)))
        checks.append(_planning_config_check(root, planning_config))
        return checks

    checks.append(ReadinessCheck("annotation", "real pilot config", "ready", str(config_path)))
    missing = [rater_id for rater_id, path in config.inputs.items() if not path.is_file()]
    if missing:
        detail = "missing configured exports for " + ", ".join(sorted(missing))
        checks.append(ReadinessCheck("annotation", "three rater exports", "blocked", detail))
    else:
        checks.append(
            ReadinessCheck(
                "annotation",
                "three rater exports",
                "ready",
                f"{len(config.inputs)} distinct files are present",
            )
        )
    checks.append(_planning_config_check(root, planning_config))
    return checks


def _planning_config_check(root: Path, planning_config: Path | None) -> ReadinessCheck:
    path = (planning_config or root / "config/annotation_planning.yaml").resolve()
    if not path.is_file():
        return ReadinessCheck(
            "annotation",
            "planning config",
            "blocked",
            "create config/annotation_planning.yaml from the tracked example",
        )
    try:
        load_annotation_planning_config(path)
    except (FileNotFoundError, TypeError, ValueError) as error:
        return ReadinessCheck("annotation", "planning config", "blocked", str(error))
    return ReadinessCheck("annotation", "planning config", "ready", str(path))


def _comparator_checks(root: Path) -> list[ReadinessCheck]:
    checks = [_python_check("comparators"), _command_check("comparators", "uv")]
    requirements = {
        "local config": root / "config/config.yaml",
        "processed reports": root / "data/processed/mimic_cxr_reports.parquet",
        "model extract": root / "data/derived/modeling/model_development_extract.parquet",
        "CLAMP predictions": root / "data/derived/clamp_ards/clamp_legacy_predictions.parquet",
        "CLAMP entities": root / "data/derived/clamp_ards/clamp_legacy_entities.parquet",
        "silver baselines": root / "data/derived/modeling/silver_baseline_predictions.parquet",
    }
    checks.extend(_file_check("comparators", name, path) for name, path in requirements.items())
    checks.append(_command_check("comparators", "git"))
    return checks


def _build_checks(root: Path) -> list[ReadinessCheck]:
    checks = [
        _python_check("build"),
        _command_check("build", "uv"),
        _command_check("build", "gcloud"),
        _file_check("build", "local config", root / "config/config.yaml"),
    ]
    report_root = root / "data/raw/mimic-cxr/files"
    report_count = sum(1 for _ in report_root.rglob("*.txt")) if report_root.is_dir() else 0
    checks.append(
        ReadinessCheck(
            "build",
            "MIMIC-CXR reports",
            "ready" if report_count else "blocked",
            f"{report_count:,} report files found" if report_count else "report tree is missing",
        )
    )
    for name in ("chexpert", "negbio", "split", "metadata"):
        checks.append(
            _file_check(
                "build",
                f"MIMIC-CXR-JPG {name}",
                root / f"data/raw/mimic-cxr-jpg/mimic-cxr-2.0.0-{name}.csv.gz",
            )
        )
    checks.append(
        _file_check(
            "build",
            "RadGraph JSON",
            root / "data/raw/radgraph/MIMIC-CXR_graphs.json",
        )
    )
    adc = Path.home() / ".config/gcloud/application_default_credentials.json"
    checks.append(_file_check("build", "Google ADC", adc))
    return checks


def _python_check(use_case: str) -> ReadinessCheck:
    version = sys.version_info
    ready = version >= (3, 11)
    return ReadinessCheck(
        use_case,
        "Python",
        "ready" if ready else "blocked",
        f"{version.major}.{version.minor}.{version.micro}",
    )


def _command_check(use_case: str, command: str) -> ReadinessCheck:
    path = shutil.which(command)
    return ReadinessCheck(
        use_case,
        command,
        "ready" if path else "blocked",
        path or f"{command} is not installed or not on PATH",
    )


def _quarto_check() -> ReadinessCheck:
    path = shutil.which("quarto")
    if not path:
        return ReadinessCheck(
            "annotation", "Quarto", "blocked", "Quarto 1.8.26 is not installed or not on PATH"
        )
    try:
        result = subprocess.run(
            [path, "--version"],
            check=True,
            capture_output=True,
            text=True,
        )
        version = result.stdout.strip()
    except subprocess.SubprocessError as error:
        return ReadinessCheck("annotation", "Quarto", "blocked", str(error))
    return ReadinessCheck("annotation", "Quarto", "ready", version)


def _file_check(use_case: str, name: str, path: Path) -> ReadinessCheck:
    exists = path.is_file()
    return ReadinessCheck(
        use_case,
        name,
        "ready" if exists else "blocked",
        str(path) if exists else f"missing: {path}",
    )
