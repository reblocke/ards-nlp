from __future__ import annotations

from ards_cxr_benchmark.config import default_config_path, load_config
from ards_cxr_benchmark.paths import get_paths


def main() -> None:
    config_path = default_config_path()
    config = load_config(config_path)
    paths = get_paths()

    print("Benchmark configuration loaded")
    print(f"- config: {config_path}")
    print(f"- repo: {paths.root}")
    print(f"- project: {config.name}")
    print(f"- BigQuery dataset: {config.bq.dataset_ref}")
    print(f"- primary label text scope: {config.primary_label_text_scope}")
    print(
        "- pipeline targets: discover, init, ingest, build, qa, sample, export, splits, "
        "modeling, modeling-qa, annotation-eval, benchmark-eval, annotation-pilot, "
        "annotation-pilot-smoke, annotation-planning, annotation-planning-smoke, doctor, "
        "release-audit, clamp-ards-sync, clamp-ards-inputs, "
        "clamp-ards-output-packet, clamp-ards-parse, clamp-ards-teacher-benchmark, "
        "clamp-ards-python, clamp-ards-python-smoke, clamp-ards-characterize, "
        "clamp-ards-parity-fixtures, clamp-ards-parity-fixture-prepare, "
        "clamp-ards-parity-fixture-validate, clamp-ards-parity-fixture-handoff, "
        "clamp-ards-parity-fixture-import, clamp-ards-parity-restricted, "
        "clamp-ards-resources-audit, clamp-ards-resources-public-audit, "
        "comparator-source, comparator-existing, comparator-clamp-python, comparator-amaral, "
        "comparator-uw-hanso-verify, "
        "comparator-uw-hanso-smoke, comparator-afshar-inspect, comparator-afshar-smoke, "
        "comparator-benchmark, comparator-snapshot, comparators-ready"
    )


if __name__ == "__main__":
    main()
