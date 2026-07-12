from __future__ import annotations

import argparse
from pathlib import Path

from ards_cxr_benchmark.clamp_ards_output_archive import parse_clamp_ards_output_archive
from ards_cxr_benchmark.clamp_ards_outputs import (
    parse_clamp_ards_outputs,
    write_clamp_teacher_outputs,
)
from ards_cxr_benchmark.config import default_config_path, load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Parse returned ARDS CLAMP outputs into teacher entity and prediction tables"
    )
    parser.add_argument("--config", type=Path, default=default_config_path())
    parser.add_argument("--input-manifest", type=Path)
    sources = parser.add_mutually_exclusive_group()
    sources.add_argument("--clamp-output-dir", type=Path)
    sources.add_argument("--clamp-output-archive", type=Path)
    parser.add_argument("--entity-output", type=Path)
    parser.add_argument("--prediction-output", type=Path)
    parser.add_argument("--probabilistic-prediction-output", type=Path)
    parser.add_argument("--audit-output", type=Path)
    parser.add_argument("--summary-output", type=Path)
    parser.add_argument("--batch-size", type=int, default=5000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    input_manifest = args.input_manifest or config.clamp_ards.input_manifest
    entity_output = args.entity_output or config.clamp_ards.teacher_entity_output
    prediction_output = args.prediction_output or config.clamp_ards.teacher_prediction_output
    probabilistic_output = (
        args.probabilistic_prediction_output
        or config.clamp_ards.teacher_probabilistic_prediction_output
    )
    audit_output = args.audit_output or config.clamp_ards.teacher_output_audit
    summary_output = args.summary_output or config.clamp_ards.teacher_summary_output

    archive = args.clamp_output_archive
    if (
        archive is None
        and args.clamp_output_dir is None
        and config.clamp_ards.output_archive.exists()
    ):
        archive = config.clamp_ards.output_archive
    if archive is not None:
        summary = parse_clamp_ards_output_archive(
            input_manifest_path=input_manifest,
            output_archive=archive,
            entity_output=entity_output,
            prediction_output=prediction_output,
            probabilistic_prediction_output=probabilistic_output,
            audit_output=audit_output,
            summary_output=summary_output,
            batch_size=args.batch_size,
            show_progress=True,
        )
    else:
        result = parse_clamp_ards_outputs(
            input_manifest_path=input_manifest,
            output_dir=args.clamp_output_dir or config.clamp_ards.output_dir,
        )
        write_clamp_teacher_outputs(
            result,
            entity_output=entity_output,
            prediction_output=prediction_output,
            probabilistic_prediction_output=probabilistic_output,
            audit_output=audit_output,
            summary_output=summary_output,
        )
        summary = result.summary
    print(
        "Parsed ARDS CLAMP outputs: "
        f"{summary['parse_success_files']:,} success file(s), "
        f"{summary['parse_error_files']:,} parse error file(s), "
        f"{summary['evaluable_prediction_rows']:,} evaluable prediction row(s)"
    )


if __name__ == "__main__":
    main()
