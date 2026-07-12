from __future__ import annotations

import argparse
from pathlib import Path

from ards_cxr_benchmark.clamp_ards.batch import run_clamp_ards_batch
from ards_cxr_benchmark.config import PipelineConfig, default_config_path, load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the deterministic Python compatibility port of ARDS CLAMP"
    )
    parser.add_argument("--config", type=Path, default=default_config_path())
    sources = parser.add_mutually_exclusive_group()
    sources.add_argument("--input", type=Path)
    sources.add_argument(
        "--fixture-root",
        type=Path,
        help="Read exact UTF-8 inputs in fixture manifest order",
    )
    parser.add_argument("--id-col", default="study_id")
    parser.add_argument("--text-col", default="report_text")
    parser.add_argument("--id-prefix")
    parser.add_argument("--entity-output", type=Path)
    parser.add_argument("--prediction-output", type=Path)
    parser.add_argument("--summary-output", type=Path)
    parser.add_argument("--project-dir", type=Path)
    parser.add_argument(
        "--resource-manifest",
        type=Path,
        help="Override the packaged production resource fingerprint manifest",
    )
    parser.add_argument("--batch-size", type=int, default=5000)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    entity_output, prediction_output, summary_output = _resolve_outputs(args, config)
    input_path = (
        None if args.fixture_root is not None else args.input or config.clamp_ards.python_input
    )
    id_prefix = args.id_prefix if args.id_prefix is not None else ("" if args.fixture_root else "s")
    summary = run_clamp_ards_batch(
        input_path=input_path,
        fixture_root=args.fixture_root,
        entity_output=entity_output,
        prediction_output=prediction_output,
        summary_output=summary_output,
        id_column=args.id_col,
        text_column=args.text_col,
        id_prefix=id_prefix,
        project_dir=args.project_dir,
        resource_manifest=args.resource_manifest,
        batch_size=args.batch_size,
        limit=args.limit,
        show_progress=not args.no_progress,
    )
    print(
        "Python ARDS CLAMP complete: "
        f"{summary.document_count:,} document(s), "
        f"{summary.positive_document_count:,} positive, "
        f"{summary.entity_count:,} entity row(s)"
    )


def _resolve_outputs(args: argparse.Namespace, config: PipelineConfig) -> tuple[Path, Path, Path]:
    clamp = config.clamp_ards
    canonical = {
        "--entity-output": clamp.python_entity_output,
        "--prediction-output": clamp.python_prediction_output,
        "--summary-output": clamp.python_summary_output,
    }
    provided = {
        "--entity-output": args.entity_output,
        "--prediction-output": args.prediction_output,
        "--summary-output": args.summary_output,
    }
    alternate_run = any(
        (
            any(path is not None for path in provided.values()),
            args.input is not None,
            args.fixture_root is not None,
            args.limit is not None,
            args.project_dir is not None,
            args.resource_manifest is not None,
            args.id_col != "study_id",
            args.text_col != "report_text",
            args.id_prefix is not None,
        )
    )
    if alternate_run:
        missing = [flag for flag, path in provided.items() if path is None]
        if missing:
            raise ValueError(
                "Alternate or limited Python CLAMP runs require explicit noncanonical "
                f"output paths for: {', '.join(missing)}"
            )
        canonical_paths = {path.expanduser().resolve() for path in canonical.values()}
        conflicts = [
            flag
            for flag, path in provided.items()
            if path is not None and path.expanduser().resolve() in canonical_paths
        ]
        if conflicts:
            raise ValueError(
                "Alternate or limited Python CLAMP runs must not overwrite configured "
                f"full-corpus outputs: {', '.join(conflicts)}"
            )
        resolved_outputs = [path.expanduser().resolve() for path in provided.values() if path]
        if len(set(resolved_outputs)) != len(resolved_outputs):
            raise ValueError(
                "Python CLAMP entity, prediction, and summary outputs must be distinct"
            )
    return tuple(provided[flag] or canonical[flag] for flag in canonical)


if __name__ == "__main__":
    main()
