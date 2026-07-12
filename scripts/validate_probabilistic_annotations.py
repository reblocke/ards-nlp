from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from ards_cxr_benchmark.annotation_reference import (
    annotation_agreement_summary,
    build_case_reference_standard,
    case_level_disagreement_flags,
    clean_rater_ratings,
    leave_one_rater_out_agreement,
    rater_pairwise_agreement,
    render_annotation_summary_markdown,
    validate_rating_dataframe,
    write_rating_validation,
)
from ards_cxr_benchmark.config import ensure_dir, ensure_parent_dir
from ards_cxr_benchmark.paths import get_paths


def default_input() -> Path:
    return get_paths().root / "tests" / "fixtures" / "probabilistic_annotations.csv"


def default_annotation_paths(input_path: Path) -> tuple[Path, Path, Path]:
    annotation_root = get_paths().root / "data" / "derived" / "annotations"
    artifact_root = get_paths().root / "artifacts" / "annotations"
    if input_path.resolve() == default_input().resolve():
        annotation_root = annotation_root / "synthetic"
        artifact_root = artifact_root / "synthetic"
    return (
        annotation_root / "validated_rater_ratings.parquet",
        annotation_root / "case_reference_standard.parquet",
        artifact_root,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate probabilistic image-only/report-only annotation ratings"
    )
    parser.add_argument("--input", type=Path, default=default_input())
    parser.add_argument("--ratings-out", type=Path)
    parser.add_argument("--reference-out", type=Path)
    parser.add_argument("--artifact-dir", type=Path)
    parser.add_argument("--require-completed", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    default_ratings_out, default_reference_out, default_artifact_dir = default_annotation_paths(
        args.input
    )
    ratings_out = args.ratings_out or default_ratings_out
    reference_out = args.reference_out or default_reference_out
    artifact_dir = args.artifact_dir or default_artifact_dir
    df = read_table(args.input)
    validation = validate_rating_dataframe(df)
    if args.require_completed and validation.n_completed_rows == 0:
        raise SystemExit("No completed probabilistic annotation rows found")
    ensure_dir(artifact_dir)
    write_rating_validation(
        validation,
        out_json=artifact_dir / "annotation_validation_summary.json",
    )
    if not validation.passed:
        raise SystemExit(
            "Probabilistic annotation validation failed; see "
            f"{artifact_dir / 'annotation_validation_summary.json'}"
        )

    ratings = clean_rater_ratings(df)
    reference = build_case_reference_standard(ratings)
    agreement = annotation_agreement_summary(ratings, reference)
    pairwise = rater_pairwise_agreement(ratings)
    leave_one_out = leave_one_rater_out_agreement(ratings)
    disagreement = case_level_disagreement_flags(reference)

    ensure_parent_dir(ratings_out)
    ensure_parent_dir(reference_out)
    ratings.to_parquet(ratings_out, index=False)
    reference.to_parquet(reference_out, index=False)
    agreement.to_csv(artifact_dir / "annotation_agreement_summary.csv", index=False)
    pairwise.to_csv(artifact_dir / "rater_pairwise_agreement.csv", index=False)
    disagreement.to_csv(artifact_dir / "case_level_disagreement_flags.csv", index=False)
    (artifact_dir / "annotation_agreement_summary.md").write_text(
        render_annotation_summary_markdown(
            validation=validation,
            agreement_summary=agreement,
            pairwise_agreement=pairwise,
            leave_one_out=leave_one_out,
        ),
        encoding="utf-8",
    )
    print(f"Wrote validated ratings to {ratings_out}")
    print(f"Wrote case reference standard to {reference_out}")
    print(f"Wrote annotation artifacts to {artifact_dir}")


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() in {".csv", ".txt"}:
        return pd.read_csv(path)
    raise ValueError(f"Unsupported input format for {path}")


if __name__ == "__main__":
    main()
