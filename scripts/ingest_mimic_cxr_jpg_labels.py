from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

from ards_cxr_benchmark.config import (
    default_config_path,
    ensure_dir,
    load_config,
    require_existing_path,
)

LABEL_COLUMNS = [
    "atelectasis",
    "cardiomegaly",
    "consolidation",
    "edema",
    "enlarged_cardiomediastinum",
    "fracture",
    "lung_lesion",
    "lung_opacity",
    "no_finding",
    "pleural_effusion",
    "pleural_other",
    "pneumonia",
    "pneumothorax",
    "support_devices",
]


def normalize_column_name(column: str) -> str:
    column = column.strip().lower()
    column = re.sub(r"[^a-z0-9]+", "_", column)
    return column.strip("_")


def normalize_label_frame(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.rename(columns={column: normalize_column_name(column) for column in df.columns})

    required = {"subject_id", "study_id"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns in {path}: {missing}")

    keep = ["subject_id", "study_id", *[column for column in LABEL_COLUMNS if column in df]]
    out = df[keep].copy()
    out["subject_id"] = pd.to_numeric(out["subject_id"], errors="raise").astype("Int64")
    out["study_id"] = pd.to_numeric(out["study_id"], errors="raise").astype("Int64")

    for column in LABEL_COLUMNS:
        if column in out:
            out[column] = pd.to_numeric(out[column], errors="coerce").astype("Int64")

    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Normalize MIMIC-CXR-JPG CheXpert/NegBio labels to Parquet"
    )
    parser.add_argument("--config", type=Path, default=default_config_path())
    parser.add_argument("--chexpert-csv", type=Path)
    parser.add_argument("--negbio-csv", type=Path)
    parser.add_argument("--out-dir", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    chexpert_csv = args.chexpert_csv or config.paths.chexpert_csv
    negbio_csv = args.negbio_csv or config.paths.negbio_csv
    out_dir = args.out_dir or config.paths.jpg_parquet_dir

    require_existing_path(chexpert_csv, "MIMIC-CXR-JPG CheXpert CSV")
    require_existing_path(negbio_csv, "MIMIC-CXR-JPG NegBio CSV")
    ensure_dir(out_dir)

    chexpert = normalize_label_frame(chexpert_csv)
    negbio = normalize_label_frame(negbio_csv)

    chexpert_path = out_dir / "mimic_cxr_jpg_chexpert.parquet"
    negbio_path = out_dir / "mimic_cxr_jpg_negbio.parquet"
    chexpert.to_parquet(chexpert_path, index=False)
    negbio.to_parquet(negbio_path, index=False)

    print(f"Wrote {len(chexpert):,} CheXpert rows to {chexpert_path}")
    print(f"Wrote {len(negbio):,} NegBio rows to {negbio_path}")


if __name__ == "__main__":
    main()
