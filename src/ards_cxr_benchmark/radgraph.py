from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd


def parse_mimic_ids_from_radgraph_key(report_key: str) -> tuple[int | None, int | None]:
    subject_match = re.search(r"p(?P<subject_id>\d{8})", report_key)
    study_match = re.search(r"s(?P<study_id>\d{8})\.txt", report_key)

    subject_id = int(subject_match.group("subject_id")) if subject_match else None
    study_id = int(study_match.group("study_id")) if study_match else None
    return subject_id, study_id


def _tokens_to_text(tokens: object) -> str:
    if isinstance(tokens, list):
        return " ".join(str(token) for token in tokens)
    if tokens is None:
        return ""
    return str(tokens)


def _label_to_text(label: object) -> str | None:
    if label is None:
        return None
    if isinstance(label, list):
        return "|".join(str(item) for item in label)
    return str(label)


def flatten_radgraph_payload(
    payload: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    report_rows: list[dict[str, object]] = []
    entity_rows: list[dict[str, object]] = []
    relation_rows: list[dict[str, object]] = []

    for report_key, report_payload in payload.items():
        subject_id, study_id = parse_mimic_ids_from_radgraph_key(report_key)
        report_uid = str(report_key)

        report_rows.append(
            {
                "report_uid": report_uid,
                "subject_id": subject_id,
                "study_id": study_id,
                "data_source": report_payload.get("data_source"),
                "data_split": report_payload.get("data_split"),
                "radgraph_text": report_payload.get("text"),
            }
        )

        entities = report_payload.get("entities", {}) or {}
        for entity_id, entity in entities.items():
            entity_id_text = str(entity_id)
            entity_rows.append(
                {
                    "report_uid": report_uid,
                    "subject_id": subject_id,
                    "study_id": study_id,
                    "entity_id": entity_id_text,
                    "tokens": _tokens_to_text(entity.get("tokens")),
                    "label": _label_to_text(entity.get("label", entity.get("labels"))),
                    "start_ix": entity.get("start_ix"),
                    "end_ix": entity.get("end_ix"),
                }
            )

            for relation in entity.get("relations", []) or []:
                if not isinstance(relation, list) or len(relation) != 2:
                    continue
                relation_type, object_id = relation
                relation_rows.append(
                    {
                        "report_uid": report_uid,
                        "subject_id": subject_id,
                        "study_id": study_id,
                        "source_entity_id": entity_id_text,
                        "target_entity_id": str(object_id),
                        "relation_type": str(relation_type),
                    }
                )

    return pd.DataFrame(report_rows), pd.DataFrame(entity_rows), pd.DataFrame(relation_rows)


def flatten_radgraph_json(path: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    if not isinstance(payload, dict):
        raise ValueError(f"RadGraph JSON root must be an object: {path}")

    reports, entities, relations = flatten_radgraph_payload(payload)
    if not reports.empty and reports["study_id"].isna().mean() > 0.01:
        raise ValueError("More than 1% of RadGraph reports could not be mapped to study_id")

    return reports, entities, relations


def write_radgraph_parquet(radgraph_json: Path, out_dir: Path) -> tuple[Path, Path, Path]:
    reports, entities, relations = flatten_radgraph_json(radgraph_json)
    out_dir.mkdir(parents=True, exist_ok=True)

    reports_path = out_dir / "radgraph_reports.parquet"
    entities_path = out_dir / "radgraph_entities.parquet"
    relations_path = out_dir / "radgraph_relations.parquet"

    reports.to_parquet(reports_path, index=False)
    entities.to_parquet(entities_path, index=False)
    relations.to_parquet(relations_path, index=False)

    return reports_path, entities_path, relations_path
