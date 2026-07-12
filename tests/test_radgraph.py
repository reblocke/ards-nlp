from __future__ import annotations

from ards_cxr_benchmark.radgraph import (
    flatten_radgraph_payload,
    parse_mimic_ids_from_radgraph_key,
)


def test_parse_mimic_ids_from_radgraph_key() -> None:
    key = "mimic-cxr-reports/files/p10/p10000032/s50414267.txt"
    assert parse_mimic_ids_from_radgraph_key(key) == (10000032, 50414267)


def test_flatten_radgraph_payload() -> None:
    payload = {
        "files/p10/p10000032/s50414267.txt": {
            "text": "Bilateral opacities.",
            "data_source": "MIMIC-CXR",
            "entities": {
                "1": {
                    "tokens": ["opacities"],
                    "label": "OBS-DP",
                    "start_ix": 1,
                    "end_ix": 2,
                    "relations": [["located_at", "2"]],
                },
                "2": {
                    "tokens": ["both", "lungs"],
                    "label": "ANAT-DP",
                    "start_ix": 3,
                    "end_ix": 5,
                    "relations": [],
                },
            },
        }
    }

    reports, entities, relations = flatten_radgraph_payload(payload)

    assert reports.shape[0] == 1
    assert entities.shape[0] == 2
    assert relations.shape[0] == 1
    assert entities.loc[0, "tokens"] == "opacities"
    assert relations.loc[0, "relation_type"] == "located_at"
