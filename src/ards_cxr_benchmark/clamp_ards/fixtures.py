from __future__ import annotations

import csv
import hashlib
import json
import re
import shutil
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path, PurePosixPath
from typing import Any

import yaml

from .resources import (
    ClampResources,
    default_project_dir,
    default_resource_manifest_path,
    load_clamp_resources,
)
from .tokenization import Utf16OffsetMap

FIXTURE_SCHEMA_VERSION = 1
FIXTURE_VERSION = "clamp-ards-golden-v1"
REFERENCE_PROJECT_COMMIT = "9f8c92fbbeb44645a1066be3510d4ab993995c1e"
CASE_ID_PATTERN = re.compile(r"^[a-z][a-z0-9_]{0,79}$")
LIFECYCLES = frozenset({"awaiting_legacy_runs", "complete"})

EXPECTED_CATEGORY_COUNTS = {
    "assertion_conjunction": 69,
    "assertion_interaction": 12,
    "assertion_post": 28,
    "assertion_pre": 127,
    "assertion_pseudo": 32,
    "dictionary_boundary": 12,
    "dictionary_case": 92,
    "dictionary_inflection": 22,
    "ruta_gap": 8,
    "ruta_rule_order": 12,
    "sentence_input": 18,
    "tokenizer_delimiter": 29,
    "tokenizer_no_split": 2,
}
EXPECTED_CASE_COUNT = sum(EXPECTED_CATEGORY_COUNTS.values())

MANIFEST_FIELDS = (
    "fixture_version",
    "case_id",
    "primary_category",
    "input_path",
    "source_kind",
    "encoding",
    "line_ending",
    "trailing_newline",
    "byte_count",
    "codepoint_count",
    "utf16_code_unit_count",
    "input_sha256",
    "resource_kind",
    "resource_index",
    "phi_automated_screen",
)

EXPECTED_ENTITY_FIELDS = (
    "clamp_doc_id",
    "start",
    "end",
    "semantic_tag",
    "assertion",
    "cui",
    "attribute",
    "entity_text",
    "raw_order",
    "duplicate_occurrence",
)

INTERMEDIATE_FIELDS = frozenset(
    {
        "schema_version",
        "case_id",
        "source_text_sha256",
        "offset_coordinate_system",
        "interval_convention",
        "legacy_output_sha256",
        "sentences",
        "tokens",
        "final_entities",
    }
)
INTERMEDIATE_ROW_FIELDS = {
    "sentences": frozenset({"start", "end", "sentence_number", "covered_text"}),
    "tokens": frozenset({"start", "end", "token_number", "covered_text"}),
    "final_entities": frozenset(
        {
            "start",
            "end",
            "semantic_tag",
            "assertion",
            "cui",
            "attribute",
            "covered_text",
            "raw_order",
        }
    ),
}

PHI_SENTINELS = {
    "mimic_deidentification_marker": re.compile(r"\[\*\*"),
    "mimic_identifier": re.compile(r"\bmimic[_-]", re.IGNORECASE),
    "record_identifier_label": re.compile(
        r"\b(?:mrn|medical record|accession|subject[_ ]?id|study[_ ]?id|"
        r"patient[_ ]?id|dob|date of birth|ssn)\b",
        re.IGNORECASE,
    ),
    "email_address": re.compile(r"\b[^\s@]+@[^\s@]+\.[^\s@]+\b"),
    "phone_number": re.compile(r"\b\d{3}[-.) ]+\d{3}[- ]+\d{4}\b"),
    "ssn_number": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "long_numeric_identifier": re.compile(r"\b\d{4,}\b"),
    "absolute_posix_path": re.compile(r"(?<!:)\/(?:Users|home|Volumes|mnt)\/"),
    "windows_path": re.compile(r"\b[A-Za-z]:\\"),
}

OBSERVED_INFLECTIONS = (
    ("aspirations", "Morphology"),
    ("bilaterality", "location"),
    ("bilaterally", "location"),
    ("congested heart failure", "ARDS"),
    ("congestion heart failure", "ARDS"),
    ("consolidate", "Morphology"),
    ("consolidated", "Morphology"),
    ("consolidations", "Morphology"),
    ("consolidative", "Morphology"),
    ("diffusely", "location"),
    ("diffusion", "location"),
    ("extension", "location"),
    ("extensively", "location"),
    ("infiltrate", "Morphology"),
    ("infiltrated", "Morphology"),
    ("infiltrating", "Morphology"),
    ("infiltration", "Morphology"),
    ("infiltrative", "Morphology"),
    ("multifocality", "location"),
    ("opacity", "Morphology"),
    ("pneumonias", "Morphology"),
    ("pulmonary edemas", "ARDS"),
)


class FixturePendingError(RuntimeError):
    """Raised when strict validation is requested before legacy outputs exist."""


@dataclass(frozen=True)
class FixtureCase:
    case_id: str
    primary_category: str
    description: str
    tags: tuple[str, ...]
    payload: bytes
    resource_kind: str = ""
    resource_index: int | None = None
    parameters: tuple[tuple[str, str], ...] = ()

    @property
    def input_path(self) -> str:
        return f"input/{self.case_id}.txt"

    def yaml_record(self) -> dict[str, Any]:
        record: dict[str, Any] = {
            "case_id": self.case_id,
            "primary_category": self.primary_category,
            "description": self.description,
            "tags": list(self.tags),
            "input_path": self.input_path,
            "source_kind": "synthetic_deterministic",
        }
        if self.resource_kind:
            record["resource_ref"] = {
                "kind": self.resource_kind,
                "index": self.resource_index,
            }
        if self.parameters:
            record["parameters"] = dict(self.parameters)
        return record


@dataclass(frozen=True)
class FixtureValidation:
    fixture_version: str
    lifecycle: str
    case_count: int
    category_counts: dict[str, int]
    pending: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "fixture_version": self.fixture_version,
            "lifecycle": self.lifecycle,
            "case_count": self.case_count,
            "category_counts": self.category_counts,
            "pending": self.pending,
        }


NormalizedExpectedEntity = tuple[int, int, str, str, str | None, str | None, str]


@dataclass(frozen=True)
class _IntermediateValidation:
    sentence_count: int
    token_count: int
    final_entities: tuple[NormalizedExpectedEntity, ...]


def build_fixture_cases(resources: ClampResources) -> tuple[FixtureCase, ...]:
    cases: list[FixtureCase] = []
    case_forms = (
        ("lower", str.lower),
        ("title", str.title),
        ("upper", str.upper),
        ("mixed", _mixed_case),
    )
    for entry_number, entry in enumerate(resources.dictionary, start=1):
        for form_name, transform in case_forms:
            surface = transform(entry.term)
            cases.append(
                _text_case(
                    f"dict_case_{entry_number:02d}_{form_name}",
                    "dictionary_case",
                    f"Dictionary row {entry_number} in {form_name} case",
                    _dictionary_context(surface, entry.semantic_tag),
                    tags=("dictionary", entry.semantic_tag, form_name),
                    resource_kind="dictionary",
                    resource_index=entry_number,
                    parameters=(("case_form", form_name), ("semantic_tag", entry.semantic_tag)),
                )
            )

    for index, (surface, semantic_tag) in enumerate(OBSERVED_INFLECTIONS, start=1):
        cases.append(
            _text_case(
                f"dict_inflection_{index:02d}",
                "dictionary_inflection",
                f"Observed aggregate surface variant {index}",
                _dictionary_context(surface, semantic_tag),
                tags=("dictionary", "stemming", semantic_tag),
                parameters=(("surface", surface), ("semantic_tag", semantic_tag)),
            )
        )

    boundary_cases = (
        ("location_substring", "bilateralism opacities."),
        ("morphology_substring", "preopacities bilateral."),
        ("direct_substring", "ARDSlike."),
        ("phrase_hyphen", "pulmonary-edema."),
        ("phrase_spaces", "pulmonary   edema."),
        ("phrase_tab", "pulmonary\tedema."),
        ("phrase_newline", "pulmonary\nedema."),
        ("direct_duplicate", "ARDS ARDS."),
        ("promoted_duplicate", "bilateral opacities bilateral opacities."),
        ("left_right_overlap", "left and right-sided infiltrates."),
        ("right_left_overlap", "right and left-sided infiltrates."),
        ("hyphen_overlap", "multifocal multi-focal pneumonia."),
    )
    for index, (name, text) in enumerate(boundary_cases, start=1):
        cases.append(
            _text_case(
                f"dict_boundary_{index:02d}",
                "dictionary_boundary",
                f"Dictionary boundary case: {name}",
                text,
                tags=("dictionary", "boundary", name),
            )
        )

    cues_by_category: dict[str, list[tuple[int, str]]] = {
        category: []
        for category in ("pseNegPhrases", "negPhrases", "postNegPhrases", "conjunctions")
    }
    for resource_index, cue in enumerate(resources.assertion_cues, start=1):
        cues_by_category[cue.category].append((resource_index, cue.phrase))

    for category_index, (resource_index, phrase) in enumerate(
        cues_by_category["pseNegPhrases"], start=1
    ):
        for context_name, text in (
            ("initial", f"{phrase} ARDS."),
            ("lexical", f"Observation: {phrase} ARDS."),
        ):
            cases.append(
                _text_case(
                    f"assert_pseudo_{category_index:03d}_{context_name}",
                    "assertion_pseudo",
                    f"Pseudo-negation cue {category_index} in {context_name} context",
                    text,
                    tags=("assertion", "pseNegPhrases", context_name),
                    resource_kind="assertion_cue",
                    resource_index=resource_index,
                    parameters=(("cue", phrase), ("context", context_name)),
                )
            )

    assertion_templates = (
        ("negPhrases", "assert_pre", "assertion_pre", lambda phrase: f"{phrase} ARDS."),
        ("postNegPhrases", "assert_post", "assertion_post", lambda phrase: f"ARDS {phrase}."),
        (
            "conjunctions",
            "assert_conjunction",
            "assertion_conjunction",
            lambda phrase: f"No ARDS {phrase} ARDS.",
        ),
    )
    for cue_category, id_prefix, primary_category, template in assertion_templates:
        for category_index, (resource_index, phrase) in enumerate(
            cues_by_category[cue_category], start=1
        ):
            cases.append(
                _text_case(
                    f"{id_prefix}_{category_index:03d}",
                    primary_category,
                    f"Assertion resource cue {category_index}: {cue_category}",
                    template(phrase),
                    tags=("assertion", cue_category),
                    resource_kind="assertion_cue",
                    resource_index=resource_index,
                    parameters=(("cue", phrase),),
                )
            )

    assertion_interactions = (
        ("pseudo_initial", "No change in ARDS."),
        ("pseudo_lexical", "Observation: no change in ARDS."),
        ("pseudo_not", "not necessarily ARDS."),
        ("multi_entity", "No ARDS or pulmonary edema."),
        ("conjunction_scope", "No ARDS but pulmonary edema."),
        ("sentence_scope", "No ARDS. Pulmonary edema."),
        ("post_scope", "ARDS unlikely."),
        ("post_conjunction", "ARDS unlikely but pulmonary edema."),
        ("overlapping_pre", "No evidence to suggest ARDS."),
        ("nested_pre", "No new evidence of ARDS."),
        ("post_punctuation", "ARDS, is ruled out."),
        ("pre_punctuation_conjunction", "No ARDS; however, pulmonary edema."),
    )
    for index, (name, text) in enumerate(assertion_interactions, start=1):
        cases.append(
            _text_case(
                f"assert_interaction_{index:02d}",
                "assertion_interaction",
                f"Assertion interaction: {name}",
                text,
                tags=("assertion", "interaction", name),
            )
        )

    fillers = ("alpha", "beta", "gamma", "delta", "epsilon", "zeta")
    for direction, left, right in (
        ("location_morphology", "bilateral", "opacities"),
        ("morphology_location", "opacities", "bilateral"),
    ):
        for gap in (0, 1, 5, 6):
            middle = "" if gap == 0 else " ".join(fillers[:gap]) + " "
            cases.append(
                _text_case(
                    f"ruta_gap_{direction}_{gap}",
                    "ruta_gap",
                    f"Ruta {direction} with {gap} intervening BaseToken candidates",
                    f"{left} {middle}{right}.",
                    tags=("ruta", direction, f"gap_{gap}"),
                    parameters=(("direction", direction), ("gap", str(gap))),
                )
            )

    for index, delimiter in enumerate(sorted(resources.delimiters, key=ord), start=1):
        cases.append(
            _text_case(
                f"token_delimiter_{index:03d}",
                "tokenizer_delimiter",
                f"Configured delimiter U+{ord(delimiter):04X}",
                f"bilateral{delimiter}opacities",
                tags=("tokenizer", "delimiter", f"u{ord(delimiter):04x}"),
                resource_kind="tokenizer_delimiter",
                resource_index=index,
                parameters=(("codepoint", f"U+{ord(delimiter):04X}"),),
            )
        )

    for index, value in enumerate(resources.no_split_strings, start=1):
        for context_name, text in (
            ("standalone", value),
            ("attached", f"synthetic{value} ARDS."),
        ):
            cases.append(
                _text_case(
                    f"token_nosplit_{index:03d}_{context_name}",
                    "tokenizer_no_split",
                    f"Configured no-split string {index} in {context_name} context",
                    text,
                    tags=("tokenizer", "no_split", context_name),
                    resource_kind="tokenizer_no_split",
                    resource_index=index,
                    parameters=(("context", context_name),),
                )
            )

    sentence_cases: tuple[tuple[str, bytes], ...] = (
        ("lf_boundary", b"bilateral\nopacities"),
        ("crlf_boundary", b"bilateral\r\nopacities"),
        ("repeated_lf", b"ARDS\n\npulmonary edema"),
        ("repeated_crlf", b"ARDS\r\n\r\npulmonary edema"),
        ("surrounding_whitespace", b" \tARDS \t"),
        ("period_boundary", b"bilateral. opacities."),
        ("abbreviation_lower", b"Dr. synthetic ARDS."),
        ("abbreviation_upper", b"Dr. ARDS."),
        ("split_pattern", b"1) ARDS. 2) pulmonary edema."),
        ("section_header", b"IMPRESSION:\nARDS."),
        ("exactly_500_tokens", " ".join(["alpha"] * 499 + ["ARDS"]).encode()),
        ("more_than_500_tokens", " ".join(["alpha"] * 500 + ["ARDS"]).encode()),
        ("empty", b""),
        ("whitespace_only", b" \t\r\n"),
        ("unicode_bmp", "caf\u00e9\u2014ARDS.".encode()),
        ("unicode_supplementary", "\U0001f600 ARDS \U0001fac1.".encode()),
        ("split_pattern_without_terminal_periods", b"1) ARDS 2) pulmonary edema"),
        ("section_header_inline", b"IMPRESSION: ARDS."),
    )
    for index, (name, payload) in enumerate(sentence_cases, start=1):
        cases.append(
            FixtureCase(
                case_id=f"sentence_input_{index:02d}",
                primary_category="sentence_input",
                description=f"Sentence/input contract: {name}",
                tags=("sentence", "input_contract", name),
                payload=payload,
                parameters=(("scenario", name),),
            )
        )

    ruta_interactions = (
        ("direct_and_promoted", "ARDS bilateral opacities."),
        ("locations_before_morphology", "diffuse bilateral opacities."),
        ("morphology_before_locations", "opacities diffuse bilateral."),
        ("location_before_morphologies", "bilateral airspace opacities."),
        ("morphologies_before_location", "airspace opacities throughout."),
        ("location_morphology_location", "diffuse opacities throughout."),
        ("morphology_location_morphology", "opacities bilateral infiltrates."),
        ("negated_location", "no bilateral opacities."),
        ("negated_morphology", "bilateral no opacities."),
        ("overlap_and_promotion", "diffuse left and right-sided infiltrates."),
        ("slash_sequence", "diffuse consolidation/pneumonia."),
        ("sequential_mutation", "extensive bilateral opacities alpha multifocal pneumonia."),
    )
    for index, (name, text) in enumerate(ruta_interactions, start=1):
        cases.append(
            _text_case(
                f"ruta_rule_order_{index:02d}",
                "ruta_rule_order",
                f"Ruta interaction: {name}",
                text,
                tags=("ruta", "rule_order", name),
            )
        )

    _validate_generated_cases(cases)
    return tuple(cases)


def generate_fixture(
    output_dir: Path,
    *,
    project_dir: Path | None = None,
    resource_manifest_path: Path | None = None,
    force: bool = False,
) -> FixtureValidation:
    requested_output = output_dir.expanduser()
    if requested_output.is_symlink():
        raise ValueError(f"Fixture output must not be a symlink: {requested_output}")
    output_dir = requested_output.resolve()
    if output_dir.exists():
        if not force:
            raise FileExistsError(f"Fixture directory already exists: {output_dir}")
    project = (project_dir or default_project_dir()).expanduser().resolve()
    repo_root = Path(__file__).resolve().parents[3]
    manifest_path = (
        (resource_manifest_path or default_resource_manifest_path()).expanduser().resolve()
    )
    _require_safe_fixture_output(
        output_dir,
        project_dir=project,
        manifest_path=manifest_path,
        repo_root=repo_root,
    )
    if output_dir.exists():
        _require_generated_fixture_marker(output_dir)
    resources = load_clamp_resources(project, manifest_path=manifest_path)
    cases = build_fixture_cases(resources)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    input_dir = output_dir / "input"
    clamp_expected = output_dir / "clamp_expected"
    intermediate_expected = output_dir / "intermediate_expected"
    input_dir.mkdir()
    clamp_expected.mkdir()
    intermediate_expected.mkdir()
    for case in cases:
        (output_dir / case.input_path).write_bytes(case.payload)

    cases_payload = {
        "schema_version": FIXTURE_SCHEMA_VERSION,
        "fixture_version": FIXTURE_VERSION,
        "case_count": len(cases),
        "category_counts": dict(sorted(Counter(case.primary_category for case in cases).items())),
        "byte_contract": {
            "encoding": "utf-8",
            "bom": False,
            "default_trailing_newline": False,
            "offset_coordinate_system": "utf16_code_units",
            "input_files_are_authoritative": True,
        },
        "expected_output_policy": "legacy_clamp_generated_only_never_hand_authored",
        "cases": [case.yaml_record() for case in cases],
    }
    (output_dir / "cases.yaml").write_text(
        yaml.safe_dump(cases_payload, sort_keys=False, allow_unicode=True, width=120),
        encoding="utf-8",
        newline="\n",
    )

    _write_manifest(output_dir / "manifest.csv", cases)
    input_tree_sha256 = _input_tree_sha256(cases)
    provenance = {
        "schema_version": FIXTURE_SCHEMA_VERSION,
        "fixture_version": FIXTURE_VERSION,
        "lifecycle": "awaiting_legacy_runs",
        "case_count": len(cases),
        "input_tree_sha256": input_tree_sha256,
        "generator": {
            "module": "ards_cxr_benchmark.clamp_ards.fixtures",
            "command": ("uv run python scripts/generate_clamp_ards_parity_fixture.py generate"),
            "source_kind": "synthetic_deterministic",
            "reference_project_commit": REFERENCE_PROJECT_COMMIT,
            "resource_manifest_sha256": _sha256(manifest_path.read_bytes()),
        },
        "legacy_runtime": {
            "clamp_version": "VERIFY",
            "clamp_build": "VERIFY",
            "operating_system": "VERIFY",
            "java_version": "VERIFY",
            "locale": "VERIFY",
            "timezone": "VERIFY",
            "pipeline_export_settings": "VERIFY",
        },
        "output_contract": {
            "encoding": "utf-8",
            "offset_coordinate_system": "half_open_utf16_code_units",
            "null_sentinel": "\\N",
            "normalized_entity_fields": list(EXPECTED_ENTITY_FIELDS),
            "intermediate_stages": ["sentences", "tokens", "final_entities"],
            "raw_xmi_is_not_committed": True,
        },
        "runs": [],
        "determinism": {
            "status": "pending",
            "required_run_count": 2,
            "raw_order_required": "VERIFY",
        },
        "reviews": {
            "phi": {
                "automated_screen": "passed",
                "manual_review": "pending",
                "reviewer": "VERIFY",
                "reviewed_at": "VERIFY",
            },
            "redistribution": {
                "status": "pending",
                "authority": "VERIFY",
                "evidence": "VERIFY",
            },
        },
    }
    (output_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    pending_text = (
        "Legacy CLAMP output is intentionally absent. Populate this directory only through the "
        "two-run import workflow; never hand-author expected spans.\n"
    )
    for directory, expected in (
        (clamp_expected, "one normalized <case_id>.tsv per case"),
        (intermediate_expected, "one normalized <case_id>.json per case"),
    ):
        (directory / "README.md").write_text(
            "# Pending legacy CLAMP output\n\n"
            f"Expected after an approved import: {expected}. Raw XMI remains ignored.\n",
            encoding="utf-8",
            newline="\n",
        )
        (directory / "PENDING").write_text(
            pending_text,
            encoding="utf-8",
            newline="\n",
        )

    write_sha256s(output_dir)
    return validate_fixture(
        output_dir,
        allow_pending=True,
        project_dir=project,
        resource_manifest_path=manifest_path,
    )


def _require_safe_fixture_output(
    output_dir: Path,
    *,
    project_dir: Path,
    manifest_path: Path,
    repo_root: Path,
) -> None:
    home = Path.home().resolve()
    if output_dir.parent == output_dir:
        raise ValueError("Fixture output must not be the filesystem root")
    if home.is_relative_to(output_dir):
        raise ValueError(f"Fixture output must not contain the user home directory: {output_dir}")
    if repo_root.is_relative_to(output_dir):
        raise ValueError(f"Fixture output must not contain the repository root: {output_dir}")
    if output_dir.is_relative_to(project_dir) or project_dir.is_relative_to(output_dir):
        raise ValueError(
            f"Fixture output must not overlap the CLAMP project: {output_dir} and {project_dir}"
        )
    if manifest_path.is_relative_to(output_dir):
        raise ValueError(
            f"Fixture output must not contain the resource manifest: {output_dir} and "
            f"{manifest_path}"
        )


def _require_generated_fixture_marker(output_dir: Path) -> None:
    if not output_dir.is_dir():
        raise ValueError(f"Forced fixture replacement requires a directory: {output_dir}")
    provenance_path = output_dir / "provenance.json"
    try:
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"Forced fixture replacement requires generated fixture provenance: {provenance_path}"
        ) from exc
    generator = provenance.get("generator") if isinstance(provenance, dict) else None
    if (
        not isinstance(generator, dict)
        or provenance.get("fixture_version") != FIXTURE_VERSION
        or generator.get("module") != "ards_cxr_benchmark.clamp_ards.fixtures"
        or generator.get("source_kind") != "synthetic_deterministic"
    ):
        raise ValueError(
            f"Forced fixture replacement is limited to generated fixture directories: {output_dir}"
        )


def validate_fixture(
    root: Path,
    *,
    allow_pending: bool = False,
    project_dir: Path | None = None,
    resource_manifest_path: Path | None = None,
) -> FixtureValidation:
    root = root.expanduser().resolve()
    resource_manifest_path = resource_manifest_path or default_resource_manifest_path()
    resource_manifest_bytes = resource_manifest_path.read_bytes()
    resource_manifest = json.loads(resource_manifest_bytes)
    if not isinstance(resource_manifest, dict) or not isinstance(
        resource_manifest.get("files"), dict
    ):
        raise ValueError("Frozen CLAMP resource manifest is invalid")
    expected_project_commit = str(resource_manifest.get("project_commit", ""))
    expected_project_files = {
        str(path): str(digest) for path, digest in resource_manifest["files"].items()
    }
    expected_resource_files = {
        path: digest
        for path, digest in expected_project_files.items()
        if path.startswith("Components/")
    }
    resources = load_clamp_resources(
        project_dir or default_project_dir(),
        manifest_path=resource_manifest_path,
    )
    expected_cases = build_fixture_cases(resources)
    cases_payload = _read_mapping(root / "cases.yaml")
    if cases_payload.get("schema_version") != FIXTURE_SCHEMA_VERSION:
        raise ValueError("Unsupported fixture schema_version")
    if cases_payload.get("fixture_version") != FIXTURE_VERSION:
        raise ValueError("Unexpected fixture_version")
    case_records = cases_payload.get("cases")
    if not isinstance(case_records, list):
        raise ValueError("cases.yaml must contain a cases list")
    expected_case_records = [case.yaml_record() for case in expected_cases]
    case_ids: list[str] = []
    case_paths: list[str] = []
    category_counts: Counter[str] = Counter()
    expected_case_by_id = {case.case_id: case for case in expected_cases}
    for record in case_records:
        if not isinstance(record, dict):
            raise ValueError("Every cases.yaml case must be a mapping")
        case_id = str(record.get("case_id", ""))
        input_path = str(record.get("input_path", ""))
        category = str(record.get("primary_category", ""))
        _validate_case_id(case_id)
        _validate_relative_path(input_path)
        if input_path != f"input/{case_id}.txt":
            raise ValueError(f"Input path does not match case ID {case_id}: {input_path}")
        if record.get("source_kind") != "synthetic_deterministic":
            raise ValueError(f"Case {case_id} is not marked synthetic_deterministic")
        case_ids.append(case_id)
        case_paths.append(input_path)
        category_counts[category] += 1
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("Duplicate case_id in cases.yaml")
    if len(case_paths) != len(set(case_paths)):
        raise ValueError("Duplicate input_path in cases.yaml")
    if len(case_ids) != EXPECTED_CASE_COUNT or len(case_ids) != cases_payload.get("case_count"):
        raise ValueError(f"Expected {EXPECTED_CASE_COUNT} fixture cases, found {len(case_ids)}")
    if dict(sorted(category_counts.items())) != EXPECTED_CATEGORY_COUNTS:
        raise ValueError(f"Unexpected fixture category counts: {dict(category_counts)}")
    if cases_payload.get("category_counts") != EXPECTED_CATEGORY_COUNTS:
        raise ValueError("cases.yaml declared category counts differ from the frozen matrix")
    if cases_payload.get("byte_contract") != {
        "encoding": "utf-8",
        "bom": False,
        "default_trailing_newline": False,
        "offset_coordinate_system": "utf16_code_units",
        "input_files_are_authoritative": True,
    }:
        raise ValueError("cases.yaml byte contract differs from the frozen contract")
    if (
        cases_payload.get("expected_output_policy")
        != "legacy_clamp_generated_only_never_hand_authored"
    ):
        raise ValueError("cases.yaml expected-output policy is invalid")
    if case_records != expected_case_records:
        raise ValueError("cases.yaml differs from the deterministic frozen coverage matrix")

    with (root / "manifest.csv").open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != MANIFEST_FIELDS:
            raise ValueError("manifest.csv has an unexpected schema")
        manifest_rows = list(reader)
    if [row["case_id"] for row in manifest_rows] != case_ids:
        raise ValueError("manifest.csv case order differs from cases.yaml")
    manifest_by_id = {row["case_id"]: row for row in manifest_rows}

    for record in case_records:
        case_id = str(record["case_id"])
        path_value = str(record["input_path"])
        payload = (root / path_value).read_bytes()
        if payload != expected_case_by_id[case_id].payload:
            raise ValueError(f"Input bytes differ from the frozen case matrix: {path_value}")
        if payload.startswith(b"\xef\xbb\xbf"):
            raise ValueError(f"UTF-8 BOM is prohibited: {path_value}")
        try:
            text = payload.decode("utf-8", errors="strict")
        except UnicodeDecodeError as exc:
            raise ValueError(f"Input is not strict UTF-8: {path_value}") from exc
        sentinels = find_phi_sentinels(text)
        if sentinels:
            raise ValueError(f"PHI sentinel(s) in {path_value}: {sentinels}")
        row = manifest_by_id[case_id]
        expected_values = {
            "fixture_version": FIXTURE_VERSION,
            "primary_category": record["primary_category"],
            "input_path": path_value,
            "source_kind": "synthetic_deterministic",
            "encoding": "utf-8",
            "line_ending": _line_ending(payload),
            "trailing_newline": str(payload.endswith((b"\n", b"\r"))).lower(),
            "byte_count": str(len(payload)),
            "codepoint_count": str(len(text)),
            "utf16_code_unit_count": str(len(text.encode("utf-16-le")) // 2),
            "input_sha256": _sha256(payload),
            "resource_kind": str(record.get("resource_ref", {}).get("kind", "")),
            "resource_index": str(record.get("resource_ref", {}).get("index", "")),
            "phi_automated_screen": "passed",
        }
        for field, expected in expected_values.items():
            if row[field] != expected:
                raise ValueError(
                    f"manifest.csv mismatch for {case_id} field {field}: "
                    f"expected {expected!r}, found {row[field]!r}"
                )

    provenance = _read_mapping(root / "provenance.json")
    lifecycle = str(provenance.get("lifecycle", ""))
    if lifecycle not in LIFECYCLES:
        raise ValueError(f"Unknown fixture lifecycle: {lifecycle!r}")
    if provenance.get("fixture_version") != FIXTURE_VERSION:
        raise ValueError("provenance fixture_version mismatch")
    if provenance.get("case_count") != len(case_ids):
        raise ValueError("provenance case_count mismatch")
    expected_input_tree_hash = _input_tree_sha256_from_manifest(manifest_rows)
    if provenance.get("input_tree_sha256") != expected_input_tree_hash:
        raise ValueError("provenance input_tree_sha256 mismatch")
    generator = provenance.get("generator", {})
    if not isinstance(generator, dict):
        raise ValueError("provenance generator must be an object")
    if generator.get("reference_project_commit") != expected_project_commit:
        raise ValueError("provenance reference project commit is stale")
    compatible_manifest_hashes = resource_manifest.get("compatible_manifest_sha256", [])
    if not isinstance(compatible_manifest_hashes, list):
        raise ValueError("Frozen CLAMP resource manifest compatibility hashes are invalid")
    accepted_manifest_hashes = {
        _sha256(resource_manifest_bytes),
        *(str(value) for value in compatible_manifest_hashes),
    }
    if generator.get("resource_manifest_sha256") not in accepted_manifest_hashes:
        raise ValueError("provenance resource-manifest SHA-256 is stale")
    phi_review = provenance.get("reviews", {}).get("phi", {})
    if phi_review.get("automated_screen") != "passed":
        raise ValueError("provenance must record a passed automated PHI screen")

    expected_files = {
        "cases.yaml",
        "manifest.csv",
        "provenance.json",
        "SHA256SUMS",
        *(str(record["input_path"]) for record in case_records),
        "clamp_expected/README.md",
        "intermediate_expected/README.md",
    }
    pending = lifecycle == "awaiting_legacy_runs"
    if pending:
        expected_files.update({"clamp_expected/PENDING", "intermediate_expected/PENDING"})
        _validate_pending_provenance(provenance)
    else:
        _validate_complete_provenance(
            provenance,
            expected_project_commit=expected_project_commit,
            expected_project_files=expected_project_files,
            expected_resource_files=expected_resource_files,
            expected_case_count=len(case_ids),
        )
        observed_fixture_counts = {
            "cases": len(case_ids),
            "sentences": 0,
            "tokens": 0,
            "final_entities": 0,
        }
        for case_id in case_ids:
            tsv_path = f"clamp_expected/{case_id}.tsv"
            json_path = f"intermediate_expected/{case_id}.json"
            expected_files.update({tsv_path, json_path})
            source_text = (
                (root / manifest_by_id[case_id]["input_path"])
                .read_bytes()
                .decode("utf-8", errors="strict")
            )
            tsv_entities = _validate_expected_tsv(
                root / tsv_path,
                case_id,
                source_text=source_text,
            )
            intermediate = _validate_intermediate_json(
                root / json_path,
                case_id,
                manifest_by_id[case_id],
                source_text=source_text,
            )
            if Counter(tsv_entities) != Counter(intermediate.final_entities):
                raise ValueError(
                    f"TSV/intermediate final-entity disagreement for fixture case {case_id}"
                )
            observed_fixture_counts["sentences"] += intermediate.sentence_count
            observed_fixture_counts["tokens"] += intermediate.token_count
            observed_fixture_counts["final_entities"] += len(intermediate.final_entities)
        _validate_complete_fixture_counts(provenance, observed_fixture_counts)

    actual_files = {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()}
    if actual_files != expected_files:
        missing = sorted(expected_files - actual_files)
        unexpected = sorted(actual_files - expected_files)
        raise ValueError(
            f"Fixture file inventory differs: missing={missing}, unexpected={unexpected}"
        )
    _validate_sha256s(root, expected_files - {"SHA256SUMS"})

    if pending and not allow_pending:
        raise FixturePendingError(
            "Fixture is awaiting two legacy CLAMP runs; rerun validation with "
            "allow_pending=True only for scaffold checks"
        )
    return FixtureValidation(
        fixture_version=FIXTURE_VERSION,
        lifecycle=lifecycle,
        case_count=len(case_ids),
        category_counts=dict(sorted(category_counts.items())),
        pending=pending,
    )


def write_sha256s(root: Path) -> None:
    root = root.expanduser().resolve()
    paths = sorted(path for path in root.rglob("*") if path.is_file() and path.name != "SHA256SUMS")
    lines = [f"{_sha256(path.read_bytes())}  {path.relative_to(root).as_posix()}" for path in paths]
    (root / "SHA256SUMS").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def find_phi_sentinels(text: str) -> tuple[str, ...]:
    if "\x00" in text:
        return ("nul_byte",)
    return tuple(name for name, pattern in PHI_SENTINELS.items() if pattern.search(text))


def _text_case(
    case_id: str,
    primary_category: str,
    description: str,
    text: str,
    *,
    tags: tuple[str, ...],
    resource_kind: str = "",
    resource_index: int | None = None,
    parameters: tuple[tuple[str, str], ...] = (),
) -> FixtureCase:
    return FixtureCase(
        case_id=case_id,
        primary_category=primary_category,
        description=description,
        tags=tags,
        payload=text.encode("utf-8"),
        resource_kind=resource_kind,
        resource_index=resource_index,
        parameters=parameters,
    )


def _dictionary_context(surface: str, semantic_tag: str) -> str:
    if semantic_tag == "ARDS":
        return f"{surface}."
    if semantic_tag == "Morphology":
        return f"bilateral {surface}."
    if semantic_tag == "location":
        return f"opacities {surface}."
    raise ValueError(f"Unexpected dictionary semantic tag: {semantic_tag}")


def _mixed_case(value: str) -> str:
    letters = 0
    result: list[str] = []
    for character in value:
        if character.isalpha():
            result.append(character.upper() if letters % 2 == 0 else character.lower())
            letters += 1
        else:
            result.append(character)
    return "".join(result)


def _validate_generated_cases(cases: list[FixtureCase]) -> None:
    ids = [case.case_id for case in cases]
    for case_id in ids:
        _validate_case_id(case_id)
    if len(ids) != len(set(ids)):
        raise ValueError("Generated fixture case IDs are not unique")
    counts = dict(sorted(Counter(case.primary_category for case in cases).items()))
    if counts != EXPECTED_CATEGORY_COUNTS:
        raise ValueError(f"Generated fixture coverage differs from frozen matrix: {counts}")
    for case in cases:
        text = case.payload.decode("utf-8", errors="strict")
        sentinels = find_phi_sentinels(text)
        if sentinels:
            raise ValueError(f"Generated case {case.case_id} contains PHI sentinel(s): {sentinels}")


def _write_manifest(path: Path, cases: tuple[FixtureCase, ...]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS, lineterminator="\n")
        writer.writeheader()
        for case in cases:
            text = case.payload.decode("utf-8", errors="strict")
            writer.writerow(
                {
                    "fixture_version": FIXTURE_VERSION,
                    "case_id": case.case_id,
                    "primary_category": case.primary_category,
                    "input_path": case.input_path,
                    "source_kind": "synthetic_deterministic",
                    "encoding": "utf-8",
                    "line_ending": _line_ending(case.payload),
                    "trailing_newline": str(case.payload.endswith((b"\n", b"\r"))).lower(),
                    "byte_count": len(case.payload),
                    "codepoint_count": len(text),
                    "utf16_code_unit_count": len(text.encode("utf-16-le")) // 2,
                    "input_sha256": _sha256(case.payload),
                    "resource_kind": case.resource_kind,
                    "resource_index": case.resource_index or "",
                    "phi_automated_screen": "passed",
                }
            )


def _line_ending(payload: bytes) -> str:
    without_crlf = payload.replace(b"\r\n", b"")
    has_crlf = b"\r\n" in payload
    has_lf = b"\n" in without_crlf
    has_cr = b"\r" in without_crlf
    present = sum((has_crlf, has_lf, has_cr))
    if present > 1:
        return "mixed"
    if has_crlf:
        return "crlf"
    if has_lf:
        return "lf"
    if has_cr:
        return "cr"
    return "none"


def _input_tree_sha256(cases: tuple[FixtureCase, ...]) -> str:
    lines = [f"{case.case_id}\t{_sha256(case.payload)}" for case in cases]
    return _sha256(("\n".join(lines) + "\n").encode())


def _input_tree_sha256_from_manifest(rows: list[dict[str, str]]) -> str:
    lines = [f"{row['case_id']}\t{row['input_sha256']}" for row in rows]
    return _sha256(("\n".join(lines) + "\n").encode())


def _validate_case_id(case_id: str) -> None:
    if not CASE_ID_PATTERN.fullmatch(case_id):
        raise ValueError(f"Unsafe fixture case_id: {case_id!r}")


def _validate_relative_path(value: str) -> None:
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or path.as_posix() != value:
        raise ValueError(f"Unsafe fixture-relative path: {value!r}")


def _read_mapping(path: Path) -> dict[str, Any]:
    if path.suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
    else:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a mapping in {path}")
    return payload


def _validate_pending_provenance(provenance: dict[str, Any]) -> None:
    if provenance.get("runs") != []:
        raise ValueError("Pending fixture provenance must not claim legacy runs")
    if provenance.get("determinism", {}).get("status") != "pending":
        raise ValueError("Pending fixture determinism status must be pending")
    reviews = provenance.get("reviews", {})
    if reviews.get("phi", {}).get("manual_review") != "pending":
        raise ValueError("Pending fixture PHI manual review must be pending")
    if reviews.get("redistribution", {}).get("status") != "pending":
        raise ValueError("Pending fixture redistribution review must be pending")


def _validate_complete_provenance(
    provenance: dict[str, Any],
    *,
    expected_project_commit: str,
    expected_project_files: dict[str, str],
    expected_resource_files: dict[str, str],
    expected_case_count: int,
) -> None:
    unresolved = {"", "VERIFY", "TODO", "PENDING", None}
    runtime = provenance.get("legacy_runtime", {})
    for field in (
        "clamp_version",
        "clamp_build",
        "operating_system",
        "java_version",
        "locale",
        "timezone",
        "pipeline_export_settings",
    ):
        if runtime.get(field) in unresolved:
            raise ValueError(f"Complete fixture provenance lacks legacy_runtime.{field}")
    runs = provenance.get("runs")
    if not isinstance(runs, list) or len(runs) != 2:
        raise ValueError("Complete fixture provenance requires exactly two legacy runs")
    for expected_number, run in enumerate(runs, start=1):
        if not isinstance(run, dict) or run.get("run_number") != expected_number:
            raise ValueError("Legacy runs must be numbered 1 and 2")
        for field in ("started_at", "completed_at", "output_manifest_sha256"):
            if run.get(field) in unresolved:
                raise ValueError(f"Legacy run {expected_number} lacks {field}")
        if not _is_sha256(run["output_manifest_sha256"]):
            raise ValueError(f"Legacy run {expected_number} manifest digest is invalid")
        started = _parse_utc_timestamp(run["started_at"], f"run {expected_number} started_at")
        completed = _parse_utc_timestamp(run["completed_at"], f"run {expected_number} completed_at")
        if started >= completed:
            raise ValueError(f"Legacy run {expected_number} timestamps are not chronological")
    determinism = provenance.get("determinism", {})
    if not isinstance(determinism, dict):
        raise ValueError("Complete fixture determinism must be an object")
    if determinism.get("status") != "passed":
        raise ValueError("Complete fixture determinism status must be passed")
    for field in (
        "raw_order_required",
        "txt_row_order_stable",
        "xmi_entity_order_stable",
        "exact_sentence_annotations",
        "exact_token_annotations",
        "exact_entity_multisets",
    ):
        if determinism.get(field) not in {True, False}:
            raise ValueError(f"Complete fixture must resolve determinism.{field}")
    if determinism["raw_order_required"] != determinism["txt_row_order_stable"]:
        raise ValueError("raw_order_required must reflect observed TXT row-order stability")
    for stable_field, count_field in (
        ("txt_row_order_stable", "txt_order_difference_documents"),
        ("xmi_entity_order_stable", "xmi_order_difference_documents"),
    ):
        difference_count = determinism.get(count_field)
        if not isinstance(difference_count, int) or isinstance(difference_count, bool):
            raise ValueError(f"Complete fixture determinism.{count_field} must be an integer")
        if difference_count < 0 or difference_count > expected_case_count:
            raise ValueError(f"Complete fixture determinism.{count_field} is out of range")
        if determinism[stable_field] != (difference_count == 0):
            raise ValueError(
                f"Complete fixture determinism.{stable_field} conflicts with {count_field}"
            )
    if determinism.get("exact_sentence_annotations") is not True:
        raise ValueError("Complete fixture repeat runs must have exact sentence annotations")
    if determinism.get("exact_token_annotations") is not True:
        raise ValueError("Complete fixture repeat runs must have exact token annotations")
    if determinism.get("exact_entity_multisets") is not True:
        raise ValueError("Complete fixture repeat runs must have exact entity multisets")
    if determinism.get("required_run_count") != 2:
        raise ValueError("Complete fixture determinism requires exactly two runs")

    output_contract = provenance.get("output_contract", {})
    if (
        not isinstance(output_contract, dict)
        or output_contract.get("raw_xmi_is_not_committed") is not True
    ):
        raise ValueError("Complete fixture must prohibit committed raw XMI")

    legacy_import = provenance.get("legacy_import")
    if not isinstance(legacy_import, dict):
        raise ValueError("Complete fixture lacks importer-generated legacy_import provenance")
    if legacy_import.get("generated_only_from_returned_legacy_clamp") is not True:
        raise ValueError("Complete fixture expected output must come from returned legacy CLAMP")
    if legacy_import.get("raw_xmi_committed") is not False:
        raise ValueError("Complete fixture legacy_import must record raw_xmi_committed=false")

    runtime_details = legacy_import.get("runtime_details")
    if not isinstance(runtime_details, dict):
        raise ValueError("Complete fixture lacks legacy runtime details")
    if runtime_details.get("project_commit") != expected_project_commit:
        raise ValueError("Complete fixture legacy project commit differs from the frozen contract")
    if runtime_details.get("project_files_sha256") != expected_project_files:
        raise ValueError("Complete fixture legacy project hashes differ from the frozen contract")
    if runtime_details.get("resources_sha256") != expected_resource_files:
        raise ValueError("Complete fixture legacy resource hashes differ from the frozen contract")
    runtime_clamp = runtime_details.get("clamp")
    if not isinstance(runtime_clamp, dict) or runtime_clamp.get("version") != "1.6.6":
        raise ValueError("Complete fixture must record the licensed CLAMP 1.6.6 runtime")
    if runtime.get("clamp_version") != runtime_clamp.get("version"):
        raise ValueError("Complete fixture CLAMP version differs across provenance sections")
    if runtime.get("clamp_build") != runtime_clamp.get("build"):
        raise ValueError("Complete fixture CLAMP build differs across provenance sections")
    if runtime.get("pipeline_export_settings") != runtime_details.get("export_settings"):
        raise ValueError("Complete fixture export settings differ across provenance sections")

    fixture_counts = legacy_import.get("fixture_counts")
    if not isinstance(fixture_counts, dict) or fixture_counts.get("cases") != expected_case_count:
        raise ValueError("Complete fixture legacy import case count is inconsistent")
    for field in ("sentences", "tokens", "final_entities"):
        count = fixture_counts.get(field) if isinstance(fixture_counts, dict) else None
        if not isinstance(count, int) or isinstance(count, bool) or count < 0:
            raise ValueError(f"Complete fixture legacy import {field} count is invalid")

    expected_output_file_count = expected_case_count * 2
    for expected_label in ("run_1", "run_2"):
        record = legacy_import.get(expected_label)
        if not isinstance(record, dict) or record.get("run_label") != expected_label:
            raise ValueError(f"Complete fixture lacks importer record for {expected_label}")
        if record.get("output_file_count") != expected_output_file_count:
            raise ValueError(f"Complete fixture {expected_label} output file count is invalid")
        if not _is_sha256(record.get("output_manifest_sha256")):
            raise ValueError(f"Complete fixture {expected_label} manifest digest is invalid")
        for field in ("started_at_utc", "finished_at_utc", "recorded_at_utc"):
            if record.get(field) in unresolved:
                raise ValueError(f"Complete fixture {expected_label} lacks {field}")
        started = _parse_utc_timestamp(record["started_at_utc"], f"{expected_label}.started_at_utc")
        finished = _parse_utc_timestamp(
            record["finished_at_utc"], f"{expected_label}.finished_at_utc"
        )
        recorded = _parse_utc_timestamp(
            record["recorded_at_utc"], f"{expected_label}.recorded_at_utc"
        )
        if not started < finished <= recorded:
            raise ValueError(f"Complete fixture {expected_label} timestamps are not chronological")

    for index, run in enumerate(runs, start=1):
        import_record = legacy_import[f"run_{index}"]
        if run["output_manifest_sha256"] != import_record["output_manifest_sha256"]:
            raise ValueError(f"Legacy run {index} digest differs across provenance sections")
        if run["started_at"] != import_record["started_at_utc"]:
            raise ValueError(f"Legacy run {index} start differs across provenance sections")
        if run["completed_at"] != import_record["finished_at_utc"]:
            raise ValueError(f"Legacy run {index} completion differs across provenance sections")
    reviews = provenance.get("reviews", {})
    phi = reviews.get("phi", {})
    if phi.get("manual_review") != "approved":
        raise ValueError("Complete fixture requires approved manual PHI review")
    for field in ("reviewer", "reviewed_at"):
        if phi.get(field) in unresolved:
            raise ValueError(f"Complete fixture PHI review lacks {field}")
    _parse_review_date(phi["reviewed_at"])
    redistribution = reviews.get("redistribution", {})
    if redistribution.get("status") != "approved":
        raise ValueError("Complete fixture requires approved redistribution review")
    for field in ("authority", "evidence"):
        if redistribution.get(field) in unresolved:
            raise ValueError(f"Complete fixture redistribution review lacks {field}")


def _is_sha256(value: Any) -> bool:
    return bool(re.fullmatch(r"[0-9a-f]{64}", str(value or "")))


def _parse_utc_timestamp(value: Any, field: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"Complete fixture {field} must be ISO-8601") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != UTC.utcoffset(parsed):
        raise ValueError(f"Complete fixture {field} must include a UTC offset")
    return parsed


def _parse_review_date(value: Any) -> None:
    text = str(value)
    try:
        date.fromisoformat(text)
    except ValueError:
        _parse_utc_timestamp(text, "PHI reviewed_at")


def _validate_complete_fixture_counts(
    provenance: dict[str, Any],
    observed: dict[str, int],
) -> None:
    legacy_import = provenance.get("legacy_import")
    if not isinstance(legacy_import, dict):
        raise ValueError("Complete fixture lacks importer-generated legacy_import provenance")
    declared = legacy_import.get("fixture_counts")
    if declared != observed:
        raise ValueError(
            "Complete fixture legacy_import.fixture_counts differs from normalized fixture "
            f"contents: expected {observed}, found {declared}"
        )


def _validate_expected_tsv(
    path: Path,
    case_id: str,
    *,
    source_text: str,
) -> tuple[NormalizedExpectedEntity, ...]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if tuple(reader.fieldnames or ()) != EXPECTED_ENTITY_FIELDS:
            raise ValueError(f"Unexpected normalized entity schema: {path}")
        result: list[NormalizedExpectedEntity] = []
        occurrences: Counter[NormalizedExpectedEntity] = Counter()
        for row_number, row in enumerate(reader, start=2):
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"Malformed normalized entity row {row_number}: {path}")
            if row["clamp_doc_id"] != case_id:
                raise ValueError(f"Wrong document ID in {path} row {row_number}")
            start = _parse_tsv_nonnegative_int(row["start"], "start", path, row_number)
            end = _parse_tsv_nonnegative_int(row["end"], "end", path, row_number)
            if end <= start:
                raise ValueError(
                    f"Normalized entity span must be non-empty in {path} row {row_number}"
                )
            entity_text = row["entity_text"]
            if not entity_text:
                raise ValueError(f"Blank entity_text in {path} row {row_number}")
            covered_text = _utf16_covered_text(
                source_text,
                start,
                end,
                path=path,
                row_number=row_number,
                stage="entity",
            )
            if entity_text != covered_text:
                raise ValueError(
                    f"Normalized entity covered text differs from input in {path} row {row_number}"
                )
            semantic_tag = _required_tsv_text(row["semantic_tag"], "semantic_tag", path, row_number)
            assertion = _required_tsv_text(row["assertion"], "assertion", path, row_number)
            cui = _parse_tsv_nullable(row["cui"], "cui", path, row_number)
            attribute = _parse_tsv_nullable(row["attribute"], "attribute", path, row_number)
            raw_order = _parse_tsv_nonnegative_int(row["raw_order"], "raw_order", path, row_number)
            if raw_order != len(result):
                raise ValueError(
                    f"Normalized entity raw_order differs from row order in {path} row {row_number}"
                )
            entity = (start, end, semantic_tag, assertion, cui, attribute, entity_text)
            duplicate_occurrence = _parse_tsv_nonnegative_int(
                row["duplicate_occurrence"],
                "duplicate_occurrence",
                path,
                row_number,
            )
            if duplicate_occurrence != occurrences[entity]:
                raise ValueError(
                    "Normalized entity duplicate_occurrence is inconsistent in "
                    f"{path} row {row_number}"
                )
            occurrences[entity] += 1
            result.append(entity)
    return tuple(result)


def _validate_intermediate_json(
    path: Path,
    case_id: str,
    manifest_row: dict[str, str],
    *,
    source_text: str,
) -> _IntermediateValidation:
    payload = _read_mapping(path)
    if set(payload) != INTERMEDIATE_FIELDS:
        raise ValueError(f"Unexpected intermediate top-level schema: {path}")
    if type(payload.get("schema_version")) is not int or (
        payload["schema_version"] != FIXTURE_SCHEMA_VERSION
    ):
        raise ValueError(f"Unexpected intermediate schema_version: {path}")
    if not isinstance(payload.get("case_id"), str) or payload["case_id"] != case_id:
        raise ValueError(f"Wrong intermediate case_id: {path}")
    if (
        not isinstance(payload.get("source_text_sha256"), str)
        or payload["source_text_sha256"] != manifest_row["input_sha256"]
        or payload["source_text_sha256"] != _sha256(source_text.encode("utf-8"))
    ):
        raise ValueError(f"Wrong intermediate source hash: {path}")
    if payload.get("offset_coordinate_system") != "utf16_code_units":
        raise ValueError(f"Wrong intermediate offset convention: {path}")
    if payload.get("interval_convention") != "half_open":
        raise ValueError(f"Wrong intermediate interval convention: {path}")
    legacy_hashes = payload.get("legacy_output_sha256")
    if not isinstance(legacy_hashes, dict) or set(legacy_hashes) != {"run_1", "run_2"}:
        raise ValueError(f"Intermediate output lacks both legacy-run hashes: {path}")
    for run_label in ("run_1", "run_2"):
        run_hashes = legacy_hashes[run_label]
        if not isinstance(run_hashes, dict) or set(run_hashes) != {"txt", "xmi"}:
            raise ValueError(f"Intermediate {run_label} hashes are invalid: {path}")
        if not all(
            isinstance(run_hashes[kind], str) and _is_sha256(run_hashes[kind])
            for kind in ("txt", "xmi")
        ):
            raise ValueError(f"Intermediate {run_label} digest is invalid: {path}")
    for field in ("sentences", "tokens", "final_entities"):
        rows = payload.get(field)
        if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
            raise ValueError(f"Intermediate {field} must be a list of objects: {path}")

    offsets = Utf16OffsetMap.from_text(source_text)
    sentence_count = _validate_intermediate_spans(
        payload["sentences"],
        stage="sentences",
        number_field="sentence_number",
        path=path,
        source_text=source_text,
        offsets=offsets,
    )
    token_count = _validate_intermediate_spans(
        payload["tokens"],
        stage="tokens",
        number_field="token_number",
        path=path,
        source_text=source_text,
        offsets=offsets,
    )
    final_entities = _validate_intermediate_entities(
        payload["final_entities"],
        path=path,
        source_text=source_text,
        offsets=offsets,
    )
    return _IntermediateValidation(
        sentence_count=sentence_count,
        token_count=token_count,
        final_entities=final_entities,
    )


def _validate_intermediate_spans(
    rows: list[dict[str, Any]],
    *,
    stage: str,
    number_field: str,
    path: Path,
    source_text: str,
    offsets: Utf16OffsetMap,
) -> int:
    previous_end = 0
    for position, row in enumerate(rows):
        if set(row) != INTERMEDIATE_ROW_FIELDS[stage]:
            raise ValueError(f"Unexpected intermediate {stage} row schema in {path}")
        start = _required_json_nonnegative_int(row["start"], "start", path, position)
        end = _required_json_nonnegative_int(row["end"], "end", path, position)
        if end <= start:
            raise ValueError(f"Intermediate {stage} span must be non-empty in {path}")
        if position and start < previous_end:
            raise ValueError(f"Intermediate {stage} spans are not in source order: {path}")
        number = _required_json_nonnegative_int(row[number_field], number_field, path, position)
        if number != position:
            raise ValueError(
                f"Intermediate {stage} {number_field} must be contiguous zero-based order: {path}"
            )
        covered_text = _required_json_text(
            row["covered_text"], "covered_text", path, position, allow_empty=False
        )
        actual = _utf16_covered_text(
            source_text,
            start,
            end,
            path=path,
            row_number=position,
            stage=stage,
            offsets=offsets,
        )
        if covered_text != actual:
            raise ValueError(f"Intermediate {stage} covered text differs from input: {path}")
        previous_end = end
    return len(rows)


def _validate_intermediate_entities(
    rows: list[dict[str, Any]],
    *,
    path: Path,
    source_text: str,
    offsets: Utf16OffsetMap,
) -> tuple[NormalizedExpectedEntity, ...]:
    result: list[NormalizedExpectedEntity] = []
    for position, row in enumerate(rows):
        if set(row) != INTERMEDIATE_ROW_FIELDS["final_entities"]:
            raise ValueError(f"Unexpected intermediate final_entities row schema in {path}")
        start = _required_json_nonnegative_int(row["start"], "start", path, position)
        end = _required_json_nonnegative_int(row["end"], "end", path, position)
        if end <= start:
            raise ValueError(f"Intermediate final entity span must be non-empty in {path}")
        raw_order = _required_json_nonnegative_int(row["raw_order"], "raw_order", path, position)
        if raw_order != position:
            raise ValueError(f"Intermediate final entity raw_order differs from row order: {path}")
        semantic_tag = _required_json_text(
            row["semantic_tag"], "semantic_tag", path, position, allow_empty=False
        )
        assertion = _required_json_text(
            row["assertion"], "assertion", path, position, allow_empty=False
        )
        cui = _nullable_json_text(row["cui"], "cui", path, position)
        attribute = _nullable_json_text(row["attribute"], "attribute", path, position)
        covered_text = _required_json_text(
            row["covered_text"], "covered_text", path, position, allow_empty=False
        )
        actual = _utf16_covered_text(
            source_text,
            start,
            end,
            path=path,
            row_number=position,
            stage="final_entities",
            offsets=offsets,
        )
        if covered_text != actual:
            raise ValueError(f"Intermediate final entity covered text differs from input: {path}")
        result.append((start, end, semantic_tag, assertion, cui, attribute, covered_text))
    return tuple(result)


def _parse_tsv_nonnegative_int(value: str, field: str, path: Path, row_number: int) -> int:
    if not re.fullmatch(r"0|[1-9][0-9]*", value):
        raise ValueError(f"Invalid normalized entity {field} in {path} row {row_number}")
    return int(value)


def _required_tsv_text(value: str, field: str, path: Path, row_number: int) -> str:
    if not value or value != value.strip():
        raise ValueError(f"Invalid normalized entity {field} in {path} row {row_number}")
    return value


def _parse_tsv_nullable(value: str, field: str, path: Path, row_number: int) -> str | None:
    if value == r"\N":
        return None
    return _required_tsv_text(value, field, path, row_number)


def _required_json_nonnegative_int(value: Any, field: str, path: Path, position: int) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"Invalid intermediate {field} at position {position}: {path}")
    return value


def _required_json_text(
    value: Any,
    field: str,
    path: Path,
    position: int,
    *,
    allow_empty: bool,
) -> str:
    if not isinstance(value, str) or (not allow_empty and not value):
        raise ValueError(f"Invalid intermediate {field} at position {position}: {path}")
    return value


def _nullable_json_text(value: Any, field: str, path: Path, position: int) -> str | None:
    if value is None:
        return None
    return _required_json_text(value, field, path, position, allow_empty=False)


def _utf16_covered_text(
    source_text: str,
    start: int,
    end: int,
    *,
    path: Path,
    row_number: int,
    stage: str,
    offsets: Utf16OffsetMap | None = None,
) -> str:
    offset_map = offsets or Utf16OffsetMap.from_text(source_text)
    try:
        source_start, source_end = offset_map.python_span(start, end)
    except ValueError as exc:
        raise ValueError(
            f"Invalid UTF-16 {stage} span in {path} row/position {row_number}: {exc}"
        ) from exc
    return source_text[source_start:source_end]


def _validate_sha256s(root: Path, expected_paths: set[str]) -> None:
    rows: dict[str, str] = {}
    for line_number, line in enumerate(
        (root / "SHA256SUMS").read_text(encoding="utf-8").splitlines(), start=1
    ):
        match = re.fullmatch(r"([0-9a-f]{64})  (.+)", line)
        if match is None:
            raise ValueError(f"Malformed SHA256SUMS line {line_number}")
        digest, relative = match.groups()
        _validate_relative_path(relative)
        if relative in rows:
            raise ValueError(f"Duplicate SHA256SUMS path: {relative}")
        rows[relative] = digest
    if set(rows) != expected_paths:
        raise ValueError("SHA256SUMS file inventory differs from fixture inventory")
    for relative, expected in rows.items():
        actual = _sha256((root / relative).read_bytes())
        if actual != expected:
            raise ValueError(f"SHA-256 mismatch for fixture file: {relative}")


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()
