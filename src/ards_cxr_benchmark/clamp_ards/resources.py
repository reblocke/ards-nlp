from __future__ import annotations

import hashlib
import importlib.resources
import json
import os
from dataclasses import dataclass
from pathlib import Path, PurePath
from typing import Any


@dataclass(frozen=True)
class DictionaryEntry:
    term: str
    semantic_tag: str
    index: int


@dataclass(frozen=True)
class AssertionCue:
    phrase: str
    category: str
    index: int


@dataclass(frozen=True)
class PhenotypeSpec:
    version: str
    sha256: str
    attribution: str
    dictionary: tuple[DictionaryEntry, ...]
    excluded_dictionary_terms: frozenset[str]
    newline_ends_sentence: bool
    break_long_sentences: bool
    max_sentence_tokens: int
    assertion_scope_tokens: int
    remove_assertions: frozenset[str]
    promotion_rules: tuple[tuple[str, str], ...]
    max_intervening_tokens: int
    final_semantic_tag: str
    remove_unpromoted_semantic_tags: frozenset[str]


@dataclass(frozen=True)
class ClampResources:
    project_dir: Path
    dictionary: tuple[DictionaryEntry, ...]
    assertion_cues: tuple[AssertionCue, ...]
    abbreviations: frozenset[str]
    split_patterns: tuple[str, ...]
    section_headers: tuple[str, ...]
    delimiters: frozenset[str]
    no_split_strings: tuple[str, ...]
    newline_ends_sentence: bool
    break_long_sentences: bool
    max_sentence_tokens: int
    assertion_scope_tokens: int
    excluded_dictionary_terms: frozenset[str]
    remove_assertions: frozenset[str]
    promotion_rules: tuple[tuple[str, str], ...]
    max_intervening_tokens: int
    final_semantic_tag: str
    remove_unpromoted_semantic_tags: frozenset[str]
    resource_sha256: dict[str, str]
    phenotype_spec_version: str
    phenotype_spec_sha256: str


PROJECT_DIR_ENV = "ARDS_CLAMP_PROJECT_DIR"
RESOURCE_MANIFEST_ENV = "ARDS_CLAMP_RESOURCE_MANIFEST"
DEFAULT_EXTERNAL_PROJECT_DIR = Path("data/external/clamp_ards_project")
PACKAGED_RESOURCE_MANIFEST = "data/clamp_ards_resource_manifest.json"
PACKAGED_PHENOTYPE_SPEC = "data/legacy_ards_phenotype_spec.json"

ABBREVIATION_RESOURCE = "Components/Sentence detector/DF_Clamp_sentence_detector/defaultAbbrs.txt"
ASSERTION_RESOURCE = "Components/Assertion classifier/DF_NegEx_assertion/defaultNegexDict.txt"
TOKEN_RULE_RESOURCE = "Components/Tokenizer/DF_Clamp_tokenizer/defaultTokenRule.txt"
EXTERNAL_RESOURCE_PATHS = frozenset(
    {
        ABBREVIATION_RESOURCE,
        ASSERTION_RESOURCE,
        TOKEN_RULE_RESOURCE,
    }
)
LEGACY_ASSERTION_CATEGORY_COUNTS = {
    "pseNegPhrases": 16,
    "negPhrases": 127,
    "postNegPhrases": 28,
    "conjunctions": 69,
}


def default_project_dir() -> Path:
    configured = os.environ.get(PROJECT_DIR_ENV)
    if configured:
        candidate = Path(configured).expanduser()
        if not candidate.is_dir():
            raise FileNotFoundError(
                f"{PROJECT_DIR_ENV} does not name an external CLAMP resource directory: "
                f"{candidate}. Required files: {_required_resource_text()}"
            )
        return candidate.resolve()

    candidate = Path.cwd() / DEFAULT_EXTERNAL_PROJECT_DIR
    if candidate.is_dir():
        return candidate.resolve()
    raise FileNotFoundError(
        "Could not locate the separately licensed CLAMP compatibility resources at "
        f"{candidate}. Set {PROJECT_DIR_ENV} or pass --project-dir. Required files: "
        f"{_required_resource_text()}"
    )


def configured_resource_manifest_path() -> Path | None:
    configured = os.environ.get(RESOURCE_MANIFEST_ENV)
    if not configured:
        return None
    path = Path(configured).expanduser()
    if not path.is_file():
        raise FileNotFoundError(
            f"{RESOURCE_MANIFEST_ENV} does not name a resource manifest: {path}"
        )
    return path.resolve()


def default_resource_manifest_path() -> Path:
    configured = configured_resource_manifest_path()
    if configured is not None:
        return configured
    checkout_manifest = (
        Path(__file__).resolve().parents[3] / "config" / "clamp_ards_resource_manifest.json"
    )
    if checkout_manifest.is_file():
        return checkout_manifest
    raise FileNotFoundError(
        f"Set {RESOURCE_MANIFEST_ENV} to a CLAMP resource manifest for this operation"
    )


def load_clamp_resources(
    project_dir: Path | None = None,
    *,
    manifest_path: Path | None = None,
) -> ClampResources:
    root = (project_dir or default_project_dir()).expanduser().resolve()
    spec = _load_phenotype_spec()
    selected_manifest = manifest_path or configured_resource_manifest_path()
    manifest, source = _load_resource_manifest(selected_manifest)
    runtime_required = _validated_runtime_required_files(manifest, source=source)
    _validate_manifest_spec(manifest, spec, source=source)
    _require_external_files(root, runtime_required)

    assertion_cues = _load_assertion_cues(root / ASSERTION_RESOURCE)
    category_counts: dict[str, int] = {}
    for cue in assertion_cues:
        category_counts[cue.category] = category_counts.get(cue.category, 0) + 1
    expected_categories = _expected_assertion_categories(manifest, source=source)
    if category_counts != expected_categories:
        raise ValueError(
            f"Unexpected CLAMP assertion cue counts: {category_counts}; "
            f"expected {expected_categories}"
        )

    delimiters, no_split_strings = _load_token_rules(root / TOKEN_RULE_RESOURCE)
    hashes = _validated_resource_hashes(
        root,
        manifest,
        source=source,
        runtime_required=runtime_required,
    )
    return ClampResources(
        project_dir=root,
        dictionary=spec.dictionary,
        assertion_cues=assertion_cues,
        abbreviations=frozenset(
            (root / ABBREVIATION_RESOURCE)
            .read_text(encoding="utf-8", errors="replace")
            .splitlines()
        ),
        split_patterns=(),
        section_headers=(),
        delimiters=frozenset(delimiters),
        no_split_strings=tuple(no_split_strings),
        newline_ends_sentence=spec.newline_ends_sentence,
        break_long_sentences=spec.break_long_sentences,
        max_sentence_tokens=spec.max_sentence_tokens,
        assertion_scope_tokens=spec.assertion_scope_tokens,
        excluded_dictionary_terms=spec.excluded_dictionary_terms,
        remove_assertions=spec.remove_assertions,
        promotion_rules=spec.promotion_rules,
        max_intervening_tokens=spec.max_intervening_tokens,
        final_semantic_tag=spec.final_semantic_tag,
        remove_unpromoted_semantic_tags=spec.remove_unpromoted_semantic_tags,
        resource_sha256=hashes,
        phenotype_spec_version=spec.version,
        phenotype_spec_sha256=spec.sha256,
    )


def _load_phenotype_spec() -> PhenotypeSpec:
    source = f"ards_cxr_benchmark.clamp_ards:{PACKAGED_PHENOTYPE_SPEC}"
    try:
        payload_bytes = (
            importlib.resources.files("ards_cxr_benchmark.clamp_ards")
            .joinpath(PACKAGED_PHENOTYPE_SPEC)
            .read_bytes()
        )
    except (FileNotFoundError, ModuleNotFoundError, OSError) as exc:
        raise FileNotFoundError(
            f"Packaged ARDS phenotype specification is unavailable: {source}"
        ) from exc
    try:
        payload = json.loads(payload_bytes)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid ARDS phenotype specification JSON: {source}") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError(f"Unsupported ARDS phenotype specification: {source}")

    attribution = _mapping(payload, "attribution", source)
    dictionary_config = _mapping(payload, "dictionary", source)
    if dictionary_config.get("case_sensitive") is not False:
        raise ValueError("ARDS phenotype compatibility requires case_sensitive=false")
    if dictionary_config.get("stemming") is not True:
        raise ValueError("ARDS phenotype compatibility requires stemming=true")
    raw_entries = dictionary_config.get("entries")
    if not isinstance(raw_entries, list):
        raise ValueError(f"ARDS phenotype dictionary entries are invalid: {source}")
    entries: list[DictionaryEntry] = []
    for index, raw in enumerate(raw_entries):
        if not isinstance(raw, dict):
            raise ValueError(f"ARDS phenotype dictionary entry {index + 1} is invalid")
        term = raw.get("term")
        semantic_tag = raw.get("semantic_tag")
        if not isinstance(term, str) or not term or not isinstance(semantic_tag, str):
            raise ValueError(f"ARDS phenotype dictionary entry {index + 1} is invalid")
        entries.append(DictionaryEntry(term=term, semantic_tag=semantic_tag, index=index))
    if len(entries) != 23:
        raise ValueError(f"Expected 23 authorized phenotype terms, found {len(entries)}")

    sentence = _mapping(payload, "sentence_detection", source)
    assertion = _mapping(payload, "assertion", source)
    postprocessing = _mapping(payload, "postprocessing", source)
    raw_rules = postprocessing.get("promotion_rules")
    if not isinstance(raw_rules, list) or not raw_rules:
        raise ValueError(f"ARDS phenotype promotion rules are invalid: {source}")
    promotion_rules: list[tuple[str, str]] = []
    for rule in raw_rules:
        if not isinstance(rule, dict):
            raise ValueError(f"ARDS phenotype promotion rule is invalid: {source}")
        first = rule.get("first_semantic_tag")
        promote = rule.get("promote_semantic_tag")
        if not isinstance(first, str) or not isinstance(promote, str):
            raise ValueError(f"ARDS phenotype promotion rule is invalid: {source}")
        promotion_rules.append((first, promote))

    return PhenotypeSpec(
        version=_required_string(payload, "phenotype_spec_version", source),
        sha256=hashlib.sha256(payload_bytes).hexdigest(),
        attribution=_required_string(attribution, "design_author", source),
        dictionary=tuple(entries),
        excluded_dictionary_terms=_string_set(
            dictionary_config,
            "excluded_legacy_surface_forms",
            source,
        ),
        newline_ends_sentence=_required_bool(sentence, "newline_ends_sentence", source),
        break_long_sentences=_required_bool(sentence, "break_long_sentences", source),
        max_sentence_tokens=_positive_int(sentence, "max_sentence_tokens", source),
        assertion_scope_tokens=_positive_int(assertion, "scope_tokens", source),
        remove_assertions=_string_set(postprocessing, "remove_assertions", source),
        promotion_rules=tuple(promotion_rules),
        max_intervening_tokens=_nonnegative_int(
            postprocessing,
            "max_intervening_tokens",
            source,
        ),
        final_semantic_tag=_required_string(
            postprocessing,
            "final_semantic_tag",
            source,
        ),
        remove_unpromoted_semantic_tags=_string_set(
            postprocessing,
            "remove_unpromoted_semantic_tags",
            source,
        ),
    )


def _load_assertion_cues(path: Path) -> tuple[AssertionCue, ...]:
    cues: list[AssertionCue] = []
    for index, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        fields = line.split("\t")
        if len(fields) != 2 or not all(fields):
            raise ValueError(f"Malformed CLAMP assertion row {index + 1}: {line!r}")
        cues.append(AssertionCue(phrase=fields[0], category=fields[1], index=len(cues)))
    return tuple(cues)


def _load_token_rules(path: Path) -> tuple[set[str], list[str]]:
    delimiters: set[str] = set()
    no_split_strings: list[str] = []
    escapes = {r"\t": "\t", r"\'": "'", r"\\": "\\"}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("DELIMETER="):
            raw = line.split("=", 1)[1]
            delimiters.update(escapes.get(item, item) for item in raw.split("DEL"))
        elif line.startswith("STRING_NO_SPLIT="):
            no_split_strings.append(line.split("=", 1)[1])
    if not delimiters:
        raise ValueError(f"No token delimiters found in {path}")
    if any(len(value) != 1 for value in delimiters):
        raise ValueError(f"CLAMP delimiter entries must be single characters: {delimiters}")
    return delimiters, no_split_strings


def _validated_runtime_required_files(
    payload: dict[str, object],
    *,
    source: str,
) -> set[str]:
    version = payload.get("manifest_version", 2)
    if version not in {2, 3}:
        raise ValueError(f"Unsupported CLAMP resource manifest version in {source}: {version}")
    expected = payload.get("files")
    required_payload = payload.get("runtime_required_files")
    if (
        not isinstance(expected, dict)
        or not isinstance(required_payload, list)
        or not all(isinstance(value, str) for value in required_payload)
    ):
        raise ValueError(f"Invalid CLAMP resource manifest: {source}")
    unknown_required = sorted(set(required_payload) - set(expected))
    if unknown_required:
        raise ValueError(f"CLAMP manifest runtime files are absent from files: {unknown_required}")
    required = set(required_payload) & EXTERNAL_RESOURCE_PATHS
    if version == 3 and required != EXTERNAL_RESOURCE_PATHS:
        missing = sorted(EXTERNAL_RESOURCE_PATHS - required)
        extra = sorted(set(required_payload) - EXTERNAL_RESOURCE_PATHS)
        raise ValueError(
            f"Manifest v3 must require exactly the three external CLAMP resources: "
            f"missing={missing}, extra={extra}"
        )
    if version == 2 and required != EXTERNAL_RESOURCE_PATHS:
        missing = sorted(EXTERNAL_RESOURCE_PATHS - required)
        raise ValueError(f"Manifest v2 lacks external compatibility resources: {missing}")
    return required


def _validate_manifest_spec(
    manifest: dict[str, object],
    spec: PhenotypeSpec,
    *,
    source: str,
) -> None:
    if manifest.get("manifest_version", 2) == 2:
        return
    raw = manifest.get("phenotype_spec")
    if not isinstance(raw, dict):
        raise ValueError(f"Manifest v3 lacks phenotype_spec metadata: {source}")
    expected = {
        "attribution": spec.attribution,
        "license": "MIT",
        "sha256": spec.sha256,
        "version": spec.version,
    }
    changed = sorted(key for key, value in expected.items() if raw.get(key) != value)
    if changed:
        raise ValueError(
            f"Manifest phenotype specification metadata differs for {changed}: {source}"
        )


def _expected_assertion_categories(
    manifest: dict[str, object],
    *,
    source: str,
) -> dict[str, int]:
    contract = manifest.get("resource_contract", {})
    if contract == {} and manifest.get("manifest_version", 2) == 2:
        return dict(LEGACY_ASSERTION_CATEGORY_COUNTS)
    if not isinstance(contract, dict):
        raise ValueError(f"Invalid resource_contract in CLAMP manifest: {source}")
    raw = contract.get("assertion_category_counts")
    if not isinstance(raw, dict) or not all(
        isinstance(key, str) and isinstance(value, int) and value > 0 for key, value in raw.items()
    ):
        raise ValueError(f"Invalid assertion_category_counts in CLAMP manifest: {source}")
    return dict(raw)


def _validated_resource_hashes(
    root: Path,
    payload: dict[str, object],
    *,
    source: str,
    runtime_required: set[str],
) -> dict[str, str]:
    expected = payload.get("files")
    expected_sizes = payload.get("file_sizes", {})
    if not isinstance(expected, dict) or not all(
        isinstance(key, str) and isinstance(value, str) for key, value in expected.items()
    ):
        raise ValueError(f"Invalid CLAMP resource manifest: {source}")
    if not isinstance(expected_sizes, dict):
        raise ValueError(f"Invalid file_sizes in CLAMP resource manifest: {source}")
    actual: dict[str, str] = {}
    changed_sizes: list[str] = []
    for key in sorted(runtime_required):
        path = root / Path(key)
        if key in expected_sizes and path.stat().st_size != int(expected_sizes[key]):
            changed_sizes.append(key)
        actual[key] = _sha256(path)
    if changed_sizes:
        raise ValueError(f"CLAMP resource sizes differ from frozen manifest: {changed_sizes}")
    changed = sorted(key for key, value in actual.items() if value != expected.get(key))
    if changed:
        raise ValueError(
            "CLAMP resource hashes differ from frozen manifest: "
            f"missing=[], unexpected=[], changed={changed}"
        )
    return actual


def _load_resource_manifest(manifest_path: Path | None) -> tuple[dict[str, object], str]:
    if manifest_path is None:
        source = f"ards_cxr_benchmark.clamp_ards:{PACKAGED_RESOURCE_MANIFEST}"
        try:
            text = (
                importlib.resources.files("ards_cxr_benchmark.clamp_ards")
                .joinpath(PACKAGED_RESOURCE_MANIFEST)
                .read_text(encoding="utf-8")
            )
        except (FileNotFoundError, ModuleNotFoundError, OSError) as exc:
            raise FileNotFoundError(
                f"Packaged frozen CLAMP resource manifest is unavailable: {source}"
            ) from exc
    else:
        source = str(manifest_path)
        try:
            text = manifest_path.read_text(encoding="utf-8")
        except OSError as exc:
            raise FileNotFoundError(f"CLAMP resource manifest is unavailable: {source}") from exc
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid CLAMP resource manifest JSON: {source}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid CLAMP resource manifest: {source}")
    return payload, source


def _require_external_files(root: Path, required: set[str]) -> None:
    missing = sorted(path for path in required if not (root / path).is_file())
    if missing:
        raise FileNotFoundError(
            "Missing separately licensed CLAMP resource file(s) under "
            f"{root}: {missing}. Supply defaultAbbrs.txt, defaultNegexDict.txt, and "
            "defaultTokenRule.txt in their documented CLAMP project paths."
        )


def _required_resource_text() -> str:
    return ", ".join(sorted(EXTERNAL_RESOURCE_PATHS))


def _mapping(payload: dict[str, Any], key: str, source: str) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"Missing or invalid {key} in {source}")
    return value


def _required_string(payload: dict[str, Any], key: str, source: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Missing or invalid {key} in {source}")
    return value


def _required_bool(payload: dict[str, Any], key: str, source: str) -> bool:
    value = payload.get(key)
    if not isinstance(value, bool):
        raise ValueError(f"Missing or invalid {key} in {source}")
    return value


def _positive_int(payload: dict[str, Any], key: str, source: str) -> int:
    value = payload.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"Missing or invalid {key} in {source}")
    return value


def _nonnegative_int(payload: dict[str, Any], key: str, source: str) -> int:
    value = payload.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"Missing or invalid {key} in {source}")
    return value


def _string_set(payload: dict[str, Any], key: str, source: str) -> frozenset[str]:
    value = payload.get(key)
    if not isinstance(value, list) or not all(isinstance(item, str) and item for item in value):
        raise ValueError(f"Missing or invalid {key} in {source}")
    return frozenset(value)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest_key(path: PurePath, root: PurePath) -> str:
    """Return the platform-independent path format used by the frozen manifest."""

    return path.relative_to(root).as_posix()
