from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath

ALLOWED_STATUSES = {"cleared", "unresolved"}
NON_AFFIRMATIVE_PREFIXES = (
    "noassertion",
    "none",
    "unknown",
    "pending",
    "verify",
    "todo",
    "tbd",
    "unresolved",
    "not established",
    "not found",
    "no evidence",
    "no license",
)
PENDING_VERIFICATION_PREFIXES = (
    "confirm",
    "determine",
    "establish",
    "obtain",
    "pending",
    "review",
    "tbd",
    "todo",
    "unresolved",
    "verify",
)


@dataclass(frozen=True)
class ResourceFinding:
    check: str
    path: str
    detail: str


@dataclass(frozen=True)
class ResourceAudit:
    file_count: int
    unresolved_count: int
    tracked_resource_count: int
    public_release_blocked: bool
    findings: tuple[ResourceFinding, ...]

    @property
    def valid(self) -> bool:
        return not self.findings

    def as_dict(self) -> dict[str, object]:
        return {
            "file_count": self.file_count,
            "unresolved_count": self.unresolved_count,
            "tracked_resource_count": self.tracked_resource_count,
            "public_release_blocked": self.public_release_blocked,
            "findings": [finding.__dict__ for finding in self.findings],
        }


def audit_clamp_resources(
    root: Path,
    *,
    public_release: bool,
    tracked_files: set[str] | None = None,
    historical_files: set[str] | None = None,
) -> ResourceAudit:
    root = root.resolve()
    manifest_path = root / "config" / "clamp_ards_resource_manifest.json"
    ledger_path = root / "docs" / "CLAMP_ARDS_RESOURCE_LEDGER.csv"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_hashes = manifest.get("files")
    expected_sizes = manifest.get("file_sizes")
    if not isinstance(expected_hashes, dict) or not isinstance(expected_sizes, dict):
        raise ValueError(f"Invalid CLAMP resource manifest: {manifest_path}")
    runtime_required = _validated_runtime_required_files(
        manifest.get("runtime_required_files", []),
        manifest_paths=set(expected_hashes),
        manifest_path=manifest_path,
    )

    rows = _read_ledger(ledger_path)
    findings: list[ResourceFinding] = []
    ledger_paths = set(rows)
    manifest_paths = set(expected_hashes)
    for path in sorted(manifest_paths - ledger_paths):
        findings.append(ResourceFinding("missing_ledger_row", path, "manifest file is omitted"))
    for path in sorted(ledger_paths - manifest_paths):
        findings.append(ResourceFinding("unexpected_ledger_row", path, "file is not in manifest"))

    unresolved_count = 0
    for path in sorted(ledger_paths & manifest_paths):
        row = rows[path]
        status = row["redistribution_status"].strip().casefold()
        if status not in ALLOWED_STATUSES:
            findings.append(ResourceFinding("invalid_status", path, f"unknown status {status!r}"))
        unresolved_count += int(status == "unresolved")
        if status == "cleared":
            non_affirmative = [
                field
                for field in ("license_spdx", "license_evidence")
                if not _is_affirmative(row[field])
            ]
            if non_affirmative:
                findings.append(
                    ResourceFinding(
                        "cleared_without_affirmative_evidence",
                        path,
                        "cleared status requires affirmative values for "
                        + ", ".join(non_affirmative),
                    )
                )
        if row["sha256"] != expected_hashes[path]:
            findings.append(ResourceFinding("ledger_hash", path, "SHA-256 differs from manifest"))
        try:
            ledger_size = int(row["bytes"])
        except ValueError:
            findings.append(ResourceFinding("ledger_size", path, "byte count is not an integer"))
        else:
            if ledger_size != int(expected_sizes[path]):
                findings.append(
                    ResourceFinding("ledger_size", path, "byte count differs from manifest")
                )

        source = root / "clamp_ARDS" / path
        if source.is_file():
            if source.stat().st_size != int(expected_sizes[path]):
                findings.append(ResourceFinding("source_size", path, "local file size differs"))
            if _sha256(source) != expected_hashes[path]:
                findings.append(ResourceFinding("source_hash", path, "local file hash differs"))

    tracked = tracked_files if tracked_files is not None else _tracked_files(root)
    tracked_resources = {
        value.removeprefix("clamp_ARDS/") for value in tracked if value.startswith("clamp_ARDS/")
    }
    if historical_files is not None:
        historical = historical_files
    elif tracked_files is None:
        historical = _historical_files(root)
    else:
        historical = set()
    historical_resources = {
        value.removeprefix("clamp_ARDS/") for value in historical if value.startswith("clamp_ARDS/")
    }
    public_findings: list[ResourceFinding] = []
    for path in sorted(ledger_paths & manifest_paths):
        row = rows[path]
        status = row["redistribution_status"].strip().casefold()
        disposition = row["disposition"].strip().casefold()
        if (
            status == "cleared"
            and _is_publicly_retained(disposition)
            and _has_pending_verification(row["verification_needed"])
        ):
            public_findings.append(
                ResourceFinding(
                    "retained_resource_pending_verification",
                    path,
                    "cleared retained resource still has a pending verification task",
                )
            )
    for path in sorted(runtime_required):
        row = rows.get(path)
        status = "" if row is None else row["redistribution_status"].strip().casefold()
        disposition = "" if row is None else row["disposition"].strip().casefold()
        external_boundary = "local resource boundary" in disposition
        if row is None or (status != "cleared" and not external_boundary):
            public_findings.append(
                ResourceFinding(
                    "unresolved_required_resource",
                    path,
                    "required behavior resource must be cleared for redistribution or kept "
                    "behind the documented external-resource boundary",
                )
            )
    for path in sorted(tracked_resources):
        row = rows.get(path)
        if row is None:
            public_findings.append(
                ResourceFinding("unreviewed_tracked_resource", path, "tracked without ledger row")
            )
            continue
        status = row["redistribution_status"].strip().casefold()
        disposition = row["disposition"].strip().casefold()
        if status != "cleared":
            public_findings.append(
                ResourceFinding(
                    "unresolved_tracked_resource",
                    path,
                    "tracked resource lacks affirmative redistribution evidence",
                )
            )
        elif _requires_public_exclusion(disposition):
            public_findings.append(
                ResourceFinding(
                    "excluded_tracked_resource",
                    path,
                    "ledger disposition requires exclusion from public Git",
                )
            )

    for path in sorted(historical_resources - tracked_resources):
        row = rows.get(path)
        if row is None:
            public_findings.append(
                ResourceFinding(
                    "unreviewed_historical_resource",
                    path,
                    "resource remains in Git history without a ledger row",
                )
            )
            continue
        status = row["redistribution_status"].strip().casefold()
        disposition = row["disposition"].strip().casefold()
        if status != "cleared" or _requires_public_exclusion(disposition):
            public_findings.append(
                ResourceFinding(
                    "historical_resource_exposure",
                    path,
                    "resource remains in public-bound history without retain clearance",
                )
            )

    if public_release and _is_git_repository(root):
        restricted_resources = {
            path: (str(expected_hashes[path]), int(expected_sizes[path]))
            for path in sorted(manifest_paths)
            if (row := rows.get(path)) is None
            or row["redistribution_status"].strip().casefold() != "cleared"
            or _requires_public_exclusion(row["disposition"].strip().casefold())
        }
        public_findings.extend(
            _audit_restricted_resource_blobs(
                root,
                restricted_resources,
                canonical_historical_resources=historical_resources,
            )
        )

    public_release_blocked = bool(findings or public_findings)
    if public_release:
        findings.extend(public_findings)
    return ResourceAudit(
        file_count=len(rows),
        unresolved_count=unresolved_count,
        tracked_resource_count=len(tracked_resources),
        public_release_blocked=public_release_blocked,
        findings=tuple(sorted(findings, key=lambda item: (item.check, item.path))),
    )


def _validated_runtime_required_files(
    raw: object,
    *,
    manifest_paths: set[object],
    manifest_path: Path,
) -> set[str]:
    if not isinstance(raw, list) or not all(isinstance(value, str) for value in raw):
        raise ValueError(f"Invalid runtime_required_files in manifest: {manifest_path}")
    duplicates = sorted(value for value in set(raw) if raw.count(value) > 1)
    if duplicates:
        raise ValueError(f"Duplicate runtime_required_files in manifest: {duplicates}")
    unsafe = sorted(value for value in raw if not _is_safe_relative_path(value))
    if unsafe:
        raise ValueError(f"Unsafe runtime_required_files path in manifest: {unsafe}")
    unknown = sorted(set(raw) - manifest_paths)
    if unknown:
        raise ValueError(f"runtime_required_files are absent from manifest files: {unknown}")
    return set(raw)


def _is_safe_relative_path(value: str) -> bool:
    posix = PurePosixPath(value)
    windows = PureWindowsPath(value)
    return (
        bool(value)
        and value == value.strip()
        and not (
            "\\" in value
            or posix.is_absolute()
            or windows.is_absolute()
            or ".." in posix.parts
            or not posix.parts
        )
    )


def _is_affirmative(value: str) -> bool:
    normalized = " ".join(value.strip().casefold().split())
    return bool(normalized) and not any(
        normalized == prefix or normalized.startswith(f"{prefix} ")
        for prefix in NON_AFFIRMATIVE_PREFIXES
    )


def _requires_public_exclusion(disposition: str) -> bool:
    return disposition.startswith("exclude") or "local resource boundary" in disposition


def _is_publicly_retained(disposition: str) -> bool:
    return disposition == "retain" or disposition.startswith("retain ")


def _has_pending_verification(value: str) -> bool:
    normalized = " ".join(value.strip().casefold().split())
    return any(
        normalized == prefix or normalized.startswith(f"{prefix} ")
        for prefix in PENDING_VERIFICATION_PREFIXES
    )


def _read_ledger(path: Path) -> dict[str, dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "path",
            "sha256",
            "bytes",
            "origin",
            "author_or_rightsholder",
            "license_spdx",
            "license_evidence",
            "runtime_role",
            "redistribution_status",
            "disposition",
            "verification_needed",
        }
        missing = sorted(required - set(reader.fieldnames or []))
        if missing:
            raise ValueError(f"CLAMP resource ledger is missing columns {missing}: {path}")
        rows: dict[str, dict[str, str]] = {}
        for row_number, row in enumerate(reader, start=2):
            resource_path = row["path"]
            if (
                not resource_path
                or Path(resource_path).is_absolute()
                or ".." in Path(resource_path).parts
            ):
                raise ValueError(f"Unsafe CLAMP resource path on ledger row {row_number}")
            if resource_path in rows:
                raise ValueError(f"Duplicate CLAMP resource ledger path: {resource_path}")
            if any(not row[column].strip() for column in required):
                raise ValueError(f"Blank required value on CLAMP resource ledger row {row_number}")
            rows[resource_path] = row
    return rows


def _tracked_files(root: Path) -> set[str]:
    result = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=root,
        check=True,
        capture_output=True,
    )
    return {value.decode("utf-8") for value in result.stdout.split(b"\0") if value}


def _historical_files(root: Path) -> set[str]:
    has_head = subprocess.run(
        ["git", "rev-parse", "--verify", "HEAD"],
        cwd=root,
        check=False,
        capture_output=True,
    )
    if has_head.returncode != 0:
        return set()
    result = subprocess.run(
        ["git", "log", "--all", "--pretty=format:", "--name-only"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return {line.strip() for line in result.stdout.splitlines() if line.strip()}


def _is_git_repository(root: Path) -> bool:
    result = subprocess.run(
        ["git", "rev-parse", "--is-inside-work-tree"],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.returncode == 0 and result.stdout.strip() == "true"


def _audit_restricted_resource_blobs(
    root: Path,
    resources: dict[str, tuple[str, int]],
    *,
    canonical_historical_resources: set[str],
) -> list[ResourceFinding]:
    """Find known restricted resource bytes anywhere in current or reachable Git state."""

    if not resources:
        return []
    resources_by_digest: dict[str, list[str]] = {}
    target_sizes: set[int] = set()
    for path, (digest, size) in resources.items():
        resources_by_digest.setdefault(digest, []).append(path)
        target_sizes.add(size)

    tracked_paths = _tracked_blob_paths(root)
    reachable_oids = _reachable_object_ids(root)
    candidate_oids = _blob_oids_with_sizes(
        root,
        set(tracked_paths) | reachable_oids,
        target_sizes,
    )
    matched_resources: dict[str, tuple[str, ...]] = {}
    for oid in sorted(candidate_oids):
        digest = hashlib.sha256(_git_blob_bytes(root, oid)).hexdigest()
        matches = resources_by_digest.get(digest)
        if matches:
            matched_resources[oid] = tuple(sorted(matches))

    findings: list[ResourceFinding] = []
    for oid, paths in sorted(tracked_paths.items()):
        for resource in matched_resources.get(oid, ()):
            for path in sorted(paths):
                if path == f"clamp_ARDS/{resource}":
                    continue
                findings.append(
                    ResourceFinding(
                        "tracked_restricted_resource_blob",
                        path,
                        f"tracked Git blob matches unresolved or excluded resource {resource}",
                    )
                )
    for oid in sorted(reachable_oids - set(tracked_paths)):
        for resource in matched_resources.get(oid, ()):
            if resource in canonical_historical_resources:
                continue
            findings.append(
                ResourceFinding(
                    "historical_restricted_resource_blob",
                    f"git-blob:{oid}",
                    f"reachable historical Git blob matches unresolved or excluded resource "
                    f"{resource}",
                )
            )
    return findings


def _tracked_blob_paths(root: Path) -> dict[str, set[str]]:
    result = subprocess.run(
        ["git", "ls-files", "--stage", "-z"],
        cwd=root,
        check=True,
        capture_output=True,
    )
    paths: dict[str, set[str]] = {}
    for raw_record in result.stdout.split(b"\0"):
        if not raw_record:
            continue
        metadata, raw_path = raw_record.split(b"\t", maxsplit=1)
        fields = metadata.split()
        if len(fields) != 3:
            raise ValueError("Unexpected git ls-files --stage record")
        oid = fields[1].decode("ascii")
        path = raw_path.decode("utf-8", errors="surrogateescape")
        paths.setdefault(oid, set()).add(path)
    return paths


def _reachable_object_ids(root: Path) -> set[str]:
    result = subprocess.run(
        ["git", "rev-list", "--objects", "--all", "--no-object-names"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return {line.strip() for line in result.stdout.splitlines() if line.strip()}


def _blob_oids_with_sizes(root: Path, oids: set[str], target_sizes: set[int]) -> set[str]:
    if not oids:
        return set()
    result = subprocess.run(
        ["git", "cat-file", "--batch-check=%(objectname) %(objecttype) %(objectsize)"],
        cwd=root,
        input="".join(f"{oid}\n" for oid in sorted(oids)),
        check=True,
        capture_output=True,
        text=True,
    )
    matches: set[str] = set()
    for line in result.stdout.splitlines():
        fields = line.split()
        if len(fields) == 3 and fields[1] == "blob" and int(fields[2]) in target_sizes:
            matches.add(fields[0])
    return matches


def _git_blob_bytes(root: Path, oid: str) -> bytes:
    result = subprocess.run(
        ["git", "cat-file", "blob", oid],
        cwd=root,
        check=True,
        capture_output=True,
    )
    return result.stdout


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
