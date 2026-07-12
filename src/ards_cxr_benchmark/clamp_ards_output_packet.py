from __future__ import annotations

import hashlib
import json
import os
import struct
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any
from uuid import uuid4

from tqdm import tqdm

from .clamp_ards_outputs import load_input_manifest, validate_distinct_clamp_paths
from .config import ensure_parent_dir


def prepare_clamp_txt_output_packet(
    *,
    source_archive: Path,
    output_archive: Path,
    input_manifest_path: Path,
    summary_output: Path,
    overwrite: bool = False,
    show_progress: bool = False,
) -> dict[str, Any]:
    """Copy only CLAMP tabular TXT outputs into a validated compact ZIP packet."""

    source_archive = source_archive.resolve()
    output_archive = output_archive.resolve()
    input_manifest_path = input_manifest_path.resolve()
    summary_output = summary_output.resolve()
    validate_distinct_clamp_paths(
        source_archive=source_archive,
        output_archive=output_archive,
        input_manifest_path=input_manifest_path,
        summary_output=summary_output,
    )
    if not source_archive.is_file() or not zipfile.is_zipfile(source_archive):
        raise ValueError(f"Source CLAMP output is not a readable ZIP archive: {source_archive}")
    if output_archive.exists() and not overwrite:
        raise FileExistsError(f"TXT-only output archive already exists: {output_archive}")

    manifest = load_input_manifest(input_manifest_path)
    manifest_ids = manifest["clamp_doc_id"].tolist()
    manifest_id_set = set(manifest_ids)

    with zipfile.ZipFile(source_archive) as source_zip:
        txt_members, xmi_count, other_count = _discover_source_members(source_zip)
        packet_ids = set(txt_members)
        missing = sorted(manifest_id_set - packet_ids)
        unexpected = sorted(packet_ids - manifest_id_set)
        if missing or unexpected:
            raise ValueError(
                "CLAMP TXT members do not exactly match input_manifest.csv: "
                f"missing={len(missing)}, unexpected={len(unexpected)}"
            )

        ensure_parent_dir(output_archive)
        temporary_archive = _temporary_path(output_archive)
        source_payload_hash = hashlib.sha256()
        payload_bytes = 0
        try:
            with zipfile.ZipFile(
                temporary_archive,
                mode="w",
                compression=zipfile.ZIP_DEFLATED,
                compresslevel=9,
            ) as packet_zip:
                for doc_id in tqdm(
                    sorted(manifest_ids),
                    desc="Building CLAMP TXT packet",
                    unit="file",
                    disable=not show_progress,
                ):
                    payload = source_zip.read(txt_members[doc_id])
                    _update_payload_hash(source_payload_hash, doc_id, payload)
                    payload_bytes += len(payload)
                    info = zipfile.ZipInfo(
                        filename=f"Output/{doc_id}.txt",
                        date_time=(1980, 1, 1, 0, 0, 0),
                    )
                    info.compress_type = zipfile.ZIP_DEFLATED
                    info.external_attr = 0o600 << 16
                    packet_zip.writestr(info, payload, compresslevel=9)

            packet_payload_hash, packet_member_count = _packet_payload_hash(temporary_archive)
            if packet_member_count != len(manifest_ids):
                raise RuntimeError(
                    "TXT-only packet validation failed: "
                    f"expected {len(manifest_ids)} members, found {packet_member_count}"
                )
            if packet_payload_hash != source_payload_hash.hexdigest():
                raise RuntimeError("TXT-only packet validation failed: payload checksum mismatch")

            source_sha256 = _file_sha256(source_archive)
            packet_sha256 = _file_sha256(temporary_archive)
            os.replace(temporary_archive, output_archive)
        finally:
            temporary_archive.unlink(missing_ok=True)

    summary = {
        "source_archive_name": source_archive.name,
        "source_archive_sha256": source_sha256,
        "source_archive_bytes": source_archive.stat().st_size,
        "source_txt_members": len(txt_members),
        "source_xmi_members_excluded": xmi_count,
        "source_other_members_excluded": other_count,
        "manifest_rows": len(manifest_ids),
        "missing_manifest_ids": 0,
        "unexpected_txt_ids": 0,
        "output_archive_name": output_archive.name,
        "output_archive_sha256": packet_sha256,
        "output_archive_bytes": output_archive.stat().st_size,
        "output_txt_members": len(manifest_ids),
        "txt_payload_bytes": payload_bytes,
        "txt_payload_sha256": source_payload_hash.hexdigest(),
    }
    _write_json_atomic(summary, summary_output)
    return summary


def _discover_source_members(
    source_zip: zipfile.ZipFile,
) -> tuple[dict[str, str], int, int]:
    txt_members: dict[str, str] = {}
    xmi_count = 0
    other_count = 0
    for info in source_zip.infolist():
        if info.is_dir():
            continue
        _validate_member_name(info.filename)
        lower = info.filename.lower()
        if lower.endswith(".txt"):
            doc_id = _member_doc_id(info.filename)
            if doc_id in txt_members:
                raise ValueError(f"Duplicate CLAMP TXT output for clamp_doc_id={doc_id}")
            txt_members[doc_id] = info.filename
        elif lower.endswith((".xmi", ".xmi.gz")):
            xmi_count += 1
        else:
            other_count += 1
    if not txt_members:
        raise ValueError("Source CLAMP output archive contains no TXT members")
    return txt_members, xmi_count, other_count


def _packet_payload_hash(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    with zipfile.ZipFile(path) as packet_zip:
        members, xmi_count, other_count = _discover_source_members(packet_zip)
        if xmi_count or other_count:
            raise RuntimeError("TXT-only packet unexpectedly contains non-TXT members")
        for doc_id in sorted(members):
            _update_payload_hash(digest, doc_id, packet_zip.read(members[doc_id]))
    return digest.hexdigest(), len(members)


def _update_payload_hash(digest: Any, doc_id: str, payload: bytes) -> None:
    encoded_id = doc_id.encode("utf-8")
    digest.update(struct.pack(">I", len(encoded_id)))
    digest.update(encoded_id)
    digest.update(struct.pack(">Q", len(payload)))
    digest.update(payload)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _member_doc_id(member_name: str) -> str:
    name = PurePosixPath(member_name).name
    doc_id = name[:-4] if name.lower().endswith(".txt") else name
    if doc_id.lower().endswith(".txt"):
        doc_id = doc_id[:-4]
    doc_id = doc_id.strip()
    if not doc_id:
        raise ValueError(f"CLAMP output member has a blank document ID: {member_name}")
    return doc_id


def _validate_member_name(member_name: str) -> None:
    path = PurePosixPath(member_name)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Unsafe path in CLAMP output archive: {member_name}")


def _temporary_path(destination: Path) -> Path:
    ensure_parent_dir(destination)
    return destination.with_name(f".{destination.name}.{uuid4().hex}.partial")


def _write_json_atomic(payload: dict[str, Any], destination: Path) -> None:
    temporary = _temporary_path(destination)
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
