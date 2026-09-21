"""Strict adapter and audit boundary for the official context archive."""

from __future__ import annotations

import hashlib
import json
import re
import stat
import unicodedata
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import urlparse
from zipfile import BadZipFile, ZipFile, ZipInfo


class CorpusError(ValueError):
    """Raised when the official corpus violates its fail-closed contract."""


_CONTEXT_NAME = re.compile(r"^context_(?P<id>[0-9]+)\.json$")
_ARTICLE_LINE = re.compile(r"(?im)^\s*Điều\s+[0-9]+")
_KNOWN_HTML = re.compile(
    r"<\s*/?\s*(?:br|p|div|span|table|thead|tbody|tr|td|th|strong|em|b|i|img)\b",
    re.IGNORECASE,
)
_REQUIRED_FIELDS = {"id", "link", "passage"}
_ALLOWED_FIELDS = _REQUIRED_FIELDS | {"name"}


@dataclass(frozen=True)
class SourceDocument:
    """Competition-neutral document emitted by the raw adapter."""

    document_id: str
    source_title: str | None
    source_url: str
    raw_text: str
    source_name: str


@dataclass(frozen=True)
class ContextSourceIdentity:
    """Canonical identity of an ordered collection of context JSON bytes."""

    archive_sha256: str
    canonical_revision: str
    record_count: int


def file_sha256(path: Path) -> str:
    """Return a lowercase SHA-256 digest without loading the file into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inspect_context_archive(path: Path) -> ContextSourceIdentity:
    """Hash exact ordered member names and bytes using the canonical contract."""

    digest = hashlib.sha256()
    count = 0
    for name, payload in iter_context_payloads(path):
        encoded_name = name.encode("utf-8")
        digest.update(len(encoded_name).to_bytes(8, "big"))
        digest.update(encoded_name)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
        count += 1
    if count == 0:
        raise CorpusError("Context archive contains no context JSON records.")
    return ContextSourceIdentity(
        archive_sha256=file_sha256(path),
        canonical_revision=f"sha256:{digest.hexdigest()}",
        record_count=count,
    )


def iter_context_payloads(path: Path) -> Iterator[tuple[str, bytes]]:
    """Yield safe context member basenames and exact bytes in lexical order."""

    if not path.is_file():
        raise CorpusError("Context archive does not exist.")
    try:
        with ZipFile(path) as archive:
            for member in _context_members(archive):
                yield PurePosixPath(member.filename).name, archive.read(member)
    except CorpusError:
        raise
    except (BadZipFile, OSError, RuntimeError) as exc:
        raise CorpusError("Context ZIP cannot be read.") from exc


def iter_documents(path: Path) -> Iterator[SourceDocument]:
    """Map raw organizer records into stable, competition-neutral documents."""

    seen_ids: set[str] = set()
    for source_name, payload in iter_context_payloads(path):
        raw = _decode_unique_json(payload, source_name)
        document = _adapt_context(raw, source_name)
        if document.document_id in seen_ids:
            raise CorpusError(f"Duplicate canonical context ID: {document.document_id}")
        seen_ids.add(document.document_id)
        yield document


def audit_context_archive(path: Path) -> dict[str, Any]:
    """Audit raw corpus properties without cleaning, dropping or inventing data."""

    identity = inspect_context_archive(path)
    lengths: list[int] = []
    passage_ids: dict[str, list[str]] = defaultdict(list)
    missing_name = 0
    blank_passage = 0
    crlf_passage = 0
    non_nfc_passage = 0
    known_html_passage = 0
    article_line_passage = 0
    total_characters = 0
    longest: list[tuple[int, str]] = []

    for document in iter_documents(path):
        text = document.raw_text
        length = len(text)
        lengths.append(length)
        total_characters += length
        passage_ids[hashlib.sha256(text.encode("utf-8")).hexdigest()].append(
            document.document_id
        )
        missing_name += document.source_title is None
        blank_passage += not text.strip()
        crlf_passage += "\r\n" in text
        non_nfc_passage += not unicodedata.is_normalized("NFC", text)
        known_html_passage += bool(_KNOWN_HTML.search(text))
        article_line_passage += bool(_ARTICLE_LINE.search(text))
        longest.append((length, document.document_id))

    duplicate_groups = [ids for ids in passage_ids.values() if len(ids) > 1]
    longest.sort(reverse=True)
    ordered_lengths = sorted(lengths)
    return {
        "schema_version": "1.0",
        "archive_sha256": identity.archive_sha256,
        "canonical_revision": identity.canonical_revision,
        "record_count": identity.record_count,
        "missing_name_count": missing_name,
        "blank_passage_count": blank_passage,
        "crlf_passage_count": crlf_passage,
        "non_nfc_passage_count": non_nfc_passage,
        "known_html_passage_count": known_html_passage,
        "article_line_passage_count": article_line_passage,
        "raw_character_count": total_characters,
        "passage_length": {
            "minimum": ordered_lengths[0],
            "median": _percentile(ordered_lengths, 0.5),
            "p95": _percentile(ordered_lengths, 0.95),
            "p99": _percentile(ordered_lengths, 0.99),
            "maximum": ordered_lengths[-1],
        },
        "longest_passages": [
            {"document_id": document_id, "characters": length}
            for length, document_id in longest[:10]
        ],
        "duplicate_passage_group_count": len(duplicate_groups),
        "duplicate_passage_extra_record_count": sum(
            len(group) - 1 for group in duplicate_groups
        ),
        "duplicate_passage_groups": duplicate_groups,
        "policy": {
            "blank_documents_preserved_in_audit": True,
            "blank_documents_emit_chunks": False,
            "duplicate_documents_dropped": False,
            "source_urls_crawled": False,
        },
    }


def require_expected_identity(
    actual: ContextSourceIdentity,
    *,
    archive_sha256: str,
    canonical_revision: str,
    record_count: int,
) -> None:
    """Fail before publication when official corpus identity has changed."""

    expected = (archive_sha256, canonical_revision, record_count)
    observed = (actual.archive_sha256, actual.canonical_revision, actual.record_count)
    if observed != expected:
        raise CorpusError(
            "Official corpus identity mismatch; run a new audit before building. "
            f"Expected {expected}, observed {observed}."
        )


def _context_members(archive: ZipFile) -> list[ZipInfo]:
    members: list[ZipInfo] = []
    seen_names: set[str] = set()
    for member in archive.infolist():
        path = PurePosixPath(member.filename)
        if path.is_absolute() or ".." in path.parts:
            raise CorpusError("Context ZIP contains an unsafe member path.")
        if member.flag_bits & 0x1:
            raise CorpusError("Context ZIP contains an encrypted member.")
        unix_mode = member.external_attr >> 16
        if unix_mode and stat.S_ISLNK(unix_mode):
            raise CorpusError("Context ZIP contains a symbolic-link member.")
        if member.is_dir():
            continue
        basename = path.name
        if not _CONTEXT_NAME.fullmatch(basename):
            raise CorpusError(f"Unexpected non-context ZIP member: {basename}")
        if basename in seen_names:
            raise CorpusError(f"Duplicate ZIP member basename: {basename}")
        seen_names.add(basename)
        members.append(member)
    return sorted(members, key=lambda item: PurePosixPath(item.filename).name)


def _decode_unique_json(payload: bytes, source_name: str) -> dict[str, Any]:
    try:
        decoded = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
        )
    except CorpusError:
        raise
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise CorpusError(f"Invalid UTF-8 JSON in {source_name}.") from exc
    if not isinstance(decoded, dict):
        raise CorpusError(f"Context root must be an object in {source_name}.")
    return decoded


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise CorpusError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _adapt_context(raw: dict[str, Any], source_name: str) -> SourceDocument:
    keys = set(raw)
    missing = _REQUIRED_FIELDS - keys
    unknown = keys - _ALLOWED_FIELDS
    if missing:
        raise CorpusError(f"{source_name} is missing fields: {sorted(missing)}")
    if unknown:
        raise CorpusError(f"{source_name} has unknown fields: {sorted(unknown)}")

    document_id = _canonical_context_id(raw["id"], source_name)
    filename_match = _CONTEXT_NAME.fullmatch(source_name)
    if filename_match is None or filename_match.group("id") != document_id:
        raise CorpusError(f"Filename and context ID mismatch in {source_name}.")

    title = raw.get("name")
    if title is not None and not isinstance(title, str):
        raise CorpusError(f"Context name must be text in {source_name}.")
    source_url = raw["link"]
    if not isinstance(source_url, str) or not source_url.strip():
        raise CorpusError(f"Context link must be non-blank text in {source_name}.")
    parsed_url = urlparse(source_url)
    if parsed_url.scheme != "https" or not parsed_url.netloc:
        raise CorpusError(f"Context link must be an absolute HTTPS URL in {source_name}.")
    passage = raw["passage"]
    if not isinstance(passage, str):
        raise CorpusError(f"Context passage must be text in {source_name}.")
    return SourceDocument(
        document_id=document_id,
        source_title=title,
        source_url=source_url,
        raw_text=passage,
        source_name=source_name,
    )


def _canonical_context_id(value: object, source_name: str) -> str:
    if isinstance(value, bool):
        raise CorpusError(f"Invalid boolean context ID in {source_name}.")
    if isinstance(value, int) and value >= 0:
        return str(value)
    if isinstance(value, str) and value.strip():
        return value.strip()
    raise CorpusError(f"Invalid context ID in {source_name}.")


def _percentile(values: list[int], ratio: float) -> int:
    if not values:
        raise CorpusError("Cannot summarize an empty corpus.")
    index = round((len(values) - 1) * ratio)
    return values[index]
