"""Reproducible E00 corpus, chunk and BM25 artifact build."""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
import sqlite3
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TextIO

from uit_dsc_fixed_rag.bm25 import SqliteBm25Writer
from uit_dsc_fixed_rag.cleaning import ConservativePassageCleaner
from uit_dsc_fixed_rag.corpus import (
    CorpusError,
    SourceDocument,
    audit_context_archive,
    file_sha256,
    inspect_context_archive,
    iter_documents,
    require_expected_identity,
)
from uit_dsc_fixed_rag.legal_chunking import ChunkingConfig, LegalChunker


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class E00Config:
    """Validated configuration required by the E00 build."""

    raw: dict[str, Any]
    dataset_name: str
    archive_sha256: str
    canonical_revision: str
    record_count: int
    chunking: ChunkingConfig
    bm25_tokenizer: str
    insert_batch_size: int
    progress_interval_documents: int

    @property
    def config_sha256(self) -> str:
        """Hash canonical JSON config bytes."""

        payload = json.dumps(
            self.raw,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


def load_e00_config(path: Path) -> E00Config:
    """Load and validate the E00 JSON configuration."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("E00 config root must be an object.")
    required_root = {"schema_version", "experiment_id", "dataset", "cleaning", "chunking", "bm25"}
    if set(payload) != required_root:
        raise ValueError("E00 config has missing or unknown root fields.")
    if payload["experiment_id"] != "E00":
        raise ValueError("E00 config has an incompatible experiment ID.")
    dataset = _require_object(payload, "dataset")
    chunking = _require_object(payload, "chunking")
    bm25 = _require_object(payload, "bm25")
    chunk_config = ChunkingConfig(
        max_whitespace_tokens=_positive_int(chunking, "max_whitespace_tokens"),
        max_characters=_positive_int(chunking, "max_characters"),
        overlap_whitespace_tokens=_non_negative_int(chunking, "overlap_whitespace_tokens"),
        document_header_characters=_non_negative_int(chunking, "document_header_characters"),
    )
    chunk_config.validate()
    tokenizer = bm25.get("tokenizer")
    if tokenizer != "unicode61 remove_diacritics 0":
        raise ValueError("E00 requires the pinned Unicode-preserving FTS5 tokenizer.")
    progress_interval = _positive_int(bm25, "progress_interval_documents")
    return E00Config(
        raw=payload,
        dataset_name=_non_blank_text(dataset, "name"),
        archive_sha256=_sha256_text(dataset, "archive_sha256", prefixed=False),
        canonical_revision=_sha256_text(dataset, "canonical_revision", prefixed=True),
        record_count=_positive_int(dataset, "record_count"),
        chunking=chunk_config,
        bm25_tokenizer=tokenizer,
        insert_batch_size=_positive_int(bm25, "insert_batch_size"),
        progress_interval_documents=progress_interval,
    )


def build_e00(
    source: Path,
    output: Path,
    config: E00Config,
    *,
    force: bool = False,
) -> dict[str, Any]:
    """Build complete E00 artifacts in a temporary directory, then publish."""

    identity = inspect_context_archive(source)
    require_expected_identity(
        identity,
        archive_sha256=config.archive_sha256,
        canonical_revision=config.canonical_revision,
        record_count=config.record_count,
    )
    audit = audit_context_archive(source)
    output_parent = output.resolve().parent
    output_parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and not force:
        raise FileExistsError(f"Artifact directory already exists: {output}")

    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.building-", dir=output_parent))
    backup = output_parent / f".{output.name}.previous"
    if backup.exists():
        shutil.rmtree(temporary)
        raise FileExistsError(f"Previous artifact backup must be resolved first: {backup}")

    try:
        manifest = _build_into(source, temporary, config, audit)
        if output.exists():
            output.rename(backup)
        temporary.rename(output)
        if backup.exists():
            shutil.rmtree(backup)
        return manifest
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        if backup.exists() and not output.exists():
            backup.rename(output)
        raise


def _build_into(
    source: Path,
    directory: Path,
    config: E00Config,
    audit: dict[str, Any],
) -> dict[str, Any]:
    cleaner = ConservativePassageCleaner()
    chunker = LegalChunker(config.chunking)
    documents_path = directory / "documents.jsonl"
    chunks_path = directory / "chunks.jsonl"
    database_path = directory / "bm25.sqlite3"
    writer = SqliteBm25Writer(
        database_path,
        tokenizer=config.bm25_tokenizer,
        batch_size=config.insert_batch_size,
    )
    counts: dict[str, int] = {
        "documents": 0,
        "documents_with_chunks": 0,
        "blank_documents": 0,
        "chunks": 0,
        "article_chunks": 0,
        "clause_chunks": 0,
        "token_fallback_chunks": 0,
        "cleaned_characters": 0,
        "html_transformations": 0,
        "notice_removals": 0,
        "oversize_chunks": 0,
    }
    try:
        with documents_path.open("w", encoding="utf-8", newline="\n") as documents_file, chunks_path.open(
            "w", encoding="utf-8", newline="\n"
        ) as chunks_file:
            for document in iter_documents(source):
                _process_document(
                    document,
                    cleaner,
                    chunker,
                    writer,
                    documents_file,
                    chunks_file,
                    counts,
                    config,
                )
                if counts["documents"] % config.progress_interval_documents == 0:
                    LOGGER.info(
                        "e00_build_progress documents=%d chunks=%d",
                        counts["documents"],
                        counts["chunks"],
                    )
        if counts["documents"] != config.record_count:
            raise CorpusError("Build document count differs from the pinned corpus count.")
        if counts["blank_documents"] != audit["blank_passage_count"]:
            raise CorpusError("Blank-document count changed between audit and build.")
        writer.finalize(
            {
                "schema_version": "1.0",
                "artifact_type": "bm25",
                "dataset_name": config.dataset_name,
                "dataset_revision": config.canonical_revision,
                "config_sha256": config.config_sha256,
                "record_count": counts["chunks"],
                "backend": "sqlite-fts5",
                "tokenizer": config.bm25_tokenizer,
            }
        )
    except Exception:
        writer.abort()
        raise

    audit["cleaning_and_chunking"] = counts
    audit_path = directory / "audit.json"
    _write_json(audit_path, audit)
    files = {
        name: {
            "bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        }
        for name, path in (
            ("audit.json", audit_path),
            ("documents.jsonl", documents_path),
            ("chunks.jsonl", chunks_path),
            ("bm25.sqlite3", database_path),
        )
    }
    manifest = {
        "schema_version": "1.0",
        "artifact_type": "e00-corpus-chunks-bm25",
        "artifact_version": "e00-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_name": config.dataset_name,
        "dataset_revision": config.canonical_revision,
        "source_archive_sha256": config.archive_sha256,
        "processing_config_sha256": config.config_sha256,
        "code_version": "0.2.0",
        "record_counts": counts,
        "backend": "sqlite-fts5",
        "model_name": None,
        "model_revision": None,
        "warnings": [
            "E00 has no retrieval relevance labels; answer-derived coverage is diagnostic only.",
            "Whitespace/character chunk budgets must be revalidated with the selected embedding tokenizer before E02."
        ],
        "files": files,
    }
    _write_json(directory / "manifest.json", manifest)
    return manifest


def _process_document(
    document: SourceDocument,
    cleaner: ConservativePassageCleaner,
    chunker: LegalChunker,
    writer: SqliteBm25Writer,
    documents_file: TextIO,
    chunks_file: TextIO,
    counts: dict[str, int],
    config: E00Config,
) -> None:
    cleaned = cleaner.clean(document.raw_text)
    chunks = chunker.chunk_document(document, cleaned.text)
    if document.raw_text.strip() and not cleaned.text:
        raise CorpusError(f"Cleaner removed all content from document {document.document_id}.")
    if cleaned.text and not chunks:
        raise CorpusError(f"Non-blank document emitted no chunks: {document.document_id}.")

    document_payload = {
        "document_id": document.document_id,
        "source_title": document.source_title,
        "source_url": document.source_url,
        "source_name": document.source_name,
        "raw_text_sha256": hashlib.sha256(document.raw_text.encode("utf-8")).hexdigest(),
        "raw_character_count": len(document.raw_text),
        "cleaned_character_count": len(cleaned.text),
        "cleaned_text": cleaned.text,
    }
    _write_json_line(documents_file, document_payload)
    for chunk in chunks:
        oversize = (
            chunk.whitespace_token_count > config.chunking.max_whitespace_tokens
            or len(chunk.text) > config.chunking.max_characters
        )
        counts["oversize_chunks"] += oversize
        _write_json_line(chunks_file, chunk.to_json())
        writer.add(chunk)
        counts["chunks"] += 1
        counts["article_chunks"] += chunk.split_method == "article"
        counts["clause_chunks"] += chunk.split_method == "clause_group"
        counts["token_fallback_chunks"] += chunk.split_method == "token_fallback"

    counts["documents"] += 1
    counts["documents_with_chunks"] += bool(chunks)
    counts["blank_documents"] += not cleaned.text
    counts["cleaned_characters"] += len(cleaned.text)
    counts["html_transformations"] += cleaned.html_transform_count
    counts["notice_removals"] += cleaned.notice_removal_count


def validate_e00_artifact(directory: Path) -> dict[str, Any]:
    """Verify manifest checksums, SQLite integrity and persisted record counts."""

    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for name, expected in manifest["files"].items():
        path = directory / name
        if not path.is_file():
            raise FileNotFoundError(f"Missing E00 artifact file: {name}")
        if path.stat().st_size != expected["bytes"] or file_sha256(path) != expected["sha256"]:
            raise CorpusError(f"E00 artifact checksum mismatch: {name}")
    connection = sqlite3.connect(directory / "bm25.sqlite3")
    try:
        integrity = connection.execute("PRAGMA integrity_check").fetchone()
        chunk_count = connection.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        fts_count = connection.execute("SELECT COUNT(*) FROM chunks_fts").fetchone()[0]
    finally:
        connection.close()
    expected_count = manifest["record_counts"]["chunks"]
    if integrity is None or integrity[0] != "ok" or chunk_count != expected_count or fts_count != expected_count:
        raise CorpusError("E00 SQLite validation failed.")
    return manifest


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _write_json_line(stream: TextIO, payload: dict[str, Any]) -> None:
    stream.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def _require_object(payload: dict[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"{key} must be an object.")
    return value


def _positive_int(payload: dict[str, Any], key: str) -> int:
    value = payload.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{key} must be a positive integer.")
    return value


def _non_negative_int(payload: dict[str, Any], key: str) -> int:
    value = payload.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{key} must be a non-negative integer.")
    return value


def _non_blank_text(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be non-blank text.")
    return value


def _sha256_text(payload: dict[str, Any], key: str, *, prefixed: bool) -> str:
    value = _non_blank_text(payload, key)
    candidate = value.removeprefix("sha256:") if prefixed else value
    if len(candidate) != 64 or any(character not in "0123456789abcdef" for character in candidate):
        raise ValueError(f"{key} must contain a lowercase SHA-256 digest.")
    if prefixed and not value.startswith("sha256:"):
        raise ValueError(f"{key} must use the sha256: prefix.")
    return value
