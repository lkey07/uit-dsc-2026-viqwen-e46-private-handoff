"""Rebuild canonical chunks and E00 BM25 with an exact embedding tokenizer."""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
import sqlite3
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from uit_dsc_fixed_rag.bm25 import SqliteBm25Writer
from uit_dsc_fixed_rag.corpus import SourceDocument, file_sha256
from uit_dsc_fixed_rag.legal_chunking import (
    ChunkingConfig,
    ExactTokenizerPolicy,
    LegalChunker,
)


LOGGER = logging.getLogger(__name__)


class E00TokenizedError(RuntimeError):
    """Raised when E00 v2 lineage, tokenizer or payload validation fails."""


@dataclass(frozen=True)
class E00TokenizedConfig:
    raw: dict[str, Any]
    source_manifest_sha256: str
    source_documents_sha256: str
    source_document_count: int
    dataset_revision: str
    source_archive_sha256: str
    inventory_path: str
    inventory_sha256: str
    model_key: str
    model_id: str
    model_revision: str
    parameter_count: int
    add_special_tokens: bool
    max_model_tokens: int
    overlap_model_tokens: int
    chunking: ChunkingConfig
    bm25_tokenizer: str
    insert_batch_size: int
    progress_interval_documents: int

    @property
    def config_sha256(self) -> str:
        encoded = json.dumps(
            self.raw, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


def load_e00_tokenized_config(path: Path) -> E00TokenizedConfig:
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected_root = {
        "schema_version", "experiment_id", "source_e00", "tokenizer",
        "chunking", "bm25", "run_contract",
    }
    if not isinstance(payload, dict) or set(payload) != expected_root:
        raise ValueError("E00 tokenized config root is incompatible.")
    if payload.get("schema_version") != "1.0" or payload.get("experiment_id") != "E00-tokenized-v2":
        raise ValueError("E00 tokenized config identity changed.")
    source = _object(payload, "source_e00")
    tokenizer = _object(payload, "tokenizer")
    chunking = _object(payload, "chunking")
    bm25 = _object(payload, "bm25")
    contract = _object(payload, "run_contract")
    if source.get("artifact_version") != "e00-v1":
        raise ValueError("E00 v2 must derive from the audited E00 v1 cleaned documents.")
    if tokenizer.get("model_key") != "embedding_harrier":
        raise ValueError("E00 v2 must use the selected Harrier tokenizer.")
    if chunking.get("version") != "article-clause-exact-harrier-v2":
        raise ValueError("E00 v2 chunking version changed.")
    if bm25.get("backend") != "sqlite-fts5" or bm25.get("tokenizer") != "unicode61 remove_diacritics 0":
        raise ValueError("E00 v2 BM25 control changed.")
    if contract != {
        "atomic_publish": True,
        "allow_external_data": False,
        "allow_synthetic_data": False,
        "allow_model_api": False,
    }:
        raise ValueError("E00 v2 run contract changed.")
    max_tokens = _positive_int(chunking, "max_model_tokens")
    overlap = _non_negative_int(chunking, "overlap_model_tokens")
    if overlap >= max_tokens:
        raise ValueError("Model-token overlap must be smaller than max tokens.")
    whitespace = _positive_int(chunking, "max_whitespace_tokens")
    characters = _positive_int(chunking, "max_characters")
    return E00TokenizedConfig(
        raw=payload,
        source_manifest_sha256=_sha256(source, "manifest_sha256"),
        source_documents_sha256=_sha256(source, "documents_sha256"),
        source_document_count=_positive_int(source, "document_count"),
        dataset_revision=_revision(source, "dataset_revision"),
        source_archive_sha256=_sha256(source, "source_archive_sha256"),
        inventory_path=_text(tokenizer, "inventory_path"),
        inventory_sha256=_sha256(tokenizer, "inventory_sha256"),
        model_key=_text(tokenizer, "model_key"),
        model_id=_text(tokenizer, "model_id"),
        model_revision=_commit(tokenizer, "revision"),
        parameter_count=_positive_int(tokenizer, "parameter_count"),
        add_special_tokens=_boolean(tokenizer, "add_special_tokens"),
        max_model_tokens=max_tokens,
        overlap_model_tokens=overlap,
        chunking=ChunkingConfig(
            max_whitespace_tokens=whitespace,
            max_characters=characters,
            overlap_whitespace_tokens=min(overlap, whitespace - 1),
            document_header_characters=_non_negative_int(chunking, "document_header_characters"),
        ),
        bm25_tokenizer="unicode61 remove_diacritics 0",
        insert_batch_size=_positive_int(bm25, "insert_batch_size"),
        progress_interval_documents=_positive_int(bm25, "progress_interval_documents"),
    )


def build_e00_tokenized(
    *,
    source_e00: Path,
    output: Path,
    config: E00TokenizedConfig,
    tokenizer: Any,
) -> dict[str, Any]:
    """Rechunk exact cleaned E00 documents, rebuild BM25 and atomically publish."""

    _validate_source(source_e00, config)
    if output.exists():
        raise FileExistsError(f"E00 v2 output already exists: {output}")
    output_parent = output.resolve().parent
    output_parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.building-", dir=output_parent))
    try:
        manifest = _build_into(source_e00, temporary, config, tokenizer)
        temporary.rename(output)
        return manifest
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def _build_into(
    source_e00: Path,
    directory: Path,
    config: E00TokenizedConfig,
    tokenizer: Any,
) -> dict[str, Any]:
    policy = ExactTokenizerPolicy(
        tokenizer=tokenizer,
        max_tokens=config.max_model_tokens,
        overlap_tokens=config.overlap_model_tokens,
        add_special_tokens=config.add_special_tokens,
    )
    chunker = LegalChunker(config.chunking, exact_token_policy=policy)
    source_documents = source_e00 / "documents.jsonl"
    documents_path = directory / "documents.jsonl"
    chunks_path = directory / "chunks.jsonl"
    database_path = directory / "bm25.sqlite3"
    shutil.copyfile(source_documents, documents_path)
    if file_sha256(documents_path) != config.source_documents_sha256:
        raise E00TokenizedError("Copied E00 documents checksum mismatch.")

    writer = SqliteBm25Writer(
        database_path,
        tokenizer=config.bm25_tokenizer,
        batch_size=config.insert_batch_size,
    )
    writer_open = True
    counts = {
        "documents": 0,
        "documents_with_chunks": 0,
        "blank_documents": 0,
        "chunks": 0,
        "article_chunks": 0,
        "clause_chunks": 0,
        "model_token_fallback_chunks": 0,
        "oversize_model_token_chunks": 0,
    }
    token_lengths: list[int] = []
    try:
        with source_documents.open("r", encoding="utf-8") as source, chunks_path.open(
            "w", encoding="utf-8", newline="\n"
        ) as chunks_file:
            for line_number, line in enumerate(source, start=1):
                record = _decode_document(line, line_number)
                document = SourceDocument(
                    document_id=record["document_id"],
                    source_title=record["source_title"],
                    source_url=record["source_url"],
                    raw_text="",
                    source_name=record["source_name"],
                )
                chunks = chunker.chunk_document(document, record["cleaned_text"])
                if record["cleaned_text"] and not chunks:
                    raise E00TokenizedError(
                        f"Non-blank source document emitted no chunks: {document.document_id}"
                    )
                for chunk in chunks:
                    if chunk.model_token_count is None:
                        raise E00TokenizedError("E00 v2 chunk has no exact model-token count.")
                    token_lengths.append(chunk.model_token_count)
                    counts["oversize_model_token_chunks"] += (
                        chunk.model_token_count > config.max_model_tokens
                    )
                    chunks_file.write(
                        json.dumps(chunk.to_json(), ensure_ascii=False, sort_keys=True) + "\n"
                    )
                    writer.add(chunk)
                    counts["chunks"] += 1
                    counts["article_chunks"] += chunk.split_method == "article"
                    counts["clause_chunks"] += chunk.split_method == "clause_group"
                    counts["model_token_fallback_chunks"] += (
                        chunk.split_method == "model_token_fallback"
                    )
                counts["documents"] += 1
                counts["documents_with_chunks"] += bool(chunks)
                counts["blank_documents"] += not record["cleaned_text"]
                if counts["documents"] % config.progress_interval_documents == 0:
                    LOGGER.info(
                        "e00_v2_build_progress documents=%d chunks=%d",
                        counts["documents"],
                        counts["chunks"],
                    )
        if counts["documents"] != config.source_document_count:
            raise E00TokenizedError("E00 v2 document count differs from source lineage.")
        if counts["oversize_model_token_chunks"]:
            raise E00TokenizedError("E00 v2 emitted chunks above the exact token budget.")
        writer.finalize(
            {
                "schema_version": "1.0",
                "artifact_type": "bm25",
                "dataset_revision": config.dataset_revision,
                "config_sha256": config.config_sha256,
                "record_count": counts["chunks"],
                "backend": "sqlite-fts5",
                "tokenizer": config.bm25_tokenizer,
            }
        )
        writer_open = False
    except Exception:
        if writer_open:
            writer.abort()
        raise

    ordered = sorted(token_lengths)
    audit = {
        "schema_version": "1.0",
        "artifact_type": "e00-v2-exact-token-audit",
        "source_e00_manifest_sha256": config.source_manifest_sha256,
        "source_documents_sha256": config.source_documents_sha256,
        "tokenizer": {
            "model_id": config.model_id,
            "revision": config.model_revision,
            "add_special_tokens": config.add_special_tokens,
        },
        "max_model_tokens": config.max_model_tokens,
        "overlap_model_tokens": config.overlap_model_tokens,
        "record_count": len(ordered),
        "oversize_count": counts["oversize_model_token_chunks"],
        "lengths": {
            "minimum": ordered[0],
            "median": _percentile(ordered, 0.50),
            "p95": _percentile(ordered, 0.95),
            "p99": _percentile(ordered, 0.99),
            "maximum": ordered[-1],
        },
    }
    audit_path = directory / "audit.json"
    _write_json(audit_path, audit)
    files = {
        name: _file_identity(path)
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
        "artifact_version": "e00-v2",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "code_version": "0.4.0",
        "dataset_name": "uit-dsc-2026-task2-selected-contexts",
        "dataset_revision": config.dataset_revision,
        "source_archive_sha256": config.source_archive_sha256,
        "source_e00_manifest_sha256": config.source_manifest_sha256,
        "source_documents_sha256": config.source_documents_sha256,
        "processing_config_sha256": config.config_sha256,
        "record_counts": counts,
        "backend": "sqlite-fts5",
        "model_name": config.model_id,
        "model_revision": config.model_revision,
        "model_role": "exact_chunk_tokenizer",
        "files": files,
        "warnings": [
            "Cleaned documents are byte-identical to audited E00 v1; only chunking and BM25 were rebuilt.",
            "The tokenizer model is the same approved embedding candidate, not an additional model.",
            "No retrieval relevance labels, synthetic data or external corpus data were used."
        ],
    }
    _write_json(directory / "manifest.json", manifest)
    validate_e00_tokenized_artifact(directory)
    return manifest


def validate_e00_tokenized_artifact(directory: Path) -> dict[str, Any]:
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("artifact_version") != "e00-v2":
        raise E00TokenizedError("E00 v2 manifest identity mismatch.")
    for name, expected in manifest["files"].items():
        path = directory / name
        if not path.is_file() or _file_identity(path) != expected:
            raise E00TokenizedError(f"E00 v2 payload checksum mismatch: {name}")
    connection = sqlite3.connect(directory / "bm25.sqlite3")
    try:
        integrity = connection.execute("PRAGMA integrity_check").fetchone()
        chunks = connection.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        fts = connection.execute("SELECT COUNT(*) FROM chunks_fts").fetchone()[0]
    finally:
        connection.close()
    expected_count = manifest["record_counts"]["chunks"]
    if integrity is None or integrity[0] != "ok" or chunks != expected_count or fts != expected_count:
        raise E00TokenizedError("E00 v2 SQLite integrity/count validation failed.")
    if manifest["record_counts"]["oversize_model_token_chunks"] != 0:
        raise E00TokenizedError("E00 v2 manifest contains oversize chunks.")
    return manifest


def _validate_source(directory: Path, config: E00TokenizedConfig) -> None:
    manifest_path = directory / "manifest.json"
    documents_path = directory / "documents.jsonl"
    if not manifest_path.is_file() or not documents_path.is_file():
        raise E00TokenizedError("Source E00 manifest or documents payload is missing.")
    if file_sha256(manifest_path) != config.source_manifest_sha256:
        raise E00TokenizedError("Source E00 manifest checksum mismatch.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("artifact_version") != "e00-v1"
        or manifest.get("dataset_revision") != config.dataset_revision
        or manifest.get("record_counts", {}).get("documents") != config.source_document_count
        or manifest.get("files", {}).get("documents.jsonl", {}).get("sha256")
        != config.source_documents_sha256
    ):
        raise E00TokenizedError("Source E00 lineage is incompatible with v2.")
    if file_sha256(documents_path) != config.source_documents_sha256:
        raise E00TokenizedError("Source E00 documents checksum mismatch.")


def _decode_document(line: str, line_number: int) -> dict[str, Any]:
    try:
        record = json.loads(line)
    except json.JSONDecodeError as exc:
        raise E00TokenizedError(f"Invalid documents JSONL line {line_number}.") from exc
    required = {
        "document_id", "source_title", "source_url", "source_name",
        "raw_text_sha256", "raw_character_count", "cleaned_character_count", "cleaned_text",
    }
    if not isinstance(record, dict) or set(record) != required:
        raise E00TokenizedError(f"Invalid E00 document schema at line {line_number}.")
    if (
        not isinstance(record["document_id"], str)
        or not isinstance(record["source_url"], str)
        or not isinstance(record["source_name"], str)
        or not isinstance(record["cleaned_text"], str)
        or (record["source_title"] is not None and not isinstance(record["source_title"], str))
    ):
        raise E00TokenizedError(f"Invalid E00 document types at line {line_number}.")
    return record


def _file_identity(path: Path) -> dict[str, Any]:
    return {"bytes": path.stat().st_size, "sha256": file_sha256(path)}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _percentile(values: list[int], ratio: float) -> int:
    if not values:
        raise E00TokenizedError("Cannot summarize an empty E00 v2 chunk set.")
    return values[round((len(values) - 1) * ratio)]


def _object(payload: dict[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"{key} must be an object.")
    return value


def _text(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be non-blank text.")
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


def _boolean(payload: dict[str, Any], key: str) -> bool:
    value = payload.get(key)
    if not isinstance(value, bool):
        raise ValueError(f"{key} must be boolean.")
    return value


def _sha256(payload: dict[str, Any], key: str) -> str:
    value = _text(payload, key)
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{key} must be a lowercase SHA-256 digest.")
    return value


def _commit(payload: dict[str, Any], key: str) -> str:
    value = _text(payload, key)
    if len(value) != 40 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{key} must be an immutable commit hash.")
    return value


def _revision(payload: dict[str, Any], key: str) -> str:
    value = _text(payload, key)
    if not value.startswith("sha256:") or len(value) != 71:
        raise ValueError(f"{key} must be a sha256: revision.")
    return value
