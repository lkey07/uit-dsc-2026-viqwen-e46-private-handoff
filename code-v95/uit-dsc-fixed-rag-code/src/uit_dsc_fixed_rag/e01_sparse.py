"""Contentless secondary sparse indexes and weighted RRF retrieval for E01."""

from __future__ import annotations

import hashlib
import json
import logging
import re
import shutil
import sqlite3
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from uit_dsc_fixed_rag.corpus import file_sha256
from uit_dsc_fixed_rag.sparse_normalization import (
    fold_vietnamese_accents,
    folded_query_tokens,
    select_trigram_query_terms,
)


LOGGER = logging.getLogger(__name__)
CODE_VERSION = "0.3.0"
_BRANCHES = ("word", "accent_word", "character")


class E01Error(RuntimeError):
    """Raised when E01 input, artifact or retrieval identity is invalid."""


@dataclass(frozen=True)
class E01Config:
    """Validated E01 index, retrieval and diagnostic policy."""

    raw: dict[str, Any]
    source_manifest_sha256: str
    source_chunks_sha256: str
    source_chunk_count: int
    dataset_revision: str
    accent_word_tokenizer: str
    character_tokenizer: str
    trigram_text_max_characters: int
    trigram_query_max_terms: int
    trigram_query_min_term_characters: int
    insert_batch_size: int
    progress_interval_chunks: int
    rrf_constant: int
    candidate_k_per_branch: int
    default_top_k: int
    word_query_policy: str
    branch_weights: dict[str, float]
    dev_sha256: str
    sample_seed: str
    sample_size: int
    diagnostic_top_k: int
    checkpoint_every_questions: int

    @property
    def config_sha256(self) -> str:
        """Return the canonical JSON identity of the complete E01 config."""

        payload = json.dumps(
            self.raw,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    @property
    def index_config_sha256(self) -> str:
        """Hash only fields that determine persisted secondary-index bytes."""

        payload = {
            "schema_version": self.raw["schema_version"],
            "source_e00": self.raw["source_e00"],
            "normalization": self.raw["normalization"],
            "secondary_index": self.raw["secondary_index"],
        }
        encoded = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class SparseBranchHit:
    """Rank-only hit from one E01 secondary sparse branch."""

    chunk_id: str
    rank: int
    score: float


@dataclass(frozen=True)
class FusedSparseHit:
    """Traceable weighted-RRF result with exact E00 legal payload."""

    chunk_id: str
    document_id: str
    article_number: str | None
    text: str
    metadata: dict[str, Any]
    rank: int
    fused_score: float
    branch_ranks: dict[str, int]
    branch_contributions: dict[str, float]


@dataclass(frozen=True)
class SparseRetrievalResponse:
    """Bounded E01 results and branch-level observability."""

    hits: tuple[FusedSparseHit, ...]
    branch_candidate_counts: dict[str, int]
    branch_latency_ms: dict[str, float]
    latency_ms: float


def load_e01_config(path: Path) -> E01Config:
    """Load E01 config with strict identities and fusion constraints."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("E01 config root must be an object.")
    expected_root = {
        "schema_version",
        "experiment_id",
        "source_e00",
        "normalization",
        "secondary_index",
        "fusion",
        "diagnostic",
    }
    if set(payload) != expected_root or payload.get("experiment_id") != "E01":
        raise ValueError("E01 config root is incompatible.")
    if payload.get("schema_version") != "1.0":
        raise ValueError("E01 config schema_version must be 1.0.")
    source = _object(payload, "source_e00")
    normalization = _object(payload, "normalization")
    index = _object(payload, "secondary_index")
    fusion = _object(payload, "fusion")
    diagnostic = _object(payload, "diagnostic")
    _require_keys(
        source,
        {
            "artifact_version",
            "manifest_sha256",
            "chunks_sha256",
            "chunk_count",
            "dataset_revision",
        },
        "source_e00",
    )
    _require_keys(
        normalization,
        {
            "version",
            "unicode_form_before_fold",
            "fold_form",
            "map_stroked_d",
            "casefold",
            "collapse_whitespace",
        },
        "normalization",
    )
    _require_keys(
        index,
        {
            "backend",
            "accent_word_tokenizer",
            "character_tokenizer",
            "trigram_input_field",
            "trigram_text_max_characters",
            "trigram_query_max_terms",
            "trigram_query_min_term_characters",
            "insert_batch_size",
            "progress_interval_chunks",
        },
        "secondary_index",
    )
    _require_keys(
        fusion,
        {
            "rrf_constant",
            "candidate_k_per_branch",
            "default_top_k",
            "word_query_policy",
            "branch_weights",
        },
        "fusion",
    )
    _require_keys(
        diagnostic,
        {"dev_sha256", "sample_seed", "sample_size", "top_k", "checkpoint_every_questions"},
        "diagnostic",
    )
    if source.get("artifact_version") != "e00-v1":
        raise ValueError("E01 must consume the e00-v1 artifact.")
    expected_normalization = {
        "version": "vietnamese-accent-fold-v1",
        "unicode_form_before_fold": "NFC",
        "fold_form": "NFD-remove-Mn",
        "map_stroked_d": True,
        "casefold": True,
        "collapse_whitespace": True,
    }
    if normalization != expected_normalization:
        raise ValueError("E01 normalization contract changed.")
    if index.get("backend") != "sqlite-fts5-contentless":
        raise ValueError("E01 secondary backend identity changed.")
    if index.get("trigram_input_field") != "text":
        raise ValueError("E01 character branch must index exact chunk text.")
    if diagnostic.get("checkpoint_every_questions") != 1:
        raise ValueError("E01 diagnostics must checkpoint every question.")
    if fusion.get("word_query_policy") != "e00-exact-or":
        raise ValueError("E01 word query must preserve the exact E00 policy.")
    weights = _object(fusion, "branch_weights")
    if set(weights) != set(_BRANCHES):
        raise ValueError("E01 branch_weights must define exactly three branches.")
    parsed_weights: dict[str, float] = {}
    for branch, value in weights.items():
        if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"Invalid positive RRF weight for {branch}.")
        parsed_weights[branch] = float(value)
    accent_tokenizer = _text(index, "accent_word_tokenizer")
    character_tokenizer = _text(index, "character_tokenizer")
    if accent_tokenizer != "unicode61 remove_diacritics 0":
        raise ValueError("Accent-word tokenizer identity changed.")
    if character_tokenizer != "trigram case_sensitive 0":
        raise ValueError("Character tokenizer identity changed.")
    return E01Config(
        raw=payload,
        source_manifest_sha256=_sha256(source, "manifest_sha256"),
        source_chunks_sha256=_sha256(source, "chunks_sha256"),
        source_chunk_count=_positive_int(source, "chunk_count"),
        dataset_revision=_revision(source, "dataset_revision"),
        accent_word_tokenizer=accent_tokenizer,
        character_tokenizer=character_tokenizer,
        trigram_text_max_characters=_positive_int(index, "trigram_text_max_characters"),
        trigram_query_max_terms=_positive_int(index, "trigram_query_max_terms"),
        trigram_query_min_term_characters=_positive_int(index, "trigram_query_min_term_characters"),
        insert_batch_size=_positive_int(index, "insert_batch_size"),
        progress_interval_chunks=_positive_int(index, "progress_interval_chunks"),
        rrf_constant=_positive_int(fusion, "rrf_constant"),
        candidate_k_per_branch=_positive_int(fusion, "candidate_k_per_branch"),
        default_top_k=_positive_int(fusion, "default_top_k"),
        word_query_policy="e00-exact-or",
        branch_weights=parsed_weights,
        dev_sha256=_sha256(diagnostic, "dev_sha256"),
        sample_seed=_text(diagnostic, "sample_seed"),
        sample_size=_positive_int(diagnostic, "sample_size"),
        diagnostic_top_k=_positive_int(diagnostic, "top_k"),
        checkpoint_every_questions=_positive_int(diagnostic, "checkpoint_every_questions"),
    )


def build_e01(
    e00_directory: Path,
    output: Path,
    config: E01Config,
    *,
    force: bool = False,
) -> dict[str, Any]:
    """Build E01 secondary indexes from exact E00 chunks without rechunking."""

    source_manifest_path = e00_directory / "manifest.json"
    source_chunks_path = e00_directory / "chunks.jsonl"
    _validate_e00_lineage(source_manifest_path, source_chunks_path, config)
    if output.exists() and not force:
        raise FileExistsError(f"E01 artifact already exists: {output}")
    output_parent = output.resolve().parent
    output_parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.building-", dir=output_parent))
    backup = output_parent / f".{output.name}.previous"
    if backup.exists():
        shutil.rmtree(temporary)
        raise FileExistsError(f"Previous E01 backup must be resolved first: {backup}")
    try:
        manifest = _build_secondary(source_chunks_path, temporary, config)
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


def _build_secondary(
    chunks_path: Path,
    directory: Path,
    config: E01Config,
) -> dict[str, Any]:
    database_path = directory / "secondary_sparse.sqlite3"
    connection = sqlite3.connect(database_path)
    _initialize_secondary_database(connection, config)
    digest = hashlib.sha256()
    pending: list[tuple[str, str, str]] = []
    seen_ids: set[str] = set()
    count = 0
    try:
        with chunks_path.open("rb") as stream:
            for line_number, raw_line in enumerate(stream, start=1):
                digest.update(raw_line)
                record = _decode_chunk_line(raw_line, line_number)
                chunk_id = record["chunk_id"]
                if chunk_id in seen_ids:
                    raise E01Error(f"Duplicate E00 chunk ID at line {line_number}.")
                seen_ids.add(chunk_id)
                folded_search_text = fold_vietnamese_accents(record["search_text"])
                folded_chunk_text = fold_vietnamese_accents(record["text"])
                pending.append(
                    (
                        chunk_id,
                        folded_search_text,
                        folded_chunk_text[: config.trigram_text_max_characters],
                    )
                )
                count += 1
                if len(pending) >= config.insert_batch_size:
                    _insert_secondary_batch(connection, pending)
                    pending.clear()
                if count % config.progress_interval_chunks == 0:
                    LOGGER.info("e01_build_progress chunks=%d", count)
        if pending:
            _insert_secondary_batch(connection, pending)
        if digest.hexdigest() != config.source_chunks_sha256:
            raise E01Error("E00 chunks checksum changed during E01 build.")
        if count != config.source_chunk_count:
            raise E01Error("E00 chunk count changed during E01 build.")
        metadata = {
            "schema_version": "1.0",
            "artifact_type": "e01-secondary-sparse",
            "artifact_version": "e01-v1",
            "code_version": CODE_VERSION,
            "source_e00_manifest_sha256": config.source_manifest_sha256,
            "source_chunks_sha256": config.source_chunks_sha256,
            "dataset_revision": config.dataset_revision,
            "index_config_sha256": config.index_config_sha256,
            "record_count": count,
            "normalization": "vietnamese-accent-fold-v1",
        }
        connection.executemany(
            "INSERT INTO metadata(key, value) VALUES (?, ?)",
            [
                (key, json.dumps(value, ensure_ascii=False, sort_keys=True))
                for key, value in sorted(metadata.items())
            ],
        )
        connection.execute("INSERT INTO accent_fts(accent_fts) VALUES ('optimize')")
        connection.execute("INSERT INTO character_fts(character_fts) VALUES ('optimize')")
        connection.commit()
        _require_database_integrity(connection, count)
    finally:
        connection.close()

    manifest = {
        "schema_version": "1.0",
        "artifact_type": "e01-secondary-sparse",
        "artifact_version": "e01-v1",
        "code_version": CODE_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_revision": config.dataset_revision,
        "source_e00_manifest_sha256": config.source_manifest_sha256,
        "source_chunks_sha256": config.source_chunks_sha256,
        "processing_config_sha256": config.index_config_sha256,
        "record_count": count,
        "backend": "sqlite-fts5-contentless",
        "branches": ["accent_word", "character"],
        "warnings": [
            "E01 reuses E00 chunks and does not create retrieval labels.",
            "E01 diagnostic metrics are answer-derived and are not retrieval recall or precision."
        ],
        "files": {
            "secondary_sparse.sqlite3": {
                "bytes": database_path.stat().st_size,
                "sha256": file_sha256(database_path),
            }
        },
    }
    _write_json(directory / "manifest.json", manifest)
    return manifest


def validate_e01_artifact(directory: Path, e00_directory: Path) -> dict[str, Any]:
    """Deep-validate E01 payload, row counts and exact E00 manifest lineage."""

    manifest_path = directory / "manifest.json"
    database_path = directory / "secondary_sparse.sqlite3"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if file_sha256(e00_directory / "manifest.json") != manifest["source_e00_manifest_sha256"]:
        raise E01Error("E01 source E00 manifest identity mismatch.")
    expected = manifest["files"]["secondary_sparse.sqlite3"]
    if database_path.stat().st_size != expected["bytes"] or file_sha256(database_path) != expected["sha256"]:
        raise E01Error("E01 database checksum mismatch.")
    connection = sqlite3.connect(database_path)
    try:
        _require_database_integrity(connection, manifest["record_count"])
    finally:
        connection.close()
    return manifest


class HybridSparseRetriever:
    """Fuse E00 word BM25 and two E01 branches using weighted RRF."""

    def __init__(
        self,
        e00_directory: Path,
        e01_directory: Path,
        config: E01Config,
    ) -> None:
        self._config = config
        e00_manifest_path = e00_directory / "manifest.json"
        e01_manifest_path = e01_directory / "manifest.json"
        if not e00_manifest_path.is_file() or not e01_manifest_path.is_file():
            raise E01Error("Required E00/E01 manifest is missing.")
        if file_sha256(e00_manifest_path) != config.source_manifest_sha256:
            raise E01Error("Runtime E00 manifest identity mismatch.")
        e01_manifest = json.loads(e01_manifest_path.read_text(encoding="utf-8"))
        if (
            e01_manifest.get("code_version") != CODE_VERSION
            or e01_manifest.get("processing_config_sha256")
            != config.index_config_sha256
            or e01_manifest.get("source_e00_manifest_sha256")
            != config.source_manifest_sha256
            or e01_manifest.get("source_chunks_sha256") != config.source_chunks_sha256
            or e01_manifest.get("record_count") != config.source_chunk_count
        ):
            raise E01Error("Runtime E01 manifest identity mismatch.")
        self._e00 = _readonly_connection(e00_directory / "bm25.sqlite3")
        self._e01 = _readonly_connection(e01_directory / "secondary_sparse.sqlite3")
        metadata = _metadata(self._e01)
        if metadata.get("index_config_sha256") != config.index_config_sha256:
            raise E01Error("E01 database index-config identity mismatch.")
        if metadata.get("source_chunks_sha256") != config.source_chunks_sha256:
            raise E01Error("E01 database chunk lineage mismatch.")
        if metadata.get("code_version") != CODE_VERSION:
            raise E01Error("E01 database code version mismatch.")

    def search(self, query: str, *, top_k: int | None = None) -> SparseRetrievalResponse:
        """Retrieve all branches independently and fuse rank contributions."""

        if not isinstance(query, str) or not query.strip():
            raise ValueError("Sparse query must be non-blank text.")
        resolved_top_k = top_k if top_k is not None else self._config.default_top_k
        if resolved_top_k <= 0:
            raise ValueError("top_k must be positive.")
        started = time.perf_counter()
        candidate_k = max(self._config.candidate_k_per_branch, resolved_top_k)
        branch_started = time.perf_counter()
        word = self._search_e00_word(query, candidate_k)
        word_latency_ms = (time.perf_counter() - branch_started) * 1000
        branch_started = time.perf_counter()
        accent = self._search_secondary_word(query, candidate_k)
        accent_latency_ms = (time.perf_counter() - branch_started) * 1000
        branch_started = time.perf_counter()
        character = self._search_secondary_character(query, candidate_k)
        character_latency_ms = (time.perf_counter() - branch_started) * 1000
        branches = {"word": word, "accent_word": accent, "character": character}
        ranked = weighted_rrf(
            {name: [hit.chunk_id for hit in hits] for name, hits in branches.items()},
            weights=self._config.branch_weights,
            rrf_constant=self._config.rrf_constant,
        )
        selected = ranked[:resolved_top_k]
        payloads = self._fetch_e00_chunks([item[0] for item in selected])
        hits: list[FusedSparseHit] = []
        for rank, (chunk_id, fused_score, branch_ranks, contributions) in enumerate(selected, start=1):
            payload = payloads.get(chunk_id)
            if payload is None:
                raise E01Error(f"Fused chunk is absent from E00: {chunk_id}")
            hits.append(
                FusedSparseHit(
                    chunk_id=chunk_id,
                    document_id=payload["document_id"],
                    article_number=payload["article_number"],
                    text=payload["text"],
                    metadata=payload["metadata"],
                    rank=rank,
                    fused_score=fused_score,
                    branch_ranks=branch_ranks,
                    branch_contributions=contributions,
                )
            )
        return SparseRetrievalResponse(
            hits=tuple(hits),
            branch_candidate_counts={name: len(hits) for name, hits in branches.items()},
            branch_latency_ms={
                "word": word_latency_ms,
                "accent_word": accent_latency_ms,
                "character": character_latency_ms,
            },
            latency_ms=(time.perf_counter() - started) * 1000,
        )

    def close(self) -> None:
        """Close both read-only databases."""

        self._e01.close()
        self._e00.close()

    def _search_e00_word(self, query: str, top_k: int) -> list[SparseBranchHit]:
        tokens = _unicode_tokens(query)
        return _fts_search(
            self._e00,
            table="chunks_fts",
            query=_or_expression(tokens),
            top_k=top_k,
            join_e00=True,
        )

    def _search_secondary_word(self, query: str, top_k: int) -> list[SparseBranchHit]:
        return _fts_search(
            self._e01,
            table="accent_fts",
            query=_or_expression(folded_query_tokens(query)),
            top_k=top_k,
            join_e00=False,
        )

    def _search_secondary_character(self, query: str, top_k: int) -> list[SparseBranchHit]:
        terms = select_trigram_query_terms(
            query,
            minimum_characters=self._config.trigram_query_min_term_characters,
            maximum_terms=self._config.trigram_query_max_terms,
        )
        return _fts_search(
            self._e01,
            table="character_fts",
            query=_character_expression(terms),
            top_k=top_k,
            join_e00=False,
        )

    def _fetch_e00_chunks(self, chunk_ids: list[str]) -> dict[str, dict[str, Any]]:
        result: dict[str, dict[str, Any]] = {}
        for start in range(0, len(chunk_ids), 400):
            batch = chunk_ids[start : start + 400]
            placeholders = ",".join("?" for _ in batch)
            rows = self._e00.execute(
                f"SELECT chunk_id, document_id, article_number, text, metadata_json FROM chunks WHERE chunk_id IN ({placeholders})",
                batch,
            ).fetchall()
            for row in rows:
                result[row["chunk_id"]] = {
                    "document_id": row["document_id"],
                    "article_number": row["article_number"],
                    "text": row["text"],
                    "metadata": json.loads(row["metadata_json"]),
                }
        return result


def weighted_rrf(
    branch_rankings: dict[str, list[str]],
    *,
    weights: dict[str, float],
    rrf_constant: int,
) -> list[tuple[str, float, dict[str, int], dict[str, float]]]:
    """Fuse independent rankings and retain exact contribution trace."""

    if rrf_constant <= 0:
        raise ValueError("RRF constant must be positive.")
    if set(branch_rankings) != set(weights):
        raise ValueError("Every RRF branch must have exactly one weight.")
    scores: dict[str, float] = {}
    ranks: dict[str, dict[str, int]] = {}
    contributions: dict[str, dict[str, float]] = {}
    for branch, ranking in branch_rankings.items():
        if len(ranking) != len(set(ranking)):
            raise ValueError(f"RRF branch {branch} contains duplicate chunk IDs.")
        weight = weights[branch]
        if weight <= 0:
            raise ValueError("RRF weights must be positive.")
        for rank, chunk_id in enumerate(ranking, start=1):
            contribution = weight / (rrf_constant + rank)
            scores[chunk_id] = scores.get(chunk_id, 0.0) + contribution
            ranks.setdefault(chunk_id, {})[branch] = rank
            contributions.setdefault(chunk_id, {})[branch] = contribution
    return [
        (chunk_id, scores[chunk_id], ranks[chunk_id], contributions[chunk_id])
        for chunk_id in sorted(
            scores,
            key=lambda item: (
                -scores[item],
                min(ranks[item].values()),
                item,
            ),
        )
    ]


def _validate_e00_lineage(
    manifest_path: Path,
    chunks_path: Path,
    config: E01Config,
) -> None:
    if not manifest_path.is_file() or not chunks_path.is_file():
        raise E01Error("Required E00 manifest or chunks payload is missing.")
    if file_sha256(manifest_path) != config.source_manifest_sha256:
        raise E01Error("Pinned E00 manifest checksum mismatch.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("artifact_version") != "e00-v1"
        or manifest.get("dataset_revision") != config.dataset_revision
        or manifest.get("record_counts", {}).get("chunks") != config.source_chunk_count
        or manifest.get("files", {}).get("chunks.jsonl", {}).get("sha256")
        != config.source_chunks_sha256
    ):
        raise E01Error("Pinned E00 manifest content is incompatible with E01.")


def _initialize_secondary_database(connection: sqlite3.Connection, config: E01Config) -> None:
    connection.execute("PRAGMA journal_mode=OFF")
    connection.execute("PRAGMA synchronous=OFF")
    connection.execute("PRAGMA locking_mode=EXCLUSIVE")
    connection.execute(
        "CREATE TABLE row_map(rowid INTEGER PRIMARY KEY, chunk_id TEXT NOT NULL UNIQUE)"
    )
    connection.execute("CREATE TABLE metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    try:
        connection.execute(
            "CREATE VIRTUAL TABLE accent_fts USING fts5("
            f"folded_text, content='', tokenize='{config.accent_word_tokenizer}')"
        )
        connection.execute(
            "CREATE VIRTUAL TABLE character_fts USING fts5("
            f"folded_text, content='', tokenize='{config.character_tokenizer}')"
        )
    except sqlite3.OperationalError as exc:
        raise E01Error("SQLite does not provide the pinned FTS5 tokenizers.") from exc


def _insert_secondary_batch(
    connection: sqlite3.Connection,
    records: list[tuple[str, str, str]],
) -> None:
    cursor = connection.cursor()
    try:
        cursor.execute("BEGIN")
        for chunk_id, folded_word_text, folded_character_text in records:
            cursor.execute("INSERT INTO row_map(chunk_id) VALUES (?)", (chunk_id,))
            rowid = cursor.lastrowid
            cursor.execute(
                "INSERT INTO accent_fts(rowid, folded_text) VALUES (?, ?)",
                (rowid, folded_word_text),
            )
            cursor.execute(
                "INSERT INTO character_fts(rowid, folded_text) VALUES (?, ?)",
                (rowid, folded_character_text),
            )
        connection.commit()
    except Exception:
        connection.rollback()
        raise


def _decode_chunk_line(raw_line: bytes, line_number: int) -> dict[str, str]:
    try:
        payload = json.loads(raw_line.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise E01Error(f"Invalid E00 chunk JSON at line {line_number}.") from exc
    if not isinstance(payload, dict):
        raise E01Error(f"E00 chunk line {line_number} is not an object.")
    chunk_id = payload.get("chunk_id")
    search_text = payload.get("search_text")
    text = payload.get("text")
    if (
        not isinstance(chunk_id, str)
        or not chunk_id
        or not isinstance(search_text, str)
        or not search_text
        or not isinstance(text, str)
        or not text
    ):
        raise E01Error(f"E00 chunk line {line_number} lacks retrieval fields.")
    return {"chunk_id": chunk_id, "search_text": search_text, "text": text}


def _require_database_integrity(connection: sqlite3.Connection, expected_count: int) -> None:
    integrity = connection.execute("PRAGMA integrity_check").fetchone()
    counts = [
        connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        for table in ("row_map", "accent_fts", "character_fts")
    ]
    if integrity is None or integrity[0] != "ok" or any(count != expected_count for count in counts):
        raise E01Error("E01 SQLite integrity or record-count validation failed.")


def _fts_search(
    connection: sqlite3.Connection,
    *,
    table: str,
    query: str | None,
    top_k: int,
    join_e00: bool,
) -> list[SparseBranchHit]:
    if query is None:
        return []
    if table not in {"chunks_fts", "accent_fts", "character_fts"}:
        raise ValueError("Unsupported FTS table.")
    if join_e00:
        statement = f"""
            SELECT c.chunk_id AS chunk_id, bm25({table}) AS raw_score
            FROM {table}
            JOIN chunks AS c ON c.rowid = {table}.rowid
            WHERE {table} MATCH ?
            ORDER BY raw_score ASC, c.chunk_id ASC
            LIMIT ?
        """
    else:
        statement = f"""
            SELECT m.chunk_id AS chunk_id, bm25({table}) AS raw_score
            FROM {table}
            JOIN row_map AS m ON m.rowid = {table}.rowid
            WHERE {table} MATCH ?
            ORDER BY raw_score ASC, m.chunk_id ASC
            LIMIT ?
        """
    try:
        rows = connection.execute(statement, (query, top_k)).fetchall()
    except sqlite3.DatabaseError as exc:
        raise E01Error(f"Sparse branch query failed for {table}.") from exc
    return [
        SparseBranchHit(
            chunk_id=row["chunk_id"],
            rank=index + 1,
            score=-float(row["raw_score"]),
        )
        for index, row in enumerate(rows)
    ]


def _unicode_tokens(text: str) -> list[str]:
    seen: set[str] = set()
    tokens: list[str] = []
    for match in re.finditer(r"[^\W_]+", text.casefold(), re.UNICODE):
        token = match.group(0)
        if token not in seen:
            seen.add(token)
            tokens.append(token)
    return tokens


def _or_expression(tokens: list[str]) -> str | None:
    return " OR ".join(f'"{token}"' for token in tokens) if tokens else None


def _minimum_two_expression(tokens: list[str]) -> str | None:
    """Require at least two selected terms to bound broad FTS scans."""

    if not tokens:
        return None
    quoted = [f'"{token}"' for token in tokens]
    if len(quoted) == 1:
        return quoted[0]
    if len(quoted) == 2:
        return f"{quoted[0]} AND {quoted[1]}"
    pairs = [
        f"({quoted[left]} AND {quoted[right]})"
        for left in range(len(quoted))
        for right in range(left + 1, len(quoted))
    ]
    return " OR ".join(pairs)


def _character_expression(tokens: list[str]) -> str | None:
    return _minimum_two_expression(tokens)


def _readonly_connection(path: Path) -> sqlite3.Connection:
    if not path.is_file():
        raise E01Error(f"Sparse database is missing: {path.name}")
    connection = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    return connection


def _metadata(connection: sqlite3.Connection) -> dict[str, Any]:
    rows = connection.execute("SELECT key, value FROM metadata").fetchall()
    return {row["key"]: json.loads(row["value"]) for row in rows}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


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


def _sha256(payload: dict[str, Any], key: str) -> str:
    value = _text(payload, key)
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{key} must be a lowercase SHA-256 digest.")
    return value


def _revision(payload: dict[str, Any], key: str) -> str:
    value = _text(payload, key)
    if not value.startswith("sha256:"):
        raise ValueError(f"{key} must be a sha256: revision.")
    digest = value.removeprefix("sha256:")
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError(f"{key} must contain a lowercase SHA-256 digest.")
    return value


def _require_keys(payload: dict[str, Any], expected: set[str], label: str) -> None:
    if set(payload) != expected:
        raise ValueError(f"{label} fields are incompatible.")
