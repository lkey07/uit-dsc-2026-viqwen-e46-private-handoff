"""Persisted SQLite FTS5 BM25 backend for the E00 sparse baseline."""

from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from uit_dsc_fixed_rag.legal_chunking import LegalChunk


class Bm25Error(RuntimeError):
    """Raised when the local sparse backend is unavailable or incompatible."""


_QUERY_TOKEN = re.compile(r"[^\W_]+", re.UNICODE)


@dataclass(frozen=True)
class Bm25Hit:
    """One ranked sparse retrieval result."""

    chunk_id: str
    document_id: str
    article_number: str | None
    score: float
    rank: int
    text: str
    metadata: dict[str, Any]


class SqliteBm25Writer:
    """Build a bounded local FTS5 index in batches."""

    backend_name = "sqlite-fts5"

    def __init__(self, path: Path, *, tokenizer: str, batch_size: int) -> None:
        if batch_size <= 0:
            raise ValueError("BM25 batch size must be positive.")
        self._path = path
        self._tokenizer = tokenizer
        self._batch_size = batch_size
        self._pending: list[tuple[Any, ...]] = []
        self._connection = sqlite3.connect(path)
        self._initialize()

    def _initialize(self) -> None:
        connection = self._connection
        connection.execute("PRAGMA journal_mode=OFF")
        connection.execute("PRAGMA synchronous=OFF")
        connection.execute("PRAGMA locking_mode=EXCLUSIVE")
        connection.execute(
            """
            CREATE TABLE chunks (
                rowid INTEGER PRIMARY KEY,
                chunk_id TEXT NOT NULL UNIQUE,
                document_id TEXT NOT NULL,
                article_number TEXT,
                text TEXT NOT NULL,
                metadata_json TEXT NOT NULL
            )
            """
        )
        try:
            connection.execute(
                "CREATE VIRTUAL TABLE chunks_fts USING fts5("
                f"search_text, tokenize='{self._tokenizer}')"
            )
        except sqlite3.OperationalError as exc:
            connection.close()
            raise Bm25Error("Python SQLite build does not provide compatible FTS5.") from exc
        connection.execute(
            "CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
        )

    def add(self, chunk: LegalChunk) -> None:
        """Queue one chunk for insertion without changing its legal text."""

        metadata = chunk.to_json()
        metadata.pop("text")
        metadata.pop("search_text")
        self._pending.append(
            (
                chunk.chunk_id,
                chunk.document_id,
                chunk.article_number,
                chunk.text,
                json.dumps(metadata, ensure_ascii=False, sort_keys=True),
                chunk.search_text,
            )
        )
        if len(self._pending) >= self._batch_size:
            self.flush()

    def flush(self) -> None:
        """Commit the current bounded insertion batch."""

        if not self._pending:
            return
        cursor = self._connection.cursor()
        try:
            cursor.execute("BEGIN")
            for chunk_id, document_id, article_number, text, metadata, search_text in self._pending:
                cursor.execute(
                    "INSERT INTO chunks(chunk_id, document_id, article_number, text, metadata_json) VALUES (?, ?, ?, ?, ?)",
                    (chunk_id, document_id, article_number, text, metadata),
                )
                cursor.execute(
                    "INSERT INTO chunks_fts(rowid, search_text) VALUES (?, ?)",
                    (cursor.lastrowid, search_text),
                )
            self._connection.commit()
            self._pending.clear()
        except Exception:
            self._connection.rollback()
            raise

    def finalize(self, metadata: dict[str, Any]) -> None:
        """Flush records, persist identity metadata and optimize the FTS index."""

        self.flush()
        self._connection.executemany(
            "INSERT INTO metadata(key, value) VALUES (?, ?)",
            [
                (key, json.dumps(value, ensure_ascii=False, sort_keys=True))
                for key, value in sorted(metadata.items())
            ],
        )
        self._connection.execute("INSERT INTO chunks_fts(chunks_fts) VALUES ('optimize')")
        self._connection.commit()
        integrity = self._connection.execute("PRAGMA integrity_check").fetchone()
        if integrity is None or integrity[0] != "ok":
            raise Bm25Error("SQLite integrity check failed after index build.")
        self._connection.close()

    def abort(self) -> None:
        """Close a partial database after a failed build."""

        self._connection.close()


class SqliteBm25Index:
    """Read-only BM25 retrieval over a persisted E00 artifact."""

    def __init__(self, path: Path) -> None:
        if not path.is_file():
            raise Bm25Error("BM25 index file does not exist.")
        uri = path.resolve().as_uri() + "?mode=ro"
        self._connection = sqlite3.connect(uri, uri=True)
        self._connection.row_factory = sqlite3.Row

    def search(self, query: str, *, top_k: int) -> list[Bm25Hit]:
        """Return deterministic top-k hits for Unicode word tokens."""

        if top_k <= 0:
            raise ValueError("top_k must be positive.")
        tokens = _query_tokens(query)
        if not tokens:
            return []
        match_query = " OR ".join(f'"{token}"' for token in tokens)
        try:
            rows = self._connection.execute(
                """
                SELECT c.chunk_id, c.document_id, c.article_number, c.text,
                       c.metadata_json, bm25(chunks_fts) AS raw_score
                FROM chunks_fts
                JOIN chunks AS c ON c.rowid = chunks_fts.rowid
                WHERE chunks_fts MATCH ?
                ORDER BY raw_score ASC, c.chunk_id ASC
                LIMIT ?
                """,
                (match_query, top_k),
            ).fetchall()
        except sqlite3.DatabaseError as exc:
            raise Bm25Error("BM25 query failed against the persisted index.") from exc
        return [
            Bm25Hit(
                chunk_id=row["chunk_id"],
                document_id=row["document_id"],
                article_number=row["article_number"],
                score=-float(row["raw_score"]),
                rank=index + 1,
                text=row["text"],
                metadata=json.loads(row["metadata_json"]),
            )
            for index, row in enumerate(rows)
        ]

    def metadata(self) -> dict[str, Any]:
        """Load the persisted index identity map."""

        rows = self._connection.execute("SELECT key, value FROM metadata").fetchall()
        return {row["key"]: json.loads(row["value"]) for row in rows}

    def close(self) -> None:
        """Close the read-only SQLite connection."""

        self._connection.close()


def _query_tokens(query: str) -> list[str]:
    if not isinstance(query, str):
        raise TypeError("BM25 query must be text.")
    seen: set[str] = set()
    tokens: list[str] = []
    for match in _QUERY_TOKEN.finditer(query.casefold()):
        token = match.group(0)
        if token not in seen:
            seen.add(token)
            tokens.append(token)
    return tokens
