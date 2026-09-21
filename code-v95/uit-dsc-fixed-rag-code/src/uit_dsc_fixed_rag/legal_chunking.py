"""Article-aware parsing and deterministic legal chunk construction."""

from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass
from typing import Any

from uit_dsc_fixed_rag.corpus import SourceDocument


_ARTICLE = re.compile(
    r"(?im)^\s*Điều\s+(?P<number>[0-9]+(?:[a-zđ])?)\s*(?:[.:-]\s*)?(?P<title>[^\n]*)"
)
_CLAUSE = re.compile(r"(?im)^\s*(?P<number>[0-9]+(?:[a-zđ])?)[.)]\s+")
_TOKEN = re.compile(r"\S+")


@dataclass(frozen=True)
class ChunkingConfig:
    """Bounded, model-independent E00 chunking policy."""

    max_whitespace_tokens: int
    max_characters: int
    overlap_whitespace_tokens: int
    document_header_characters: int

    def validate(self) -> None:
        """Reject unsafe or non-progressing chunk windows."""

        if self.max_whitespace_tokens < 16:
            raise ValueError("max_whitespace_tokens must be at least 16")
        if self.max_characters < 128:
            raise ValueError("max_characters must be at least 128")
        if not 0 <= self.overlap_whitespace_tokens < self.max_whitespace_tokens:
            raise ValueError("overlap must be non-negative and smaller than max tokens")
        if self.document_header_characters < 0:
            raise ValueError("document_header_characters must be non-negative")


@dataclass(frozen=True)
class ExactTokenizerPolicy:
    """Exact model-token budget and offset windows for the final fallback."""

    tokenizer: Any
    max_tokens: int
    overlap_tokens: int
    add_special_tokens: bool = True

    def validate(self) -> None:
        if not getattr(self.tokenizer, "is_fast", False):
            raise ValueError("Exact token fallback requires a fast tokenizer with offsets.")
        if self.max_tokens < 16:
            raise ValueError("Exact max_tokens must be at least 16.")
        if not 0 <= self.overlap_tokens < self.max_tokens:
            raise ValueError("Exact overlap must be smaller than max_tokens.")
        if self.content_capacity <= 0:
            raise ValueError("Tokenizer special tokens consume the complete budget.")

    @property
    def content_capacity(self) -> int:
        special = (
            int(self.tokenizer.num_special_tokens_to_add(pair=False))
            if self.add_special_tokens
            else 0
        )
        return self.max_tokens - special

    def count(self, text: str) -> int:
        encoded = self.tokenizer(
            text,
            add_special_tokens=self.add_special_tokens,
            truncation=False,
            padding=False,
        )
        return len(encoded["input_ids"])

    def window_spans(self, text: str, start: int, end: int) -> list[tuple[int, int]]:
        """Return bounded exact character spans with deterministic token overlap."""

        fragment = text[start:end]
        encoded = self.tokenizer(
            fragment,
            add_special_tokens=False,
            truncation=False,
            padding=False,
            return_offsets_mapping=True,
        )
        offsets = [
            (int(offset_start), int(offset_end))
            for offset_start, offset_end in encoded["offset_mapping"]
            if int(offset_end) > int(offset_start)
        ]
        if not offsets:
            return []
        windows: list[tuple[int, int]] = []
        token_start = 0
        while token_start < len(offsets):
            token_end = min(len(offsets), token_start + self.content_capacity)
            while token_end > token_start:
                candidate_start = start + offsets[token_start][0]
                candidate_end = start + offsets[token_end - 1][1]
                trimmed = _trim_span(text, candidate_start, candidate_end)
                if trimmed is not None and self.count(text[trimmed[0] : trimmed[1]]) <= self.max_tokens:
                    break
                token_end -= 1
            if token_end <= token_start or trimmed is None:
                raise ValueError("Exact tokenizer fallback could not make bounded progress.")
            windows.append(trimmed)
            if token_end >= len(offsets):
                break
            token_start = max(token_start + 1, token_end - self.overlap_tokens)
        return windows


@dataclass(frozen=True)
class LegalChunk:
    """One traceable retrieval unit over exact cleaned-document character spans."""

    chunk_id: str
    document_id: str
    source_title: str | None
    source_url: str
    article_number: str | None
    article_title: str | None
    clause_numbers: tuple[str, ...]
    section_kind: str
    split_method: str
    chunk_index: int
    start_char: int
    end_char: int
    whitespace_token_count: int
    model_token_count: int | None
    text: str
    search_text: str

    def to_json(self) -> dict[str, Any]:
        """Return a stable JSON-compatible representation."""

        payload = asdict(self)
        payload["clause_numbers"] = list(self.clause_numbers)
        return payload


@dataclass(frozen=True)
class _ArticleSpan:
    number: str
    title: str | None
    heading: str
    start: int
    heading_end: int
    end: int


class LegalChunker:
    """Prefer Điều, then Khoản groups, then bounded token windows."""

    version = "article-clause-whitespace-v1"

    def __init__(
        self,
        config: ChunkingConfig,
        *,
        exact_token_policy: ExactTokenizerPolicy | None = None,
    ) -> None:
        config.validate()
        if exact_token_policy is not None:
            exact_token_policy.validate()
        self._config = config
        self._exact_token_policy = exact_token_policy

    def chunk_document(
        self,
        document: SourceDocument,
        cleaned_text: str,
    ) -> list[LegalChunk]:
        """Chunk one cleaned document without dropping non-blank characters."""

        if not cleaned_text.strip():
            return []
        articles = _article_spans(cleaned_text)
        header_end = articles[0].start if articles else min(len(cleaned_text), self._config.document_header_characters)
        document_header = _compact_header(
            cleaned_text[:header_end], self._config.document_header_characters
        )
        specs: list[tuple[int, int, str | None, str | None, str, str]] = []

        if articles:
            preamble = _trim_span(cleaned_text, 0, articles[0].start)
            if preamble is not None:
                for start, end in self._window_spans(cleaned_text, *preamble):
                    specs.append((start, end, None, None, "preamble", self._fallback_method))
            for article in articles:
                specs.extend(self._chunk_article(cleaned_text, article))
        else:
            entire = _trim_span(cleaned_text, 0, len(cleaned_text))
            if entire is not None:
                for start, end in self._window_spans(cleaned_text, *entire):
                    specs.append((start, end, None, None, "unstructured", self._fallback_method))

        chunks: list[LegalChunk] = []
        for index, (start, end, article_number, article_title, kind, method) in enumerate(specs):
            text = cleaned_text[start:end]
            article_heading = None
            if article_number is not None:
                matching = next(
                    article for article in articles if article.number == article_number and article.start <= start < article.end
                )
                article_heading = matching.heading
            search_text = _search_text(
                document.source_title,
                document_header,
                article_heading,
                text,
            )
            digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]
            chunk_id = f"ctx-{document.document_id}::{index:05d}::{digest}"
            chunks.append(
                LegalChunk(
                    chunk_id=chunk_id,
                    document_id=document.document_id,
                    source_title=document.source_title,
                    source_url=document.source_url,
                    article_number=article_number,
                    article_title=article_title,
                    clause_numbers=tuple(match.group("number") for match in _CLAUSE.finditer(text)),
                    section_kind=kind,
                    split_method=method,
                    chunk_index=index,
                    start_char=start,
                    end_char=end,
                    whitespace_token_count=len(_TOKEN.findall(text)),
                    model_token_count=(
                        self._exact_token_policy.count(text)
                        if self._exact_token_policy is not None
                        else None
                    ),
                    text=text,
                    search_text=search_text,
                )
            )
        return chunks

    def _chunk_article(
        self,
        text: str,
        article: _ArticleSpan,
    ) -> list[tuple[int, int, str | None, str | None, str, str]]:
        trimmed = _trim_span(text, article.start, article.end)
        if trimmed is None:
            return []
        start, end = trimmed
        if self._within_budget(text[start:end]):
            return [(start, end, article.number, article.title, "article", "article")]

        clauses = list(_CLAUSE.finditer(text, article.heading_end, article.end))
        if not clauses:
            return [
                (window_start, window_end, article.number, article.title, "article", self._fallback_method)
                for window_start, window_end in self._window_spans(text, start, end)
            ]

        units: list[tuple[int, int]] = []
        intro = _trim_span(text, article.start, clauses[0].start())
        if intro is not None:
            units.append(intro)
        for index, clause in enumerate(clauses):
            clause_end = clauses[index + 1].start() if index + 1 < len(clauses) else article.end
            unit = _trim_span(text, clause.start(), clause_end)
            if unit is not None:
                units.append(unit)

        specs: list[tuple[int, int, str | None, str | None, str, str]] = []
        group_start: int | None = None
        group_end: int | None = None
        for unit_start, unit_end in units:
            if not self._within_budget(text[unit_start:unit_end]):
                if group_start is not None and group_end is not None:
                    specs.append((group_start, group_end, article.number, article.title, "clause_group", "clause_group"))
                    group_start = group_end = None
                specs.extend(
                    (window_start, window_end, article.number, article.title, "clause", self._fallback_method)
                    for window_start, window_end in self._window_spans(text, unit_start, unit_end)
                )
                continue
            candidate_start = group_start if group_start is not None else unit_start
            if group_end is None or self._within_budget(text[candidate_start:unit_end]):
                group_start, group_end = candidate_start, unit_end
            else:
                specs.append((group_start, group_end, article.number, article.title, "clause_group", "clause_group"))
                group_start, group_end = unit_start, unit_end
        if group_start is not None and group_end is not None:
            specs.append((group_start, group_end, article.number, article.title, "clause_group", "clause_group"))
        return specs

    def _within_budget(self, text: str) -> bool:
        return (
            len(text) <= self._config.max_characters
            and len(_TOKEN.findall(text)) <= self._config.max_whitespace_tokens
            and (
                self._exact_token_policy is None
                or self._exact_token_policy.count(text) <= self._exact_token_policy.max_tokens
            )
        )

    def _window_spans(self, text: str, start: int, end: int) -> list[tuple[int, int]]:
        if self._exact_token_policy is not None:
            return self._exact_token_policy.window_spans(text, start, end)
        tokens = list(_TOKEN.finditer(text, start, end))
        if not tokens:
            return []
        windows: list[tuple[int, int]] = []
        token_start = 0
        while token_start < len(tokens):
            token_end = token_start
            first_char = tokens[token_start].start()
            while token_end < len(tokens):
                candidate = tokens[token_end]
                if token_end > token_start and (
                    token_end - token_start >= self._config.max_whitespace_tokens
                    or candidate.end() - first_char > self._config.max_characters
                ):
                    break
                token_end += 1
            if token_end == token_start:
                token_end += 1
            windows.append((tokens[token_start].start(), tokens[token_end - 1].end()))
            if token_end >= len(tokens):
                break
            next_start = max(token_start + 1, token_end - self._config.overlap_whitespace_tokens)
            token_start = next_start
        return windows

    @property
    def _fallback_method(self) -> str:
        return "model_token_fallback" if self._exact_token_policy is not None else "token_fallback"


def _article_spans(text: str) -> list[_ArticleSpan]:
    matches = list(_ARTICLE.finditer(text))
    spans: list[_ArticleSpan] = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        title = match.group("title").strip() or None
        spans.append(
            _ArticleSpan(
                number=match.group("number"),
                title=title,
                heading=match.group(0).strip(),
                start=match.start(),
                heading_end=match.end(),
                end=end,
            )
        )
    return spans


def _trim_span(text: str, start: int, end: int) -> tuple[int, int] | None:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return (start, end) if start < end else None


def _compact_header(text: str, limit: int) -> str:
    if limit == 0:
        return ""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    compact = "\n".join(lines)
    return compact[:limit].rstrip()


def _search_text(
    source_title: str | None,
    document_header: str,
    article_heading: str | None,
    text: str,
) -> str:
    parts: list[str] = []
    for value in (source_title, document_header, article_heading, text):
        if value and value not in parts:
            parts.append(value)
    return "\n".join(parts)
