"""Leakage-safe, resumable answer-derived diagnostics for E00 versus E01."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any

from uit_dsc_fixed_rag.bm25 import SqliteBm25Index
from uit_dsc_fixed_rag.corpus import file_sha256
from uit_dsc_fixed_rag.e01_sparse import E01Config, HybridSparseRetriever
from uit_dsc_fixed_rag.sparse_normalization import (
    fold_vietnamese_accents,
    folded_query_tokens,
)


LOGGER = logging.getLogger(__name__)


class DiagnosticError(RuntimeError):
    """Raised when diagnostic inputs or checkpoint identity are incompatible."""


_ARTICLE_REFERENCE = re.compile(r"(?i)\bĐiều\s+([0-9]+(?:[a-zđ])?)")
_DOCUMENT_REFERENCE = re.compile(
    r"\b[0-9]{1,4}/[0-9]{4}/[A-ZĐ]+(?:-[A-ZĐ0-9]+)+\b",
    re.UNICODE,
)


def select_dev_sample(
    records: dict[str, dict[str, Any]],
    *,
    seed: str,
    size: int,
) -> list[str]:
    """Select a deterministic question-ID sample without reading answers."""

    if size <= 0 or size > len(records):
        raise ValueError("Diagnostic sample size is outside the dev record range.")
    if not seed:
        raise ValueError("Diagnostic sample seed must not be blank.")
    return sorted(
        records,
        key=lambda question_id: (
            hashlib.sha256(f"{seed}\0{question_id}".encode("utf-8")).digest(),
            question_id,
        ),
    )[:size]


def answer_derived_diagnostics(
    answer: str,
    hits: list[dict[str, Any]],
) -> dict[str, Any]:
    """Measure answer/context overlap after retrieval without creating labels."""

    context_text = "\n".join(str(hit["text"]) for hit in hits)
    folded_context = fold_vietnamese_accents(context_text)
    answer_tokens = set(folded_query_tokens(answer))
    context_tokens = set(folded_query_tokens(context_text))
    coverage = (
        len(answer_tokens & context_tokens) / len(answer_tokens)
        if answer_tokens
        else 0.0
    )

    article_references = sorted(set(_ARTICLE_REFERENCE.findall(answer.casefold())))
    document_references = sorted(
        set(match.group(0) for match in _DOCUMENT_REFERENCE.finditer(answer.upper()))
    )
    article_hits = 0
    for reference in article_references:
        folded_label = fold_vietnamese_accents(f"Điều {reference}")
        if any(
            str(hit.get("article_number") or "").casefold() == reference.casefold()
            or folded_label in fold_vietnamese_accents(str(hit["text"]))
            for hit in hits
        ):
            article_hits += 1
    document_hits = sum(
        fold_vietnamese_accents(reference) in folded_context
        for reference in document_references
    )
    reference_count = len(article_references) + len(document_references)
    reference_hit_count = article_hits + document_hits
    return {
        "answer_unique_token_count": len(answer_tokens),
        "answer_unique_token_coverage": coverage,
        "explicit_reference_count": reference_count,
        "explicit_reference_hit_count": reference_hit_count,
        "explicit_reference_any_hit": bool(reference_hit_count),
        "explicit_reference_all_hit": (
            reference_count > 0 and reference_hit_count == reference_count
        ),
        "context_character_count": len(context_text),
    }


def run_dev_diagnostics(
    *,
    dev_path: Path,
    e00_directory: Path,
    e01_directory: Path,
    output_directory: Path,
    config: E01Config,
) -> dict[str, Any]:
    """Run or resume dev diagnostics with one durable record per question."""

    if file_sha256(dev_path) != config.dev_sha256:
        raise DiagnosticError("Dev split checksum differs from the pinned E01 config.")
    dev = _load_dev(dev_path)
    sample_ids = select_dev_sample(dev, seed=config.sample_seed, size=config.sample_size)
    identity = _run_identity(
        sample_ids=sample_ids,
        config=config,
        e00_manifest=e00_directory / "manifest.json",
        e01_manifest=e01_directory / "manifest.json",
    )
    output_directory.mkdir(parents=True, exist_ok=True)
    results_path = output_directory / "results.jsonl"
    state_path = output_directory / "state.json"
    completed = _load_checkpoint(results_path, state_path, identity, sample_ids)

    e00 = SqliteBm25Index(e00_directory / "bm25.sqlite3")
    e01 = HybridSparseRetriever(
        e00_directory,
        e01_directory,
        config,
    )
    try:
        with results_path.open("a", encoding="utf-8", newline="\n") as stream:
            for index, question_id in enumerate(sample_ids[len(completed) :], start=len(completed)):
                record = dev[question_id]
                question = record["question"]
                answer = record["answer"]

                started = time.perf_counter()
                e00_hits = e00.search(question, top_k=config.diagnostic_top_k)
                e00_latency_ms = (time.perf_counter() - started) * 1000
                e01_response = e01.search(question, top_k=config.diagnostic_top_k)
                e00_payload = [
                    {
                        "chunk_id": hit.chunk_id,
                        "article_number": hit.article_number,
                        "text": hit.text,
                    }
                    for hit in e00_hits
                ]
                e01_payload = [
                    {
                        "chunk_id": hit.chunk_id,
                        "article_number": hit.article_number,
                        "text": hit.text,
                    }
                    for hit in e01_response.hits
                ]
                e00_metrics = answer_derived_diagnostics(answer, e00_payload)
                e01_metrics = answer_derived_diagnostics(answer, e01_payload)
                e00_ids = {item["chunk_id"] for item in e00_payload}
                e01_ids = {item["chunk_id"] for item in e01_payload}
                union = e00_ids | e01_ids
                result = {
                    "question_id": question_id,
                    "sample_index": index,
                    "e00": {
                        **e00_metrics,
                        "latency_ms": e00_latency_ms,
                        "top_chunk_ids": [item["chunk_id"] for item in e00_payload],
                    },
                    "e01": {
                        **e01_metrics,
                        "latency_ms": e01_response.latency_ms,
                        "top_chunk_ids": [item["chunk_id"] for item in e01_payload],
                        "branch_candidate_counts": e01_response.branch_candidate_counts,
                    },
                    "top_k_jaccard": len(e00_ids & e01_ids) / len(union) if union else 1.0,
                }
                stream.write(json.dumps(result, ensure_ascii=False, sort_keys=True) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
                completed.append(result)
                if len(completed) % config.checkpoint_every_questions == 0:
                    _write_state(state_path, identity, len(completed), complete=False)
                    LOGGER.info(
                        "e01_diagnostic_progress completed=%d total=%d question_id=%s",
                        len(completed),
                        len(sample_ids),
                        question_id,
                    )
    finally:
        e01.close()
        e00.close()

    report = _aggregate(completed, identity, sample_ids)
    _atomic_write_json(output_directory / "report.json", report)
    _write_state(state_path, identity, len(completed), complete=True)
    return report


def _aggregate(
    records: list[dict[str, Any]],
    identity: dict[str, Any],
    sample_ids: list[str],
) -> dict[str, Any]:
    if len(records) != len(sample_ids):
        raise DiagnosticError("Cannot aggregate an incomplete diagnostic run.")

    def mean(path: tuple[str, str]) -> float:
        return sum(float(record[path[0]][path[1]]) for record in records) / len(records)

    def reference_rate(system: str, field: str) -> float | None:
        eligible = [
            record
            for record in records
            if int(record[system]["explicit_reference_count"]) > 0
        ]
        if not eligible:
            return None
        return sum(bool(record[system][field]) for record in eligible) / len(eligible)

    return {
        "schema_version": "1.0",
        "experiment_id": "E01-dev200-retrieval-diagnostic",
        "run_identity": identity,
        "sample_size": len(records),
        "sample_ids_sha256": hashlib.sha256(
            "\n".join(sample_ids).encode("utf-8")
        ).hexdigest(),
        "metrics": {
            "e00": {
                "mean_answer_unique_token_coverage": mean(("e00", "answer_unique_token_coverage")),
                "explicit_reference_any_hit_rate": reference_rate("e00", "explicit_reference_any_hit"),
                "explicit_reference_all_hit_rate": reference_rate("e00", "explicit_reference_all_hit"),
                "mean_context_characters": mean(("e00", "context_character_count")),
                "mean_latency_ms": mean(("e00", "latency_ms")),
            },
            "e01": {
                "mean_answer_unique_token_coverage": mean(("e01", "answer_unique_token_coverage")),
                "explicit_reference_any_hit_rate": reference_rate("e01", "explicit_reference_any_hit"),
                "explicit_reference_all_hit_rate": reference_rate("e01", "explicit_reference_all_hit"),
                "mean_context_characters": mean(("e01", "context_character_count")),
                "mean_latency_ms": mean(("e01", "latency_ms")),
            },
            "delta_e01_minus_e00": {
                "mean_answer_unique_token_coverage": (
                    mean(("e01", "answer_unique_token_coverage"))
                    - mean(("e00", "answer_unique_token_coverage"))
                ),
                "explicit_reference_any_hit_rate": _optional_delta(
                    reference_rate("e01", "explicit_reference_any_hit"),
                    reference_rate("e00", "explicit_reference_any_hit"),
                ),
                "explicit_reference_all_hit_rate": _optional_delta(
                    reference_rate("e01", "explicit_reference_all_hit"),
                    reference_rate("e00", "explicit_reference_all_hit"),
                ),
            },
            "mean_top_k_jaccard": sum(record["top_k_jaccard"] for record in records) / len(records),
        },
        "warnings": [
            "Gold answers were used only after retrieval to compute diagnostics.",
            "These answer-derived metrics are not evidence labels, retrieval recall or precision.",
            "No E01 promotion decision may be made from these diagnostics alone; answer-level METEOR remains primary."
        ],
    }


def _run_identity(
    *,
    sample_ids: list[str],
    config: E01Config,
    e00_manifest: Path,
    e01_manifest: Path,
) -> dict[str, Any]:
    identity = {
        "config_sha256": config.config_sha256,
        "dev_sha256": config.dev_sha256,
        "e00_manifest_sha256": file_sha256(e00_manifest),
        "e01_manifest_sha256": file_sha256(e01_manifest),
        "sample_ids_sha256": hashlib.sha256("\n".join(sample_ids).encode("utf-8")).hexdigest(),
        "sample_size": len(sample_ids),
    }
    identity["identity_sha256"] = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return identity


def _load_checkpoint(
    results_path: Path,
    state_path: Path,
    identity: dict[str, Any],
    sample_ids: list[str],
) -> list[dict[str, Any]]:
    if results_path.exists() != state_path.exists():
        raise DiagnosticError("Diagnostic checkpoint files are incomplete.")
    if not results_path.exists():
        _write_state(state_path, identity, 0, complete=False)
        results_path.touch()
        return []
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if state.get("run_identity") != identity:
        raise DiagnosticError("Diagnostic checkpoint identity mismatch.")
    records: list[dict[str, Any]] = []
    with results_path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise DiagnosticError(f"Invalid checkpoint JSONL line {line_number}.") from exc
            records.append(record)
    completed_ids = [record.get("question_id") for record in records]
    if completed_ids != sample_ids[: len(completed_ids)] or len(records) > len(sample_ids):
        raise DiagnosticError("Diagnostic results are not an exact sample-order prefix.")
    if state.get("completed_count") > len(records):
        raise DiagnosticError("Diagnostic state is ahead of durable JSONL results.")
    _write_state(state_path, identity, len(records), complete=len(records) == len(sample_ids))
    return records


def _load_dev(path: Path) -> dict[str, dict[str, str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not payload:
        raise DiagnosticError("Dev split must be a non-empty mapping.")
    result: dict[str, dict[str, str]] = {}
    for question_id, record in payload.items():
        if (
            not isinstance(question_id, str)
            or not isinstance(record, dict)
            or set(record) != {"question", "answer"}
            or not isinstance(record["question"], str)
            or not record["question"].strip()
            or not isinstance(record["answer"], str)
        ):
            raise DiagnosticError(f"Invalid dev record: {question_id}")
        result[question_id] = record
    return result


def _write_state(
    path: Path,
    identity: dict[str, Any],
    completed_count: int,
    *,
    complete: bool,
) -> None:
    _atomic_write_json(
        path,
        {
            "schema_version": "1.0",
            "run_identity": identity,
            "completed_count": completed_count,
            "complete": complete,
        },
    )


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(path)


def _optional_delta(candidate: float | None, control: float | None) -> float | None:
    return candidate - control if candidate is not None and control is not None else None
