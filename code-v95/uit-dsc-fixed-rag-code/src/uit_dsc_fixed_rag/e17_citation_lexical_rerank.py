"""E17 deterministic citation/lexical context-order A/B on fresh dev-200."""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean
from typing import Any

from uit_dsc_fixed_rag.corpus import file_sha256
from uit_dsc_fixed_rag.e02_answer import _atomic_json, _atomic_jsonl, pack_contexts
from uit_dsc_fixed_rag.e03_rrf_grid import (
    _bootstrap_ci,
    _json_sha256,
    _load_worker_progress,
    _read_jsonl,
    _write_worker_state,
)
from uit_dsc_fixed_rag.e08b_context_lora import (
    inference_packing,
    load_context_lora_generator,
    load_e08b_config,
)
from uit_dsc_fixed_rag.e10_repetition_grid import answer_diagnostics
from uit_dsc_fixed_rag.evaluation.official import (
    ensure_nltk_resources,
    nltk_meteor_score,
    rouge_l_fmeasure,
)
from uit_dsc_fixed_rag.retrieval_diagnostics import select_dev_sample
from uit_dsc_fixed_rag.sparse_normalization import fold_vietnamese_accents


CODE_VERSION = "0.48.0"
LOGGER = logging.getLogger(__name__)
_WORD = re.compile(r"[a-z0-9]+")
_DOCUMENT_REF = re.compile(r"\b[0-9]{1,4}/[0-9]{4}/[a-z0-9]+(?:-[a-z0-9]+)*\b")
_ARTICLE_REF = re.compile(r"\bdieu\s+([0-9]+[a-z]?)\b")
_CLAUSE_REF = re.compile(r"\bkhoan\s+([0-9]+[a-z]?)\b")
_STOPWORDS = frozenset({
    "ai", "bao", "bi", "cac", "can", "co", "cua", "duoc", "gi", "hay",
    "hien", "khi", "la", "lam", "mot", "nao", "nhung", "phai", "quy",
    "ra", "sau", "the", "thi", "theo", "trong", "tren", "truoc", "tu",
    "va", "ve", "voi",
})


class E17Error(RuntimeError):
    """Raised when E17 evidence or checkpoint identity changes."""


@dataclass(frozen=True)
class E17Config:
    raw: dict[str, Any]
    path: Path

    @property
    def config_sha256(self) -> str:
        return file_sha256(self.path)

    def section(self, key: str) -> dict[str, Any]:
        value = self.raw.get(key)
        if not isinstance(value, dict):
            raise ValueError(f"E17 section must be an object: {key}")
        return value


def load_config(path: Path) -> E17Config:
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "schema_version", "experiment_id", "source_e08a", "source_training",
        "dev", "rerank", "inference", "parameter_budget", "execution",
        "scoring", "run_contract",
    }
    if (
        set(payload) != required
        or payload.get("schema_version") != "1.0"
        or payload.get("experiment_id")
        != "E17-citation-lexical-rerank-fresh-dev200-v1"
    ):
        raise ValueError("E17 config root is incompatible.")
    config = E17Config(payload, path)
    source = config.section("source_e08a")
    if (
        source.get("experiment_id")
        != "E08A-context-retrieval-train5636-dev521-v1"
        or source.get("path") != "retrieval/dev521/results.jsonl"
        or source.get("record_count") != 521
        or source.get("contexts_per_question") != 12
    ):
        raise ValueError("E17 E08A source changed.")
    training = config.section("source_training")
    if (
        training.get("experiment_id")
        != "E08B-context-aware-lora-train5636-dev521-v3"
        or training.get("train_sample_size") != 5636
        or training.get("rank") != 8
        or training.get("alpha") != 16
        or training.get("fresh_from_base") is not True
    ):
        raise ValueError("E17 E08B adapter source changed.")
    dev = config.section("dev")
    if (
        dev.get("full_size") != 721
        or dev.get("excluded_tuned_prefix_size") != 200
        or dev.get("dev521_size") != 521
        or dev.get("evaluation_offset_within_dev521") != 0
        or dev.get("sample_size") != 200
        or dev.get("answer_usage") != "final-scoring-only"
    ):
        raise ValueError("E17 fresh dev sample changed.")
    rerank = config.section("rerank")
    expected_rerank = {
        "control_key": "ranked_top12_control",
        "candidate_key": "citation_lexical_top12",
        "candidate_limit": 12,
        "lexical_window": 6,
        "minimum_token_characters": 3,
        "original_rank_weight": 1.5,
        "token_overlap_weight": 1.0,
        "bigram_overlap_weight": 4.0,
        "document_reference_bonus": 1000.0,
        "article_reference_bonus": 500.0,
        "clause_reference_bonus": 100.0,
        "stable_original_rank_tiebreak": True,
    }
    if rerank != expected_rerank:
        raise ValueError("E17 rerank policy changed.")
    inference = config.section("inference")
    if inference != {
        "max_input_tokens": 8192,
        "minimum_contexts": 1,
        "context_limit": 12,
        "max_new_tokens": 704,
        "do_sample": False,
        "num_beams": 1,
        "enable_thinking": False,
        "use_cache": True,
        "repetition_penalty": 1.0,
        "no_repeat_ngram_size": 0,
    }:
        raise ValueError("E17 inference contract changed.")
    if config.section("execution") != {
        "workers": 2,
        "partition": "one-retrieval-order-variant-per-worker-all-fresh-dev200-ids",
        "questions_per_worker": 200,
        "checkpoint_after_questions": 1,
    }:
        raise ValueError("E17 execution contract changed.")
    budget = config.section("parameter_budget")
    if (
        budget["maximum_stack_total"]
        != budget["embedding"] + budget["generator"] + budget["adapter_parameter_cap"]
        or budget["maximum_stack_total"] >= budget["exclusive_limit"]
    ):
        raise ValueError("E17 parameter budget failed.")
    if config.section("scoring").get("promotion_allowed") is not False:
        raise ValueError("E17 cannot auto-promote.")
    contract = config.section("run_contract")
    if not contract or not all(value is True for value in contract.values()):
        raise ValueError("E17 run contract lost an invariant.")
    return config


def _ids_sha(ids: list[str]) -> str:
    return hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()


def _load_dev(path: Path, config: E17Config) -> tuple[dict[str, Any], list[str]]:
    section = config.section("dev")
    if not path.is_file() or file_sha256(path) != section["sha256"]:
        raise E17Error("Pinned dev split changed.")
    records = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(records, dict) or len(records) != section["full_size"]:
        raise E17Error("Pinned dev record count changed.")
    full_ids = select_dev_sample(
        records, seed=section["sample_seed"], size=section["full_size"]
    )
    start = section["excluded_tuned_prefix_size"] + section[
        "evaluation_offset_within_dev521"
    ]
    ids = full_ids[start : start + section["sample_size"]]
    if len(ids) != section["sample_size"] or _ids_sha(ids) != section[
        "sample_ids_sha256"
    ]:
        raise E17Error("Fresh E17 dev-200 identity changed.")
    return records, ids


def _content_tokens(text: str, minimum: int) -> list[str]:
    folded = fold_vietnamese_accents(text)
    return [
        token for token in _WORD.findall(folded)
        if len(token) >= minimum and token not in _STOPWORDS
    ]


def context_relevance(
    question: str, context: dict[str, Any], original_rank: int,
    config: E17Config,
) -> dict[str, Any]:
    policy = config.section("rerank")
    query_folded = fold_vietnamese_accents(question)
    context_folded = fold_vietnamese_accents(
        " ".join(str(context.get(key) or "") for key in ("document_id", "article_number", "text"))
    )
    documents = sorted(set(_DOCUMENT_REF.findall(query_folded)))
    articles = sorted(set(_ARTICLE_REF.findall(query_folded)))
    clauses = sorted(set(_CLAUSE_REF.findall(query_folded)))
    document_hits = sum(reference in context_folded for reference in documents)
    article_hits = sum(
        bool(re.search(rf"\bdieu\s+{re.escape(reference)}\b", context_folded))
        or fold_vietnamese_accents(str(context.get("article_number") or "")) == reference
        for reference in articles
    )
    clause_hits = sum(
        bool(re.search(rf"\bkhoan\s+{re.escape(reference)}\b", context_folded))
        for reference in clauses
    )
    query_tokens = _content_tokens(question, policy["minimum_token_characters"])
    unique_query = list(dict.fromkeys(query_tokens))
    context_token_set = set(_WORD.findall(context_folded))
    token_hits = sum(token in context_token_set for token in unique_query)
    bigrams = list(dict.fromkeys(zip(query_tokens, query_tokens[1:])))
    bigram_hits = sum(f"{left} {right}" in context_folded for left, right in bigrams)
    score = (
        document_hits * policy["document_reference_bonus"]
        + article_hits * policy["article_reference_bonus"]
        + clause_hits * policy["clause_reference_bonus"]
        + token_hits * policy["token_overlap_weight"]
        + bigram_hits * policy["bigram_overlap_weight"]
        + (policy["candidate_limit"] - original_rank)
        * policy["original_rank_weight"]
    )
    return {
        "score": float(score),
        "document_hits": document_hits,
        "article_hits": article_hits,
        "clause_hits": clause_hits,
        "token_hits": token_hits,
        "bigram_hits": bigram_hits,
        "has_explicit_reference": bool(documents or articles or clauses),
    }


def citation_lexical_order(
    question: str, contexts: list[dict[str, Any]], config: E17Config,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    policy = config.section("rerank")
    if len(contexts) != policy["candidate_limit"]:
        raise E17Error("E17 requires exactly twelve source contexts.")
    diagnostics = [
        context_relevance(question, context, rank, config)
        for rank, context in enumerate(contexts)
    ]
    explicit = any(item["has_explicit_reference"] for item in diagnostics)
    window = len(contexts) if explicit else policy["lexical_window"]
    ranked = sorted(
        range(window), key=lambda rank: (-diagnostics[rank]["score"], rank)
    )
    order = [*ranked, *range(window, len(contexts))]
    return [contexts[index] for index in order], [
        {"source_rank": index + 1, **diagnostics[index]} for index in order
    ]


def _source_contexts(
    e08a_directory: Path, dev_path: Path, config: E17Config,
) -> tuple[dict[str, Any], list[str], list[dict[str, Any]]]:
    dev, ids = _load_dev(dev_path, config)
    source = config.section("source_e08a")
    path = e08a_directory / source["path"]
    if not path.is_file() or file_sha256(path) != source["sha256"]:
        raise E17Error("Pinned E08A dev-521 contexts changed.")
    all_rows = _read_jsonl(path)
    if len(all_rows) != source["record_count"]:
        raise E17Error("E08A dev-521 context count changed.")
    offset = config.section("dev")["evaluation_offset_within_dev521"]
    rows = all_rows[offset : offset + len(ids)]
    for index, (question_id, row) in enumerate(zip(ids, rows)):
        contexts = row.get("contexts")
        if (
            row.get("question_id") != question_id
            or row.get("sample_index") != offset + index
            or row.get("target") != "dev521"
            or row.get("answer_included") is not False
            or "answer" in row
            or not isinstance(contexts, list)
            or len(contexts) != source["contexts_per_question"]
        ):
            raise E17Error(f"Invalid E08A context row: {index}")
    return dev, ids, rows


def _training_config(project_root: Path, config: E17Config):
    section = config.section("source_training")
    path = project_root / section["config_path"]
    if not path.is_file() or file_sha256(path) != section["config_sha256"]:
        raise E17Error("Pinned E08B training config changed.")
    return load_e08b_config(path)


def _adapter_hash(training_directory: Path, config: E17Config) -> str:
    section = config.section("source_training")
    adapter = training_directory / section["adapter_path"]
    complete_path = training_directory / "adapter-final" / "complete.json"
    if not adapter.is_file() or not complete_path.is_file():
        raise E17Error("Completed E08B adapter is missing.")
    observed = file_sha256(adapter)
    complete = json.loads(complete_path.read_text(encoding="utf-8"))
    if (
        observed != section["adapter_sha256"]
        or complete.get("experiment_id") != section["experiment_id"]
        or complete.get("config_sha256") != section["config_sha256"]
        or complete.get("adapter_sha256") != observed
        or complete.get("fresh_from_base") is not True
        or complete.get("e07_adapter_loaded") is not False
    ):
        raise E17Error("Completed E08B adapter evidence changed.")
    return observed


def validate_preflight(
    *, project_root: Path, e08a_directory: Path, training_directory: Path,
    dev_path: Path, config: E17Config,
) -> dict[str, Any]:
    scorer = project_root / config.section("scoring")["official_scorer_path"]
    if not scorer.is_file() or file_sha256(scorer) != config.section("scoring")[
        "official_scorer_sha256"
    ]:
        raise E17Error("Pinned official scorer changed.")
    report_path = e08a_directory / "report.json"
    if not report_path.is_file() or file_sha256(report_path) != config.section(
        "source_e08a"
    )["report_sha256"]:
        raise E17Error("Pinned E08A report changed.")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if (
        report.get("experiment_id") != config.section("source_e08a")["experiment_id"]
        or report.get("dev_answers_used_by_retrieval") is not False
        or report.get("holdout_untouched") is not True
        or report.get("public_read") is not False
    ):
        raise E17Error("E08A leakage contract changed.")
    training = _training_config(project_root, config)
    adapter_hash = _adapter_hash(training_directory, config)
    _, ids, rows = _source_contexts(e08a_directory, dev_path, config)
    return {
        "code_version": CODE_VERSION,
        "config_sha256": config.config_sha256,
        "training_config_sha256": training.config_sha256,
        "adapter_sha256": adapter_hash,
        "source_contexts_sha256": config.section("source_e08a")["sha256"],
        "sample_ids_sha256": _ids_sha(ids),
        "sample_size": len(ids),
        "source_context_rows": len(rows),
        "variants": [
            config.section("rerank")["control_key"],
            config.section("rerank")["candidate_key"],
        ],
        "max_new_tokens": 704,
    }


def prepare_variants(
    *, e08a_directory: Path, dev_path: Path, output_directory: Path,
    config: E17Config,
) -> dict[str, Any]:
    dev, ids, source_rows = _source_contexts(e08a_directory, dev_path, config)
    rows = []
    changed = explicit = top1_changed = 0
    for index, (question_id, source_row) in enumerate(zip(ids, source_rows)):
        control = source_row["contexts"]
        candidate, diagnostics = citation_lexical_order(
            dev[question_id]["question"], control, config
        )
        control_ids = [item["chunk_id"] for item in control]
        candidate_ids = [item["chunk_id"] for item in candidate]
        changed += candidate_ids != control_ids
        top1_changed += candidate_ids[0] != control_ids[0]
        explicit += any(item["has_explicit_reference"] for item in diagnostics)
        rows.append({
            "question_id": question_id,
            "sample_index": index,
            "answer_included": False,
            "variants": {
                config.section("rerank")["control_key"]: {"contexts": control},
                config.section("rerank")["candidate_key"]: {
                    "contexts": candidate,
                    "ranking_diagnostics": diagnostics,
                },
            },
        })
    root = output_directory / "prepared"
    root.mkdir(parents=True, exist_ok=True)
    path = root / "results.jsonl"
    _atomic_jsonl(path, rows)
    summary = {
        "schema_version": "1.0",
        "sample_size": len(rows),
        "sample_ids_sha256": _ids_sha(ids),
        "source_contexts_sha256": config.section("source_e08a")["sha256"],
        "results_sha256": file_sha256(path),
        "reordered_question_count": changed,
        "reordered_question_rate": changed / len(rows),
        "top1_changed_count": top1_changed,
        "explicit_reference_question_count": explicit,
        "answers_used": False,
    }
    _atomic_json(root / "summary.json", summary)
    return summary


def variant_for_worker(config: E17Config, worker_rank: int) -> str:
    keys = [
        config.section("rerank")["control_key"],
        config.section("rerank")["candidate_key"],
    ]
    if worker_rank not in range(len(keys)):
        raise E17Error(f"Invalid E17 worker rank: {worker_rank}")
    return keys[worker_rank]


def load_generator(
    *, project_root: Path, training_directory: Path, config: E17Config,
    device: str,
) -> tuple[Any, Any, dict[str, Any], int]:
    training = _training_config(project_root, config)
    try:
        return load_context_lora_generator(
            config=training, training_directory=training_directory, device=device
        )
    except Exception as exc:
        raise E17Error(str(exc)) from exc


def run_variant_worker(
    *, worker_rank: int, device: str, model: Any, tokenizer: Any,
    project_root: Path, training_directory: Path, dev_path: Path,
    output_directory: Path, config: E17Config, device_map: dict[str, Any],
    adapter_parameters: int,
) -> dict[str, Any]:
    import torch

    variant = variant_for_worker(config, worker_rank)
    if device != f"cuda:{worker_rank}":
        raise E17Error("E17 worker/device mapping changed.")
    training = _training_config(project_root, config)
    dev, ids = _load_dev(dev_path, config)
    prepared_path = output_directory / "prepared" / "results.jsonl"
    prepared = _read_jsonl(prepared_path)
    if len(prepared) != len(ids):
        raise E17Error("Run complete E17 preparation first.")
    adapter_hash = _adapter_hash(training_directory, config)
    assigned = list(range(len(ids)))
    identity = {
        "code_version": CODE_VERSION,
        "stage": "e17-fresh-dev200-retrieval-order-worker",
        "config_sha256": config.config_sha256,
        "prepared_results_sha256": file_sha256(prepared_path),
        "adapter_sha256": adapter_hash,
        "adapter_parameters": adapter_parameters,
        "variant": variant,
        "worker_rank": worker_rank,
        "device": device,
        "device_map": device_map,
        "sample_ids_sha256": _ids_sha(ids),
    }
    identity["identity_sha256"] = _json_sha256(identity)
    root = output_directory / "evaluation" / variant
    records = root / "records"
    records.mkdir(parents=True, exist_ok=True)
    state_path = root / "state.json"
    completed = _load_worker_progress(
        records=records, state_path=state_path, identity=identity,
        assigned_indices=assigned, sample_ids=ids,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    def token_count(messages: list[dict[str, Any]]) -> int:
        rendered = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
        return len(tokenizer(rendered, add_special_tokens=False)["input_ids"])

    for position in range(completed, len(assigned)):
        index = assigned[position]
        question_id = ids[index]
        row = prepared[index]
        if row.get("question_id") != question_id or row.get("sample_index") != index:
            raise E17Error(f"Prepared E17 row changed: {index}")
        candidates = row["variants"][variant]["contexts"]
        selected, messages, input_tokens = pack_contexts(
            question=dev[question_id]["question"], contexts=candidates,
            config=inference_packing(training), token_counter=token_count,
        )
        rendered = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = tokenizer(rendered, add_special_tokens=False, return_tensors="pt")
        inputs = {key: value.to(device) for key, value in inputs.items()}
        started = time.perf_counter()
        with torch.inference_mode():
            generated = model.generate(
                **inputs, do_sample=False, num_beams=1, max_new_tokens=704,
                use_cache=True, repetition_penalty=1.0, no_repeat_ngram_size=0,
            )
        latency_ms = (time.perf_counter() - started) * 1000
        new_ids = generated[0, inputs["input_ids"].shape[1]:]
        answer = tokenizer.decode(
            new_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        ).strip()
        if not answer:
            raise E17Error(f"Empty E17 answer: {question_id}")
        eos = tokenizer.eos_token_id
        eos_ids = set(eos if isinstance(eos, list) else [eos])
        generated_count = int(new_ids.shape[0])
        finish_reason = (
            "eos" if generated_count and int(new_ids[-1]) in eos_ids
            else "length" if generated_count >= 704
            else "other"
        )
        _atomic_json(records / f"{index:04d}.json", {
            "question_id": question_id,
            "sample_index": index,
            "worker_rank": worker_rank,
            "worker_identity_sha256": identity["identity_sha256"],
            "variant": variant,
            "answer": answer,
            "max_new_tokens": 704,
            "candidate_chunk_ids": [item["chunk_id"] for item in candidates],
            "selected_chunk_ids": [item["chunk_id"] for item in selected],
            "selected_context_count": len(selected),
            "input_tokens": input_tokens,
            "output_tokens": len(tokenizer(answer, add_special_tokens=False)["input_ids"]),
            "generated_tokens_including_special": generated_count,
            "finish_reason": finish_reason,
            "generation_latency_ms": latency_ms,
        })
        _write_worker_state(state_path, identity, position + 1, len(assigned))
        LOGGER.info(
            "e17_generation_progress variant=%s device=%s completed=%d total=%d "
            "question_id=%s finish=%s", variant, device, position + 1,
            len(assigned), question_id, finish_reason,
        )
    _write_worker_state(state_path, identity, len(assigned), len(assigned))
    return {"variant": variant, "device": device, "completed": len(assigned)}


def _metrics(rows: list[dict[str, Any]], scores: list[dict[str, float]]) -> dict[str, Any]:
    diagnostics = [answer_diagnostics(row["answer"]) for row in rows]
    return {
        "meteor": fmean(item["meteor"] for item in scores),
        "rouge_l": fmean(item["rouge_l"] for item in scores),
        "mean_output_tokens": fmean(row["output_tokens"] for row in rows),
        "mean_answer_characters": fmean(len(row["answer"]) for row in rows),
        "mean_selected_contexts": fmean(row["selected_context_count"] for row in rows),
        "length_finish_rate": fmean(row["finish_reason"] == "length" for row in rows),
        "duplicate_line_rate": fmean(item["duplicate_line"] for item in diagnostics),
        "duplicate_sentence_rate": fmean(item["duplicate_sentence"] for item in diagnostics),
        "non_sentence_ending_rate": fmean(item["non_sentence_ending"] for item in diagnostics),
        "mean_generation_latency_ms": fmean(row["generation_latency_ms"] for row in rows),
    }


def finalize_score(
    *, training_directory: Path, dev_path: Path, output_directory: Path,
    config: E17Config,
) -> dict[str, Any]:
    ensure_nltk_resources(download=False)
    dev, ids = _load_dev(dev_path, config)
    prepared_path = output_directory / "prepared" / "results.jsonl"
    prepared_summary_path = output_directory / "prepared" / "summary.json"
    if not prepared_path.is_file() or not prepared_summary_path.is_file():
        raise E17Error("Missing prepared E17 retrieval-order artifact.")
    prepared_rows = _read_jsonl(prepared_path)
    prepared_summary = json.loads(prepared_summary_path.read_text(encoding="utf-8"))
    if (
        len(prepared_rows) != len(ids)
        or prepared_summary.get("sample_ids_sha256") != _ids_sha(ids)
        or prepared_summary.get("results_sha256") != file_sha256(prepared_path)
        or prepared_summary.get("answers_used") is not False
    ):
        raise E17Error("Prepared E17 retrieval-order identity changed.")
    keys = [
        config.section("rerank")["control_key"],
        config.section("rerank")["candidate_key"],
    ]
    rows_by_key: dict[str, list[dict[str, Any]]] = {}
    for worker_rank, key in enumerate(keys):
        root = output_directory / "evaluation" / key
        state_path = root / "state.json"
        if not state_path.is_file():
            raise E17Error(f"Missing E17 worker state: {key}")
        state = json.loads(state_path.read_text(encoding="utf-8"))
        identity = state.get("run_identity", {})
        if (
            state.get("complete") is not True
            or state.get("completed_count") != len(ids)
            or state.get("assigned_count") != len(ids)
            or identity.get("config_sha256") != config.config_sha256
            or identity.get("variant") != key
            or identity.get("worker_rank") != worker_rank
        ):
            raise E17Error(f"Incomplete E17 worker: {key}")
        rows = []
        for index, question_id in enumerate(ids):
            path = root / "records" / f"{index:04d}.json"
            if not path.is_file():
                raise E17Error(f"Missing E17 answer: {key}:{index}")
            row = json.loads(path.read_text(encoding="utf-8"))
            if (
                row.get("question_id") != question_id
                or row.get("sample_index") != index
                or row.get("variant") != key
                or row.get("worker_rank") != worker_rank
                or row.get("max_new_tokens") != 704
                or row.get("finish_reason") not in {"eos", "length", "other"}
                or not isinstance(row.get("answer"), str)
                or not row["answer"].strip()
                or row.get("candidate_chunk_ids")
                != [
                    item["chunk_id"]
                    for item in prepared_rows[index]["variants"][key]["contexts"]
                ]
            ):
                raise E17Error(f"Invalid E17 answer: {key}:{index}")
            rows.append(row)
        rows_by_key[key] = rows
        _atomic_jsonl(root / "results.jsonl", rows)
    scores: dict[str, list[dict[str, float]]] = {key: [] for key in keys}
    merged = []
    for index, question_id in enumerate(ids):
        variants = {key: rows_by_key[key][index] for key in keys}
        merged.append({"question_id": question_id, "sample_index": index, "variants": variants})
        reference = dev[question_id]["answer"]
        for key in keys:
            scores[key].append({
                "meteor": nltk_meteor_score(reference, variants[key]["answer"]),
                "rouge_l": rouge_l_fmeasure(reference, variants[key]["answer"]),
            })
    results_path = output_directory / "results.jsonl"
    _atomic_jsonl(results_path, merged)
    metrics = {key: _metrics(rows_by_key[key], scores[key]) for key in keys}
    control, candidate = keys
    meteor_delta = [
        a["meteor"] - b["meteor"] for a, b in zip(scores[candidate], scores[control])
    ]
    rouge_delta = [
        a["rouge_l"] - b["rouge_l"] for a, b in zip(scores[candidate], scores[control])
    ]
    scoring = config.section("scoring")
    paired = {
        "meteor_mean": fmean(meteor_delta),
        "meteor_bootstrap_95_ci": _bootstrap_ci(
            meteor_delta, seed=f"{scoring['bootstrap_seed']}:meteor",
            iterations=scoring["bootstrap_iterations"],
        ),
        "rouge_l_mean": fmean(rouge_delta),
        "rouge_l_bootstrap_95_ci": _bootstrap_ci(
            rouge_delta, seed=f"{scoring['bootstrap_seed']}:rouge_l",
            iterations=scoring["bootstrap_iterations"],
        ),
    }
    report = {
        "schema_version": "1.0",
        "experiment_id": config.raw["experiment_id"],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "sample_size": len(ids),
        "sample_scope": "first-200-of-dev521-after-excluding-prior-tuned-dev200",
        "control_variant": control,
        "candidate_variant": candidate,
        "only_changed_factor": "deterministic ordering of the same twelve contexts",
        "retrieval_order_diagnostics": prepared_summary,
        "metrics": metrics,
        "paired_delta_candidate_minus_control": paired,
        "smoke_leader": max(keys, key=lambda key: (metrics[key]["meteor"], metrics[key]["rouge_l"])),
        "promotion_allowed": False,
        "evidence": {
            "config_sha256": config.config_sha256,
            "source_contexts_sha256": config.section("source_e08a")["sha256"],
            "prepared_results_sha256": prepared_summary["results_sha256"],
            "adapter_sha256": _adapter_hash(training_directory, config),
            "sample_ids_sha256": _ids_sha(ids),
            "results_sha256": file_sha256(results_path),
        },
        "warning": (
            "E17 is a paired fresh-dev200 retrieval-order experiment. It does not "
            "read public or holdout and cannot automatically promote a stack."
        ),
    }
    _atomic_json(output_directory / "report.json", report)
    return report


__all__ = [
    "E17Config", "E17Error", "citation_lexical_order", "context_relevance",
    "finalize_score", "load_config", "load_generator", "prepare_variants",
    "run_variant_worker", "validate_preflight", "variant_for_worker",
]
