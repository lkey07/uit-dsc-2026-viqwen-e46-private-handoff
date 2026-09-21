"""E20 high-precision source-aware expansion from frozen E08A top-20."""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean
from typing import Any

from .corpus import file_sha256
from .e02_answer import _atomic_json, _atomic_jsonl, pack_contexts
from .e03_rrf_grid import (
    _bootstrap_ci,
    _json_sha256,
    _load_worker_progress,
    _read_jsonl,
    _write_worker_state,
)
from .e08b_context_lora import _validate_context_rows, inference_packing
from .e18_source_metadata import enrich_context, scan_selected, source_number
from .e19_metadata_lora import (
    _metrics,
    eval_sample,
    load_candidate_generator,
    load_config as load_e19_config,
    validate_candidate_adapter,
    validate_training_preflight,
)
from .evaluation.official import (
    ensure_nltk_resources,
    nltk_meteor_score,
    rouge_l_fmeasure,
)
from .sparse_normalization import fold_vietnamese_accents


EXPERIMENT = "E20-source-aware-top20-expansion-dev200-v1"
CONTROL = "e19_metadata_trained_rank8_max704"
CANDIDATE = "e20_source_aware_top20_to_top12"
CODE_VERSION = "0.53.0"
LOGGER = logging.getLogger(__name__)

_WORD = re.compile(r"[a-z0-9]+")
_DOCUMENT_NUMBER = re.compile(
    r"\b[0-9]{1,4}\s*/\s*[0-9]{4}\s*/\s*[a-z0-9]+(?:\s*-\s*[a-z0-9]+)*\b"
)
_TITLE_STOPWORDS = frozenset(
    {
        "ban", "bo", "boi", "cac", "chi", "chi-tiet", "cho", "co", "cua",
        "duoc", "hanh", "huong", "mot", "nay", "nhung", "quy", "quy-dinh",
        "so", "sua", "thi", "thuc", "ve", "va", "viec",
    }
)
_LEGAL_TYPE_TOKENS = frozenset(
    {"bo", "chi", "dinh", "luat", "nghi", "phap", "quyet", "thong", "tu"}
)


class E20Error(RuntimeError):
    """Raised when an E20 source, decision or checkpoint changes."""


@dataclass(frozen=True)
class E20Config:
    raw: dict[str, Any]
    path: Path
    source: Any

    @property
    def config_sha256(self) -> str:
        return file_sha256(self.path)

    def section(self, key: str) -> dict[str, Any]:
        value = self.raw.get(key)
        if not isinstance(value, dict):
            raise E20Error(f"E20 section must be an object: {key}")
        return value


def load_config(project_root: Path, path: Path) -> E20Config:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if set(raw) != {
        "schema_version", "experiment_id", "source_e19_config_path",
        "source_e19_config_sha256", "source_control", "resolver", "evaluation",
        "parameter_budget", "scoring", "run_contract",
    }:
        raise E20Error("E20 config root changed.")
    if raw.get("schema_version") != "1.0" or raw.get("experiment_id") != EXPERIMENT:
        raise E20Error("E20 experiment identity changed.")
    source_path = project_root / raw["source_e19_config_path"]
    if (
        raw["source_e19_config_path"]
        != "configs/e19-metadata-aware-lora-train5636-eval200-v1.json"
        or file_sha256(source_path) != raw["source_e19_config_sha256"]
    ):
        raise E20Error("Pinned E19 contract changed.")
    source = load_e19_config(project_root, source_path)
    control = raw["source_control"]
    if control != {
        "experiment_id": source.raw["experiment_id"],
        "variant": CONTROL,
        "results_path": f"evaluation/{CONTROL}/results.jsonl",
        "results_sha256": "a79e0ba7afadb6d76a2ed60884551145de1e102ffe2b0bdf7a9f9b7178ce0a9c",
        "adapter_sha256": "bfb9d8120337c4013b5dcb0b59c6d03beccfe9f9748e5a01ef62e1de64fb5edd",
        "sample_ids_sha256": "dc17d6c9af9e941868c03cfc61fd17cdd1dc55aa5834beb3e75ba0fde32a5c73",
        "sample_size": 200,
        "max_new_tokens": 704,
        "context_count": 12,
    }:
        raise E20Error("E20 control contract changed.")
    resolver = raw["resolver"]
    if resolver != {
        "candidate_pool_size": 20,
        "output_context_count": 12,
        "tail_start_rank": 13,
        "maximum_tail_promotions": 2,
        "document_number_match": "exact-after-accent-fold-casefold-and-whitespace-removal",
        "source_title_match": "unique-best-legal-title-token-coverage-v1",
        "minimum_title_token_matches": 2,
        "minimum_title_token_coverage": 0.5,
        "require_legal_type_token_overlap": True,
        "require_unique_best_document": True,
        "require_matching_tail_candidate": True,
        "ambiguous_or_unmatched_policy": "preserve-original-top12-byte-order",
        "insertion_policy": "after-last-existing-source-match-then-preserve-fused-order",
        "answers_used": False,
    }:
        raise E20Error("E20 resolver policy changed.")
    evaluation = raw["evaluation"]
    if evaluation != {
        "dev521_offset": 200,
        "sample_size": 200,
        "sample_ids_sha256": control["sample_ids_sha256"],
        "control_variant": CONTROL,
        "candidate_variant": CANDIDATE,
        "max_input_tokens": 8192,
        "max_new_tokens": 704,
        "do_sample": False,
        "num_beams": 1,
        "workers": 2,
        "questions_per_worker": 100,
        "worker_partition": "contiguous-100-100",
        "promotion_allowed": False,
    }:
        raise E20Error("E20 evaluation contract changed.")
    budget = raw["parameter_budget"]
    if (
        budget["maximum_stack_total"]
        != budget["embedding"] + budget["generator"] + budget["adapter_parameter_cap"]
        or budget["maximum_stack_total"] >= budget["exclusive_limit"]
    ):
        raise E20Error("E20 parameter budget failed.")
    if raw["scoring"] != {
        "primary_metric": "meteor",
        "secondary_metric": "rouge_l",
        "official_scorer_path": "src/uit_dsc_fixed_rag/evaluation/official.py",
        "official_scorer_sha256": "5ad785fb19d0b4b02bad1030c8360e42c5c409d5090905d233b6787905c6d689",
        "bootstrap_seed": "uit-dsc-2026-e20-source-aware-top20-v1",
        "bootstrap_iterations": 10000,
        "promotion_allowed": False,
    }:
        raise E20Error("E20 scoring contract changed.")
    if not raw["run_contract"] or not all(raw["run_contract"].values()):
        raise E20Error("E20 run contract lost an invariant.")
    return E20Config(raw=raw, path=path, source=source)


def code_sha(project_root: Path) -> str:
    paths = sorted((project_root / "src/uit_dsc_fixed_rag").rglob("*.py"))
    paths.append(project_root / "scripts/run_e20_source_aware_expansion_kaggle.py")
    return _json_sha256(
        {
            path.relative_to(project_root).as_posix(): file_sha256(path)
            for path in paths
            if path.is_file()
        }
    )


def _normalize(text: str) -> str:
    return " ".join(
        _WORD.findall(
            fold_vietnamese_accents(unicodedata.normalize("NFC", str(text))).casefold()
        )
    )


def _normalize_number(text: str) -> str:
    return re.sub(r"\s+", "", _normalize(text)).strip(".-")


def document_number_references(question: str) -> list[str]:
    folded = fold_vietnamese_accents(unicodedata.normalize("NFC", question)).casefold()
    return list(dict.fromkeys(_normalize_number(match.group(0)) for match in _DOCUMENT_NUMBER.finditer(folded)))


def _title_tokens(title: str | None) -> tuple[str, ...]:
    if not title:
        return ()
    return tuple(
        dict.fromkeys(
            token
            for token in _normalize(title).split()
            if len(token) >= 2 and token not in _TITLE_STOPWORDS
        )
    )


def title_match_key(question: str, title: str | None, config: E20Config) -> tuple[int, int, int, int] | None:
    """Return an integer-only match key, or None when a title cue is weak."""

    tokens = _title_tokens(title)
    if not tokens:
        return None
    question_norm = _normalize(question)
    question_tokens = set(question_norm.split())
    overlap = [token for token in tokens if token in question_tokens]
    policy = config.section("resolver")
    if len(overlap) < policy["minimum_title_token_matches"]:
        return None
    if len(overlap) / len(tokens) < policy["minimum_title_token_coverage"]:
        return None
    if policy["require_legal_type_token_overlap"] and not (
        set(overlap) & _LEGAL_TYPE_TOKENS
    ):
        return None
    title_norm = _normalize(title or "")
    phrase = int(bool(title_norm) and title_norm in question_norm)
    # Integer cross-products make equality and tie handling stable across Python builds.
    return phrase, len(overlap), len(overlap) * 10_000 // len(tokens), len(tokens)


def _document_groups(
    pool: list[dict[str, Any]], documents: dict[str, dict[str, Any]]
) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = {}
    for rank, context in enumerate(pool):
        document_id = context["document_id"]
        if document_id not in documents:
            raise E20Error(f"Missing source document: {document_id}")
        groups.setdefault(document_id, []).append(rank)
    return groups


def resolve_document(
    question: str,
    pool: list[dict[str, Any]],
    documents: dict[str, dict[str, Any]],
    config: E20Config,
) -> tuple[str | None, dict[str, Any]]:
    """Resolve one unambiguous source among the frozen top-20 candidates."""

    groups = _document_groups(pool, documents)
    references = document_number_references(question)
    number_matches = []
    for document_id in groups:
        number = source_number(documents[document_id]["cleaned_text"])
        if number and _normalize_number(number) in references:
            number_matches.append(document_id)
    if len(number_matches) == 1:
        return number_matches[0], {
            "resolution": "document_number",
            "document_number_references": references,
            "eligible_document_count": 1,
        }
    if len(number_matches) > 1:
        return None, {
            "resolution": "ambiguous_document_number",
            "document_number_references": references,
            "eligible_document_count": len(number_matches),
        }

    scored: dict[str, tuple[int, int, int, int]] = {}
    for document_id in groups:
        key = title_match_key(question, documents[document_id].get("source_title"), config)
        if key is not None:
            scored[document_id] = key
    if not scored:
        return None, {
            "resolution": "no_high_precision_source_cue",
            "document_number_references": references,
            "eligible_document_count": 0,
        }
    best = max(scored.values())
    winners = sorted(document_id for document_id, key in scored.items() if key == best)
    if len(winners) != 1:
        return None, {
            "resolution": "ambiguous_source_title",
            "document_number_references": references,
            "eligible_document_count": len(scored),
            "best_tie_count": len(winners),
            "best_title_match_key": list(best),
        }
    return winners[0], {
        "resolution": "source_title",
        "document_number_references": references,
        "eligible_document_count": len(scored),
        "best_title_match_key": list(best),
    }


def source_aware_top12(
    question: str,
    pool: list[dict[str, Any]],
    documents: dict[str, dict[str, Any]],
    config: E20Config,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Replace at most two weak tail slots with same-source hits from ranks 13-20."""

    policy = config.section("resolver")
    if len(pool) != policy["candidate_pool_size"]:
        raise E20Error("E20 requires the exact frozen E08A top-20 pool.")
    ids = [context.get("chunk_id") for context in pool]
    if len(set(ids)) != len(ids) or any(not value for value in ids):
        raise E20Error("E20 candidate pool has duplicate or missing chunk IDs.")
    original = pool[: policy["output_context_count"]]
    document_id, diagnostics = resolve_document(question, pool, documents, config)
    diagnostics = {**diagnostics, "changed": False, "tail_promoted_ranks": []}
    if document_id is None:
        return original, diagnostics
    tail = [
        (rank, context)
        for rank, context in enumerate(pool[policy["tail_start_rank"] - 1 :], start=policy["tail_start_rank"])
        if context["document_id"] == document_id
    ][: policy["maximum_tail_promotions"]]
    if not tail:
        return original, {**diagnostics, "resolution": diagnostics["resolution"] + "_without_tail_match"}
    removable = [
        index for index, context in enumerate(original)
        if context["document_id"] != document_id
    ]
    if len(removable) < len(tail):
        return original, {**diagnostics, "resolution": "insufficient_nonmatching_slots"}
    remove = set(removable[-len(tail) :])
    kept = [context for index, context in enumerate(original) if index not in remove]
    existing = [index for index, context in enumerate(kept) if context["document_id"] == document_id]
    insertion = existing[-1] + 1 if existing else 0
    promoted = [context for _, context in tail]
    candidate = [*kept[:insertion], *promoted, *kept[insertion:]]
    if len(candidate) != policy["output_context_count"] or len({x["chunk_id"] for x in candidate}) != len(candidate):
        raise E20Error("E20 selection did not produce twelve unique contexts.")
    changed = [x["chunk_id"] for x in candidate] != [x["chunk_id"] for x in original]
    return candidate, {
        **diagnostics,
        "changed": changed,
        "resolved_document_id": document_id,
        "tail_promoted_ranks": [rank for rank, _ in tail],
        "tail_promoted_chunk_ids": [context["chunk_id"] for _, context in tail],
        "dropped_chunk_ids": [original[index]["chunk_id"] for index in sorted(remove)],
    }


def _control_rows(
    control_directory: Path, ids: list[str], config: E20Config
) -> list[dict[str, Any]]:
    section = config.section("source_control")
    report_path = control_directory / "report.json"
    results_path = control_directory / section["results_path"]
    if not report_path.is_file() or not results_path.is_file():
        raise E20Error("Saved E19 Notebook 2 control artifact is incomplete.")
    if file_sha256(results_path) != section["results_sha256"]:
        raise E20Error("Saved E19 candidate results changed.")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    evidence = report.get("evidence", {})
    if (
        report.get("experiment_id") != section["experiment_id"]
        or report.get("sample_size") != section["sample_size"]
        or report.get("candidate_variant") != section["variant"]
        or report.get("smoke_leader") != section["variant"]
        or evidence.get("sample_ids_sha256") != section["sample_ids_sha256"]
        or evidence.get("candidate_adapter_sha256") != section["adapter_sha256"]
        or evidence.get("candidate_results_sha256") != section["results_sha256"]
        or report.get("public_read") is not False
        or report.get("holdout_untouched") is not True
    ):
        raise E20Error("Saved E19 control report changed.")
    rows = _read_jsonl(results_path)
    if len(rows) != len(ids):
        raise E20Error("Saved E19 control count changed.")
    for index, (question_id, row) in enumerate(zip(ids, rows)):
        payload = {key: value for key, value in row.items() if key != "record_sha256"}
        if (
            row.get("question_id") != question_id
            or row.get("sample_index") != index
            or row.get("variant") != section["variant"]
            or row.get("selected_context_count") != section["context_count"]
            or not isinstance(row.get("answer"), str)
            or not row["answer"].strip()
            or row.get("record_sha256") != _json_sha256(payload)
        ):
            raise E20Error(f"Invalid saved E19 control result: {index}")
    return rows


def validate_preflight(
    *, project_root: Path, e08a_directory: Path, e00_directory: Path,
    training_directory: Path, control_directory: Path, train_path: Path,
    dev_path: Path, config: E20Config,
) -> dict[str, Any]:
    source = validate_training_preflight(
        project_root=project_root,
        e08a_directory=e08a_directory,
        e00_directory=e00_directory,
        train_path=train_path,
        dev_path=dev_path,
        config=config.source,
    )
    adapter_hash, complete = validate_candidate_adapter(training_directory, config.source)
    _, _, ids = eval_sample(train_path, dev_path, config.source)
    _control_rows(control_directory, ids, config)
    scorer = project_root / config.section("scoring")["official_scorer_path"]
    if file_sha256(scorer) != config.section("scoring")["official_scorer_sha256"]:
        raise E20Error("Pinned official scorer changed.")
    if adapter_hash != config.section("source_control")["adapter_sha256"]:
        raise E20Error("E20 is not using the exact E19 adapter.")
    return {
        "code_version": CODE_VERSION,
        "code_sha256": code_sha(project_root),
        "config_sha256": config.config_sha256,
        "source_e19_config_sha256": config.raw["source_e19_config_sha256"],
        "source_contexts_sha256": config.source.section("source_e08a")["dev_results_sha256"],
        "control_results_sha256": config.section("source_control")["results_sha256"],
        "adapter_sha256": adapter_hash,
        "training_identity_sha256": complete["identity_sha256"],
        "sample_ids_sha256": config.section("evaluation")["sample_ids_sha256"],
        "sample_size": len(ids),
        "candidate_pool_size": 20,
        "output_context_count": 12,
        "max_new_tokens": 704,
        "maximum_stack_parameters": source["maximum_stack_parameters"],
    }


def prepare(
    *, e08a_directory: Path, e00_directory: Path, train_path: Path,
    dev_path: Path, output_directory: Path, config: E20Config,
) -> dict[str, Any]:
    dev, dev521_ids, ids = eval_sample(train_path, dev_path, config.source)
    source = config.source.section("source_e08a")
    all_rows = _validate_context_rows(
        e08a_directory / source["dev_results_path"], dev521_ids, "dev521", source
    )
    start = config.section("evaluation")["dev521_offset"]
    selected_rows = all_rows[start : start + len(ids)]
    wanted = {
        candidate["chunk_id"]
        for row in selected_rows
        for candidate in row["fused"]
    }
    metadata = config.source.contract["metadata_source"]
    chunks = scan_selected(
        e00_directory / "chunks.jsonl", "chunk_id", wanted, metadata["chunks_sha256"]
    )
    documents = scan_selected(
        e00_directory / "documents.jsonl",
        "document_id",
        {chunk["document_id"] for chunk in chunks.values()},
        metadata["documents_sha256"],
    )
    prepared: list[dict[str, Any]] = []
    for index, (question_id, row) in enumerate(zip(ids, selected_rows)):
        pool = [
            {
                "chunk_id": chunks[item["chunk_id"]]["chunk_id"],
                "document_id": chunks[item["chunk_id"]]["document_id"],
                "article_number": chunks[item["chunk_id"]].get("article_number"),
                "text": chunks[item["chunk_id"]]["text"],
            }
            for item in row["fused"]
        ]
        if pool[:12] != row["contexts"]:
            raise E20Error(f"E00 top-20 no longer reconstructs E08A top-12: {question_id}")
        selected, diagnostics = source_aware_top12(
            dev[question_id]["question"], pool, documents, config
        )
        enriched = [
            enrich_context(context, chunks[context["chunk_id"]], documents[context["document_id"]])[0]
            for context in selected
        ]
        prepared.append(
            {
                "question_id": question_id,
                "sample_index": index,
                "answer_included": False,
                "source_pool_chunk_ids": [context["chunk_id"] for context in pool],
                "control_chunk_ids": [context["chunk_id"] for context in pool[:12]],
                "contexts": enriched,
                "selection_diagnostics": diagnostics,
            }
        )
    root = output_directory / "prepared"
    root.mkdir(parents=True, exist_ok=True)
    results_path = root / "results.jsonl"
    _atomic_jsonl(results_path, prepared)
    diagnostics = [row["selection_diagnostics"] for row in prepared]
    changed = [row for row in diagnostics if row["changed"]]
    summary = {
        "schema_version": "1.0",
        "experiment_id": EXPERIMENT,
        "sample_size": len(ids),
        "sample_ids_sha256": config.section("evaluation")["sample_ids_sha256"],
        "results_sha256": file_sha256(results_path),
        "source_contexts_sha256": source["dev_results_sha256"],
        "candidate_pool_size": 20,
        "contexts_per_question": 12,
        "changed_question_count": len(changed),
        "changed_question_rate": len(changed) / len(prepared),
        "tail_promoted_chunk_count": sum(len(row["tail_promoted_ranks"]) for row in changed),
        "document_number_resolution_count": sum(row["resolution"] == "document_number" for row in diagnostics),
        "source_title_resolution_count": sum(row["resolution"] == "source_title" for row in diagnostics),
        "ambiguous_resolution_count": sum(row["resolution"].startswith("ambiguous") for row in diagnostics),
        "unchanged_question_count": len(prepared) - len(changed),
        "answers_used": False,
        "fallback_preserves_original_top12": True,
        "metadata_policy": metadata["policy"],
    }
    _atomic_json(root / "summary.json", summary)
    return summary


def _assigned_indices(worker_rank: int, config: E20Config) -> list[int]:
    if worker_rank not in (0, 1):
        raise E20Error("E20 worker rank must be 0 or 1.")
    size = config.section("evaluation")["questions_per_worker"]
    return list(range(worker_rank * size, (worker_rank + 1) * size))


def run_worker(
    *, worker_rank: int, device: str, model: Any, tokenizer: Any,
    adapter_hash: str, adapter_parameters: int, device_map: dict[str, Any],
    project_root: Path, train_path: Path, dev_path: Path,
    output_directory: Path, config: E20Config,
) -> dict[str, Any]:
    if device != f"cuda:{worker_rank}":
        raise E20Error("E20 worker/device mapping changed.")
    dev, _, ids = eval_sample(train_path, dev_path, config.source)
    prepared_path = output_directory / "prepared/results.jsonl"
    prepared = _read_jsonl(prepared_path)
    summary = json.loads((output_directory / "prepared/summary.json").read_text(encoding="utf-8"))
    if (
        [row.get("question_id") for row in prepared] != ids
        or summary.get("experiment_id") != EXPERIMENT
        or summary.get("results_sha256") != file_sha256(prepared_path)
        or summary.get("sample_ids_sha256") != config.section("evaluation")["sample_ids_sha256"]
        or summary.get("answers_used") is not False
    ):
        raise E20Error("Prepared E20 contexts changed.")
    assigned = _assigned_indices(worker_rank, config)
    identity = {
        "code_version": CODE_VERSION,
        "code_sha256": code_sha(project_root),
        "config_sha256": config.config_sha256,
        "prepared_sha256": file_sha256(prepared_path),
        "sample_ids_sha256": config.section("evaluation")["sample_ids_sha256"],
        "variant": CANDIDATE,
        "adapter_sha256": adapter_hash,
        "adapter_parameters": adapter_parameters,
        "worker_rank": worker_rank,
        "device": device,
        "device_map": device_map,
        "assigned_indices_sha256": hashlib.sha256(
            ",".join(str(index) for index in assigned).encode("ascii")
        ).hexdigest(),
    }
    identity["identity_sha256"] = _json_sha256(identity)
    root = output_directory / "evaluation" / CANDIDATE
    records = root / "records"
    records.mkdir(parents=True, exist_ok=True)
    state = root / f"worker-{worker_rank}-state.json"
    completed = _load_worker_progress(
        records=records,
        state_path=state,
        identity=identity,
        assigned_indices=assigned,
        sample_ids=ids,
    )
    for index in assigned[:completed]:
        row = json.loads((records / f"{index:04d}.json").read_text(encoding="utf-8"))
        if row.get("record_sha256") != _json_sha256(
            {key: value for key, value in row.items() if key != "record_sha256"}
        ):
            raise E20Error(f"Changed E20 checkpoint record: {index}")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    packing = inference_packing(config.source)

    def count(messages: list[dict[str, Any]]) -> int:
        rendered = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        return len(tokenizer(rendered, add_special_tokens=False)["input_ids"])

    import torch

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    for completed_position in range(completed, len(assigned)):
        index = assigned[completed_position]
        question_id = ids[index]
        selected, messages, input_tokens = pack_contexts(
            question=dev[question_id]["question"],
            contexts=prepared[index]["contexts"],
            config=packing,
            token_counter=count,
        )
        if len(selected) != 12:
            raise E20Error(f"E20 prompt did not fit all twelve contexts: {question_id}")
        rendered = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = tokenizer(rendered, add_special_tokens=False, return_tensors="pt")
        inputs = {key: value.to(device) for key, value in inputs.items()}
        started = time.perf_counter()
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                do_sample=False,
                num_beams=1,
                max_new_tokens=704,
                use_cache=True,
            )
        latency = (time.perf_counter() - started) * 1000
        new_ids = generated[0, inputs["input_ids"].shape[1] :]
        answer = tokenizer.decode(
            new_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        ).strip()
        if not answer:
            raise E20Error(f"Empty E20 answer: {question_id}")
        eos = tokenizer.eos_token_id
        eos_ids = set(eos if isinstance(eos, list) else [eos])
        generated_count = int(new_ids.shape[0])
        finish = (
            "eos"
            if generated_count and int(new_ids[-1]) in eos_ids
            else "length"
            if generated_count >= 704
            else "other"
        )
        record = {
            "question_id": question_id,
            "sample_index": index,
            "worker_rank": worker_rank,
            "variant": CANDIDATE,
            "worker_identity_sha256": identity["identity_sha256"],
            "answer": answer,
            "selected_chunk_ids": [context["chunk_id"] for context in selected],
            "selected_context_count": 12,
            "input_tokens": input_tokens,
            "output_tokens": len(tokenizer(answer, add_special_tokens=False)["input_ids"]),
            "generated_tokens_including_special": generated_count,
            "finish_reason": finish,
            "generation_latency_ms": latency,
            "retrieval_changed": prepared[index]["selection_diagnostics"]["changed"],
        }
        record["record_sha256"] = _json_sha256(record)
        _atomic_json(records / f"{index:04d}.json", record)
        _write_worker_state(state, identity, completed_position + 1, len(assigned))
        LOGGER.info(
            "e20_generation_progress worker=%d device=%s completed=%d total=100 "
            "global_completed_at_least=%d global_total=200 question_id=%s changed=%s finish=%s",
            worker_rank,
            device,
            completed_position + 1,
            (completed_position + 1) * 2,
            question_id,
            record["retrieval_changed"],
            finish,
        )
    _write_worker_state(state, identity, len(assigned), len(assigned))
    return {"worker_rank": worker_rank, "device": device, "completed": len(assigned)}


def finalize_score(
    *, control_directory: Path, train_path: Path, dev_path: Path,
    output_directory: Path, config: E20Config,
) -> dict[str, Any]:
    ensure_nltk_resources(download=False)
    dev, _, ids = eval_sample(train_path, dev_path, config.source)
    control_rows = _control_rows(control_directory, ids, config)
    root = output_directory / "evaluation" / CANDIDATE
    for worker_rank in (0, 1):
        state = json.loads((root / f"worker-{worker_rank}-state.json").read_text(encoding="utf-8"))
        if state.get("complete") is not True or state.get("completed_count") != 100:
            raise E20Error(f"Incomplete E20 worker: {worker_rank}")
    candidate_rows = []
    for index, question_id in enumerate(ids):
        row = json.loads((root / "records" / f"{index:04d}.json").read_text(encoding="utf-8"))
        payload = {key: value for key, value in row.items() if key != "record_sha256"}
        if (
            row.get("question_id") != question_id
            or row.get("sample_index") != index
            or row.get("worker_rank") != index // 100
            or row.get("variant") != CANDIDATE
            or row.get("selected_context_count") != 12
            or not isinstance(row.get("answer"), str)
            or not row["answer"].strip()
            or row.get("record_sha256") != _json_sha256(payload)
        ):
            raise E20Error(f"Invalid E20 candidate result: {index}")
        candidate_rows.append(row)
    results_path = root / "results.jsonl"
    _atomic_jsonl(results_path, candidate_rows)
    prepared = _read_jsonl(output_directory / "prepared/results.jsonl")
    summary = json.loads((output_directory / "prepared/summary.json").read_text(encoding="utf-8"))
    scores = {CONTROL: [], CANDIDATE: []}
    per_question = []
    deltas = []
    changed_deltas = []
    for index, question_id in enumerate(ids):
        values = {}
        for variant, rows in ((CONTROL, control_rows), (CANDIDATE, candidate_rows)):
            values[variant] = {
                "meteor": nltk_meteor_score(dev[question_id]["answer"], rows[index]["answer"]),
                "rouge_l": rouge_l_fmeasure(dev[question_id]["answer"], rows[index]["answer"]),
            }
            scores[variant].append(values[variant])
        delta = values[CANDIDATE]["meteor"] - values[CONTROL]["meteor"]
        changed = bool(prepared[index]["selection_diagnostics"]["changed"])
        deltas.append(delta)
        if changed:
            changed_deltas.append(delta)
        per_question.append(
            {
                "question_id": question_id,
                "sample_index": index,
                "retrieval_changed": changed,
                "scores": values,
                "meteor_delta": delta,
            }
        )
    _atomic_jsonl(output_directory / "per_question_scores.jsonl", per_question)
    metrics = {
        CONTROL: _metrics(control_rows, scores[CONTROL]),
        CANDIDATE: _metrics(candidate_rows, scores[CANDIDATE]),
    }
    paired = {
        "meteor_mean": fmean(deltas),
        "meteor_bootstrap_95_ci": _bootstrap_ci(
            deltas,
            seed=config.section("scoring")["bootstrap_seed"],
            iterations=config.section("scoring")["bootstrap_iterations"],
        ),
        "improved_questions": sum(value > 0 for value in deltas),
        "worsened_questions": sum(value < 0 for value in deltas),
        "tied_questions": sum(value == 0 for value in deltas),
    }
    changed_effect = {
        "question_count": len(changed_deltas),
        "meteor_mean": fmean(changed_deltas) if changed_deltas else None,
        "meteor_bootstrap_95_ci": (
            _bootstrap_ci(
                changed_deltas,
                seed=config.section("scoring")["bootstrap_seed"] + "-changed",
                iterations=config.section("scoring")["bootstrap_iterations"],
            )
            if changed_deltas
            else None
        ),
        "improved_questions": sum(value > 0 for value in changed_deltas),
        "worsened_questions": sum(value < 0 for value in changed_deltas),
        "tied_questions": sum(value == 0 for value in changed_deltas),
    }
    report = {
        "schema_version": "1.0",
        "experiment_id": EXPERIMENT,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "sample_size": len(ids),
        "sample_scope": "same-already-used-dev200-as-e19-no-final121-read",
        "control_variant": CONTROL,
        "candidate_variant": CANDIDATE,
        "only_changed_factor": "high-precision source-aware replacement from frozen fused ranks 13-20",
        "retrieval_diagnostics": summary,
        "metrics": metrics,
        "paired_delta_candidate_minus_control": paired,
        "paired_effect_on_changed_questions": changed_effect,
        "smoke_leader": max((CONTROL, CANDIDATE), key=lambda variant: metrics[variant]["meteor"]),
        "promotion_allowed": False,
        "evidence": {
            "config_sha256": config.config_sha256,
            "sample_ids_sha256": config.section("evaluation")["sample_ids_sha256"],
            "source_contexts_sha256": config.source.section("source_e08a")["dev_results_sha256"],
            "prepared_contexts_sha256": file_sha256(output_directory / "prepared/results.jsonl"),
            "adapter_sha256": config.section("source_control")["adapter_sha256"],
            "control_results_sha256": config.section("source_control")["results_sha256"],
            "candidate_results_sha256": file_sha256(results_path),
        },
        "public_read": False,
        "holdout_untouched": True,
        "warning": "E20 reuses the repeatedly inspected E19 dev200; it is diagnostic and cannot auto-promote.",
    }
    _atomic_json(output_directory / "report.json", report)
    return report


__all__ = [
    "CANDIDATE", "CODE_VERSION", "CONTROL", "E20Config", "E20Error", "EXPERIMENT",
    "code_sha", "document_number_references", "finalize_score", "load_config",
    "prepare", "resolve_document", "run_worker", "source_aware_top12",
    "title_match_key", "validate_preflight",
]
