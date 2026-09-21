"""Final deterministic public-1000 inference and submission packaging."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from uit_dsc_fixed_rag.bm25 import SqliteBm25Index
from uit_dsc_fixed_rag.corpus import file_sha256
from uit_dsc_fixed_rag.e02_answer import _atomic_json, _atomic_jsonl, pack_contexts
from uit_dsc_fixed_rag.e02_compare import (
    _ChunkStore,
    _TorchGpuFlatIpIndex,
    _load_mapping,
    weighted_rrf,
)
from uit_dsc_fixed_rag.e03_rrf_grid import (
    _json_sha256,
    _load_contiguous,
    _load_worker_progress,
    _read_jsonl,
    _write_jsonl_from_records,
    _write_state,
    _write_worker_state,
)
from uit_dsc_fixed_rag.e07_full_dev import (
    _validate_full_train_report,
    load_embedding as _load_embedding,
    load_generator_with_adapter as _load_generator_with_adapter,
    prepare_local_bm25 as _prepare_local_bm25,
)
from uit_dsc_fixed_rag.e07_lora import InferencePacking


CODE_VERSION = "0.18.1"
LOGGER = logging.getLogger(__name__)


class FinalPublicError(RuntimeError):
    """Raised when final public inference cannot continue safely."""


@dataclass(frozen=True)
class FinalPublicConfig:
    raw: dict[str, Any]
    path: Path

    @property
    def config_sha256(self) -> str:
        return file_sha256(self.path)

    def section(self, key: str) -> dict[str, Any]:
        value = self.raw.get(key)
        if not isinstance(value, dict):
            raise ValueError(f"Final public section must be an object: {key}")
        return value


def load_final_public_config(path: Path) -> FinalPublicConfig:
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "schema_version",
        "experiment_id",
        "model_selection",
        "full_lora_source",
        "public",
        "source_e00",
        "source_dense",
        "retrieval",
        "generator",
        "inference",
        "parameter_budget",
        "execution",
        "submission_contract",
        "run_contract",
    }
    if (
        set(payload) != required
        or payload.get("schema_version") != "1.0"
        or payload.get("experiment_id")
        != "FINAL-public1000-aiteam-qwen35-lora-v1"
    ):
        raise ValueError("Final public config root is incompatible.")
    config = FinalPublicConfig(payload, path)
    public = config.section("public")
    if (
        public.get("sample_size") != 1000
        or public.get("order") != "lexicographic-question-id"
        or public.get("answer_contract") != "all-null-and-never-read-for-inference"
    ):
        raise ValueError("Final public question contract changed.")
    retrieval = config.section("retrieval")
    if retrieval != {
        "candidate_k_per_branch": 40,
        "rrf_constant": 60,
        "sparse_weight": 0.5,
        "dense_weight": 0.5,
        "fused_top_k": 20,
        "selected_contexts": 12,
        "normalize_query_embeddings": True,
        "dense_search_backend": "torch-cuda-exact-flat-ip-float32-batched",
        "reranker": None,
    }:
        raise ValueError("Final public retrieval contract changed.")
    inference = config.section("inference")
    if (
        inference.get("max_input_tokens") != 8192
        or inference.get("max_new_tokens") != 384
        or inference.get("do_sample") is not False
        or inference.get("num_beams") != 1
        or inference.get("enable_thinking") is not False
    ):
        raise ValueError("Final public inference contract changed.")
    budget = config.section("parameter_budget")
    if (
        budget.get("maximum_stack_total")
        != budget.get("embedding")
        + budget.get("generator")
        + budget.get("adapter_parameter_cap")
        or budget["maximum_stack_total"] >= budget["exclusive_limit"]
    ):
        raise ValueError("Final public parameter budget failed.")
    execution = config.section("execution")
    if (
        execution.get("generation_workers") != 2
        or execution.get("checkpoint_after_questions") != 1
    ):
        raise ValueError("Final public checkpoint contract changed.")
    submission = config.section("submission_contract")
    if (
        submission.get("archive_name") != "submission.zip"
        or submission.get("json_name") != "submission.json"
        or submission.get("archive_members") != ["submission.json"]
        or submission.get("record_schema") != {"answer": "non-empty-string"}
    ):
        raise ValueError("Final submission contract changed.")
    contract = config.section("run_contract")
    if not contract or not all(value is True for value in contract.values()):
        raise ValueError("Final public run contract lost an invariant.")
    return config


def load_public_questions(path: Path, config: FinalPublicConfig) -> tuple[dict[str, str], list[str]]:
    public_cfg = config.section("public")
    if not path.is_file() or file_sha256(path) != public_cfg["sha256"]:
        raise FinalPublicError("Pinned official public file is missing or changed.")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or len(payload) != public_cfg["sample_size"]:
        raise FinalPublicError("Official public file must contain exactly 1,000 records.")
    questions: dict[str, str] = {}
    for raw_id, record in payload.items():
        question_id = str(raw_id)
        if (
            not isinstance(record, dict)
            or set(record) != {"question", "answer"}
            or not isinstance(record["question"], str)
            or not record["question"].strip()
            or record["answer"] is not None
        ):
            raise FinalPublicError(f"Invalid or answer-bearing public record: {question_id}")
        questions[question_id] = record["question"]
    ids = sorted(questions)
    if _ids_sha(ids) != public_cfg["sample_ids_sha256"]:
        raise FinalPublicError("Deterministic public question order changed.")
    return questions, ids


def validate_final_preflight(
    *,
    project_root: Path,
    e00_directory: Path,
    dense_directory: Path,
    full_lora_directory: Path,
    selection_directory: Path,
    public_path: Path,
    config: FinalPublicConfig,
) -> dict[str, Any]:
    questions, ids = load_public_questions(public_path, config)
    del questions

    scorer_cfg = config.section("submission_contract")
    scorer_bytes, scorer_container = _load_official_scorer_bytes(
        project_root, scorer_cfg
    )
    if hashlib.sha256(scorer_bytes).hexdigest() != scorer_cfg[
        "official_scorer_entry_sha256"
    ]:
        raise FinalPublicError("Official scorer mapping implementation changed.")
    scorer_text = scorer_bytes.decode("utf-8")
    if "v['answer']" not in scorer_text or "len(ids_preds) != len(ids_truth)" not in scorer_text:
        raise FinalPublicError("Official scorer no longer exposes the reviewed mapping contract.")

    e00_cfg = config.section("source_e00")
    manifest_path = e00_directory / "manifest.json"
    bm25_path = e00_directory / "bm25.sqlite3"
    if (
        not manifest_path.is_file()
        or file_sha256(manifest_path) != e00_cfg["manifest_sha256"]
        or not bm25_path.is_file()
        or bm25_path.stat().st_size != e00_cfg["bm25_bytes"]
        or file_sha256(bm25_path) != e00_cfg["bm25_sha256"]
    ):
        raise FinalPublicError("Pinned E00 v2 artifact is missing or changed.")

    dense_cfg = config.section("source_dense")
    dense_manifest_path = dense_directory / "manifest.json"
    dense_path = dense_directory / "dense.faiss"
    mapping_path = dense_directory / "chunk_ids.jsonl"
    if not all(path.is_file() for path in (dense_manifest_path, dense_path, mapping_path)):
        raise FinalPublicError("Pinned AITeam dense artifact is incomplete.")
    dense_manifest = json.loads(dense_manifest_path.read_text(encoding="utf-8"))
    if (
        dense_manifest.get("record_count") != dense_cfg["record_count"]
        or dense_manifest.get("dimension") != dense_cfg["dimension"]
        or dense_manifest.get("candidate", {}).get("key") != dense_cfg["candidate"]
        or file_sha256(dense_path) != dense_cfg["dense_faiss_sha256"]
        or file_sha256(mapping_path) != dense_cfg["chunk_ids_sha256"]
    ):
        raise FinalPublicError("Pinned AITeam dense artifact changed.")

    lora_cfg = config.section("full_lora_source")
    lora_report_path = full_lora_directory / "report.json"
    lora_results_path = full_lora_directory / "evaluation" / "results.jsonl"
    adapter_path = (
        full_lora_directory
        / "training"
        / "adapter-final"
        / "adapter_model.safetensors"
    )
    if not all(path.is_file() for path in (lora_report_path, lora_results_path, adapter_path)):
        raise FinalPublicError("Saved full-train LoRA artifact is incomplete.")
    lora_report = json.loads(lora_report_path.read_text(encoding="utf-8"))
    _validate_full_train_report(lora_report, lora_cfg)
    if (
        file_sha256(lora_results_path) != lora_cfg["evaluation_results_sha256"]
        or file_sha256(adapter_path) != lora_cfg["adapter_sha256"]
    ):
        raise FinalPublicError("Saved full-train LoRA evidence changed.")

    selection_cfg = config.section("model_selection")
    selection_report_path = selection_directory / "report.json"
    selection_results_path = selection_directory / "generation" / "results.jsonl"
    selection_scores_path = selection_directory / "generation" / "scores.jsonl"
    if not all(
        path.is_file()
        for path in (selection_report_path, selection_results_path, selection_scores_path)
    ):
        raise FinalPublicError("Saved E07G model-selection artifact is incomplete.")
    selection_report = json.loads(selection_report_path.read_text(encoding="utf-8"))
    _validate_model_selection_report(selection_report, selection_cfg)
    if (
        file_sha256(selection_results_path) != selection_cfg["generation_results_sha256"]
        or file_sha256(selection_scores_path) != selection_cfg["scores_sha256"]
    ):
        raise FinalPublicError("Saved E07G model-selection evidence changed.")

    return {
        "config_sha256": config.config_sha256,
        "public_sha256": config.section("public")["sha256"],
        "sample_ids_sha256": _ids_sha(ids),
        "sample_size": len(ids),
        "official_scorer_zip_sha256": scorer_cfg["official_scorer_zip_sha256"],
        "official_scorer_entry_sha256": scorer_cfg["official_scorer_entry_sha256"],
        "official_scorer_container": scorer_container,
        "e00_manifest_sha256": e00_cfg["manifest_sha256"],
        "bm25_sha256": e00_cfg["bm25_sha256"],
        "dense_faiss_sha256": dense_cfg["dense_faiss_sha256"],
        "full_lora_adapter_sha256": lora_cfg["adapter_sha256"],
        "selection_generation_results_sha256": selection_cfg[
            "generation_results_sha256"
        ],
        "selected_generator": selection_cfg["winner"],
        "maximum_stack_parameters": config.section("parameter_budget")[
            "maximum_stack_total"
        ],
    }


def prepare_local_bm25(
    *, source: Path, runtime_directory: Path, config: FinalPublicConfig
) -> Path:
    return _prepare_local_bm25(
        source=source, runtime_directory=runtime_directory, config=config
    )


def run_sparse_retrieval(
    *,
    bm25_path: Path,
    public_path: Path,
    output_directory: Path,
    config: FinalPublicConfig,
) -> dict[str, Any]:
    questions, ids = load_public_questions(public_path, config)
    identity = {
        "code_version": CODE_VERSION,
        "stage": "final-public1000-bm25",
        "config_sha256": config.config_sha256,
        "bm25_sha256": config.section("source_e00")["bm25_sha256"],
        "sample_ids_sha256": _ids_sha(ids),
    }
    identity["identity_sha256"] = _json_sha256(identity)
    root = output_directory / "retrieval" / "sparse"
    records = root / "records"
    records.mkdir(parents=True, exist_ok=True)
    state_path = root / "state.json"
    completed = _load_contiguous(records, state_path, identity, ids)
    pending = ids[completed:]
    local = threading.local()
    depth = config.section("retrieval")["candidate_k_per_branch"]

    def search(question_id: str) -> dict[str, Any]:
        index = getattr(local, "index", None)
        if index is None:
            index = SqliteBm25Index(bm25_path)
            local.index = index
        started = time.perf_counter()
        hits = index.search(questions[question_id], top_k=depth)
        return {
            "ids": [hit.chunk_id for hit in hits],
            "scores": [float(hit.score) for hit in hits],
            "latency_ms": (time.perf_counter() - started) * 1000,
        }

    workers = config.section("execution")["sparse_worker_threads"]
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(search, question_id) for question_id in pending]
        for offset, (question_id, future) in enumerate(zip(pending, futures)):
            index = completed + offset
            result = future.result()
            if not result["ids"] or len(result["ids"]) > depth:
                raise FinalPublicError(f"Invalid BM25 result: {question_id}")
            _atomic_json(
                records / f"{index:04d}.json",
                {
                    "question_id": question_id,
                    "sample_index": index,
                    "sparse_top_ids": result["ids"],
                    "sparse_scores": result["scores"],
                    "latency_ms": result["latency_ms"],
                },
            )
            _write_state(state_path, identity, index + 1, complete=False)
            LOGGER.info(
                "final_public_sparse_progress completed=%d total=%d question_id=%s",
                index + 1,
                len(ids),
                question_id,
            )
    _write_jsonl_from_records(root / "results.jsonl", records, len(ids))
    _write_state(state_path, identity, len(ids), complete=True)
    return {
        "sample_size": len(ids),
        "results_sha256": file_sha256(root / "results.jsonl"),
    }


def load_embedding(config: FinalPublicConfig, device: str) -> Any:
    return _load_embedding(config, device)


def run_dense_and_fusion(
    *,
    model: Any,
    dense_directory: Path,
    bm25_path: Path,
    public_path: Path,
    output_directory: Path,
    config: FinalPublicConfig,
) -> dict[str, Any]:
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover
        raise FinalPublicError("Install numpy for final dense retrieval.") from exc
    questions, ids = load_public_questions(public_path, config)
    sparse_path = output_directory / "retrieval" / "sparse" / "results.jsonl"
    sparse = _read_jsonl(sparse_path)
    if [row.get("question_id") for row in sparse] != ids:
        raise FinalPublicError("Final sparse cache is incomplete or reordered.")
    dense_cfg = config.section("source_dense")
    retrieval = config.section("retrieval")
    identity = {
        "code_version": CODE_VERSION,
        "stage": "final-public1000-aiteam-rrf-top12",
        "config_sha256": config.config_sha256,
        "sparse_results_sha256": file_sha256(sparse_path),
        "dense_faiss_sha256": dense_cfg["dense_faiss_sha256"],
        "sample_ids_sha256": _ids_sha(ids),
    }
    identity["identity_sha256"] = _json_sha256(identity)
    root = output_directory / "retrieval"
    records = root / "records"
    records.mkdir(parents=True, exist_ok=True)
    state_path = root / "state.json"
    completed = _load_contiguous(records, state_path, identity, ids)
    mapping = _load_mapping(
        dense_directory / "chunk_ids.jsonl", dense_cfg["record_count"]
    )
    search = _TorchGpuFlatIpIndex(
        dense_directory / "dense.faiss",
        device=config.section("execution")["dense_device"],
        expected_count=dense_cfg["record_count"],
        expected_dimension=dense_cfg["dimension"],
    )
    store = _ChunkStore(bm25_path)
    batch_size = config.section("execution")["dense_query_batch_size"]
    try:
        for start in range(completed, len(ids), batch_size):
            stop = min(start + batch_size, len(ids))
            batch_ids = ids[start:stop]
            vectors = model.encode(
                [dense_cfg["query_prefix"] + questions[qid] for qid in batch_ids],
                batch_size=len(batch_ids),
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
            vectors = np.asarray(vectors, dtype=np.float32)
            if (
                vectors.shape != (len(batch_ids), dense_cfg["dimension"])
                or not np.isfinite(vectors).all()
            ):
                raise FinalPublicError("Invalid final public query embeddings.")
            dense_scores, dense_rows = search.search(
                vectors, retrieval["candidate_k_per_branch"]
            )
            for offset, question_id in enumerate(batch_ids):
                index = start + offset
                dense_ids = [
                    mapping[int(row)] for row in dense_rows[offset] if int(row) >= 0
                ]
                fused = weighted_rrf(
                    {
                        "sparse": sparse[index]["sparse_top_ids"],
                        "dense": dense_ids,
                    },
                    weights={
                        "sparse": retrieval["sparse_weight"],
                        "dense": retrieval["dense_weight"],
                    },
                    constant=retrieval["rrf_constant"],
                    top_k=retrieval["fused_top_k"],
                )
                contexts = store.fetch([row["chunk_id"] for row in fused])
                selected = contexts[: retrieval["selected_contexts"]]
                if not selected or len(selected) > retrieval["selected_contexts"]:
                    raise FinalPublicError(f"Invalid selected contexts: {question_id}")
                _atomic_json(
                    records / f"{index:04d}.json",
                    {
                        "question_id": question_id,
                        "sample_index": index,
                        "sparse_top_ids": sparse[index]["sparse_top_ids"],
                        "dense_top_ids": dense_ids,
                        "dense_scores": [
                            float(value)
                            for value in dense_scores[offset][: len(dense_ids)]
                        ],
                        "fused": fused,
                        "contexts": selected,
                    },
                )
                _write_state(state_path, identity, index + 1, complete=False)
            LOGGER.info(
                "final_public_dense_progress completed=%d total=%d", stop, len(ids)
            )
    finally:
        search.close()
        store.close()
    _write_jsonl_from_records(root / "results.jsonl", records, len(ids))
    _write_state(state_path, identity, len(ids), complete=True)
    return {
        "sample_size": len(ids),
        "results_sha256": file_sha256(root / "results.jsonl"),
    }


def load_generator_with_adapter(
    *, full_lora_directory: Path, config: FinalPublicConfig, device: str
) -> tuple[Any, Any, dict[str, Any], int]:
    return _load_generator_with_adapter(
        full_lora_directory=full_lora_directory, config=config, device=device
    )


def run_generation_worker(
    *,
    worker_rank: int,
    device: str,
    model: Any,
    tokenizer: Any,
    full_lora_directory: Path,
    public_path: Path,
    output_directory: Path,
    config: FinalPublicConfig,
    device_map: dict[str, Any],
    adapter_parameters: int,
) -> dict[str, Any]:
    workers = config.section("execution")["generation_workers"]
    if worker_rank not in range(workers) or device != f"cuda:{worker_rank}":
        raise FinalPublicError("Final generation worker/device mapping changed.")
    questions, ids = load_public_questions(public_path, config)
    retrieval_path = output_directory / "retrieval" / "results.jsonl"
    retrieval = _read_jsonl(retrieval_path)
    if [row.get("question_id") for row in retrieval] != ids:
        raise FinalPublicError("Final retrieval results are incomplete or reordered.")
    assigned = [index for index in range(len(ids)) if index % workers == worker_rank]
    adapter_path = (
        full_lora_directory
        / "training"
        / "adapter-final"
        / "adapter_model.safetensors"
    )
    identity = {
        "code_version": CODE_VERSION,
        "stage": "final-public1000-qwen35-full-lora-generation",
        "config_sha256": config.config_sha256,
        "retrieval_results_sha256": file_sha256(retrieval_path),
        "adapter_sha256": file_sha256(adapter_path),
        "adapter_parameters": adapter_parameters,
        "worker_rank": worker_rank,
        "device": device,
        "device_map": device_map,
        "assigned_indices_sha256": hashlib.sha256(
            ",".join(map(str, assigned)).encode("ascii")
        ).hexdigest(),
    }
    identity["identity_sha256"] = _json_sha256(identity)
    root = output_directory / "generation"
    records = root / "records"
    records.mkdir(parents=True, exist_ok=True)
    state_path = root / f"worker-{worker_rank}-state.json"
    completed = _load_worker_progress(
        records=records,
        state_path=state_path,
        identity=identity,
        assigned_indices=assigned,
        sample_ids=ids,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    inference = config.section("inference")
    context_limit = config.section("retrieval")["selected_contexts"]
    packing = InferencePacking(
        max_input_tokens=inference["max_input_tokens"],
        minimum_contexts=inference["minimum_contexts"],
        system_prompt=inference["system_prompt"],
        answer_instruction=inference["answer_instruction"],
    )

    def token_count(messages: list[dict[str, Any]]) -> int:
        rendered = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        return len(tokenizer(rendered, add_special_tokens=False)["input_ids"])

    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise FinalPublicError("Install PyTorch for final generation.") from exc

    for position in range(completed, len(assigned)):
        index = assigned[position]
        question_id = ids[index]
        selected, messages, input_tokens = pack_contexts(
            question=questions[question_id],
            contexts=retrieval[index]["contexts"][:context_limit],
            config=packing,
            token_counter=token_count,
        )
        rendered = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        tensors = tokenizer(
            rendered, add_special_tokens=False, return_tensors="pt"
        )
        tensors = {key: value.to(device) for key, value in tensors.items()}
        started = time.perf_counter()
        with torch.inference_mode():
            generated = model.generate(
                **tensors,
                do_sample=False,
                num_beams=1,
                max_new_tokens=inference["max_new_tokens"],
                use_cache=True,
            )
        latency_ms = (time.perf_counter() - started) * 1000
        answer = tokenizer.decode(
            generated[0, tensors["input_ids"].shape[1] :],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        ).strip()
        if not answer:
            raise FinalPublicError(f"Generator produced an empty answer: {question_id}")
        _atomic_json(
            records / f"{index:04d}.json",
            {
                "question_id": question_id,
                "sample_index": index,
                "worker_rank": worker_rank,
                "worker_identity_sha256": identity["identity_sha256"],
                "answer": answer,
                "selected_chunk_ids": [row["chunk_id"] for row in selected],
                "selected_context_count": len(selected),
                "input_tokens": input_tokens,
                "output_tokens": len(
                    tokenizer(answer, add_special_tokens=False)["input_ids"]
                ),
                "generation_latency_ms": latency_ms,
            },
        )
        _write_worker_state(state_path, identity, position + 1, len(assigned))
        LOGGER.info(
            "final_public_generation_progress worker=%d device=%s worker_completed=%d "
            "worker_total=%d question_id=%s",
            worker_rank,
            device,
            position + 1,
            len(assigned),
            question_id,
        )
    _write_worker_state(state_path, identity, len(assigned), len(assigned))
    return {
        "worker_rank": worker_rank,
        "device": device,
        "completed": len(assigned),
        "adapter_parameters": adapter_parameters,
    }


def _model_selection_results_sha256(config: FinalPublicConfig) -> str:
    """Resolve the selected generation hash across old and paired schemas."""

    model_selection = config.section("model_selection")
    value = model_selection.get("generation_results_sha256")
    if value is None:
        # Later paired experiments name the promoted side explicitly instead of
        # storing both variants in one generic generation artifact.
        value = model_selection.get("candidate_results_sha256")
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise FinalPublicError(
            "Model-selection evidence lacks a valid generation/candidate results hash."
        )
    return value


def finalize_submission(
    *, output_directory: Path, public_path: Path, config: FinalPublicConfig
) -> dict[str, Any]:
    _, ids = load_public_questions(public_path, config)
    root = output_directory / "generation"
    records = root / "records"
    rows: list[dict[str, Any]] = []
    submission: dict[str, dict[str, str]] = {}
    for index, question_id in enumerate(ids):
        path = records / f"{index:04d}.json"
        if not path.is_file():
            raise FinalPublicError(f"Missing final generation record: {index}")
        row = json.loads(path.read_text(encoding="utf-8"))
        answer = row.get("answer")
        if (
            row.get("sample_index") != index
            or row.get("question_id") != question_id
            or not isinstance(answer, str)
            or not answer.strip()
        ):
            raise FinalPublicError(f"Invalid final generation record: {index}")
        rows.append(row)
        submission[question_id] = {"answer": answer}
    workers = config.section("execution")["generation_workers"]
    for rank in range(workers):
        state_path = root / f"worker-{rank}-state.json"
        if not state_path.is_file():
            raise FinalPublicError(f"Missing final worker state: {rank}")
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if (
            state.get("complete") is not True
            or state.get("completed_count") != state.get("assigned_count")
            or state.get("assigned_count") != len(
                [index for index in range(len(ids)) if index % workers == rank]
            )
        ):
            raise FinalPublicError(f"Final generation worker is incomplete: {rank}")
    _atomic_jsonl(root / "results.jsonl", rows)

    submission_cfg = config.section("submission_contract")
    submission_path = output_directory / submission_cfg["json_name"]
    submission_bytes = (
        json.dumps(submission, ensure_ascii=False, indent=2).encode("utf-8") + b"\n"
    )
    if submission_bytes.startswith(b"\xef\xbb\xbf"):
        raise FinalPublicError("Submission JSON unexpectedly contains a UTF-8 BOM.")
    _atomic_bytes(submission_path, submission_bytes)
    _validate_submission_payload(submission_bytes, ids)

    archive_path = output_directory / submission_cfg["archive_name"]
    _write_submission_zip(archive_path, submission_cfg["json_name"], submission_bytes)
    with zipfile.ZipFile(archive_path) as archive:
        if archive.namelist() != submission_cfg["archive_members"]:
            raise FinalPublicError("Submission archive contains unexpected members.")
        archived_bytes = archive.read(submission_cfg["json_name"])
    if archived_bytes != submission_bytes:
        raise FinalPublicError("Archived submission JSON differs from the validated bytes.")
    _validate_submission_payload(archived_bytes, ids)

    model_selection_results_sha256 = _model_selection_results_sha256(config)

    report = {
        "schema_version": "1.0",
        "experiment_id": config.raw["experiment_id"],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "sample_size": len(ids),
        "selected_stack": {
            "embedding": config.section("source_dense")["candidate"],
            "fusion": "bm25-dense-weighted-rrf-0.5-0.5",
            "reranker": None,
            "contexts": (
                f"ranked-top{config.section('retrieval')['selected_contexts']}"
            ),
            "generator": config.section("model_selection")["winner"],
            "generator_base": config.section("generator")["model_id"],
            "adapter": config.section("generator")["adapter_variant"],
        },
        "validation": {
            "all_question_ids_present": True,
            "all_answers_non_empty": True,
            "utf8_without_bom": True,
            "archive_members": submission_cfg["archive_members"],
            "official_mapping_schema": {"question_id": {"answer": "string"}},
        },
        "files": {
            "generation/results.jsonl": {
                "sha256": file_sha256(root / "results.jsonl"),
                "bytes": (root / "results.jsonl").stat().st_size,
            },
            "submission.json": {
                "sha256": file_sha256(submission_path),
                "bytes": submission_path.stat().st_size,
            },
            "submission.zip": {
                "sha256": file_sha256(archive_path),
                "bytes": archive_path.stat().st_size,
            },
        },
        "evidence": {
            "config_sha256": config.config_sha256,
            "public_sha256": config.section("public")["sha256"],
            "sample_ids_sha256": config.section("public")["sample_ids_sha256"],
            "retrieval_results_sha256": file_sha256(
                output_directory / "retrieval" / "results.jsonl"
            ),
            "adapter_sha256": config.section("full_lora_source")["adapter_sha256"],
            "model_selection_results_sha256": model_selection_results_sha256,
        },
        "holdout_untouched": True,
        "public_answers_read": False,
    }
    _atomic_json(output_directory / "report.json", report)
    return report


def _validate_model_selection_report(
    report: dict[str, Any], expected: dict[str, Any]
) -> None:
    metrics = report.get("metrics", {})
    control = metrics.get("qwen35_2b_lora_fulltrain", {})
    candidate = metrics.get("vi_qwen2_3b_rag", {})
    paired = report.get("paired_delta_viqwen_minus_qwen_lora", {})
    evidence = report.get("evidence", {})
    if (
        report.get("experiment_id") != expected["report_experiment_id"]
        or report.get("comparison") != expected["comparison"]
        or report.get("sample_size") != expected["sample_size"]
        or report.get("smoke_leader") != expected["winner"]
        or report.get("promotion_allowed") is not expected["promotion_allowed"]
        or report.get("holdout_untouched") is not True
        or control.get("meteor") != expected["control_meteor"]
        or control.get("rouge_l") != expected["control_rouge_l"]
        or candidate.get("meteor") != expected["candidate_meteor"]
        or candidate.get("rouge_l") != expected["candidate_rouge_l"]
        or paired.get("meteor_mean") != expected["candidate_minus_control_meteor"]
        or paired.get("meteor_bootstrap_95_ci")
        != expected["candidate_minus_control_meteor_ci"]
        or paired.get("rouge_l_mean") != expected["candidate_minus_control_rouge_l"]
        or paired.get("rouge_l_bootstrap_95_ci")
        != expected["candidate_minus_control_rouge_l_ci"]
        or evidence.get("config_sha256") != expected["report_config_sha256"]
        or evidence.get("generation_results_sha256")
        != expected["generation_results_sha256"]
        or evidence.get("scores_sha256") != expected["scores_sha256"]
        or evidence.get("control_results_sha256")
        != expected["control_results_sha256"]
        or evidence.get("candidate_revision") != expected["candidate_revision"]
        or evidence.get("candidate_parameter_count")
        != expected["candidate_parameter_count"]
    ):
        raise FinalPublicError("Saved E07G model-selection report changed.")


def _load_official_scorer_bytes(
    project_root: Path, scorer_cfg: dict[str, Any]
) -> tuple[bytes, str]:
    """Accept the reviewed ZIP or Kaggle's automatic extraction of that ZIP."""

    scorer_zip = project_root / scorer_cfg["official_scorer_zip_path"]
    if scorer_zip.is_file():
        if file_sha256(scorer_zip) != scorer_cfg["official_scorer_zip_sha256"]:
            raise FinalPublicError("Pinned official scorer archive changed.")
        try:
            with zipfile.ZipFile(scorer_zip) as archive:
                payload = archive.read(scorer_cfg["official_scorer_entry"])
        except (KeyError, zipfile.BadZipFile) as exc:
            raise FinalPublicError("Official scorer archive is invalid.") from exc
        return payload, "pinned-zip"

    # Kaggle expands nested ZIP files while ingesting some datasets. The exact
    # reviewed scoring.py bytes remain pinned even when the outer ZIP bytes do
    # not survive ingestion.
    extracted_root = scorer_zip.with_suffix("")
    extracted_entry = extracted_root / scorer_cfg["official_scorer_entry"]
    if not extracted_entry.is_file():
        raise FinalPublicError(
            "Pinned official scorer is missing as both ZIP and extracted directory."
        )
    payload = extracted_entry.read_bytes()
    if hashlib.sha256(payload).hexdigest() != scorer_cfg[
        "official_scorer_entry_sha256"
    ]:
        raise FinalPublicError("Extracted official scorer implementation changed.")
    return payload, "kaggle-extracted-directory"


def _validate_submission_payload(payload: bytes, ids: list[str]) -> None:
    try:
        decoded = payload.decode("utf-8")
        data = json.loads(decoded)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FinalPublicError("Submission JSON is not valid UTF-8 JSON.") from exc
    if not isinstance(data, dict) or list(data) != ids:
        raise FinalPublicError("Submission question IDs are missing, extra or reordered.")
    for question_id, record in data.items():
        if (
            not isinstance(record, dict)
            or set(record) != {"answer"}
            or not isinstance(record["answer"], str)
            or not record["answer"].strip()
        ):
            raise FinalPublicError(f"Submission record is invalid: {question_id}")


def _write_submission_zip(path: Path, member_name: str, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    if temporary.exists():
        raise FinalPublicError("A partial submission archive exists; remove only its .tmp file.")
    info = zipfile.ZipInfo(member_name, date_time=(2026, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o600 << 16
    try:
        with zipfile.ZipFile(
            temporary, mode="x", compression=zipfile.ZIP_DEFLATED, compresslevel=9
        ) as archive:
            archive.writestr(info, payload)
        os.replace(temporary, path)
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _ids_sha(ids: list[str]) -> str:
    return hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()
