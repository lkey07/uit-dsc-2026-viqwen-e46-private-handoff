"""E08A frozen context retrieval for train-5636 and untouched dev-521."""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from uit_dsc_fixed_rag.bm25 import SqliteBm25Index
from uit_dsc_fixed_rag.corpus import file_sha256
from uit_dsc_fixed_rag.e02_answer import _atomic_json
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
from uit_dsc_fixed_rag.e07_lora import select_train_sample
from uit_dsc_fixed_rag.final_public import (
    _validate_model_selection_report,
    load_embedding as _load_embedding,
    prepare_local_bm25 as _prepare_local_bm25,
)
from uit_dsc_fixed_rag.retrieval_diagnostics import select_dev_sample


CODE_VERSION = "0.19.0"
LOGGER = logging.getLogger(__name__)


class E08ContextError(RuntimeError):
    """Raised when E08A retrieval cannot continue safely."""


@dataclass(frozen=True)
class E08ContextConfig:
    raw: dict[str, Any]
    path: Path

    @property
    def config_sha256(self) -> str:
        return file_sha256(self.path)

    def section(self, key: str) -> dict[str, Any]:
        value = self.raw.get(key)
        if not isinstance(value, dict):
            raise ValueError(f"E08A section must be an object: {key}")
        return value


def load_e08_context_config(path: Path) -> E08ContextConfig:
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "schema_version",
        "experiment_id",
        "advancement_source",
        "source_e00",
        "source_dense",
        "train",
        "dev",
        "retrieval",
        "execution",
        "parameter_budget",
        "run_contract",
    }
    if (
        set(payload) != required
        or payload.get("schema_version") != "1.0"
        or payload.get("experiment_id")
        != "E08A-context-retrieval-train5636-dev521-v1"
    ):
        raise ValueError("E08A config root is incompatible.")
    config = E08ContextConfig(payload, path)
    train = config.section("train")
    if (
        train.get("record_count") != 5636
        or train.get("sample_size") != 5636
        or train.get("answer_usage")
        != "reserved-for-later-generator-supervision-not-retrieval"
    ):
        raise ValueError("E08A train contract changed.")
    dev = config.section("dev")
    if (
        dev.get("full_size") != 721
        or dev.get("excluded_tuned_prefix_size") != 200
        or dev.get("evaluation_size") != 521
        or dev.get("answer_usage") != "later-scoring-only-never-retrieval"
    ):
        raise ValueError("E08A dev-521 contract changed.")
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
        raise ValueError("E08A retrieval contract changed.")
    execution = config.section("execution")
    if (
        execution.get("targets") != ["train5636", "dev521"]
        or execution.get("dense_devices") != ["cuda:0", "cuda:1"]
        or execution.get("dense_worker_count") != 2
        or execution.get("checkpoint_after_questions") != 1
    ):
        raise ValueError("E08A execution contract changed.")
    budget = config.section("parameter_budget")
    if (
        budget.get("future_stack_total")
        != budget.get("embedding")
        + budget.get("generator_reserved")
        + budget.get("adapter_parameter_cap_reserved")
        or budget["future_stack_total"] >= budget["exclusive_limit"]
    ):
        raise ValueError("E08A future parameter budget failed.")
    contract = config.section("run_contract")
    if not contract or not all(value is True for value in contract.values()):
        raise ValueError("E08A run contract lost an invariant.")
    return config


def load_e08_targets(
    train_path: Path, dev_path: Path, config: E08ContextConfig
) -> dict[str, tuple[dict[str, str], list[str]]]:
    train_cfg, dev_cfg = config.section("train"), config.section("dev")
    if not train_path.is_file() or file_sha256(train_path) != train_cfg["sha256"]:
        raise E08ContextError("Pinned official train split is missing or changed.")
    if not dev_path.is_file() or file_sha256(dev_path) != dev_cfg["sha256"]:
        raise E08ContextError("Pinned dev split is missing or changed.")
    train_records = _load_question_records(train_path, answer_type=str)
    dev_records = _load_question_records(dev_path, answer_type=str)
    if len(train_records) != train_cfg["record_count"] or len(dev_records) != dev_cfg[
        "full_size"
    ]:
        raise E08ContextError("E08A split record counts changed.")

    train_ids = select_train_sample(
        train_records, seed=train_cfg["sample_seed"], size=train_cfg["sample_size"]
    )
    dev_ids = select_dev_sample(
        dev_records, seed=dev_cfg["sample_seed"], size=dev_cfg["full_size"]
    )
    prefix = dev_cfg["excluded_tuned_prefix_size"]
    evaluation_ids = dev_ids[prefix:]
    if (
        _ids_sha(train_ids) != train_cfg["sample_ids_sha256"]
        or _ids_sha(dev_ids) != dev_cfg["full_ids_sha256"]
        or _ids_sha(dev_ids[:prefix]) != dev_cfg["excluded_tuned_prefix_ids_sha256"]
        or _ids_sha(evaluation_ids) != dev_cfg["evaluation_ids_sha256"]
        or len(evaluation_ids) != dev_cfg["evaluation_size"]
    ):
        raise E08ContextError("E08A deterministic target order changed.")
    if set(train_ids) & set(evaluation_ids):
        raise E08ContextError("Train and E08 dev-521 question groups overlap.")
    train_questions = {question_id: train_records[question_id] for question_id in train_ids}
    dev_questions = {
        question_id: dev_records[question_id] for question_id in evaluation_ids
    }
    return {
        "train5636": (train_questions, train_ids),
        "dev521": (dev_questions, evaluation_ids),
    }


def validate_e08_context_preflight(
    *,
    e00_directory: Path,
    dense_directory: Path,
    selection_directory: Path,
    train_path: Path,
    dev_path: Path,
    config: E08ContextConfig,
) -> dict[str, Any]:
    targets = load_e08_targets(train_path, dev_path, config)
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
        raise E08ContextError("Pinned E00 v2 artifact is missing or changed.")

    dense_cfg = config.section("source_dense")
    dense_manifest_path = dense_directory / "manifest.json"
    dense_path = dense_directory / "dense.faiss"
    mapping_path = dense_directory / "chunk_ids.jsonl"
    if not all(path.is_file() for path in (dense_manifest_path, dense_path, mapping_path)):
        raise E08ContextError("Pinned AITeam dense artifact is incomplete.")
    dense_manifest = json.loads(dense_manifest_path.read_text(encoding="utf-8"))
    if (
        dense_manifest.get("record_count") != dense_cfg["record_count"]
        or dense_manifest.get("dimension") != dense_cfg["dimension"]
        or dense_manifest.get("candidate", {}).get("key") != dense_cfg["candidate"]
        or file_sha256(dense_path) != dense_cfg["dense_faiss_sha256"]
        or file_sha256(mapping_path) != dense_cfg["chunk_ids_sha256"]
    ):
        raise E08ContextError("Pinned AITeam dense artifact changed.")

    advancement = config.section("advancement_source")
    report_path = selection_directory / "report.json"
    results_path = selection_directory / "generation" / "results.jsonl"
    scores_path = selection_directory / "generation" / "scores.jsonl"
    if not all(path.is_file() for path in (report_path, results_path, scores_path)):
        raise E08ContextError("Saved E07G advancement artifact is incomplete.")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    _validate_model_selection_report(report, advancement)
    if (
        file_sha256(results_path) != advancement["generation_results_sha256"]
        or file_sha256(scores_path) != advancement["scores_sha256"]
    ):
        raise E08ContextError("Saved E07G advancement evidence changed.")

    return {
        "config_sha256": config.config_sha256,
        "train_sha256": config.section("train")["sha256"],
        "train_ids_sha256": _ids_sha(targets["train5636"][1]),
        "train_size": len(targets["train5636"][1]),
        "dev_sha256": config.section("dev")["sha256"],
        "dev521_ids_sha256": _ids_sha(targets["dev521"][1]),
        "dev521_size": len(targets["dev521"][1]),
        "e00_manifest_sha256": e00_cfg["manifest_sha256"],
        "bm25_sha256": e00_cfg["bm25_sha256"],
        "dense_faiss_sha256": dense_cfg["dense_faiss_sha256"],
        "advancement_generation_results_sha256": advancement[
            "generation_results_sha256"
        ],
        "future_stack_parameters": config.section("parameter_budget")[
            "future_stack_total"
        ],
    }


def prepare_local_bm25(
    *, source: Path, runtime_directory: Path, config: E08ContextConfig
) -> Path:
    return _prepare_local_bm25(
        source=source, runtime_directory=runtime_directory, config=config
    )


def run_sparse_target(
    *,
    target: str,
    bm25_path: Path,
    train_path: Path,
    dev_path: Path,
    output_directory: Path,
    config: E08ContextConfig,
) -> dict[str, Any]:
    targets = load_e08_targets(train_path, dev_path, config)
    if target not in targets:
        raise E08ContextError(f"Unknown E08A retrieval target: {target}")
    questions, ids = targets[target]
    identity = {
        "code_version": CODE_VERSION,
        "stage": f"e08a-{target}-bm25",
        "target": target,
        "config_sha256": config.config_sha256,
        "bm25_sha256": config.section("source_e00")["bm25_sha256"],
        "sample_ids_sha256": _ids_sha(ids),
    }
    identity["identity_sha256"] = _json_sha256(identity)
    root = output_directory / "retrieval" / target / "sparse"
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
                raise E08ContextError(f"Invalid E08A BM25 result: {question_id}")
            _atomic_json(
                records / f"{index:04d}.json",
                {
                    "target": target,
                    "question_id": question_id,
                    "sample_index": index,
                    "sparse_top_ids": result["ids"],
                    "sparse_scores": result["scores"],
                    "latency_ms": result["latency_ms"],
                },
            )
            _write_state(state_path, identity, index + 1, complete=False)
            LOGGER.info(
                "e08a_sparse_progress target=%s completed=%d total=%d question_id=%s",
                target,
                index + 1,
                len(ids),
                question_id,
            )
    _write_jsonl_from_records(root / "results.jsonl", records, len(ids))
    _write_state(state_path, identity, len(ids), complete=True)
    return {
        "target": target,
        "sample_size": len(ids),
        "results_sha256": file_sha256(root / "results.jsonl"),
    }


def load_embedding(config: E08ContextConfig, device: str) -> Any:
    return _load_embedding(config, device)


def run_dense_worker(
    *,
    worker_rank: int,
    device: str,
    model: Any,
    dense_directory: Path,
    bm25_path: Path,
    train_path: Path,
    dev_path: Path,
    output_directory: Path,
    config: E08ContextConfig,
) -> dict[str, Any]:
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover
        raise E08ContextError("Install numpy for E08A dense retrieval.") from exc
    execution = config.section("execution")
    devices = execution["dense_devices"]
    if (
        worker_rank not in range(execution["dense_worker_count"])
        or device != devices[worker_rank]
    ):
        raise E08ContextError("E08A dense worker rank/device assignment changed.")
    targets = load_e08_targets(train_path, dev_path, config)
    dense_cfg = config.section("source_dense")
    retrieval = config.section("retrieval")
    mapping = _load_mapping(
        dense_directory / "chunk_ids.jsonl", dense_cfg["record_count"]
    )
    search = _TorchGpuFlatIpIndex(
        dense_directory / "dense.faiss",
        device=device,
        expected_count=dense_cfg["record_count"],
        expected_dimension=dense_cfg["dimension"],
    )
    store = _ChunkStore(bm25_path)
    summaries: dict[str, Any] = {}
    try:
        for target in config.section("execution")["targets"]:
            questions, ids = targets[target]
            summaries[target] = _run_dense_worker_target(
                target=target,
                worker_rank=worker_rank,
                device=device,
                questions=questions,
                ids=ids,
                model=model,
                mapping=mapping,
                search=search,
                store=store,
                output_directory=output_directory,
                config=config,
                np=np,
                retrieval=retrieval,
                dense_cfg=dense_cfg,
            )
    finally:
        search.close()
        store.close()
    return summaries


def _run_dense_worker_target(
    *,
    target: str,
    worker_rank: int,
    device: str,
    questions: dict[str, str],
    ids: list[str],
    model: Any,
    mapping: list[str],
    search: Any,
    store: Any,
    output_directory: Path,
    config: E08ContextConfig,
    np: Any,
    retrieval: dict[str, Any],
    dense_cfg: dict[str, Any],
) -> dict[str, Any]:
    sparse_path = output_directory / "retrieval" / target / "sparse" / "results.jsonl"
    sparse = _read_jsonl(sparse_path)
    if [row.get("question_id") for row in sparse] != ids:
        raise E08ContextError(f"E08A sparse cache is incomplete or reordered: {target}")
    identity = _dense_worker_identity(
        target=target,
        worker_rank=worker_rank,
        device=device,
        ids=ids,
        sparse_path=sparse_path,
        dense_cfg=dense_cfg,
        config=config,
    )
    root = output_directory / "retrieval" / target
    records = root / "records"
    records.mkdir(parents=True, exist_ok=True)
    state_path = root / f"dense-worker-{worker_rank}-state.json"
    worker_count = config.section("execution")["dense_worker_count"]
    assigned_indices = list(range(worker_rank, len(ids), worker_count))
    completed = _load_worker_progress(
        records=records,
        state_path=state_path,
        identity=identity,
        assigned_indices=assigned_indices,
        sample_ids=ids,
    )
    batch_size = config.section("execution")["dense_query_batch_size"]
    for start in range(completed, len(assigned_indices), batch_size):
        stop = min(start + batch_size, len(assigned_indices))
        batch_indices = assigned_indices[start:stop]
        batch_ids = [ids[index] for index in batch_indices]
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
            raise E08ContextError(f"Invalid E08A query embeddings: {target}")
        dense_scores, dense_rows = search.search(
            vectors, retrieval["candidate_k_per_branch"]
        )
        for offset, (index, question_id) in enumerate(zip(batch_indices, batch_ids)):
            dense_ids = [mapping[int(row)] for row in dense_rows[offset] if int(row) >= 0]
            fused = weighted_rrf(
                {"sparse": sparse[index]["sparse_top_ids"], "dense": dense_ids},
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
                raise E08ContextError(f"Invalid E08A selected contexts: {question_id}")
            _atomic_json(
                records / f"{index:04d}.json",
                {
                    "target": target,
                    "question_id": question_id,
                    "sample_index": index,
                    "sparse_top_ids": sparse[index]["sparse_top_ids"],
                    "dense_top_ids": dense_ids,
                    "dense_scores": [
                        float(value) for value in dense_scores[offset][: len(dense_ids)]
                    ],
                    "fused": fused,
                    "contexts": selected,
                    "answer_included": False,
                    "worker_rank": worker_rank,
                    "worker_identity_sha256": identity["identity_sha256"],
                },
            )
            _write_worker_state(
                state_path,
                identity,
                start + offset + 1,
                len(assigned_indices),
            )
        LOGGER.info(
            "e08a_dense_progress target=%s worker=%d device=%s "
            "worker_completed=%d worker_total=%d global_completed_at_least=%d total=%d",
            target,
            worker_rank,
            device,
            stop,
            len(assigned_indices),
            min(len(ids), stop * worker_count),
            len(ids),
        )
    _write_worker_state(
        state_path, identity, len(assigned_indices), len(assigned_indices)
    )
    return {
        "target": target,
        "worker_rank": worker_rank,
        "device": device,
        "assigned_count": len(assigned_indices),
        "complete": True,
    }


def _dense_worker_identity(
    *,
    target: str,
    worker_rank: int,
    device: str,
    ids: list[str],
    sparse_path: Path,
    dense_cfg: dict[str, Any],
    config: E08ContextConfig,
) -> dict[str, Any]:
    identity = {
        "code_version": CODE_VERSION,
        "stage": f"e08a-{target}-aiteam-rrf-top12-dual-gpu",
        "target": target,
        "worker_rank": worker_rank,
        "device": device,
        "worker_count": config.section("execution")["dense_worker_count"],
        "config_sha256": config.config_sha256,
        "sparse_results_sha256": file_sha256(sparse_path),
        "dense_faiss_sha256": dense_cfg["dense_faiss_sha256"],
        "sample_ids_sha256": _ids_sha(ids),
    }
    identity["identity_sha256"] = _json_sha256(identity)
    return identity


def _merge_dense_target(
    *,
    target: str,
    ids: list[str],
    output_directory: Path,
    config: E08ContextConfig,
) -> None:
    root = output_directory / "retrieval" / target
    sparse_path = root / "sparse" / "results.jsonl"
    records = root / "records"
    dense_cfg = config.section("source_dense")
    execution = config.section("execution")
    expected_identities: dict[int, dict[str, Any]] = {}
    worker_state_hashes: list[str] = []
    for worker_rank, device in enumerate(execution["dense_devices"]):
        identity = _dense_worker_identity(
            target=target,
            worker_rank=worker_rank,
            device=device,
            ids=ids,
            sparse_path=sparse_path,
            dense_cfg=dense_cfg,
            config=config,
        )
        expected_identities[worker_rank] = identity
        assigned_count = len(range(worker_rank, len(ids), execution["dense_worker_count"]))
        state_path = root / f"dense-worker-{worker_rank}-state.json"
        if not state_path.is_file():
            raise E08ContextError(f"Missing E08A dense worker state: {target}/{worker_rank}")
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if (
            state.get("run_identity") != identity
            or state.get("completed_count") != assigned_count
            or state.get("assigned_count") != assigned_count
            or state.get("complete") is not True
        ):
            raise E08ContextError(f"Incomplete E08A dense worker: {target}/{worker_rank}")
        worker_state_hashes.append(file_sha256(state_path))
    record_files = sorted(records.glob("*.json")) if records.is_dir() else []
    if len(record_files) != len(ids):
        raise E08ContextError(f"E08A dense records are incomplete: {target}")
    for index, question_id in enumerate(ids):
        path = records / f"{index:04d}.json"
        if not path.is_file():
            raise E08ContextError(f"Missing E08A dense record: {target}/{index}")
        row = json.loads(path.read_text(encoding="utf-8"))
        worker_rank = index % execution["dense_worker_count"]
        if (
            row.get("sample_index") != index
            or row.get("question_id") != question_id
            or row.get("worker_rank") != worker_rank
            or row.get("worker_identity_sha256")
            != expected_identities[worker_rank]["identity_sha256"]
        ):
            raise E08ContextError(f"E08A dense record identity changed: {target}/{index}")
    results_path = root / "results.jsonl"
    _write_jsonl_from_records(results_path, records, len(ids))
    merged_identity = {
        "code_version": CODE_VERSION,
        "stage": f"e08a-{target}-dense-merge",
        "target": target,
        "config_sha256": config.config_sha256,
        "sample_ids_sha256": _ids_sha(ids),
        "worker_state_sha256": worker_state_hashes,
        "results_sha256": file_sha256(results_path),
    }
    merged_identity["identity_sha256"] = _json_sha256(merged_identity)
    _write_state(root / "state.json", merged_identity, len(ids), complete=True)


def finalize_e08_context(
    *,
    train_path: Path,
    dev_path: Path,
    output_directory: Path,
    config: E08ContextConfig,
) -> dict[str, Any]:
    targets = load_e08_targets(train_path, dev_path, config)
    files: dict[str, Any] = {}
    target_reports: dict[str, Any] = {}
    for target in config.section("execution")["targets"]:
        _, ids = targets[target]
        _merge_dense_target(
            target=target,
            ids=ids,
            output_directory=output_directory,
            config=config,
        )
        root = output_directory / "retrieval" / target
        sparse_path = root / "sparse" / "results.jsonl"
        dense_path = root / "results.jsonl"
        sparse_state_path = root / "sparse" / "state.json"
        dense_state_path = root / "state.json"
        if not all(
            path.is_file()
            for path in (sparse_path, dense_path, sparse_state_path, dense_state_path)
        ):
            raise E08ContextError(f"E08A target output is incomplete: {target}")
        sparse_state = json.loads(sparse_state_path.read_text(encoding="utf-8"))
        dense_state = json.loads(dense_state_path.read_text(encoding="utf-8"))
        if (
            sparse_state.get("complete") is not True
            or dense_state.get("complete") is not True
            or sparse_state.get("completed_count") != len(ids)
            or dense_state.get("completed_count") != len(ids)
        ):
            raise E08ContextError(f"E08A checkpoint is incomplete: {target}")
        rows = _read_jsonl(dense_path)
        if len(rows) != len(ids):
            raise E08ContextError(f"E08A result count changed: {target}")
        context_counts = []
        for index, (question_id, row) in enumerate(zip(ids, rows)):
            contexts = row.get("contexts")
            if (
                row.get("sample_index") != index
                or row.get("question_id") != question_id
                or row.get("target") != target
                or row.get("answer_included") is not False
                or "answer" in row
                or not isinstance(contexts, list)
                or not 0 < len(contexts) <= config.section("retrieval")[
                    "selected_contexts"
                ]
            ):
                raise E08ContextError(f"Invalid E08A result row: {target}/{index}")
            context_counts.append(len(contexts))
        target_reports[target] = {
            "sample_size": len(ids),
            "sample_ids_sha256": _ids_sha(ids),
            "minimum_contexts": min(context_counts),
            "maximum_contexts": max(context_counts),
            "answers_included": False,
        }
        for relative, path in (
            (f"retrieval/{target}/sparse/results.jsonl", sparse_path),
            (f"retrieval/{target}/results.jsonl", dense_path),
        ):
            files[relative] = {"bytes": path.stat().st_size, "sha256": file_sha256(path)}
    report = {
        "schema_version": "1.0",
        "experiment_id": config.raw["experiment_id"],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "targets": target_reports,
        "retrieval": config.section("retrieval"),
        "files": files,
        "evidence": {
            "config_sha256": config.config_sha256,
            "train_sha256": config.section("train")["sha256"],
            "dev_sha256": config.section("dev")["sha256"],
            "e00_manifest_sha256": config.section("source_e00")["manifest_sha256"],
            "bm25_sha256": config.section("source_e00")["bm25_sha256"],
            "dense_faiss_sha256": config.section("source_dense")[
                "dense_faiss_sha256"
            ],
            "advancement_results_sha256": config.section("advancement_source")[
                "generation_results_sha256"
            ],
        },
        "train_answers_used_by_retrieval": False,
        "dev_answers_used_by_retrieval": False,
        "holdout_untouched": True,
        "public_read": False,
    }
    _atomic_json(output_directory / "report.json", report)
    return report


def _load_question_records(path: Path, *, answer_type: type) -> dict[str, str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not payload:
        raise E08ContextError(f"Question split must be a non-empty mapping: {path}")
    questions: dict[str, str] = {}
    for raw_id, record in payload.items():
        question_id = str(raw_id)
        if (
            not isinstance(record, dict)
            or set(record) != {"question", "answer"}
            or not isinstance(record["question"], str)
            or not record["question"].strip()
            or not isinstance(record["answer"], answer_type)
        ):
            raise E08ContextError(f"Invalid E08A question record: {question_id}")
        questions[question_id] = record["question"]
    return questions


def _ids_sha(ids: list[str]) -> str:
    return hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()
