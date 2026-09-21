"""Shared deterministic private retrieval and parent-context preparation."""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean
from typing import Any
from concurrent.futures import ThreadPoolExecutor

from .bm25 import SqliteBm25Index
from .corpus import file_sha256
from .e02_answer import _atomic_json, _atomic_jsonl
from .e02_compare import _ChunkStore, _TorchGpuFlatIpIndex, _load_mapping, weighted_rrf
from .e03_rrf_grid import (
    _json_sha256,
    _load_contiguous,
    _load_worker_progress,
    _read_jsonl,
    _write_jsonl_from_records,
    _write_state,
    _write_worker_state,
)
from .e07_full_dev import load_embedding as _load_embedding
from .e18_source_metadata import enrich_context, scan_selected
from .e21_parent_context import article_blocks, seed_unit
from .final_public import prepare_local_bm25 as _prepare_local_bm25


EXPERIMENT = "FINAL-private-p00-retrieval-parent-v1"
CODE_VERSION = "1.0.0"
LOG = logging.getLogger(__name__)


class PrivateRetrievalError(RuntimeError):
    """Raised when private retrieval cannot proceed without changing identity."""


@dataclass(frozen=True)
class Config:
    raw: dict[str, Any]
    path: Path

    @property
    def sha(self) -> str:
        return file_sha256(self.path)

    @property
    def config_sha256(self) -> str:
        """Compatibility alias used by the frozen BM25 cache helper."""
        return self.sha

    def section(self, key: str) -> dict[str, Any]:
        value = self.raw.get(key)
        if not isinstance(value, dict):
            raise PrivateRetrievalError(f"Missing config section: {key}")
        return value


def load_config(path: Path) -> Config:
    raw = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "schema_version", "experiment_id", "source_e00", "metadata_source",
        "source_dense", "retrieval", "context_policy", "execution",
        "parameter_budget", "run_contract",
    }
    if set(raw) != required or raw.get("schema_version") != "1.0" or raw.get("experiment_id") != EXPERIMENT:
        raise PrivateRetrievalError("P00 config root changed.")
    retrieval = raw["retrieval"]
    if retrieval != {
        "candidate_k_per_branch": 40, "rrf_constant": 60,
        "sparse_weight": 0.5, "dense_weight": 0.5,
        "fused_top_k": 20, "selected_contexts": 12,
        "normalize_query_embeddings": True,
        "dense_search_backend": "torch-cuda-exact-flat-ip-float32-batched",
        "reranker": None,
    }:
        raise PrivateRetrievalError("P00 retrieval policy changed.")
    execution = raw["execution"]
    if (execution.get("dense_devices") != ["cuda:0", "cuda:1"]
            or execution.get("dense_worker_count") != 2
            or execution.get("checkpoint_after_questions") != 1):
        raise PrivateRetrievalError("P00 execution/checkpoint policy changed.")
    policy = raw["context_policy"]
    if (policy.get("seed_contexts") != 12 or policy.get("max_input_tokens") != 8192
            or policy.get("evict_seed_for_expansion") is not False):
        raise PrivateRetrievalError("P00 parent-context policy changed.")
    budget = raw["parameter_budget"]
    if (budget.get("maximum_stack_total") != budget.get("embedding") + budget.get("downstream_generator") + budget.get("adapter_parameter_cap")
            or budget["maximum_stack_total"] >= budget["exclusive_limit"]):
        raise PrivateRetrievalError("P00 downstream stack exceeds the BTC limit.")
    if not raw["run_contract"] or not all(value is True for value in raw["run_contract"].values()):
        raise PrivateRetrievalError("P00 run contract lost an invariant.")
    return Config(raw, path)


def _ids_sha(ids: list[str]) -> str:
    return hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()


def load_private_questions(path: Path) -> tuple[dict[str, str], list[str], dict[str, Any]]:
    if not path.is_file():
        raise PrivateRetrievalError(f"Private question file is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not payload:
        raise PrivateRetrievalError("Private file must be a non-empty question-ID mapping.")
    questions: dict[str, str] = {}
    for raw_id, record in payload.items():
        qid = str(raw_id)
        if not isinstance(record, dict) or set(record) not in ({"question"}, {"question", "answer"}):
            raise PrivateRetrievalError(f"Invalid private record schema: {qid}")
        question = record.get("question")
        if not isinstance(question, str) or not question.strip():
            raise PrivateRetrievalError(f"Empty private question: {qid}")
        if "answer" in record and record["answer"] is not None:
            raise PrivateRetrievalError(f"Private reference answer is present and must not be read: {qid}")
        if qid in questions:
            raise PrivateRetrievalError(f"Duplicate private question ID: {qid}")
        questions[qid] = question
    ids = sorted(questions)
    normalized = [" ".join(unicodedata.normalize("NFC", questions[qid]).casefold().split()) for qid in ids]
    identity = {
        "private_sha256": file_sha256(path),
        "sample_ids_sha256": _ids_sha(ids),
        "question_text_sha256": _json_sha256({qid: questions[qid] for qid in ids}),
        "sample_size": len(ids),
        "normalized_duplicate_questions": len(normalized) - len(set(normalized)),
        "answers_read": False,
    }
    return questions, ids, identity


def code_sha(root: Path) -> str:
    paths = [
        root / "src/uit_dsc_fixed_rag/final_private_p00.py",
        root / "scripts/run_final_private_p00_kaggle.py",
    ]
    return _json_sha256({p.relative_to(root).as_posix(): file_sha256(p) for p in paths})


def validate_preflight(*, root: Path, e00: Path, dense: Path, private: Path, config: Config) -> dict[str, Any]:
    _, _, private_identity = load_private_questions(private)
    e00_cfg = config.section("source_e00")
    metadata = config.section("metadata_source")
    required_e00 = {
        "manifest.json": e00_cfg["manifest_sha256"],
        "bm25.sqlite3": e00_cfg["bm25_sha256"],
        metadata["chunks_path"]: metadata["chunks_sha256"],
        metadata["documents_path"]: metadata["documents_sha256"],
    }
    for name, expected in required_e00.items():
        path = e00 / name
        if not path.is_file() or file_sha256(path) != expected:
            raise PrivateRetrievalError(f"Pinned E00 file is missing or changed: {name}")
    if (e00 / "bm25.sqlite3").stat().st_size != e00_cfg["bm25_bytes"]:
        raise PrivateRetrievalError("Pinned E00 BM25 size changed.")
    dense_cfg = config.section("source_dense")
    manifest_path, index_path, mapping_path = dense / "manifest.json", dense / "dense.faiss", dense / "chunk_ids.jsonl"
    if not all(path.is_file() for path in (manifest_path, index_path, mapping_path)):
        raise PrivateRetrievalError("Pinned dense artifact is incomplete.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (manifest.get("record_count") != dense_cfg["record_count"]
            or manifest.get("dimension") != dense_cfg["dimension"]
            or manifest.get("candidate", {}).get("key") != dense_cfg["candidate"]
            or file_sha256(index_path) != dense_cfg["dense_faiss_sha256"]
            or file_sha256(mapping_path) != dense_cfg["chunk_ids_sha256"]):
        raise PrivateRetrievalError("Pinned dense artifact changed.")
    return {
        "experiment_id": EXPERIMENT, "code_sha256": code_sha(root),
        "config_sha256": config.sha, **private_identity,
        "e00_manifest_sha256": e00_cfg["manifest_sha256"],
        "bm25_sha256": e00_cfg["bm25_sha256"],
        "dense_faiss_sha256": dense_cfg["dense_faiss_sha256"],
        "retrieval": config.section("retrieval"),
        "context_policy": config.section("context_policy"),
        "maximum_stack_parameters": config.section("parameter_budget")["maximum_stack_total"],
    }


def check_preflight(*, root: Path, output: Path, private: Path, config: Config) -> tuple[dict[str, str], list[str], dict[str, Any]]:
    questions, ids, private_identity = load_private_questions(private)
    path = output / "preflight.json"
    if not path.is_file():
        raise PrivateRetrievalError("Run P00 preflight first.")
    payload = json.loads(path.read_text(encoding="utf-8"))
    evidence = payload.get("evidence", {})
    expected = {
        "code_sha256": code_sha(root), "config_sha256": config.sha,
        "private_sha256": private_identity["private_sha256"],
        "sample_ids_sha256": private_identity["sample_ids_sha256"],
        "question_text_sha256": private_identity["question_text_sha256"],
        "sample_size": private_identity["sample_size"],
    }
    if payload.get("experiment_id") != EXPERIMENT or any(evidence.get(k) != v for k, v in expected.items()):
        raise PrivateRetrievalError("P00 preflight/checkpoint identity changed.")
    return questions, ids, evidence


def prepare_local_bm25(*, source: Path, runtime_directory: Path, config: Config) -> Path:
    return _prepare_local_bm25(source=source, runtime_directory=runtime_directory, config=config)


def run_sparse(*, root: Path, bm25_path: Path, private: Path, output: Path, config: Config) -> dict[str, Any]:
    questions, ids, evidence = check_preflight(root=root, output=output, private=private, config=config)
    identity = {
        "code_version": CODE_VERSION, "stage": "private-p00-bm25",
        "code_sha256": evidence["code_sha256"], "config_sha256": config.sha,
        "bm25_sha256": config.section("source_e00")["bm25_sha256"],
        "private_sha256": evidence["private_sha256"], "sample_ids_sha256": _ids_sha(ids),
    }
    identity["identity_sha256"] = _json_sha256(identity)
    stage = output / "retrieval/sparse"
    records, state = stage / "records", stage / "state.json"
    records.mkdir(parents=True, exist_ok=True)
    completed = _load_contiguous(records, state, identity, ids)
    pending = ids[completed:]
    local = threading.local()
    depth = config.section("retrieval")["candidate_k_per_branch"]

    def search(qid: str) -> dict[str, Any]:
        index = getattr(local, "index", None)
        if index is None:
            index = SqliteBm25Index(bm25_path)
            local.index = index
        started = time.perf_counter()
        hits = index.search(questions[qid], top_k=depth)
        return {"ids": [hit.chunk_id for hit in hits], "scores": [float(hit.score) for hit in hits],
                "latency_ms": (time.perf_counter() - started) * 1000}

    with ThreadPoolExecutor(max_workers=config.section("execution")["sparse_worker_threads"]) as executor:
        futures = [executor.submit(search, qid) for qid in pending]
        for offset, (qid, future) in enumerate(zip(pending, futures)):
            index, result = completed + offset, future.result()
            if not result["ids"] or len(result["ids"]) > depth:
                raise PrivateRetrievalError(f"Invalid BM25 result: {qid}")
            _atomic_json(records / f"{index:04d}.json", {
                "question_id": qid, "sample_index": index,
                "sparse_top_ids": result["ids"], "sparse_scores": result["scores"],
                "latency_ms": result["latency_ms"],
            })
            _write_state(state, identity, index + 1, complete=False)
            LOG.info("p00_sparse completed=%d total=%d qid=%s", index + 1, len(ids), qid)
    _write_records(stage / "results.jsonl", records, len(ids))
    _write_state(state, identity, len(ids), complete=True)
    return {"sample_size": len(ids), "results_sha256": file_sha256(stage / "results.jsonl")}


def load_embedding(config: Config, device: str) -> Any:
    return _load_embedding(config, device)


def _dense_identity(*, rank: int, device: str, ids: list[str], sparse_path: Path, evidence: dict[str, Any], config: Config) -> dict[str, Any]:
    identity = {
        "code_version": CODE_VERSION, "stage": "private-p00-aiteam-rrf",
        "worker_rank": rank, "device": device,
        "worker_count": config.section("execution")["dense_worker_count"],
        "code_sha256": evidence["code_sha256"], "config_sha256": config.sha,
        "private_sha256": evidence["private_sha256"], "sample_ids_sha256": _ids_sha(ids),
        "sparse_results_sha256": file_sha256(sparse_path),
        "dense_faiss_sha256": config.section("source_dense")["dense_faiss_sha256"],
    }
    identity["identity_sha256"] = _json_sha256(identity)
    return identity


def run_dense_worker(*, root: Path, rank: int, device: str, model: Any, dense: Path,
                     bm25_path: Path, private: Path, output: Path, config: Config) -> dict[str, Any]:
    import numpy as np
    questions, ids, evidence = check_preflight(root=root, output=output, private=private, config=config)
    execution = config.section("execution")
    if rank not in range(execution["dense_worker_count"]) or device != execution["dense_devices"][rank]:
        raise PrivateRetrievalError("Dense worker rank/device mismatch.")
    sparse_path = output / "retrieval/sparse/results.jsonl"
    sparse = _read_jsonl(sparse_path)
    if [row.get("question_id") for row in sparse] != ids:
        raise PrivateRetrievalError("Sparse cache is incomplete or reordered.")
    dense_cfg, retrieval = config.section("source_dense"), config.section("retrieval")
    identity = _dense_identity(rank=rank, device=device, ids=ids, sparse_path=sparse_path, evidence=evidence, config=config)
    stage, records = output / "retrieval", output / "retrieval/records"
    state = stage / f"dense-worker-{rank}-state.json"
    records.mkdir(parents=True, exist_ok=True)
    assigned = list(range(rank, len(ids), execution["dense_worker_count"]))
    completed = _load_worker_progress(records=records, state_path=state, identity=identity,
                                      assigned_indices=assigned, sample_ids=ids)
    mapping = _load_mapping(dense / "chunk_ids.jsonl", dense_cfg["record_count"])
    search = _TorchGpuFlatIpIndex(dense / "dense.faiss", device=device,
                                  expected_count=dense_cfg["record_count"], expected_dimension=dense_cfg["dimension"])
    store = _ChunkStore(bm25_path)
    batch = execution["dense_query_batch_size"]
    try:
        for start in range(completed, len(assigned), batch):
            subset = assigned[start:min(start + batch, len(assigned))]
            qids = [ids[index] for index in subset]
            vectors = model.encode([dense_cfg["query_prefix"] + questions[qid] for qid in qids],
                                   batch_size=len(qids), show_progress_bar=False,
                                   convert_to_numpy=True, normalize_embeddings=True)
            vectors = np.asarray(vectors, dtype=np.float32)
            if vectors.shape != (len(qids), dense_cfg["dimension"]) or not np.isfinite(vectors).all():
                raise PrivateRetrievalError("Invalid private query embeddings.")
            scores, rows = search.search(vectors, retrieval["candidate_k_per_branch"])
            for offset, (index, qid) in enumerate(zip(subset, qids)):
                dense_ids = [mapping[int(row)] for row in rows[offset] if int(row) >= 0]
                fused = weighted_rrf(
                    {"sparse": sparse[index]["sparse_top_ids"], "dense": dense_ids},
                    weights={"sparse": retrieval["sparse_weight"], "dense": retrieval["dense_weight"]},
                    constant=retrieval["rrf_constant"], top_k=retrieval["fused_top_k"],
                )
                contexts = store.fetch([item["chunk_id"] for item in fused[:retrieval["selected_contexts"]]])
                if len(contexts) != retrieval["selected_contexts"]:
                    raise PrivateRetrievalError(f"Expected 12 contexts: {qid}")
                row = {
                    "question_id": qid, "sample_index": index,
                    "sparse_top_ids": sparse[index]["sparse_top_ids"], "dense_top_ids": dense_ids,
                    "dense_scores": [float(value) for value in scores[offset][:len(dense_ids)]],
                    "fused_pool": fused, "contexts": contexts, "answers_used": False,
                    "worker_rank": rank, "worker_identity_sha256": identity["identity_sha256"],
                }
                _atomic_json(records / f"{index:04d}.json", row)
                _write_worker_state(state, identity, start + offset + 1, len(assigned))
                LOG.info("p00_dense worker=%d completed=%d total=%d qid=%s", rank, start + offset + 1, len(assigned), qid)
    finally:
        search.close(); store.close()
    _write_worker_state(state, identity, len(assigned), len(assigned))
    return {"worker_rank": rank, "completed": len(assigned), "device": device}


def _write_records(path: Path, records: Path, count: int) -> None:
    _atomic_jsonl(path, [json.loads((records / f"{index:04d}.json").read_text(encoding="utf-8")) for index in range(count)])


def _validate_dense_workers(*, ids: list[str], evidence: dict[str, Any], output: Path, config: Config) -> list[str]:
    sparse_path, records = output / "retrieval/sparse/results.jsonl", output / "retrieval/records"
    identities = []
    for rank, device in enumerate(config.section("execution")["dense_devices"]):
        identity = _dense_identity(rank=rank, device=device, ids=ids, sparse_path=sparse_path, evidence=evidence, config=config)
        state_path = output / f"retrieval/dense-worker-{rank}-state.json"
        state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.is_file() else {}
        assigned = list(range(rank, len(ids), 2))
        if (state.get("run_identity") != identity or state.get("complete") is not True
                or state.get("completed_count") != len(assigned) or state.get("assigned_count") != len(assigned)):
            raise PrivateRetrievalError(f"Incomplete dense worker: {rank}")
        identities.append(identity["identity_sha256"])
    for index, qid in enumerate(ids):
        path = records / f"{index:04d}.json"
        if not path.is_file():
            raise PrivateRetrievalError(f"Missing dense record: {index}")
        row = json.loads(path.read_text(encoding="utf-8"))
        rank = index % 2
        if (row.get("question_id") != qid or row.get("sample_index") != index
                or row.get("worker_rank") != rank or row.get("worker_identity_sha256") != identities[rank]
                or row.get("answers_used") is not False or "answer" in row):
            raise PrivateRetrievalError(f"Dense record identity changed: {index}")
    return identities


def finalize(*, root: Path, e00: Path, private: Path, output: Path, config: Config) -> dict[str, Any]:
    questions, ids, evidence = check_preflight(root=root, output=output, private=private, config=config)
    identities = _validate_dense_workers(ids=ids, evidence=evidence, output=output, config=config)
    records = output / "retrieval/records"
    raw_path = output / "retrieval/raw-results.jsonl"
    _write_records(raw_path, records, len(ids))
    rows = _read_jsonl(raw_path)
    pool_path = output / "retrieval/candidate-pool-top20.jsonl"
    _atomic_jsonl(pool_path, [{"question_id": row["question_id"], "sample_index": row["sample_index"],
                              "fused_pool": row["fused_pool"], "answers_used": False} for row in rows])

    metadata = config.section("metadata_source")
    wanted = {context["chunk_id"] for row in rows for context in row["contexts"]}
    seeds = scan_selected(e00 / metadata["chunks_path"], "chunk_id", wanted, metadata["chunks_sha256"])
    document_ids = {seed["document_id"] for seed in seeds.values()}
    documents = scan_selected(e00 / metadata["documents_path"], "document_id", document_ids, metadata["documents_sha256"])
    by_document = {document_id: [] for document_id in document_ids}
    digest = hashlib.sha256()
    with (e00 / metadata["chunks_path"]).open("rb") as stream:
        for line in stream:
            digest.update(line)
            chunk = json.loads(line)
            if chunk["document_id"] in by_document:
                by_document[chunk["document_id"]].append(chunk)
    if digest.hexdigest() != metadata["chunks_sha256"]:
        raise PrivateRetrievalError("E00 chunks changed during parent scan.")
    block_by_chunk: dict[str, list[dict[str, Any]]] = {}
    for document_id, chunks in by_document.items():
        for block in article_blocks(chunks, documents[document_id]):
            for chunk in block:
                if chunk["chunk_id"] in wanted:
                    block_by_chunk[chunk["chunk_id"]] = block
    prepared = []
    for index, (qid, row) in enumerate(zip(ids, rows)):
        if row["question_id"] != qid or row["sample_index"] != index or len(row["contexts"]) != 12:
            raise PrivateRetrievalError(f"Raw retrieval row changed: {index}")
        units = []
        for rank, context in enumerate(row["contexts"]):
            seed = seeds[context["chunk_id"]]
            enrich_context(context, seed, documents[seed["document_id"]])
            units.append(seed_unit(seed, rank, block_by_chunk[seed["chunk_id"]],
                                   documents[seed["document_id"]], config.section("context_policy")))
        prepared.append({"question_id": qid, "sample_index": index, "answers_used": False, "units": units})
    prepared_path = output / "prepared/results.jsonl"
    _atomic_jsonl(prepared_path, prepared)
    expansion_counts = [sum(bool(unit["expansions"]) for unit in row["units"]) for row in prepared]
    report = {
        "schema_version": "1.0", "experiment_id": EXPERIMENT,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "sample_size": len(ids), "private_questions_sha256": evidence["private_sha256"],
        "sample_ids_sha256": _ids_sha(ids), "answers_used": False,
        "retrieval": config.section("retrieval"), "context_policy": config.section("context_policy"),
        "diagnostics": {
            "seed_contexts_per_question": 12,
            "candidate_pool_size_min": min(len(row["fused_pool"]) for row in rows),
            "candidate_pool_size_max": max(len(row["fused_pool"]) for row in rows),
            "questions_with_parent_expansion": sum(value > 0 for value in expansion_counts),
            "mean_expandable_seed_units": fmean(expansion_counts),
            "source_document_count": len(documents),
            "normalized_duplicate_questions": evidence["normalized_duplicate_questions"],
        },
        "files": {
            "retrieval/raw-results.jsonl": {"sha256": file_sha256(raw_path), "bytes": raw_path.stat().st_size},
            "retrieval/candidate-pool-top20.jsonl": {"sha256": file_sha256(pool_path), "bytes": pool_path.stat().st_size},
            "prepared/results.jsonl": {"sha256": file_sha256(prepared_path), "bytes": prepared_path.stat().st_size},
        },
        "evidence": {**evidence, "dense_worker_identity_sha256": identities},
        "private_reference_answers_read": False,
        "downstream_candidates_must_reuse_prepared_sha256": file_sha256(prepared_path),
    }
    _atomic_json(output / "report.json", report)
    _atomic_json(output / "identity.json", {
        "experiment_id": EXPERIMENT, "private_sha256": evidence["private_sha256"],
        "sample_ids_sha256": _ids_sha(ids), "config_sha256": config.sha,
        "code_sha256": code_sha(root), "raw_results_sha256": file_sha256(raw_path),
        "prepared_results_sha256": file_sha256(prepared_path),
    })
    return report
