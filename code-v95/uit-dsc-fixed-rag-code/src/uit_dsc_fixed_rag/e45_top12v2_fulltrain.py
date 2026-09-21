"""Deterministic full-train retrieval and two-stage legal top-12 selection."""
from __future__ import annotations

import hashlib
import json
import logging
import re
import threading
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean
from typing import Any

from .bm25 import SqliteBm25Index
from .corpus import file_sha256
from .e02_answer import _atomic_json, _atomic_jsonl
from .e02_compare import _ChunkStore, _TorchGpuFlatIpIndex, _load_mapping, weighted_rrf
from .e03_rrf_grid import (
    _json_sha256,
    _load_contiguous,
    _load_worker_progress,
    _read_jsonl,
    _write_state,
    _write_worker_state,
)
from .e07_full_dev import load_embedding as _load_embedding
from .e18_source_metadata import enrich_context, scan_selected
from .final_public import prepare_local_bm25 as _prepare_local_bm25


EXPERIMENT = "E45-top12v2-fulltrain7000-v1"
CODE_VERSION = "1.0.0"
LOG = logging.getLogger(__name__)

_ARTICLE = re.compile(r"\bđiều\s+([0-9]+[a-zđ]?)\b", re.IGNORECASE)
_CLAUSE = re.compile(r"\b(khoản\s+[0-9]+|điểm\s+[a-zđ])\b", re.IGNORECASE)
_TOKEN = re.compile(r"[0-9a-zà-ỹđ]+", re.IGNORECASE)
_IMPORTANT_STOPWORDS = {
    "theo", "quy", "định", "pháp", "luật", "được", "không", "những", "trường",
    "hợp", "nào", "như", "thế", "với", "của", "trong", "tại", "điều", "khoản",
    "điểm", "và", "hoặc", "khi", "có", "phải", "bao", "gồm", "về", "cho", "một",
}


class E45Error(RuntimeError):
    pass


@dataclass(frozen=True)
class Config:
    raw: dict[str, Any]
    path: Path

    @property
    def sha(self) -> str:
        return file_sha256(self.path)

    @property
    def config_sha256(self) -> str:
        return self.sha

    def section(self, key: str) -> dict[str, Any]:
        value = self.raw.get(key)
        if not isinstance(value, dict):
            raise E45Error(f"Missing E45 config section: {key}")
        return value


def load_config(path: Path) -> Config:
    raw = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "schema_version", "experiment_id", "training_source", "source_e00",
        "metadata_source", "source_dense", "retrieval", "selector", "execution",
        "parameter_budget", "run_contract",
    }
    if set(raw) != required or raw.get("schema_version") != "1.0" or raw.get("experiment_id") != EXPERIMENT:
        raise E45Error("E45 config root changed.")
    if raw["training_source"] != {
        "path": "data/raw/train.json",
        "sha256": "2a52501cc065d266f2f832475950bcf1e7c75c386efa9b2f568f251d745f5988",
        "record_count": 7000,
        "answer_usage": "generation-supervision-only-never-retrieval-selection",
    }:
        raise E45Error("E45 official training source changed.")
    retrieval = raw["retrieval"]
    if retrieval != {
        "candidate_k_per_branch": 40, "rrf_constant": 60,
        "sparse_weight": 0.5, "dense_weight": 0.5,
        "fused_top_k": 20, "selected_contexts": 12,
        "normalize_query_embeddings": True,
        "dense_search_backend": "torch-cuda-exact-flat-ip-float32-batched",
        "reranker": None,
    }:
        raise E45Error("E45 retrieval policy changed.")
    selector = raw["selector"]
    if (selector.get("name") != "rrf-top6-then-legal-priority-v2"
            or selector.get("locked_rrf_contexts") != 6
            or selector.get("selected_contexts") != 12
            or selector.get("weighted_bonus_score") is not False
            or selector.get("preserve_rrf_order_inside_each_priority") is not True):
        raise E45Error("E45 selector policy changed.")
    execution = raw["execution"]
    if (execution.get("dense_devices") != ["cuda:0", "cuda:1"]
            or execution.get("dense_worker_count") != 2
            or execution.get("checkpoint_after_questions") != 1):
        raise E45Error("E45 execution policy changed.")
    budget = raw["parameter_budget"]
    if (budget.get("maximum_stack_total")
            != budget.get("embedding") + budget.get("downstream_generator") + budget.get("adapter_parameter_cap")
            or budget["maximum_stack_total"] >= budget["exclusive_limit"]):
        raise E45Error("E45 model stack exceeds the BTC parameter limit.")
    if not raw["run_contract"] or not all(value is True for value in raw["run_contract"].values()):
        raise E45Error("E45 run contract lost an invariant.")
    return Config(raw, path)


def _fold(text: str) -> str:
    return " ".join(unicodedata.normalize("NFC", text).casefold().split())


def _ids_sha(ids: list[str]) -> str:
    return hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()


def load_official_train(path: Path, config: Config) -> tuple[dict[str, dict[str, str]], list[str]]:
    source = config.section("training_source")
    if not path.is_file() or file_sha256(path) != source["sha256"]:
        raise E45Error("Pinned official train.json is missing or changed.")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or len(payload) != source["record_count"]:
        raise E45Error("Official train.json record count changed.")
    records: dict[str, dict[str, str]] = {}
    for raw_id, value in payload.items():
        qid = str(raw_id)
        if (not isinstance(value, dict) or set(value) != {"question", "answer"}
                or not isinstance(value["question"], str) or not value["question"].strip()
                or not isinstance(value["answer"], str) or not value["answer"].strip()
                or qid in records):
            raise E45Error(f"Invalid official training record: {qid}")
        records[qid] = {"question": value["question"], "answer": value["answer"]}
    return records, sorted(records)


def code_sha(root: Path) -> str:
    paths = [
        root / "src/uit_dsc_fixed_rag/e45_top12v2_fulltrain.py",
        root / "scripts/run_e45_top12v2_fulltrain_kaggle.py",
    ]
    return _json_sha256({p.relative_to(root).as_posix(): file_sha256(p) for p in paths})


def validate_preflight(*, root: Path, e00: Path, dense: Path, train: Path, config: Config) -> dict[str, Any]:
    records, ids = load_official_train(train, config)
    e00_cfg, metadata = config.section("source_e00"), config.section("metadata_source")
    required_e00 = {
        "manifest.json": e00_cfg["manifest_sha256"],
        "bm25.sqlite3": e00_cfg["bm25_sha256"],
        metadata["chunks_path"]: metadata["chunks_sha256"],
        metadata["documents_path"]: metadata["documents_sha256"],
    }
    for name, expected in required_e00.items():
        path = e00 / name
        if not path.is_file() or file_sha256(path) != expected:
            raise E45Error(f"Pinned E00 file is missing or changed: {name}")
    if (e00 / "bm25.sqlite3").stat().st_size != e00_cfg["bm25_bytes"]:
        raise E45Error("Pinned E00 BM25 size changed.")
    dense_cfg = config.section("source_dense")
    manifest_path, index_path, mapping_path = dense / "manifest.json", dense / "dense.faiss", dense / "chunk_ids.jsonl"
    if not all(path.is_file() for path in (manifest_path, index_path, mapping_path)):
        raise E45Error("Pinned dense artifact is incomplete.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (manifest.get("record_count") != dense_cfg["record_count"]
            or manifest.get("dimension") != dense_cfg["dimension"]
            or manifest.get("candidate", {}).get("key") != dense_cfg["candidate"]
            or file_sha256(index_path) != dense_cfg["dense_faiss_sha256"]
            or file_sha256(mapping_path) != dense_cfg["chunk_ids_sha256"]):
        raise E45Error("Pinned dense artifact changed.")
    normalized = [_fold(records[qid]["question"]) for qid in ids]
    return {
        "experiment_id": EXPERIMENT, "code_sha256": code_sha(root), "config_sha256": config.sha,
        "train_sha256": file_sha256(train), "sample_size": len(ids), "sample_ids_sha256": _ids_sha(ids),
        "question_text_sha256": _json_sha256({qid: records[qid]["question"] for qid in ids}),
        "normalized_duplicate_questions": len(normalized) - len(set(normalized)),
        "e00_manifest_sha256": e00_cfg["manifest_sha256"], "bm25_sha256": e00_cfg["bm25_sha256"],
        "dense_faiss_sha256": dense_cfg["dense_faiss_sha256"],
        "answers_used_by_retrieval": False,
    }


def check_preflight(*, root: Path, output: Path, train: Path, config: Config) -> tuple[dict[str, dict[str, str]], list[str], dict[str, Any]]:
    records, ids = load_official_train(train, config)
    path = output / "preflight.json"
    if not path.is_file():
        raise E45Error("Run E45 preflight first.")
    payload = json.loads(path.read_text(encoding="utf-8"))
    evidence = payload.get("evidence", {})
    expected = {
        "code_sha256": code_sha(root), "config_sha256": config.sha,
        "train_sha256": file_sha256(train), "sample_size": len(ids), "sample_ids_sha256": _ids_sha(ids),
        "question_text_sha256": _json_sha256({qid: records[qid]["question"] for qid in ids}),
    }
    if payload.get("experiment_id") != EXPERIMENT or any(evidence.get(key) != value for key, value in expected.items()):
        raise E45Error("E45 preflight/checkpoint identity changed.")
    return records, ids, evidence


def prepare_local_bm25(*, source: Path, runtime_directory: Path, config: Config) -> Path:
    return _prepare_local_bm25(source=source, runtime_directory=runtime_directory, config=config)


def _write_records(path: Path, records: Path, count: int) -> None:
    _atomic_jsonl(path, [json.loads((records / f"{index:04d}.json").read_text(encoding="utf-8"))
                         for index in range(count)])


def run_sparse(*, root: Path, bm25_path: Path, train: Path, output: Path, config: Config) -> dict[str, Any]:
    records_by_id, ids, evidence = check_preflight(root=root, output=output, train=train, config=config)
    identity = {
        "code_version": CODE_VERSION, "stage": "e45-fulltrain-bm25",
        "code_sha256": evidence["code_sha256"], "config_sha256": config.sha,
        "train_sha256": evidence["train_sha256"], "sample_ids_sha256": _ids_sha(ids),
        "bm25_sha256": config.section("source_e00")["bm25_sha256"],
    }
    identity["identity_sha256"] = _json_sha256(identity)
    stage, record_dir = output / "retrieval/sparse", output / "retrieval/sparse/records"
    state = stage / "state.json"
    record_dir.mkdir(parents=True, exist_ok=True)
    completed = _load_contiguous(record_dir, state, identity, ids)
    pending = ids[completed:]
    local = threading.local()
    depth = config.section("retrieval")["candidate_k_per_branch"]

    def search(qid: str) -> dict[str, Any]:
        index = getattr(local, "index", None)
        if index is None:
            index = SqliteBm25Index(bm25_path); local.index = index
        started = time.perf_counter()
        hits = index.search(records_by_id[qid]["question"], top_k=depth)
        return {"ids": [hit.chunk_id for hit in hits], "scores": [float(hit.score) for hit in hits],
                "latency_ms": (time.perf_counter() - started) * 1000}

    with ThreadPoolExecutor(max_workers=config.section("execution")["sparse_worker_threads"]) as executor:
        futures = [executor.submit(search, qid) for qid in pending]
        for offset, (qid, future) in enumerate(zip(pending, futures)):
            index, result = completed + offset, future.result()
            if not result["ids"] or len(result["ids"]) > depth:
                raise E45Error(f"Invalid E45 BM25 result: {qid}")
            _atomic_json(record_dir / f"{index:04d}.json", {
                "question_id": qid, "sample_index": index, "sparse_top_ids": result["ids"],
                "sparse_scores": result["scores"], "latency_ms": result["latency_ms"],
            })
            _write_state(state, identity, index + 1, complete=False)
            LOG.info("e45_sparse completed=%d total=%d qid=%s", index + 1, len(ids), qid)
    result_path = stage / "results.jsonl"
    _write_records(result_path, record_dir, len(ids))
    _write_state(state, identity, len(ids), complete=True)
    return {"sample_size": len(ids), "results_sha256": file_sha256(result_path)}


def load_embedding(config: Config, device: str) -> Any:
    return _load_embedding(config, device)


def _dense_identity(*, rank: int, device: str, ids: list[str], sparse_path: Path,
                    evidence: dict[str, Any], config: Config) -> dict[str, Any]:
    identity = {
        "code_version": CODE_VERSION, "stage": "e45-fulltrain-aiteam-rrf20",
        "worker_rank": rank, "device": device,
        "worker_count": config.section("execution")["dense_worker_count"],
        "code_sha256": evidence["code_sha256"], "config_sha256": config.sha,
        "train_sha256": evidence["train_sha256"], "sample_ids_sha256": _ids_sha(ids),
        "sparse_results_sha256": file_sha256(sparse_path),
        "dense_faiss_sha256": config.section("source_dense")["dense_faiss_sha256"],
    }
    identity["identity_sha256"] = _json_sha256(identity)
    return identity


def run_dense_worker(*, root: Path, rank: int, device: str, model: Any, dense: Path,
                     bm25_path: Path, train: Path, output: Path, config: Config) -> dict[str, Any]:
    import numpy as np

    records_by_id, ids, evidence = check_preflight(root=root, output=output, train=train, config=config)
    execution = config.section("execution")
    if rank not in range(execution["dense_worker_count"]) or device != execution["dense_devices"][rank]:
        raise E45Error("E45 dense worker rank/device mismatch.")
    sparse_path = output / "retrieval/sparse/results.jsonl"
    sparse = _read_jsonl(sparse_path)
    if [row.get("question_id") for row in sparse] != ids:
        raise E45Error("E45 sparse cache is incomplete or reordered.")
    dense_cfg, retrieval = config.section("source_dense"), config.section("retrieval")
    identity = _dense_identity(rank=rank, device=device, ids=ids, sparse_path=sparse_path,
                               evidence=evidence, config=config)
    stage, record_dir = output / "retrieval", output / "retrieval/records"
    state = stage / f"dense-worker-{rank}-state.json"
    record_dir.mkdir(parents=True, exist_ok=True)
    assigned = list(range(rank, len(ids), execution["dense_worker_count"]))
    completed = _load_worker_progress(records=record_dir, state_path=state, identity=identity,
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
            vectors = model.encode([dense_cfg["query_prefix"] + records_by_id[qid]["question"] for qid in qids],
                                   batch_size=len(qids), show_progress_bar=False,
                                   convert_to_numpy=True, normalize_embeddings=True)
            vectors = np.asarray(vectors, dtype=np.float32)
            if vectors.shape != (len(qids), dense_cfg["dimension"]) or not np.isfinite(vectors).all():
                raise E45Error("Invalid E45 query embeddings.")
            scores, rows = search.search(vectors, retrieval["candidate_k_per_branch"])
            for offset, (index, qid) in enumerate(zip(subset, qids)):
                dense_ids = [mapping[int(row)] for row in rows[offset] if int(row) >= 0]
                fused = weighted_rrf(
                    {"sparse": sparse[index]["sparse_top_ids"], "dense": dense_ids},
                    weights={"sparse": retrieval["sparse_weight"], "dense": retrieval["dense_weight"]},
                    constant=retrieval["rrf_constant"], top_k=retrieval["fused_top_k"],
                )
                candidate_contexts = store.fetch([item["chunk_id"] for item in fused])
                if len(candidate_contexts) != retrieval["fused_top_k"]:
                    raise E45Error(f"Expected E45 top-20 candidate contexts: {qid}")
                row = {
                    "question_id": qid, "sample_index": index,
                    "sparse_top_ids": sparse[index]["sparse_top_ids"], "dense_top_ids": dense_ids,
                    "dense_scores": [float(value) for value in scores[offset][:len(dense_ids)]],
                    "fused_pool": fused, "candidate_contexts": candidate_contexts,
                    "answers_used": False, "worker_rank": rank,
                    "worker_identity_sha256": identity["identity_sha256"],
                }
                _atomic_json(record_dir / f"{index:04d}.json", row)
                _write_worker_state(state, identity, start + offset + 1, len(assigned))
                LOG.info("e45_dense worker=%d completed=%d total=%d qid=%s",
                         rank, start + offset + 1, len(assigned), qid)
    finally:
        search.close(); store.close()
    _write_worker_state(state, identity, len(assigned), len(assigned))
    return {"worker_rank": rank, "completed": len(assigned), "device": device}


def _span_overlap(left: dict[str, Any], right: dict[str, Any]) -> float:
    if left["document_id"] != right["document_id"]:
        return 0.0
    overlap = max(0, min(left["end_char"], right["end_char"]) - max(left["start_char"], right["start_char"]))
    shortest = min(left["end_char"] - left["start_char"], right["end_char"] - right["start_char"])
    return overlap / shortest if shortest else 0.0


def _signals(question: str, item: dict[str, Any]) -> tuple[bool, bool, bool, bool]:
    q = _fold(question)
    evidence, body = item["evidence"], _fold(item["body"])
    document_number = _fold(evidence.get("document_number") or "")
    source_title = _fold(evidence.get("source_title") or "")
    doc_match = bool((document_number and document_number in q)
                     or (source_title and len(source_title) >= 12 and source_title in q))
    question_articles = {_fold(value) for value in _ARTICLE.findall(q)}
    article = _fold(str(item["context"].get("article_number") or ""))
    article_match = bool(article and article in question_articles)
    clauses = {_fold(value) for value in _CLAUSE.findall(q)}
    clause_match = bool(clauses and any(value in body for value in clauses))
    important = {token for token in _TOKEN.findall(q) if len(token) >= 4 and token not in _IMPORTANT_STOPWORDS}
    keyword_match = len(important.intersection(_TOKEN.findall(body))) >= 3
    return doc_match, article_match, clause_match, keyword_match


def _priority(question: str, item: dict[str, Any]) -> int:
    document, article, clause, keyword = _signals(question, item)
    if document and article and clause:
        return 0
    if document and article:
        return 1
    if document:
        return 2
    if article or clause:
        return 3
    if keyword:
        return 4
    return 5


def select_top12_v2(question: str, candidates: list[dict[str, Any]], selector: dict[str, Any],
                    *, allow_content_relaxed_fill: bool = False) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Select six RRF anchors then six legal-priority contexts without a bonus score."""
    if len(candidates) != 20:
        raise E45Error("Top12-v2 requires the complete ordered RRF top-20 pool.")
    selected: list[dict[str, Any]] = []
    seen_bodies: set[str] = set()
    duplicate_rejections = 0

    def unique(item: dict[str, Any]) -> bool:
        nonlocal duplicate_rejections
        body = _fold(item["body"])
        if body in seen_bodies or any(_span_overlap(item["evidence"], old["evidence"]) >= 0.80 for old in selected):
            duplicate_rejections += 1
            return False
        return True

    def add(item: dict[str, Any], phase: str, priority: int | None) -> bool:
        if not unique(item):
            return False
        item["selection_phase"] = phase
        item["legal_priority"] = priority
        item["selected_rank"] = len(selected)
        selected.append(item)
        seen_bodies.add(_fold(item["body"]))
        return True

    # Phase 1: preserve the strongest RRF evidence, dropping only exact/strong span duplicates.
    for item in candidates:
        if len(selected) == selector["locked_rrf_contexts"]:
            break
        add(item, "locked-rrf", None)

    locked_ids = {item["context"]["chunk_id"] for item in selected}
    remaining = [item for item in candidates if item["context"]["chunk_id"] not in locked_ids]
    priorities = {item["context"]["chunk_id"]: _priority(question, item) for item in remaining}
    group_counts: dict[tuple[str, str], int] = {}
    for item in selected:
        key = (str(item["context"]["document_id"]), str(item["context"].get("article_number")))
        group_counts[key] = group_counts.get(key, 0) + 1
    cap = selector["supplementary_max_per_document_article"]

    # Phase 2: lexicographic legal tiers. RRF order is unchanged inside every tier.
    deferred: list[dict[str, Any]] = []
    for priority in range(6):
        for item in remaining:
            if priorities[item["context"]["chunk_id"]] != priority:
                continue
            key = (str(item["context"]["document_id"]), str(item["context"].get("article_number")))
            if group_counts.get(key, 0) >= cap:
                deferred.append(item)
                continue
            if add(item, "legal-priority", priority):
                group_counts[key] = group_counts.get(key, 0) + 1
            if len(selected) == selector["selected_contexts"]:
                break
        if len(selected) == selector["selected_contexts"]:
            break

    # Diversity is a preference, never a reason to emit fewer than twelve contexts.
    if len(selected) < selector["selected_contexts"]:
        deferred_ids = {item["context"]["chunk_id"] for item in deferred}
        fallback = deferred + [item for item in remaining if item["context"]["chunk_id"] not in deferred_ids]
        already = {item["context"]["chunk_id"] for item in selected}
        for item in fallback:
            if item["context"]["chunk_id"] in already:
                continue
            if add(item, "diversity-relaxed-fill", priorities[item["context"]["chunk_id"]]):
                already.add(item["context"]["chunk_id"])
            if len(selected) == selector["selected_contexts"]:
                break
    # The saved E45 run used an audited final fill for pools with fewer than
    # twelve distinct bodies/spans. This is opt-in for the private E46 lineage;
    # it relaxes content overlap but never repeats a chunk ID.
    if allow_content_relaxed_fill and len(selected) < selector["selected_contexts"]:
        already = {item["context"]["chunk_id"] for item in selected}
        for item in candidates:
            chunk_id = item["context"]["chunk_id"]
            if chunk_id in already:
                continue
            item["selection_phase"] = "content-relaxed-fill"
            item["legal_priority"] = None
            item["selected_rank"] = len(selected)
            selected.append(item)
            already.add(chunk_id)
            if len(selected) == selector["selected_contexts"]:
                break
    if len(selected) != selector["selected_contexts"]:
        raise E45Error(f"Top12-v2 cannot fill 12 unique contexts; got {len(selected)}.")
    return selected, {
        "duplicate_rejections": duplicate_rejections,
        "legal_supplementary": sum(item["selection_phase"] == "legal-priority" for item in selected),
        "diversity_relaxed": sum(item["selection_phase"] == "diversity-relaxed-fill" for item in selected),
        "priority_counts": {str(level): sum(item["legal_priority"] == level for item in selected) for level in range(6)},
    }


def _validate_workers(*, ids: list[str], evidence: dict[str, Any], output: Path, config: Config) -> list[str]:
    sparse_path, record_dir = output / "retrieval/sparse/results.jsonl", output / "retrieval/records"
    identities = []
    worker_count = config.section("execution")["dense_worker_count"]
    for rank, device in enumerate(config.section("execution")["dense_devices"]):
        identity = _dense_identity(rank=rank, device=device, ids=ids, sparse_path=sparse_path,
                                   evidence=evidence, config=config)
        state_path = output / f"retrieval/dense-worker-{rank}-state.json"
        state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.is_file() else {}
        assigned = list(range(rank, len(ids), worker_count))
        if (state.get("run_identity") != identity or state.get("complete") is not True
                or state.get("completed_count") != len(assigned) or state.get("assigned_count") != len(assigned)):
            raise E45Error(f"Incomplete E45 dense worker: {rank}")
        identities.append(identity["identity_sha256"])
    for index, qid in enumerate(ids):
        path = record_dir / f"{index:04d}.json"
        if not path.is_file():
            raise E45Error(f"Missing E45 dense record: {index}")
        row = json.loads(path.read_text(encoding="utf-8"))
        rank = index % worker_count
        if (row.get("question_id") != qid or row.get("sample_index") != index
                or row.get("worker_rank") != rank or row.get("worker_identity_sha256") != identities[rank]
                or row.get("answers_used") is not False or "answer" in row
                or len(row.get("fused_pool", [])) != 20 or len(row.get("candidate_contexts", [])) != 20):
            raise E45Error(f"E45 dense record identity changed: {index}")
    return identities


def finalize(*, root: Path, e00: Path, train: Path, output: Path, config: Config) -> dict[str, Any]:
    official, ids, evidence = check_preflight(root=root, output=output, train=train, config=config)
    identities = _validate_workers(ids=ids, evidence=evidence, output=output, config=config)
    record_dir = output / "retrieval/records"
    raw_path = output / "retrieval/raw-results.jsonl"
    _write_records(raw_path, record_dir, len(ids))
    rows = _read_jsonl(raw_path)
    pool_path = output / "retrieval/candidate-pool-top20.jsonl"
    _atomic_jsonl(pool_path, [{"question_id": row["question_id"], "sample_index": row["sample_index"],
                              "fused_pool": row["fused_pool"], "answers_used": False} for row in rows])

    metadata = config.section("metadata_source")
    wanted = {context["chunk_id"] for row in rows for context in row["candidate_contexts"]}
    chunks = scan_selected(e00 / metadata["chunks_path"], "chunk_id", wanted, metadata["chunks_sha256"])
    document_ids = {chunk["document_id"] for chunk in chunks.values()}
    documents = scan_selected(e00 / metadata["documents_path"], "document_id", document_ids,
                              metadata["documents_sha256"])
    output_rows, diagnostics = [], []
    for index, (qid, row) in enumerate(zip(ids, rows)):
        candidates = []
        for rrf_rank, context in enumerate(row["candidate_contexts"]):
            chunk = chunks[context["chunk_id"]]
            enriched, meta = enrich_context(dict(context), chunk, documents[chunk["document_id"]])
            candidates.append({
                "context": enriched, "body": context["text"], "evidence": meta,
                "rrf_rank": rrf_rank, "rrf_score": row["fused_pool"][rrf_rank]["rrf_score"],
            })
        selected, diag = select_top12_v2(official[qid]["question"], candidates, config.section("selector"))
        contexts = [item["context"] for item in selected]
        output_rows.append({
            "question_id": qid, "sample_index": index, "question": official[qid]["question"],
            "answer": official[qid]["answer"], "contexts": contexts,
            "selected_chunk_ids": [context["chunk_id"] for context in contexts],
            "selected_rrf_ranks": [item["rrf_rank"] for item in selected],
            "selection_phases": [item["selection_phase"] for item in selected],
            "legal_priorities": [item["legal_priority"] for item in selected],
            "supervision": "official-answer-only", "answers_are_retrieval_labels": False,
            "selector": config.section("selector")["name"],
            "metadata_policy": "exact-source-title-and-article-title-plus-unambiguous-header-So-number-v1",
        })
        diagnostics.append(diag)
    records_path = output / "training-data/records.jsonl"
    _atomic_jsonl(records_path, output_rows)
    summary = {
        "schema_version": "1.0", "experiment_id": EXPERIMENT,
        "record_count": len(output_rows), "records_sha256": file_sha256(records_path),
        "sample_ids_sha256": _ids_sha(ids), "context_candidates_per_record": 12,
        "selector": config.section("selector")["name"],
        "same_context_bodies_and_order": True,
        "answers_are_retrieval_labels": False, "contains_public_or_private": False,
        "official_answers_used_for_generation_supervision_only": True,
        "metadata_policy": "exact-source-title-and-article-title-plus-unambiguous-header-So-number-v1",
    }
    _atomic_json(output / "training-data/summary.json", summary)
    changed_from_rrf12 = sum(row["selected_rrf_ranks"] != list(range(12)) for row in output_rows)
    report = {
        "schema_version": "1.0", "experiment_id": EXPERIMENT,
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "sample_size": len(ids),
        "train_sha256": evidence["train_sha256"], "sample_ids_sha256": _ids_sha(ids),
        "retrieval": config.section("retrieval"), "selector": config.section("selector"),
        "diagnostics": {
            "changed_from_plain_rrf_top12": changed_from_rrf12,
            "unchanged_from_plain_rrf_top12": len(ids) - changed_from_rrf12,
            "mean_duplicate_rejections": fmean(item["duplicate_rejections"] for item in diagnostics),
            "questions_with_diversity_relaxation": sum(item["diversity_relaxed"] > 0 for item in diagnostics),
            "normalized_duplicate_questions": evidence["normalized_duplicate_questions"],
        },
        "files": {
            "retrieval/raw-results.jsonl": {"sha256": file_sha256(raw_path), "bytes": raw_path.stat().st_size},
            "retrieval/candidate-pool-top20.jsonl": {"sha256": file_sha256(pool_path), "bytes": pool_path.stat().st_size},
            "training-data/records.jsonl": {"sha256": file_sha256(records_path), "bytes": records_path.stat().st_size},
            "training-data/summary.json": {"sha256": file_sha256(output / "training-data/summary.json"),
                                                    "bytes": (output / "training-data/summary.json").stat().st_size},
        },
        "evidence": {**evidence, "dense_worker_identity_sha256": identities},
        "answers_used_by_retrieval_or_selector": False,
        "public_read": False, "private_read": False, "automatic_promotion": False,
    }
    _atomic_json(output / "report.json", report)
    return report


__all__ = [
    "Config", "E45Error", "EXPERIMENT", "check_preflight", "finalize", "load_config",
    "load_embedding", "load_official_train", "prepare_local_bm25", "run_dense_worker",
    "run_sparse", "select_top12_v2", "validate_preflight",
]
