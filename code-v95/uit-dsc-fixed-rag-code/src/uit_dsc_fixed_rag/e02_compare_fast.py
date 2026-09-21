"""Fast execution layer for the unchanged E02 retrieval-only comparison."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from uit_dsc_fixed_rag.bm25 import SqliteBm25Index
from uit_dsc_fixed_rag.corpus import file_sha256
from uit_dsc_fixed_rag.e02_compare import (
    CompareConfig,
    E02CompareError,
    _ChunkStore,
    _TorchGpuFlatIpIndex,
    _atomic_json,
    _candidate_summary,
    _load_completed,
    _load_dev,
    _load_mapping,
    _read_records,
    _write_results_jsonl,
    _write_state,
    aggregate_comparison,
    weighted_rrf,
)
from uit_dsc_fixed_rag.retrieval_diagnostics import (
    answer_derived_diagnostics,
    select_dev_sample,
)


CODE_VERSION = "0.1.0"
LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class FastCompareConfig:
    raw: dict[str, Any]
    path: Path
    semantic_config_path: str
    semantic_config_sha256: str
    bm25_bytes: int
    bm25_sha256: str
    sparse_workers: int
    dense_query_batch_size: int
    dense_devices: dict[str, str]

    @property
    def config_sha256(self) -> str:
        return file_sha256(self.path)


def load_fast_config(path: Path) -> FastCompareConfig:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if set(payload) != {
        "schema_version", "experiment_id", "semantic_config", "source_bm25",
        "execution", "run_contract",
    } or payload.get("schema_version") != "1.0":
        raise ValueError("Fast E02 config root is incompatible.")
    if payload.get("experiment_id") != "E02-compare-dev200-fast-v1":
        raise ValueError("Unexpected fast E02 experiment ID.")
    semantic = _object(payload, "semantic_config")
    source = _object(payload, "source_bm25")
    execution = _object(payload, "execution")
    contract = _object(payload, "run_contract")
    required_execution = {
        "copy_bm25_to_local_ssd": True,
        "sparse_worker_threads": 4,
        "sparse_compute_once": True,
        "dense_query_batch_size": 32,
        "dense_candidates_parallel": True,
        "dense_device_map": {
            "embedding_aiteam": "cuda:0",
            "embedding_harrier": "cuda:1",
        },
        "dense_search_backend": "torch-cuda-exact-flat-ip-float32-batched",
    }
    if execution != required_execution:
        raise ValueError("Fast E02 execution settings changed.")
    required_contract = {
        "retrieval_semantics_unchanged": True,
        "checkpoint_every_questions": 1,
        "atomic_checkpoint_write": True,
        "resume_fail_closed": True,
        "answers_used_only_after_retrieval": True,
        "promotion_allowed": False,
        "allow_holdout": False,
        "allow_public": False,
    }
    if contract != required_contract:
        raise ValueError("Fast E02 run contract changed.")
    return FastCompareConfig(
        raw=payload,
        path=path,
        semantic_config_path=_text(semantic, "path"),
        semantic_config_sha256=_sha256(semantic, "sha256"),
        bm25_bytes=_positive_int(source, "bytes"),
        bm25_sha256=_sha256(source, "sha256"),
        sparse_workers=_positive_int(execution, "sparse_worker_threads"),
        dense_query_batch_size=_positive_int(execution, "dense_query_batch_size"),
        dense_devices=dict(_object(execution, "dense_device_map")),
    )


def validate_fast_config(*, project_root: Path, config: FastCompareConfig) -> dict[str, Any]:
    semantic_path = project_root / config.semantic_config_path
    if not semantic_path.is_file() or file_sha256(semantic_path) != config.semantic_config_sha256:
        raise E02CompareError("Pinned semantic E02 config is missing or changed.")
    return {
        "execution_config_sha256": config.config_sha256,
        "semantic_config_sha256": config.semantic_config_sha256,
        "sparse_workers": config.sparse_workers,
        "dense_query_batch_size": config.dense_query_batch_size,
        "dense_devices": config.dense_devices,
    }


def prepare_local_bm25(
    *, source_path: Path, runtime_directory: Path, config: FastCompareConfig
) -> Path:
    if not source_path.is_file() or source_path.stat().st_size != config.bm25_bytes:
        raise E02CompareError("Pinned BM25 source file size differs.")
    runtime_directory.mkdir(parents=True, exist_ok=True)
    destination = runtime_directory / "bm25.sqlite3"
    manifest_path = runtime_directory / "cache-manifest.json"
    expected_manifest = {
        "schema_version": "1.0",
        "source_sha256": config.bm25_sha256,
        "bytes": config.bm25_bytes,
        "execution_config_sha256": config.config_sha256,
    }
    if destination.is_file() and manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest == expected_manifest and destination.stat().st_size == config.bm25_bytes:
            LOGGER.info("e02_fast_bm25_local_cache_reused path=%s", destination)
            return destination
        raise E02CompareError("Existing local BM25 cache has a different identity.")
    if destination.exists() or manifest_path.exists():
        raise E02CompareError("Partial local BM25 cache exists; use a clean runtime directory.")

    temporary = runtime_directory / "bm25.sqlite3.tmp"
    LOGGER.info(
        "e02_fast_bm25_copy_started bytes=%d source=%s destination=%s",
        config.bm25_bytes, source_path, destination,
    )
    with source_path.open("rb") as source, temporary.open("xb") as target:
        shutil.copyfileobj(source, target, length=16 * 1024 * 1024)
        target.flush()
        os.fsync(target.fileno())
    if temporary.stat().st_size != config.bm25_bytes:
        raise E02CompareError("Local BM25 copy size mismatch.")
    observed_sha = file_sha256(temporary)
    if observed_sha != config.bm25_sha256:
        raise E02CompareError("Local BM25 copy checksum mismatch.")
    os.replace(temporary, destination)
    _atomic_json(manifest_path, expected_manifest)
    LOGGER.info("e02_fast_bm25_copy_completed path=%s", destination)
    return destination


def prepare_sparse_cache(
    *,
    bm25_path: Path,
    dev_path: Path,
    output_directory: Path,
    semantic_config: CompareConfig,
    fast_config: FastCompareConfig,
) -> dict[str, Any]:
    _validate_local_bm25(bm25_path, fast_config)
    dev = _load_dev(dev_path)
    sample_ids = select_dev_sample(
        dev, seed=semantic_config.sample_seed, size=semantic_config.sample_size
    )
    identity = {
        "code_version": CODE_VERSION,
        "stage": "sparse-once",
        "execution_config_sha256": fast_config.config_sha256,
        "semantic_config_sha256": semantic_config.config_sha256,
        "bm25_sha256": fast_config.bm25_sha256,
        "dev_sha256": semantic_config.dev_sha256,
        "sample_ids_sha256": hashlib.sha256("\n".join(sample_ids).encode("utf-8")).hexdigest(),
    }
    identity["identity_sha256"] = _json_sha256(identity)
    sparse_root = output_directory / "sparse"
    records_directory = sparse_root / "records"
    records_directory.mkdir(parents=True, exist_ok=True)
    state_path = sparse_root / "state.json"
    completed = _load_completed(records_directory, state_path, identity, sample_ids)
    pending_ids = sample_ids[completed:]
    thread_state = threading.local()

    def search_one(question_id: str) -> dict[str, Any]:
        index = getattr(thread_state, "index", None)
        if index is None:
            index = SqliteBm25Index(bm25_path)
            thread_state.index = index
        started = time.perf_counter()
        hits = index.search(dev[question_id]["question"], top_k=semantic_config.candidate_k)
        return {
            "ids": [hit.chunk_id for hit in hits],
            "scores": [float(hit.score) for hit in hits],
            "latency_ms": (time.perf_counter() - started) * 1000,
        }

    with ThreadPoolExecutor(max_workers=fast_config.sparse_workers) as executor:
        futures = [executor.submit(search_one, question_id) for question_id in pending_ids]
        for relative_index, (question_id, future) in enumerate(zip(pending_ids, futures)):
            sample_index = completed + relative_index
            result = future.result()
            row = {
                "question_id": question_id,
                "sample_index": sample_index,
                "sparse_top_ids": result["ids"],
                "sparse_scores": result["scores"],
                "sparse_latency_ms": result["latency_ms"],
            }
            _atomic_json(records_directory / f"{sample_index:04d}.json", row)
            _write_state(state_path, identity, sample_index + 1, complete=False)
            LOGGER.info(
                "e02_fast_sparse_progress completed=%d total=%d question_id=%s latency_ms=%.1f",
                sample_index + 1, len(sample_ids), question_id, result["latency_ms"],
            )
    _write_results_jsonl(sparse_root / "results.jsonl", records_directory, len(sample_ids))
    _write_state(state_path, identity, len(sample_ids), complete=True)
    rows = _read_records(records_directory, sample_ids)
    return {
        "sample_size": len(rows),
        "results_sha256": file_sha256(sparse_root / "results.jsonl"),
        "run_identity": identity,
    }


def run_fast_candidate(
    *,
    candidate_key: str,
    model: Any,
    dense_directory: Path,
    bm25_path: Path,
    dev_path: Path,
    output_directory: Path,
    semantic_config: CompareConfig,
    fast_config: FastCompareConfig,
) -> dict[str, Any]:
    candidate = semantic_config.candidates.get(candidate_key)
    if candidate is None or fast_config.dense_devices.get(candidate_key) != str(model.device):
        raise E02CompareError("Fast candidate/device mapping differs from the pinned run.")
    _validate_local_bm25(bm25_path, fast_config)
    dev = _load_dev(dev_path)
    sample_ids = select_dev_sample(
        dev, seed=semantic_config.sample_seed, size=semantic_config.sample_size
    )
    sparse_path = output_directory / "sparse" / "results.jsonl"
    sparse_state_path = output_directory / "sparse" / "state.json"
    if not sparse_path.is_file() or not sparse_state_path.is_file():
        raise E02CompareError("Shared sparse cache is missing.")
    sparse_state = json.loads(sparse_state_path.read_text(encoding="utf-8"))
    if sparse_state.get("complete") is not True or sparse_state.get("completed_count") != len(sample_ids):
        raise E02CompareError("Shared sparse cache is incomplete.")
    sparse_rows = [json.loads(line) for line in sparse_path.read_text(encoding="utf-8").splitlines()]
    if [row.get("question_id") for row in sparse_rows] != sample_ids:
        raise E02CompareError("Shared sparse cache question order differs.")

    identity = {
        "code_version": CODE_VERSION,
        "stage": "batched-dense-and-fusion",
        "execution_config_sha256": fast_config.config_sha256,
        "semantic_config_sha256": semantic_config.config_sha256,
        "candidate": candidate.__dict__,
        "dense_manifest_sha256": file_sha256(dense_directory / "manifest.json"),
        "sparse_results_sha256": file_sha256(sparse_path),
        "dev_sha256": semantic_config.dev_sha256,
        "sample_ids_sha256": hashlib.sha256("\n".join(sample_ids).encode("utf-8")).hexdigest(),
        "device": str(model.device),
        "query_batch_size": fast_config.dense_query_batch_size,
    }
    identity["identity_sha256"] = _json_sha256(identity)
    candidate_root = output_directory / candidate_key
    records_directory = candidate_root / "records"
    records_directory.mkdir(parents=True, exist_ok=True)
    state_path = candidate_root / "state.json"
    completed = _load_completed(records_directory, state_path, identity, sample_ids)

    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - Kaggle dependency
        raise E02CompareError("Install numpy in the Kaggle runtime.") from exc
    mapping = _load_mapping(dense_directory / "chunk_ids.jsonl", semantic_config.chunk_count)
    search_index = _TorchGpuFlatIpIndex(
        dense_directory / "dense.faiss",
        device=str(model.device),
        expected_count=semantic_config.chunk_count,
        expected_dimension=candidate.dimension,
    )
    store = _ChunkStore(bm25_path)
    try:
        for start in range(completed, len(sample_ids), fast_config.dense_query_batch_size):
            stop = min(start + fast_config.dense_query_batch_size, len(sample_ids))
            batch_ids = sample_ids[start:stop]
            batch_questions = [candidate.query_prefix + dev[qid]["question"] for qid in batch_ids]
            dense_started = time.perf_counter()
            query_vectors = model.encode(
                batch_questions,
                batch_size=len(batch_questions),
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
            query_vectors = np.asarray(query_vectors, dtype=np.float32)
            if (
                query_vectors.shape != (len(batch_ids), candidate.dimension)
                or not np.isfinite(query_vectors).all()
            ):
                raise E02CompareError(f"Invalid batched query embeddings: {candidate_key}")
            dense_scores, dense_rows = search_index.search(
                query_vectors, semantic_config.candidate_k
            )
            batch_latency_ms = (time.perf_counter() - dense_started) * 1000
            per_question_latency = batch_latency_ms / len(batch_ids)

            for offset, question_id in enumerate(batch_ids):
                sample_index = start + offset
                sparse = sparse_rows[sample_index]
                dense_ids = [
                    mapping[int(row)] for row in dense_rows[offset] if int(row) >= 0
                ]
                fused = weighted_rrf(
                    {"sparse": sparse["sparse_top_ids"], "dense": dense_ids},
                    weights=semantic_config.branch_weights,
                    constant=semantic_config.rrf_constant,
                    top_k=semantic_config.top_k,
                )
                contexts = store.fetch([item["chunk_id"] for item in fused])
                diagnostics = answer_derived_diagnostics(
                    dev[question_id]["answer"], contexts
                )
                row = {
                    "question_id": question_id,
                    "sample_index": sample_index,
                    "candidate": candidate_key,
                    "sparse_top_ids": sparse["sparse_top_ids"],
                    "dense_top_ids": dense_ids,
                    "dense_scores": [float(value) for value in dense_scores[offset][:len(dense_ids)]],
                    "fused": fused,
                    "contexts": contexts,
                    "diagnostics": diagnostics,
                    "latency_ms": {
                        "sparse": sparse["sparse_latency_ms"],
                        "dense_encode_and_search": per_question_latency,
                    },
                }
                _atomic_json(records_directory / f"{sample_index:04d}.json", row)
                _write_state(state_path, identity, sample_index + 1, complete=False)
                LOGGER.info(
                    "e02_fast_dense_progress candidate=%s completed=%d total=%d question_id=%s "
                    "batch=%d batch_latency_ms=%.1f",
                    candidate_key, sample_index + 1, len(sample_ids), question_id,
                    len(batch_ids), batch_latency_ms,
                )
        _write_results_jsonl(candidate_root / "results.jsonl", records_directory, len(sample_ids))
        _write_state(state_path, identity, len(sample_ids), complete=True)
    finally:
        search_index.close()
        store.close()
    rows = _read_records(records_directory, sample_ids)
    return _candidate_summary(rows, identity)


def aggregate_fast_comparison(
    *, output_directory: Path, semantic_config: CompareConfig, fast_config: FastCompareConfig
) -> dict[str, Any]:
    report = aggregate_comparison(output_directory=output_directory, config=semantic_config)
    report["experiment_id"] = "E02-compare-dev200-retrieval-diagnostic-fast-v1"
    report["semantic_config_sha256"] = semantic_config.config_sha256
    report["execution_config_sha256"] = fast_config.config_sha256
    report["execution_only_changes"] = [
        "BM25 was computed once from an SHA-verified local SSD copy using four CPU threads.",
        "Both embedding candidates encoded queries and searched exact FlatIP in batches on separate GPUs.",
        "Retrieval depths, RRF weights, chunks, questions and answer-derived diagnostics were unchanged.",
    ]
    _atomic_json(output_directory / "report.json", report)
    return report


def _validate_local_bm25(path: Path, config: FastCompareConfig) -> None:
    manifest_path = path.parent / "cache-manifest.json"
    if not path.is_file() or path.stat().st_size != config.bm25_bytes or not manifest_path.is_file():
        raise E02CompareError("Local BM25 cache is missing or incomplete.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("source_sha256") != config.bm25_sha256
        or manifest.get("bytes") != config.bm25_bytes
        or manifest.get("execution_config_sha256") != config.config_sha256
    ):
        raise E02CompareError("Local BM25 cache manifest differs.")


def _json_sha256(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


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
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{key} must be a positive integer.")
    return value


def _sha256(payload: dict[str, Any], key: str) -> str:
    value = _text(payload, key).lower()
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{key} must be a SHA-256 hex digest.")
    return value
