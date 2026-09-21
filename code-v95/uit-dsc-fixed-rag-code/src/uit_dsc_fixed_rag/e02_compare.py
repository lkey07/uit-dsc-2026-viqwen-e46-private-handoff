"""Fail-closed E02 dense retrieval comparison on a deterministic dev sample."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean
from typing import Any

from uit_dsc_fixed_rag.bm25 import SqliteBm25Index
from uit_dsc_fixed_rag.corpus import file_sha256
from uit_dsc_fixed_rag.retrieval_diagnostics import (
    answer_derived_diagnostics,
    select_dev_sample,
)


CODE_VERSION = "0.2.0"
LOGGER = logging.getLogger(__name__)


class E02CompareError(RuntimeError):
    """Raised when comparison inputs or resumable state are incompatible."""


@dataclass(frozen=True)
class CompareCandidate:
    key: str
    model_id: str
    revision: str
    parameter_count: int
    dimension: int
    query_prefix: str
    dense_faiss_sha256: str
    chunk_ids_sha256: str


@dataclass(frozen=True)
class CompareConfig:
    raw: dict[str, Any]
    path: Path
    e00_manifest_sha256: str
    e00_chunks_sha256: str
    chunk_count: int
    candidates: dict[str, CompareCandidate]
    dev_path: str
    dev_sha256: str
    sample_seed: str
    sample_size: int
    sample_ids_sha256: str
    candidate_k: int
    top_k: int
    rrf_constant: int
    branch_weights: dict[str, float]

    @property
    def config_sha256(self) -> str:
        return file_sha256(self.path)


def load_compare_config(path: Path) -> CompareConfig:
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "schema_version", "experiment_id", "source_e00", "dense_candidates",
        "dev", "retrieval", "run_contract",
    }
    if set(payload) != required or payload.get("schema_version") != "1.0":
        raise ValueError("E02 comparison config root is incompatible.")
    if payload.get("experiment_id") != "E02-compare-dev200":
        raise ValueError("Unexpected E02 comparison experiment ID.")

    source = _object(payload, "source_e00")
    if source.get("artifact_version") != "e00-v2":
        raise ValueError("E02 comparison must use E00 v2.")
    candidates_payload = _object(payload, "dense_candidates")
    if set(candidates_payload) != {"embedding_harrier", "embedding_aiteam"}:
        raise ValueError("E02 comparison requires exactly the two approved candidates.")
    candidates: dict[str, CompareCandidate] = {}
    for key, raw_candidate in candidates_payload.items():
        if not isinstance(raw_candidate, dict):
            raise ValueError(f"Invalid candidate config: {key}")
        candidates[key] = CompareCandidate(
            key=key,
            model_id=_text(raw_candidate, "model_id"),
            revision=_revision(raw_candidate, "revision"),
            parameter_count=_positive_int(raw_candidate, "parameter_count"),
            dimension=_positive_int(raw_candidate, "dimension"),
            query_prefix=_string(raw_candidate, "query_prefix"),
            dense_faiss_sha256=_sha256(raw_candidate, "dense_faiss_sha256"),
            chunk_ids_sha256=_sha256(raw_candidate, "chunk_ids_sha256"),
        )
    if len({candidate.dimension for candidate in candidates.values()}) != 1:
        raise ValueError("Dense candidates must have the same output dimension.")

    dev = _object(payload, "dev")
    retrieval = _object(payload, "retrieval")
    weights = _object(retrieval, "branch_weights")
    if set(weights) != {"sparse", "dense"}:
        raise ValueError("E02 comparison requires sparse and dense RRF branches.")
    branch_weights = {key: float(value) for key, value in weights.items()}
    if any(value <= 0 for value in branch_weights.values()):
        raise ValueError("RRF weights must be positive.")
    contract = _object(payload, "run_contract")
    required_contract = {
        "checkpoint_every_questions": 1,
        "atomic_checkpoint_write": True,
        "resume_fail_closed": True,
        "answers_used_only_after_retrieval": True,
        "promotion_allowed": False,
        "allow_holdout": False,
        "allow_public": False,
    }
    if contract != required_contract:
        raise ValueError("E02 comparison run contract changed.")
    if retrieval.get("normalize_query_embeddings") is not True:
        raise ValueError("Dense query embeddings must be normalized.")
    if retrieval.get("comparison_search_backend") != "torch-cuda-exact-flat-ip-float32":
        raise ValueError("E02 comparison must use exact float32 GPU inner-product search.")

    return CompareConfig(
        raw=payload,
        path=path,
        e00_manifest_sha256=_sha256(source, "manifest_sha256"),
        e00_chunks_sha256=_sha256(source, "chunks_sha256"),
        chunk_count=_positive_int(source, "chunk_count"),
        candidates=candidates,
        dev_path=_text(dev, "path"),
        dev_sha256=_sha256(dev, "sha256"),
        sample_seed=_text(dev, "sample_seed"),
        sample_size=_positive_int(dev, "sample_size"),
        sample_ids_sha256=_sha256(dev, "sample_ids_sha256"),
        candidate_k=_positive_int(retrieval, "candidate_k_per_branch"),
        top_k=_positive_int(retrieval, "top_k"),
        rrf_constant=_positive_int(retrieval, "rrf_constant"),
        branch_weights=branch_weights,
    )


def validate_inputs(
    *,
    e00_directory: Path,
    dense_directories: dict[str, Path],
    dev_path: Path,
    config: CompareConfig,
) -> dict[str, Any]:
    if set(dense_directories) != set(config.candidates):
        raise E02CompareError("Dense candidate directory mapping is incomplete.")
    required_e00 = (
        e00_directory / "manifest.json",
        e00_directory / "chunks.jsonl",
        e00_directory / "bm25.sqlite3",
    )
    if not all(path.is_file() for path in required_e00):
        raise E02CompareError("E00 v2 files are incomplete.")
    if file_sha256(required_e00[0]) != config.e00_manifest_sha256:
        raise E02CompareError("E00 manifest checksum mismatch.")
    if file_sha256(required_e00[1]) != config.e00_chunks_sha256:
        raise E02CompareError("E00 chunks checksum mismatch.")
    if file_sha256(dev_path) != config.dev_sha256:
        raise E02CompareError("Dev split checksum mismatch.")

    evidence: dict[str, Any] = {}
    for key, directory in dense_directories.items():
        candidate = config.candidates[key]
        manifest_path = directory / "manifest.json"
        index_path = directory / "dense.faiss"
        mapping_path = directory / "chunk_ids.jsonl"
        if not all(path.is_file() for path in (manifest_path, index_path, mapping_path)):
            raise E02CompareError(f"Dense artifact is incomplete: {key}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest_candidate = manifest.get("candidate", {})
        expected_candidate = {
            "key": key,
            "model_id": candidate.model_id,
            "revision": candidate.revision,
            "parameter_count": candidate.parameter_count,
            "output_dimension": candidate.dimension,
        }
        if (
            manifest.get("artifact_type") != "e02-dense-flat-ip"
            or manifest.get("artifact_version") != "e02-v2"
            or manifest.get("record_count") != config.chunk_count
            or manifest.get("dimension") != candidate.dimension
            or manifest.get("normalized") is not True
            or manifest.get("source_chunks_sha256") != config.e00_chunks_sha256
            or manifest_candidate != expected_candidate
        ):
            raise E02CompareError(f"Dense manifest is incompatible: {key}")
        index_sha = file_sha256(index_path)
        mapping_sha = file_sha256(mapping_path)
        if index_sha != candidate.dense_faiss_sha256:
            raise E02CompareError(f"Dense index checksum mismatch: {key}")
        if mapping_sha != candidate.chunk_ids_sha256:
            raise E02CompareError(f"Chunk mapping checksum mismatch: {key}")
        evidence[key] = {
            "manifest_sha256": file_sha256(manifest_path),
            "dense_faiss_sha256": index_sha,
            "chunk_ids_sha256": mapping_sha,
        }

    dev = _load_dev(dev_path)
    sample_ids = select_dev_sample(dev, seed=config.sample_seed, size=config.sample_size)
    observed_sample_sha = hashlib.sha256("\n".join(sample_ids).encode("utf-8")).hexdigest()
    if observed_sample_sha != config.sample_ids_sha256:
        raise E02CompareError("Deterministic dev sample identity mismatch.")
    return {
        "config_sha256": config.config_sha256,
        "dev_sha256": config.dev_sha256,
        "sample_ids_sha256": observed_sample_sha,
        "sample_size": len(sample_ids),
        "dense": evidence,
    }


def weighted_rrf(
    branches: dict[str, list[str]],
    *,
    weights: dict[str, float],
    constant: int,
    top_k: int,
) -> list[dict[str, Any]]:
    if set(branches) != set(weights) or constant <= 0 or top_k <= 0:
        raise ValueError("Invalid weighted RRF inputs.")
    scores: dict[str, float] = {}
    ranks: dict[str, dict[str, int]] = {}
    for branch, ids in branches.items():
        if len(ids) != len(set(ids)):
            raise ValueError(f"Duplicate chunk ID in RRF branch: {branch}")
        for rank, chunk_id in enumerate(ids, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + weights[branch] / (constant + rank)
            ranks.setdefault(chunk_id, {})[branch] = rank
    ordered = sorted(scores, key=lambda chunk_id: (-scores[chunk_id], chunk_id))[:top_k]
    return [
        {"chunk_id": chunk_id, "rrf_score": scores[chunk_id], "branch_ranks": ranks[chunk_id]}
        for chunk_id in ordered
    ]


def run_candidate_diagnostics(
    *,
    candidate_key: str,
    model: Any,
    e00_directory: Path,
    dense_directory: Path,
    dev_path: Path,
    output_directory: Path,
    config: CompareConfig,
) -> dict[str, Any]:
    candidate = config.candidates.get(candidate_key)
    if candidate is None:
        raise E02CompareError(f"Unknown comparison candidate: {candidate_key}")
    validate_inputs(
        e00_directory=e00_directory,
        dense_directories={candidate_key: dense_directory},
        dev_path=dev_path,
        config=_single_candidate_validation_config(config, candidate_key),
    )

    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - Kaggle dependency
        raise E02CompareError("Install numpy in the Kaggle runtime.") from exc

    dev = _load_dev(dev_path)
    sample_ids = select_dev_sample(dev, seed=config.sample_seed, size=config.sample_size)
    identity = _candidate_run_identity(config, candidate, dense_directory, sample_ids)
    candidate_output = output_directory / candidate_key
    records_directory = candidate_output / "records"
    records_directory.mkdir(parents=True, exist_ok=True)
    state_path = candidate_output / "state.json"
    completed = _load_completed(records_directory, state_path, identity, sample_ids)

    mapping = _load_mapping(dense_directory / "chunk_ids.jsonl", config.chunk_count)
    search_index = _TorchGpuFlatIpIndex(
        dense_directory / "dense.faiss",
        device=str(model.device),
        expected_count=config.chunk_count,
        expected_dimension=candidate.dimension,
    )

    bm25 = SqliteBm25Index(e00_directory / "bm25.sqlite3")
    store = _ChunkStore(e00_directory / "bm25.sqlite3")
    try:
        for sample_index in range(completed, len(sample_ids)):
            question_id = sample_ids[sample_index]
            record = dev[question_id]
            question = record["question"]

            sparse_started = time.perf_counter()
            sparse_hits = bm25.search(question, top_k=config.candidate_k)
            sparse_latency_ms = (time.perf_counter() - sparse_started) * 1000
            sparse_ids = [hit.chunk_id for hit in sparse_hits]

            dense_started = time.perf_counter()
            query_vector = model.encode(
                [candidate.query_prefix + question],
                batch_size=1,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
            query_vector = np.asarray(query_vector, dtype=np.float32)
            if query_vector.shape != (1, candidate.dimension) or not np.isfinite(query_vector).all():
                raise E02CompareError(f"Invalid query embedding for question {question_id}.")
            dense_scores, dense_rows = search_index.search(query_vector, config.candidate_k)
            dense_ids = [mapping[int(row)] for row in dense_rows[0] if int(row) >= 0]
            dense_latency_ms = (time.perf_counter() - dense_started) * 1000

            fused = weighted_rrf(
                {"sparse": sparse_ids, "dense": dense_ids},
                weights=config.branch_weights,
                constant=config.rrf_constant,
                top_k=config.top_k,
            )
            fused_ids = [item["chunk_id"] for item in fused]
            payload = store.fetch(fused_ids)
            diagnostics = answer_derived_diagnostics(record["answer"], payload)
            result = {
                "question_id": question_id,
                "sample_index": sample_index,
                "candidate": candidate_key,
                "sparse_top_ids": sparse_ids,
                "dense_top_ids": dense_ids,
                "dense_scores": [float(score) for score in dense_scores[0][:len(dense_ids)]],
                "fused": fused,
                "contexts": payload,
                "diagnostics": diagnostics,
                "latency_ms": {
                    "sparse": sparse_latency_ms,
                    "dense_encode_and_search": dense_latency_ms,
                },
            }
            _atomic_json(records_directory / f"{sample_index:04d}.json", result)
            _write_state(state_path, identity, sample_index + 1, complete=False)
            LOGGER.info(
                "e02_compare_progress candidate=%s completed=%d total=%d question_id=%s",
                candidate_key,
                sample_index + 1,
                len(sample_ids),
                question_id,
            )
        _write_results_jsonl(candidate_output / "results.jsonl", records_directory, len(sample_ids))
        _write_state(state_path, identity, len(sample_ids), complete=True)
    finally:
        search_index.close()
        store.close()
        bm25.close()

    records = _read_records(records_directory, sample_ids)
    return _candidate_summary(records, identity)


def aggregate_comparison(
    *,
    output_directory: Path,
    config: CompareConfig,
) -> dict[str, Any]:
    by_candidate: dict[str, list[dict[str, Any]]] = {}
    for key in config.candidates:
        path = output_directory / key / "results.jsonl"
        if not path.is_file():
            raise E02CompareError(f"Candidate results are missing: {key}")
        by_candidate[key] = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        if len(by_candidate[key]) != config.sample_size:
            raise E02CompareError(f"Candidate results are incomplete: {key}")
    harrier = by_candidate["embedding_harrier"]
    aiteam = by_candidate["embedding_aiteam"]
    if [row["question_id"] for row in harrier] != [row["question_id"] for row in aiteam]:
        raise E02CompareError("Candidate result question order differs.")

    summaries = {
        key: _candidate_summary(rows, None)["metrics"]
        for key, rows in by_candidate.items()
    }
    jaccards = []
    for left, right in zip(harrier, aiteam):
        left_ids = {item["chunk_id"] for item in left["fused"]}
        right_ids = {item["chunk_id"] for item in right["fused"]}
        union = left_ids | right_ids
        jaccards.append(len(left_ids & right_ids) / len(union) if union else 1.0)
    report = {
        "schema_version": "1.0",
        "experiment_id": "E02-compare-dev200-retrieval-diagnostic",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_sha256": config.config_sha256,
        "sample_size": config.sample_size,
        "metrics": summaries,
        "mean_fused_top_k_jaccard": fmean(jaccards),
        "warnings": [
            "Gold answers were used only after retrieval for diagnostics.",
            "These metrics are not retrieval relevance labels, recall or precision.",
            "This report cannot select or promote an embedding; answer-level generator evaluation is required.",
        ],
    }
    _atomic_json(output_directory / "report.json", report)
    return report


class _ChunkStore:
    def __init__(self, path: Path) -> None:
        uri = path.resolve().as_uri() + "?mode=ro"
        self.connection = sqlite3.connect(uri, uri=True)
        self.connection.row_factory = sqlite3.Row

    def fetch(self, chunk_ids: list[str]) -> list[dict[str, Any]]:
        if not chunk_ids:
            return []
        placeholders = ",".join("?" for _ in chunk_ids)
        rows = self.connection.execute(
            f"SELECT chunk_id, document_id, article_number, text FROM chunks WHERE chunk_id IN ({placeholders})",
            chunk_ids,
        ).fetchall()
        by_id = {row["chunk_id"]: row for row in rows}
        if set(by_id) != set(chunk_ids):
            raise E02CompareError("Fused retrieval refers to unknown chunk IDs.")
        return [
            {
                "chunk_id": chunk_id,
                "document_id": by_id[chunk_id]["document_id"],
                "article_number": by_id[chunk_id]["article_number"],
                "text": by_id[chunk_id]["text"],
            }
            for chunk_id in chunk_ids
        ]

    def close(self) -> None:
        self.connection.close()


class _TorchGpuFlatIpIndex:
    """Load a measured FAISS FlatIP matrix and execute the same exact IP on one GPU."""

    def __init__(
        self,
        path: Path,
        *,
        device: str,
        expected_count: int,
        expected_dimension: int,
    ) -> None:
        try:
            import faiss
            import numpy as np
            import torch
        except ImportError as exc:  # pragma: no cover - Kaggle dependency
            raise E02CompareError("Install faiss-cpu, numpy and torch.") from exc
        if not device.startswith("cuda:") or not torch.cuda.is_available():
            raise E02CompareError("E02 comparison exact search requires an explicit CUDA device.")
        index = faiss.read_index(str(path))
        if index.ntotal != expected_count or index.d != expected_dimension:
            raise E02CompareError("Loaded FAISS shape differs from the pinned artifact.")
        vectors = np.asarray(index.reconstruct_n(0, expected_count), dtype=np.float32)
        if vectors.shape != (expected_count, expected_dimension) or not np.isfinite(vectors).all():
            raise E02CompareError("Reconstructed FAISS vectors are invalid.")
        self._torch = torch
        self._np = np
        self._device = torch.device(device)
        self._vectors = torch.from_numpy(vectors).to(self._device, dtype=torch.float32)
        del vectors
        del index
        LOGGER.info(
            "e02_compare_gpu_index_loaded device=%s records=%d dimension=%d bytes=%d",
            device,
            expected_count,
            expected_dimension,
            self._vectors.numel() * self._vectors.element_size(),
        )

    def search(self, query: Any, top_k: int) -> tuple[Any, Any]:
        query_array = self._np.asarray(query, dtype=self._np.float32)
        if (
            query_array.ndim != 2
            or query_array.shape[0] <= 0
            or query_array.shape[1] != self._vectors.shape[1]
        ):
            raise E02CompareError("GPU search query shape is invalid.")
        with self._torch.inference_mode():
            query_tensor = self._torch.from_numpy(query_array).to(self._device)
            scores = self._torch.matmul(query_tensor, self._vectors.T)
            values, rows = self._torch.topk(
                scores, k=top_k, dim=1, largest=True, sorted=True
            )
        return values.cpu().numpy(), rows.cpu().numpy()

    def close(self) -> None:
        del self._vectors
        self._torch.cuda.empty_cache()


def _candidate_summary(records: list[dict[str, Any]], identity: dict[str, Any] | None) -> dict[str, Any]:
    if not records:
        raise E02CompareError("Cannot summarize empty candidate results.")
    reference_eligible = [
        row for row in records if row["diagnostics"]["explicit_reference_count"] > 0
    ]
    metrics = {
        "mean_answer_unique_token_coverage": fmean(
            row["diagnostics"]["answer_unique_token_coverage"] for row in records
        ),
        "explicit_reference_any_hit_rate": (
            fmean(bool(row["diagnostics"]["explicit_reference_any_hit"]) for row in reference_eligible)
            if reference_eligible else None
        ),
        "explicit_reference_all_hit_rate": (
            fmean(bool(row["diagnostics"]["explicit_reference_all_hit"]) for row in reference_eligible)
            if reference_eligible else None
        ),
        "mean_context_characters": fmean(
            row["diagnostics"]["context_character_count"] for row in records
        ),
        "mean_sparse_latency_ms": fmean(row["latency_ms"]["sparse"] for row in records),
        "mean_dense_latency_ms": fmean(
            row["latency_ms"]["dense_encode_and_search"] for row in records
        ),
    }
    return {"run_identity": identity, "sample_size": len(records), "metrics": metrics}


def _load_mapping(path: Path, expected_count: int) -> list[str]:
    mapping: list[str] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            try:
                chunk_id = json.loads(line)
            except json.JSONDecodeError as exc:
                raise E02CompareError(f"Invalid chunk mapping line {line_number}.") from exc
            if not isinstance(chunk_id, str) or not chunk_id or chunk_id in seen:
                raise E02CompareError(f"Invalid or duplicate chunk mapping at line {line_number}.")
            seen.add(chunk_id)
            mapping.append(chunk_id)
    if len(mapping) != expected_count:
        raise E02CompareError("Chunk mapping row count mismatch.")
    return mapping


def _candidate_run_identity(
    config: CompareConfig,
    candidate: CompareCandidate,
    dense_directory: Path,
    sample_ids: list[str],
) -> dict[str, Any]:
    identity = {
        "code_version": CODE_VERSION,
        "config_sha256": config.config_sha256,
        "candidate": candidate.__dict__,
        "dense_manifest_sha256": file_sha256(dense_directory / "manifest.json"),
        "dev_sha256": config.dev_sha256,
        "sample_ids_sha256": hashlib.sha256("\n".join(sample_ids).encode("utf-8")).hexdigest(),
    }
    identity["identity_sha256"] = _json_sha256(identity)
    return identity


def _load_completed(
    records_directory: Path,
    state_path: Path,
    identity: dict[str, Any],
    sample_ids: list[str],
) -> int:
    files = sorted(records_directory.glob("*.json"))
    if not state_path.exists():
        if files:
            raise E02CompareError("Checkpoint records exist without state.")
        _write_state(state_path, identity, 0, complete=False)
        return 0
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if state.get("run_identity") != identity:
        raise E02CompareError("Checkpoint belongs to a different comparison run.")
    if len(files) > len(sample_ids):
        raise E02CompareError("Checkpoint contains too many question records.")
    for index, path in enumerate(files):
        if path.name != f"{index:04d}.json":
            raise E02CompareError("Checkpoint record sequence is non-contiguous.")
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("sample_index") != index or record.get("question_id") != sample_ids[index]:
            raise E02CompareError("Checkpoint question identity mismatch.")
    if int(state.get("completed_count", -1)) > len(files):
        raise E02CompareError("Checkpoint state is ahead of durable records.")
    _write_state(state_path, identity, len(files), complete=len(files) == len(sample_ids))
    return len(files)


def _read_records(records_directory: Path, sample_ids: list[str]) -> list[dict[str, Any]]:
    records = [
        json.loads((records_directory / f"{index:04d}.json").read_text(encoding="utf-8"))
        for index in range(len(sample_ids))
    ]
    if [record.get("question_id") for record in records] != sample_ids:
        raise E02CompareError("Completed record order differs from the sample manifest.")
    return records


def _write_results_jsonl(path: Path, records_directory: Path, count: int) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        for index in range(count):
            record = json.loads((records_directory / f"{index:04d}.json").read_text(encoding="utf-8"))
            stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _write_state(path: Path, identity: dict[str, Any], completed: int, *, complete: bool) -> None:
    _atomic_json(path, {
        "schema_version": "1.0",
        "run_identity": identity,
        "completed_count": completed,
        "complete": complete,
    })


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _load_dev(path: Path) -> dict[str, dict[str, str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not payload:
        raise E02CompareError("Dev split must be a non-empty mapping.")
    result: dict[str, dict[str, str]] = {}
    for raw_id, record in payload.items():
        question_id = str(raw_id)
        if (
            not isinstance(record, dict)
            or set(record) != {"question", "answer"}
            or not isinstance(record["question"], str)
            or not record["question"].strip()
            or not isinstance(record["answer"], str)
        ):
            raise E02CompareError(f"Invalid dev record: {question_id}")
        result[question_id] = record
    return result


def _single_candidate_validation_config(config: CompareConfig, candidate_key: str) -> CompareConfig:
    # Validation can be scoped to the one candidate loaded by a process while
    # preserving the full run identity and configuration bytes.
    return CompareConfig(
        raw=config.raw,
        path=config.path,
        e00_manifest_sha256=config.e00_manifest_sha256,
        e00_chunks_sha256=config.e00_chunks_sha256,
        chunk_count=config.chunk_count,
        candidates={candidate_key: config.candidates[candidate_key]},
        dev_path=config.dev_path,
        dev_sha256=config.dev_sha256,
        sample_seed=config.sample_seed,
        sample_size=config.sample_size,
        sample_ids_sha256=config.sample_ids_sha256,
        candidate_k=config.candidate_k,
        top_k=config.top_k,
        rrf_constant=config.rrf_constant,
        branch_weights=config.branch_weights,
    )


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


def _string(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str):
        raise ValueError(f"{key} must be text.")
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


def _revision(payload: dict[str, Any], key: str) -> str:
    value = _text(payload, key).lower()
    if len(value) != 40 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{key} must be a 40-character immutable commit.")
    return value
