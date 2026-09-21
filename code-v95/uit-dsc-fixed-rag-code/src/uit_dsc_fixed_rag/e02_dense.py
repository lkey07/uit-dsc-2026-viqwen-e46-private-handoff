"""Fail-closed E02 dense A/B preparation and sharded FAISS build.

Heavy imports are intentionally lazy so repository validation and unit tests do
not download models or require the Kaggle-only numerical stack.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from uit_dsc_fixed_rag.corpus import file_sha256


CODE_VERSION = "0.6.0"
APPROVAL_KEYS = (
    "inventory_reviewed",
    "experiment_plan_reviewed",
    "weights_download_authorized",
)


class E02Error(RuntimeError):
    """Raised when E02 identity, authorization or artifact checks fail."""


@dataclass(frozen=True)
class DenseCandidate:
    """One immutable embedding candidate."""

    key: str
    model_id: str
    revision: str
    parameter_count: int
    output_dimension: int
    native_max_tokens: int
    query_prefix: str
    passage_prefix: str


@dataclass(frozen=True)
class E02Config:
    """Validated configuration for a one-candidate E02 run."""

    raw: dict[str, Any]
    source_artifact_version: str
    source_manifest_sha256: str
    source_chunks_sha256: str
    source_chunk_count: int
    dataset_revision: str
    inventory_path: str
    inventory_sha256: str
    experiment_plan_path: str
    experiment_plan_sha256: str
    candidates: dict[str, DenseCandidate]
    input_field: str
    target_max_tokens: int
    shared_hard_max_tokens: int
    add_special_tokens: bool
    audit_batch_size: int
    normalize_embeddings: bool
    stored_dtype: str
    shard_size: int
    encode_batch_size: int
    output_dimension: int

    @property
    def config_sha256(self) -> str:
        """Return the canonical identity of all E02 decisions."""

        return _json_sha256(self.raw)


def load_e02_config(path: Path) -> E02Config:
    """Load and strictly validate the E02 experiment contract."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("E02 config root must be an object.")
    required_root = {
        "schema_version",
        "experiment_id",
        "source_e00",
        "governance",
        "candidates",
        "token_audit",
        "encoding",
        "index",
        "fusion",
        "diagnostic",
    }
    if set(payload) != required_root or payload.get("experiment_id") != "E02":
        raise ValueError("E02 config root is incompatible.")
    if payload.get("schema_version") != "1.0":
        raise ValueError("E02 config schema_version must be 1.0.")

    source = _object(payload, "source_e00")
    governance = _object(payload, "governance")
    candidates_raw = _object(payload, "candidates")
    audit = _object(payload, "token_audit")
    encoding = _object(payload, "encoding")
    index = _object(payload, "index")
    fusion = _object(payload, "fusion")
    diagnostic = _object(payload, "diagnostic")

    source_artifact_version = source.get("artifact_version")
    if source_artifact_version not in {"e00-v1", "e00-v2"}:
        raise ValueError("E02 must use a pinned E00 sparse and chunk control.")
    if any(governance.get(key) is not False for key in (
        "allow_external_data", "allow_synthetic_data", "allow_model_api"
    )):
        raise ValueError("E02 governance must prohibit external/synthetic data and APIs.")
    if governance.get("require_runtime_download_authorization") is not True:
        raise ValueError("E02 must require explicit runtime model-download authorization.")
    if audit.get("input_field") != "text":
        raise ValueError("E02 embeds exact canonical chunk text only.")
    if encoding.get("checkpoint_after_each_shard") is not True:
        raise ValueError("E02 encoding must checkpoint after each shard.")
    if encoding.get("stored_dtype") != "float16" or encoding.get("compute_dtype") != "float32":
        raise ValueError("E02 storage/compute dtype contract changed.")
    if index != {
        "backend": "faiss-cpu",
        "factory": "Flat",
        "metric": "inner_product",
        "exact_search": True,
    }:
        raise ValueError("E02 must use the pinned exact CPU inner-product index.")
    expected_sparse_control = (
        "E00-v2-word-BM25" if source_artifact_version == "e00-v2" else "E00-word-BM25"
    )
    if fusion.get("sparse_control") != expected_sparse_control:
        raise ValueError("Rejected E01 must not be used as the E02 sparse control.")
    if diagnostic.get("checkpoint_every_questions") != 1:
        raise ValueError("E02 question diagnostics must checkpoint after every ID.")
    if diagnostic.get("promotion_allowed") is not False:
        raise ValueError("Answer-derived retrieval diagnostics cannot promote E02.")

    candidates: dict[str, DenseCandidate] = {}
    dimensions: set[int] = set()
    for key, raw in candidates_raw.items():
        if not isinstance(raw, dict):
            raise ValueError(f"E02 candidate {key} must be an object.")
        candidate = DenseCandidate(
            key=key,
            model_id=_text(raw, "model_id"),
            revision=_commit(raw, "revision"),
            parameter_count=_positive_int(raw, "parameter_count"),
            output_dimension=_positive_int(raw, "output_dimension"),
            native_max_tokens=_positive_int(raw, "native_max_tokens"),
            query_prefix=_string(raw, "query_prefix"),
            passage_prefix=_string(raw, "passage_prefix"),
        )
        candidates[key] = candidate
        dimensions.add(candidate.output_dimension)
    if set(candidates) != {"embedding_harrier", "embedding_aiteam"}:
        raise ValueError("E02 must define exactly the two approved embedding candidates.")
    if len(dimensions) != 1:
        raise ValueError("E02 candidates must emit the same vector dimension.")

    target = _positive_int(audit, "target_max_tokens")
    hard = _positive_int(audit, "shared_hard_max_tokens")
    if target > hard or audit.get("fail_on_hard_limit_exceeded") is not True:
        raise ValueError("E02 tokenizer limits are invalid or not fail-closed.")
    if any(hard > candidate.native_max_tokens for candidate in candidates.values()):
        raise ValueError("Shared hard token limit exceeds a candidate's native limit.")

    return E02Config(
        raw=payload,
        source_artifact_version=source_artifact_version,
        source_manifest_sha256=_sha256(source, "manifest_sha256"),
        source_chunks_sha256=_sha256(source, "chunks_sha256"),
        source_chunk_count=_positive_int(source, "chunk_count"),
        dataset_revision=_revision(source, "dataset_revision"),
        inventory_path=_text(governance, "model_inventory_path"),
        inventory_sha256=_sha256(governance, "model_inventory_sha256"),
        experiment_plan_path=_text(governance, "experiment_plan_path"),
        experiment_plan_sha256=_sha256(governance, "experiment_plan_sha256"),
        candidates=candidates,
        input_field="text",
        target_max_tokens=target,
        shared_hard_max_tokens=hard,
        add_special_tokens=_boolean(audit, "add_special_tokens"),
        audit_batch_size=_positive_int(audit, "batch_size"),
        normalize_embeddings=_boolean(encoding, "normalize_embeddings"),
        stored_dtype="float16",
        shard_size=_positive_int(encoding, "shard_size"),
        encode_batch_size=_positive_int(encoding, "batch_size"),
        output_dimension=next(iter(dimensions)),
    )


def validate_preflight(
    *,
    project_root: Path,
    e00_directory: Path,
    config: E02Config,
    candidate_key: str,
    approvals: dict[str, bool],
) -> DenseCandidate:
    """Validate immutable lineage and explicit runtime approvals before download."""

    if set(approvals) != set(APPROVAL_KEYS) or not all(approvals.values()):
        missing = [key for key in APPROVAL_KEYS if approvals.get(key) is not True]
        raise E02Error(f"E02 runtime approvals are incomplete: {missing}")
    candidate = config.candidates.get(candidate_key)
    if candidate is None:
        raise E02Error(f"Unknown E02 candidate: {candidate_key}")

    inventory_path = project_root / config.inventory_path
    plan_path = project_root / config.experiment_plan_path
    if file_sha256(inventory_path) != config.inventory_sha256:
        raise E02Error("Pinned model inventory checksum mismatch.")
    if file_sha256(plan_path) != config.experiment_plan_sha256:
        raise E02Error("Pinned experiment plan checksum mismatch.")
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if plan.get("experiment_sequence", [None, None, {}])[2].get("id") != "E02":
        raise E02Error("Pinned experiment plan does not contain E02 at the expected step.")
    models = {model.get("key"): model for model in inventory.get("models", [])}
    registered = models.get(candidate_key)
    expected = {
        "model_id": candidate.model_id,
        "revision": candidate.revision,
        "parameter_count": candidate.parameter_count,
    }
    if registered is None or any(registered.get(key) != value for key, value in expected.items()):
        raise E02Error("E02 candidate differs from the pinned model inventory.")

    _validate_e00(e00_directory, config)
    return candidate


def load_sentence_transformer(
    candidate: DenseCandidate,
    config: E02Config,
    *,
    device: str,
) -> Any:
    """Load exactly one pinned candidate after preflight authorization."""

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:  # pragma: no cover - Kaggle-only dependency
        raise E02Error("Install sentence-transformers in the Kaggle runtime.") from exc
    model = SentenceTransformer(
        candidate.model_id,
        revision=candidate.revision,
        device=device,
        trust_remote_code=False,
    )
    # Token counts are tokenizer-specific. The same canonical text can be 384
    # Harrier tokens but more than 512 AITeam tokens, so enforce each pinned
    # model's native context limit rather than truncating a fair A/B input.
    model.max_seq_length = candidate.native_max_tokens
    observed_parameters = sum(parameter.numel() for parameter in model.parameters())
    if observed_parameters != candidate.parameter_count:
        raise E02Error(
            "Downloaded model parameter count mismatch: "
            f"expected {candidate.parameter_count}, observed {observed_parameters}."
        )
    dimension_getter = getattr(model, "get_embedding_dimension", None)
    dimension = (
        dimension_getter()
        if dimension_getter is not None
        else model.get_sentence_embedding_dimension()
    )
    if dimension != candidate.output_dimension:
        raise E02Error(
            f"Embedding dimension mismatch: expected {candidate.output_dimension}, got {dimension}."
        )
    return model


def audit_token_lengths(
    *,
    e00_directory: Path,
    output_path: Path,
    config: E02Config,
    candidate: DenseCandidate,
    model: Any,
) -> dict[str, Any]:
    """Measure every canonical chunk with the exact downloaded tokenizer."""

    tokenizer = model.tokenizer
    lengths: list[int] = []
    longest: list[tuple[int, str]] = []
    target_exceeded = 0
    shared_hard_exceeded = 0
    hard_exceeded = 0
    batch_ids: list[str] = []
    batch_texts: list[str] = []

    def consume() -> None:
        nonlocal target_exceeded, shared_hard_exceeded, hard_exceeded
        if not batch_texts:
            return
        encoded = tokenizer(
            [candidate.passage_prefix + text for text in batch_texts],
            add_special_tokens=config.add_special_tokens,
            truncation=False,
            padding=False,
            return_length=True,
        )
        observed = encoded["length"]
        if len(observed) != len(batch_texts):
            raise E02Error("Tokenizer length output is incomplete.")
        for chunk_id, length in zip(batch_ids, observed):
            value = int(length)
            lengths.append(value)
            target_exceeded += value > config.target_max_tokens
            shared_hard_exceeded += value > config.shared_hard_max_tokens
            hard_exceeded += value > candidate.native_max_tokens
            longest.append((value, chunk_id))
        batch_ids.clear()
        batch_texts.clear()

    for record in iter_e00_chunks(e00_directory / "chunks.jsonl", config):
        batch_ids.append(record["chunk_id"])
        batch_texts.append(record[config.input_field])
        if len(batch_texts) >= config.audit_batch_size:
            consume()
    consume()
    ordered = sorted(lengths)
    longest.sort(reverse=True)
    report = {
        "schema_version": "1.0",
        "artifact_type": "e02-exact-token-audit",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_sha256": config.config_sha256,
        "candidate": _candidate_identity(candidate),
        "source_chunks_sha256": config.source_chunks_sha256,
        "record_count": len(lengths),
        "target_max_tokens": config.target_max_tokens,
        "shared_hard_max_tokens": config.shared_hard_max_tokens,
        "candidate_hard_max_tokens": candidate.native_max_tokens,
        "target_exceeded_count": target_exceeded,
        "shared_hard_exceeded_count": shared_hard_exceeded,
        "hard_exceeded_count": hard_exceeded,
        "lengths": {
            "minimum": ordered[0],
            "median": _percentile(ordered, 0.50),
            "p95": _percentile(ordered, 0.95),
            "p99": _percentile(ordered, 0.99),
            "maximum": ordered[-1],
        },
        "longest_chunks": [
            {"chunk_id": chunk_id, "tokens": tokens}
            for tokens, chunk_id in longest[:20]
        ],
        "hard_gate_passed": hard_exceeded == 0,
    }
    _atomic_json(output_path, report)
    if hard_exceeded:
        raise E02Error(
            f"Exact tokenizer audit found {hard_exceeded} chunks above the candidate native limit; "
            "rechunk and rebuild E00 before dense encoding."
        )
    return report


def encode_shards_and_build_index(
    *,
    e00_directory: Path,
    output_directory: Path,
    config: E02Config,
    candidate: DenseCandidate,
    model: Any,
    token_audit_path: Path,
    encode_devices: tuple[str, ...] | list[str] | None = None,
) -> dict[str, Any]:
    """Resume sharded encoding, then publish an exact CPU FAISS index."""

    try:
        import faiss
        import numpy as np
    except ImportError as exc:  # pragma: no cover - Kaggle-only dependencies
        raise E02Error("Install faiss-cpu and numpy in the Kaggle runtime.") from exc

    audit = json.loads(token_audit_path.read_text(encoding="utf-8"))
    if (
        audit.get("config_sha256") != config.config_sha256
        or audit.get("candidate") != _candidate_identity(candidate)
        or audit.get("source_chunks_sha256") != config.source_chunks_sha256
        or audit.get("hard_gate_passed") is not True
    ):
        raise E02Error("Exact tokenizer audit is missing, stale or failed.")

    devices = tuple(encode_devices or ())
    if not devices:
        devices = (str(model.device),)
    if any(not device.strip() for device in devices) or len(set(devices)) != len(devices):
        raise E02Error("Encoding devices must be non-empty and unique.")

    output_directory.mkdir(parents=True, exist_ok=True)
    shards_directory = output_directory / "shards"
    shards_directory.mkdir(exist_ok=True)
    state_path = output_directory / "state.json"
    identity = _run_identity(config, candidate, encode_devices=devices)
    state = _load_or_create_state(state_path, identity)

    pool = None
    try:
        if len(devices) > 1:
            pool = model.start_multi_process_pool(target_devices=list(devices))

        shard_ids: list[str] = []
        shard_texts: list[str] = []
        shard_index = 0
        for record in iter_e00_chunks(e00_directory / "chunks.jsonl", config):
            shard_ids.append(record["chunk_id"])
            shard_texts.append(candidate.passage_prefix + record[config.input_field])
            if len(shard_ids) < config.shard_size:
                continue
            _encode_one_shard(
                shard_index, shard_ids, shard_texts, shards_directory,
                state, state_path, identity, config, model, np, pool,
            )
            shard_index += 1
            shard_ids, shard_texts = [], []
        if shard_ids:
            _encode_one_shard(
                shard_index, shard_ids, shard_texts, shards_directory,
                state, state_path, identity, config, model, np, pool,
            )
    finally:
        if pool is not None:
            model.stop_multi_process_pool(pool)

    expected_shards = (config.source_chunk_count + config.shard_size - 1) // config.shard_size
    if len(state["shards"]) != expected_shards:
        raise E02Error("Encoding checkpoint does not contain every expected shard.")

    index = faiss.IndexFlatIP(config.output_dimension)
    map_temp = output_directory / ".chunk_ids.jsonl.tmp"
    with map_temp.open("w", encoding="utf-8", newline="\n") as mapping:
        for shard in state["shards"]:
            vector_path = shards_directory / shard["vectors_file"]
            ids_path = shards_directory / shard["ids_file"]
            if file_sha256(vector_path) != shard["vectors_sha256"]:
                raise E02Error(f"Vector shard checksum mismatch: {vector_path.name}")
            if file_sha256(ids_path) != shard["ids_sha256"]:
                raise E02Error(f"ID shard checksum mismatch: {ids_path.name}")
            vectors = np.load(vector_path, allow_pickle=False)
            if vectors.shape != (shard["record_count"], config.output_dimension):
                raise E02Error(f"Vector shard shape mismatch: {vector_path.name}")
            index.add(np.asarray(vectors, dtype=np.float32))
            mapping.write(ids_path.read_text(encoding="utf-8"))
    map_path = output_directory / "chunk_ids.jsonl"
    os.replace(map_temp, map_path)
    if index.ntotal != config.source_chunk_count:
        raise E02Error("FAISS row count differs from the canonical chunk count.")

    index_temp = output_directory / ".dense.faiss.tmp"
    index_path = output_directory / "dense.faiss"
    faiss.write_index(index, str(index_temp))
    os.replace(index_temp, index_path)
    manifest = {
        "schema_version": "1.0",
        "artifact_type": "e02-dense-flat-ip",
        "artifact_version": "e02-v2",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "code_version": CODE_VERSION,
        "run_identity": identity,
        "candidate": _candidate_identity(candidate),
        "source_e00_manifest_sha256": config.source_manifest_sha256,
        "source_chunks_sha256": config.source_chunks_sha256,
        "record_count": config.source_chunk_count,
        "dimension": config.output_dimension,
        "normalized": config.normalize_embeddings,
        "encoding_devices": list(devices),
        "encoding_processes": len(devices),
        "index": {"backend": "faiss-cpu", "factory": "Flat", "metric": "inner_product"},
        "files": {
            "dense.faiss": _file_identity(index_path),
            "chunk_ids.jsonl": _file_identity(map_path),
            "state.json": _file_identity(state_path),
            "token-audit.json": _file_identity(token_audit_path),
        },
        "warnings": [
            "This artifact contains no retrieval labels.",
            "E02 promotion requires answer-level evaluation with the frozen generator."
        ],
    }
    _atomic_json(output_directory / "manifest.json", manifest)
    return manifest


def iter_e00_chunks(path: Path, config: E02Config) -> Iterator[dict[str, str]]:
    """Stream exact E00 chunks while verifying count and byte checksum."""

    digest = hashlib.sha256()
    count = 0
    with path.open("rb") as stream:
        for line_number, raw_line in enumerate(stream, start=1):
            digest.update(raw_line)
            try:
                record = json.loads(raw_line.decode("utf-8"))
            except (UnicodeError, json.JSONDecodeError) as exc:
                raise E02Error(f"Invalid E00 chunk JSON at line {line_number}.") from exc
            chunk_id = record.get("chunk_id") if isinstance(record, dict) else None
            text = record.get(config.input_field) if isinstance(record, dict) else None
            if not isinstance(chunk_id, str) or not chunk_id or not isinstance(text, str) or not text:
                raise E02Error(f"Invalid E00 dense input at line {line_number}.")
            count += 1
            yield {"chunk_id": chunk_id, config.input_field: text}
    if count != config.source_chunk_count or digest.hexdigest() != config.source_chunks_sha256:
        raise E02Error("E00 chunks changed while being streamed.")


def _encode_one_shard(
    shard_index: int,
    ids: list[str],
    texts: list[str],
    directory: Path,
    state: dict[str, Any],
    state_path: Path,
    identity: dict[str, Any],
    config: E02Config,
    model: Any,
    np: Any,
    pool: Any | None,
) -> None:
    vectors_name = f"vectors-{shard_index:05d}.npy"
    ids_name = f"ids-{shard_index:05d}.jsonl"
    if shard_index < len(state["shards"]):
        saved = state["shards"][shard_index]
        if saved.get("record_count") != len(ids) or saved.get("ids_order_sha256") != _ids_sha256(ids):
            raise E02Error(f"Resume shard identity mismatch at shard {shard_index}.")
        return
    if shard_index != len(state["shards"]):
        raise E02Error("Encoding state contains a non-contiguous shard sequence.")

    encode_kwargs = {
        "batch_size": config.encode_batch_size,
        "show_progress_bar": True,
        "convert_to_numpy": True,
        "normalize_embeddings": config.normalize_embeddings,
    }
    if pool is not None:
        encode_kwargs["pool"] = pool
    vectors = model.encode(texts, **encode_kwargs)
    vectors = np.asarray(vectors, dtype=np.float16)
    if vectors.shape != (len(ids), config.output_dimension) or not np.isfinite(vectors).all():
        raise E02Error(f"Invalid vectors produced for shard {shard_index}.")
    vector_path = directory / vectors_name
    ids_path = directory / ids_name
    with tempfile.NamedTemporaryFile(dir=directory, prefix=".vectors-", suffix=".npy", delete=False) as tmp:
        vector_tmp = Path(tmp.name)
        np.save(tmp, vectors, allow_pickle=False)
    ids_tmp = directory / f".{ids_name}.tmp"
    ids_tmp.write_text("".join(json.dumps(item) + "\n" for item in ids), encoding="utf-8", newline="\n")
    os.replace(vector_tmp, vector_path)
    os.replace(ids_tmp, ids_path)
    state["shards"].append({
        "shard_index": shard_index,
        "record_count": len(ids),
        "ids_order_sha256": _ids_sha256(ids),
        "vectors_file": vectors_name,
        "vectors_sha256": file_sha256(vector_path),
        "ids_file": ids_name,
        "ids_sha256": file_sha256(ids_path),
    })
    _atomic_json(state_path, state)


def _validate_e00(directory: Path, config: E02Config) -> None:
    manifest_path = directory / "manifest.json"
    chunks_path = directory / "chunks.jsonl"
    database_path = directory / "bm25.sqlite3"
    if not all(path.is_file() for path in (manifest_path, chunks_path, database_path)):
        raise E02Error("E00 manifest, chunks or BM25 control is missing.")
    if file_sha256(manifest_path) != config.source_manifest_sha256:
        raise E02Error("Pinned E00 manifest checksum mismatch.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("artifact_version") != config.source_artifact_version
        or manifest.get("dataset_revision") != config.dataset_revision
        or manifest.get("record_counts", {}).get("chunks") != config.source_chunk_count
        or manifest.get("files", {}).get("chunks.jsonl", {}).get("sha256") != config.source_chunks_sha256
    ):
        raise E02Error("Pinned E00 manifest content is incompatible with E02.")


def _load_or_create_state(path: Path, identity: dict[str, Any]) -> dict[str, Any]:
    if path.exists():
        state = json.loads(path.read_text(encoding="utf-8"))
        if state.get("run_identity") != identity or not isinstance(state.get("shards"), list):
            raise E02Error("Existing E02 checkpoint belongs to a different run.")
        return state
    state = {"schema_version": "1.0", "run_identity": identity, "shards": []}
    _atomic_json(path, state)
    return state


def _run_identity(
    config: E02Config,
    candidate: DenseCandidate,
    *,
    encode_devices: tuple[str, ...],
) -> dict[str, Any]:
    packages = {}
    for package in ("sentence-transformers", "transformers", "torch", "numpy", "faiss-cpu"):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = "missing"
    identity = {
        "config_sha256": config.config_sha256,
        "candidate": _candidate_identity(candidate),
        "source_chunks_sha256": config.source_chunks_sha256,
        "code_version": CODE_VERSION,
        "encoding_devices": list(encode_devices),
        "python": platform.python_version(),
        "packages": packages,
    }
    identity["identity_sha256"] = _json_sha256(identity)
    return identity


def _candidate_identity(candidate: DenseCandidate) -> dict[str, Any]:
    return {
        "key": candidate.key,
        "model_id": candidate.model_id,
        "revision": candidate.revision,
        "parameter_count": candidate.parameter_count,
        "output_dimension": candidate.output_dimension,
    }


def _file_identity(path: Path) -> dict[str, Any]:
    return {"bytes": path.stat().st_size, "sha256": file_sha256(path)}


def _ids_sha256(ids: list[str]) -> str:
    digest = hashlib.sha256()
    for item in ids:
        digest.update(item.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, path)


def _json_sha256(payload: Any) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _percentile(values: list[int], ratio: float) -> int:
    if not values:
        raise E02Error("Cannot summarize an empty token audit.")
    return values[round((len(values) - 1) * ratio)]


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
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{key} must be a positive integer.")
    return value


def _boolean(payload: dict[str, Any], key: str) -> bool:
    value = payload.get(key)
    if not isinstance(value, bool):
        raise ValueError(f"{key} must be boolean.")
    return value


def _sha256(payload: dict[str, Any], key: str) -> str:
    value = _text(payload, key)
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{key} must be a lowercase SHA-256 digest.")
    return value


def _commit(payload: dict[str, Any], key: str) -> str:
    value = _text(payload, key)
    if len(value) != 40 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{key} must be an immutable lowercase commit hash.")
    return value


def _revision(payload: dict[str, Any], key: str) -> str:
    value = _text(payload, key)
    if not value.startswith("sha256:") or len(value) != 71:
        raise ValueError(f"{key} must be a sha256: revision.")
    return value
