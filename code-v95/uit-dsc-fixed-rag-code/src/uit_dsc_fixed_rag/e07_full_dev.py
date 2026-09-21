"""E07 full-dev-721 promotion for the full-train LoRA adapter."""

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
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean
from typing import Any

from uit_dsc_fixed_rag.bm25 import SqliteBm25Index
from uit_dsc_fixed_rag.corpus import file_sha256
from uit_dsc_fixed_rag.e02_answer import _atomic_json, _atomic_jsonl, pack_contexts
from uit_dsc_fixed_rag.e02_compare import (
    _ChunkStore, _TorchGpuFlatIpIndex, _load_dev, _load_mapping, weighted_rrf,
)
from uit_dsc_fixed_rag.e03_rrf_grid import (
    _bootstrap_ci, _json_sha256, _load_contiguous, _load_worker_progress,
    _read_jsonl, _validate_worker_placement, _write_jsonl_from_records,
    _write_state, _write_worker_state,
)
from uit_dsc_fixed_rag.e07_lora import InferencePacking
from uit_dsc_fixed_rag.evaluation.official import (
    ensure_nltk_resources, nltk_meteor_score, rouge_l_fmeasure,
)
from uit_dsc_fixed_rag.retrieval_diagnostics import select_dev_sample


CODE_VERSION = "0.14.0"
LOGGER = logging.getLogger(__name__)


class E07FullDevError(RuntimeError):
    """Raised when full-dev promotion cannot continue safely."""


@dataclass(frozen=True)
class E07FullDevConfig:
    raw: dict[str, Any]
    path: Path

    @property
    def config_sha256(self) -> str:
        return file_sha256(self.path)

    def section(self, key: str) -> dict[str, Any]:
        value = self.raw.get(key)
        if not isinstance(value, dict):
            raise ValueError(f"E07 full-dev section must be an object: {key}")
        return value


def load_full_dev_config(path: Path) -> E07FullDevConfig:
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "schema_version", "experiment_id", "advancement_source", "source_e06",
        "source_e00", "source_dense", "dev", "retrieval", "generator",
        "inference", "parameter_budget", "execution", "scoring", "run_contract",
    }
    if (
        set(payload) != required
        or payload.get("schema_version") != "1.0"
        or payload.get("experiment_id")
        != "E07-lora-aiteam-fulltrain-fulldev721-promotion-v1"
    ):
        raise ValueError("E07 full-dev config root is incompatible.")
    config = E07FullDevConfig(payload, path)
    dev = config.section("dev")
    if (
        dev.get("sample_size") != 721 or dev.get("control_prefix_size") != 200
        or dev.get("sha256")
        != "2209b5e066ee354aae00f3b2b1aa69dbf71097c2ccd354f18b2f60daab128dcb"
    ):
        raise ValueError("E07 full-dev split contract changed.")
    retrieval = config.section("retrieval")
    if retrieval != {
        "candidate_k_per_branch": 40, "rrf_constant": 60,
        "sparse_weight": 0.5, "dense_weight": 0.5, "fused_top_k": 20,
        "selected_contexts": 12, "normalize_query_embeddings": True,
        "dense_search_backend": "torch-cuda-exact-flat-ip-float32-batched",
    }:
        raise ValueError("E07 full-dev retrieval contract changed.")
    inference = config.section("inference")
    if (
        inference.get("max_input_tokens") != 8192
        or inference.get("max_new_tokens") != 384
        or inference.get("do_sample") is not False
        or inference.get("num_beams") != 1
        or inference.get("enable_thinking") is not False
    ):
        raise ValueError("E07 full-dev inference contract changed.")
    budget = config.section("parameter_budget")
    if (
        budget.get("maximum_stack_total")
        != budget.get("embedding") + budget.get("generator")
        + budget.get("adapter_parameter_cap")
        or budget["maximum_stack_total"] >= budget["exclusive_limit"]
    ):
        raise ValueError("E07 full-dev parameter budget failed.")
    contract = config.section("run_contract")
    if not all(value is True for value in contract.values()):
        raise ValueError("E07 full-dev run contract lost an invariant.")
    return config


def validate_full_dev_preflight(
    *, project_root: Path, e00_directory: Path, dense_directory: Path,
    e06_directory: Path, full_lora_directory: Path, dev_path: Path,
    config: E07FullDevConfig,
) -> dict[str, Any]:
    scorer_cfg = config.section("scoring")
    scorer = project_root / scorer_cfg["official_scorer_path"]
    if not scorer.is_file() or file_sha256(scorer) != scorer_cfg["official_scorer_sha256"]:
        raise E07FullDevError("Pinned official scorer is missing or changed.")
    dev_cfg = config.section("dev")
    if not dev_path.is_file() or file_sha256(dev_path) != dev_cfg["sha256"]:
        raise E07FullDevError("Pinned dev split is missing or changed.")
    dev = _load_dev(dev_path)
    ids = select_dev_sample(dev, seed=dev_cfg["sample_seed"], size=dev_cfg["sample_size"])
    ids_sha = _ids_sha(ids)
    prefix_sha = _ids_sha(ids[:dev_cfg["control_prefix_size"]])
    if ids_sha != dev_cfg["sample_ids_sha256"] or prefix_sha != dev_cfg[
        "control_prefix_ids_sha256"
    ]:
        raise E07FullDevError("Full-dev deterministic question order changed.")

    e00_cfg = config.section("source_e00")
    manifest_path, bm25_path = e00_directory / "manifest.json", e00_directory / "bm25.sqlite3"
    if (
        not manifest_path.is_file() or file_sha256(manifest_path) != e00_cfg["manifest_sha256"]
        or not bm25_path.is_file() or bm25_path.stat().st_size != e00_cfg["bm25_bytes"]
        or file_sha256(bm25_path) != e00_cfg["bm25_sha256"]
    ):
        raise E07FullDevError("Pinned E00 v2 artifact is missing or changed.")

    dense_cfg = config.section("source_dense")
    dense_manifest_path = dense_directory / "manifest.json"
    dense_path, mapping_path = dense_directory / "dense.faiss", dense_directory / "chunk_ids.jsonl"
    if not all(path.is_file() for path in (dense_manifest_path, dense_path, mapping_path)):
        raise E07FullDevError("Pinned AITeam dense artifact is incomplete.")
    dense_manifest = json.loads(dense_manifest_path.read_text(encoding="utf-8"))
    if (
        dense_manifest.get("record_count") != dense_cfg["record_count"]
        or dense_manifest.get("dimension") != dense_cfg["dimension"]
        or dense_manifest.get("candidate", {}).get("key") != dense_cfg["candidate"]
        or file_sha256(dense_path) != dense_cfg["dense_faiss_sha256"]
        or file_sha256(mapping_path) != dense_cfg["chunk_ids_sha256"]
    ):
        raise E07FullDevError("Pinned AITeam dense artifact changed.")

    e06_cfg = config.section("source_e06")
    e06_report_path = e06_directory / "report.json"
    e06_generation = e06_directory / "generation" / "results.jsonl"
    e06_prepared = e06_directory / "prepared" / "results.jsonl"
    if not all(path.is_file() for path in (e06_report_path, e06_generation, e06_prepared)):
        raise E07FullDevError("Saved E06 artifact is incomplete.")
    e06_report = json.loads(e06_report_path.read_text(encoding="utf-8"))
    if (
        e06_report.get("experiment_id") != e06_cfg["report_experiment_id"]
        or e06_report.get("sample_size") != e06_cfg["sample_size"]
        or e06_report.get("evidence", {}).get("config_sha256")
        != e06_cfg["report_config_sha256"]
        or file_sha256(e06_generation) != e06_cfg["generation_results_sha256"]
        or file_sha256(e06_prepared) != e06_cfg["prepared_results_sha256"]
    ):
        raise E07FullDevError("Saved E06 lineage changed.")

    advance = config.section("advancement_source")
    full_report_path = full_lora_directory / "report.json"
    full_results = full_lora_directory / "evaluation" / "results.jsonl"
    adapter = full_lora_directory / "training" / "adapter-final" / "adapter_model.safetensors"
    if not all(path.is_file() for path in (full_report_path, full_results, adapter)):
        raise E07FullDevError("Saved full-train LoRA artifact is incomplete.")
    full_report = json.loads(full_report_path.read_text(encoding="utf-8"))
    _validate_full_train_report(full_report, advance)
    if (
        file_sha256(full_results) != advance["evaluation_results_sha256"]
        or file_sha256(adapter) != advance["adapter_sha256"]
    ):
        raise E07FullDevError("Saved full-train LoRA evidence changed.")
    return {
        "config_sha256": config.config_sha256, "dev_sha256": dev_cfg["sha256"],
        "sample_ids_sha256": ids_sha, "sample_size": len(ids),
        "control_prefix_ids_sha256": prefix_sha,
        "e00_manifest_sha256": e00_cfg["manifest_sha256"],
        "bm25_sha256": e00_cfg["bm25_sha256"],
        "dense_faiss_sha256": dense_cfg["dense_faiss_sha256"],
        "e06_generation_results_sha256": e06_cfg["generation_results_sha256"],
        "e06_prepared_results_sha256": e06_cfg["prepared_results_sha256"],
        "full_lora_report_sha256": file_sha256(full_report_path),
        "full_lora_results_sha256": advance["evaluation_results_sha256"],
        "adapter_sha256": advance["adapter_sha256"],
        "maximum_stack_parameters": config.section("parameter_budget")["maximum_stack_total"],
    }


def prepare_local_bm25(
    *, source: Path, runtime_directory: Path, config: E07FullDevConfig,
) -> Path:
    source_cfg = config.section("source_e00")
    runtime_directory.mkdir(parents=True, exist_ok=True)
    destination, manifest_path = runtime_directory / "bm25.sqlite3", runtime_directory / "manifest.json"
    identity = {
        "schema_version": "1.0", "config_sha256": config.config_sha256,
        "source_sha256": source_cfg["bm25_sha256"], "bytes": source_cfg["bm25_bytes"],
    }
    if destination.is_file() and manifest_path.is_file():
        if json.loads(manifest_path.read_text(encoding="utf-8")) == identity:
            return destination
        raise E07FullDevError("Local BM25 cache belongs to another run.")
    if destination.exists() or manifest_path.exists():
        raise E07FullDevError("Partial local BM25 cache exists.")
    temporary = runtime_directory / "bm25.sqlite3.tmp"
    with source.open("rb") as input_stream, temporary.open("xb") as output_stream:
        shutil.copyfileobj(input_stream, output_stream, length=16 * 1024 * 1024)
        output_stream.flush()
        os.fsync(output_stream.fileno())
    if (
        temporary.stat().st_size != source_cfg["bm25_bytes"]
        or file_sha256(temporary) != source_cfg["bm25_sha256"]
    ):
        raise E07FullDevError("Local BM25 copy verification failed.")
    os.replace(temporary, destination)
    _atomic_json(manifest_path, identity)
    return destination


def run_sparse_retrieval(
    *, bm25_path: Path, dev_path: Path, output_directory: Path,
    config: E07FullDevConfig,
) -> dict[str, Any]:
    dev, dev_cfg = _load_dev(dev_path), config.section("dev")
    ids = select_dev_sample(dev, seed=dev_cfg["sample_seed"], size=dev_cfg["sample_size"])
    identity = {
        "code_version": CODE_VERSION, "stage": "full-dev-bm25",
        "config_sha256": config.config_sha256,
        "bm25_sha256": config.section("source_e00")["bm25_sha256"],
        "sample_ids_sha256": _ids_sha(ids),
    }
    identity["identity_sha256"] = _json_sha256(identity)
    root, records = output_directory / "retrieval" / "sparse", output_directory / "retrieval" / "sparse" / "records"
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
        hits = index.search(dev[question_id]["question"], top_k=depth)
        return {
            "ids": [hit.chunk_id for hit in hits],
            "scores": [float(hit.score) for hit in hits],
            "latency_ms": (time.perf_counter() - started) * 1000,
        }

    workers = config.section("execution")["sparse_worker_threads"]
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(search, question_id) for question_id in pending]
        for offset, (question_id, future) in enumerate(zip(pending, futures)):
            index, result = completed + offset, future.result()
            _atomic_json(records / f"{index:04d}.json", {
                "question_id": question_id, "sample_index": index,
                "sparse_top_ids": result["ids"], "sparse_scores": result["scores"],
                "latency_ms": result["latency_ms"],
            })
            _write_state(state_path, identity, index + 1, complete=False)
            LOGGER.info("e07_full_dev_sparse_progress completed=%d total=%d", index + 1, len(ids))
    _write_jsonl_from_records(root / "results.jsonl", records, len(ids))
    _write_state(state_path, identity, len(ids), complete=True)
    return {"sample_size": len(ids), "results_sha256": file_sha256(root / "results.jsonl")}


def load_embedding(config: E07FullDevConfig, device: str) -> Any:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:  # pragma: no cover
        raise E07FullDevError("Install sentence-transformers for full-dev retrieval.") from exc
    dense = config.section("source_dense")
    model = SentenceTransformer(
        dense["model_id"], revision=dense["revision"], device=device,
        trust_remote_code=False,
    )
    model.max_seq_length = 512
    observed = sum(parameter.numel() for parameter in model.parameters())
    dimension_getter = getattr(model, "get_embedding_dimension", None)
    dimension = dimension_getter() if dimension_getter else model.get_sentence_embedding_dimension()
    if observed != dense["parameter_count"] or dimension != dense["dimension"]:
        raise E07FullDevError("AITeam embedding identity changed.")
    return model


def run_dense_and_fusion(
    *, model: Any, dense_directory: Path, bm25_path: Path, dev_path: Path,
    e06_directory: Path, output_directory: Path, config: E07FullDevConfig,
) -> dict[str, Any]:
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover
        raise E07FullDevError("Install numpy for dense retrieval.") from exc
    dev, dev_cfg = _load_dev(dev_path), config.section("dev")
    ids = select_dev_sample(dev, seed=dev_cfg["sample_seed"], size=dev_cfg["sample_size"])
    sparse_path = output_directory / "retrieval" / "sparse" / "results.jsonl"
    sparse = _read_jsonl(sparse_path)
    if [row.get("question_id") for row in sparse] != ids:
        raise E07FullDevError("Full-dev sparse cache order changed.")
    e06_cfg = config.section("source_e06")
    e06_prepared_path = e06_directory / "prepared" / "results.jsonl"
    if file_sha256(e06_prepared_path) != e06_cfg["prepared_results_sha256"]:
        raise E07FullDevError("E06 context control changed.")
    e06_prepared = _read_jsonl(e06_prepared_path)
    dense_cfg, retrieval = config.section("source_dense"), config.section("retrieval")
    identity = {
        "code_version": CODE_VERSION, "stage": "full-dev-aiteam-rrf-top12",
        "config_sha256": config.config_sha256,
        "sparse_results_sha256": file_sha256(sparse_path),
        "dense_faiss_sha256": dense_cfg["dense_faiss_sha256"],
        "sample_ids_sha256": _ids_sha(ids),
    }
    identity["identity_sha256"] = _json_sha256(identity)
    root, records = output_directory / "retrieval", output_directory / "retrieval" / "records"
    records.mkdir(parents=True, exist_ok=True)
    state_path = root / "state.json"
    completed = _load_contiguous(records, state_path, identity, ids)
    mapping = _load_mapping(dense_directory / "chunk_ids.jsonl", dense_cfg["record_count"])
    search = _TorchGpuFlatIpIndex(
        dense_directory / "dense.faiss", device=config.section("execution")["dense_device"],
        expected_count=dense_cfg["record_count"], expected_dimension=dense_cfg["dimension"],
    )
    store = _ChunkStore(bm25_path)
    batch_size = config.section("execution")["dense_query_batch_size"]
    try:
        for start in range(completed, len(ids), batch_size):
            stop = min(start + batch_size, len(ids))
            batch_ids = ids[start:stop]
            vectors = model.encode(
                [dense_cfg["query_prefix"] + dev[qid]["question"] for qid in batch_ids],
                batch_size=len(batch_ids), show_progress_bar=False,
                convert_to_numpy=True, normalize_embeddings=True,
            )
            vectors = np.asarray(vectors, dtype=np.float32)
            if vectors.shape != (len(batch_ids), dense_cfg["dimension"]) or not np.isfinite(vectors).all():
                raise E07FullDevError("Invalid full-dev query embeddings.")
            dense_scores, dense_rows = search.search(vectors, retrieval["candidate_k_per_branch"])
            for offset, question_id in enumerate(batch_ids):
                index = start + offset
                dense_ids = [mapping[int(row)] for row in dense_rows[offset] if int(row) >= 0]
                fused = weighted_rrf(
                    {"sparse": sparse[index]["sparse_top_ids"], "dense": dense_ids},
                    weights={"sparse": retrieval["sparse_weight"], "dense": retrieval["dense_weight"]},
                    constant=retrieval["rrf_constant"], top_k=retrieval["fused_top_k"],
                )
                contexts = store.fetch([row["chunk_id"] for row in fused])
                selected = contexts[:retrieval["selected_contexts"]]
                if index < dev_cfg["control_prefix_size"]:
                    expected = [row["chunk_id"] for row in e06_prepared[index]["contexts"]]
                    observed = [row["chunk_id"] for row in selected]
                    if observed != expected:
                        raise E07FullDevError(f"Full-dev retrieval differs from E06 control: {question_id}")
                _atomic_json(records / f"{index:04d}.json", {
                    "question_id": question_id, "sample_index": index,
                    "sparse_top_ids": sparse[index]["sparse_top_ids"],
                    "dense_top_ids": dense_ids,
                    "dense_scores": [float(value) for value in dense_scores[offset][:len(dense_ids)]],
                    "fused": fused, "contexts": selected,
                })
                _write_state(state_path, identity, index + 1, complete=False)
            LOGGER.info("e07_full_dev_dense_progress completed=%d total=%d", stop, len(ids))
    finally:
        search.close()
        store.close()
    _write_jsonl_from_records(root / "results.jsonl", records, len(ids))
    _write_state(state_path, identity, len(ids), complete=True)
    return {"sample_size": len(ids), "results_sha256": file_sha256(root / "results.jsonl")}


def load_generator_with_adapter(
    *, full_lora_directory: Path, config: E07FullDevConfig, device: str,
) -> tuple[Any, Any, dict[str, Any], int]:
    try:
        import torch
        from peft import PeftModel
        from transformers import AutoTokenizer, Qwen3_5ForCausalLM
    except ImportError as exc:  # pragma: no cover
        raise E07FullDevError("Install Transformers and PEFT for full-dev generation.") from exc
    if device not in {"cuda:0", "cuda:1"} or torch.cuda.device_count() != 2:
        raise E07FullDevError("Full-dev generation requires T4 x2.")
    adapter = full_lora_directory / "training" / "adapter-final"
    torch.cuda.set_device(int(device[-1]))
    generator = config.section("generator")
    tokenizer = AutoTokenizer.from_pretrained(
        generator["model_id"], revision=generator["revision"], trust_remote_code=False,
    )
    base = Qwen3_5ForCausalLM.from_pretrained(
        generator["model_id"], revision=generator["revision"], dtype=torch.float16,
        device_map={"": device}, low_cpu_mem_usage=True, trust_remote_code=False,
    )
    model = PeftModel.from_pretrained(base, adapter, is_trainable=False)
    model.eval()
    adapter_parameters = sum(
        parameter.numel() for name, parameter in model.named_parameters() if "lora_" in name
    )
    if not 0 < adapter_parameters <= config.section("parameter_budget")["adapter_parameter_cap"]:
        raise E07FullDevError("Full-dev adapter parameter count violates the cap.")
    device_map = _validate_worker_placement(model, device)
    return model, tokenizer, device_map, adapter_parameters


def run_generation_worker(
    *, worker_rank: int, device: str, model: Any, tokenizer: Any,
    full_lora_directory: Path, e06_directory: Path, dev_path: Path,
    output_directory: Path, config: E07FullDevConfig,
    device_map: dict[str, Any], adapter_parameters: int,
) -> dict[str, Any]:
    workers = config.section("execution")["generation_workers"]
    if worker_rank not in range(workers) or device != f"cuda:{worker_rank}":
        raise E07FullDevError("Full-dev worker/device mapping changed.")
    dev, dev_cfg = _load_dev(dev_path), config.section("dev")
    ids = select_dev_sample(dev, seed=dev_cfg["sample_seed"], size=dev_cfg["sample_size"])
    retrieval_path = output_directory / "retrieval" / "results.jsonl"
    retrieval = _read_jsonl(retrieval_path)
    if [row.get("question_id") for row in retrieval] != ids:
        raise E07FullDevError("Full-dev retrieval results are incomplete or reordered.")
    e06_path = e06_directory / "generation" / "results.jsonl"
    full_path = full_lora_directory / "evaluation" / "results.jsonl"
    e06_rows, full_rows = _read_jsonl(e06_path), _read_jsonl(full_path)
    source_e06, advance = config.section("source_e06"), config.section("advancement_source")
    if (
        file_sha256(e06_path) != source_e06["generation_results_sha256"]
        or file_sha256(full_path) != advance["evaluation_results_sha256"]
    ):
        raise E07FullDevError("Reused dev-200 answers changed.")
    assigned = [index for index in range(len(ids)) if index % workers == worker_rank]
    identity = {
        "code_version": CODE_VERSION, "stage": "full-dev-paired-generation",
        "config_sha256": config.config_sha256,
        "retrieval_results_sha256": file_sha256(retrieval_path),
        "adapter_sha256": advance["adapter_sha256"],
        "adapter_parameters": adapter_parameters, "worker_rank": worker_rank,
        "device": device, "device_map": device_map,
        "assigned_indices_sha256": hashlib.sha256(
            ",".join(map(str, assigned)).encode("ascii")
        ).hexdigest(),
    }
    identity["identity_sha256"] = _json_sha256(identity)
    root, records = output_directory / "generation", output_directory / "generation" / "records"
    records.mkdir(parents=True, exist_ok=True)
    state_path = root / f"worker-{worker_rank}-state.json"
    completed = _load_worker_progress(
        records=records, state_path=state_path, identity=identity,
        assigned_indices=assigned, sample_ids=ids,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    inference = config.section("inference")
    packing = InferencePacking(
        max_input_tokens=inference["max_input_tokens"],
        minimum_contexts=inference["minimum_contexts"],
        system_prompt=inference["system_prompt"],
        answer_instruction=inference["answer_instruction"],
    )

    def token_count(messages: list[dict[str, Any]]) -> int:
        rendered = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        return len(tokenizer(rendered, add_special_tokens=False)["input_ids"])

    def generate(inputs: dict[str, Any]) -> tuple[str, float]:
        started = time.perf_counter()
        output = model.generate(
            **inputs, do_sample=False, num_beams=1,
            max_new_tokens=inference["max_new_tokens"], use_cache=True,
        )
        latency = (time.perf_counter() - started) * 1000
        answer = tokenizer.decode(
            output[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        ).strip()
        if not answer:
            raise E07FullDevError("Generator produced an empty full-dev answer.")
        return answer, latency

    for position in range(completed, len(assigned)):
        index, question_id = assigned[position], ids[assigned[position]]
        if index < dev_cfg["control_prefix_size"]:
            frozen = e06_rows[index]["variants"][source_e06["selected_variant"]]
            prior = full_rows[index]["variants"]
            if frozen["answer"] != prior["frozen_e06"]["answer"]:
                raise E07FullDevError("Frozen answer differs across saved dev-200 artifacts.")
            variants = {
                "frozen_e06": {**prior["frozen_e06"], "answer_source": "reused-e06-dev200"},
                "lora_full": {**prior["lora_e07"], "answer_source": "reused-fulltrain-dev200"},
            }
        else:
            selected, messages, input_tokens = pack_contexts(
                question=dev[question_id]["question"], contexts=retrieval[index]["contexts"],
                config=packing, token_counter=token_count,
            )
            rendered = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
            )
            tensors = tokenizer(rendered, add_special_tokens=False, return_tensors="pt")
            tensors = {key: value.to(device) for key, value in tensors.items()}
            with model.disable_adapter():
                frozen_answer, frozen_latency = generate(tensors)
            lora_answer, lora_latency = generate(tensors)
            common = {
                "selected_chunk_ids": [row["chunk_id"] for row in selected],
                "selected_context_count": len(selected), "input_tokens": input_tokens,
            }
            variants = {
                "frozen_e06": {
                    **common, "answer": frozen_answer, "answer_source": "generated-full-dev-frozen",
                    "output_tokens": len(tokenizer(frozen_answer, add_special_tokens=False)["input_ids"]),
                    "generation_latency_ms": frozen_latency,
                },
                "lora_full": {
                    **common, "answer": lora_answer, "answer_source": "generated-full-dev-lora",
                    "output_tokens": len(tokenizer(lora_answer, add_special_tokens=False)["input_ids"]),
                    "generation_latency_ms": lora_latency,
                },
            }
        _atomic_json(records / f"{index:04d}.json", {
            "question_id": question_id, "sample_index": index,
            "worker_rank": worker_rank, "worker_identity_sha256": identity["identity_sha256"],
            "variants": variants,
        })
        _write_worker_state(state_path, identity, position + 1, len(assigned))
        LOGGER.info(
            "e07_full_dev_generation_progress worker=%d completed=%d total=%d question_id=%s reused=%s",
            worker_rank, position + 1, len(assigned), question_id,
            index < dev_cfg["control_prefix_size"],
        )
    _write_worker_state(state_path, identity, len(assigned), len(assigned))
    return {"worker_rank": worker_rank, "completed": len(assigned)}


def finalize_full_dev(
    *, output_directory: Path, dev_path: Path, config: E07FullDevConfig,
) -> dict[str, Any]:
    ensure_nltk_resources(download=False)
    dev, dev_cfg = _load_dev(dev_path), config.section("dev")
    ids = select_dev_sample(dev, seed=dev_cfg["sample_seed"], size=dev_cfg["sample_size"])
    root, records = output_directory / "generation", output_directory / "generation" / "records"
    rows = []
    for index, question_id in enumerate(ids):
        path = records / f"{index:04d}.json"
        if not path.is_file():
            raise E07FullDevError(f"Missing full-dev generation record: {index}")
        row = json.loads(path.read_text(encoding="utf-8"))
        if row.get("question_id") != question_id or set(row.get("variants", {})) != {
            "frozen_e06", "lora_full"
        }:
            raise E07FullDevError(f"Invalid full-dev generation record: {index}")
        rows.append(row)
    for rank in range(config.section("execution")["generation_workers"]):
        state = json.loads((root / f"worker-{rank}-state.json").read_text(encoding="utf-8"))
        if state.get("complete") is not True:
            raise E07FullDevError(f"Full-dev generation worker incomplete: {rank}")
    _atomic_jsonl(root / "results.jsonl", rows)
    scored = {key: [] for key in ("frozen_e06", "lora_full")}
    score_rows = []
    for row in rows:
        reference = dev[row["question_id"]]["answer"]
        item = {"question_id": row["question_id"]}
        for key in scored:
            answer = row["variants"][key]["answer"]
            score = {
                "question_id": row["question_id"],
                "meteor": nltk_meteor_score(reference, answer),
                "rouge_l": rouge_l_fmeasure(reference, answer),
            }
            scored[key].append(score)
            item[key] = {"meteor": score["meteor"], "rouge_l": score["rouge_l"]}
        score_rows.append(item)
    _atomic_jsonl(root / "scores.jsonl", score_rows)
    metrics = {
        key: {
            "meteor": fmean(item["meteor"] for item in values),
            "rouge_l": fmean(item["rouge_l"] for item in values),
            "mean_output_tokens": fmean(row["variants"][key]["output_tokens"] for row in rows),
        }
        for key, values in scored.items()
    }
    advance = config.section("advancement_source")
    prefix = dev_cfg["control_prefix_size"]
    prefix_metrics = {
        key: {
            "meteor": fmean(item["meteor"] for item in scored[key][:prefix]),
            "rouge_l": fmean(item["rouge_l"] for item in scored[key][:prefix]),
        }
        for key in scored
    }
    if (
        prefix_metrics["frozen_e06"]["meteor"] != advance["frozen_meteor"]
        or prefix_metrics["frozen_e06"]["rouge_l"] != advance["frozen_rouge_l"]
        or prefix_metrics["lora_full"]["meteor"] != advance["lora_meteor"]
        or prefix_metrics["lora_full"]["rouge_l"] != advance["lora_rouge_l"]
    ):
        raise E07FullDevError("Re-scored dev-200 prefix changed during promotion.")
    meteor_delta = [
        candidate["meteor"] - baseline["meteor"]
        for candidate, baseline in zip(scored["lora_full"], scored["frozen_e06"])
    ]
    rouge_delta = [
        candidate["rouge_l"] - baseline["rouge_l"]
        for candidate, baseline in zip(scored["lora_full"], scored["frozen_e06"])
    ]
    scoring = config.section("scoring")
    meteor_ci = _bootstrap_ci(
        meteor_delta, seed=f"{scoring['bootstrap_seed']}:meteor",
        iterations=scoring["bootstrap_iterations"],
    )
    rouge_ci = _bootstrap_ci(
        rouge_delta, seed=f"{scoring['bootstrap_seed']}:rouge_l",
        iterations=scoring["bootstrap_iterations"],
    )
    promotion_allowed = fmean(meteor_delta) > 0 and meteor_ci[0] > 0
    report = {
        "schema_version": "1.0", "experiment_id": config.raw["experiment_id"],
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "sample_size": len(rows),
        "train_sample_size": 5636, "metrics": metrics,
        "dev200_prefix_metrics": prefix_metrics,
        "paired_delta_lora_minus_frozen": {
            "meteor_mean": fmean(meteor_delta), "meteor_bootstrap_95_ci": meteor_ci,
            "rouge_l_mean": fmean(rouge_delta), "rouge_l_bootstrap_95_ci": rouge_ci,
        },
        "promotion_allowed": promotion_allowed,
        "promotion_rule": scoring["promotion_rule"],
        "selected_variant": "lora_full" if promotion_allowed else "frozen_e06",
        "holdout_untouched": True,
        "evidence": {
            "config_sha256": config.config_sha256,
            "sample_ids_sha256": dev_cfg["sample_ids_sha256"],
            "retrieval_results_sha256": file_sha256(output_directory / "retrieval" / "results.jsonl"),
            "generation_results_sha256": file_sha256(root / "results.jsonl"),
            "scores_sha256": file_sha256(root / "scores.jsonl"),
            "adapter_sha256": advance["adapter_sha256"],
        },
    }
    _atomic_json(output_directory / "report.json", report)
    return report


def _validate_full_train_report(report: dict[str, Any], expected: dict[str, Any]) -> None:
    metrics, evidence = report.get("metrics", {}), report.get("evidence", {})
    paired = report.get("paired_delta_lora_minus_frozen", {})
    if (
        report.get("experiment_id") != expected["report_experiment_id"]
        or report.get("sample_size") != expected["sample_size"]
        or report.get("train_sample_size") != expected["train_sample_size"]
        or report.get("smoke_leader") != expected["smoke_leader"]
        or evidence.get("config_sha256") != expected["report_config_sha256"]
        or evidence.get("evaluation_results_sha256") != expected["evaluation_results_sha256"]
        or evidence.get("adapter_sha256") != expected["adapter_sha256"]
        or evidence.get("training_records_sha256") != expected["training_records_sha256"]
        or metrics.get("frozen_e06", {}).get("meteor") != expected["frozen_meteor"]
        or metrics.get("frozen_e06", {}).get("rouge_l") != expected["frozen_rouge_l"]
        or metrics.get("lora_e07", {}).get("meteor") != expected["lora_meteor"]
        or metrics.get("lora_e07", {}).get("rouge_l") != expected["lora_rouge_l"]
        or paired.get("meteor_bootstrap_95_ci", [None])[0] != expected["meteor_ci_lower"]
        or paired.get("rouge_l_bootstrap_95_ci", [None])[0] != expected["rouge_l_ci_lower"]
    ):
        raise E07FullDevError("Saved full-train dev-200 report changed.")


def _ids_sha(ids: list[str]) -> str:
    return hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()
