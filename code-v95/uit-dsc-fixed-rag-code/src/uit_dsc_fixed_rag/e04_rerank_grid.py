"""E04 reranker candidate-depth grid over the selected AITeam RRF stack."""

from __future__ import annotations

import hashlib
import json
import logging
import random
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean
from typing import Any

from uit_dsc_fixed_rag.corpus import file_sha256
from uit_dsc_fixed_rag.e02_answer import _atomic_json, _atomic_jsonl, pack_contexts
from uit_dsc_fixed_rag.e02_compare import _ChunkStore, _load_dev, weighted_rrf
from uit_dsc_fixed_rag.e03_rrf_grid import (
    _bootstrap_ci,
    _json_sha256,
    _load_worker_progress,
    _normalize_device,
    _read_jsonl,
    _validate_worker_placement,
    _write_jsonl_from_records,
    _write_state,
    _write_worker_state,
)
from uit_dsc_fixed_rag.evaluation.official import (
    ensure_nltk_resources,
    nltk_meteor_score,
    rouge_l_fmeasure,
)
from uit_dsc_fixed_rag.retrieval_diagnostics import select_dev_sample


CODE_VERSION = "0.1.0"
LOGGER = logging.getLogger(__name__)


class E04GridError(RuntimeError):
    """Raised when the E04 grid cannot safely continue."""


@dataclass(frozen=True)
class E04Variant:
    key: str
    reranker_input_top_k: int
    output_top_k: int
    answer_source: str


@dataclass(frozen=True)
class E04GridConfig:
    raw: dict[str, Any]
    path: Path
    selection: dict[str, Any]
    retrieval: dict[str, Any]
    e00_manifest_sha256: str
    bm25_bytes: int
    bm25_sha256: str
    variants: tuple[E04Variant, ...]
    reranker_model_id: str
    reranker_revision: str
    reranker_parameter_count: int
    reranker_runtime_unique_parameter_count: int
    reranker_max_tokens: int
    reranker_batch_size: int
    generator_model_id: str
    generator_revision: str
    generator_parameter_count: int
    generator_runtime_unique_parameter_count: int
    required_cuda_devices: int
    parameter_limit: int
    stack_total: int
    dev_path: str
    dev_sha256: str
    sample_seed: str
    sample_size: int
    sample_ids_sha256: str
    max_input_tokens: int
    minimum_contexts: int
    system_prompt: str
    answer_instruction: str
    max_new_tokens: int
    worker_count: int
    scorer_path: str
    scorer_sha256: str
    control_variant: str
    bootstrap_seed: str
    bootstrap_iterations: int

    @property
    def config_sha256(self) -> str:
        return file_sha256(self.path)


def load_e04_grid_config(path: Path) -> E04GridConfig:
    payload = json.loads(path.read_text(encoding="utf-8"))
    required_root = {
        "schema_version", "experiment_id", "selection_source", "retrieval_input",
        "source_e00", "variants", "reranker", "generator", "parameter_budget",
        "dev", "context", "decoding", "execution", "scoring", "run_contract",
    }
    if set(payload) != required_root or payload.get("schema_version") != "1.0":
        raise ValueError("E04 grid config root is incompatible.")
    if payload.get("experiment_id") != "E04-rerank-grid-aiteam-dev200-v2":
        raise ValueError("Unexpected E04 grid experiment ID.")
    selection = _object(payload, "selection_source")
    if selection != {
        "mode": "operator-selected-from-e03-smoke",
        "report_experiment_id": "E03-rrf-grid-aiteam-dev200-v1",
        "report_config_sha256": "485ee1c3b9d7e19bcea96db2f9ae01081e04165d80d84da43cab881c02b173a7",
        "generation_results_sha256": "fee068bffc9e7cc7a7ed1393fa07df06293545cfd07bb0aecd3835c73734d338",
        "sample_size": 200,
        "selected_embedding": "embedding_aiteam",
        "selected_variant": "sparse050_dense050",
        "sparse_weight": 0.5,
        "dense_weight": 0.5,
        "meteor": 0.3630209902348587,
        "rouge_l": 0.4216082800213014,
        "promotion_rule_satisfied": False,
    }:
        raise ValueError("E04 operator selection evidence changed.")
    retrieval = _object(payload, "retrieval_input")
    if retrieval != {
        "experiment_id": "E02-compare-dev200-retrieval-diagnostic-fast-v1",
        "semantic_config_sha256": "34196f0320b09142820f4ae859564ce5773627d048f1c736380c97079a02faac",
        "execution_config_sha256": "7d8faaec425cc1bfa2636be007c366b69cff7dae1423531ae4b961d0961ea8b8",
        "candidate": "embedding_aiteam",
        "candidate_k_per_branch": 40,
        "rrf_constant": 60,
        "fused_input_top_k": 40,
        "output_top_k": 20,
    }:
        raise ValueError("E04 retrieval lineage changed.")
    source = _object(payload, "source_e00")
    if source.get("artifact_version") != "e00-v2":
        raise ValueError("E04 requires E00 v2.")

    variants_payload = payload.get("variants")
    if not isinstance(variants_payload, list) or len(variants_payload) != 3:
        raise ValueError("E04 requires control, rerank-20 and rerank-40.")
    variants = tuple(
        E04Variant(
            key=_text(item, "key"),
            reranker_input_top_k=_nonnegative_int(item, "reranker_input_top_k"),
            output_top_k=_positive_int(item, "output_top_k"),
            answer_source=_text(item, "answer_source"),
        )
        for item in variants_payload
        if isinstance(item, dict)
    )
    if tuple(
        (row.key, row.reranker_input_top_k, row.output_top_k, row.answer_source)
        for row in variants
    ) != (
        ("control_no_rerank", 0, 20, "reuse-e03-sparse050-dense050"),
        ("rerank_top20", 20, 20, "generate-e04"),
        ("rerank_top40", 40, 20, "generate-e04"),
    ):
        raise ValueError("E04 reranker grid changed from the experiment plan.")

    reranker = _object(payload, "reranker")
    if (
        reranker.get("key") != "reranker_vietnamese"
        or reranker.get("model_id") != "AITeamVN/Vietnamese_Reranker"
        or reranker.get("score") != "sequence-classification-logit-descending"
    ):
        raise ValueError("E04 reranker contract changed.")
    generator = _object(payload, "generator")
    if (
        generator.get("key") != "generator_qwen35_2b"
        or generator.get("model_id") != "Qwen/Qwen3.5-2B"
        or generator.get("dtype") != "float16"
        or generator.get("device_mode") != "replicated-single-gpu-workers"
    ):
        raise ValueError("E04 generator contract changed.")
    budget = _object(payload, "parameter_budget")
    limit = _positive_int(budget, "exclusive_limit")
    stack_total = _positive_int(budget, "stack_total")
    observed_total = sum(
        _positive_int(budget, key) for key in ("embedding", "reranker", "generator")
    )
    if (
        stack_total != observed_total
        or stack_total >= limit
        or _positive_int(budget, "reranker") != _positive_int(reranker, "parameter_count")
        or _positive_int(budget, "generator") != _positive_int(generator, "parameter_count")
    ):
        raise ValueError("E04 stack violates the parameter budget.")
    dev = _object(payload, "dev")
    context = _object(payload, "context")
    decoding = _object(payload, "decoding")
    if context.get("packing") != "ranked-whole-chunks-greedy":
        raise ValueError("E04 context packing changed.")
    if decoding != {
        "do_sample": False, "enable_thinking": False, "max_new_tokens": 384,
        "num_beams": 1, "use_cache": True,
    }:
        raise ValueError("E04 deterministic decoding changed.")
    execution = _object(payload, "execution")
    if execution != {
        "worker_count": 2,
        "partition": "sample-index-mod-worker-count",
        "reranker_batch_size": 4,
        "prompts_per_generate_call": 1,
    }:
        raise ValueError("E04 dual-GPU execution contract changed.")
    scoring = _object(payload, "scoring")
    if (
        scoring.get("primary_metric") != "meteor"
        or scoring.get("secondary_metric") != "rouge_l"
        or scoring.get("control_variant") != "control_no_rerank"
    ):
        raise ValueError("E04 scoring contract changed.")
    contract = _object(payload, "run_contract")
    if contract != {
        "one_factor_reranker_only": True,
        "reuse_control_answer_exactly": True,
        "checkpoint_every_questions": 1,
        "atomic_checkpoint_write": True,
        "resume_fail_closed": True,
        "same_embedding_fusion_generator_prompt_decoding": True,
        "allow_holdout": False,
        "allow_public": False,
        "allow_external_data": False,
        "allow_synthetic_data": False,
        "allow_model_api": False,
        "promotion_allowed": False,
    }:
        raise ValueError("E04 run contract changed.")

    return E04GridConfig(
        raw=payload,
        path=path,
        selection=dict(selection),
        retrieval=dict(retrieval),
        e00_manifest_sha256=_sha256(source, "manifest_sha256"),
        bm25_bytes=_positive_int(source, "bm25_bytes"),
        bm25_sha256=_sha256(source, "bm25_sha256"),
        variants=variants,
        reranker_model_id=_text(reranker, "model_id"),
        reranker_revision=_revision(reranker, "revision"),
        reranker_parameter_count=_positive_int(reranker, "parameter_count"),
        reranker_runtime_unique_parameter_count=_positive_int(
            reranker, "runtime_unique_parameter_count"
        ),
        reranker_max_tokens=_positive_int(reranker, "trained_pair_max_tokens"),
        reranker_batch_size=_positive_int(reranker, "batch_size"),
        generator_model_id=_text(generator, "model_id"),
        generator_revision=_revision(generator, "revision"),
        generator_parameter_count=_positive_int(generator, "parameter_count"),
        generator_runtime_unique_parameter_count=_positive_int(
            generator, "runtime_unique_parameter_count"
        ),
        required_cuda_devices=_positive_int(generator, "required_cuda_devices"),
        parameter_limit=limit,
        stack_total=stack_total,
        dev_path=_text(dev, "path"),
        dev_sha256=_sha256(dev, "sha256"),
        sample_seed=_text(dev, "sample_seed"),
        sample_size=_positive_int(dev, "sample_size"),
        sample_ids_sha256=_sha256(dev, "sample_ids_sha256"),
        max_input_tokens=_positive_int(context, "max_input_tokens"),
        minimum_contexts=_positive_int(context, "minimum_contexts"),
        system_prompt=_text(context, "system_prompt"),
        answer_instruction=_text(context, "answer_instruction"),
        max_new_tokens=_positive_int(decoding, "max_new_tokens"),
        worker_count=_positive_int(execution, "worker_count"),
        scorer_path=_text(scoring, "official_scorer_path"),
        scorer_sha256=_sha256(scoring, "official_scorer_sha256"),
        control_variant=_text(scoring, "control_variant"),
        bootstrap_seed=_text(scoring, "bootstrap_seed"),
        bootstrap_iterations=_positive_int(scoring, "bootstrap_iterations"),
    )


def validate_preflight(
    *, project_root: Path, e00_directory: Path, retrieval_directory: Path,
    selection_directory: Path, dev_path: Path, config: E04GridConfig,
) -> dict[str, Any]:
    scorer = project_root / config.scorer_path
    if not scorer.is_file() or file_sha256(scorer) != config.scorer_sha256:
        raise E04GridError("Pinned official scorer is missing or changed.")
    if not dev_path.is_file() or file_sha256(dev_path) != config.dev_sha256:
        raise E04GridError("Pinned dev split is missing or changed.")
    dev = _load_dev(dev_path)
    sample_ids = select_dev_sample(dev, seed=config.sample_seed, size=config.sample_size)
    sample_sha = hashlib.sha256("\n".join(sample_ids).encode("utf-8")).hexdigest()
    if sample_sha != config.sample_ids_sha256:
        raise E04GridError("Deterministic E04 sample identity changed.")

    manifest_path = e00_directory / "manifest.json"
    bm25_path = e00_directory / "bm25.sqlite3"
    if (
        not manifest_path.is_file()
        or file_sha256(manifest_path) != config.e00_manifest_sha256
        or not bm25_path.is_file()
        or bm25_path.stat().st_size != config.bm25_bytes
        or file_sha256(bm25_path) != config.bm25_sha256
    ):
        raise E04GridError("Pinned E00 v2 BM25 artifact is missing or changed.")

    retrieval_results = retrieval_directory / "embedding_aiteam" / "results.jsonl"
    retrieval_state_path = retrieval_directory / "embedding_aiteam" / "state.json"
    retrieval_report_path = retrieval_directory / "report.json"
    if not all(path.is_file() for path in (
        retrieval_results, retrieval_state_path, retrieval_report_path
    )):
        raise E04GridError("Completed E02 AITeam retrieval artifact is missing.")
    retrieval_report = json.loads(retrieval_report_path.read_text(encoding="utf-8"))
    retrieval_state = json.loads(retrieval_state_path.read_text(encoding="utf-8"))
    retrieval_identity = retrieval_state.get("run_identity", {})
    if (
        retrieval_report.get("experiment_id") != config.retrieval["experiment_id"]
        or retrieval_report.get("semantic_config_sha256")
        != config.retrieval["semantic_config_sha256"]
        or retrieval_report.get("execution_config_sha256")
        != config.retrieval["execution_config_sha256"]
        or retrieval_report.get("sample_size") != config.sample_size
        or retrieval_state.get("complete") is not True
        or retrieval_state.get("completed_count") != config.sample_size
        or retrieval_identity.get("semantic_config_sha256")
        != config.retrieval["semantic_config_sha256"]
        or retrieval_identity.get("candidate", {}).get("key") != "embedding_aiteam"
    ):
        raise E04GridError("E02 retrieval identity differs from E04.")
    retrieval_rows = _read_jsonl(retrieval_results)
    _validate_retrieval_rows(retrieval_rows, sample_ids, config)

    selection_report_path = selection_directory / "report.json"
    selection_results_path = selection_directory / "generation" / "results.jsonl"
    if not selection_report_path.is_file() or not selection_results_path.is_file():
        raise E04GridError("Saved E03 selection output is incomplete.")
    selection_report = json.loads(selection_report_path.read_text(encoding="utf-8"))
    _validate_selection_report(selection_report, config)
    if file_sha256(selection_results_path) != config.selection["generation_results_sha256"]:
        raise E04GridError("Saved E03 generation results checksum changed.")
    selection_rows = _read_jsonl(selection_results_path)
    _validate_selection_rows(selection_rows, sample_ids, config)

    return {
        "config_sha256": config.config_sha256,
        "dev_sha256": config.dev_sha256,
        "sample_ids_sha256": sample_sha,
        "sample_size": len(sample_ids),
        "bm25_sha256": config.bm25_sha256,
        "retrieval_results_sha256": file_sha256(retrieval_results),
        "selection_report_sha256": file_sha256(selection_report_path),
        "selection_generation_results_sha256": file_sha256(selection_results_path),
        "selection_mode": config.selection["mode"],
        "selected_stack": {
            "embedding": "embedding_aiteam",
            "sparse_weight": 0.5,
            "dense_weight": 0.5,
        },
        "stack_parameter_total": config.stack_total,
        "parameter_limit_exclusive": config.parameter_limit,
    }


def prepare_fused_top40(
    *, e00_directory: Path, retrieval_directory: Path, dev_path: Path,
    output_directory: Path, config: E04GridConfig, preflight: dict[str, Any],
) -> dict[str, Any]:
    dev = _load_dev(dev_path)
    sample_ids = select_dev_sample(dev, seed=config.sample_seed, size=config.sample_size)
    source_path = retrieval_directory / "embedding_aiteam" / "results.jsonl"
    if file_sha256(source_path) != preflight.get("retrieval_results_sha256"):
        raise E04GridError("E02 retrieval results changed after E04 preflight.")
    source_rows = _read_jsonl(source_path)
    _validate_retrieval_rows(source_rows, sample_ids, config)
    identity = {
        "code_version": CODE_VERSION,
        "stage": "reuse-e02-rankings-fuse-aiteam-050-050-top40",
        "config_sha256": config.config_sha256,
        "retrieval_results_sha256": preflight["retrieval_results_sha256"],
        "bm25_sha256": config.bm25_sha256,
        "sample_ids_sha256": config.sample_ids_sha256,
    }
    identity["identity_sha256"] = _json_sha256(identity)
    root = output_directory / "prepared"
    records = root / "records"
    records.mkdir(parents=True, exist_ok=True)
    state_path = root / "state.json"
    completed = _load_contiguous(records, state_path, identity, sample_ids)
    store = _ChunkStore(e00_directory / "bm25.sqlite3")
    try:
        for index in range(completed, len(sample_ids)):
            source = source_rows[index]
            fused = weighted_rrf(
                {"sparse": source["sparse_top_ids"], "dense": source["dense_top_ids"]},
                weights={"sparse": 0.5, "dense": 0.5},
                constant=int(config.retrieval["rrf_constant"]),
                top_k=int(config.retrieval["fused_input_top_k"]),
            )
            contexts = store.fetch([row["chunk_id"] for row in fused])
            _atomic_json(records / f"{index:04d}.json", {
                "question_id": sample_ids[index],
                "sample_index": index,
                "candidate": "embedding_aiteam",
                "fused": fused,
                "contexts": contexts,
            })
            _write_state(state_path, identity, index + 1, complete=False)
            LOGGER.info(
                "e04_prepare_progress completed=%d total=%d question_id=%s",
                index + 1, len(sample_ids), sample_ids[index],
            )
    finally:
        store.close()
    _write_jsonl_from_records(root / "results.jsonl", records, len(sample_ids))
    _write_state(state_path, identity, len(sample_ids), complete=True)
    return {
        "sample_size": len(sample_ids),
        "fused_top_k": 40,
        "results_sha256": file_sha256(root / "results.jsonl"),
        "run_identity": identity,
    }


def load_reranker_on_device(
    config: E04GridConfig, device: str
) -> tuple[Any, Any, dict[str, Any]]:
    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as exc:  # pragma: no cover - Kaggle dependency
        raise E04GridError("Install PyTorch and Transformers for reranking.") from exc
    if device not in {"cuda:0", "cuda:1"} or torch.cuda.device_count() != 2:
        raise E04GridError("E04 reranker workers require GPU T4 x2.")
    torch.cuda.set_device(int(device[-1]))
    tokenizer = AutoTokenizer.from_pretrained(
        config.reranker_model_id,
        revision=config.reranker_revision,
        trust_remote_code=False,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        config.reranker_model_id,
        revision=config.reranker_revision,
        dtype=torch.float16,
        trust_remote_code=False,
    ).to(device)
    model.eval()
    observed = sum(parameter.numel() for parameter in model.parameters())
    if observed != config.reranker_runtime_unique_parameter_count:
        raise E04GridError(
            "Reranker runtime unique parameter count changed: "
            f"{observed} != {config.reranker_runtime_unique_parameter_count}"
        )
    try:
        device_map = _validate_worker_placement(model, device)
    except RuntimeError as exc:
        raise E04GridError(str(exc)) from exc
    LOGGER.info(
        "e04_reranker_loaded device=%s runtime_unique_parameters=%d "
        "published_tensor_parameters=%d",
        device, observed, config.reranker_parameter_count,
    )
    return model, tokenizer, device_map


def run_rerank_worker(
    *, worker_rank: int, device: str, model: Any, tokenizer: Any,
    dev_path: Path, output_directory: Path, config: E04GridConfig,
    device_map: dict[str, Any],
) -> dict[str, Any]:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise E04GridError("PyTorch is required for reranking.") from exc
    if worker_rank not in range(config.worker_count) or device != f"cuda:{worker_rank}":
        raise E04GridError("E04 reranker worker rank/device assignment changed.")
    prepared_root = output_directory / "prepared"
    state = json.loads((prepared_root / "state.json").read_text(encoding="utf-8"))
    if state.get("complete") is not True:
        raise E04GridError("Prepare fused top-40 before reranking.")
    prepared_rows = _read_jsonl(prepared_root / "results.jsonl")
    dev = _load_dev(dev_path)
    sample_ids = select_dev_sample(dev, seed=config.sample_seed, size=config.sample_size)
    if [row.get("question_id") for row in prepared_rows] != sample_ids:
        raise E04GridError("Prepared E04 question order changed.")
    assigned = [index for index in range(len(sample_ids)) if index % 2 == worker_rank]
    identity = {
        "code_version": CODE_VERSION,
        "stage": "dual-gpu-reranker-worker",
        "config_sha256": config.config_sha256,
        "prepared_results_sha256": file_sha256(prepared_root / "results.jsonl"),
        "sample_ids_sha256": config.sample_ids_sha256,
        "worker_rank": worker_rank,
        "device": device,
        "assigned_indices_sha256": hashlib.sha256(
            ",".join(str(index) for index in assigned).encode("ascii")
        ).hexdigest(),
        "model": {
            "model_id": config.reranker_model_id,
            "revision": config.reranker_revision,
            "parameter_count": config.reranker_parameter_count,
            "runtime_unique_parameter_count": config.reranker_runtime_unique_parameter_count,
            "observed_device_map": device_map,
        },
    }
    identity["identity_sha256"] = _json_sha256(identity)
    root = output_directory / "reranked"
    records = root / "records"
    records.mkdir(parents=True, exist_ok=True)
    state_path = root / f"worker-{worker_rank}-state.json"
    completed = _load_worker_progress(
        records=records, state_path=state_path, identity=identity,
        assigned_indices=assigned, sample_ids=sample_ids,
    )
    for position in range(completed, len(assigned)):
        sample_index = assigned[position]
        question_id = sample_ids[sample_index]
        contexts = prepared_rows[sample_index]["contexts"]
        pairs = [(dev[question_id]["question"], context["text"]) for context in contexts]
        scores: list[float] = []
        maximum_tokens = 0
        started = time.perf_counter()
        for offset in range(0, len(pairs), config.reranker_batch_size):
            batch = pairs[offset:offset + config.reranker_batch_size]
            inputs = tokenizer(batch, padding=True, truncation=False, return_tensors="pt")
            lengths = inputs["attention_mask"].sum(dim=1)
            maximum_tokens = max(maximum_tokens, int(lengths.max().item()))
            if maximum_tokens > config.reranker_max_tokens:
                raise E04GridError(
                    f"Reranker pair exceeds {config.reranker_max_tokens} tokens: "
                    f"question_id={question_id}, observed={maximum_tokens}"
                )
            inputs = {key: value.to(device) for key, value in inputs.items()}
            with torch.inference_mode():
                logits = model(**inputs, return_dict=True).logits.view(-1).float()
            scores.extend(float(value) for value in logits.detach().cpu().tolist())
        latency_ms = (time.perf_counter() - started) * 1000
        if len(scores) != 40:
            raise E04GridError("Reranker score count differs from fused top-40.")
        order20 = reranker_order(scores[:20], contexts[:20], top_k=20)
        order40 = reranker_order(scores, contexts, top_k=20)
        variants = {
            "control_no_rerank": {
                "contexts": contexts[:20],
                "reranker_input_top_k": 0,
                "reranker_trace": [],
            },
            "rerank_top20": {
                "contexts": [contexts[index] for index in order20],
                "reranker_input_top_k": 20,
                "reranker_trace": _reranker_trace(order20, scores, contexts),
            },
            "rerank_top40": {
                "contexts": [contexts[index] for index in order40],
                "reranker_input_top_k": 40,
                "reranker_trace": _reranker_trace(order40, scores, contexts),
            },
        }
        _atomic_json(records / f"{sample_index:04d}.json", {
            "question_id": question_id,
            "sample_index": sample_index,
            "worker_rank": worker_rank,
            "worker_identity_sha256": identity["identity_sha256"],
            "variants": variants,
            "reranker_latency_ms": latency_ms,
            "maximum_pair_tokens": maximum_tokens,
        })
        _write_worker_state(state_path, identity, position + 1, len(assigned))
        LOGGER.info(
            "e04_rerank_progress worker=%d device=%s worker_completed=%d "
            "worker_total=%d question_id=%s latency_ms=%.1f max_pair_tokens=%d",
            worker_rank, device, position + 1, len(assigned), question_id,
            latency_ms, maximum_tokens,
        )
    _write_worker_state(state_path, identity, len(assigned), len(assigned))
    return {"worker_rank": worker_rank, "device": device, "completed": len(assigned)}


def finalize_reranking(
    *, output_directory: Path, dev_path: Path, config: E04GridConfig
) -> dict[str, Any]:
    sample_ids = select_dev_sample(
        _load_dev(dev_path), seed=config.sample_seed, size=config.sample_size
    )
    root = output_directory / "reranked"
    records = root / "records"
    rows = _validate_complete_partitioned_records(
        root=root, records=records, sample_ids=sample_ids,
        variant_keys={variant.key for variant in config.variants}, config=config,
    )
    _atomic_jsonl(root / "results.jsonl", rows)
    summary = {
        "schema_version": "1.0",
        "sample_size": len(rows),
        "results_sha256": file_sha256(root / "results.jsonl"),
        "mean_reranker_latency_ms": fmean(row["reranker_latency_ms"] for row in rows),
        "maximum_pair_tokens": max(row["maximum_pair_tokens"] for row in rows),
    }
    _atomic_json(root / "summary.json", summary)
    return summary


def load_generator_on_device(
    config: E04GridConfig, device: str
) -> tuple[Any, Any, dict[str, Any]]:
    try:
        import torch
        from transformers import AutoProcessor
        try:
            from transformers import AutoModelForMultimodalLM as AutoGenerator
        except ImportError:  # pragma: no cover
            from transformers import AutoModelForImageTextToText as AutoGenerator
    except ImportError as exc:  # pragma: no cover
        raise E04GridError("Install Transformers and Accelerate for E04 generation.") from exc
    if device not in {"cuda:0", "cuda:1"} or torch.cuda.device_count() != 2:
        raise E04GridError("E04 generation workers require GPU T4 x2.")
    torch.cuda.set_device(int(device[-1]))
    random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    processor = AutoProcessor.from_pretrained(
        config.generator_model_id,
        revision=config.generator_revision,
        trust_remote_code=False,
    )
    tokenizer = processor.tokenizer
    model = AutoGenerator.from_pretrained(
        config.generator_model_id,
        revision=config.generator_revision,
        dtype=torch.float16,
        device_map={"": device},
        low_cpu_mem_usage=True,
        trust_remote_code=False,
    )
    model.eval()
    observed = sum(parameter.numel() for parameter in model.parameters())
    if observed != config.generator_runtime_unique_parameter_count:
        raise E04GridError(
            "Generator runtime unique parameter count changed: "
            f"{observed} != {config.generator_runtime_unique_parameter_count}"
        )
    try:
        device_map = _validate_worker_placement(model, device)
    except RuntimeError as exc:
        raise E04GridError(str(exc)) from exc
    LOGGER.info(
        "e04_generator_loaded device=%s runtime_unique_parameters=%d "
        "published_tensor_parameters=%d",
        device, observed, config.generator_parameter_count,
    )
    return model, tokenizer, device_map


def run_generation_worker(
    *, worker_rank: int, device: str, model: Any, tokenizer: Any,
    selection_directory: Path, dev_path: Path, output_directory: Path,
    config: E04GridConfig, device_map: dict[str, Any],
) -> dict[str, Any]:
    if worker_rank not in range(config.worker_count) or device != f"cuda:{worker_rank}":
        raise E04GridError("E04 generation worker rank/device assignment changed.")
    reranked_root = output_directory / "reranked"
    reranked_rows = _read_jsonl(reranked_root / "results.jsonl")
    selection_path = selection_directory / "generation" / "results.jsonl"
    if file_sha256(selection_path) != config.selection["generation_results_sha256"]:
        raise E04GridError("E03 control answers changed before E04 generation.")
    selection_rows = _read_jsonl(selection_path)
    dev = _load_dev(dev_path)
    sample_ids = select_dev_sample(dev, seed=config.sample_seed, size=config.sample_size)
    _validate_selection_rows(selection_rows, sample_ids, config)
    if [row.get("question_id") for row in reranked_rows] != sample_ids:
        raise E04GridError("Reranked E04 question order changed.")
    assigned = [index for index in range(len(sample_ids)) if index % 2 == worker_rank]
    identity = {
        "code_version": CODE_VERSION,
        "stage": "dual-gpu-generator-worker-two-reranked-variants",
        "config_sha256": config.config_sha256,
        "reranked_results_sha256": file_sha256(reranked_root / "results.jsonl"),
        "control_results_sha256": config.selection["generation_results_sha256"],
        "sample_ids_sha256": config.sample_ids_sha256,
        "worker_rank": worker_rank,
        "device": device,
        "assigned_indices_sha256": hashlib.sha256(
            ",".join(str(index) for index in assigned).encode("ascii")
        ).hexdigest(),
        "generator": {
            "model_id": config.generator_model_id,
            "revision": config.generator_revision,
            "parameter_count": config.generator_parameter_count,
            "runtime_unique_parameter_count": config.generator_runtime_unique_parameter_count,
            "observed_device_map": device_map,
        },
    }
    identity["identity_sha256"] = _json_sha256(identity)
    root = output_directory / "generation"
    records = root / "records"
    records.mkdir(parents=True, exist_ok=True)
    state_path = root / f"worker-{worker_rank}-state.json"
    completed = _load_worker_progress(
        records=records, state_path=state_path, identity=identity,
        assigned_indices=assigned, sample_ids=sample_ids,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    def token_count(messages: list[dict[str, Any]]) -> int:
        rendered = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        return len(tokenizer(rendered, add_special_tokens=False)["input_ids"])

    for position in range(completed, len(assigned)):
        sample_index = assigned[position]
        question_id = sample_ids[sample_index]
        source_control = selection_rows[sample_index]["variants"]["sparse050_dense050"]
        control_contexts = reranked_rows[sample_index]["variants"]["control_no_rerank"]["contexts"]
        selected_control, _, control_input_tokens = pack_contexts(
            question=dev[question_id]["question"], contexts=control_contexts,
            config=config, token_counter=token_count,
        )
        selected_control_ids = [context["chunk_id"] for context in selected_control]
        if (
            selected_control_ids != source_control.get("selected_chunk_ids")
            or control_input_tokens != source_control.get("input_tokens")
        ):
            raise E04GridError(f"E03 control context lineage changed: {question_id}")
        answers: dict[str, Any] = {
            "control_no_rerank": {
                **source_control,
                "answer_source": "reused-e03-sparse050-dense050",
            }
        }
        for variant_key in ("rerank_top20", "rerank_top40"):
            contexts = reranked_rows[sample_index]["variants"][variant_key]["contexts"]
            selected, messages, input_tokens = pack_contexts(
                question=dev[question_id]["question"], contexts=contexts,
                config=config, token_counter=token_count,
            )
            rendered = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
            inputs = tokenizer(rendered, add_special_tokens=False, return_tensors="pt")
            inputs = {key: value.to(device) for key, value in inputs.items()}
            started = time.perf_counter()
            generated = model.generate(
                **inputs, do_sample=False, num_beams=1,
                max_new_tokens=config.max_new_tokens, use_cache=True,
            )
            latency_ms = (time.perf_counter() - started) * 1000
            prompt_width = inputs["input_ids"].shape[1]
            answer = tokenizer.decode(
                generated[0, prompt_width:], skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            ).strip()
            if not answer:
                raise E04GridError(f"Generator returned an empty answer: {variant_key}/{question_id}")
            answers[variant_key] = {
                "answer": answer,
                "answer_source": "generated-e04",
                "selected_chunk_ids": [context["chunk_id"] for context in selected],
                "selected_context_count": len(selected),
                "input_tokens": input_tokens,
                "output_tokens": len(tokenizer(answer, add_special_tokens=False)["input_ids"]),
                "generation_latency_ms": latency_ms,
            }
        _atomic_json(records / f"{sample_index:04d}.json", {
            "question_id": question_id,
            "sample_index": sample_index,
            "worker_rank": worker_rank,
            "worker_identity_sha256": identity["identity_sha256"],
            "variants": answers,
        })
        _write_worker_state(state_path, identity, position + 1, len(assigned))
        LOGGER.info(
            "e04_generation_progress worker=%d device=%s worker_completed=%d "
            "worker_total=%d question_id=%s",
            worker_rank, device, position + 1, len(assigned), question_id,
        )
    _write_worker_state(state_path, identity, len(assigned), len(assigned))
    return {"worker_rank": worker_rank, "device": device, "completed": len(assigned)}


def finalize_generation(
    *, output_directory: Path, dev_path: Path, config: E04GridConfig
) -> dict[str, Any]:
    sample_ids = select_dev_sample(
        _load_dev(dev_path), seed=config.sample_seed, size=config.sample_size
    )
    root = output_directory / "generation"
    records = root / "records"
    rows = _validate_complete_partitioned_records(
        root=root, records=records, sample_ids=sample_ids,
        variant_keys={variant.key for variant in config.variants}, config=config,
    )
    _atomic_jsonl(root / "results.jsonl", rows)
    summary = {
        "schema_version": "1.0",
        "sample_size": len(rows),
        "answer_count": len(rows) * len(config.variants),
        "reused_control_answers": len(rows),
        "newly_generated_answers": len(rows) * 2,
        "results_sha256": file_sha256(root / "results.jsonl"),
    }
    _atomic_json(root / "summary.json", summary)
    return summary


def score_grid(
    *, output_directory: Path, dev_path: Path, config: E04GridConfig
) -> dict[str, Any]:
    ensure_nltk_resources(download=False)
    results_path = output_directory / "generation" / "results.jsonl"
    if not results_path.is_file():
        raise E04GridError("Finalize E04 generation before scoring.")
    rows = _read_jsonl(results_path)
    dev = _load_dev(dev_path)
    sample_ids = select_dev_sample(dev, seed=config.sample_seed, size=config.sample_size)
    if [row.get("question_id") for row in rows] != sample_ids:
        raise E04GridError("E04 generation order changed before scoring.")
    scored = {variant.key: [] for variant in config.variants}
    for row in rows:
        reference = dev[row["question_id"]]["answer"]
        for variant in config.variants:
            answer = row["variants"][variant.key]["answer"]
            scored[variant.key].append({
                "question_id": row["question_id"],
                "sample_index": row["sample_index"],
                "meteor": nltk_meteor_score(reference, answer),
                "rouge_l": rouge_l_fmeasure(reference, answer),
            })
    metrics = {}
    for variant in config.variants:
        key = variant.key
        _atomic_jsonl(output_directory / f"scores-{key}.jsonl", scored[key])
        _atomic_json(output_directory / f"predictions-{key}.json", {
            row["question_id"]: {"answer": source["variants"][key]["answer"]}
            for row, source in zip(scored[key], rows)
        })
        latencies = [
            row["variants"][key].get("generation_latency_ms") for row in rows
            if isinstance(row["variants"][key].get("generation_latency_ms"), (int, float))
        ]
        metrics[key] = {
            "reranker_input_top_k": variant.reranker_input_top_k,
            "meteor": fmean(row["meteor"] for row in scored[key]),
            "rouge_l": fmean(row["rouge_l"] for row in scored[key]),
            "mean_generation_latency_ms": fmean(latencies) if latencies else None,
            "answer_source": variant.answer_source,
        }
    if (
        metrics[config.control_variant]["meteor"] != config.selection["meteor"]
        or metrics[config.control_variant]["rouge_l"] != config.selection["rouge_l"]
    ):
        raise E04GridError("Re-scored reused E03 control metrics changed.")
    control = scored[config.control_variant]
    paired = {}
    for variant in config.variants:
        if variant.key == config.control_variant:
            continue
        meteor = [left["meteor"] - right["meteor"] for left, right in zip(scored[variant.key], control)]
        rouge = [left["rouge_l"] - right["rouge_l"] for left, right in zip(scored[variant.key], control)]
        paired[f"{variant.key}-minus-{config.control_variant}"] = {
            "meteor_mean": fmean(meteor),
            "meteor_bootstrap_95_ci": _bootstrap_ci(
                meteor, seed=f"{config.bootstrap_seed}:{variant.key}:meteor",
                iterations=config.bootstrap_iterations,
            ),
            "rouge_l_mean": fmean(rouge),
            "rouge_l_bootstrap_95_ci": _bootstrap_ci(
                rouge, seed=f"{config.bootstrap_seed}:{variant.key}:rouge_l",
                iterations=config.bootstrap_iterations,
            ),
        }
    smoke_leader = max(metrics, key=lambda key: (metrics[key]["meteor"], metrics[key]["rouge_l"]))
    rerank_summary = json.loads(
        (output_directory / "reranked" / "summary.json").read_text(encoding="utf-8")
    )
    report = {
        "schema_version": "1.0",
        "experiment_id": "E04-rerank-grid-aiteam-dev200-v2",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "sample_size": len(rows),
        "selected_embedding": "embedding_aiteam",
        "fusion_weights": {"sparse": 0.5, "dense": 0.5},
        "metrics": metrics,
        "control_variant": config.control_variant,
        "paired_deltas_vs_control": paired,
        "smoke_leader": smoke_leader,
        "reranker_runtime": rerank_summary,
        "promotion_allowed": False,
        "evidence": {
            "config_sha256": config.config_sha256,
            "dev_sha256": config.dev_sha256,
            "sample_ids_sha256": config.sample_ids_sha256,
            "generation_results_sha256": file_sha256(results_path),
            "reused_e03_generation_results_sha256": config.selection["generation_results_sha256"],
        },
        "warning": "Dev-200 is an E04 smoke grid; formal promotion requires full dev.",
    }
    _atomic_json(output_directory / "report.json", report)
    return report


def reranker_order(scores: list[float], contexts: list[dict[str, Any]], *, top_k: int) -> list[int]:
    if len(scores) != len(contexts) or not 0 < top_k <= len(contexts):
        raise ValueError("Invalid reranker ordering inputs.")
    return sorted(
        range(len(contexts)),
        key=lambda index: (-float(scores[index]), index, str(contexts[index]["chunk_id"])),
    )[:top_k]


def _reranker_trace(
    order: list[int], scores: list[float], contexts: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    return [
        {
            "chunk_id": contexts[index]["chunk_id"],
            "score": scores[index],
            "fused_rank": index + 1,
            "reranked_rank": rank + 1,
        }
        for rank, index in enumerate(order)
    ]


def _validate_complete_partitioned_records(
    *, root: Path, records: Path, sample_ids: list[str],
    variant_keys: set[str], config: E04GridConfig,
) -> list[dict[str, Any]]:
    rows = []
    for index, question_id in enumerate(sample_ids):
        path = records / f"{index:04d}.json"
        if not path.is_file():
            raise E04GridError(f"Partitioned record is missing: {index}")
        row = json.loads(path.read_text(encoding="utf-8"))
        if (
            row.get("sample_index") != index
            or row.get("question_id") != question_id
            or set(row.get("variants", {})) != variant_keys
        ):
            raise E04GridError(f"Partitioned record identity changed: {index}")
        rows.append(row)
    for rank in range(config.worker_count):
        state_path = root / f"worker-{rank}-state.json"
        if not state_path.is_file() or json.loads(
            state_path.read_text(encoding="utf-8")
        ).get("complete") is not True:
            raise E04GridError(f"Worker is incomplete: {rank}")
    return rows


def _validate_retrieval_rows(
    rows: list[dict[str, Any]], sample_ids: list[str], config: E04GridConfig
) -> None:
    depth = int(config.retrieval["candidate_k_per_branch"])
    if len(rows) != len(sample_ids):
        raise E04GridError("E02 retrieval count differs from E04.")
    for index, (question_id, row) in enumerate(zip(sample_ids, rows)):
        sparse = row.get("sparse_top_ids")
        dense = row.get("dense_top_ids")
        if (
            row.get("sample_index") != index
            or row.get("question_id") != question_id
            or row.get("candidate") != "embedding_aiteam"
            or not isinstance(sparse, list) or not isinstance(dense, list)
            or not 0 < len(sparse) <= depth or not 0 < len(dense) <= depth
            or len(set(sparse)) != len(sparse) or len(set(dense)) != len(dense)
        ):
            raise E04GridError(f"E02 retrieval row is incompatible: {index}")


def _validate_selection_report(report: dict[str, Any], config: E04GridConfig) -> None:
    selection = config.selection
    selected_metrics = report.get("metrics", {}).get(selection["selected_variant"], {})
    evidence = report.get("evidence", {})
    if (
        report.get("experiment_id") != selection["report_experiment_id"]
        or report.get("sample_size") != selection["sample_size"]
        or report.get("smoke_leader") != selection["selected_variant"]
        or report.get("promotion_allowed") is not False
        or report.get("selected_embedding") != selection["selected_embedding"]
        or evidence.get("config_sha256") != selection["report_config_sha256"]
        or evidence.get("generation_results_sha256") != selection["generation_results_sha256"]
        or selected_metrics.get("meteor") != selection["meteor"]
        or selected_metrics.get("rouge_l") != selection["rouge_l"]
    ):
        raise E04GridError("Saved E03 report differs from E04 selection evidence.")


def _validate_selection_rows(
    rows: list[dict[str, Any]], sample_ids: list[str], config: E04GridConfig
) -> None:
    key = config.selection["selected_variant"]
    if len(rows) != len(sample_ids):
        raise E04GridError("E03 control answer count differs from E04.")
    for index, (question_id, row) in enumerate(zip(sample_ids, rows)):
        answer = row.get("variants", {}).get(key, {}).get("answer")
        if (
            row.get("sample_index") != index
            or row.get("question_id") != question_id
            or not isinstance(answer, str) or not answer.strip()
        ):
            raise E04GridError(f"E03 control answer row is incompatible: {index}")


def _load_contiguous(
    records: Path, state_path: Path, identity: dict[str, Any], sample_ids: list[str]
) -> int:
    files = sorted(records.glob("*.json"))
    if not state_path.is_file():
        if files:
            raise E04GridError("Prepared records exist without state.")
        _write_state(state_path, identity, 0, complete=False)
        return 0
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if state.get("run_identity") != identity:
        raise E04GridError("Prepared checkpoint belongs to a different run.")
    for index, path in enumerate(files):
        row = json.loads(path.read_text(encoding="utf-8"))
        if (
            path.name != f"{index:04d}.json"
            or row.get("sample_index") != index
            or row.get("question_id") != sample_ids[index]
        ):
            raise E04GridError("Prepared records are non-contiguous or changed.")
    if int(state.get("completed_count", -1)) > len(files):
        raise E04GridError("Prepared state is ahead of durable records.")
    _write_state(state_path, identity, len(files), complete=len(files) == len(sample_ids))
    return len(files)


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


def _nonnegative_int(payload: dict[str, Any], key: str) -> int:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{key} must be a non-negative integer.")
    return value


def _sha256(payload: dict[str, Any], key: str) -> str:
    value = _text(payload, key).lower()
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{key} must be a SHA-256 digest.")
    return value


def _revision(payload: dict[str, Any], key: str) -> str:
    value = _text(payload, key).lower()
    if len(value) != 40 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{key} must be an immutable 40-character revision.")
    return value
