"""E03 weighted-RRF grid over the operator-selected AITeam E02 branch."""

from __future__ import annotations

import hashlib
import json
import logging
import os
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
from uit_dsc_fixed_rag.evaluation.official import (
    ensure_nltk_resources,
    nltk_meteor_score,
    rouge_l_fmeasure,
)
from uit_dsc_fixed_rag.retrieval_diagnostics import select_dev_sample


CODE_VERSION = "0.1.1"
RETRIEVAL_CODE_VERSION = "0.1.0"
LOGGER = logging.getLogger(__name__)


class E03Error(RuntimeError):
    """Raised when the E03 grid cannot continue without changing its identity."""


@dataclass(frozen=True)
class RrfVariant:
    key: str
    sparse_weight: float
    dense_weight: float


@dataclass(frozen=True)
class E03Config:
    raw: dict[str, Any]
    path: Path
    selection: dict[str, Any]
    retrieval: dict[str, Any]
    e00_manifest_sha256: str
    bm25_bytes: int
    bm25_sha256: str
    variants: tuple[RrfVariant, ...]
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


def load_e03_config(path: Path) -> E03Config:
    payload = json.loads(path.read_text(encoding="utf-8"))
    required_root = {
        "schema_version", "experiment_id", "selection_source", "retrieval_input",
        "source_e00", "grid", "generator", "parameter_budget", "dev", "context",
        "decoding", "execution", "scoring", "run_contract",
    }
    if set(payload) != required_root or payload.get("schema_version") != "1.0":
        raise ValueError("E03 config root is incompatible.")
    if payload.get("experiment_id") != "E03-rrf-grid-aiteam-dev200-v1":
        raise ValueError("Unexpected E03 experiment ID.")

    selection = _object(payload, "selection_source")
    expected_selection_keys = {
        "mode", "selected_embedding", "report_experiment_id", "report_config_sha256",
        "generation_results_sha256", "sample_size", "promotion_rule_satisfied",
        "observed_metrics",
    }
    if set(selection) != expected_selection_keys:
        raise ValueError("E03 selection evidence schema changed.")
    if (
        selection.get("mode") != "operator-selected-from-e02-smoke"
        or selection.get("selected_embedding") != "embedding_aiteam"
        or selection.get("promotion_rule_satisfied") is not False
    ):
        raise ValueError("E03 must preserve the recorded operator selection status.")
    _sha256(selection, "report_config_sha256")
    _sha256(selection, "generation_results_sha256")
    if _positive_int(selection, "sample_size") != 200:
        raise ValueError("E03 selection must come from the measured dev-200 run.")
    observed_metrics = _object(selection, "observed_metrics")
    if set(observed_metrics) != {
        "embedding_aiteam_meteor", "embedding_harrier_meteor", "meteor_delta",
        "meteor_ci_lower", "meteor_ci_upper",
    }:
        raise ValueError("E03 measured selection metrics changed.")
    for key in observed_metrics:
        _number(observed_metrics, key)

    retrieval = _object(payload, "retrieval_input")
    if retrieval != {
        "experiment_id": "E02-compare-dev200-retrieval-diagnostic-fast-v1",
        "semantic_config_sha256": "34196f0320b09142820f4ae859564ce5773627d048f1c736380c97079a02faac",
        "execution_config_sha256": "7d8faaec425cc1bfa2636be007c366b69cff7dae1423531ae4b961d0961ea8b8",
        "candidate": "embedding_aiteam",
        "candidate_k_per_branch": 40,
        "rrf_constant": 60,
        "fused_top_k": 20,
    }:
        raise ValueError("E03 retrieval lineage changed.")

    source = _object(payload, "source_e00")
    if source.get("artifact_version") != "e00-v2":
        raise ValueError("E03 requires E00 v2.")
    variants_payload = payload.get("grid")
    if not isinstance(variants_payload, list) or len(variants_payload) != 3:
        raise ValueError("E03 requires exactly three RRF variants.")
    variants = tuple(
        RrfVariant(
            key=_text(item, "key"),
            sparse_weight=_number(item, "sparse_weight"),
            dense_weight=_number(item, "dense_weight"),
        )
        for item in variants_payload
        if isinstance(item, dict)
    )
    expected_grid = (
        ("sparse060_dense040", 0.6, 0.4),
        ("sparse050_dense050", 0.5, 0.5),
        ("sparse040_dense060", 0.4, 0.6),
    )
    if tuple((row.key, row.sparse_weight, row.dense_weight) for row in variants) != expected_grid:
        raise ValueError("E03 RRF grid changed from the experiment plan.")

    generator = _object(payload, "generator")
    if (
        generator.get("key") != "generator_qwen35_2b"
        or generator.get("model_id") != "Qwen/Qwen3.5-2B"
        or generator.get("dtype") != "float16"
        or generator.get("device_mode") != "replicated-single-gpu-workers"
    ):
        raise ValueError("E03 generator contract changed.")
    budget = _object(payload, "parameter_budget")
    parameter_limit = _positive_int(budget, "exclusive_limit")
    stack_total = _positive_int(budget, "stack_total")
    generator_parameters = _positive_int(generator, "parameter_count")
    if (
        stack_total != _positive_int(budget, "embedding") + _positive_int(budget, "generator")
        or _positive_int(budget, "generator") != generator_parameters
        or stack_total >= parameter_limit
    ):
        raise ValueError("E03 model stack violates the parameter budget.")

    dev = _object(payload, "dev")
    context = _object(payload, "context")
    decoding = _object(payload, "decoding")
    if context.get("packing") != "fused-rank-whole-chunks-greedy":
        raise ValueError("E03 context packing changed.")
    if decoding != {
        "do_sample": False, "enable_thinking": False, "max_new_tokens": 384,
        "num_beams": 1, "use_cache": True,
    }:
        raise ValueError("E03 deterministic decoding changed.")
    execution = _object(payload, "execution")
    if execution != {
        "worker_count": 2,
        "partition": "sample-index-mod-worker-count",
        "prompts_per_generate_call": 1,
    }:
        raise ValueError("E03 dual-GPU execution contract changed.")
    scoring = _object(payload, "scoring")
    if (
        scoring.get("primary_metric") != "meteor"
        or scoring.get("secondary_metric") != "rouge_l"
        or scoring.get("control_variant") != "sparse050_dense050"
    ):
        raise ValueError("E03 scoring contract changed.")
    contract = _object(payload, "run_contract")
    if contract != {
        "one_factor_rrf_weights_only": True,
        "checkpoint_every_questions": 1,
        "atomic_checkpoint_write": True,
        "resume_fail_closed": True,
        "same_embedding_generator_prompt_decoding": True,
        "reranker_allowed": False,
        "allow_holdout": False,
        "allow_public": False,
        "allow_external_data": False,
        "allow_synthetic_data": False,
        "allow_model_api": False,
        "promotion_allowed": False,
    }:
        raise ValueError("E03 run contract changed.")

    return E03Config(
        raw=payload,
        path=path,
        selection=dict(selection),
        retrieval=dict(retrieval),
        e00_manifest_sha256=_sha256(source, "manifest_sha256"),
        bm25_bytes=_positive_int(source, "bm25_bytes"),
        bm25_sha256=_sha256(source, "bm25_sha256"),
        variants=variants,
        generator_model_id=_text(generator, "model_id"),
        generator_revision=_revision(generator, "revision"),
        generator_parameter_count=generator_parameters,
        generator_runtime_unique_parameter_count=_positive_int(
            generator, "runtime_unique_parameter_count"
        ),
        required_cuda_devices=_positive_int(generator, "required_cuda_devices"),
        parameter_limit=parameter_limit,
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
    *,
    project_root: Path,
    e00_directory: Path,
    retrieval_directory: Path,
    selection_directory: Path,
    dev_path: Path,
    config: E03Config,
) -> dict[str, Any]:
    scorer = project_root / config.scorer_path
    if not scorer.is_file() or file_sha256(scorer) != config.scorer_sha256:
        raise E03Error("Pinned official scorer is missing or changed.")
    if not dev_path.is_file() or file_sha256(dev_path) != config.dev_sha256:
        raise E03Error("Pinned dev split is missing or changed.")
    dev = _load_dev(dev_path)
    sample_ids = select_dev_sample(dev, seed=config.sample_seed, size=config.sample_size)
    sample_sha = hashlib.sha256("\n".join(sample_ids).encode("utf-8")).hexdigest()
    if sample_sha != config.sample_ids_sha256:
        raise E03Error("Deterministic E03 sample identity changed.")

    manifest_path = e00_directory / "manifest.json"
    bm25_path = e00_directory / "bm25.sqlite3"
    if (
        not manifest_path.is_file()
        or file_sha256(manifest_path) != config.e00_manifest_sha256
        or not bm25_path.is_file()
        or bm25_path.stat().st_size != config.bm25_bytes
        or file_sha256(bm25_path) != config.bm25_sha256
    ):
        raise E03Error("Pinned E00 v2 BM25 artifact is missing or changed.")

    retrieval_report_path = retrieval_directory / "report.json"
    retrieval_state_path = retrieval_directory / "embedding_aiteam" / "state.json"
    retrieval_results_path = retrieval_directory / "embedding_aiteam" / "results.jsonl"
    if not all(path.is_file() for path in (
        retrieval_report_path, retrieval_state_path, retrieval_results_path
    )):
        raise E03Error("Completed E02 AITeam retrieval artifact is missing.")
    retrieval_report = json.loads(retrieval_report_path.read_text(encoding="utf-8"))
    if (
        retrieval_report.get("experiment_id") != config.retrieval["experiment_id"]
        or retrieval_report.get("semantic_config_sha256")
        != config.retrieval["semantic_config_sha256"]
        or retrieval_report.get("execution_config_sha256")
        != config.retrieval["execution_config_sha256"]
        or retrieval_report.get("sample_size") != config.sample_size
    ):
        raise E03Error("E02 retrieval report identity differs from E03.")
    retrieval_state = json.loads(retrieval_state_path.read_text(encoding="utf-8"))
    retrieval_identity = retrieval_state.get("run_identity", {})
    if (
        retrieval_state.get("complete") is not True
        or retrieval_state.get("completed_count") != config.sample_size
        or retrieval_identity.get("semantic_config_sha256")
        != config.retrieval["semantic_config_sha256"]
        or retrieval_identity.get("candidate", {}).get("key") != "embedding_aiteam"
    ):
        raise E03Error("E02 AITeam retrieval checkpoint is incompatible.")
    retrieval_rows = _read_jsonl(retrieval_results_path)
    _validate_retrieval_rows(retrieval_rows, sample_ids, config)

    selection_report_path = selection_directory / "report.json"
    if not selection_report_path.is_file():
        raise E03Error("Saved E02 answer comparison report is missing.")
    selection_report = json.loads(selection_report_path.read_text(encoding="utf-8"))
    _validate_selection_report(selection_report, config)

    return {
        "config_sha256": config.config_sha256,
        "dev_sha256": config.dev_sha256,
        "sample_ids_sha256": sample_sha,
        "sample_size": len(sample_ids),
        "e00_manifest_sha256": config.e00_manifest_sha256,
        "bm25_sha256": config.bm25_sha256,
        "retrieval_results_sha256": file_sha256(retrieval_results_path),
        "selection_report_sha256": file_sha256(selection_report_path),
        "selection_mode": config.selection["mode"],
        "selected_embedding": "embedding_aiteam",
        "promotion_rule_satisfied": False,
        "stack_parameter_total": config.stack_total,
        "parameter_limit_exclusive": config.parameter_limit,
    }


def prepare_grid_retrieval(
    *,
    e00_directory: Path,
    retrieval_directory: Path,
    dev_path: Path,
    output_directory: Path,
    config: E03Config,
    preflight: dict[str, Any],
) -> dict[str, Any]:
    dev = _load_dev(dev_path)
    sample_ids = select_dev_sample(dev, seed=config.sample_seed, size=config.sample_size)
    source_path = retrieval_directory / "embedding_aiteam" / "results.jsonl"
    if file_sha256(source_path) != preflight.get("retrieval_results_sha256"):
        raise E03Error("E02 retrieval results changed after E03 preflight.")
    source_rows = _read_jsonl(source_path)
    _validate_retrieval_rows(source_rows, sample_ids, config)
    identity = {
        "code_version": RETRIEVAL_CODE_VERSION,
        "stage": "rrf-grid-reuse-e02-rankings",
        "config_sha256": config.config_sha256,
        "retrieval_results_sha256": preflight["retrieval_results_sha256"],
        "bm25_sha256": config.bm25_sha256,
        "sample_ids_sha256": config.sample_ids_sha256,
    }
    identity["identity_sha256"] = _json_sha256(identity)
    root = output_directory / "retrieval"
    records = root / "records"
    records.mkdir(parents=True, exist_ok=True)
    state_path = root / "state.json"
    completed = _load_contiguous(records, state_path, identity, sample_ids)
    store = _ChunkStore(e00_directory / "bm25.sqlite3")
    try:
        for index in range(completed, len(sample_ids)):
            source = source_rows[index]
            variants: dict[str, Any] = {}
            for variant in config.variants:
                fused = weighted_rrf(
                    {
                        "sparse": source["sparse_top_ids"],
                        "dense": source["dense_top_ids"],
                    },
                    weights={
                        "sparse": variant.sparse_weight,
                        "dense": variant.dense_weight,
                    },
                    constant=int(config.retrieval["rrf_constant"]),
                    top_k=int(config.retrieval["fused_top_k"]),
                )
                contexts = store.fetch([row["chunk_id"] for row in fused])
                variants[variant.key] = {"fused": fused, "contexts": contexts}
            row = {
                "question_id": sample_ids[index],
                "sample_index": index,
                "candidate": "embedding_aiteam",
                "variants": variants,
            }
            _atomic_json(records / f"{index:04d}.json", row)
            _write_state(state_path, identity, index + 1, complete=False)
            LOGGER.info(
                "e03_retrieval_progress completed=%d total=%d question_id=%s",
                index + 1, len(sample_ids), sample_ids[index],
            )
    finally:
        store.close()
    _write_jsonl_from_records(root / "results.jsonl", records, len(sample_ids))
    _write_state(state_path, identity, len(sample_ids), complete=True)
    return {
        "sample_size": len(sample_ids),
        "variant_count": len(config.variants),
        "results_sha256": file_sha256(root / "results.jsonl"),
        "run_identity": identity,
    }


def load_generator_on_device(config: E03Config, device: str) -> tuple[Any, Any, dict[str, Any]]:
    try:
        import torch
        from transformers import AutoProcessor
        try:
            from transformers import AutoModelForMultimodalLM as AutoGenerator
        except ImportError:  # pragma: no cover - Transformers compatibility
            from transformers import AutoModelForImageTextToText as AutoGenerator
    except ImportError as exc:  # pragma: no cover - Kaggle dependency
        raise E03Error("Install the reviewed Transformers and Accelerate dependencies.") from exc
    if device not in {"cuda:0", "cuda:1"} or torch.cuda.device_count() != 2:
        raise E03Error("E03 workers require Kaggle GPU T4 x2 and an explicit device.")
    device_index = int(device[-1])
    torch.cuda.set_device(device_index)
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
        raise E03Error(
            "Generator runtime unique parameter count changed: "
            f"{observed} != {config.generator_runtime_unique_parameter_count}"
        )
    device_map = _validate_worker_placement(model, device)
    LOGGER.info(
        "e03_generator_loaded device=%s runtime_unique_parameters=%d "
        "published_tensor_parameters=%d",
        device, observed, config.generator_parameter_count,
    )
    return model, tokenizer, device_map


def run_generation_worker(
    *,
    worker_rank: int,
    device: str,
    model: Any,
    tokenizer: Any,
    dev_path: Path,
    output_directory: Path,
    config: E03Config,
    device_map: dict[str, Any],
) -> dict[str, Any]:
    if worker_rank not in range(config.worker_count) or device != f"cuda:{worker_rank}":
        raise E03Error("E03 worker rank/device assignment changed.")
    retrieval_root = output_directory / "retrieval"
    retrieval_state = json.loads((retrieval_root / "state.json").read_text(encoding="utf-8"))
    if retrieval_state.get("complete") is not True:
        raise E03Error("Prepare the complete E03 retrieval grid before generation.")
    retrieval_rows = _read_jsonl(retrieval_root / "results.jsonl")
    dev = _load_dev(dev_path)
    sample_ids = select_dev_sample(dev, seed=config.sample_seed, size=config.sample_size)
    if [row.get("question_id") for row in retrieval_rows] != sample_ids:
        raise E03Error("E03 retrieval result order changed.")
    assigned = [index for index in range(len(sample_ids)) if index % config.worker_count == worker_rank]
    identity = {
        "code_version": CODE_VERSION,
        "stage": "dual-gpu-independent-question-worker",
        "config_sha256": config.config_sha256,
        "retrieval_results_sha256": file_sha256(retrieval_root / "results.jsonl"),
        "sample_ids_sha256": config.sample_ids_sha256,
        "worker_rank": worker_rank,
        "worker_count": config.worker_count,
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
    generation_root = output_directory / "generation"
    records = generation_root / "records"
    records.mkdir(parents=True, exist_ok=True)
    state_path = generation_root / f"worker-{worker_rank}-state.json"
    completed_positions = _load_worker_progress(
        records=records,
        state_path=state_path,
        identity=identity,
        assigned_indices=assigned,
        sample_ids=sample_ids,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    def token_count(messages: list[dict[str, Any]]) -> int:
        rendered = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        return len(tokenizer(rendered, add_special_tokens=False)["input_ids"])

    for position in range(completed_positions, len(assigned)):
        sample_index = assigned[position]
        question_id = sample_ids[sample_index]
        answers: dict[str, Any] = {}
        for variant in config.variants:
            source = retrieval_rows[sample_index]["variants"][variant.key]
            selected, messages, input_tokens = pack_contexts(
                question=dev[question_id]["question"],
                contexts=source["contexts"],
                config=config,
                token_counter=token_count,
            )
            rendered = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
            inputs = tokenizer(
                rendered, add_special_tokens=False, return_tensors="pt"
            )
            inputs = {key: value.to(device) for key, value in inputs.items()}
            started = time.perf_counter()
            generated = model.generate(
                **inputs,
                do_sample=False,
                num_beams=1,
                max_new_tokens=config.max_new_tokens,
                use_cache=True,
            )
            latency_ms = (time.perf_counter() - started) * 1000
            prompt_width = inputs["input_ids"].shape[1]
            answer = tokenizer.decode(
                generated[0, prompt_width:],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            ).strip()
            if not answer:
                raise E03Error(f"Generator returned an empty E03 answer: {variant.key}/{question_id}")
            answers[variant.key] = {
                "answer": answer,
                "selected_chunk_ids": [context["chunk_id"] for context in selected],
                "selected_context_count": len(selected),
                "input_tokens": input_tokens,
                "output_tokens": len(tokenizer(answer, add_special_tokens=False)["input_ids"]),
                "generation_latency_ms": latency_ms,
            }
        row = {
            "question_id": question_id,
            "sample_index": sample_index,
            "worker_rank": worker_rank,
            "worker_identity_sha256": identity["identity_sha256"],
            "variants": answers,
        }
        _atomic_json(records / f"{sample_index:04d}.json", row)
        _write_worker_state(state_path, identity, position + 1, len(assigned))
        LOGGER.info(
            "e03_generation_progress worker=%d device=%s worker_completed=%d "
            "worker_total=%d global_completed_at_least=%d total=%d question_id=%s",
            worker_rank, device, position + 1, len(assigned),
            (position + 1) * config.worker_count, len(sample_ids), question_id,
        )
    _write_worker_state(state_path, identity, len(assigned), len(assigned))
    return {
        "worker_rank": worker_rank,
        "device": device,
        "completed_questions": len(assigned),
        "run_identity": identity,
    }


def finalize_generation(
    *, output_directory: Path, dev_path: Path, config: E03Config
) -> dict[str, Any]:
    dev = _load_dev(dev_path)
    sample_ids = select_dev_sample(dev, seed=config.sample_seed, size=config.sample_size)
    generation_root = output_directory / "generation"
    records = generation_root / "records"
    rows = []
    for index, question_id in enumerate(sample_ids):
        path = records / f"{index:04d}.json"
        if not path.is_file():
            raise E03Error(f"Generation record is missing: {index}")
        row = json.loads(path.read_text(encoding="utf-8"))
        if (
            row.get("sample_index") != index
            or row.get("question_id") != question_id
            or set(row.get("variants", {})) != {variant.key for variant in config.variants}
        ):
            raise E03Error(f"Generation record identity changed: {index}")
        rows.append(row)
    for rank in range(config.worker_count):
        state_path = generation_root / f"worker-{rank}-state.json"
        if not state_path.is_file():
            raise E03Error(f"Generation worker state is missing: {rank}")
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("complete") is not True:
            raise E03Error(f"Generation worker is incomplete: {rank}")
    _atomic_jsonl(generation_root / "results.jsonl", rows)
    summary = {
        "schema_version": "1.0",
        "sample_size": len(sample_ids),
        "answer_count": len(sample_ids) * len(config.variants),
        "worker_count": config.worker_count,
        "results_sha256": file_sha256(generation_root / "results.jsonl"),
    }
    _atomic_json(generation_root / "summary.json", summary)
    return summary


def score_grid(
    *, output_directory: Path, dev_path: Path, config: E03Config
) -> dict[str, Any]:
    ensure_nltk_resources(download=False)
    summary_path = output_directory / "generation" / "summary.json"
    results_path = output_directory / "generation" / "results.jsonl"
    if not summary_path.is_file() or not results_path.is_file():
        raise E03Error("Finalize complete E03 generation before scoring.")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("sample_size") != config.sample_size:
        raise E03Error("E03 generation summary is incomplete.")
    rows = _read_jsonl(results_path)
    dev = _load_dev(dev_path)
    sample_ids = select_dev_sample(dev, seed=config.sample_seed, size=config.sample_size)
    if [row.get("question_id") for row in rows] != sample_ids:
        raise E03Error("E03 generation result order changed before scoring.")
    scored: dict[str, list[dict[str, Any]]] = {
        variant.key: [] for variant in config.variants
    }
    for row in rows:
        question_id = row["question_id"]
        reference = dev[question_id]["answer"]
        for variant in config.variants:
            answer = row["variants"][variant.key]["answer"]
            scored[variant.key].append({
                "question_id": question_id,
                "sample_index": row["sample_index"],
                "meteor": nltk_meteor_score(reference, answer),
                "rouge_l": rouge_l_fmeasure(reference, answer),
            })
    metrics = {}
    for variant in config.variants:
        key = variant.key
        _atomic_jsonl(output_directory / f"scores-{key}.jsonl", scored[key])
        predictions = {
            row["question_id"]: {"answer": source["variants"][key]["answer"]}
            for row, source in zip(scored[key], rows)
        }
        _atomic_json(output_directory / f"predictions-{key}.json", predictions)
        metrics[key] = {
            "sparse_weight": variant.sparse_weight,
            "dense_weight": variant.dense_weight,
            "meteor": fmean(row["meteor"] for row in scored[key]),
            "rouge_l": fmean(row["rouge_l"] for row in scored[key]),
            "mean_generation_latency_ms": fmean(
                row["variants"][key]["generation_latency_ms"] for row in rows
            ),
            "mean_selected_contexts": fmean(
                row["variants"][key]["selected_context_count"] for row in rows
            ),
        }
    control = scored[config.control_variant]
    paired = {}
    for variant in config.variants:
        if variant.key == config.control_variant:
            continue
        meteor_deltas = [
            candidate["meteor"] - baseline["meteor"]
            for candidate, baseline in zip(scored[variant.key], control)
        ]
        rouge_deltas = [
            candidate["rouge_l"] - baseline["rouge_l"]
            for candidate, baseline in zip(scored[variant.key], control)
        ]
        paired[f"{variant.key}-minus-{config.control_variant}"] = {
            "meteor_mean": fmean(meteor_deltas),
            "meteor_bootstrap_95_ci": _bootstrap_ci(
                meteor_deltas,
                seed=f"{config.bootstrap_seed}:{variant.key}:meteor",
                iterations=config.bootstrap_iterations,
            ),
            "rouge_l_mean": fmean(rouge_deltas),
            "rouge_l_bootstrap_95_ci": _bootstrap_ci(
                rouge_deltas,
                seed=f"{config.bootstrap_seed}:{variant.key}:rouge_l",
                iterations=config.bootstrap_iterations,
            ),
        }
    smoke_leader = max(metrics, key=lambda key: (metrics[key]["meteor"], metrics[key]["rouge_l"]))
    report = {
        "schema_version": "1.0",
        "experiment_id": "E03-rrf-grid-aiteam-dev200-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "sample_size": len(rows),
        "selected_embedding": "embedding_aiteam",
        "selection_mode": config.selection["mode"],
        "metrics": metrics,
        "control_variant": config.control_variant,
        "paired_deltas_vs_control": paired,
        "smoke_leader": smoke_leader,
        "promotion_allowed": False,
        "evidence": {
            "config_sha256": config.config_sha256,
            "dev_sha256": config.dev_sha256,
            "sample_ids_sha256": config.sample_ids_sha256,
            "generation_results_sha256": file_sha256(results_path),
        },
        "warning": (
            "Dev-200 is an E03 smoke grid. The leader is not a formal promotion; "
            "the predeclared promotion gate requires full dev."
        ),
    }
    _atomic_json(output_directory / "report.json", report)
    return report


def _validate_selection_report(report: dict[str, Any], config: E03Config) -> None:
    selection = config.selection
    metrics = report.get("metrics", {})
    delta = report.get("paired_delta_aiteam_minus_harrier", {})
    evidence = report.get("evidence", {})
    expected = selection["observed_metrics"]
    if (
        report.get("experiment_id") != selection["report_experiment_id"]
        or report.get("sample_size") != selection["sample_size"]
        or report.get("promotion_allowed") is not False
        or evidence.get("config_sha256") != selection["report_config_sha256"]
        or evidence.get("generation_results_sha256")
        != selection["generation_results_sha256"]
        or metrics.get("embedding_aiteam", {}).get("meteor")
        != expected["embedding_aiteam_meteor"]
        or metrics.get("embedding_harrier", {}).get("meteor")
        != expected["embedding_harrier_meteor"]
        or delta.get("meteor_mean") != expected["meteor_delta"]
        or delta.get("meteor_bootstrap_95_ci")
        != [expected["meteor_ci_lower"], expected["meteor_ci_upper"]]
    ):
        raise E03Error("Saved E02 answer report differs from the operator selection evidence.")


def _validate_retrieval_rows(
    rows: list[dict[str, Any]], sample_ids: list[str], config: E03Config
) -> None:
    if len(rows) != len(sample_ids):
        raise E03Error("E02 AITeam retrieval result count differs from E03.")
    depth = int(config.retrieval["candidate_k_per_branch"])
    for index, (question_id, row) in enumerate(zip(sample_ids, rows)):
        if (
            row.get("sample_index") != index
            or row.get("question_id") != question_id
            or row.get("candidate") != "embedding_aiteam"
            or not isinstance(row.get("sparse_top_ids"), list)
            or not isinstance(row.get("dense_top_ids"), list)
            or not 0 < len(row["sparse_top_ids"]) <= depth
            or not 0 < len(row["dense_top_ids"]) <= depth
            or len(set(row["sparse_top_ids"])) != len(row["sparse_top_ids"])
            or len(set(row["dense_top_ids"])) != len(row["dense_top_ids"])
        ):
            raise E03Error(f"E02 AITeam retrieval row is incompatible: {index}")


def _load_contiguous(
    records: Path,
    state_path: Path,
    identity: dict[str, Any],
    sample_ids: list[str],
) -> int:
    files = sorted(records.glob("*.json"))
    if not state_path.is_file():
        if files:
            raise E03Error("E03 records exist without checkpoint state.")
        _write_state(state_path, identity, 0, complete=False)
        return 0
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if state.get("run_identity") != identity:
        raise E03Error("E03 checkpoint belongs to a different run.")
    for index, path in enumerate(files):
        row = json.loads(path.read_text(encoding="utf-8"))
        if (
            path.name != f"{index:04d}.json"
            or row.get("sample_index") != index
            or row.get("question_id") != sample_ids[index]
        ):
            raise E03Error("E03 checkpoint records are non-contiguous or changed.")
    if int(state.get("completed_count", -1)) > len(files):
        raise E03Error("E03 checkpoint state is ahead of durable records.")
    _write_state(state_path, identity, len(files), complete=len(files) == len(sample_ids))
    return len(files)


def _load_worker_progress(
    *,
    records: Path,
    state_path: Path,
    identity: dict[str, Any],
    assigned_indices: list[int],
    sample_ids: list[str],
) -> int:
    if not state_path.is_file():
        existing = [index for index in assigned_indices if (records / f"{index:04d}.json").is_file()]
        if existing:
            raise E03Error("Worker records exist without worker checkpoint state.")
        _write_worker_state(state_path, identity, 0, len(assigned_indices))
        return 0
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if state.get("run_identity") != identity:
        raise E03Error("Generation worker checkpoint belongs to a different run.")
    completed = 0
    gap_seen = False
    for index in assigned_indices:
        path = records / f"{index:04d}.json"
        if not path.is_file():
            gap_seen = True
            continue
        if gap_seen:
            raise E03Error("Generation worker records are non-contiguous within its partition.")
        row = json.loads(path.read_text(encoding="utf-8"))
        if (
            row.get("sample_index") != index
            or row.get("question_id") != sample_ids[index]
            or row.get("worker_identity_sha256") != identity["identity_sha256"]
        ):
            raise E03Error("Generation worker record identity changed.")
        completed += 1
    if int(state.get("completed_count", -1)) > completed:
        raise E03Error("Generation worker state is ahead of durable records.")
    _write_worker_state(state_path, identity, completed, len(assigned_indices))
    return completed


def _write_worker_state(
    path: Path, identity: dict[str, Any], completed: int, total: int
) -> None:
    _atomic_json(path, {
        "schema_version": "1.0",
        "run_identity": identity,
        "completed_count": completed,
        "assigned_count": total,
        "complete": completed == total,
    })


def _write_state(
    path: Path, identity: dict[str, Any], completed: int, *, complete: bool
) -> None:
    _atomic_json(path, {
        "schema_version": "1.0",
        "run_identity": identity,
        "completed_count": completed,
        "complete": complete,
    })


def _write_jsonl_from_records(path: Path, records: Path, count: int) -> None:
    _atomic_jsonl(
        path,
        [json.loads((records / f"{index:04d}.json").read_text(encoding="utf-8"))
         for index in range(count)],
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        # JSON may legally contain raw Unicode line-separator characters such as
        # U+2028 when ensure_ascii=False is used. str.splitlines() treats those
        # characters as record boundaries even though JSONL is delimited only by
        # the physical LF bytes written by _atomic_jsonl.
        return [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").split("\n")
            if line
        ]
    except (OSError, json.JSONDecodeError) as exc:
        raise E03Error(f"Invalid JSONL artifact: {path}") from exc


def _bootstrap_ci(values: list[float], *, seed: str, iterations: int) -> list[float]:
    if not values or iterations <= 0:
        raise ValueError("Bootstrap inputs must be non-empty and positive.")
    generator = random.Random(hashlib.sha256(seed.encode("utf-8")).digest())
    count = len(values)
    samples = sorted(
        fmean(values[generator.randrange(count)] for _ in range(count))
        for _ in range(iterations)
    )
    return [
        samples[int(0.025 * (iterations - 1))],
        samples[int(0.975 * (iterations - 1))],
    ]


def _normalize_device(value: Any) -> str:
    if isinstance(value, int):
        return f"cuda:{value}"
    text = str(value)
    if text in {"0", "1"}:
        return f"cuda:{text}"
    return text


def _validate_worker_placement(model: Any, expected_device: str) -> dict[str, Any]:
    """Verify actual parameter placement even when Transformers omits hf_device_map."""

    parameter_devices = {
        _normalize_device(getattr(parameter, "device", "unknown"))
        for parameter in model.parameters()
    }
    if parameter_devices != {expected_device}:
        raise E03Error(
            f"E03 generator parameters escaped {expected_device}: {sorted(parameter_devices)}"
        )
    reported = getattr(model, "hf_device_map", None)
    if isinstance(reported, dict) and reported:
        reported_devices = {_normalize_device(value) for value in reported.values()}
        if reported_devices != {expected_device}:
            raise E03Error(
                f"E03 reported device map escaped {expected_device}: {reported}"
            )
        return dict(reported)
    return {"": expected_device}


def _json_sha256(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        .encode("utf-8")
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


def _number(payload: dict[str, Any], key: str) -> float:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{key} must be numeric.")
    return float(value)


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
