"""E08B context-aware LoRA training and new-adapter-only dev-521 evaluation."""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean
from typing import Any

from uit_dsc_fixed_rag.corpus import file_sha256
from uit_dsc_fixed_rag.e02_answer import _atomic_json, _atomic_jsonl, pack_contexts
from uit_dsc_fixed_rag.e02_compare import _load_dev
from uit_dsc_fixed_rag.e03_rrf_grid import (
    _bootstrap_ci,
    _json_sha256,
    _load_worker_progress,
    _read_jsonl,
    _validate_worker_placement,
    _write_worker_state,
)
from uit_dsc_fixed_rag.e07_lora import InferencePacking, select_train_sample
from uit_dsc_fixed_rag.evaluation.official import (
    ensure_nltk_resources,
    nltk_meteor_score,
    rouge_l_fmeasure,
)
from uit_dsc_fixed_rag.retrieval_diagnostics import select_dev_sample


CODE_VERSION = "0.28.0"
LOGGER = logging.getLogger(__name__)


class E08BError(RuntimeError):
    """Raised when E08B evidence, training or evaluation changes."""


@dataclass(frozen=True)
class E08BConfig:
    raw: dict[str, Any]
    path: Path

    @property
    def config_sha256(self) -> str:
        return file_sha256(self.path)

    def section(self, key: str) -> dict[str, Any]:
        value = self.raw.get(key)
        if not isinstance(value, dict):
            raise ValueError(f"E08B section must be an object: {key}")
        return value

    @property
    def generator_model_id(self) -> str:
        return str(self.section("generator")["model_id"])

    @property
    def generator_revision(self) -> str:
        return str(self.section("generator")["revision"])


def load_e08b_config(path: Path) -> E08BConfig:
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "schema_version",
        "experiment_id",
        "source_e08a",
        "train",
        "dev",
        "prompt",
        "generator",
        "lora",
        "optimization",
        "inference",
        "parameter_budget",
        "execution",
        "scoring",
        "run_contract",
    }
    if (
        set(payload) != required
        or payload.get("schema_version") != "1.0"
        or payload.get("experiment_id")
        != "E08B-context-aware-lora-train5636-dev521-v3"
    ):
        raise ValueError("E08B config root is incompatible.")
    config = E08BConfig(payload, path)
    source = config.section("source_e08a")
    if source != {
        "experiment_id": "E08A-context-retrieval-train5636-dev521-v1",
        "report_sha256": "37404e8c2acc689a86871bc9c8c6165a94a9d8b0942ee09b3351461d3b8ecaea",
        "config_sha256": "a13103c6a7ad7432792c658f7a1242aa8e1187ab5a251c7dde6101dc9ff644f7",
        "train_results_path": "retrieval/train5636/results.jsonl",
        "train_results_bytes": 123142591,
        "train_results_sha256": "8bb590020510bc512dbcf738165480cd164059826849a001d117c2374e8e0786",
        "train_sparse_results_sha256": "170fc816926cdd55b3e0bce5666febf18be19102ab8679a9269b31a0790ba9e6",
        "dev_results_path": "retrieval/dev521/results.jsonl",
        "dev_results_bytes": 11421025,
        "dev_results_sha256": "69dbd80c6f070764717edd386c0ac6a8cc6ec22e3f940ce0008fc9367699defe",
        "dev_sparse_results_sha256": "93ce7ee52f327f16340f43e2c3b877c6036073f3e75375eace4c8830f0c141ca",
        "candidate_k_per_branch": 40,
        "sparse_weight": 0.5,
        "dense_weight": 0.5,
        "fused_top_k": 20,
        "selected_contexts": 12,
        "reranker": None,
    }:
        raise ValueError("E08B E08A-source contract changed.")
    train = config.section("train")
    if (
        train.get("record_count") != 5636
        or train.get("sample_size") != 5636
        or train.get("supervision") != "official-answer-only"
        or train.get("input")
        != "question-plus-frozen-ranked-top12-context-candidates"
        or train.get("context_candidate_limit") != 12
        or train.get("minimum_contexts") != 1
        or train.get("maximum_sequence_tokens") != 3072
        or train.get("context_tail_truncation_fallback")
        != "first-ranked-context-maximal-character-prefix"
        or train.get("minimum_truncated_context_characters") != 1
        or train.get("answer_truncation_allowed") is not False
        or train.get("assistant_loss_only") is not True
    ):
        raise ValueError("E08B train contract changed.")
    dev = config.section("dev")
    if (
        dev.get("full_size") != 721
        or dev.get("excluded_prefix_size") != 200
        or dev.get("evaluation_size") != 521
        or dev.get("answer_usage") != "scoring-only"
        or dev.get("control_generation") != "skipped-by-operator-decision"
    ):
        raise ValueError("E08B dev-521 contract changed.")
    lora = config.section("lora")
    expected_targets = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "in_proj_qkv",
        "in_proj_z",
        "in_proj_b",
        "in_proj_a",
        "out_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]
    if (
        lora.get("initialization") != "fresh-from-pinned-base-not-e07-adapter"
        or lora.get("rank") != 8
        or lora.get("alpha") != 16
        or lora.get("dropout") != 0.05
        or lora.get("target_modules") != expected_targets
        or lora.get("quantization") != "nf4-double-quant"
        or lora.get("compute_dtype") != "float16"
    ):
        raise ValueError("E08B LoRA contract changed.")
    if config.section("optimization") != {
        "epochs": 1.0,
        "learning_rate": 0.0001,
        "per_device_batch_size": 1,
        "gradient_accumulation_steps": 4,
        "effective_global_batch_size": 8,
        "warmup_steps": 0.03,
        "lr_scheduler": "cosine",
        "weight_decay": 0.0,
        "gradient_checkpointing": True,
        "save_steps": 32,
        "seed": 20260830,
    }:
        raise ValueError("E08B optimization contract changed.")
    inference = config.section("inference")
    if (
        inference.get("max_input_tokens") != 8192
        or inference.get("max_new_tokens") != 384
        or inference.get("do_sample") is not False
        or inference.get("num_beams") != 1
        or inference.get("enable_thinking") is not False
    ):
        raise ValueError("E08B inference contract changed.")
    budget = config.section("parameter_budget")
    if (
        budget.get("maximum_stack_total")
        != budget.get("embedding")
        + budget.get("generator")
        + budget.get("adapter_parameter_cap")
        or budget["maximum_stack_total"] >= budget["exclusive_limit"]
        or budget["adapter_parameter_cap"] != lora["trainable_parameter_cap"]
    ):
        raise ValueError("E08B parameter budget failed.")
    execution = config.section("execution")
    if execution.get("training_workers") != 2 or execution.get(
        "evaluation_workers"
    ) != 2:
        raise ValueError("E08B dual-GPU execution changed.")
    contract = config.section("run_contract")
    if not contract or not all(value is True for value in contract.values()):
        raise ValueError("E08B run contract lost an invariant.")
    if config.section("scoring").get("promotion_allowed") is not False:
        raise ValueError("E08B new-adapter-only score cannot promote automatically.")
    return config


def resolve_evaluation_max_new_tokens(
    config: E08BConfig, override: int | None
) -> int:
    """Resolve an audited inference-only output cap without changing training."""
    value = (
        config.section("inference")["max_new_tokens"]
        if override is None
        else override
    )
    if value not in {384, 512, 640, 704}:
        raise E08BError(
            "E08B evaluation max_new_tokens must be one of 384, 512, 640 or 704."
        )
    return value


def validate_e08b_preflight(
    *,
    project_root: Path,
    e08a_directory: Path,
    train_path: Path,
    dev_path: Path,
    config: E08BConfig,
) -> dict[str, Any]:
    source = config.section("source_e08a")
    report_path = e08a_directory / "report.json"
    train_results = e08a_directory / source["train_results_path"]
    dev_results = e08a_directory / source["dev_results_path"]
    train_sparse = e08a_directory / "retrieval/train5636/sparse/results.jsonl"
    dev_sparse = e08a_directory / "retrieval/dev521/sparse/results.jsonl"
    required = (report_path, train_results, dev_results, train_sparse, dev_sparse)
    if not all(path.is_file() for path in required):
        raise E08BError("Saved E08A context artifact is incomplete.")
    if file_sha256(report_path) != source["report_sha256"]:
        raise E08BError("Saved E08A report bytes changed.")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    _validate_e08a_report(report, source)
    if (
        train_results.stat().st_size != source["train_results_bytes"]
        or file_sha256(train_results) != source["train_results_sha256"]
        or dev_results.stat().st_size != source["dev_results_bytes"]
        or file_sha256(dev_results) != source["dev_results_sha256"]
        or file_sha256(train_sparse) != source["train_sparse_results_sha256"]
        or file_sha256(dev_sparse) != source["dev_sparse_results_sha256"]
    ):
        raise E08BError("Saved E08A retrieval-result evidence changed.")
    train, train_ids, dev, dev_ids = _load_splits(train_path, dev_path, config)
    _validate_context_rows(train_results, train_ids, "train5636", source)
    _validate_context_rows(dev_results, dev_ids, "dev521", source)
    scorer_cfg = config.section("scoring")
    scorer_path = project_root / scorer_cfg["official_scorer_path"]
    if (
        not scorer_path.is_file()
        or file_sha256(scorer_path) != scorer_cfg["official_scorer_sha256"]
    ):
        raise E08BError("Pinned official scorer is missing or changed.")
    return {
        "config_sha256": config.config_sha256,
        "e08a_report_sha256": source["report_sha256"],
        "train_sha256": config.section("train")["sha256"],
        "train_size": len(train_ids),
        "train_ids_sha256": _ids_sha(train_ids),
        "train_contexts_sha256": source["train_results_sha256"],
        "dev_sha256": config.section("dev")["sha256"],
        "dev_size": len(dev_ids),
        "dev_ids_sha256": _ids_sha(dev_ids),
        "dev_contexts_sha256": source["dev_results_sha256"],
        "maximum_stack_parameters": config.section("parameter_budget")[
            "maximum_stack_total"
        ],
        "train_answer_count": len(train),
        "dev_answer_count": len(dev_ids),
    }


def prepare_context_train_records(
    *,
    e08a_directory: Path,
    train_path: Path,
    dev_path: Path,
    output_directory: Path,
    config: E08BConfig,
    preflight: dict[str, Any],
) -> dict[str, Any]:
    train, train_ids, _, _ = _load_splits(train_path, dev_path, config)
    source = config.section("source_e08a")
    context_path = e08a_directory / source["train_results_path"]
    if (
        file_sha256(context_path) != preflight.get("train_contexts_sha256")
        or file_sha256(train_path) != preflight.get("train_sha256")
    ):
        raise E08BError("E08B train inputs changed after preflight.")
    context_rows = _validate_context_rows(
        context_path, train_ids, "train5636", source
    )
    rows: list[dict[str, Any]] = []
    for index, (question_id, context_row) in enumerate(zip(train_ids, context_rows)):
        official = train[question_id]
        rows.append(
            {
                "sample_index": index,
                "question_id": question_id,
                "question": official["question"],
                "answer": official["answer"],
                "contexts": context_row["contexts"],
                "supervision": "official-answer-only",
                "answers_are_retrieval_labels": False,
            }
        )
    root = output_directory / "training-data"
    root.mkdir(parents=True, exist_ok=True)
    records_path = root / "records.jsonl"
    _atomic_jsonl(records_path, rows)
    summary = {
        "schema_version": "1.0",
        "record_count": len(rows),
        "records_sha256": file_sha256(records_path),
        "sample_ids_sha256": _ids_sha(train_ids),
        "source_contexts_sha256": source["train_results_sha256"],
        "context_candidates_per_record": 12,
        "answers_are_retrieval_labels": False,
        "contains_dev_holdout_or_public": False,
    }
    _atomic_json(root / "summary.json", summary)
    return summary


def inference_packing(config: E08BConfig) -> InferencePacking:
    inference, prompt = config.section("inference"), config.section("prompt")
    return InferencePacking(
        max_input_tokens=inference["max_input_tokens"],
        minimum_contexts=inference["minimum_contexts"],
        system_prompt=prompt["system_prompt"],
        answer_instruction=prompt["answer_instruction"],
    )


def load_context_lora_generator(
    *, config: E08BConfig, training_directory: Path, device: str
) -> tuple[Any, Any, dict[str, Any], int]:
    try:
        import torch
        from peft import PeftModel
        from transformers import AutoTokenizer, Qwen3_5ForCausalLM
    except ImportError as exc:  # pragma: no cover
        raise E08BError("Install Transformers and PEFT for E08B inference.") from exc
    if device not in {"cuda:0", "cuda:1"} or torch.cuda.device_count() != 2:
        raise E08BError("E08B evaluation requires GPU T4 x2.")
    final = training_directory / "adapter-final"
    complete_path = final / "complete.json"
    adapter_path = final / "adapter_model.safetensors"
    if not complete_path.is_file() or not adapter_path.is_file():
        raise E08BError("Completed E08B context-aware adapter is missing.")
    complete = json.loads(complete_path.read_text(encoding="utf-8"))
    if (
        complete.get("experiment_id") != config.raw["experiment_id"]
        or complete.get("config_sha256") != config.config_sha256
        or complete.get("source_contexts_sha256")
        != config.section("source_e08a")["train_results_sha256"]
        or complete.get("adapter_sha256") != file_sha256(adapter_path)
        or complete.get("fresh_from_base") is not True
        or complete.get("e07_adapter_loaded") is not False
    ):
        raise E08BError("Completed E08B adapter evidence changed.")
    torch.cuda.set_device(int(device[-1]))
    tokenizer = AutoTokenizer.from_pretrained(
        config.generator_model_id,
        revision=config.generator_revision,
        trust_remote_code=False,
    )
    base = Qwen3_5ForCausalLM.from_pretrained(
        config.generator_model_id,
        revision=config.generator_revision,
        dtype=torch.float16,
        device_map={"": device},
        low_cpu_mem_usage=True,
        trust_remote_code=False,
    )
    model = PeftModel.from_pretrained(base, final, is_trainable=False)
    model.eval()
    adapter_parameters = sum(
        parameter.numel()
        for name, parameter in model.named_parameters()
        if "lora_" in name
    )
    if not 0 < adapter_parameters <= config.section("lora")[
        "trainable_parameter_cap"
    ]:
        raise E08BError("Observed E08B adapter parameter count violates its cap.")
    device_map = _validate_worker_placement(model, device)
    LOGGER.info(
        "e08b_generator_loaded device=%s adapter_parameters=%d",
        device,
        adapter_parameters,
    )
    return model, tokenizer, device_map, adapter_parameters


def run_dev521_worker(
    *,
    worker_rank: int,
    device: str,
    model: Any,
    tokenizer: Any,
    training_directory: Path,
    e08a_directory: Path,
    train_path: Path,
    dev_path: Path,
    output_directory: Path,
    config: E08BConfig,
    device_map: dict[str, Any],
    adapter_parameters: int,
    evaluation_max_new_tokens: int,
) -> dict[str, Any]:
    evaluation_max_new_tokens = resolve_evaluation_max_new_tokens(
        config, evaluation_max_new_tokens
    )
    workers = config.section("execution")["evaluation_workers"]
    if worker_rank not in range(workers) or device != f"cuda:{worker_rank}":
        raise E08BError("E08B evaluation worker mapping changed.")
    _, _, dev, dev_ids = _load_splits(train_path, dev_path, config)
    source = config.section("source_e08a")
    context_path = e08a_directory / source["dev_results_path"]
    contexts = _validate_context_rows(context_path, dev_ids, "dev521", source)
    assigned = [index for index in range(len(dev_ids)) if index % workers == worker_rank]
    adapter_path = training_directory / "adapter-final" / "adapter_model.safetensors"
    adapter_hash = file_sha256(adapter_path)
    identity = {
        "code_version": CODE_VERSION,
        "stage": "e08b-context-aware-dev521-worker",
        "config_sha256": config.config_sha256,
        "adapter_sha256": adapter_hash,
        "adapter_parameters": adapter_parameters,
        "evaluation_max_new_tokens": evaluation_max_new_tokens,
        "dev_contexts_sha256": source["dev_results_sha256"],
        "worker_rank": worker_rank,
        "device": device,
        "device_map": device_map,
        "assigned_indices_sha256": hashlib.sha256(
            ",".join(map(str, assigned)).encode("ascii")
        ).hexdigest(),
    }
    identity["identity_sha256"] = _json_sha256(identity)
    root = output_directory / "evaluation"
    records = root / "records"
    records.mkdir(parents=True, exist_ok=True)
    state_path = root / f"worker-{worker_rank}-state.json"
    completed = _load_worker_progress(
        records=records,
        state_path=state_path,
        identity=identity,
        assigned_indices=assigned,
        sample_ids=dev_ids,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    def token_count(messages: list[dict[str, Any]]) -> int:
        rendered = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        return len(tokenizer(rendered, add_special_tokens=False)["input_ids"])

    for position in range(completed, len(assigned)):
        index = assigned[position]
        question_id = dev_ids[index]
        selected, messages, input_tokens = pack_contexts(
            question=dev[question_id]["question"],
            contexts=contexts[index]["contexts"],
            config=inference_packing(config),
            token_counter=token_count,
        )
        rendered = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = tokenizer(rendered, add_special_tokens=False, return_tensors="pt")
        inputs = {key: value.to(device) for key, value in inputs.items()}
        started = time.perf_counter()
        generated = model.generate(
            **inputs,
            do_sample=False,
            num_beams=1,
            max_new_tokens=evaluation_max_new_tokens,
            use_cache=True,
        )
        latency_ms = (time.perf_counter() - started) * 1000
        new_ids = generated[0, inputs["input_ids"].shape[1] :]
        answer = tokenizer.decode(
            new_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        ).strip()
        if not answer:
            raise E08BError(f"Empty E08B answer: {question_id}")
        eos = tokenizer.eos_token_id
        eos_ids = set(eos if isinstance(eos, list) else [eos])
        generated_count = int(new_ids.shape[0])
        ended_by_eos = bool(generated_count and int(new_ids[-1]) in eos_ids)
        finish_reason = (
            "eos"
            if ended_by_eos
            else "length"
            if generated_count >= evaluation_max_new_tokens
            else "other"
        )
        _atomic_json(
            records / f"{index:04d}.json",
            {
                "question_id": question_id,
                "sample_index": index,
                "worker_rank": worker_rank,
                "worker_identity_sha256": identity["identity_sha256"],
                "answer": answer,
                "answer_source": "generated-e08b-context-aware-lora",
                "max_new_tokens": evaluation_max_new_tokens,
                "selected_chunk_ids": [row["chunk_id"] for row in selected],
                "selected_context_count": len(selected),
                "input_tokens": input_tokens,
                "output_tokens": len(
                    tokenizer(answer, add_special_tokens=False)["input_ids"]
                ),
                "generated_tokens_including_special": generated_count,
                "finish_reason": finish_reason,
                "generation_latency_ms": latency_ms,
            },
        )
        _write_worker_state(state_path, identity, position + 1, len(assigned))
        LOGGER.info(
            "e08b_generation_progress worker=%d device=%s completed=%d total=%d "
            "global_completed_at_least=%d global_total=%d question_id=%s",
            worker_rank,
            device,
            position + 1,
            len(assigned),
            min(len(dev_ids), (position + 1) * workers),
            len(dev_ids),
            question_id,
        )
    _write_worker_state(state_path, identity, len(assigned), len(assigned))
    return {"worker_rank": worker_rank, "completed": len(assigned), "device": device}


def finalize_dev521_score(
    *,
    output_directory: Path,
    training_directory: Path,
    e08a_directory: Path,
    train_path: Path,
    dev_path: Path,
    config: E08BConfig,
    evaluation_max_new_tokens: int,
) -> dict[str, Any]:
    evaluation_max_new_tokens = resolve_evaluation_max_new_tokens(
        config, evaluation_max_new_tokens
    )
    ensure_nltk_resources(download=False)
    _, _, dev, dev_ids = _load_splits(train_path, dev_path, config)
    root = output_directory / "evaluation"
    records = root / "records"
    rows: list[dict[str, Any]] = []
    for index, question_id in enumerate(dev_ids):
        path = records / f"{index:04d}.json"
        if not path.is_file():
            raise E08BError(f"Missing E08B evaluation record: {index}")
        row = json.loads(path.read_text(encoding="utf-8"))
        if (
            row.get("question_id") != question_id
            or row.get("sample_index") != index
            or not isinstance(row.get("answer"), str)
            or not row["answer"].strip()
            or not 0 < row.get("selected_context_count", 0) <= 12
            or row.get("max_new_tokens") != evaluation_max_new_tokens
            or row.get("finish_reason") not in {"eos", "length", "other"}
        ):
            raise E08BError(f"Invalid E08B evaluation record: {index}")
        rows.append(row)
    for rank in range(config.section("execution")["evaluation_workers"]):
        state_path = root / f"worker-{rank}-state.json"
        if not state_path.is_file():
            raise E08BError(f"Missing E08B evaluation worker state: {rank}")
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("complete") is not True:
            raise E08BError(f"E08B evaluation worker incomplete: {rank}")
    results_path = root / "results.jsonl"
    _atomic_jsonl(results_path, rows)
    scores = []
    for row in rows:
        reference = dev[row["question_id"]]["answer"]
        scores.append(
            {
                "question_id": row["question_id"],
                "sample_index": row["sample_index"],
                "meteor": nltk_meteor_score(reference, row["answer"]),
                "rouge_l": rouge_l_fmeasure(reference, row["answer"]),
            }
        )
    scores_path = root / "scores.jsonl"
    _atomic_jsonl(scores_path, scores)
    meteor_values = [row["meteor"] for row in scores]
    rouge_values = [row["rouge_l"] for row in scores]
    scoring = config.section("scoring")
    report = {
        "schema_version": "1.0",
        "experiment_id": (
            f"E08B-context-aware-lora-dev521-max"
            f"{evaluation_max_new_tokens}-v1"
        ),
        "training_experiment_id": config.raw["experiment_id"],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "sample_size": len(rows),
        "train_sample_size": config.section("train")["sample_size"],
        "evaluation_mode": (
            "new-context-aware-adapter-only-on-dev521-with-output-cap-override"
        ),
        "evaluation_max_new_tokens": evaluation_max_new_tokens,
        "metrics": {
            "meteor": fmean(meteor_values),
            "meteor_bootstrap_95_ci": _bootstrap_ci(
                meteor_values,
                seed=f"{scoring['bootstrap_seed']}:meteor",
                iterations=scoring["bootstrap_iterations"],
            ),
            "rouge_l": fmean(rouge_values),
            "rouge_l_bootstrap_95_ci": _bootstrap_ci(
                rouge_values,
                seed=f"{scoring['bootstrap_seed']}:rouge_l",
                iterations=scoring["bootstrap_iterations"],
            ),
            "mean_output_tokens": fmean(row["output_tokens"] for row in rows),
            "mean_answer_characters": fmean(len(row["answer"]) for row in rows),
            "length_finish_rate": fmean(
                row["finish_reason"] == "length" for row in rows
            ),
            "mean_input_tokens": fmean(row["input_tokens"] for row in rows),
            "mean_selected_contexts": fmean(
                row["selected_context_count"] for row in rows
            ),
            "mean_generation_latency_ms": fmean(
                row["generation_latency_ms"] for row in rows
            ),
        },
        "promotion_allowed": False,
        "control_regenerated_on_dev521": False,
        "evidence": {
            "config_sha256": config.config_sha256,
            "e08a_report_sha256": config.section("source_e08a")["report_sha256"],
            "dev_contexts_sha256": config.section("source_e08a")[
                "dev_results_sha256"
            ],
            "adapter_sha256": file_sha256(
                training_directory / "adapter-final" / "adapter_model.safetensors"
            ),
            "training_records_sha256": _training_records_hash(training_directory),
            "evaluation_results_sha256": file_sha256(results_path),
            "scores_sha256": file_sha256(scores_path),
            "evaluation_max_new_tokens": evaluation_max_new_tokens,
        },
        "warning": (
            "The operator requested new-adapter-only scoring on untouched dev-521. "
            "The evaluation output cap was explicitly overridden without changing "
            "the saved training adapter. Without regenerating the E07 control on "
            "the same IDs, this absolute score cannot isolate the LoRA or output-cap "
            "effect, establish a paired improvement, or auto-promote the adapter."
        ),
        "holdout_untouched": True,
        "public_read": False,
    }
    _atomic_json(output_directory / "report.json", report)
    return report


def _load_splits(
    train_path: Path, dev_path: Path, config: E08BConfig
) -> tuple[dict[str, Any], list[str], dict[str, Any], list[str]]:
    train_cfg, dev_cfg = config.section("train"), config.section("dev")
    if not train_path.is_file() or file_sha256(train_path) != train_cfg["sha256"]:
        raise E08BError("Pinned official train split is missing or changed.")
    if not dev_path.is_file() or file_sha256(dev_path) != dev_cfg["sha256"]:
        raise E08BError("Pinned official dev split is missing or changed.")
    train = json.loads(train_path.read_text(encoding="utf-8"))
    dev = _load_dev(dev_path)
    if not isinstance(train, dict) or len(train) != train_cfg["record_count"]:
        raise E08BError("Official train record count changed.")
    train_ids = select_train_sample(
        train, seed=train_cfg["sample_seed"], size=train_cfg["sample_size"]
    )
    full_dev_ids = select_dev_sample(
        dev, seed=dev_cfg["sample_seed"], size=dev_cfg["full_size"]
    )
    dev_ids = full_dev_ids[dev_cfg["excluded_prefix_size"] :]
    if (
        _ids_sha(train_ids) != train_cfg["sample_ids_sha256"]
        or len(dev_ids) != dev_cfg["evaluation_size"]
        or _ids_sha(dev_ids) != dev_cfg["evaluation_ids_sha256"]
        or set(train_ids) & set(dev_ids)
    ):
        raise E08BError("E08B deterministic split identity or disjointness changed.")
    return train, train_ids, dev, dev_ids


def _validate_context_rows(
    path: Path,
    ids: list[str],
    target: str,
    source: dict[str, Any],
) -> list[dict[str, Any]]:
    rows = _read_jsonl(path)
    if len(rows) != len(ids):
        raise E08BError(f"E08A context result count changed: {target}")
    selected_count = source["selected_contexts"]
    fused_count = source["fused_top_k"]
    for index, (question_id, row) in enumerate(zip(ids, rows)):
        contexts, fused = row.get("contexts"), row.get("fused")
        if (
            row.get("target") != target
            or row.get("sample_index") != index
            or row.get("question_id") != question_id
            or row.get("answer_included") is not False
            or "answer" in row
            or not isinstance(contexts, list)
            or len(contexts) != selected_count
            or not isinstance(fused, list)
            or len(fused) != fused_count
        ):
            raise E08BError(f"Invalid E08A context row: {target}/{index}")
        context_ids = [context.get("chunk_id") for context in contexts]
        fused_ids = [candidate.get("chunk_id") for candidate in fused[:selected_count]]
        if (
            context_ids != fused_ids
            or len(set(context_ids)) != selected_count
            or any(
                not isinstance(context.get("text"), str) or not context["text"].strip()
                for context in contexts
            )
        ):
            raise E08BError(f"E08A ranked context identity changed: {target}/{index}")
    return rows


def _validate_e08a_report(report: dict[str, Any], source: dict[str, Any]) -> None:
    targets, files = report.get("targets", {}), report.get("files", {})
    train_target, dev_target = targets.get("train5636", {}), targets.get("dev521", {})
    train_file = files.get(source["train_results_path"], {})
    dev_file = files.get(source["dev_results_path"], {})
    retrieval = report.get("retrieval", {})
    if (
        report.get("experiment_id") != source["experiment_id"]
        or report.get("train_answers_used_by_retrieval") is not False
        or report.get("dev_answers_used_by_retrieval") is not False
        or report.get("holdout_untouched") is not True
        or report.get("public_read") is not False
        or report.get("evidence", {}).get("config_sha256")
        != source["config_sha256"]
        or train_target.get("sample_size") != 5636
        or dev_target.get("sample_size") != 521
        or train_target.get("minimum_contexts") != 12
        or train_target.get("maximum_contexts") != 12
        or dev_target.get("minimum_contexts") != 12
        or dev_target.get("maximum_contexts") != 12
        or train_target.get("answers_included") is not False
        or dev_target.get("answers_included") is not False
        or train_file.get("bytes") != source["train_results_bytes"]
        or train_file.get("sha256") != source["train_results_sha256"]
        or dev_file.get("bytes") != source["dev_results_bytes"]
        or dev_file.get("sha256") != source["dev_results_sha256"]
        or retrieval.get("candidate_k_per_branch") != source["candidate_k_per_branch"]
        or retrieval.get("sparse_weight") != source["sparse_weight"]
        or retrieval.get("dense_weight") != source["dense_weight"]
        or retrieval.get("fused_top_k") != source["fused_top_k"]
        or retrieval.get("selected_contexts") != source["selected_contexts"]
        or retrieval.get("reranker") is not None
    ):
        raise E08BError("Saved E08A report changed.")


def _training_records_hash(training_directory: Path) -> str:
    summary_path = training_directory.parent / "training-data" / "summary.json"
    if not summary_path.is_file():
        raise E08BError("E08B training-data summary is missing beside training output.")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    value = summary.get("records_sha256")
    if not isinstance(value, str):
        raise E08BError("E08B training records hash is missing.")
    return value


def _ids_sha(ids: list[str]) -> str:
    return hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()
