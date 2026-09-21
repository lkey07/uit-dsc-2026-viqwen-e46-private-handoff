"""E07 answer-supervised LoRA training and paired frozen-vs-adapter evaluation."""

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
    _bootstrap_ci, _json_sha256, _load_worker_progress, _read_jsonl,
    _validate_worker_placement, _write_worker_state,
)
from uit_dsc_fixed_rag.evaluation.official import (
    ensure_nltk_resources, nltk_meteor_score, rouge_l_fmeasure,
)
from uit_dsc_fixed_rag.retrieval_diagnostics import select_dev_sample


CODE_VERSION = "0.13.0"
LOGGER = logging.getLogger(__name__)


class E07Error(RuntimeError):
    """Raised when E07 evidence, training or evaluation is incompatible."""


@dataclass(frozen=True)
class E07Config:
    raw: dict[str, Any]
    path: Path

    @property
    def config_sha256(self) -> str:
        return file_sha256(self.path)

    def section(self, key: str) -> dict[str, Any]:
        value = self.raw.get(key)
        if not isinstance(value, dict):
            raise ValueError(f"E07 section must be an object: {key}")
        return value

    @property
    def generator_model_id(self) -> str:
        return str(self.section("generator")["model_id"])

    @property
    def generator_revision(self) -> str:
        return str(self.section("generator")["revision"])


@dataclass(frozen=True)
class InferencePacking:
    max_input_tokens: int
    minimum_contexts: int
    system_prompt: str
    answer_instruction: str


def load_e07_config(path: Path) -> E07Config:
    payload = json.loads(path.read_text(encoding="utf-8"))
    experiment_id = payload.get("experiment_id")
    supported = {
        "E07-lora-aiteam-train1024-dev200-v1",
        "E07-lora-aiteam-fulltrain-dev200-v2",
    }
    if experiment_id not in supported:
        raise ValueError("Unexpected E07 experiment ID.")
    required = {
        "schema_version", "experiment_id", "selection_source", "train", "dev",
        "training_prompt", "generator", "lora", "optimization", "inference",
        "parameter_budget", "execution", "scoring", "run_contract",
    }
    if experiment_id == "E07-lora-aiteam-fulltrain-dev200-v2":
        required.add("advancement_source")
    if set(payload) != required or payload.get("schema_version") != "1.0":
        raise ValueError("E07 config root is incompatible.")
    config = E07Config(raw=payload, path=path)
    selection = config.section("selection_source")
    if selection != {
        "mode": "operator-selected-e06-control-smoke",
        "report_experiment_id": "E06-prompt-length-grid-aiteam-dev200-v1",
        "report_config_sha256": "3f397b5cc543fb240d64a4d60a4d3c79fe2f4863bbdcb03bc6d85b2962a3b7c9",
        "generation_results_sha256": "dda4a3f3a215e65c85bc1e96939b5947357a46d00d07987ea30c58c3ff1e83c8",
        "prepared_results_sha256": "56ed911cba621d9d30d582dea9388efca14e16f9f3ab6a5bc23d250b0bb6017c",
        "sample_size": 200, "selected_variant": "control_e05_ranked_top12",
        "meteor": 0.37659844658685926, "rouge_l": 0.4321430184618054,
        "promotion_rule_satisfied": False,
    }:
        raise ValueError("E07 E06-selection evidence changed.")
    train = config.section("train")
    expected_sample_size = (
        5636 if experiment_id == "E07-lora-aiteam-fulltrain-dev200-v2" else 1024
    )
    if (
        train.get("record_count") != 5636
        or train.get("sample_size") != expected_sample_size
        or train.get("supervision") != "official-answer-only"
        or train.get("input") != "question-only-no-retrieval-labels"
        or train.get("assistant_loss_only") is not True
        or train.get("maximum_sequence_tokens") != 2048
    ):
        raise ValueError("E07 training-data contract changed.")
    if experiment_id == "E07-lora-aiteam-fulltrain-dev200-v2":
        advancement = config.section("advancement_source")
        if (
            advancement.get("mode") != "full-train-authorized-after-e07-smoke"
            or advancement.get("report_experiment_id")
            != "E07-lora-aiteam-train1024-dev200-v1"
            or advancement.get("sample_size") != 200
            or advancement.get("train_sample_size") != 1024
            or advancement.get("smoke_leader") != "lora_e07"
            or advancement.get("meteor_delta_ci_lower", 0) <= 0
            or advancement.get("rouge_l_delta_ci_lower", 0) <= 0
            or advancement.get("promotion_allowed") is not False
        ):
            raise ValueError("E07 full-train advancement evidence changed.")
    if config.section("training_prompt") != {
        "system_prompt": "Bạn là trợ lý pháp luật Việt Nam.",
        "answer_instruction": (
            "Hãy trả lời trực tiếp câu hỏi bằng tiếng Việt, nêu căn cứ pháp lý "
            "và các điều kiện hoặc mốc thời gian quan trọng nếu có."
        ),
    }:
        raise ValueError("E07 training prompt changed.")
    lora = config.section("lora")
    expected_targets = [
        "q_proj", "k_proj", "v_proj", "o_proj", "in_proj_qkv", "in_proj_z",
        "in_proj_b", "in_proj_a", "out_proj", "gate_proj", "up_proj", "down_proj",
    ]
    if (
        lora.get("rank") != 8 or lora.get("alpha") != 16
        or lora.get("dropout") != 0.05 or lora.get("target_modules") != expected_targets
        or lora.get("quantization") != "nf4-double-quant"
        or lora.get("compute_dtype") != "float16"
    ):
        raise ValueError("E07 LoRA contract changed.")
    optimization = config.section("optimization")
    if optimization != {
        "epochs": 1.0, "learning_rate": 0.0001, "per_device_batch_size": 1,
        "gradient_accumulation_steps": 4, "effective_global_batch_size": 8,
        "warmup_steps": 0.03, "lr_scheduler": "cosine", "weight_decay": 0.0,
        "gradient_checkpointing": True, "save_steps": 32, "seed": 20260828,
    }:
        raise ValueError("E07 optimization contract changed.")
    inference = config.section("inference")
    if (
        inference.get("max_input_tokens") != 8192
        or inference.get("max_new_tokens") != 384
        or inference.get("do_sample") is not False
        or inference.get("num_beams") != 1
        or inference.get("enable_thinking") is not False
    ):
        raise ValueError("E07 frozen inference contract changed.")
    budget = config.section("parameter_budget")
    if (
        budget.get("maximum_stack_total")
        != budget.get("embedding") + budget.get("generator") + budget.get("adapter_parameter_cap")
        or budget["maximum_stack_total"] >= budget["exclusive_limit"]
        or budget["adapter_parameter_cap"] != lora["trainable_parameter_cap"]
    ):
        raise ValueError("E07 parameter budget changed or failed.")
    contract = config.section("run_contract")
    if not all(value is True for key, value in contract.items() if key != "promotion_allowed"):
        raise ValueError("E07 run contract lost a required invariant.")
    if contract.get("promotion_allowed") is not False:
        raise ValueError("E07 smoke promotion must remain disabled.")
    return config


def select_train_sample(train: dict[str, Any], *, seed: str, size: int) -> list[str]:
    if not 0 < size <= len(train):
        raise ValueError("Invalid E07 train sample size.")
    return sorted(
        train,
        key=lambda question_id: (
            hashlib.sha256(f"{seed}:{question_id}".encode("utf-8")).digest(),
            question_id,
        ),
    )[:size]


def validate_preflight(
    *, project_root: Path, selection_directory: Path, train_path: Path,
    dev_path: Path, config: E07Config, advancement_directory: Path | None = None,
) -> dict[str, Any]:
    train_cfg, dev_cfg = config.section("train"), config.section("dev")
    scorer_cfg = config.section("scoring")
    scorer = project_root / str(scorer_cfg["official_scorer_path"])
    if not scorer.is_file() or file_sha256(scorer) != scorer_cfg["official_scorer_sha256"]:
        raise E07Error("Pinned scorer is missing or changed.")
    if not train_path.is_file() or file_sha256(train_path) != train_cfg["sha256"]:
        raise E07Error("Pinned official train split is missing or changed.")
    if not dev_path.is_file() or file_sha256(dev_path) != dev_cfg["sha256"]:
        raise E07Error("Pinned dev split is missing or changed.")
    train = json.loads(train_path.read_text(encoding="utf-8"))
    if not isinstance(train, dict) or len(train) != train_cfg["record_count"]:
        raise E07Error("Official train record count changed.")
    train_ids = select_train_sample(
        train, seed=train_cfg["sample_seed"], size=train_cfg["sample_size"]
    )
    train_ids_sha = hashlib.sha256("\n".join(train_ids).encode("utf-8")).hexdigest()
    if train_ids_sha != train_cfg["sample_ids_sha256"]:
        raise E07Error("E07 deterministic train sample changed.")
    dev_ids = select_dev_sample(
        _load_dev(dev_path), seed=dev_cfg["sample_seed"], size=dev_cfg["sample_size"]
    )
    dev_ids_sha = hashlib.sha256("\n".join(dev_ids).encode("utf-8")).hexdigest()
    if dev_ids_sha != dev_cfg["sample_ids_sha256"]:
        raise E07Error("E07 deterministic dev sample changed.")

    report_path = selection_directory / "report.json"
    generation_path = selection_directory / "generation" / "results.jsonl"
    prepared_path = selection_directory / "prepared" / "results.jsonl"
    if not all(path.is_file() for path in (report_path, generation_path, prepared_path)):
        raise E07Error("Saved E06 artifact is incomplete.")
    selection = config.section("selection_source")
    if file_sha256(generation_path) != selection["generation_results_sha256"]:
        raise E07Error("E06 generation results changed.")
    if file_sha256(prepared_path) != selection["prepared_results_sha256"]:
        raise E07Error("E06 prepared contexts changed.")
    _validate_e06_report(json.loads(report_path.read_text(encoding="utf-8")), selection)
    advancement_evidence = None
    if "advancement_source" in config.raw:
        if advancement_directory is None:
            raise E07Error("E07 full-train requires the saved train-1024 smoke artifact.")
        smoke_report_path = advancement_directory / "report.json"
        if not smoke_report_path.is_file():
            raise E07Error("Saved E07 train-1024 report is missing.")
        smoke_report = json.loads(smoke_report_path.read_text(encoding="utf-8"))
        _validate_smoke_report(smoke_report, config.section("advancement_source"))
        advancement_evidence = {
            "report_sha256": file_sha256(smoke_report_path),
            "adapter_sha256": smoke_report["evidence"]["adapter_sha256"],
            "evaluation_results_sha256": smoke_report["evidence"][
                "evaluation_results_sha256"
            ],
        }
    evidence = {
        "config_sha256": config.config_sha256,
        "train_sha256": train_cfg["sha256"],
        "train_sample_ids_sha256": train_ids_sha,
        "train_sample_size": len(train_ids),
        "dev_sha256": dev_cfg["sha256"],
        "dev_sample_ids_sha256": dev_ids_sha,
        "dev_sample_size": len(dev_ids),
        "e06_report_sha256": file_sha256(report_path),
        "e06_generation_results_sha256": file_sha256(generation_path),
        "e06_prepared_results_sha256": file_sha256(prepared_path),
        "maximum_stack_parameters": config.section("parameter_budget")["maximum_stack_total"],
    }
    if advancement_evidence is not None:
        evidence["e07_smoke_advancement"] = advancement_evidence
    return evidence


def prepare_train_records(
    *, train_path: Path, output_directory: Path, config: E07Config,
    preflight: dict[str, Any],
) -> dict[str, Any]:
    train_cfg = config.section("train")
    if file_sha256(train_path) != preflight.get("train_sha256"):
        raise E07Error("Train split changed after preflight.")
    train = json.loads(train_path.read_text(encoding="utf-8"))
    ids = select_train_sample(train, seed=train_cfg["sample_seed"], size=train_cfg["sample_size"])
    rows = []
    for index, question_id in enumerate(ids):
        row = train[question_id]
        if not isinstance(row.get("question"), str) or not isinstance(row.get("answer"), str):
            raise E07Error(f"Invalid official train row: {question_id}")
        rows.append({
            "sample_index": index, "question_id": question_id,
            "question": row["question"], "answer": row["answer"],
            "supervision": "official-answer-only",
        })
    root = output_directory / "training-data"
    root.mkdir(parents=True, exist_ok=True)
    _atomic_jsonl(root / "records.jsonl", rows)
    summary = {
        "schema_version": "1.0", "record_count": len(rows),
        "records_sha256": file_sha256(root / "records.jsonl"),
        "sample_ids_sha256": train_cfg["sample_ids_sha256"],
        "answers_are_retrieval_labels": False,
        "contains_dev_holdout_or_public": False,
    }
    _atomic_json(root / "summary.json", summary)
    return summary


def inference_packing(config: E07Config) -> InferencePacking:
    section = config.section("inference")
    return InferencePacking(
        max_input_tokens=section["max_input_tokens"],
        minimum_contexts=section["minimum_contexts"],
        system_prompt=section["system_prompt"],
        answer_instruction=section["answer_instruction"],
    )


def load_lora_generator(
    *, config: E07Config, adapter_directory: Path, device: str,
) -> tuple[Any, Any, dict[str, Any], int]:
    try:
        import torch
        from peft import PeftModel
        from transformers import AutoTokenizer, Qwen3_5ForCausalLM
    except ImportError as exc:  # pragma: no cover
        raise E07Error("Install Transformers and PEFT for E07 inference.") from exc
    if device not in {"cuda:0", "cuda:1"} or torch.cuda.device_count() != 2:
        raise E07Error("E07 evaluation requires GPU T4 x2.")
    final_adapter = adapter_directory / "adapter-final"
    if not (final_adapter / "adapter_config.json").is_file():
        raise E07Error("Completed E07 adapter is missing.")
    torch.cuda.set_device(int(device[-1]))
    tokenizer = AutoTokenizer.from_pretrained(
        config.generator_model_id, revision=config.generator_revision,
        trust_remote_code=False,
    )
    base = Qwen3_5ForCausalLM.from_pretrained(
        config.generator_model_id, revision=config.generator_revision,
        dtype=torch.float16, device_map={"": device}, low_cpu_mem_usage=True,
        trust_remote_code=False,
    )
    model = PeftModel.from_pretrained(base, final_adapter, is_trainable=False)
    model.eval()
    trainable_adapter_parameters = sum(
        parameter.numel() for name, parameter in model.named_parameters() if "lora_" in name
    )
    cap = config.section("lora")["trainable_parameter_cap"]
    if not 0 < trainable_adapter_parameters <= cap:
        raise E07Error("Observed adapter parameter count violates the pinned cap.")
    device_map = _validate_worker_placement(model, device)
    LOGGER.info(
        "e07_lora_generator_loaded device=%s adapter_parameters=%d",
        device, trainable_adapter_parameters,
    )
    return model, tokenizer, device_map, trainable_adapter_parameters


def run_eval_worker(
    *, worker_rank: int, device: str, model: Any, tokenizer: Any,
    adapter_directory: Path, selection_directory: Path, dev_path: Path,
    output_directory: Path, config: E07Config, device_map: dict[str, Any],
    adapter_parameters: int,
) -> dict[str, Any]:
    workers = config.section("execution")["evaluation_workers"]
    if worker_rank not in range(workers) or device != f"cuda:{worker_rank}":
        raise E07Error("E07 evaluation worker mapping changed.")
    selection = config.section("selection_source")
    prepared_path = selection_directory / "prepared" / "results.jsonl"
    source_path = selection_directory / "generation" / "results.jsonl"
    if (
        file_sha256(prepared_path) != selection["prepared_results_sha256"]
        or file_sha256(source_path) != selection["generation_results_sha256"]
    ):
        raise E07Error("E06 evaluation source changed.")
    prepared_rows, source_rows = _read_jsonl(prepared_path), _read_jsonl(source_path)
    dev = _load_dev(dev_path)
    dev_cfg = config.section("dev")
    sample_ids = select_dev_sample(dev, seed=dev_cfg["sample_seed"], size=dev_cfg["sample_size"])
    assigned = [index for index in range(len(sample_ids)) if index % workers == worker_rank]
    adapter_hash = file_sha256(adapter_directory / "adapter-final" / "adapter_model.safetensors")
    identity = {
        "code_version": CODE_VERSION, "stage": "e07-lora-eval-worker",
        "config_sha256": config.config_sha256, "adapter_sha256": adapter_hash,
        "adapter_parameters": adapter_parameters, "worker_rank": worker_rank,
        "device": device, "device_map": device_map,
        "assigned_indices_sha256": hashlib.sha256(
            ",".join(map(str, assigned)).encode("ascii")
        ).hexdigest(),
    }
    identity["identity_sha256"] = _json_sha256(identity)
    root, records = output_directory / "evaluation", output_directory / "evaluation" / "records"
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
        index, question_id = assigned[position], sample_ids[assigned[position]]
        contexts = prepared_rows[index]["contexts"]
        selected, messages, input_tokens = pack_contexts(
            question=dev[question_id]["question"], contexts=contexts,
            config=inference_packing(config), token_counter=token_count,
        )
        rendered = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        inputs = tokenizer(rendered, add_special_tokens=False, return_tensors="pt")
        inputs = {key: value.to(device) for key, value in inputs.items()}
        started = time.perf_counter()
        generated = model.generate(
            **inputs, do_sample=False, num_beams=1,
            max_new_tokens=config.section("inference")["max_new_tokens"], use_cache=True,
        )
        latency_ms = (time.perf_counter() - started) * 1000
        answer = tokenizer.decode(
            generated[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        ).strip()
        if not answer:
            raise E07Error(f"Empty E07 LoRA answer: {question_id}")
        control = source_rows[index]["variants"]["control_e05_ranked_top12"]
        _atomic_json(records / f"{index:04d}.json", {
            "question_id": question_id, "sample_index": index,
            "worker_rank": worker_rank, "worker_identity_sha256": identity["identity_sha256"],
            "variants": {
                "frozen_e06": {**control, "answer_source": "reused-e06-frozen"},
                "lora_e07": {
                    "answer": answer, "answer_source": "generated-e07-lora",
                    "selected_chunk_ids": [row["chunk_id"] for row in selected],
                    "selected_context_count": len(selected), "input_tokens": input_tokens,
                    "output_tokens": len(tokenizer(answer, add_special_tokens=False)["input_ids"]),
                    "generation_latency_ms": latency_ms,
                },
            },
        })
        _write_worker_state(state_path, identity, position + 1, len(assigned))
        LOGGER.info(
            "e07_eval_progress worker=%d device=%s completed=%d total=%d question_id=%s",
            worker_rank, device, position + 1, len(assigned), question_id,
        )
    _write_worker_state(state_path, identity, len(assigned), len(assigned))
    return {"worker_rank": worker_rank, "completed": len(assigned)}


def finalize_and_score(
    *, output_directory: Path, adapter_directory: Path, dev_path: Path,
    config: E07Config,
) -> dict[str, Any]:
    ensure_nltk_resources(download=False)
    dev, dev_cfg = _load_dev(dev_path), config.section("dev")
    sample_ids = select_dev_sample(dev, seed=dev_cfg["sample_seed"], size=dev_cfg["sample_size"])
    root, records = output_directory / "evaluation", output_directory / "evaluation" / "records"
    rows = []
    for index, question_id in enumerate(sample_ids):
        path = records / f"{index:04d}.json"
        if not path.is_file():
            raise E07Error(f"Missing E07 evaluation record: {index}")
        row = json.loads(path.read_text(encoding="utf-8"))
        if row.get("question_id") != question_id or set(row.get("variants", {})) != {
            "frozen_e06", "lora_e07"
        }:
            raise E07Error(f"Invalid E07 evaluation record: {index}")
        rows.append(row)
    for rank in range(config.section("execution")["evaluation_workers"]):
        state = json.loads((root / f"worker-{rank}-state.json").read_text(encoding="utf-8"))
        if state.get("complete") is not True:
            raise E07Error(f"E07 evaluation worker incomplete: {rank}")
    _atomic_jsonl(root / "results.jsonl", rows)
    scored = {key: [] for key in ("frozen_e06", "lora_e07")}
    for row in rows:
        reference = dev[row["question_id"]]["answer"]
        for key in scored:
            answer = row["variants"][key]["answer"]
            scored[key].append({
                "question_id": row["question_id"],
                "meteor": nltk_meteor_score(reference, answer),
                "rouge_l": rouge_l_fmeasure(reference, answer),
            })
    metrics = {
        key: {
            "meteor": fmean(item["meteor"] for item in values),
            "rouge_l": fmean(item["rouge_l"] for item in values),
            "mean_output_tokens": fmean(row["variants"][key]["output_tokens"] for row in rows),
        }
        for key, values in scored.items()
    }
    selection = config.section("selection_source")
    if metrics["frozen_e06"]["meteor"] != selection["meteor"] or metrics[
        "frozen_e06"
    ]["rouge_l"] != selection["rouge_l"]:
        raise E07Error("Re-scored frozen E06 control changed.")
    meteor_delta = [a["meteor"] - b["meteor"] for a, b in zip(scored["lora_e07"], scored["frozen_e06"])]
    rouge_delta = [a["rouge_l"] - b["rouge_l"] for a, b in zip(scored["lora_e07"], scored["frozen_e06"])]
    scoring = config.section("scoring")
    report = {
        "schema_version": "1.0", "experiment_id": config.raw["experiment_id"],
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "sample_size": len(rows),
        "train_sample_size": config.section("train")["sample_size"],
        "metrics": metrics,
        "paired_delta_lora_minus_frozen": {
            "meteor_mean": fmean(meteor_delta),
            "meteor_bootstrap_95_ci": _bootstrap_ci(
                meteor_delta, seed=f"{scoring['bootstrap_seed']}:meteor",
                iterations=scoring["bootstrap_iterations"],
            ),
            "rouge_l_mean": fmean(rouge_delta),
            "rouge_l_bootstrap_95_ci": _bootstrap_ci(
                rouge_delta, seed=f"{scoring['bootstrap_seed']}:rouge_l",
                iterations=scoring["bootstrap_iterations"],
            ),
        },
        "smoke_leader": max(metrics, key=lambda key: (metrics[key]["meteor"], metrics[key]["rouge_l"])),
        "promotion_allowed": False,
        "evidence": {
            "config_sha256": config.config_sha256,
            "evaluation_results_sha256": file_sha256(root / "results.jsonl"),
            "adapter_sha256": file_sha256(adapter_directory / "adapter-final" / "adapter_model.safetensors"),
            "training_records_sha256": file_sha256(output_directory / "training-data" / "records.jsonl"),
        },
        "warning": (
            "E07 full-train/dev200 is a confirmation smoke; formal promotion requires "
            "full-dev-721."
            if config.section("train")["sample_size"] == 5636
            else "E07 train1024/dev200 is a smoke LoRA experiment; formal promotion "
            "requires full train/full dev."
        ),
    }
    _atomic_json(output_directory / "report.json", report)
    return report


def _validate_e06_report(report: dict[str, Any], selection: dict[str, Any]) -> None:
    metrics = report.get("metrics", {}).get(selection["selected_variant"], {})
    evidence = report.get("evidence", {})
    if (
        report.get("experiment_id") != selection["report_experiment_id"]
        or report.get("sample_size") != selection["sample_size"]
        or report.get("smoke_leader") != selection["selected_variant"]
        or report.get("promotion_allowed") is not False
        or evidence.get("config_sha256") != selection["report_config_sha256"]
        or evidence.get("generation_results_sha256") != selection["generation_results_sha256"]
        or evidence.get("prepared_results_sha256") != selection["prepared_results_sha256"]
        or metrics.get("meteor") != selection["meteor"]
        or metrics.get("rouge_l") != selection["rouge_l"]
    ):
        raise E07Error("Saved E06 report differs from E07 selection evidence.")


def _validate_smoke_report(report: dict[str, Any], advancement: dict[str, Any]) -> None:
    metrics = report.get("metrics", {})
    frozen, lora = metrics.get("frozen_e06", {}), metrics.get("lora_e07", {})
    paired = report.get("paired_delta_lora_minus_frozen", {})
    evidence = report.get("evidence", {})
    if (
        report.get("experiment_id") != advancement["report_experiment_id"]
        or report.get("sample_size") != advancement["sample_size"]
        or report.get("train_sample_size") != advancement["train_sample_size"]
        or report.get("smoke_leader") != advancement["smoke_leader"]
        or report.get("promotion_allowed") is not advancement["promotion_allowed"]
        or evidence.get("config_sha256") != advancement["report_config_sha256"]
        or evidence.get("evaluation_results_sha256")
        != advancement["evaluation_results_sha256"]
        or evidence.get("adapter_sha256") != advancement["adapter_sha256"]
        or evidence.get("training_records_sha256")
        != advancement["training_records_sha256"]
        or frozen.get("meteor") != advancement["frozen_meteor"]
        or frozen.get("rouge_l") != advancement["frozen_rouge_l"]
        or lora.get("meteor") != advancement["lora_meteor"]
        or lora.get("rouge_l") != advancement["lora_rouge_l"]
        or paired.get("meteor_bootstrap_95_ci", [None])[0]
        != advancement["meteor_delta_ci_lower"]
        or paired.get("rouge_l_bootstrap_95_ci", [None])[0]
        != advancement["rouge_l_delta_ci_lower"]
    ):
        raise E07Error("Saved E07 train-1024 smoke report changed.")
