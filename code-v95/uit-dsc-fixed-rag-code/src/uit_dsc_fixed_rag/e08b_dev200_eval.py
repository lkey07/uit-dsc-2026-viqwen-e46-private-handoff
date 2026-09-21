"""Paired dev-200 evaluation for the context-aware E08B LoRA adapter."""

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
    _write_worker_state,
)
from uit_dsc_fixed_rag.e07_lora import InferencePacking
from uit_dsc_fixed_rag.e08b_context_lora import (
    E08BConfig,
    load_context_lora_generator,
    load_e08b_config,
)
from uit_dsc_fixed_rag.e10_repetition_grid import answer_diagnostics
from uit_dsc_fixed_rag.evaluation.official import (
    ensure_nltk_resources,
    nltk_meteor_score,
    rouge_l_fmeasure,
)
from uit_dsc_fixed_rag.retrieval_diagnostics import select_dev_sample


CODE_VERSION = "0.30.0"
LOGGER = logging.getLogger(__name__)


class E08BDev200Error(RuntimeError):
    """Raised when the paired E08B dev-200 experiment loses identity."""


@dataclass(frozen=True)
class E08BDev200Config:
    raw: dict[str, Any]
    path: Path

    @property
    def config_sha256(self) -> str:
        return file_sha256(self.path)

    def section(self, key: str) -> dict[str, Any]:
        value = self.raw.get(key)
        if not isinstance(value, dict):
            raise ValueError(f"E08B dev-200 section must be an object: {key}")
        return value


def load_config(path: Path) -> E08BDev200Config:
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "schema_version", "experiment_id", "source_contexts", "source_control",
        "source_training", "dev", "generator", "lora", "inference",
        "parameter_budget", "execution", "scoring", "run_contract",
    }
    if (
        set(payload) != required
        or payload.get("schema_version") != "1.0"
        or payload.get("experiment_id")
        != "E08B-context-aware-lora-dev200-max704-paired-v1"
    ):
        raise ValueError("E08B dev-200 config root is incompatible.")
    config = E08BDev200Config(payload, path)
    contexts = config.section("source_contexts")
    if (
        contexts.get("experiment_id") != "E06-prompt-length-grid-aiteam-dev200-v1"
        or contexts.get("path") != "prepared/results.jsonl"
        or contexts.get("record_count") != 200
        or contexts.get("contexts_per_question") != 12
    ):
        raise ValueError("E08B dev-200 context identity changed.")
    control = config.section("source_control")
    if (
        control.get("experiment_id")
        != "E11-output-length-refinement-qwen35-lora-dev200-v1"
        or control.get("path") != "results.jsonl"
        or control.get("variant") != "max704"
        or control.get("max_new_tokens") != 704
        or control.get("sample_size") != 200
    ):
        raise ValueError("E08B dev-200 control identity changed.")
    training = config.section("source_training")
    if (
        training.get("experiment_id")
        != "E08B-context-aware-lora-train5636-dev521-v3"
        or training.get("config_path") != "configs/e08b-context-aware-lora-v3.json"
        or training.get("fresh_from_base") is not True
        or training.get("train_sample_size") != 5636
    ):
        raise ValueError("E08B training identity changed.")
    inference = config.section("inference")
    if (
        inference.get("max_input_tokens") != 8192
        or inference.get("minimum_contexts") != 1
        or inference.get("max_new_tokens") != 704
        or inference.get("do_sample") is not False
        or inference.get("num_beams") != 1
        or inference.get("enable_thinking") is not False
        or inference.get("use_cache") is not True
        or inference.get("repetition_penalty") != 1.0
        or inference.get("no_repeat_ngram_size") != 0
    ):
        raise ValueError("E08B dev-200 inference identity changed.")
    execution = config.section("execution")
    if (
        execution.get("workers") != 2
        or execution.get("partition") != "sample-index-mod-worker-count"
        or execution.get("questions_per_worker") != 100
        or execution.get("checkpoint_after_questions") != 1
    ):
        raise ValueError("E08B dev-200 execution identity changed.")
    budget = config.section("parameter_budget")
    if (
        budget.get("maximum_stack_total")
        != budget.get("embedding") + budget.get("generator")
        + budget.get("adapter_parameter_cap")
        or budget["maximum_stack_total"] >= budget["exclusive_limit"]
    ):
        raise ValueError("E08B dev-200 parameter budget failed.")
    contract = config.section("run_contract")
    if not contract or not all(value is True for value in contract.values()):
        raise ValueError("E08B dev-200 run contract lost an invariant.")
    if config.section("scoring").get("promotion_allowed") is not False:
        raise ValueError("E08B dev-200 cannot auto-promote.")
    return config


def _sample_ids(
    dev_path: Path, config: E08BDev200Config,
) -> tuple[dict[str, Any], list[str]]:
    section = config.section("dev")
    if not dev_path.is_file() or file_sha256(dev_path) != section["sha256"]:
        raise E08BDev200Error("Pinned dev split is missing or changed.")
    dev = _load_dev(dev_path)
    ids = select_dev_sample(dev, seed=section["sample_seed"], size=section["sample_size"])
    identity = hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()
    if identity != section["sample_ids_sha256"]:
        raise E08BDev200Error("Deterministic dev-200 IDs changed.")
    return dev, ids


def _context_rows(
    contexts_directory: Path, dev_path: Path, config: E08BDev200Config,
) -> list[dict[str, Any]]:
    section = config.section("source_contexts")
    path = contexts_directory / section["path"]
    if not path.is_file() or file_sha256(path) != section["sha256"]:
        raise E08BDev200Error("Saved E06 ranked top-12 contexts changed.")
    _, ids = _sample_ids(dev_path, config)
    rows = _read_jsonl(path)
    if len(rows) != len(ids):
        raise E08BDev200Error("Saved E06 context count changed.")
    for index, (question_id, row) in enumerate(zip(ids, rows)):
        contexts = row.get("contexts")
        if (
            row.get("question_id") != question_id
            or row.get("sample_index") != index
            or not isinstance(contexts, list)
            or len(contexts) != section["contexts_per_question"]
            or any(not isinstance(item.get("text"), str) for item in contexts)
        ):
            raise E08BDev200Error(f"Invalid E06 context row: {index}")
    return rows


def _control_rows(
    control_directory: Path, dev_path: Path, config: E08BDev200Config,
) -> list[dict[str, Any]]:
    section = config.section("source_control")
    path = control_directory / section["path"]
    if not path.is_file() or file_sha256(path) != section["sha256"]:
        raise E08BDev200Error("Saved E11 max704 results changed.")
    _, ids = _sample_ids(dev_path, config)
    rows = _read_jsonl(path)
    if len(rows) != len(ids):
        raise E08BDev200Error("Saved E11 result count changed.")
    for index, (question_id, row) in enumerate(zip(ids, rows)):
        variant = row.get("variants", {}).get(section["variant"], {})
        if (
            row.get("question_id") != question_id
            or row.get("sample_index") != index
            or not isinstance(variant.get("answer"), str)
            or not variant["answer"].strip()
            or variant.get("max_new_tokens") != section["max_new_tokens"]
            or not isinstance(variant.get("output_tokens"), int)
            or variant.get("finish_reason") not in {"eos", "length", "other"}
        ):
            raise E08BDev200Error(f"Invalid E11 max704 row: {index}")
    report_path = control_directory / "report.json"
    if not report_path.is_file():
        raise E08BDev200Error("Saved E11 report is missing.")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    metric = report.get("metrics", {}).get(section["variant"], {})
    if (
        report.get("experiment_id") != section["experiment_id"]
        or report.get("sample_size") != section["sample_size"]
        or report.get("evidence", {}).get("config_sha256") != section["config_sha256"]
        or report.get("evidence", {}).get("results_sha256") != section["sha256"]
        or metric.get("meteor") != section["meteor"]
        or metric.get("rouge_l") != section["rouge_l"]
        or metric.get("mean_output_tokens") != section["mean_output_tokens"]
    ):
        raise E08BDev200Error("Saved E11 report identity changed.")
    return rows


def _training_config(
    project_root: Path, config: E08BDev200Config,
) -> E08BConfig:
    section = config.section("source_training")
    path = project_root / section["config_path"]
    if not path.is_file() or file_sha256(path) != section["config_sha256"]:
        raise E08BDev200Error("Pinned E08B training config changed.")
    return load_e08b_config(path)


def _adapter_evidence(
    training_directory: Path,
    training_config: E08BConfig,
    config: E08BDev200Config,
) -> dict[str, Any]:
    source = config.section("source_training")
    final = training_directory / "adapter-final"
    adapter_path = training_directory / source["adapter_path"]
    complete_path = final / "complete.json"
    if not adapter_path.is_file() or not complete_path.is_file():
        raise E08BDev200Error("Completed E08B adapter is missing.")
    complete = json.loads(complete_path.read_text(encoding="utf-8"))
    adapter_sha = file_sha256(adapter_path)
    if (
        complete.get("experiment_id") != source["experiment_id"]
        or complete.get("config_sha256") != training_config.config_sha256
        or complete.get("source_contexts_sha256") != source["source_contexts_sha256"]
        or complete.get("adapter_sha256") != adapter_sha
        or complete.get("fresh_from_base") is not True
        or complete.get("e07_adapter_loaded") is not False
    ):
        raise E08BDev200Error("Completed E08B adapter evidence changed.")
    return {"adapter_sha256": adapter_sha, "complete": complete}


def validate_preflight(
    *, project_root: Path, contexts_directory: Path, control_directory: Path,
    training_directory: Path, dev_path: Path, config: E08BDev200Config,
) -> dict[str, Any]:
    scorer = project_root / config.section("scoring")["official_scorer_path"]
    if (
        not scorer.is_file()
        or file_sha256(scorer)
        != config.section("scoring")["official_scorer_sha256"]
    ):
        raise E08BDev200Error("Pinned official scorer changed.")
    training_config = _training_config(project_root, config)
    adapter = _adapter_evidence(training_directory, training_config, config)
    contexts = _context_rows(contexts_directory, dev_path, config)
    controls = _control_rows(control_directory, dev_path, config)
    return {
        "config_sha256": config.config_sha256,
        "training_config_sha256": training_config.config_sha256,
        "adapter_sha256": adapter["adapter_sha256"],
        "contexts_sha256": config.section("source_contexts")["sha256"],
        "control_results_sha256": config.section("source_control")["sha256"],
        "sample_ids_sha256": config.section("dev")["sample_ids_sha256"],
        "sample_size": len(contexts),
        "control_size": len(controls),
    }


def load_generator(
    *, project_root: Path, training_directory: Path,
    config: E08BDev200Config, device: str,
) -> tuple[Any, Any, dict[str, Any], int]:
    training_config = _training_config(project_root, config)
    try:
        return load_context_lora_generator(
            config=training_config,
            training_directory=training_directory,
            device=device,
        )
    except Exception as exc:
        raise E08BDev200Error(str(exc)) from exc


def _packing(config: E08BDev200Config) -> InferencePacking:
    section = config.section("inference")
    return InferencePacking(
        max_input_tokens=section["max_input_tokens"],
        minimum_contexts=section["minimum_contexts"],
        system_prompt=section["system_prompt"],
        answer_instruction=section["answer_instruction"],
    )


def assigned_indices(config: E08BDev200Config, worker_rank: int) -> list[int]:
    workers = config.section("execution")["workers"]
    if worker_rank not in range(workers):
        raise E08BDev200Error("Invalid E08B dev-200 worker rank.")
    return [index for index in range(config.section("dev")["sample_size"])
            if index % workers == worker_rank]


def run_worker(
    *, worker_rank: int, device: str, model: Any, tokenizer: Any,
    project_root: Path, contexts_directory: Path, control_directory: Path,
    training_directory: Path, dev_path: Path, output_directory: Path,
    config: E08BDev200Config, device_map: dict[str, Any], adapter_parameters: int,
) -> dict[str, Any]:
    if device != f"cuda:{worker_rank}":
        raise E08BDev200Error("E08B dev-200 worker-to-device mapping changed.")
    contexts = _context_rows(contexts_directory, dev_path, config)
    _control_rows(control_directory, dev_path, config)
    dev, ids = _sample_ids(dev_path, config)
    assigned = assigned_indices(config, worker_rank)
    adapter_path = training_directory / config.section("source_training")["adapter_path"]
    identity = {
        "code_version": CODE_VERSION,
        "stage": "e08b-context-aware-dev200-max704-worker",
        "config_sha256": config.config_sha256,
        "adapter_sha256": file_sha256(adapter_path),
        "adapter_parameters": adapter_parameters,
        "contexts_sha256": config.section("source_contexts")["sha256"],
        "control_results_sha256": config.section("source_control")["sha256"],
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
        records=records, state_path=state_path, identity=identity,
        assigned_indices=assigned, sample_ids=ids,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    def token_count(messages: list[dict[str, Any]]) -> int:
        rendered = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
        return len(tokenizer(rendered, add_special_tokens=False)["input_ids"])

    inference = config.section("inference")
    for position in range(completed, len(assigned)):
        index = assigned[position]
        question_id = ids[index]
        selected, messages, input_tokens = pack_contexts(
            question=dev[question_id]["question"],
            contexts=contexts[index]["contexts"], config=_packing(config),
            token_counter=token_count,
        )
        rendered = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = tokenizer(rendered, add_special_tokens=False, return_tensors="pt")
        inputs = {key: value.to(device) for key, value in inputs.items()}
        started = time.perf_counter()
        generated = model.generate(
            **inputs, do_sample=False, num_beams=1,
            max_new_tokens=inference["max_new_tokens"], use_cache=True,
            repetition_penalty=1.0, no_repeat_ngram_size=0,
        )
        latency_ms = (time.perf_counter() - started) * 1000
        new_ids = generated[0, inputs["input_ids"].shape[1]:]
        answer = tokenizer.decode(
            new_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False,
        ).strip()
        if not answer:
            raise E08BDev200Error(f"Empty E08B dev-200 answer: {question_id}")
        eos = tokenizer.eos_token_id
        eos_ids = set(eos if isinstance(eos, list) else [eos])
        generated_count = int(new_ids.shape[0])
        finish_reason = (
            "eos" if generated_count and int(new_ids[-1]) in eos_ids
            else "length" if generated_count >= inference["max_new_tokens"]
            else "other"
        )
        _atomic_json(records / f"{index:04d}.json", {
            "question_id": question_id,
            "sample_index": index,
            "worker_rank": worker_rank,
            "worker_identity_sha256": identity["identity_sha256"],
            "variant": "e08b_context_lora_max704",
            "answer": answer,
            "max_new_tokens": inference["max_new_tokens"],
            "selected_chunk_ids": [item["chunk_id"] for item in selected],
            "selected_context_count": len(selected),
            "input_tokens": input_tokens,
            "output_tokens": len(tokenizer(answer, add_special_tokens=False)["input_ids"]),
            "generated_tokens_including_special": generated_count,
            "finish_reason": finish_reason,
            "generation_latency_ms": latency_ms,
        })
        _write_worker_state(state_path, identity, position + 1, len(assigned))
        LOGGER.info(
            "e08b_dev200_progress worker=%d device=%s completed=%d total=%d "
            "global_completed_at_least=%d global_total=%d question_id=%s finish=%s",
            worker_rank, device, position + 1, len(assigned),
            min(len(ids), (position + 1) * 2), len(ids), question_id, finish_reason,
        )
    _write_worker_state(state_path, identity, len(assigned), len(assigned))
    return {"worker_rank": worker_rank, "device": device, "completed": len(assigned)}


def finalize_score(
    *, contexts_directory: Path, control_directory: Path,
    training_directory: Path, dev_path: Path, output_directory: Path,
    config: E08BDev200Config,
) -> dict[str, Any]:
    ensure_nltk_resources(download=False)
    dev, ids = _sample_ids(dev_path, config)
    controls = _control_rows(control_directory, dev_path, config)
    _context_rows(contexts_directory, dev_path, config)
    candidate_rows: list[dict[str, Any]] = []
    root = output_directory / "evaluation"
    for rank in range(config.section("execution")["workers"]):
        state_path = root / f"worker-{rank}-state.json"
        if not state_path.is_file():
            raise E08BDev200Error(f"Missing evaluation worker state: {rank}")
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if (
            state.get("complete") is not True
            or state.get("completed_count") != len(assigned_indices(config, rank))
        ):
            raise E08BDev200Error(f"Incomplete evaluation worker: {rank}")
    for index, question_id in enumerate(ids):
        path = root / "records" / f"{index:04d}.json"
        if not path.is_file():
            raise E08BDev200Error(f"Missing E08B dev-200 record: {index}")
        row = json.loads(path.read_text(encoding="utf-8"))
        if (
            row.get("question_id") != question_id
            or row.get("sample_index") != index
            or row.get("worker_rank") != index % 2
            or row.get("variant") != "e08b_context_lora_max704"
            or row.get("max_new_tokens") != 704
            or row.get("finish_reason") not in {"eos", "length", "other"}
            or not isinstance(row.get("answer"), str)
            or not row["answer"].strip()
        ):
            raise E08BDev200Error(f"Invalid E08B dev-200 record: {index}")
        candidate_rows.append(row)
    _atomic_jsonl(root / "results.jsonl", candidate_rows)

    control_key = "e11_max704_control"
    candidate_key = "e08b_context_lora_max704"
    scored = {control_key: [], candidate_key: []}
    merged = []
    source_control = config.section("source_control")
    for index, question_id in enumerate(ids):
        control = controls[index]["variants"][source_control["variant"]]
        candidate = candidate_rows[index]
        variants = {
            control_key: {
                **control,
                "answer_source": "reused-byte-verified-e11-max704-control",
            },
            candidate_key: candidate,
        }
        merged.append({"question_id": question_id, "sample_index": index, "variants": variants})
        reference = dev[question_id]["answer"]
        for key in (control_key, candidate_key):
            scored[key].append({
                "meteor": nltk_meteor_score(reference, variants[key]["answer"]),
                "rouge_l": rouge_l_fmeasure(reference, variants[key]["answer"]),
            })
    results_path = output_directory / "results.jsonl"
    _atomic_jsonl(results_path, merged)
    metrics: dict[str, dict[str, Any]] = {}
    for key in (control_key, candidate_key):
        rows = [row["variants"][key] for row in merged]
        diagnostics = [answer_diagnostics(row["answer"]) for row in rows]
        metric = {
            "meteor": fmean(item["meteor"] for item in scored[key]),
            "rouge_l": fmean(item["rouge_l"] for item in scored[key]),
            "mean_output_tokens": fmean(row["output_tokens"] for row in rows),
            "mean_answer_characters": fmean(len(row["answer"]) for row in rows),
            "length_finish_rate": fmean(row["finish_reason"] == "length" for row in rows),
            "duplicate_line_rate": fmean(item["duplicate_line"] for item in diagnostics),
            "duplicate_sentence_rate": fmean(item["duplicate_sentence"] for item in diagnostics),
            "abbreviation_loop_rate": fmean(item["abbreviation_loop"] for item in diagnostics),
            "non_sentence_ending_rate": fmean(item["non_sentence_ending"] for item in diagnostics),
        }
        if key == candidate_key:
            metric["mean_generation_latency_ms"] = fmean(
                row["generation_latency_ms"] for row in rows
            )
        metrics[key] = metric
    if (
        abs(metrics[control_key]["meteor"] - source_control["meteor"]) > 1e-12
        or abs(metrics[control_key]["rouge_l"] - source_control["rouge_l"]) > 1e-12
        or abs(metrics[control_key]["mean_output_tokens"]
               - source_control["mean_output_tokens"]) > 1e-12
    ):
        raise E08BDev200Error("Re-scored E11 max704 control changed.")
    meteor_deltas = [
        candidate["meteor"] - control["meteor"]
        for candidate, control in zip(scored[candidate_key], scored[control_key])
    ]
    rouge_deltas = [
        candidate["rouge_l"] - control["rouge_l"]
        for candidate, control in zip(scored[candidate_key], scored[control_key])
    ]
    scoring = config.section("scoring")
    paired = {
        "meteor_mean": fmean(meteor_deltas),
        "meteor_bootstrap_95_ci": _bootstrap_ci(
            meteor_deltas, seed=f"{scoring['bootstrap_seed']}:meteor",
            iterations=scoring["bootstrap_iterations"],
        ),
        "rouge_l_mean": fmean(rouge_deltas),
        "rouge_l_bootstrap_95_ci": _bootstrap_ci(
            rouge_deltas, seed=f"{scoring['bootstrap_seed']}:rouge_l",
            iterations=scoring["bootstrap_iterations"],
        ),
    }
    adapter_path = training_directory / config.section("source_training")["adapter_path"]
    leader = max(
        (control_key, candidate_key),
        key=lambda key: (metrics[key]["meteor"], metrics[key]["rouge_l"]),
    )
    report = {
        "schema_version": "1.0",
        "experiment_id": config.raw["experiment_id"],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "sample_size": len(ids),
        "control_variant": control_key,
        "candidate_variant": candidate_key,
        "only_changed_factor": "E07 answer-only LoRA versus E08B context-aware LoRA",
        "metrics": metrics,
        "paired_delta_e08b_minus_e11": paired,
        "smoke_leader": leader,
        "promotion_allowed": False,
        "evidence": {
            "config_sha256": config.config_sha256,
            "contexts_sha256": config.section("source_contexts")["sha256"],
            "control_results_sha256": config.section("source_control")["sha256"],
            "adapter_sha256": file_sha256(adapter_path),
            "sample_ids_sha256": config.section("dev")["sample_ids_sha256"],
            "results_sha256": file_sha256(results_path),
        },
        "warning": (
            "E08B dev-200 is a paired smoke comparison on the repeatedly used tuning "
            "sample. It reads neither public nor holdout and cannot auto-promote."
        ),
    }
    _atomic_json(output_directory / "report.json", report)
    return report


__all__ = [
    "E08BDev200Config", "E08BDev200Error", "assigned_indices",
    "finalize_score", "load_config", "load_generator", "run_worker",
    "validate_preflight",
]
