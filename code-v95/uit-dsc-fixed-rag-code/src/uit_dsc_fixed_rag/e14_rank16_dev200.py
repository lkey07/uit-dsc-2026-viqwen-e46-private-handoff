"""Paired max-704 dev-200 evaluation of fresh E14 rank-16 versus E08B rank-8."""

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
from uit_dsc_fixed_rag.e03_rrf_grid import (
    _bootstrap_ci,
    _json_sha256,
    _load_worker_progress,
    _write_worker_state,
)
from uit_dsc_fixed_rag.e08b_context_lora import load_context_lora_generator
from uit_dsc_fixed_rag.e08b_dev200_eval import _context_rows, _packing, _sample_ids
from uit_dsc_fixed_rag.e10_repetition_grid import answer_diagnostics
from uit_dsc_fixed_rag.e13_e08b_max768_tailtrim import _control_rows
from uit_dsc_fixed_rag.e14_rank16_lora import E14Config, load_config as load_training_config
from uit_dsc_fixed_rag.evaluation.official import (
    ensure_nltk_resources,
    nltk_meteor_score,
    rouge_l_fmeasure,
)


CODE_VERSION = "0.42.0"
LOGGER = logging.getLogger(__name__)


class E14Dev200Error(RuntimeError):
    """Raised when E14 evaluation evidence or checkpoint identity changes."""


@dataclass(frozen=True)
class E14Dev200Config:
    raw: dict[str, Any]
    path: Path

    @property
    def config_sha256(self) -> str:
        return file_sha256(self.path)

    def section(self, key: str) -> dict[str, Any]:
        value = self.raw.get(key)
        if not isinstance(value, dict):
            raise ValueError(f"E14 dev-200 section must be an object: {key}")
        return value


def load_config(path: Path) -> E14Dev200Config:
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
        != "E14-rank16-vs-rank8-dev200-max704-v1"
    ):
        raise ValueError("E14 dev-200 config root is incompatible.")
    config = E14Dev200Config(payload, path)
    contexts = config.section("source_contexts")
    if (
        contexts.get("experiment_id") != "E06-prompt-length-grid-aiteam-dev200-v1"
        or contexts.get("path") != "prepared/results.jsonl"
        or contexts.get("record_count") != 200
        or contexts.get("contexts_per_question") != 12
    ):
        raise ValueError("E14 context identity changed.")
    control = config.section("source_control")
    if (
        control.get("experiment_id")
        != "E08B-context-aware-lora-dev200-max704-paired-v1"
        or control.get("path") != "results.jsonl"
        or control.get("variant") != "e08b_context_lora_max704"
        or control.get("max_new_tokens") != 704
        or control.get("length_finish_rate") != 0.275
        or control.get("sample_size") != 200
    ):
        raise ValueError("E14 rank-8 control identity changed.")
    training = config.section("source_training")
    if (
        training.get("experiment_id")
        != "E14-context-aware-lora-rank16-train5636-v1"
        or training.get("config_path")
        != "configs/e14-context-aware-lora-rank16-train5636-v1.json"
        or training.get("fresh_from_base") is not True
        or training.get("old_rank8_adapter_loaded") is not False
        or training.get("train_sample_size") != 5636
        or training.get("rank") != 16
        or training.get("alpha") != 32
    ):
        raise ValueError("E14 rank-16 training identity changed.")
    lora = config.section("lora")
    if (
        lora.get("rank") != 16
        or lora.get("alpha") != 32
        or lora.get("initialization")
        != "reuse-completed-fresh-e14-rank16-adapter"
    ):
        raise ValueError("E14 rank-16 adapter contract changed.")
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
        raise ValueError("E14 max704 inference identity changed.")
    execution = config.section("execution")
    if (
        execution.get("workers") != 2
        or execution.get("partition") != "sample-index-mod-worker-count"
        or execution.get("questions_per_worker") != 100
        or execution.get("checkpoint_after_questions") != 1
    ):
        raise ValueError("E14 execution identity changed.")
    budget = config.section("parameter_budget")
    if (
        budget.get("maximum_stack_total")
        != budget.get("embedding") + budget.get("generator")
        + budget.get("adapter_parameter_cap")
        or budget["maximum_stack_total"] >= budget["exclusive_limit"]
    ):
        raise ValueError("E14 parameter budget failed.")
    contract = config.section("run_contract")
    if not contract or not all(value is True for value in contract.values()):
        raise ValueError("E14 run contract lost an invariant.")
    if config.section("scoring").get("promotion_allowed") is not False:
        raise ValueError("E14 dev-200 cannot auto-promote.")
    return config


def _training_config(project_root: Path, config: E14Dev200Config) -> E14Config:
    section = config.section("source_training")
    path = project_root / section["config_path"]
    if not path.is_file() or file_sha256(path) != section["config_sha256"]:
        raise E14Dev200Error("Pinned E14 training config changed.")
    training = load_training_config(path)
    if (
        training.section("prompt")["system_prompt"]
        != config.section("inference")["system_prompt"]
        or training.section("prompt")["answer_instruction"]
        != config.section("inference")["answer_instruction"]
        or training.section("inference")["max_new_tokens"] != 704
    ):
        raise E14Dev200Error("E14 prompt or max704 contract changed.")
    return training


def _adapter_evidence(
    training_directory: Path,
    training_config: E14Config,
    config: E14Dev200Config,
) -> dict[str, Any]:
    source = config.section("source_training")
    adapter_path = training_directory / source["adapter_path"]
    complete_path = training_directory / "adapter-final" / "complete.json"
    if not adapter_path.is_file() or not complete_path.is_file():
        raise E14Dev200Error("Completed E14 rank-16 adapter is missing.")
    complete = json.loads(complete_path.read_text(encoding="utf-8"))
    adapter_sha = file_sha256(adapter_path)
    trainable = complete.get("trainable_parameters")
    if (
        complete.get("experiment_id") != source["experiment_id"]
        or complete.get("config_sha256") != training_config.config_sha256
        or complete.get("source_contexts_sha256") != source["source_contexts_sha256"]
        or complete.get("adapter_sha256") != adapter_sha
        or complete.get("fresh_from_base") is not True
        or complete.get("e07_adapter_loaded") is not False
        or complete.get("e08b_adapter_loaded") is not False
        or complete.get("lora_rank") != 16
        or complete.get("lora_alpha") != 32
        or complete.get("evaluation_max_new_tokens") != 704
        or not isinstance(trainable, int)
        or not 0 < trainable <= config.section("lora")["adapter_parameter_cap"]
    ):
        raise E14Dev200Error("Completed E14 rank-16 adapter evidence changed.")
    return {
        "adapter_sha256": adapter_sha,
        "trainable_parameters": trainable,
        "complete": complete,
    }


def validate_preflight(
    *, project_root: Path, contexts_directory: Path, control_directory: Path,
    training_directory: Path, dev_path: Path, config: E14Dev200Config,
) -> dict[str, Any]:
    scorer = project_root / config.section("scoring")["official_scorer_path"]
    if (
        not scorer.is_file()
        or file_sha256(scorer)
        != config.section("scoring")["official_scorer_sha256"]
    ):
        raise E14Dev200Error("Pinned official scorer changed.")
    training = _training_config(project_root, config)
    adapter = _adapter_evidence(training_directory, training, config)
    contexts = _context_rows(contexts_directory, dev_path, config)
    controls = _control_rows(control_directory, dev_path, config)
    return {
        "config_sha256": config.config_sha256,
        "training_config_sha256": training.config_sha256,
        "adapter_sha256": adapter["adapter_sha256"],
        "adapter_trainable_parameters": adapter["trainable_parameters"],
        "contexts_sha256": config.section("source_contexts")["sha256"],
        "control_results_sha256": config.section("source_control")["sha256"],
        "sample_ids_sha256": config.section("dev")["sample_ids_sha256"],
        "sample_size": len(contexts),
        "control_size": len(controls),
        "lora_rank": 16,
        "lora_alpha": 32,
        "max_new_tokens": 704,
    }


def load_generator(
    *, project_root: Path, training_directory: Path,
    config: E14Dev200Config, device: str,
) -> tuple[Any, Any, dict[str, Any], int]:
    training = _training_config(project_root, config)
    _adapter_evidence(training_directory, training, config)
    try:
        return load_context_lora_generator(
            config=training,
            training_directory=training_directory,
            device=device,
        )
    except Exception as exc:
        raise E14Dev200Error(str(exc)) from exc


def assigned_indices(config: E14Dev200Config, worker_rank: int) -> list[int]:
    workers = config.section("execution")["workers"]
    if worker_rank not in range(workers):
        raise E14Dev200Error("Invalid E14 worker rank.")
    return [
        index for index in range(config.section("dev")["sample_size"])
        if index % workers == worker_rank
    ]


def run_worker(
    *, worker_rank: int, device: str, model: Any, tokenizer: Any,
    project_root: Path, contexts_directory: Path, control_directory: Path,
    training_directory: Path, dev_path: Path, output_directory: Path,
    config: E14Dev200Config, device_map: dict[str, Any], adapter_parameters: int,
) -> dict[str, Any]:
    if device != f"cuda:{worker_rank}":
        raise E14Dev200Error("E14 worker-to-device mapping changed.")
    contexts = _context_rows(contexts_directory, dev_path, config)
    _control_rows(control_directory, dev_path, config)
    dev, ids = _sample_ids(dev_path, config)
    assigned = assigned_indices(config, worker_rank)
    adapter_path = training_directory / config.section("source_training")["adapter_path"]
    identity = {
        "code_version": CODE_VERSION,
        "stage": "e14-rank16-vs-rank8-dev200-max704-worker",
        "config_sha256": config.config_sha256,
        "adapter_sha256": file_sha256(adapter_path),
        "adapter_parameters": adapter_parameters,
        "lora_rank": 16,
        "lora_alpha": 32,
        "max_new_tokens": 704,
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
        records=records,
        state_path=state_path,
        identity=identity,
        assigned_indices=assigned,
        sample_ids=ids,
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

    inference = config.section("inference")
    for position in range(completed, len(assigned)):
        index = assigned[position]
        question_id = ids[index]
        selected, messages, input_tokens = pack_contexts(
            question=dev[question_id]["question"],
            contexts=contexts[index]["contexts"],
            config=_packing(config),
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
            max_new_tokens=704,
            use_cache=True,
            repetition_penalty=1.0,
            no_repeat_ngram_size=0,
        )
        latency_ms = (time.perf_counter() - started) * 1000
        new_ids = generated[0, inputs["input_ids"].shape[1]:]
        answer = tokenizer.decode(
            new_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        ).strip()
        if not answer:
            raise E14Dev200Error(f"Empty E14 answer: {question_id}")
        eos = tokenizer.eos_token_id
        eos_ids = set(eos if isinstance(eos, list) else [eos])
        generated_count = int(new_ids.shape[0])
        finish_reason = (
            "eos" if generated_count and int(new_ids[-1]) in eos_ids
            else "length" if generated_count >= 704
            else "other"
        )
        _atomic_json(records / f"{index:04d}.json", {
            "question_id": question_id,
            "sample_index": index,
            "worker_rank": worker_rank,
            "worker_identity_sha256": identity["identity_sha256"],
            "variant": "e14_rank16_max704",
            "answer": answer,
            "max_new_tokens": 704,
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
            "e14_generation_progress worker=%d device=%s completed=%d total=%d "
            "global_completed_at_least=%d global_total=%d question_id=%s finish=%s",
            worker_rank,
            device,
            position + 1,
            len(assigned),
            min(len(ids), (position + 1) * 2),
            len(ids),
            question_id,
            finish_reason,
        )
    _write_worker_state(state_path, identity, len(assigned), len(assigned))
    return {"worker_rank": worker_rank, "device": device, "completed": len(assigned)}


def finalize_score(
    *, contexts_directory: Path, control_directory: Path,
    training_directory: Path, dev_path: Path, output_directory: Path,
    config: E14Dev200Config,
) -> dict[str, Any]:
    ensure_nltk_resources(download=False)
    dev, ids = _sample_ids(dev_path, config)
    controls = _control_rows(control_directory, dev_path, config)
    _context_rows(contexts_directory, dev_path, config)
    candidate_rows: list[dict[str, Any]] = []
    root = output_directory / "evaluation"
    for rank in range(2):
        state_path = root / f"worker-{rank}-state.json"
        if not state_path.is_file():
            raise E14Dev200Error(f"Missing E14 worker state: {rank}")
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if (
            state.get("complete") is not True
            or state.get("completed_count") != len(assigned_indices(config, rank))
        ):
            raise E14Dev200Error(f"Incomplete E14 worker: {rank}")
    for index, question_id in enumerate(ids):
        path = root / "records" / f"{index:04d}.json"
        if not path.is_file():
            raise E14Dev200Error(f"Missing E14 record: {index}")
        row = json.loads(path.read_text(encoding="utf-8"))
        if (
            row.get("question_id") != question_id
            or row.get("sample_index") != index
            or row.get("worker_rank") != index % 2
            or row.get("variant") != "e14_rank16_max704"
            or row.get("max_new_tokens") != 704
            or row.get("finish_reason") not in {"eos", "length", "other"}
            or not isinstance(row.get("answer"), str)
            or not row["answer"].strip()
        ):
            raise E14Dev200Error(f"Invalid E14 record: {index}")
        candidate_rows.append(row)
    _atomic_jsonl(root / "results.jsonl", candidate_rows)

    control_key = "e08b_rank8_max704_control"
    candidate_key = "e14_rank16_max704"
    scored = {control_key: [], candidate_key: []}
    merged: list[dict[str, Any]] = []
    source_control = config.section("source_control")
    for index, question_id in enumerate(ids):
        control = controls[index]["variants"][source_control["variant"]]
        candidate = candidate_rows[index]
        variants = {
            control_key: {
                **control,
                "answer_source": "reused-byte-verified-e08b-rank8-max704-control",
            },
            candidate_key: candidate,
        }
        merged.append({
            "question_id": question_id,
            "sample_index": index,
            "variants": variants,
        })
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
            "duplicate_sentence_rate": fmean(
                item["duplicate_sentence"] for item in diagnostics
            ),
            "abbreviation_loop_rate": fmean(
                item["abbreviation_loop"] for item in diagnostics
            ),
            "non_sentence_ending_rate": fmean(
                item["non_sentence_ending"] for item in diagnostics
            ),
        }
        if key == candidate_key:
            metric["mean_generation_latency_ms"] = fmean(
                row["generation_latency_ms"] for row in rows
            )
        metrics[key] = metric
    if (
        abs(metrics[control_key]["meteor"] - source_control["meteor"]) > 1e-12
        or abs(metrics[control_key]["rouge_l"] - source_control["rouge_l"]) > 1e-12
        or abs(
            metrics[control_key]["mean_output_tokens"]
            - source_control["mean_output_tokens"]
        ) > 1e-12
    ):
        raise E14Dev200Error("Re-scored E08B rank-8 max704 control changed.")
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
            meteor_deltas,
            seed=f"{scoring['bootstrap_seed']}:meteor",
            iterations=scoring["bootstrap_iterations"],
        ),
        "rouge_l_mean": fmean(rouge_deltas),
        "rouge_l_bootstrap_95_ci": _bootstrap_ci(
            rouge_deltas,
            seed=f"{scoring['bootstrap_seed']}:rouge_l",
            iterations=scoring["bootstrap_iterations"],
        ),
    }
    leader = max(
        (control_key, candidate_key),
        key=lambda key: (metrics[key]["meteor"], metrics[key]["rouge_l"]),
    )
    adapter_path = training_directory / config.section("source_training")["adapter_path"]
    report = {
        "schema_version": "1.0",
        "experiment_id": config.raw["experiment_id"],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "sample_size": len(ids),
        "control_variant": control_key,
        "candidate_variant": candidate_key,
        "only_changed_factor": (
            "fresh context-aware LoRA rank8/alpha16 versus rank16/alpha32; "
            "max_new_tokens remains 704"
        ),
        "metrics": metrics,
        "paired_delta_rank16_minus_rank8": paired,
        "smoke_leader": leader,
        "promotion_allowed": False,
        "evidence": {
            "config_sha256": config.config_sha256,
            "training_config_sha256": config.section("source_training")["config_sha256"],
            "contexts_sha256": config.section("source_contexts")["sha256"],
            "control_results_sha256": source_control["sha256"],
            "rank8_adapter_sha256": source_control["adapter_sha256"],
            "rank16_adapter_sha256": file_sha256(adapter_path),
            "sample_ids_sha256": config.section("dev")["sample_ids_sha256"],
            "results_sha256": file_sha256(results_path),
        },
        "warning": (
            "E14 is a dev-200 tuning smoke on a repeatedly used selection sample. "
            "Public and holdout are not read; this report cannot auto-promote."
        ),
    }
    _atomic_json(output_directory / "report.json", report)
    return report


__all__ = [
    "E14Dev200Config", "E14Dev200Error", "assigned_indices", "finalize_score",
    "load_config", "load_generator", "run_worker", "validate_preflight",
]
