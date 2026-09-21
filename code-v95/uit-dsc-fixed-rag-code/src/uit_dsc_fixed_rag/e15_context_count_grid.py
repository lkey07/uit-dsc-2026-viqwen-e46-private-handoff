"""E15 rank-16 top-8/top-10 context-count grid on paired dev-200."""

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
    _read_jsonl,
    _write_worker_state,
)
from uit_dsc_fixed_rag.e08b_dev200_eval import _context_rows, _packing, _sample_ids
from uit_dsc_fixed_rag.e10_repetition_grid import answer_diagnostics
from uit_dsc_fixed_rag.e14_rank16_dev200 import (
    _adapter_evidence,
    _training_config,
    load_generator as load_e14_generator,
)
from uit_dsc_fixed_rag.evaluation.official import (
    ensure_nltk_resources,
    nltk_meteor_score,
    rouge_l_fmeasure,
)


CODE_VERSION = "0.45.0"
LOGGER = logging.getLogger(__name__)


class E15Error(RuntimeError):
    """Raised when E15 evidence or checkpoint identity changes."""


@dataclass(frozen=True)
class E15Config:
    raw: dict[str, Any]
    path: Path

    @property
    def config_sha256(self) -> str:
        return file_sha256(self.path)

    def section(self, key: str) -> dict[str, Any]:
        value = self.raw.get(key)
        if not isinstance(value, dict):
            raise ValueError(f"E15 section must be an object: {key}")
        return value

    @property
    def variants(self) -> list[dict[str, Any]]:
        value = self.section("inference").get("variants")
        if not isinstance(value, list):
            raise ValueError("E15 variants must be a list.")
        return value


def load_config(path: Path) -> E15Config:
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
        != "E15-rank16-context-count-grid-dev200-max704-v1"
    ):
        raise ValueError("E15 config root is incompatible.")
    config = E15Config(payload, path)
    contexts = config.section("source_contexts")
    if (
        contexts.get("experiment_id") != "E06-prompt-length-grid-aiteam-dev200-v1"
        or contexts.get("path") != "prepared/results.jsonl"
        or contexts.get("record_count") != 200
        or contexts.get("contexts_per_question") != 12
    ):
        raise ValueError("E15 source context identity changed.")
    control = config.section("source_control")
    if (
        control.get("experiment_id")
        != "E14-rank16-vs-rank8-dev200-max704-v1"
        or control.get("path") != "results.jsonl"
        or control.get("variant") != "e14_rank16_max704"
        or control.get("context_limit") != 12
        or control.get("max_new_tokens") != 704
        or control.get("sample_size") != 200
        or control.get("adapter_sha256")
        != "e6ffd1c1f6d64b9cad8a603b09ba09e0274374d50f21e0a848f1e46f93ef27da"
    ):
        raise ValueError("E15 E14 top-12 control identity changed.")
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
        raise ValueError("E15 rank-16 training identity changed.")
    expected_variants = [
        {
            "key": "rank16_top8_max704", "context_limit": 8,
            "worker_rank": 0, "device": "cuda:0",
        },
        {
            "key": "rank16_top10_max704", "context_limit": 10,
            "worker_rank": 1, "device": "cuda:1",
        },
    ]
    inference = config.section("inference")
    if (
        inference.get("max_input_tokens") != 8192
        or inference.get("minimum_contexts") != 1
        or inference.get("control_context_limit") != 12
        or inference.get("max_new_tokens") != 704
        or inference.get("variants") != expected_variants
        or inference.get("do_sample") is not False
        or inference.get("num_beams") != 1
        or inference.get("enable_thinking") is not False
        or inference.get("use_cache") is not True
        or inference.get("repetition_penalty") != 1.0
        or inference.get("no_repeat_ngram_size") != 0
    ):
        raise ValueError("E15 context-count grid changed.")
    execution = config.section("execution")
    if execution != {
        "workers": 2,
        "partition": "one-context-limit-variant-per-worker-all-dev200-ids",
        "questions_per_worker": 200,
        "checkpoint_after_questions": 1,
    }:
        raise ValueError("E15 dual-variant execution changed.")
    budget = config.section("parameter_budget")
    if (
        budget.get("maximum_stack_total")
        != budget.get("embedding") + budget.get("generator")
        + budget.get("adapter_parameter_cap")
        or budget["maximum_stack_total"] >= budget["exclusive_limit"]
    ):
        raise ValueError("E15 parameter budget failed.")
    if config.section("scoring").get("promotion_allowed") is not False:
        raise ValueError("E15 dev-200 cannot auto-promote.")
    contract = config.section("run_contract")
    if not contract or not all(value is True for value in contract.values()):
        raise ValueError("E15 run contract lost an invariant.")
    return config


def variant_for_worker(config: E15Config, worker_rank: int) -> dict[str, Any]:
    matches = [item for item in config.variants if item["worker_rank"] == worker_rank]
    if len(matches) != 1:
        raise E15Error(f"Invalid E15 worker rank: {worker_rank}")
    return matches[0]


def assigned_indices(config: E15Config, worker_rank: int) -> list[int]:
    variant_for_worker(config, worker_rank)
    return list(range(config.section("dev")["sample_size"]))


def _control_rows(
    control_directory: Path, dev_path: Path, config: E15Config,
) -> list[dict[str, Any]]:
    section = config.section("source_control")
    path = control_directory / section["path"]
    if not path.is_file() or file_sha256(path) != section["sha256"]:
        raise E15Error("Saved E14 rank-16 top-12 results changed.")
    _, ids = _sample_ids(dev_path, config)
    rows = _read_jsonl(path)
    if len(rows) != len(ids):
        raise E15Error("Saved E14 rank-16 top-12 result count changed.")
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
            raise E15Error(f"Invalid E14 rank-16 top-12 control row: {index}")
    report_path = control_directory / "report.json"
    if not report_path.is_file():
        raise E15Error("Saved E14 report is missing.")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    metric = report.get("metrics", {}).get(section["variant"], {})
    evidence = report.get("evidence", {})
    if (
        report.get("experiment_id") != section["experiment_id"]
        or report.get("sample_size") != section["sample_size"]
        or evidence.get("config_sha256") != section["config_sha256"]
        or evidence.get("results_sha256") != section["sha256"]
        or evidence.get("rank16_adapter_sha256") != section["adapter_sha256"]
        or metric.get("meteor") != section["meteor"]
        or metric.get("rouge_l") != section["rouge_l"]
        or metric.get("mean_output_tokens") != section["mean_output_tokens"]
        or metric.get("length_finish_rate") != section["length_finish_rate"]
    ):
        raise E15Error("Saved E14 report identity changed.")
    return rows


def validate_preflight(
    *, project_root: Path, contexts_directory: Path, control_directory: Path,
    training_directory: Path, dev_path: Path, config: E15Config,
) -> dict[str, Any]:
    scorer = project_root / config.section("scoring")["official_scorer_path"]
    if (
        not scorer.is_file()
        or file_sha256(scorer)
        != config.section("scoring")["official_scorer_sha256"]
    ):
        raise E15Error("Pinned official scorer changed.")
    training = _training_config(project_root, config)
    adapter = _adapter_evidence(training_directory, training, config)
    if adapter["adapter_sha256"] != config.section("source_control")["adapter_sha256"]:
        raise E15Error("E15 control and generator adapter hashes differ.")
    contexts = _context_rows(contexts_directory, dev_path, config)
    controls = _control_rows(control_directory, dev_path, config)
    return {
        "code_version": CODE_VERSION,
        "config_sha256": config.config_sha256,
        "training_config_sha256": training.config_sha256,
        "adapter_sha256": adapter["adapter_sha256"],
        "adapter_trainable_parameters": adapter["trainable_parameters"],
        "contexts_sha256": config.section("source_contexts")["sha256"],
        "control_results_sha256": config.section("source_control")["sha256"],
        "sample_ids_sha256": config.section("dev")["sample_ids_sha256"],
        "sample_size": len(contexts),
        "control_size": len(controls),
        "variants": [item["key"] for item in config.variants],
        "context_limits": [item["context_limit"] for item in config.variants],
        "max_new_tokens": 704,
    }


def load_generator(
    *, project_root: Path, training_directory: Path,
    config: E15Config, device: str,
) -> tuple[Any, Any, dict[str, Any], int]:
    try:
        return load_e14_generator(
            project_root=project_root,
            training_directory=training_directory,
            config=config,
            device=device,
        )
    except Exception as exc:
        raise E15Error(str(exc)) from exc


def run_variant_worker(
    *, worker_rank: int, device: str, model: Any, tokenizer: Any,
    project_root: Path, contexts_directory: Path, control_directory: Path,
    training_directory: Path, dev_path: Path, output_directory: Path,
    config: E15Config, device_map: dict[str, Any], adapter_parameters: int,
) -> dict[str, Any]:
    variant = variant_for_worker(config, worker_rank)
    if device != variant["device"]:
        raise E15Error("E15 worker-to-device mapping changed.")
    contexts = _context_rows(contexts_directory, dev_path, config)
    _control_rows(control_directory, dev_path, config)
    dev, ids = _sample_ids(dev_path, config)
    assigned = assigned_indices(config, worker_rank)
    adapter_path = training_directory / config.section("source_training")["adapter_path"]
    identity = {
        "code_version": CODE_VERSION,
        "stage": "e15-rank16-context-count-grid-dev200-worker",
        "config_sha256": config.config_sha256,
        "adapter_sha256": file_sha256(adapter_path),
        "adapter_parameters": adapter_parameters,
        "lora_rank": 16,
        "lora_alpha": 32,
        "variant": variant["key"],
        "context_limit": variant["context_limit"],
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
    root = output_directory / "evaluation" / variant["key"]
    records = root / "records"
    records.mkdir(parents=True, exist_ok=True)
    state_path = root / "state.json"
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
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
        return len(tokenizer(rendered, add_special_tokens=False)["input_ids"])

    for position in range(completed, len(assigned)):
        index = assigned[position]
        question_id = ids[index]
        candidates = contexts[index]["contexts"][:variant["context_limit"]]
        if len(candidates) != variant["context_limit"]:
            raise E15Error(f"Context prefix is incomplete: {question_id}")
        selected, messages, input_tokens = pack_contexts(
            question=dev[question_id]["question"],
            contexts=candidates,
            config=_packing(config),
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
            raise E15Error(f"Empty E15 answer: {question_id}")
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
            "variant": variant["key"],
            "context_limit": variant["context_limit"],
            "answer": answer,
            "max_new_tokens": 704,
            "candidate_chunk_ids": [item["chunk_id"] for item in candidates],
            "selected_chunk_ids": [item["chunk_id"] for item in selected],
            "selected_context_count": len(selected),
            "input_tokens": input_tokens,
            "output_tokens": len(
                tokenizer(answer, add_special_tokens=False)["input_ids"]
            ),
            "generated_tokens_including_special": generated_count,
            "finish_reason": finish_reason,
            "generation_latency_ms": latency_ms,
        })
        _write_worker_state(state_path, identity, position + 1, len(assigned))
        LOGGER.info(
            "e15_generation_progress variant=%s device=%s completed=%d total=%d "
            "question_id=%s selected_contexts=%d finish=%s",
            variant["key"], device, position + 1, len(assigned), question_id,
            len(selected), finish_reason,
        )
    _write_worker_state(state_path, identity, len(assigned), len(assigned))
    return {
        "worker_rank": worker_rank,
        "device": device,
        "variant": variant["key"],
        "context_limit": variant["context_limit"],
        "completed": len(assigned),
    }


def _metrics(
    rows: list[dict[str, Any]], scores: list[dict[str, float]],
    *, include_latency: bool,
) -> dict[str, Any]:
    diagnostics = [answer_diagnostics(row["answer"]) for row in rows]
    result = {
        "meteor": fmean(item["meteor"] for item in scores),
        "rouge_l": fmean(item["rouge_l"] for item in scores),
        "mean_output_tokens": fmean(row["output_tokens"] for row in rows),
        "mean_answer_characters": fmean(len(row["answer"]) for row in rows),
        "mean_selected_contexts": fmean(
            row.get("selected_context_count", 12) for row in rows
        ),
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
    if include_latency:
        result["mean_generation_latency_ms"] = fmean(
            row["generation_latency_ms"] for row in rows
        )
    return result


def _paired_delta(
    candidate: list[dict[str, float]], control: list[dict[str, float]],
    *, config: E15Config, label: str,
) -> dict[str, Any]:
    meteor = [a["meteor"] - b["meteor"] for a, b in zip(candidate, control)]
    rouge = [a["rouge_l"] - b["rouge_l"] for a, b in zip(candidate, control)]
    scoring = config.section("scoring")
    return {
        "meteor_mean": fmean(meteor),
        "meteor_bootstrap_95_ci": _bootstrap_ci(
            meteor,
            seed=f"{scoring['bootstrap_seed']}:{label}:meteor",
            iterations=scoring["bootstrap_iterations"],
        ),
        "rouge_l_mean": fmean(rouge),
        "rouge_l_bootstrap_95_ci": _bootstrap_ci(
            rouge,
            seed=f"{scoring['bootstrap_seed']}:{label}:rouge_l",
            iterations=scoring["bootstrap_iterations"],
        ),
    }


def finalize_score(
    *, contexts_directory: Path, control_directory: Path,
    training_directory: Path, dev_path: Path, output_directory: Path,
    config: E15Config,
) -> dict[str, Any]:
    ensure_nltk_resources(download=False)
    dev, ids = _sample_ids(dev_path, config)
    contexts = _context_rows(contexts_directory, dev_path, config)
    controls = _control_rows(control_directory, dev_path, config)
    control_key = "e14_rank16_top12_max704_control"
    keys = [control_key, *(item["key"] for item in config.variants)]
    rows_by_key: dict[str, list[dict[str, Any]]] = {key: [] for key in keys}
    source_control = config.section("source_control")
    for index, question_id in enumerate(ids):
        control = dict(controls[index]["variants"][source_control["variant"]])
        control["context_limit"] = 12
        control["answer_source"] = "reused-byte-verified-e14-rank16-top12-control"
        rows_by_key[control_key].append(control)
    for variant in config.variants:
        root = output_directory / "evaluation" / variant["key"]
        state_path = root / "state.json"
        if not state_path.is_file():
            raise E15Error(f"Missing E15 state: {variant['key']}")
        state = json.loads(state_path.read_text(encoding="utf-8"))
        state_identity = state.get("run_identity", {})
        if (
            state.get("complete") is not True
            or state.get("completed_count") != 200
            or state.get("assigned_count") != 200
            or state_identity.get("config_sha256") != config.config_sha256
            or state_identity.get("adapter_sha256")
            != config.section("source_control")["adapter_sha256"]
            or state_identity.get("variant") != variant["key"]
            or state_identity.get("context_limit") != variant["context_limit"]
            or state_identity.get("max_new_tokens") != 704
        ):
            raise E15Error(f"Incomplete E15 worker: {variant['key']}")
        for index, question_id in enumerate(ids):
            path = root / "records" / f"{index:04d}.json"
            if not path.is_file():
                raise E15Error(f"Missing E15 record: {variant['key']}:{index}")
            row = json.loads(path.read_text(encoding="utf-8"))
            expected_prefix = [
                item["chunk_id"]
                for item in contexts[index]["contexts"][:variant["context_limit"]]
            ]
            if (
                row.get("question_id") != question_id
                or row.get("sample_index") != index
                or row.get("worker_rank") != variant["worker_rank"]
                or row.get("variant") != variant["key"]
                or row.get("context_limit") != variant["context_limit"]
                or row.get("max_new_tokens") != 704
                or row.get("candidate_chunk_ids") != expected_prefix
                or row.get("finish_reason") not in {"eos", "length", "other"}
                or not isinstance(row.get("answer"), str)
                or not row["answer"].strip()
            ):
                raise E15Error(f"Invalid E15 record: {variant['key']}:{index}")
            rows_by_key[variant["key"]].append(row)
        _atomic_jsonl(root / "results.jsonl", rows_by_key[variant["key"]])

    scored: dict[str, list[dict[str, float]]] = {key: [] for key in keys}
    merged: list[dict[str, Any]] = []
    for index, question_id in enumerate(ids):
        variants = {key: rows_by_key[key][index] for key in keys}
        merged.append({
            "question_id": question_id,
            "sample_index": index,
            "variants": variants,
        })
        reference = dev[question_id]["answer"]
        for key in keys:
            scored[key].append({
                "meteor": nltk_meteor_score(reference, variants[key]["answer"]),
                "rouge_l": rouge_l_fmeasure(reference, variants[key]["answer"]),
            })
    results_path = output_directory / "results.jsonl"
    _atomic_jsonl(results_path, merged)
    metrics = {
        key: _metrics(
            rows_by_key[key], scored[key], include_latency=(key != control_key)
        )
        for key in keys
    }
    if (
        abs(metrics[control_key]["meteor"] - source_control["meteor"]) > 1e-12
        or abs(metrics[control_key]["rouge_l"] - source_control["rouge_l"]) > 1e-12
        or abs(
            metrics[control_key]["mean_output_tokens"]
            - source_control["mean_output_tokens"]
        ) > 1e-12
        or abs(
            metrics[control_key]["length_finish_rate"]
            - source_control["length_finish_rate"]
        ) > 1e-12
    ):
        raise E15Error("Re-scored E14 top-12 control changed.")
    paired_vs_control = {
        key: _paired_delta(
            scored[key], scored[control_key], config=config, label=f"{key}-minus-top12"
        )
        for key in keys[1:]
    }
    paired_top10_minus_top8 = _paired_delta(
        scored["rank16_top10_max704"],
        scored["rank16_top8_max704"],
        config=config,
        label="top10-minus-top8",
    )
    leader = max(keys, key=lambda key: (metrics[key]["meteor"], metrics[key]["rouge_l"]))
    adapter_path = training_directory / config.section("source_training")["adapter_path"]
    report = {
        "schema_version": "1.0",
        "experiment_id": config.raw["experiment_id"],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "sample_size": len(ids),
        "control_variant": control_key,
        "candidate_variants": keys[1:],
        "only_changed_factor": "ranked context prefix limit: top12 control vs top8/top10",
        "metrics": metrics,
        "paired_deltas_vs_top12_control": paired_vs_control,
        "paired_delta_top10_minus_top8": paired_top10_minus_top8,
        "smoke_leader": leader,
        "promotion_allowed": False,
        "evidence": {
            "config_sha256": config.config_sha256,
            "training_config_sha256": config.section("source_training")["config_sha256"],
            "contexts_sha256": config.section("source_contexts")["sha256"],
            "control_results_sha256": source_control["sha256"],
            "rank16_adapter_sha256": file_sha256(adapter_path),
            "sample_ids_sha256": config.section("dev")["sample_ids_sha256"],
            "results_sha256": file_sha256(results_path),
        },
        "warning": (
            "E15 is a dev-200 tuning grid on a repeatedly used selection sample. "
            "Public and holdout are not read; this report cannot auto-promote."
        ),
    }
    _atomic_json(output_directory / "report.json", report)
    return report


__all__ = [
    "E15Config", "E15Error", "assigned_indices", "finalize_score",
    "load_config", "load_generator", "run_variant_worker",
    "validate_preflight", "variant_for_worker",
]
