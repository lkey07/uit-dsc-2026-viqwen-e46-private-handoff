"""E13 max-768 and conservative repeated-tail trimming on paired dev-200."""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean
from typing import Any, Callable

from uit_dsc_fixed_rag.corpus import file_sha256
from uit_dsc_fixed_rag.e02_answer import _atomic_json, _atomic_jsonl, pack_contexts
from uit_dsc_fixed_rag.e03_rrf_grid import (
    _bootstrap_ci,
    _json_sha256,
    _load_worker_progress,
    _read_jsonl,
    _write_worker_state,
)
from uit_dsc_fixed_rag.e08b_context_lora import load_context_lora_generator
from uit_dsc_fixed_rag.e08b_dev200_eval import (
    _adapter_evidence,
    _context_rows,
    _packing,
    _sample_ids,
    _training_config,
)
from uit_dsc_fixed_rag.e10_repetition_grid import answer_diagnostics
from uit_dsc_fixed_rag.evaluation.official import (
    ensure_nltk_resources,
    nltk_meteor_score,
    rouge_l_fmeasure,
)


CODE_VERSION = "0.41.0"
LOGGER = logging.getLogger(__name__)


class E13Error(RuntimeError):
    """Raised when E13 evidence or checkpoint identity changes."""


@dataclass(frozen=True)
class E13Config:
    raw: dict[str, Any]
    path: Path

    @property
    def config_sha256(self) -> str:
        return file_sha256(self.path)

    def section(self, key: str) -> dict[str, Any]:
        value = self.raw.get(key)
        if not isinstance(value, dict):
            raise ValueError(f"E13 section must be an object: {key}")
        return value


def load_config(path: Path) -> E13Config:
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "schema_version", "experiment_id", "source_contexts", "source_control",
        "source_training", "dev", "generator", "lora", "inference",
        "postprocess", "parameter_budget", "execution", "scoring", "run_contract",
    }
    if (
        set(payload) != required
        or payload.get("schema_version") != "1.0"
        or payload.get("experiment_id") != "E13-e08b-max768-tailtrim-dev200-v1"
    ):
        raise ValueError("E13 config root is incompatible.")
    config = E13Config(payload, path)
    contexts = config.section("source_contexts")
    if (
        contexts.get("experiment_id") != "E06-prompt-length-grid-aiteam-dev200-v1"
        or contexts.get("path") != "prepared/results.jsonl"
        or contexts.get("record_count") != 200
        or contexts.get("contexts_per_question") != 12
    ):
        raise ValueError("E13 context identity changed.")
    control = config.section("source_control")
    if (
        control.get("experiment_id")
        != "E08B-context-aware-lora-dev200-max704-paired-v1"
        or control.get("variant") != "e08b_context_lora_max704"
        or control.get("max_new_tokens") != 704
        or control.get("sample_size") != 200
    ):
        raise ValueError("E13 max704 control identity changed.")
    training = config.section("source_training")
    if (
        training.get("experiment_id")
        != "E08B-context-aware-lora-train5636-dev521-v3"
        or training.get("config_path") != "configs/e08b-context-aware-lora-v3.json"
        or training.get("fresh_from_base") is not True
        or training.get("train_sample_size") != 5636
        or training.get("adapter_sha256") != control.get("adapter_sha256")
    ):
        raise ValueError("E13 E08B adapter identity changed.")
    inference = config.section("inference")
    if (
        inference.get("max_input_tokens") != 8192
        or inference.get("minimum_contexts") != 1
        or inference.get("max_new_tokens") != 768
        or inference.get("do_sample") is not False
        or inference.get("num_beams") != 1
        or inference.get("enable_thinking") is not False
        or inference.get("use_cache") is not True
        or inference.get("repetition_penalty") != 1.0
        or inference.get("no_repeat_ngram_size") != 0
    ):
        raise ValueError("E13 deterministic inference identity changed.")
    post = config.section("postprocess")
    if post != {
        "raw_variant": "e08b_max768_raw",
        "candidate_variant": "e08b_max768_tailtrim",
        "method": "conservative-consecutive-tail-block-trim-v1",
        "line_block_sizes": [3, 2, 1],
        "sentence_block_sizes": [3, 2, 1],
        "normalization": "unicode-casefold-collapse-whitespace",
        "only_exact_consecutive_suffix": True,
        "never_rewrite_when_no_suffix_repeat": True,
    }:
        raise ValueError("E13 conservative tail-trim contract changed.")
    execution = config.section("execution")
    if execution != {
        "workers": 2,
        "partition": "sample-index-mod-worker-count",
        "questions_per_worker": 100,
        "checkpoint_after_questions": 1,
    }:
        raise ValueError("E13 execution identity changed.")
    budget = config.section("parameter_budget")
    if (
        budget.get("maximum_stack_total")
        != budget.get("embedding") + budget.get("generator")
        + budget.get("adapter_parameter_cap")
        or budget["maximum_stack_total"] >= budget["exclusive_limit"]
    ):
        raise ValueError("E13 parameter budget failed.")
    if config.section("scoring").get("promotion_allowed") is not False:
        raise ValueError("E13 cannot auto-promote.")
    contract = config.section("run_contract")
    if not contract or not all(value is True for value in contract.values()):
        raise ValueError("E13 run contract lost an invariant.")
    return config


def _control_rows(
    control_directory: Path, dev_path: Path, config: E13Config,
) -> list[dict[str, Any]]:
    section = config.section("source_control")
    path = control_directory / section["path"]
    if not path.is_file() or file_sha256(path) != section["sha256"]:
        raise E13Error("Saved E08B max704 control results changed.")
    _, ids = _sample_ids(dev_path, config)
    rows = _read_jsonl(path)
    if len(rows) != len(ids):
        raise E13Error("Saved E08B max704 control count changed.")
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
            raise E13Error(f"Invalid E08B max704 control row: {index}")
    report_path = control_directory / "report.json"
    if not report_path.is_file():
        raise E13Error("Saved E08B max704 control report is missing.")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    metric = report.get("metrics", {}).get(section["variant"], {})
    evidence = report.get("evidence", {})
    if (
        report.get("experiment_id") != section["experiment_id"]
        or report.get("sample_size") != section["sample_size"]
        or evidence.get("config_sha256") != section["config_sha256"]
        or evidence.get("results_sha256") != section["sha256"]
        or evidence.get("adapter_sha256") != section["adapter_sha256"]
        or metric.get("meteor") != section["meteor"]
        or metric.get("rouge_l") != section["rouge_l"]
        or metric.get("mean_output_tokens") != section["mean_output_tokens"]
        or metric.get("length_finish_rate") != section["length_finish_rate"]
    ):
        raise E13Error("Saved E08B max704 control report identity changed.")
    return rows


def validate_preflight(
    *, project_root: Path, contexts_directory: Path, control_directory: Path,
    training_directory: Path, dev_path: Path, config: E13Config,
) -> dict[str, Any]:
    scorer = project_root / config.section("scoring")["official_scorer_path"]
    if (
        not scorer.is_file()
        or file_sha256(scorer) != config.section("scoring")["official_scorer_sha256"]
    ):
        raise E13Error("Pinned official scorer changed.")
    training_config = _training_config(project_root, config)
    adapter = _adapter_evidence(training_directory, training_config, config)
    if adapter["adapter_sha256"] != config.section("source_training")["adapter_sha256"]:
        raise E13Error("Pinned E08B adapter hash changed.")
    contexts = _context_rows(contexts_directory, dev_path, config)
    controls = _control_rows(control_directory, dev_path, config)
    return {
        "code_version": CODE_VERSION,
        "config_sha256": config.config_sha256,
        "training_config_sha256": training_config.config_sha256,
        "adapter_sha256": adapter["adapter_sha256"],
        "contexts_sha256": config.section("source_contexts")["sha256"],
        "control_results_sha256": config.section("source_control")["sha256"],
        "sample_ids_sha256": config.section("dev")["sample_ids_sha256"],
        "sample_size": len(contexts),
        "control_size": len(controls),
        "max_new_tokens": 768,
    }


def load_generator(
    *, project_root: Path, training_directory: Path, config: E13Config,
    device: str,
) -> tuple[Any, Any, dict[str, Any], int]:
    training_config = _training_config(project_root, config)
    try:
        return load_context_lora_generator(
            config=training_config,
            training_directory=training_directory,
            device=device,
        )
    except Exception as exc:
        raise E13Error(str(exc)) from exc


def assigned_indices(config: E13Config, worker_rank: int) -> list[int]:
    workers = config.section("execution")["workers"]
    if worker_rank not in range(workers):
        raise E13Error("Invalid E13 worker rank.")
    return [
        index for index in range(config.section("dev")["sample_size"])
        if index % workers == worker_rank
    ]


def run_worker(
    *, worker_rank: int, device: str, model: Any, tokenizer: Any,
    project_root: Path, contexts_directory: Path, control_directory: Path,
    training_directory: Path, dev_path: Path, output_directory: Path,
    config: E13Config, device_map: dict[str, Any], adapter_parameters: int,
) -> dict[str, Any]:
    if device != f"cuda:{worker_rank}":
        raise E13Error("E13 worker-to-device mapping changed.")
    contexts = _context_rows(contexts_directory, dev_path, config)
    _control_rows(control_directory, dev_path, config)
    dev, ids = _sample_ids(dev_path, config)
    assigned = assigned_indices(config, worker_rank)
    adapter_path = training_directory / config.section("source_training")["adapter_path"]
    identity = {
        "code_version": CODE_VERSION,
        "stage": "e13-e08b-max768-dev200-worker",
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
            contexts=contexts[index]["contexts"],
            config=_packing(config), token_counter=token_count,
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
            max_new_tokens=768, use_cache=True,
            repetition_penalty=1.0, no_repeat_ngram_size=0,
        )
        latency_ms = (time.perf_counter() - started) * 1000
        new_ids = generated[0, inputs["input_ids"].shape[1]:]
        answer = tokenizer.decode(
            new_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False,
        ).strip()
        if not answer:
            raise E13Error(f"Empty E13 max768 answer: {question_id}")
        eos = tokenizer.eos_token_id
        eos_ids = set(eos if isinstance(eos, list) else [eos])
        generated_count = int(new_ids.shape[0])
        finish_reason = (
            "eos" if generated_count and int(new_ids[-1]) in eos_ids
            else "length" if generated_count >= 768
            else "other"
        )
        _atomic_json(records / f"{index:04d}.json", {
            "question_id": question_id,
            "sample_index": index,
            "worker_rank": worker_rank,
            "worker_identity_sha256": identity["identity_sha256"],
            "variant": "e08b_max768_raw",
            "answer": answer,
            "max_new_tokens": 768,
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
            "e13_generation_progress worker=%d device=%s completed=%d total=%d "
            "global_completed_at_least=%d global_total=%d question_id=%s finish=%s",
            worker_rank, device, position + 1, len(assigned),
            min(len(ids), (position + 1) * 2), len(ids), question_id, finish_reason,
        )
    _write_worker_state(state_path, identity, len(assigned), len(assigned))
    return {"worker_rank": worker_rank, "device": device, "completed": len(assigned)}


def _normalize_block(value: str) -> str:
    return " ".join(value.casefold().split())


def _line_spans(value: str) -> list[tuple[int, int]]:
    spans = []
    offset = 0
    for line in value.splitlines(keepends=True):
        end = offset + len(line)
        if _normalize_block(line):
            spans.append((offset, end))
        offset = end
    if offset < len(value) and _normalize_block(value[offset:]):
        spans.append((offset, len(value)))
    return spans


def _sentence_spans(value: str) -> list[tuple[int, int]]:
    spans = []
    pattern = re.compile(r"\S(?:.*?)(?:[.!?]+(?=\s|$)|$)", re.DOTALL)
    for match in pattern.finditer(value):
        if _normalize_block(match.group(0)):
            spans.append((match.start(), match.end()))
    return spans


def _trim_exact_suffix_blocks(
    value: str, span_factory: Callable[[str], list[tuple[int, int]]],
    sizes: list[int],
) -> tuple[str, int]:
    removed = 0
    while True:
        spans = span_factory(value)
        matched = False
        for size in sizes:
            if len(spans) < size * 2:
                continue
            previous_start = spans[-2 * size][0]
            repeated_start = spans[-size][0]
            end = spans[-1][1]
            previous = _normalize_block(value[previous_start:repeated_start])
            repeated = _normalize_block(value[repeated_start:end])
            if previous and previous == repeated:
                value = value[:repeated_start].rstrip()
                removed += 1
                matched = True
                break
        if not matched:
            return value, removed


def trim_repeated_tail(answer: str) -> dict[str, Any]:
    """Remove only exact, consecutive repeated line/sentence blocks at the suffix."""

    original = answer
    value, line_blocks = _trim_exact_suffix_blocks(
        original, _line_spans, [3, 2, 1]
    )
    value, sentence_blocks = _trim_exact_suffix_blocks(
        value, _sentence_spans, [3, 2, 1]
    )
    if not value.strip():
        raise E13Error("Tail trimming unexpectedly removed the complete answer.")
    return {
        "answer": value,
        "changed": value != original,
        "removed_characters": len(original) - len(value),
        "removed_line_blocks": line_blocks,
        "removed_sentence_blocks": sentence_blocks,
    }


def _variant_metrics(
    rows: list[dict[str, Any]], scores: list[dict[str, float]],
) -> dict[str, Any]:
    diagnostics = [answer_diagnostics(row["answer"]) for row in rows]
    metric = {
        "meteor": fmean(item["meteor"] for item in scores),
        "rouge_l": fmean(item["rouge_l"] for item in scores),
        "mean_output_tokens": fmean(row["output_tokens"] for row in rows),
        "mean_answer_characters": fmean(len(row["answer"]) for row in rows),
        "length_finish_rate": fmean(
            row.get("source_finish_reason", row.get("finish_reason")) == "length"
            for row in rows
        ),
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
    if all("generation_latency_ms" in row for row in rows):
        metric["mean_generation_latency_ms"] = fmean(
            row["generation_latency_ms"] for row in rows
        )
    if all("postprocess_changed" in row for row in rows):
        metric["postprocess_changed_rate"] = fmean(
            row["postprocess_changed"] for row in rows
        )
        metric["mean_removed_characters"] = fmean(
            row["removed_characters"] for row in rows
        )
    return metric


def _paired(
    candidate: list[dict[str, float]], control: list[dict[str, float]],
    *, seed: str, iterations: int,
) -> dict[str, Any]:
    meteor = [a["meteor"] - b["meteor"] for a, b in zip(candidate, control)]
    rouge = [a["rouge_l"] - b["rouge_l"] for a, b in zip(candidate, control)]
    return {
        "meteor_mean": fmean(meteor),
        "meteor_bootstrap_95_ci": _bootstrap_ci(
            meteor, seed=f"{seed}:meteor", iterations=iterations
        ),
        "rouge_l_mean": fmean(rouge),
        "rouge_l_bootstrap_95_ci": _bootstrap_ci(
            rouge, seed=f"{seed}:rouge_l", iterations=iterations
        ),
    }


def finalize_score(
    *, project_root: Path, contexts_directory: Path, control_directory: Path,
    training_directory: Path, dev_path: Path, output_directory: Path,
    config: E13Config,
) -> dict[str, Any]:
    ensure_nltk_resources(download=False)
    dev, ids = _sample_ids(dev_path, config)
    controls = _control_rows(control_directory, dev_path, config)
    _context_rows(contexts_directory, dev_path, config)
    root = output_directory / "evaluation"
    for rank in range(2):
        state_path = root / f"worker-{rank}-state.json"
        if not state_path.is_file():
            raise E13Error(f"Missing E13 worker state: {rank}")
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if (
            state.get("complete") is not True
            or state.get("completed_count") != len(assigned_indices(config, rank))
        ):
            raise E13Error(f"Incomplete E13 worker: {rank}")
    raw_rows = []
    for index, question_id in enumerate(ids):
        path = root / "records" / f"{index:04d}.json"
        if not path.is_file():
            raise E13Error(f"Missing E13 record: {index}")
        row = json.loads(path.read_text(encoding="utf-8"))
        if (
            row.get("question_id") != question_id
            or row.get("sample_index") != index
            or row.get("worker_rank") != index % 2
            or row.get("variant") != "e08b_max768_raw"
            or row.get("max_new_tokens") != 768
            or row.get("finish_reason") not in {"eos", "length", "other"}
            or not isinstance(row.get("answer"), str)
            or not row["answer"].strip()
        ):
            raise E13Error(f"Invalid E13 record: {index}")
        raw_rows.append(row)
    _atomic_jsonl(root / "results.jsonl", raw_rows)

    try:
        from transformers import AutoTokenizer
    except ImportError as exc:  # pragma: no cover
        raise E13Error("Transformers is required to score trimmed token counts.") from exc
    tokenizer = AutoTokenizer.from_pretrained(
        config.section("generator")["model_id"],
        revision=config.section("generator")["revision"],
        trust_remote_code=False,
    )
    control_key = "e08b_max704_control"
    raw_key = "e08b_max768_raw"
    trim_key = "e08b_max768_tailtrim"
    variant_rows = {control_key: [], raw_key: raw_rows, trim_key: []}
    for index, row in enumerate(controls):
        variant_rows[control_key].append(
            row["variants"][config.section("source_control")["variant"]]
        )
        trimmed = trim_repeated_tail(raw_rows[index]["answer"])
        variant_rows[trim_key].append({
            **raw_rows[index],
            "variant": trim_key,
            "answer": trimmed["answer"],
            "output_tokens": len(
                tokenizer(trimmed["answer"], add_special_tokens=False)["input_ids"]
            ),
            "source_finish_reason": raw_rows[index]["finish_reason"],
            "finish_reason": "postprocessed" if trimmed["changed"]
            else raw_rows[index]["finish_reason"],
            "postprocess_changed": trimmed["changed"],
            "removed_characters": trimmed["removed_characters"],
            "removed_line_blocks": trimmed["removed_line_blocks"],
            "removed_sentence_blocks": trimmed["removed_sentence_blocks"],
        })

    scores = {key: [] for key in variant_rows}
    merged = []
    for index, question_id in enumerate(ids):
        variants = {key: rows[index] for key, rows in variant_rows.items()}
        merged.append({"question_id": question_id, "sample_index": index, "variants": variants})
        reference = dev[question_id]["answer"]
        for key, row in variants.items():
            scores[key].append({
                "meteor": nltk_meteor_score(reference, row["answer"]),
                "rouge_l": rouge_l_fmeasure(reference, row["answer"]),
            })
    results_path = output_directory / "results.jsonl"
    _atomic_jsonl(results_path, merged)
    metrics = {
        key: _variant_metrics(variant_rows[key], scores[key])
        for key in variant_rows
    }
    source = config.section("source_control")
    if (
        abs(metrics[control_key]["meteor"] - source["meteor"]) > 1e-12
        or abs(metrics[control_key]["rouge_l"] - source["rouge_l"]) > 1e-12
        or abs(metrics[control_key]["mean_output_tokens"]
               - source["mean_output_tokens"]) > 1e-12
    ):
        raise E13Error("Re-scored E08B max704 control changed.")
    scoring = config.section("scoring")
    paired = {
        f"{raw_key}-minus-{control_key}": _paired(
            scores[raw_key], scores[control_key],
            seed=f"{scoring['bootstrap_seed']}:raw-v-control",
            iterations=scoring["bootstrap_iterations"],
        ),
        f"{trim_key}-minus-{control_key}": _paired(
            scores[trim_key], scores[control_key],
            seed=f"{scoring['bootstrap_seed']}:trim-v-control",
            iterations=scoring["bootstrap_iterations"],
        ),
        f"{trim_key}-minus-{raw_key}": _paired(
            scores[trim_key], scores[raw_key],
            seed=f"{scoring['bootstrap_seed']}:trim-v-raw",
            iterations=scoring["bootstrap_iterations"],
        ),
    }
    leader = max(
        variant_rows,
        key=lambda key: (metrics[key]["meteor"], metrics[key]["rouge_l"]),
    )
    adapter_path = training_directory / config.section("source_training")["adapter_path"]
    report = {
        "schema_version": "1.0",
        "experiment_id": config.raw["experiment_id"],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "sample_size": len(ids),
        "control_variant": control_key,
        "candidate_variants": [raw_key, trim_key],
        "only_changed_factors": {
            raw_key: "max_new_tokens 704 to 768",
            trim_key: "raw max768 plus conservative exact repeated-suffix trimming",
        },
        "metrics": metrics,
        "paired_deltas": paired,
        "smoke_leader": leader,
        "promotion_allowed": False,
        "evidence": {
            "config_sha256": config.config_sha256,
            "contexts_sha256": config.section("source_contexts")["sha256"],
            "control_results_sha256": config.section("source_control")["sha256"],
            "adapter_sha256": file_sha256(adapter_path),
            "sample_ids_sha256": config.section("dev")["sample_ids_sha256"],
            "raw_results_sha256": file_sha256(root / "results.jsonl"),
            "results_sha256": file_sha256(results_path),
        },
        "warning": (
            "E13 is a dev-200 tuning smoke. Public and holdout are not read; "
            "selection cannot auto-promote."
        ),
    }
    _atomic_json(output_directory / "report.json", report)
    return report


__all__ = [
    "E13Config", "E13Error", "assigned_indices", "finalize_score",
    "load_config", "load_generator", "run_worker", "trim_repeated_tail",
    "validate_preflight",
]
