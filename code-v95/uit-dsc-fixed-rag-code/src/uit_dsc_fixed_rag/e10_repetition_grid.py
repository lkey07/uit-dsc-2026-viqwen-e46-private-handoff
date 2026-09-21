"""E10 deterministic anti-repetition decoding grid on saved E09 max640."""

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
from uit_dsc_fixed_rag.e09_output_length_grid import (
    inference_packing,
    load_generator as load_e09_generator,
    validate_preflight as validate_e09_sources,
)
from uit_dsc_fixed_rag.evaluation.official import (
    ensure_nltk_resources,
    nltk_meteor_score,
    rouge_l_fmeasure,
)
from uit_dsc_fixed_rag.retrieval_diagnostics import select_dev_sample


CODE_VERSION = "0.26.0"
LOGGER = logging.getLogger(__name__)


class E10Error(RuntimeError):
    """Raised when E10 loses its frozen experiment identity."""


@dataclass(frozen=True)
class E10Config:
    raw: dict[str, Any]
    path: Path

    @property
    def config_sha256(self) -> str:
        return file_sha256(self.path)

    def section(self, key: str) -> dict[str, Any]:
        value = self.raw.get(key)
        if not isinstance(value, dict):
            raise ValueError(f"E10 section must be an object: {key}")
        return value

    @property
    def variants(self) -> list[dict[str, Any]]:
        value = self.section("inference").get("variants")
        if not isinstance(value, list):
            raise ValueError("E10 variants must be a list.")
        return value


def load_e10_config(path: Path) -> E10Config:
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "schema_version", "experiment_id", "source_contexts", "source_control",
        "source_max640", "dev", "generator", "lora", "inference",
        "parameter_budget", "execution", "scoring", "run_contract",
    }
    if (
        set(payload) != required
        or payload.get("schema_version") != "1.0"
        or payload.get("experiment_id")
        != "E10-repetition-decoding-grid-qwen35-lora-dev200-v1"
    ):
        raise ValueError("E10 config root is incompatible.")
    config = E10Config(payload, path)
    source = config.section("source_max640")
    if (
        source.get("experiment_id")
        != "E09-output-length-grid-qwen35-lora-dev200-v2"
        or source.get("path") != "results.jsonl"
        or source.get("sha256")
        != "e7e7f319bee4ac10e2aeffea70fffa2e5c4b728e5cc275d84e76b166438e1fa8"
        or source.get("variant") != "max640"
        or source.get("max_new_tokens") != 640
        or source.get("sample_size") != 200
    ):
        raise ValueError("E10 max640 control identity changed.")
    expected_variants = [
        {
            "key": "rep_penalty_105", "max_new_tokens": 640,
            "repetition_penalty": 1.05, "no_repeat_ngram_size": 0,
            "worker_rank": 0, "device": "cuda:0",
        },
        {
            "key": "no_repeat_ngram8", "max_new_tokens": 640,
            "repetition_penalty": 1.0, "no_repeat_ngram_size": 8,
            "worker_rank": 1, "device": "cuda:1",
        },
    ]
    inference = config.section("inference")
    if (
        inference.get("max_input_tokens") != 8192
        or inference.get("minimum_contexts") != 1
        or inference.get("control_max_new_tokens") != 640
        or inference.get("variants") != expected_variants
        or inference.get("do_sample") is not False
        or inference.get("num_beams") != 1
        or inference.get("enable_thinking") is not False
        or inference.get("use_cache") is not True
    ):
        raise ValueError("E10 inference grid changed.")
    generator = config.section("generator")
    if (
        generator.get("model_id") != "Qwen/Qwen3.5-2B"
        or generator.get("revision")
        != "15852e8c16360a2fea060d615a32b45270f8a8fc"
        or generator.get("published_parameter_count") != 2_274_069_824
        or generator.get("accepted_runtime_unique_parameter_counts")
        != [1_881_825_088, 2_213_241_664]
    ):
        raise ValueError("E10 generator identity changed.")
    execution = config.section("execution")
    if execution.get("workers") != 2 or execution.get("checkpoint_after_questions") != 1:
        raise ValueError("E10 dual-GPU execution changed.")
    budget = config.section("parameter_budget")
    if (
        budget.get("maximum_stack_total")
        != budget.get("embedding") + budget.get("generator")
        + budget.get("adapter_parameter_cap")
        or budget["maximum_stack_total"] >= budget["exclusive_limit"]
    ):
        raise ValueError("E10 parameter budget failed.")
    contract = config.section("run_contract")
    if not contract or not all(value is True for value in contract.values()):
        raise ValueError("E10 run contract lost an invariant.")
    if config.section("scoring").get("promotion_allowed") is not False:
        raise ValueError("E10 dev-200 grid cannot auto-promote.")
    return config


def _sample_ids(dev_path: Path, config: E10Config) -> tuple[dict[str, Any], list[str]]:
    dev_cfg = config.section("dev")
    if not dev_path.is_file() or file_sha256(dev_path) != dev_cfg["sha256"]:
        raise E10Error("Pinned dev split is missing or changed.")
    dev = _load_dev(dev_path)
    ids = select_dev_sample(
        dev, seed=dev_cfg["sample_seed"], size=dev_cfg["sample_size"]
    )
    identity = hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()
    if identity != dev_cfg["sample_ids_sha256"]:
        raise E10Error("E10 deterministic dev-200 IDs changed.")
    return dev, ids


def _max640_rows(
    *, max640_directory: Path, dev_path: Path, config: E10Config,
) -> list[dict[str, Any]]:
    source = config.section("source_max640")
    path = max640_directory / source["path"]
    if not path.is_file() or file_sha256(path) != source["sha256"]:
        raise E10Error("Saved E09 max640 results are missing or changed.")
    _, ids = _sample_ids(dev_path, config)
    rows = _read_jsonl(path)
    if len(rows) != len(ids):
        raise E10Error("Saved E09 max640 row count changed.")
    for index, (question_id, row) in enumerate(zip(ids, rows)):
        control = row.get("variants", {}).get(source["variant"], {})
        if (
            row.get("question_id") != question_id
            or row.get("sample_index") != index
            or not isinstance(control.get("answer"), str)
            or not control["answer"].strip()
            or control.get("max_new_tokens") != 640
            or not isinstance(control.get("output_tokens"), int)
            or control.get("finish_reason") not in {"eos", "length", "other"}
        ):
            raise E10Error(f"Invalid saved E09 max640 row: {index}")
    return rows


def validate_preflight(
    *, project_root: Path, contexts_directory: Path, control_directory: Path,
    max640_directory: Path, dev_path: Path, config: E10Config,
) -> dict[str, Any]:
    try:
        evidence = validate_e09_sources(
            project_root=project_root,
            contexts_directory=contexts_directory,
            control_directory=control_directory,
            dev_path=dev_path,
            config=config,
        )
    except Exception as exc:
        raise E10Error(str(exc)) from exc
    rows = _max640_rows(
        max640_directory=max640_directory, dev_path=dev_path, config=config
    )
    report_path = max640_directory / "report.json"
    if not report_path.is_file():
        raise E10Error("Saved E09 report is missing.")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    source = config.section("source_max640")
    metric = report.get("metrics", {}).get(source["variant"], {})
    if (
        report.get("experiment_id") != source["experiment_id"]
        or report.get("sample_size") != source["sample_size"]
        or report.get("promotion_allowed") is not False
        or report.get("evidence", {}).get("config_sha256") != source["config_sha256"]
        or report.get("evidence", {}).get("results_sha256") != source["sha256"]
        or metric.get("meteor") != source["meteor"]
        or metric.get("length_finish_rate") != source["length_finish_rate"]
    ):
        raise E10Error("Saved E09 max640 report changed.")
    evidence.update({
        "max640_results_sha256": file_sha256(max640_directory / source["path"]),
        "max640_report_sha256": file_sha256(report_path),
        "max640_row_count": len(rows),
    })
    return evidence


def load_generator(
    *, config: E10Config, control_directory: Path, device: str,
) -> tuple[Any, Any, dict[str, Any], int, int]:
    try:
        return load_e09_generator(
            config=config, control_directory=control_directory, device=device
        )
    except Exception as exc:
        raise E10Error(str(exc)) from exc


def _variant_for_worker(
    config: E10Config, worker_rank: int, device: str,
) -> dict[str, Any]:
    matches = [
        item for item in config.variants
        if item["worker_rank"] == worker_rank and item["device"] == device
    ]
    if len(matches) != 1:
        raise E10Error("E10 worker-to-policy mapping changed.")
    return matches[0]


def generation_kwargs(variant: dict[str, Any]) -> dict[str, Any]:
    return {
        "do_sample": False,
        "num_beams": 1,
        "max_new_tokens": variant["max_new_tokens"],
        "use_cache": True,
        "repetition_penalty": variant["repetition_penalty"],
        "no_repeat_ngram_size": variant["no_repeat_ngram_size"],
    }


def run_variant_worker(
    *, worker_rank: int, device: str, model: Any, tokenizer: Any,
    contexts_directory: Path, control_directory: Path, max640_directory: Path,
    dev_path: Path, output_directory: Path, config: E10Config,
    device_map: dict[str, Any], adapter_parameters: int,
    runtime_unique_parameter_count: int,
) -> dict[str, Any]:
    variant = _variant_for_worker(config, worker_rank, device)
    contexts_cfg = config.section("source_contexts")
    contexts_path = contexts_directory / contexts_cfg["path"]
    if file_sha256(contexts_path) != contexts_cfg["sha256"]:
        raise E10Error("E10 frozen contexts changed after preflight.")
    _max640_rows(
        max640_directory=max640_directory, dev_path=dev_path, config=config
    )
    contexts = _read_jsonl(contexts_path)
    dev, ids = _sample_ids(dev_path, config)
    assigned = list(range(len(ids)))
    adapter_path = control_directory / config.section("source_control")["adapter_path"]
    experiment_label = str(config.raw["experiment_id"]).split("-", 1)[0].lower()
    identity = {
        "code_version": getattr(config, "code_version", CODE_VERSION),
        "stage": f"{experiment_label}-one-variant-per-gpu",
        "config_sha256": config.config_sha256,
        "contexts_sha256": contexts_cfg["sha256"],
        "max640_results_sha256": config.section("source_max640")["sha256"],
        "adapter_sha256": file_sha256(adapter_path),
        "adapter_parameters": adapter_parameters,
        "runtime_unique_parameter_count": runtime_unique_parameter_count,
        "variant": variant,
        "device_map": device_map,
        "sample_ids_sha256": config.section("dev")["sample_ids_sha256"],
    }
    identity["identity_sha256"] = _json_sha256(identity)
    root = output_directory / "generation" / variant["key"]
    records = root / "records"
    records.mkdir(parents=True, exist_ok=True)
    state_path = root / "state.json"
    completed = _load_worker_progress(
        records=records, state_path=state_path, identity=identity,
        assigned_indices=assigned, sample_ids=ids,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    packing = inference_packing(config)

    def token_count(messages: list[dict[str, Any]]) -> int:
        rendered = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
        return len(tokenizer(rendered, add_special_tokens=False)["input_ids"])

    for position in range(completed, len(assigned)):
        index = assigned[position]
        question_id = ids[index]
        selected, messages, input_tokens = pack_contexts(
            question=dev[question_id]["question"],
            contexts=contexts[index]["contexts"], config=packing,
            token_counter=token_count,
        )
        rendered = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = tokenizer(rendered, add_special_tokens=False, return_tensors="pt")
        inputs = {key: value.to(device) for key, value in inputs.items()}
        started = time.perf_counter()
        generated = model.generate(**inputs, **generation_kwargs(variant))
        latency_ms = (time.perf_counter() - started) * 1000
        new_ids = generated[0, inputs["input_ids"].shape[1]:]
        answer = tokenizer.decode(
            new_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        ).strip()
        if not answer:
            raise E10Error(f"Empty {experiment_label.upper()} answer: {question_id}")
        eos = tokenizer.eos_token_id
        eos_ids = set(eos if isinstance(eos, list) else [eos])
        generated_count = int(new_ids.shape[0])
        ended_by_eos = bool(generated_count and int(new_ids[-1]) in eos_ids)
        finish_reason = (
            "eos" if ended_by_eos
            else "length" if generated_count >= variant["max_new_tokens"]
            else "other"
        )
        _atomic_json(records / f"{index:04d}.json", {
            "question_id": question_id,
            "sample_index": index,
            "worker_rank": worker_rank,
            "worker_identity_sha256": identity["identity_sha256"],
            "variant": variant["key"],
            "max_new_tokens": variant["max_new_tokens"],
            "repetition_penalty": variant["repetition_penalty"],
            "no_repeat_ngram_size": variant["no_repeat_ngram_size"],
            "runtime_unique_parameter_count": runtime_unique_parameter_count,
            "answer": answer,
            "selected_chunk_ids": [row["chunk_id"] for row in selected],
            "selected_context_count": len(selected),
            "input_tokens": input_tokens,
            "output_tokens": len(tokenizer(answer, add_special_tokens=False)["input_ids"]),
            "generated_tokens_including_special": generated_count,
            "finish_reason": finish_reason,
            "generation_latency_ms": latency_ms,
        })
        _write_worker_state(state_path, identity, position + 1, len(assigned))
        LOGGER.info(
            "%s_generation_progress variant=%s device=%s completed=%d total=%d "
            "question_id=%s finish=%s",
            experiment_label, variant["key"], device, position + 1, len(assigned), question_id,
            finish_reason,
        )
    _write_worker_state(state_path, identity, len(assigned), len(assigned))
    return {"variant": variant["key"], "completed": len(assigned)}


def answer_diagnostics(answer: str) -> dict[str, bool]:
    lines = [line.strip() for line in re.split(r"\r?\n", answer) if len(line.strip()) >= 8]
    sentences = [
        item.strip() for item in re.split(r"(?<=[.!?])\s+", answer)
        if len(item.strip()) >= 20
    ]
    return {
        "duplicate_line": len(lines) != len(set(lines)),
        "duplicate_sentence": len(sentences) != len(set(sentences)),
        "abbreviation_loop": bool(re.search(r"(?:-[A-ZĐ]{2,10}){10,}", answer)),
        "non_sentence_ending": not bool(re.search(r"[.!?;:]$", answer.strip())),
    }


def finalize_score(
    *, contexts_directory: Path, control_directory: Path,
    max640_directory: Path, dev_path: Path, output_directory: Path,
    config: E10Config,
) -> dict[str, Any]:
    ensure_nltk_resources(download=False)
    dev, ids = _sample_ids(dev_path, config)
    max640_rows = _max640_rows(
        max640_directory=max640_directory, dev_path=dev_path, config=config
    )
    generated_by_key: dict[str, list[dict[str, Any]]] = {}
    for variant in config.variants:
        root = output_directory / "generation" / variant["key"]
        state_path = root / "state.json"
        if not state_path.is_file():
            raise E10Error(f"Missing E10 worker state: {variant['key']}")
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("complete") is not True or state.get("completed_count") != len(ids):
            raise E10Error(f"Incomplete E10 worker: {variant['key']}")
        rows = []
        for index, question_id in enumerate(ids):
            path = root / "records" / f"{index:04d}.json"
            if not path.is_file():
                raise E10Error(f"Missing E10 record: {variant['key']}:{index}")
            row = json.loads(path.read_text(encoding="utf-8"))
            if (
                row.get("question_id") != question_id
                or row.get("sample_index") != index
                or row.get("variant") != variant["key"]
                or row.get("max_new_tokens") != variant["max_new_tokens"]
                or row.get("repetition_penalty") != variant["repetition_penalty"]
                or row.get("no_repeat_ngram_size") != variant["no_repeat_ngram_size"]
                or row.get("finish_reason") not in {"eos", "length", "other"}
            ):
                raise E10Error(f"Invalid E10 record: {variant['key']}:{index}")
            rows.append(row)
        _atomic_jsonl(root / "results.jsonl", rows)
        generated_by_key[variant["key"]] = rows

    control_key = "max640_control"
    keys = [control_key, *[item["key"] for item in config.variants]]
    scored: dict[str, list[dict[str, Any]]] = {key: [] for key in keys}
    merged = []
    for index, question_id in enumerate(ids):
        control = max640_rows[index]["variants"][config.section("source_max640")["variant"]]
        variants = {
            control_key: {
                **control,
                "answer_source": "reused-byte-verified-e09-max640-control",
            }
        }
        for variant in config.variants:
            variants[variant["key"]] = generated_by_key[variant["key"]][index]
        merged.append({"question_id": question_id, "sample_index": index, "variants": variants})
        reference = dev[question_id]["answer"]
        for key in keys:
            answer = variants[key]["answer"]
            scored[key].append({
                "question_id": question_id,
                "meteor": nltk_meteor_score(reference, answer),
                "rouge_l": rouge_l_fmeasure(reference, answer),
            })
    results_path = output_directory / "results.jsonl"
    _atomic_jsonl(results_path, merged)
    metrics: dict[str, dict[str, Any]] = {}
    for key in keys:
        rows = [row["variants"][key] for row in merged]
        diagnostics = [answer_diagnostics(row["answer"]) for row in rows]
        metric = {
            "meteor": fmean(row["meteor"] for row in scored[key]),
            "rouge_l": fmean(row["rouge_l"] for row in scored[key]),
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
        if key != control_key:
            metric["mean_generation_latency_ms"] = fmean(
                row["generation_latency_ms"] for row in rows
            )
        metrics[key] = metric
    source = config.section("source_max640")
    if (
        abs(metrics[control_key]["meteor"] - source["meteor"]) > 1e-12
        or metrics[control_key]["length_finish_rate"] != source["length_finish_rate"]
    ):
        raise E10Error("Re-scored E09 max640 control changed.")

    scoring = config.section("scoring")
    paired: dict[str, Any] = {}
    for variant in config.variants:
        key = variant["key"]
        meteor = [
            candidate["meteor"] - control["meteor"]
            for candidate, control in zip(scored[key], scored[control_key])
        ]
        rouge = [
            candidate["rouge_l"] - control["rouge_l"]
            for candidate, control in zip(scored[key], scored[control_key])
        ]
        paired[f"{key}-minus-{control_key}"] = {
            "meteor_mean": fmean(meteor),
            "meteor_bootstrap_95_ci": _bootstrap_ci(
                meteor, seed=f"{scoring['bootstrap_seed']}:{key}:meteor",
                iterations=scoring["bootstrap_iterations"],
            ),
            "rouge_l_mean": fmean(rouge),
            "rouge_l_bootstrap_95_ci": _bootstrap_ci(
                rouge, seed=f"{scoring['bootstrap_seed']}:{key}:rouge_l",
                iterations=scoring["bootstrap_iterations"],
            ),
        }
    leader = max(keys, key=lambda key: (metrics[key]["meteor"], metrics[key]["rouge_l"]))
    is_e10 = str(config.raw["experiment_id"]).startswith("E10-")
    report = {
        "schema_version": "1.0",
        "experiment_id": config.raw["experiment_id"],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "sample_size": len(ids),
        "control_variant": control_key,
        "only_changed_factor": (
            "anti-repetition decoding policy at max_new_tokens=640"
            if is_e10 else "max_new_tokens only: 640 control versus 704 and 768"
        ),
        "metrics": metrics,
        "paired_deltas_vs_max640_control": paired,
        "smoke_leader": leader,
        "promotion_allowed": False,
        "evidence": {
            "config_sha256": config.config_sha256,
            "contexts_sha256": file_sha256(
                contexts_directory / config.section("source_contexts")["path"]
            ),
            "max640_results_sha256": file_sha256(
                max640_directory / config.section("source_max640")["path"]
            ),
            "adapter_sha256": file_sha256(
                control_directory / config.section("source_control")["adapter_path"]
            ),
            "results_sha256": file_sha256(results_path),
        },
        "warning": (
            "E10 is a dev-200 anti-repetition smoke grid. It does not read public or "
            "holdout and cannot automatically promote a competition stack."
            if is_e10 else
            "E11 is a dev-200 output-length refinement. It does not read public or "
            "holdout and cannot automatically promote a competition stack."
        ),
    }
    _atomic_json(output_directory / "report.json", report)
    return report


__all__ = [
    "E10Config", "E10Error", "answer_diagnostics", "finalize_score",
    "generation_kwargs", "load_e10_config", "load_generator",
    "run_variant_worker", "validate_preflight",
]
