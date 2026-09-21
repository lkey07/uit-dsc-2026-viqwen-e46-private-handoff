"""E12 Vi-Qwen2-3B-RAG output-length grid on deterministic dev-200."""

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
from uit_dsc_fixed_rag.e07_generator_ab import (
    _as_text_chat_messages,
    load_candidate_generator as load_e07g_candidate_generator,
)
from uit_dsc_fixed_rag.e07_lora import InferencePacking
from uit_dsc_fixed_rag.e10_repetition_grid import answer_diagnostics
from uit_dsc_fixed_rag.evaluation.official import (
    ensure_nltk_resources,
    nltk_meteor_score,
    rouge_l_fmeasure,
)
from uit_dsc_fixed_rag.retrieval_diagnostics import select_dev_sample


CODE_VERSION = "0.29.0"
LOGGER = logging.getLogger(__name__)


class E12Error(RuntimeError):
    """Raised when E12 loses a frozen experiment invariant."""


@dataclass(frozen=True)
class E12Config:
    raw: dict[str, Any]
    path: Path

    @property
    def config_sha256(self) -> str:
        return file_sha256(self.path)

    def section(self, key: str) -> dict[str, Any]:
        value = self.raw.get(key)
        if not isinstance(value, dict):
            raise ValueError(f"E12 section must be an object: {key}")
        return value

    @property
    def variants(self) -> list[dict[str, Any]]:
        value = self.section("inference").get("variants")
        if not isinstance(value, list):
            raise ValueError("E12 variants must be a list.")
        return value


def load_e12_config(path: Path) -> E12Config:
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "schema_version", "experiment_id", "source_contexts", "source_e07g",
        "dev", "candidate_generator", "inference", "parameter_budget",
        "execution", "scoring", "run_contract",
    }
    if (
        set(payload) != required
        or payload.get("schema_version") != "1.0"
        or payload.get("experiment_id")
        != "E12-viqwen2-3b-rag-output-length-grid-dev200-v1"
    ):
        raise ValueError("E12 config root is incompatible.")
    config = E12Config(payload, path)
    candidate = config.section("candidate_generator")
    expected_candidate = {
        "key": "generator_vi_qwen2_3b_rag",
        "model_id": "AITeamVN/Vi-Qwen2-3B-RAG",
        "revision": "eaf427c24d86066a2b35828c499b7db3af321227",
        "architecture": "Qwen2ForCausalLM",
        "parameter_count": 3_085_938_688,
        "license": "apache-2.0",
        "trained_rag_context_tokens": 8192,
        "inventory_path": "configs/model-inventory-generator-ab-v2.json",
        "inventory_sha256": "5bb041a04fb89bacb3f90a1bb934d7d5337975876e5b167a4a67261b74d7e691",
    }
    if candidate != expected_candidate:
        raise ValueError("E12 candidate generator identity changed.")
    source = config.section("source_e07g")
    if (
        source.get("report_experiment_id")
        != "E07G-generator-ab-viqwen2-3b-rag-dev200-v1"
        or source.get("results_path") != "generation/results.jsonl"
        or source.get("results_sha256")
        != "40208338acd2c1f15e2c2d5db564ab3a3153aec18fbd1462d9b947a870368f9a"
        or source.get("sample_size") != 200
        or source.get("viqwen_max_new_tokens") != 384
    ):
        raise ValueError("E12 E07G source identity changed.")
    expected_variants = [
        {"key": "viqwen_max640", "max_new_tokens": 640,
         "worker_rank": 0, "device": "cuda:0"},
        {"key": "viqwen_max768", "max_new_tokens": 768,
         "worker_rank": 1, "device": "cuda:1"},
    ]
    inference = config.section("inference")
    if (
        inference.get("max_input_tokens") != 8192
        or inference.get("minimum_contexts") != 1
        or inference.get("control_max_new_tokens") != 384
        or inference.get("variants") != expected_variants
        or inference.get("do_sample") is not False
        or inference.get("num_beams") != 1
        or inference.get("use_cache") is not True
    ):
        raise ValueError("E12 inference grid changed.")
    budget = config.section("parameter_budget")
    if (
        budget.get("candidate_stack_total")
        != budget.get("embedding") + budget.get("candidate_generator")
        or budget.get("headroom")
        != budget.get("exclusive_limit") - budget.get("candidate_stack_total")
        or budget["candidate_stack_total"] >= budget["exclusive_limit"]
    ):
        raise ValueError("E12 parameter budget failed.")
    execution = config.section("execution")
    if execution.get("workers") != 2 or execution.get("checkpoint_after_questions") != 1:
        raise ValueError("E12 dual-GPU execution changed.")
    contract = config.section("run_contract")
    if not contract or not all(value is True for value in contract.values()):
        raise ValueError("E12 run contract lost an invariant.")
    if config.section("scoring").get("promotion_allowed") is not False:
        raise ValueError("E12 dev-200 grid cannot auto-promote.")
    return config


def _sample_ids(dev_path: Path, config: E12Config) -> tuple[dict[str, Any], list[str]]:
    dev_cfg = config.section("dev")
    if not dev_path.is_file() or file_sha256(dev_path) != dev_cfg["sha256"]:
        raise E12Error("Pinned dev split is missing or changed.")
    dev = _load_dev(dev_path)
    ids = select_dev_sample(dev, seed=dev_cfg["sample_seed"], size=dev_cfg["sample_size"])
    identity = hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()
    if identity != dev_cfg["sample_ids_sha256"]:
        raise E12Error("E12 deterministic dev-200 IDs changed.")
    return dev, ids


def _source_rows(
    *, e07g_directory: Path, dev_path: Path, config: E12Config,
) -> list[dict[str, Any]]:
    source = config.section("source_e07g")
    path = e07g_directory / source["results_path"]
    if not path.is_file() or file_sha256(path) != source["results_sha256"]:
        raise E12Error("Saved E07G results are missing or changed.")
    _, ids = _sample_ids(dev_path, config)
    rows = _read_jsonl(path)
    if len(rows) != len(ids):
        raise E12Error("Saved E07G row count changed.")
    expected_keys = {source["qwen_variant"], source["viqwen_variant"]}
    for index, (question_id, row) in enumerate(zip(ids, rows)):
        variants = row.get("variants", {})
        if (
            row.get("question_id") != question_id
            or row.get("sample_index") != index
            or set(variants) != expected_keys
        ):
            raise E12Error(f"Invalid saved E07G row: {index}")
        for key in expected_keys:
            variant = variants[key]
            if (
                not isinstance(variant.get("answer"), str)
                or not variant["answer"].strip()
                or not isinstance(variant.get("output_tokens"), int)
                or not isinstance(variant.get("selected_chunk_ids"), list)
            ):
                raise E12Error(f"Invalid saved E07G answer: {index}/{key}")
    return rows


def validate_preflight(
    *, project_root: Path, e06_directory: Path, e07g_directory: Path,
    dev_path: Path, config: E12Config,
) -> dict[str, Any]:
    scorer_cfg = config.section("scoring")
    scorer = project_root / scorer_cfg["official_scorer_path"]
    if not scorer.is_file() or file_sha256(scorer) != scorer_cfg["official_scorer_sha256"]:
        raise E12Error("Pinned official scorer is missing or changed.")
    candidate = config.section("candidate_generator")
    inventory = project_root / candidate["inventory_path"]
    if not inventory.is_file() or file_sha256(inventory) != candidate["inventory_sha256"]:
        raise E12Error("Pinned generator inventory is missing or changed.")
    _, ids = _sample_ids(dev_path, config)
    contexts_cfg = config.section("source_contexts")
    contexts_path = e06_directory / contexts_cfg["path"]
    report_path = e06_directory / "report.json"
    if (
        not contexts_path.is_file()
        or file_sha256(contexts_path) != contexts_cfg["sha256"]
        or not report_path.is_file()
    ):
        raise E12Error("Saved E06 contexts are missing or changed.")
    e06_report = json.loads(report_path.read_text(encoding="utf-8"))
    if (
        e06_report.get("experiment_id") != contexts_cfg["report_experiment_id"]
        or e06_report.get("sample_size") != contexts_cfg["record_count"]
        or e06_report.get("evidence", {}).get("config_sha256")
        != contexts_cfg["report_config_sha256"]
    ):
        raise E12Error("Saved E06 report changed.")
    contexts = _read_jsonl(contexts_path)
    if len(contexts) != len(ids):
        raise E12Error("Saved E06 context row count changed.")
    source_rows = _source_rows(
        e07g_directory=e07g_directory, dev_path=dev_path, config=config
    )
    for index, question_id in enumerate(ids):
        row = contexts[index]
        selected = source_rows[index]["variants"][
            config.section("source_e07g")["viqwen_variant"]
        ]["selected_chunk_ids"]
        context_ids = [item["chunk_id"] for item in row.get("contexts", [])]
        if (
            row.get("question_id") != question_id
            or row.get("sample_index") != index
            or len(row.get("contexts", [])) != contexts_cfg["contexts_per_question"]
            or not selected
            or context_ids[:len(selected)] != selected
        ):
            raise E12Error(f"E12 context identity changed: {index}")
    source = config.section("source_e07g")
    e07g_report_path = e07g_directory / "report.json"
    if not e07g_report_path.is_file():
        raise E12Error("Saved E07G report is missing.")
    report = json.loads(e07g_report_path.read_text(encoding="utf-8"))
    metrics = report.get("metrics", {})
    evidence = report.get("evidence", {})
    qwen, viqwen = metrics.get(source["qwen_variant"], {}), metrics.get(source["viqwen_variant"], {})
    if (
        report.get("experiment_id") != source["report_experiment_id"]
        or report.get("sample_size") != source["sample_size"]
        or evidence.get("config_sha256") != source["report_config_sha256"]
        or evidence.get("generation_results_sha256") != source["results_sha256"]
        or evidence.get("scores_sha256") != source["scores_sha256"]
        or qwen.get("meteor") != source["qwen_meteor"]
        or qwen.get("rouge_l") != source["qwen_rouge_l"]
        or qwen.get("mean_output_tokens") != source["qwen_mean_output_tokens"]
        or viqwen.get("meteor") != source["viqwen_meteor"]
        or viqwen.get("rouge_l") != source["viqwen_rouge_l"]
        or viqwen.get("mean_output_tokens") != source["viqwen_mean_output_tokens"]
    ):
        raise E12Error("Saved E07G report changed.")
    return {
        "config_sha256": config.config_sha256,
        "dev_sha256": config.section("dev")["sha256"],
        "sample_ids_sha256": config.section("dev")["sample_ids_sha256"],
        "contexts_sha256": file_sha256(contexts_path),
        "e07g_results_sha256": file_sha256(
            e07g_directory / source["results_path"]
        ),
        "e07g_report_sha256": file_sha256(e07g_report_path),
        "inventory_sha256": file_sha256(inventory),
        "sample_size": len(ids),
    }


def load_candidate_generator(
    config: E12Config, device: str
) -> tuple[Any, Any, dict[str, Any], int]:
    try:
        return load_e07g_candidate_generator(config, device)
    except Exception as exc:
        raise E12Error(str(exc)) from exc


def _variant_for_worker(config: E12Config, worker_rank: int, device: str) -> dict[str, Any]:
    matches = [
        item for item in config.variants
        if item["worker_rank"] == worker_rank and item["device"] == device
    ]
    if len(matches) != 1:
        raise E12Error("E12 worker-to-output-cap mapping changed.")
    return matches[0]


def generation_kwargs(variant: dict[str, Any]) -> dict[str, Any]:
    return {
        "do_sample": False,
        "num_beams": 1,
        "max_new_tokens": variant["max_new_tokens"],
        "use_cache": True,
    }


def run_variant_worker(
    *, worker_rank: int, device: str, model: Any, tokenizer: Any,
    e06_directory: Path, e07g_directory: Path, dev_path: Path,
    output_directory: Path, config: E12Config, device_map: dict[str, Any],
    observed_parameters: int,
) -> dict[str, Any]:
    variant = _variant_for_worker(config, worker_rank, device)
    contexts_cfg = config.section("source_contexts")
    contexts_path = e06_directory / contexts_cfg["path"]
    if file_sha256(contexts_path) != contexts_cfg["sha256"]:
        raise E12Error("E12 frozen contexts changed after preflight.")
    source_rows = _source_rows(
        e07g_directory=e07g_directory, dev_path=dev_path, config=config
    )
    contexts = _read_jsonl(contexts_path)
    dev, ids = _sample_ids(dev_path, config)
    assigned = list(range(len(ids)))
    candidate = config.section("candidate_generator")
    identity = {
        "code_version": CODE_VERSION,
        "stage": "e12-one-viqwen-output-cap-per-gpu",
        "config_sha256": config.config_sha256,
        "contexts_sha256": contexts_cfg["sha256"],
        "e07g_results_sha256": config.section("source_e07g")["results_sha256"],
        "candidate_model_id": candidate["model_id"],
        "candidate_revision": candidate["revision"],
        "observed_parameters": observed_parameters,
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
    inference = config.section("inference")
    packing = InferencePacking(
        max_input_tokens=inference["max_input_tokens"],
        minimum_contexts=inference["minimum_contexts"],
        system_prompt=inference["system_prompt"],
        answer_instruction=inference["answer_instruction"],
    )

    def render(messages: list[dict[str, Any]]) -> str:
        return tokenizer.apply_chat_template(
            _as_text_chat_messages(messages), tokenize=False, add_generation_prompt=True
        )

    def token_count(messages: list[dict[str, Any]]) -> int:
        return len(tokenizer(render(messages), add_special_tokens=False)["input_ids"])

    for position in range(completed, len(assigned)):
        index = assigned[position]
        question_id = ids[index]
        selected, messages, input_tokens = pack_contexts(
            question=dev[question_id]["question"], contexts=contexts[index]["contexts"],
            config=packing, token_counter=token_count,
        )
        source_selected = source_rows[index]["variants"][
            config.section("source_e07g")["viqwen_variant"]
        ]["selected_chunk_ids"]
        if [row["chunk_id"] for row in selected] != source_selected:
            raise E12Error(f"Vi-Qwen packed contexts changed: {question_id}")
        inputs = tokenizer(render(messages), add_special_tokens=False, return_tensors="pt")
        inputs = {key: value.to(device) for key, value in inputs.items()}
        started = time.perf_counter()
        generated = model.generate(**inputs, **generation_kwargs(variant))
        latency_ms = (time.perf_counter() - started) * 1000
        new_ids = generated[0, inputs["input_ids"].shape[1]:]
        answer = tokenizer.decode(
            new_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        ).strip()
        if not answer:
            raise E12Error(f"Vi-Qwen produced an empty E12 answer: {question_id}")
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
            "candidate_model_id": candidate["model_id"],
            "candidate_revision": candidate["revision"],
            "observed_parameters": observed_parameters,
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
            "e12_generation_progress variant=%s device=%s completed=%d total=%d "
            "question_id=%s finish=%s",
            variant["key"], device, position + 1, len(assigned), question_id,
            finish_reason,
        )
    _write_worker_state(state_path, identity, len(assigned), len(assigned))
    return {
        "variant": variant["key"], "completed": len(assigned),
        "observed_parameters": observed_parameters,
    }


def _source_variant(
    source_row: dict[str, Any], source_key: str, output_key: str, max_tokens: int,
) -> dict[str, Any]:
    value = dict(source_row["variants"][source_key])
    output_tokens = value["output_tokens"]
    value.update({
        "variant": output_key,
        "max_new_tokens": max_tokens,
        "finish_reason": "length" if output_tokens >= max_tokens else "other",
        "answer_source": "reused-byte-verified-e07g",
    })
    return value


def finalize_score(
    *, e06_directory: Path, e07g_directory: Path, dev_path: Path,
    output_directory: Path, config: E12Config,
) -> dict[str, Any]:
    ensure_nltk_resources(download=False)
    dev, ids = _sample_ids(dev_path, config)
    source_rows = _source_rows(
        e07g_directory=e07g_directory, dev_path=dev_path, config=config
    )
    generated: dict[str, list[dict[str, Any]]] = {}
    for variant in config.variants:
        root = output_directory / "generation" / variant["key"]
        state_path = root / "state.json"
        if not state_path.is_file():
            raise E12Error(f"Missing E12 worker state: {variant['key']}")
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("complete") is not True or state.get("completed_count") != len(ids):
            raise E12Error(f"Incomplete E12 worker: {variant['key']}")
        rows = []
        for index, question_id in enumerate(ids):
            path = root / "records" / f"{index:04d}.json"
            if not path.is_file():
                raise E12Error(f"Missing E12 record: {variant['key']}:{index}")
            row = json.loads(path.read_text(encoding="utf-8"))
            if (
                row.get("question_id") != question_id
                or row.get("sample_index") != index
                or row.get("variant") != variant["key"]
                or row.get("max_new_tokens") != variant["max_new_tokens"]
                or row.get("finish_reason") not in {"eos", "length", "other"}
                or row.get("observed_parameters")
                != config.section("candidate_generator")["parameter_count"]
            ):
                raise E12Error(f"Invalid E12 record: {variant['key']}:{index}")
            rows.append(row)
        _atomic_jsonl(root / "results.jsonl", rows)
        generated[variant["key"]] = rows

    source = config.section("source_e07g")
    control_keys = ("qwen35_lora_max384", "viqwen_max384")
    keys = [*control_keys, *[item["key"] for item in config.variants]]
    merged = []
    scored: dict[str, list[dict[str, Any]]] = {key: [] for key in keys}
    for index, question_id in enumerate(ids):
        variants = {
            control_keys[0]: _source_variant(
                source_rows[index], source["qwen_variant"], control_keys[0], 384
            ),
            control_keys[1]: _source_variant(
                source_rows[index], source["viqwen_variant"], control_keys[1], 384
            ),
        }
        for variant in config.variants:
            variants[variant["key"]] = generated[variant["key"]][index]
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
            "duplicate_sentence_rate": fmean(item["duplicate_sentence"] for item in diagnostics),
            "abbreviation_loop_rate": fmean(item["abbreviation_loop"] for item in diagnostics),
            "non_sentence_ending_rate": fmean(item["non_sentence_ending"] for item in diagnostics),
        }
        if key not in control_keys:
            metric["mean_generation_latency_ms"] = fmean(
                row["generation_latency_ms"] for row in rows
            )
        metrics[key] = metric
    if (
        metrics[control_keys[0]]["meteor"] != source["qwen_meteor"]
        or metrics[control_keys[0]]["rouge_l"] != source["qwen_rouge_l"]
        or metrics[control_keys[1]]["meteor"] != source["viqwen_meteor"]
        or metrics[control_keys[1]]["rouge_l"] != source["viqwen_rouge_l"]
    ):
        raise E12Error("Re-scored E07G controls changed.")

    scoring = config.section("scoring")
    paired: dict[str, Any] = {}
    for variant in config.variants:
        key = variant["key"]
        for baseline in control_keys:
            meteor = [
                candidate["meteor"] - control["meteor"]
                for candidate, control in zip(scored[key], scored[baseline])
            ]
            rouge = [
                candidate["rouge_l"] - control["rouge_l"]
                for candidate, control in zip(scored[key], scored[baseline])
            ]
            label = f"{key}-minus-{baseline}"
            paired[label] = {
                "meteor_mean": fmean(meteor),
                "meteor_bootstrap_95_ci": _bootstrap_ci(
                    meteor, seed=f"{scoring['bootstrap_seed']}:{label}:meteor",
                    iterations=scoring["bootstrap_iterations"],
                ),
                "rouge_l_mean": fmean(rouge),
                "rouge_l_bootstrap_95_ci": _bootstrap_ci(
                    rouge, seed=f"{scoring['bootstrap_seed']}:{label}:rouge_l",
                    iterations=scoring["bootstrap_iterations"],
                ),
            }
    leader = max(keys, key=lambda key: (metrics[key]["meteor"], metrics[key]["rouge_l"]))
    report = {
        "schema_version": "1.0",
        "experiment_id": config.raw["experiment_id"],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "sample_size": len(ids),
        "only_changed_factor_within_viqwen": "max_new_tokens: 384 control, 640, 768",
        "metrics": metrics,
        "paired_deltas": paired,
        "smoke_leader": leader,
        "promotion_allowed": False,
        "evidence": {
            "config_sha256": config.config_sha256,
            "contexts_sha256": file_sha256(
                e06_directory / config.section("source_contexts")["path"]
            ),
            "e07g_results_sha256": file_sha256(
                e07g_directory / source["results_path"]
            ),
            "candidate_revision": config.section("candidate_generator")["revision"],
            "candidate_parameter_count": config.section("candidate_generator")["parameter_count"],
            "results_sha256": file_sha256(results_path),
        },
        "warning": (
            "E12 is a dev-200 Vi-Qwen output-length smoke grid. It reads neither "
            "public nor holdout and cannot automatically promote a generator."
        ),
    }
    _atomic_json(output_directory / "report.json", report)
    return report


__all__ = [
    "E12Config", "E12Error", "finalize_score", "generation_kwargs",
    "load_candidate_generator", "load_e12_config", "run_variant_worker",
    "validate_preflight",
]
