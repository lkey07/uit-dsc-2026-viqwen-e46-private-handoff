"""E09 deterministic output-length grid on the saved E07 dev-200 stack."""

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
from uit_dsc_fixed_rag.e02_compare import _load_dev
from uit_dsc_fixed_rag.e03_rrf_grid import (
    _bootstrap_ci,
    _json_sha256,
    _load_worker_progress,
    _read_jsonl,
    _validate_worker_placement,
    _write_worker_state,
)
from uit_dsc_fixed_rag.e07_lora import InferencePacking
from uit_dsc_fixed_rag.evaluation.official import (
    ensure_nltk_resources,
    nltk_meteor_score,
    rouge_l_fmeasure,
)
from uit_dsc_fixed_rag.retrieval_diagnostics import select_dev_sample


CODE_VERSION = "0.25.0"
LOGGER = logging.getLogger(__name__)


class E09Error(RuntimeError):
    """Raised when the output-length experiment loses its frozen identity."""


@dataclass(frozen=True)
class E09Config:
    raw: dict[str, Any]
    path: Path

    @property
    def config_sha256(self) -> str:
        return file_sha256(self.path)

    def section(self, key: str) -> dict[str, Any]:
        value = self.raw.get(key)
        if not isinstance(value, dict):
            raise ValueError(f"E09 section must be an object: {key}")
        return value

    @property
    def variants(self) -> list[dict[str, Any]]:
        variants = self.section("inference").get("variants")
        if not isinstance(variants, list):
            raise ValueError("E09 variants must be a list.")
        return variants


def load_e09_config(path: Path) -> E09Config:
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "schema_version",
        "experiment_id",
        "source_contexts",
        "source_control",
        "dev",
        "generator",
        "lora",
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
        not in {
            "E09-output-length-grid-qwen35-lora-dev200-v1",
            "E09-output-length-grid-qwen35-lora-dev200-v2",
        }
    ):
        raise ValueError("E09 config root is incompatible.")
    config = E09Config(payload, path)
    contexts = config.section("source_contexts")
    if contexts != {
        "experiment_id": "E06-prompt-length-grid-aiteam-dev200-v1",
        "path": "prepared/results.jsonl",
        "sha256": "56ed911cba621d9d30d582dea9388efca14e16f9f3ab6a5bc23d250b0bb6017c",
        "record_count": 200,
        "contexts_per_question": 12,
        "stack": "aiteam-bm25-dense-rrf050-050-no-reranker-ranked-top12",
    }:
        raise ValueError("E09 frozen context source changed.")
    control = config.section("source_control")
    if (
        control.get("experiment_id") != "E07-lora-aiteam-fulltrain-dev200-v2"
        or control.get("path") != "evaluation/results.jsonl"
        or control.get("sha256")
        != "0f9f62c97df96d5ab4c9759f641d0a86d53f9c3dc7afe97eb7db219d98e1f6b3"
        or control.get("variant") != "lora_e07"
        or control.get("max_new_tokens") != 384
        or control.get("adapter_sha256")
        != "c5bc928a956774988c3013d544b3f6fb2cadac005e504298b00910cf97dc4308"
    ):
        raise ValueError("E09 saved E07 control changed.")
    dev = config.section("dev")
    if (
        dev.get("sample_size") != 200
        or dev.get("sample_ids_sha256")
        != "0e5f5ddf95a0c4c7b4b1ae76471725d448e7a365a331bed997b6cf5922e9a345"
        or dev.get("answer_usage") != "scoring-only"
    ):
        raise ValueError("E09 dev sample changed.")
    generator = config.section("generator")
    accepted_counts = generator.get(
        "accepted_runtime_unique_parameter_counts",
        [generator.get("runtime_unique_parameter_count")],
    )
    if (
        generator.get("model_id") != "Qwen/Qwen3.5-2B"
        or generator.get("revision")
        != "15852e8c16360a2fea060d615a32b45270f8a8fc"
        or generator.get("published_parameter_count") != 2_274_069_824
        or accepted_counts
        not in ([2_213_241_664], [1_881_825_088, 2_213_241_664])
    ):
        raise ValueError("E09 generator identity or accepted runtime counts changed.")
    inference = config.section("inference")
    expected_variants = [
        {"key": "max512", "max_new_tokens": 512, "worker_rank": 0, "device": "cuda:0"},
        {"key": "max640", "max_new_tokens": 640, "worker_rank": 1, "device": "cuda:1"},
    ]
    if (
        inference.get("max_input_tokens") != 8192
        or inference.get("minimum_contexts") != 1
        or inference.get("control_max_new_tokens") != 384
        or inference.get("variants") != expected_variants
        or inference.get("do_sample") is not False
        or inference.get("num_beams") != 1
        or inference.get("enable_thinking") is not False
        or inference.get("use_cache") is not True
    ):
        raise ValueError("E09 inference grid changed.")
    execution = config.section("execution")
    if execution.get("workers") != 2 or execution.get("checkpoint_after_questions") != 1:
        raise ValueError("E09 dual-GPU execution changed.")
    budget = config.section("parameter_budget")
    if (
        budget.get("maximum_stack_total")
        != budget.get("embedding")
        + budget.get("generator")
        + budget.get("adapter_parameter_cap")
        or budget["maximum_stack_total"] >= budget["exclusive_limit"]
    ):
        raise ValueError("E09 parameter budget failed.")
    contract = config.section("run_contract")
    if not contract or not all(value is True for value in contract.values()):
        raise ValueError("E09 run contract lost an invariant.")
    if config.section("scoring").get("promotion_allowed") is not False:
        raise ValueError("E09 dev-200 grid cannot auto-promote.")
    return config


def inference_packing(config: E09Config) -> InferencePacking:
    section = config.section("inference")
    return InferencePacking(
        max_input_tokens=section["max_input_tokens"],
        minimum_contexts=section["minimum_contexts"],
        system_prompt=section["system_prompt"],
        answer_instruction=section["answer_instruction"],
    )


def _sample_ids(dev_path: Path, config: E09Config) -> tuple[dict[str, Any], list[str]]:
    dev_cfg = config.section("dev")
    if not dev_path.is_file() or file_sha256(dev_path) != dev_cfg["sha256"]:
        raise E09Error("Pinned dev split is missing or changed.")
    dev = _load_dev(dev_path)
    ids = select_dev_sample(
        dev, seed=dev_cfg["sample_seed"], size=dev_cfg["sample_size"]
    )
    identity = hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()
    if identity != dev_cfg["sample_ids_sha256"]:
        raise E09Error("E09 deterministic dev-200 IDs changed.")
    return dev, ids


def validate_preflight(
    *, project_root: Path, contexts_directory: Path, control_directory: Path,
    dev_path: Path, config: E09Config,
) -> dict[str, Any]:
    scorer_cfg = config.section("scoring")
    scorer = project_root / scorer_cfg["official_scorer_path"]
    if not scorer.is_file() or file_sha256(scorer) != scorer_cfg["official_scorer_sha256"]:
        raise E09Error("Pinned official scorer is missing or changed.")
    _, ids = _sample_ids(dev_path, config)
    context_cfg = config.section("source_contexts")
    contexts_path = contexts_directory / context_cfg["path"]
    if not contexts_path.is_file() or file_sha256(contexts_path) != context_cfg["sha256"]:
        raise E09Error("Saved E06 ranked-top12 contexts changed.")
    contexts = _read_jsonl(contexts_path)
    if len(contexts) != len(ids):
        raise E09Error("E09 context row count changed.")
    for index, (question_id, row) in enumerate(zip(ids, contexts)):
        if (
            row.get("question_id") != question_id
            or row.get("sample_index") != index
            or not isinstance(row.get("contexts"), list)
            or len(row["contexts"]) != context_cfg["contexts_per_question"]
        ):
            raise E09Error(f"Invalid frozen context row: {index}")

    control_cfg = config.section("source_control")
    control_path = control_directory / control_cfg["path"]
    adapter_path = control_directory / control_cfg["adapter_path"]
    training_records = control_directory / "training-data/records.jsonl"
    report_path = control_directory / "report.json"
    complete_path = adapter_path.parent / "complete.json"
    required = (control_path, adapter_path, training_records, report_path, complete_path)
    if not all(path.is_file() for path in required):
        raise E09Error("Saved E07 full-train artifact is incomplete.")
    if file_sha256(control_path) != control_cfg["sha256"]:
        raise E09Error("Saved E07 dev-200 control rows changed.")
    if file_sha256(adapter_path) != control_cfg["adapter_sha256"]:
        raise E09Error("Saved E07 adapter changed.")
    if file_sha256(training_records) != control_cfg["training_records_sha256"]:
        raise E09Error("Saved E07 training records changed.")
    complete = json.loads(complete_path.read_text(encoding="utf-8"))
    if complete.get("adapter_sha256") != control_cfg["adapter_sha256"]:
        raise E09Error("Saved E07 adapter completion evidence changed.")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    metrics = report.get("metrics", {}).get(control_cfg["variant"], {})
    evidence = report.get("evidence", {})
    if (
        report.get("experiment_id") != control_cfg["experiment_id"]
        or report.get("sample_size") != 200
        or report.get("train_sample_size") != 5636
        or report.get("smoke_leader") != control_cfg["variant"]
        or report.get("promotion_allowed") is not False
        or metrics.get("meteor") != control_cfg["meteor"]
        or metrics.get("rouge_l") != control_cfg["rouge_l"]
        or metrics.get("mean_output_tokens") != control_cfg["mean_output_tokens"]
        or evidence.get("config_sha256") != control_cfg["config_sha256"]
        or evidence.get("evaluation_results_sha256") != control_cfg["sha256"]
        or evidence.get("adapter_sha256") != control_cfg["adapter_sha256"]
        or evidence.get("training_records_sha256")
        != control_cfg["training_records_sha256"]
    ):
        raise E09Error("Saved E07 report changed.")
    control_rows = _read_jsonl(control_path)
    if len(control_rows) != len(ids):
        raise E09Error("Saved E07 control count changed.")
    for index, (question_id, row) in enumerate(zip(ids, control_rows)):
        variant = row.get("variants", {}).get(control_cfg["variant"], {})
        if (
            row.get("question_id") != question_id
            or row.get("sample_index") != index
            or not isinstance(variant.get("answer"), str)
            or not variant["answer"].strip()
            or not isinstance(variant.get("output_tokens"), int)
        ):
            raise E09Error(f"Invalid saved E07 control row: {index}")
    return {
        "config_sha256": config.config_sha256,
        "dev_sha256": config.section("dev")["sha256"],
        "sample_ids_sha256": config.section("dev")["sample_ids_sha256"],
        "contexts_sha256": file_sha256(contexts_path),
        "control_results_sha256": file_sha256(control_path),
        "adapter_sha256": file_sha256(adapter_path),
        "control_report_sha256": file_sha256(report_path),
        "sample_size": len(ids),
    }


def load_generator(
    *, config: E09Config, control_directory: Path, device: str,
) -> tuple[Any, Any, dict[str, Any], int, int]:
    try:
        import torch
        from peft import PeftModel
        from transformers import AutoTokenizer, Qwen3_5ForCausalLM
    except ImportError as exc:  # pragma: no cover
        raise E09Error("Install Transformers and PEFT for E09 inference.") from exc
    if device not in {"cuda:0", "cuda:1"} or torch.cuda.device_count() != 2:
        raise E09Error("E09 requires exactly two T4 GPUs.")
    torch.cuda.set_device(int(device[-1]))
    seed = 20260830
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    generator = config.section("generator")
    tokenizer = AutoTokenizer.from_pretrained(
        generator["model_id"], revision=generator["revision"], trust_remote_code=False
    )
    base = Qwen3_5ForCausalLM.from_pretrained(
        generator["model_id"], revision=generator["revision"], dtype=torch.float16,
        device_map={"": device}, low_cpu_mem_usage=True, trust_remote_code=False,
    )
    observed = sum(parameter.numel() for parameter in base.parameters())
    accepted = generator.get(
        "accepted_runtime_unique_parameter_counts",
        [generator["runtime_unique_parameter_count"]],
    )
    if observed not in accepted:
        raise E09Error(f"Qwen runtime parameter count changed: {observed}")
    final_adapter = (control_directory / config.section("source_control")["adapter_path"]).parent
    model = PeftModel.from_pretrained(base, final_adapter, is_trainable=False)
    model.eval()
    adapter_parameters = sum(
        parameter.numel() for name, parameter in model.named_parameters() if "lora_" in name
    )
    if not 0 < adapter_parameters <= config.section("lora")["adapter_parameter_cap"]:
        raise E09Error("Observed E07 adapter parameter count violates its cap.")
    try:
        device_map = _validate_worker_placement(model, device)
    except RuntimeError as exc:
        raise E09Error(str(exc)) from exc
    LOGGER.info(
        "e09_generator_loaded device=%s runtime_unique_parameters=%d "
        "adapter_parameters=%d",
        device, observed, adapter_parameters,
    )
    return model, tokenizer, device_map, adapter_parameters, observed


def _variant_for_worker(config: E09Config, worker_rank: int, device: str) -> dict[str, Any]:
    matches = [
        variant for variant in config.variants
        if variant["worker_rank"] == worker_rank and variant["device"] == device
    ]
    if len(matches) != 1:
        raise E09Error("E09 worker-to-length mapping changed.")
    return matches[0]


def run_variant_worker(
    *, worker_rank: int, device: str, model: Any, tokenizer: Any,
    contexts_directory: Path, control_directory: Path, dev_path: Path,
    output_directory: Path, config: E09Config, device_map: dict[str, Any],
    adapter_parameters: int, runtime_unique_parameter_count: int,
) -> dict[str, Any]:
    variant = _variant_for_worker(config, worker_rank, device)
    context_cfg, control_cfg = (
        config.section("source_contexts"), config.section("source_control")
    )
    contexts_path = contexts_directory / context_cfg["path"]
    control_path = control_directory / control_cfg["path"]
    if (
        file_sha256(contexts_path) != context_cfg["sha256"]
        or file_sha256(control_path) != control_cfg["sha256"]
    ):
        raise E09Error("E09 frozen input changed after preflight.")
    contexts = _read_jsonl(contexts_path)
    dev, ids = _sample_ids(dev_path, config)
    assigned = list(range(len(ids)))
    adapter_hash = file_sha256(
        control_directory / control_cfg["adapter_path"]
    )
    identity = {
        "code_version": CODE_VERSION,
        "stage": "e09-one-length-variant-per-gpu",
        "config_sha256": config.config_sha256,
        "contexts_sha256": context_cfg["sha256"],
        "control_results_sha256": control_cfg["sha256"],
        "adapter_sha256": adapter_hash,
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
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        return len(tokenizer(rendered, add_special_tokens=False)["input_ids"])

    for position in range(completed, len(assigned)):
        index = assigned[position]
        question_id = ids[index]
        selected, messages, input_tokens = pack_contexts(
            question=dev[question_id]["question"], contexts=contexts[index]["contexts"],
            config=packing, token_counter=token_count,
        )
        rendered = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        inputs = tokenizer(rendered, add_special_tokens=False, return_tensors="pt")
        inputs = {key: value.to(device) for key, value in inputs.items()}
        started = time.perf_counter()
        generated = model.generate(
            **inputs, do_sample=False, num_beams=1,
            max_new_tokens=variant["max_new_tokens"], use_cache=True,
        )
        latency_ms = (time.perf_counter() - started) * 1000
        input_width = inputs["input_ids"].shape[1]
        new_ids = generated[0, input_width:]
        answer = tokenizer.decode(
            new_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        ).strip()
        if not answer:
            raise E09Error(f"Empty E09 answer: {question_id}")
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
            "e09_generation_progress variant=%s device=%s completed=%d total=%d "
            "question_id=%s finish=%s",
            variant["key"], device, position + 1, len(assigned), question_id,
            finish_reason,
        )
    _write_worker_state(state_path, identity, len(assigned), len(assigned))
    return {"variant": variant["key"], "completed": len(assigned)}


def finalize_score(
    *, contexts_directory: Path, control_directory: Path, dev_path: Path,
    output_directory: Path, config: E09Config,
) -> dict[str, Any]:
    ensure_nltk_resources(download=False)
    dev, ids = _sample_ids(dev_path, config)
    control_cfg = config.section("source_control")
    control_path = control_directory / control_cfg["path"]
    if file_sha256(control_path) != control_cfg["sha256"]:
        raise E09Error("E09 control changed before scoring.")
    control_rows = _read_jsonl(control_path)
    generated_by_key: dict[str, list[dict[str, Any]]] = {}
    for variant in config.variants:
        root = output_directory / "generation" / variant["key"]
        state_path = root / "state.json"
        if not state_path.is_file():
            raise E09Error(f"Missing E09 worker state: {variant['key']}")
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("complete") is not True or state.get("completed_count") != len(ids):
            raise E09Error(f"Incomplete E09 worker: {variant['key']}")
        rows = []
        for index, question_id in enumerate(ids):
            path = root / "records" / f"{index:04d}.json"
            if not path.is_file():
                raise E09Error(f"Missing E09 record: {variant['key']}:{index}")
            row = json.loads(path.read_text(encoding="utf-8"))
            if (
                row.get("question_id") != question_id
                or row.get("sample_index") != index
                or row.get("variant") != variant["key"]
                or row.get("max_new_tokens") != variant["max_new_tokens"]
                or row.get("runtime_unique_parameter_count")
                not in config.section("generator").get(
                    "accepted_runtime_unique_parameter_counts",
                    [config.section("generator")["runtime_unique_parameter_count"]],
                )
                or row.get("finish_reason") not in {"eos", "length", "other"}
            ):
                raise E09Error(f"Invalid E09 record: {variant['key']}:{index}")
            rows.append(row)
        _atomic_jsonl(root / "results.jsonl", rows)
        generated_by_key[variant["key"]] = rows

    merged = []
    keys = ["control384", *[variant["key"] for variant in config.variants]]
    scored: dict[str, list[dict[str, Any]]] = {key: [] for key in keys}
    for index, question_id in enumerate(ids):
        control = control_rows[index]["variants"][control_cfg["variant"]]
        variants = {
            "control384": {
                **control,
                "max_new_tokens": control_cfg["max_new_tokens"],
                "answer_source": "reused-byte-verified-e07-fulltrain-control",
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
        metric = {
            "meteor": fmean(item["meteor"] for item in scored[key]),
            "rouge_l": fmean(item["rouge_l"] for item in scored[key]),
            "mean_output_tokens": fmean(row["output_tokens"] for row in rows),
            "mean_answer_characters": fmean(len(row["answer"]) for row in rows),
        }
        if key == "control384":
            metric["near_cap_rate_output_tokens_ge_380"] = fmean(
                row["output_tokens"] >= 380 for row in rows
            )
        else:
            metric["length_finish_rate"] = fmean(
                row["finish_reason"] == "length" for row in rows
            )
            metric["mean_generation_latency_ms"] = fmean(
                row["generation_latency_ms"] for row in rows
            )
        metrics[key] = metric
    if (
        abs(metrics["control384"]["meteor"] - control_cfg["meteor"]) > 1e-12
        or abs(metrics["control384"]["rouge_l"] - control_cfg["rouge_l"]) > 1e-12
        or abs(metrics["control384"]["mean_output_tokens"] - control_cfg["mean_output_tokens"])
        > 1e-12
    ):
        raise E09Error("Re-scored E07 384-token control changed.")

    scoring = config.section("scoring")
    paired: dict[str, Any] = {}
    for variant in config.variants:
        key = variant["key"]
        meteor = [
            candidate["meteor"] - control["meteor"]
            for candidate, control in zip(scored[key], scored["control384"])
        ]
        rouge = [
            candidate["rouge_l"] - control["rouge_l"]
            for candidate, control in zip(scored[key], scored["control384"])
        ]
        paired[f"{key}-minus-control384"] = {
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
    report = {
        "schema_version": "1.0",
        "experiment_id": config.raw["experiment_id"],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "sample_size": len(ids),
        "only_changed_factor": "max_new_tokens",
        "metrics": metrics,
        "paired_deltas_vs_control384": paired,
        "smoke_leader": leader,
        "promotion_allowed": False,
        "evidence": {
            "config_sha256": config.config_sha256,
            "contexts_sha256": file_sha256(
                contexts_directory / config.section("source_contexts")["path"]
            ),
            "control_results_sha256": file_sha256(control_path),
            "adapter_sha256": file_sha256(
                control_directory / control_cfg["adapter_path"]
            ),
            "results_sha256": file_sha256(results_path),
            "runtime_unique_parameter_counts": sorted(
                {
                    row["runtime_unique_parameter_count"]
                    for variant in config.variants
                    for row in generated_by_key[variant["key"]]
                }
            ),
        },
        "warning": (
            "E09 is a reused dev-200 output-length diagnostic. Select a length only "
            "by operator review; this report cannot auto-promote a competition stack."
        ),
    }
    _atomic_json(output_directory / "report.json", report)
    return report
