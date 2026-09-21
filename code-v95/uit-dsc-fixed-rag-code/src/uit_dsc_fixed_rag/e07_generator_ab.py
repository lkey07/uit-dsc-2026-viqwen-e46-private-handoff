"""Post-E07 dev-200 generator A/B with a Vietnamese RAG generator."""

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
from uit_dsc_fixed_rag.e07_lora import InferencePacking
from uit_dsc_fixed_rag.evaluation.official import (
    ensure_nltk_resources,
    nltk_meteor_score,
    rouge_l_fmeasure,
)
from uit_dsc_fixed_rag.retrieval_diagnostics import select_dev_sample


CODE_VERSION = "0.17.0"
LOGGER = logging.getLogger(__name__)


class E07GeneratorAbError(RuntimeError):
    """Raised when the generator A/B cannot continue safely."""


@dataclass(frozen=True)
class E07GeneratorAbConfig:
    raw: dict[str, Any]
    path: Path

    @property
    def config_sha256(self) -> str:
        return file_sha256(self.path)

    def section(self, key: str) -> dict[str, Any]:
        value = self.raw.get(key)
        if not isinstance(value, dict):
            raise ValueError(f"E07G section must be an object: {key}")
        return value


def load_generator_ab_config(path: Path) -> E07GeneratorAbConfig:
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "schema_version",
        "experiment_id",
        "plan_addendum",
        "context_source",
        "control_source",
        "dev",
        "candidate_generator",
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
        != "E07G-generator-ab-viqwen2-3b-rag-dev200-v1"
    ):
        raise ValueError("E07G config root is incompatible.")
    config = E07GeneratorAbConfig(payload, path)
    candidate = config.section("candidate_generator")
    if candidate != {
        "key": "generator_vi_qwen2_3b_rag",
        "model_id": "AITeamVN/Vi-Qwen2-3B-RAG",
        "revision": "eaf427c24d86066a2b35828c499b7db3af321227",
        "architecture": "Qwen2ForCausalLM",
        "parameter_count": 3085938688,
        "license": "apache-2.0",
        "trained_rag_context_tokens": 8192,
    }:
        raise ValueError("E07G candidate generator identity changed.")
    dev = config.section("dev")
    if (
        dev.get("sample_size") != 200
        or dev.get("sample_ids_sha256")
        != "0e5f5ddf95a0c4c7b4b1ae76471725d448e7a365a331bed997b6cf5922e9a345"
    ):
        raise ValueError("E07G dev-200 contract changed.")
    inference = config.section("inference")
    if (
        inference.get("max_input_tokens") != 8192
        or inference.get("max_new_tokens") != 384
        or inference.get("do_sample") is not False
        or inference.get("num_beams") != 1
        or inference.get("use_cache") is not True
    ):
        raise ValueError("E07G inference contract changed.")
    budget = config.section("parameter_budget")
    if (
        budget.get("candidate_stack_total")
        != budget.get("embedding") + budget.get("candidate_generator")
        or budget.get("headroom")
        != budget.get("exclusive_limit") - budget.get("candidate_stack_total")
        or budget["candidate_stack_total"] >= budget["exclusive_limit"]
    ):
        raise ValueError("E07G parameter budget failed.")
    contract = config.section("run_contract")
    if not all(value is True for key, value in contract.items() if key != "promotion_allowed"):
        raise ValueError("E07G run contract lost an invariant.")
    if contract.get("promotion_allowed") is not False:
        raise ValueError("E07G dev-200 cannot formally promote a generator.")
    return config


def validate_preflight(
    *,
    project_root: Path,
    e06_directory: Path,
    full_lora_directory: Path,
    dev_path: Path,
    config: E07GeneratorAbConfig,
) -> dict[str, Any]:
    plan = config.section("plan_addendum")
    inventory_path = project_root / plan["inventory_path"]
    if (
        not inventory_path.is_file()
        or file_sha256(inventory_path) != plan["inventory_sha256"]
    ):
        raise E07GeneratorAbError("Pinned generator A/B inventory is missing or changed.")
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    candidate = config.section("candidate_generator")
    inventory_candidate = inventory.get("candidate_generator", {})
    if any(
        inventory_candidate.get(key) != candidate[key]
        for key in ("model_id", "revision", "architecture", "parameter_count", "license")
    ):
        raise E07GeneratorAbError("Candidate identity differs from reviewed inventory.")

    scorer_cfg = config.section("scoring")
    scorer = project_root / scorer_cfg["official_scorer_path"]
    if not scorer.is_file() or file_sha256(scorer) != scorer_cfg["official_scorer_sha256"]:
        raise E07GeneratorAbError("Pinned official scorer is missing or changed.")
    dev_cfg = config.section("dev")
    if not dev_path.is_file() or file_sha256(dev_path) != dev_cfg["sha256"]:
        raise E07GeneratorAbError("Pinned dev split is missing or changed.")
    dev = _load_dev(dev_path)
    sample_ids = select_dev_sample(
        dev, seed=dev_cfg["sample_seed"], size=dev_cfg["sample_size"]
    )
    if _ids_sha(sample_ids) != dev_cfg["sample_ids_sha256"]:
        raise E07GeneratorAbError("Deterministic dev-200 order changed.")

    context = config.section("context_source")
    e06_report_path = e06_directory / "report.json"
    prepared_path = e06_directory / "prepared" / "results.jsonl"
    e06_generation_path = e06_directory / "generation" / "results.jsonl"
    if not all(path.is_file() for path in (e06_report_path, prepared_path, e06_generation_path)):
        raise E07GeneratorAbError("Saved E06 context artifact is incomplete.")
    e06_report = json.loads(e06_report_path.read_text(encoding="utf-8"))
    if (
        e06_report.get("experiment_id") != context["report_experiment_id"]
        or e06_report.get("sample_size") != context["sample_size"]
        or e06_report.get("evidence", {}).get("config_sha256")
        != context["report_config_sha256"]
        or file_sha256(prepared_path) != context["prepared_results_sha256"]
        or file_sha256(e06_generation_path) != context["generation_results_sha256"]
    ):
        raise E07GeneratorAbError("Saved E06 context lineage changed.")

    control = config.section("control_source")
    control_report_path = full_lora_directory / "report.json"
    control_results_path = full_lora_directory / "evaluation" / "results.jsonl"
    adapter_path = full_lora_directory / "training" / "adapter-final" / "adapter_model.safetensors"
    training_records_path = full_lora_directory / "training-data" / "records.jsonl"
    if not all(
        path.is_file()
        for path in (control_report_path, control_results_path, adapter_path, training_records_path)
    ):
        raise E07GeneratorAbError("Saved full-train LoRA control is incomplete.")
    control_report = json.loads(control_report_path.read_text(encoding="utf-8"))
    _validate_control_report(control_report, control)
    if (
        file_sha256(control_results_path) != control["evaluation_results_sha256"]
        or file_sha256(adapter_path) != control["adapter_sha256"]
        or file_sha256(training_records_path) != control["training_records_sha256"]
    ):
        raise E07GeneratorAbError("Saved full-train LoRA evidence changed.")

    prepared_rows = _read_jsonl(prepared_path)
    control_rows = _read_jsonl(control_results_path)
    if len(prepared_rows) != len(sample_ids) or len(control_rows) != len(sample_ids):
        raise E07GeneratorAbError("Control rows do not cover exactly dev-200.")
    for index, question_id in enumerate(sample_ids):
        prepared, row = prepared_rows[index], control_rows[index]
        variant = row.get("variants", {}).get(control["control_variant"], {})
        available_ids = [item["chunk_id"] for item in prepared.get("contexts", [])]
        selected_ids = variant.get("selected_chunk_ids")
        if (
            prepared.get("question_id") != question_id
            or row.get("question_id") != question_id
            or not isinstance(selected_ids, list)
            or not selected_ids
            or selected_ids != available_ids[: len(selected_ids)]
            or not isinstance(variant.get("answer"), str)
            or not variant["answer"].strip()
        ):
            raise E07GeneratorAbError(f"Context/control mismatch at dev index {index}.")
    return {
        "config_sha256": config.config_sha256,
        "inventory_sha256": plan["inventory_sha256"],
        "dev_sha256": dev_cfg["sha256"],
        "sample_ids_sha256": dev_cfg["sample_ids_sha256"],
        "sample_size": len(sample_ids),
        "prepared_results_sha256": context["prepared_results_sha256"],
        "control_results_sha256": control["evaluation_results_sha256"],
        "control_adapter_sha256": control["adapter_sha256"],
        "candidate_stack_parameters": config.section("parameter_budget")[
            "candidate_stack_total"
        ],
    }


def load_candidate_generator(
    config: E07GeneratorAbConfig, device: str
) -> tuple[Any, Any, dict[str, Any], int]:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:  # pragma: no cover
        raise E07GeneratorAbError("Install Transformers for E07G generation.") from exc
    if device not in {"cuda:0", "cuda:1"} or torch.cuda.device_count() != 2:
        raise E07GeneratorAbError("E07G requires Kaggle GPU T4 x2.")
    torch.cuda.set_device(int(device[-1]))
    candidate = config.section("candidate_generator")
    tokenizer = AutoTokenizer.from_pretrained(
        candidate["model_id"],
        revision=candidate["revision"],
        trust_remote_code=False,
    )
    model = AutoModelForCausalLM.from_pretrained(
        candidate["model_id"],
        revision=candidate["revision"],
        dtype=torch.float16,
        device_map={"": device},
        low_cpu_mem_usage=True,
        trust_remote_code=False,
    )
    architectures = list(getattr(model.config, "architectures", None) or [])
    observed_parameters = sum(parameter.numel() for parameter in model.parameters())
    if (
        candidate["architecture"] not in architectures
        or observed_parameters != candidate["parameter_count"]
    ):
        raise E07GeneratorAbError(
            "Vi-Qwen identity changed: "
            f"architectures={architectures}, parameters={observed_parameters}."
        )
    model.eval()
    device_map = _validate_worker_placement(model, device)
    LOGGER.info(
        "e07g_candidate_loaded device=%s parameters=%d architecture=%s",
        device,
        observed_parameters,
        candidate["architecture"],
    )
    return model, tokenizer, device_map, observed_parameters


def run_generation_worker(
    *,
    worker_rank: int,
    device: str,
    model: Any,
    tokenizer: Any,
    e06_directory: Path,
    full_lora_directory: Path,
    dev_path: Path,
    output_directory: Path,
    config: E07GeneratorAbConfig,
    device_map: dict[str, Any],
    observed_parameters: int,
) -> dict[str, Any]:
    execution = config.section("execution")
    workers = execution["workers"]
    if worker_rank not in range(workers) or device != f"cuda:{worker_rank}":
        raise E07GeneratorAbError("E07G worker/device mapping changed.")
    context, control = config.section("context_source"), config.section("control_source")
    prepared_path = e06_directory / "prepared" / "results.jsonl"
    control_path = full_lora_directory / "evaluation" / "results.jsonl"
    if (
        file_sha256(prepared_path) != context["prepared_results_sha256"]
        or file_sha256(control_path) != control["evaluation_results_sha256"]
    ):
        raise E07GeneratorAbError("E07G sources changed after preflight.")
    prepared_rows, control_rows = _read_jsonl(prepared_path), _read_jsonl(control_path)
    dev, dev_cfg = _load_dev(dev_path), config.section("dev")
    sample_ids = select_dev_sample(
        dev, seed=dev_cfg["sample_seed"], size=dev_cfg["sample_size"]
    )
    assigned = [index for index in range(len(sample_ids)) if index % workers == worker_rank]
    candidate = config.section("candidate_generator")
    identity = {
        "code_version": CODE_VERSION,
        "stage": "e07g-viqwen-dev200-worker",
        "config_sha256": config.config_sha256,
        "prepared_results_sha256": context["prepared_results_sha256"],
        "control_results_sha256": control["evaluation_results_sha256"],
        "candidate_model_id": candidate["model_id"],
        "candidate_revision": candidate["revision"],
        "observed_parameters": observed_parameters,
        "worker_rank": worker_rank,
        "device": device,
        "device_map": device_map,
        "assigned_indices_sha256": hashlib.sha256(
            ",".join(map(str, assigned)).encode("ascii")
        ).hexdigest(),
    }
    identity["identity_sha256"] = _json_sha256(identity)
    root = output_directory / "generation"
    records = root / "records"
    records.mkdir(parents=True, exist_ok=True)
    state_path = root / f"worker-{worker_rank}-state.json"
    completed = _load_worker_progress(
        records=records,
        state_path=state_path,
        identity=identity,
        assigned_indices=assigned,
        sample_ids=sample_ids,
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
            _as_text_chat_messages(messages),
            tokenize=False,
            add_generation_prompt=True,
        )

    def token_count(messages: list[dict[str, Any]]) -> int:
        return len(tokenizer(render(messages), add_special_tokens=False)["input_ids"])

    for position in range(completed, len(assigned)):
        sample_index = assigned[position]
        question_id = sample_ids[sample_index]
        contexts = prepared_rows[sample_index]["contexts"]
        selected, messages, input_tokens = pack_contexts(
            question=dev[question_id]["question"],
            contexts=contexts,
            config=packing,
            token_counter=token_count,
        )
        rendered = render(messages)
        inputs = tokenizer(rendered, add_special_tokens=False, return_tensors="pt")
        inputs = {key: value.to(device) for key, value in inputs.items()}
        started = time.perf_counter()
        generated = model.generate(
            **inputs,
            do_sample=False,
            num_beams=1,
            max_new_tokens=inference["max_new_tokens"],
            use_cache=True,
        )
        latency_ms = (time.perf_counter() - started) * 1000
        answer = tokenizer.decode(
            generated[0, inputs["input_ids"].shape[1] :],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        ).strip()
        if not answer:
            raise E07GeneratorAbError(f"Vi-Qwen produced an empty answer: {question_id}")
        selected_ids = [row["chunk_id"] for row in selected]
        saved_control = control_rows[sample_index]["variants"][control["control_variant"]]
        if selected_ids != saved_control.get("selected_chunk_ids"):
            raise E07GeneratorAbError(f"Control contexts changed: {question_id}")
        _atomic_json(
            records / f"{sample_index:04d}.json",
            {
                "question_id": question_id,
                "sample_index": sample_index,
                "worker_rank": worker_rank,
                "worker_identity_sha256": identity["identity_sha256"],
                "variants": {
                    "qwen35_2b_lora_fulltrain": {
                        **saved_control,
                        "answer_source": "reused-e07-fulltrain-dev200",
                    },
                    "vi_qwen2_3b_rag": {
                        "answer": answer,
                        "answer_source": "generated-e07g-viqwen",
                        "selected_chunk_ids": selected_ids,
                        "selected_context_count": len(selected),
                        "input_tokens": input_tokens,
                        "output_tokens": len(
                            tokenizer(answer, add_special_tokens=False)["input_ids"]
                        ),
                        "generation_latency_ms": latency_ms,
                    },
                },
            },
        )
        _write_worker_state(state_path, identity, position + 1, len(assigned))
        LOGGER.info(
            "e07g_generation_progress worker=%d device=%s completed=%d total=%d question_id=%s",
            worker_rank,
            device,
            position + 1,
            len(assigned),
            question_id,
        )
    _write_worker_state(state_path, identity, len(assigned), len(assigned))
    return {
        "worker_rank": worker_rank,
        "completed": len(assigned),
        "candidate_model_id": candidate["model_id"],
        "candidate_revision": candidate["revision"],
        "observed_parameters": observed_parameters,
    }


def finalize_and_score(
    *, output_directory: Path, dev_path: Path, config: E07GeneratorAbConfig
) -> dict[str, Any]:
    ensure_nltk_resources(download=False)
    dev, dev_cfg = _load_dev(dev_path), config.section("dev")
    sample_ids = select_dev_sample(
        dev, seed=dev_cfg["sample_seed"], size=dev_cfg["sample_size"]
    )
    root = output_directory / "generation"
    records = root / "records"
    rows: list[dict[str, Any]] = []
    variants = ("qwen35_2b_lora_fulltrain", "vi_qwen2_3b_rag")
    for index, question_id in enumerate(sample_ids):
        path = records / f"{index:04d}.json"
        if not path.is_file():
            raise E07GeneratorAbError(f"Missing E07G record: {index}")
        row = json.loads(path.read_text(encoding="utf-8"))
        if row.get("question_id") != question_id or set(row.get("variants", {})) != set(
            variants
        ):
            raise E07GeneratorAbError(f"Invalid E07G record: {index}")
        rows.append(row)
    observed_counts = set()
    for rank in range(config.section("execution")["workers"]):
        state_path = root / f"worker-{rank}-state.json"
        summary_path = root / f"worker-{rank}-summary.json"
        if not state_path.is_file() or not summary_path.is_file():
            raise E07GeneratorAbError(f"Missing E07G worker evidence: {rank}")
        state = json.loads(state_path.read_text(encoding="utf-8"))
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if state.get("complete") is not True:
            raise E07GeneratorAbError(f"E07G worker incomplete: {rank}")
        observed_counts.add(summary.get("observed_parameters"))
    expected_count = config.section("candidate_generator")["parameter_count"]
    if observed_counts != {expected_count}:
        raise E07GeneratorAbError("Worker parameter audits disagree.")
    _atomic_jsonl(root / "results.jsonl", rows)

    scored: dict[str, list[dict[str, Any]]] = {key: [] for key in variants}
    score_rows = []
    for row in rows:
        reference = dev[row["question_id"]]["answer"]
        score_row: dict[str, Any] = {"question_id": row["question_id"]}
        for key in variants:
            answer = row["variants"][key]["answer"]
            item = {
                "meteor": nltk_meteor_score(reference, answer),
                "rouge_l": rouge_l_fmeasure(reference, answer),
            }
            scored[key].append(item)
            score_row[key] = item
        score_rows.append(score_row)
    _atomic_jsonl(root / "scores.jsonl", score_rows)
    metrics = {
        key: {
            "meteor": fmean(item["meteor"] for item in scored[key]),
            "rouge_l": fmean(item["rouge_l"] for item in scored[key]),
            "mean_output_tokens": fmean(
                row["variants"][key]["output_tokens"] for row in rows
            ),
        }
        for key in variants
    }
    control = config.section("control_source")
    if (
        metrics["qwen35_2b_lora_fulltrain"]["meteor"] != control["meteor"]
        or metrics["qwen35_2b_lora_fulltrain"]["rouge_l"] != control["rouge_l"]
    ):
        raise E07GeneratorAbError("Re-scored full-train LoRA control changed.")
    meteor_delta = [
        candidate["meteor"] - baseline["meteor"]
        for candidate, baseline in zip(
            scored["vi_qwen2_3b_rag"], scored["qwen35_2b_lora_fulltrain"]
        )
    ]
    rouge_delta = [
        candidate["rouge_l"] - baseline["rouge_l"]
        for candidate, baseline in zip(
            scored["vi_qwen2_3b_rag"], scored["qwen35_2b_lora_fulltrain"]
        )
    ]
    scoring = config.section("scoring")
    report = {
        "schema_version": "1.0",
        "experiment_id": config.raw["experiment_id"],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "sample_size": len(rows),
        "comparison": "Vi-Qwen2-3B-RAG vs Qwen3.5-2B LoRA full-train-5636",
        "metrics": metrics,
        "paired_delta_viqwen_minus_qwen_lora": {
            "meteor_mean": fmean(meteor_delta),
            "meteor_bootstrap_95_ci": _bootstrap_ci(
                meteor_delta,
                seed=f"{scoring['bootstrap_seed']}:meteor",
                iterations=scoring["bootstrap_iterations"],
            ),
            "rouge_l_mean": fmean(rouge_delta),
            "rouge_l_bootstrap_95_ci": _bootstrap_ci(
                rouge_delta,
                seed=f"{scoring['bootstrap_seed']}:rouge_l",
                iterations=scoring["bootstrap_iterations"],
            ),
        },
        "smoke_leader": max(
            metrics, key=lambda key: (metrics[key]["meteor"], metrics[key]["rouge_l"])
        ),
        "promotion_allowed": False,
        "holdout_untouched": True,
        "evidence": {
            "config_sha256": config.config_sha256,
            "sample_ids_sha256": dev_cfg["sample_ids_sha256"],
            "generation_results_sha256": file_sha256(root / "results.jsonl"),
            "scores_sha256": file_sha256(root / "scores.jsonl"),
            "control_results_sha256": control["evaluation_results_sha256"],
            "candidate_revision": config.section("candidate_generator")["revision"],
            "candidate_parameter_count": expected_count,
        },
        "warning": (
            "This is a dev-200 generator smoke comparison. It does not use holdout and "
            "cannot formally promote a competition stack."
        ),
    }
    _atomic_json(output_directory / "report.json", report)
    return report


def _validate_control_report(report: dict[str, Any], expected: dict[str, Any]) -> None:
    metrics = report.get("metrics", {}).get("lora_e07", {})
    evidence = report.get("evidence", {})
    if (
        report.get("experiment_id") != expected["report_experiment_id"]
        or report.get("sample_size") != expected["sample_size"]
        or report.get("train_sample_size") != expected["train_sample_size"]
        or report.get("smoke_leader") != expected["control_variant"]
        or evidence.get("config_sha256") != expected["report_config_sha256"]
        or evidence.get("evaluation_results_sha256")
        != expected["evaluation_results_sha256"]
        or evidence.get("adapter_sha256") != expected["adapter_sha256"]
        or evidence.get("training_records_sha256") != expected["training_records_sha256"]
        or metrics.get("meteor") != expected["meteor"]
        or metrics.get("rouge_l") != expected["rouge_l"]
    ):
        raise E07GeneratorAbError("Saved full-train LoRA report changed.")


def _as_text_chat_messages(messages: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Adapt text-only multimodal chat content to Qwen2's string-only template."""

    adapted: list[dict[str, str]] = []
    for message in messages:
        role = message.get("role")
        content = message.get("content")
        if not isinstance(role, str) or not role:
            raise E07GeneratorAbError("Chat message has an invalid role.")
        if isinstance(content, str):
            text = content
        elif isinstance(content, list) and content:
            pieces: list[str] = []
            for item in content:
                if (
                    not isinstance(item, dict)
                    or item.get("type") != "text"
                    or not isinstance(item.get("text"), str)
                ):
                    raise E07GeneratorAbError(
                        "Vi-Qwen comparison accepts text-only chat content."
                    )
                pieces.append(item["text"])
            text = "".join(pieces)
        else:
            raise E07GeneratorAbError("Chat message has invalid content.")
        adapted.append({"role": role, "content": text})
    return adapted


def _ids_sha(ids: list[str]) -> str:
    return hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()
