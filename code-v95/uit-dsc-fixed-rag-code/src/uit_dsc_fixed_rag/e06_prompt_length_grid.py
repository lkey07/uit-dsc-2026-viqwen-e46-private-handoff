"""E06 zero-shot prompt and deterministic answer-length grid."""

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
from uit_dsc_fixed_rag.evaluation.official import (
    ensure_nltk_resources,
    nltk_meteor_score,
    rouge_l_fmeasure,
)
from uit_dsc_fixed_rag.retrieval_diagnostics import select_dev_sample


CODE_VERSION = "0.11.0"
LOGGER = logging.getLogger(__name__)


class E06Error(RuntimeError):
    """Raised when E06 cannot safely continue."""


@dataclass(frozen=True)
class E06Variant:
    key: str
    prompt_profile: str
    max_new_tokens: int
    answer_source: str


@dataclass(frozen=True)
class PackingProfile:
    max_input_tokens: int
    minimum_contexts: int
    system_prompt: str
    answer_instruction: str


@dataclass(frozen=True)
class E06Config:
    raw: dict[str, Any]
    path: Path
    selection: dict[str, Any]
    variants: tuple[E06Variant, ...]
    prompt_profiles: dict[str, dict[str, str]]
    generator_model_id: str
    generator_revision: str
    generator_parameter_count: int
    generator_runtime_unique_parameter_count: int
    required_cuda_devices: int
    parameter_limit: int
    stack_total: int
    dev_path: str
    dev_sha256: str
    sample_seed: str
    sample_size: int
    sample_ids_sha256: str
    max_input_tokens: int
    minimum_contexts: int
    worker_count: int
    scorer_path: str
    scorer_sha256: str
    control_variant: str
    bootstrap_seed: str
    bootstrap_iterations: int

    @property
    def config_sha256(self) -> str:
        return file_sha256(self.path)


def load_e06_config(path: Path) -> E06Config:
    payload = json.loads(path.read_text(encoding="utf-8"))
    required_root = {
        "schema_version", "experiment_id", "selection_source", "variants",
        "prompt_profiles", "generator", "parameter_budget", "dev", "context",
        "decoding", "execution", "scoring", "run_contract",
    }
    if set(payload) != required_root or payload.get("schema_version") != "1.0":
        raise ValueError("E06 config root is incompatible.")
    if payload.get("experiment_id") != "E06-prompt-length-grid-aiteam-dev200-v1":
        raise ValueError("Unexpected E06 experiment ID.")

    selection = _object(payload, "selection_source")
    if selection != {
        "mode": "operator-selected-ranked-top12-from-e05-smoke",
        "report_experiment_id": "E05-context-grid-aiteam-dev200-v1",
        "report_config_sha256": "d9d050201b7b303b77af359bf85a2d9aad3db3b61172c8e8861cdb7ff3f56047",
        "generation_results_sha256": "47443101ae3a696cf047d235f10fe94b961c1557308af304155ca62a7aed5df3",
        "prepared_results_sha256": "06329045dcbd68ce7aa98dcdbb344434c0ea64a468102392c0aa8ad156a88c4e",
        "sample_size": 200,
        "selected_embedding": "embedding_aiteam",
        "sparse_weight": 0.5,
        "dense_weight": 0.5,
        "reranker": None,
        "selected_context_variant": "ranked_top12",
        "meteor": 0.37659844658685926,
        "rouge_l": 0.4321430184618054,
        "promotion_rule_satisfied": False,
    }:
        raise ValueError("E06 operator selection evidence changed.")

    variants_payload = payload.get("variants")
    if not isinstance(variants_payload, list) or len(variants_payload) != 3:
        raise ValueError("E06 requires exactly three prompt/length variants.")
    variants = tuple(
        E06Variant(
            key=_text(item, "key"),
            prompt_profile=_text(item, "prompt_profile"),
            max_new_tokens=_positive_int(item, "max_new_tokens"),
            answer_source=_text(item, "answer_source"),
        )
        for item in variants_payload
        if isinstance(item, dict)
    )
    if tuple(
        (row.key, row.prompt_profile, row.max_new_tokens, row.answer_source)
        for row in variants
    ) != (
        ("control_e05_ranked_top12", "baseline_e05", 384, "reuse-e05-ranked-top12"),
        ("structured_prompt_384", "structured_evidence_v1", 384, "generate-e06"),
        ("structured_prompt_256", "structured_evidence_v1", 256, "generate-e06"),
    ):
        raise ValueError("E06 grid changed from the experiment plan.")

    prompts = _object(payload, "prompt_profiles")
    if set(prompts) != {"baseline_e05", "structured_evidence_v1"}:
        raise ValueError("E06 prompt profile set changed.")
    prompt_profiles: dict[str, dict[str, str]] = {}
    for key, value in prompts.items():
        if not isinstance(value, dict) or set(value) != {"system_prompt", "answer_instruction"}:
            raise ValueError(f"E06 prompt profile is incompatible: {key}")
        prompt_profiles[key] = {
            "system_prompt": _text(value, "system_prompt"),
            "answer_instruction": _text(value, "answer_instruction"),
        }

    generator = _object(payload, "generator")
    if (
        generator.get("key") != "generator_qwen35_2b"
        or generator.get("model_id") != "Qwen/Qwen3.5-2B"
        or generator.get("dtype") != "float16"
        or generator.get("device_mode") != "replicated-single-gpu-workers"
    ):
        raise ValueError("E06 generator contract changed.")
    budget = _object(payload, "parameter_budget")
    limit = _positive_int(budget, "exclusive_limit")
    stack_total = _positive_int(budget, "stack_total")
    observed_total = _positive_int(budget, "embedding") + _positive_int(budget, "generator")
    if (
        stack_total != observed_total or stack_total >= limit
        or _positive_int(budget, "generator") != _positive_int(generator, "parameter_count")
    ):
        raise ValueError("E06 stack violates the parameter budget.")

    dev = _object(payload, "dev")
    context = _object(payload, "context")
    if context != {
        "source_variant": "ranked_top12", "candidate_limit": 12,
        "max_input_tokens": 8192, "packing": "ordered-whole-chunks-greedy",
        "minimum_contexts": 1,
    }:
        raise ValueError("E06 context selection changed from E05 winner.")
    decoding = _object(payload, "decoding")
    if decoding != {
        "do_sample": False, "enable_thinking": False,
        "num_beams": 1, "use_cache": True,
    }:
        raise ValueError("E06 deterministic decoding changed.")
    execution = _object(payload, "execution")
    if execution != {
        "worker_count": 2, "partition": "sample-index-mod-worker-count",
        "prompts_per_generate_call": 1,
    }:
        raise ValueError("E06 dual-GPU execution contract changed.")
    scoring = _object(payload, "scoring")
    if (
        scoring.get("primary_metric") != "meteor"
        or scoring.get("secondary_metric") != "rouge_l"
        or scoring.get("control_variant") != "control_e05_ranked_top12"
    ):
        raise ValueError("E06 scoring contract changed.")
    contract = _object(payload, "run_contract")
    if contract != {
        "retrieval_rankings_fixed": True,
        "context_selection_fixed": True,
        "reranker_allowed": False,
        "reuse_control_answer_exactly": True,
        "sequential_one_factor_comparisons": True,
        "checkpoint_every_questions": 1,
        "atomic_checkpoint_write": True,
        "resume_fail_closed": True,
        "official_dev_answers_scoring_only": True,
        "allow_holdout": False,
        "allow_public": False,
        "allow_external_data": False,
        "allow_synthetic_data": False,
        "allow_model_api": False,
        "promotion_allowed": False,
    }:
        raise ValueError("E06 run contract changed.")

    return E06Config(
        raw=payload,
        path=path,
        selection=dict(selection),
        variants=variants,
        prompt_profiles=prompt_profiles,
        generator_model_id=_text(generator, "model_id"),
        generator_revision=_revision(generator, "revision"),
        generator_parameter_count=_positive_int(generator, "parameter_count"),
        generator_runtime_unique_parameter_count=_positive_int(
            generator, "runtime_unique_parameter_count"
        ),
        required_cuda_devices=_positive_int(generator, "required_cuda_devices"),
        parameter_limit=limit,
        stack_total=stack_total,
        dev_path=_text(dev, "path"),
        dev_sha256=_sha256(dev, "sha256"),
        sample_seed=_text(dev, "sample_seed"),
        sample_size=_positive_int(dev, "sample_size"),
        sample_ids_sha256=_sha256(dev, "sample_ids_sha256"),
        max_input_tokens=_positive_int(context, "max_input_tokens"),
        minimum_contexts=_positive_int(context, "minimum_contexts"),
        worker_count=_positive_int(execution, "worker_count"),
        scorer_path=_text(scoring, "official_scorer_path"),
        scorer_sha256=_sha256(scoring, "official_scorer_sha256"),
        control_variant=_text(scoring, "control_variant"),
        bootstrap_seed=_text(scoring, "bootstrap_seed"),
        bootstrap_iterations=_positive_int(scoring, "bootstrap_iterations"),
    )


def packing_profile(config: E06Config, variant: E06Variant) -> PackingProfile:
    prompt = config.prompt_profiles[variant.prompt_profile]
    return PackingProfile(
        max_input_tokens=config.max_input_tokens,
        minimum_contexts=config.minimum_contexts,
        system_prompt=prompt["system_prompt"],
        answer_instruction=prompt["answer_instruction"],
    )


def validate_preflight(
    *, project_root: Path, selection_directory: Path, dev_path: Path,
    config: E06Config,
) -> dict[str, Any]:
    scorer = project_root / config.scorer_path
    if not scorer.is_file() or file_sha256(scorer) != config.scorer_sha256:
        raise E06Error("Pinned official scorer is missing or changed.")
    if not dev_path.is_file() or file_sha256(dev_path) != config.dev_sha256:
        raise E06Error("Pinned dev split is missing or changed.")
    sample_ids = select_dev_sample(
        _load_dev(dev_path), seed=config.sample_seed, size=config.sample_size
    )
    sample_sha = hashlib.sha256("\n".join(sample_ids).encode("utf-8")).hexdigest()
    if sample_sha != config.sample_ids_sha256:
        raise E06Error("Deterministic E06 sample identity changed.")

    report_path = selection_directory / "report.json"
    prepared_path = selection_directory / "prepared" / "results.jsonl"
    generation_path = selection_directory / "generation" / "results.jsonl"
    if not all(path.is_file() for path in (report_path, prepared_path, generation_path)):
        raise E06Error("Saved E05 selection artifact is incomplete.")
    if file_sha256(prepared_path) != config.selection["prepared_results_sha256"]:
        raise E06Error("Saved E05 ranked-top12 contexts changed.")
    if file_sha256(generation_path) != config.selection["generation_results_sha256"]:
        raise E06Error("Saved E05 generation results changed.")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    _validate_selection_report(report, config)
    prepared_rows = _read_jsonl(prepared_path)
    generation_rows = _read_jsonl(generation_path)
    _validate_selection_rows(prepared_rows, generation_rows, sample_ids)
    return {
        "config_sha256": config.config_sha256,
        "dev_sha256": config.dev_sha256,
        "sample_ids_sha256": sample_sha,
        "sample_size": len(sample_ids),
        "selection_report_sha256": file_sha256(report_path),
        "selection_prepared_results_sha256": file_sha256(prepared_path),
        "selection_generation_results_sha256": file_sha256(generation_path),
        "selection_mode": config.selection["mode"],
        "selected_stack": {
            "embedding": "embedding_aiteam", "sparse_weight": 0.5,
            "dense_weight": 0.5, "reranker": None, "contexts": "ranked_top12",
        },
        "stack_parameter_total": config.stack_total,
        "parameter_limit_exclusive": config.parameter_limit,
    }


def prepare_ranked_top12(
    *, selection_directory: Path, dev_path: Path, output_directory: Path,
    config: E06Config, preflight: dict[str, Any],
) -> dict[str, Any]:
    sample_ids = select_dev_sample(
        _load_dev(dev_path), seed=config.sample_seed, size=config.sample_size
    )
    source_path = selection_directory / "prepared" / "results.jsonl"
    if file_sha256(source_path) != preflight.get("selection_prepared_results_sha256"):
        raise E06Error("E05 contexts changed after E06 preflight.")
    source_rows = _read_jsonl(source_path)
    rows = []
    for index, question_id in enumerate(sample_ids):
        source = source_rows[index]["variants"]["ranked_top12"]
        contexts = source["contexts"]
        if len(contexts) != 12:
            raise E06Error(f"E05 ranked-top12 context count changed: {question_id}")
        rows.append({
            "question_id": question_id,
            "sample_index": index,
            "contexts": contexts,
            "source_fused_ranks": source["source_fused_ranks"],
        })
    root = output_directory / "prepared"
    root.mkdir(parents=True, exist_ok=True)
    _atomic_jsonl(root / "results.jsonl", rows)
    identity = {
        "code_version": CODE_VERSION,
        "stage": "reuse-e05-ranked-top12-contexts",
        "config_sha256": config.config_sha256,
        "selection_prepared_results_sha256": preflight[
            "selection_prepared_results_sha256"
        ],
        "sample_ids_sha256": config.sample_ids_sha256,
    }
    identity["identity_sha256"] = _json_sha256(identity)
    summary = {
        "schema_version": "1.0",
        "sample_size": len(rows),
        "contexts_per_question": 12,
        "results_sha256": file_sha256(root / "results.jsonl"),
        "run_identity": identity,
    }
    _atomic_json(root / "summary.json", summary)
    return summary


def load_generator_on_device(
    config: E06Config, device: str
) -> tuple[Any, Any, dict[str, Any]]:
    try:
        import torch
        from transformers import AutoProcessor
        try:
            from transformers import AutoModelForMultimodalLM as AutoGenerator
        except ImportError:  # pragma: no cover
            from transformers import AutoModelForImageTextToText as AutoGenerator
    except ImportError as exc:  # pragma: no cover
        raise E06Error("Install Transformers and Accelerate for E06 generation.") from exc
    if device not in {"cuda:0", "cuda:1"} or torch.cuda.device_count() != 2:
        raise E06Error("E06 generation workers require GPU T4 x2.")
    torch.cuda.set_device(int(device[-1]))
    random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    processor = AutoProcessor.from_pretrained(
        config.generator_model_id, revision=config.generator_revision,
        trust_remote_code=False,
    )
    tokenizer = processor.tokenizer
    model = AutoGenerator.from_pretrained(
        config.generator_model_id, revision=config.generator_revision,
        dtype=torch.float16, device_map={"": device}, low_cpu_mem_usage=True,
        trust_remote_code=False,
    )
    model.eval()
    observed = sum(parameter.numel() for parameter in model.parameters())
    if observed != config.generator_runtime_unique_parameter_count:
        raise E06Error(
            "Generator runtime unique parameter count changed: "
            f"{observed} != {config.generator_runtime_unique_parameter_count}"
        )
    try:
        device_map = _validate_worker_placement(model, device)
    except RuntimeError as exc:
        raise E06Error(str(exc)) from exc
    LOGGER.info(
        "e06_generator_loaded device=%s runtime_unique_parameters=%d "
        "published_tensor_parameters=%d",
        device, observed, config.generator_parameter_count,
    )
    return model, tokenizer, device_map


def run_generation_worker(
    *, worker_rank: int, device: str, model: Any, tokenizer: Any,
    selection_directory: Path, dev_path: Path, output_directory: Path,
    config: E06Config, device_map: dict[str, Any],
) -> dict[str, Any]:
    if worker_rank not in range(config.worker_count) or device != f"cuda:{worker_rank}":
        raise E06Error("E06 worker rank/device assignment changed.")
    prepared_path = output_directory / "prepared" / "results.jsonl"
    prepared_summary_path = output_directory / "prepared" / "summary.json"
    if not prepared_path.is_file() or not prepared_summary_path.is_file():
        raise E06Error("Run E06 prepare before generation.")
    prepared_summary = json.loads(prepared_summary_path.read_text(encoding="utf-8"))
    if file_sha256(prepared_path) != prepared_summary.get("results_sha256"):
        raise E06Error("Prepared E06 contexts changed.")
    prepared_rows = _read_jsonl(prepared_path)
    source_path = selection_directory / "generation" / "results.jsonl"
    if file_sha256(source_path) != config.selection["generation_results_sha256"]:
        raise E06Error("E05 control answers changed before E06 generation.")
    source_rows = _read_jsonl(source_path)
    dev = _load_dev(dev_path)
    sample_ids = select_dev_sample(dev, seed=config.sample_seed, size=config.sample_size)
    if [row.get("question_id") for row in prepared_rows] != sample_ids:
        raise E06Error("Prepared E06 question order changed.")

    assigned = [index for index in range(len(sample_ids)) if index % 2 == worker_rank]
    identity = {
        "code_version": CODE_VERSION,
        "stage": "dual-gpu-generator-worker-two-e06-variants",
        "config_sha256": config.config_sha256,
        "prepared_results_sha256": file_sha256(prepared_path),
        "control_results_sha256": config.selection["generation_results_sha256"],
        "sample_ids_sha256": config.sample_ids_sha256,
        "worker_rank": worker_rank,
        "device": device,
        "assigned_indices_sha256": hashlib.sha256(
            ",".join(str(index) for index in assigned).encode("ascii")
        ).hexdigest(),
        "generator": {
            "model_id": config.generator_model_id,
            "revision": config.generator_revision,
            "parameter_count": config.generator_parameter_count,
            "runtime_unique_parameter_count": config.generator_runtime_unique_parameter_count,
            "observed_device_map": device_map,
        },
    }
    identity["identity_sha256"] = _json_sha256(identity)
    root = output_directory / "generation"
    records = root / "records"
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
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
        return len(tokenizer(rendered, add_special_tokens=False)["input_ids"])

    control_variant = config.variants[0]
    control_packing = packing_profile(config, control_variant)
    for position in range(completed, len(assigned)):
        sample_index = assigned[position]
        question_id = sample_ids[sample_index]
        contexts = prepared_rows[sample_index]["contexts"]
        source_control = source_rows[sample_index]["variants"]["ranked_top12"]
        selected_control, _, control_tokens = pack_contexts(
            question=dev[question_id]["question"], contexts=contexts,
            config=control_packing, token_counter=token_count,
        )
        if (
            [context["chunk_id"] for context in selected_control]
            != source_control.get("selected_chunk_ids")
            or control_tokens != source_control.get("input_tokens")
        ):
            raise E06Error(f"E05 control prompt lineage changed: {question_id}")
        answers: dict[str, Any] = {
            control_variant.key: {
                **source_control,
                "answer_source": "reused-e05-ranked-top12",
                "generation_latency_ms": None,
            }
        }
        for variant in config.variants[1:]:
            selected, messages, input_tokens = pack_contexts(
                question=dev[question_id]["question"], contexts=contexts,
                config=packing_profile(config, variant), token_counter=token_count,
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
                max_new_tokens=variant.max_new_tokens, use_cache=True,
            )
            latency_ms = (time.perf_counter() - started) * 1000
            prompt_width = inputs["input_ids"].shape[1]
            answer = tokenizer.decode(
                generated[0, prompt_width:], skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            ).strip()
            if not answer:
                raise E06Error(f"Generator returned an empty answer: {variant.key}/{question_id}")
            answers[variant.key] = {
                "answer": answer,
                "answer_source": "generated-e06",
                "prompt_profile": variant.prompt_profile,
                "max_new_tokens": variant.max_new_tokens,
                "selected_chunk_ids": [context["chunk_id"] for context in selected],
                "selected_context_count": len(selected),
                "input_tokens": input_tokens,
                "output_tokens": len(tokenizer(answer, add_special_tokens=False)["input_ids"]),
                "generation_latency_ms": latency_ms,
            }
        _atomic_json(records / f"{sample_index:04d}.json", {
            "question_id": question_id,
            "sample_index": sample_index,
            "worker_rank": worker_rank,
            "worker_identity_sha256": identity["identity_sha256"],
            "variants": answers,
        })
        _write_worker_state(state_path, identity, position + 1, len(assigned))
        LOGGER.info(
            "e06_generation_progress worker=%d device=%s worker_completed=%d "
            "worker_total=%d question_id=%s",
            worker_rank, device, position + 1, len(assigned), question_id,
        )
    _write_worker_state(state_path, identity, len(assigned), len(assigned))
    return {"worker_rank": worker_rank, "device": device, "completed": len(assigned)}


def finalize_generation(
    *, output_directory: Path, dev_path: Path, config: E06Config,
) -> dict[str, Any]:
    sample_ids = select_dev_sample(
        _load_dev(dev_path), seed=config.sample_seed, size=config.sample_size
    )
    root = output_directory / "generation"
    rows = _validate_complete_records(root, sample_ids, config)
    _atomic_jsonl(root / "results.jsonl", rows)
    summary = {
        "schema_version": "1.0",
        "sample_size": len(rows),
        "answer_count": len(rows) * len(config.variants),
        "reused_control_answers": len(rows),
        "newly_generated_answers": len(rows) * 2,
        "results_sha256": file_sha256(root / "results.jsonl"),
    }
    _atomic_json(root / "summary.json", summary)
    return summary


def score_grid(
    *, output_directory: Path, dev_path: Path, config: E06Config,
) -> dict[str, Any]:
    ensure_nltk_resources(download=False)
    results_path = output_directory / "generation" / "results.jsonl"
    prepared_path = output_directory / "prepared" / "results.jsonl"
    if not results_path.is_file() or not prepared_path.is_file():
        raise E06Error("Finalize E06 generation before scoring.")
    rows = _read_jsonl(results_path)
    dev = _load_dev(dev_path)
    sample_ids = select_dev_sample(dev, seed=config.sample_seed, size=config.sample_size)
    if [row.get("question_id") for row in rows] != sample_ids:
        raise E06Error("E06 generation order changed before scoring.")
    scored = {variant.key: [] for variant in config.variants}
    for row in rows:
        reference = dev[row["question_id"]]["answer"]
        for variant in config.variants:
            answer = row["variants"][variant.key]["answer"]
            scored[variant.key].append({
                "question_id": row["question_id"],
                "sample_index": row["sample_index"],
                "meteor": nltk_meteor_score(reference, answer),
                "rouge_l": rouge_l_fmeasure(reference, answer),
            })
    metrics: dict[str, Any] = {}
    for variant in config.variants:
        key = variant.key
        _atomic_jsonl(output_directory / f"scores-{key}.jsonl", scored[key])
        _atomic_json(output_directory / f"predictions-{key}.json", {
            row["question_id"]: {"answer": source["variants"][key]["answer"]}
            for row, source in zip(scored[key], rows)
        })
        latencies = [
            row["variants"][key].get("generation_latency_ms") for row in rows
            if isinstance(row["variants"][key].get("generation_latency_ms"), (int, float))
        ]
        metrics[key] = {
            "prompt_profile": variant.prompt_profile,
            "max_new_tokens": variant.max_new_tokens,
            "meteor": fmean(row["meteor"] for row in scored[key]),
            "rouge_l": fmean(row["rouge_l"] for row in scored[key]),
            "mean_generation_latency_ms": fmean(latencies) if latencies else None,
            "mean_selected_contexts": fmean(
                row["variants"][key]["selected_context_count"] for row in rows
            ),
            "mean_input_tokens": fmean(row["variants"][key]["input_tokens"] for row in rows),
            "mean_output_tokens": fmean(row["variants"][key]["output_tokens"] for row in rows),
            "answer_source": variant.answer_source,
        }
    control = config.control_variant
    if (
        metrics[control]["meteor"] != config.selection["meteor"]
        or metrics[control]["rouge_l"] != config.selection["rouge_l"]
    ):
        raise E06Error("Re-scored reused E05 control metrics changed.")

    comparisons = (
        ("structured_prompt_384", control),
        ("structured_prompt_256", "structured_prompt_384"),
        ("structured_prompt_256", control),
    )
    paired = {}
    for candidate, baseline in comparisons:
        meteor = [
            left["meteor"] - right["meteor"]
            for left, right in zip(scored[candidate], scored[baseline])
        ]
        rouge = [
            left["rouge_l"] - right["rouge_l"]
            for left, right in zip(scored[candidate], scored[baseline])
        ]
        key = f"{candidate}-minus-{baseline}"
        paired[key] = {
            "meteor_mean": fmean(meteor),
            "meteor_bootstrap_95_ci": _bootstrap_ci(
                meteor, seed=f"{config.bootstrap_seed}:{key}:meteor",
                iterations=config.bootstrap_iterations,
            ),
            "rouge_l_mean": fmean(rouge),
            "rouge_l_bootstrap_95_ci": _bootstrap_ci(
                rouge, seed=f"{config.bootstrap_seed}:{key}:rouge_l",
                iterations=config.bootstrap_iterations,
            ),
        }
    smoke_leader = max(
        metrics, key=lambda key: (metrics[key]["meteor"], metrics[key]["rouge_l"])
    )
    report = {
        "schema_version": "1.0",
        "experiment_id": "E06-prompt-length-grid-aiteam-dev200-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "sample_size": len(rows),
        "fixed_stack": {
            "embedding": "embedding_aiteam", "sparse_weight": 0.5,
            "dense_weight": 0.5, "reranker": None, "contexts": "ranked_top12",
            "generator": "Qwen/Qwen3.5-2B",
        },
        "metrics": metrics,
        "control_variant": control,
        "paired_deltas": paired,
        "smoke_leader": smoke_leader,
        "promotion_allowed": False,
        "evidence": {
            "config_sha256": config.config_sha256,
            "dev_sha256": config.dev_sha256,
            "sample_ids_sha256": config.sample_ids_sha256,
            "prepared_results_sha256": file_sha256(prepared_path),
            "generation_results_sha256": file_sha256(results_path),
            "reused_e05_generation_results_sha256": config.selection[
                "generation_results_sha256"
            ],
            "reused_e05_prepared_results_sha256": config.selection[
                "prepared_results_sha256"
            ],
        },
        "warning": "Dev-200 is an E06 smoke grid; formal promotion requires full dev.",
    }
    _atomic_json(output_directory / "report.json", report)
    return report


def _validate_selection_report(report: dict[str, Any], config: E06Config) -> None:
    selection = config.selection
    metrics = report.get("metrics", {}).get(selection["selected_context_variant"], {})
    evidence = report.get("evidence", {})
    if (
        report.get("experiment_id") != selection["report_experiment_id"]
        or report.get("sample_size") != selection["sample_size"]
        or report.get("smoke_leader") != selection["selected_context_variant"]
        or report.get("promotion_allowed") is not False
        or report.get("selected_embedding") != selection["selected_embedding"]
        or report.get("reranker") is not None
        or evidence.get("config_sha256") != selection["report_config_sha256"]
        or evidence.get("generation_results_sha256")
        != selection["generation_results_sha256"]
        or evidence.get("prepared_results_sha256")
        != selection["prepared_results_sha256"]
        or metrics.get("meteor") != selection["meteor"]
        or metrics.get("rouge_l") != selection["rouge_l"]
    ):
        raise E06Error("Saved E05 report differs from the ranked-top12 selection evidence.")


def _validate_selection_rows(
    prepared_rows: list[dict[str, Any]], generation_rows: list[dict[str, Any]],
    sample_ids: list[str],
) -> None:
    if len(prepared_rows) != len(sample_ids) or len(generation_rows) != len(sample_ids):
        raise E06Error("E05 selection record count differs from E06.")
    for index, question_id in enumerate(sample_ids):
        prepared = prepared_rows[index]
        generated = generation_rows[index]
        source = prepared.get("variants", {}).get("ranked_top12", {})
        contexts = source.get("contexts")
        answer = generated.get("variants", {}).get("ranked_top12", {}).get("answer")
        if (
            prepared.get("sample_index") != index
            or prepared.get("question_id") != question_id
            or generated.get("sample_index") != index
            or generated.get("question_id") != question_id
            or not isinstance(contexts, list) or len(contexts) != 12
            or source.get("source_fused_ranks") != list(range(1, 13))
            or len({context.get("chunk_id") for context in contexts}) != 12
            or any(not isinstance(context.get("text"), str) for context in contexts)
            or not isinstance(answer, str) or not answer.strip()
        ):
            raise E06Error(f"E05 ranked-top12 selection row is incompatible: {index}")


def _validate_complete_records(
    root: Path, sample_ids: list[str], config: E06Config,
) -> list[dict[str, Any]]:
    records = root / "records"
    expected_variants = {variant.key for variant in config.variants}
    rows = []
    for index, question_id in enumerate(sample_ids):
        path = records / f"{index:04d}.json"
        if not path.is_file():
            raise E06Error(f"Generation record is missing: {index}")
        row = json.loads(path.read_text(encoding="utf-8"))
        if (
            row.get("sample_index") != index
            or row.get("question_id") != question_id
            or set(row.get("variants", {})) != expected_variants
        ):
            raise E06Error(f"Generation record identity changed: {index}")
        rows.append(row)
    for rank in range(config.worker_count):
        state_path = root / f"worker-{rank}-state.json"
        if not state_path.is_file() or json.loads(
            state_path.read_text(encoding="utf-8")
        ).get("complete") is not True:
            raise E06Error(f"Generation worker is incomplete: {rank}")
    return rows


def _object(payload: dict[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"{key} must be an object.")
    return value


def _text(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be non-blank text.")
    return value


def _positive_int(payload: dict[str, Any], key: str) -> int:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{key} must be a positive integer.")
    return value


def _sha256(payload: dict[str, Any], key: str) -> str:
    value = _text(payload, key).lower()
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{key} must be a SHA-256 hex digest.")
    return value


def _revision(payload: dict[str, Any], key: str) -> str:
    value = _text(payload, key).lower()
    if len(value) != 40 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{key} must be a 40-character immutable commit.")
    return value
