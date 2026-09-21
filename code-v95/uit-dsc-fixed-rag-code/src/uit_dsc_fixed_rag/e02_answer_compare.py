"""Frozen-generator paired answer evaluation for the two E02 retrieval candidates."""

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
from uit_dsc_fixed_rag.e02_answer import (
    _atomic_json,
    _atomic_jsonl,
    _load_completed,
    _load_dev,
    _read_records,
    _write_results_jsonl,
    _write_state,
    pack_contexts,
)
from uit_dsc_fixed_rag.evaluation.official import (
    ensure_nltk_resources,
    nltk_meteor_score,
    rouge_l_fmeasure,
)
from uit_dsc_fixed_rag.retrieval_diagnostics import select_dev_sample


CODE_VERSION = "0.1.1"
LOGGER = logging.getLogger(__name__)


class E02AnswerCompareError(RuntimeError):
    """Raised when paired answer evaluation cannot safely continue."""


@dataclass(frozen=True)
class E02AnswerCompareConfig:
    raw: dict[str, Any]
    path: Path
    retrieval_experiment_id: str
    retrieval_semantic_sha256: str
    retrieval_execution_sha256: str
    candidates: tuple[str, ...]
    fused_top_k: int
    generator_key: str
    generator_model_id: str
    generator_revision: str
    generator_parameter_count: int
    generator_runtime_unique_parameter_count: int
    generator_dtype: str
    generator_device_map: str
    required_cuda_devices: int
    parameter_limit: int
    stack_totals: dict[str, int]
    dev_path: str
    dev_sha256: str
    sample_seed: str
    sample_size: int
    sample_ids_sha256: str
    max_input_tokens: int
    minimum_contexts: int
    system_prompt: str
    answer_instruction: str
    max_new_tokens: int
    scorer_path: str
    scorer_sha256: str
    bootstrap_seed: str
    bootstrap_iterations: int

    @property
    def config_sha256(self) -> str:
        return file_sha256(self.path)


def load_answer_compare_config(path: Path) -> E02AnswerCompareConfig:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if set(payload) != {
        "schema_version", "experiment_id", "retrieval_input", "generator",
        "parameter_budget", "dev", "context", "decoding", "scoring", "run_contract",
    } or payload.get("schema_version") != "1.0":
        raise ValueError("E02 answer comparison config root is incompatible.")
    if payload.get("experiment_id") != "E02-answer-compare-dev200-v1":
        raise ValueError("Unexpected E02 answer comparison experiment ID.")
    retrieval = _object(payload, "retrieval_input")
    generator = _object(payload, "generator")
    budget = _object(payload, "parameter_budget")
    dev = _object(payload, "dev")
    context = _object(payload, "context")
    decoding = _object(payload, "decoding")
    scoring = _object(payload, "scoring")
    contract = _object(payload, "run_contract")
    candidates = retrieval.get("candidates")
    if candidates != ["embedding_aiteam", "embedding_harrier"]:
        raise ValueError("Answer comparison requires AITeam then Harrier.")
    if retrieval.get("fused_top_k") != 20 or retrieval.get("reranker_used") is not False:
        raise ValueError("E02 answer comparison must consume unrereanked fused top 20.")
    if generator.get("dtype") != "float16" or generator.get("device_map") != "balanced":
        raise ValueError("Generator must use one reviewed dual-GPU float16 copy.")
    if context.get("packing") != "fused-rank-whole-chunks-greedy":
        raise ValueError("Unexpected E02 context packing policy.")
    if decoding != {
        "do_sample": False,
        "enable_thinking": False,
        "max_new_tokens": 384,
        "num_beams": 1,
        "use_cache": True,
    }:
        raise ValueError("Deterministic E02 decoding settings changed.")
    if scoring.get("primary_metric") != "meteor" or scoring.get("secondary_metric") != "rouge_l":
        raise ValueError("Official metric selection changed.")
    if scoring.get("paired_delta") != "embedding_aiteam-minus-embedding_harrier":
        raise ValueError("Paired comparison direction changed.")
    required_contract = {
        "checkpoint_every_questions": 1,
        "atomic_checkpoint_write": True,
        "resume_fail_closed": True,
        "same_generator_prompt_decoding": True,
        "reranker_allowed": False,
        "allow_holdout": False,
        "allow_public": False,
        "allow_external_data": False,
        "allow_synthetic_data": False,
        "allow_model_api": False,
        "promotion_allowed": False,
    }
    if contract != required_contract:
        raise ValueError("E02 answer comparison run contract changed.")
    limit = _positive_int(budget, "exclusive_limit")
    stacks = _object(budget, "candidate_stacks")
    if set(stacks) != set(candidates):
        raise ValueError("Parameter budget does not cover both embeddings.")
    stack_totals: dict[str, int] = {}
    for candidate in candidates:
        stack = _object(stacks, candidate)
        total = _positive_int(stack, "total")
        observed = _positive_int(stack, "embedding") + _positive_int(stack, "generator")
        if total != observed or total >= limit:
            raise ValueError(f"Invalid E02 parameter budget for {candidate}.")
        if _positive_int(stack, "generator") != _positive_int(generator, "parameter_count"):
            raise ValueError("Generator parameter count differs between config sections.")
        stack_totals[candidate] = total
    return E02AnswerCompareConfig(
        raw=payload,
        path=path,
        retrieval_experiment_id=_text(retrieval, "experiment_id"),
        retrieval_semantic_sha256=_sha256(retrieval, "semantic_config_sha256"),
        retrieval_execution_sha256=_sha256(retrieval, "execution_config_sha256"),
        candidates=tuple(candidates),
        fused_top_k=_positive_int(retrieval, "fused_top_k"),
        generator_key=_text(generator, "key"),
        generator_model_id=_text(generator, "model_id"),
        generator_revision=_revision(generator, "revision"),
        generator_parameter_count=_positive_int(generator, "parameter_count"),
        generator_runtime_unique_parameter_count=_positive_int(
            generator, "runtime_unique_parameter_count"
        ),
        generator_dtype=_text(generator, "dtype"),
        generator_device_map=_text(generator, "device_map"),
        required_cuda_devices=_positive_int(generator, "required_cuda_devices"),
        parameter_limit=limit,
        stack_totals=stack_totals,
        dev_path=_text(dev, "path"),
        dev_sha256=_sha256(dev, "sha256"),
        sample_seed=_text(dev, "sample_seed"),
        sample_size=_positive_int(dev, "sample_size"),
        sample_ids_sha256=_sha256(dev, "sample_ids_sha256"),
        max_input_tokens=_positive_int(context, "max_input_tokens"),
        minimum_contexts=_positive_int(context, "minimum_contexts"),
        system_prompt=_text(context, "system_prompt"),
        answer_instruction=_text(context, "answer_instruction"),
        max_new_tokens=_positive_int(decoding, "max_new_tokens"),
        scorer_path=_text(scoring, "official_scorer_path"),
        scorer_sha256=_sha256(scoring, "official_scorer_sha256"),
        bootstrap_seed=_text(scoring, "bootstrap_seed"),
        bootstrap_iterations=_positive_int(scoring, "bootstrap_iterations"),
    )


def validate_inputs(
    *,
    project_root: Path,
    retrieval_directory: Path,
    dev_path: Path,
    config: E02AnswerCompareConfig,
) -> dict[str, Any]:
    scorer = project_root / config.scorer_path
    if not scorer.is_file() or file_sha256(scorer) != config.scorer_sha256:
        raise E02AnswerCompareError("Pinned official scorer is missing or changed.")
    if not dev_path.is_file() or file_sha256(dev_path) != config.dev_sha256:
        raise E02AnswerCompareError("Pinned dev split is missing or changed.")
    report_path = retrieval_directory / "report.json"
    if not report_path.is_file():
        raise E02AnswerCompareError("Saved E02 retrieval report is missing.")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if (
        report.get("experiment_id") != config.retrieval_experiment_id
        or report.get("semantic_config_sha256") != config.retrieval_semantic_sha256
        or report.get("execution_config_sha256") != config.retrieval_execution_sha256
        or report.get("sample_size") != config.sample_size
    ):
        raise E02AnswerCompareError("Saved E02 retrieval report has a different identity.")
    dev = _load_dev(dev_path)
    sample_ids = select_dev_sample(dev, seed=config.sample_seed, size=config.sample_size)
    observed_sample_sha = hashlib.sha256("\n".join(sample_ids).encode("utf-8")).hexdigest()
    if observed_sample_sha != config.sample_ids_sha256:
        raise E02AnswerCompareError("Deterministic dev-200 sample identity changed.")
    result_hashes = {}
    for candidate in config.candidates:
        results_path = retrieval_directory / candidate / "results.jsonl"
        state_path = retrieval_directory / candidate / "state.json"
        if not results_path.is_file() or not state_path.is_file():
            raise E02AnswerCompareError(f"Saved retrieval candidate is missing: {candidate}")
        state = json.loads(state_path.read_text(encoding="utf-8"))
        identity = state.get("run_identity", {})
        if (
            state.get("complete") is not True
            or state.get("completed_count") != config.sample_size
            or identity.get("semantic_config_sha256") != config.retrieval_semantic_sha256
            or identity.get("candidate", {}).get("key") != candidate
        ):
            raise E02AnswerCompareError(f"Saved retrieval state differs: {candidate}")
        rows = [json.loads(line) for line in results_path.read_text(encoding="utf-8").splitlines()]
        if len(rows) != len(sample_ids):
            raise E02AnswerCompareError(f"Saved retrieval count differs: {candidate}")
        for index, (question_id, row) in enumerate(zip(sample_ids, rows)):
            if (
                row.get("question_id") != question_id
                or row.get("sample_index") != index
                or row.get("candidate") != candidate
                or len(row.get("contexts", [])) != config.fused_top_k
            ):
                raise E02AnswerCompareError(f"Saved retrieval row differs: {candidate}/{index}")
        result_hashes[candidate] = file_sha256(results_path)
    return {
        "config_sha256": config.config_sha256,
        "retrieval_report_sha256": file_sha256(report_path),
        "retrieval_results_sha256": result_hashes,
        "dev_sha256": config.dev_sha256,
        "sample_ids_sha256": config.sample_ids_sha256,
        "sample_size": config.sample_size,
        "stack_parameter_totals": config.stack_totals,
        "parameter_limit_exclusive": config.parameter_limit,
    }


def run_paired_generation(
    *,
    model: Any,
    tokenizer: Any,
    retrieval_directory: Path,
    dev_path: Path,
    output_directory: Path,
    config: E02AnswerCompareConfig,
    device_map: dict[str, Any],
) -> dict[str, Any]:
    evidence = validate_inputs(
        project_root=config.path.parents[1],
        retrieval_directory=retrieval_directory,
        dev_path=dev_path,
        config=config,
    )
    dev = _load_dev(dev_path)
    sample_ids = select_dev_sample(dev, seed=config.sample_seed, size=config.sample_size)
    retrieval_rows = {
        candidate: [
            json.loads(line)
            for line in (retrieval_directory / candidate / "results.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        for candidate in config.candidates
    }
    identity = {
        "code_version": CODE_VERSION,
        "config_sha256": config.config_sha256,
        "retrieval_results_sha256": evidence["retrieval_results_sha256"],
        "dev_sha256": config.dev_sha256,
        "sample_ids_sha256": config.sample_ids_sha256,
        "generator": {
            "model_id": config.generator_model_id,
            "revision": config.generator_revision,
            "parameter_count": config.generator_parameter_count,
            "runtime_unique_parameter_count": (
                config.generator_runtime_unique_parameter_count
            ),
            "dtype": config.generator_dtype,
            "device_map": config.generator_device_map,
            "observed_devices": sorted({str(value) for value in device_map.values()}),
        },
        "paired_batch_size": 2,
    }
    identity["identity_sha256"] = _json_sha256(identity)
    generation_root = output_directory / "generation"
    records_directory = generation_root / "records"
    records_directory.mkdir(parents=True, exist_ok=True)
    state_path = generation_root / "state.json"
    completed = _load_completed(records_directory, state_path, identity, sample_ids)
    input_device = getattr(model, "device", None)
    if input_device is None:
        raise E02AnswerCompareError("Cannot determine generator input device.")
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

    for index in range(completed, len(sample_ids)):
        question_id = sample_ids[index]
        rendered_prompts = []
        packed: dict[str, dict[str, Any]] = {}
        for candidate in config.candidates:
            selected, messages, input_tokens = pack_contexts(
                question=dev[question_id]["question"],
                contexts=retrieval_rows[candidate][index]["contexts"],
                config=config,
                token_counter=token_count,
            )
            rendered = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            rendered_prompts.append(rendered)
            packed[candidate] = {"selected": selected, "input_tokens": input_tokens}
        inputs = tokenizer(
            rendered_prompts,
            padding=True,
            add_special_tokens=False,
            return_tensors="pt",
        )
        inputs = {key: value.to(input_device) for key, value in inputs.items()}
        started = time.perf_counter()
        generated = model.generate(
            **inputs,
            do_sample=False,
            num_beams=1,
            max_new_tokens=config.max_new_tokens,
            use_cache=True,
        )
        latency_ms = (time.perf_counter() - started) * 1000
        prompt_width = inputs["input_ids"].shape[1]
        new_ids = generated[:, prompt_width:]
        answers = tokenizer.batch_decode(
            new_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        candidate_payload = {}
        for row_index, candidate in enumerate(config.candidates):
            answer = answers[row_index].strip()
            if not answer:
                raise E02AnswerCompareError(
                    f"Generator returned an empty answer: {candidate}/{question_id}"
                )
            selected = packed[candidate]["selected"]
            candidate_payload[candidate] = {
                "answer": answer,
                "selected_chunk_ids": [context["chunk_id"] for context in selected],
                "selected_context_count": len(selected),
                "input_tokens": packed[candidate]["input_tokens"],
                "output_tokens": len(tokenizer(answer, add_special_tokens=False)["input_ids"]),
            }
        result = {
            "question_id": question_id,
            "sample_index": index,
            "candidates": candidate_payload,
            "paired_generation_latency_ms": latency_ms,
        }
        _atomic_json(records_directory / f"{index:04d}.json", result)
        _write_state(state_path, identity, index + 1, complete=False)
        LOGGER.info(
            "e02_answer_compare_progress completed=%d total=%d question_id=%s "
            "aiteam_input=%d harrier_input=%d latency_ms=%.1f",
            index + 1,
            len(sample_ids),
            question_id,
            candidate_payload["embedding_aiteam"]["input_tokens"],
            candidate_payload["embedding_harrier"]["input_tokens"],
            latency_ms,
        )
    _write_results_jsonl(generation_root / "results.jsonl", records_directory, len(sample_ids))
    _write_state(state_path, identity, len(sample_ids), complete=True)
    return {
        "sample_size": len(sample_ids),
        "paired_batch_size": 2,
        "results_sha256": file_sha256(generation_root / "results.jsonl"),
        "run_identity": identity,
    }


def score_paired_generation(
    *, dev_path: Path, output_directory: Path, config: E02AnswerCompareConfig
) -> dict[str, Any]:
    ensure_nltk_resources(download=False)
    dev = _load_dev(dev_path)
    sample_ids = select_dev_sample(dev, seed=config.sample_seed, size=config.sample_size)
    state_path = output_directory / "generation" / "state.json"
    records_directory = output_directory / "generation" / "records"
    if not state_path.is_file():
        raise E02AnswerCompareError("Generation checkpoint state is missing.")
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if state.get("complete") is not True or state.get("completed_count") != len(sample_ids):
        raise E02AnswerCompareError("Generation is incomplete; refusing partial scoring.")
    rows = _read_records(records_directory, sample_ids)
    scored: dict[str, list[dict[str, Any]]] = {candidate: [] for candidate in config.candidates}
    predictions: dict[str, dict[str, dict[str, str]]] = {
        candidate: {} for candidate in config.candidates
    }
    for row in rows:
        question_id = row["question_id"]
        reference = dev[question_id]["answer"]
        for candidate in config.candidates:
            answer = row["candidates"][candidate]["answer"]
            scored[candidate].append({
                "question_id": question_id,
                "sample_index": row["sample_index"],
                "meteor": nltk_meteor_score(reference, answer),
                "rouge_l": rouge_l_fmeasure(reference, answer),
            })
            predictions[candidate][question_id] = {"answer": answer}
    for candidate in config.candidates:
        _atomic_jsonl(output_directory / f"scores-{candidate}.jsonl", scored[candidate])
        _atomic_json(output_directory / f"predictions-{candidate}.json", predictions[candidate])
    aiteam = scored["embedding_aiteam"]
    harrier = scored["embedding_harrier"]
    meteor_deltas = [left["meteor"] - right["meteor"] for left, right in zip(aiteam, harrier)]
    rouge_deltas = [left["rouge_l"] - right["rouge_l"] for left, right in zip(aiteam, harrier)]
    metrics = {
        candidate: {
            "meteor": fmean(row["meteor"] for row in candidate_rows),
            "rouge_l": fmean(row["rouge_l"] for row in candidate_rows),
        }
        for candidate, candidate_rows in scored.items()
    }
    report = {
        "schema_version": "1.0",
        "experiment_id": "E02-answer-compare-dev200-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "sample_size": len(sample_ids),
        "metrics": metrics,
        "paired_delta_aiteam_minus_harrier": {
            "meteor_mean": fmean(meteor_deltas),
            "meteor_bootstrap_95_ci": _bootstrap_ci(
                meteor_deltas,
                seed=config.bootstrap_seed + ":meteor",
                iterations=config.bootstrap_iterations,
            ),
            "rouge_l_mean": fmean(rouge_deltas),
            "rouge_l_bootstrap_95_ci": _bootstrap_ci(
                rouge_deltas,
                seed=config.bootstrap_seed + ":rouge_l",
                iterations=config.bootstrap_iterations,
            ),
        },
        "runtime": {
            "mean_paired_generation_latency_ms": fmean(
                row["paired_generation_latency_ms"] for row in rows
            ),
            "paired_batch_size": 2,
            "mean_selected_contexts": {
                candidate: fmean(
                    row["candidates"][candidate]["selected_context_count"] for row in rows
                )
                for candidate in config.candidates
            },
        },
        "evidence": {
            "config_sha256": config.config_sha256,
            "dev_sha256": config.dev_sha256,
            "sample_ids_sha256": config.sample_ids_sha256,
            "generation_results_sha256": file_sha256(
                output_directory / "generation" / "results.jsonl"
            ),
        },
        "promotion_allowed": False,
        "warning": "Dev-200 is the E02 answer-level smoke comparison. Apply the predeclared full-dev paired promotion rule before selecting the embedding.",
    }
    _atomic_json(output_directory / "report.json", report)
    return report


def _bootstrap_ci(values: list[float], *, seed: str, iterations: int) -> list[float]:
    if not values or iterations <= 0:
        raise ValueError("Bootstrap inputs must be non-empty and positive.")
    generator = random.Random(hashlib.sha256(seed.encode("utf-8")).digest())
    count = len(values)
    samples = sorted(
        fmean(values[generator.randrange(count)] for _ in range(count))
        for _ in range(iterations)
    )
    return [
        samples[int(0.025 * (iterations - 1))],
        samples[int(0.975 * (iterations - 1))],
    ]


def _json_sha256(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


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
