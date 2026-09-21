"""Paired AITeam/Harrier reranked answer-level evaluation on dev-200."""

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
    _read_records,
    _write_results_jsonl,
    _write_state,
    pack_contexts,
)
from uit_dsc_fixed_rag.e02_compare import _load_dev
from uit_dsc_fixed_rag.evaluation.official import (
    ensure_nltk_resources,
    nltk_meteor_score,
    rouge_l_fmeasure,
)
from uit_dsc_fixed_rag.retrieval_diagnostics import select_dev_sample


CODE_VERSION = "0.1.0"
LOGGER = logging.getLogger(__name__)


class E04Error(RuntimeError):
    """Raised when the paired reranker/generator run cannot safely continue."""


@dataclass(frozen=True)
class E04Config:
    raw: dict[str, Any]
    path: Path
    candidates: tuple[str, ...]
    compare_config_path: str
    compare_config_sha256: str
    reranker_key: str
    reranker_model_id: str
    reranker_revision: str
    reranker_parameter_count: int
    reranker_max_tokens: int
    reranker_input_top_k: int
    reranker_output_top_k: int
    reranker_batch_size: int
    generator_key: str
    generator_model_id: str
    generator_revision: str
    generator_parameter_count: int
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


def load_e04_config(path: Path) -> E04Config:
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected_root = {
        "schema_version", "experiment_id", "retrieval", "reranker", "generator",
        "parameter_budget", "dev", "context", "decoding", "scoring", "run_contract",
    }
    if set(payload) != expected_root or payload.get("schema_version") != "1.0":
        raise ValueError("E04 config root is incompatible.")
    if payload.get("experiment_id") != "E04-rerank-answer-compare-dev200-v1":
        raise ValueError("Unexpected E04 experiment ID.")

    retrieval = _object(payload, "retrieval")
    reranker = _object(payload, "reranker")
    generator = _object(payload, "generator")
    budget = _object(payload, "parameter_budget")
    dev = _object(payload, "dev")
    context = _object(payload, "context")
    decoding = _object(payload, "decoding")
    scoring = _object(payload, "scoring")
    contract = _object(payload, "run_contract")

    candidates = retrieval.get("candidates")
    if candidates != ["embedding_aiteam", "embedding_harrier"]:
        raise ValueError("E04 requires the two pinned embeddings in the reviewed order.")
    if retrieval.get("candidate_k_per_branch") != 40 or retrieval.get("fused_top_k") != 20:
        raise ValueError("Frozen E02 retrieval depth changed.")
    if retrieval.get("rrf_constant") != 60 or retrieval.get("branch_weights") != {
        "sparse": 1.0, "dense": 1.0
    }:
        raise ValueError("Frozen E02 RRF settings changed.")
    if reranker.get("score") != "sequence-classification-logit-descending":
        raise ValueError("Unexpected reranker scoring function.")
    if reranker.get("input_top_k") != 20 or reranker.get("output_top_k") != 20:
        raise ValueError("E04 v1 reranks exactly the fused top 20.")
    if generator.get("dtype") != "float16" or generator.get("device_map") != "balanced":
        raise ValueError("Generator must use one reviewed dual-GPU float16 copy.")
    if context.get("packing") != "reranked-whole-chunks-greedy":
        raise ValueError("Unexpected context packing policy.")
    if decoding != {
        "do_sample": False,
        "enable_thinking": False,
        "max_new_tokens": 384,
        "num_beams": 1,
        "use_cache": True,
    }:
        raise ValueError("Deterministic decoding settings changed.")
    if scoring.get("primary_metric") != "meteor" or scoring.get("secondary_metric") != "rouge_l":
        raise ValueError("Official metric selection changed.")
    if scoring.get("paired_delta") != "embedding_aiteam-minus-embedding_harrier":
        raise ValueError("Paired comparison direction changed.")
    required_contract = {
        "checkpoint_every_questions": 1,
        "atomic_checkpoint_write": True,
        "resume_fail_closed": True,
        "same_reranker_generator_prompt_decoding": True,
        "allow_holdout": False,
        "allow_public": False,
        "allow_external_data": False,
        "allow_synthetic_data": False,
        "allow_model_api": False,
        "promotion_allowed": False,
    }
    if contract != required_contract:
        raise ValueError("E04 run contract changed.")

    limit = _positive_int(budget, "exclusive_limit")
    stack_payload = _object(budget, "candidate_stacks")
    if set(stack_payload) != set(candidates):
        raise ValueError("Parameter budget does not cover both candidates.")
    stack_totals: dict[str, int] = {}
    for candidate in candidates:
        stack = _object(stack_payload, candidate)
        total = _positive_int(stack, "total")
        observed = sum(_positive_int(stack, key) for key in ("embedding", "reranker", "generator"))
        if total != observed or total >= limit:
            raise ValueError(f"Invalid parameter budget for {candidate}.")
        stack_totals[candidate] = total
    if any(
        _positive_int(_object(stack_payload, candidate), "reranker")
        != _positive_int(reranker, "parameter_count")
        for candidate in candidates
    ):
        raise ValueError("Reranker parameter counts disagree.")
    if any(
        _positive_int(_object(stack_payload, candidate), "generator")
        != _positive_int(generator, "parameter_count")
        for candidate in candidates
    ):
        raise ValueError("Generator parameter counts disagree.")

    return E04Config(
        raw=payload,
        path=path,
        candidates=tuple(candidates),
        compare_config_path=_text(retrieval, "compare_config_path"),
        compare_config_sha256=_sha256(retrieval, "compare_config_sha256"),
        reranker_key=_text(reranker, "key"),
        reranker_model_id=_text(reranker, "model_id"),
        reranker_revision=_revision(reranker, "revision"),
        reranker_parameter_count=_positive_int(reranker, "parameter_count"),
        reranker_max_tokens=_positive_int(reranker, "trained_pair_max_tokens"),
        reranker_input_top_k=_positive_int(reranker, "input_top_k"),
        reranker_output_top_k=_positive_int(reranker, "output_top_k"),
        reranker_batch_size=_positive_int(reranker, "batch_size"),
        generator_key=_text(generator, "key"),
        generator_model_id=_text(generator, "model_id"),
        generator_revision=_revision(generator, "revision"),
        generator_parameter_count=_positive_int(generator, "parameter_count"),
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


def validate_static_inputs(*, project_root: Path, dev_path: Path, config: E04Config) -> dict[str, Any]:
    compare_path = project_root / config.compare_config_path
    scorer_path = project_root / config.scorer_path
    if not compare_path.is_file() or file_sha256(compare_path) != config.compare_config_sha256:
        raise E04Error("Pinned E02 retrieval config is missing or changed.")
    if not scorer_path.is_file() or file_sha256(scorer_path) != config.scorer_sha256:
        raise E04Error("Pinned official scorer is missing or changed.")
    if not dev_path.is_file() or file_sha256(dev_path) != config.dev_sha256:
        raise E04Error("Pinned dev split is missing or changed.")
    dev = _load_dev(dev_path)
    sample_ids = select_dev_sample(dev, seed=config.sample_seed, size=config.sample_size)
    sample_sha = hashlib.sha256("\n".join(sample_ids).encode("utf-8")).hexdigest()
    if sample_sha != config.sample_ids_sha256:
        raise E04Error("Deterministic dev-200 sample identity changed.")
    return {
        "config_sha256": config.config_sha256,
        "compare_config_sha256": config.compare_config_sha256,
        "official_scorer_sha256": config.scorer_sha256,
        "dev_sha256": config.dev_sha256,
        "sample_ids_sha256": sample_sha,
        "sample_size": len(sample_ids),
        "stack_parameter_totals": config.stack_totals,
        "parameter_limit_exclusive": config.parameter_limit,
    }


def load_retrieval_records(
    *, candidate: str, retrieval_directory: Path, dev_path: Path, config: E04Config
) -> tuple[list[dict[str, Any]], list[str], str]:
    if candidate not in config.candidates:
        raise E04Error(f"Unexpected embedding candidate: {candidate}")
    root = retrieval_directory / candidate
    results_path = root / "results.jsonl"
    state_path = root / "state.json"
    if not results_path.is_file() or not state_path.is_file():
        raise E04Error(f"Completed retrieval output is missing: {candidate}")
    state = json.loads(state_path.read_text(encoding="utf-8"))
    identity = state.get("run_identity", {})
    if state.get("complete") is not True or state.get("completed_count") != config.sample_size:
        raise E04Error(f"Retrieval output is incomplete: {candidate}")
    if identity.get("config_sha256") != config.compare_config_sha256:
        raise E04Error(f"Retrieval config differs for {candidate}.")
    if identity.get("candidate", {}).get("key") != candidate:
        raise E04Error(f"Retrieval candidate identity differs for {candidate}.")

    dev = _load_dev(dev_path)
    sample_ids = select_dev_sample(dev, seed=config.sample_seed, size=config.sample_size)
    rows = [json.loads(line) for line in results_path.read_text(encoding="utf-8").splitlines()]
    if len(rows) != len(sample_ids):
        raise E04Error(f"Retrieval count differs from dev-200: {candidate}")
    for index, (question_id, row) in enumerate(zip(sample_ids, rows)):
        if (
            row.get("sample_index") != index
            or row.get("question_id") != question_id
            or row.get("candidate") != candidate
            or len(row.get("contexts", [])) != config.reranker_input_top_k
        ):
            raise E04Error(f"Invalid retrieval row {index}: {candidate}")
    return rows, sample_ids, file_sha256(results_path)


def run_reranking(
    *,
    model: Any,
    tokenizer: Any,
    device: Any,
    retrieval_directory: Path,
    dev_path: Path,
    output_directory: Path,
    config: E04Config,
) -> dict[str, Any]:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - Kaggle dependency
        raise E04Error("PyTorch is required for reranking.") from exc
    dev = _load_dev(dev_path)
    summaries: dict[str, Any] = {}
    for candidate in config.candidates:
        retrieval_rows, sample_ids, retrieval_sha = load_retrieval_records(
            candidate=candidate,
            retrieval_directory=retrieval_directory,
            dev_path=dev_path,
            config=config,
        )
        identity = _stage_identity(
            stage="rerank",
            candidate=candidate,
            config=config,
            source_sha256=retrieval_sha,
            sample_ids=sample_ids,
            model={
                "model_id": config.reranker_model_id,
                "revision": config.reranker_revision,
                "parameter_count": config.reranker_parameter_count,
            },
        )
        candidate_root = output_directory / "reranked" / candidate
        records_directory = candidate_root / "records"
        records_directory.mkdir(parents=True, exist_ok=True)
        state_path = candidate_root / "state.json"
        completed = _load_completed(records_directory, state_path, identity, sample_ids)

        for index in range(completed, len(sample_ids)):
            question_id = sample_ids[index]
            contexts = retrieval_rows[index]["contexts"]
            pairs = [(dev[question_id]["question"], context["text"]) for context in contexts]
            scores: list[float] = []
            started = time.perf_counter()
            for offset in range(0, len(pairs), config.reranker_batch_size):
                batch = pairs[offset:offset + config.reranker_batch_size]
                inputs = tokenizer(
                    batch,
                    padding=True,
                    truncation=False,
                    return_tensors="pt",
                )
                lengths = inputs["attention_mask"].sum(dim=1)
                maximum = int(lengths.max().item())
                if maximum > config.reranker_max_tokens:
                    raise E04Error(
                        f"Reranker pair exceeds {config.reranker_max_tokens} tokens: "
                        f"question_id={question_id}, observed={maximum}"
                    )
                inputs = {key: value.to(device) for key, value in inputs.items()}
                with torch.inference_mode():
                    logits = model(**inputs, return_dict=True).logits.view(-1).float()
                scores.extend(float(value) for value in logits.detach().cpu().tolist())
            latency_ms = (time.perf_counter() - started) * 1000
            if len(scores) != len(contexts):
                raise E04Error("Reranker score count differs from candidate count.")
            order = reranker_order(
                scores=scores,
                contexts=contexts,
                top_k=config.reranker_output_top_k,
            )
            reranked_contexts = [contexts[row] for row in order]
            result = {
                "question_id": question_id,
                "sample_index": index,
                "candidate": candidate,
                "contexts": reranked_contexts,
                "reranker": [
                    {
                        "chunk_id": contexts[row]["chunk_id"],
                        "score": scores[row],
                        "fused_rank": row + 1,
                        "reranked_rank": rank + 1,
                    }
                    for rank, row in enumerate(order)
                ],
                "reranker_latency_ms": latency_ms,
            }
            _atomic_json(records_directory / f"{index:04d}.json", result)
            _write_state(state_path, identity, index + 1, complete=False)
            LOGGER.info(
                "e04_rerank_progress candidate=%s completed=%d total=%d question_id=%s latency_ms=%.1f",
                candidate, index + 1, len(sample_ids), question_id, latency_ms,
            )
        _write_results_jsonl(candidate_root / "results.jsonl", records_directory, len(sample_ids))
        _write_state(state_path, identity, len(sample_ids), complete=True)
        rows = _read_records(records_directory, sample_ids)
        summaries[candidate] = {
            "sample_size": len(rows),
            "mean_latency_ms": fmean(row["reranker_latency_ms"] for row in rows),
            "results_sha256": file_sha256(candidate_root / "results.jsonl"),
            "run_identity": identity,
        }
    return summaries


def run_generation(
    *,
    model: Any,
    processor: Any,
    dev_path: Path,
    output_directory: Path,
    config: E04Config,
    device_map: dict[str, Any],
) -> dict[str, Any]:
    dev = _load_dev(dev_path)
    sample_ids = select_dev_sample(dev, seed=config.sample_seed, size=config.sample_size)
    summaries: dict[str, Any] = {}

    def token_count(messages: list[dict[str, Any]]) -> int:
        encoded = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
            return_dict=True,
        )
        ids = encoded["input_ids"]
        if ids and isinstance(ids[0], list):
            ids = ids[0]
        return len(ids)

    input_device = getattr(model, "device", None)
    if input_device is None:
        raise E04Error("Cannot determine generator input device.")
    observed_devices = sorted({str(value) for value in device_map.values()})

    for candidate in config.candidates:
        reranked_path = output_directory / "reranked" / candidate / "results.jsonl"
        reranked_state = output_directory / "reranked" / candidate / "state.json"
        if not reranked_path.is_file() or not reranked_state.is_file():
            raise E04Error(f"Reranked output is missing: {candidate}")
        state = json.loads(reranked_state.read_text(encoding="utf-8"))
        if state.get("complete") is not True or state.get("completed_count") != len(sample_ids):
            raise E04Error(f"Reranked output is incomplete: {candidate}")
        reranked_rows = [
            json.loads(line) for line in reranked_path.read_text(encoding="utf-8").splitlines()
        ]
        if [row.get("question_id") for row in reranked_rows] != sample_ids:
            raise E04Error(f"Reranked question order differs: {candidate}")
        identity = _stage_identity(
            stage="generate",
            candidate=candidate,
            config=config,
            source_sha256=file_sha256(reranked_path),
            sample_ids=sample_ids,
            model={
                "model_id": config.generator_model_id,
                "revision": config.generator_revision,
                "parameter_count": config.generator_parameter_count,
                "dtype": config.generator_dtype,
                "device_map": config.generator_device_map,
                "observed_devices": observed_devices,
            },
        )
        candidate_root = output_directory / "generated" / candidate
        records_directory = candidate_root / "records"
        records_directory.mkdir(parents=True, exist_ok=True)
        state_path = candidate_root / "state.json"
        completed = _load_completed(records_directory, state_path, identity, sample_ids)

        for index in range(completed, len(sample_ids)):
            question_id = sample_ids[index]
            selected, messages, input_tokens = pack_contexts(
                question=dev[question_id]["question"],
                contexts=reranked_rows[index]["contexts"],
                config=config,
                token_counter=token_count,
            )
            inputs = processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=False,
                return_dict=True,
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
            prompt_length = inputs["input_ids"].shape[1]
            new_ids = generated[:, prompt_length:]
            answer = processor.batch_decode(
                new_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0].strip()
            if not answer:
                raise E04Error(f"Generator returned an empty answer: {candidate}/{question_id}")
            result = {
                "question_id": question_id,
                "sample_index": index,
                "candidate": candidate,
                "answer": answer,
                "selected_chunk_ids": [context["chunk_id"] for context in selected],
                "selected_context_count": len(selected),
                "input_tokens": input_tokens,
                "output_tokens": int(new_ids.shape[1]),
                "generation_latency_ms": latency_ms,
            }
            _atomic_json(records_directory / f"{index:04d}.json", result)
            _write_state(state_path, identity, index + 1, complete=False)
            LOGGER.info(
                "e04_generate_progress candidate=%s completed=%d total=%d question_id=%s "
                "contexts=%d input_tokens=%d output_tokens=%d latency_ms=%.1f",
                candidate, index + 1, len(sample_ids), question_id, len(selected),
                input_tokens, new_ids.shape[1], latency_ms,
            )
        _write_results_jsonl(candidate_root / "results.jsonl", records_directory, len(sample_ids))
        _write_state(state_path, identity, len(sample_ids), complete=True)
        rows = _read_records(records_directory, sample_ids)
        summaries[candidate] = {
            "sample_size": len(rows),
            "results_sha256": file_sha256(candidate_root / "results.jsonl"),
            "mean_latency_ms": fmean(row["generation_latency_ms"] for row in rows),
            "run_identity": identity,
        }
    return summaries


def score_and_compare(*, dev_path: Path, output_directory: Path, config: E04Config) -> dict[str, Any]:
    ensure_nltk_resources(download=False)
    dev = _load_dev(dev_path)
    sample_ids = select_dev_sample(dev, seed=config.sample_seed, size=config.sample_size)
    scored: dict[str, list[dict[str, Any]]] = {}
    generated: dict[str, list[dict[str, Any]]] = {}

    for candidate in config.candidates:
        root = output_directory / "generated" / candidate
        state = json.loads((root / "state.json").read_text(encoding="utf-8"))
        if state.get("complete") is not True or state.get("completed_count") != len(sample_ids):
            raise E04Error(f"Generation is incomplete; refusing partial score: {candidate}")
        rows = [json.loads(line) for line in (root / "results.jsonl").read_text(encoding="utf-8").splitlines()]
        if [row.get("question_id") for row in rows] != sample_ids:
            raise E04Error(f"Generated question order differs: {candidate}")
        generated[candidate] = rows
        score_rows = []
        predictions = {}
        for row in rows:
            question_id = row["question_id"]
            reference = dev[question_id]["answer"]
            score_rows.append({
                "question_id": question_id,
                "sample_index": row["sample_index"],
                "meteor": nltk_meteor_score(reference, row["answer"]),
                "rouge_l": rouge_l_fmeasure(reference, row["answer"]),
            })
            predictions[question_id] = {"answer": row["answer"]}
        scored[candidate] = score_rows
        _atomic_jsonl(output_directory / f"scores-{candidate}.jsonl", score_rows)
        _atomic_json(output_directory / f"predictions-{candidate}.json", predictions)

    aiteam = scored["embedding_aiteam"]
    harrier = scored["embedding_harrier"]
    meteor_deltas = [left["meteor"] - right["meteor"] for left, right in zip(aiteam, harrier)]
    rouge_deltas = [left["rouge_l"] - right["rouge_l"] for left, right in zip(aiteam, harrier)]
    metrics = {
        candidate: {
            "meteor": fmean(row["meteor"] for row in rows),
            "rouge_l": fmean(row["rouge_l"] for row in rows),
        }
        for candidate, rows in scored.items()
    }
    report = {
        "schema_version": "1.0",
        "experiment_id": "E04-rerank-answer-compare-dev200-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "sample_size": len(sample_ids),
        "metrics": metrics,
        "paired_delta_aiteam_minus_harrier": {
            "meteor_mean": fmean(meteor_deltas),
            "meteor_bootstrap_95_ci": _bootstrap_ci(
                meteor_deltas, seed=config.bootstrap_seed + ":meteor",
                iterations=config.bootstrap_iterations,
            ),
            "rouge_l_mean": fmean(rouge_deltas),
            "rouge_l_bootstrap_95_ci": _bootstrap_ci(
                rouge_deltas, seed=config.bootstrap_seed + ":rouge_l",
                iterations=config.bootstrap_iterations,
            ),
        },
        "runtime": {
            candidate: {
                "mean_generation_latency_ms": fmean(row["generation_latency_ms"] for row in rows),
                "mean_input_tokens": fmean(row["input_tokens"] for row in rows),
                "mean_output_tokens": fmean(row["output_tokens"] for row in rows),
                "mean_selected_contexts": fmean(row["selected_context_count"] for row in rows),
            }
            for candidate, rows in generated.items()
        },
        "evidence": {
            "config_sha256": config.config_sha256,
            "dev_sha256": config.dev_sha256,
            "sample_ids_sha256": config.sample_ids_sha256,
            "generated_results_sha256": {
                candidate: file_sha256(output_directory / "generated" / candidate / "results.jsonl")
                for candidate in config.candidates
            },
        },
        "promotion_allowed": False,
        "warning": "Dev-200 is a paired smoke evaluation. Freeze the better configuration and apply the predeclared full-dev promotion rule before selecting an embedding.",
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
    lower = samples[int(0.025 * (iterations - 1))]
    upper = samples[int(0.975 * (iterations - 1))]
    return [lower, upper]


def reranker_order(
    *, scores: list[float], contexts: list[dict[str, Any]], top_k: int
) -> list[int]:
    """Return a deterministic descending-logit order with fused-rank tie breaking."""

    if len(scores) != len(contexts) or not scores or top_k <= 0 or top_k > len(scores):
        raise ValueError("Invalid reranker ordering inputs.")
    if any(not isinstance(context.get("chunk_id"), str) for context in contexts):
        raise ValueError("Every reranker context requires a string chunk ID.")
    return sorted(
        range(len(contexts)),
        key=lambda row: (-float(scores[row]), row, contexts[row]["chunk_id"]),
    )[:top_k]


def _stage_identity(
    *,
    stage: str,
    candidate: str,
    config: E04Config,
    source_sha256: str,
    sample_ids: list[str],
    model: dict[str, Any],
) -> dict[str, Any]:
    identity = {
        "code_version": CODE_VERSION,
        "stage": stage,
        "candidate": candidate,
        "config_sha256": config.config_sha256,
        "source_sha256": source_sha256,
        "sample_ids_sha256": hashlib.sha256("\n".join(sample_ids).encode("utf-8")).hexdigest(),
        "model": model,
    }
    identity["identity_sha256"] = hashlib.sha256(
        json.dumps(identity, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return identity


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
