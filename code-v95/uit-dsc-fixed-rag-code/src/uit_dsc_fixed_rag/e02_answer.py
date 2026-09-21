"""Deterministic AITeam E02 answer-level smoke evaluation."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean
from typing import Any, Callable

from uit_dsc_fixed_rag.corpus import file_sha256
from uit_dsc_fixed_rag.e02_compare import E02CompareError, _load_dev
from uit_dsc_fixed_rag.evaluation.official import (
    ensure_nltk_resources,
    nltk_meteor_score,
    rouge_l_fmeasure,
)
from uit_dsc_fixed_rag.retrieval_diagnostics import select_dev_sample


CODE_VERSION = "0.1.0"
LOGGER = logging.getLogger(__name__)


class E02AnswerError(RuntimeError):
    """Raised when the answer-level run cannot safely continue."""


@dataclass(frozen=True)
class AnswerConfig:
    raw: dict[str, Any]
    path: Path
    candidate_key: str
    compare_config_path: str
    compare_config_sha256: str
    generator_key: str
    generator_model_id: str
    generator_revision: str
    generator_parameter_count: int
    generator_dtype: str
    generator_device_map: str
    required_cuda_devices: int
    embedding_parameter_count: int
    total_parameter_count: int
    parameter_limit: int
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

    @property
    def config_sha256(self) -> str:
        return file_sha256(self.path)


def load_answer_config(path: Path) -> AnswerConfig:
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected_root = {
        "schema_version", "experiment_id", "retrieval", "generator",
        "parameter_budget", "dev", "context", "decoding", "scoring",
        "run_contract",
    }
    if set(payload) != expected_root or payload.get("schema_version") != "1.0":
        raise ValueError("E02 answer config root is incompatible.")
    if payload.get("experiment_id") != "E02-answer-aiteam-dev200-v1":
        raise ValueError("Unexpected E02 answer experiment ID.")

    retrieval = _object(payload, "retrieval")
    generator = _object(payload, "generator")
    budget = _object(payload, "parameter_budget")
    dev = _object(payload, "dev")
    context = _object(payload, "context")
    decoding = _object(payload, "decoding")
    scoring = _object(payload, "scoring")
    contract = _object(payload, "run_contract")

    if retrieval != {
        "candidate": "embedding_aiteam",
        "compare_config_path": retrieval.get("compare_config_path"),
        "compare_config_sha256": retrieval.get("compare_config_sha256"),
        "candidate_k_per_branch": 40,
        "fused_top_k": 20,
        "rrf_constant": 60,
        "branch_weights": {"sparse": 1.0, "dense": 1.0},
    }:
        raise ValueError("AITeam retrieval settings changed from the frozen E02 baseline.")
    if generator.get("dtype") != "float16" or generator.get("device_map") != "balanced":
        raise ValueError("Generator must use the reviewed dual-GPU float16 placement.")
    if decoding != {
        "do_sample": False,
        "enable_thinking": False,
        "max_new_tokens": 384,
        "num_beams": 1,
        "use_cache": True,
    }:
        raise ValueError("Deterministic decoding settings changed.")
    if context.get("packing") != "fused-rank-whole-chunks-greedy":
        raise ValueError("Unexpected context packing policy.")
    required_contract = {
        "checkpoint_every_questions": 1,
        "atomic_checkpoint_write": True,
        "resume_fail_closed": True,
        "allow_holdout": False,
        "allow_public": False,
        "allow_external_data": False,
        "allow_synthetic_data": False,
        "allow_model_api": False,
        "promotion_allowed": False,
    }
    if contract != required_contract:
        raise ValueError("E02 answer run contract changed.")

    embedding_count = _positive_int(budget, "embedding_parameter_count")
    generator_count = _positive_int(budget, "generator_parameter_count")
    total_count = _positive_int(budget, "loaded_model_parameter_total")
    limit = _positive_int(budget, "exclusive_limit")
    if total_count != embedding_count + generator_count or total_count >= limit:
        raise ValueError("Reviewed model parameter budget is invalid.")
    if generator_count != _positive_int(generator, "parameter_count"):
        raise ValueError("Generator parameter counts disagree.")
    if scoring.get("primary_metric") != "meteor" or scoring.get("secondary_metric") != "rouge_l":
        raise ValueError("Official metric selection changed.")

    return AnswerConfig(
        raw=payload,
        path=path,
        candidate_key=_text(retrieval, "candidate"),
        compare_config_path=_text(retrieval, "compare_config_path"),
        compare_config_sha256=_sha256(retrieval, "compare_config_sha256"),
        generator_key=_text(generator, "key"),
        generator_model_id=_text(generator, "model_id"),
        generator_revision=_revision(generator, "revision"),
        generator_parameter_count=generator_count,
        generator_dtype=_text(generator, "dtype"),
        generator_device_map=_text(generator, "device_map"),
        required_cuda_devices=_positive_int(generator, "required_cuda_devices"),
        embedding_parameter_count=embedding_count,
        total_parameter_count=total_count,
        parameter_limit=limit,
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
    )


def validate_static_inputs(*, project_root: Path, dev_path: Path, config: AnswerConfig) -> dict[str, Any]:
    compare_config = project_root / config.compare_config_path
    scorer = project_root / config.scorer_path
    if not compare_config.is_file() or file_sha256(compare_config) != config.compare_config_sha256:
        raise E02AnswerError("Pinned retrieval comparison config is missing or changed.")
    if not scorer.is_file() or file_sha256(scorer) != config.scorer_sha256:
        raise E02AnswerError("Pinned official scorer implementation is missing or changed.")
    if not dev_path.is_file() or file_sha256(dev_path) != config.dev_sha256:
        raise E02AnswerError("Pinned dev split is missing or changed.")
    dev = _load_dev(dev_path)
    sample_ids = select_dev_sample(dev, seed=config.sample_seed, size=config.sample_size)
    sample_sha = hashlib.sha256("\n".join(sample_ids).encode("utf-8")).hexdigest()
    if sample_sha != config.sample_ids_sha256:
        raise E02AnswerError("Deterministic dev-200 sample identity changed.")
    return {
        "config_sha256": config.config_sha256,
        "compare_config_sha256": config.compare_config_sha256,
        "official_scorer_sha256": config.scorer_sha256,
        "dev_sha256": config.dev_sha256,
        "sample_ids_sha256": sample_sha,
        "sample_size": len(sample_ids),
        "parameter_total": config.total_parameter_count,
        "parameter_limit_exclusive": config.parameter_limit,
    }


def make_messages(
    *, question: str, contexts: list[dict[str, Any]], config: AnswerConfig
) -> list[dict[str, Any]]:
    evidence = []
    for index, context in enumerate(contexts, start=1):
        article = context.get("article_number")
        article_label = f", điều {article}" if article not in (None, "") else ""
        evidence.append(
            f"[Tài liệu {index}{article_label}]\n{str(context['text']).strip()}"
        )
    joined_evidence = "\n\n".join(evidence)
    user_text = (
        f"Câu hỏi:\n{question.strip()}\n\n"
        f"Tài liệu:\n{joined_evidence}\n\n"
        f"Yêu cầu:\n{config.answer_instruction}"
    )
    return [
        {"role": "system", "content": [{"type": "text", "text": config.system_prompt}]},
        {"role": "user", "content": [{"type": "text", "text": user_text}]},
    ]


def pack_contexts(
    *,
    question: str,
    contexts: list[dict[str, Any]],
    config: AnswerConfig,
    token_counter: Callable[[list[dict[str, Any]]], int],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    """Greedily retain whole fused-ranked chunks under the exact chat-token cap."""

    selected: list[dict[str, Any]] = []
    final_messages: list[dict[str, Any]] | None = None
    final_count = 0
    for context in contexts:
        if not isinstance(context, dict) or not isinstance(context.get("text"), str):
            raise E02AnswerError("Retrieval context has an invalid schema.")
        candidate_contexts = [*selected, context]
        candidate_messages = make_messages(
            question=question, contexts=candidate_contexts, config=config
        )
        candidate_count = int(token_counter(candidate_messages))
        if candidate_count <= 0:
            raise E02AnswerError("Generator token counter returned an invalid length.")
        if candidate_count <= config.max_input_tokens:
            selected = candidate_contexts
            final_messages = candidate_messages
            final_count = candidate_count
    if len(selected) < config.minimum_contexts or final_messages is None:
        raise E02AnswerError("No retrieved context fits the reviewed generator input cap.")
    return selected, final_messages, final_count


def load_retrieval_records(
    *, retrieval_directory: Path, dev_path: Path, config: AnswerConfig
) -> tuple[list[dict[str, Any]], list[str], str]:
    results_path = retrieval_directory / config.candidate_key / "results.jsonl"
    state_path = retrieval_directory / config.candidate_key / "state.json"
    if not results_path.is_file() or not state_path.is_file():
        raise E02AnswerError("Completed AITeam retrieval output is missing.")
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if state.get("complete") is not True or state.get("completed_count") != config.sample_size:
        raise E02AnswerError("AITeam retrieval checkpoint is incomplete.")
    identity = state.get("run_identity", {})
    if identity.get("config_sha256") != config.compare_config_sha256:
        raise E02AnswerError("AITeam retrieval used a different frozen config.")
    if identity.get("candidate", {}).get("key") != config.candidate_key:
        raise E02AnswerError("Retrieval candidate is not AITeam.")

    dev = _load_dev(dev_path)
    sample_ids = select_dev_sample(dev, seed=config.sample_seed, size=config.sample_size)
    try:
        rows = [json.loads(line) for line in results_path.read_text(encoding="utf-8").splitlines()]
    except json.JSONDecodeError as exc:
        raise E02AnswerError("AITeam retrieval results are invalid JSONL.") from exc
    if len(rows) != len(sample_ids):
        raise E02AnswerError("AITeam retrieval result count differs from dev-200.")
    for index, (question_id, row) in enumerate(zip(sample_ids, rows)):
        if (
            row.get("sample_index") != index
            or row.get("question_id") != question_id
            or row.get("candidate") != config.candidate_key
            or not isinstance(row.get("contexts"), list)
        ):
            raise E02AnswerError(f"AITeam retrieval row {index} has an incompatible identity.")
    return rows, sample_ids, file_sha256(results_path)


def run_generation(
    *,
    model: Any,
    processor: Any,
    retrieval_directory: Path,
    dev_path: Path,
    output_directory: Path,
    config: AnswerConfig,
    device_map: dict[str, Any],
) -> dict[str, Any]:
    retrieval_rows, sample_ids, retrieval_sha = load_retrieval_records(
        retrieval_directory=retrieval_directory, dev_path=dev_path, config=config
    )
    dev = _load_dev(dev_path)
    identity = _generation_identity(config, retrieval_sha, sample_ids, device_map)
    generation_root = output_directory / "generation"
    records_directory = generation_root / "records"
    records_directory.mkdir(parents=True, exist_ok=True)
    state_path = generation_root / "state.json"
    completed = _load_completed(records_directory, state_path, identity, sample_ids)

    def count_messages(messages: list[dict[str, Any]]) -> int:
        encoded = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
            return_dict=True,
        )
        input_ids = encoded["input_ids"]
        if input_ids and isinstance(input_ids[0], list):
            input_ids = input_ids[0]
        return len(input_ids)

    input_device = getattr(model, "device", None)
    if input_device is None:
        raise E02AnswerError("Cannot determine the generator input device.")

    for index in range(completed, len(sample_ids)):
        question_id = sample_ids[index]
        question = dev[question_id]["question"]
        selected, messages, input_tokens = pack_contexts(
            question=question,
            contexts=retrieval_rows[index]["contexts"],
            config=config,
            token_counter=count_messages,
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
        generation_started = time.perf_counter()
        generated = model.generate(
            **inputs,
            do_sample=False,
            num_beams=1,
            max_new_tokens=config.max_new_tokens,
            use_cache=True,
        )
        latency_ms = (time.perf_counter() - generation_started) * 1000
        prompt_length = inputs["input_ids"].shape[1]
        new_ids = generated[:, prompt_length:]
        answer = processor.batch_decode(
            new_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()
        if not answer:
            raise E02AnswerError(f"Generator returned an empty answer for {question_id}.")
        result = {
            "question_id": question_id,
            "sample_index": index,
            "candidate": config.candidate_key,
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
            "e02_answer_progress candidate=%s completed=%d total=%d question_id=%s "
            "contexts=%d input_tokens=%d output_tokens=%d latency_ms=%.1f",
            config.candidate_key, index + 1, len(sample_ids), question_id,
            len(selected), input_tokens, new_ids.shape[1], latency_ms,
        )

    _write_results_jsonl(generation_root / "results.jsonl", records_directory, len(sample_ids))
    _write_state(state_path, identity, len(sample_ids), complete=True)
    return {
        "sample_size": len(sample_ids),
        "results_sha256": file_sha256(generation_root / "results.jsonl"),
        "run_identity": identity,
    }


def score_generation(
    *, dev_path: Path, output_directory: Path, config: AnswerConfig
) -> dict[str, Any]:
    ensure_nltk_resources(download=False)
    dev = _load_dev(dev_path)
    sample_ids = select_dev_sample(dev, seed=config.sample_seed, size=config.sample_size)
    records_directory = output_directory / "generation" / "records"
    state_path = output_directory / "generation" / "state.json"
    if not state_path.is_file():
        raise E02AnswerError("Generation checkpoint state is missing.")
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if state.get("complete") is not True or state.get("completed_count") != len(sample_ids):
        raise E02AnswerError("Generation is incomplete; refusing to score partial answers.")

    rows = _read_records(records_directory, sample_ids)
    scored_rows = []
    predictions: dict[str, dict[str, str]] = {}
    for row in rows:
        question_id = row["question_id"]
        answer = row["answer"]
        reference = dev[question_id]["answer"]
        meteor = nltk_meteor_score(reference, answer)
        rouge_l = rouge_l_fmeasure(reference, answer)
        scored_rows.append({
            "question_id": question_id,
            "sample_index": row["sample_index"],
            "meteor": meteor,
            "rouge_l": rouge_l,
        })
        predictions[question_id] = {"answer": answer}

    scores_path = output_directory / "scores.jsonl"
    _atomic_jsonl(scores_path, scored_rows)
    _atomic_json(output_directory / "predictions.json", predictions)
    report = {
        "schema_version": "1.0",
        "experiment_id": "E02-answer-aiteam-dev200-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "candidate": config.candidate_key,
        "generator": {
            "key": config.generator_key,
            "model_id": config.generator_model_id,
            "revision": config.generator_revision,
            "parameter_count": config.generator_parameter_count,
        },
        "sample_size": len(rows),
        "metrics": {
            "meteor": fmean(row["meteor"] for row in scored_rows),
            "rouge_l": fmean(row["rouge_l"] for row in scored_rows),
        },
        "runtime": {
            "mean_generation_latency_ms": fmean(row["generation_latency_ms"] for row in rows),
            "mean_input_tokens": fmean(row["input_tokens"] for row in rows),
            "mean_output_tokens": fmean(row["output_tokens"] for row in rows),
            "mean_selected_contexts": fmean(row["selected_context_count"] for row in rows),
        },
        "evidence": {
            "config_sha256": config.config_sha256,
            "dev_sha256": config.dev_sha256,
            "sample_ids_sha256": config.sample_ids_sha256,
            "generation_results_sha256": file_sha256(output_directory / "generation" / "results.jsonl"),
            "scores_sha256": file_sha256(scores_path),
        },
        "promotion_allowed": False,
        "warning": "Dev-200 is a smoke evaluation. It cannot promote AITeam without the paired frozen-generator comparison and the predeclared full-dev rule.",
    }
    _atomic_json(output_directory / "report.json", report)
    return report


def _generation_identity(
    config: AnswerConfig,
    retrieval_sha256: str,
    sample_ids: list[str],
    device_map: dict[str, Any],
) -> dict[str, Any]:
    normalized_devices = sorted({str(value) for value in device_map.values()})
    identity = {
        "code_version": CODE_VERSION,
        "config_sha256": config.config_sha256,
        "retrieval_results_sha256": retrieval_sha256,
        "dev_sha256": config.dev_sha256,
        "sample_ids_sha256": hashlib.sha256("\n".join(sample_ids).encode("utf-8")).hexdigest(),
        "generator": {
            "model_id": config.generator_model_id,
            "revision": config.generator_revision,
            "parameter_count": config.generator_parameter_count,
            "dtype": config.generator_dtype,
            "device_map": config.generator_device_map,
            "observed_devices": normalized_devices,
        },
    }
    identity["identity_sha256"] = _json_sha256(identity)
    return identity


def _load_completed(
    records_directory: Path,
    state_path: Path,
    identity: dict[str, Any],
    sample_ids: list[str],
) -> int:
    files = sorted(records_directory.glob("*.json"))
    if not state_path.exists():
        if files:
            raise E02AnswerError("Generation records exist without checkpoint state.")
        _write_state(state_path, identity, 0, complete=False)
        return 0
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if state.get("run_identity") != identity:
        raise E02AnswerError("Checkpoint belongs to a different generation run.")
    if len(files) > len(sample_ids):
        raise E02AnswerError("Generation checkpoint has too many records.")
    for index, path in enumerate(files):
        if path.name != f"{index:04d}.json":
            raise E02AnswerError("Generation checkpoint sequence is non-contiguous.")
        row = json.loads(path.read_text(encoding="utf-8"))
        if row.get("sample_index") != index or row.get("question_id") != sample_ids[index]:
            raise E02AnswerError("Generation checkpoint question identity mismatch.")
    if int(state.get("completed_count", -1)) > len(files):
        raise E02AnswerError("Generation state is ahead of durable records.")
    _write_state(state_path, identity, len(files), complete=len(files) == len(sample_ids))
    return len(files)


def _read_records(records_directory: Path, sample_ids: list[str]) -> list[dict[str, Any]]:
    rows = [
        json.loads((records_directory / f"{index:04d}.json").read_text(encoding="utf-8"))
        for index in range(len(sample_ids))
    ]
    if [row.get("question_id") for row in rows] != sample_ids:
        raise E02AnswerError("Generation result order differs from dev-200.")
    return rows


def _write_results_jsonl(path: Path, records_directory: Path, count: int) -> None:
    rows = [
        json.loads((records_directory / f"{index:04d}.json").read_text(encoding="utf-8"))
        for index in range(count)
    ]
    _atomic_jsonl(path, rows)


def _write_state(path: Path, identity: dict[str, Any], completed: int, *, complete: bool) -> None:
    _atomic_json(path, {
        "schema_version": "1.0",
        "run_identity": identity,
        "completed_count": completed,
        "complete": complete,
    })


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


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
