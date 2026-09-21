"""E05 context-count and document/article-diversity grid for the fixed RAG stack."""

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


CODE_VERSION = "0.10.0"
LOGGER = logging.getLogger(__name__)


class E05Error(RuntimeError):
    """Raised when an E05 artifact cannot be used or resumed safely."""


@dataclass(frozen=True)
class E05Variant:
    key: str
    candidate_limit: int
    ordering: str
    answer_source: str


@dataclass(frozen=True)
class E05Config:
    raw: dict[str, Any]
    path: Path
    selection: dict[str, Any]
    variants: tuple[E05Variant, ...]
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
    system_prompt: str
    answer_instruction: str
    max_new_tokens: int
    worker_count: int
    scorer_path: str
    scorer_sha256: str
    control_variant: str
    bootstrap_seed: str
    bootstrap_iterations: int

    @property
    def config_sha256(self) -> str:
        return file_sha256(self.path)


def load_e05_config(path: Path) -> E05Config:
    payload = json.loads(path.read_text(encoding="utf-8"))
    required_root = {
        "schema_version", "experiment_id", "selection_source", "variants",
        "generator", "parameter_budget", "dev", "context", "decoding",
        "execution", "scoring", "run_contract",
    }
    if set(payload) != required_root or payload.get("schema_version") != "1.0":
        raise ValueError("E05 config root is incompatible.")
    if payload.get("experiment_id") != "E05-context-grid-aiteam-dev200-v1":
        raise ValueError("Unexpected E05 experiment ID.")

    selection = _object(payload, "selection_source")
    if selection != {
        "mode": "operator-selected-no-reranker-from-e04-smoke",
        "report_experiment_id": "E04-rerank-grid-aiteam-dev200-v2",
        "report_config_sha256": "d2ae8e5349721cae0dbfc1dd87160cfb664e10674cd3a4fbb934b74b8922b3b0",
        "generation_results_sha256": "de96e8469162f67cc441defff1ca44d3ef21efbfd9e935c6f1d8e46157a22ddc",
        "context_results_sha256": "d721ddc0d0052d2d8364a3d0e463b3c94669bfd7efeed1f9402404f0607e2c3c",
        "sample_size": 200,
        "selected_embedding": "embedding_aiteam",
        "sparse_weight": 0.5,
        "dense_weight": 0.5,
        "selected_variant": "control_no_rerank",
        "meteor": 0.3630209902348587,
        "rouge_l": 0.4216082800213014,
        "promotion_rule_satisfied": False,
    }:
        raise ValueError("E05 operator selection evidence changed.")

    variants_payload = payload.get("variants")
    if not isinstance(variants_payload, list) or len(variants_payload) != 3:
        raise ValueError("E05 requires exactly three context variants.")
    variants = tuple(
        E05Variant(
            key=_text(item, "key"),
            candidate_limit=_positive_int(item, "candidate_limit"),
            ordering=_text(item, "ordering"),
            answer_source=_text(item, "answer_source"),
        )
        for item in variants_payload
        if isinstance(item, dict)
    )
    if tuple(
        (row.key, row.candidate_limit, row.ordering, row.answer_source)
        for row in variants
    ) != (
        ("control_ranked_top20", 20, "fused-rank", "reuse-e04-control-no-rerank"),
        ("ranked_top12", 12, "fused-rank", "generate-e05"),
        (
            "document_article_diverse_top12", 12,
            "first-document-article-occurrence-then-fused-rank-fill", "generate-e05",
        ),
    ):
        raise ValueError("E05 context grid changed from the experiment plan.")

    generator = _object(payload, "generator")
    if (
        generator.get("key") != "generator_qwen35_2b"
        or generator.get("model_id") != "Qwen/Qwen3.5-2B"
        or generator.get("dtype") != "float16"
        or generator.get("device_mode") != "replicated-single-gpu-workers"
    ):
        raise ValueError("E05 generator contract changed.")
    budget = _object(payload, "parameter_budget")
    limit = _positive_int(budget, "exclusive_limit")
    stack_total = _positive_int(budget, "stack_total")
    observed_total = _positive_int(budget, "embedding") + _positive_int(budget, "generator")
    if (
        observed_total != stack_total or stack_total >= limit
        or _positive_int(budget, "generator") != _positive_int(generator, "parameter_count")
    ):
        raise ValueError("E05 stack violates the parameter budget.")

    dev = _object(payload, "dev")
    context = _object(payload, "context")
    if context.get("packing") != "ordered-whole-chunks-greedy":
        raise ValueError("E05 context packing changed.")
    decoding = _object(payload, "decoding")
    if decoding != {
        "do_sample": False, "enable_thinking": False, "max_new_tokens": 384,
        "num_beams": 1, "use_cache": True,
    }:
        raise ValueError("E05 deterministic decoding changed.")
    execution = _object(payload, "execution")
    if execution != {
        "worker_count": 2,
        "partition": "sample-index-mod-worker-count",
        "prompts_per_generate_call": 1,
    }:
        raise ValueError("E05 dual-GPU execution contract changed.")
    scoring = _object(payload, "scoring")
    if (
        scoring.get("primary_metric") != "meteor"
        or scoring.get("secondary_metric") != "rouge_l"
        or scoring.get("control_variant") != "control_ranked_top20"
    ):
        raise ValueError("E05 scoring contract changed.")
    contract = _object(payload, "run_contract")
    if contract != {
        "retrieval_rankings_fixed": True,
        "reranker_allowed": False,
        "reuse_control_answer_exactly": True,
        "checkpoint_every_questions": 1,
        "atomic_checkpoint_write": True,
        "resume_fail_closed": True,
        "same_embedding_fusion_generator_prompt_decoding": True,
        "allow_holdout": False,
        "allow_public": False,
        "allow_external_data": False,
        "allow_synthetic_data": False,
        "allow_model_api": False,
        "promotion_allowed": False,
    }:
        raise ValueError("E05 run contract changed.")

    return E05Config(
        raw=payload,
        path=path,
        selection=dict(selection),
        variants=variants,
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
        system_prompt=_text(context, "system_prompt"),
        answer_instruction=_text(context, "answer_instruction"),
        max_new_tokens=_positive_int(decoding, "max_new_tokens"),
        worker_count=_positive_int(execution, "worker_count"),
        scorer_path=_text(scoring, "official_scorer_path"),
        scorer_sha256=_sha256(scoring, "official_scorer_sha256"),
        control_variant=_text(scoring, "control_variant"),
        bootstrap_seed=_text(scoring, "bootstrap_seed"),
        bootstrap_iterations=_positive_int(scoring, "bootstrap_iterations"),
    )


def validate_preflight(
    *, project_root: Path, selection_directory: Path, dev_path: Path,
    config: E05Config,
) -> dict[str, Any]:
    scorer = project_root / config.scorer_path
    if not scorer.is_file() or file_sha256(scorer) != config.scorer_sha256:
        raise E05Error("Pinned official scorer is missing or changed.")
    if not dev_path.is_file() or file_sha256(dev_path) != config.dev_sha256:
        raise E05Error("Pinned dev split is missing or changed.")
    sample_ids = select_dev_sample(
        _load_dev(dev_path), seed=config.sample_seed, size=config.sample_size
    )
    sample_sha = hashlib.sha256("\n".join(sample_ids).encode("utf-8")).hexdigest()
    if sample_sha != config.sample_ids_sha256:
        raise E05Error("Deterministic E05 sample identity changed.")

    report_path = selection_directory / "report.json"
    context_path = selection_directory / "reranked" / "results.jsonl"
    generation_path = selection_directory / "generation" / "results.jsonl"
    if not all(path.is_file() for path in (report_path, context_path, generation_path)):
        raise E05Error("Saved E04 selection artifact is incomplete.")
    if file_sha256(context_path) != config.selection["context_results_sha256"]:
        raise E05Error("Saved E04 no-reranker contexts changed.")
    if file_sha256(generation_path) != config.selection["generation_results_sha256"]:
        raise E05Error("Saved E04 generation results changed.")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    _validate_selection_report(report, config)
    context_rows = _read_jsonl(context_path)
    generation_rows = _read_jsonl(generation_path)
    _validate_selection_rows(context_rows, generation_rows, sample_ids)
    return {
        "config_sha256": config.config_sha256,
        "dev_sha256": config.dev_sha256,
        "sample_ids_sha256": sample_sha,
        "sample_size": len(sample_ids),
        "selection_report_sha256": file_sha256(report_path),
        "selection_context_results_sha256": file_sha256(context_path),
        "selection_generation_results_sha256": file_sha256(generation_path),
        "selection_mode": config.selection["mode"],
        "selected_stack": {
            "embedding": "embedding_aiteam", "sparse_weight": 0.5,
            "dense_weight": 0.5, "reranker": None,
        },
        "stack_parameter_total": config.stack_total,
        "parameter_limit_exclusive": config.parameter_limit,
    }


def document_article_diverse_order(
    contexts: list[dict[str, Any]], *, top_k: int
) -> list[int]:
    """Keep first occurrence of each document/article unit, then fill by fused rank."""

    if not 0 < top_k <= len(contexts):
        raise ValueError("Invalid document/article-diversity top_k.")
    selected: list[int] = []
    seen_units: set[tuple[str, str | None]] = set()
    for index, context in enumerate(contexts):
        document_id = context.get("document_id")
        if not isinstance(document_id, str) or not document_id:
            raise E05Error("Every E05 context must have a non-blank document_id.")
        article_number = context.get("article_number")
        if article_number is not None and (
            not isinstance(article_number, str) or not article_number.strip()
        ):
            raise E05Error("E05 article_number must be null or non-blank text.")
        unit = (document_id, article_number)
        if unit not in seen_units:
            selected.append(index)
            seen_units.add(unit)
            if len(selected) == top_k:
                return selected
    selected_set = set(selected)
    for index in range(len(contexts)):
        if index not in selected_set:
            selected.append(index)
            if len(selected) == top_k:
                return selected
    raise E05Error("Could not construct the document/article-diverse context list.")


def prepare_context_variants(
    *, selection_directory: Path, dev_path: Path, output_directory: Path,
    config: E05Config, preflight: dict[str, Any],
) -> dict[str, Any]:
    sample_ids = select_dev_sample(
        _load_dev(dev_path), seed=config.sample_seed, size=config.sample_size
    )
    context_path = selection_directory / "reranked" / "results.jsonl"
    generation_path = selection_directory / "generation" / "results.jsonl"
    if (
        file_sha256(context_path) != preflight.get("selection_context_results_sha256")
        or file_sha256(generation_path)
        != preflight.get("selection_generation_results_sha256")
    ):
        raise E05Error("E04 selection artifacts changed after E05 preflight.")
    context_rows = _read_jsonl(context_path)
    generation_rows = _read_jsonl(generation_path)
    _validate_selection_rows(context_rows, generation_rows, sample_ids)

    rows: list[dict[str, Any]] = []
    for index, question_id in enumerate(sample_ids):
        source = context_rows[index]["variants"]["control_no_rerank"]["contexts"]
        control = list(source[:20])
        ranked = list(source[:12])
        diverse_indices = document_article_diverse_order(control, top_k=12)
        diverse = [control[position] for position in diverse_indices]
        rows.append({
            "question_id": question_id,
            "sample_index": index,
            "variants": {
                "control_ranked_top20": {
                    "contexts": control,
                    "source_fused_ranks": list(range(1, 21)),
                },
                "ranked_top12": {
                    "contexts": ranked,
                    "source_fused_ranks": list(range(1, 13)),
                },
                "document_article_diverse_top12": {
                    "contexts": diverse,
                    "source_fused_ranks": [position + 1 for position in diverse_indices],
                },
            },
        })
    root = output_directory / "prepared"
    root.mkdir(parents=True, exist_ok=True)
    _atomic_jsonl(root / "results.jsonl", rows)
    identity = {
        "code_version": CODE_VERSION,
        "stage": "reuse-e04-control-and-materialize-context-grid",
        "config_sha256": config.config_sha256,
        "selection_context_results_sha256": preflight["selection_context_results_sha256"],
        "selection_generation_results_sha256": preflight[
            "selection_generation_results_sha256"
        ],
        "sample_ids_sha256": config.sample_ids_sha256,
    }
    identity["identity_sha256"] = _json_sha256(identity)
    summary = {
        "schema_version": "1.0",
        "sample_size": len(rows),
        "results_sha256": file_sha256(root / "results.jsonl"),
        "run_identity": identity,
        "mean_unique_documents": {
            key: fmean(
                len({context["document_id"] for context in row["variants"][key]["contexts"]})
                for row in rows
            )
            for key in (
                "control_ranked_top20", "ranked_top12",
                "document_article_diverse_top12",
            )
        },
        "mean_unique_document_article_units": {
            key: fmean(
                len({
                    (context["document_id"], context.get("article_number"))
                    for context in row["variants"][key]["contexts"]
                })
                for row in rows
            )
            for key in (
                "control_ranked_top20", "ranked_top12",
                "document_article_diverse_top12",
            )
        },
    }
    _atomic_json(root / "summary.json", summary)
    return summary


def load_generator_on_device(
    config: E05Config, device: str
) -> tuple[Any, Any, dict[str, Any]]:
    try:
        import torch
        from transformers import AutoProcessor
        try:
            from transformers import AutoModelForMultimodalLM as AutoGenerator
        except ImportError:  # pragma: no cover
            from transformers import AutoModelForImageTextToText as AutoGenerator
    except ImportError as exc:  # pragma: no cover
        raise E05Error("Install Transformers and Accelerate for E05 generation.") from exc
    if device not in {"cuda:0", "cuda:1"} or torch.cuda.device_count() != 2:
        raise E05Error("E05 generation workers require GPU T4 x2.")
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
        raise E05Error(
            "Generator runtime unique parameter count changed: "
            f"{observed} != {config.generator_runtime_unique_parameter_count}"
        )
    try:
        device_map = _validate_worker_placement(model, device)
    except RuntimeError as exc:
        raise E05Error(str(exc)) from exc
    LOGGER.info(
        "e05_generator_loaded device=%s runtime_unique_parameters=%d "
        "published_tensor_parameters=%d",
        device, observed, config.generator_parameter_count,
    )
    return model, tokenizer, device_map


def run_generation_worker(
    *, worker_rank: int, device: str, model: Any, tokenizer: Any,
    selection_directory: Path, dev_path: Path, output_directory: Path,
    config: E05Config, device_map: dict[str, Any],
) -> dict[str, Any]:
    if worker_rank not in range(config.worker_count) or device != f"cuda:{worker_rank}":
        raise E05Error("E05 worker rank/device assignment changed.")
    prepared_path = output_directory / "prepared" / "results.jsonl"
    prepared_summary_path = output_directory / "prepared" / "summary.json"
    if not prepared_path.is_file() or not prepared_summary_path.is_file():
        raise E05Error("Run E05 prepare before generation.")
    prepared_summary = json.loads(prepared_summary_path.read_text(encoding="utf-8"))
    if file_sha256(prepared_path) != prepared_summary.get("results_sha256"):
        raise E05Error("Prepared E05 contexts changed.")
    prepared_rows = _read_jsonl(prepared_path)
    source_path = selection_directory / "generation" / "results.jsonl"
    if file_sha256(source_path) != config.selection["generation_results_sha256"]:
        raise E05Error("E04 control answers changed before E05 generation.")
    source_rows = _read_jsonl(source_path)
    dev = _load_dev(dev_path)
    sample_ids = select_dev_sample(dev, seed=config.sample_seed, size=config.sample_size)
    if [row.get("question_id") for row in prepared_rows] != sample_ids:
        raise E05Error("Prepared E05 question order changed.")

    assigned = [index for index in range(len(sample_ids)) if index % 2 == worker_rank]
    identity = {
        "code_version": CODE_VERSION,
        "stage": "dual-gpu-generator-worker-two-context-variants",
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

    for position in range(completed, len(assigned)):
        sample_index = assigned[position]
        question_id = sample_ids[sample_index]
        source_control = source_rows[sample_index]["variants"]["control_no_rerank"]
        control_contexts = prepared_rows[sample_index]["variants"][
            "control_ranked_top20"
        ]["contexts"]
        selected_control, _, control_tokens = pack_contexts(
            question=dev[question_id]["question"], contexts=control_contexts,
            config=config, token_counter=token_count,
        )
        if (
            [context["chunk_id"] for context in selected_control]
            != source_control.get("selected_chunk_ids")
            or control_tokens != source_control.get("input_tokens")
        ):
            raise E05Error(f"E04 control prompt lineage changed: {question_id}")
        answers: dict[str, Any] = {
            "control_ranked_top20": {
                **source_control,
                "answer_source": "reused-e04-control-no-rerank",
                "generation_latency_ms": None,
            }
        }
        for variant_key in ("ranked_top12", "document_article_diverse_top12"):
            contexts = prepared_rows[sample_index]["variants"][variant_key]["contexts"]
            selected, messages, input_tokens = pack_contexts(
                question=dev[question_id]["question"], contexts=contexts,
                config=config, token_counter=token_count,
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
                max_new_tokens=config.max_new_tokens, use_cache=True,
            )
            latency_ms = (time.perf_counter() - started) * 1000
            prompt_width = inputs["input_ids"].shape[1]
            answer = tokenizer.decode(
                generated[0, prompt_width:], skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            ).strip()
            if not answer:
                raise E05Error(f"Generator returned an empty answer: {variant_key}/{question_id}")
            answers[variant_key] = {
                "answer": answer,
                "answer_source": "generated-e05",
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
            "e05_generation_progress worker=%d device=%s worker_completed=%d "
            "worker_total=%d question_id=%s",
            worker_rank, device, position + 1, len(assigned), question_id,
        )
    _write_worker_state(state_path, identity, len(assigned), len(assigned))
    return {"worker_rank": worker_rank, "device": device, "completed": len(assigned)}


def finalize_generation(
    *, output_directory: Path, dev_path: Path, config: E05Config,
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
    *, output_directory: Path, dev_path: Path, config: E05Config,
) -> dict[str, Any]:
    ensure_nltk_resources(download=False)
    results_path = output_directory / "generation" / "results.jsonl"
    prepared_path = output_directory / "prepared" / "results.jsonl"
    if not results_path.is_file() or not prepared_path.is_file():
        raise E05Error("Finalize E05 generation before scoring.")
    rows = _read_jsonl(results_path)
    prepared_rows = _read_jsonl(prepared_path)
    dev = _load_dev(dev_path)
    sample_ids = select_dev_sample(dev, seed=config.sample_seed, size=config.sample_size)
    if [row.get("question_id") for row in rows] != sample_ids:
        raise E05Error("E05 generation order changed before scoring.")
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
            "candidate_limit": variant.candidate_limit,
            "ordering": variant.ordering,
            "meteor": fmean(row["meteor"] for row in scored[key]),
            "rouge_l": fmean(row["rouge_l"] for row in scored[key]),
            "mean_generation_latency_ms": fmean(latencies) if latencies else None,
            "mean_selected_contexts": fmean(
                row["variants"][key]["selected_context_count"] for row in rows
            ),
            "mean_input_tokens": fmean(row["variants"][key]["input_tokens"] for row in rows),
            "mean_unique_documents": fmean(
                len({context["document_id"] for context in source["variants"][key]["contexts"]})
                for source in prepared_rows
            ),
            "mean_unique_document_article_units": fmean(
                len({
                    (context["document_id"], context.get("article_number"))
                    for context in source["variants"][key]["contexts"]
                })
                for source in prepared_rows
            ),
            "answer_source": variant.answer_source,
        }
    control = config.control_variant
    if (
        metrics[control]["meteor"] != config.selection["meteor"]
        or metrics[control]["rouge_l"] != config.selection["rouge_l"]
    ):
        raise E05Error("Re-scored reused E04 control metrics changed.")

    comparisons = (
        ("ranked_top12", control),
        ("document_article_diverse_top12", control),
        ("document_article_diverse_top12", "ranked_top12"),
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
        "experiment_id": "E05-context-grid-aiteam-dev200-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "sample_size": len(rows),
        "selected_embedding": "embedding_aiteam",
        "fusion_weights": {"sparse": 0.5, "dense": 0.5},
        "reranker": None,
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
            "reused_e04_generation_results_sha256": config.selection[
                "generation_results_sha256"
            ],
            "reused_e04_context_results_sha256": config.selection[
                "context_results_sha256"
            ],
        },
        "warning": "Dev-200 is an E05 smoke grid; formal promotion requires full dev.",
    }
    _atomic_json(output_directory / "report.json", report)
    return report


def _validate_selection_report(report: dict[str, Any], config: E05Config) -> None:
    selection = config.selection
    metrics = report.get("metrics", {}).get(selection["selected_variant"], {})
    evidence = report.get("evidence", {})
    reranker_runtime = report.get("reranker_runtime", {})
    if (
        report.get("experiment_id") != selection["report_experiment_id"]
        or report.get("sample_size") != selection["sample_size"]
        or report.get("smoke_leader") != selection["selected_variant"]
        or report.get("promotion_allowed") is not False
        or report.get("selected_embedding") != selection["selected_embedding"]
        or evidence.get("config_sha256") != selection["report_config_sha256"]
        or evidence.get("generation_results_sha256")
        != selection["generation_results_sha256"]
        or reranker_runtime.get("results_sha256") != selection["context_results_sha256"]
        or metrics.get("meteor") != selection["meteor"]
        or metrics.get("rouge_l") != selection["rouge_l"]
    ):
        raise E05Error("Saved E04 report differs from the no-reranker selection evidence.")


def _validate_selection_rows(
    context_rows: list[dict[str, Any]], generation_rows: list[dict[str, Any]],
    sample_ids: list[str],
) -> None:
    if len(context_rows) != len(sample_ids) or len(generation_rows) != len(sample_ids):
        raise E05Error("E04 selection record count differs from E05.")
    for index, question_id in enumerate(sample_ids):
        context_row = context_rows[index]
        generation_row = generation_rows[index]
        contexts = context_row.get("variants", {}).get("control_no_rerank", {}).get("contexts")
        answer = generation_row.get("variants", {}).get("control_no_rerank", {}).get("answer")
        if (
            context_row.get("sample_index") != index
            or context_row.get("question_id") != question_id
            or generation_row.get("sample_index") != index
            or generation_row.get("question_id") != question_id
            or not isinstance(contexts, list) or len(contexts) != 20
            or len({context.get("chunk_id") for context in contexts}) != 20
            or any(not isinstance(context.get("text"), str) for context in contexts)
            or any(not isinstance(context.get("document_id"), str) for context in contexts)
            or not isinstance(answer, str) or not answer.strip()
        ):
            raise E05Error(f"E04 no-reranker selection row is incompatible: {index}")


def _validate_complete_records(
    root: Path, sample_ids: list[str], config: E05Config,
) -> list[dict[str, Any]]:
    records = root / "records"
    expected_variants = {variant.key for variant in config.variants}
    rows = []
    for index, question_id in enumerate(sample_ids):
        path = records / f"{index:04d}.json"
        if not path.is_file():
            raise E05Error(f"Generation record is missing: {index}")
        row = json.loads(path.read_text(encoding="utf-8"))
        if (
            row.get("sample_index") != index
            or row.get("question_id") != question_id
            or set(row.get("variants", {})) != expected_variants
        ):
            raise E05Error(f"Generation record identity changed: {index}")
        rows.append(row)
    for rank in range(config.worker_count):
        state_path = root / f"worker-{rank}-state.json"
        if not state_path.is_file() or json.loads(
            state_path.read_text(encoding="utf-8")
        ).get("complete") is not True:
            raise E05Error(f"Generation worker is incomplete: {rank}")
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
