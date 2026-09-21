"""Public-1000 confirmation run for the fresh E14 rank-16 max704 adapter."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from uit_dsc_fixed_rag.corpus import file_sha256
from uit_dsc_fixed_rag.e02_answer import _atomic_json
from uit_dsc_fixed_rag.e14_rank16_lora import E14Config, load_config as load_training_config
from uit_dsc_fixed_rag.final_public import (
    FinalPublicConfig,
    FinalPublicError,
    _load_official_scorer_bytes,
    finalize_submission,
    load_public_questions,
    run_generation_worker,
)
from uit_dsc_fixed_rag.final_public_e08b import (
    _validate_retrieval,
    prepare_reused_retrieval,
)
from uit_dsc_fixed_rag.e08b_context_lora import load_context_lora_generator


CODE_VERSION = "0.44.0"


def load_config(path: Path) -> FinalPublicConfig:
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "schema_version", "experiment_id", "model_selection", "retrieval_reuse",
        "full_lora_source", "public", "source_dense", "retrieval", "generator",
        "inference", "parameter_budget", "execution", "submission_contract",
        "run_contract",
    }
    if (
        set(payload) != required
        or payload.get("schema_version") != "1.0"
        or payload.get("experiment_id")
        != "FINAL-public1000-aiteam-e14-rank16-max704-v1"
    ):
        raise ValueError("E14 final-public config root is incompatible.")
    config = FinalPublicConfig(payload, path)
    selection = config.section("model_selection")
    if (
        selection.get("report_experiment_id")
        != "E14-rank16-vs-rank8-dev200-max704-v1"
        or selection.get("winner") != "e14_rank16_max704"
        or selection.get("control") != "e08b_rank8_max704_control"
        or selection.get("sample_size") != 200
        or selection.get("candidate_meteor") != 0.5634773414246171
        or selection.get("meteor_delta") != 0.0012616265122396724
        or selection.get("meteor_ci")
        != [-0.017371953990342245, 0.019256096794968126]
        or selection.get("promotion_allowed") is not False
        or selection.get("operator_selected_for_public_confirmation") is not True
    ):
        raise ValueError("E14 operator-selection evidence changed.")
    source = config.section("full_lora_source")
    if (
        source.get("experiment_id")
        != "E14-context-aware-lora-rank16-train5636-v1"
        or source.get("train_sample_size") != 5636
        or source.get("fresh_from_base") is not True
        or source.get("rank") != 16
        or source.get("alpha") != 32
        or source.get("trainable_parameters") != 16819200
        or source.get("old_rank8_adapter_loaded") is not False
    ):
        raise ValueError("E14 rank-16 training-source identity changed.")
    public = config.section("public")
    if (
        public.get("sample_size") != 1000
        or public.get("order") != "lexicographic-question-id"
        or public.get("answer_contract") != "all-null-and-never-read-for-inference"
    ):
        raise ValueError("E14 public question contract changed.")
    if config.section("retrieval") != {
        "candidate_k_per_branch": 40,
        "rrf_constant": 60,
        "sparse_weight": 0.5,
        "dense_weight": 0.5,
        "fused_top_k": 20,
        "selected_contexts": 12,
        "normalize_query_embeddings": True,
        "dense_search_backend": "torch-cuda-exact-flat-ip-float32-batched",
        "reranker": None,
    }:
        raise ValueError("Reused public retrieval contract changed.")
    inference = config.section("inference")
    if (
        inference.get("max_input_tokens") != 8192
        or inference.get("minimum_contexts") != 1
        or inference.get("max_new_tokens") != 704
        or inference.get("do_sample") is not False
        or inference.get("num_beams") != 1
        or inference.get("enable_thinking") is not False
        or inference.get("use_cache") is not True
    ):
        raise ValueError("E14 final max704 inference contract changed.")
    budget = config.section("parameter_budget")
    if (
        budget.get("maximum_stack_total")
        != budget.get("embedding") + budget.get("generator")
        + budget.get("adapter_parameter_cap")
        or budget["maximum_stack_total"] >= budget["exclusive_limit"]
    ):
        raise ValueError("E14 final parameter budget failed.")
    execution = config.section("execution")
    if (
        execution.get("generation_workers") != 2
        or execution.get("generation_partition")
        != "sample-index-mod-worker-count"
        or execution.get("questions_per_worker") != 500
        or execution.get("checkpoint_after_questions") != 1
    ):
        raise ValueError("E14 final execution contract changed.")
    submission = config.section("submission_contract")
    if (
        submission.get("archive_name") != "submission.zip"
        or submission.get("json_name") != "submission.json"
        or submission.get("archive_members") != ["submission.json"]
        or submission.get("record_schema") != {"answer": "non-empty-string"}
    ):
        raise ValueError("E14 final submission contract changed.")
    contract = config.section("run_contract")
    if not contract or not all(value is True for value in contract.values()):
        raise ValueError("E14 final run contract lost an invariant.")
    return config


def _training_config(project_root: Path, config: FinalPublicConfig) -> E14Config:
    expected = config.section("full_lora_source")
    path = project_root / expected["training_config_path"]
    if not path.is_file():
        raise FinalPublicError("Pinned E14 training config is missing.")
    training = load_training_config(path)
    if training.config_sha256 != expected["training_config_sha256"]:
        raise FinalPublicError("Pinned E14 training config changed.")
    if (
        training.section("prompt")["system_prompt"]
        != config.section("inference")["system_prompt"]
        or training.section("prompt")["answer_instruction"]
        != config.section("inference")["answer_instruction"]
        or training.section("inference")["max_new_tokens"] != 704
    ):
        raise FinalPublicError("E14 final prompt or output cap changed.")
    return training


def _validate_adapter(
    *, project_root: Path, e14_directory: Path, config: FinalPublicConfig,
) -> tuple[E14Config, str]:
    expected = config.section("full_lora_source")
    training = _training_config(project_root, config)
    final = e14_directory / "training" / "adapter-final"
    adapter_path = e14_directory / expected["adapter_relative_path"]
    complete_path = final / "complete.json"
    if not adapter_path.is_file() or not complete_path.is_file():
        raise FinalPublicError("Saved E14 rank-16 adapter is incomplete.")
    complete = json.loads(complete_path.read_text(encoding="utf-8"))
    adapter_hash = file_sha256(adapter_path)
    if (
        adapter_hash != expected["adapter_sha256"]
        or complete.get("experiment_id") != expected["experiment_id"]
        or complete.get("config_sha256") != expected["training_config_sha256"]
        or complete.get("source_contexts_sha256")
        != expected["source_contexts_sha256"]
        or complete.get("adapter_sha256") != adapter_hash
        or complete.get("fresh_from_base") is not True
        or complete.get("e07_adapter_loaded") is not False
        or complete.get("e08b_adapter_loaded") is not False
        or complete.get("lora_rank") != 16
        or complete.get("lora_alpha") != 32
        or complete.get("trainable_parameters") != expected["trainable_parameters"]
        or complete.get("evaluation_max_new_tokens") != 704
    ):
        raise FinalPublicError("Saved E14 rank-16 adapter evidence changed.")
    return training, adapter_hash


def _validate_selection(
    *, selection_directory: Path, config: FinalPublicConfig,
) -> str:
    expected = config.section("model_selection")
    report_path = selection_directory / "report.json"
    results_path = selection_directory / "results.jsonl"
    if not report_path.is_file() or not results_path.is_file():
        raise FinalPublicError("Saved E14 dev-200 selection is incomplete.")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    metrics = report.get("metrics", {})
    candidate = metrics.get(expected["winner"], {})
    control = metrics.get(expected["control"], {})
    paired = report.get("paired_delta_rank16_minus_rank8", {})
    evidence = report.get("evidence", {})
    if (
        report.get("experiment_id") != expected["report_experiment_id"]
        or report.get("sample_size") != expected["sample_size"]
        or report.get("smoke_leader") != expected["winner"]
        or report.get("promotion_allowed") is not expected["promotion_allowed"]
        or candidate.get("meteor") != expected["candidate_meteor"]
        or candidate.get("rouge_l") != expected["candidate_rouge_l"]
        or control.get("meteor") != expected["control_meteor"]
        or paired.get("meteor_mean") != expected["meteor_delta"]
        or paired.get("meteor_bootstrap_95_ci") != expected["meteor_ci"]
        or paired.get("rouge_l_mean") != expected["rouge_l_delta"]
        or paired.get("rouge_l_bootstrap_95_ci") != expected["rouge_l_ci"]
        or evidence.get("config_sha256") != expected["config_sha256"]
        or evidence.get("contexts_sha256") != expected["contexts_sha256"]
        or evidence.get("control_results_sha256")
        != expected["control_results_sha256"]
        or evidence.get("rank8_adapter_sha256") != expected["rank8_adapter_sha256"]
        or evidence.get("rank16_adapter_sha256")
        != expected["rank16_adapter_sha256"]
        or evidence.get("results_sha256")
        != expected["generation_results_sha256"]
    ):
        raise FinalPublicError("Saved E14 dev-200 selection evidence changed.")
    observed = file_sha256(results_path)
    if observed != expected["generation_results_sha256"]:
        raise FinalPublicError("Saved E14 dev-200 selection results changed.")
    return observed


def validate_preflight(
    *, project_root: Path, retrieval_directory: Path, e14_directory: Path,
    selection_directory: Path, public_path: Path, config: FinalPublicConfig,
) -> dict[str, Any]:
    _, ids = load_public_questions(public_path, config)
    scorer_cfg = config.section("submission_contract")
    scorer_bytes, scorer_container = _load_official_scorer_bytes(
        project_root, scorer_cfg
    )
    if hashlib.sha256(scorer_bytes).hexdigest() != scorer_cfg[
        "official_scorer_entry_sha256"
    ]:
        raise FinalPublicError("Official scorer mapping implementation changed.")
    scorer_text = scorer_bytes.decode("utf-8")
    if "v['answer']" not in scorer_text or "len(ids_preds) != len(ids_truth)" not in scorer_text:
        raise FinalPublicError("Official scorer mapping contract changed.")
    _, adapter_hash = _validate_adapter(
        project_root=project_root, e14_directory=e14_directory, config=config
    )
    selection_hash = _validate_selection(
        selection_directory=selection_directory, config=config
    )
    retrieval_cfg = config.section("retrieval_reuse")
    report_path = retrieval_directory / "report.json"
    retrieval_path = retrieval_directory / retrieval_cfg["path"]
    if not report_path.is_file() or not retrieval_path.is_file():
        raise FinalPublicError("Saved public retrieval artifact is incomplete.")
    retrieval_hash = _validate_retrieval(
        report=json.loads(report_path.read_text(encoding="utf-8")),
        path=retrieval_path,
        ids=ids,
        config=config,
    )
    return {
        "code_version": CODE_VERSION,
        "config_sha256": config.config_sha256,
        "public_sha256": config.section("public")["sha256"],
        "sample_ids_sha256": config.section("public")["sample_ids_sha256"],
        "sample_size": len(ids),
        "official_scorer_container": scorer_container,
        "official_scorer_entry_sha256": scorer_cfg["official_scorer_entry_sha256"],
        "rank16_adapter_sha256": adapter_hash,
        "model_selection_results_sha256": selection_hash,
        "retrieval_results_sha256": retrieval_hash,
        "lora_rank": 16,
        "max_new_tokens": 704,
    }


def load_generator(
    *, project_root: Path, e14_directory: Path, config: FinalPublicConfig,
    device: str,
) -> tuple[Any, Any, dict[str, Any], int]:
    training, _ = _validate_adapter(
        project_root=project_root, e14_directory=e14_directory, config=config
    )
    try:
        return load_context_lora_generator(
            config=training,
            training_directory=e14_directory / "training",
            device=device,
        )
    except Exception as exc:
        raise FinalPublicError(str(exc)) from exc


def finalize(
    *, output_directory: Path, public_path: Path, config: FinalPublicConfig,
) -> dict[str, Any]:
    report = finalize_submission(
        output_directory=output_directory, public_path=public_path, config=config
    )
    selection = config.section("model_selection")
    report["selected_stack"]["max_new_tokens"] = 704
    report["selected_stack"]["lora_rank"] = 16
    report["selected_stack"]["generator"] = selection["winner"]
    report["model_selection"] = {
        "primary_metric": "meteor",
        "dev200_control_meteor": selection["control_meteor"],
        "dev200_candidate_meteor": selection["candidate_meteor"],
        "paired_meteor_delta": selection["meteor_delta"],
        "paired_meteor_ci": selection["meteor_ci"],
        "operator_selected_for_public_confirmation": True,
        "formal_promotion": False,
    }
    report["evidence"]["reused_retrieval_results_sha256"] = file_sha256(
        output_directory / "retrieval" / "results.jsonl"
    )
    report["evidence"]["model_selection_results_sha256"] = selection[
        "generation_results_sha256"
    ]
    _atomic_json(output_directory / "report.json", report)
    return report


__all__ = [
    "FinalPublicError", "finalize", "load_config", "load_generator",
    "prepare_reused_retrieval", "run_generation_worker", "validate_preflight",
]
