"""Public-1000 trial for E14 rank-16, top-8 contexts and max1024 output."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from uit_dsc_fixed_rag.corpus import file_sha256
from uit_dsc_fixed_rag.e02_answer import _atomic_json
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
from uit_dsc_fixed_rag.final_public_e14 import (
    _validate_adapter,
    load_generator,
)


CODE_VERSION = "0.47.0"
EXPERIMENT_ID = "FINAL-public1000-aiteam-e16-rank16-top8-max1024-v1"


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
        or payload.get("experiment_id") != EXPERIMENT_ID
    ):
        raise ValueError("E16 final-public config root is incompatible.")
    config = FinalPublicConfig(payload, path)
    selection = config.section("model_selection")
    if (
        selection.get("report_experiment_id")
        != "E16-rank16-top8-output-length-grid-dev200-v1"
        or selection.get("smoke_leader") != "rank16_top8_max704_control"
        or selection.get("operator_selected_variant") != "rank16_top8_max1024"
        or selection.get("winner") != "rank16_top8_max1024"
        or selection.get("sample_size") != 200
        or selection.get("control_meteor") != 0.5673159604551824
        or selection.get("candidate_meteor") != 0.5624775916726549
        or selection.get("meteor_delta") != -0.004838368782527493
        or selection.get("candidate_length_finish_rate") != 0.22
        or selection.get("promotion_allowed") is not False
        or selection.get("operator_selected_for_public_trial") is not True
    ):
        raise ValueError("E16 max1024 operator-selection evidence changed.")
    source = config.section("full_lora_source")
    if (
        source.get("experiment_id")
        != "E14-context-aware-lora-rank16-train5636-v1"
        or source.get("train_sample_size") != 5636
        or source.get("fresh_from_base") is not True
        or source.get("rank") != 16
        or source.get("alpha") != 32
        or source.get("trainable_parameters") != 16_819_200
        or source.get("old_rank8_adapter_loaded") is not False
    ):
        raise ValueError("E16 public rank-16 adapter identity changed.")
    public = config.section("public")
    if (
        public.get("sample_size") != 1000
        or public.get("order") != "lexicographic-question-id"
        or public.get("answer_contract") != "all-null-and-never-read-for-inference"
    ):
        raise ValueError("E16 public question contract changed.")
    if config.section("retrieval") != {
        "candidate_k_per_branch": 40,
        "rrf_constant": 60,
        "sparse_weight": 0.5,
        "dense_weight": 0.5,
        "fused_top_k": 20,
        "selected_contexts": 8,
        "normalize_query_embeddings": True,
        "dense_search_backend": "torch-cuda-exact-flat-ip-float32-batched",
        "reranker": None,
    }:
        raise ValueError("E16 public top-8 retrieval contract changed.")
    inference = config.section("inference")
    if (
        inference.get("max_input_tokens") != 8192
        or inference.get("minimum_contexts") != 1
        or inference.get("max_new_tokens") != 1024
        or inference.get("do_sample") is not False
        or inference.get("num_beams") != 1
        or inference.get("enable_thinking") is not False
        or inference.get("use_cache") is not True
        or inference.get("repetition_penalty") != 1.0
        or inference.get("no_repeat_ngram_size") != 0
    ):
        raise ValueError("E16 public max1024 inference contract changed.")
    budget = config.section("parameter_budget")
    if (
        budget.get("maximum_stack_total")
        != budget.get("embedding") + budget.get("generator")
        + budget.get("adapter_parameter_cap")
        or budget["maximum_stack_total"] >= budget["exclusive_limit"]
    ):
        raise ValueError("E16 public parameter budget failed.")
    execution = config.section("execution")
    if (
        execution.get("generation_workers") != 2
        or execution.get("generation_partition")
        != "sample-index-mod-worker-count"
        or execution.get("questions_per_worker") != 500
        or execution.get("checkpoint_after_questions") != 1
    ):
        raise ValueError("E16 public execution contract changed.")
    submission = config.section("submission_contract")
    if (
        submission.get("archive_name") != "submission.zip"
        or submission.get("json_name") != "submission.json"
        or submission.get("archive_members") != ["submission.json"]
        or submission.get("record_schema") != {"answer": "non-empty-string"}
    ):
        raise ValueError("E16 public submission contract changed.")
    contract = config.section("run_contract")
    if not contract or not all(value is True for value in contract.values()):
        raise ValueError("E16 public run contract lost an invariant.")
    return config


def _validate_selection(
    *, selection_directory: Path, config: FinalPublicConfig,
) -> str:
    expected = config.section("model_selection")
    report_path = selection_directory / "report.json"
    results_path = selection_directory / "results.jsonl"
    if not report_path.is_file() or not results_path.is_file():
        raise FinalPublicError("Saved E16 dev-200 output-length grid is incomplete.")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    metrics = report.get("metrics", {})
    control = metrics.get("rank16_top8_max704_control", {})
    candidate = metrics.get("rank16_top8_max1024", {})
    paired = report.get("paired_deltas_vs_max704_control", {}).get(
        "rank16_top8_max1024", {}
    )
    evidence = report.get("evidence", {})
    if (
        report.get("experiment_id") != expected["report_experiment_id"]
        or report.get("sample_size") != expected["sample_size"]
        or report.get("fixed_context_limit") != 8
        or report.get("smoke_leader") != expected["smoke_leader"]
        or report.get("promotion_allowed") is not expected["promotion_allowed"]
        or control.get("meteor") != expected["control_meteor"]
        or control.get("rouge_l") != expected["control_rouge_l"]
        or control.get("length_finish_rate")
        != expected["control_length_finish_rate"]
        or candidate.get("meteor") != expected["candidate_meteor"]
        or candidate.get("rouge_l") != expected["candidate_rouge_l"]
        or candidate.get("length_finish_rate")
        != expected["candidate_length_finish_rate"]
        or paired.get("meteor_mean") != expected["meteor_delta"]
        or paired.get("meteor_bootstrap_95_ci") != expected["meteor_ci"]
        or paired.get("rouge_l_mean") != expected["rouge_l_delta"]
        or paired.get("rouge_l_bootstrap_95_ci") != expected["rouge_l_ci"]
        or evidence.get("config_sha256") != expected["config_sha256"]
        or evidence.get("contexts_sha256") != expected["contexts_sha256"]
        or evidence.get("control_results_sha256")
        != expected["control_results_sha256"]
        or evidence.get("rank16_adapter_sha256")
        != expected["rank16_adapter_sha256"]
        or evidence.get("results_sha256")
        != expected["generation_results_sha256"]
    ):
        raise FinalPublicError("Saved E16 dev-200 evidence changed.")
    observed = file_sha256(results_path)
    if observed != expected["generation_results_sha256"]:
        raise FinalPublicError("Saved E16 dev-200 results changed.")
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
        "source_contexts_per_question": retrieval_cfg["contexts_per_question"],
        "selected_contexts": 8,
        "lora_rank": 16,
        "max_new_tokens": 1024,
    }


def finalize(
    *, output_directory: Path, public_path: Path, config: FinalPublicConfig,
) -> dict[str, Any]:
    report = finalize_submission(
        output_directory=output_directory, public_path=public_path, config=config
    )
    selection = config.section("model_selection")
    report["selected_stack"]["max_new_tokens"] = 1024
    report["selected_stack"]["lora_rank"] = 16
    report["selected_stack"]["generator"] = selection[
        "operator_selected_variant"
    ]
    report["model_selection"] = {
        "primary_metric": "meteor",
        "dev200_smoke_leader": selection["smoke_leader"],
        "dev200_control_meteor": selection["control_meteor"],
        "dev200_candidate_meteor": selection["candidate_meteor"],
        "paired_meteor_delta": selection["meteor_delta"],
        "paired_meteor_ci": selection["meteor_ci"],
        "control_length_finish_rate": selection["control_length_finish_rate"],
        "candidate_length_finish_rate": selection["candidate_length_finish_rate"],
        "operator_selected_for_public_trial": True,
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
