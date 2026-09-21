"""Final public-1000 generation with the selected E08B context-aware LoRA."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from uit_dsc_fixed_rag.corpus import file_sha256
from uit_dsc_fixed_rag.e02_answer import _atomic_json
from uit_dsc_fixed_rag.e03_rrf_grid import _read_jsonl
from uit_dsc_fixed_rag.e08b_context_lora import (
    E08BConfig,
    load_context_lora_generator,
    load_e08b_config,
)
from uit_dsc_fixed_rag.final_public import (
    FinalPublicConfig,
    FinalPublicError,
    _load_official_scorer_bytes,
    finalize_submission,
    load_public_questions,
    run_generation_worker,
)


CODE_VERSION = "0.40.0"


def load_config(path: Path) -> FinalPublicConfig:
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "schema_version",
        "experiment_id",
        "model_selection",
        "retrieval_reuse",
        "full_lora_source",
        "public",
        "source_dense",
        "retrieval",
        "generator",
        "inference",
        "parameter_budget",
        "execution",
        "submission_contract",
        "run_contract",
    }
    if (
        set(payload) != required
        or payload.get("schema_version") != "1.0"
        or payload.get("experiment_id")
        != "FINAL-public1000-aiteam-e08b-context-lora-max704-v1"
    ):
        raise ValueError("E08B final-public config root is incompatible.")
    config = FinalPublicConfig(payload, path)
    selection = config.section("model_selection")
    if (
        selection.get("winner") != "e08b_context_lora_max704"
        or selection.get("sample_size") != 200
        or selection.get("candidate_meteor") != 0.5622157149123774
        or selection.get("meteor_delta") != 0.09255685064640623
        or selection.get("meteor_ci_lower") != 0.06405177826665513
        or selection.get("promotion_allowed") is not False
    ):
        raise ValueError("E08B operator-selection evidence changed.")
    source = config.section("full_lora_source")
    if (
        source.get("experiment_id")
        != "E08B-context-aware-lora-train5636-dev521-v3"
        or source.get("train_sample_size") != 5636
        or source.get("fresh_from_base") is not True
    ):
        raise ValueError("E08B training-source identity changed.")
    public = config.section("public")
    if (
        public.get("sample_size") != 1000
        or public.get("order") != "lexicographic-question-id"
        or public.get("answer_contract") != "all-null-and-never-read-for-inference"
    ):
        raise ValueError("Final public question contract changed.")
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
        raise ValueError("E08B final deterministic inference contract changed.")
    budget = config.section("parameter_budget")
    if (
        budget.get("maximum_stack_total")
        != budget.get("embedding")
        + budget.get("generator")
        + budget.get("adapter_parameter_cap")
        or budget["maximum_stack_total"] >= budget["exclusive_limit"]
    ):
        raise ValueError("E08B final parameter budget failed.")
    execution = config.section("execution")
    if (
        execution.get("generation_workers") != 2
        or execution.get("questions_per_worker") != 500
        or execution.get("checkpoint_after_questions") != 1
    ):
        raise ValueError("E08B final execution contract changed.")
    submission = config.section("submission_contract")
    if (
        submission.get("archive_name") != "submission.zip"
        or submission.get("json_name") != "submission.json"
        or submission.get("archive_members") != ["submission.json"]
        or submission.get("record_schema") != {"answer": "non-empty-string"}
    ):
        raise ValueError("E08B final submission contract changed.")
    contract = config.section("run_contract")
    if not contract or not all(value is True for value in contract.values()):
        raise ValueError("E08B final run contract lost an invariant.")
    return config


def _training_config(project_root: Path, config: FinalPublicConfig) -> E08BConfig:
    expected = config.section("full_lora_source")
    path = project_root / expected["training_config_path"]
    training_config = load_e08b_config(path)
    if training_config.config_sha256 != expected["training_config_sha256"]:
        raise FinalPublicError("Pinned E08B training config changed.")
    return training_config


def _validate_adapter(
    *, project_root: Path, e08b_directory: Path, config: FinalPublicConfig
) -> tuple[E08BConfig, str]:
    expected = config.section("full_lora_source")
    training_config = _training_config(project_root, config)
    final = e08b_directory / "training" / "adapter-final"
    adapter_path = e08b_directory / expected["adapter_relative_path"]
    complete_path = final / "complete.json"
    if not adapter_path.is_file() or not complete_path.is_file():
        raise FinalPublicError("Saved E08B context-aware adapter is incomplete.")
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
    ):
        raise FinalPublicError("Saved E08B context-aware adapter evidence changed.")
    return training_config, adapter_hash


def _validate_selection(
    *, selection_directory: Path, config: FinalPublicConfig
) -> str:
    expected = config.section("model_selection")
    report_path = selection_directory / "report.json"
    results_path = selection_directory / "results.jsonl"
    if not report_path.is_file() or not results_path.is_file():
        raise FinalPublicError("Saved E08B paired dev-200 selection is incomplete.")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    metrics = report.get("metrics", {})
    candidate = metrics.get("e08b_context_lora_max704", {})
    control = metrics.get("e11_max704_control", {})
    paired = report.get("paired_delta_e08b_minus_e11", {})
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
        or paired.get("meteor_bootstrap_95_ci", [None])[0]
        != expected["meteor_ci_lower"]
        or paired.get("rouge_l_mean") != expected["rouge_l_delta"]
        or paired.get("rouge_l_bootstrap_95_ci", [None])[0]
        != expected["rouge_l_ci_lower"]
        or evidence.get("config_sha256") != expected["config_sha256"]
        or evidence.get("contexts_sha256") != expected["contexts_sha256"]
        or evidence.get("control_results_sha256")
        != expected["control_results_sha256"]
        or evidence.get("adapter_sha256") != expected["adapter_sha256"]
        or evidence.get("results_sha256")
        != expected["generation_results_sha256"]
    ):
        raise FinalPublicError("Saved E08B paired dev-200 evidence changed.")
    observed = file_sha256(results_path)
    if observed != expected["generation_results_sha256"]:
        raise FinalPublicError("Saved E08B paired dev-200 results changed.")
    return observed


def _validate_retrieval(
    *, report: dict[str, Any], path: Path, ids: list[str], config: FinalPublicConfig
) -> str:
    expected = config.section("retrieval_reuse")
    stack = report.get("selected_stack", {})
    validation = report.get("validation", {})
    evidence = report.get("evidence", {})
    if (
        report.get("experiment_id") != expected["report_experiment_id"]
        or report.get("sample_size") != expected["sample_size"]
        or report.get("public_answers_read") is not False
        or report.get("holdout_untouched") is not True
        or stack.get("embedding") != "embedding_aiteam"
        or stack.get("fusion") != "bm25-dense-weighted-rrf-0.5-0.5"
        or stack.get("reranker") is not None
        or stack.get("contexts") != "ranked-top12"
        or validation.get("all_question_ids_present") is not True
        or validation.get("all_answers_non_empty") is not True
        or evidence.get("config_sha256") != expected["config_sha256"]
        or evidence.get("public_sha256") != config.section("public")["sha256"]
        or evidence.get("sample_ids_sha256")
        != config.section("public")["sample_ids_sha256"]
    ):
        raise FinalPublicError("Saved public retrieval report changed.")
    observed = file_sha256(path)
    if evidence.get("retrieval_results_sha256") != observed:
        raise FinalPublicError("Saved public retrieval bytes changed.")
    rows = _read_jsonl(path)
    if len(rows) != len(ids):
        raise FinalPublicError("Saved public retrieval count changed.")
    for index, (question_id, row) in enumerate(zip(ids, rows)):
        contexts = row.get("contexts")
        if (
            row.get("sample_index") != index
            or row.get("question_id") != question_id
            or "answer" in row
            or not isinstance(contexts, list)
            or len(contexts) != expected["contexts_per_question"]
        ):
            raise FinalPublicError(f"Invalid saved public retrieval row: {index}")
    return observed


def validate_preflight(
    *, project_root: Path, retrieval_directory: Path, e08b_directory: Path,
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
        project_root=project_root, e08b_directory=e08b_directory, config=config
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
        "adapter_sha256": adapter_hash,
        "model_selection_results_sha256": selection_hash,
        "retrieval_results_sha256": retrieval_hash,
        "max_new_tokens": 704,
    }


def prepare_reused_retrieval(
    *, retrieval_directory: Path, public_path: Path, output_directory: Path,
    config: FinalPublicConfig,
) -> dict[str, Any]:
    _, ids = load_public_questions(public_path, config)
    expected = config.section("retrieval_reuse")
    source = retrieval_directory / expected["path"]
    report = json.loads((retrieval_directory / "report.json").read_text(encoding="utf-8"))
    source_hash = _validate_retrieval(
        report=report, path=source, ids=ids, config=config
    )
    destination = output_directory / "retrieval" / "results.jsonl"
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".jsonl.tmp")
    temporary.write_bytes(source.read_bytes())
    temporary.replace(destination)
    if file_sha256(destination) != source_hash:
        raise FinalPublicError("Copied public retrieval bytes changed.")
    evidence = {
        "source_experiment_id": expected["report_experiment_id"],
        "source_config_sha256": expected["config_sha256"],
        "results_sha256": source_hash,
        "sample_size": len(ids),
        "contexts_per_question": expected["contexts_per_question"],
    }
    _atomic_json(output_directory / "retrieval-source.json", evidence)
    return evidence


def load_generator(
    *, project_root: Path, e08b_directory: Path, config: FinalPublicConfig,
    device: str,
) -> tuple[Any, Any, dict[str, Any], int]:
    training_config, _ = _validate_adapter(
        project_root=project_root, e08b_directory=e08b_directory, config=config
    )
    try:
        return load_context_lora_generator(
            config=training_config,
            training_directory=e08b_directory / "training",
            device=device,
        )
    except Exception as exc:
        raise FinalPublicError(str(exc)) from exc


def finalize(
    *, output_directory: Path, public_path: Path, config: FinalPublicConfig
) -> dict[str, Any]:
    report = finalize_submission(
        output_directory=output_directory, public_path=public_path, config=config
    )
    selection = config.section("model_selection")
    report["selected_stack"]["max_new_tokens"] = 704
    report["selected_stack"]["generator"] = selection["winner"]
    report["model_selection"] = {
        "primary_metric": "meteor",
        "dev200_meteor": selection["candidate_meteor"],
        "dev200_rouge_l": selection["candidate_rouge_l"],
        "paired_meteor_delta": selection["meteor_delta"],
        "paired_meteor_ci_lower": selection["meteor_ci_lower"],
        "operator_selected": True,
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
    "FinalPublicError",
    "finalize",
    "load_config",
    "load_generator",
    "prepare_reused_retrieval",
    "run_generation_worker",
    "validate_preflight",
]
