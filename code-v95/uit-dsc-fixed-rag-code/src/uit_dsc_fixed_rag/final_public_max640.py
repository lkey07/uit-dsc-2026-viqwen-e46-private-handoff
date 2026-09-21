"""Public-1000 long-output generation over byte-verified saved retrieval."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from uit_dsc_fixed_rag.corpus import file_sha256
from uit_dsc_fixed_rag.e02_answer import _atomic_json
from uit_dsc_fixed_rag.e03_rrf_grid import _read_jsonl
from uit_dsc_fixed_rag.e07_full_dev import _validate_full_train_report
from uit_dsc_fixed_rag.final_public import (
    FinalPublicConfig,
    FinalPublicError,
    _load_official_scorer_bytes,
    finalize_submission,
    load_generator_with_adapter,
    load_public_questions,
    run_generation_worker,
)


CODE_VERSION = "0.26.0"


def load_max640_config(path: Path) -> FinalPublicConfig:
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected_lengths = {
        "FINAL-public1000-aiteam-qwen35-lora-max640-v2": 640,
        "FINAL-public1000-aiteam-qwen35-lora-max896-v3": 896,
    }
    required = {
        "schema_version",
        "experiment_id",
        "model_selection",
        "output_length_selection",
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
        or payload.get("experiment_id") not in expected_lengths
    ):
        raise ValueError("Long-output public config root is incompatible.")
    config = FinalPublicConfig(payload, path)
    inference = config.section("inference")
    expected_length = expected_lengths[payload["experiment_id"]]
    if (
        inference.get("max_input_tokens") != 8192
        or inference.get("minimum_contexts") != 1
        or inference.get("max_new_tokens") != expected_length
        or inference.get("do_sample") is not False
        or inference.get("num_beams") != 1
        or inference.get("enable_thinking") is not False
        or inference.get("use_cache") is not True
    ):
        raise ValueError("Long-output deterministic inference contract changed.")
    length_selection = config.section("output_length_selection")
    if expected_length == 896 and (
        length_selection.get("public_requested_max_new_tokens") != 896
        or length_selection.get("public_selection_mode")
        != "operator-extrapolated-from-max640-because-meteor-is-primary-and-half-hit-cap"
    ):
        raise ValueError("Max-896 operator-selection evidence changed.")
    retrieval = config.section("retrieval")
    if retrieval != {
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
        raise ValueError("Reused retrieval contract changed.")
    execution = config.section("execution")
    if (
        execution.get("generation_workers") != 2
        or execution.get("checkpoint_after_questions") != 1
    ):
        raise ValueError("Long-output checkpoint contract changed.")
    budget = config.section("parameter_budget")
    if (
        budget.get("maximum_stack_total")
        != budget.get("embedding")
        + budget.get("generator")
        + budget.get("adapter_parameter_cap")
        or budget["maximum_stack_total"] >= budget["exclusive_limit"]
    ):
        raise ValueError("Long-output parameter budget failed.")
    submission = config.section("submission_contract")
    if (
        submission.get("archive_name") != "submission.zip"
        or submission.get("json_name") != "submission.json"
        or submission.get("archive_members") != ["submission.json"]
        or submission.get("record_schema") != {"answer": "non-empty-string"}
    ):
        raise ValueError("Long-output submission contract changed.")
    contract = config.section("run_contract")
    if not contract or not all(value is True for value in contract.values()):
        raise ValueError("Long-output run contract lost an invariant.")
    return config


def validate_max640_preflight(
    *,
    project_root: Path,
    retrieval_directory: Path,
    full_lora_directory: Path,
    e09_directory: Path,
    public_path: Path,
    config: FinalPublicConfig,
) -> dict[str, Any]:
    _, ids = load_public_questions(public_path, config)
    scorer_cfg = config.section("submission_contract")
    scorer_bytes, scorer_container = _load_official_scorer_bytes(
        project_root, scorer_cfg
    )
    if file_sha256_bytes(scorer_bytes) != scorer_cfg["official_scorer_entry_sha256"]:
        raise FinalPublicError("Official scorer mapping implementation changed.")
    scorer_text = scorer_bytes.decode("utf-8")
    if "v['answer']" not in scorer_text or "len(ids_preds) != len(ids_truth)" not in scorer_text:
        raise FinalPublicError("Official scorer mapping contract changed.")

    lora_cfg = config.section("full_lora_source")
    lora_report_path = full_lora_directory / "report.json"
    lora_results_path = full_lora_directory / "evaluation" / "results.jsonl"
    adapter_path = (
        full_lora_directory
        / "training"
        / "adapter-final"
        / "adapter_model.safetensors"
    )
    if not all(
        path.is_file()
        for path in (lora_report_path, lora_results_path, adapter_path)
    ):
        raise FinalPublicError("Saved full-train LoRA artifact is incomplete.")
    _validate_full_train_report(
        json.loads(lora_report_path.read_text(encoding="utf-8")), lora_cfg
    )
    if (
        file_sha256(lora_results_path) != lora_cfg["evaluation_results_sha256"]
        or file_sha256(adapter_path) != lora_cfg["adapter_sha256"]
    ):
        raise FinalPublicError("Saved full-train LoRA evidence changed.")

    e09_cfg = config.section("output_length_selection")
    e09_report_path = e09_directory / "report.json"
    e09_results_path = e09_directory / "results.jsonl"
    if not e09_report_path.is_file() or not e09_results_path.is_file():
        raise FinalPublicError("Saved E09 output-length artifact is incomplete.")
    e09_report = json.loads(e09_report_path.read_text(encoding="utf-8"))
    _validate_e09_report(e09_report, e09_cfg)
    if file_sha256(e09_results_path) != e09_cfg["results_sha256"]:
        raise FinalPublicError("Saved E09 generation results changed.")

    retrieval_cfg = config.section("retrieval_reuse")
    retrieval_report_path = retrieval_directory / "report.json"
    retrieval_path = retrieval_directory / retrieval_cfg["path"]
    if not retrieval_report_path.is_file() or not retrieval_path.is_file():
        raise FinalPublicError("Saved public retrieval artifact is incomplete.")
    retrieval_report = json.loads(
        retrieval_report_path.read_text(encoding="utf-8")
    )
    retrieval_hash = _validate_retrieval_artifact(
        report=retrieval_report,
        retrieval_path=retrieval_path,
        ids=ids,
        config=config,
    )
    return {
        "config_sha256": config.config_sha256,
        "public_sha256": config.section("public")["sha256"],
        "sample_ids_sha256": config.section("public")["sample_ids_sha256"],
        "sample_size": len(ids),
        "official_scorer_container": scorer_container,
        "official_scorer_entry_sha256": scorer_cfg["official_scorer_entry_sha256"],
        "full_lora_adapter_sha256": lora_cfg["adapter_sha256"],
        "e09_results_sha256": e09_cfg["results_sha256"],
        "selected_output_length": f"max{config.section('inference')['max_new_tokens']}",
        "max_new_tokens": config.section("inference")["max_new_tokens"],
        "retrieval_results_sha256": retrieval_hash,
    }


def prepare_reused_retrieval(
    *,
    retrieval_directory: Path,
    public_path: Path,
    output_directory: Path,
    config: FinalPublicConfig,
) -> dict[str, Any]:
    _, ids = load_public_questions(public_path, config)
    source_cfg = config.section("retrieval_reuse")
    source = retrieval_directory / source_cfg["path"]
    report = json.loads(
        (retrieval_directory / "report.json").read_text(encoding="utf-8")
    )
    expected_hash = _validate_retrieval_artifact(
        report=report, retrieval_path=source, ids=ids, config=config
    )
    destination = output_directory / "retrieval" / "results.jsonl"
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".jsonl.tmp")
    temporary.write_bytes(source.read_bytes())
    temporary.replace(destination)
    if file_sha256(destination) != expected_hash:
        raise FinalPublicError("Copied public retrieval bytes changed.")
    evidence = {
        "source_experiment_id": source_cfg["report_experiment_id"],
        "source_config_sha256": source_cfg["config_sha256"],
        "results_sha256": expected_hash,
        "sample_size": len(ids),
        "contexts_per_question": source_cfg["contexts_per_question"],
    }
    _atomic_json(output_directory / "retrieval-source.json", evidence)
    return evidence


def finalize_max640_submission(
    *,
    output_directory: Path,
    public_path: Path,
    config: FinalPublicConfig,
) -> dict[str, Any]:
    report = finalize_submission(
        output_directory=output_directory,
        public_path=public_path,
        config=config,
    )
    selection = config.section("output_length_selection")
    requested_length = config.section("inference")["max_new_tokens"]
    report["selected_stack"]["max_new_tokens"] = requested_length
    report["output_length_selection"] = {
        "metric": "meteor",
        "requested_variant": f"max{requested_length}",
        "selection_mode": selection.get(
            "public_selection_mode", "operator-selected-e09-leader"
        ),
        "e09_measured_leader": "max640",
        "e09_max640_meteor": selection["max640_meteor"],
        "e09_max640_meteor_delta_vs_control384": selection[
            "max640_meteor_delta_vs_control"
        ],
        "operator_selected": True,
    }
    report["evidence"]["e09_results_sha256"] = selection["results_sha256"]
    report["evidence"]["reused_retrieval_results_sha256"] = file_sha256(
        output_directory / "retrieval" / "results.jsonl"
    )
    _atomic_json(output_directory / "report.json", report)
    return report


def _validate_e09_report(report: dict[str, Any], expected: dict[str, Any]) -> None:
    metrics = report.get("metrics", {})
    paired = report.get("paired_deltas_vs_control384", {}).get(
        "max640-minus-control384", {}
    )
    evidence = report.get("evidence", {})
    if (
        report.get("experiment_id") != expected["report_experiment_id"]
        or report.get("sample_size") != expected["sample_size"]
        or report.get("only_changed_factor") != expected["only_changed_factor"]
        or report.get("smoke_leader") != expected["winner"]
        or report.get("promotion_allowed") is not expected["promotion_allowed"]
        or metrics.get("control384", {}).get("meteor")
        != expected["control384_meteor"]
        or metrics.get("max512", {}).get("meteor") != expected["max512_meteor"]
        or metrics.get("max640", {}).get("meteor") != expected["max640_meteor"]
        or metrics.get("max640", {}).get("length_finish_rate")
        != expected["max640_length_finish_rate"]
        or paired.get("meteor_mean")
        != expected["max640_meteor_delta_vs_control"]
        or paired.get("meteor_bootstrap_95_ci", [None])[0]
        != expected["max640_meteor_ci_lower"]
        or evidence.get("config_sha256") != expected["config_sha256"]
        or evidence.get("contexts_sha256") != expected["contexts_sha256"]
        or evidence.get("control_results_sha256")
        != expected["control_results_sha256"]
        or evidence.get("adapter_sha256") != expected["adapter_sha256"]
        or evidence.get("results_sha256") != expected["results_sha256"]
    ):
        raise FinalPublicError("Saved E09 output-length report changed.")


def _validate_retrieval_artifact(
    *,
    report: dict[str, Any],
    retrieval_path: Path,
    ids: list[str],
    config: FinalPublicConfig,
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
        or evidence.get("adapter_sha256")
        != config.section("full_lora_source")["adapter_sha256"]
    ):
        raise FinalPublicError("Saved public retrieval report changed.")
    observed_hash = file_sha256(retrieval_path)
    if evidence.get("retrieval_results_sha256") != observed_hash:
        raise FinalPublicError("Saved public retrieval bytes changed.")
    rows = _read_jsonl(retrieval_path)
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
    return observed_hash


def file_sha256_bytes(payload: bytes) -> str:
    import hashlib

    return hashlib.sha256(payload).hexdigest()


__all__ = [
    "FinalPublicError",
    "finalize_max640_submission",
    "load_generator_with_adapter",
    "load_max640_config",
    "prepare_reused_retrieval",
    "run_generation_worker",
    "validate_max640_preflight",
]
