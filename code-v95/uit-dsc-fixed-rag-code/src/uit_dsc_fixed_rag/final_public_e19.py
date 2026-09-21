"""Operator-selected public-1000 trial of the E19 metadata-trained adapter."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

from .corpus import file_sha256
from .e02_answer import _atomic_json, _atomic_jsonl
from .e03_rrf_grid import _json_sha256, _read_jsonl
from .e18_source_metadata import enrich_context, scan_selected
from .e19_metadata_lora import (
    load_candidate_generator,
    load_config as load_e19_training_config,
    validate_candidate_adapter,
)
from .final_public import (
    FinalPublicConfig,
    FinalPublicError,
    _load_official_scorer_bytes,
    finalize_submission,
    load_public_questions,
    run_generation_worker as _run_generation_worker,
)
from .final_public_e08b import _validate_retrieval
from .final_public_e18 import load_config as load_e18_public_config


EXPERIMENT = "FINAL-public1000-e19-metadata-trained-max704-v1"
CODE_VERSION = "0.52.0"


def _load_trial(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if set(payload) != {
        "schema_version", "experiment_id", "base_config_path",
        "base_config_sha256", "training_config_path",
        "training_config_sha256", "model_selection", "full_lora_source",
        "public_trial",
    }:
        raise FinalPublicError("E19 public config root changed.")
    if payload.get("schema_version") != "1.0" or payload.get("experiment_id") != EXPERIMENT:
        raise FinalPublicError("E19 public experiment identity changed.")
    if payload["public_trial"] != {
        "context_count": 12,
        "max_input_tokens": 8192,
        "max_new_tokens": 704,
        "generation_workers": 2,
        "questions_per_worker": 500,
        "public_answers_never_read": True,
        "retrieval_and_metadata_unchanged_from_e18_public": True,
        "only_generator_adapter_changed": True,
        "rollback_public_meteor": 0.5251,
        "rollback_public_rouge_l": 0.5623,
    }:
        raise FinalPublicError("E19 public trial contract changed.")
    return payload


def _selection_contract(selection: dict[str, Any]) -> None:
    if selection != {
        "report_experiment_id": "E19-metadata-aware-lora-train5636-eval200-v1",
        "winner": "e19_metadata_trained_rank8_max704",
        "control": "e08b_rank8_metadata_max704",
        "sample_scope": "next-200-after-old-dev200-and-e18-dev200",
        "sample_size": 200,
        "config_sha256": "ce7d391f40475d69e0505c17e1644783d65c56e31d5f7a6b8169c2420fb5e480",
        "sample_ids_sha256": "dc17d6c9af9e941868c03cfc61fd17cdd1dc55aa5834beb3e75ba0fde32a5c73",
        "control_results_sha256": "7c19a35ef13496c297ce6a65b13bc1b6423abefb7e296a26908b911529dfe611",
        "candidate_results_sha256": "a79e0ba7afadb6d76a2ed60884551145de1e102ffe2b0bdf7a9f9b7178ce0a9c",
        "control_adapter_sha256": "14316bdb1a999e67c9baae90620c785977856e68cd540d46dbc57f1b2176499d",
        "candidate_adapter_sha256": "bfb9d8120337c4013b5dcb0b59c6d03beccfe9f9748e5a01ef62e1de64fb5edd",
        "control_meteor": 0.510773073430154,
        "candidate_meteor": 0.5233168277186363,
        "control_rouge_l": 0.5620342954062643,
        "candidate_rouge_l": 0.572299998528628,
        "meteor_delta": 0.012543754288482267,
        "meteor_ci": [-0.008864353648602323, 0.0338959974877113],
        "improved_questions": 91,
        "worsened_questions": 83,
        "tied_questions": 26,
        "promotion_allowed": False,
        "operator_selected_for_public_trial": True,
    }:
        raise FinalPublicError("E19 public selection contract changed.")


def load_config(project_root: Path, path: Path) -> FinalPublicConfig:
    trial = _load_trial(path)
    base_path = project_root / trial["base_config_path"]
    training_path = project_root / trial["training_config_path"]
    if file_sha256(base_path) != trial["base_config_sha256"]:
        raise FinalPublicError("Pinned E18 public config changed.")
    if file_sha256(training_path) != trial["training_config_sha256"]:
        raise FinalPublicError("Pinned E19 training config changed.")
    base = load_e18_public_config(project_root, base_path)
    training = load_e19_training_config(project_root, training_path)
    _selection_contract(trial["model_selection"])
    if trial["full_lora_source"] != {
        "experiment_id": training.raw["experiment_id"],
        "training_config_path": trial["training_config_path"],
        "training_config_sha256": trial["training_config_sha256"],
        "training_code_sha256": "9bb0f7bc25a2aa1553cd9a86b9257dc6aaf04574683ef7e4f94702a3b9974f8b",
        "training_identity_sha256": "9a9b5f9eb0854097af5147ba4a9d42d1be9a890075a86f7123cd1dc86c218a78",
        "adapter_relative_path": "training/adapter-final/adapter_model.safetensors",
        "adapter_sha256": trial["model_selection"]["candidate_adapter_sha256"],
        "training_records_sha256": "1d7c5b97e539cfe33ce036a66de4595c959d30b0a39b10e10c2fba415ce3d46b",
        "raw_source_contexts_sha256": training.section("source_e08a")["train_results_sha256"],
        "metadata_policy": training.contract["metadata_source"]["policy"],
        "train_sample_size": 5636,
        "fresh_from_base": True,
        "lora_rank": 8,
        "lora_alpha": 16,
    }:
        raise FinalPublicError("E19 public adapter source changed.")
    base_trial = json.loads(base_path.read_text(encoding="utf-8"))
    if base_trial.get("metadata_source") != training.contract["metadata_source"]:
        raise FinalPublicError("E18 public and E19 training metadata policies differ.")
    raw = copy.deepcopy(base.raw)
    raw["experiment_id"] = EXPERIMENT
    raw["model_selection"] = copy.deepcopy(trial["model_selection"])
    raw["full_lora_source"] = copy.deepcopy(trial["full_lora_source"])
    raw["metadata_source"] = copy.deepcopy(base_trial["metadata_source"])
    raw["generator"]["adapter_variant"] = "e19-metadata-aware-context-full-train-5636"
    raw["run_contract"].update({
        "operator_selected_e19_directional_public_trial": True,
        "e19_adapter_trained_with_exact_inference_metadata": True,
        "retrieval_and_metadata_reused_from_e18_public": True,
        "only_generator_adapter_changed_from_e18_public": True,
        "e18_public_05251_preserved_as_rollback": True,
    })
    config = FinalPublicConfig(raw=raw, path=path)
    inference = config.section("inference")
    if (
        inference.get("max_input_tokens") != 8192
        or inference.get("minimum_contexts") != 12
        or inference.get("max_new_tokens") != 704
        or inference.get("do_sample") is not False
        or inference.get("num_beams") != 1
        or inference.get("enable_thinking") is not False
        or inference.get("use_cache") is not True
    ):
        raise FinalPublicError("E19 public deterministic inference changed.")
    if config.section("execution") != {
        "generation_workers": 2,
        "generation_partition": "sample-index-mod-worker-count",
        "questions_per_worker": 500,
        "checkpoint_after_questions": 1,
    }:
        raise FinalPublicError("E19 public dual-GPU execution changed.")
    if not all(config.section("run_contract").values()):
        raise FinalPublicError("E19 public run contract lost an invariant.")
    return config


def code_sha(project_root: Path) -> str:
    paths = sorted((project_root / "src/uit_dsc_fixed_rag").rglob("*.py"))
    paths.append(project_root / "scripts/run_final_public_e19_kaggle.py")
    return _json_sha256({
        path.relative_to(project_root).as_posix(): file_sha256(path)
        for path in paths if path.is_file()
    })


def _validate_selection(selection_directory: Path, config: FinalPublicConfig) -> str:
    expected = config.section("model_selection")
    report_path = selection_directory / "report.json"
    control_path = selection_directory / "evaluation" / expected["control"] / "results.jsonl"
    candidate_path = selection_directory / "evaluation" / expected["winner"] / "results.jsonl"
    if not all(path.is_file() for path in (report_path, control_path, candidate_path)):
        raise FinalPublicError("Saved E19 paired selection output is incomplete.")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    metrics = report.get("metrics", {})
    control, candidate = metrics.get(expected["control"], {}), metrics.get(expected["winner"], {})
    paired, evidence = report.get("paired_delta_candidate_minus_control", {}), report.get("evidence", {})
    if (
        report.get("experiment_id") != expected["report_experiment_id"]
        or report.get("sample_scope") != expected["sample_scope"]
        or report.get("sample_size") != expected["sample_size"]
        or report.get("control_variant") != expected["control"]
        or report.get("candidate_variant") != expected["winner"]
        or report.get("smoke_leader") != expected["winner"]
        or report.get("promotion_allowed") is not False
        or report.get("public_read") is not False
        or report.get("holdout_untouched") is not True
        or control.get("meteor") != expected["control_meteor"]
        or control.get("rouge_l") != expected["control_rouge_l"]
        or candidate.get("meteor") != expected["candidate_meteor"]
        or candidate.get("rouge_l") != expected["candidate_rouge_l"]
        or paired.get("meteor_mean") != expected["meteor_delta"]
        or paired.get("meteor_bootstrap_95_ci") != expected["meteor_ci"]
        or paired.get("improved_questions") != expected["improved_questions"]
        or paired.get("worsened_questions") != expected["worsened_questions"]
        or paired.get("tied_questions") != expected["tied_questions"]
        or evidence.get("config_sha256") != expected["config_sha256"]
        or evidence.get("sample_ids_sha256") != expected["sample_ids_sha256"]
        or evidence.get("control_adapter_sha256") != expected["control_adapter_sha256"]
        or evidence.get("candidate_adapter_sha256") != expected["candidate_adapter_sha256"]
        or evidence.get("control_results_sha256") != expected["control_results_sha256"]
        or evidence.get("candidate_results_sha256") != expected["candidate_results_sha256"]
        or file_sha256(control_path) != expected["control_results_sha256"]
        or file_sha256(candidate_path) != expected["candidate_results_sha256"]
    ):
        raise FinalPublicError("Saved E19 paired selection evidence changed.")
    return file_sha256(candidate_path)


def _validate_adapter(
    *, project_root: Path, e19_directory: Path, config: FinalPublicConfig
) -> tuple[Any, str]:
    source = config.section("full_lora_source")
    training_config = load_e19_training_config(
        project_root, project_root / source["training_config_path"]
    )
    adapter_hash, complete = validate_candidate_adapter(
        e19_directory / "training", training_config
    )
    if (
        adapter_hash != source["adapter_sha256"]
        or complete.get("code_sha256") != source["training_code_sha256"]
        or complete.get("identity_sha256") != source["training_identity_sha256"]
        or complete.get("training_records_sha256") != source["training_records_sha256"]
        or complete.get("raw_source_contexts_sha256") != source["raw_source_contexts_sha256"]
        or complete.get("metadata_policy") != source["metadata_policy"]
        or complete.get("trainable_parameters") != 8_409_600
    ):
        raise FinalPublicError("Saved E19 metadata-trained adapter evidence changed.")
    return training_config, adapter_hash


def _validate_e00(e00_directory: Path, config: FinalPublicConfig) -> None:
    metadata = config.section("metadata_source")
    for filename, key in (
        ("manifest.json", "manifest_sha256"),
        ("chunks.jsonl", "chunks_sha256"),
        ("documents.jsonl", "documents_sha256"),
    ):
        path = e00_directory / filename
        if not path.is_file() or file_sha256(path) != metadata[key]:
            raise FinalPublicError(f"Pinned E00-v2 metadata source changed: {filename}")


def validate_preflight(
    *, project_root: Path, retrieval_directory: Path, e19_directory: Path,
    selection_directory: Path, e00_directory: Path, public_path: Path,
    config: FinalPublicConfig,
) -> dict[str, Any]:
    _, ids = load_public_questions(public_path, config)
    scorer_cfg = config.section("submission_contract")
    scorer_bytes, scorer_container = _load_official_scorer_bytes(project_root, scorer_cfg)
    if hashlib.sha256(scorer_bytes).hexdigest() != scorer_cfg["official_scorer_entry_sha256"]:
        raise FinalPublicError("Official scorer mapping changed.")
    _, adapter_hash = _validate_adapter(
        project_root=project_root, e19_directory=e19_directory, config=config
    )
    selection_hash = _validate_selection(selection_directory, config)
    source_cfg = config.section("retrieval_reuse")
    source_path = retrieval_directory / source_cfg["path"]
    report_path = retrieval_directory / "report.json"
    if not source_path.is_file() or not report_path.is_file():
        raise FinalPublicError("Saved public retrieval artifact is incomplete.")
    raw_hash = _validate_retrieval(
        report=json.loads(report_path.read_text(encoding="utf-8")),
        path=source_path, ids=ids, config=config,
    )
    _validate_e00(e00_directory, config)
    return {
        "code_version": CODE_VERSION,
        "code_sha256": code_sha(project_root),
        "config_sha256": config.config_sha256,
        "public_sha256": config.section("public")["sha256"],
        "sample_ids_sha256": config.section("public")["sample_ids_sha256"],
        "sample_size": len(ids),
        "official_scorer_container": scorer_container,
        "official_scorer_entry_sha256": scorer_cfg["official_scorer_entry_sha256"],
        "adapter_sha256": adapter_hash,
        "model_selection_results_sha256": selection_hash,
        "raw_retrieval_results_sha256": raw_hash,
        "metadata_source": config.section("metadata_source"),
        "max_new_tokens": 704,
        "minimum_contexts": 12,
    }


def prepare_metadata_retrieval(
    *, retrieval_directory: Path, e00_directory: Path, public_path: Path,
    output_directory: Path, config: FinalPublicConfig,
) -> dict[str, Any]:
    _, ids = load_public_questions(public_path, config)
    source_cfg = config.section("retrieval_reuse")
    source_path = retrieval_directory / source_cfg["path"]
    source_report = json.loads((retrieval_directory / "report.json").read_text(encoding="utf-8"))
    raw_hash = _validate_retrieval(
        report=source_report, path=source_path, ids=ids, config=config
    )
    rows = _read_jsonl(source_path)
    wanted_chunks = {context["chunk_id"] for row in rows for context in row["contexts"]}
    metadata = config.section("metadata_source")
    chunks = scan_selected(
        e00_directory / "chunks.jsonl", "chunk_id", wanted_chunks,
        metadata["chunks_sha256"],
    )
    documents = scan_selected(
        e00_directory / "documents.jsonl", "document_id",
        {chunk["document_id"] for chunk in chunks.values()},
        metadata["documents_sha256"],
    )
    enriched_rows, coverage = [], []
    for index, (question_id, row) in enumerate(zip(ids, rows)):
        if row.get("sample_index") != index or row.get("question_id") != question_id:
            raise FinalPublicError(f"Public retrieval identity changed: {index}")
        if "answer" in row or len(row.get("contexts", [])) != 12:
            raise FinalPublicError(f"Invalid public retrieval contexts: {index}")
        contexts = []
        for context in row["contexts"]:
            enriched, evidence = enrich_context(
                context, chunks[context["chunk_id"]], documents[context["document_id"]]
            )
            contexts.append(enriched)
            coverage.append(evidence)
        if [c["chunk_id"] for c in contexts] != [c["chunk_id"] for c in row["contexts"]]:
            raise FinalPublicError("Metadata preparation changed retrieval order.")
        enriched_rows.append({**row, "contexts": contexts})
    destination = output_directory / "retrieval/results.jsonl"
    destination.parent.mkdir(parents=True, exist_ok=True)
    _atomic_jsonl(destination, enriched_rows)
    summary = {
        "experiment_id": EXPERIMENT,
        "raw_retrieval_results_sha256": raw_hash,
        "metadata_retrieval_results_sha256": file_sha256(destination),
        "sample_size": len(ids),
        "context_instances": len(coverage),
        "contexts_per_question": 12,
        "with_source_title": sum(bool(item["source_title"]) for item in coverage),
        "with_document_number": sum(bool(item["document_number"]) for item in coverage),
        "with_article_title": sum(bool(item["article_title"]) for item in coverage),
        "unchanged_contexts": sum(not item["prefix"] for item in coverage),
        "same_context_bodies_and_order": True,
        "answers_used": False,
        "metadata_policy": metadata["policy"],
    }
    _atomic_json(output_directory / "metadata-retrieval.json", summary)
    return summary


def load_generator(
    *, project_root: Path, e19_directory: Path,
    config: FinalPublicConfig, device: str,
) -> tuple[Any, Any, dict[str, Any], int]:
    training_config, _ = _validate_adapter(
        project_root=project_root, e19_directory=e19_directory, config=config
    )
    return load_candidate_generator(
        config=training_config,
        training_directory=e19_directory / "training",
        device=device,
    )


def run_generation_worker(**kwargs: Any) -> dict[str, Any]:
    return _run_generation_worker(**kwargs)


def finalize(
    *, output_directory: Path, public_path: Path, config: FinalPublicConfig
) -> dict[str, Any]:
    _, ids = load_public_questions(public_path, config)
    retrieval = _read_jsonl(output_directory / "retrieval/results.jsonl")
    records = output_directory / "generation/records"
    for index, question_id in enumerate(ids):
        record_path = records / f"{index:04d}.json"
        if not record_path.is_file():
            raise FinalPublicError(f"Missing generation record: {index}")
        record = json.loads(record_path.read_text(encoding="utf-8"))
        expected_chunks = [context["chunk_id"] for context in retrieval[index]["contexts"]]
        if (
            record.get("question_id") != question_id
            or record.get("selected_context_count") != 12
            or record.get("selected_chunk_ids") != expected_chunks
        ):
            raise FinalPublicError(f"Public E19 did not use all twelve contexts: {index}")
    metadata_summary = json.loads(
        (output_directory / "metadata-retrieval.json").read_text(encoding="utf-8")
    )
    if metadata_summary.get("metadata_retrieval_results_sha256") != file_sha256(
        output_directory / "retrieval/results.jsonl"
    ):
        raise FinalPublicError("Prepared metadata retrieval changed after generation.")
    report = finalize_submission(
        output_directory=output_directory, public_path=public_path, config=config
    )
    selection = config.section("model_selection")
    report["selected_stack"].update({
        "contexts": "ranked-top12-with-e00-source-metadata",
        "generator": "e19_metadata_trained_rank8_max704",
        "max_new_tokens": 704,
    })
    report["model_selection"] = {
        "primary_metric": "meteor",
        "fresh_dev200_control_meteor": selection["control_meteor"],
        "fresh_dev200_candidate_meteor": selection["candidate_meteor"],
        "paired_meteor_delta": selection["meteor_delta"],
        "paired_meteor_ci": selection["meteor_ci"],
        "operator_selected_public_trial": True,
        "formal_promotion": False,
        "rollback_public_meteor": 0.5251,
    }
    report["metadata_retrieval"] = metadata_summary
    report["evidence"]["raw_retrieval_results_sha256"] = metadata_summary[
        "raw_retrieval_results_sha256"
    ]
    report["evidence"]["metadata_retrieval_results_sha256"] = metadata_summary[
        "metadata_retrieval_results_sha256"
    ]
    _atomic_json(output_directory / "report.json", report)
    return report


__all__ = [
    "EXPERIMENT", "FinalPublicError", "code_sha", "finalize", "load_config",
    "load_generator", "prepare_metadata_retrieval", "run_generation_worker",
    "validate_preflight",
]
