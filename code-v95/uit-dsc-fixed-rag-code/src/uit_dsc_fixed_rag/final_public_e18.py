"""Public-1000 trial of E18 source metadata with the preserved E08B stack."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

from .corpus import file_sha256
from .e02_answer import _atomic_json, _atomic_jsonl
from .e03_rrf_grid import _read_jsonl
from .e18_source_metadata import enrich_context, scan_selected
from .final_public import (
    FinalPublicConfig,
    FinalPublicError,
    _load_official_scorer_bytes,
    finalize_submission,
    load_public_questions,
    run_generation_worker,
)
from .final_public_e08b import (
    _validate_adapter,
    _validate_retrieval,
    load_config as load_base_config,
    load_generator,
)


EXPERIMENT = "FINAL-public1000-e08b-source-metadata-max704-v1"
CODE_VERSION = "0.50.0"


def _load_trial(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if set(payload) != {
        "schema_version", "experiment_id", "base_config_path",
        "base_config_sha256", "model_selection", "metadata_source",
        "public_trial",
    }:
        raise FinalPublicError("E18 public config root changed.")
    if payload.get("schema_version") != "1.0" or payload.get("experiment_id") != EXPERIMENT:
        raise FinalPublicError("E18 public experiment identity changed.")
    trial = payload.get("public_trial", {})
    required_trial = {
        "context_count": 12,
        "max_input_tokens": 8192,
        "max_new_tokens": 704,
        "overflow_policy": "fail-before-generation-never-drop-contexts",
        "generation_workers": 2,
        "questions_per_worker": 500,
        "public_answers_never_read": True,
        "retrieval_scores_and_order_unchanged": True,
        "context_bodies_unchanged": True,
        "metadata_only_change": True,
    }
    if trial != required_trial:
        raise FinalPublicError("E18 public trial contract changed.")
    return payload


def load_config(project_root: Path, path: Path) -> FinalPublicConfig:
    trial = _load_trial(path)
    base_path = project_root / trial["base_config_path"]
    if file_sha256(base_path) != trial["base_config_sha256"]:
        raise FinalPublicError("Pinned E08B public base config changed.")
    base = load_base_config(base_path)
    raw = copy.deepcopy(base.raw)
    raw["experiment_id"] = EXPERIMENT
    raw["model_selection"] = copy.deepcopy(trial["model_selection"])
    raw["inference"]["minimum_contexts"] = 12
    raw["run_contract"].update({
        "operator_selected_e18_directional_public_trial": True,
        "source_metadata_from_pinned_e00_only": True,
        "retrieval_scores_and_order_unchanged": True,
        "context_bodies_unchanged": True,
        "all_twelve_contexts_required": True,
    })
    config = FinalPublicConfig(raw=raw, path=path)
    selection = config.section("model_selection")
    if (
        selection.get("report_experiment_id")
        != "E18-e08b-source-metadata-fresh-dev200-max704-v1"
        or selection.get("winner") != "source_metadata_top12"
        or selection.get("control") != "ranked_top12_control"
        or selection.get("sample_size") != 200
        or selection.get("candidate_meteor") != 0.5533650437609198
        or selection.get("meteor_delta") != 0.012567387456554715
        or selection.get("meteor_ci")
        != [-0.005081754550967019, 0.030618916988666846]
        or selection.get("promotion_allowed") is not False
        or selection.get("operator_selected_for_public_trial") is not True
    ):
        raise FinalPublicError("E18 public selection evidence changed.")
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
        raise FinalPublicError("E18 public deterministic inference changed.")
    if not all(config.section("run_contract").values()):
        raise FinalPublicError("E18 public run contract lost an invariant.")
    return config


def _validate_selection(selection_directory: Path, config: FinalPublicConfig) -> str:
    expected = config.section("model_selection")
    report_path = selection_directory / "report.json"
    results_path = selection_directory / "results.jsonl"
    if not report_path.is_file() or not results_path.is_file():
        raise FinalPublicError("Saved E18 selection output is incomplete.")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    metrics = report.get("metrics", {})
    control = metrics.get(expected["control"], {})
    candidate = metrics.get(expected["winner"], {})
    paired = report.get("paired_delta_candidate_minus_control", {})
    evidence = report.get("evidence", {})
    if (
        report.get("experiment_id") != expected["report_experiment_id"]
        or report.get("sample_size") != expected["sample_size"]
        or report.get("control_variant") != expected["control"]
        or report.get("candidate_variant") != expected["winner"]
        or report.get("smoke_leader") != expected["winner"]
        or report.get("promotion_allowed") is not expected["promotion_allowed"]
        or report.get("public_read") is not False
        or candidate.get("meteor") != expected["candidate_meteor"]
        or candidate.get("rouge_l") != expected["candidate_rouge_l"]
        or control.get("meteor") != expected["control_meteor"]
        or control.get("rouge_l") != expected["control_rouge_l"]
        or paired.get("meteor_mean") != expected["meteor_delta"]
        or paired.get("meteor_bootstrap_95_ci") != expected["meteor_ci"]
        or paired.get("improved_questions") != expected["improved_questions"]
        or paired.get("worsened_questions") != expected["worsened_questions"]
        or paired.get("tied_questions") != expected["tied_questions"]
        or evidence.get("config_sha256") != expected["config_sha256"]
        or evidence.get("sample_ids_sha256") != expected["sample_ids_sha256"]
        or evidence.get("adapter_sha256") != expected["adapter_sha256"]
        or evidence.get("results_sha256") != expected["generation_results_sha256"]
    ):
        raise FinalPublicError("Saved E18 selection evidence changed.")
    observed = file_sha256(results_path)
    if observed != expected["generation_results_sha256"]:
        raise FinalPublicError("Saved E18 results bytes changed.")
    return observed


def _validate_e00(e00_directory: Path, trial: dict[str, Any]) -> None:
    expected = trial["metadata_source"]
    for filename, key in (
        ("manifest.json", "manifest_sha256"),
        ("chunks.jsonl", "chunks_sha256"),
        ("documents.jsonl", "documents_sha256"),
    ):
        path = e00_directory / filename
        if not path.is_file() or file_sha256(path) != expected[key]:
            raise FinalPublicError(f"Pinned E00-v2 metadata source changed: {filename}")


def validate_preflight(
    *, project_root: Path, retrieval_directory: Path, e08b_directory: Path,
    selection_directory: Path, e00_directory: Path, public_path: Path,
    config: FinalPublicConfig,
) -> dict[str, Any]:
    _, ids = load_public_questions(public_path, config)
    scorer_cfg = config.section("submission_contract")
    scorer_bytes, scorer_container = _load_official_scorer_bytes(project_root, scorer_cfg)
    if hashlib.sha256(scorer_bytes).hexdigest() != scorer_cfg["official_scorer_entry_sha256"]:
        raise FinalPublicError("Official scorer mapping changed.")
    _, adapter_hash = _validate_adapter(
        project_root=project_root, e08b_directory=e08b_directory, config=config
    )
    selection_hash = _validate_selection(selection_directory, config)
    source_cfg = config.section("retrieval_reuse")
    source_path = retrieval_directory / source_cfg["path"]
    report_path = retrieval_directory / "report.json"
    if not source_path.is_file() or not report_path.is_file():
        raise FinalPublicError("Saved public retrieval artifact is incomplete.")
    raw_hash = _validate_retrieval(
        report=json.loads(report_path.read_text(encoding="utf-8")),
        path=source_path,
        ids=ids,
        config=config,
    )
    trial = _load_trial(config.path)
    _validate_e00(e00_directory, trial)
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
        "raw_retrieval_results_sha256": raw_hash,
        "metadata_source": trial["metadata_source"],
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
    wanted_chunks = {c["chunk_id"] for row in rows for c in row["contexts"]}
    metadata = _load_trial(config.path)["metadata_source"]
    chunks = scan_selected(
        e00_directory / "chunks.jsonl", "chunk_id", wanted_chunks,
        metadata["chunks_sha256"],
    )
    wanted_documents = {chunk["document_id"] for chunk in chunks.values()}
    documents = scan_selected(
        e00_directory / "documents.jsonl", "document_id", wanted_documents,
        metadata["documents_sha256"],
    )
    enriched_rows: list[dict[str, Any]] = []
    coverage: list[dict[str, Any]] = []
    for index, (question_id, row) in enumerate(zip(ids, rows)):
        if row.get("sample_index") != index or row.get("question_id") != question_id:
            raise FinalPublicError(f"Public retrieval identity changed: {index}")
        if "answer" in row or len(row.get("contexts", [])) != 12:
            raise FinalPublicError(f"Invalid public retrieval contexts: {index}")
        enriched_contexts = []
        for context in row["contexts"]:
            enriched, evidence = enrich_context(
                context, chunks[context["chunk_id"]], documents[context["document_id"]]
            )
            enriched_contexts.append(enriched)
            coverage.append(evidence)
        if [c["chunk_id"] for c in enriched_contexts] != [c["chunk_id"] for c in row["contexts"]]:
            raise FinalPublicError("Metadata preparation changed retrieval order.")
        enriched_rows.append({**row, "contexts": enriched_contexts})
    destination = output_directory / "retrieval" / "results.jsonl"
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


def finalize(
    *, output_directory: Path, public_path: Path, config: FinalPublicConfig
) -> dict[str, Any]:
    _, ids = load_public_questions(public_path, config)
    retrieval = _read_jsonl(output_directory / "retrieval" / "results.jsonl")
    records = output_directory / "generation" / "records"
    for index, question_id in enumerate(ids):
        record_path = records / f"{index:04d}.json"
        if not record_path.is_file():
            raise FinalPublicError(f"Missing generation record: {index}")
        record = json.loads(record_path.read_text(encoding="utf-8"))
        expected_chunks = [c["chunk_id"] for c in retrieval[index]["contexts"]]
        if (
            record.get("question_id") != question_id
            or record.get("selected_context_count") != 12
            or record.get("selected_chunk_ids") != expected_chunks
        ):
            raise FinalPublicError(f"Public E18 did not use all twelve contexts: {index}")
    metadata_summary = json.loads(
        (output_directory / "metadata-retrieval.json").read_text(encoding="utf-8")
    )
    if metadata_summary.get("metadata_retrieval_results_sha256") != file_sha256(
        output_directory / "retrieval" / "results.jsonl"
    ):
        raise FinalPublicError("Prepared metadata retrieval changed after generation.")
    report = finalize_submission(
        output_directory=output_directory, public_path=public_path, config=config
    )
    selection = config.section("model_selection")
    report["selected_stack"].update({
        "contexts": "ranked-top12-with-e00-source-metadata",
        "generator": "e08b_context_lora_max704",
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
    "FinalPublicError", "finalize", "load_config", "load_generator",
    "prepare_metadata_retrieval", "run_generation_worker", "validate_preflight",
]
