"""Operator-selected public1000 trial of E21 parent-expanded context assembly."""
from __future__ import annotations

import copy
import hashlib
import importlib.metadata
import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean
from typing import Any

from . import e21_parent_context as parent
from .corpus import file_sha256
from .e02_answer import _atomic_json, _atomic_jsonl
from .e03_rrf_grid import _json_sha256, _read_jsonl, _load_worker_progress, _write_worker_state
from .e10_repetition_grid import answer_diagnostics
from .e18_source_metadata import enrich_context, save_once, scan_selected
from .final_public import FinalPublicConfig, FinalPublicError, _load_official_scorer_bytes, finalize_submission, load_public_questions
from .final_public_e08b import _validate_retrieval
from .final_public_e19 import _validate_adapter, _validate_e00, load_config as load_base_config, load_generator

EXPERIMENT = "FINAL-public1000-e23-parent-expanded-max704-v1"
VARIANT = "e23_parent_expanded_max704"
LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class Config:
    raw: dict[str, Any]
    path: Path
    base: FinalPublicConfig
    context: Any

    @property
    def sha(self) -> str:
        return file_sha256(self.path)

    @property
    def public(self) -> FinalPublicConfig:
        raw = copy.deepcopy(self.base.raw)
        raw["experiment_id"] = EXPERIMENT
        raw["model_selection"] = {
            "winner": self.raw["selection"]["candidate_variant"],
            "candidate_results_sha256": self.raw["selection"]["results_sha256"],
        }
        raw["run_contract"].update({
            "operator_selected_e21_parent_expansion_public_trial": True,
            "same_e19_adapter_and_max704": True,
            "saved_public_top12_reused_without_search": True,
            "parent_text_from_exact_e00_spans": True,
        })
        return FinalPublicConfig(raw=raw, path=self.path)


def load_config(root: Path, path: Path) -> Config:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if set(raw) != {"schema_version", "experiment_id", "base_config_path", "base_config_sha256",
                    "selection", "context_policy", "public_trial"}:
        raise FinalPublicError("E23 config root changed.")
    if raw.get("schema_version") != "1.0" or raw.get("experiment_id") != EXPERIMENT:
        raise FinalPublicError("E23 identity changed.")
    base_path = root / raw["base_config_path"]
    if raw["base_config_path"] != "configs/final-public-e19-metadata-trained-max704-v1.json" or file_sha256(base_path) != raw["base_config_sha256"]:
        raise FinalPublicError("Pinned E19 public contract changed.")
    base = load_base_config(root, base_path)
    policy = raw["context_policy"]
    context_path = root / policy["source_config_path"]
    context = parent.load_config(root, context_path)
    if policy != {
        "source_config_path": "configs/e21-parent-context-dev200-v1.json",
        "source_config_sha256": context.sha, "variant": parent.VARIANTS[1],
        "seed_contexts": 12, "merge": context.policy["merge"], "metadata": context.policy["metadata"],
        "max_parent_tokens": 1200, "max_parent_characters": 6000,
        "max_input_tokens": 8192, "max_new_tokens": 704, "evict_seed_for_expansion": False,
    }:
        raise FinalPublicError("E23 must reproduce the E21 GPU1 context policy.")
    expected_selection = {
        "experiment_id": parent.EXPERIMENT, "candidate_variant": parent.VARIANTS[1],
        "sample_size": 200, "sample_scope": "already-used-e19-dev400-600-final121-not-scored",
        "config_sha256": context.sha, "code_sha256": "e5574861eb34ae3fdee1220d5260109ddd9f84feacea9c1ec46a78df8b5527c8",
        "results_sha256": "cbe3d5e163e1983f031a06c767596fcebd91177e3b90625998a1deed28f6afe4",
        "control_meteor": 0.5233168277186363, "candidate_meteor": 0.5633893466813007,
        "candidate_rouge_l": 0.5864053775086489, "meteor_delta": 0.04007251896266439,
        "meteor_ci": [0.012369872150859176, 0.06805563198334254],
        "improved": 110, "worsened": 76, "tied": 14,
        "promotion_allowed": False, "operator_selected_for_public_trial": True,
    }
    if raw["selection"] != expected_selection:
        raise FinalPublicError("E23 selection evidence contract changed.")
    if raw["public_trial"] != {
        "sample_size": 1000, "generation_workers": 2, "questions_per_worker": 500,
        "partition": "sample-index-mod-worker-count", "public_answers_never_read": True,
        "saved_public_top12_reused": True, "same_e19_adapter": True,
        "rollback_public_meteor": 0.5309, "rollback_public_rouge_l": 0.5632,
    }:
        raise FinalPublicError("E23 public execution contract changed.")
    return Config(raw, path, base, context)


def code_sha(root: Path) -> str:
    paths = sorted((root / "src/uit_dsc_fixed_rag").rglob("*.py"))
    paths.append(root / "scripts/run_final_public_e23_kaggle.py")
    return _json_sha256({p.relative_to(root).as_posix(): file_sha256(p) for p in paths})


def validate_selection(directory: Path, config: Config) -> dict[str, Any]:
    expected = config.raw["selection"]
    report_path = directory / "report.json"
    results_path = directory / "evaluation" / expected["candidate_variant"] / "results.jsonl"
    state_path = results_path.parent / "state.json"
    if not all(p.is_file() for p in (report_path, results_path, state_path)):
        raise FinalPublicError("Add the FULL saved E21 output.")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    metric = report.get("metrics", {}).get(expected["candidate_variant"], {})
    paired = report.get("paired_expanded_minus_e19", {})
    evidence = report.get("evidence", {})
    if (report.get("experiment_id") != expected["experiment_id"] or report.get("sample_size") != 200
            or report.get("sample_scope") != expected["sample_scope"] or report.get("smoke_leader") != expected["candidate_variant"]
            or report.get("promotion_allowed") is not False or report.get("public_read") is not False
            or report.get("holdout_untouched") is not True
            or report["metrics"][parent.CONTROL]["meteor"] != expected["control_meteor"]
            or metric.get("meteor") != expected["candidate_meteor"] or metric.get("rouge_l") != expected["candidate_rouge_l"]
            or paired.get("meteor_mean") != expected["meteor_delta"] or paired.get("meteor_bootstrap_95_ci") != expected["meteor_ci"]
            or paired.get("improved") != expected["improved"] or paired.get("worsened") != expected["worsened"] or paired.get("tied") != expected["tied"]
            or evidence.get("config_sha256") != expected["config_sha256"] or evidence.get("code_sha256") != expected["code_sha256"]
            or evidence.get("candidate_results_sha256", {}).get(expected["candidate_variant"]) != expected["results_sha256"]
            or file_sha256(results_path) != expected["results_sha256"]):
        raise FinalPublicError("Saved E21 selection evidence changed.")
    state = json.loads(state_path.read_text(encoding="utf-8"))
    identity = state.get("run_identity", {})
    rows = _read_jsonl(results_path)
    if (state.get("complete") is not True or state.get("completed_count") != 200 or state.get("assigned_count") != 200
            or identity.get("variant") != expected["candidate_variant"] or identity.get("worker_rank") != 1
            or identity.get("device") != "cuda:1" or identity.get("config_sha256") != expected["config_sha256"]
            or identity.get("code_sha256") != expected["code_sha256"] or len(rows) != 200
            or identity.get("identity_sha256") != _json_sha256({k: v for k, v in identity.items() if k != "identity_sha256"})):
        raise FinalPublicError("Saved E21 worker state changed.")
    for i, row in enumerate(rows):
        parent.validate_record(row, row.get("question_id"), i, 1, identity)
    return {"results_sha256": file_sha256(results_path), "runtime": identity["runtime"],
            "generation_config_sha256": identity["generation_config_sha256"], "identity_sha256": identity["identity_sha256"]}


def validate_preflight(*, root: Path, retrieval: Path, e00: Path, e19: Path,
                       selection: Path, public: Path, config: Config) -> dict[str, Any]:
    public_config = config.public
    _, ids = load_public_questions(public, public_config)
    scorer_cfg = public_config.section("submission_contract")
    scorer_bytes, scorer_container = _load_official_scorer_bytes(root, scorer_cfg)
    if hashlib.sha256(scorer_bytes).hexdigest() != scorer_cfg["official_scorer_entry_sha256"]:
        raise FinalPublicError("Official scorer mapping changed.")
    _, adapter_sha = _validate_adapter(project_root=root, e19_directory=e19, config=public_config)
    _validate_e00(e00, public_config)
    source_cfg = public_config.section("retrieval_reuse")
    raw_path = retrieval / source_cfg["path"]
    raw_sha = _validate_retrieval(report=json.loads((retrieval / "report.json").read_text(encoding="utf-8")),
                                  path=raw_path, ids=ids, config=public_config)
    selected = validate_selection(selection, config)
    budget = public_config.section("parameter_budget")
    total = budget["embedding"] + budget["generator"] + budget["adapter_parameter_cap"]
    if total != budget["maximum_stack_total"] or total >= budget["exclusive_limit"]:
        raise FinalPublicError("Parameter budget changed.")
    return {"experiment_id": EXPERIMENT, "code_sha256": code_sha(root), "config_sha256": config.sha,
            "public_sha256": public_config.section("public")["sha256"], "sample_ids_sha256": public_config.section("public")["sample_ids_sha256"],
            "sample_size": len(ids), "raw_retrieval_results_sha256": raw_sha, "adapter_sha256": adapter_sha,
            "selection": selected, "official_scorer_container": scorer_container,
            "official_scorer_entry_sha256": scorer_cfg["official_scorer_entry_sha256"],
            "maximum_stack_parameters": total, "max_new_tokens": 704, "workers": [500, 500]}


def check_preflight(root: Path, output: Path, config: Config) -> dict[str, Any]:
    payload = json.loads((output / "preflight.json").read_text(encoding="utf-8"))
    evidence = payload.get("evidence", {})
    if (payload.get("experiment_id") != EXPERIMENT or evidence.get("config_sha256") != config.sha
            or evidence.get("code_sha256") != code_sha(root) or evidence.get("sample_size") != 1000
            or evidence.get("max_new_tokens") != 704):
        raise FinalPublicError("E23 preflight identity changed.")
    return evidence


def prepare(*, retrieval: Path, e00: Path, public: Path, output: Path, config: Config) -> dict[str, Any]:
    public_config = config.public
    _, ids = load_public_questions(public, public_config)
    source_cfg = public_config.section("retrieval_reuse")
    raw_path = retrieval / source_cfg["path"]
    raw_sha = _validate_retrieval(report=json.loads((retrieval / "report.json").read_text(encoding="utf-8")),
                                  path=raw_path, ids=ids, config=public_config)
    rows = _read_jsonl(raw_path)
    metadata = public_config.section("metadata_source")
    wanted = {c["chunk_id"] for row in rows for c in row["contexts"]}
    seeds = scan_selected(e00 / "chunks.jsonl", "chunk_id", wanted, metadata["chunks_sha256"])
    doc_ids = {c["document_id"] for c in seeds.values()}
    documents = scan_selected(e00 / "documents.jsonl", "document_id", doc_ids, metadata["documents_sha256"])
    by_doc, digest = {d: [] for d in doc_ids}, hashlib.sha256()
    with (e00 / "chunks.jsonl").open("rb") as stream:
        for line in stream:
            digest.update(line)
            chunk = json.loads(line)
            if chunk["document_id"] in by_doc:
                by_doc[chunk["document_id"]].append(chunk)
    if digest.hexdigest() != metadata["chunks_sha256"]:
        raise FinalPublicError("E00 chunks changed during parent scan.")
    block_by_chunk = {}
    for document_id, chunks in by_doc.items():
        for block in parent.article_blocks(chunks, documents[document_id]):
            for chunk in block:
                if chunk["chunk_id"] in wanted:
                    block_by_chunk[chunk["chunk_id"]] = block
    prepared = []
    for i, (qid, row) in enumerate(zip(ids, rows)):
        if row.get("sample_index") != i or row.get("question_id") != qid or "answer" in row or len(row.get("contexts", [])) != 12:
            raise FinalPublicError(f"Public top12 changed: {i}")
        units = []
        for rank, context in enumerate(row["contexts"]):
            seed = seeds[context["chunk_id"]]
            enrich_context(context, seed, documents[seed["document_id"]])
            units.append(parent.seed_unit(seed, rank, block_by_chunk[seed["chunk_id"]], documents[seed["document_id"]], config.context.policy))
        prepared.append({"question_id": qid, "sample_index": i, "answers_used": False, "units": units})
    destination = output / "retrieval/results.jsonl"
    save_once(destination, prepared, jsonl=True)
    summary = {"experiment_id": EXPERIMENT, "sample_size": 1000, "seed_contexts_per_question": 12,
               "answers_used": False, "raw_retrieval_results_sha256": raw_sha,
               "prepared_results_sha256": file_sha256(destination),
               "expandable_questions": sum(any(u["expansions"] for u in row["units"]) for row in prepared),
               "source_document_count": len(documents), "context_policy": config.raw["context_policy"]}
    save_once(output / "prepared-summary.json", summary)
    return summary


def load_prepared(output: Path, ids: list[str], config: Config) -> list[dict[str, Any]]:
    path = output / "retrieval/results.jsonl"
    summary = json.loads((output / "prepared-summary.json").read_text(encoding="utf-8"))
    rows = _read_jsonl(path)
    if (summary.get("experiment_id") != EXPERIMENT or summary.get("sample_size") != 1000
            or summary.get("answers_used") is not False or summary.get("prepared_results_sha256") != file_sha256(path)
            or summary.get("context_policy") != config.raw["context_policy"]
            or [r.get("question_id") for r in rows] != ids
            or any(r.get("sample_index") != i or r.get("answers_used") is not False
                   or [u.get("rank") for u in r.get("units", [])] != list(range(12)) for i, r in enumerate(rows))):
        raise FinalPublicError("Prepared public parent contexts changed.")
    return rows


def assigned(rank: int) -> list[int]:
    if rank not in (0, 1):
        raise FinalPublicError("Worker rank must be 0 or 1.")
    return list(range(rank, 1000, 2))


def validate_record(row: dict[str, Any], qid: str, index: int, rank: int, identity: dict[str, Any]) -> None:
    if (row.get("question_id") != qid or row.get("sample_index") != index or row.get("worker_rank") != rank
            or row.get("variant") != VARIANT or row.get("worker_identity_sha256") != identity["identity_sha256"]
            or not isinstance(row.get("answer"), str) or not row["answer"].strip()
            or row.get("record_sha256") != _json_sha256({k: v for k, v in row.items() if k != "record_sha256"})):
        raise FinalPublicError(f"Invalid E23 record: {rank}/{index}")


def run_worker(*, root: Path, output: Path, e19: Path, public: Path,
               config: Config, rank: int, device: str) -> dict[str, Any]:
    import torch
    indices = assigned(rank)
    if device != f"cuda:{rank}":
        raise FinalPublicError("Worker/GPU mismatch.")
    checked = check_preflight(root, output, config)
    questions, ids = load_public_questions(public, config.public)
    prepared = load_prepared(output, ids, config)
    expected_runtime = checked["selection"]["runtime"]
    runtime = {key: importlib.metadata.version(key) for key in expected_runtime}
    if runtime != expected_runtime:
        raise FinalPublicError("Use the E21 runtime versions; do not mix environments.")
    model, tokenizer, placement, parameters = load_generator(project_root=root, e19_directory=e19, config=config.public, device=device)
    generation_sha = _json_sha256(model.generation_config.to_dict())
    if generation_sha != checked["selection"]["generation_config_sha256"]:
        raise FinalPublicError("Generation defaults differ from E21.")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    def rendered(messages):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    def text_count(text):
        return len(tokenizer(text, add_special_tokens=False)["input_ids"])
    identity = {"code_sha256": code_sha(root), "config_sha256": config.sha,
                "prepared_sha256": file_sha256(output / "retrieval/results.jsonl"),
                "public_ids_sha256": config.public.section("public")["sample_ids_sha256"],
                "adapter_sha256": checked["adapter_sha256"], "adapter_parameters": parameters,
                "runtime": runtime, "generation_config_sha256": generation_sha,
                "worker_rank": rank, "device": device, "device_map": placement,
                "assigned_indices_sha256": hashlib.sha256(",".join(map(str, indices)).encode("ascii")).hexdigest(),
                "variant": VARIANT}
    identity["identity_sha256"] = _json_sha256(identity)
    records = output / "generation/records"
    state = output / f"generation/worker-{rank}-state.json"
    records.mkdir(parents=True, exist_ok=True)
    done = _load_worker_progress(records=records, state_path=state, identity=identity, assigned_indices=indices, sample_ids=ids)
    for i in indices[:done]:
        validate_record(json.loads((records / f"{i:04d}.json").read_text(encoding="utf-8")), ids[i], i, rank, identity)
    for completed, i in enumerate(indices[done:], start=done + 1):
        messages, spans, diagnostics = parent.pack(questions[ids[i]], prepared[i], config.context,
                                                    lambda m: text_count(rendered(m)), text_count, parent.VARIANTS[1])
        prompt = rendered(messages)
        input_tokens = text_count(prompt)
        if input_tokens > 8192 or diagnostics["seed_ranks"] != list(range(12)) or diagnostics["skipped_seed_ranks"]:
            raise FinalPublicError("E23 packing violated mandatory seed/input contract.")
        tensors = {k: v.to(device) for k, v in tokenizer(prompt, add_special_tokens=False, return_tensors="pt").items()}
        started = time.perf_counter()
        with torch.inference_mode():
            generated = model.generate(**tensors, do_sample=False, num_beams=1, max_new_tokens=704, use_cache=True)
        latency = (time.perf_counter() - started) * 1000
        new_ids = generated[0, tensors["input_ids"].shape[1]:]
        answer = tokenizer.decode(new_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False).strip()
        if not answer:
            raise FinalPublicError(f"Empty answer: {ids[i]}")
        eos = tokenizer.eos_token_id
        eos_ids = set(eos if isinstance(eos, list) else [eos])
        finish = "eos" if len(new_ids) and int(new_ids[-1]) in eos_ids else "length" if len(new_ids) >= 704 else "other"
        row = {"question_id": ids[i], "sample_index": i, "worker_rank": rank, "variant": VARIANT,
               "worker_identity_sha256": identity["identity_sha256"], "answer": answer,
               "input_tokens": input_tokens, "output_tokens": text_count(answer),
               "generated_tokens_including_special": len(new_ids), "finish_reason": finish,
               "generation_latency_ms": latency, "selected_context_count": len(spans),
               "selected_chunk_ids": diagnostics["selected_chunk_ids"], "packing": diagnostics,
               "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
               "evidence_spans": [{k: v for k, v in span.items() if k != "text"} for span in spans]}
        row["record_sha256"] = _json_sha256(row)
        _atomic_json(records / f"{i:04d}.json", row)
        _write_worker_state(state, identity, completed, 500)
        LOG.info("e23_public_generation worker=%d device=%s completed=%d total=500 question_id=%s finish=%s", rank, device, completed, ids[i], finish)
    return {"worker_rank": rank, "completed": 500, "device": device}


def finalize(*, root: Path, output: Path, public: Path, config: Config) -> dict[str, Any]:
    checked = check_preflight(root, output, config)
    _, ids = load_public_questions(public, config.public)
    prepared = load_prepared(output, ids, config)
    rows = [None] * 1000
    identities = []
    for rank in range(2):
        state = json.loads((output / f"generation/worker-{rank}-state.json").read_text(encoding="utf-8"))
        identity = state.get("run_identity", {})
        expected_indices = assigned(rank)
        if (state.get("complete") is not True or state.get("completed_count") != 500 or state.get("assigned_count") != 500
                or identity.get("worker_rank") != rank or identity.get("device") != f"cuda:{rank}" or identity.get("variant") != VARIANT
                or identity.get("config_sha256") != config.sha or identity.get("code_sha256") != code_sha(root)
                or identity.get("prepared_sha256") != file_sha256(output / "retrieval/results.jsonl")
                or identity.get("adapter_sha256") != checked["adapter_sha256"] or identity.get("runtime") != checked["selection"]["runtime"]
                or identity.get("generation_config_sha256") != checked["selection"]["generation_config_sha256"]
                or identity.get("assigned_indices_sha256") != hashlib.sha256(",".join(map(str, expected_indices)).encode("ascii")).hexdigest()
                or identity.get("identity_sha256") != _json_sha256({k: v for k, v in identity.items() if k != "identity_sha256"})):
            raise FinalPublicError("Incomplete or incompatible E23 worker state.")
        identities.append(identity["identity_sha256"])
        for i in expected_indices:
            row = json.loads((output / f"generation/records/{i:04d}.json").read_text(encoding="utf-8"))
            validate_record(row, ids[i], i, rank, identity)
            if row["input_tokens"] > 8192 or row["packing"]["seed_ranks"] != list(range(12)) or row["packing"]["skipped_seed_ranks"]:
                raise FinalPublicError(f"Invalid saved packing: {i}")
            rows[i] = row
    _atomic_jsonl(output / "generation/results.jsonl", rows)
    report = finalize_submission(output_directory=output, public_path=public, config=config.public)
    report["selected_stack"].update({"contexts": "saved-public-top12-compact-merged-plus-bounded-same-article-parent",
                                     "generator": VARIANT, "adapter": "e19-metadata-aware-context-full-train-5636", "max_new_tokens": 704})
    report["model_selection"] = {**config.raw["selection"], "formal_promotion": False,
                                 "operator_selected_public_trial": True, "rollback_public_meteor": 0.5309}
    report["context_packing"] = {"seed_contexts_per_question": 12,
        "questions_with_expansion": sum(bool(r["packing"]["expansions"]) for r in rows),
        "mean_input_tokens": fmean(r["input_tokens"] for r in rows),
        "maximum_input_tokens": max(r["input_tokens"] for r in rows),
        "mean_body_characters": fmean(r["packing"]["body_characters"] for r in rows),
        "mean_merged_spans": fmean(r["packing"]["merged_span_count"] for r in rows),
        "questions_with_skipped_seeds": sum(bool(r["packing"]["skipped_seed_ranks"]) for r in rows),
        "length_finish_rate": fmean(r["finish_reason"] == "length" for r in rows)}
    answer_checks = [answer_diagnostics(r["answer"]) for r in rows]
    report["output_diagnostics"] = {
        "mean_output_tokens": fmean(r["output_tokens"] for r in rows),
        "mean_answer_characters": fmean(len(r["answer"]) for r in rows),
        "duplicate_line_rate": fmean(item["duplicate_line"] for item in answer_checks),
        "duplicate_sentence_rate": fmean(item["duplicate_sentence"] for item in answer_checks),
        "abbreviation_loop_rate": fmean(item["abbreviation_loop"] for item in answer_checks),
        "non_sentence_ending_rate": fmean(item["non_sentence_ending"] for item in answer_checks),
        "mean_generation_latency_ms": fmean(r["generation_latency_ms"] for r in rows),
    }
    report["evidence"].update({"raw_retrieval_results_sha256": json.loads((output / "prepared-summary.json").read_text(encoding="utf-8"))["raw_retrieval_results_sha256"],
                               "selection_results_sha256": config.raw["selection"]["results_sha256"],
                               "worker_identity_sha256": identities})
    report["public_answers_read"] = False
    report["holdout_untouched"] = True
    _atomic_json(output / "report.json", report)
    return report
