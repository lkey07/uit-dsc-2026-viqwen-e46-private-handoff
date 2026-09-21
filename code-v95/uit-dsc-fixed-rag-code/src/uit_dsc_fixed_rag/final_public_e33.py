"""E32-selected max1024/two-trim public trial on frozen E23 parent contexts."""
from __future__ import annotations

import copy
import hashlib
import importlib.metadata
import json
import logging
import time
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean
from typing import Any

from . import final_public_e23 as e23
from .corpus import file_sha256
from .e02_answer import _atomic_json, _atomic_jsonl
from .e03_rrf_grid import _json_sha256, _load_worker_progress, _write_worker_state
from .e10_repetition_grid import answer_diagnostics
from .e13_e08b_max768_tailtrim import trim_repeated_tail
from .e18_source_metadata import save_once
from .e31_long_token_suffix_trim import trim_long_token_suffix
from .final_public import (
    FinalPublicConfig, FinalPublicError, _atomic_bytes,
    _validate_submission_payload, _write_submission_zip, load_public_questions,
)
from .final_public_e19 import load_generator

EXPERIMENT = "FINAL-public1000-e33-parent-max1024-two-trims-v1"
VARIANT = "e33_parent_max1024_tailtrim_longtoken"
LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class Config:
    raw: dict[str, Any]
    path: Path
    source: e23.Config

    @property
    def sha(self) -> str:
        return file_sha256(self.path)

    @property
    def public(self) -> FinalPublicConfig:
        raw = copy.deepcopy(self.source.public.raw)
        raw["experiment_id"] = EXPERIMENT
        raw["inference"]["max_new_tokens"] = 1024
        raw["model_selection"] = {
            "winner": VARIANT,
            "candidate_results_sha256": self.raw["selection"]["candidate_results_sha256"],
        }
        raw["run_contract"].update({
            "e32_clean_dev120_operator_selection": True,
            "same_e23_parent_context_and_e19_adapter": True,
            "deterministic_max1024_and_two_suffix_trims": True,
            "saved_public_top12_reused_without_search": True,
        })
        return FinalPublicConfig(raw, self.path)


def load_config(root: Path, path: Path) -> Config:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if set(raw) != {"schema_version", "experiment_id", "source_e23_config_path",
                    "source_e23_config_sha256", "selection", "inference", "public_trial"}:
        raise FinalPublicError("E33 config root changed.")
    if raw["schema_version"] != "1.0" or raw["experiment_id"] != EXPERIMENT:
        raise FinalPublicError("E33 experiment identity changed.")
    source_path = root / raw["source_e23_config_path"]
    if (raw["source_e23_config_path"] != "configs/final-public-e23-parent-expanded-max704-v1.json"
            or file_sha256(source_path) != raw["source_e23_config_sha256"]):
        raise FinalPublicError("Pinned E23 public context contract changed.")
    source = e23.load_config(root, source_path)
    expected_inference = {
        "max_input_tokens": 8192, "max_new_tokens": 1024, "do_sample": False,
        "num_beams": 1, "repetition_penalty": 1.0, "no_repeat_ngram_size": 0,
        "enable_thinking": False, "use_cache": True,
        "postprocess_first": "conservative-consecutive-tail-block-trim-v1",
        "postprocess_second": "exact-consecutive-long-token-suffix-trim-v1",
        "minimum_block_tokens": 32, "maximum_block_tokens": 256,
    }
    if raw["inference"] != expected_inference:
        raise FinalPublicError("E33 generation/trim policy changed.")
    if raw["public_trial"] != {
        "sample_size": 1000, "generation_workers": 2, "questions_per_worker": 500,
        "partition": "sample-index-mod-worker-count", "saved_public_top12_reused": True,
        "e21_parent_policy_unchanged": True, "e19_adapter_unchanged": True,
        "public_answers_never_read": True, "rollback_public_meteor": 0.5584,
        "rollback_public_rouge_l": 0.5727,
    }:
        raise FinalPublicError("E33 public trial contract changed.")
    selection = raw["selection"]
    if (set(selection) != {
            "experiment_id", "config_sha256", "code_sha256", "sample_ids_sha256",
            "candidate_variant", "candidate_results_sha256", "candidate_meteor",
            "candidate_rouge_l", "competitor_variant", "competitor_results_sha256",
            "competitor_meteor", "meteor_delta_1280_minus_1024",
            "meteor_ci_1280_minus_1024", "trim_delta_1024_minus_raw",
            "trim_ci_1024_minus_raw", "automatic_promotion_allowed",
            "operator_selected_public_trial",
        } or selection["experiment_id"] != "E32-final-dev120-two-caps-tailtrim-v1"
            or selection["candidate_variant"] != "e32_parent_max1024_tailtrim_longtoken"
            or selection["competitor_variant"] != "e32_parent_max1280_tailtrim_longtoken"
            or selection["candidate_meteor"] <= selection["competitor_meteor"]
            or selection["automatic_promotion_allowed"] is not False
            or selection["operator_selected_public_trial"] is not True):
        raise FinalPublicError("E33 selection contract changed.")
    return Config(raw, path, source)


def code_sha(root: Path) -> str:
    paths = sorted((root / "src/uit_dsc_fixed_rag").rglob("*.py"))
    paths.append(root / "scripts/run_final_public_e33_kaggle.py")
    return _json_sha256({p.relative_to(root).as_posix(): file_sha256(p) for p in paths})


def validate_selection(directory: Path, config: Config) -> dict[str, Any]:
    expected = config.raw["selection"]
    report_path = directory / "report.json"
    candidate = directory / "evaluation" / expected["candidate_variant"] / "results.jsonl"
    competitor = directory / "evaluation" / expected["competitor_variant"] / "results.jsonl"
    if not all(path.is_file() for path in (report_path, candidate, competitor)):
        raise FinalPublicError("Add the FULL saved E32 output/dataset, including both derived results.")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    evidence = report.get("evidence", {})
    metrics = report.get("metrics", {})
    comparison = report.get("paired_final1280_minus_final1024", {})
    trim = report.get("paired_final1024_minus_raw1024", {})
    if (report.get("experiment_id") != expected["experiment_id"]
            or report.get("clean_scored_size") != 120
            or report.get("source_reserved_dev_size") != 121
            or report.get("excluded_prior_dev_duplicate") != {"142213": ["25227", "112319"]}
            or report.get("sample_ids_sha256") != expected["sample_ids_sha256"]
            or report.get("dev121_leader_on_clean120") != expected["candidate_variant"]
            or report.get("automatic_promotion_allowed") is not False
            or report.get("public_read") is not False or report.get("private_untouched") is not True
            or metrics.get(expected["candidate_variant"], {}).get("meteor") != expected["candidate_meteor"]
            or metrics.get(expected["candidate_variant"], {}).get("rouge_l") != expected["candidate_rouge_l"]
            or metrics.get(expected["competitor_variant"], {}).get("meteor") != expected["competitor_meteor"]
            or comparison.get("meteor_mean") != expected["meteor_delta_1280_minus_1024"]
            or comparison.get("meteor_bootstrap_95_ci") != expected["meteor_ci_1280_minus_1024"]
            or trim.get("meteor_mean") != expected["trim_delta_1024_minus_raw"]
            or trim.get("meteor_bootstrap_95_ci") != expected["trim_ci_1024_minus_raw"]
            or evidence.get("config_sha256") != expected["config_sha256"]
            or evidence.get("code_sha256") != expected["code_sha256"]
            or evidence.get("adapter_sha256") != config.source.base.section("full_lora_source")["adapter_sha256"]
            or evidence.get("candidate_results_sha256", {}).get(expected["candidate_variant"]) != expected["candidate_results_sha256"]
            or evidence.get("candidate_results_sha256", {}).get(expected["competitor_variant"]) != expected["competitor_results_sha256"]
            or file_sha256(candidate) != expected["candidate_results_sha256"]
            or file_sha256(competitor) != expected["competitor_results_sha256"]):
        raise FinalPublicError("Saved E32 clean-dev selection evidence changed.")
    for path, variant in ((candidate, expected["candidate_variant"]),
                          (competitor, expected["competitor_variant"])):
        with path.open(encoding="utf-8") as stream:
            rows = [json.loads(line) for line in stream]
        if (len(rows) != 120 or len({r.get("question_id") for r in rows}) != 120
                or any(r.get("variant") != variant or not isinstance(r.get("answer"), str)
                       or not r["answer"].strip()
                       or r.get("record_sha256") != _json_sha256({k: v for k, v in r.items() if k != "record_sha256"})
                       for r in rows)):
            raise FinalPublicError("Saved E32 derived records are incomplete or changed.")
    return {"candidate_results_sha256": file_sha256(candidate),
            "competitor_results_sha256": file_sha256(competitor),
            "clean_scored_size": 120, "candidate_meteor": expected["candidate_meteor"]}


def validate_preflight(*, root: Path, retrieval: Path, e00: Path, e19: Path,
                       e21: Path, e32: Path, public: Path, config: Config) -> dict[str, Any]:
    source = e23.validate_preflight(root=root, retrieval=retrieval, e00=e00, e19=e19,
                                    selection=e21, public=public, config=config.source)
    selected = validate_selection(e32, config)
    return {**source, "experiment_id": EXPERIMENT, "code_sha256": code_sha(root),
            "config_sha256": config.sha, "max_new_tokens": 1024,
            "e21_runtime": source["selection"]["runtime"],
            "e21_generation_config_sha256": source["selection"]["generation_config_sha256"],
            "e32_selection": selected, "rollback_public_meteor": 0.5584}


def check_preflight(root: Path, output: Path, config: Config) -> dict[str, Any]:
    payload = json.loads((output / "preflight.json").read_text(encoding="utf-8"))
    evidence = payload.get("evidence", {})
    if (payload.get("experiment_id") != EXPERIMENT or evidence.get("config_sha256") != config.sha
            or evidence.get("code_sha256") != code_sha(root)
            or evidence.get("sample_size") != 1000 or evidence.get("max_new_tokens") != 1024
            or evidence.get("e32_selection", {}).get("candidate_results_sha256")
            != config.raw["selection"]["candidate_results_sha256"]):
        raise FinalPublicError("E33 preflight identity changed.")
    return evidence


def prepare(*, retrieval: Path, e00: Path, public: Path, output: Path,
            config: Config) -> dict[str, Any]:
    # Preserve the exact E23 context assembly, including its source experiment ID.
    return e23.prepare(retrieval=retrieval, e00=e00, public=public,
                       output=output, config=config.source)


def validate_record(row: dict[str, Any], qid: str, index: int, rank: int,
                    identity: dict[str, Any]) -> None:
    if (row.get("question_id") != qid or row.get("sample_index") != index
            or row.get("worker_rank") != rank or row.get("variant") != VARIANT
            or row.get("worker_identity_sha256") != identity["identity_sha256"]
            or not isinstance(row.get("answer"), str) or not row["answer"].strip()
            or row.get("record_sha256") != _json_sha256({k: v for k, v in row.items() if k != "record_sha256"})):
        raise FinalPublicError(f"Invalid E33 raw generation record: {rank}/{index}")


def run_worker(*, root: Path, output: Path, e19: Path, public: Path,
               config: Config, rank: int, device: str) -> dict[str, Any]:
    import torch
    indices = e23.assigned(rank)
    if device != f"cuda:{rank}":
        raise FinalPublicError("Worker/GPU mismatch.")
    checked = check_preflight(root, output, config)
    questions, ids = load_public_questions(public, config.public)
    prepared = e23.load_prepared(output, ids, config.source)
    expected_runtime = checked["e21_runtime"]
    runtime = {key: importlib.metadata.version(key) for key in expected_runtime}
    if runtime != expected_runtime:
        raise FinalPublicError("Use the exact E21/E32 generation runtime versions.")
    model, tokenizer, placement, parameters = load_generator(
        project_root=root, e19_directory=e19, config=config.public, device=device)
    generation_sha = _json_sha256(model.generation_config.to_dict())
    if generation_sha != checked["e21_generation_config_sha256"]:
        raise FinalPublicError("Generation defaults differ from E21/E32.")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    def rendered(messages):
        return tokenizer.apply_chat_template(messages, tokenize=False,
                                             add_generation_prompt=True, enable_thinking=False)

    def text_count(value):
        return len(tokenizer(value, add_special_tokens=False)["input_ids"])

    identity = {
        "code_sha256": code_sha(root), "config_sha256": config.sha,
        "prepared_sha256": file_sha256(output / "retrieval/results.jsonl"),
        "public_ids_sha256": config.public.section("public")["sample_ids_sha256"],
        "adapter_sha256": checked["adapter_sha256"], "adapter_parameters": parameters,
        "runtime": runtime, "generation_config_sha256": generation_sha,
        "worker_rank": rank, "device": device, "device_map": placement,
        "assigned_indices_sha256": hashlib.sha256(",".join(map(str, indices)).encode("ascii")).hexdigest(),
        "variant": VARIANT, "max_new_tokens": 1024,
    }
    identity["identity_sha256"] = _json_sha256(identity)
    records = output / "generation/records"
    state = output / f"generation/worker-{rank}-state.json"
    records.mkdir(parents=True, exist_ok=True)
    done = _load_worker_progress(records=records, state_path=state, identity=identity,
                                 assigned_indices=indices, sample_ids=ids)
    for index in indices[:done]:
        validate_record(json.loads((records / f"{index:04d}.json").read_text(encoding="utf-8")),
                        ids[index], index, rank, identity)
    for completed, index in enumerate(indices[done:], start=done + 1):
        messages, spans, diagnostics = e23.parent.pack(
            questions[ids[index]], prepared[index], config.source.context,
            lambda m: text_count(rendered(m)), text_count, e23.parent.VARIANTS[1])
        prompt = rendered(messages)
        input_tokens = text_count(prompt)
        if (input_tokens > 8192 or diagnostics["seed_ranks"] != list(range(12))
                or diagnostics["skipped_seed_ranks"]):
            raise FinalPublicError("E33 packing violated the frozen E23 context contract.")
        tensors = {k: v.to(device) for k, v in tokenizer(
            prompt, add_special_tokens=False, return_tensors="pt").items()}
        started = time.perf_counter()
        with torch.inference_mode():
            generated = model.generate(
                **tensors, do_sample=False, num_beams=1, repetition_penalty=1.0,
                no_repeat_ngram_size=0, max_new_tokens=1024, use_cache=True)
        latency = (time.perf_counter() - started) * 1000
        new_ids = generated[0, tensors["input_ids"].shape[1]:]
        answer = tokenizer.decode(new_ids, skip_special_tokens=True,
                                  clean_up_tokenization_spaces=False).strip()
        if not answer:
            raise FinalPublicError(f"Empty raw answer: {ids[index]}")
        eos = tokenizer.eos_token_id
        eos_ids = set(eos if isinstance(eos, list) else [eos])
        finish = ("eos" if len(new_ids) and int(new_ids[-1]) in eos_ids
                  else "length" if len(new_ids) >= 1024 else "other")
        row = {
            "question_id": ids[index], "sample_index": index, "worker_rank": rank,
            "variant": VARIANT, "worker_identity_sha256": identity["identity_sha256"],
            "answer": answer, "input_tokens": input_tokens, "output_tokens": text_count(answer),
            "generated_tokens_including_special": len(new_ids), "finish_reason": finish,
            "generation_latency_ms": latency, "selected_context_count": len(spans),
            "selected_chunk_ids": diagnostics["selected_chunk_ids"], "packing": diagnostics,
            "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "evidence_spans": [{k: v for k, v in span.items() if k != "text"} for span in spans],
        }
        row["record_sha256"] = _json_sha256(row)
        _atomic_json(records / f"{index:04d}.json", row)
        _write_worker_state(state, identity, completed, 500)
        LOG.info("e33_public_generation worker=%d device=%s completed=%d total=500 question_id=%s finish=%s",
                 rank, device, completed, ids[index], finish)
    return {"worker_rank": rank, "completed": 500, "device": device}


def trim_answer(answer: str) -> tuple[str, dict[str, Any]]:
    first = trim_repeated_tail(answer)
    second = trim_long_token_suffix(first["answer"], minimum_block_tokens=32,
                                    maximum_block_tokens=256)
    if not second["answer"].strip():
        raise FinalPublicError("E33 suffix trims removed the complete answer.")
    return second["answer"], {
        "line_sentence_changed": first["changed"],
        "line_sentence_removed_characters": first["removed_characters"],
        "long_token_changed": second["changed"],
        "long_token_removed_characters": second["removed_characters"],
        "long_token_removed_whitespace_tokens": second["removed_whitespace_tokens"],
        "long_token_matched_block_sizes": second["matched_block_sizes"],
    }


def finalize(*, root: Path, output: Path, public: Path, config: Config) -> dict[str, Any]:
    checked = check_preflight(root, output, config)
    _, ids = load_public_questions(public, config.public)
    e23.load_prepared(output, ids, config.source)
    prepared_sha = file_sha256(output / "retrieval/results.jsonl")
    rows: list[dict[str, Any] | None] = [None] * 1000
    identities = []
    for rank in range(2):
        state = json.loads((output / f"generation/worker-{rank}-state.json").read_text(encoding="utf-8"))
        identity = state.get("run_identity", {})
        indices = e23.assigned(rank)
        if (state.get("complete") is not True or state.get("completed_count") != 500
                or state.get("assigned_count") != 500 or identity.get("worker_rank") != rank
                or identity.get("device") != f"cuda:{rank}" or identity.get("variant") != VARIANT
                or identity.get("max_new_tokens") != 1024
                or identity.get("config_sha256") != config.sha
                or identity.get("code_sha256") != code_sha(root)
                or identity.get("prepared_sha256") != prepared_sha
                or identity.get("adapter_sha256") != checked["adapter_sha256"]
                or identity.get("runtime") != checked["e21_runtime"]
                or identity.get("generation_config_sha256") != checked["e21_generation_config_sha256"]
                or identity.get("assigned_indices_sha256")
                != hashlib.sha256(",".join(map(str, indices)).encode("ascii")).hexdigest()
                or identity.get("identity_sha256")
                != _json_sha256({k: v for k, v in identity.items() if k != "identity_sha256"})):
            raise FinalPublicError("Incomplete or incompatible E33 worker state.")
        identities.append(identity["identity_sha256"])
        for index in indices:
            row = json.loads((output / f"generation/records/{index:04d}.json").read_text(encoding="utf-8"))
            validate_record(row, ids[index], index, rank, identity)
            if (row["input_tokens"] > 8192 or row["packing"]["seed_ranks"] != list(range(12))
                    or row["packing"]["skipped_seed_ranks"]):
                raise FinalPublicError(f"Invalid saved E33 packing: {index}")
            rows[index] = row
    if any(row is None for row in rows):
        raise FinalPublicError("Missing E33 public record.")
    raw_rows = rows
    _atomic_jsonl(output / "generation/raw-results.jsonl", raw_rows)
    derived = []
    submission = {}
    for row in raw_rows:
        answer, diagnostics = trim_answer(row["answer"])
        result = {**row, "answer": answer, "source_record_sha256": row["record_sha256"],
                  "postprocess": diagnostics}
        result.pop("record_sha256")
        result["record_sha256"] = _json_sha256(result)
        derived.append(result)
        submission[row["question_id"]] = {"answer": answer}
    _atomic_jsonl(output / "generation/results.jsonl", derived)
    payload = (json.dumps(submission, ensure_ascii=False, indent=2).encode("utf-8") + b"\n")
    if payload.startswith(b"\xef\xbb\xbf"):
        raise FinalPublicError("Submission unexpectedly contains a UTF-8 BOM.")
    _validate_submission_payload(payload, ids)
    submission_path = output / "submission.json"
    archive_path = output / "submission.zip"
    _atomic_bytes(submission_path, payload)
    _write_submission_zip(archive_path, "submission.json", payload)
    with zipfile.ZipFile(archive_path) as archive:
        if archive.namelist() != ["submission.json"] or archive.read("submission.json") != payload:
            raise FinalPublicError("E33 submission archive validation failed.")
    checks = [answer_diagnostics(row["answer"]) for row in derived]
    report = {
        "schema_version": "1.0", "experiment_id": EXPERIMENT,
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "sample_size": 1000,
        "selected_stack": {
            "embedding": config.public.section("source_dense")["candidate"],
            "fusion": "bm25-dense-weighted-rrf-0.5-0.5", "reranker": None,
            "contexts": "saved-public-top12-compact-merged-plus-bounded-same-article-parent",
            "generator": VARIANT, "generator_base": config.public.section("generator")["model_id"],
            "adapter": "e19-metadata-aware-context-full-train-5636",
            "max_new_tokens": 1024, "decoding": "greedy",
            "postprocess": ["exact-line-sentence-suffix-trim", "exact-32-to-256-token-suffix-trim"],
        },
        "validation": {
            "all_question_ids_present": True, "all_answers_non_empty": True,
            "utf8_without_bom": True, "archive_members": ["submission.json"],
            "official_mapping_schema": {"question_id": {"answer": "string"}},
        },
        "files": {
            name: {"sha256": file_sha256(output / name), "bytes": (output / name).stat().st_size}
            for name in ("generation/raw-results.jsonl", "generation/results.jsonl",
                         "submission.json", "submission.zip")
        },
        "model_selection": {**config.raw["selection"], "formal_promotion": False,
                            "rollback_public_meteor": 0.5584},
        "context_packing": {
            "seed_contexts_per_question": 12,
            "questions_with_expansion": sum(bool(row["packing"]["expansions"]) for row in raw_rows),
            "mean_input_tokens": fmean(row["input_tokens"] for row in raw_rows),
            "maximum_input_tokens": max(row["input_tokens"] for row in raw_rows),
            "questions_with_skipped_seeds": sum(bool(row["packing"]["skipped_seed_ranks"]) for row in raw_rows),
            "length_finish_rate": fmean(row["finish_reason"] == "length" for row in raw_rows),
        },
        "output_diagnostics": {
            "mean_raw_output_tokens": fmean(row["output_tokens"] for row in raw_rows),
            "mean_raw_answer_characters": fmean(len(row["answer"]) for row in raw_rows),
            "mean_final_answer_characters": fmean(len(row["answer"]) for row in derived),
            "line_sentence_changed_questions": sum(row["postprocess"]["line_sentence_changed"] for row in derived),
            "long_token_changed_questions": sum(row["postprocess"]["long_token_changed"] for row in derived),
            "both_unchanged_questions": sum(not row["postprocess"]["line_sentence_changed"]
                                            and not row["postprocess"]["long_token_changed"] for row in derived),
            "total_removed_characters": sum(len(raw["answer"]) - len(final["answer"])
                                            for raw, final in zip(raw_rows, derived)),
            "duplicate_line_rate": fmean(item["duplicate_line"] for item in checks),
            "duplicate_sentence_rate": fmean(item["duplicate_sentence"] for item in checks),
            "mean_generation_latency_ms": fmean(row["generation_latency_ms"] for row in raw_rows),
        },
        "evidence": {
            "config_sha256": config.sha,
            "public_sha256": config.public.section("public")["sha256"],
            "sample_ids_sha256": config.public.section("public")["sample_ids_sha256"],
            "raw_retrieval_results_sha256": json.loads((output / "prepared-summary.json").read_text(encoding="utf-8"))["raw_retrieval_results_sha256"],
            "prepared_results_sha256": prepared_sha,
            "adapter_sha256": checked["adapter_sha256"],
            "e21_selection_results_sha256": checked["selection"]["results_sha256"],
            "e32_selection_results_sha256": checked["e32_selection"]["candidate_results_sha256"],
            "worker_identity_sha256": identities,
        },
        "public_answers_read": False, "holdout_untouched": True,
        "warning": "This is a public trial, not an automatic promotion; retain E23 if the public score drops.",
    }
    _atomic_json(output / "report.json", report)
    return report
