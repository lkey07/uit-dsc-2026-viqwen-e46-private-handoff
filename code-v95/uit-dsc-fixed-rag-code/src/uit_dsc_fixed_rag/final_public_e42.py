"""Regenerate only E41 length-finished public answers at max1536 in sharded FP16."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import time
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean
from typing import Any

from . import final_public_e39 as e39
from . import final_public_e40 as e40
from . import final_public_e41 as e41
from .corpus import file_sha256
from .e02_answer import _atomic_json, _atomic_jsonl
from .e03_rrf_grid import _json_sha256, _load_worker_progress, _write_worker_state
from .e10_repetition_grid import answer_diagnostics
from .final_public import (FinalPublicError, _atomic_bytes,
                           _validate_submission_payload, _write_submission_zip,
                           load_public_questions)

EXPERIMENT = "FINAL-public1000-e42-e41-length-max1536-sharded-v1"
VARIANT = "e42_e41_length_only_fp16_sharded_max1536_two_trims"
E41_CONFIG_SHA = "2b584ec70888fa0217b870462ffe932c4ca2dc79628dbe4f078b8dec3c702aed"


@dataclass(frozen=True)
class Config:
    raw: dict[str, Any]
    path: Path
    e41: e41.Config

    @property
    def sha(self) -> str:
        return file_sha256(self.path)


def load_config(root: Path, path: Path) -> Config:
    raw = json.loads(path.read_text(encoding="utf-8"))
    expected = {"schema_version", "experiment_id", "source_e41_config_path",
                "source_e41_config_sha256", "source_e41", "inference",
                "execution", "runtime", "public_trial"}
    if (set(raw) != expected or raw["schema_version"] != "1.0"
            or raw["experiment_id"] != EXPERIMENT
            or raw["source_e41_config_path"] != "configs/final-public-e41-viqwen-e38-fp16-hybrid-v1.json"
            or raw["source_e41_config_sha256"] != E41_CONFIG_SHA):
        raise FinalPublicError("E42 reviewed root contract changed.")
    source_path = root / raw["source_e41_config_path"]
    if file_sha256(source_path) != E41_CONFIG_SHA:
        raise FinalPublicError("Pinned E41 config changed.")
    if raw["inference"] != {"source_max_new_tokens": 1024,
            "candidate_max_new_tokens": 1536,
            "selection": "source-finish-reason-length-only",
            "reuse_source_eos_answers_byte_exact": True, "do_sample": False,
            "num_beams": 1, "repetition_penalty": 1.0, "no_repeat_ngram_size": 0,
            "use_cache": True,
            "postprocess_first": "conservative-consecutive-tail-block-trim-v1",
            "postprocess_second": "exact-consecutive-long-token-suffix-trim-v1",
            "minimum_block_tokens": 32, "maximum_block_tokens": 256}:
        raise FinalPublicError("E42 inference contract changed.")
    if raw["execution"] != {"precision": "float16", "device_map": "balanced",
            "cuda_devices": 2, "model_replicas": 1, "generation_workers": 1,
            "cpu_or_disk_offload": False, "checkpoint_after_each_question": True}:
        raise FinalPublicError("E42 execution contract changed.")
    if raw["runtime"] != {"torch": "2.10.0+cu128", "transformers": "5.16.1",
                           "peft": "0.19.1", "accelerate": "1.13.0"}:
        raise FinalPublicError("E42 runtime contract changed.")
    if raw["public_trial"] != {"sample_size": 1000, "new_generation_questions": 240,
            "public_answers_never_read": True, "rollback_public_meteor": 0.5795,
            "automatic_promotion": False}:
        raise FinalPublicError("E42 public-trial contract changed.")
    return Config(raw, path, e41.load_config(root, source_path))


def code_sha(root: Path) -> str:
    paths = sorted((root / "src/uit_dsc_fixed_rag").rglob("*.py"))
    paths.append(root / "scripts/run_final_public_e42_kaggle.py")
    return _json_sha256({path.relative_to(root).as_posix(): file_sha256(path) for path in paths})


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if line.strip(): rows.append(json.loads(line))
    return rows


def _base(config: Config) -> e40.Config:
    return config.e41.e40


def validate_e41_source(source: Path, ids: list[str], config: Config) -> tuple[
        list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], list[int]]:
    report_path = source / "report.json"
    plan_path = source / "plan.json"
    raw_path = source / "generation/raw-results.jsonl"
    results_path = source / "generation/results.jsonl"
    submission_path = source / "submission.json"
    zip_path = source / "submission.zip"
    if not all(path.is_file() for path in (report_path, plan_path, raw_path, results_path,
                                            submission_path, zip_path)):
        raise FinalPublicError("Add the complete E41 output dataset.")
    pin = config.raw["source_e41"]
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if (report.get("experiment_id") != pin["experiment_id"]
            or report.get("sample_size") != pin["sample_size"]
            or report.get("public_reference_answers_read") is not False
            or report.get("private_untouched") is not True
            or report.get("selected_stack", {}).get("precision") != "float16"
            or report.get("selected_stack", {}).get("max_new_tokens") != 1024
            or report.get("evidence", {}).get("adapter_sha256") != pin["adapter_sha256"]
            or report.get("evidence", {}).get("plan_sha256") != pin["plan_identity_sha256"]):
        raise FinalPublicError("E41 source report changed.")
    file_pins = {raw_path: pin["raw_results_sha256"], results_path: pin["results_sha256"],
                 submission_path: pin["submission_sha256"], zip_path: pin["submission_zip_sha256"]}
    for path, expected in file_pins.items():
        if file_sha256(path) != expected:
            raise FinalPublicError(f"Pinned E41 file changed: {path.name}")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if (plan.get("plan_sha256") != pin["plan_identity_sha256"]
            or plan.get("plan_sha256") != _json_sha256(
                {key: value for key, value in plan.items() if key != "plan_sha256"})):
        raise FinalPublicError("E41 prompt plan changed.")
    raw_rows, final_rows = _read_jsonl(raw_path), _read_jsonl(results_path)
    if len(raw_rows) != 1000 or len(final_rows) != 1000 or len(plan.get("prompt_rows", [])) != 1000:
        raise FinalPublicError("E41 source must contain exactly 1,000 rows.")
    length_indices = []
    for index, qid in enumerate(ids):
        raw, final, prompt = raw_rows[index], final_rows[index], plan["prompt_rows"][index]
        if (raw.get("question_id") != qid or raw.get("sample_index") != index
                or final.get("question_id") != qid or final.get("sample_index") != index
                or prompt.get("question_id") != qid or prompt.get("sample_index") != index
                or not e41._valid_hash(raw) or not e41._valid_hash(final)
                or final.get("source_record_sha256") != raw.get("record_sha256")
                or raw.get("prompt_sha256") != prompt.get("prompt_sha256")
                or raw.get("input_tokens") != prompt.get("input_tokens")):
            raise FinalPublicError(f"E41 source row changed: {index}")
        if raw.get("finish_reason") == "length": length_indices.append(index)
        elif raw.get("finish_reason") not in {"eos", "other"}:
            raise FinalPublicError(f"Unknown E41 finish reason: {index}")
    if len(length_indices) != pin["length_finished_questions"]:
        raise FinalPublicError(f"Expected 240 E41 length rows, got {len(length_indices)}.")
    return raw_rows, final_rows, plan, length_indices


def preflight(*, root: Path, source_e41: Path, source_e33: Path, training: Path,
              public: Path, output: Path, config: Config) -> dict[str, Any]:
    _, ids = e39.validate_e33_source(source_e33, public, _base(config))
    adapter_sha, complete = e39.validate_training(training, _base(config))
    _, _, _, indices = validate_e41_source(source_e41, ids, config)
    evidence = {"experiment_id": EXPERIMENT, "code_sha256": code_sha(root),
        "config_sha256": config.sha, "sample_size": len(ids),
        "sample_ids_sha256": config.e41.e40.raw["source_e33"]["sample_ids_sha256"],
        "e33_prepared_results_sha256": file_sha256(source_e33 / "retrieval/results.jsonl"),
        "e41_raw_results_sha256": file_sha256(source_e41 / "generation/raw-results.jsonl"),
        "e41_results_sha256": file_sha256(source_e41 / "generation/results.jsonl"),
        "e41_length_indices_sha256": _json_sha256(indices), "new_generation_questions": len(indices),
        "adapter_sha256": adapter_sha, "training_identity_sha256": complete["identity_sha256"],
        "model_id": config.e41.e40.raw["e38_training"]["model_id"],
        "model_revision": config.e41.e40.raw["e38_training"]["model_revision"],
        "precision": "float16", "public_reference_answers_read": False}
    saved_path = output / "preflight.json"
    if saved_path.is_file():
        saved = json.loads(saved_path.read_text(encoding="utf-8"))
        if saved != evidence: raise FinalPublicError("Saved E42 preflight belongs to different inputs.")
        return saved
    _atomic_json(saved_path, evidence)
    return evidence


def check_preflight(root: Path, source_e41: Path, source_e33: Path, training: Path,
                    public: Path, output: Path, config: Config) -> dict[str, Any]:
    saved = json.loads((output / "preflight.json").read_text(encoding="utf-8"))
    _, ids = e39.validate_e33_source(source_e33, public, _base(config))
    adapter_sha, complete = e39.validate_training(training, _base(config))
    _, _, _, indices = validate_e41_source(source_e41, ids, config)
    if (saved.get("experiment_id") != EXPERIMENT or saved.get("code_sha256") != code_sha(root)
            or saved.get("config_sha256") != config.sha
            or saved.get("e33_prepared_results_sha256") != file_sha256(source_e33 / "retrieval/results.jsonl")
            or saved.get("e41_raw_results_sha256") != file_sha256(source_e41 / "generation/raw-results.jsonl")
            or saved.get("e41_results_sha256") != file_sha256(source_e41 / "generation/results.jsonl")
            or saved.get("e41_length_indices_sha256") != _json_sha256(indices)
            or saved.get("adapter_sha256") != adapter_sha
            or saved.get("training_identity_sha256") != complete["identity_sha256"]):
        raise FinalPublicError("E42 inputs changed after preflight.")
    return saved


def _validate_record(row: dict[str, Any], qid: str, index: int,
                     identity: dict[str, Any]) -> None:
    if (row.get("question_id") != qid or row.get("sample_index") != index
            or row.get("variant") != VARIANT or row.get("source_finish_reason") != "length"
            or row.get("worker_identity_sha256") != identity["identity_sha256"]
            or not isinstance(row.get("answer"), str) or not row["answer"].strip()
            or row.get("record_sha256") != _json_sha256(
                {key: value for key, value in row.items() if key != "record_sha256"})):
        raise FinalPublicError(f"Invalid E42 record: {index}")


def run(*, root: Path, source_e41: Path, source_e33: Path, training: Path,
        public: Path, output: Path, config: Config, model_cache: Path) -> dict[str, Any]:
    import torch
    from peft import PeftModel
    from transformers import Qwen2ForCausalLM
    if torch.cuda.device_count() != 2:
        raise FinalPublicError("E42 requires T4 x2.")
    checked = check_preflight(root, source_e41, source_e33, training, public, output, config)
    prepared, ids = e39.validate_e33_source(source_e33, public, _base(config))
    raw_source, _, plan, indices = validate_e41_source(source_e41, ids, config)
    questions, _ = load_public_questions(public, config.e41.e40.e33.public)
    revision = config.e41.e40.raw["e38_training"]["model_revision"]
    if model_cache.resolve().name != revision:
        raise FinalPublicError("Use the pinned Vi-Qwen revision.")
    e39.checkpoint_parameters(model_cache, _base(config))
    runtime = {name: importlib.metadata.version(name) for name in config.raw["runtime"]}
    if runtime != config.raw["runtime"]:
        raise FinalPublicError(f"Use exact E38 runtime: {runtime}")
    tokenizer = e41._tokenizer(model_cache)
    base = Qwen2ForCausalLM.from_pretrained(str(model_cache), dtype=torch.float16,
        device_map="balanced", max_memory={0: "14GiB", 1: "14GiB"},
        low_cpu_mem_usage=True, trust_remote_code=False)
    device_map = dict(getattr(base, "hf_device_map", {}) or {})
    if e40._devices(device_map) != {"cuda:0", "cuda:1"}:
        raise FinalPublicError(f"E42 model was not sharded across both GPUs: {device_map}")
    model = PeftModel.from_pretrained(base, training / "adapter-final", is_trainable=False)
    model.eval()
    if any(parameter.device.type != "cuda" for parameter in model.parameters()):
        raise FinalPublicError("E42 forbids CPU/disk parameter offload.")
    input_device = model.get_input_embeddings().weight.device
    identity = {"code_sha256": code_sha(root), "config_sha256": config.sha,
        "e41_raw_results_sha256": checked["e41_raw_results_sha256"],
        "e41_length_indices_sha256": checked["e41_length_indices_sha256"],
        "adapter_sha256": checked["adapter_sha256"], "runtime": runtime,
        "generation_config_sha256": _json_sha256(model.generation_config.to_dict()),
        "device_map": device_map, "input_device": str(input_device),
        "assigned_indices": indices, "variant": VARIANT, "precision": "float16",
        "max_new_tokens": 1536}
    identity["identity_sha256"] = _json_sha256(identity)
    records = output / "generation/records"; records.mkdir(parents=True, exist_ok=True)
    state = output / "generation/worker-0-state.json"
    done = _load_worker_progress(records=records, state_path=state, identity=identity,
                                 assigned_indices=indices, sample_ids=ids)
    for index in indices[:done]:
        _validate_record(json.loads((records / f"{index:04d}.json").read_text(encoding="utf-8")),
                         ids[index], index, identity)
    prompt_lookup = {row["sample_index"]: row for row in plan["prompt_rows"]}
    for completed, index in enumerate(indices[done:], start=done + 1):
        prompt, input_tokens, spans, packing, count = e41._prompt(
            tokenizer, questions[ids[index]], prepared[index], config.e41)
        prompt_sha = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        expected = prompt_lookup[index]
        if (prompt_sha != expected["prompt_sha256"] or input_tokens != expected["input_tokens"]
                or raw_source[index]["prompt_sha256"] != prompt_sha
                or raw_source[index]["finish_reason"] != "length"):
            raise FinalPublicError(f"E42 prompt/source selection changed: {ids[index]}")
        tensors = {key: value.to(input_device) for key, value in tokenizer(
            prompt, add_special_tokens=False, return_tensors="pt").items()}
        started = time.perf_counter()
        with torch.inference_mode():
            generated = model.generate(**tensors, do_sample=False, num_beams=1,
                repetition_penalty=1.0, no_repeat_ngram_size=0,
                max_new_tokens=1536, use_cache=True)
        latency = (time.perf_counter() - started) * 1000
        new_ids = generated[0, tensors["input_ids"].shape[1]:]
        answer = tokenizer.decode(new_ids.detach().cpu(), skip_special_tokens=True,
                                  clean_up_tokenization_spaces=False).strip()
        if not answer: raise FinalPublicError(f"Empty E42 answer: {ids[index]}")
        eos = model.generation_config.eos_token_id
        eos_ids = set(eos if isinstance(eos, list) else [eos])
        finish = "eos" if len(new_ids) and int(new_ids[-1]) in eos_ids else (
            "length" if len(new_ids) >= 1536 else "other")
        row = {"question_id": ids[index], "sample_index": index, "variant": VARIANT,
            "worker_identity_sha256": identity["identity_sha256"], "answer": answer,
            "input_tokens": input_tokens, "output_tokens": count(answer),
            "generated_tokens_including_special": len(new_ids), "finish_reason": finish,
            "source_finish_reason": "length", "source_record_sha256": raw_source[index]["record_sha256"],
            "generation_latency_ms": latency, "selected_context_count": len(spans),
            "packing": packing, "prompt_sha256": prompt_sha}
        row["record_sha256"] = _json_sha256(row)
        _atomic_json(records / f"{index:04d}.json", row)
        _write_worker_state(state, identity, completed, len(indices))
        print(f"E42 sharded max1536: {completed}/{len(indices)} qid={ids[index]} finish={finish}", flush=True)
    return {"completed": len(indices), "precision": "float16",
            "max_new_tokens": 1536, "devices": sorted(e40._devices(device_map))}


def finalize(*, root: Path, source_e41: Path, source_e33: Path, training: Path,
             public: Path, output: Path, config: Config) -> dict[str, Any]:
    checked = check_preflight(root, source_e41, source_e33, training, public, output, config)
    _, ids = e39.validate_e33_source(source_e33, public, _base(config))
    raw_source, final_source, _, indices = validate_e41_source(source_e41, ids, config)
    selected = set(indices)
    state = json.loads((output / "generation/worker-0-state.json").read_text(encoding="utf-8"))
    identity = state.get("run_identity", {})
    if (state.get("complete") is not True or state.get("completed_count") != len(indices)
            or identity.get("assigned_indices") != indices or identity.get("code_sha256") != code_sha(root)
            or identity.get("config_sha256") != config.sha
            or identity.get("adapter_sha256") != checked["adapter_sha256"]
            or identity.get("precision") != "float16" or identity.get("max_new_tokens") != 1536
            or e40._devices(identity.get("device_map", {})) != {"cuda:0", "cuda:1"}
            or identity.get("identity_sha256") != _json_sha256(
                {key: value for key, value in identity.items() if key != "identity_sha256"})):
        raise FinalPublicError("Incomplete or changed E42 worker state.")
    candidates = {}
    for index in indices:
        row = json.loads((output / f"generation/records/{index:04d}.json").read_text(encoding="utf-8"))
        _validate_record(row, ids[index], index, identity); candidates[index] = row
    raw_rows, final_rows, submission = [], [], {}
    reused_eos = 0
    for index, qid in enumerate(ids):
        if index in selected:
            raw = candidates[index]
            answer, post = e39._trim(raw["answer"])
            mode = "regenerated-max1536"
            source_final_sha = None
        else:
            raw = raw_source[index]
            answer = final_source[index]["answer"]
            post = final_source[index].get("postprocess")
            mode = "reused-e41-nonlength"
            source_final_sha = final_source[index]["record_sha256"]
            reused_eos += raw.get("finish_reason") == "eos"
        merged_raw = {"question_id": qid, "sample_index": index, "mode": mode,
            "answer": raw["answer"], "finish_reason": raw["finish_reason"],
            "input_tokens": raw["input_tokens"], "output_tokens": raw["output_tokens"],
            "prompt_sha256": raw["prompt_sha256"], "source_raw_record_sha256":
                raw_source[index]["record_sha256"] if index not in selected else raw["source_record_sha256"]}
        merged_raw["record_sha256"] = _json_sha256(merged_raw); raw_rows.append(merged_raw)
        final = {**merged_raw, "answer": answer, "postprocess": post,
                 "source_generation_record_sha256": raw["record_sha256"],
                 "source_e41_final_record_sha256": source_final_sha}
        final.pop("record_sha256"); final["record_sha256"] = _json_sha256(final)
        final_rows.append(final); submission[qid] = {"answer": answer}
    if reused_eos != sum(row.get("finish_reason") == "eos" for row in raw_source):
        raise FinalPublicError("E42 did not preserve every E41 EOS answer.")
    _atomic_jsonl(output / "generation/raw-results.jsonl", raw_rows)
    _atomic_jsonl(output / "generation/results.jsonl", final_rows)
    payload = json.dumps(submission, ensure_ascii=False, indent=2).encode("utf-8") + b"\n"
    _validate_submission_payload(payload, ids); _atomic_bytes(output / "submission.json", payload)
    _write_submission_zip(output / "submission.zip", "submission.json", payload)
    with zipfile.ZipFile(output / "submission.zip") as archive:
        if archive.namelist() != ["submission.json"] or archive.read("submission.json") != payload:
            raise FinalPublicError("Invalid E42 submission archive.")
    candidate_rows = [candidates[index] for index in indices]
    checks = [answer_diagnostics(row["answer"]) for row in final_rows]
    changed = sum(final_rows[index]["answer"] != final_source[index]["answer"] for index in indices)
    report = {"schema_version": "1.0", "experiment_id": EXPERIMENT,
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "sample_size": 1000,
        "source_public_experiment_id": config.raw["source_e41"]["experiment_id"],
        "source_public_meteor_operator_reported": 0.5795,
        "selected_stack": {"contexts": "exact-saved-E33-parent-contexts",
            "generator": config.e41.e40.raw["e38_training"]["model_id"],
            "adapter": "E38-v2-rank8-max3328", "precision": "float16",
            "selection": "E41-finish-reason-length-only", "new_generation_questions": len(indices),
            "execution": "one-model-balanced-over-two-T4", "max_new_tokens": 1536,
            "postprocess": [config.raw["inference"]["postprocess_first"],
                            config.raw["inference"]["postprocess_second"]]},
        "diagnostics": {"reused_questions": 1000 - len(indices),
            "regenerated_questions": len(indices), "changed_regenerated_answers": changed,
            "candidate_eos_rate": fmean(row["finish_reason"] == "eos" for row in candidate_rows),
            "candidate_length_finish_rate": fmean(row["finish_reason"] == "length" for row in candidate_rows),
            "candidate_mean_output_tokens": fmean(row["output_tokens"] for row in candidate_rows),
            "overall_length_finish_rate": fmean(row["finish_reason"] == "length" for row in raw_rows),
            "line_sentence_changed_regenerated": sum(
                final_rows[index]["postprocess"]["line_sentence_changed"] for index in indices),
            "long_token_changed_regenerated": sum(
                final_rows[index]["postprocess"]["long_token_changed"] for index in indices),
            "duplicate_line_rate": fmean(value["duplicate_line"] for value in checks),
            "duplicate_sentence_rate": fmean(value["duplicate_sentence"] for value in checks)},
        "files": {name: {"sha256": file_sha256(output / name), "bytes": (output / name).stat().st_size}
                  for name in ("generation/raw-results.jsonl", "generation/results.jsonl",
                               "submission.json", "submission.zip")},
        "evidence": {**checked, "worker_identity_sha256": identity["identity_sha256"]},
        "public_reference_answers_read": False, "private_untouched": True,
        "automatic_promotion": False, "rollback_public_meteor": 0.5795,
        "warning": "Length-only public trial; keep E41 as rollback until leaderboard scoring."}
    _atomic_json(output / "report.json", report)
    return report
