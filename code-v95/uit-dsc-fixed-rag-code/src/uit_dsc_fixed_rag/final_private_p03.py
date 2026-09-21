"""P02 Qwen/E19 length-only restart at max1536, then Unified Clean."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import shutil
import time
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean
from typing import Any

from . import final_private_p01 as p01
from . import final_private_p02 as p02
from . import final_public_e40 as e40
from .corpus import file_sha256
from .e02_answer import _atomic_json, _atomic_jsonl
from .e03_rrf_grid import _json_sha256, _write_worker_state
from .final_public import FinalPublicError


EXPERIMENT = "FINAL-private-p03-qwen-e19-length1536-unified-v1"
VARIANT = "p03_qwen35_e19_fp16_length_restart_max1536"


class PrivateLengthError(FinalPublicError):
    """Raised when P03 cannot continue without mixing immutable runs."""


@dataclass(frozen=True)
class Config:
    raw: dict[str, Any]
    path: Path
    base: p02.Config

    @property
    def sha(self) -> str:
        return file_sha256(self.path)

    def section(self, key: str) -> dict[str, Any]:
        value = self.raw.get(key)
        if isinstance(value, dict):
            return value
        return self.base.section(key)


def load_config(root: Path, path: Path) -> Config:
    raw = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "schema_version", "experiment_id", "base_p02_config", "source_p02",
        "inference", "execution", "runtime", "parameter_budget",
        "submission_contract", "run_contract",
    }
    if (set(raw) != required or raw.get("schema_version") != "1.0"
            or raw.get("experiment_id") != EXPERIMENT):
        raise PrivateLengthError("P03 config root changed.")
    base_pin = raw["base_p02_config"]
    base_path = root / base_pin.get("path", "")
    if (not base_path.is_file()
            or file_sha256(base_path) != base_pin.get("sha256")):
        raise PrivateLengthError("Pinned P02 base config changed.")
    base = p02.load_config(base_path)
    source = raw["source_p02"]
    if (source.get("experiment_id") != p02.EXPERIMENT
            or source.get("config_sha256") != base.sha
            or source.get("sample_size") != 1918
            or source.get("eos_questions") != 1463
            or source.get("length_questions") != 455
            or source.get("private_reference_answers_read") is not False
            or source.get("reranker") is not None):
        raise PrivateLengthError("P03 source P02 contract changed.")
    inference = raw["inference"]
    if inference != {
        "source_max_new_tokens": 1024,
        "candidate_max_new_tokens": 1536,
        "regenerate_selection": "source-raw-finish-reason-length-only",
        "reuse_source_eos_answers_byte_exact": True,
        "regenerate_from_original_prompt_not_continuation": True,
        "do_sample": False, "num_beams": 1, "repetition_penalty": 1.0,
        "no_repeat_ngram_size": 0, "enable_thinking": False,
        "use_cache": True,
    }:
        raise PrivateLengthError("P03 deterministic generation policy changed.")
    execution = raw["execution"]
    if (execution.get("precision") != "float16"
            or execution.get("workers") != 1
            or execution.get("model_placement") != "balanced-over-two-T4"
            or execution.get("cpu_or_disk_offload") is not False
            or execution.get("checkpoint_after_each_question") is not True):
        raise PrivateLengthError("P03 execution policy changed.")
    budget = raw["parameter_budget"]
    if (budget.get("maximum_stack_total")
            != budget.get("embedding") + budget.get("generator")
            + budget.get("adapter_parameter_cap")
            or budget["maximum_stack_total"] >= budget["exclusive_limit"]):
        raise PrivateLengthError("P03 stack exceeds the BTC parameter limit.")
    if not raw["run_contract"] or not all(raw["run_contract"].values()):
        raise PrivateLengthError("P03 run contract lost an invariant.")
    return Config(raw=raw, path=path, base=base)


def code_sha(root: Path) -> str:
    paths = [
        root / "src/uit_dsc_fixed_rag/final_private_p03.py",
        root / "src/uit_dsc_fixed_rag/final_private_p02.py",
        root / "src/uit_dsc_fixed_rag/final_private_p01.py",
        root / "src/uit_dsc_fixed_rag/final_public_e43.py",
        root / "src/uit_dsc_fixed_rag/final_public_e44.py",
        root / "scripts/run_final_private_p03_kaggle.py",
    ]
    return _json_sha256({
        path.relative_to(root).as_posix(): file_sha256(path) for path in paths
    })


def _runtime(config: Config) -> dict[str, str]:
    observed = {
        name: importlib.metadata.version(name)
        for name in config.section("runtime")
    }
    if observed != config.section("runtime"):
        raise PrivateLengthError(f"Use the exact P03 runtime: {observed}")
    return observed


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def validate_source_p02(
    source: Path, ids: list[str], config: Config,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    report_path = source / "report.json"
    raw_path = source / "generation/raw-results.jsonl"
    results_path = source / "generation/results.jsonl"
    submission_path = source / "submission.json"
    zip_path = source / "submission.zip"
    required = (report_path, raw_path, results_path, submission_path, zip_path)
    if not all(path.is_file() for path in required):
        raise PrivateLengthError("Add the complete finalized P02 output dataset.")
    pin = config.section("source_p02")
    if (file_sha256(raw_path) != pin["raw_results_sha256"]
            or file_sha256(results_path) != pin["results_sha256"]
            or file_sha256(zip_path) != pin["submission_zip_sha256"]):
        raise PrivateLengthError("Saved P02 prediction files changed.")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    files = report.get("files", {})
    evidence = report.get("evidence", {})
    diagnostics = report.get("diagnostics", {})
    if (report.get("experiment_id") != pin["experiment_id"]
            or report.get("sample_size") != len(ids)
            or report.get("private_reference_answers_read") is not False
            or report.get("selected_stack", {}).get("reranker") is not None
            or report.get("selected_stack", {}).get("max_new_tokens") != 1024
            or evidence.get("code_sha256") != pin["generation_code_sha256"]
            or evidence.get("config_sha256") != pin["config_sha256"]
            or diagnostics.get("eos_questions") != pin["eos_questions"]
            or diagnostics.get("length_questions") != pin["length_questions"]
            or diagnostics.get("changed_questions")
            != pin["postprocess_changed_questions"]):
        raise PrivateLengthError("Saved P02 report changed.")
    for name, path in (
        ("generation/raw-results.jsonl", raw_path),
        ("generation/results.jsonl", results_path),
        ("submission.json", submission_path),
        ("submission.zip", zip_path),
    ):
        details = files.get(name, {})
        if (details.get("sha256") != file_sha256(path)
                or details.get("bytes") != path.stat().st_size):
            raise PrivateLengthError(f"P02 report/file mismatch: {name}")
    raw, final = _read_jsonl(raw_path), _read_jsonl(results_path)
    if len(raw) != len(ids) or len(final) != len(ids):
        raise PrivateLengthError("P02 prediction count changed.")
    submission_bytes = submission_path.read_bytes()
    if submission_bytes.startswith(b"\xef\xbb\xbf"):
        raise PrivateLengthError("P02 submission contains a UTF-8 BOM.")
    submission = json.loads(submission_bytes.decode("utf-8"))
    if list(submission) != ids:
        raise PrivateLengthError("P02 submission ID order changed.")
    with zipfile.ZipFile(zip_path) as archive:
        if (archive.namelist() != ["submission.json"]
                or archive.read("submission.json") != submission_bytes
                or archive.testzip() is not None):
            raise PrivateLengthError("P02 submission archive changed.")
    for index, qid in enumerate(ids):
        source_row, final_row = raw[index], final[index]
        expected_answer, expected_post = p01.unified_clean(source_row["answer"])
        if (source_row.get("question_id") != qid
                or source_row.get("sample_index") != index
                or source_row.get("variant") != p02.VARIANT
                or not p02._valid_hash(source_row)
                or final_row.get("question_id") != qid
                or final_row.get("sample_index") != index
                or not p02._valid_hash(final_row)
                or final_row.get("source_raw_record_sha256")
                != source_row["record_sha256"]
                or final_row.get("answer") != expected_answer
                or final_row.get("postprocess") != expected_post
                or submission.get(qid, {}).get("answer") != final_row["answer"]):
            raise PrivateLengthError(f"Invalid saved P02 prediction row: {index}")
    counts = {
        reason: sum(row.get("finish_reason") == reason for row in raw)
        for reason in ("eos", "length", "other")
    }
    if counts != {"eos": 1463, "length": 455, "other": 0}:
        raise PrivateLengthError(f"P02 finish-reason counts changed: {counts}")
    return raw, final, report


def preflight(
    *, root: Path, source_p00: Path, source_p02: Path, training: Path,
    private: Path, output: Path, config: Config,
) -> dict[str, Any]:
    _, ids, _, p00_report = p01.validate_p00(source_p00, private, config)
    _, adapter_sha, complete = p02._validate_training(root, training, config)
    raw, _, _ = validate_source_p02(source_p02, ids, config)
    evidence = {
        "experiment_id": EXPERIMENT,
        "code_sha256": code_sha(root),
        "config_sha256": config.sha,
        "sample_size": len(ids),
        "private_sha256": p00_report["private_questions_sha256"],
        "sample_ids_sha256": p00_report["sample_ids_sha256"],
        "p00_report_sha256": file_sha256(source_p00 / "report.json"),
        "p00_prepared_results_sha256": file_sha256(
            source_p00 / "prepared/results.jsonl"),
        "source_p02_report_sha256": file_sha256(source_p02 / "report.json"),
        "source_p02_raw_results_sha256": file_sha256(
            source_p02 / "generation/raw-results.jsonl"),
        "source_p02_results_sha256": file_sha256(
            source_p02 / "generation/results.jsonl"),
        "length_indices_sha256": _json_sha256([
            index for index, row in enumerate(raw)
            if row["finish_reason"] == "length"
        ]),
        "new_generation_questions": 455,
        "adapter_sha256": adapter_sha,
        "training_identity_sha256": complete["identity_sha256"],
        "model_id": config.section("e19_training")["model_id"],
        "model_revision": config.section("e19_training")["model_revision"],
        "precision": "float16",
        "reranker": None,
        "private_reference_answers_read": False,
    }
    path = output / "preflight.json"
    if path.is_file() and json.loads(path.read_text(encoding="utf-8")) != evidence:
        raise PrivateLengthError("Saved P03 preflight belongs to different inputs.")
    if not path.is_file():
        _atomic_json(path, evidence)
    return evidence


def check_preflight(**kwargs: Any) -> dict[str, Any]:
    output = kwargs["output"]
    if not (output / "preflight.json").is_file():
        raise PrivateLengthError("Run P03 preflight first.")
    return preflight(**kwargs)


def prepare_plan(
    *, root: Path, source_p00: Path, source_p02: Path, training: Path,
    private: Path, output: Path, config: Config, model_cache: Path,
) -> dict[str, Any]:
    checked = check_preflight(
        root=root, source_p00=source_p00, source_p02=source_p02,
        training=training, private=private, output=output, config=config)
    pin = config.section("e19_training")
    if model_cache.resolve().name != pin["model_revision"]:
        raise PrivateLengthError("Use the pinned Qwen3.5-2B revision.")
    _runtime(config)
    questions, ids, prepared, _ = p01.validate_p00(source_p00, private, config)
    raw, _, _ = validate_source_p02(source_p02, ids, config)
    indices = [
        index for index, row in enumerate(raw)
        if row["finish_reason"] == "length"
    ]
    if len(indices) != 455:
        raise PrivateLengthError("P03 length-only selection changed.")
    tokenizer = p02._tokenizer(model_cache)
    rows = []
    for index in indices:
        qid = ids[index]
        prompt, tokens, _, _, _ = p02._prompt(
            tokenizer, questions[qid], prepared[index], config)
        prompt_sha = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        if (prompt_sha != raw[index]["prompt_sha256"]
                or tokens != raw[index]["input_tokens"]):
            raise PrivateLengthError(
                f"P03 cannot reconstruct the original P02 prompt: {index}")
        rows.append({
            "sample_index": index, "question_id": qid,
            "input_tokens": tokens, "prompt_sha256": prompt_sha,
            "source_max1024_record_sha256": raw[index]["record_sha256"],
        })
    plan = {
        "experiment_id": EXPERIMENT,
        "config_sha256": config.sha,
        "source_p02_raw_results_sha256": checked[
            "source_p02_raw_results_sha256"],
        "length_indices": indices,
        "prompt_rows": rows,
    }
    plan["plan_sha256"] = _json_sha256(plan)
    path = output / "plan.json"
    if path.is_file() and json.loads(path.read_text(encoding="utf-8")) != plan:
        raise PrivateLengthError("Saved P03 execution plan changed.")
    if not path.is_file():
        _atomic_json(path, plan)
    return {
        "sample_size": len(ids), "regenerated_length_questions": len(indices),
        "reused_eos_questions": len(ids) - len(indices),
        "minimum_input_tokens": min(row["input_tokens"] for row in rows),
        "maximum_input_tokens": max(row["input_tokens"] for row in rows),
        "plan_sha256": plan["plan_sha256"],
    }


def _load_plan(output: Path, config: Config) -> dict[str, Any]:
    path = output / "plan.json"
    if not path.is_file():
        raise PrivateLengthError("Run P03 plan first.")
    plan = json.loads(path.read_text(encoding="utf-8"))
    if (plan.get("experiment_id") != EXPERIMENT
            or plan.get("config_sha256") != config.sha
            or plan.get("plan_sha256") != _json_sha256({
                key: value for key, value in plan.items()
                if key != "plan_sha256"
            })):
        raise PrivateLengthError("P03 plan identity changed.")
    return plan


def _load_sharded(model_cache: Path, training: Path, config: Config):
    import torch
    from peft import PeftModel
    from transformers import Qwen3_5ForCausalLM

    if (torch.cuda.device_count() != 2
            or any(torch.cuda.get_device_name(index) != "Tesla T4"
                   for index in range(2))):
        raise PrivateLengthError("P03 requires matching T4 x2.")
    base = Qwen3_5ForCausalLM.from_pretrained(
        str(model_cache), dtype=torch.float16, device_map="balanced",
        max_memory={0: "14GiB", 1: "14GiB"}, low_cpu_mem_usage=True,
        trust_remote_code=False)
    base_parameters = sum(parameter.numel() for parameter in base.parameters())
    accepted = config.section("e19_training")["accepted_runtime_parameter_counts"]
    if base_parameters not in accepted:
        raise PrivateLengthError(
            f"Unexpected Qwen3.5-2B parameter count: {base_parameters}")
    device_map = dict(getattr(base, "hf_device_map", {}) or {})
    if e40._devices(device_map) != {"cuda:0", "cuda:1"}:
        raise PrivateLengthError(
            f"P03 model was not sharded across both GPUs: {device_map}")
    model = PeftModel.from_pretrained(
        base, training / "training/adapter-final", is_trainable=False)
    model.eval()
    adapter_parameters = sum(
        parameter.numel() for name, parameter in model.named_parameters()
        if "lora_" in name)
    if adapter_parameters != config.section("e19_training")["adapter_parameters"]:
        raise PrivateLengthError("E19 adapter parameter count changed.")
    if any(parameter.device.type != "cuda" for parameter in model.parameters()):
        raise PrivateLengthError("P03 forbids CPU/disk parameter offload.")
    input_device = model.get_input_embeddings().weight.device
    return model, base_parameters, adapter_parameters, device_map, input_device


def _generate(model: Any, tokenizer: Any, prompt: str, device: Any, count: Any):
    import torch

    tensors = {
        key: value.to(device) for key, value in tokenizer(
            prompt, add_special_tokens=False, return_tensors="pt").items()
    }
    started = time.perf_counter()
    with torch.inference_mode():
        generated = model.generate(
            **tensors, do_sample=False, num_beams=1, repetition_penalty=1.0,
            no_repeat_ngram_size=0, max_new_tokens=1536, use_cache=True)
    latency = (time.perf_counter() - started) * 1000
    new_ids = generated[0, tensors["input_ids"].shape[1]:]
    answer = tokenizer.decode(
        new_ids.detach().cpu(), skip_special_tokens=True,
        clean_up_tokenization_spaces=False).strip()
    if not answer:
        raise PrivateLengthError("Qwen/E19 returned an empty P03 answer.")
    eos = model.generation_config.eos_token_id
    eos_ids = set(eos if isinstance(eos, list) else [eos])
    finish = "eos" if len(new_ids) and int(new_ids[-1]) in eos_ids else (
        "length" if len(new_ids) >= 1536 else "other")
    return answer, count(answer), len(new_ids), finish, latency


def run_length1536(
    *, root: Path, source_p00: Path, source_p02: Path, training: Path,
    private: Path, output: Path, config: Config, model_cache: Path,
) -> dict[str, Any]:
    checked = check_preflight(
        root=root, source_p00=source_p00, source_p02=source_p02,
        training=training, private=private, output=output, config=config)
    plan = _load_plan(output, config)
    questions, ids, prepared, _ = p01.validate_p00(source_p00, private, config)
    raw1024, _, _ = validate_source_p02(source_p02, ids, config)
    pin = config.section("e19_training")
    if model_cache.resolve().name != pin["model_revision"]:
        raise PrivateLengthError("Use the pinned Qwen3.5-2B revision.")
    runtime = _runtime(config)
    tokenizer = p02._tokenizer(model_cache)
    model, base_parameters, adapter_parameters, device_map, input_device = (
        _load_sharded(model_cache, training, config))
    indices = plan["length_indices"]
    identity = {
        "code_sha256": code_sha(root), "config_sha256": config.sha,
        "plan_sha256": plan["plan_sha256"],
        "adapter_sha256": checked["adapter_sha256"],
        "source_p02_raw_results_sha256": checked[
            "source_p02_raw_results_sha256"],
        "length_indices_sha256": checked["length_indices_sha256"],
        "assigned_indices": indices, "mode": "sharded-fp16-restart",
        "max_new_tokens": 1536, "continuation": False,
        "runtime": runtime, "base_parameters": base_parameters,
        "adapter_parameters": adapter_parameters,
        "generation_config_sha256": _json_sha256(
            model.generation_config.to_dict()),
        "device_map": device_map, "input_device": str(input_device),
    }
    identity["identity_sha256"] = _json_sha256(identity)
    records = output / "generation/records"
    records.mkdir(parents=True, exist_ok=True)
    state = output / "generation/worker-state.json"
    done = p02._progress(records, state, identity, indices, ids)
    prompt_lookup = {
        row["sample_index"]: row for row in plan["prompt_rows"]
    }
    for completed, index in enumerate(indices[done:], start=done + 1):
        qid = ids[index]
        prompt, tokens, spans, packing, count = p02._prompt(
            tokenizer, questions[qid], prepared[index], config)
        prompt_sha = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        expected = prompt_lookup[index]
        if (tokens != expected["input_tokens"]
                or prompt_sha != expected["prompt_sha256"]
                or raw1024[index]["prompt_sha256"] != prompt_sha
                or raw1024[index]["finish_reason"] != "length"):
            raise PrivateLengthError(
                "P03 did not restart the exact original P02 prompt.")
        answer, output_tokens, generated_tokens, finish, latency = _generate(
            model, tokenizer, prompt, input_device, count)
        row = {
            "question_id": qid, "sample_index": index,
            "variant": VARIANT, "mode": "sharded-fp16-restart",
            "worker_identity_sha256": identity["identity_sha256"],
            "answer": answer, "input_tokens": tokens,
            "output_tokens": output_tokens,
            "generated_tokens_including_special": generated_tokens,
            "finish_reason": finish, "generation_latency_ms": latency,
            "selected_context_count": len(spans), "packing": packing,
            "prompt_sha256": prompt_sha,
            "source_max1024_record_sha256": raw1024[index]["record_sha256"],
            "regenerated_from_original_prompt": True,
        }
        row["record_sha256"] = _json_sha256(row)
        _atomic_json(records / f"{index:04d}.json", row)
        _write_worker_state(state, identity, completed, len(indices))
        print(
            f"P03 sharded max1536: {completed}/{len(indices)} "
            f"qid={qid} finish={finish}", flush=True)
    return {"regenerated": len(indices), "reused_eos": len(ids) - len(indices)}


def finalize(
    *, root: Path, source_p00: Path, source_p02: Path, training: Path,
    private: Path, output: Path, config: Config,
) -> dict[str, Any]:
    checked = check_preflight(
        root=root, source_p00=source_p00, source_p02=source_p02,
        training=training, private=private, output=output, config=config)
    plan = _load_plan(output, config)
    _, ids, _, _ = p01.validate_p00(source_p00, private, config)
    raw1024, final1024, source_report = validate_source_p02(
        source_p02, ids, config)
    indices = plan["length_indices"]
    selected = set(indices)
    state_path = output / "generation/worker-state.json"
    if not state_path.is_file():
        raise PrivateLengthError("Missing P03 worker state.")
    state = json.loads(state_path.read_text(encoding="utf-8"))
    identity = state.get("run_identity", {})
    if (state.get("complete") is not True
            or state.get("completed_count") != len(indices)
            or state.get("assigned_count") != len(indices)
            or identity.get("code_sha256") != code_sha(root)
            or identity.get("config_sha256") != config.sha
            or identity.get("plan_sha256") != plan["plan_sha256"]
            or identity.get("adapter_sha256") != checked["adapter_sha256"]
            or identity.get("source_p02_raw_results_sha256")
            != checked["source_p02_raw_results_sha256"]
            or identity.get("length_indices_sha256")
            != checked["length_indices_sha256"]
            or identity.get("assigned_indices") != indices
            or identity.get("mode") != "sharded-fp16-restart"
            or identity.get("max_new_tokens") != 1536
            or identity.get("continuation") is not False
            or identity.get("identity_sha256") != _json_sha256({
                key: value for key, value in identity.items()
                if key != "identity_sha256"
            })):
        raise PrivateLengthError("Incomplete or incompatible P03 worker state.")
    candidates: dict[int, dict[str, Any]] = {}
    for index in indices:
        path = output / f"generation/records/{index:04d}.json"
        row = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
        if (row.get("question_id") != ids[index]
                or row.get("sample_index") != index
                or row.get("variant") != VARIANT
                or row.get("mode") != "sharded-fp16-restart"
                or row.get("worker_identity_sha256")
                != identity["identity_sha256"]
                or row.get("regenerated_from_original_prompt") is not True
                or row.get("source_max1024_record_sha256")
                != raw1024[index]["record_sha256"]
                or row.get("prompt_sha256") != raw1024[index]["prompt_sha256"]
                or not isinstance(row.get("answer"), str)
                or not row["answer"].strip()
                or not p02._valid_hash(row)):
            raise PrivateLengthError(f"Invalid P03 regenerated row: {index}")
        candidates[index] = row
    composite = []
    for index, qid in enumerate(ids):
        source = candidates[index] if index in selected else raw1024[index]
        row = {
            "question_id": qid, "sample_index": index, "variant": VARIANT,
            "mode": ("regenerated-max1536" if index in selected
                     else "reused-max1024-eos"),
            "answer": source["answer"],
            "finish_reason": source["finish_reason"],
            "input_tokens": source["input_tokens"],
            "output_tokens": source["output_tokens"],
            "generated_tokens_including_special": source[
                "generated_tokens_including_special"],
            "prompt_sha256": source["prompt_sha256"],
            "source_generation_record_sha256": source["record_sha256"],
            "source_max1024_record_sha256": raw1024[index]["record_sha256"],
            "regenerated_from_original_prompt": index in selected,
        }
        row["record_sha256"] = _json_sha256(row)
        composite.append(row)
    final, review = p01._clean_rows(composite)
    for index in range(len(ids)):
        if index not in selected and final[index]["answer"] != final1024[index]["answer"]:
            raise PrivateLengthError(
                "A reused P02 EOS answer changed in the P03 candidate.")
    generation = output / "generation"
    _atomic_jsonl(generation / "raw-results.jsonl", composite)
    _atomic_jsonl(generation / "results.jsonl", final)
    _atomic_jsonl(output / "review_changed.jsonl", review)
    p02._write_submission(output, ids, final)
    prefix_matches = sum(
        raw1024[index]["answer"]
        == candidates[index]["answer"][:len(raw1024[index]["answer"])]
        for index in indices)
    diagnostics = p01._diagnostics(composite, final)
    diagnostics.update({
        "reused_eos_questions": len(ids) - len(indices),
        "regenerated_length_questions": len(indices),
        "regenerated_prefix_matches_max1024": prefix_matches,
        "regenerated_prefix_differs_max1024": len(indices) - prefix_matches,
        "candidate_eos_questions": sum(
            row["finish_reason"] == "eos" for row in composite),
        "candidate_length_questions": sum(
            row["finish_reason"] == "length" for row in composite),
        "regenerated_eos_rate": fmean(
            candidates[index]["finish_reason"] == "eos" for index in indices),
    })
    files = {}
    for name in (
        "generation/raw-results.jsonl", "generation/results.jsonl",
        "review_changed.jsonl", "submission.json", "submission.zip",
    ):
        path = output / name
        files[name] = {"sha256": file_sha256(path), "bytes": path.stat().st_size}
    report = {
        "schema_version": "1.0", "experiment_id": EXPERIMENT,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "sample_size": len(ids),
        "selected_stack": {
            "retrieval": "exact-saved-P00-bm25-aiteam-rrf-top12-parent",
            "reranker": None,
            "generator": config.section("e19_training")["model_id"],
            "adapter": "E19-metadata-aware-rank8",
            "precision": "float16",
            "source_generation": "P02-two-replica-max1024",
            "selection": "P02-raw-finish-reason-length-only",
            "length_generation": "restart-original-prompt-sharded-max1536",
            "postprocess": config.section("unified_clean")["stages"],
        },
        "diagnostics": diagnostics,
        "validation": {
            "all_question_ids_present": True,
            "all_answers_non_empty": True,
            "utf8_without_bom": True,
            "archive_members": ["submission.json"],
            "official_mapping_schema": {"question_id": {"answer": "string"}},
        },
        "files": files,
        "source_p02": {
            "report_sha256": file_sha256(source_p02 / "report.json"),
            "raw_results_sha256": file_sha256(
                source_p02 / "generation/raw-results.jsonl"),
            "results_sha256": file_sha256(
                source_p02 / "generation/results.jsonl"),
            "submission_zip_sha256": file_sha256(source_p02 / "submission.zip"),
            "source_diagnostics": source_report["diagnostics"],
        },
        "evidence": {**checked, "plan_sha256": plan["plan_sha256"],
                     "worker_identity_sha256": identity["identity_sha256"]},
        "new_retrieval_performed": False,
        "reranker_performed": False,
        "private_reference_answers_read": False,
        "automatic_promotion": False,
        "warning": (
            "Length-only private candidate; preserve P02 max1024 as rollback "
            "and respect the three-submission budget."),
    }
    _atomic_json(output / "report.json", report)
    shutil.copy2(source_p02 / "submission.zip", output / "source-p02-max1024.zip")
    shutil.copy2(output / "submission.zip", output / "submission-p03-max1536.zip")
    return report


__all__ = [
    "Config", "EXPERIMENT", "PrivateLengthError", "VARIANT", "check_preflight",
    "code_sha", "finalize", "load_config", "preflight", "prepare_plan",
    "run_length1536", "validate_source_p02",
]
