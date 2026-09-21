"""Full-FP16 Vi-Qwen public trial with one model sharded over two T4 GPUs."""
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

from . import final_public_e23 as e23
from . import final_public_e33 as e33
from . import final_public_e39 as e39
from .corpus import file_sha256
from .e02_answer import _atomic_json, _atomic_jsonl
from .e03_rrf_grid import _json_sha256, _load_worker_progress, _write_worker_state
from .e07_generator_ab import _as_text_chat_messages
from .e10_repetition_grid import answer_diagnostics
from .final_public import (FinalPublicError, _atomic_bytes,
                           _validate_submission_payload, _write_submission_zip,
                           load_public_questions)

EXPERIMENT = "FINAL-public1000-e40-viqwen-e38-fp16-sharded-v1"
VARIANT = "e40_viqwen_e38_fp16_sharded_parent_max1024_two_trims"


@dataclass(frozen=True)
class Config:
    raw: dict[str, Any]
    path: Path
    e33: e33.Config

    @property
    def sha(self) -> str:
        return file_sha256(self.path)


def load_config(root: Path, path: Path) -> Config:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if (set(raw) != {"schema_version", "experiment_id", "source_e33_config_path",
                     "source_e33_config_sha256", "source_e33", "e38_training",
                     "inference", "runtime", "execution", "public_trial"}
            or raw["schema_version"] != "1.0" or raw["experiment_id"] != EXPERIMENT
            or raw["source_e33_config_path"] != "configs/final-public-e33-parent-max1024-two-trims-v1.json"
            or raw["source_e33_config_sha256"] != "1e7524279df82606c5bf7a4d5d74e1d8a2cc06123e2430d560e4d02eec4b9bea"):
        raise FinalPublicError("E40 reviewed root contract changed.")
    source_path = root / raw["source_e33_config_path"]
    if file_sha256(source_path) != raw["source_e33_config_sha256"]:
        raise FinalPublicError("Pinned E33 config changed.")
    if raw["execution"] != {"precision": "float16", "device_map": "balanced",
            "cuda_devices": 2, "model_replicas": 1, "generation_workers": 1,
            "questions_per_worker": 1000, "per_gpu_weight_budget": "14GiB",
            "cpu_or_disk_offload": False}:
        raise FinalPublicError("E40 FP16 sharding contract changed.")
    if raw["inference"] != {"max_input_tokens": 8192, "max_new_tokens": 1024,
            "do_sample": False, "num_beams": 1, "repetition_penalty": 1.0,
            "no_repeat_ngram_size": 0, "use_cache": True,
            "postprocess_first": "conservative-consecutive-tail-block-trim-v1",
            "postprocess_second": "exact-consecutive-long-token-suffix-trim-v1",
            "minimum_block_tokens": 32, "maximum_block_tokens": 256}:
        raise FinalPublicError("E40 inference contract changed.")
    if raw["runtime"] != {"torch": "2.10.0+cu128", "transformers": "5.16.1",
                           "peft": "0.19.1", "accelerate": "1.13.0"}:
        raise FinalPublicError("E40 runtime contract changed.")
    if raw["public_trial"] != {"sample_size": 1000,
            "reuse_exact_e33_prepared_contexts": True, "public_answers_never_read": True,
            "rollback_public_meteor": 0.562, "automatic_promotion": False}:
        raise FinalPublicError("E40 public trial contract changed.")
    return Config(raw, path, e33.load_config(root, source_path))


def code_sha(root: Path) -> str:
    paths = sorted((root / "src/uit_dsc_fixed_rag").rglob("*.py"))
    paths.append(root / "scripts/run_final_public_e40_kaggle.py")
    return _json_sha256({p.relative_to(root).as_posix(): file_sha256(p) for p in paths})


def _as_e39_config(config: Config) -> Any:
    """The source/training validators need only raw/path/e33 shared fields."""
    return config


def preflight(*, root: Path, source: Path, training: Path, public: Path,
              output: Path, config: Config) -> dict[str, Any]:
    prepared, ids = e39.validate_e33_source(source, public, _as_e39_config(config))
    adapter_sha, complete = e39.validate_training(training, _as_e39_config(config))
    evidence = {"experiment_id": EXPERIMENT, "code_sha256": code_sha(root),
        "config_sha256": config.sha, "sample_size": len(ids),
        "sample_ids_sha256": config.raw["source_e33"]["sample_ids_sha256"],
        "prepared_results_sha256": file_sha256(source / "retrieval/results.jsonl"),
        "adapter_sha256": adapter_sha, "training_identity_sha256": complete["identity_sha256"],
        "model_id": config.raw["e38_training"]["model_id"],
        "model_revision": config.raw["e38_training"]["model_revision"],
        "prepared_questions": len(prepared), "precision": "float16",
        "model_replicas": 1, "public_reference_answers_read": False}
    _atomic_json(output / "preflight.json", evidence)
    return evidence


def check_preflight(root: Path, source: Path, training: Path, public: Path,
                    output: Path, config: Config) -> dict[str, Any]:
    saved = json.loads((output / "preflight.json").read_text(encoding="utf-8"))
    _, ids = e39.validate_e33_source(source, public, _as_e39_config(config))
    adapter_sha, complete = e39.validate_training(training, _as_e39_config(config))
    if (saved.get("experiment_id") != EXPERIMENT or saved.get("code_sha256") != code_sha(root)
            or saved.get("config_sha256") != config.sha or saved.get("sample_size") != len(ids)
            or saved.get("prepared_results_sha256") != file_sha256(source / "retrieval/results.jsonl")
            or saved.get("adapter_sha256") != adapter_sha
            or saved.get("training_identity_sha256") != complete["identity_sha256"]
            or saved.get("precision") != "float16" or saved.get("model_replicas") != 1):
        raise FinalPublicError("E40 inputs changed after preflight.")
    return saved


def _devices(device_map: dict[str, Any]) -> set[str]:
    result = set()
    for value in device_map.values():
        if isinstance(value, int):
            result.add(f"cuda:{value}")
        else:
            result.add(str(value))
    return result


def _validate_record(row: dict[str, Any], qid: str, index: int,
                     identity: dict[str, Any]) -> None:
    if (row.get("question_id") != qid or row.get("sample_index") != index
            or row.get("variant") != VARIANT
            or row.get("worker_identity_sha256") != identity["identity_sha256"]
            or not isinstance(row.get("answer"), str) or not row["answer"].strip()
            or row.get("record_sha256") != _json_sha256({k: v for k, v in row.items()
                                                          if k != "record_sha256"})):
        raise FinalPublicError(f"Invalid E40 record: {index}")


def run(*, root: Path, source: Path, training: Path, public: Path,
        output: Path, config: Config, model_cache: Path) -> dict[str, Any]:
    import torch
    from peft import PeftModel
    from transformers import AutoTokenizer, Qwen2ForCausalLM
    if torch.cuda.device_count() != 2:
        raise FinalPublicError("E40 requires exactly T4 x2.")
    checked = check_preflight(root, source, training, public, output, config)
    prepared, ids = e39.validate_e33_source(source, public, _as_e39_config(config))
    questions, _ = load_public_questions(public, config.e33.public)
    pin = config.raw["e38_training"]
    if model_cache.resolve().name != pin["model_revision"]:
        raise FinalPublicError("Use the pinned Vi-Qwen revision.")
    e39.checkpoint_parameters(model_cache, _as_e39_config(config))
    runtime = {name: importlib.metadata.version(name) for name in config.raw["runtime"]}
    if runtime != config.raw["runtime"]:
        raise FinalPublicError(f"Use exact E38 runtime: {runtime}")
    tokenizer = AutoTokenizer.from_pretrained(str(model_cache), trust_remote_code=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    base = Qwen2ForCausalLM.from_pretrained(
        str(model_cache), dtype=torch.float16, device_map="balanced",
        max_memory={0: "14GiB", 1: "14GiB"}, low_cpu_mem_usage=True,
        trust_remote_code=False,
    )
    device_map = dict(getattr(base, "hf_device_map", {}) or {})
    devices = _devices(device_map)
    if devices != {"cuda:0", "cuda:1"}:
        raise FinalPublicError(f"Model was not fully sharded across both GPUs: {device_map}")
    model = PeftModel.from_pretrained(base, training / "adapter-final", is_trainable=False)
    model.eval()
    if any(parameter.device.type != "cuda" for parameter in model.parameters()):
        raise FinalPublicError("E40 forbids CPU/disk parameter offload.")
    input_device = model.get_input_embeddings().weight.device
    if str(input_device) not in {"cuda:0", "cuda:1"}:
        raise FinalPublicError("Embedding layer is not on a GPU.")
    generation_sha = _json_sha256(model.generation_config.to_dict())
    indices = list(range(1000))
    identity = {"code_sha256": code_sha(root), "config_sha256": config.sha,
        "prepared_sha256": checked["prepared_results_sha256"],
        "adapter_sha256": checked["adapter_sha256"], "runtime": runtime,
        "generation_config_sha256": generation_sha, "device_map": device_map,
        "input_device": str(input_device), "assigned_indices": indices,
        "variant": VARIANT, "precision": "float16", "model_replicas": 1,
        "max_new_tokens": 1024}
    identity["identity_sha256"] = _json_sha256(identity)
    records = output / "generation/records"
    state = output / "generation/worker-0-state.json"
    records.mkdir(parents=True, exist_ok=True)
    done = _load_worker_progress(records=records, state_path=state, identity=identity,
                                 assigned_indices=indices, sample_ids=ids)
    for index in indices[:done]:
        _validate_record(json.loads((records / f"{index:04d}.json").read_text(encoding="utf-8")),
                         ids[index], index, identity)

    def render(messages):
        return tokenizer.apply_chat_template(_as_text_chat_messages(messages), tokenize=False,
                                             add_generation_prompt=True)
    def count(text):
        return len(tokenizer(text, add_special_tokens=False)["input_ids"])
    for completed, index in enumerate(indices[done:], start=done + 1):
        messages, spans, packing = e23.parent.pack(
            questions[ids[index]], prepared[index], config.e33.source.context,
            lambda value: count(render(value)), count, e23.parent.VARIANTS[1])
        prompt = render(messages)
        input_tokens = count(prompt)
        if (input_tokens > 8192 or packing["seed_ranks"] != list(range(12))
                or packing["skipped_seed_ranks"]):
            raise FinalPublicError(f"E40 cannot preserve all E33 top-12 seeds: {ids[index]}")
        tensors = {key: value.to(input_device) for key, value in tokenizer(
            prompt, add_special_tokens=False, return_tensors="pt").items()}
        started = time.perf_counter()
        with torch.inference_mode():
            generated = model.generate(**tensors, do_sample=False, num_beams=1,
                repetition_penalty=1.0, no_repeat_ngram_size=0,
                max_new_tokens=1024, use_cache=True)
        latency = (time.perf_counter() - started) * 1000
        new_ids = generated[0, tensors["input_ids"].shape[1]:]
        answer = tokenizer.decode(new_ids.detach().cpu(), skip_special_tokens=True,
                                  clean_up_tokenization_spaces=False).strip()
        if not answer:
            raise FinalPublicError(f"Empty E40 answer: {ids[index]}")
        eos = model.generation_config.eos_token_id
        eos_ids = set(eos if isinstance(eos, list) else [eos])
        finish = "eos" if len(new_ids) and int(new_ids[-1]) in eos_ids else (
            "length" if len(new_ids) >= 1024 else "other")
        row = {"question_id": ids[index], "sample_index": index, "variant": VARIANT,
            "worker_identity_sha256": identity["identity_sha256"], "answer": answer,
            "input_tokens": input_tokens, "output_tokens": count(answer),
            "generated_tokens_including_special": len(new_ids), "finish_reason": finish,
            "generation_latency_ms": latency, "selected_context_count": len(spans),
            "packing": packing, "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest()}
        row["record_sha256"] = _json_sha256(row)
        _atomic_json(records / f"{index:04d}.json", row)
        _write_worker_state(state, identity, completed, 1000)
        print(f"E40 FP16-sharded: {completed}/1000 qid={ids[index]} finish={finish}", flush=True)
    return {"completed": 1000, "precision": "float16", "devices": sorted(devices)}


def finalize(*, root: Path, source: Path, training: Path, public: Path,
             output: Path, config: Config) -> dict[str, Any]:
    checked = check_preflight(root, source, training, public, output, config)
    _, ids = e39.validate_e33_source(source, public, _as_e39_config(config))
    state = json.loads((output / "generation/worker-0-state.json").read_text(encoding="utf-8"))
    identity = state.get("run_identity", {})
    if (state.get("complete") is not True or state.get("completed_count") != 1000
            or identity.get("assigned_indices") != list(range(1000))
            or identity.get("code_sha256") != code_sha(root)
            or identity.get("config_sha256") != config.sha
            or identity.get("adapter_sha256") != checked["adapter_sha256"]
            or identity.get("precision") != "float16" or identity.get("model_replicas") != 1
            or _devices(identity.get("device_map", {})) != {"cuda:0", "cuda:1"}
            or identity.get("identity_sha256") != _json_sha256({k: v for k, v in identity.items()
                                                                  if k != "identity_sha256"})):
        raise FinalPublicError("Incomplete or changed E40 state.")
    raw_rows = []
    for index, qid in enumerate(ids):
        row = json.loads((output / f"generation/records/{index:04d}.json").read_text(encoding="utf-8"))
        _validate_record(row, qid, index, identity)
        raw_rows.append(row)
    _atomic_jsonl(output / "generation/raw-results.jsonl", raw_rows)
    final_rows, submission = [], {}
    for row in raw_rows:
        answer, post = e39._trim(row["answer"])
        final = {**row, "answer": answer, "source_record_sha256": row["record_sha256"],
                 "postprocess": post}
        final.pop("record_sha256")
        final["record_sha256"] = _json_sha256(final)
        final_rows.append(final)
        submission[row["question_id"]] = {"answer": answer}
    _atomic_jsonl(output / "generation/results.jsonl", final_rows)
    payload = json.dumps(submission, ensure_ascii=False, indent=2).encode("utf-8") + b"\n"
    _validate_submission_payload(payload, ids)
    _atomic_bytes(output / "submission.json", payload)
    _write_submission_zip(output / "submission.zip", "submission.json", payload)
    with zipfile.ZipFile(output / "submission.zip") as archive:
        if archive.namelist() != ["submission.json"] or archive.read("submission.json") != payload:
            raise FinalPublicError("Invalid E40 submission archive.")
    checks = [answer_diagnostics(row["answer"]) for row in final_rows]
    report = {"schema_version": "1.0", "experiment_id": EXPERIMENT,
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "sample_size": 1000,
        "selected_stack": {"contexts": "exact-saved-E33-parent-contexts",
            "generator": config.raw["e38_training"]["model_id"], "adapter": "E38-v2-rank8-max3328",
            "precision": "float16", "execution": "one-model-balanced-over-two-T4",
            "max_new_tokens": 1024, "postprocess": [config.raw["inference"]["postprocess_first"],
                                                       config.raw["inference"]["postprocess_second"]]},
        "diagnostics": {"length_finish_rate": fmean(r["finish_reason"] == "length" for r in raw_rows),
            "mean_output_tokens": fmean(r["output_tokens"] for r in raw_rows),
            "line_sentence_changed_questions": sum(r["postprocess"]["line_sentence_changed"] for r in final_rows),
            "long_token_changed_questions": sum(r["postprocess"]["long_token_changed"] for r in final_rows),
            "duplicate_line_rate": fmean(x["duplicate_line"] for x in checks),
            "duplicate_sentence_rate": fmean(x["duplicate_sentence"] for x in checks),
            "mean_generation_latency_ms": fmean(r["generation_latency_ms"] for r in raw_rows)},
        "files": {name: {"sha256": file_sha256(output / name), "bytes": (output / name).stat().st_size}
                  for name in ("generation/raw-results.jsonl", "generation/results.jsonl",
                               "submission.json", "submission.zip")},
        "evidence": {**checked, "worker_identity_sha256": identity["identity_sha256"],
                     "device_map": identity["device_map"]},
        "public_reference_answers_read": False, "private_untouched": True,
        "automatic_promotion": False, "rollback_public_meteor": 0.562,
        "warning": "Direct public trial without E38 dev selection; retain E33 as rollback."}
    _atomic_json(output / "report.json", report)
    return report
