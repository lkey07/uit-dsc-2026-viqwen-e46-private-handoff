"""Hybrid full-FP16 public inference: two replicas for short prompts, sharding for long prompts."""
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
from . import final_public_e39 as e39
from . import final_public_e40 as e40
from .corpus import file_sha256
from .e02_answer import _atomic_json, _atomic_jsonl
from .e03_rrf_grid import _json_sha256, _write_worker_state
from .e07_generator_ab import _as_text_chat_messages
from .e10_repetition_grid import answer_diagnostics
from .final_public import (FinalPublicError, _atomic_bytes,
                           _validate_submission_payload, _write_submission_zip,
                           load_public_questions)

EXPERIMENT = "FINAL-public1000-e41-viqwen-e38-fp16-hybrid-v1"
VARIANT = "e41_viqwen_e38_fp16_hybrid_parent_max1024_two_trims"
E40_CONFIG_SHA = "cb57fa694bbab7086d9411a790f8c0caf3f0ad3513fb18833768ed02eba3dcdd"


@dataclass(frozen=True)
class Config:
    raw: dict[str, Any]
    path: Path
    e40: e40.Config

    @property
    def sha(self) -> str:
        return file_sha256(self.path)


def load_config(root: Path, path: Path) -> Config:
    raw = json.loads(path.read_text(encoding="utf-8"))
    expected = {"schema_version", "experiment_id", "source_e40_config_path",
                "source_e40_config_sha256", "execution", "runtime", "resume", "public_trial"}
    if (set(raw) != expected or raw["schema_version"] != "1.0"
            or raw["experiment_id"] != EXPERIMENT
            or raw["source_e40_config_path"] != "configs/final-public-e40-viqwen-e38-fp16-sharded-v1.json"
            or raw["source_e40_config_sha256"] != E40_CONFIG_SHA
            or raw["execution"] != {"precision": "float16", "replica_max_input_tokens": 6000,
                "max_new_tokens": 1024, "short_workers": 2,
                "short_questions_per_gpu": "balanced-alternating", "long_worker": 1,
                "long_device_map": "balanced", "oom_fallback_to_sharded": True,
                "cpu_or_disk_offload": False}
            or raw["runtime"] != {"torch": "2.10.0+cu128", "transformers": "5.16.1",
                                   "peft": "0.19.1", "accelerate": "1.13.0"}
            or raw["resume"] != {"accept_partial_e40": True,
                "e40_experiment_id": e40.EXPERIMENT, "e40_variant": e40.VARIANT}
            or raw["public_trial"] != {"sample_size": 1000,
                "public_answers_never_read": True, "rollback_public_meteor": 0.562,
                "automatic_promotion": False}):
        raise FinalPublicError("E41 reviewed contract changed.")
    source_path = root / raw["source_e40_config_path"]
    if file_sha256(source_path) != E40_CONFIG_SHA:
        raise FinalPublicError("Pinned E40 config changed.")
    return Config(raw, path, e40.load_config(root, source_path))


def code_sha(root: Path) -> str:
    paths = sorted((root / "src/uit_dsc_fixed_rag").rglob("*.py"))
    paths.append(root / "scripts/run_final_public_e41_kaggle.py")
    return _json_sha256({p.relative_to(root).as_posix(): file_sha256(p) for p in paths})


def _base(config: Config) -> e40.Config:
    return config.e40


def _valid_hash(row: dict[str, Any]) -> bool:
    return row.get("record_sha256") == _json_sha256(
        {key: value for key, value in row.items() if key != "record_sha256"})


def _import_e40(partial: Path | None, output: Path, ids: list[str], adapter_sha: str) -> int:
    target = output / "generation/imported"
    existing = sorted(target.glob("*.json")) if target.is_dir() else []
    if existing:
        for index, path in enumerate(existing):
            row = json.loads(path.read_text(encoding="utf-8"))
            if path.name != f"{index:04d}.json" or row.get("sample_index") != index \
                    or row.get("question_id") != ids[index] or row.get("mode") != "imported-e40" \
                    or not _valid_hash(row):
                raise FinalPublicError("Saved E41 imported records changed.")
        return len(existing)
    target.mkdir(parents=True, exist_ok=True)
    if partial is None:
        return 0
    state_path = partial / "generation/worker-0-state.json"
    if not state_path.is_file():
        raise FinalPublicError("The supplied E40 partial output has no worker checkpoint.")
    state = json.loads(state_path.read_text(encoding="utf-8"))
    identity = state.get("run_identity", {})
    completed = int(state.get("completed_count", -1))
    if (not 0 <= completed <= 1000 or identity.get("config_sha256") != E40_CONFIG_SHA
            or identity.get("adapter_sha256") != adapter_sha or identity.get("precision") != "float16"
            or identity.get("model_replicas") != 1 or identity.get("max_new_tokens") != 1024
            or identity.get("variant") != e40.VARIANT
            or identity.get("assigned_indices") != list(range(1000))
            or identity.get("identity_sha256") != _json_sha256(
                {key: value for key, value in identity.items() if key != "identity_sha256"})):
        raise FinalPublicError("The supplied partial artifact is not the reviewed E40 run.")
    records = partial / "generation/records"
    for index in range(completed):
        source_path = records / f"{index:04d}.json"
        if not source_path.is_file():
            raise FinalPublicError("E40 checkpoint is ahead of its durable records.")
        source = json.loads(source_path.read_text(encoding="utf-8"))
        e40._validate_record(source, ids[index], index, identity)
        row = {"question_id": ids[index], "sample_index": index, "mode": "imported-e40",
               "answer": source["answer"], "input_tokens": source["input_tokens"],
               "output_tokens": source["output_tokens"], "finish_reason": source["finish_reason"],
               "generation_latency_ms": source["generation_latency_ms"],
               "selected_context_count": source["selected_context_count"],
               "packing": source["packing"], "prompt_sha256": source["prompt_sha256"],
               "source_record_sha256": source["record_sha256"],
               "source_worker_identity_sha256": identity["identity_sha256"]}
        row["record_sha256"] = _json_sha256(row)
        _atomic_json(target / f"{index:04d}.json", row)
    return completed


def preflight(*, root: Path, source: Path, training: Path, public: Path, output: Path,
              config: Config, e40_partial: Path | None = None) -> dict[str, Any]:
    prepared, ids = e39.validate_e33_source(source, public, _base(config))
    adapter_sha, complete = e39.validate_training(training, _base(config))
    saved_path = output / "preflight.json"
    if saved_path.is_file():
        saved = json.loads(saved_path.read_text(encoding="utf-8"))
        if (saved.get("experiment_id") != EXPERIMENT or saved.get("code_sha256") != code_sha(root)
                or saved.get("config_sha256") != config.sha
                or saved.get("prepared_results_sha256") != file_sha256(source / "retrieval/results.jsonl")
                or saved.get("adapter_sha256") != adapter_sha):
            raise FinalPublicError("Saved E41 preflight belongs to different inputs.")
        _import_e40(None, output, ids, adapter_sha)
        return saved
    imported = _import_e40(e40_partial, output, ids, adapter_sha)
    evidence = {"experiment_id": EXPERIMENT, "code_sha256": code_sha(root),
        "config_sha256": config.sha, "sample_size": len(ids),
        "sample_ids_sha256": config.e40.raw["source_e33"]["sample_ids_sha256"],
        "prepared_results_sha256": file_sha256(source / "retrieval/results.jsonl"),
        "adapter_sha256": adapter_sha, "training_identity_sha256": complete["identity_sha256"],
        "model_id": config.e40.raw["e38_training"]["model_id"],
        "model_revision": config.e40.raw["e38_training"]["model_revision"],
        "imported_e40_questions": imported, "precision": "float16",
        "public_reference_answers_read": False}
    _atomic_json(saved_path, evidence)
    return evidence


def check_preflight(root: Path, source: Path, training: Path, public: Path,
                    output: Path, config: Config) -> dict[str, Any]:
    return preflight(root=root, source=source, training=training, public=public,
                     output=output, config=config, e40_partial=None)


def _runtime(config: Config) -> dict[str, str]:
    observed = {name: importlib.metadata.version(name) for name in config.raw["runtime"]}
    if observed != config.raw["runtime"]:
        raise FinalPublicError(f"Use exact E38 runtime: {observed}")
    return observed


def _tokenizer(model_cache: Path):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(model_cache), trust_remote_code=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer


def _prompt(tokenizer, question: dict[str, Any], prepared: dict[str, Any], config: Config):
    def render(messages):
        return tokenizer.apply_chat_template(_as_text_chat_messages(messages), tokenize=False,
                                             add_generation_prompt=True)
    def count(text):
        return len(tokenizer(text, add_special_tokens=False)["input_ids"])
    messages, spans, packing = e23.parent.pack(
        question, prepared, config.e40.e33.source.context,
        lambda value: count(render(value)), count, e23.parent.VARIANTS[1])
    prompt = render(messages)
    tokens = count(prompt)
    if tokens > 8192 or packing["seed_ranks"] != list(range(12)) or packing["skipped_seed_ranks"]:
        raise FinalPublicError("E41 failed to reproduce the exact E33 context contract.")
    return prompt, tokens, spans, packing, count


def prepare_plan(*, root: Path, source: Path, training: Path, public: Path, output: Path,
                 config: Config, model_cache: Path) -> dict[str, Any]:
    checked = check_preflight(root, source, training, public, output, config)
    if model_cache.resolve().name != config.e40.raw["e38_training"]["model_revision"]:
        raise FinalPublicError("Use the pinned Vi-Qwen revision.")
    e39.checkpoint_parameters(model_cache, _base(config))
    _runtime(config)
    prepared, ids = e39.validate_e33_source(source, public, _base(config))
    questions, _ = load_public_questions(public, config.e40.e33.public)
    tokenizer = _tokenizer(model_cache)
    imported = set(range(checked["imported_e40_questions"]))
    prompt_rows, short, long = [], [], []
    threshold = config.raw["execution"]["replica_max_input_tokens"]
    for index, qid in enumerate(ids):
        prompt, tokens, _, _, _ = _prompt(tokenizer, questions[qid], prepared[index], config)
        row = {"sample_index": index, "question_id": qid, "input_tokens": tokens,
               "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
               "imported_e40": index in imported}
        prompt_rows.append(row)
        if index not in imported:
            (short if tokens <= threshold else long).append(index)
    partitions = [short[::2], short[1::2]]
    plan = {"experiment_id": EXPERIMENT, "config_sha256": config.sha,
            "prepared_results_sha256": checked["prepared_results_sha256"],
            "imported_indices": sorted(imported), "short_indices": short,
            "short_partitions": partitions, "long_indices": long,
            "replica_max_input_tokens": threshold, "prompt_rows": prompt_rows}
    plan["plan_sha256"] = _json_sha256(plan)
    path = output / "plan.json"
    if path.is_file() and json.loads(path.read_text(encoding="utf-8")) != plan:
        raise FinalPublicError("Saved E41 execution plan changed.")
    _atomic_json(path, plan)
    return {"imported": len(imported), "short": len(short), "long": len(long),
            "short_partitions": [len(value) for value in partitions],
            "threshold": threshold, "plan_sha256": plan["plan_sha256"]}


def _load_plan(output: Path, config: Config) -> dict[str, Any]:
    plan = json.loads((output / "plan.json").read_text(encoding="utf-8"))
    if (plan.get("experiment_id") != EXPERIMENT or plan.get("config_sha256") != config.sha
            or plan.get("plan_sha256") != _json_sha256(
                {key: value for key, value in plan.items() if key != "plan_sha256"})):
        raise FinalPublicError("E41 plan changed.")
    return plan


def _progress(records: Path, state: Path, identity: dict[str, Any], assigned: list[int],
              ids: list[str]) -> int:
    if not state.is_file():
        if any((records / f"{index:04d}.json").is_file() for index in assigned):
            raise FinalPublicError("E41 records exist without their checkpoint state.")
        _write_worker_state(state, identity, 0, len(assigned))
        return 0
    saved = json.loads(state.read_text(encoding="utf-8"))
    if saved.get("run_identity") != identity:
        raise FinalPublicError("E41 checkpoint identity changed.")
    completed = 0
    gap = False
    for index in assigned:
        path = records / f"{index:04d}.json"
        if not path.is_file():
            gap = True
            continue
        if gap:
            raise FinalPublicError("E41 records are non-contiguous in a worker partition.")
        row = json.loads(path.read_text(encoding="utf-8"))
        if row.get("sample_index") != index or row.get("question_id") != ids[index] or not _valid_hash(row):
            raise FinalPublicError("E41 worker record changed.")
        completed += 1
    if int(saved.get("completed_count", -1)) > completed:
        raise FinalPublicError("E41 checkpoint is ahead of durable records.")
    _write_worker_state(state, identity, completed, len(assigned))
    return completed


def _answer_row(*, qid: str, index: int, mode: str, identity: dict[str, Any], answer: str,
                input_tokens: int, output_tokens: int, generated_tokens: int, finish: str,
                latency: float, spans: list[Any], packing: dict[str, Any], prompt_sha: str) -> dict[str, Any]:
    row = {"question_id": qid, "sample_index": index, "mode": mode,
           "worker_identity_sha256": identity["identity_sha256"], "answer": answer,
           "input_tokens": input_tokens, "output_tokens": output_tokens,
           "generated_tokens_including_special": generated_tokens, "finish_reason": finish,
           "generation_latency_ms": latency, "selected_context_count": len(spans),
           "packing": packing, "prompt_sha256": prompt_sha}
    row["record_sha256"] = _json_sha256(row)
    return row


def run_short(*, root: Path, source: Path, training: Path, public: Path, output: Path,
              config: Config, model_cache: Path, rank: int, device: str) -> dict[str, Any]:
    import torch
    from peft import PeftModel
    from transformers import Qwen2ForCausalLM
    if rank not in (0, 1) or device != f"cuda:{rank}" or torch.cuda.device_count() != 2:
        raise FinalPublicError("E41 short workers require matching T4 x2 ranks.")
    checked = check_preflight(root, source, training, public, output, config)
    plan = _load_plan(output, config)
    assigned = plan["short_partitions"][rank]
    prepared, ids = e39.validate_e33_source(source, public, _base(config))
    questions, _ = load_public_questions(public, config.e40.e33.public)
    if model_cache.resolve().name != config.e40.raw["e38_training"]["model_revision"]:
        raise FinalPublicError("Use the pinned Vi-Qwen revision.")
    e39.checkpoint_parameters(model_cache, _base(config)); runtime = _runtime(config)
    torch.cuda.set_device(rank)
    tokenizer = _tokenizer(model_cache)
    base = Qwen2ForCausalLM.from_pretrained(str(model_cache), dtype=torch.float16,
        device_map={"": device}, low_cpu_mem_usage=True, trust_remote_code=False)
    model = PeftModel.from_pretrained(base, training / "adapter-final", is_trainable=False)
    model.eval()
    if any(str(parameter.device) != device for parameter in model.parameters()):
        raise FinalPublicError("E41 replica is not wholly resident on its assigned GPU.")
    identity = {"code_sha256": code_sha(root), "config_sha256": config.sha,
        "plan_sha256": plan["plan_sha256"], "adapter_sha256": checked["adapter_sha256"],
        "runtime": runtime, "generation_config_sha256": _json_sha256(model.generation_config.to_dict()),
        "mode": "replica-fp16", "rank": rank, "device": device,
        "assigned_indices": assigned, "max_new_tokens": 1024}
    identity["identity_sha256"] = _json_sha256(identity)
    records = output / "generation/replica-records"; records.mkdir(parents=True, exist_ok=True)
    state = output / f"generation/short-worker-{rank}-state.json"
    done = _progress(records, state, identity, assigned, ids)
    prompt_lookup = {row["sample_index"]: row for row in plan["prompt_rows"]}
    for completed, index in enumerate(assigned[done:], start=done + 1):
        prompt, tokens, spans, packing, count = _prompt(
            tokenizer, questions[ids[index]], prepared[index], config)
        expected = prompt_lookup[index]
        prompt_sha = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        if tokens != expected["input_tokens"] or prompt_sha != expected["prompt_sha256"]:
            raise FinalPublicError("E41 prompt changed after planning.")
        tensors = {key: value.to(device) for key, value in tokenizer(
            prompt, add_special_tokens=False, return_tensors="pt").items()}
        started = time.perf_counter()
        try:
            with torch.inference_mode():
                generated = model.generate(**tensors, do_sample=False, num_beams=1,
                    repetition_penalty=1.0, no_repeat_ngram_size=0,
                    max_new_tokens=1024, use_cache=True)
            latency = (time.perf_counter() - started) * 1000
            new_ids = generated[0, tensors["input_ids"].shape[1]:]
            answer = tokenizer.decode(new_ids.detach().cpu(), skip_special_tokens=True,
                                      clean_up_tokenization_spaces=False).strip()
            if not answer:
                raise FinalPublicError(f"Empty E41 answer: {ids[index]}")
            eos = model.generation_config.eos_token_id
            eos_ids = set(eos if isinstance(eos, list) else [eos])
            finish = "eos" if len(new_ids) and int(new_ids[-1]) in eos_ids else (
                "length" if len(new_ids) >= 1024 else "other")
            row = _answer_row(qid=ids[index], index=index, mode="replica-fp16",
                identity=identity, answer=answer, input_tokens=tokens, output_tokens=count(answer),
                generated_tokens=len(new_ids), finish=finish, latency=latency, spans=spans,
                packing=packing, prompt_sha=prompt_sha)
        except torch.cuda.OutOfMemoryError:
            row = {"question_id": ids[index], "sample_index": index, "mode": "fallback-oom",
                   "worker_identity_sha256": identity["identity_sha256"],
                   "input_tokens": tokens, "prompt_sha256": prompt_sha}
            row["record_sha256"] = _json_sha256(row)
            del tensors
            torch.cuda.empty_cache()
        _atomic_json(records / f"{index:04d}.json", row)
        _write_worker_state(state, identity, completed, len(assigned))
        print(f"E41 GPU{rank} short: {completed}/{len(assigned)} qid={ids[index]} mode={row['mode']}", flush=True)
    return {"rank": rank, "completed": len(assigned)}


def run_long(*, root: Path, source: Path, training: Path, public: Path, output: Path,
             config: Config, model_cache: Path) -> dict[str, Any]:
    import torch
    from peft import PeftModel
    from transformers import Qwen2ForCausalLM
    checked = check_preflight(root, source, training, public, output, config)
    plan = _load_plan(output, config)
    replica = output / "generation/replica-records"
    fallbacks = []
    for rank in (0, 1):
        state = json.loads((output / f"generation/short-worker-{rank}-state.json").read_text(encoding="utf-8"))
        if state.get("complete") is not True:
            raise FinalPublicError("Finish both short workers before sharded generation.")
        for index in plan["short_partitions"][rank]:
            row = json.loads((replica / f"{index:04d}.json").read_text(encoding="utf-8"))
            if row.get("mode") == "fallback-oom": fallbacks.append(index)
    assigned = sorted(plan["long_indices"] + fallbacks)
    prepared, ids = e39.validate_e33_source(source, public, _base(config))
    questions, _ = load_public_questions(public, config.e40.e33.public)
    if model_cache.resolve().name != config.e40.raw["e38_training"]["model_revision"]:
        raise FinalPublicError("Use the pinned Vi-Qwen revision.")
    e39.checkpoint_parameters(model_cache, _base(config)); runtime = _runtime(config)
    tokenizer = _tokenizer(model_cache)
    base = Qwen2ForCausalLM.from_pretrained(str(model_cache), dtype=torch.float16,
        device_map="balanced", max_memory={0: "14GiB", 1: "14GiB"},
        low_cpu_mem_usage=True, trust_remote_code=False)
    device_map = dict(getattr(base, "hf_device_map", {}) or {})
    if e40._devices(device_map) != {"cuda:0", "cuda:1"}:
        raise FinalPublicError("E41 long model was not sharded across both GPUs.")
    model = PeftModel.from_pretrained(base, training / "adapter-final", is_trainable=False)
    model.eval()
    if any(parameter.device.type != "cuda" for parameter in model.parameters()):
        raise FinalPublicError("E41 forbids CPU/disk offload.")
    input_device = model.get_input_embeddings().weight.device
    identity = {"code_sha256": code_sha(root), "config_sha256": config.sha,
        "plan_sha256": plan["plan_sha256"], "adapter_sha256": checked["adapter_sha256"],
        "runtime": runtime, "generation_config_sha256": _json_sha256(model.generation_config.to_dict()),
        "mode": "sharded-fp16", "device_map": device_map,
        "input_device": str(input_device), "assigned_indices": assigned, "max_new_tokens": 1024}
    identity["identity_sha256"] = _json_sha256(identity)
    records = output / "generation/sharded-records"; records.mkdir(parents=True, exist_ok=True)
    state = output / "generation/long-worker-state.json"
    done = _progress(records, state, identity, assigned, ids)
    prompt_lookup = {row["sample_index"]: row for row in plan["prompt_rows"]}
    for completed, index in enumerate(assigned[done:], start=done + 1):
        prompt, tokens, spans, packing, count = _prompt(
            tokenizer, questions[ids[index]], prepared[index], config)
        expected = prompt_lookup[index]; prompt_sha = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        if tokens != expected["input_tokens"] or prompt_sha != expected["prompt_sha256"]:
            raise FinalPublicError("E41 prompt changed after planning.")
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
        if not answer: raise FinalPublicError(f"Empty E41 answer: {ids[index]}")
        eos = model.generation_config.eos_token_id; eos_ids = set(eos if isinstance(eos, list) else [eos])
        finish = "eos" if len(new_ids) and int(new_ids[-1]) in eos_ids else (
            "length" if len(new_ids) >= 1024 else "other")
        row = _answer_row(qid=ids[index], index=index, mode="sharded-fp16", identity=identity,
            answer=answer, input_tokens=tokens, output_tokens=count(answer), generated_tokens=len(new_ids),
            finish=finish, latency=latency, spans=spans, packing=packing, prompt_sha=prompt_sha)
        _atomic_json(records / f"{index:04d}.json", row)
        _write_worker_state(state, identity, completed, len(assigned))
        print(f"E41 sharded long: {completed}/{len(assigned)} qid={ids[index]} finish={finish}", flush=True)
    return {"completed": len(assigned), "long": len(plan["long_indices"]), "oom_fallbacks": len(fallbacks)}


def finalize(*, root: Path, source: Path, training: Path, public: Path,
             output: Path, config: Config) -> dict[str, Any]:
    checked = check_preflight(root, source, training, public, output, config)
    plan = _load_plan(output, config); _, ids = e39.validate_e33_source(source, public, _base(config))
    imported = set(plan["imported_indices"]); short = set(plan["short_indices"])
    raw_rows = []
    for index, qid in enumerate(ids):
        if index in imported: path = output / f"generation/imported/{index:04d}.json"
        elif index in short:
            candidate = output / f"generation/replica-records/{index:04d}.json"
            row0 = json.loads(candidate.read_text(encoding="utf-8"))
            path = (output / f"generation/sharded-records/{index:04d}.json"
                    if row0.get("mode") == "fallback-oom" else candidate)
        else: path = output / f"generation/sharded-records/{index:04d}.json"
        row = json.loads(path.read_text(encoding="utf-8"))
        if row.get("question_id") != qid or row.get("sample_index") != index \
                or row.get("mode") not in {"imported-e40", "replica-fp16", "sharded-fp16"} \
                or not isinstance(row.get("answer"), str) or not row["answer"].strip() or not _valid_hash(row):
            raise FinalPublicError(f"Invalid final E41 record: {index}")
        raw_rows.append(row)
    _atomic_jsonl(output / "generation/raw-results.jsonl", raw_rows)
    final_rows, submission = [], {}
    for row in raw_rows:
        answer, post = e39._trim(row["answer"])
        final = {**row, "answer": answer, "source_record_sha256": row["record_sha256"],
                 "postprocess": post}; final.pop("record_sha256")
        final["record_sha256"] = _json_sha256(final); final_rows.append(final)
        submission[row["question_id"]] = {"answer": answer}
    _atomic_jsonl(output / "generation/results.jsonl", final_rows)
    payload = json.dumps(submission, ensure_ascii=False, indent=2).encode("utf-8") + b"\n"
    _validate_submission_payload(payload, ids); _atomic_bytes(output / "submission.json", payload)
    _write_submission_zip(output / "submission.zip", "submission.json", payload)
    with zipfile.ZipFile(output / "submission.zip") as archive:
        if archive.namelist() != ["submission.json"] or archive.read("submission.json") != payload:
            raise FinalPublicError("Invalid E41 submission archive.")
    checks = [answer_diagnostics(row["answer"]) for row in final_rows]
    modes = {mode: sum(row["mode"] == mode for row in raw_rows)
             for mode in ("imported-e40", "replica-fp16", "sharded-fp16")}
    report = {"schema_version": "1.0", "experiment_id": EXPERIMENT,
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "sample_size": 1000,
        "selected_stack": {"contexts": "exact-saved-E33-parent-contexts",
            "generator": config.e40.raw["e38_training"]["model_id"],
            "adapter": "E38-v2-rank8-max3328", "precision": "float16",
            "execution": "hybrid-two-replicas-short-one-sharded-long",
            "replica_max_input_tokens": 6000, "max_new_tokens": 1024,
            "postprocess": [config.e40.raw["inference"]["postprocess_first"],
                            config.e40.raw["inference"]["postprocess_second"]]},
        "diagnostics": {"mode_counts": modes,
            "length_finish_rate": fmean(row["finish_reason"] == "length" for row in raw_rows),
            "mean_output_tokens": fmean(row["output_tokens"] for row in raw_rows),
            "line_sentence_changed_questions": sum(row["postprocess"]["line_sentence_changed"] for row in final_rows),
            "long_token_changed_questions": sum(row["postprocess"]["long_token_changed"] for row in final_rows),
            "duplicate_line_rate": fmean(value["duplicate_line"] for value in checks),
            "duplicate_sentence_rate": fmean(value["duplicate_sentence"] for value in checks)},
        "files": {name: {"sha256": file_sha256(output / name), "bytes": (output / name).stat().st_size}
                  for name in ("generation/raw-results.jsonl", "generation/results.jsonl",
                               "submission.json", "submission.zip")},
        "evidence": {**checked, "plan_sha256": plan["plan_sha256"]},
        "public_reference_answers_read": False, "private_untouched": True,
        "automatic_promotion": False, "rollback_public_meteor": 0.562,
        "warning": "Direct public trial without E38 dev selection; retain E33 as rollback."}
    _atomic_json(output / "report.json", report)
    return report
