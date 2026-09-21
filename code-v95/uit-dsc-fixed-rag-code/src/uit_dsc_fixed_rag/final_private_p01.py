"""Private Vi-Qwen FP16 generation at max1024 and length-only max1536."""

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
from types import SimpleNamespace
from typing import Any

from . import final_public_e23 as e23
from . import final_public_e39 as e39
from . import final_public_e40 as e40
from . import final_public_e43 as e43
from . import final_public_e44 as e44
from .corpus import file_sha256
from .e02_answer import _atomic_json, _atomic_jsonl
from .e03_rrf_grid import _json_sha256, _write_worker_state
from .e07_generator_ab import _as_text_chat_messages
from .e07_lora import InferencePacking
from .e10_repetition_grid import answer_diagnostics
from .final_private_p00 import load_private_questions
from .final_public import (
    FinalPublicError,
    _atomic_bytes,
    _validate_submission_payload,
    _write_submission_zip,
)


EXPERIMENT = "FINAL-private-p01-viqwen-1024-1536-unified-v1"
VARIANT_1024 = "p01_viqwen_e38_fp16_hybrid_max1024"
VARIANT_1536 = "p01_viqwen_e38_fp16_length_restart_max1536"


class PrivateGenerationError(FinalPublicError):
    """Raised when P01 cannot continue without mixing run identities."""


@dataclass(frozen=True)
class Config:
    raw: dict[str, Any]
    path: Path

    @property
    def sha(self) -> str:
        return file_sha256(self.path)

    def section(self, key: str) -> dict[str, Any]:
        value = self.raw.get(key)
        if not isinstance(value, dict):
            raise PrivateGenerationError(f"Missing P01 config section: {key}")
        return value


def load_config(path: Path) -> Config:
    raw = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "schema_version", "experiment_id", "source_p00", "e38_training",
        "prompt", "context_policy", "inference", "execution", "runtime",
        "unified_clean", "parameter_budget", "submission_contract", "run_contract",
    }
    if set(raw) != required or raw.get("schema_version") != "1.0" or raw.get("experiment_id") != EXPERIMENT:
        raise PrivateGenerationError("P01 config root changed.")
    source = raw["source_p00"]
    if (source.get("experiment_id") != "FINAL-private-p00-retrieval-parent-v1"
            or source.get("selected_contexts") != 12 or source.get("fused_top_k") != 20
            or source.get("answers_used") is not False):
        raise PrivateGenerationError("P01/P00 retrieval contract changed.")
    inference = raw["inference"]
    if inference != {
        "source_max_new_tokens": 1024,
        "candidate_max_new_tokens": 1536,
        "regenerate_selection": "source-finish-reason-length-only",
        "reuse_source_nonlength_answers_byte_exact": True,
        "regenerate_from_original_prompt_not_continuation": True,
        "do_sample": False, "num_beams": 1, "repetition_penalty": 1.0,
        "no_repeat_ngram_size": 0, "use_cache": True,
    }:
        raise PrivateGenerationError("P01 deterministic two-cap policy changed.")
    execution = raw["execution"]
    if (execution.get("precision") != "float16"
            or execution.get("replica_max_input_tokens") != 6000
            or execution.get("short_workers") != 2
            or execution.get("checkpoint_after_each_question") is not True
            or execution.get("cpu_or_disk_offload") is not False):
        raise PrivateGenerationError("P01 FP16/checkpoint policy changed.")
    clean = raw["unified_clean"]
    if (clean.get("normal_partial_min_tokens") != 8
            or clean.get("normal_partial_max_tokens") != 256
            or clean.get("short_structured_partial_min_tokens") != 2
            or clean.get("short_structured_partial_max_tokens") != 7
            or clean.get("long_block_min_tokens") != 32
            or clean.get("e44_long_block_max_tokens") != 512
            or clean.get("maximum_fixed_point_passes_per_stage") != 16
            or clean.get("suffix_only") is not True
            or clean.get("interior_trim") is not False):
        raise PrivateGenerationError("P01 Unified Clean policy changed.")
    budget = raw["parameter_budget"]
    if (budget.get("maximum_stack_total") != budget.get("embedding") + budget.get("generator") + budget.get("adapter_parameter_cap")
            or budget["maximum_stack_total"] >= budget["exclusive_limit"]):
        raise PrivateGenerationError("P01 stack exceeds the BTC parameter limit.")
    contract = raw["run_contract"]
    if not contract or not all(value is True for value in contract.values()):
        raise PrivateGenerationError("P01 run contract lost an invariant.")
    return Config(raw, path)


def code_sha(root: Path) -> str:
    paths = [root / "src/uit_dsc_fixed_rag/final_private_p01.py",
             root / "scripts/run_final_private_p01_kaggle.py"]
    return _json_sha256({path.relative_to(root).as_posix(): file_sha256(path) for path in paths})


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _valid_hash(row: dict[str, Any]) -> bool:
    return row.get("record_sha256") == _json_sha256(
        {key: value for key, value in row.items() if key != "record_sha256"})


def validate_p00(source: Path, private: Path, config: Config) -> tuple[
        dict[str, str], list[str], list[dict[str, Any]], dict[str, Any]]:
    report_path, identity_path = source / "report.json", source / "identity.json"
    raw_path = source / "retrieval/raw-results.jsonl"
    pool_path = source / "retrieval/candidate-pool-top20.jsonl"
    prepared_path = source / "prepared/results.jsonl"
    required = (report_path, identity_path, raw_path, pool_path, prepared_path)
    if not all(path.is_file() for path in required):
        raise PrivateGenerationError("Add the complete P00 output dataset.")
    questions, ids, private_identity = load_private_questions(private)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    identity = json.loads(identity_path.read_text(encoding="utf-8"))
    pin = config.section("source_p00")
    files = report.get("files", {})
    expected_files = {
        "retrieval/raw-results.jsonl": raw_path,
        "retrieval/candidate-pool-top20.jsonl": pool_path,
        "prepared/results.jsonl": prepared_path,
    }
    if (report.get("experiment_id") != pin["experiment_id"]
            or report.get("sample_size") != len(ids)
            or report.get("answers_used") is not False
            or report.get("private_reference_answers_read") is not False
            or report.get("private_questions_sha256") != private_identity["private_sha256"]
            or report.get("sample_ids_sha256") != private_identity["sample_ids_sha256"]
            or report.get("retrieval", {}).get("fused_top_k") != 20
            or report.get("retrieval", {}).get("selected_contexts") != 12
            or report.get("evidence", {}).get("config_sha256") != pin["config_sha256"]
            or report.get("evidence", {}).get("code_sha256") != pin["code_sha256"]
            or report.get("evidence", {}).get("question_text_sha256") != private_identity["question_text_sha256"]
            or report.get("downstream_candidates_must_reuse_prepared_sha256") != file_sha256(prepared_path)):
        raise PrivateGenerationError("P00 report/private identity changed.")
    for name, path in expected_files.items():
        details = files.get(name, {})
        if details.get("sha256") != file_sha256(path) or details.get("bytes") != path.stat().st_size:
            raise PrivateGenerationError(f"P00 output file changed: {name}")
    if (identity.get("experiment_id") != pin["experiment_id"]
            or identity.get("private_sha256") != private_identity["private_sha256"]
            or identity.get("sample_ids_sha256") != private_identity["sample_ids_sha256"]
            or identity.get("config_sha256") != pin["config_sha256"]
            or identity.get("code_sha256") != pin["code_sha256"]
            or identity.get("raw_results_sha256") != file_sha256(raw_path)
            or identity.get("prepared_results_sha256") != file_sha256(prepared_path)):
        raise PrivateGenerationError("P00 identity.json changed.")
    prepared = _read_jsonl(prepared_path)
    if len(prepared) != len(ids):
        raise PrivateGenerationError("P00 prepared row count changed.")
    for index, (qid, row) in enumerate(zip(ids, prepared)):
        if (row.get("question_id") != qid or row.get("sample_index") != index
                or row.get("answers_used") is not False
                or [unit.get("rank") for unit in row.get("units", [])] != list(range(12))):
            raise PrivateGenerationError(f"Invalid P00 prepared row: {index}")
    return questions, ids, prepared, report


def validate_training(training: Path, config: Config) -> tuple[str, dict[str, Any]]:
    return e39.validate_training(training, config)


def preflight(*, root: Path, source_p00: Path, training: Path, private: Path,
              output: Path, config: Config) -> dict[str, Any]:
    _, ids, _, p00 = validate_p00(source_p00, private, config)
    adapter_sha, complete = validate_training(training, config)
    evidence = {
        "experiment_id": EXPERIMENT, "code_sha256": code_sha(root),
        "config_sha256": config.sha, "sample_size": len(ids),
        "private_sha256": p00["private_questions_sha256"],
        "sample_ids_sha256": p00["sample_ids_sha256"],
        "p00_report_sha256": file_sha256(source_p00 / "report.json"),
        "p00_prepared_results_sha256": file_sha256(source_p00 / "prepared/results.jsonl"),
        "adapter_sha256": adapter_sha, "training_identity_sha256": complete["identity_sha256"],
        "model_id": config.section("e38_training")["model_id"],
        "model_revision": config.section("e38_training")["model_revision"],
        "precision": "float16", "private_reference_answers_read": False,
    }
    path = output / "preflight.json"
    if path.is_file():
        if json.loads(path.read_text(encoding="utf-8")) != evidence:
            raise PrivateGenerationError("Saved P01 preflight belongs to different inputs.")
    else:
        _atomic_json(path, evidence)
    return evidence


def check_preflight(*, root: Path, source_p00: Path, training: Path, private: Path,
                    output: Path, config: Config) -> dict[str, Any]:
    if not (output / "preflight.json").is_file():
        raise PrivateGenerationError("Run P01 preflight first.")
    return preflight(root=root, source_p00=source_p00, training=training,
                     private=private, output=output, config=config)


def _runtime(config: Config) -> dict[str, str]:
    observed = {name: importlib.metadata.version(name) for name in config.section("runtime")}
    if observed != config.section("runtime"):
        raise PrivateGenerationError(f"Use the exact E38 runtime: {observed}")
    return observed


def _tokenizer(model_cache: Path):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(model_cache), trust_remote_code=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer


class _PromptRoot:
    def __init__(self, config: Config):
        self.config = config

    def section(self, key: str) -> dict[str, Any]:
        if key == "inference":
            return {"max_input_tokens": 8192, "minimum_contexts": 1}
        if key == "prompt":
            return self.config.section("prompt")
        raise PrivateGenerationError(f"Unexpected prompt section: {key}")


class _ContextSource:
    def __init__(self, config: Config):
        self.source = _PromptRoot(config)


class _ContextConfig:
    def __init__(self, config: Config):
        self.policy = config.section("context_policy")
        self.source = _ContextSource(config)


def _prompt(tokenizer: Any, question: str, prepared: dict[str, Any], config: Config):
    def render(messages: list[dict[str, Any]]) -> str:
        return tokenizer.apply_chat_template(
            _as_text_chat_messages(messages), tokenize=False, add_generation_prompt=True)

    def count(text: str) -> int:
        return len(tokenizer(text, add_special_tokens=False)["input_ids"])

    messages, spans, packing = e23.parent.pack(
        question, prepared, _ContextConfig(config),
        lambda value: count(render(value)), count, e23.parent.VARIANTS[1])
    prompt = render(messages)
    tokens = count(prompt)
    if (tokens > 8192 or packing["seed_ranks"] != list(range(12))
            or packing["skipped_seed_ranks"]):
        raise PrivateGenerationError("P01 could not preserve all P00 top-12 seeds.")
    return prompt, tokens, spans, packing, count


def prepare_plan(*, root: Path, source_p00: Path, training: Path, private: Path,
                 output: Path, config: Config, model_cache: Path) -> dict[str, Any]:
    checked = check_preflight(root=root, source_p00=source_p00, training=training,
                              private=private, output=output, config=config)
    pin = config.section("e38_training")
    if model_cache.resolve().name != pin["model_revision"]:
        raise PrivateGenerationError("Use the pinned Vi-Qwen revision.")
    e39.checkpoint_parameters(model_cache, config); _runtime(config)
    questions, ids, prepared, _ = validate_p00(source_p00, private, config)
    tokenizer = _tokenizer(model_cache)
    rows, short, long = [], [], []
    threshold = config.section("execution")["replica_max_input_tokens"]
    for index, qid in enumerate(ids):
        prompt, tokens, _, _, _ = _prompt(tokenizer, questions[qid], prepared[index], config)
        rows.append({"sample_index": index, "question_id": qid, "input_tokens": tokens,
                     "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest()})
        (short if tokens <= threshold else long).append(index)
    plan = {
        "experiment_id": EXPERIMENT, "config_sha256": config.sha,
        "p00_prepared_results_sha256": checked["p00_prepared_results_sha256"],
        "private_sha256": checked["private_sha256"],
        "short_indices": short, "short_partitions": [short[::2], short[1::2]],
        "long_indices": long, "replica_max_input_tokens": threshold,
        "prompt_rows": rows,
    }
    plan["plan_sha256"] = _json_sha256(plan)
    path = output / "plan.json"
    if path.is_file() and json.loads(path.read_text(encoding="utf-8")) != plan:
        raise PrivateGenerationError("Saved P01 execution plan changed.")
    _atomic_json(path, plan)
    return {"sample_size": len(ids), "short": len(short), "long": len(long),
            "short_partitions": [len(value) for value in plan["short_partitions"]],
            "threshold": threshold, "plan_sha256": plan["plan_sha256"]}


def _load_plan(output: Path, config: Config) -> dict[str, Any]:
    path = output / "plan.json"
    if not path.is_file():
        raise PrivateGenerationError("Run P01 plan first.")
    plan = json.loads(path.read_text(encoding="utf-8"))
    if (plan.get("experiment_id") != EXPERIMENT or plan.get("config_sha256") != config.sha
            or plan.get("plan_sha256") != _json_sha256(
                {key: value for key, value in plan.items() if key != "plan_sha256"})):
        raise PrivateGenerationError("P01 plan identity changed.")
    return plan


def _progress(records: Path, state: Path, identity: dict[str, Any], assigned: list[int],
              ids: list[str]) -> int:
    if not state.is_file():
        if any((records / f"{index:04d}.json").is_file() for index in assigned):
            raise PrivateGenerationError("P01 records exist without their checkpoint state.")
        _write_worker_state(state, identity, 0, len(assigned)); return 0
    saved = json.loads(state.read_text(encoding="utf-8"))
    if saved.get("run_identity") != identity:
        raise PrivateGenerationError("P01 checkpoint belongs to a different run.")
    completed, gap = 0, False
    for index in assigned:
        path = records / f"{index:04d}.json"
        if not path.is_file():
            gap = True; continue
        if gap:
            raise PrivateGenerationError("P01 records are non-contiguous in their partition.")
        row = json.loads(path.read_text(encoding="utf-8"))
        if row.get("sample_index") != index or row.get("question_id") != ids[index] or not _valid_hash(row):
            raise PrivateGenerationError("P01 durable record changed.")
        completed += 1
    if int(saved.get("completed_count", -1)) > completed:
        raise PrivateGenerationError("P01 checkpoint is ahead of durable records.")
    _write_worker_state(state, identity, completed, len(assigned))
    return completed


def _answer_row(*, qid: str, index: int, variant: str, mode: str,
                identity: dict[str, Any], answer: str, input_tokens: int,
                output_tokens: int, generated_tokens: int, finish: str,
                latency: float, spans: list[Any], packing: dict[str, Any],
                prompt_sha: str, source_record_sha: str | None = None) -> dict[str, Any]:
    row = {
        "question_id": qid, "sample_index": index, "variant": variant, "mode": mode,
        "worker_identity_sha256": identity["identity_sha256"], "answer": answer,
        "input_tokens": input_tokens, "output_tokens": output_tokens,
        "generated_tokens_including_special": generated_tokens,
        "finish_reason": finish, "generation_latency_ms": latency,
        "selected_context_count": len(spans), "packing": packing,
        "prompt_sha256": prompt_sha,
    }
    if source_record_sha is not None:
        row["source_max1024_record_sha256"] = source_record_sha
        row["regenerated_from_original_prompt"] = True
    row["record_sha256"] = _json_sha256(row)
    return row


def _generate(model: Any, tokenizer: Any, prompt: str, device: Any,
              max_new_tokens: int, count: Any) -> tuple[str, int, int, str, float]:
    import torch
    tensors = {key: value.to(device) for key, value in tokenizer(
        prompt, add_special_tokens=False, return_tensors="pt").items()}
    started = time.perf_counter()
    with torch.inference_mode():
        generated = model.generate(
            **tensors, do_sample=False, num_beams=1, repetition_penalty=1.0,
            no_repeat_ngram_size=0, max_new_tokens=max_new_tokens, use_cache=True)
    latency = (time.perf_counter() - started) * 1000
    new_ids = generated[0, tensors["input_ids"].shape[1]:]
    answer = tokenizer.decode(new_ids.detach().cpu(), skip_special_tokens=True,
                              clean_up_tokenization_spaces=False).strip()
    if not answer:
        raise PrivateGenerationError("Vi-Qwen returned an empty answer.")
    eos = model.generation_config.eos_token_id
    eos_ids = set(eos if isinstance(eos, list) else [eos])
    finish = "eos" if len(new_ids) and int(new_ids[-1]) in eos_ids else (
        "length" if len(new_ids) >= max_new_tokens else "other")
    return answer, count(answer), len(new_ids), finish, latency


def run_short(*, root: Path, source_p00: Path, training: Path, private: Path,
              output: Path, config: Config, model_cache: Path,
              rank: int, device: str) -> dict[str, Any]:
    import torch
    from peft import PeftModel
    from transformers import Qwen2ForCausalLM
    if rank not in (0, 1) or device != f"cuda:{rank}" or torch.cuda.device_count() != 2:
        raise PrivateGenerationError("P01 short workers require matching T4 x2 ranks.")
    checked = check_preflight(root=root, source_p00=source_p00, training=training,
                              private=private, output=output, config=config)
    plan = _load_plan(output, config); assigned = plan["short_partitions"][rank]
    questions, ids, prepared, _ = validate_p00(source_p00, private, config)
    pin = config.section("e38_training")
    if model_cache.resolve().name != pin["model_revision"]:
        raise PrivateGenerationError("Use the pinned Vi-Qwen revision.")
    e39.checkpoint_parameters(model_cache, config); runtime = _runtime(config)
    torch.cuda.set_device(rank); tokenizer = _tokenizer(model_cache)
    base = Qwen2ForCausalLM.from_pretrained(
        str(model_cache), dtype=torch.float16, device_map={"": device},
        low_cpu_mem_usage=True, trust_remote_code=False)
    model = PeftModel.from_pretrained(base, training / "adapter-final", is_trainable=False)
    model.eval()
    if any(str(parameter.device) != device for parameter in model.parameters()):
        raise PrivateGenerationError("P01 replica is not wholly resident on its assigned GPU.")
    identity = {
        "code_sha256": code_sha(root), "config_sha256": config.sha,
        "plan_sha256": plan["plan_sha256"], "adapter_sha256": checked["adapter_sha256"],
        "runtime": runtime, "generation_config_sha256": _json_sha256(model.generation_config.to_dict()),
        "mode": "replica-fp16", "rank": rank, "device": device,
        "assigned_indices": assigned, "max_new_tokens": 1024,
    }
    identity["identity_sha256"] = _json_sha256(identity)
    records = output / "generation/max1024/replica-records"; records.mkdir(parents=True, exist_ok=True)
    state = output / f"generation/max1024/short-worker-{rank}-state.json"
    done = _progress(records, state, identity, assigned, ids)
    prompt_lookup = {row["sample_index"]: row for row in plan["prompt_rows"]}
    for completed, index in enumerate(assigned[done:], start=done + 1):
        prompt, tokens, spans, packing, count = _prompt(tokenizer, questions[ids[index]], prepared[index], config)
        expected = prompt_lookup[index]; prompt_sha = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        if tokens != expected["input_tokens"] or prompt_sha != expected["prompt_sha256"]:
            raise PrivateGenerationError("P01 prompt changed after planning.")
        try:
            answer, output_tokens, generated_tokens, finish, latency = _generate(
                model, tokenizer, prompt, device, 1024, count)
            row = _answer_row(
                qid=ids[index], index=index, variant=VARIANT_1024, mode="replica-fp16",
                identity=identity, answer=answer, input_tokens=tokens,
                output_tokens=output_tokens, generated_tokens=generated_tokens,
                finish=finish, latency=latency, spans=spans, packing=packing,
                prompt_sha=prompt_sha)
        except torch.cuda.OutOfMemoryError:
            row = {"question_id": ids[index], "sample_index": index,
                   "variant": VARIANT_1024, "mode": "fallback-oom",
                   "worker_identity_sha256": identity["identity_sha256"],
                   "input_tokens": tokens, "prompt_sha256": prompt_sha}
            row["record_sha256"] = _json_sha256(row); torch.cuda.empty_cache()
        _atomic_json(records / f"{index:04d}.json", row)
        _write_worker_state(state, identity, completed, len(assigned))
        print(f"P01 GPU{rank} max1024: {completed}/{len(assigned)} qid={ids[index]} mode={row['mode']}", flush=True)
    return {"rank": rank, "completed": len(assigned)}


def _load_sharded(model_cache: Path, training: Path):
    import torch
    from peft import PeftModel
    from transformers import Qwen2ForCausalLM
    base = Qwen2ForCausalLM.from_pretrained(
        str(model_cache), dtype=torch.float16, device_map="balanced",
        max_memory={0: "14GiB", 1: "14GiB"}, low_cpu_mem_usage=True,
        trust_remote_code=False)
    device_map = dict(getattr(base, "hf_device_map", {}) or {})
    if e40._devices(device_map) != {"cuda:0", "cuda:1"}:
        raise PrivateGenerationError(f"P01 model was not sharded across both GPUs: {device_map}")
    model = PeftModel.from_pretrained(base, training / "adapter-final", is_trainable=False)
    model.eval()
    if any(parameter.device.type != "cuda" for parameter in model.parameters()):
        raise PrivateGenerationError("P01 forbids CPU/disk parameter offload.")
    return model, device_map, model.get_input_embeddings().weight.device


def run_long1024(*, root: Path, source_p00: Path, training: Path, private: Path,
                 output: Path, config: Config, model_cache: Path) -> dict[str, Any]:
    checked = check_preflight(root=root, source_p00=source_p00, training=training,
                              private=private, output=output, config=config)
    plan = _load_plan(output, config)
    replica = output / "generation/max1024/replica-records"
    fallbacks = []
    for rank in (0, 1):
        state_path = output / f"generation/max1024/short-worker-{rank}-state.json"
        state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.is_file() else {}
        if state.get("complete") is not True:
            raise PrivateGenerationError("Finish both max1024 replica workers first.")
        for index in plan["short_partitions"][rank]:
            row = json.loads((replica / f"{index:04d}.json").read_text(encoding="utf-8"))
            if row.get("mode") == "fallback-oom": fallbacks.append(index)
    assigned = sorted(plan["long_indices"] + fallbacks)
    questions, ids, prepared, _ = validate_p00(source_p00, private, config)
    pin = config.section("e38_training")
    if model_cache.resolve().name != pin["model_revision"]:
        raise PrivateGenerationError("Use the pinned Vi-Qwen revision.")
    e39.checkpoint_parameters(model_cache, config); runtime = _runtime(config)
    tokenizer = _tokenizer(model_cache)
    records = output / "generation/max1024/sharded-records"; records.mkdir(parents=True, exist_ok=True)
    state_path = output / "generation/max1024/long-worker-state.json"
    if not assigned:
        identity = {"code_sha256": code_sha(root), "config_sha256": config.sha,
                    "plan_sha256": plan["plan_sha256"], "adapter_sha256": checked["adapter_sha256"],
                    "mode": "sharded-fp16", "assigned_indices": [], "max_new_tokens": 1024}
        identity["identity_sha256"] = _json_sha256(identity)
        _progress(records, state_path, identity, [], ids)
        return {"completed": 0, "long": 0, "oom_fallbacks": 0}
    model, device_map, input_device = _load_sharded(model_cache, training)
    identity = {
        "code_sha256": code_sha(root), "config_sha256": config.sha,
        "plan_sha256": plan["plan_sha256"], "adapter_sha256": checked["adapter_sha256"],
        "runtime": runtime, "generation_config_sha256": _json_sha256(model.generation_config.to_dict()),
        "mode": "sharded-fp16", "device_map": device_map,
        "input_device": str(input_device), "assigned_indices": assigned,
        "max_new_tokens": 1024,
    }
    identity["identity_sha256"] = _json_sha256(identity)
    done = _progress(records, state_path, identity, assigned, ids)
    prompt_lookup = {row["sample_index"]: row for row in plan["prompt_rows"]}
    for completed, index in enumerate(assigned[done:], start=done + 1):
        prompt, tokens, spans, packing, count = _prompt(tokenizer, questions[ids[index]], prepared[index], config)
        expected = prompt_lookup[index]; prompt_sha = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        if tokens != expected["input_tokens"] or prompt_sha != expected["prompt_sha256"]:
            raise PrivateGenerationError("P01 max1024 sharded prompt changed.")
        answer, output_tokens, generated_tokens, finish, latency = _generate(
            model, tokenizer, prompt, input_device, 1024, count)
        row = _answer_row(
            qid=ids[index], index=index, variant=VARIANT_1024, mode="sharded-fp16",
            identity=identity, answer=answer, input_tokens=tokens,
            output_tokens=output_tokens, generated_tokens=generated_tokens,
            finish=finish, latency=latency, spans=spans, packing=packing,
            prompt_sha=prompt_sha)
        _atomic_json(records / f"{index:04d}.json", row)
        _write_worker_state(state_path, identity, completed, len(assigned))
        print(f"P01 sharded max1024: {completed}/{len(assigned)} qid={ids[index]} finish={finish}", flush=True)
    return {"completed": len(assigned), "long": len(plan["long_indices"]), "oom_fallbacks": len(fallbacks)}


_E43_POLICY = {
    "pipeline": ["exact-partial-repeated-suffix-trim-v1",
                 "conservative-consecutive-tail-block-trim-v1",
                 "exact-consecutive-long-token-suffix-trim-v1"],
    "minimum_partial_whitespace_tokens": 8,
    "maximum_partial_whitespace_tokens": 256,
    "minimum_prior_complete_copies": 2,
    "partial_boundaries": ["line", "sentence"],
    "partial_match": "unicode-nfkc-casefold-collapse-whitespace-exact-prefix",
    "minimum_long_block_tokens": 32, "maximum_long_block_tokens": 256,
    "maximum_fixed_point_passes": 16, "suffix_only": True,
    "ordinary_answers_byte_unchanged": True,
}

_E44_POLICY = {
    "pipeline": ["consecutive-numbered-identical-body-suffix-trim-v1",
                 "structured-short-partial-repeated-line-trim-v1",
                 "conservative-consecutive-tail-block-trim-v1",
                 "exact-consecutive-long-token-suffix-trim-max512-v1"],
    "minimum_partial_whitespace_tokens": 2,
    "maximum_partial_whitespace_tokens": 7,
    "minimum_prior_complete_copies": 2,
    "minimum_numbered_body_copies": 3,
    "require_consecutive_numbers": True,
    "allow_final_bare_number": True,
    "require_structured_line_marker": True,
    "require_no_terminal_punctuation": True,
    "partial_match": "unicode-nfkc-casefold-collapse-whitespace-exact-prefix",
    "minimum_long_block_tokens": 32, "maximum_long_block_tokens": 512,
    "maximum_fixed_point_passes": 16, "suffix_only": True,
    "ordinary_answers_byte_unchanged": True,
}


def unified_clean(answer: str) -> tuple[str, dict[str, Any]]:
    after_e43, diag43 = e43.clean_answer(answer, SimpleNamespace(raw={"postprocess": _E43_POLICY}))
    after_e44, diag44 = e44.clean_answer(after_e43, SimpleNamespace(raw={"postprocess": _E44_POLICY}))
    if not after_e44.strip() or len(after_e44) > len(answer) or not answer.startswith(after_e44):
        raise PrivateGenerationError("Unified Clean violated monotonic suffix-only removal.")
    return after_e44, {
        "changed": after_e44 != answer,
        "removed_characters": len(answer) - len(after_e44),
        "normal_partial_changed": diag43["partial_changed"],
        "short_partial_changed": diag44["short_partial_changed"],
        "numbered_body_changed": diag44["numbered_body_changed"],
        "line_sentence_changed": diag43["line_sentence_changed"] or diag44["line_sentence_changed"],
        "long_token_changed": diag43["long_token_changed"] or diag44["long_token_changed"],
        "e43": diag43, "e44": diag44,
    }


def _write_submission(directory: Path, ids: list[str], rows: list[dict[str, Any]]) -> None:
    submission = {qid: {"answer": rows[index]["answer"]} for index, qid in enumerate(ids)}
    payload = json.dumps(submission, ensure_ascii=False, indent=2).encode("utf-8") + b"\n"
    _validate_submission_payload(payload, ids)
    _atomic_bytes(directory / "submission.json", payload)
    _write_submission_zip(directory / "submission.zip", "submission.json", payload)
    with zipfile.ZipFile(directory / "submission.zip") as archive:
        if archive.namelist() != ["submission.json"] or archive.read("submission.json") != payload:
            raise PrivateGenerationError("Invalid P01 submission archive.")


def _clean_rows(raw_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    final_rows, review = [], []
    for row in raw_rows:
        answer, post = unified_clean(row["answer"])
        final = {**row, "answer": answer, "source_raw_record_sha256": row["record_sha256"],
                 "postprocess": post}
        final.pop("record_sha256"); final["record_sha256"] = _json_sha256(final)
        final_rows.append(final)
        if post["changed"]:
            review.append({"question_id": row["question_id"], "sample_index": row["sample_index"],
                           "raw_answer": row["answer"], "clean_answer": answer,
                           "removed_characters": post["removed_characters"],
                           "postprocess": post})
    return final_rows, review


def _diagnostics(raw: list[dict[str, Any]], final: list[dict[str, Any]]) -> dict[str, Any]:
    checks = [answer_diagnostics(row["answer"]) for row in final]
    return {
        "length_finish_rate": fmean(row["finish_reason"] == "length" for row in raw),
        "mean_output_tokens": fmean(row["output_tokens"] for row in raw),
        "changed_questions": sum(row["postprocess"]["changed"] for row in final),
        "normal_partial_changed_questions": sum(row["postprocess"]["normal_partial_changed"] for row in final),
        "short_partial_changed_questions": sum(row["postprocess"]["short_partial_changed"] for row in final),
        "numbered_body_changed_questions": sum(row["postprocess"]["numbered_body_changed"] for row in final),
        "line_sentence_changed_questions": sum(row["postprocess"]["line_sentence_changed"] for row in final),
        "long_token_changed_questions": sum(row["postprocess"]["long_token_changed"] for row in final),
        "total_removed_characters": sum(row["postprocess"]["removed_characters"] for row in final),
        "duplicate_line_rate": fmean(value["duplicate_line"] for value in checks),
        "duplicate_sentence_rate": fmean(value["duplicate_sentence"] for value in checks),
    }


def _collect_max1024(*, output: Path, ids: list[str], plan: dict[str, Any]) -> list[dict[str, Any]]:
    replica = output / "generation/max1024/replica-records"
    sharded = output / "generation/max1024/sharded-records"
    short = set(plan["short_indices"])
    rows = []
    for index, qid in enumerate(ids):
        if index in short:
            candidate = replica / f"{index:04d}.json"
            first = json.loads(candidate.read_text(encoding="utf-8"))
            path = sharded / f"{index:04d}.json" if first.get("mode") == "fallback-oom" else candidate
        else:
            path = sharded / f"{index:04d}.json"
        if not path.is_file():
            raise PrivateGenerationError(f"Missing max1024 generation row: {index}")
        row = json.loads(path.read_text(encoding="utf-8"))
        if (row.get("question_id") != qid or row.get("sample_index") != index
                or row.get("variant") != VARIANT_1024
                or row.get("mode") not in {"replica-fp16", "sharded-fp16"}
                or not isinstance(row.get("answer"), str) or not row["answer"].strip()
                or not _valid_hash(row)):
            raise PrivateGenerationError(f"Invalid max1024 generation row: {index}")
        rows.append(row)
    return rows


def finalize1024(*, root: Path, source_p00: Path, training: Path, private: Path,
                 output: Path, config: Config) -> dict[str, Any]:
    checked = check_preflight(root=root, source_p00=source_p00, training=training,
                              private=private, output=output, config=config)
    _, ids, _, _ = validate_p00(source_p00, private, config)
    plan = _load_plan(output, config)
    raw_rows = _collect_max1024(output=output, ids=ids, plan=plan)
    directory = output / "max1024"; directory.mkdir(parents=True, exist_ok=True)
    _atomic_jsonl(directory / "raw-results.jsonl", raw_rows)
    final_rows, review = _clean_rows(raw_rows)
    _atomic_jsonl(directory / "results.jsonl", final_rows)
    _atomic_jsonl(directory / "review-changed.jsonl", review)
    _write_submission(directory, ids, final_rows)
    report = {
        "schema_version": "1.0", "experiment_id": EXPERIMENT + "-max1024",
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "sample_size": len(ids),
        "selected_stack": {"contexts": "exact-P00-parent-contexts",
            "generator": config.section("e38_training")["model_id"],
            "adapter": "E38-v2-rank8-max3328", "precision": "float16",
            "execution": "hybrid-two-replicas-short-one-sharded-long",
            "replica_max_input_tokens": 6000, "max_new_tokens": 1024,
            "postprocess": config.section("unified_clean")["stages"]},
        "diagnostics": {**_diagnostics(raw_rows, final_rows),
            "mode_counts": {mode: sum(row["mode"] == mode for row in raw_rows)
                            for mode in ("replica-fp16", "sharded-fp16")}},
        "files": {name: {"sha256": file_sha256(directory / name),
                          "bytes": (directory / name).stat().st_size}
                  for name in ("raw-results.jsonl", "results.jsonl", "review-changed.jsonl",
                               "submission.json", "submission.zip")},
        "evidence": {**checked, "plan_sha256": plan["plan_sha256"]},
        "private_reference_answers_read": False,
    }
    _atomic_json(directory / "report.json", report)
    return report


def _load_max1024(output: Path, ids: list[str]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    directory = output / "max1024"
    report_path = directory / "report.json"
    raw_path, results_path = directory / "raw-results.jsonl", directory / "results.jsonl"
    if not all(path.is_file() for path in (report_path, raw_path, results_path,
                                            directory / "submission.json", directory / "submission.zip")):
        raise PrivateGenerationError("Run P01 finalize1024 before max1536.")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    raw, final = _read_jsonl(raw_path), _read_jsonl(results_path)
    if len(raw) != len(ids) or len(final) != len(ids):
        raise PrivateGenerationError("P01 max1024 output is incomplete.")
    for index, qid in enumerate(ids):
        if (raw[index].get("question_id") != qid or final[index].get("question_id") != qid
                or not _valid_hash(raw[index]) or not _valid_hash(final[index])
                or final[index].get("source_raw_record_sha256") != raw[index]["record_sha256"]):
            raise PrivateGenerationError(f"P01 max1024 output changed: {index}")
    for name, details in report.get("files", {}).items():
        path = directory / name
        if not path.is_file() or file_sha256(path) != details.get("sha256"):
            raise PrivateGenerationError(f"P01 max1024 file changed: {name}")
    return raw, final, report


def run_length1536(*, root: Path, source_p00: Path, training: Path, private: Path,
                   output: Path, config: Config, model_cache: Path) -> dict[str, Any]:
    checked = check_preflight(root=root, source_p00=source_p00, training=training,
                              private=private, output=output, config=config)
    questions, ids, prepared, _ = validate_p00(source_p00, private, config)
    raw1024, _, report1024 = _load_max1024(output, ids)
    indices = [index for index, row in enumerate(raw1024) if row["finish_reason"] == "length"]
    plan = _load_plan(output, config)
    pin = config.section("e38_training")
    if model_cache.resolve().name != pin["model_revision"]:
        raise PrivateGenerationError("Use the pinned Vi-Qwen revision.")
    e39.checkpoint_parameters(model_cache, config); runtime = _runtime(config)
    records = output / "generation/max1536/records"; records.mkdir(parents=True, exist_ok=True)
    state_path = output / "generation/max1536/worker-state.json"
    base_identity = {
        "code_sha256": code_sha(root), "config_sha256": config.sha,
        "plan_sha256": plan["plan_sha256"], "adapter_sha256": checked["adapter_sha256"],
        "max1024_raw_results_sha256": report1024["files"]["raw-results.jsonl"]["sha256"],
        "length_indices_sha256": _json_sha256(indices),
        "assigned_indices": indices, "mode": "sharded-fp16-restart",
        "max_new_tokens": 1536, "continuation": False,
    }
    if not indices:
        base_identity["identity_sha256"] = _json_sha256(base_identity)
        _progress(records, state_path, base_identity, [], ids)
        return {"regenerated": 0, "reused_nonlength": len(ids)}
    tokenizer = _tokenizer(model_cache)
    model, device_map, input_device = _load_sharded(model_cache, training)
    identity = {**base_identity, "runtime": runtime,
        "generation_config_sha256": _json_sha256(model.generation_config.to_dict()),
        "device_map": device_map, "input_device": str(input_device)}
    identity["identity_sha256"] = _json_sha256(identity)
    done = _progress(records, state_path, identity, indices, ids)
    prompt_lookup = {row["sample_index"]: row for row in plan["prompt_rows"]}
    for completed, index in enumerate(indices[done:], start=done + 1):
        prompt, tokens, spans, packing, count = _prompt(tokenizer, questions[ids[index]], prepared[index], config)
        expected = prompt_lookup[index]; prompt_sha = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        if (tokens != expected["input_tokens"] or prompt_sha != expected["prompt_sha256"]
                or raw1024[index]["prompt_sha256"] != prompt_sha
                or raw1024[index]["finish_reason"] != "length"):
            raise PrivateGenerationError("P01 max1536 did not restart the exact original prompt.")
        answer, output_tokens, generated_tokens, finish, latency = _generate(
            model, tokenizer, prompt, input_device, 1536, count)
        row = _answer_row(
            qid=ids[index], index=index, variant=VARIANT_1536,
            mode="sharded-fp16-restart", identity=identity, answer=answer,
            input_tokens=tokens, output_tokens=output_tokens,
            generated_tokens=generated_tokens, finish=finish, latency=latency,
            spans=spans, packing=packing, prompt_sha=prompt_sha,
            source_record_sha=raw1024[index]["record_sha256"])
        _atomic_json(records / f"{index:04d}.json", row)
        _write_worker_state(state_path, identity, completed, len(indices))
        print(f"P01 sharded restart max1536: {completed}/{len(indices)} qid={ids[index]} finish={finish}", flush=True)
    return {"regenerated": len(indices), "reused_nonlength": len(ids) - len(indices)}


def finalize(*, root: Path, source_p00: Path, training: Path, private: Path,
             output: Path, config: Config) -> dict[str, Any]:
    checked = check_preflight(root=root, source_p00=source_p00, training=training,
                              private=private, output=output, config=config)
    _, ids, _, _ = validate_p00(source_p00, private, config)
    raw1024, final1024, report1024 = _load_max1024(output, ids)
    indices = [index for index, row in enumerate(raw1024) if row["finish_reason"] == "length"]
    selected = set(indices)
    state_path = output / "generation/max1536/worker-state.json"
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.is_file() else {}
    if (state.get("complete") is not True or state.get("completed_count") != len(indices)
            or state.get("assigned_count") != len(indices)):
        raise PrivateGenerationError("P01 max1536 worker is incomplete.")
    candidates = {}
    for index in indices:
        path = output / f"generation/max1536/records/{index:04d}.json"
        row = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
        if (row.get("question_id") != ids[index] or row.get("sample_index") != index
                or row.get("variant") != VARIANT_1536
                or row.get("mode") != "sharded-fp16-restart"
                or row.get("regenerated_from_original_prompt") is not True
                or row.get("source_max1024_record_sha256") != raw1024[index]["record_sha256"]
                or not _valid_hash(row)):
            raise PrivateGenerationError(f"Invalid max1536 regenerated row: {index}")
        candidates[index] = row
    composite = []
    for index, qid in enumerate(ids):
        source = candidates[index] if index in selected else raw1024[index]
        row = {
            "question_id": qid, "sample_index": index, "variant": VARIANT_1536,
            "mode": "regenerated-max1536" if index in selected else "reused-max1024-nonlength",
            "answer": source["answer"], "finish_reason": source["finish_reason"],
            "input_tokens": source["input_tokens"], "output_tokens": source["output_tokens"],
            "generated_tokens_including_special": source["generated_tokens_including_special"],
            "prompt_sha256": source["prompt_sha256"],
            "source_generation_record_sha256": source["record_sha256"],
            "source_max1024_record_sha256": raw1024[index]["record_sha256"],
            "regenerated_from_original_prompt": index in selected,
        }
        row["record_sha256"] = _json_sha256(row); composite.append(row)
    directory = output / "max1536"; directory.mkdir(parents=True, exist_ok=True)
    _atomic_jsonl(directory / "raw-results.jsonl", composite)
    final1536, review = _clean_rows(composite)
    for index in range(len(ids)):
        if index not in selected and final1536[index]["answer"] != final1024[index]["answer"]:
            raise PrivateGenerationError("A reused non-length answer changed between candidates.")
    _atomic_jsonl(directory / "results.jsonl", final1536)
    _atomic_jsonl(directory / "review-changed.jsonl", review)
    _write_submission(directory, ids, final1536)
    prefix_matches = sum(raw1024[index]["answer"] == candidates[index]["answer"][:len(raw1024[index]["answer"])]
                         for index in indices)
    report = {
        "schema_version": "1.0", "experiment_id": EXPERIMENT,
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "sample_size": len(ids),
        "selected_stack": {"contexts": "exact-P00-parent-contexts",
            "generator": config.section("e38_training")["model_id"],
            "adapter": "E38-v2-rank8-max3328", "precision": "float16",
            "source_generation": "hybrid-max1024",
            "selection": "max1024-finish-reason-length-only",
            "length_generation": "restart-original-prompt-sharded-max1536",
            "postprocess": config.section("unified_clean")["stages"]},
        "diagnostics": {**_diagnostics(composite, final1536),
            "reused_nonlength_questions": len(ids) - len(indices),
            "regenerated_length_questions": len(indices),
            "regenerated_prefix_matches_max1024": prefix_matches,
            "regenerated_prefix_differs_max1024": len(indices) - prefix_matches,
            "max1024_length_finish_rate": report1024["diagnostics"]["length_finish_rate"],
            "max1536_regenerated_eos_rate": (fmean(candidates[index]["finish_reason"] == "eos" for index in indices)
                                               if indices else 0.0)},
        "files": {name: {"sha256": file_sha256(directory / name),
                          "bytes": (directory / name).stat().st_size}
                  for name in ("raw-results.jsonl", "results.jsonl", "review-changed.jsonl",
                               "submission.json", "submission.zip")},
        "source_max1024": {"report_sha256": file_sha256(output / "max1024/report.json"),
                           "raw_results_sha256": report1024["files"]["raw-results.jsonl"]["sha256"],
                           "results_sha256": report1024["files"]["results.jsonl"]["sha256"],
                           "submission_zip_sha256": report1024["files"]["submission.zip"]["sha256"]},
        "evidence": checked,
        "private_reference_answers_read": False,
        "automatic_promotion": False,
        "warning": "Private inference only; three-submission budget requires operator review before upload.",
    }
    _atomic_json(directory / "report.json", report)
    shutil.copy2(output / "max1024/submission.zip", output / "submission-max1024.zip")
    shutil.copy2(output / "max1536/submission.zip", output / "submission-max1536-unified-clean.zip")
    _atomic_json(output / "report.json", report)
    return report
