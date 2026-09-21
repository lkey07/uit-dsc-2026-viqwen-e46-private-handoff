"""Private E33-lineage Qwen/E19 max1024 generation with Unified Clean."""

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

from . import final_private_p01 as p01
from . import e19_metadata_lora as e19
from . import final_public_e23 as e23
from .corpus import file_sha256
from .e02_answer import _atomic_json, _atomic_jsonl
from .e03_rrf_grid import _json_sha256, _write_worker_state
from .e10_repetition_grid import answer_diagnostics
from .final_public import (
    FinalPublicError,
    _atomic_bytes,
    _validate_submission_payload,
    _write_submission_zip,
)


EXPERIMENT = "FINAL-private-p02-qwen-e19-max1024-unified-v1"
VARIANT = "p02_qwen35_e19_fp16_max1024_unified_clean"


class PrivateQwenError(FinalPublicError):
    """Raised when P02 cannot continue without mixing run identities."""


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
            raise PrivateQwenError(f"Missing P02 config section: {key}")
        return value


def load_config(path: Path) -> Config:
    raw = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "schema_version", "experiment_id", "source_p00", "e19_training",
        "prompt", "context_policy", "inference", "execution", "runtime",
        "unified_clean", "parameter_budget", "submission_contract", "run_contract",
    }
    if set(raw) != required or raw.get("schema_version") != "1.0" or raw.get("experiment_id") != EXPERIMENT:
        raise PrivateQwenError("P02 config root changed.")
    source = raw["source_p00"]
    if (source.get("experiment_id") != "FINAL-private-p00-retrieval-parent-v1"
            or source.get("selected_contexts") != 12
            or source.get("fused_top_k") != 20
            or source.get("answers_used") is not False):
        raise PrivateQwenError("P02/P00 retrieval contract changed.")
    training = raw["e19_training"]
    if (training.get("experiment_id") != "E19-metadata-aware-lora-train5636-eval200-v1"
            or training.get("model_id") != "Qwen/Qwen3.5-2B"
            or training.get("model_revision") != "15852e8c16360a2fea060d615a32b45270f8a8fc"
            or training.get("accepted_runtime_parameter_counts") != [1_881_825_088, 2_213_241_664]
            or training.get("adapter_sha256") != "bfb9d8120337c4013b5dcb0b59c6d03beccfe9f9748e5a01ef62e1de64fb5edd"
            or training.get("adapter_parameters") != 8_409_600
            or training.get("fresh_from_pinned_base") is not True):
        raise PrivateQwenError("P02 E19 generator contract changed.")
    inference = raw["inference"]
    if inference != {
        "max_input_tokens": 8192, "max_new_tokens": 1024,
        "do_sample": False, "num_beams": 1, "repetition_penalty": 1.0,
        "no_repeat_ngram_size": 0, "enable_thinking": False, "use_cache": True,
    }:
        raise PrivateQwenError("P02 deterministic generation policy changed.")
    execution = raw["execution"]
    if (execution.get("precision") != "float16" or execution.get("workers") != 2
            or execution.get("partition") != "sample-index-mod-worker-count"
            or execution.get("questions_per_worker") != 959
            or execution.get("cpu_or_disk_offload") is not False
            or execution.get("checkpoint_after_each_question") is not True):
        raise PrivateQwenError("P02 execution policy changed.")
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
        raise PrivateQwenError("P02 Unified Clean policy changed.")
    budget = raw["parameter_budget"]
    if (budget.get("maximum_stack_total")
            != budget.get("embedding") + budget.get("generator") + budget.get("adapter_parameter_cap")
            or budget["maximum_stack_total"] >= budget["exclusive_limit"]):
        raise PrivateQwenError("P02 stack exceeds the BTC parameter limit.")
    if not raw["run_contract"] or not all(value is True for value in raw["run_contract"].values()):
        raise PrivateQwenError("P02 run contract lost an invariant.")
    return Config(raw=raw, path=path)


def code_sha(root: Path) -> str:
    paths = [
        root / "src/uit_dsc_fixed_rag/final_private_p02.py",
        root / "src/uit_dsc_fixed_rag/final_private_p01.py",
        root / "src/uit_dsc_fixed_rag/final_public_e43.py",
        root / "src/uit_dsc_fixed_rag/final_public_e44.py",
        root / "scripts/run_final_private_p02_kaggle.py",
    ]
    return _json_sha256({path.relative_to(root).as_posix(): file_sha256(path) for path in paths})


def _runtime(config: Config) -> dict[str, str]:
    observed = {name: importlib.metadata.version(name) for name in config.section("runtime")}
    if observed != config.section("runtime"):
        raise PrivateQwenError(f"Use the exact P02 runtime: {observed}")
    return observed


def _validate_training(root: Path, training: Path, config: Config) -> tuple[Any, str, dict[str, Any]]:
    pin = config.section("e19_training")
    config_path = root / pin["training_config_path"]
    if not config_path.is_file() or file_sha256(config_path) != pin["training_config_sha256"]:
        raise PrivateQwenError("Pinned E19 training config changed.")
    training_config = e19.load_config(root, config_path)
    adapter_sha, complete = e19.validate_candidate_adapter(training / "training", training_config)
    if (adapter_sha != pin["adapter_sha256"]
            or complete.get("code_sha256") != pin["training_code_sha256"]
            or complete.get("identity_sha256") != pin["training_identity_sha256"]
            or complete.get("training_records_sha256") != pin["training_records_sha256"]
            or complete.get("raw_source_contexts_sha256") != pin["raw_source_contexts_sha256"]
            or complete.get("metadata_policy") != pin["metadata_policy"]
            or complete.get("trainable_parameters") != pin["adapter_parameters"]):
        raise PrivateQwenError("Saved E19 adapter evidence changed.")
    if (training_config.generator_model_id != pin["model_id"]
            or training_config.generator_revision != pin["model_revision"]
            or training_config.section("prompt") != config.section("prompt")):
        raise PrivateQwenError("P02 no longer matches the E19/E33 prompt or model.")
    return training_config, adapter_sha, complete


def preflight(*, root: Path, source_p00: Path, training: Path, private: Path,
              output: Path, config: Config) -> dict[str, Any]:
    _, ids, _, p00_report = p01.validate_p00(source_p00, private, config)
    _, adapter_sha, complete = _validate_training(root, training, config)
    evidence = {
        "experiment_id": EXPERIMENT,
        "code_sha256": code_sha(root),
        "config_sha256": config.sha,
        "sample_size": len(ids),
        "private_sha256": p00_report["private_questions_sha256"],
        "sample_ids_sha256": p00_report["sample_ids_sha256"],
        "p00_report_sha256": file_sha256(source_p00 / "report.json"),
        "p00_prepared_results_sha256": file_sha256(source_p00 / "prepared/results.jsonl"),
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
        raise PrivateQwenError("Saved P02 preflight belongs to different inputs.")
    if not path.is_file():
        _atomic_json(path, evidence)
    return evidence


def check_preflight(*, root: Path, source_p00: Path, training: Path, private: Path,
                    output: Path, config: Config) -> dict[str, Any]:
    if not (output / "preflight.json").is_file():
        raise PrivateQwenError("Run P02 preflight first.")
    return preflight(root=root, source_p00=source_p00, training=training,
                     private=private, output=output, config=config)


def _tokenizer(model_cache: Path):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(model_cache), trust_remote_code=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer


def _prompt(tokenizer: Any, question: str, prepared: dict[str, Any], config: Config):
    def render(messages: list[dict[str, Any]]) -> str:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False)

    def count(text: str) -> int:
        return len(tokenizer(text, add_special_tokens=False)["input_ids"])

    messages, spans, packing = e23.parent.pack(
        question, prepared, p01._ContextConfig(config),
        lambda value: count(render(value)), count, e23.parent.VARIANTS[1])
    prompt = render(messages)
    tokens = count(prompt)
    if (tokens > 8192 or packing["seed_ranks"] != list(range(12))
            or packing["skipped_seed_ranks"]):
        raise PrivateQwenError("P02 could not preserve all P00 top-12 seeds.")
    return prompt, tokens, spans, packing, count


def prepare_plan(*, root: Path, source_p00: Path, training: Path, private: Path,
                 output: Path, config: Config, model_cache: Path) -> dict[str, Any]:
    checked = check_preflight(root=root, source_p00=source_p00, training=training,
                              private=private, output=output, config=config)
    pin = config.section("e19_training")
    if model_cache.resolve().name != pin["model_revision"]:
        raise PrivateQwenError("Use the pinned Qwen3.5-2B revision.")
    _runtime(config)
    questions, ids, prepared, _ = p01.validate_p00(source_p00, private, config)
    tokenizer = _tokenizer(model_cache)
    rows = []
    for index, qid in enumerate(ids):
        prompt, tokens, _, _, _ = _prompt(tokenizer, questions[qid], prepared[index], config)
        rows.append({
            "sample_index": index, "question_id": qid, "input_tokens": tokens,
            "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        })
    partitions = [list(range(rank, len(ids), 2)) for rank in (0, 1)]
    if [len(value) for value in partitions] != [959, 959]:
        raise PrivateQwenError("P02 private partition size changed.")
    plan = {
        "experiment_id": EXPERIMENT, "config_sha256": config.sha,
        "p00_prepared_results_sha256": checked["p00_prepared_results_sha256"],
        "private_sha256": checked["private_sha256"],
        "partitions": partitions, "prompt_rows": rows,
    }
    plan["plan_sha256"] = _json_sha256(plan)
    path = output / "plan.json"
    if path.is_file() and json.loads(path.read_text(encoding="utf-8")) != plan:
        raise PrivateQwenError("Saved P02 execution plan changed.")
    _atomic_json(path, plan)
    return {
        "sample_size": len(ids), "partitions": [len(value) for value in partitions],
        "minimum_input_tokens": min(row["input_tokens"] for row in rows),
        "maximum_input_tokens": max(row["input_tokens"] for row in rows),
        "plan_sha256": plan["plan_sha256"],
    }


def _load_plan(output: Path, config: Config) -> dict[str, Any]:
    path = output / "plan.json"
    if not path.is_file():
        raise PrivateQwenError("Run P02 plan first.")
    plan = json.loads(path.read_text(encoding="utf-8"))
    if (plan.get("experiment_id") != EXPERIMENT or plan.get("config_sha256") != config.sha
            or plan.get("plan_sha256") != _json_sha256(
                {key: value for key, value in plan.items() if key != "plan_sha256"})):
        raise PrivateQwenError("P02 plan identity changed.")
    return plan


def _valid_hash(row: dict[str, Any]) -> bool:
    return row.get("record_sha256") == _json_sha256(
        {key: value for key, value in row.items() if key != "record_sha256"})


def _progress(records: Path, state: Path, identity: dict[str, Any], assigned: list[int],
              ids: list[str]) -> int:
    if not state.is_file():
        if any((records / f"{index:04d}.json").is_file() for index in assigned):
            raise PrivateQwenError("P02 records exist without their checkpoint state.")
        _write_worker_state(state, identity, 0, len(assigned))
        return 0
    saved = json.loads(state.read_text(encoding="utf-8"))
    if saved.get("run_identity") != identity:
        raise PrivateQwenError("P02 checkpoint belongs to a different run.")
    completed, gap = 0, False
    for index in assigned:
        path = records / f"{index:04d}.json"
        if not path.is_file():
            gap = True
            continue
        if gap:
            raise PrivateQwenError("P02 records are non-contiguous in their partition.")
        row = json.loads(path.read_text(encoding="utf-8"))
        if (row.get("sample_index") != index or row.get("question_id") != ids[index]
                or not _valid_hash(row)):
            raise PrivateQwenError("P02 durable record changed.")
        completed += 1
    if int(saved.get("completed_count", -1)) > completed:
        raise PrivateQwenError("P02 checkpoint is ahead of durable records.")
    _write_worker_state(state, identity, completed, len(assigned))
    return completed


def _load_model(model_cache: Path, training: Path, config: Config, device: str):
    import torch
    from peft import PeftModel
    from transformers import Qwen3_5ForCausalLM

    base = Qwen3_5ForCausalLM.from_pretrained(
        str(model_cache), dtype=torch.float16, device_map={"": device},
        low_cpu_mem_usage=True, trust_remote_code=False)
    base_parameters = sum(parameter.numel() for parameter in base.parameters())
    if base_parameters not in config.section("e19_training")["accepted_runtime_parameter_counts"]:
        raise PrivateQwenError(f"Unexpected Qwen3.5-2B parameter count: {base_parameters}")
    model = PeftModel.from_pretrained(
        base, training / "training/adapter-final", is_trainable=False)
    model.eval()
    adapter_parameters = sum(
        parameter.numel() for name, parameter in model.named_parameters() if "lora_" in name)
    if adapter_parameters != config.section("e19_training")["adapter_parameters"]:
        raise PrivateQwenError("E19 adapter parameter count changed.")
    if any(str(parameter.device) != device for parameter in model.parameters()):
        raise PrivateQwenError("P02 replica is not wholly resident on its assigned GPU.")
    return model, adapter_parameters, base_parameters


def _generate(model: Any, tokenizer: Any, prompt: str, device: str, count: Any):
    import torch
    tensors = {key: value.to(device) for key, value in tokenizer(
        prompt, add_special_tokens=False, return_tensors="pt").items()}
    started = time.perf_counter()
    with torch.inference_mode():
        generated = model.generate(
            **tensors, do_sample=False, num_beams=1, repetition_penalty=1.0,
            no_repeat_ngram_size=0, max_new_tokens=1024, use_cache=True)
    latency = (time.perf_counter() - started) * 1000
    new_ids = generated[0, tensors["input_ids"].shape[1]:]
    answer = tokenizer.decode(
        new_ids.detach().cpu(), skip_special_tokens=True,
        clean_up_tokenization_spaces=False).strip()
    if not answer:
        raise PrivateQwenError("Qwen/E19 returned an empty answer.")
    eos = model.generation_config.eos_token_id
    eos_ids = set(eos if isinstance(eos, list) else [eos])
    finish = "eos" if len(new_ids) and int(new_ids[-1]) in eos_ids else (
        "length" if len(new_ids) >= 1024 else "other")
    return answer, count(answer), len(new_ids), finish, latency


def run_worker(*, root: Path, source_p00: Path, training: Path, private: Path,
               output: Path, config: Config, model_cache: Path,
               rank: int, device: str) -> dict[str, Any]:
    import torch
    if rank not in (0, 1) or device != f"cuda:{rank}" or torch.cuda.device_count() != 2:
        raise PrivateQwenError("P02 workers require matching T4 x2 ranks.")
    checked = check_preflight(root=root, source_p00=source_p00, training=training,
                              private=private, output=output, config=config)
    plan = _load_plan(output, config)
    assigned = plan["partitions"][rank]
    questions, ids, prepared, _ = p01.validate_p00(source_p00, private, config)
    pin = config.section("e19_training")
    if model_cache.resolve().name != pin["model_revision"]:
        raise PrivateQwenError("Use the pinned Qwen3.5-2B revision.")
    runtime = _runtime(config)
    torch.cuda.set_device(rank)
    tokenizer = _tokenizer(model_cache)
    model, adapter_parameters, base_parameters = _load_model(
        model_cache, training, config, device)
    generation_sha = _json_sha256(model.generation_config.to_dict())
    identity = {
        "code_sha256": code_sha(root), "config_sha256": config.sha,
        "plan_sha256": plan["plan_sha256"], "adapter_sha256": checked["adapter_sha256"],
        "adapter_parameters": adapter_parameters, "runtime": runtime,
        "base_parameters": base_parameters,
        "generation_config_sha256": generation_sha,
        "mode": "replica-fp16", "rank": rank, "device": device,
        "assigned_indices": assigned, "max_new_tokens": 1024,
    }
    identity["identity_sha256"] = _json_sha256(identity)
    records = output / "generation/records"
    records.mkdir(parents=True, exist_ok=True)
    state = output / f"generation/worker-{rank}-state.json"
    done = _progress(records, state, identity, assigned, ids)
    prompt_lookup = {row["sample_index"]: row for row in plan["prompt_rows"]}
    for completed, index in enumerate(assigned[done:], start=done + 1):
        prompt, tokens, spans, packing, count = _prompt(
            tokenizer, questions[ids[index]], prepared[index], config)
        expected = prompt_lookup[index]
        prompt_sha = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        if tokens != expected["input_tokens"] or prompt_sha != expected["prompt_sha256"]:
            raise PrivateQwenError("P02 prompt changed after planning.")
        answer, output_tokens, generated_tokens, finish, latency = _generate(
            model, tokenizer, prompt, device, count)
        row = {
            "question_id": ids[index], "sample_index": index,
            "worker_rank": rank, "variant": VARIANT, "mode": "replica-fp16",
            "worker_identity_sha256": identity["identity_sha256"],
            "answer": answer, "input_tokens": tokens, "output_tokens": output_tokens,
            "generated_tokens_including_special": generated_tokens,
            "finish_reason": finish, "generation_latency_ms": latency,
            "selected_context_count": len(spans), "packing": packing,
            "prompt_sha256": prompt_sha,
        }
        row["record_sha256"] = _json_sha256(row)
        _atomic_json(records / f"{index:04d}.json", row)
        _write_worker_state(state, identity, completed, len(assigned))
        print(
            f"P02 GPU{rank}: {completed}/{len(assigned)} qid={ids[index]} finish={finish}",
            flush=True)
    return {"rank": rank, "completed": len(assigned), "device": device}


def _write_submission(output: Path, ids: list[str], rows: list[dict[str, Any]]) -> None:
    submission = {qid: {"answer": rows[index]["answer"]} for index, qid in enumerate(ids)}
    payload = json.dumps(submission, ensure_ascii=False, indent=2).encode("utf-8") + b"\n"
    if payload.startswith(b"\xef\xbb\xbf"):
        raise PrivateQwenError("P02 submission unexpectedly contains a UTF-8 BOM.")
    _validate_submission_payload(payload, ids)
    _atomic_bytes(output / "submission.json", payload)
    _write_submission_zip(output / "submission.zip", "submission.json", payload)
    with zipfile.ZipFile(output / "submission.zip") as archive:
        if archive.namelist() != ["submission.json"] or archive.read("submission.json") != payload:
            raise PrivateQwenError("Invalid P02 submission archive.")


def finalize(*, root: Path, source_p00: Path, training: Path, private: Path,
             output: Path, config: Config) -> dict[str, Any]:
    checked = check_preflight(root=root, source_p00=source_p00, training=training,
                              private=private, output=output, config=config)
    plan = _load_plan(output, config)
    _, ids, _, p00_report = p01.validate_p00(source_p00, private, config)
    records = output / "generation/records"
    raw_rows: list[dict[str, Any]] = []
    worker_identities = []
    for rank in (0, 1):
        state_path = output / f"generation/worker-{rank}-state.json"
        if not state_path.is_file():
            raise PrivateQwenError(f"Missing P02 worker state: {rank}")
        state = json.loads(state_path.read_text(encoding="utf-8"))
        identity = state.get("run_identity", {})
        assigned = plan["partitions"][rank]
        if (state.get("complete") is not True
                or state.get("completed_count") != len(assigned)
                or state.get("assigned_count") != len(assigned)
                or identity.get("rank") != rank
                or identity.get("device") != f"cuda:{rank}"
                or identity.get("config_sha256") != config.sha
                or identity.get("code_sha256") != code_sha(root)
                or identity.get("plan_sha256") != plan["plan_sha256"]
                or identity.get("adapter_sha256") != checked["adapter_sha256"]
                or identity.get("base_parameters") not in config.section("e19_training")["accepted_runtime_parameter_counts"]
                or identity.get("assigned_indices") != assigned
                or identity.get("max_new_tokens") != 1024
                or identity.get("identity_sha256") != _json_sha256(
                    {key: value for key, value in identity.items() if key != "identity_sha256"})):
            raise PrivateQwenError("Incomplete or incompatible P02 worker state.")
        worker_identities.append(identity["identity_sha256"])
    prompt_lookup = {row["sample_index"]: row for row in plan["prompt_rows"]}
    for index, qid in enumerate(ids):
        path = records / f"{index:04d}.json"
        if not path.is_file():
            raise PrivateQwenError(f"Missing P02 generation row: {index}")
        row = json.loads(path.read_text(encoding="utf-8"))
        if (row.get("question_id") != qid or row.get("sample_index") != index
                or row.get("worker_rank") != index % 2 or row.get("variant") != VARIANT
                or row.get("prompt_sha256") != prompt_lookup[index]["prompt_sha256"]
                or row.get("input_tokens") != prompt_lookup[index]["input_tokens"]
                or not isinstance(row.get("selected_context_count"), int)
                or row.get("selected_context_count") < 1
                or row.get("packing", {}).get("seed_ranks") != list(range(12))
                or bool(row.get("packing", {}).get("skipped_seed_ranks"))
                or not isinstance(row.get("answer"), str) or not row["answer"].strip()
                or not _valid_hash(row)):
            raise PrivateQwenError(f"Invalid P02 generation row: {index}")
        raw_rows.append(row)
    _atomic_jsonl(output / "generation/raw-results.jsonl", raw_rows)
    final_rows, review = p01._clean_rows(raw_rows)
    _atomic_jsonl(output / "generation/results.jsonl", final_rows)
    _atomic_jsonl(output / "review_changed.jsonl", review)
    _write_submission(output, ids, final_rows)
    checks = [answer_diagnostics(row["answer"]) for row in final_rows]
    diagnostics = p01._diagnostics(raw_rows, final_rows)
    diagnostics.update({
        "eos_questions": sum(row["finish_reason"] == "eos" for row in raw_rows),
        "length_questions": sum(row["finish_reason"] == "length" for row in raw_rows),
        "other_finish_questions": sum(row["finish_reason"] == "other" for row in raw_rows),
        "minimum_input_tokens": min(row["input_tokens"] for row in raw_rows),
        "maximum_input_tokens": max(row["input_tokens"] for row in raw_rows),
        "mean_input_tokens": fmean(row["input_tokens"] for row in raw_rows),
        "mean_generation_latency_ms": fmean(row["generation_latency_ms"] for row in raw_rows),
        "duplicate_line_rate": fmean(value["duplicate_line"] for value in checks),
        "duplicate_sentence_rate": fmean(value["duplicate_sentence"] for value in checks),
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
            "precision": "float16", "execution": "two-independent-replicas",
            "max_new_tokens": 1024, "decoding": "greedy",
            "postprocess": config.section("unified_clean")["stages"],
        },
        "diagnostics": diagnostics,
        "validation": {
            "all_question_ids_present": True, "all_answers_non_empty": True,
            "utf8_without_bom": True, "archive_members": ["submission.json"],
            "official_mapping_schema": {"question_id": {"answer": "string"}},
        },
        "files": files,
        "evidence": {
            **checked, "plan_sha256": plan["plan_sha256"],
            "p00_raw_results_sha256": p00_report["files"]["retrieval/raw-results.jsonl"]["sha256"],
            "worker_identity_sha256": worker_identities,
        },
        "new_retrieval_performed": False,
        "reranker_performed": False,
        "private_reference_answers_read": False,
        "automatic_promotion": False,
        "public_lineage_meteor": 0.5620,
        "warning": "Third private candidate from the E33/Qwen lineage; compare only through an authorized submission.",
    }
    _atomic_json(output / "report.json", report)
    return report


__all__ = [
    "Config", "EXPERIMENT", "PrivateQwenError", "VARIANT", "check_preflight",
    "finalize", "load_config", "preflight", "prepare_plan", "run_worker",
]
