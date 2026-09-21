"""E38 Vi-Qwen public trial on the exact saved E33 parent-context artifact."""
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
from .corpus import file_sha256
from .e02_answer import _atomic_json, _atomic_jsonl
from .e03_rrf_grid import (_json_sha256, _load_worker_progress,
                           _validate_worker_placement, _write_worker_state)
from .e07_generator_ab import _as_text_chat_messages
from .e10_repetition_grid import answer_diagnostics
from .e13_e08b_max768_tailtrim import trim_repeated_tail
from .e31_long_token_suffix_trim import trim_long_token_suffix
from .final_public import (FinalPublicError, _atomic_bytes,
                           _validate_submission_payload, _write_submission_zip,
                           load_public_questions)

EXPERIMENT = "FINAL-public1000-e39-viqwen-e38-max1024-two-trims-v1"
VARIANT = "e39_viqwen_e38_parent_max1024_two_trims"


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
                     "inference", "runtime", "public_trial"}
            or raw["schema_version"] != "1.0" or raw["experiment_id"] != EXPERIMENT
            or raw["source_e33_config_path"] != "configs/final-public-e33-parent-max1024-two-trims-v1.json"
            or raw["source_e33_config_sha256"] != "1e7524279df82606c5bf7a4d5d74e1d8a2cc06123e2430d560e4d02eec4b9bea"):
        raise FinalPublicError("E39 reviewed contract changed.")
    source_path = root / raw["source_e33_config_path"]
    if file_sha256(source_path) != raw["source_e33_config_sha256"]:
        raise FinalPublicError("Pinned E33 configuration changed.")
    if raw["inference"] != {
        "max_input_tokens": 8192, "max_new_tokens": 1024, "do_sample": False,
        "num_beams": 1, "repetition_penalty": 1.0, "no_repeat_ngram_size": 0,
        "use_cache": True, "postprocess_first": "conservative-consecutive-tail-block-trim-v1",
        "postprocess_second": "exact-consecutive-long-token-suffix-trim-v1",
        "minimum_block_tokens": 32, "maximum_block_tokens": 256,
    } or raw["runtime"] != {"torch": "2.10.0+cu128", "transformers": "5.16.1",
                              "peft": "0.19.1", "accelerate": "1.13.0"}:
        raise FinalPublicError("E39 inference/runtime contract changed.")
    if raw["public_trial"] != {"sample_size": 1000, "workers": 2,
        "questions_per_worker": 500, "reuse_exact_e33_prepared_contexts": True,
        "public_answers_never_read": True, "rollback_public_meteor": 0.562,
        "automatic_promotion": False}:
        raise FinalPublicError("E39 public-trial contract changed.")
    return Config(raw, path, e33.load_config(root, source_path))


def code_sha(root: Path) -> str:
    paths = sorted((root / "src/uit_dsc_fixed_rag").rglob("*.py"))
    paths.append(root / "scripts/run_final_public_e39_kaggle.py")
    return _json_sha256({p.relative_to(root).as_posix(): file_sha256(p) for p in paths})


def validate_training(training: Path, config: Config) -> tuple[str, dict[str, Any]]:
    complete_path = training / "adapter-final/complete.json"
    adapter = training / "adapter-final/adapter_model.safetensors"
    if not complete_path.is_file() or not adapter.is_file():
        raise FinalPublicError("Add complete E38 Notebook 1 v2 output.")
    complete = json.loads(complete_path.read_text(encoding="utf-8"))
    pin = config.raw["e38_training"]
    adapter_sha = file_sha256(adapter)
    if (complete.get("experiment_id") != pin["experiment_id"]
            or complete.get("config_sha256") != pin["config_sha256"]
            or complete.get("model_id") != pin["model_id"]
            or complete.get("model_revision") != pin["model_revision"]
            or complete.get("model_parameters") != pin["model_parameters"]
            or complete.get("training_records_sha256") != pin["training_records_sha256"]
            or complete.get("training_sample_ids_sha256") != pin["training_sample_ids_sha256"]
            or complete.get("packing", {}).get("maximum_sequence_tokens") > pin["maximum_sequence_tokens"]
            or complete.get("packing", {}).get("answer_truncation_count") != 0
            or complete.get("fresh_from_pinned_checkpoint") is not True
            or complete.get("e19_adapter_loaded") is not False
            or complete.get("adapter_sha256") != adapter_sha
            or not 0 < complete.get("trainable_parameters", 0) <= 50_000_000
            or len(str(complete.get("code_sha256", ""))) != 64):
        raise FinalPublicError("E38 v2 training identity changed.")
    if pin["model_parameters"] + 567_754_752 + complete["trainable_parameters"] >= 4_000_000_000:
        raise FinalPublicError("E39 model stack exceeds four billion parameters.")
    return adapter_sha, complete


def validate_e33_source(source: Path, public: Path, config: Config) -> tuple[list[dict[str, Any]], list[str]]:
    report_path = source / "report.json"
    if not report_path.is_file():
        raise FinalPublicError("Add complete E33 output including report and retrieval contexts.")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    pin = config.raw["source_e33"]
    if (report.get("experiment_id") != pin["experiment_id"] or report.get("sample_size") != 1000
            or report.get("public_answers_read") is not False or report.get("holdout_untouched") is not True
            or report.get("evidence", {}).get("prepared_results_sha256") != pin["prepared_results_sha256"]
            or report.get("evidence", {}).get("public_sha256") != pin["public_sha256"]
            or report.get("evidence", {}).get("sample_ids_sha256") != pin["sample_ids_sha256"]):
        raise FinalPublicError("E33 source report changed.")
    questions, ids = load_public_questions(public, config.e33.public)
    prepared = e23.load_prepared(source, ids, config.e33.source)
    if file_sha256(source / "retrieval/results.jsonl") != pin["prepared_results_sha256"]:
        raise FinalPublicError("E33 prepared contexts changed.")
    return prepared, ids


def checkpoint_parameters(model_cache: Path, config: Config) -> int:
    """Count immutable checkpoint tensors before 4-bit materialization."""
    from safetensors import safe_open
    model_config = json.loads((model_cache / "config.json").read_text(encoding="utf-8"))
    if "Qwen2ForCausalLM" not in model_config.get("architectures", []):
        raise FinalPublicError("Vi-Qwen architecture changed.")
    single = model_cache / "model.safetensors"
    shards = [single] if single.is_file() else sorted(model_cache.glob("model-*-of-*.safetensors"))
    if not shards:
        raise FinalPublicError("Pinned Vi-Qwen checkpoint tensors are missing.")
    names, total = set(), 0
    for path in shards:
        with safe_open(str(path), framework="pt", device="cpu") as reader:
            for name in reader.keys():
                if name in names:
                    raise FinalPublicError("Duplicate checkpoint tensor across shards.")
                names.add(name)
                size = 1
                for dimension in reader.get_slice(name).get_shape():
                    size *= dimension
                total += size
    if total != config.raw["e38_training"]["model_parameters"]:
        raise FinalPublicError(f"Vi-Qwen checkpoint parameter count changed: {total}")
    return total


def preflight(*, root: Path, source: Path, training: Path, public: Path,
              output: Path, config: Config) -> dict[str, Any]:
    prepared, ids = validate_e33_source(source, public, config)
    adapter_sha, complete = validate_training(training, config)
    evidence = {"experiment_id": EXPERIMENT, "code_sha256": code_sha(root),
                "config_sha256": config.sha, "sample_size": len(ids),
                "sample_ids_sha256": config.raw["source_e33"]["sample_ids_sha256"],
                "prepared_results_sha256": file_sha256(source / "retrieval/results.jsonl"),
                "adapter_sha256": adapter_sha, "training_identity_sha256": complete["identity_sha256"],
                "model_id": config.raw["e38_training"]["model_id"],
                "model_revision": config.raw["e38_training"]["model_revision"],
                "prepared_questions": len(prepared), "public_reference_answers_read": False}
    _atomic_json(output / "preflight.json", evidence)
    return evidence


def check_preflight(root: Path, source: Path, training: Path, public: Path,
                    output: Path, config: Config) -> dict[str, Any]:
    saved = json.loads((output / "preflight.json").read_text(encoding="utf-8"))
    _, ids = validate_e33_source(source, public, config)
    adapter_sha, complete = validate_training(training, config)
    if (saved.get("experiment_id") != EXPERIMENT or saved.get("code_sha256") != code_sha(root)
            or saved.get("config_sha256") != config.sha or saved.get("sample_size") != len(ids)
            or saved.get("prepared_results_sha256") != file_sha256(source / "retrieval/results.jsonl")
            or saved.get("adapter_sha256") != adapter_sha
            or saved.get("training_identity_sha256") != complete["identity_sha256"]):
        raise FinalPublicError("E39 inputs changed after preflight.")
    return saved


def _validate_record(row: dict[str, Any], qid: str, index: int, rank: int, identity: dict[str, Any]) -> None:
    if (row.get("question_id") != qid or row.get("sample_index") != index
            or row.get("worker_rank") != rank or row.get("variant") != VARIANT
            or row.get("worker_identity_sha256") != identity["identity_sha256"]
            or not isinstance(row.get("answer"), str) or not row["answer"].strip()
            or row.get("record_sha256") != _json_sha256({k: v for k, v in row.items() if k != "record_sha256"})):
        raise FinalPublicError(f"Invalid E39 record: {rank}/{index}")


def run_worker(*, root: Path, source: Path, training: Path, public: Path,
               output: Path, config: Config, model_cache: Path, rank: int, device: str) -> dict[str, Any]:
    import torch
    from peft import PeftModel
    from transformers import AutoTokenizer, BitsAndBytesConfig, Qwen2ForCausalLM
    if rank not in (0, 1) or device != f"cuda:{rank}" or torch.cuda.device_count() != 2:
        raise FinalPublicError("E39 requires T4 x2 with matching worker/GPU ranks.")
    checked = check_preflight(root, source, training, public, output, config)
    prepared, ids = validate_e33_source(source, public, config)
    questions, _ = load_public_questions(public, config.e33.public)
    pin = config.raw["e38_training"]
    if model_cache.resolve().name != pin["model_revision"]:
        raise FinalPublicError("Use the pinned Vi-Qwen checkpoint revision.")
    checkpoint_parameters(model_cache, config)
    runtime = {name: importlib.metadata.version(name) for name in config.raw["runtime"]}
    if runtime != config.raw["runtime"]:
        raise FinalPublicError(f"Use the exact E38 runtime: {runtime}")
    torch.cuda.set_device(rank)
    tokenizer = AutoTokenizer.from_pretrained(str(model_cache), trust_remote_code=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    quantization = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16,
    )
    base = Qwen2ForCausalLM.from_pretrained(
        str(model_cache), quantization_config=quantization, dtype=torch.float16,
        device_map={"": device}, low_cpu_mem_usage=True, trust_remote_code=False,
    )
    model = PeftModel.from_pretrained(base, training / "adapter-final", is_trainable=False)
    model.eval()
    placement = _validate_worker_placement(model, device)
    generation_sha = _json_sha256(model.generation_config.to_dict())
    indices = e23.assigned(rank)
    identity = {"code_sha256": code_sha(root), "config_sha256": config.sha,
        "prepared_sha256": checked["prepared_results_sha256"], "adapter_sha256": checked["adapter_sha256"],
        "runtime": runtime, "generation_config_sha256": generation_sha, "worker_rank": rank,
        "device": device, "device_map": placement, "assigned_indices": indices,
        "variant": VARIANT, "max_new_tokens": 1024,
        "inference_quantization": "nf4-double-quant-float16"}
    identity["identity_sha256"] = _json_sha256(identity)
    records = output / "generation/records"
    state = output / f"generation/worker-{rank}-state.json"
    records.mkdir(parents=True, exist_ok=True)
    done = _load_worker_progress(records=records, state_path=state, identity=identity,
                                 assigned_indices=indices, sample_ids=ids)
    for index in indices[:done]:
        _validate_record(json.loads((records / f"{index:04d}.json").read_text(encoding="utf-8")),
                         ids[index], index, rank, identity)

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
            raise FinalPublicError(f"E39 cannot preserve all E33 top-12 seeds: {ids[index]}")
        tensors = {k: v.to(device) for k, v in tokenizer(
            prompt, add_special_tokens=False, return_tensors="pt").items()}
        started = time.perf_counter()
        with torch.inference_mode():
            generated = model.generate(**tensors, do_sample=False, num_beams=1,
                repetition_penalty=1.0, no_repeat_ngram_size=0, max_new_tokens=1024, use_cache=True)
        latency = (time.perf_counter() - started) * 1000
        new_ids = generated[0, tensors["input_ids"].shape[1]:]
        answer = tokenizer.decode(new_ids, skip_special_tokens=True,
                                  clean_up_tokenization_spaces=False).strip()
        if not answer:
            raise FinalPublicError(f"Empty E39 answer: {ids[index]}")
        eos = model.generation_config.eos_token_id
        eos_ids = set(eos if isinstance(eos, list) else [eos])
        finish = "eos" if len(new_ids) and int(new_ids[-1]) in eos_ids else (
            "length" if len(new_ids) >= 1024 else "other")
        row = {"question_id": ids[index], "sample_index": index, "worker_rank": rank,
               "variant": VARIANT, "worker_identity_sha256": identity["identity_sha256"],
               "answer": answer, "input_tokens": input_tokens,
               "output_tokens": count(answer), "generated_tokens_including_special": len(new_ids),
               "finish_reason": finish, "generation_latency_ms": latency,
               "selected_context_count": len(spans), "packing": packing,
               "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest()}
        row["record_sha256"] = _json_sha256(row)
        _atomic_json(records / f"{index:04d}.json", row)
        _write_worker_state(state, identity, completed, 500)
        print(f"E39 GPU{rank}: {completed}/500 qid={ids[index]} finish={finish}", flush=True)
    return {"worker_rank": rank, "completed": 500}


def _trim(answer: str) -> tuple[str, dict[str, Any]]:
    first = trim_repeated_tail(answer)
    second = trim_long_token_suffix(first["answer"], minimum_block_tokens=32, maximum_block_tokens=256)
    if not second["answer"].strip():
        raise FinalPublicError("E39 trims removed a complete answer.")
    return second["answer"], {"line_sentence_changed": first["changed"],
        "long_token_changed": second["changed"],
        "removed_characters": len(answer) - len(second["answer"])}


def finalize(*, root: Path, source: Path, training: Path, public: Path,
             output: Path, config: Config) -> dict[str, Any]:
    checked = check_preflight(root, source, training, public, output, config)
    _, ids = validate_e33_source(source, public, config)
    rows: list[dict[str, Any] | None] = [None] * 1000
    identities = []
    for rank in (0, 1):
        state = json.loads((output / f"generation/worker-{rank}-state.json").read_text(encoding="utf-8"))
        identity = state.get("run_identity", {})
        indices = e23.assigned(rank)
        if (state.get("complete") is not True or state.get("completed_count") != 500
                or identity.get("assigned_indices") != indices or identity.get("worker_rank") != rank
                or identity.get("code_sha256") != code_sha(root) or identity.get("config_sha256") != config.sha
                or identity.get("adapter_sha256") != checked["adapter_sha256"]
                or identity.get("identity_sha256") != _json_sha256({k: v for k, v in identity.items() if k != "identity_sha256"})):
            raise FinalPublicError("Incomplete or changed E39 worker state.")
        identities.append(identity["identity_sha256"])
        for index in indices:
            row = json.loads((output / f"generation/records/{index:04d}.json").read_text(encoding="utf-8"))
            _validate_record(row, ids[index], index, rank, identity)
            rows[index] = row
    if any(row is None for row in rows):
        raise FinalPublicError("Missing E39 public answer.")
    raw_rows = rows
    _atomic_jsonl(output / "generation/raw-results.jsonl", raw_rows)
    final_rows, submission = [], {}
    for row in raw_rows:
        answer, post = _trim(row["answer"])
        final = {**row, "answer": answer, "source_record_sha256": row["record_sha256"], "postprocess": post}
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
            raise FinalPublicError("Invalid E39 submission archive.")
    checks = [answer_diagnostics(row["answer"]) for row in final_rows]
    report = {"schema_version": "1.0", "experiment_id": EXPERIMENT,
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "sample_size": 1000,
        "selected_stack": {"retrieval_contexts": "exact-saved-E33-parent-contexts",
            "generator": config.raw["e38_training"]["model_id"], "adapter": "E38-v2-rank8-max3328",
            "max_new_tokens": 1024, "decoding": "greedy",
            "postprocess": [config.raw["inference"]["postprocess_first"],
                            config.raw["inference"]["postprocess_second"]]},
        "diagnostics": {"length_finish_rate": fmean(r["finish_reason"] == "length" for r in raw_rows),
            "mean_output_tokens": fmean(r["output_tokens"] for r in raw_rows),
            "line_sentence_changed_questions": sum(r["postprocess"]["line_sentence_changed"] for r in final_rows),
            "long_token_changed_questions": sum(r["postprocess"]["long_token_changed"] for r in final_rows),
            "duplicate_line_rate": fmean(x["duplicate_line"] for x in checks),
            "duplicate_sentence_rate": fmean(x["duplicate_sentence"] for x in checks)},
        "files": {name: {"sha256": file_sha256(output / name), "bytes": (output / name).stat().st_size}
                  for name in ("generation/raw-results.jsonl", "generation/results.jsonl", "submission.json", "submission.zip")},
        "evidence": {**checked, "worker_identity_sha256": identities},
        "public_reference_answers_read": False, "private_untouched": True,
        "automatic_promotion": False, "rollback_public_meteor": 0.562,
        "warning": "Direct operator-authorized public trial without E38 dev selection; keep E33 as rollback."}
    _atomic_json(output / "report.json", report)
    return report
