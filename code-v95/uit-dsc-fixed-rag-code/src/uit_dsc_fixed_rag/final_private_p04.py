"""Hybrid two-replica continuation of a saved P03 max1536 run."""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean
from typing import Any

from . import final_private_p01 as p01
from . import final_private_p02 as p02
from . import final_private_p03 as p03
from .corpus import file_sha256
from .e02_answer import _atomic_json, _atomic_jsonl
from .e03_rrf_grid import _json_sha256, _write_worker_state
from .final_public import FinalPublicError


EXPERIMENT = "FINAL-private-p04-qwen-e19-hybrid-length1536-unified-v1"
VARIANT = "p04_qwen35_e19_fp16_hybrid_length_restart_max1536"


class PrivateHybridError(FinalPublicError):
    """Raised when P04 cannot continue without mixing run identities."""


@dataclass(frozen=True)
class Config:
    raw: dict[str, Any]
    path: Path
    base: p03.Config

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
    if (set(raw) != {"schema_version", "experiment_id", "base_p03_config",
                     "source_partial_p03", "execution", "run_contract"}
            or raw.get("schema_version") != "1.0"
            or raw.get("experiment_id") != EXPERIMENT):
        raise PrivateHybridError("P04 config root changed.")
    base_pin = raw["base_p03_config"]
    base_path = root / base_pin.get("path", "")
    if (not base_path.is_file()
            or file_sha256(base_path) != base_pin.get("sha256")):
        raise PrivateHybridError("Pinned P03 base config changed.")
    base = p03.load_config(root, base_path)
    source = raw["source_partial_p03"]
    if (source.get("experiment_id") != p03.EXPERIMENT
            or source.get("config_sha256") != base.sha
            or source.get("assigned_questions") != 455
            or source.get("mode") != "sharded-fp16-restart"
            or source.get("max_new_tokens") != 1536
            or source.get("continuation") is not False):
        raise PrivateHybridError("P04 saved-P03 contract changed.")
    execution = raw["execution"]
    if (execution.get("precision") != "float16"
            or execution.get("replica_workers") != 2
            or execution.get("oom_policy")
            != "checkpoint-marker-then-sharded-fallback"
            or execution.get("cpu_or_disk_offload") is not False
            or execution.get("checkpoint_after_each_question") is not True):
        raise PrivateHybridError("P04 execution policy changed.")
    if not raw["run_contract"] or not all(raw["run_contract"].values()):
        raise PrivateHybridError("P04 run contract lost an invariant.")
    return Config(raw=raw, path=path, base=base)


def code_sha(root: Path) -> str:
    paths = [
        root / "src/uit_dsc_fixed_rag/final_private_p04.py",
        root / "src/uit_dsc_fixed_rag/final_private_p03.py",
        root / "src/uit_dsc_fixed_rag/final_private_p02.py",
        root / "src/uit_dsc_fixed_rag/final_private_p01.py",
        root / "src/uit_dsc_fixed_rag/final_public_e43.py",
        root / "src/uit_dsc_fixed_rag/final_public_e44.py",
        root / "scripts/run_final_private_p04_kaggle.py",
    ]
    return _json_sha256({
        path.relative_to(root).as_posix(): file_sha256(path) for path in paths
    })


def _load_old_plan(partial: Path, config: Config) -> dict[str, Any]:
    path = partial / "plan.json"
    if not path.is_file():
        raise PrivateHybridError("Saved P03 plan is missing.")
    plan = json.loads(path.read_text(encoding="utf-8"))
    pin = config.section("source_partial_p03")
    if (plan.get("experiment_id") != pin["experiment_id"]
            or plan.get("config_sha256") != pin["config_sha256"]
            or plan.get("source_p02_raw_results_sha256")
            != pin["source_p02_raw_results_sha256"]
            or len(plan.get("length_indices", [])) != 455
            or len(plan.get("prompt_rows", [])) != 455
            or plan.get("plan_sha256") != _json_sha256({
                key: value for key, value in plan.items()
                if key != "plan_sha256"
            })):
        raise PrivateHybridError("Saved P03 plan identity changed.")
    return plan


def validate_partial_p03(
    partial: Path, ids: list[str], raw1024: list[dict[str, Any]], config: Config,
) -> tuple[dict[int, dict[str, Any]], dict[str, Any], dict[str, Any]]:
    preflight_path = partial / "preflight.json"
    if not preflight_path.is_file():
        raise PrivateHybridError("Add the saved P03 output, including preflight.json.")
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    pin = config.section("source_partial_p03")
    if (preflight.get("experiment_id") != pin["experiment_id"]
            or preflight.get("code_sha256") != pin["code_sha256"]
            or preflight.get("config_sha256") != pin["config_sha256"]
            or preflight.get("sample_size") != len(ids)
            or preflight.get("source_p02_raw_results_sha256")
            != pin["source_p02_raw_results_sha256"]
            or preflight.get("new_generation_questions") != 455
            or preflight.get("private_reference_answers_read") is not False):
        raise PrivateHybridError("Saved P03 preflight changed.")
    plan = _load_old_plan(partial, config)
    prompt_lookup = {row["sample_index"]: row for row in plan["prompt_rows"]}
    for index in plan["length_indices"]:
        row = prompt_lookup.get(index, {})
        if (row.get("question_id") != ids[index]
                or row.get("prompt_sha256") != raw1024[index]["prompt_sha256"]
                or row.get("input_tokens") != raw1024[index]["input_tokens"]
                or row.get("source_max1024_record_sha256")
                != raw1024[index]["record_sha256"]):
            raise PrivateHybridError(f"Saved P03 prompt row changed: {index}")
    records_dir = partial / "generation/records"
    state_path = partial / "generation/worker-state.json"
    if not state_path.is_file():
        if records_dir.is_dir() and any(records_dir.glob("*.json")):
            raise PrivateHybridError("Saved P03 records have no worker state.")
        return {}, preflight, plan
    state = json.loads(state_path.read_text(encoding="utf-8"))
    identity = state.get("run_identity", {})
    assigned = plan["length_indices"]
    if (identity.get("code_sha256") != pin["code_sha256"]
            or identity.get("config_sha256") != pin["config_sha256"]
            or identity.get("plan_sha256") != plan["plan_sha256"]
            or identity.get("source_p02_raw_results_sha256")
            != pin["source_p02_raw_results_sha256"]
            or identity.get("assigned_indices") != assigned
            or identity.get("mode") != pin["mode"]
            or identity.get("max_new_tokens") != pin["max_new_tokens"]
            or identity.get("continuation") is not pin["continuation"]
            or identity.get("identity_sha256") != _json_sha256({
                key: value for key, value in identity.items()
                if key != "identity_sha256"
            })):
        raise PrivateHybridError("Saved P03 worker identity changed.")
    imported: dict[int, dict[str, Any]] = {}
    gap = False
    for index in assigned:
        path = records_dir / f"{index:04d}.json"
        if not path.is_file():
            gap = True
            continue
        if gap:
            raise PrivateHybridError("Saved P03 records are not a contiguous prefix.")
        row = json.loads(path.read_text(encoding="utf-8"))
        if (row.get("question_id") != ids[index]
                or row.get("sample_index") != index
                or row.get("variant") != p03.VARIANT
                or row.get("mode") != pin["mode"]
                or row.get("worker_identity_sha256")
                != identity["identity_sha256"]
                or row.get("source_max1024_record_sha256")
                != raw1024[index]["record_sha256"]
                or row.get("prompt_sha256") != raw1024[index]["prompt_sha256"]
                or row.get("regenerated_from_original_prompt") is not True
                or not isinstance(row.get("answer"), str)
                or not row["answer"].strip()
                or not p02._valid_hash(row)):
            raise PrivateHybridError(f"Invalid saved P03 record: {index}")
        imported[index] = row
    if int(state.get("completed_count", -1)) > len(imported):
        raise PrivateHybridError("Saved P03 state is ahead of durable records.")
    return imported, preflight, plan


def preflight(
    *, root: Path, source_p00: Path, source_p02: Path, partial_p03: Path,
    training: Path, private: Path, output: Path, config: Config,
) -> dict[str, Any]:
    _, ids, _, p00_report = p01.validate_p00(source_p00, private, config)
    _, adapter_sha, complete = p02._validate_training(root, training, config)
    raw1024, _, _ = p03.validate_source_p02(source_p02, ids, config)
    imported, old_preflight, old_plan = validate_partial_p03(
        partial_p03, ids, raw1024, config)
    imported_manifest = [
        [index, imported[index]["record_sha256"]] for index in sorted(imported)
    ]
    evidence = {
        "experiment_id": EXPERIMENT,
        "code_sha256": code_sha(root), "config_sha256": config.sha,
        "sample_size": len(ids),
        "private_sha256": p00_report["private_questions_sha256"],
        "sample_ids_sha256": p00_report["sample_ids_sha256"],
        "p00_report_sha256": file_sha256(source_p00 / "report.json"),
        "p00_prepared_results_sha256": file_sha256(
            source_p00 / "prepared/results.jsonl"),
        "source_p02_raw_results_sha256": file_sha256(
            source_p02 / "generation/raw-results.jsonl"),
        "source_partial_p03_preflight_sha256": file_sha256(
            partial_p03 / "preflight.json"),
        "source_partial_p03_plan_sha256": old_plan["plan_sha256"],
        "source_partial_p03_code_sha256": old_preflight["code_sha256"],
        "imported_questions": len(imported),
        "imported_manifest_sha256": _json_sha256(imported_manifest),
        "remaining_questions": 455 - len(imported),
        "adapter_sha256": adapter_sha,
        "training_identity_sha256": complete["identity_sha256"],
        "model_id": config.section("e19_training")["model_id"],
        "model_revision": config.section("e19_training")["model_revision"],
        "precision": "float16", "reranker": None,
        "private_reference_answers_read": False,
    }
    path = output / "preflight.json"
    if path.is_file() and json.loads(path.read_text(encoding="utf-8")) != evidence:
        raise PrivateHybridError("Saved P04 preflight belongs to different inputs.")
    if not path.is_file():
        _atomic_json(path, evidence)
    return evidence


def check_preflight(**kwargs: Any) -> dict[str, Any]:
    if not (kwargs["output"] / "preflight.json").is_file():
        raise PrivateHybridError("Run P04 preflight first.")
    return preflight(**kwargs)


def prepare_plan(
    *, root: Path, source_p00: Path, source_p02: Path, partial_p03: Path,
    training: Path, private: Path, output: Path, config: Config,
    model_cache: Path,
) -> dict[str, Any]:
    checked = check_preflight(
        root=root, source_p00=source_p00, source_p02=source_p02,
        partial_p03=partial_p03, training=training, private=private,
        output=output, config=config)
    if model_cache.resolve().name != config.section("e19_training")["model_revision"]:
        raise PrivateHybridError("Use the pinned Qwen3.5-2B revision.")
    p03._runtime(config)
    _, ids, _, _ = p01.validate_p00(source_p00, private, config)
    raw1024, _, _ = p03.validate_source_p02(source_p02, ids, config)
    imported, _, old_plan = validate_partial_p03(
        partial_p03, ids, raw1024, config)
    remaining = [index for index in old_plan["length_indices"] if index not in imported]
    partitions = [remaining[rank::2] for rank in (0, 1)]
    plan = {
        "experiment_id": EXPERIMENT, "config_sha256": config.sha,
        "source_p02_raw_results_sha256": checked[
            "source_p02_raw_results_sha256"],
        "source_partial_p03_plan_sha256": old_plan["plan_sha256"],
        "imported_indices": sorted(imported),
        "imported_manifest_sha256": checked["imported_manifest_sha256"],
        "remaining_indices": remaining, "partitions": partitions,
        "prompt_rows": old_plan["prompt_rows"],
    }
    plan["plan_sha256"] = _json_sha256(plan)
    path = output / "plan.json"
    if path.is_file() and json.loads(path.read_text(encoding="utf-8")) != plan:
        raise PrivateHybridError("Saved P04 plan changed.")
    if not path.is_file():
        _atomic_json(path, plan)
    return {
        "imported": len(imported), "remaining": len(remaining),
        "per_gpu": [len(value) for value in partitions],
        "plan_sha256": plan["plan_sha256"],
    }


def _load_plan(output: Path, config: Config) -> dict[str, Any]:
    path = output / "plan.json"
    if not path.is_file():
        raise PrivateHybridError("Run P04 plan first.")
    plan = json.loads(path.read_text(encoding="utf-8"))
    if (plan.get("experiment_id") != EXPERIMENT
            or plan.get("config_sha256") != config.sha
            or plan.get("plan_sha256") != _json_sha256({
                key: value for key, value in plan.items()
                if key != "plan_sha256"
            })):
        raise PrivateHybridError("P04 plan identity changed.")
    return plan


def run_worker(
    *, root: Path, source_p00: Path, source_p02: Path, partial_p03: Path,
    training: Path, private: Path, output: Path, config: Config,
    model_cache: Path, rank: int, device: str,
) -> dict[str, Any]:
    import torch

    if rank not in (0, 1) or device != f"cuda:{rank}" or torch.cuda.device_count() != 2:
        raise PrivateHybridError("P04 replica workers require matching T4 x2 ranks.")
    checked = check_preflight(
        root=root, source_p00=source_p00, source_p02=source_p02,
        partial_p03=partial_p03, training=training, private=private,
        output=output, config=config)
    plan = _load_plan(output, config)
    assigned = plan["partitions"][rank]
    questions, ids, prepared, _ = p01.validate_p00(source_p00, private, config)
    raw1024, _, _ = p03.validate_source_p02(source_p02, ids, config)
    if model_cache.resolve().name != config.section("e19_training")["model_revision"]:
        raise PrivateHybridError("Use the pinned Qwen3.5-2B revision.")
    runtime = p03._runtime(config)
    torch.cuda.set_device(rank)
    tokenizer = p02._tokenizer(model_cache)
    model, adapter_parameters, base_parameters = p02._load_model(
        model_cache, training, config, device)
    identity = {
        "code_sha256": code_sha(root), "config_sha256": config.sha,
        "plan_sha256": plan["plan_sha256"],
        "adapter_sha256": checked["adapter_sha256"],
        "imported_manifest_sha256": checked["imported_manifest_sha256"],
        "runtime": runtime, "base_parameters": base_parameters,
        "adapter_parameters": adapter_parameters,
        "generation_config_sha256": _json_sha256(
            model.generation_config.to_dict()),
        "mode": "replica-fp16", "rank": rank, "device": device,
        "assigned_indices": assigned, "max_new_tokens": 1536,
        "continuation": False,
    }
    identity["identity_sha256"] = _json_sha256(identity)
    records = output / "generation/replica-records"
    records.mkdir(parents=True, exist_ok=True)
    state = output / f"generation/replica-worker-{rank}-state.json"
    done = p02._progress(records, state, identity, assigned, ids)
    prompt_lookup = {row["sample_index"]: row for row in plan["prompt_rows"]}
    for completed, index in enumerate(assigned[done:], start=done + 1):
        qid = ids[index]
        prompt, tokens, spans, packing, count = p02._prompt(
            tokenizer, questions[qid], prepared[index], config)
        prompt_sha = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        expected = prompt_lookup[index]
        if (tokens != expected["input_tokens"]
                or prompt_sha != expected["prompt_sha256"]
                or raw1024[index]["finish_reason"] != "length"):
            raise PrivateHybridError("P04 prompt differs from the original P02 prompt.")
        try:
            answer, output_tokens, generated_tokens, finish, latency = p03._generate(
                model, tokenizer, prompt, device, count)
            row = {
                "question_id": qid, "sample_index": index,
                "variant": VARIANT, "mode": "replica-fp16",
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
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            row = {
                "question_id": qid, "sample_index": index,
                "variant": VARIANT, "mode": "fallback-oom",
                "worker_identity_sha256": identity["identity_sha256"],
                "prompt_sha256": prompt_sha,
                "source_max1024_record_sha256": raw1024[index]["record_sha256"],
                "regenerated_from_original_prompt": True,
                "error": "torch.cuda.OutOfMemoryError",
            }
        row["record_sha256"] = _json_sha256(row)
        _atomic_json(records / f"{index:04d}.json", row)
        _write_worker_state(state, identity, completed, len(assigned))
        print(
            f"P04 GPU{rank}: {completed}/{len(assigned)} qid={qid} "
            f"mode={row['mode']} finish={row.get('finish_reason')}", flush=True)
    return {"rank": rank, "completed": len(assigned)}


def _replica_rows(
    output: Path, plan: dict[str, Any], ids: list[str], root: Path,
    config: Config, checked: dict[str, Any],
) -> dict[int, dict[str, Any]]:
    records = output / "generation/replica-records"
    rows: dict[int, dict[str, Any]] = {}
    for rank in (0, 1):
        assigned = plan["partitions"][rank]
        state_path = output / f"generation/replica-worker-{rank}-state.json"
        state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.is_file() else {}
        if (state.get("complete") is not True
                or state.get("completed_count") != len(assigned)
                or state.get("assigned_count") != len(assigned)):
            raise PrivateHybridError(f"P04 replica worker {rank} is incomplete.")
        identity = state.get("run_identity", {})
        if (identity.get("code_sha256") != code_sha(root)
                or identity.get("config_sha256") != config.sha
                or identity.get("plan_sha256") != plan["plan_sha256"]
                or identity.get("adapter_sha256") != checked["adapter_sha256"]
                or identity.get("imported_manifest_sha256")
                != checked["imported_manifest_sha256"]
                or identity.get("mode") != "replica-fp16"
                or identity.get("rank") != rank
                or identity.get("device") != f"cuda:{rank}"
                or identity.get("assigned_indices") != assigned
                or identity.get("max_new_tokens") != 1536
                or identity.get("continuation") is not False
                or identity.get("identity_sha256") != _json_sha256({
                key: value for key, value in identity.items()
                if key != "identity_sha256"})):
            raise PrivateHybridError(f"P04 replica worker {rank} identity changed.")
        for index in assigned:
            path = records / f"{index:04d}.json"
            row = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
            if (row.get("question_id") != ids[index]
                    or row.get("sample_index") != index
                    or row.get("variant") != VARIANT
                    or row.get("worker_identity_sha256") != identity["identity_sha256"]
                    or row.get("mode") not in {"replica-fp16", "fallback-oom"}
                    or not p02._valid_hash(row)):
                raise PrivateHybridError(f"Invalid P04 replica row: {index}")
            rows[index] = row
    return rows


def run_fallback(
    *, root: Path, source_p00: Path, source_p02: Path, partial_p03: Path,
    training: Path, private: Path, output: Path, config: Config,
    model_cache: Path,
) -> dict[str, Any]:
    checked = check_preflight(
        root=root, source_p00=source_p00, source_p02=source_p02,
        partial_p03=partial_p03, training=training, private=private,
        output=output, config=config)
    plan = _load_plan(output, config)
    questions, ids, prepared, _ = p01.validate_p00(source_p00, private, config)
    raw1024, _, _ = p03.validate_source_p02(source_p02, ids, config)
    replica = _replica_rows(output, plan, ids, root, config, checked)
    assigned = sorted(index for index, row in replica.items() if row["mode"] == "fallback-oom")
    records = output / "generation/fallback-records"
    records.mkdir(parents=True, exist_ok=True)
    state = output / "generation/fallback-worker-state.json"
    base_identity = {
        "code_sha256": code_sha(root), "config_sha256": config.sha,
        "plan_sha256": plan["plan_sha256"],
        "adapter_sha256": checked["adapter_sha256"],
        "assigned_indices": assigned, "mode": "sharded-fp16-fallback",
        "max_new_tokens": 1536, "continuation": False,
    }
    if not assigned:
        base_identity["identity_sha256"] = _json_sha256(base_identity)
        p02._progress(records, state, base_identity, [], ids)
        return {"fallback_questions": 0}
    tokenizer = p02._tokenizer(model_cache)
    model, base_parameters, adapter_parameters, device_map, input_device = (
        p03._load_sharded(model_cache, training, config))
    identity = {
        **base_identity, "runtime": p03._runtime(config),
        "base_parameters": base_parameters,
        "adapter_parameters": adapter_parameters,
        "generation_config_sha256": _json_sha256(
            model.generation_config.to_dict()),
        "device_map": device_map, "input_device": str(input_device),
    }
    identity["identity_sha256"] = _json_sha256(identity)
    done = p02._progress(records, state, identity, assigned, ids)
    prompt_lookup = {row["sample_index"]: row for row in plan["prompt_rows"]}
    for completed, index in enumerate(assigned[done:], start=done + 1):
        qid = ids[index]
        prompt, tokens, spans, packing, count = p02._prompt(
            tokenizer, questions[qid], prepared[index], config)
        prompt_sha = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        if prompt_sha != prompt_lookup[index]["prompt_sha256"]:
            raise PrivateHybridError("P04 fallback prompt changed.")
        answer, output_tokens, generated_tokens, finish, latency = p03._generate(
            model, tokenizer, prompt, input_device, count)
        row = {
            "question_id": qid, "sample_index": index,
            "variant": VARIANT, "mode": "sharded-fp16-fallback",
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
        _write_worker_state(state, identity, completed, len(assigned))
        print(
            f"P04 sharded fallback: {completed}/{len(assigned)} "
            f"qid={qid} finish={finish}", flush=True)
    return {"fallback_questions": len(assigned)}


def finalize(
    *, root: Path, source_p00: Path, source_p02: Path, partial_p03: Path,
    training: Path, private: Path, output: Path, config: Config,
) -> dict[str, Any]:
    checked = check_preflight(
        root=root, source_p00=source_p00, source_p02=source_p02,
        partial_p03=partial_p03, training=training, private=private,
        output=output, config=config)
    plan = _load_plan(output, config)
    _, ids, _, _ = p01.validate_p00(source_p00, private, config)
    raw1024, final1024, source_report = p03.validate_source_p02(
        source_p02, ids, config)
    imported, _, old_plan = validate_partial_p03(
        partial_p03, ids, raw1024, config)
    replica = _replica_rows(output, plan, ids, root, config, checked)
    fallback_indices = sorted(
        index for index, row in replica.items() if row["mode"] == "fallback-oom")
    fallback_state_path = output / "generation/fallback-worker-state.json"
    fallback_state = json.loads(fallback_state_path.read_text(encoding="utf-8")) if fallback_state_path.is_file() else {}
    if (fallback_state.get("complete") is not True
            or fallback_state.get("completed_count") != len(fallback_indices)
            or fallback_state.get("assigned_count") != len(fallback_indices)):
        raise PrivateHybridError("P04 fallback worker is incomplete.")
    fallback_identity = fallback_state.get("run_identity", {})
    if (fallback_identity.get("code_sha256") != code_sha(root)
            or fallback_identity.get("config_sha256") != config.sha
            or fallback_identity.get("plan_sha256") != plan["plan_sha256"]
            or fallback_identity.get("adapter_sha256") != checked["adapter_sha256"]
            or fallback_identity.get("assigned_indices") != fallback_indices
            or fallback_identity.get("mode") != "sharded-fp16-fallback"
            or fallback_identity.get("max_new_tokens") != 1536
            or fallback_identity.get("continuation") is not False
            or fallback_identity.get("identity_sha256") != _json_sha256({
                key: value for key, value in fallback_identity.items()
                if key != "identity_sha256"
            })):
        raise PrivateHybridError("P04 fallback worker identity changed.")
    candidates = dict(imported)
    for index, row in replica.items():
        if row["mode"] == "fallback-oom":
            path = output / f"generation/fallback-records/{index:04d}.json"
            row = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
            if (row.get("mode") != "sharded-fp16-fallback"
                    or row.get("worker_identity_sha256")
                    != fallback_identity["identity_sha256"]
                    or not p02._valid_hash(row)):
                raise PrivateHybridError(f"Invalid P04 fallback row: {index}")
        if (not isinstance(row.get("answer"), str) or not row["answer"].strip()
                or row.get("prompt_sha256") != raw1024[index]["prompt_sha256"]
                or row.get("source_max1024_record_sha256")
                != raw1024[index]["record_sha256"]):
            raise PrivateHybridError(f"Invalid P04 generated answer: {index}")
        candidates[index] = row
    selected = set(old_plan["length_indices"])
    if set(candidates) != selected:
        raise PrivateHybridError("P04 did not produce every length-selected answer.")
    composite = []
    for index, qid in enumerate(ids):
        source = candidates[index] if index in selected else raw1024[index]
        mode = ("reused-max1024-eos" if index not in selected else (
            "imported-v92-sharded" if index in imported else source["mode"]))
        row = {
            "question_id": qid, "sample_index": index, "variant": VARIANT,
            "mode": mode, "answer": source["answer"],
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
            raise PrivateHybridError("A reused P02 EOS answer changed.")
    generation = output / "generation"
    _atomic_jsonl(generation / "raw-results.jsonl", composite)
    _atomic_jsonl(generation / "results.jsonl", final)
    _atomic_jsonl(output / "review_changed.jsonl", review)
    p02._write_submission(output, ids, final)
    diagnostics = p01._diagnostics(composite, final)
    diagnostics.update({
        "reused_eos_questions": len(ids) - len(selected),
        "regenerated_length_questions": len(selected),
        "imported_v92_sharded_questions": len(imported),
        "new_replica_questions": len(replica) - len(fallback_indices),
        "new_sharded_fallback_questions": len(fallback_indices),
        "candidate_eos_questions": sum(
            row["finish_reason"] == "eos" for row in composite),
        "candidate_length_questions": sum(
            row["finish_reason"] == "length" for row in composite),
    })
    files = {}
    for name in ("generation/raw-results.jsonl", "generation/results.jsonl",
                 "review_changed.jsonl", "submission.json", "submission.zip"):
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
            "adapter": "E19-metadata-aware-rank8", "precision": "float16",
            "source_generation": "P02-two-replica-max1024",
            "selection": "P02-raw-finish-reason-length-only",
            "length_generation": "import-saved-sharded-then-two-replicas-with-sharded-oom-fallback",
            "postprocess": config.section("unified_clean")["stages"],
        },
        "diagnostics": diagnostics,
        "validation": {
            "all_question_ids_present": True, "all_answers_non_empty": True,
            "utf8_without_bom": True, "archive_members": ["submission.json"],
            "official_mapping_schema": {"question_id": {"answer": "string"}},
        },
        "files": files,
        "source_p02": {
            "report_sha256": file_sha256(source_p02 / "report.json"),
            "raw_results_sha256": file_sha256(
                source_p02 / "generation/raw-results.jsonl"),
            "source_diagnostics": source_report["diagnostics"],
        },
        "evidence": {**checked, "plan_sha256": plan["plan_sha256"]},
        "new_retrieval_performed": False, "reranker_performed": False,
        "private_reference_answers_read": False, "automatic_promotion": False,
        "warning": "Hybrid private candidate; preserve P02 max1024 as rollback.",
    }
    _atomic_json(output / "report.json", report)
    shutil.copy2(source_p02 / "submission.zip", output / "source-p02-max1024.zip")
    shutil.copy2(output / "submission.zip", output / "submission-p04-hybrid-max1536.zip")
    return report


__all__ = [
    "Config", "EXPERIMENT", "PrivateHybridError", "VARIANT", "check_preflight",
    "code_sha", "finalize", "load_config", "preflight", "prepare_plan",
    "run_fallback", "run_worker", "validate_partial_p03",
]
