"""One-GPU E46 private inference with E45 selection and Unified Clean."""

from __future__ import annotations

import hashlib
import importlib.metadata
import itertools
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .corpus import file_sha256
from .e02_answer import _atomic_json, _atomic_jsonl, make_messages
from .e03_rrf_grid import _json_sha256, _read_jsonl
from .e08b_context_lora import inference_packing
from .e18_source_metadata import enrich_context, scan_selected
from .e45_top12v2_fulltrain import select_top12_v2
from .e46_viqwen_top12v2_fulltrain import load_config as load_e46_config, render_viqwen
from .final_private_p00 import load_private_questions
from .final_private_p01 import _clean_rows, _diagnostics, _generate, _valid_hash, _write_submission


EXPERIMENT = "FINAL-private-p05-e46-top12v2-max1536-unified-v1"


class PrivateE46Error(RuntimeError):
    pass


@dataclass(frozen=True)
class Config:
    raw: dict[str, Any]
    path: Path

    @property
    def sha(self) -> str:
        return file_sha256(self.path)


def load_config(path: Path) -> Config:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if raw != {
        "schema_version": "1.0",
        "experiment_id": EXPERIMENT,
        "source_e46_config_path": "configs/e46-viqwen-top12v2-fulltrain7000-v1.json",
        "source_p00_experiment": "FINAL-private-p00-retrieval-parent-v1",
        "sample_size": 1918,
        "inference": {"device": "cuda:0", "precision": "float16", "max_input_tokens": 8192,
                      "max_new_tokens": 1536, "do_sample": False, "num_beams": 1,
                      "repetition_penalty": 1.0, "no_repeat_ngram_size": 0},
        "selector": "rrf-top6-then-legal-priority-v2",
        "postprocess": "E43-then-E44-unified-suffix-only",
        "checkpoint_after_questions": 1,
        "private_reference_answers_read": False,
        "reranker": None,
    }:
        raise PrivateE46Error("P05 inference contract changed.")
    return Config(raw, path)


def _code_sha(root: Path) -> str:
    paths = [root / "src/uit_dsc_fixed_rag/final_private_p05_e46.py",
             root / "scripts/run_final_private_p05_e46_kaggle.py"]
    return _json_sha256({path.relative_to(root).as_posix(): file_sha256(path) for path in paths})


def _verify_sources(*, root: Path, p00: Path, e00: Path, e45: Path, e46: Path,
                    private: Path, config: Config) -> tuple[dict[str, str], list[str], dict[str, Any], Any]:
    questions, ids, identity = load_private_questions(private)
    if len(ids) != config.raw["sample_size"]:
        raise PrivateE46Error("Private question count changed.")
    ec = load_e46_config(root, root / config.raw["source_e46_config_path"])
    e45_report = json.loads((e45 / "report.json").read_text(encoding="utf-8"))
    e45_summary = json.loads((e45 / "training-data/summary.json").read_text(encoding="utf-8"))
    e45_records = e45 / "training-data/records.jsonl"
    e46_complete = json.loads((e46 / "adapter-final/complete.json").read_text(encoding="utf-8"))
    e46_identity = json.loads((e46 / "training-identity.json").read_text(encoding="utf-8"))
    adapter = e46 / "adapter-final/adapter_model.safetensors"
    if (e45_report.get("experiment_id") != "E45-top12v2-fulltrain7000-v1"
            or e45_report.get("sample_size") != 7000
            or e45_report.get("answers_used_by_retrieval_or_selector") is not False
            or e45_report.get("private_read") is not False
            or e45_summary.get("records_sha256") != e46_complete.get("training_records_sha256")
            or file_sha256(e45_records) != e45_summary.get("records_sha256")
            or e45_summary.get("selector") != config.raw["selector"]
            or e46_complete.get("experiment_id") != "E46-viqwen-top12v2-fulltrain7000-v1"
            or e46_complete.get("identity_sha256") != e46_identity.get("identity_sha256")
            or e46_identity.get("e45_report_sha256") != file_sha256(e45 / "report.json")
            or e46_complete.get("config_sha256") != ec.sha
            or e46_complete.get("model_id") != ec.raw["model"]["id"]
            or e46_complete.get("model_revision") != ec.raw["model"]["revision"]
            or e46_complete.get("model_parameters") != ec.raw["model"]["checkpoint_parameters"]
            or e46_complete.get("training_records") != 7000
            or e46_complete.get("selector") != config.raw["selector"]
            or e46_complete.get("fresh_from_pinned_checkpoint") is not True
            or e46_complete.get("official_answer_truncation_count") != 0
            or e46_complete.get("e38_adapter_loaded") is not False
            or e46_complete.get("e19_adapter_loaded") is not False
            or not adapter.is_file() or file_sha256(adapter) != e46_complete.get("adapter_sha256")):
        raise PrivateE46Error("E45/E46 artifact identity changed or E46 is incomplete.")
    p00_report = json.loads((p00 / "report.json").read_text(encoding="utf-8"))
    pool_path = p00 / "retrieval/candidate-pool-top20.jsonl"
    pool_pin = p00_report.get("files", {}).get("retrieval/candidate-pool-top20.jsonl", {})
    if (p00_report.get("experiment_id") != config.raw["source_p00_experiment"]
            or p00_report.get("sample_size") != len(ids)
            or p00_report.get("private_questions_sha256") != identity["private_sha256"]
            or p00_report.get("sample_ids_sha256") != identity["sample_ids_sha256"]
            or p00_report.get("answers_used") is not False
            or p00_report.get("private_reference_answers_read") is not False
            or p00_report.get("retrieval", {}).get("fused_top_k") != 20
            or p00_report.get("retrieval", {}).get("reranker") is not None
            or file_sha256(pool_path) != pool_pin.get("sha256")
            or pool_path.stat().st_size != pool_pin.get("bytes")):
        raise PrivateE46Error("P00/private identity changed.")
    metadata = ec.e45.section("metadata_source")
    p00_config_path = root / "configs/final-private-p00-retrieval-parent-v1.json"
    p00_config = json.loads(p00_config_path.read_text(encoding="utf-8"))
    if (file_sha256(p00_config_path) != p00_report.get("evidence", {}).get("config_sha256")
            or p00_config["metadata_source"] != metadata):
        raise PrivateE46Error("E45/P00 metadata source differs.")
    for key in ("chunks_path", "documents_path", "chunks_sha256", "documents_sha256"):
        if metadata[key] != p00_config["metadata_source"][key]:
            raise PrivateE46Error("E45/P00 metadata pin mismatch.")
    if (file_sha256(e00 / metadata["chunks_path"]) != metadata["chunks_sha256"]
            or file_sha256(e00 / metadata["documents_path"]) != metadata["documents_sha256"]):
        raise PrivateE46Error("E00 metadata files changed.")
    pins = {
        "experiment_id": EXPERIMENT, "code_sha256": _code_sha(root), "config_sha256": config.sha,
        "sample_size": len(ids), "private_sha256": identity["private_sha256"],
        "sample_ids_sha256": identity["sample_ids_sha256"],
        "question_text_sha256": identity["question_text_sha256"],
        "p00_report_sha256": file_sha256(p00 / "report.json"),
        "p00_pool_sha256": file_sha256(pool_path),
        "e00_chunks_sha256": metadata["chunks_sha256"],
        "e00_documents_sha256": metadata["documents_sha256"],
        "e45_report_sha256": file_sha256(e45 / "report.json"),
        "e46_adapter_sha256": e46_complete["adapter_sha256"],
        "e46_training_identity_sha256": e46_complete["identity_sha256"],
        "model_id": ec.raw["model"]["id"], "model_revision": ec.raw["model"]["revision"],
        "parameter_stack_maximum": sum(ec.raw["model"][key] for key in
                                       ("checkpoint_parameters", "embedding_parameters", "adapter_parameter_cap")),
        "private_reference_answers_read": False, "reranker": None,
    }
    if pins["parameter_stack_maximum"] >= 4_000_000_000:
        raise PrivateE46Error("BTC parameter budget exceeded.")
    return questions, ids, pins, ec


def _audit_e45_selector(*, e45: Path, e00: Path, output: Path, ec: Any) -> dict[str, Any]:
    """Prove the current selector reproduces all saved E45 training context IDs."""
    report = json.loads((e45 / "report.json").read_text(encoding="utf-8"))
    raw_path = e45 / "retrieval/raw-results.jsonl"
    train_path = e45 / "training-data/records.jsonl"
    if (file_sha256(raw_path) != report.get("files", {}).get("retrieval/raw-results.jsonl", {}).get("sha256")
            or file_sha256(train_path) != report.get("files", {}).get("training-data/records.jsonl", {}).get("sha256")):
        raise PrivateE46Error("E45 audit source files changed.")
    expected = {"raw_sha256": file_sha256(raw_path), "training_sha256": file_sha256(train_path),
                "chunks_sha256": ec.e45.section("metadata_source")["chunks_sha256"],
                "documents_sha256": ec.e45.section("metadata_source")["documents_sha256"],
                "audited_rows": 7000, "selector_matches_saved_e45": True}
    path = output / "selector-audit.json"
    if path.is_file():
        if json.loads(path.read_text(encoding="utf-8")) != expected:
            raise PrivateE46Error("Saved E45 selector audit changed.")
        return expected
    wanted: set[str] = set()
    with raw_path.open("r", encoding="utf-8") as stream:
        for line in stream:
            raw = json.loads(line)
            wanted.update(item["chunk_id"] for item in raw["fused_pool"])
    metadata = ec.e45.section("metadata_source")
    chunks = scan_selected(e00 / metadata["chunks_path"], "chunk_id", wanted,
                           metadata["chunks_sha256"])
    documents = scan_selected(e00 / metadata["documents_path"], "document_id",
                              {chunk["document_id"] for chunk in chunks.values()},
                              metadata["documents_sha256"])
    count = 0
    with raw_path.open("r", encoding="utf-8") as raw_stream, train_path.open("r", encoding="utf-8") as train_stream:
        for raw_line, train_line in itertools.zip_longest(raw_stream, train_stream):
            if raw_line is None or train_line is None:
                raise PrivateE46Error("E45 raw/training row count differs.")
            raw, trained = json.loads(raw_line), json.loads(train_line)
            if (raw.get("question_id") != trained.get("question_id")
                    or raw.get("sample_index") != trained.get("sample_index") or raw.get("sample_index") != count
                    or len(raw.get("candidate_contexts", [])) != 20):
                raise PrivateE46Error(f"E45 audit row identity changed: {count}")
            candidates = []
            for rank, source in enumerate(raw["candidate_contexts"]):
                fused = raw["fused_pool"][rank]
                if source["chunk_id"] != fused["chunk_id"]:
                    raise PrivateE46Error(f"E45 candidate rank changed: {count}:{rank}")
                chunk = chunks[source["chunk_id"]]
                context, evidence = enrich_context(dict(source), chunk, documents[chunk["document_id"]])
                candidates.append({"context": context, "body": source["text"], "evidence": evidence,
                                   "rrf_rank": rank, "rrf_score": fused["rrf_score"]})
            selected, _ = select_top12_v2(trained["question"], candidates, ec.e45.section("selector"),
                                           allow_content_relaxed_fill=True)
            if [item["context"]["chunk_id"] for item in selected] != trained.get("selected_chunk_ids"):
                raise PrivateE46Error(f"Current selector differs from saved E45 at training row {count}.")
            count += 1
    if count != 7000:
        raise PrivateE46Error(f"E45 selector audit expected 7000 rows, got {count}.")
    _atomic_json(path, expected)
    return expected


def preflight(*, root: Path, p00: Path, e00: Path, e45: Path, e46: Path,
              private: Path, output: Path, config: Config) -> dict[str, Any]:
    _, _, pins, _ = _verify_sources(root=root, p00=p00, e00=e00, e45=e45, e46=e46,
                                   private=private, config=config)
    path = output / "preflight.json"
    if path.is_file() and json.loads(path.read_text(encoding="utf-8")) != pins:
        raise PrivateE46Error("Saved P05 run belongs to different input artifacts.")
    _atomic_json(path, pins)
    return pins


def _checked(**kwargs: Any) -> tuple[dict[str, str], list[str], dict[str, Any], Any]:
    output = kwargs["output"]
    path = output / "preflight.json"
    if not path.is_file():
        raise PrivateE46Error("Run preflight first.")
    result = _verify_sources(**{key: value for key, value in kwargs.items() if key != "output"})
    if json.loads(path.read_text(encoding="utf-8")) != result[2]:
        raise PrivateE46Error("Preflight identity changed.")
    return result


def prepare(*, root: Path, p00: Path, e00: Path, e45: Path, e46: Path,
            private: Path, output: Path, config: Config) -> dict[str, Any]:
    shared = dict(root=root, p00=p00, e00=e00, e45=e45, e46=e46,
                  private=private, output=output, config=config)
    questions, ids, pins, ec = _checked(**shared)
    _audit_e45_selector(e45=e45, e00=e00, output=output, ec=ec)
    pool = _read_jsonl(p00 / "retrieval/candidate-pool-top20.jsonl")
    if len(pool) != len(ids):
        raise PrivateE46Error("P00 top20 pool has wrong row count.")
    wanted: set[str] = set()
    for index, (qid, row) in enumerate(zip(ids, pool)):
        fused = row.get("fused_pool", [])
        chunk_ids = [item.get("chunk_id") for item in fused]
        if (row.get("question_id") != qid or row.get("sample_index") != index
                or row.get("answers_used") is not False or len(fused) != 20
                or len(set(chunk_ids)) != 20 or any(not isinstance(item.get("rrf_score"), (int, float)) for item in fused)):
            raise PrivateE46Error(f"Invalid P00 top20 row: {index}")
        wanted.update(chunk_ids)
    metadata = ec.e45.section("metadata_source")
    chunks = scan_selected(e00 / metadata["chunks_path"], "chunk_id", wanted,
                           metadata["chunks_sha256"])
    documents = scan_selected(e00 / metadata["documents_path"], "document_id",
                              {chunk["document_id"] for chunk in chunks.values()},
                              metadata["documents_sha256"])
    rows = []
    for index, (qid, row) in enumerate(zip(ids, pool)):
        candidates = []
        for rank, fused in enumerate(row["fused_pool"]):
            chunk = chunks[fused["chunk_id"]]
            context = {key: chunk[key] for key in ("chunk_id", "document_id", "article_number", "text")}
            enriched, evidence = enrich_context(context, chunk, documents[chunk["document_id"]])
            candidates.append({"context": enriched, "body": context["text"],
                               "evidence": evidence, "rrf_rank": rank,
                               "rrf_score": fused["rrf_score"]})
        selected, diagnostic = select_top12_v2(
            questions[qid], candidates, ec.e45.section("selector"), allow_content_relaxed_fill=True)
        rows.append({"question_id": qid, "sample_index": index, "question": questions[qid],
                     "contexts": [item["context"] for item in selected],
                     "selected_rrf_ranks": [item["rrf_rank"] for item in selected],
                     "selection_phases": [item["selection_phase"] for item in selected],
                     "selection_diagnostic": diagnostic, "answers_used": False})
    plan_path = output / "plan.jsonl"
    if plan_path.is_file():
        if _read_jsonl(plan_path) != rows:
            raise PrivateE46Error("Saved P05 selector plan changed.")
    else:
        _atomic_jsonl(plan_path, rows)
    plan = {"experiment_id": EXPERIMENT, "preflight_sha256": _json_sha256(pins),
            "plan_sha256": file_sha256(plan_path), "questions": len(rows),
            "selector_audit_sha256": file_sha256(output / "selector-audit.json"),
            "changed_from_rrf12": sum(row["selected_rrf_ranks"] != list(range(12)) for row in rows),
            "answers_used_by_selector": False}
    if (output / "plan.json").is_file() and json.loads((output / "plan.json").read_text(encoding="utf-8")) != plan:
        raise PrivateE46Error("Saved P05 plan identity changed.")
    _atomic_json(output / "plan.json", plan)
    return plan


def _plan(output: Path, ids: list[str], pins: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    path = output / "plan.jsonl"
    plan = json.loads((output / "plan.json").read_text(encoding="utf-8"))
    if (plan.get("experiment_id") != EXPERIMENT
            or plan.get("preflight_sha256") != _json_sha256(pins)
            or plan.get("plan_sha256") != file_sha256(path)
            or plan.get("selector_audit_sha256") != file_sha256(output / "selector-audit.json")
            or plan.get("questions") != len(ids)
            or plan.get("answers_used_by_selector") is not False):
        raise PrivateE46Error("P05 plan identity changed.")
    audit = json.loads((output / "selector-audit.json").read_text(encoding="utf-8"))
    if audit.get("selector_matches_saved_e45") is not True or audit.get("audited_rows") != 7000:
        raise PrivateE46Error("P05 E45 selector audit is incomplete.")
    rows = _read_jsonl(path)
    if len(rows) != len(ids) or any(row.get("question_id") != qid or row.get("sample_index") != i
                                    or row.get("answers_used") is not False
                                    or len(row.get("contexts", [])) != 12
                                    for i, (qid, row) in enumerate(zip(ids, rows))):
        raise PrivateE46Error("P05 plan row order/schema changed.")
    return rows, plan


def _prompt(tokenizer: Any, row: dict[str, Any], ec: Any, max_input: int) -> tuple[str, int, int, int]:
    packing = inference_packing(ec.e19)
    selected: list[dict[str, Any]] = []
    prompt = ""
    token_count = 0

    def render(contexts: list[dict[str, Any]]) -> tuple[str, int]:
        value = render_viqwen(tokenizer, make_messages(question=row["question"],
                                                     contexts=contexts, config=packing))
        return value, len(tokenizer(value, add_special_tokens=False)["input_ids"])

    for context in row["contexts"]:
        trial, count = render([*selected, context])
        if count <= max_input:
            selected.append(context)
            prompt, token_count = trial, count
    trimmed = 0
    if not selected:
        first = row["contexts"][0]
        body = first["text"].strip()
        low = ec.e19.section("train")["minimum_truncated_context_characters"]
        high = len(body)
        best = None
        while low <= high:
            mid = (low + high) // 2
            trial_context = {**first, "text": body[:mid].rstrip()}
            trial, count = render([trial_context])
            if count <= max_input:
                best = (trial, count, len(body) - mid)
                low = mid + 1
            else:
                high = mid - 1
        if best is None:
            raise PrivateE46Error(f"No context fits the input cap: {row['question_id']}")
        prompt, token_count, trimmed = best
        selected = [first]
    return prompt, token_count, len(selected), trimmed


def _runtime(ec: Any) -> None:
    expected = ec.raw["runtime"]
    observed = {name: importlib.metadata.version(name) for name in expected}
    if observed != expected:
        raise PrivateE46Error(f"E46 inference runtime mismatch: {observed}")


def _load_model(model_cache: Path, e46: Path, ec: Any):
    import torch
    from peft import PeftModel
    from peft.utils import save_and_load
    from transformers import AutoTokenizer, Qwen2ForCausalLM

    if model_cache.resolve().name != ec.raw["model"]["revision"]:
        raise PrivateE46Error("Model cache is not the pinned Vi-Qwen revision.")
    if not torch.cuda.is_available():
        raise PrivateE46Error("P05 requires one visible CUDA GPU.")
    device = torch.device("cuda:0")
    tokenizer = AutoTokenizer.from_pretrained(str(model_cache), trust_remote_code=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = Qwen2ForCausalLM.from_pretrained(str(model_cache), dtype=torch.float16,
                                            device_map={"": "cuda:0"}, low_cpu_mem_usage=True,
                                            trust_remote_code=False)
    observed = sum(parameter.numel() for parameter in base.parameters())
    if observed != ec.raw["model"]["checkpoint_parameters"]:
        raise PrivateE46Error(f"Unexpected Vi-Qwen parameter count: {observed}")

    # PEFT 0.19.1 imports a removed Transformers TP symbol on every adapter load.
    # This run has one device and no tensor-parallel plan, so only that no-op path is bypassed.
    original = save_and_load._maybe_shard_state_dict_for_tp

    def no_tensor_parallel(model: Any, state: dict[str, Any], adapter_name: str) -> None:
        if any(getattr(module, "_hf_tp_plan", None) is not None
               or getattr(module, "_hf_device_mesh", None) is not None
               for module in model.modules()):
            raise PrivateE46Error("Tensor-parallel adapter encountered in one-GPU inference.")

    save_and_load._maybe_shard_state_dict_for_tp = no_tensor_parallel
    try:
        model = PeftModel.from_pretrained(base, str(e46 / "adapter-final"), is_trainable=False)
    finally:
        save_and_load._maybe_shard_state_dict_for_tp = original
    model.eval()
    return model, tokenizer, device


def generate(*, root: Path, p00: Path, e00: Path, e45: Path, e46: Path,
             private: Path, output: Path, config: Config, model_cache: Path) -> dict[str, Any]:
    shared = dict(root=root, p00=p00, e00=e00, e45=e45, e46=e46,
                  private=private, output=output, config=config)
    _, ids, pins, ec = _checked(**shared)
    rows, plan = _plan(output, ids, pins)
    _runtime(ec)
    records = output / "generation/records"
    records.mkdir(parents=True, exist_ok=True)
    state_path = output / "generation/state.json"
    run_identity = _json_sha256({"preflight": pins, "plan_sha256": plan["plan_sha256"]})
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.is_file() else None
    if state is not None and state.get("run_identity_sha256") != run_identity:
        raise PrivateE46Error("Saved generation belongs to another run.")
    completed, gap = 0, False
    for index, qid in enumerate(ids):
        path = records / f"{index:04d}.json"
        if not path.is_file():
            gap = True
            continue
        if gap:
            raise PrivateE46Error("Generation records are non-contiguous.")
        record = json.loads(path.read_text(encoding="utf-8"))
        if (record.get("question_id") != qid or record.get("sample_index") != index
                or record.get("run_identity_sha256") != run_identity
                or record.get("max_new_tokens") != 1536 or not _valid_hash(record)):
            raise PrivateE46Error(f"Generation record changed: {index}")
        completed += 1
    if state is None and completed:
        raise PrivateE46Error("Generation records exist without checkpoint state.")
    if state is not None and int(state.get("completed_count", -1)) > completed:
        raise PrivateE46Error("Checkpoint state is ahead of durable records.")
    _atomic_json(state_path, {"run_identity_sha256": run_identity,
                              "completed_count": completed, "total": len(ids),
                              "complete": completed == len(ids)})
    if completed == len(ids):
        return {"completed": completed, "total": len(ids), "resumed": True}
    model, tokenizer, device = _load_model(model_cache, e46, ec)
    from tqdm.auto import tqdm
    for index in tqdm(range(completed, len(ids)), desc="P05 one-GPU E46 max1536"):
        prompt, tokens, count_contexts, trimmed = _prompt(
            tokenizer, rows[index], ec, config.raw["inference"]["max_input_tokens"])
        answer, output_tokens, generated_tokens, finish, latency = _generate(
            model, tokenizer, prompt, device, 1536,
            lambda text: len(tokenizer(text, add_special_tokens=False)["input_ids"]))
        record = {"question_id": ids[index], "sample_index": index,
                  "run_identity_sha256": run_identity,
                  "answer": answer, "finish_reason": finish,
                  "input_tokens": tokens, "output_tokens": output_tokens,
                  "generated_tokens_including_special": generated_tokens,
                  "selected_context_count": count_contexts,
                  "truncated_first_context_characters": trimmed,
                  "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                  "generation_latency_ms": latency, "max_new_tokens": 1536,
                  "mode": "single-gpu-fp16-original-prompt", "variant": EXPERIMENT}
        record["record_sha256"] = _json_sha256(record)
        _atomic_json(records / f"{index:04d}.json", record)
        _atomic_json(state_path, {"run_identity_sha256": run_identity,
                                  "completed_count": index + 1, "total": len(ids),
                                  "complete": index + 1 == len(ids)})
    return {"completed": len(ids), "total": len(ids), "resumed_from": completed}


def finalize(*, root: Path, p00: Path, e00: Path, e45: Path, e46: Path,
             private: Path, output: Path, config: Config) -> dict[str, Any]:
    shared = dict(root=root, p00=p00, e00=e00, e45=e45, e46=e46,
                  private=private, output=output, config=config)
    _, ids, pins, _ = _checked(**shared)
    _, plan = _plan(output, ids, pins)
    identity = _json_sha256({"preflight": pins, "plan_sha256": plan["plan_sha256"]})
    state = json.loads((output / "generation/state.json").read_text(encoding="utf-8"))
    if (state.get("run_identity_sha256") != identity or state.get("complete") is not True
            or state.get("completed_count") != len(ids)):
        raise PrivateE46Error("P05 generation is incomplete.")
    raw = []
    for index, qid in enumerate(ids):
        path = output / f"generation/records/{index:04d}.json"
        record = json.loads(path.read_text(encoding="utf-8"))
        if (record.get("question_id") != qid or record.get("sample_index") != index
                or record.get("run_identity_sha256") != identity
                or record.get("max_new_tokens") != 1536
                or not isinstance(record.get("answer"), str) or not record["answer"].strip()
                or not _valid_hash(record)):
            raise PrivateE46Error(f"Invalid P05 generation record: {index}")
        raw.append(record)
    _atomic_jsonl(output / "raw-results.jsonl", raw)
    final, review = _clean_rows(raw)
    _atomic_jsonl(output / "results.jsonl", final)
    _atomic_jsonl(output / "review-changed.jsonl", review)
    _write_submission(output, ids, final)
    names = ("raw-results.jsonl", "results.jsonl", "review-changed.jsonl",
             "submission.json", "submission.zip")
    report = {"schema_version": "1.0", "experiment_id": EXPERIMENT,
              "created_at_utc": datetime.now(timezone.utc).isoformat(),
              "sample_size": len(ids), "selected_stack": {
                  "retrieval": "saved-P00-top20", "selector": config.raw["selector"],
                  "contexts": "E00-metadata-enriched-E45-top12v2",
                  "generator": pins["model_id"], "adapter": "fresh-E46-fulltrain7000",
                  "precision": "float16", "device": "cuda:0", "max_new_tokens": 1536,
                  "postprocess": "E43-then-E44-unified-suffix-only", "reranker": None},
              "diagnostics": {**_diagnostics(raw, final),
                              "eos_questions": sum(row["finish_reason"] == "eos" for row in raw),
                              "length_questions": sum(row["finish_reason"] == "length" for row in raw),
                              "contexts_per_prompt_min": min(row["selected_context_count"] for row in raw),
                              "contexts_per_prompt_max": max(row["selected_context_count"] for row in raw)},
              "files": {name: {"sha256": file_sha256(output / name),
                                "bytes": (output / name).stat().st_size} for name in names},
              "evidence": pins, "private_reference_answers_read": False,
              "automatic_promotion": False}
    _atomic_json(output / "report.json", report)
    return report
