"""One-GPU E46/top12-v2 private variant with E21/P00 parent expansion."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from .corpus import file_sha256
from .e02_answer import _atomic_json, _atomic_jsonl
from .e03_rrf_grid import _json_sha256, _read_jsonl
from .e18_source_metadata import scan_selected
from .e21_parent_context import E21Error, VARIANTS, article_blocks, pack, seed_unit
from .e46_viqwen_top12v2_fulltrain import render_viqwen
from . import final_private_p05_e46 as p05


EXPERIMENT = "FINAL-private-p06-e46-top12v2-parent-max1536-unified-v1"


class ParentVariantError(RuntimeError):
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
        "schema_version": "1.0", "experiment_id": EXPERIMENT,
        "source_p05_config_path": "configs/final-private-p05-e46-top12v2-max1536-unified-v1.json",
        "source_p00_config_path": "configs/final-private-p00-retrieval-parent-v1.json",
        "selector": "rrf-top6-then-legal-priority-v2",
        "parent_policy": "E21-bounded-same-article-over-E45-top12v2-seeds",
        "max_input_tokens": 8192, "max_new_tokens": 1536,
        "single_gpu_fp16": True,
        "seed_overflow_fallback": "P05-top12v2-no-parent",
        "postprocess": "E43-then-E44-unified-suffix-only",
        "private_reference_answers_read": False, "reranker": None,
    }:
        raise ParentVariantError("P06 parent variant contract changed.")
    return Config(raw, path)


def _code_sha(root: Path) -> str:
    paths = (root / "src/uit_dsc_fixed_rag/final_private_p06_e46_parent.py",
             root / "scripts/run_final_private_p06_e46_parent.py")
    return _json_sha256({path.relative_to(root).as_posix(): file_sha256(path) for path in paths})


def _base_dir(output: Path) -> Path:
    return output / "p05-selector-source"


def _sources(*, root: Path, p00: Path, e00: Path, e45: Path, e46: Path,
             private: Path, config: Config) -> tuple[list[str], dict[str, Any], Any, Any, dict[str, Any]]:
    base_config = p05.load_config(root / config.raw["source_p05_config_path"])
    _, ids, base_pins, ec = p05._verify_sources(
        root=root, p00=p00, e00=e00, e45=e45, e46=e46,
        private=private, config=base_config)
    p00_config_path = root / config.raw["source_p00_config_path"]
    p00_config = json.loads(p00_config_path.read_text(encoding="utf-8"))
    policy = p00_config["context_policy"]
    report = json.loads((p00 / "report.json").read_text(encoding="utf-8"))
    prepared_path = p00 / "prepared/results.jsonl"
    prepared_pin = report.get("files", {}).get("prepared/results.jsonl", {})
    if (not prepared_path.is_file()
            or file_sha256(prepared_path) != prepared_pin.get("sha256")
            or prepared_path.stat().st_size != prepared_pin.get("bytes")
            or report.get("downstream_candidates_must_reuse_prepared_sha256") != prepared_pin.get("sha256")
            or file_sha256(p00_config_path) != report.get("evidence", {}).get("config_sha256")
            or policy != report.get("context_policy")
            or policy.get("seed_contexts") != 12
            or policy.get("max_input_tokens") != config.raw["max_input_tokens"]
            or policy.get("max_parent_tokens") != 1200
            or policy.get("max_parent_characters") != 6000
            or policy.get("evict_seed_for_expansion") is not False):
        raise ParentVariantError("Saved P00 parent policy/artifact changed.")
    pins = {
        "experiment_id": EXPERIMENT, "code_sha256": _code_sha(root),
        "config_sha256": config.sha, "p05_source_pins_sha256": _json_sha256(base_pins),
        "p00_prepared_sha256": prepared_pin["sha256"],
        "p00_context_policy_sha256": _json_sha256(policy),
        "sample_size": len(ids), "private_sha256": base_pins["private_sha256"],
        "sample_ids_sha256": base_pins["sample_ids_sha256"],
        "e46_adapter_sha256": base_pins["e46_adapter_sha256"],
        "model_id": base_pins["model_id"], "model_revision": base_pins["model_revision"],
        "private_reference_answers_read": False, "reranker": None,
    }
    return ids, pins, ec, base_config, policy


def preflight(*, root: Path, p00: Path, e00: Path, e45: Path, e46: Path,
              private: Path, output: Path, config: Config) -> dict[str, Any]:
    ids, pins, _, base_config, _ = _sources(
        root=root, p00=p00, e00=e00, e45=e45, e46=e46,
        private=private, config=config)
    p05.preflight(root=root, p00=p00, e00=e00, e45=e45, e46=e46,
                  private=private, output=_base_dir(output), config=base_config)
    path = output / "preflight.json"
    if path.is_file() and json.loads(path.read_text(encoding="utf-8")) != pins:
        raise ParentVariantError("Saved P06 preflight belongs to another run.")
    _atomic_json(path, pins)
    return {**pins, "questions": len(ids)}


def _checked(**shared: Any) -> tuple[list[str], dict[str, Any], Any, Any, dict[str, Any]]:
    path = shared["output"] / "preflight.json"
    if not path.is_file():
        raise ParentVariantError("Run P06 preflight first.")
    result = _sources(**{key: value for key, value in shared.items() if key != "output"})
    if json.loads(path.read_text(encoding="utf-8")) != result[1]:
        raise ParentVariantError("P06 preflight identity changed.")
    return result


def prepare(*, root: Path, p00: Path, e00: Path, e45: Path, e46: Path,
            private: Path, output: Path, config: Config) -> dict[str, Any]:
    shared = dict(root=root, p00=p00, e00=e00, e45=e45, e46=e46,
                  private=private, output=output, config=config)
    ids, pins, ec, base_config, policy = _checked(**shared)
    # Reuse the audited E45/top12-v2 selector, not P00's older RRF top-12.
    p05.prepare(root=root, p00=p00, e00=e00, e45=e45, e46=e46,
                private=private, output=_base_dir(output), config=base_config)
    selected = _read_jsonl(_base_dir(output) / "plan.jsonl")
    if len(selected) != len(ids):
        raise ParentVariantError("P05 selector plan count changed.")
    wanted = {context["chunk_id"] for row in selected for context in row["contexts"]}
    metadata = ec.e45.section("metadata_source")
    chunks_path = e00 / metadata["chunks_path"]
    seeds = scan_selected(chunks_path, "chunk_id", wanted, metadata["chunks_sha256"])
    document_ids = {seed["document_id"] for seed in seeds.values()}
    documents = scan_selected(e00 / metadata["documents_path"], "document_id", document_ids,
                              metadata["documents_sha256"])
    by_document = {document_id: [] for document_id in document_ids}
    digest = hashlib.sha256()
    with chunks_path.open("rb") as stream:
        for line in stream:
            digest.update(line)
            chunk = json.loads(line)
            if chunk["document_id"] in by_document:
                by_document[chunk["document_id"]].append(chunk)
    if digest.hexdigest() != metadata["chunks_sha256"]:
        raise ParentVariantError("E00 changed during parent block scan.")
    blocks = {}
    for document_id, chunks in by_document.items():
        for block in article_blocks(chunks, documents[document_id]):
            for chunk in block:
                if chunk["chunk_id"] in wanted:
                    blocks[chunk["chunk_id"]] = block
    if set(blocks) != wanted:
        raise ParentVariantError("Missing top12-v2 seed parent blocks.")
    rows = []
    for index, (qid, source) in enumerate(zip(ids, selected)):
        if source.get("question_id") != qid or source.get("sample_index") != index:
            raise ParentVariantError(f"P05 selector row order changed: {index}")
        units = []
        for rank, context in enumerate(source["contexts"]):
            seed = seeds[context["chunk_id"]]
            units.append(seed_unit(seed, rank, blocks[seed["chunk_id"]],
                                   documents[seed["document_id"]], policy))
        rows.append({"question_id": qid, "sample_index": index,
                     "question": source["question"], "answers_used": False,
                     "selected_chunk_ids": [x["chunk_id"] for x in source["contexts"]],
                     "units": units})
    path = output / "parent-plan.jsonl"
    if path.is_file():
        if _read_jsonl(path) != rows:
            raise ParentVariantError("Saved P06 parent plan changed.")
    else:
        _atomic_jsonl(path, rows)
    plan = {"experiment_id": EXPERIMENT, "preflight_sha256": _json_sha256(pins),
            "p05_plan_sha256": file_sha256(_base_dir(output) / "plan.jsonl"),
            "parent_plan_sha256": file_sha256(path), "questions": len(ids),
            "expandable_questions": sum(any(unit["expansions"] for unit in row["units"])
                                        for row in rows),
            "answers_used_by_selector_or_parent": False}
    plan_path = output / "plan.json"
    if plan_path.is_file() and json.loads(plan_path.read_text(encoding="utf-8")) != plan:
        raise ParentVariantError("Saved P06 plan identity changed.")
    _atomic_json(plan_path, plan)
    return plan


def _plan(output: Path, ids: list[str], pins: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    plan = json.loads((output / "plan.json").read_text(encoding="utf-8"))
    path = output / "parent-plan.jsonl"
    if (plan.get("experiment_id") != EXPERIMENT
            or plan.get("preflight_sha256") != _json_sha256(pins)
            or plan.get("p05_plan_sha256") != file_sha256(_base_dir(output) / "plan.jsonl")
            or plan.get("parent_plan_sha256") != file_sha256(path)
            or plan.get("questions") != len(ids)
            or plan.get("answers_used_by_selector_or_parent") is not False):
        raise ParentVariantError("P06 parent plan identity changed.")
    rows = _read_jsonl(path)
    if (len(rows) != len(ids)
            or any(row.get("question_id") != qid or row.get("sample_index") != i
                   or row.get("answers_used") is not False
                   or [u.get("rank") for u in row.get("units", [])] != list(range(12))
                   or [u.get("seed", {}).get("chunk_ids") for u in row["units"]]
                   != [[cid] for cid in row.get("selected_chunk_ids", [])]
                   for i, (qid, row) in enumerate(zip(ids, rows)))):
        raise ParentVariantError("P06 parent rows changed.")
    return rows, plan


def _prompt(tokenizer: Any, row: dict[str, Any], ec: Any, policy: dict[str, Any],
            p05_row: dict[str, Any]) -> tuple[str, int, dict[str, Any]]:
    def render(messages: list[dict[str, Any]]) -> str:
        return render_viqwen(tokenizer, messages)

    def count(value: str) -> int:
        return len(tokenizer(value, add_special_tokens=False)["input_ids"])

    context_config = SimpleNamespace(policy=policy, source=SimpleNamespace(source=ec.e19))
    try:
        messages, spans, diagnostics = pack(
            row["question"], row, context_config,
            lambda messages: count(render(messages)), count, VARIANTS[1])
    except E21Error as error:
        if not str(error).startswith(("Compact seeds exceed the input budget",
                                      "No seed fits E21 input budget")):
            raise
        # Parent formatting must never force an unsafe partial prompt. Reuse P05
        # packing for only these over-budget QIDs; make the fallback auditable.
        prompt, tokens, contexts, trimmed = p05._prompt(tokenizer, p05_row, ec, 8192)
        return prompt, tokens, {"mode": "p05-seed-budget-fallback",
                                "parent_expansions": [], "selected_seed_count": contexts,
                                "trimmed_first_context_characters": trimmed}
    prompt = render(messages)
    tokens = count(prompt)
    if (tokens > 8192 or diagnostics["seed_ranks"] != list(range(12))
            or diagnostics["skipped_seed_ranks"]):
        raise ParentVariantError("P06 parent packing lost a seed or exceeded input cap.")
    return prompt, tokens, {"mode": "e21-parent-top12v2",
                            "parent_expansions": diagnostics["expansions"],
                            "selected_seed_count": 12,
                            "merged_span_count": len(spans),
                            "trimmed_first_context_characters": 0}


def _generation(output: Path, ids: list[str], run_identity: str) -> tuple[list[dict[str, Any]], int]:
    records = output / "generation/records"
    state_path = output / "generation/state.json"
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.is_file() else None
    if state is not None and state.get("run_identity_sha256") != run_identity:
        raise ParentVariantError("Saved P06 generation belongs to another run.")
    rows, gap = [], False
    for index, qid in enumerate(ids):
        path = records / f"{index:04d}.json"
        if not path.is_file():
            gap = True
            continue
        if gap:
            raise ParentVariantError("P06 generation records are non-contiguous.")
        row = json.loads(path.read_text(encoding="utf-8"))
        if (row.get("question_id") != qid or row.get("sample_index") != index
                or row.get("run_identity_sha256") != run_identity
                or row.get("variant") != EXPERIMENT
                or row.get("max_new_tokens") != 1536
                or not isinstance(row.get("answer"), str) or not row["answer"].strip()
                or not p05._valid_hash(row)):
            raise ParentVariantError(f"P06 generation row changed: {index}")
        rows.append(row)
    if state is None and rows:
        raise ParentVariantError("P06 records exist without checkpoint state.")
    if state is not None and int(state.get("completed_count", -1)) > len(rows):
        raise ParentVariantError("P06 checkpoint ahead of durable records.")
    return rows, len(rows)


def generate(*, root: Path, p00: Path, e00: Path, e45: Path, e46: Path,
             private: Path, output: Path, config: Config, model_cache: Path) -> dict[str, Any]:
    shared = dict(root=root, p00=p00, e00=e00, e45=e45, e46=e46,
                  private=private, output=output, config=config)
    ids, pins, ec, _, policy = _checked(**shared)
    rows, plan = _plan(output, ids, pins)
    p05_rows = _read_jsonl(_base_dir(output) / "plan.jsonl")
    p05._runtime(ec)
    run_identity = _json_sha256({"preflight": pins, "plan_sha256": plan["parent_plan_sha256"]})
    records = output / "generation/records"
    records.mkdir(parents=True, exist_ok=True)
    state_path = output / "generation/state.json"
    _, completed = _generation(output, ids, run_identity)
    _atomic_json(state_path, {"run_identity_sha256": run_identity,
                              "completed_count": completed, "total": len(ids),
                              "complete": completed == len(ids)})
    if completed == len(ids):
        return {"completed": completed, "total": len(ids), "resumed": True}
    model, tokenizer, device = p05._load_model(model_cache, e46, ec)
    from tqdm.auto import tqdm
    for index in tqdm(range(completed, len(ids)), desc="P06 E46 top12-v2 + parent max1536"):
        prompt, tokens, packing = _prompt(tokenizer, rows[index], ec, policy, p05_rows[index])
        answer, output_tokens, generated_tokens, finish, latency = p05._generate(
            model, tokenizer, prompt, device, 1536,
            lambda value: len(tokenizer(value, add_special_tokens=False)["input_ids"]))
        record = {"question_id": ids[index], "sample_index": index,
                  "run_identity_sha256": run_identity,
                  "answer": answer, "finish_reason": finish,
                  "input_tokens": tokens, "output_tokens": output_tokens,
                  "generated_tokens_including_special": generated_tokens,
                  "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                  "generation_latency_ms": latency, "max_new_tokens": 1536,
                  "packing": packing, "variant": EXPERIMENT}
        record["record_sha256"] = _json_sha256(record)
        _atomic_json(records / f"{index:04d}.json", record)
        _atomic_json(state_path, {"run_identity_sha256": run_identity,
                                  "completed_count": index + 1, "total": len(ids),
                                  "complete": index + 1 == len(ids)})
    return {"completed": len(ids), "total": len(ids), "resumed_from": completed}


def finalize(*, root: Path, p00: Path, e00: Path, e45: Path, e46: Path,
             private: Path, output: Path, config: Config) -> dict[str, Any]:
    ids, pins, _, _, _ = _checked(
        root=root, p00=p00, e00=e00, e45=e45, e46=e46,
        private=private, output=output, config=config)
    _, plan = _plan(output, ids, pins)
    identity = _json_sha256({"preflight": pins, "plan_sha256": plan["parent_plan_sha256"]})
    raw, completed = _generation(output, ids, identity)
    state = json.loads((output / "generation/state.json").read_text(encoding="utf-8"))
    if (completed != len(ids) or state.get("run_identity_sha256") != identity
            or state.get("completed_count") != len(ids) or state.get("complete") is not True):
        raise ParentVariantError("P06 generation is incomplete.")
    _atomic_jsonl(output / "raw-results.jsonl", raw)
    final, review = p05._clean_rows(raw)
    _atomic_jsonl(output / "results.jsonl", final)
    _atomic_jsonl(output / "review-changed.jsonl", review)
    p05._write_submission(output, ids, final)
    names = ("raw-results.jsonl", "results.jsonl", "review-changed.jsonl",
             "submission.json", "submission.zip")
    report = {"schema_version": "1.0", "experiment_id": EXPERIMENT,
              "created_at_utc": datetime.now(timezone.utc).isoformat(),
              "sample_size": len(ids), "selected_stack": {
                  "retrieval": "saved-P00-top20", "selector": config.raw["selector"],
                  "contexts": "E45-top12v2-E21-bounded-same-article-parent",
                  "generator": pins["model_id"], "adapter": "fresh-E46-fulltrain7000",
                  "precision": "float16", "device": "cuda:0", "max_new_tokens": 1536,
                  "postprocess": config.raw["postprocess"], "reranker": None},
              "diagnostics": {**p05._diagnostics(raw, final),
                  "eos_questions": sum(r["finish_reason"] == "eos" for r in raw),
                  "length_questions": sum(r["finish_reason"] == "length" for r in raw),
                  "questions_with_parent_expansion": sum(bool(r["packing"]["parent_expansions"]) for r in raw),
                  "seed_budget_fallback_questions": sum(r["packing"]["mode"] == "p05-seed-budget-fallback" for r in raw)},
              "files": {name: {"sha256": file_sha256(output / name),
                               "bytes": (output / name).stat().st_size} for name in names},
              "evidence": {**pins, **plan},
              "private_reference_answers_read": False,
              "automatic_promotion": False,
              "parent_improvement_unproven": True}
    _atomic_json(output / "report.json", report)
    return report
