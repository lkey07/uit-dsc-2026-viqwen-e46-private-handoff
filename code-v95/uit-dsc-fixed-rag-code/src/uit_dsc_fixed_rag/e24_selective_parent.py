"""Reranker-selected same-article parent segments on the used E21 dev-200."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import logging
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean
from typing import Any

from . import e21_parent_context as parent
from .corpus import file_sha256
from .e02_answer import _atomic_json, _atomic_jsonl
from .e03_rrf_grid import (
    _bootstrap_ci,
    _json_sha256,
    _load_worker_progress,
    _read_jsonl,
    _validate_worker_placement,
    _write_worker_state,
)
from .e18_source_metadata import save_once
from .e19_metadata_lora import _metrics, load_candidate_generator, validate_candidate_adapter
from .evaluation.official import ensure_nltk_resources, nltk_meteor_score, rouge_l_fmeasure

EXPERIMENT = "E24-selective-parent-reranker-dev200-v1"
CONTROL = "parent_expanded_max704"
VARIANTS = ("selective_parent_top4_max704", "selective_parent_top8_max704")
LOG = logging.getLogger(__name__)


class E24Error(RuntimeError):
    pass


@dataclass(frozen=True)
class Config:
    raw: dict[str, Any]
    path: Path
    source: Any

    @property
    def sha(self) -> str:
        return file_sha256(self.path)

    @property
    def e19(self) -> Any:
        return self.source.source.source

    @property
    def variants(self) -> list[dict[str, Any]]:
        return self.raw["variants"]


def load_config(root: Path, path: Path) -> Config:
    raw = json.loads(path.read_text(encoding="utf-8"))
    expected_keys = {
        "schema_version", "experiment_id", "source_config_path", "source_config_sha256",
        "source_code_sha256", "control", "segmentation", "reranker", "variants",
        "inference", "parameter_budget", "evaluation", "run_contract",
    }
    if set(raw) != expected_keys or raw.get("schema_version") != "1.0" or raw.get("experiment_id") != EXPERIMENT:
        raise E24Error("E24 config identity changed.")
    if (raw["source_config_path"] != "configs/e21-parent-context-dev200-v1.json"
            or raw["source_config_sha256"] != "fe8554d5f222e03a8f60ee181f8cfc4d1570719efaa6c56e2f7559fad1672ade"
            or raw["source_code_sha256"] != "e5574861eb34ae3fdee1220d5260109ddd9f84feacea9c1ec46a78df8b5527c8"):
        raise E24Error("Pinned E21 source changed.")
    source = parent.load_config(root, root / raw["source_config_path"])
    if source.sha != raw["source_config_sha256"]:
        raise E24Error("Pinned E21 config bytes changed.")
    if raw["control"] != {
        "variant": CONTROL,
        "results_sha256": "cbe3d5e163e1983f031a06c767596fcebd91177e3b90625998a1deed28f6afe4",
        "meteor": 0.5633893466813007,
        "rouge_l": 0.5864053775086489,
        "sample_size": 200,
    }:
        raise E24Error("E21 control evidence changed.")
    if raw["segmentation"] != {
        "source": "widest-available-e21-same-article-expansion",
        "paragraph_boundary": "newline-then-bounded-whitespace-window",
        "minimum_segment_characters": 40,
        "maximum_segment_characters": 1600,
        "exact_source_substrings_only": True,
        "deduplicate_document_block_offsets": True,
    }:
        raise E24Error("Selective segmentation policy changed.")
    if raw["reranker"] != {
        "model_id": "AITeamVN/Vietnamese_Reranker",
        "revision": "f536976248403314225d7fdfdbc87f0e9516a54e",
        "parameter_count": 567755777,
        "runtime_unique_parameter_count": 567755777,
        "trained_pair_max_tokens": 2304,
        "batch_size": 4,
        "score": "sequence-classification-logit-descending",
    }:
        raise E24Error("E24 reranker contract changed.")
    if raw["variants"] != [
        {"key": VARIANTS[0], "worker_rank": 0, "maximum_added_segments": 4},
        {"key": VARIANTS[1], "worker_rank": 1, "maximum_added_segments": 8},
    ]:
        raise E24Error("E24 variants changed.")
    if raw["inference"] != {
        "seed_contexts": 12, "all_seeds_mandatory": True, "max_input_tokens": 8192,
        "max_new_tokens": 704, "do_sample": False, "num_beams": 1,
        "enable_thinking": False, "use_cache": True,
    }:
        raise E24Error("E24 inference changed.")
    budget = raw["parameter_budget"]
    if (budget["maximum_stack_total"] != budget["embedding"] + budget["reranker"]
            + budget["generator"] + budget["adapter_parameter_cap"]
            or budget["maximum_stack_total"] >= budget["exclusive_limit"]):
        raise E24Error("E24 parameter budget failed.")
    if not raw["run_contract"] or not all(value is True for value in raw["run_contract"].values()):
        raise E24Error("E24 run contract lost an invariant.")
    return Config(raw, path, source)


def code_sha(root: Path) -> str:
    paths = sorted((root / "src/uit_dsc_fixed_rag").rglob("*.py"))
    paths.append(root / "scripts/run_e24_selective_parent_kaggle.py")
    return _json_sha256({path.relative_to(root).as_posix(): file_sha256(path) for path in paths})


def sample(train: Path, dev: Path, config: Config):
    return parent.sample(train, dev, config.source)


def load_source(directory: Path, ids: list[str], config: Config):
    result_path = directory / "evaluation" / CONTROL / "results.jsonl"
    prepared_path = directory / "prepared" / "results.jsonl"
    state_path = result_path.parent / "state.json"
    if (not result_path.is_file() or not prepared_path.is_file() or not state_path.is_file()
            or file_sha256(result_path) != config.raw["control"]["results_sha256"]):
        raise E24Error("Add the full byte-exact E21 output.")
    rows = _read_jsonl(result_path)
    state = json.loads(state_path.read_text(encoding="utf-8"))
    identity = state.get("run_identity", {})
    adapter_sha = config.source.source.section("source_control")["adapter_sha256"]
    if (len(rows) != 200 or state.get("complete") is not True or state.get("completed_count") != 200
            or state.get("assigned_count") != 200 or identity.get("worker_rank") != 1
            or identity.get("variant") != CONTROL or identity.get("device") != "cuda:1"
            or identity.get("config_sha256") != config.raw["source_config_sha256"]
            or identity.get("code_sha256") != config.raw["source_code_sha256"]
            or identity.get("adapter_sha256") != adapter_sha
            or identity.get("prepared_sha256") != file_sha256(prepared_path)
            or identity.get("identity_sha256") != _json_sha256({k: v for k, v in identity.items() if k != "identity_sha256"})):
        raise E24Error("E21 state or prepared provenance changed.")
    for index, qid in enumerate(ids):
        parent.validate_record(rows[index], qid, index, 1, identity)
    prepared = parent.load_prepared(directory, ids, config.source)
    return prepared, rows, identity


def preflight(root: Path, output: Path, source: Path, training: Path,
              train: Path, dev: Path, config: Config) -> dict[str, Any]:
    _, _, ids = sample(train, dev, config)
    prepared, _, identity = load_source(source, ids, config)
    if len(prepared) != 200:
        raise E24Error("E21 prepared sample changed.")
    adapter_sha, adapter_parameters = validate_candidate_adapter(training, config.e19)
    if adapter_sha != identity["adapter_sha256"]:
        raise E24Error("Wrong E19 adapter.")
    scorer = config.e19.section("scoring")
    if file_sha256(root / scorer["official_scorer_path"]) != scorer["official_scorer_sha256"]:
        raise E24Error("Pinned scorer changed.")
    payload = {
        "experiment_id": EXPERIMENT, "code_sha256": code_sha(root), "config_sha256": config.sha,
        "source_identity_sha256": identity["identity_sha256"], "adapter_sha256": adapter_sha,
        "adapter_parameters": adapter_parameters, "sample_size": 200,
        "variants": list(VARIANTS), "questions_per_variant": 200,
        "maximum_stack_parameters": config.raw["parameter_budget"]["maximum_stack_total"],
        "reference_answers_used_for_selection": False,
    }
    save_once(output / "preflight.json", payload)
    return payload


def check_preflight(root: Path, output: Path, config: Config) -> dict[str, Any]:
    path = output / "preflight.json"
    if not path.is_file():
        raise E24Error("Run E24 preflight first.")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (payload.get("experiment_id") != EXPERIMENT or payload.get("config_sha256") != config.sha
            or payload.get("code_sha256") != code_sha(root) or payload.get("sample_size") != 200):
        raise E24Error("E24 preflight code/config changed.")
    return payload


def _trim_range(text: str, start: int, end: int) -> tuple[int, int]:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def split_exact(text: str, *, absolute_start: int, minimum: int, maximum: int) -> list[dict[str, Any]]:
    """Split into exact paragraph/whitespace-bounded source spans."""
    raw = []
    for match in re.finditer(r"[^\n]+", text):
        start, end = _trim_range(text, match.start(), match.end())
        while end - start > maximum:
            bound = start + maximum
            candidates = [text.rfind(mark, start + maximum // 2, bound + 1) for mark in (". ", "; ", ": ", " ")]
            cut = max(candidates)
            if cut < start + maximum // 2:
                cut = bound
            elif text[cut:cut + 2] in (". ", "; ", ": "):
                cut += 1
            left_start, left_end = _trim_range(text, start, cut)
            if left_end > left_start:
                raw.append((left_start, left_end))
            start, end = _trim_range(text, cut, end)
        if end > start:
            raw.append((start, end))
    merged: list[tuple[int, int]] = []
    for start, end in raw:
        if end - start < minimum and merged and end - merged[-1][0] <= maximum:
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))
    if len(merged) > 1 and merged[0][1] - merged[0][0] < minimum and merged[1][1] - merged[0][0] <= maximum:
        merged[1] = (merged[0][0], merged[1][1])
        merged.pop(0)
    return [{"start": absolute_start + start, "end": absolute_start + end, "text": text[start:end]}
            for start, end in merged if end - start >= minimum]


def seed_spans(prepared: dict[str, Any]) -> list[dict[str, Any]]:
    return [{**{k: v for k, v in unit.items() if k not in ("seed", "expansions")}, **unit["seed"]}
            for unit in prepared["units"]]


def selective_candidates(prepared: dict[str, Any], config: Config) -> list[dict[str, Any]]:
    policy = config.raw["segmentation"]
    seeds = seed_spans(prepared)
    baseline = sum(len(span["text"]) for span in parent.merge_spans(seeds))
    candidates: dict[tuple[Any, ...], dict[str, Any]] = {}
    for unit in prepared["units"]:
        if not unit["expansions"]:
            continue
        widest = max(unit["expansions"], key=lambda item: (item["end"] - item["start"], -item["start"]))
        for segment in split_exact(
            widest["text"], absolute_start=widest["start"],
            minimum=policy["minimum_segment_characters"], maximum=policy["maximum_segment_characters"],
        ):
            item = {
                "document_id": unit["document_id"], "article_number": unit["article_number"],
                "article_title": unit["article_title"], "source_title": unit["source_title"],
                "document_number": unit["document_number"], "rank": unit["rank"],
                "block_id": unit["block_id"], "chunk_ids": widest["chunk_ids"],
                **segment, "source_seed_ranks": [unit["rank"]],
            }
            key = (item["document_id"], item["block_id"], item["start"], item["end"])
            if key in candidates:
                old = candidates[key]
                if old["text"] != item["text"]:
                    raise E24Error("Duplicate selective source offsets disagree.")
                old["source_seed_ranks"] = sorted(set(old["source_seed_ranks"] + [unit["rank"]]))
                old["rank"] = min(old["rank"], unit["rank"])
                continue
            if sum(len(span["text"]) for span in parent.merge_spans(seeds + [item])) > baseline:
                item["segment_id"] = hashlib.sha256(
                    f"{item['document_id']}|{item['block_id']}|{item['start']}|{item['end']}".encode("utf-8")
                ).hexdigest()[:20]
                candidates[key] = item
    return sorted(candidates.values(), key=lambda item: (item["rank"], item["document_id"], item["start"]))


def prepare(root: Path, output: Path, source: Path, train: Path, dev: Path, config: Config) -> dict[str, Any]:
    check_preflight(root, output, config)
    questions, _, ids = sample(train, dev, config)
    prepared, _, source_identity = load_source(source, ids, config)
    rows = []
    for index, qid in enumerate(ids):
        candidates = selective_candidates(prepared[index], config)
        rows.append({"question_id": qid, "sample_index": index, "answers_used": False, "candidates": candidates})
    path = output / "selective/prepared.jsonl"
    save_once(path, rows, jsonl=True)
    summary = {
        "experiment_id": EXPERIMENT, "sample_size": 200, "answers_used": False,
        "source_identity_sha256": source_identity["identity_sha256"],
        "prepared_sha256": file_sha256(path),
        "mean_candidates": fmean(len(row["candidates"]) for row in rows),
        "minimum_candidates": min(len(row["candidates"]) for row in rows),
        "maximum_candidates": max(len(row["candidates"]) for row in rows),
        "questions_with_candidates": sum(bool(row["candidates"]) for row in rows),
    }
    save_once(output / "selective/prepared-summary.json", summary)
    return summary


def load_prepared(output: Path, ids: list[str]) -> list[dict[str, Any]]:
    path = output / "selective/prepared.jsonl"
    summary = json.loads((output / "selective/prepared-summary.json").read_text(encoding="utf-8"))
    rows = _read_jsonl(path)
    if (summary.get("experiment_id") != EXPERIMENT or summary.get("prepared_sha256") != file_sha256(path)
            or summary.get("answers_used") is not False or [row.get("question_id") for row in rows] != ids
            or any(row.get("sample_index") != index or row.get("answers_used") is not False for index, row in enumerate(rows))):
        raise E24Error("Prepared selective candidates changed.")
    return rows


def rerank_indices(rank: int) -> list[int]:
    if rank not in (0, 1):
        raise E24Error("Reranker worker rank must be 0 or 1.")
    return list(range(rank * 100, (rank + 1) * 100))


def load_reranker(config: Config, device: str):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    spec = config.raw["reranker"]
    if device not in ("cuda:0", "cuda:1") or torch.cuda.device_count() != 2:
        raise E24Error("Select T4 x2 for E24 reranking.")
    torch.cuda.set_device(int(device[-1]))
    tokenizer = AutoTokenizer.from_pretrained(spec["model_id"], revision=spec["revision"], trust_remote_code=False)
    model = AutoModelForSequenceClassification.from_pretrained(
        spec["model_id"], revision=spec["revision"], dtype=torch.float16, trust_remote_code=False,
    ).to(device)
    model.eval()
    observed = sum(parameter.numel() for parameter in model.parameters())
    if observed != spec["runtime_unique_parameter_count"]:
        raise E24Error("Reranker parameter count changed.")
    placement = _validate_worker_placement(model, device)
    return model, tokenizer, placement


def candidate_text(item: dict[str, Any]) -> str:
    lines = []
    for label, value in (("Tên nguồn", item["source_title"]), ("Số văn bản", item["document_number"])):
        if value:
            lines.append(f"{label}: {value}")
    article = f"Điều {item['article_number']}" if item["article_number"] else "Trích đoạn"
    if item["article_title"]:
        article += f": {item['article_title']}"
    lines.extend([article, "Nội dung:", item["text"]])
    return "\n".join(lines)


def validate_rerank_record(row: dict[str, Any], qid: str, index: int, rank: int, identity: dict[str, Any]) -> None:
    if (row.get("question_id") != qid or row.get("sample_index") != index or row.get("worker_rank") != rank
            or row.get("worker_identity_sha256") != identity["identity_sha256"]
            or row.get("record_sha256") != _json_sha256({k: v for k, v in row.items() if k != "record_sha256"})):
        raise E24Error(f"Invalid reranker record: {rank}/{index}")


def run_reranker_worker(root: Path, output: Path, train: Path, dev: Path,
                        config: Config, rank: int, device: str) -> dict[str, Any]:
    import torch
    checked = check_preflight(root, output, config)
    questions, _, ids = sample(train, dev, config)
    prepared = load_prepared(output, ids)
    assigned = rerank_indices(rank)
    if device != f"cuda:{rank}":
        raise E24Error("Reranker worker/GPU mismatch.")
    model, tokenizer, placement = load_reranker(config, device)
    spec = config.raw["reranker"]
    identity = {
        "code_sha256": code_sha(root), "config_sha256": config.sha,
        "prepared_sha256": file_sha256(output / "selective/prepared.jsonl"),
        "source_identity_sha256": checked["source_identity_sha256"],
        "worker_rank": rank, "device": device, "assigned_indices": assigned,
        "model_id": spec["model_id"], "revision": spec["revision"],
        "runtime_unique_parameter_count": spec["runtime_unique_parameter_count"],
        "device_map": placement,
        "runtime": {key: importlib.metadata.version(key) for key in ("torch", "transformers")},
    }
    identity["identity_sha256"] = _json_sha256(identity)
    records = output / "selective/rerank-records"
    state = output / f"selective/rerank-worker-{rank}-state.json"
    records.mkdir(parents=True, exist_ok=True)
    done = _load_worker_progress(records=records, state_path=state, identity=identity,
                                 assigned_indices=assigned, sample_ids=ids)
    for index in assigned[:done]:
        validate_rerank_record(json.loads((records / f"{index:04d}.json").read_text(encoding="utf-8")), ids[index], index, rank, identity)
    for completed, index in enumerate(assigned[done:], start=done + 1):
        candidates = prepared[index]["candidates"]
        scores: list[float] = []
        maximum_tokens = 0
        started = time.perf_counter()
        pairs = [(questions[ids[index]]["question"], candidate_text(item)) for item in candidates]
        for offset in range(0, len(pairs), spec["batch_size"]):
            inputs = tokenizer(pairs[offset:offset + spec["batch_size"]], padding=True, truncation=False, return_tensors="pt")
            maximum_tokens = max(maximum_tokens, int(inputs["attention_mask"].sum(dim=1).max().item()))
            if maximum_tokens > spec["trained_pair_max_tokens"]:
                raise E24Error(f"Reranker pair exceeds trained limit: {ids[index]}/{maximum_tokens}")
            with torch.inference_mode():
                logits = model(**{key: value.to(device) for key, value in inputs.items()}, return_dict=True).logits.view(-1).float()
            scores.extend(float(value) for value in logits.detach().cpu().tolist())
        if len(scores) != len(candidates):
            raise E24Error("Reranker score count changed.")
        order = sorted(range(len(candidates)), key=lambda item: (-scores[item], item, candidates[item]["segment_id"]))
        ranked = [{**candidates[item], "reranker_score": scores[item], "reranker_rank": position + 1}
                  for position, item in enumerate(order)]
        row = {
            "question_id": ids[index], "sample_index": index, "worker_rank": rank,
            "worker_identity_sha256": identity["identity_sha256"], "answers_used": False,
            "candidates": ranked, "maximum_pair_tokens": maximum_tokens,
            "reranker_latency_ms": (time.perf_counter() - started) * 1000,
        }
        row["record_sha256"] = _json_sha256(row)
        _atomic_json(records / f"{index:04d}.json", row)
        _write_worker_state(state, identity, completed, 100)
        LOG.info("e24_rerank worker=%d completed=%d total=100 question_id=%s candidates=%d", rank, completed, ids[index], len(candidates))
    return {"worker": rank, "completed": 100}


def finalize_reranking(root: Path, output: Path, train: Path, dev: Path, config: Config) -> dict[str, Any]:
    checked = check_preflight(root, output, config)
    _, _, ids = sample(train, dev, config)
    prepared = load_prepared(output, ids)
    rows: list[dict[str, Any] | None] = [None] * 200
    identities = []
    for rank in range(2):
        state = json.loads((output / f"selective/rerank-worker-{rank}-state.json").read_text(encoding="utf-8"))
        identity = state.get("run_identity", {})
        if (state.get("complete") is not True or state.get("completed_count") != 100 or state.get("assigned_count") != 100
                or identity.get("worker_rank") != rank or identity.get("device") != f"cuda:{rank}"
                or identity.get("assigned_indices") != rerank_indices(rank)
                or identity.get("config_sha256") != config.sha or identity.get("code_sha256") != code_sha(root)
                or identity.get("prepared_sha256") != file_sha256(output / "selective/prepared.jsonl")
                or identity.get("source_identity_sha256") != checked["source_identity_sha256"]
                or identity.get("identity_sha256") != _json_sha256({k: v for k, v in identity.items() if k != "identity_sha256"})):
            raise E24Error("Incomplete or changed E24 reranker worker.")
        identities.append(identity["identity_sha256"])
        for index in rerank_indices(rank):
            row = json.loads((output / f"selective/rerank-records/{index:04d}.json").read_text(encoding="utf-8"))
            validate_rerank_record(row, ids[index], index, rank, identity)
            if len(row["candidates"]) != len(prepared[index]["candidates"]):
                raise E24Error("Reranked candidate count changed.")
            rows[index] = row
    path = output / "selective/results.jsonl"
    _atomic_jsonl(path, rows)
    summary = {
        "experiment_id": EXPERIMENT, "sample_size": 200, "answers_used": False,
        "results_sha256": file_sha256(path), "worker_identity_sha256": identities,
        "mean_latency_ms": fmean(row["reranker_latency_ms"] for row in rows),
        "maximum_pair_tokens": max(row["maximum_pair_tokens"] for row in rows),
    }
    _atomic_json(output / "selective/summary.json", summary)
    return summary


def load_reranked(output: Path, ids: list[str]) -> list[dict[str, Any]]:
    path = output / "selective/results.jsonl"
    summary = json.loads((output / "selective/summary.json").read_text(encoding="utf-8"))
    rows = _read_jsonl(path)
    if (summary.get("experiment_id") != EXPERIMENT or summary.get("results_sha256") != file_sha256(path)
            or summary.get("answers_used") is not False or [row.get("question_id") for row in rows] != ids):
        raise E24Error("Finalized reranker results changed.")
    return rows


def pack(question: str, prepared: dict[str, Any], reranked: dict[str, Any], config: Config,
         message_counter, variant: dict[str, Any]):
    chosen = seed_spans(prepared)
    messages, merged = parent.render(question, chosen, parent.inference_packing(config.e19))
    if len(chosen) != 12 or message_counter(messages) > 8192:
        raise E24Error("All 12 compact seeds must fit before selective additions.")
    baseline = sum(len(span["text"]) for span in merged)
    selected = []
    for candidate in reranked["candidates"]:
        if len(selected) >= variant["maximum_added_segments"]:
            break
        trial = chosen + [{k: v for k, v in candidate.items() if k not in ("segment_id", "source_seed_ranks", "reranker_score", "reranker_rank")}]
        trial_messages, trial_merged = parent.render(question, trial, parent.inference_packing(config.e19))
        body = sum(len(span["text"]) for span in trial_merged)
        if body <= baseline or message_counter(trial_messages) > 8192:
            continue
        chosen, messages, merged, baseline = trial, trial_messages, trial_merged, body
        selected.append({"segment_id": candidate["segment_id"], "reranker_rank": candidate["reranker_rank"],
                         "reranker_score": candidate["reranker_score"], "start": candidate["start"],
                         "end": candidate["end"], "document_id": candidate["document_id"]})
    diagnostics = {
        "seed_ranks": list(range(12)), "skipped_seed_ranks": [],
        "maximum_added_segments": variant["maximum_added_segments"], "selected_segments": selected,
        "selected_segment_count": len(selected), "candidate_count": len(reranked["candidates"]),
        "merged_span_count": len(merged), "body_characters": baseline,
        "selected_chunk_ids": sorted({chunk_id for span in merged for chunk_id in span["chunk_ids"]}),
    }
    return messages, merged, diagnostics


def validate_generation_record(row: dict[str, Any], qid: str, index: int, rank: int,
                               variant: str, identity: dict[str, Any]) -> None:
    if (row.get("question_id") != qid or row.get("sample_index") != index or row.get("worker_rank") != rank
            or row.get("variant") != variant or row.get("worker_identity_sha256") != identity["identity_sha256"]
            or not isinstance(row.get("answer"), str) or not row["answer"].strip()
            or row.get("record_sha256") != _json_sha256({k: v for k, v in row.items() if k != "record_sha256"})):
        raise E24Error(f"Invalid E24 generation record: {rank}/{index}")


def run_generation_worker(root: Path, output: Path, source: Path, training: Path,
                          train: Path, dev: Path, config: Config, rank: int, device: str) -> dict[str, Any]:
    import torch
    if rank not in (0, 1) or device != f"cuda:{rank}":
        raise E24Error("GPU0=top4 all200 and GPU1=top8 all200.")
    checked = check_preflight(root, output, config)
    questions, _, ids = sample(train, dev, config)
    prepared, _, source_identity = load_source(source, ids, config)
    reranked = load_reranked(output, ids)
    runtime = {key: importlib.metadata.version(key) for key in source_identity["runtime"]}
    if runtime != source_identity["runtime"]:
        raise E24Error("Use the exact E21 runtime versions.")
    model, tokenizer, placement, parameters = load_candidate_generator(config=config.e19, training_directory=training, device=device)
    adapter_sha, _ = validate_candidate_adapter(training, config.e19)
    generation_sha = _json_sha256(model.generation_config.to_dict())
    if adapter_sha != checked["adapter_sha256"] or generation_sha != source_identity["generation_config_sha256"]:
        raise E24Error("E19 adapter or generation defaults differ from E21.")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    def rendered(messages):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    def count(text):
        return len(tokenizer(text, add_special_tokens=False)["input_ids"])
    variant = config.variants[rank]
    identity = {
        "code_sha256": code_sha(root), "config_sha256": config.sha,
        "source_identity_sha256": source_identity["identity_sha256"],
        "reranked_sha256": file_sha256(output / "selective/results.jsonl"),
        "adapter_sha256": adapter_sha, "adapter_parameters": parameters,
        "runtime": runtime, "generation_config_sha256": generation_sha,
        "worker_rank": rank, "device": device, "variant": variant["key"], "device_map": placement,
        "assigned_indices": list(range(200)),
    }
    identity["identity_sha256"] = _json_sha256(identity)
    folder = output / "evaluation" / variant["key"]
    records, state = folder / "records", folder / "state.json"
    records.mkdir(parents=True, exist_ok=True)
    indices = list(range(200))
    done = _load_worker_progress(records=records, state_path=state, identity=identity,
                                 assigned_indices=indices, sample_ids=ids)
    for index in indices[:done]:
        validate_generation_record(json.loads((records / f"{index:04d}.json").read_text(encoding="utf-8")),
                                   ids[index], index, rank, variant["key"], identity)
    for completed, index in enumerate(indices[done:], start=done + 1):
        messages, spans, diagnostics = pack(questions[ids[index]]["question"], prepared[index], reranked[index],
                                             config, lambda value: count(rendered(value)), variant)
        prompt = rendered(messages)
        input_tokens = count(prompt)
        inputs = {key: value.to(device) for key, value in tokenizer(prompt, add_special_tokens=False, return_tensors="pt").items()}
        started = time.perf_counter()
        with torch.inference_mode():
            generated = model.generate(**inputs, do_sample=False, num_beams=1, max_new_tokens=704, use_cache=True)
        latency = (time.perf_counter() - started) * 1000
        new_ids = generated[0, inputs["input_ids"].shape[1]:]
        answer = tokenizer.decode(new_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False).strip()
        if not answer:
            raise E24Error(f"Empty answer: {ids[index]}")
        eos = tokenizer.eos_token_id
        eos_ids = set(eos if isinstance(eos, list) else [eos])
        finish = "eos" if len(new_ids) and int(new_ids[-1]) in eos_ids else "length" if len(new_ids) >= 704 else "other"
        row = {
            "question_id": ids[index], "sample_index": index, "worker_rank": rank,
            "variant": variant["key"], "worker_identity_sha256": identity["identity_sha256"],
            "answer": answer, "input_tokens": input_tokens, "output_tokens": count(answer),
            "generated_tokens_including_special": len(new_ids), "finish_reason": finish,
            "generation_latency_ms": latency, "selected_context_count": len(spans),
            "packing": diagnostics, "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "evidence_spans": [{k: v for k, v in span.items() if k != "text"} for span in spans],
        }
        row["record_sha256"] = _json_sha256(row)
        _atomic_json(records / f"{index:04d}.json", row)
        _write_worker_state(state, identity, completed, 200)
        LOG.info("e24_generation variant=%s device=%s completed=%d total=200 question_id=%s finish=%s",
                 variant["key"], device, completed, ids[index], finish)
    return {"variant": variant["key"], "completed": 200, "device": device}


def finalize(root: Path, output: Path, source: Path, train: Path, dev: Path, config: Config) -> dict[str, Any]:
    checked = check_preflight(root, output, config)
    ensure_nltk_resources(download=False)
    questions, _, ids = sample(train, dev, config)
    prepared, controls, source_identity = load_source(source, ids, config)
    reranked = load_reranked(output, ids)
    del prepared, reranked
    by_variant = {CONTROL: controls}
    for rank, variant in enumerate(config.variants):
        folder = output / "evaluation" / variant["key"]
        state = json.loads((folder / "state.json").read_text(encoding="utf-8"))
        identity = state.get("run_identity", {})
        if (state.get("complete") is not True or state.get("completed_count") != 200 or state.get("assigned_count") != 200
                or identity.get("worker_rank") != rank or identity.get("device") != f"cuda:{rank}"
                or identity.get("variant") != variant["key"] or identity.get("assigned_indices") != list(range(200))
                or identity.get("config_sha256") != config.sha or identity.get("code_sha256") != code_sha(root)
                or identity.get("source_identity_sha256") != source_identity["identity_sha256"]
                or identity.get("reranked_sha256") != file_sha256(output / "selective/results.jsonl")
                or identity.get("adapter_sha256") != checked["adapter_sha256"]
                or identity.get("runtime") != source_identity["runtime"]
                or identity.get("generation_config_sha256") != source_identity["generation_config_sha256"]
                or identity.get("identity_sha256") != _json_sha256({k: v for k, v in identity.items() if k != "identity_sha256"})):
            raise E24Error("Incomplete or changed E24 generation state.")
        rows = []
        for index, qid in enumerate(ids):
            row = json.loads((folder / "records" / f"{index:04d}.json").read_text(encoding="utf-8"))
            validate_generation_record(row, qid, index, rank, variant["key"], identity)
            rows.append(row)
        _atomic_jsonl(folder / "results.jsonl", rows)
        by_variant[variant["key"]] = rows
    scores = {
        key: [{"meteor": nltk_meteor_score(questions[qid]["answer"], row["answer"]),
               "rouge_l": rouge_l_fmeasure(questions[qid]["answer"], row["answer"])}
              for qid, row in zip(ids, rows)]
        for key, rows in by_variant.items()
    }
    def paired(candidate: str, baseline: str) -> dict[str, Any]:
        delta = [left["meteor"] - right["meteor"] for left, right in zip(scores[candidate], scores[baseline])]
        return {
            "meteor_mean": fmean(delta),
            "meteor_bootstrap_95_ci": _bootstrap_ci(delta, seed=f"e24-{candidate}-{baseline}", iterations=10000),
            "improved": sum(value > 0 for value in delta),
            "worsened": sum(value < 0 for value in delta),
            "tied": sum(value == 0 for value in delta),
        }
    _atomic_jsonl(output / "per_question_scores.jsonl", [
        {"question_id": qid, "scores": {key: scores[key][index] for key in scores}}
        for index, qid in enumerate(ids)
    ])
    report = {
        "schema_version": "1.0", "experiment_id": EXPERIMENT,
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "sample_size": 200,
        "sample_scope": "same-repeated-e19-e21-dev400-600", "control_variant": CONTROL,
        "metrics": {key: _metrics(rows, scores[key]) for key, rows in by_variant.items()},
        "paired_top4_minus_e21": paired(VARIANTS[0], CONTROL),
        "paired_top8_minus_e21": paired(VARIANTS[1], CONTROL),
        "paired_top8_minus_top4": paired(VARIANTS[1], VARIANTS[0]),
        "selection": {key: {
            "mean_selected_segments": fmean(row["packing"]["selected_segment_count"] for row in by_variant[key]),
            "mean_candidate_count": fmean(row["packing"]["candidate_count"] for row in by_variant[key]),
            "mean_input_tokens": fmean(row["input_tokens"] for row in by_variant[key]),
            "mean_body_characters": fmean(row["packing"]["body_characters"] for row in by_variant[key]),
        } for key in VARIANTS},
        "smoke_leader": max(by_variant, key=lambda key: fmean(item["meteor"] for item in scores[key])),
        "evidence": {
            "config_sha256": config.sha, "code_sha256": code_sha(root),
            "source_identity_sha256": source_identity["identity_sha256"],
            "control_results_sha256": config.raw["control"]["results_sha256"],
            "reranked_results_sha256": file_sha256(output / "selective/results.jsonl"),
            "candidate_results_sha256": {key: file_sha256(output / f"evaluation/{key}/results.jsonl") for key in VARIANTS},
        },
        "promotion_allowed": False, "public_read": False, "holdout_untouched": True,
        "warning": "Repeated dev-200. E24 tests question-only reranker selection inside exact E21 parents; no private/public guarantee.",
    }
    _atomic_json(output / "report.json", report)
    return report
