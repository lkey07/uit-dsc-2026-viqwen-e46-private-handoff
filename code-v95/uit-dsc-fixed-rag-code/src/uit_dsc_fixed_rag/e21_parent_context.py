"""Bounded parent context A/B over original E19 retrieval seeds; no new model."""
from __future__ import annotations

import copy
import hashlib
import importlib.metadata
import json
import logging
import time
from bisect import bisect_right
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean

from .corpus import file_sha256
from .legal_chunking import _article_spans
from .e02_answer import _atomic_json, _atomic_jsonl, make_messages
from .e03_rrf_grid import _json_sha256, _read_jsonl, _bootstrap_ci, _load_worker_progress, _write_worker_state
from .e08b_context_lora import inference_packing, _validate_context_rows
from .e18_source_metadata import scan_selected, enrich_context, source_number, save_once
from .e19_metadata_lora import eval_sample, _metrics
from .e20_source_aware_expansion import load_config as load_source, _control_rows, validate_preflight as source_preflight
from .evaluation.official import ensure_nltk_resources, nltk_meteor_score, rouge_l_fmeasure

EXPERIMENT = "E21-parent-context-dev200-v1"
VARIANTS = ("compact_seeds_max704", "parent_expanded_max704")
CONTROL = "e19_metadata_trained_rank8_max704"
LOG = logging.getLogger(__name__)


class E21Error(RuntimeError):
    pass


@dataclass(frozen=True)
class Config:
    raw: dict
    path: Path
    source: object

    @property
    def sha(self):
        return file_sha256(self.path)

    @property
    def policy(self):
        return self.raw["policy"]


def load_config(root, path):
    raw = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "seed_contexts": 12, "max_input_tokens": 8192, "max_new_tokens": 704,
        "max_parent_tokens": 1200, "max_parent_characters": 6000,
        "neighbor_radius": 1,
        "expansion_order": ["whole_article", "both_neighbors", "previous_neighbor", "next_neighbor"],
        "evict_seed_for_expansion": False,
        "merge": "overlapping-or-touching-exact-document-spans",
        "metadata": "one-source-header-per-document-article-label-per-span",
        "variants": list(VARIANTS), "questions_per_worker": 200,
        "sample": "reuse-e19-dev400-600",
        "reference_answers": "final-scoring-and-human-review-only",
    }
    if (set(raw) != {"schema_version", "experiment_id", "source_contract_path", "policy", "promotion_allowed"}
            or raw["schema_version"] != "1.0" or raw["experiment_id"] != EXPERIMENT
            or raw["policy"] != expected or raw["promotion_allowed"] is not False
            or raw["source_contract_path"] != "configs/e20-source-aware-top20-expansion-dev200-v1.json"):
        raise E21Error("E21 reviewed contract changed.")
    return Config(raw, path, load_source(root, root / raw["source_contract_path"]))


def code_sha(root):
    paths = sorted((root / "src/uit_dsc_fixed_rag").rglob("*.py"))
    paths.append(root / "scripts/run_e21_parent_context_kaggle.py")
    return _json_sha256({p.relative_to(root).as_posix(): file_sha256(p) for p in paths})


def preflight(root, output, config, **sources):
    evidence = source_preflight(project_root=root, config=config.source, **sources)
    evidence.pop("candidate_pool_size", None)
    evidence.pop("output_context_count", None)
    evidence.update(code_version="0.54.0", code_sha256=code_sha(root), config_sha256=config.sha,
                    seed_context_count=12, variants=list(VARIANTS), questions_per_worker=200)
    payload = {"experiment_id": EXPERIMENT, "evidence": evidence}
    save_once(output / "preflight.json", payload)
    return payload


def check_preflight(root, output, config):
    p = json.loads((output / "preflight.json").read_text(encoding="utf-8"))
    if (p.get("experiment_id") != EXPERIMENT
            or p["evidence"].get("code_sha256") != code_sha(root)
            or p["evidence"].get("config_sha256") != config.sha):
        raise E21Error("E21 preflight changed; never mix checkpoints across code/config.")
    return p


def sample(train, dev, config):
    return eval_sample(train, dev, config.source.source)


def article_blocks(chunks, document=None):
    """Repeated article numbers are separated by contiguous chunk runs."""
    blocks, previous, previous_key = [], None, None
    parents = _article_spans(document["cleaned_text"]) if document is not None else []
    starts = [p.start for p in parents]
    for chunk in sorted(chunks, key=lambda c: c["chunk_index"]):
        boundary = None
        if document is not None and chunk.get("article_number") is not None:
            index = bisect_right(starts, chunk["start_char"]) - 1
            parent = parents[index] if index >= 0 else None
            if (parent is None or parent.number != chunk["article_number"]
                    or chunk["end_char"] > parent.end):
                raise E21Error("Chunk has no unambiguous original article boundary.")
            boundary = parent.start
        key = (chunk["document_id"], chunk.get("article_number"), chunk.get("article_title"), boundary)
        if (previous is None or key != previous_key
                or chunk["chunk_index"] != previous["chunk_index"] + 1
                or key[1] is None):
            blocks.append([])
        blocks[-1].append(chunk)
        previous = chunk
        previous_key = key
    return blocks


def span(document, start, end, chunk_ids):
    if not 0 <= start < end <= len(document["cleaned_text"]):
        raise E21Error("Invalid document offsets.")
    return {"start": start, "end": end, "text": document["cleaned_text"][start:end],
            "chunk_ids": sorted(set(chunk_ids))}


def seed_unit(seed, rank, block, document, policy):
    enrich_context({k: seed.get(k) for k in ("chunk_id", "document_id", "article_number", "text")}, seed, document)
    first = block[0]
    unit = {
        "document_id": seed["document_id"], "article_number": seed.get("article_number"),
        "article_title": seed.get("article_title"), "source_title": document.get("source_title"),
        "document_number": source_number(document["cleaned_text"]), "rank": rank,
        "block_id": first["chunk_id"],
        "seed": span(document, seed["start_char"], seed["end_char"], [seed["chunk_id"]]),
        "expansions": [],
    }
    if seed.get("article_number") is None:
        return unit
    index = next(i for i, c in enumerate(block) if c["chunk_id"] == seed["chunk_id"])
    before, after = max(0, index - 1), min(len(block) - 1, index + 1)
    slices = {"whole_article": block, "both_neighbors": block[before:after + 1],
              "previous_neighbor": block[before:index + 1], "next_neighbor": block[index:after + 1]}
    seen = {(seed["start_char"], seed["end_char"])}
    for label in policy["expansion_order"]:
        subset = slices[label]
        start, end = min(c["start_char"] for c in subset), max(c["end_char"] for c in subset)
        if (start, end) in seen or end - start > policy["max_parent_characters"]:
            continue
        seen.add((start, end))
        # All added text must be backed by this exact E00 document, never a title inference.
        for c in subset:
            if document["cleaned_text"][c["start_char"]:c["end_char"]] != c["text"]:
                raise E21Error("Neighbor chunk text differs from E00 source span.")
        unit["expansions"].append({"kind": label, **span(document, start, end, [c["chunk_id"] for c in subset])})
    return unit


def merge_spans(units):
    """Merge only overlapping/touching spans of the same document/article run."""
    merged = []
    for original in sorted(units, key=lambda u: (u["document_id"], u["block_id"], u["start"], u["end"])):
        item = copy.deepcopy(original)
        if (merged and (merged[-1]["document_id"], merged[-1]["block_id"]) == (item["document_id"], item["block_id"])
                and item["start"] <= merged[-1]["end"]):
            old = merged[-1]
            overlap = min(old["end"], item["end"]) - item["start"]
            if old["text"][item["start"] - old["start"]:item["start"] - old["start"] + overlap] != item["text"][:overlap]:
                raise E21Error("Overlapping source spans disagree.")
            if item["end"] > old["end"]:
                old["text"] += item["text"][old["end"] - item["start"]:]
                old["end"] = item["end"]
            old["rank"] = min(old["rank"], item["rank"])
            old["chunk_ids"] = sorted(set(old["chunk_ids"] + item["chunk_ids"]))
        else:
            merged.append(item)
    return sorted(merged, key=lambda u: (u["rank"], u["document_id"], u["start"]))


def render(question, units, packing):
    merged = merge_spans(units)
    documents = {}
    for u in merged:
        documents.setdefault(u["document_id"], []).append(u)
    contexts = []
    for parts in documents.values():
        first = parts[0]
        lines = []
        for label, value in (("Tên nguồn", first["source_title"]), ("Số văn bản", first["document_number"])):
            if value:
                lines.append(f"{label}: {value.strip()}")
        for part in parts:
            label = f"Điều {part['article_number']}" if part["article_number"] else "Trích đoạn"
            if part["article_title"]:
                label += f": {part['article_title']}"
            lines.extend([label, "Nội dung:", part["text"]])
        contexts.append({"text": "\n".join(lines), "article_number": None})
    return make_messages(question=question, contexts=contexts, config=packing), merged


def pack(question, prepared, config, message_counter, text_counter, variant):
    if variant not in VARIANTS:
        raise E21Error("Unknown packing variant.")
    policy, packing = config.policy, inference_packing(config.source.source)
    chosen, skipped = [], []
    for unit in prepared["units"]:
        value = {k: v for k, v in unit.items() if k not in ("seed", "expansions")}
        value.update(unit["seed"])
        trial = chosen + [value]
        messages, _ = render(question, trial, packing)
        if message_counter(messages) <= policy["max_input_tokens"]:
            chosen = trial
        else:
            skipped.append(unit["rank"])
    if not chosen:
        raise E21Error("No seed fits E21 input budget.")
    if skipped:
        raise E21Error(f"Compact seeds exceed the input budget; refusing to drop original evidence: {skipped}")
    seed_ranks = [u["rank"] for u in chosen]
    actions = []
    if variant == VARIANTS[1]:
        for i, selected in enumerate(list(chosen)):
            for extension in prepared["units"][selected["rank"]]["expansions"]:
                if text_counter(extension["text"]) > policy["max_parent_tokens"]:
                    continue
                replacement = {**selected, **{k: v for k, v in extension.items() if k != "kind"}}
                trial = chosen[:i] + [replacement] + chosen[i + 1:]
                messages, trial_spans = render(question, trial, packing)
                if sum(len(s["text"]) for s in trial_spans) <= sum(len(s["text"]) for s in merge_spans(chosen)):
                    continue
                if message_counter(messages) <= policy["max_input_tokens"]:
                    chosen = trial
                    actions.append({"seed_rank": selected["rank"], "kind": extension["kind"]})
                    break
    messages, merged = render(question, chosen, packing)
    diagnostics = {
        "seed_ranks": seed_ranks, "skipped_seed_ranks": skipped,
        "expansions": actions, "merged_span_count": len(merged),
        "document_count": len({u["document_id"] for u in merged}),
        "body_characters": sum(len(u["text"]) for u in merged),
        "selected_chunk_ids": sorted({cid for u in merged for cid in u["chunk_ids"]}),
    }
    return messages, merged, diagnostics


def prepare(e08a, e00, train, dev, output, config):
    _, all_ids, ids = sample(train, dev, config)
    source = config.source.source.section("source_e08a")
    raw_path = e08a / source["dev_results_path"]
    if file_sha256(raw_path) != source["dev_results_sha256"]:
        raise E21Error("E08A source changed.")
    rows = _validate_context_rows(raw_path, all_ids, "dev521", source)[200:400]
    metadata = config.source.source.contract["metadata_source"]
    seeds = scan_selected(e00 / "chunks.jsonl", "chunk_id",
                          {c["chunk_id"] for r in rows for c in r["contexts"]}, metadata["chunks_sha256"])
    doc_ids = {c["document_id"] for c in seeds.values()}
    documents = scan_selected(e00 / "documents.jsonl", "document_id", doc_ids, metadata["documents_sha256"])
    by_doc, digest = {d: [] for d in doc_ids}, hashlib.sha256()
    with (e00 / "chunks.jsonl").open("rb") as stream:
        for line in stream:
            digest.update(line)
            c = json.loads(line)
            if c["document_id"] in by_doc:
                by_doc[c["document_id"]].append(c)
    if digest.hexdigest() != metadata["chunks_sha256"]:
        raise E21Error("E00 changed during parent scan.")
    lookup = {}
    for document_id, chunks in by_doc.items():
        for block in article_blocks(chunks, documents[document_id]):
            for c in block:
                if c["chunk_id"] in seeds:
                    lookup[c["chunk_id"]] = block
    prepared = []
    for i, (qid, row) in enumerate(zip(ids, rows)):
        units = []
        for rank, c in enumerate(row["contexts"]):
            seed = seeds[c["chunk_id"]]
            enrich_context(c, seed, documents[seed["document_id"]])
            units.append(seed_unit(seed, rank, lookup[c["chunk_id"]], documents[seed["document_id"]], config.policy))
        prepared.append({"question_id": qid, "sample_index": i, "answers_used": False, "units": units})
    path = output / "prepared/results.jsonl"
    save_once(path, prepared, jsonl=True)
    summary = {"experiment_id": EXPERIMENT, "sample_size": 200, "config_sha256": config.sha,
               "results_sha256": file_sha256(path), "source_contexts_sha256": file_sha256(raw_path),
               "expandable_questions": sum(any(u["expansions"] for u in r["units"]) for r in prepared),
               "answers_used": False, "variants": list(VARIANTS)}
    save_once(output / "prepared/summary.json", summary)
    return summary


def load_prepared(output, ids, config):
    path = output / "prepared/results.jsonl"
    summary = json.loads((output / "prepared/summary.json").read_text(encoding="utf-8"))
    rows = _read_jsonl(path)
    if (summary.get("experiment_id") != EXPERIMENT or summary.get("config_sha256") != config.sha
            or summary.get("results_sha256") != file_sha256(path)
            or [r.get("question_id") for r in rows] != ids
            or any(r.get("answers_used") is not False or r.get("sample_index") != i
                   or [u.get("rank") for u in r.get("units", [])] != list(range(12)) for i, r in enumerate(rows))):
        raise E21Error("Prepared contexts changed.")
    return rows


def validate_record(row, qid, index, rank, identity):
    if (row.get("question_id") != qid or row.get("sample_index") != index
            or row.get("variant") != VARIANTS[rank] or row.get("worker_rank") != rank
            or row.get("worker_identity_sha256") != identity["identity_sha256"]
            or not isinstance(row.get("answer"), str) or not row["answer"].strip()
            or row.get("record_sha256") != _json_sha256({k: v for k, v in row.items() if k != "record_sha256"})):
        raise E21Error(f"Changed/incompatible answer record: {rank}/{index}")


def run_worker(root, train, dev, output, training, config, rank, device):
    import torch
    from .e19_metadata_lora import load_candidate_generator, validate_candidate_adapter
    if rank not in (0, 1) or device != f"cuda:{rank}":
        raise E21Error("GPU0=compact all200; GPU1=expanded all200.")
    check_preflight(root, output, config)
    records_dev, _, ids = sample(train, dev, config)
    prepared = load_prepared(output, ids, config)
    adapter_sha, _ = validate_candidate_adapter(training, config.source.source)
    if adapter_sha != config.source.section("source_control")["adapter_sha256"]:
        raise E21Error("Wrong E19 adapter.")
    model, tokenizer, placement, params = load_candidate_generator(config=config.source.source, training_directory=training, device=device)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    def rendered(messages):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    def count_text(text):
        return len(tokenizer(text, add_special_tokens=False)["input_ids"])
    def count_messages(messages):
        return count_text(rendered(messages))
    identity = {
        "code_sha256": code_sha(root), "config_sha256": config.sha,
        "prepared_sha256": file_sha256(output / "prepared/results.jsonl"),
        "adapter_sha256": adapter_sha, "adapter_parameters": params,
        "variant": VARIANTS[rank], "worker_rank": rank, "device": device,
        "device_map": placement, "sample_ids_sha256": config.source.section("evaluation")["sample_ids_sha256"],
        "runtime": {k: importlib.metadata.version(k) for k in ("torch", "transformers", "peft", "accelerate")},
        "generation_config_sha256": _json_sha256(model.generation_config.to_dict()),
    }
    identity["identity_sha256"] = _json_sha256(identity)
    folder = output / "evaluation" / VARIANTS[rank]
    records, state_path = folder / "records", folder / "state.json"
    records.mkdir(parents=True, exist_ok=True)
    done = _load_worker_progress(records=records, state_path=state_path, identity=identity,
                                 assigned_indices=list(range(200)), sample_ids=ids)
    for i in range(done):
        validate_record(json.loads((records / f"{i:04d}.json").read_text(encoding="utf-8")), ids[i], i, rank, identity)
    for i in range(done, 200):
        qid = ids[i]
        messages, spans, diagnostics = pack(records_dev[qid]["question"], prepared[i], config, count_messages, count_text, VARIANTS[rank])
        prompt = rendered(messages)
        inputs = {k: v.to(device) for k, v in tokenizer(prompt, add_special_tokens=False, return_tensors="pt").items()}
        started = time.perf_counter()
        with torch.inference_mode():
            result = model.generate(**inputs, do_sample=False, num_beams=1, max_new_tokens=704, use_cache=True)
        latency = (time.perf_counter() - started) * 1000
        new_ids = result[0, inputs["input_ids"].shape[1]:]
        answer = tokenizer.decode(new_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False).strip()
        if not answer:
            raise E21Error(f"Empty answer: {qid}")
        eos = tokenizer.eos_token_id
        eos_ids = set(eos if isinstance(eos, list) else [eos])
        finish = "eos" if len(new_ids) and int(new_ids[-1]) in eos_ids else "length" if len(new_ids) >= 704 else "other"
        row = {"question_id": qid, "sample_index": i, "worker_rank": rank,
               "variant": VARIANTS[rank], "worker_identity_sha256": identity["identity_sha256"],
               "answer": answer, "input_tokens": count_text(prompt), "output_tokens": count_text(answer),
               "generated_tokens_including_special": len(new_ids), "finish_reason": finish,
               "generation_latency_ms": latency, "selected_context_count": len(spans),
               "selected_chunk_ids": diagnostics["selected_chunk_ids"], "packing": diagnostics,
               "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
               "evidence_spans": [{k: v for k, v in s.items() if k != "text"} for s in spans]}
        row["record_sha256"] = _json_sha256(row)
        _atomic_json(records / f"{i:04d}.json", row)
        _write_worker_state(state_path, identity, i + 1, 200)
        LOG.info("e21_generation variant=%s device=%s completed=%d total=200 question_id=%s finish=%s", VARIANTS[rank], device, i + 1, qid, finish)
    return {"variant": VARIANTS[rank], "completed": 200}


def finalize(root, train, dev, output, control, config):
    check_preflight(root, output, config)
    ensure_nltk_resources(download=False)
    questions, _, ids = sample(train, dev, config)
    prepared = load_prepared(output, ids, config)
    by_variant = {CONTROL: _control_rows(control, ids, config.source)}
    for rank, variant in enumerate(VARIANTS):
        folder = output / "evaluation" / variant
        state = json.loads((folder / "state.json").read_text(encoding="utf-8"))
        identity = state["run_identity"]
        if (state.get("complete") is not True or state.get("completed_count") != 200
                or state.get("assigned_count") != 200 or identity.get("worker_rank") != rank
                or identity.get("variant") != variant or identity.get("device") != f"cuda:{rank}"
                or identity.get("sample_ids_sha256") != config.source.section("evaluation")["sample_ids_sha256"]
                or identity.get("code_sha256") != code_sha(root) or identity.get("config_sha256") != config.sha
                or identity.get("prepared_sha256") != file_sha256(output / "prepared/results.jsonl")
                or identity.get("adapter_sha256") != config.source.section("source_control")["adapter_sha256"]
                or identity.get("identity_sha256") != _json_sha256({k: v for k, v in identity.items() if k != "identity_sha256"})):
            raise E21Error("Incomplete or incompatible worker state.")
        rows = []
        for i, qid in enumerate(ids):
            row = json.loads((folder / "records" / f"{i:04d}.json").read_text(encoding="utf-8"))
            validate_record(row, qid, i, rank, identity)
            rows.append(row)
        _atomic_jsonl(folder / "results.jsonl", rows)
        by_variant[variant] = rows
    scores = {v: [{"meteor": nltk_meteor_score(questions[q]["answer"], row["answer"]),
                   "rouge_l": rouge_l_fmeasure(questions[q]["answer"], row["answer"])}
                  for q, row in zip(ids, rows)] for v, rows in by_variant.items()}
    def paired(a, b):
        delta = [x["meteor"] - y["meteor"] for x, y in zip(scores[a], scores[b])]
        return {"meteor_mean": fmean(delta), "meteor_bootstrap_95_ci": _bootstrap_ci(delta, seed=f"e21-{a}-{b}", iterations=10000),
                "improved": sum(x > 0 for x in delta), "worsened": sum(x < 0 for x in delta), "tied": sum(x == 0 for x in delta)}
    _atomic_jsonl(output / "per_question_scores.jsonl", [{"question_id": q, "scores": {v: scores[v][i] for v in scores}} for i, q in enumerate(ids)])
    # Human diagnostic export only, produced after generation is complete.
    review = []
    for i in sorted(range(200), key=lambda i: (scores[CONTROL][i]["meteor"], i))[:50]:
        q = ids[i]
        review.append({"question_id": q, "question": questions[q]["question"], "reference_answer": questions[q]["answer"],
                       "answers": {v: by_variant[v][i]["answer"] for v in by_variant},
                       "scores": {v: scores[v][i] for v in scores}, "seed_units": prepared[i]["units"],
                       "packing": {v: by_variant[v][i]["packing"] for v in VARIANTS},
                       "manual_error_category": None,
                       "allowed_review_categories": ["missing_evidence", "wrong_source", "evidence_present_answer_incomplete", "repetition", "wording_or_scoring", "unclear"]})
    _atomic_jsonl(output / "review_low50.jsonl", review)
    metrics = {v: _metrics(rows, scores[v]) for v, rows in by_variant.items()}
    report = {"schema_version": "1.0", "experiment_id": EXPERIMENT,
              "created_at_utc": datetime.now(timezone.utc).isoformat(), "sample_size": 200,
              "sample_scope": "already-used-e19-dev400-600-final121-not-scored",
              "control_variant": CONTROL, "metrics": metrics,
              "paired_compact_minus_e19": paired(VARIANTS[0], CONTROL),
              "paired_expanded_minus_e19": paired(VARIANTS[1], CONTROL),
              "paired_expanded_minus_compact": paired(VARIANTS[1], VARIANTS[0]),
              "packing": {v: {"questions_with_expansion": sum(bool(r["packing"]["expansions"]) for r in by_variant[v]),
                               "mean_body_characters": fmean(r["packing"]["body_characters"] for r in by_variant[v]),
                               "mean_input_tokens": fmean(r["input_tokens"] for r in by_variant[v]),
                               "questions_with_skipped_seeds": sum(bool(r["packing"]["skipped_seed_ranks"]) for r in by_variant[v])} for v in VARIANTS},
              "smoke_leader": max(metrics, key=lambda v: metrics[v]["meteor"]),
              "promotion_allowed": False, "public_read": False, "holdout_untouched": True,
              "evidence": {"config_sha256": config.sha, "code_sha256": code_sha(root),
                           "control_results_sha256": config.source.section("source_control")["results_sha256"],
                           "candidate_results_sha256": {v: file_sha256(output / "evaluation" / v / "results.jsonl") for v in VARIANTS}},
              "warning": "Repeatedly used dev; inspect paired effects and untouched validation before promotion. No private-performance guarantee."}
    _atomic_json(output / "report.json", report)
    return report
