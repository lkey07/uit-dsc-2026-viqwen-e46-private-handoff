"""E18 source-metadata A/B; exact E08B bodies/ranks, new dev200, max704."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import logging
import re
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean
from time import perf_counter

from .corpus import file_sha256
from .e02_answer import _atomic_json, _atomic_jsonl, make_messages
from .e03_rrf_grid import (
    _bootstrap_ci, _json_sha256, _load_worker_progress, _read_jsonl,
    _write_worker_state,
)
from .e08b_context_lora import inference_packing, load_context_lora_generator
from .e17_citation_lexical_rerank import (
    _adapter_hash, _ids_sha, _load_dev, _metrics, _source_contexts,
    _training_config, load_config as load_source_contract,
    validate_preflight as validate_sources,
)
from .evaluation.official import ensure_nltk_resources, nltk_meteor_score, rouge_l_fmeasure

LOGGER = logging.getLogger(__name__)
EXPERIMENT = "E18-e08b-source-metadata-fresh-dev200-max704-v1"
VARIANTS = ("ranked_top12_control", "source_metadata_top12")
_ARTICLE = re.compile(r"(?im)^\s*Điều\s+[0-9]+")
# Only an explicit header 'Số:' is eligible; never infer a number from a URL,
# slug, cited document in the body, or the question/answer.
_NUMBER = re.compile(r"(?im)^[ \t]*Số[ \t]*:[ \t\r\n]*([0-9][0-9A-Za-zĐđ/.-]*)")


class E18Error(RuntimeError):
    pass


@dataclass(frozen=True)
class Config:
    raw: dict
    path: Path
    source: object

    @property
    def sha256(self):
        return file_sha256(self.path)


def load_config(project_root: Path, path: Path) -> Config:
    raw = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "schema_version": "1.0", "experiment_id": EXPERIMENT,
        "source_contract_path": "configs/e17-citation-lexical-rerank-dev200-v1.json",
        "source_contract_sha256": "b2a846a383ae90da98424e96fccd9ca4b00c4d79d9b44ff57e0895b14d7865c4",
        "variants": list(VARIANTS),
        "metadata_policy": "exact-source-title-and-article-title-plus-unambiguous-header-So-number-v1",
        "context_count": 12, "max_input_tokens": 8192, "max_new_tokens": 704,
        "sample_size": 200, "overflow_policy": "fail-before-generation-never-drop-contexts",
        "primary_metric": "meteor", "promotion_allowed": False,
        "metadata_source": {
            "artifact_version": "e00-v2",
            "manifest_sha256": "04efd3905ad6d2758461587ca68d7d70fa2c568855b78f0c762c7a37ba547b2e",
            "chunks_sha256": "1e48c7762765ac2dd169045e9f5327c5311db3f1da8a6007ef70fff58718e367",
            "documents_sha256": "f2968724e8a25124034b9ff2144427f8853ed37359443bd149ec3832f4c1fed7",
        },
    }
    if raw != expected:
        raise E18Error("E18 reviewed config changed.")
    source_path = project_root / raw["source_contract_path"]
    if file_sha256(source_path) != raw["source_contract_sha256"]:
        raise E18Error("E18 source contract changed.")
    return Config(raw, path, load_source_contract(source_path))


def load_questions(dev_path, config):
    records, ids = _load_dev(dev_path, config.source)
    # Check normalized-question groups, not just disjoint IDs.
    from .retrieval_diagnostics import select_dev_sample
    all_ids = select_dev_sample(records, seed=config.source.section("dev")["sample_seed"], size=len(records))
    normalize = lambda s: " ".join(unicodedata.normalize("NFC", s).casefold().split())
    old_groups = {normalize(records[i]["question"]) for i in all_ids[:200]}
    groups = [normalize(records[i]["question"]) for i in ids]
    if old_groups.intersection(groups) or len(set(groups)) != len(ids):
        raise E18Error("E18 dev subset has repeated or previously tuned question groups.")
    return {i: records[i]["question"] for i in ids}, ids


def source_number(cleaned_text):
    heading = _ARTICLE.search(cleaned_text)
    header = cleaned_text[:heading.start()] if heading else cleaned_text[:600]
    matches = list(dict.fromkeys(m.group(1).rstrip(".-") for m in _NUMBER.finditer(header)))
    matches = [number for number in matches if "/" in number]
    return matches[0] if len(matches) == 1 else None


def enrich_context(context, chunk, document):
    for field in ("chunk_id", "document_id", "article_number", "text"):
        if context.get(field) != chunk.get(field):
            raise E18Error(f"E00/E08A context mismatch: {context.get('chunk_id')}:{field}")
    if document.get("document_id") != chunk["document_id"]:
        raise E18Error("Wrong metadata document.")
    start, end = chunk["start_char"], chunk["end_char"]
    if not (0 <= start < end <= len(document["cleaned_text"])) or document["cleaned_text"][start:end] != chunk["text"]:
        raise E18Error("Chunk no longer matches its original document span.")
    if chunk.get("source_title") != document.get("source_title"):
        raise E18Error("Chunk and document source titles disagree.")
    title, article = chunk.get("source_title"), chunk.get("article_title")
    number = source_number(document["cleaned_text"])
    lines = []
    for label, value in (("Tên nguồn", title), ("Số văn bản", number), ("Tiêu đề Điều", article)):
        if value is not None and not isinstance(value, str):
            raise E18Error(f"Invalid metadata value: {label}")
        if value and value.strip():
            lines.append(f"{label}: {value.strip()}")
    prefix = "\n".join(lines) + "\nNội dung:\n" if lines else ""
    enriched = {**context, "text": prefix + context["text"]}
    evidence = {
        "chunk_id": chunk["chunk_id"], "document_id": chunk["document_id"],
        "source_title": title, "document_number": number, "article_title": article,
        "body_sha256": hashlib.sha256(context["text"].encode("utf-8")).hexdigest(),
        "prefix": prefix, "start_char": start, "end_char": end,
    }
    return enriched, evidence


def scan_selected(path, key, wanted, expected_sha):
    """One full byte-hashed pass, retaining only metadata for selected IDs."""
    found, digest = {}, hashlib.sha256()
    with path.open("rb") as stream:
        for line in stream:
            digest.update(line)
            if not line.strip():
                raise E18Error(f"Blank JSONL row: {path}")
            row = json.loads(line)
            if row[key] in wanted:
                if row[key] in found:
                    raise E18Error(f"Duplicate source ID: {row[key]}")
                found[row[key]] = row
    if digest.hexdigest() != expected_sha or set(found) != set(wanted):
        raise E18Error(f"Missing or changed pinned metadata source: {path}")
    return found


def save_once(path, payload, *, jsonl=False):
    if path.exists():
        old = _read_jsonl(path) if jsonl else json.loads(path.read_text(encoding="utf-8"))
        if old != payload:
            raise E18Error(f"Refusing to overwrite a different E18 artifact: {path}")
    else:
        (_atomic_jsonl if jsonl else _atomic_json)(path, payload)


def code_sha(project_root):
    paths = sorted((project_root / "src/uit_dsc_fixed_rag").rglob("*.py"))
    paths.append(project_root / "scripts/run_e18_source_metadata_kaggle.py")
    return _json_sha256({p.relative_to(project_root).as_posix(): file_sha256(p) for p in paths})


def preflight(*, project_root, e08a_directory, training_directory, e00_directory, dev_path, output_directory, config):
    evidence = validate_sources(project_root=project_root, e08a_directory=e08a_directory,
        training_directory=training_directory, dev_path=dev_path, config=config.source)
    _, ids = load_questions(dev_path, config)
    if file_sha256(e00_directory / "manifest.json") != config.raw["metadata_source"]["manifest_sha256"]:
        raise E18Error("Use the exact saved E00-v2 output.")
    payload = {"experiment_id": EXPERIMENT, "config_sha256": config.sha256,
        "code_sha256": code_sha(project_root), "source_contract_sha256": config.source.config_sha256,
        "adapter_sha256": evidence["adapter_sha256"], "sample_ids_sha256": _ids_sha(ids),
        "sample_size": len(ids), "metadata_source": config.raw["metadata_source"],
        "variants": list(VARIANTS), "max_new_tokens": 704}
    output_directory.mkdir(parents=True, exist_ok=True)
    save_once(output_directory / "preflight.json", payload)
    return payload


def check_preflight(project_root, output_directory, config):
    payload = json.loads((output_directory / "preflight.json").read_text(encoding="utf-8"))
    if payload.get("config_sha256") != config.sha256 or payload.get("code_sha256") != code_sha(project_root):
        raise E18Error("Preflight code/config changed; never mix runs.")
    return payload


def prepare(*, e08a_directory, e00_directory, dev_path, output_directory, config):
    _, ids = load_questions(dev_path, config)
    _, source_ids, rows = _source_contexts(e08a_directory, dev_path, config.source)
    if ids != source_ids:
        raise E18Error("Source sample changed.")
    wanted = {c["chunk_id"] for r in rows for c in r["contexts"]}
    LOGGER.info("e18_reading_metadata chunks=%d; streaming pinned E00 files once", len(wanted))
    chunks = scan_selected(e00_directory / "chunks.jsonl", "chunk_id", wanted,
        config.raw["metadata_source"]["chunks_sha256"])
    documents = scan_selected(e00_directory / "documents.jsonl", "document_id",
        {c["document_id"] for c in chunks.values()}, config.raw["metadata_source"]["documents_sha256"])
    prepared = []
    for index, row in enumerate(rows):
        candidate, metadata = [], []
        for c in row["contexts"]:
            enriched, ev = enrich_context(c, chunks[c["chunk_id"]], documents[c["document_id"]])
            candidate.append(enriched)
            metadata.append(ev)
        prepared.append({"question_id": ids[index], "sample_index": index,
            "answer_included": False, "metadata": metadata,
            "variants": {VARIANTS[0]: row["contexts"], VARIANTS[1]: candidate}})
    root = output_directory / "prepared"
    root.mkdir(parents=True, exist_ok=True)
    save_once(root / "results.jsonl", prepared, jsonl=True)
    meta = [m for r in prepared for m in r["metadata"]]
    summary = {"experiment_id": EXPERIMENT, "config_sha256": config.sha256,
        "results_sha256": file_sha256(root / "results.jsonl"), "sample_ids_sha256": _ids_sha(ids),
        "source_contexts_sha256": config.source.section("source_e08a")["sha256"],
        "metadata_source": config.raw["metadata_source"], "sample_size": len(ids),
        "context_instances": len(meta), "with_source_title": sum(bool(m["source_title"]) for m in meta),
        "with_document_number": sum(bool(m["document_number"]) for m in meta),
        "with_article_title": sum(bool(m["article_title"]) for m in meta),
        "unchanged_contexts": sum(not m["prefix"] for m in meta),
        "same_context_bodies_and_order": True, "answers_used": False}
    save_once(root / "summary.json", summary)
    return summary


def load_prepared(output_directory, ids, config):
    root = output_directory / "prepared"
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    if (summary.get("config_sha256") != config.sha256 or summary.get("sample_ids_sha256") != _ids_sha(ids)
            or summary.get("results_sha256") != file_sha256(root / "results.jsonl")):
        raise E18Error("Prepared metadata hash changed.")
    rows = _read_jsonl(root / "results.jsonl")
    if len(rows) != len(ids):
        raise E18Error("Prepared count changed.")
    for index, row in enumerate(rows):
        if row.get("question_id") != ids[index] or row.get("sample_index") != index or row.get("answer_included") is not False:
            raise E18Error("Prepared question identity changed.")
        a, b = (row["variants"][v] for v in VARIANTS)
        if len(a) != 12 or len(b) != 12 or len(row["metadata"]) != 12:
            raise E18Error("Both variants must have all twelve contexts.")
        for old, new, meta in zip(a, b, row["metadata"]):
            if (old["chunk_id"] != new["chunk_id"] or old["chunk_id"] != meta["chunk_id"]
                    or new != {**old, "text": meta["prefix"] + old["text"]}
                    or hashlib.sha256(old["text"].encode("utf-8")).hexdigest() != meta["body_sha256"]):
                raise E18Error("Metadata variant altered context bodies/order.")
    return rows, summary


def render_pair(question, row, tokenizer, packing):
    rendered, counts = {}, {}
    for key in VARIANTS:
        messages = make_messages(question=question, contexts=row["variants"][key], config=packing)
        rendered[key] = tokenizer.apply_chat_template(messages, tokenize=False,
            add_generation_prompt=True, enable_thinking=False)
        counts[key] = len(tokenizer(rendered[key], add_special_tokens=False)["input_ids"])
        if not 0 < counts[key] <= 8192:
            raise E18Error(f"Question {row['question_id']} {key} has {counts[key]} input tokens; no context may be dropped.")
    return rendered, counts


def validate_record(row, index, question_id, variant, identity, chunks, prompt_sha=None):
    unsigned = {k: v for k, v in row.items() if k != "record_sha256"}
    if (row.get("record_sha256") != _json_sha256(unsigned)
            or row.get("sample_index") != index or row.get("question_id") != question_id
            or row.get("variant") != variant or row.get("worker_rank") != identity["worker_rank"]
            or row.get("worker_identity_sha256") != identity["identity_sha256"]
            or row.get("selected_chunk_ids") != chunks or row.get("selected_context_count") != 12
            or row.get("max_new_tokens") != 704 or row.get("finish_reason") not in {"eos", "length", "other"}
            or not isinstance(row.get("answer"), str) or not row["answer"].strip()
            or (prompt_sha is not None and row.get("prompt_sha256") != prompt_sha)):
        raise E18Error(f"Changed or invalid checkpoint answer: {variant}:{index}")


def run_worker(*, project_root, training_directory, dev_path, output_directory, config, worker_rank, device):
    import torch
    if worker_rank not in (0, 1) or device != f"cuda:{worker_rank}":
        raise E18Error("Worker 0=control/cuda:0; worker 1=metadata/cuda:1.")
    pre = check_preflight(project_root, output_directory, config)
    questions, ids = load_questions(dev_path, config)
    prepared, summary = load_prepared(output_directory, ids, config)
    adapter_sha = _adapter_hash(training_directory, config.source)
    training = _training_config(project_root, config.source)
    model, tokenizer, device_map, adapter_parameters = load_context_lora_generator(
        config=training, training_directory=training_directory, device=device)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    variant = VARIANTS[worker_rank]
    prompts, audit = [], []
    for row in prepared:
        pair, counts = render_pair(questions[row["question_id"]], row, tokenizer, inference_packing(training))
        prompts.append(pair[variant])
        audit.append({"question_id": row["question_id"], "input_tokens": counts,
            "prompt_sha256": {v: hashlib.sha256(pair[v].encode("utf-8")).hexdigest() for v in VARIANTS}})
    root = output_directory / "evaluation" / variant
    records = root / "records"
    records.mkdir(parents=True, exist_ok=True)
    save_once(root / "prompt_audit.jsonl", audit, jsonl=True)
    identity = {"experiment_id": EXPERIMENT, "config_sha256": config.sha256,
        "code_sha256": pre["code_sha256"], "prepared_sha256": summary["results_sha256"],
        "prompt_audit_sha256": _json_sha256(audit), "sample_ids_sha256": _ids_sha(ids),
        "adapter_sha256": adapter_sha, "adapter_parameters": adapter_parameters,
        "worker_rank": worker_rank, "variant": variant, "device": device, "device_map": device_map,
        "runtime": {k: importlib.metadata.version(k) for k in ("torch", "transformers", "peft", "accelerate")}}
    identity["identity_sha256"] = _json_sha256(identity)
    state_path = root / "state.json"
    expected_files = {f"{i:04d}.json" for i in range(len(ids))}
    if any(p.name not in expected_files for p in records.glob("*.json")):
        raise E18Error("Unexpected checkpoint IDs.")
    completed = _load_worker_progress(records=records, state_path=state_path, identity=identity,
        assigned_indices=list(range(len(ids))), sample_ids=ids)
    for index in range(completed):
        saved = json.loads((records / f"{index:04d}.json").read_text(encoding="utf-8"))
        validate_record(saved, index, ids[index], variant, identity,
            [c["chunk_id"] for c in prepared[index]["variants"][variant]], audit[index]["prompt_sha256"][variant])
    LOGGER.info("e18_prompts_verified variant=%s total=%d resume_completed=%d max_input=%d",
        variant, len(ids), completed, max(a["input_tokens"][variant] for a in audit))
    for index in range(completed, len(ids)):
        tensors = tokenizer(prompts[index], add_special_tokens=False, return_tensors="pt")
        tensors = {k: v.to(device) for k, v in tensors.items()}
        started = perf_counter()
        with torch.inference_mode():
            output = model.generate(**tensors, do_sample=False, num_beams=1, max_new_tokens=704,
                use_cache=True, repetition_penalty=1.0, no_repeat_ngram_size=0)
        latency = (perf_counter() - started) * 1000
        tokens = output[0, tensors["input_ids"].shape[1]:]
        answer = tokenizer.decode(tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False).strip()
        if not answer:
            raise E18Error(f"Empty answer: {ids[index]}")
        eos = model.generation_config.eos_token_id
        eos = eos if eos is not None else tokenizer.eos_token_id
        eos_ids = set(eos if isinstance(eos, (list, tuple)) else [eos])
        count = int(tokens.shape[0])
        finish = "eos" if count and int(tokens[-1]) in eos_ids else "length" if count >= 704 else "other"
        row = {"question_id": ids[index], "sample_index": index, "variant": variant,
            "worker_rank": worker_rank, "worker_identity_sha256": identity["identity_sha256"],
            "prompt_sha256": audit[index]["prompt_sha256"][variant], "answer": answer,
            "selected_chunk_ids": [c["chunk_id"] for c in prepared[index]["variants"][variant]],
            "selected_context_count": 12, "max_new_tokens": 704,
            "input_tokens": audit[index]["input_tokens"][variant],
            "output_tokens": len(tokenizer(answer, add_special_tokens=False)["input_ids"]),
            "generated_tokens_including_special": count, "finish_reason": finish,
            "generation_latency_ms": latency}
        row["record_sha256"] = _json_sha256(row)
        _atomic_json(records / f"{index:04d}.json", row)
        _write_worker_state(state_path, identity, index + 1, len(ids))
        LOGGER.info("e18_generation_progress variant=%s device=%s completed=%d total=%d question_id=%s finish=%s",
            variant, device, index + 1, len(ids), ids[index], finish)
    return {"variant": variant, "completed": len(ids)}


def finalize(*, project_root, training_directory, dev_path, output_directory, config):
    ensure_nltk_resources(download=False)
    pre = check_preflight(project_root, output_directory, config)
    questions, ids = load_questions(dev_path, config)
    prepared, summary = load_prepared(output_directory, ids, config)
    adapter_sha = _adapter_hash(training_directory, config.source)
    by_variant, audits, runtimes = {}, [], []
    for rank, variant in enumerate(VARIANTS):
        root = output_directory / "evaluation" / variant
        state = json.loads((root / "state.json").read_text(encoding="utf-8"))
        identity = state["run_identity"]
        runtimes.append(identity.get("runtime"))
        if (state.get("complete") is not True or state.get("completed_count") != len(ids)
                or state.get("assigned_count") != len(ids) or identity.get("worker_rank") != rank
                or identity.get("variant") != variant or identity.get("adapter_sha256") != adapter_sha
                or identity.get("prepared_sha256") != summary["results_sha256"]
                or identity.get("config_sha256") != config.sha256 or identity.get("code_sha256") != pre["code_sha256"]
                or identity.get("sample_ids_sha256") != _ids_sha(ids)
                or identity.get("identity_sha256") != _json_sha256({k:v for k,v in identity.items() if k != "identity_sha256"})):
            raise E18Error(f"Incomplete or mixed worker: {variant}")
        audit = _read_jsonl(root / "prompt_audit.jsonl")
        if _json_sha256(audit) != identity["prompt_audit_sha256"] or len(audit) != len(ids):
            raise E18Error("Prompt audit changed.")
        audits.append(audit)
        if {p.name for p in (root / "records").glob("*.json")} != {f"{i:04d}.json" for i in range(len(ids))}:
            raise E18Error("Missing or extra answer IDs.")
        rows = []
        for index, question_id in enumerate(ids):
            row = json.loads((root / "records" / f"{index:04d}.json").read_text(encoding="utf-8"))
            validate_record(row, index, question_id, variant, identity,
                [c["chunk_id"] for c in prepared[index]["variants"][variant]], audit[index]["prompt_sha256"][variant])
            rows.append(row)
        by_variant[variant] = rows
        save_once(root / "results.jsonl", rows, jsonl=True)
    if audits[0] != audits[1]:
        raise E18Error("Workers did not verify identical paired prompts.")
    if runtimes[0] != runtimes[1]:
        raise E18Error("Workers used different inference library versions.")
    dev = json.loads(dev_path.read_text(encoding="utf-8"))
    scores = {v: [] for v in VARIANTS}
    merged, per_question = [], []
    for index, qid in enumerate(ids):
        predictions = {v: by_variant[v][index] for v in VARIANTS}
        values = {v: {"meteor": nltk_meteor_score(dev[qid]["answer"], predictions[v]["answer"]),
                      "rouge_l": rouge_l_fmeasure(dev[qid]["answer"], predictions[v]["answer"])} for v in VARIANTS}
        for v in VARIANTS:
            scores[v].append(values[v])
        merged.append({"question_id": qid, "sample_index": index, "variants": predictions})
        per_question.append({"question_id": qid, "sample_index": index, "scores": values,
            "meteor_delta": values[VARIANTS[1]]["meteor"] - values[VARIANTS[0]]["meteor"]})
    save_once(output_directory / "results.jsonl", merged, jsonl=True)
    save_once(output_directory / "per_question_scores.jsonl", per_question, jsonl=True)
    # Reference answers enter this diagnostic export only after generation/scoring.
    review = []
    for item in sorted(per_question, key=lambda r: (r["scores"][VARIANTS[0]]["meteor"], r["sample_index"]))[:30]:
        index, qid = item["sample_index"], item["question_id"]
        review.append({**item, "diagnostic_only": True, "question": questions[qid],
            "reference_answer": dev[qid]["answer"], "contexts": prepared[index],
            "predictions": {v: by_variant[v][index]["answer"] for v in VARIANTS}})
    save_once(output_directory / "error_review.jsonl", review, jsonl=True)
    metrics = {v: _metrics(by_variant[v], scores[v]) for v in VARIANTS}
    deltas = [r["meteor_delta"] for r in per_question]
    report = {"schema_version": "1.0", "experiment_id": EXPERIMENT,
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "sample_size": len(ids),
        "control_variant": VARIANTS[0], "candidate_variant": VARIANTS[1],
        "only_changed_factor": "corpus source metadata prepended to unchanged top12 context bodies",
        "max_new_tokens": 704, "metrics": metrics,
        "paired_delta_candidate_minus_control": {"meteor_mean": fmean(deltas),
            "meteor_bootstrap_95_ci": _bootstrap_ci(deltas, seed="e18-metadata-meteor-v1", iterations=10000),
            "improved_questions": sum(d > 0 for d in deltas), "worsened_questions": sum(d < 0 for d in deltas),
            "tied_questions": sum(d == 0 for d in deltas)},
        "metadata_coverage": summary,
        "mean_added_input_tokens": fmean(a["input_tokens"][VARIANTS[1]] - a["input_tokens"][VARIANTS[0]] for a in audits[0]),
        "evidence": {**pre, "prepared_sha256": summary["results_sha256"],
            "results_sha256": file_sha256(output_directory / "results.jsonl")},
        "smoke_leader": max(VARIANTS, key=lambda v: metrics[v]["meteor"]),
        "promotion_allowed": False, "public_read": False, "holdout_untouched": True,
        "warning": "Dev subset excludes prior tuned dev200 by normalized question; once evaluated it is development data, not holdout."}
    _atomic_json(output_directory / "report.json", report)
    return report
