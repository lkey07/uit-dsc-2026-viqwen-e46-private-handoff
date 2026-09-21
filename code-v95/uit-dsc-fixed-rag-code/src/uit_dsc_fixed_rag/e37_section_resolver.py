"""Question-only section selection inside documents already present in E32 prompts."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import logging
import re
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from typing import Any

from .corpus import file_sha256
from .e03_rrf_grid import (
    _bootstrap_ci, _json_sha256, _load_worker_progress, _read_jsonl,
    _write_worker_state,
)
from .e08b_context_lora import inference_packing
from .e13_e08b_max768_tailtrim import trim_repeated_tail
from .e18_source_metadata import save_once, scan_selected
from .e19_metadata_lora import load_candidate_generator, validate_candidate_adapter
from .e21_parent_context import render as render_parent
from .e31_long_token_suffix_trim import trim_long_token_suffix
from .e36_failure_audit import load_config as load_e36_config
from .evaluation.official import ensure_nltk_resources, nltk_meteor_score, rouge_l_fmeasure
from .legal_chunking import _article_spans

EXPERIMENT = "E37-document-section-resolver-dev120-v1"
SOURCE_VARIANT = "e32_parent_max1024_tailtrim_longtoken"
LOG = logging.getLogger(__name__)
_STOP = frozenset("""ai bao cac can chi cho co cua da de den dieu duoc gi khi la lam mot
nao nay nguoi nhieu nhu nhung quy dinh sao sau se theo thi tren trong tu va ve voi
phap luat hien tai thuc hien truong hop doi tuong thanh pho nam bao gom noi dung
""".split())
_WORD = re.compile(r"[a-z0-9]+")
_PROCEDURE = re.compile(r"^(?:PHẪU THUẬT|KỸ THUẬT|QUY TRÌNH|ĐIỀU TRỊ)\b", re.IGNORECASE)
_ROMAN = re.compile(r"^(?:[IVX]+|\d+)\s*[.)-]\s+", re.IGNORECASE)


class E37Error(RuntimeError):
    pass


@dataclass(frozen=True)
class Config:
    raw: dict[str, Any]
    path: Path
    e36: Any

    @property
    def sha(self) -> str:
        return file_sha256(self.path)

    @property
    def e32(self) -> Any:
        return self.e36.e32_config

    @property
    def policy(self) -> dict[str, Any]:
        return self.raw["policy"]


def load_config(root: Path, path: Path) -> Config:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if (set(raw) != {"schema_version", "experiment_id", "source_e36_config_path",
                    "source_e36_config_sha256", "source", "policy", "runtime",
                    "run_contract"}
            or raw["schema_version"] != "1.0" or raw["experiment_id"] != EXPERIMENT
            or raw["source_e36_config_path"] != "configs/e36-e32-failure-audit-dev120-v1.json"
            or raw["source_e36_config_sha256"] != "c98e779efd2be83c71402dffa9627e60bc5b9fce12b780dabce98ff1d566d0e7"
            or raw["source"] != {
                "e36_experiment_id": "E36-e32-failure-audit-dev120-v1",
                "sample_size": 120,
                "sample_ids_sha256": "ffc390de97c137c571a768e1cc1e59b10be1245ed29529da39725528d05bd3e5",
                "baseline_meteor": 0.5627316657089687,
                "baseline_rouge_l": 0.561259109088349,
                "adapter_sha256": "bfb9d8120337c4013b5dcb0b59c6d03beccfe9f9748e5a01ef62e1de64fb5edd",
            }
            or raw["policy"] != {
                "documents": "only-documents-in-original-top12-packed-spans",
                "sections": ["exact-article", "uppercase-procedure-heading"],
                "minimum_shared_heading_terms": 2,
                "minimum_heading_score": 2.0,
                "minimum_best_vs_second_margin": 0.25,
                "maximum_section_characters": 6000,
                "maximum_section_tokens": 1200,
                "maximum_input_tokens": 8192,
                "replacement": "one-selected-document-spans-with-one-exact-section-first",
                "fallback": "byte-exact-e32-prompt-and-answer",
                "max_new_tokens": 1024,
                "postprocessing": ["conservative-consecutive-tail-block-trim-v1",
                                   "exact-consecutive-long-token-suffix-trim-v1"],
            }
            or raw["runtime"] != {
                "torch": "2.10.0+cu128", "transformers": "5.16.1",
                "peft": "0.19.1", "accelerate": "1.13.0", "nltk": "3.7",
                "generation_config_sha256": "cb07aad7984cd3cf30c1de668fb00db44bfb75f44b64bb02e36badff2de0ed8b",
            }
            or raw["run_contract"] != {
                "same_repeated_clean_dev120_as_e32": True,
                "exploratory_only_no_automatic_promotion": True,
                "question_and_source_only_during_selection": True,
                "reference_answers_only_after_generation": True,
                "unchanged_prompts_reuse_exact_e32_answers": True,
                "no_retrieval_index_rebuild_or_finetuning": True,
                "no_public_or_private_read": True,
                "fixed_deterministic_no_external_or_synthetic_data": True,
                "checkpoint_per_generated_question_id": True,
                "resume_fail_closed": True,
                "no_submission": True,
            }):
        raise E37Error("E37 reviewed experiment contract changed.")
    source = root / raw["source_e36_config_path"]
    if file_sha256(source) != raw["source_e36_config_sha256"]:
        raise E37Error("Pinned E36 config changed.")
    return Config(raw, path, load_e36_config(root, source))


def code_sha(root: Path) -> str:
    paths = [root / "src/uit_dsc_fixed_rag/e37_section_resolver.py",
             root / "scripts/run_e37_section_resolver_kaggle.py"]
    return _json_sha256({p.relative_to(root).as_posix(): file_sha256(p) for p in paths})


def _ids_sha(ids: list[str]) -> str:
    return hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()


def source_rows(e36: Path, config: Config) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    report_path = e36 / "report.json"
    rows_path = e36 / "review_all120.jsonl"
    if not report_path.is_file() or not rows_path.is_file():
        raise E37Error("Add the complete E36 notebook output, including review_all120.jsonl.")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    pin = config.raw["source"]
    if (report.get("experiment_id") != pin["e36_experiment_id"]
            or report.get("sample_size") != pin["sample_size"]
            or report.get("exact_prompts_verified") != 120
            or report.get("baseline_meteor") != pin["baseline_meteor"]
            or report.get("baseline_rouge_l") != pin["baseline_rouge_l"]
            or report.get("evidence", {}).get("sample_ids_sha256") != pin["sample_ids_sha256"]
            or report.get("evidence", {}).get("source_results_sha256")
            != config.e36.raw["source_e32"]["selected_results_sha256"]
            or report.get("files", {}).get("review_all120.jsonl") != file_sha256(rows_path)
            or report.get("public_read") is not False
            or report.get("private_untouched") is not True):
        raise E37Error("E36 source report or review packets changed.")
    rows = _read_jsonl(rows_path)
    ids = [str(row.get("question_id")) for row in rows]
    if (len(rows) != 120 or _ids_sha(ids) != pin["sample_ids_sha256"]
            or any(row.get("sample_index") != i or row.get("prompt_verified") is not True
                   or not isinstance(row.get("actual_prompt"), str)
                   or not isinstance(row.get("model_answer"), str) or not row["model_answer"].strip()
                   or row.get("record_sha256") != _json_sha256({k: v for k, v in row.items() if k != "record_sha256"})
                   for i, row in enumerate(rows))):
        raise E37Error("E36 row order, prompt verification, or record identity changed.")
    return report, rows


def _norm(value: str) -> str:
    value = unicodedata.normalize("NFD", value.casefold().replace("đ", "d"))
    return " ".join(_WORD.findall("".join(c for c in value if unicodedata.category(c) != "Mn")))


def _terms(value: str) -> set[str]:
    return {token for token in _norm(value).split() if token not in _STOP}


def _source_terms(source_title: str | None) -> set[str]:
    return _terms((source_title or "").replace("-", " ").replace("_", " "))


def _is_upper_heading(value: str) -> bool:
    letters = [c for c in value if c.isalpha()]
    return len(letters) >= 12 and sum(c.isupper() for c in letters) / len(letters) >= 0.9


def _procedure_headings(text: str) -> list[tuple[int, str]]:
    lines = list(re.finditer(r"(?m)^[^\n]+", text))
    found: list[tuple[int, str]] = []
    for i, line in enumerate(lines):
        head = line.group().strip()
        if not _PROCEDURE.match(head) or not _is_upper_heading(head):
            continue
        parts = [head]
        for following in lines[i + 1:i + 4]:
            continuation = following.group().strip()
            if not continuation or _ROMAN.match(continuation) or not _is_upper_heading(continuation):
                break
            if _PROCEDURE.match(continuation):
                break
            parts.append(continuation)
        title = " ".join(parts)
        if len(title) <= 220:
            found.append((line.start(), title))
    return found


def _sections(document: dict[str, Any]) -> list[dict[str, Any]]:
    text = document["cleaned_text"]
    articles = _article_spans(text)
    procedures = _procedure_headings(text)
    result: list[dict[str, Any]] = []
    for article in articles:
        result.append({"start": article.start, "end": article.end,
                       "heading": article.heading, "kind": "article",
                       "article_number": article.number, "article_title": article.title})
    for i, (start, heading) in enumerate(procedures):
        article = next((a for a in articles if a.start <= start < a.end), None)
        next_start = procedures[i + 1][0] if i + 1 < len(procedures) else len(text)
        end = min(next_start, article.end if article else len(text))
        if end > start:
            result.append({"start": start, "end": end, "heading": heading,
                           "kind": "procedure", "article_number": None,
                           "article_title": heading})
    return result


def _candidate_score(question: str, heading: str, source_title: str | None) -> tuple[float, int]:
    query, head = _terms(question), _terms(heading)
    shared = query & head
    if len(shared) < 2 or not head:
        return 0.0, len(shared)
    precision = len(shared) / len(head)
    recall = len(shared) / max(1, len(query))
    source_recall = len(query & _source_terms(source_title)) / max(1, len(query))
    return round(2.0 * precision + 1.5 * recall + 0.5 * source_recall
                 + 0.4 * len(shared), 6), len(shared)


def resolve(question: str, packed_spans: list[dict[str, Any]],
            documents: dict[str, dict[str, Any]], policy: dict[str, Any]) -> dict[str, Any] | None:
    """Select one exact source section using the question and packed source IDs only."""
    if not isinstance(question, str) or not packed_spans:
        raise E37Error("Missing question or packed E32 evidence.")
    docs = {span["document_id"] for span in packed_spans}
    if docs - documents.keys():
        raise E37Error("Packed document is missing from pinned E00 source.")
    candidates: list[dict[str, Any]] = []
    for document_id in docs:
        document = documents[document_id]
        first_rank = min(s["rank"] for s in packed_spans if s["document_id"] == document_id)
        for section in _sections(document):
            if section["end"] - section["start"] > policy["maximum_section_characters"]:
                continue
            score, shared = _candidate_score(question, section["heading"], document.get("source_title"))
            if (shared < policy["minimum_shared_heading_terms"]
                    or score < policy["minimum_heading_score"]):
                continue
            candidates.append({**section, "document_id": document_id,
                               "source_title": document.get("source_title"),
                               "first_seed_rank": first_rank, "score": score,
                               "shared_heading_terms": shared})
    candidates.sort(key=lambda c: (-c["score"], c["first_seed_rank"], c["document_id"], c["start"]))
    if not candidates:
        return None
    best = candidates[0]
    if (len(candidates) > 1
            and best["score"] - candidates[1]["score"] < policy["minimum_best_vs_second_margin"]):
        return None
    for span in packed_spans:
        if (span["document_id"] == best["document_id"] and span["rank"] <= 2
                and span["start"] <= best["start"] and span["end"] >= best["end"]):
            return None
    text = documents[best["document_id"]]["cleaned_text"]
    selected_text = text[best["start"]:best["end"]]
    if not selected_text.strip():
        raise E37Error("Selected source section is empty.")
    return {**best, "text": selected_text}


def _replacement_units(spans: list[dict[str, Any]], chosen: dict[str, Any]) -> list[dict[str, Any]]:
    same_doc = [s for s in spans if s["document_id"] == chosen["document_id"]]
    if not same_doc:
        raise E37Error("Chosen source document was not retrieved.")
    first = min(same_doc, key=lambda s: s["rank"])
    replacement = {
        "document_id": chosen["document_id"], "article_number": chosen["article_number"],
        "article_title": chosen["article_title"], "source_title": chosen["source_title"],
        "document_number": first.get("document_number"), "rank": -1,
        "block_id": "e37-section:" + chosen["document_id"] + ":" + str(chosen["start"]),
        "start": chosen["start"], "end": chosen["end"], "text": chosen["text"],
        "chunk_ids": sorted({cid for s in same_doc for cid in s["chunk_ids"]}),
    }
    return [replacement] + [s for s in spans if s["document_id"] != chosen["document_id"]]


def _rendered(tokenizer: Any, messages: list[dict[str, Any]]) -> str:
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                         enable_thinking=False)


def _count(tokenizer: Any, value: str) -> int:
    return len(tokenizer(value, add_special_tokens=False)["input_ids"])


def validate_preflight(*, root: Path, e36: Path, e00: Path, training: Path,
                       output: Path, config: Config) -> dict[str, Any]:
    report, _ = source_rows(e36, config)
    docs = e00 / "documents.jsonl"
    expected_docs = config.e32.e19.contract["metadata_source"]["documents_sha256"]
    if not docs.is_file() or file_sha256(docs) != expected_docs:
        raise E37Error("Add the exact E00-v2 documents.jsonl source.")
    adapter_sha, _ = validate_candidate_adapter(training, config.e32.e19)
    if adapter_sha != config.raw["source"]["adapter_sha256"]:
        raise E37Error("Wrong E19 adapter.")
    payload = {
        "experiment_id": EXPERIMENT, "code_sha256": code_sha(root),
        "config_sha256": config.sha, "e36_report_sha256": file_sha256(e36 / "report.json"),
        "e36_rows_sha256": report["files"]["review_all120.jsonl"],
        "e00_documents_sha256": expected_docs, "adapter_sha256": adapter_sha,
        "sample_ids_sha256": config.raw["source"]["sample_ids_sha256"],
        "reference_answers_used": False,
    }
    save_once(output / "preflight.json", payload)
    return payload


def _preflight(root: Path, output: Path, config: Config) -> dict[str, Any]:
    path = output / "preflight.json"
    if not path.is_file():
        raise E37Error("Run preflight first.")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (payload.get("experiment_id") != EXPERIMENT
            or payload.get("code_sha256") != code_sha(root)
            or payload.get("config_sha256") != config.sha
            or payload.get("sample_ids_sha256") != config.raw["source"]["sample_ids_sha256"]):
        raise E37Error("E37 preflight code/config changed.")
    return payload


def _verify_input_identity(checked: dict[str, Any], e36: Path, e00: Path) -> None:
    if (file_sha256(e36 / "report.json") != checked["e36_report_sha256"]
            or file_sha256(e36 / "review_all120.jsonl") != checked["e36_rows_sha256"]
            or file_sha256(e00 / "documents.jsonl") != checked["e00_documents_sha256"]):
        raise E37Error("Pinned E36 or E00 input changed after preflight.")


def prepare(*, root: Path, e36: Path, e00: Path, training: Path, output: Path,
            config: Config, tokenizer: Any) -> dict[str, Any]:
    checked = _preflight(root, output, config)
    _verify_input_identity(checked, e36, e00)
    _, source = source_rows(e36, config)
    needed = {s["document_id"] for r in source for s in r["packed_spans"]}
    docs = scan_selected(e00 / "documents.jsonl", "document_id", needed,
                         checked["e00_documents_sha256"])
    packing = inference_packing(config.e32.e21.source.source)
    prepared = []
    for index, row in enumerate(source):
        question = row["question"]
        baseline_messages, _ = render_parent(question, row["packed_spans"], packing)
        baseline_prompt = _rendered(tokenizer, baseline_messages)
        if (baseline_prompt != row["actual_prompt"]
                or hashlib.sha256(baseline_prompt.encode("utf-8")).hexdigest() != row["prompt_sha256"]
                or _count(tokenizer, baseline_prompt) != row["input_tokens"]):
            raise E37Error(f"Could not reproduce exact E32 prompt: {row['question_id']}")
        choice = resolve(question, row["packed_spans"], docs, config.policy)
        candidate_prompt = baseline_prompt
        reason = "no_confident_new_section"
        if choice is not None:
            if _count(tokenizer, choice["text"]) <= config.policy["maximum_section_tokens"]:
                units = _replacement_units(row["packed_spans"], choice)
                messages, _ = render_parent(question, units, packing)
                trial = _rendered(tokenizer, messages)
                if _count(tokenizer, trial) <= config.policy["maximum_input_tokens"]:
                    candidate_prompt = trial
                    reason = "selected_section" if trial != baseline_prompt else "unchanged_prompt"
                else:
                    reason = "input_budget_fallback"
            else:
                reason = "section_token_budget_fallback"
        selected = None if reason != "selected_section" else {
            k: choice[k] for k in ("document_id", "start", "end", "heading", "kind",
                                   "first_seed_rank", "score", "shared_heading_terms")
        }
        prepared.append({
            "question_id": row["question_id"], "sample_index": index,
            "baseline_prompt_sha256": row["prompt_sha256"],
            "candidate_prompt_sha256": hashlib.sha256(candidate_prompt.encode("utf-8")).hexdigest(),
            "candidate_prompt": candidate_prompt if reason == "selected_section" else None,
            "changed": reason == "selected_section", "reason": reason,
            "selected_section": selected, "answers_used": False,
        })
    path = output / "prepared/results.jsonl"
    save_once(path, prepared, jsonl=True)
    summary = {
        "experiment_id": EXPERIMENT, "sample_size": 120,
        "sample_ids_sha256": config.raw["source"]["sample_ids_sha256"],
        "results_sha256": file_sha256(path),
        "changed_questions": sum(r["changed"] for r in prepared),
        "reasons": {reason: sum(r["reason"] == reason for r in prepared)
                    for reason in sorted({r["reason"] for r in prepared})},
        "answers_used": False,
    }
    save_once(output / "prepared/summary.json", summary)
    return summary


def _prepared(output: Path, config: Config) -> list[dict[str, Any]]:
    path = output / "prepared/results.jsonl"
    summary = json.loads((output / "prepared/summary.json").read_text(encoding="utf-8"))
    rows = _read_jsonl(path)
    ids = [str(r.get("question_id")) for r in rows]
    if (summary.get("experiment_id") != EXPERIMENT or summary.get("results_sha256") != file_sha256(path)
            or summary.get("sample_ids_sha256") != _ids_sha(ids)
            or len(rows) != 120
            or any(r.get("sample_index") != i or r.get("answers_used") is not False
                   or r.get("changed") is not isinstance(r.get("candidate_prompt"), str)
                   for i, r in enumerate(rows))):
        raise E37Error("E37 prepared contexts changed or are incomplete.")
    return rows


def run_worker(*, root: Path, e36: Path, e00: Path, training: Path,
               output: Path, config: Config, rank: int, device: str) -> dict[str, Any]:
    import torch

    if rank not in (0, 1) or device != f"cuda:{rank}" or torch.cuda.device_count() != 2:
        raise E37Error("Choose T4 x2; worker ranks are cuda:0 and cuda:1.")
    checked = _preflight(root, output, config)
    _verify_input_identity(checked, e36, e00)
    prepared = _prepared(output, config)
    indices = [i for i, row in enumerate(prepared) if row["changed"]][rank::2]
    runtime = {name: importlib.metadata.version(name)
               for name in ("torch", "transformers", "peft", "accelerate")}
    if runtime != {k: config.raw["runtime"][k] for k in runtime}:
        raise E37Error(f"Use exact E32 generation runtime: {runtime}")
    model, tokenizer, placement, parameters = load_candidate_generator(
        config=config.e32.e19, training_directory=training, device=device,
    )
    adapter_sha, _ = validate_candidate_adapter(training, config.e32.e19)
    generation_sha = _json_sha256(model.generation_config.to_dict())
    if (adapter_sha != checked["adapter_sha256"]
            or generation_sha != config.raw["runtime"]["generation_config_sha256"]):
        raise E37Error("E19 adapter or E32 generation defaults changed.")
    identity = {
        "code_sha256": code_sha(root), "config_sha256": config.sha,
        "prepared_sha256": file_sha256(output / "prepared/results.jsonl"),
        "sample_ids_sha256": config.raw["source"]["sample_ids_sha256"],
        "e36_rows_sha256": checked["e36_rows_sha256"],
        "adapter_sha256": adapter_sha, "generation_config_sha256": generation_sha,
        "runtime": runtime, "worker_rank": rank, "device": device,
        "device_map": placement, "adapter_parameters": parameters,
        "assigned_indices": indices, "max_new_tokens": 1024,
    }
    identity["identity_sha256"] = _json_sha256(identity)
    folder = output / f"generation/worker-{rank}"
    records = folder / "records"
    state = folder / "state.json"
    records.mkdir(parents=True, exist_ok=True)
    ids = [r["question_id"] for r in prepared]
    done = _load_worker_progress(records=records, state_path=state, identity=identity,
                                 assigned_indices=indices, sample_ids=ids)
    if {p.name for p in records.glob("*.json")} != {f"{i:04d}.json" for i in indices[:done]}:
        raise E37Error("Unexpected or incomplete resumed worker records.")
    for index in indices[:done]:
        old = json.loads((records / f"{index:04d}.json").read_text(encoding="utf-8"))
        if (old.get("candidate_prompt_sha256") != prepared[index]["candidate_prompt_sha256"]
                or old.get("record_sha256") != _json_sha256({k: v for k, v in old.items() if k != "record_sha256"})):
            raise E37Error(f"Changed resumed worker record: {index}")
    for offset, index in enumerate(indices[done:], start=done + 1):
        row = prepared[index]
        prompt = row["candidate_prompt"]
        inputs = {key: value.to(device) for key, value in tokenizer(
            prompt, add_special_tokens=False, return_tensors="pt").items()}
        started = time.perf_counter()
        with torch.inference_mode():
            generated = model.generate(**inputs, do_sample=False, num_beams=1,
                                       repetition_penalty=1.0, no_repeat_ngram_size=0,
                                       max_new_tokens=1024, use_cache=True)
        latency = (time.perf_counter() - started) * 1000
        new_ids = generated[0, inputs["input_ids"].shape[1]:]
        answer = tokenizer.decode(new_ids, skip_special_tokens=True,
                                  clean_up_tokenization_spaces=False).strip()
        if not answer:
            raise E37Error(f"Empty E37 answer: {row['question_id']}")
        eos = model.generation_config.eos_token_id
        eos_ids = set(eos if isinstance(eos, list) else [eos])
        finish = "eos" if len(new_ids) and int(new_ids[-1]) in eos_ids else (
            "length" if len(new_ids) >= 1024 else "other")
        result = {
            "question_id": row["question_id"], "sample_index": index,
            "worker_rank": rank, "worker_identity_sha256": identity["identity_sha256"],
            "candidate_prompt_sha256": row["candidate_prompt_sha256"],
            "answer": answer, "finish_reason": finish,
            "generated_tokens_including_special": len(new_ids),
            "generation_latency_ms": latency,
        }
        result["record_sha256"] = _json_sha256(result)
        from .e02_answer import _atomic_json
        _atomic_json(records / f"{index:04d}.json", result)
        _write_worker_state(state, identity, offset, len(indices))
        LOG.info("e37_generation device=%s completed=%d/%d question_id=%s finish=%s",
                 device, offset, len(indices), row["question_id"], finish)
    return {"worker_rank": rank, "completed": len(indices), "generated": len(indices)}


def finalize(*, root: Path, e36: Path, e00: Path, training: Path,
             output: Path, config: Config) -> dict[str, Any]:
    checked = _preflight(root, output, config)
    _verify_input_identity(checked, e36, e00)
    ensure_nltk_resources(download=False)
    _, source = source_rows(e36, config)
    prepared = _prepared(output, config)
    if [r["question_id"] for r in prepared] != [r["question_id"] for r in source]:
        raise E37Error("Prepared and source question order differs.")
    generated: dict[int, dict[str, Any]] = {}
    for rank in (0, 1):
        folder = output / f"generation/worker-{rank}"
        state = json.loads((folder / "state.json").read_text(encoding="utf-8"))
        identity = state.get("run_identity", {})
        indices = [i for i, row in enumerate(prepared) if row["changed"]][rank::2]
        if (state.get("complete") is not True or state.get("completed_count") != len(indices)
                or state.get("assigned_count") != len(indices)
                or identity.get("assigned_indices") != indices
                or identity.get("prepared_sha256") != file_sha256(output / "prepared/results.jsonl")
                or identity.get("code_sha256") != code_sha(root)
                or identity.get("config_sha256") != config.sha
                or identity.get("identity_sha256")
                != _json_sha256({k: v for k, v in identity.items() if k != "identity_sha256"})):
            raise E37Error("Incomplete or changed E37 generation worker.")
        for index in indices:
            result = json.loads((folder / f"records/{index:04d}.json").read_text(encoding="utf-8"))
            if (result.get("question_id") != prepared[index]["question_id"]
                    or result.get("sample_index") != index
                    or result.get("worker_identity_sha256") != identity["identity_sha256"]
                    or result.get("candidate_prompt_sha256") != prepared[index]["candidate_prompt_sha256"]
                    or result.get("record_sha256") != _json_sha256({k: v for k, v in result.items() if k != "record_sha256"})):
                raise E37Error(f"Changed generated record: {index}")
            generated[index] = result
    if len(generated) != sum(r["changed"] for r in prepared):
        raise E37Error("Missing or duplicate E37 generated answer.")
    rows = []
    deltas = []
    for index, (baseline, selected) in enumerate(zip(source, prepared)):
        answer = baseline["model_answer"]
        finish = baseline["finish_reason"]
        if selected["changed"]:
            raw = generated[index]
            first = trim_repeated_tail(raw["answer"])
            second = trim_long_token_suffix(first["answer"], minimum_block_tokens=32,
                                            maximum_block_tokens=256)
            answer = second["answer"]
            finish = raw["finish_reason"]
        reference = baseline["reference_answer"]
        control = {"meteor": nltk_meteor_score(reference, baseline["model_answer"]),
                   "rouge_l": rouge_l_fmeasure(reference, baseline["model_answer"])}
        candidate = {"meteor": nltk_meteor_score(reference, answer),
                     "rouge_l": rouge_l_fmeasure(reference, answer)}
        if (abs(control["meteor"] - baseline["scores"]["meteor"]) > 1e-10
                or abs(control["rouge_l"] - baseline["scores"]["rouge_l"]) > 1e-10):
            raise E37Error(f"E36 baseline score changed: {baseline['question_id']}")
        deltas.append(candidate["meteor"] - control["meteor"])
        rows.append({"question_id": baseline["question_id"], "sample_index": index,
                     "question": baseline["question"], "reference_answer": reference,
                     "control_answer": baseline["model_answer"], "candidate_answer": answer,
                     "control_scores": control, "candidate_scores": candidate,
                     "changed_prompt": selected["changed"], "selected_section": selected["selected_section"],
                     "finish_reason": finish})
    metrics = {
        "control_meteor": fmean(r["control_scores"]["meteor"] for r in rows),
        "candidate_meteor": fmean(r["candidate_scores"]["meteor"] for r in rows),
        "control_rouge_l": fmean(r["control_scores"]["rouge_l"] for r in rows),
        "candidate_rouge_l": fmean(r["candidate_scores"]["rouge_l"] for r in rows),
        "paired_meteor_delta": fmean(deltas),
        "paired_meteor_95_ci": _bootstrap_ci(deltas, seed="e37-section-vs-e32-dev120", iterations=10000),
        "improved": sum(x > 0 for x in deltas), "worsened": sum(x < 0 for x in deltas),
        "tied": sum(x == 0 for x in deltas),
    }
    if (abs(metrics["control_meteor"] - config.raw["source"]["baseline_meteor"]) > 1e-10
            or abs(metrics["control_rouge_l"] - config.raw["source"]["baseline_rouge_l"]) > 1e-10):
        raise E37Error("Control mean differs from E36 official scorer.")
    save_once(output / "evaluation/results.jsonl", rows, jsonl=True)
    review = sorted(rows, key=lambda r: r["candidate_scores"]["meteor"] - r["control_scores"]["meteor"])
    save_once(output / "review_changed.jsonl", [r for r in review if r["changed_prompt"]], jsonl=True)
    report = {
        "schema_version": "1.0", "experiment_id": EXPERIMENT,
        "sample_size": 120, "generated_questions": len(generated),
        "unchanged_answers_reused": 120 - len(generated),
        "metrics": metrics,
        "selection_reasons": json.loads((output / "prepared/summary.json").read_text(encoding="utf-8"))["reasons"],
        "evidence": {"code_sha256": code_sha(root), "config_sha256": config.sha,
                     "prepared_sha256": file_sha256(output / "prepared/results.jsonl"),
                     "e36_rows_sha256": file_sha256(e36 / "review_all120.jsonl"),
                     "evaluation_sha256": file_sha256(output / "evaluation/results.jsonl")},
        "reference_answers_used_only_after_generation": True,
        "public_read": False, "private_untouched": True,
        "automatic_promotion_allowed": False,
        "warning": "Repeatedly used dev-120: exploratory smoke evidence, not an independent promotion test.",
    }
    save_once(output / "report.json", report)
    return report
