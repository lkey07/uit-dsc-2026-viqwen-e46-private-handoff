"""Reconstruct and audit the exact saved E32 dev-120 prompts; no inference."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean
from typing import Any

from . import e21_parent_context as parent
from . import e32_final_dev120_two_caps as e32
from .corpus import file_sha256
from .e02_answer import _atomic_json, _atomic_jsonl
from .e03_rrf_grid import _json_sha256, _read_jsonl
from .e10_repetition_grid import answer_diagnostics
from .evaluation.official import ensure_nltk_resources, nltk_meteor_score, rouge_l_fmeasure

EXPERIMENT = "E36-e32-failure-audit-dev120-v1"
SOURCE_VARIANT = "e32_parent_max1024_tailtrim_longtoken"


class E36Error(RuntimeError):
    """Raised for changed artifacts, prompt mismatch or mixed audit checkpoints."""


@dataclass(frozen=True)
class Config:
    raw: dict[str, Any]
    path: Path
    e32_config: e32.Config

    @property
    def sha(self) -> str:
        return file_sha256(self.path)


def load_config(root: Path, path: Path) -> Config:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if (set(raw) != {"schema_version", "experiment_id", "source_e32_config_path",
                     "source_e32_config_sha256", "source_e32", "audit", "run_contract"}
            or raw["schema_version"] != "1.0" or raw["experiment_id"] != EXPERIMENT
            or raw["source_e32_config_path"] != "configs/e32-final-dev120-two-caps-tailtrim-v1.json"
            or raw["source_e32_config_sha256"] != "edc55d0776a86fcf3ced46184a0067aee7701978ae2481fbe6ce0e16932a5c17"):
        raise E36Error("E36 configuration identity changed.")
    if raw["source_e32"] != {
        "experiment_id": "E32-final-dev120-two-caps-tailtrim-v1",
        "report_sha256": "f5e3cc2e9dda13e9bad289c2c03276183c4b0ec8dfc405b770b8cd0ba9e2f513",
        "sample_ids_sha256": "ffc390de97c137c571a768e1cc1e59b10be1245ed29529da39725528d05bd3e5",
        "selected_results_sha256": "3a42bc0f71b7b8edaa04b5b4bf537189c6ee81307b7d9262d2e4d8c2787484c5",
        "selected_meteor": 0.5627316657089687,
        "selected_rouge_l": 0.561259109088349,
    } or raw["audit"] != {
        "sample_size": 120, "review_lowest_meteor": 40,
        "reconstruct_all_prompts": True,
        "verify_prompt_sha256_and_token_count": True,
        "manual_categories": [
            "evidence_absent_from_retrieved_parent_candidates",
            "evidence_available_but_not_packed",
            "evidence_in_prompt_but_answer_wrong_or_incomplete",
            "answer_legally_adequate_but_metric_wording_mismatch",
            "output_truncated_or_repeated",
            "uncertain",
        ],
    } or raw["run_contract"] != {
        "offline_cpu_only": True, "no_model_weights_or_generation": True,
        "no_retrieval_or_finetuning": True, "no_external_or_synthetic_data": True,
        "reference_answers_for_posthoc_diagnosis_only": True,
        "no_automatic_retrieval_labels": True,
        "public_not_read": True, "private_untouched": True,
        "no_submission_created": True,
    }:
        raise E36Error("E36 audit contract changed.")
    e32_path = root / raw["source_e32_config_path"]
    if file_sha256(e32_path) != raw["source_e32_config_sha256"]:
        raise E36Error("Pinned E32 config bytes changed.")
    return Config(raw, path, e32.load_config(root, e32_path))


def code_sha(root: Path) -> str:
    paths = [root / "src/uit_dsc_fixed_rag/e36_failure_audit.py",
             root / "scripts/run_e36_failure_audit_kaggle.py"]
    return _json_sha256({path.relative_to(root).as_posix(): file_sha256(path) for path in paths})


def _ids_sha(ids: list[str]) -> str:
    return hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()


def validate_source(root: Path, source: Path, config: Config):
    pin = config.raw["source_e32"]
    report_path = source / "report.json"
    results_path = source / f"evaluation/{SOURCE_VARIANT}/results.jsonl"
    prepared_path = source / "prepared/results.jsonl"
    if not all(path.is_file() for path in (report_path, results_path, prepared_path)):
        raise E36Error("Add the complete saved E32 output, including prepared and selected results.")
    if file_sha256(report_path) != pin["report_sha256"] or file_sha256(results_path) != pin["selected_results_sha256"]:
        raise E36Error("Saved E32 report or selected answers changed.")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if (report.get("experiment_id") != pin["experiment_id"]
            or report.get("sample_ids_sha256") != pin["sample_ids_sha256"]
            or report.get("clean_scored_size") != 120
            or report.get("evidence", {}).get("config_sha256") != config.e32_config.sha
            or report.get("evidence", {}).get("prepared_results_sha256") != file_sha256(prepared_path)
            or report.get("evidence", {}).get("candidate_results_sha256", {}).get(SOURCE_VARIANT)
            != pin["selected_results_sha256"]
            or report.get("metrics", {}).get(SOURCE_VARIANT, {}).get("meteor") != pin["selected_meteor"]
            or report.get("metrics", {}).get(SOURCE_VARIANT, {}).get("rouge_l") != pin["selected_rouge_l"]
            or report.get("public_read") is not False or report.get("private_untouched") is not True):
        raise E36Error("Saved E32 report contract changed.")
    train = root / "artifacts/splits/v1/train.json"
    dev = root / "artifacts/splits/v1/dev.json"
    questions, _, ids, _ = e32.sample(train, dev, config.e32_config)
    if _ids_sha(ids) != pin["sample_ids_sha256"]:
        raise E36Error("Pinned dev-120 sample changed.")
    prepared = e32.load_prepared(source, ids, config.e32_config)
    results = _read_jsonl(results_path)
    if len(results) != 120:
        raise E36Error("Saved E32 selected answer count changed.")
    for index, (qid, row) in enumerate(zip(ids, results)):
        if (row.get("question_id") != qid or row.get("sample_index") != index
                or row.get("variant") != SOURCE_VARIANT
                or not isinstance(row.get("answer"), str) or not row["answer"].strip()
                or row.get("record_sha256") != _json_sha256({k: v for k, v in row.items() if k != "record_sha256"})):
            raise E36Error(f"Changed E32 answer row: {qid}")
    return questions, ids, prepared, results, {
        "source_report_sha256": file_sha256(report_path),
        "source_results_sha256": file_sha256(results_path),
        "source_prepared_sha256": file_sha256(prepared_path),
        "train_sha256": file_sha256(train), "dev_sha256": file_sha256(dev),
    }


def reconstruct_prompt(question: str, prepared: dict[str, Any], row: dict[str, Any],
                       config: Config, tokenizer: Any) -> tuple[str, list[dict[str, Any]], dict[str, Any]]:
    def render(messages):
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )

    def count(value):
        return len(tokenizer(value, add_special_tokens=False)["input_ids"])

    messages, spans, diagnostics = parent.pack(
        question, prepared, config.e32_config.e21,
        lambda items: count(render(items)), count, parent.VARIANTS[1],
    )
    prompt = render(messages)
    if (hashlib.sha256(prompt.encode("utf-8")).hexdigest() != row.get("prompt_sha256")
            or count(prompt) != row.get("input_tokens")
            or len(spans) != row.get("selected_context_count")
            or diagnostics["selected_chunk_ids"] != row.get("selected_chunk_ids")):
        raise E36Error(f"Reconstructed prompt differs from saved E32 generation: {row['question_id']}")
    return prompt, spans, diagnostics


def _candidate_evidence(prepared: dict[str, Any]) -> list[dict[str, Any]]:
    return [{
        "rank": unit["rank"], "document_id": unit["document_id"],
        "article_number": unit["article_number"], "article_title": unit["article_title"],
        "source_title": unit["source_title"], "document_number": unit["document_number"],
        "seed_text": unit["seed"]["text"],
        "expansions": [{"kind": extension["kind"], "text": extension["text"]}
                       for extension in unit["expansions"]],
    } for unit in prepared["units"]]


def _verify_record(record: dict[str, Any], qid: str, identity_sha: str) -> None:
    if (record.get("question_id") != qid
            or record.get("audit_identity_sha256") != identity_sha
            or record.get("prompt_verified") is not True
            or record.get("record_sha256") != _json_sha256({k: v for k, v in record.items() if k != "record_sha256"})):
        raise E36Error(f"Incomplete or mixed E36 checkpoint: {qid}")


def run(*, root: Path, source: Path, output: Path,
        config: Config, tokenizer: Any, transformers_version: str) -> dict[str, Any]:
    ensure_nltk_resources(download=False)
    import nltk
    if nltk.__version__ != "3.7" or transformers_version != "5.16.1":
        raise E36Error("Use nltk==3.7 and transformers==5.16.1 for exact E32 reconstruction.")
    questions, ids, prepared, results, evidence = validate_source(root, source, config)
    identity = {
        "experiment_id": EXPERIMENT, "code_sha256": code_sha(root),
        "config_sha256": config.sha, **evidence,
        "sample_ids_sha256": _ids_sha(ids),
        "transformers_version": transformers_version,
        "tokenizer_model_id": config.e32_config.e19.generator_model_id,
        "tokenizer_revision": config.e32_config.e19.generator_revision,
    }
    identity["identity_sha256"] = _json_sha256(identity)
    identity_path = output / "audit/identity.json"
    if identity_path.is_file():
        old = json.loads(identity_path.read_text(encoding="utf-8"))
        if old != identity:
            raise E36Error("E36 audit identity changed; refusing to merge checkpoints.")
    else:
        _atomic_json(identity_path, identity)
    completed = output / "report.json"
    if completed.is_file():
        old_report = json.loads(completed.read_text(encoding="utf-8"))
        expected_files = {"review_all120.jsonl", "review_low40.jsonl"}
        if (old_report.get("experiment_id") != EXPERIMENT
                or old_report.get("evidence") != identity
                or set(old_report.get("files", {})) != expected_files
                or any(not (output / name).is_file()
                       or file_sha256(output / name) != old_report["files"][name]
                       for name in expected_files)):
            raise E36Error("Completed E36 output changed; refusing to merge or overwrite it.")
        return old_report
    records = []
    for index, (qid, prepared_row, source_row) in enumerate(zip(ids, prepared, results)):
        path = output / f"audit/records/{index:04d}.json"
        if path.is_file():
            record = json.loads(path.read_text(encoding="utf-8"))
            _verify_record(record, qid, identity["identity_sha256"])
        else:
            question = questions[qid]["question"]
            prompt, spans, packing = reconstruct_prompt(
                question, prepared_row, source_row, config, tokenizer,
            )
            reference = questions[qid]["answer"]
            score = {"meteor": nltk_meteor_score(reference, source_row["answer"]),
                     "rouge_l": rouge_l_fmeasure(reference, source_row["answer"])}
            record = {
                "question_id": qid, "sample_index": index,
                "audit_identity_sha256": identity["identity_sha256"],
                "source_record_sha256": source_row["record_sha256"],
                "question": question, "reference_answer": reference,
                "model_answer": source_row["answer"], "scores": score,
                "finish_reason": source_row["finish_reason"],
                "input_tokens": source_row["input_tokens"],
                "output_tokens": source_row["output_tokens"],
                "answer_diagnostics": answer_diagnostics(source_row["answer"]),
                "prompt_verified": True, "prompt_sha256": source_row["prompt_sha256"],
                "actual_prompt": prompt,
                "packed_spans": spans, "packing_diagnostics": packing,
                "retrieved_parent_candidates": _candidate_evidence(prepared_row),
                "manual_cause": None, "manual_evidence_excerpt": None,
                "manual_notes": None,
            }
            record["record_sha256"] = _json_sha256(record)
            _atomic_json(path, record)
        records.append(record)
        if (index + 1) % 5 == 0 or index == 0:
            print(f"E36 audit progress: verified {index + 1}/120, question_id={qid}", flush=True)
    meteor = fmean(record["scores"]["meteor"] for record in records)
    rouge = fmean(record["scores"]["rouge_l"] for record in records)
    pin = config.raw["source_e32"]
    if abs(meteor - pin["selected_meteor"]) > 1e-10 or abs(rouge - pin["selected_rouge_l"]) > 1e-10:
        raise E36Error("Recomputed scorer differs from the pinned E32 result.")
    ranked = sorted(records, key=lambda row: (row["scores"]["meteor"], row["sample_index"]))
    low = ranked[:config.raw["audit"]["review_lowest_meteor"]]
    _atomic_jsonl(output / "review_all120.jsonl", records)
    _atomic_jsonl(output / "review_low40.jsonl", low)
    summary = {
        "schema_version": "1.0", "experiment_id": EXPERIMENT,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "sample_size": 120, "review_low40_ids": [r["question_id"] for r in low],
        "baseline_meteor": meteor, "baseline_rouge_l": rouge,
        "exact_prompts_verified": len(records),
        "automatic_signals": {
            "length_finished_all120": sum(r["finish_reason"] == "length" for r in records),
            "length_finished_low40": sum(r["finish_reason"] == "length" for r in low),
            "duplicate_line_all120": sum(r["answer_diagnostics"]["duplicate_line"] for r in records),
            "duplicate_sentence_all120": sum(r["answer_diagnostics"]["duplicate_sentence"] for r in records),
            "mean_meteor_low40": fmean(r["scores"]["meteor"] for r in low),
        },
        "manual_categories": config.raw["audit"]["manual_categories"],
        "manual_causes_not_inferred": True,
        "evidence": identity,
        "files": {name: file_sha256(output / name) for name in
                  ("review_all120.jsonl", "review_low40.jsonl")},
        "new_generation_performed": False, "retrieval_performed": False,
        "public_read": False, "private_untouched": True,
        "warning": "Repeatedly used dev-120: diagnostic only; do not treat manual categories as retrieval training labels.",
    }
    _atomic_json(output / "report.json", summary)
    return summary
