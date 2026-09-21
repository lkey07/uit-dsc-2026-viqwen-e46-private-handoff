"""Offline public trial: trim the exact saved E23 max704 answers, without inference."""
from __future__ import annotations

import copy
import hashlib
import json
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import final_public_e23 as e23
from . import final_public_e33 as e33
from .corpus import file_sha256
from .e02_answer import _atomic_json, _atomic_jsonl
from .e03_rrf_grid import _json_sha256, _read_jsonl
from .final_public import (
    FinalPublicError, _atomic_bytes, _validate_submission_payload,
    _write_submission_zip, load_public_questions,
)

EXPERIMENT = "FINAL-public1000-e34-e23-max704-two-trims-v1"
VARIANT = "e34_e23_max704_two_suffix_trims"


@dataclass(frozen=True)
class Config:
    raw: dict[str, Any]
    path: Path
    e33_config: e33.Config

    @property
    def sha(self) -> str:
        return file_sha256(self.path)


def load_config(root: Path, path: Path) -> Config:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if (set(raw) != {"schema_version", "experiment_id", "source_e33_config_path",
                     "source_e33_config_sha256", "source_public", "postprocess", "run_contract"}
            or raw.get("schema_version") != "1.0" or raw.get("experiment_id") != EXPERIMENT):
        raise FinalPublicError("E34 experiment contract changed.")
    parent_path = root / raw["source_e33_config_path"]
    if (raw["source_e33_config_path"] != "configs/final-public-e33-parent-max1024-two-trims-v1.json"
            or file_sha256(parent_path) != raw["source_e33_config_sha256"]):
        raise FinalPublicError("Pinned E33 two-trim policy source changed.")
    parent = e33.load_config(root, parent_path)
    if raw["source_public"] != {
        "experiment_id": e23.EXPERIMENT, "variant": e23.VARIANT,
        "sample_size": 1000, "max_new_tokens": 704,
        "operator_reported_public_meteor": 0.5584,
        "operator_reported_public_rouge_l": 0.5727,
        "require_complete_saved_output": True,
    }:
        raise FinalPublicError("E34 must trim only the saved E23 max704 public result.")
    if raw["postprocess"] != {
        "first": "conservative-consecutive-tail-block-trim-v1",
        "second": "exact-consecutive-long-token-suffix-trim-v1",
        "minimum_block_tokens": 32, "maximum_block_tokens": 256,
        "only_exact_consecutive_suffix_repeats": True,
        "ordinary_answers_byte_unchanged": True,
    }:
        raise FinalPublicError("E34 must use the exact E32 two-suffix-trim policy.")
    if raw["run_contract"] != {
        "offline_no_model_load": True, "reuse_exact_e23_public_answers": True,
        "no_new_generation_or_retrieval": True,
        "official_public_reference_answers_never_read": True,
        "private_untouched": True, "no_question_specific_routing": True,
        "automatic_promotion_allowed": False,
        "submission_only_official_mapping": True,
    }:
        raise FinalPublicError("E34 offline-only execution contract changed.")
    return Config(raw, path, parent)


def code_sha(root: Path) -> str:
    paths = sorted((root / "src/uit_dsc_fixed_rag").rglob("*.py"))
    paths.append(root / "scripts/run_final_public_e34_kaggle.py")
    return _json_sha256({p.relative_to(root).as_posix(): file_sha256(p) for p in paths})


def validate_e23_source(source: Path, public: Path, config: Config) -> tuple[list[dict[str, Any]], list[str], dict[str, Any]]:
    _, ids = load_public_questions(public, config.e33_config.source.public)
    report_path = source / "report.json"
    result_path = source / "generation/results.jsonl"
    submission_path = source / "submission.json"
    archive_path = source / "submission.zip"
    if not all(path.is_file() for path in (report_path, result_path, submission_path, archive_path)):
        raise FinalPublicError("Add the complete saved E23 public output/dataset, not just its ZIP.")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    stack = report.get("selected_stack", {})
    evidence = report.get("evidence", {})
    validation = report.get("validation", {})
    if (report.get("experiment_id") != e23.EXPERIMENT or report.get("sample_size") != 1000
            or report.get("public_answers_read") is not False
            or report.get("holdout_untouched") is not True
            or stack.get("generator") != e23.VARIANT or stack.get("max_new_tokens") != 704
            or stack.get("adapter") != "e19-metadata-aware-context-full-train-5636"
            or stack.get("contexts") != "saved-public-top12-compact-merged-plus-bounded-same-article-parent"
            or evidence.get("config_sha256") != config.e33_config.source.sha
            or evidence.get("public_sha256") != config.e33_config.source.public.section("public")["sha256"]
            or evidence.get("sample_ids_sha256") != config.e33_config.source.public.section("public")["sample_ids_sha256"]
            or evidence.get("adapter_sha256") != config.e33_config.source.base.section("full_lora_source")["adapter_sha256"]
            or evidence.get("selection_results_sha256") != config.e33_config.source.raw["selection"]["results_sha256"]
            or validation.get("all_question_ids_present") is not True
            or validation.get("all_answers_non_empty") is not True
            or validation.get("utf8_without_bom") is not True
            or validation.get("archive_members") != ["submission.json"]):
        raise FinalPublicError("Saved E23 report is not the pinned best-public source.")
    for name, path in (("generation/results.jsonl", result_path),
                       ("submission.json", submission_path), ("submission.zip", archive_path)):
        details = report.get("files", {}).get(name, {})
        if (details.get("sha256") != file_sha256(path)
                or details.get("bytes") != path.stat().st_size):
            raise FinalPublicError(f"Saved E23 source file changed: {name}")
    original_bytes = submission_path.read_bytes()
    if original_bytes.startswith(b"\xef\xbb\xbf"):
        raise FinalPublicError("Saved E23 submission has a UTF-8 BOM.")
    _validate_submission_payload(original_bytes, ids)
    with zipfile.ZipFile(archive_path) as archive:
        if archive.namelist() != ["submission.json"] or archive.read("submission.json") != original_bytes:
            raise FinalPublicError("Saved E23 ZIP is not the validated submission.")
    original = json.loads(original_bytes)
    rows = _read_jsonl(result_path)
    if len(rows) != 1000:
        raise FinalPublicError("Saved E23 result count changed.")
    worker_ids = report.get("evidence", {}).get("worker_identity_sha256")
    if not isinstance(worker_ids, list) or len(worker_ids) != 2:
        raise FinalPublicError("Saved E23 worker evidence is incomplete.")
    for rank in range(2):
        state_path = source / f"generation/worker-{rank}-state.json"
        if not state_path.is_file():
            raise FinalPublicError("Add the complete saved E23 output, including worker states.")
        state = json.loads(state_path.read_text(encoding="utf-8"))
        identity = state.get("run_identity", {})
        if (state.get("complete") is not True or state.get("completed_count") != 500
                or state.get("assigned_count") != 500 or identity.get("worker_rank") != rank
                or identity.get("device") != f"cuda:{rank}" or identity.get("variant") != e23.VARIANT
                or identity.get("identity_sha256") != worker_ids[rank]
                or identity.get("identity_sha256") != _json_sha256({k: v for k, v in identity.items() if k != "identity_sha256"})):
            raise FinalPublicError("Saved E23 worker state changed.")
        for index in e23.assigned(rank):
            row = rows[index]
            e23.validate_record(row, ids[index], index, rank, identity)
            if row["answer"] != original[ids[index]]["answer"]:
                raise FinalPublicError(f"E23 ZIP and generated answer differ: {ids[index]}")
    return rows, ids, {
        "source_report_sha256": file_sha256(report_path),
        "source_results_sha256": file_sha256(result_path),
        "source_submission_sha256": file_sha256(submission_path),
        "source_zip_sha256": file_sha256(archive_path),
    }


def run(*, root: Path, source: Path, e32_output: Path, public: Path,
        output: Path, config: Config) -> dict[str, Any]:
    rows, ids, source_evidence = validate_e23_source(source, public, config)
    selected = e33.validate_selection(e32_output, config.e33_config)
    existing = output / "report.json"
    if existing.is_file():
        old = json.loads(existing.read_text(encoding="utf-8"))
        if (old.get("experiment_id") != EXPERIMENT or old.get("evidence", {}).get("config_sha256") != config.sha
                or old.get("evidence", {}).get("code_sha256") != code_sha(root)
                or any(old.get("evidence", {}).get(key) != value for key, value in source_evidence.items())
                or old.get("evidence", {}).get("e32_selected_results_sha256") != selected["candidate_results_sha256"]):
            raise FinalPublicError("Existing E34 output belongs to another run; do not merge it.")
        for name, details in old.get("files", {}).items():
            path = output / name
            if not path.is_file() or file_sha256(path) != details.get("sha256"):
                raise FinalPublicError("Existing E34 output is incomplete or changed.")
        return old
    output.mkdir(parents=True, exist_ok=True)
    derived = []
    submission = {}
    for index, row in enumerate(rows):
        answer, diagnostics = e33.trim_answer(row["answer"])
        if not answer.strip() or len(answer) > len(row["answer"]):
            raise FinalPublicError(f"Invalid E34 trimmed answer: {ids[index]}")
        changed = answer != row["answer"]
        if changed is not (diagnostics["line_sentence_changed"] or diagnostics["long_token_changed"]):
            raise FinalPublicError("E34 trim diagnostics do not match output.")
        record = {
            "question_id": ids[index], "sample_index": index, "variant": VARIANT,
            "source_variant": e23.VARIANT, "source_record_sha256": row["record_sha256"],
            "source_answer_sha256": hashlib.sha256(row["answer"].encode("utf-8")).hexdigest(),
            "answer": answer, "changed": changed, "postprocess": diagnostics,
        }
        record["record_sha256"] = _json_sha256(record)
        derived.append(record)
        submission[ids[index]] = {"answer": answer}
    _atomic_jsonl(output / "derivation/results.jsonl", derived)
    payload = json.dumps(submission, ensure_ascii=False, indent=2).encode("utf-8") + b"\n"
    _validate_submission_payload(payload, ids)
    submission_path = output / "submission.json"
    zip_path = output / "submission.zip"
    _atomic_bytes(submission_path, payload)
    _write_submission_zip(zip_path, "submission.json", payload)
    with zipfile.ZipFile(zip_path) as archive:
        if archive.namelist() != ["submission.json"] or archive.read("submission.json") != payload:
            raise FinalPublicError("E34 ZIP does not contain the exact validated JSON.")
    changed_count = sum(row["changed"] for row in derived)
    report = {
        "schema_version": "1.0", "experiment_id": EXPERIMENT,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "sample_size": 1000,
        "source_public_experiment_id": e23.EXPERIMENT,
        "source_public_max_new_tokens": 704,
        "source_public_meteor_operator_reported": 0.5584,
        "new_generation_performed": False,
        "retrieval_performed": False,
        "postprocessing": [config.raw["postprocess"]["first"], config.raw["postprocess"]["second"]],
        "diagnostics": {
            "changed_questions": changed_count,
            "unchanged_questions": 1000 - changed_count,
            "line_sentence_changed_questions": sum(r["postprocess"]["line_sentence_changed"] for r in derived),
            "long_token_changed_questions": sum(r["postprocess"]["long_token_changed"] for r in derived),
            "total_removed_characters": sum(len(raw["answer"]) - len(final["answer"])
                                            for raw, final in zip(rows, derived)),
            "identical_to_e23_submission": payload == (source / "submission.json").read_bytes(),
        },
        "selection": {"e32_candidate_variant": config.e33_config.raw["selection"]["candidate_variant"],
                      "e32_candidate_results_sha256": selected["candidate_results_sha256"],
                      "automatic_promotion_allowed": False},
        "validation": {"all_question_ids_present": True, "all_answers_non_empty": True,
                       "utf8_without_bom": True, "archive_members": ["submission.json"],
                       "official_mapping_schema": {"question_id": {"answer": "string"}}},
        "files": {
            name: {"sha256": file_sha256(output / name), "bytes": (output / name).stat().st_size}
            for name in ("derivation/results.jsonl", "submission.json", "submission.zip")
        },
        "evidence": {"config_sha256": config.sha, "code_sha256": code_sha(root),
                     "public_sha256": config.e33_config.source.public.section("public")["sha256"],
                     "sample_ids_sha256": config.e33_config.source.public.section("public")["sample_ids_sha256"],
                     **source_evidence,
                     "e32_selected_results_sha256": selected["candidate_results_sha256"]},
        "saved_e23_predictions_read": True,
        "official_public_reference_answers_read": False,
        "private_untouched": True,
        "warning": "Offline public trial only; public METEOR is unknown until submitted. Keep E23 as rollback.",
    }
    _atomic_json(existing, report)
    return report
