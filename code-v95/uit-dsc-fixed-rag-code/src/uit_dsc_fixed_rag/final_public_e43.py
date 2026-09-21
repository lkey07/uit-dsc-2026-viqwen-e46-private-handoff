"""Offline partial-loop cleanup of the complete saved E42 public answers."""
from __future__ import annotations

import hashlib
import json
import re
import unicodedata
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean
from typing import Any

from . import final_public_e39 as e39
from . import final_public_e42 as e42
from .corpus import file_sha256
from .e02_answer import _atomic_json, _atomic_jsonl
from .e03_rrf_grid import _json_sha256
from .e10_repetition_grid import answer_diagnostics
from .e13_e08b_max768_tailtrim import trim_repeated_tail
from .e31_long_token_suffix_trim import trim_long_token_suffix
from .final_public import (FinalPublicError, _atomic_bytes,
                           _validate_submission_payload, _write_submission_zip,
                           load_public_questions)


EXPERIMENT = "FINAL-public1000-e43-e42-partial-loop-clean-v1"
VARIANT = "e43_e42_partial_loop_fixed_point_clean"
E42_CONFIG_SHA = "0ee51336557a7a671d0ecb146e60390fce3bd78b2a2c4e53bc4dc7adc29edb9c"
_SPACE = re.compile(r"\s+")
_TOKEN = re.compile(r"\S+")
_TERMINAL = re.compile(r"[.!?;…][\"'’”\)\]]*$")
_COMPLETE_SENTENCE = re.compile(r"[^.!?;…]+[.!?;…][\"'’”\)\]]*")


@dataclass(frozen=True)
class Config:
    raw: dict[str, Any]
    path: Path
    e42: e42.Config

    @property
    def sha(self) -> str:
        return file_sha256(self.path)


def load_config(root: Path, path: Path) -> Config:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if (set(raw) != {"schema_version", "experiment_id", "source_e42_config_path",
                     "source_e42_config_sha256", "source_public", "postprocess",
                     "run_contract"}
            or raw.get("schema_version") != "1.0"
            or raw.get("experiment_id") != EXPERIMENT
            or raw.get("source_e42_config_path")
            != "configs/final-public-e42-e41-length-max1536-sharded-v1.json"
            or raw.get("source_e42_config_sha256") != E42_CONFIG_SHA):
        raise FinalPublicError("E43 reviewed root contract changed.")
    source_path = root / raw["source_e42_config_path"]
    if file_sha256(source_path) != E42_CONFIG_SHA:
        raise FinalPublicError("Pinned E42 config changed.")
    if raw["source_public"] != {
        "experiment_id": e42.EXPERIMENT, "sample_size": 1000,
        "max_new_tokens": 1536, "require_complete_saved_output": True,
        "rollback_public_meteor": 0.5795,
    }:
        raise FinalPublicError("E43 source-public contract changed.")
    if raw["postprocess"] != {
        "pipeline": ["exact-partial-repeated-suffix-trim-v1",
                     "conservative-consecutive-tail-block-trim-v1",
                     "exact-consecutive-long-token-suffix-trim-v1"],
        "minimum_partial_whitespace_tokens": 8,
        "maximum_partial_whitespace_tokens": 256,
        "minimum_prior_complete_copies": 2,
        "partial_boundaries": ["line", "sentence"],
        "partial_match": "unicode-nfkc-casefold-collapse-whitespace-exact-prefix",
        "minimum_long_block_tokens": 32, "maximum_long_block_tokens": 256,
        "maximum_fixed_point_passes": 16, "suffix_only": True,
        "ordinary_answers_byte_unchanged": True,
    }:
        raise FinalPublicError("E43 partial-loop policy changed.")
    if raw["run_contract"] != {
        "offline_no_model_load": True, "source_final_answers_reused": True,
        "no_new_generation_or_retrieval": True,
        "same_policy_for_every_question": True, "no_question_id_routing": True,
        "official_public_reference_answers_never_read": True,
        "private_untouched": True, "automatic_promotion_allowed": False,
        "submission_only_official_mapping": True,
        "review_every_changed_answer": True,
    }:
        raise FinalPublicError("E43 offline execution contract changed.")
    return Config(raw=raw, path=path, e42=e42.load_config(root, source_path))


def code_sha(root: Path) -> str:
    paths = sorted((root / "src/uit_dsc_fixed_rag").rglob("*.py"))
    paths.append(root / "scripts/run_final_public_e43_kaggle.py")
    return _json_sha256({p.relative_to(root).as_posix(): file_sha256(p)
                         for p in paths if p.is_file()})


def _normalize(value: str) -> str:
    return _SPACE.sub(" ", unicodedata.normalize("NFKC", value).casefold()).strip()


def _count_tokens(value: str) -> int:
    return len(_TOKEN.findall(value))


def _is_complete(value: str) -> bool:
    return bool(_TERMINAL.search(value.rstrip()))


def _partial_line(value: str) -> tuple[int, str, list[str]] | None:
    spans = list(re.finditer(r"[^\r\n]+", value))
    nonempty = [span for span in spans if span.group(0).strip()]
    if len(nonempty) < 2:
        return None
    last = nonempty[-1]
    candidate = last.group(0).strip()
    if _is_complete(candidate):
        return None
    previous = [span.group(0).strip() for span in nonempty[:-1]
                if _is_complete(span.group(0).strip())]
    return last.start(), candidate, previous


def _partial_sentence(value: str) -> tuple[int, str, list[str]] | None:
    complete = list(_COMPLETE_SENTENCE.finditer(value))
    if not complete:
        return None
    start = complete[-1].end()
    while start < len(value) and value[start].isspace():
        start += 1
    candidate = value[start:].strip()
    if not candidate or _is_complete(candidate):
        return None
    previous = [match.group(0).strip() for match in complete]
    return start, candidate, previous


def trim_partial_repeated_suffix(
    answer: str, *, minimum_tokens: int = 8, maximum_tokens: int = 256,
    minimum_prior_copies: int = 2,
) -> dict[str, Any]:
    """Drop only an incomplete suffix proven to prefix a repeated complete unit."""

    if (minimum_tokens < 2 or maximum_tokens < minimum_tokens
            or minimum_prior_copies < 2):
        raise FinalPublicError("Invalid E43 partial-suffix bounds.")
    for boundary, factory in (("line", _partial_line), ("sentence", _partial_sentence)):
        found = factory(answer)
        if found is None:
            continue
        start, candidate, previous = found
        token_count = _count_tokens(candidate)
        if not minimum_tokens <= token_count <= maximum_tokens:
            continue
        partial = _normalize(candidate)
        matches: dict[str, tuple[str, int]] = {}
        for unit in previous:
            normalized = _normalize(unit)
            if normalized != partial and normalized.startswith(partial):
                original, count = matches.get(normalized, (unit, 0))
                matches[normalized] = (original, count + 1)
        eligible = [(normalized, original, count)
                    for normalized, (original, count) in matches.items()
                    if count >= minimum_prior_copies]
        if not eligible:
            continue
        normalized, original, copies = max(
            eligible, key=lambda item: (item[2], len(item[0])))
        cleaned = answer[:start].rstrip()
        if not cleaned:
            raise FinalPublicError("E43 partial trim would remove the complete answer.")
        return {"answer": cleaned, "changed": True, "boundary": boundary,
                "partial_tokens": token_count, "partial_characters": len(answer) - len(cleaned),
                "prior_complete_copies": copies,
                "matched_complete_sha256": hashlib.sha256(
                    original.encode("utf-8")).hexdigest()}
    return {"answer": answer, "changed": False, "boundary": None,
            "partial_tokens": 0, "partial_characters": 0,
            "prior_complete_copies": 0, "matched_complete_sha256": None}


def clean_answer(answer: str, config: Config) -> tuple[str, dict[str, Any]]:
    policy = config.raw["postprocess"]
    value = answer
    passes: list[dict[str, Any]] = []
    for pass_index in range(1, policy["maximum_fixed_point_passes"] + 1):
        before = value
        partial = trim_partial_repeated_suffix(
            value, minimum_tokens=policy["minimum_partial_whitespace_tokens"],
            maximum_tokens=policy["maximum_partial_whitespace_tokens"],
            minimum_prior_copies=policy["minimum_prior_complete_copies"])
        line = trim_repeated_tail(partial["answer"])
        long = trim_long_token_suffix(
            line["answer"], minimum_block_tokens=policy["minimum_long_block_tokens"],
            maximum_block_tokens=policy["maximum_long_block_tokens"])
        value = long["answer"]
        if not value.strip() or len(value) > len(before):
            raise FinalPublicError("Invalid E43 fixed-point cleanup.")
        changed = value != before
        if changed:
            passes.append({
                "pass": pass_index,
                "partial": {key: partial[key] for key in (
                    "changed", "boundary", "partial_tokens", "partial_characters",
                    "prior_complete_copies", "matched_complete_sha256")},
                "line_sentence": {key: line[key] for key in (
                    "changed", "removed_characters", "removed_line_blocks",
                    "removed_sentence_blocks")},
                "long_token": {key: long[key] for key in (
                    "changed", "removed_characters", "removed_whitespace_tokens",
                    "removed_token_blocks", "matched_block_sizes")},
                "removed_characters": len(before) - len(value),
            })
        if not changed:
            return value, {"changed": value != answer, "passes": passes,
                           "fixed_point_passes": pass_index,
                           "partial_changed": any(p["partial"]["changed"] for p in passes),
                           "line_sentence_changed": any(
                               p["line_sentence"]["changed"] for p in passes),
                           "long_token_changed": any(
                               p["long_token"]["changed"] for p in passes),
                           "removed_characters": len(answer) - len(value)}
    raise FinalPublicError("E43 cleanup did not reach its reviewed fixed point.")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def validate_e42_source(source: Path, public: Path, config: Config) -> tuple[
        list[dict[str, Any]], list[dict[str, Any]], list[str], dict[str, Any]]:
    report_path = source / "report.json"
    raw_path = source / "generation/raw-results.jsonl"
    final_path = source / "generation/results.jsonl"
    submission_path = source / "submission.json"
    archive_path = source / "submission.zip"
    paths = (report_path, raw_path, final_path, submission_path, archive_path)
    if not all(path.is_file() for path in paths):
        raise FinalPublicError("Add the complete saved E42 output, not a partial session.")
    _, ids = load_public_questions(public, config.e42.e41.e40.e33.public)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    stack = report.get("selected_stack", {})
    expected_post = [config.e42.raw["inference"]["postprocess_first"],
                     config.e42.raw["inference"]["postprocess_second"]]
    if (report.get("experiment_id") != e42.EXPERIMENT
            or report.get("sample_size") != 1000
            or report.get("public_reference_answers_read") is not False
            or report.get("private_untouched") is not True
            or report.get("automatic_promotion") is not False
            or report.get("evidence", {}).get("config_sha256") != E42_CONFIG_SHA
            or stack.get("max_new_tokens") != 1536
            or stack.get("new_generation_questions") != 240
            or stack.get("postprocess") != expected_post):
        raise FinalPublicError("Saved E42 report is not the reviewed source.")
    for name, path in (("generation/raw-results.jsonl", raw_path),
                       ("generation/results.jsonl", final_path),
                       ("submission.json", submission_path),
                       ("submission.zip", archive_path)):
        details = report.get("files", {}).get(name, {})
        if (details.get("sha256") != file_sha256(path)
                or details.get("bytes") != path.stat().st_size):
            raise FinalPublicError(f"Saved E42 source file changed: {name}")
    payload = submission_path.read_bytes()
    if payload.startswith(b"\xef\xbb\xbf"):
        raise FinalPublicError("Saved E42 submission has a UTF-8 BOM.")
    _validate_submission_payload(payload, ids)
    with zipfile.ZipFile(archive_path) as archive:
        if archive.namelist() != ["submission.json"] or archive.read("submission.json") != payload:
            raise FinalPublicError("Saved E42 ZIP is not the validated submission.")
    submission = json.loads(payload)
    raw_rows, final_rows = _read_jsonl(raw_path), _read_jsonl(final_path)
    if len(raw_rows) != 1000 or len(final_rows) != 1000:
        raise FinalPublicError("Saved E42 source must contain exactly 1,000 rows.")
    for index, qid in enumerate(ids):
        raw, final = raw_rows[index], final_rows[index]
        if (raw.get("question_id") != qid or raw.get("sample_index") != index
                or final.get("question_id") != qid or final.get("sample_index") != index
                or raw.get("record_sha256") != _json_sha256(
                    {key: value for key, value in raw.items() if key != "record_sha256"})
                or final.get("record_sha256") != _json_sha256(
                    {key: value for key, value in final.items() if key != "record_sha256"})
                or not isinstance(raw.get("answer"), str) or not raw["answer"].strip()
                or not isinstance(final.get("answer"), str) or not final["answer"].strip()
                or submission[qid]["answer"] != final["answer"]):
            raise FinalPublicError(f"Saved E42 row changed: {index}")
        baseline, _ = e39._trim(raw["answer"])
        if baseline != final["answer"]:
            raise FinalPublicError(f"E42 two-trim baseline cannot be reproduced: {index}")
    return raw_rows, final_rows, ids, {
        "source_report_sha256": file_sha256(report_path),
        "source_raw_results_sha256": file_sha256(raw_path),
        "source_results_sha256": file_sha256(final_path),
        "source_submission_sha256": file_sha256(submission_path),
        "source_zip_sha256": file_sha256(archive_path),
    }


def run(*, root: Path, source: Path, public: Path, output: Path,
        config: Config) -> dict[str, Any]:
    raw_rows, baseline_rows, ids, source_evidence = validate_e42_source(
        source, public, config)
    report_path = output / "report.json"
    if report_path.is_file():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if (report.get("experiment_id") != EXPERIMENT
                or report.get("evidence", {}).get("config_sha256") != config.sha
                or report.get("evidence", {}).get("code_sha256") != code_sha(root)
                or any(report.get("evidence", {}).get(key) != value
                       for key, value in source_evidence.items())):
            raise FinalPublicError("Existing E43 output belongs to another run.")
        for name, details in report.get("files", {}).items():
            path = output / name
            if not path.is_file() or file_sha256(path) != details.get("sha256"):
                raise FinalPublicError("Existing E43 output is incomplete or changed.")
        return report
    output.mkdir(parents=True, exist_ok=True)
    derived, changed_review, submission = [], [], {}
    for index, qid in enumerate(ids):
        baseline = baseline_rows[index]["answer"]
        # E43 is a suffix-only derivation of the *submitted E42 answer*.  Starting
        # again from the longer raw generation is not safe: changing the order of
        # the three suffix rules can retain text that E42 had already removed.
        answer, diagnostics = clean_answer(baseline, config)
        if len(answer) > len(baseline) or not baseline.startswith(answer):
            raise FinalPublicError(
                f"E43 cleanup is not a monotonic suffix removal: {index}")
        changed = answer != baseline
        record = {"question_id": qid, "sample_index": index, "variant": VARIANT,
                  "source_raw_record_sha256": raw_rows[index]["record_sha256"],
                  "source_final_record_sha256": baseline_rows[index]["record_sha256"],
                  "source_answer_sha256": hashlib.sha256(
                      baseline.encode("utf-8")).hexdigest(),
                  "answer": answer, "changed_from_e42": changed,
                  "postprocess": diagnostics}
        record["record_sha256"] = _json_sha256(record)
        derived.append(record); submission[qid] = {"answer": answer}
        if changed:
            changed_review.append({
                "question_id": qid, "sample_index": index,
                "finish_reason": raw_rows[index].get("finish_reason"),
                "e42_answer": baseline, "e43_answer": answer,
                "removed_characters": len(baseline) - len(answer),
                "postprocess": diagnostics,
            })
    _atomic_jsonl(output / "derivation/results.jsonl", derived)
    _atomic_jsonl(output / "review_changed.jsonl", changed_review)
    payload = json.dumps(submission, ensure_ascii=False, indent=2).encode("utf-8") + b"\n"
    _validate_submission_payload(payload, ids)
    _atomic_bytes(output / "submission.json", payload)
    _write_submission_zip(output / "submission.zip", "submission.json", payload)
    with zipfile.ZipFile(output / "submission.zip") as archive:
        if archive.namelist() != ["submission.json"] or archive.read("submission.json") != payload:
            raise FinalPublicError("E43 ZIP does not contain the exact validated JSON.")
    before_diag = [answer_diagnostics(row["answer"]) for row in baseline_rows]
    after_diag = [answer_diagnostics(row["answer"]) for row in derived]
    report = {
        "schema_version": "1.0", "experiment_id": EXPERIMENT,
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "sample_size": 1000,
        "source_public_experiment_id": e42.EXPERIMENT,
        "source_public_meteor_operator_reported": None,
        "rollback_public_meteor": 0.5795,
        "new_generation_performed": False, "retrieval_performed": False,
        "postprocessing": config.raw["postprocess"]["pipeline"],
        "diagnostics": {
            "changed_questions": len(changed_review),
            "unchanged_questions": 1000 - len(changed_review),
            "partial_changed_questions": sum(
                row["postprocess"]["partial_changed"] for row in derived),
            "line_sentence_changed_questions": sum(
                row["postprocess"]["line_sentence_changed"] for row in derived),
            "long_token_changed_questions": sum(
                row["postprocess"]["long_token_changed"] for row in derived),
            "total_removed_characters_vs_e42": sum(
                len(before["answer"]) - len(after["answer"])
                for before, after in zip(baseline_rows, derived)),
            "baseline_duplicate_line_rate": fmean(x["duplicate_line"] for x in before_diag),
            "candidate_duplicate_line_rate": fmean(x["duplicate_line"] for x in after_diag),
            "baseline_duplicate_sentence_rate": fmean(
                x["duplicate_sentence"] for x in before_diag),
            "candidate_duplicate_sentence_rate": fmean(
                x["duplicate_sentence"] for x in after_diag),
        },
        "validation": {"all_question_ids_present": True, "all_answers_non_empty": True,
                       "utf8_without_bom": True, "archive_members": ["submission.json"],
                       "official_mapping_schema": {"question_id": {"answer": "string"}}},
        "files": {name: {"sha256": file_sha256(output / name),
                           "bytes": (output / name).stat().st_size}
                  for name in ("derivation/results.jsonl", "review_changed.jsonl",
                               "submission.json", "submission.zip")},
        "evidence": {"config_sha256": config.sha, "code_sha256": code_sha(root),
                     "public_sha256": config.e42.e41.e40.e33.public.section("public")["sha256"],
                     "sample_ids_sha256": config.e42.e41.e40.raw[
                         "source_e33"]["sample_ids_sha256"], **source_evidence},
        "saved_e42_predictions_read": True,
        "official_public_reference_answers_read": False, "private_untouched": True,
        "automatic_promotion": False,
        "warning": "Offline cleanup trial; review every changed answer before submission.",
    }
    _atomic_json(report_path, report)
    return report
