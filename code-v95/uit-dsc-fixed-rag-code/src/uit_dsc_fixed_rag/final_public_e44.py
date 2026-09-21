"""Offline residual suffix-loop cleanup of the complete saved E43 answers."""
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

from . import final_public_e43 as e43
from .corpus import file_sha256
from .e02_answer import _atomic_json, _atomic_jsonl
from .e03_rrf_grid import _json_sha256
from .e10_repetition_grid import answer_diagnostics
from .e13_e08b_max768_tailtrim import trim_repeated_tail
from .e31_long_token_suffix_trim import trim_long_token_suffix
from .final_public import (FinalPublicError, _atomic_bytes,
                           _validate_submission_payload, _write_submission_zip,
                           load_public_questions)


EXPERIMENT = "FINAL-public1000-e44-e43-residual-suffix-clean-v1"
VARIANT = "e44_e43_residual_suffix_clean"
E43_CONFIG_SHA = "85edca19630e2f354fbda924e8df26a8dd66024a32f97ad795d3af37c7538643"
E43_CODE_SHA = "d53ee336ebb55280b570366387eeb3e72ffaa7ff465456f8a2a9893064c84264"
SPACE = re.compile(r"\s+")
TOKEN = re.compile(r"\S+")
TERMINAL = re.compile(r"[.!?;:…][\"'’”\)\]]*$")
STRUCTURED = re.compile(r"^(?:[-+*•]|\d+[.)])\s+")
NUMBERED = re.compile(r"^(\d+)\.\s*(.*)$")


@dataclass(frozen=True)
class Config:
    raw: dict[str, Any]
    path: Path
    e43: e43.Config

    @property
    def sha(self) -> str:
        return file_sha256(self.path)


def load_config(root: Path, path: Path) -> Config:
    raw = json.loads(path.read_text(encoding="utf-8"))
    required = {"schema_version", "experiment_id", "source_e43_config_path",
                "source_e43_config_sha256", "source_public", "postprocess",
                "run_contract"}
    if (set(raw) != required or raw.get("schema_version") != "1.0"
            or raw.get("experiment_id") != EXPERIMENT
            or raw.get("source_e43_config_path")
            != "configs/final-public-e43-e42-partial-loop-clean-v1.json"
            or raw.get("source_e43_config_sha256") != E43_CONFIG_SHA):
        raise FinalPublicError("E44 reviewed root contract changed.")
    e43_path = root / raw["source_e43_config_path"]
    if file_sha256(e43_path) != E43_CONFIG_SHA:
        raise FinalPublicError("Pinned E43 config changed.")
    if raw["source_public"] != {
        "experiment_id": e43.EXPERIMENT, "sample_size": 1000,
        "operator_reported_meteor": 0.5817,
        "require_complete_saved_output": True,
        "config_sha256": E43_CONFIG_SHA, "code_sha256": E43_CODE_SHA,
        "results_sha256": "a106dfbc39f89b0f8fa4f7481d2c3e77707406c8de7edb5edb6848c788367c75",
        "review_sha256": "feed6cf1feb2a242a87788bf068fcc6a5579444751eb01406ddf11cb51aeb7fa",
        "submission_sha256": "a587010d8d717d7153317c58cb00a11d8d72d2c3e86020c658a45bfac3bae7b1",
        "zip_sha256": "d0ae10abc5aa581584d0ec9ae5f5fd232d51f8153e913d4a66bf39748c35937b",
    }:
        raise FinalPublicError("E44 source-public contract changed.")
    if raw["postprocess"] != {
        "pipeline": ["consecutive-numbered-identical-body-suffix-trim-v1",
                     "structured-short-partial-repeated-line-trim-v1",
                     "conservative-consecutive-tail-block-trim-v1",
                     "exact-consecutive-long-token-suffix-trim-max512-v1"],
        "minimum_partial_whitespace_tokens": 2,
        "maximum_partial_whitespace_tokens": 7,
        "minimum_prior_complete_copies": 2,
        "minimum_numbered_body_copies": 3,
        "require_consecutive_numbers": True,
        "allow_final_bare_number": True,
        "require_structured_line_marker": True,
        "require_no_terminal_punctuation": True,
        "partial_match": "unicode-nfkc-casefold-collapse-whitespace-exact-prefix",
        "minimum_long_block_tokens": 32, "maximum_long_block_tokens": 512,
        "maximum_fixed_point_passes": 16, "suffix_only": True,
        "ordinary_answers_byte_unchanged": True,
    }:
        raise FinalPublicError("E44 residual-loop policy changed.")
    if raw["run_contract"] != {
        "offline_no_model_load": True, "source_e43_final_answers_reused": True,
        "no_new_generation_or_retrieval": True,
        "same_policy_for_every_question": True, "no_question_id_routing": True,
        "official_public_reference_answers_never_read": True,
        "private_untouched": True, "automatic_promotion_allowed": False,
        "submission_only_official_mapping": True,
        "review_every_changed_answer": True,
        "monotonic_suffix_removal_only": True,
    }:
        raise FinalPublicError("E44 offline execution contract changed.")
    return Config(raw=raw, path=path, e43=e43.load_config(root, e43_path))


def code_sha(root: Path) -> str:
    paths = sorted((root / "src/uit_dsc_fixed_rag").rglob("*.py"))
    paths.append(root / "scripts/run_final_public_e44_kaggle.py")
    return _json_sha256({path.relative_to(root).as_posix(): file_sha256(path)
                         for path in paths if path.is_file()})


def _normalize(value: str) -> str:
    return SPACE.sub(" ", unicodedata.normalize("NFKC", value).casefold()).strip()


def trim_structured_short_partial(answer: str, config: Config) -> dict[str, Any]:
    """Remove a short unfinished list item proven to prefix two prior copies."""

    policy = config.raw["postprocess"]
    spans = [match for match in re.finditer(r"[^\r\n]+", answer)
             if match.group(0).strip()]
    if len(spans) < 3:
        return {"answer": answer, "changed": False, "partial_tokens": 0,
                "prior_copies": 0, "matched_complete_sha256": None}
    last = spans[-1]
    candidate = last.group(0).strip()
    tokens = len(TOKEN.findall(candidate))
    if (not policy["minimum_partial_whitespace_tokens"] <= tokens
            <= policy["maximum_partial_whitespace_tokens"]
            or not STRUCTURED.match(candidate)
            or TERMINAL.search(candidate.rstrip())):
        return {"answer": answer, "changed": False, "partial_tokens": 0,
                "prior_copies": 0, "matched_complete_sha256": None}
    partial = _normalize(candidate)
    matches: dict[str, tuple[str, int]] = {}
    for span in spans[:-1]:
        original = span.group(0).strip()
        normalized = _normalize(original)
        if normalized != partial and normalized.startswith(partial):
            saved, count = matches.get(normalized, (original, 0))
            matches[normalized] = (saved, count + 1)
    eligible = [(normalized, original, copies)
                for normalized, (original, copies) in matches.items()
                if copies >= policy["minimum_prior_complete_copies"]]
    if not eligible:
        return {"answer": answer, "changed": False, "partial_tokens": 0,
                "prior_copies": 0, "matched_complete_sha256": None}
    _, original, copies = max(eligible, key=lambda item: (item[2], len(item[0])))
    cleaned = answer[:last.start()].rstrip()
    if not cleaned or not answer.startswith(cleaned):
        raise FinalPublicError("E44 short-partial trim violated suffix-only cleanup.")
    return {
        "answer": cleaned, "changed": True, "partial_tokens": tokens,
        "prior_copies": copies,
        "matched_complete_sha256": hashlib.sha256(original.encode("utf-8")).hexdigest(),
    }


def trim_numbered_identical_body_suffix(answer: str, config: Config) -> dict[str, Any]:
    """Keep one copy of a suffix run whose number increments but body repeats."""

    minimum = config.raw["postprocess"]["minimum_numbered_body_copies"]
    spans = [match for match in re.finditer(r"[^\r\n]+", answer)
             if match.group(0).strip()]
    if len(spans) < minimum:
        return {"answer": answer, "changed": False, "complete_copies": 0,
                "first_number": None, "last_number": None,
                "removed_bare_number": False, "matched_body_sha256": None}
    parsed = []
    for span in spans:
        match = NUMBERED.match(span.group(0).strip())
        parsed.append((int(match.group(1)), match.group(2).strip()) if match else None)
    cursor = len(parsed) - 1
    bare = bool(parsed[cursor] and not parsed[cursor][1])
    bare_number = parsed[cursor][0] if bare else None
    if bare:
        cursor -= 1
    if cursor < 0 or not parsed[cursor] or not parsed[cursor][1]:
        return {"answer": answer, "changed": False, "complete_copies": 0,
                "first_number": None, "last_number": None,
                "removed_bare_number": False, "matched_body_sha256": None}
    body = _normalize(parsed[cursor][1])
    end = cursor
    start = cursor
    while start > 0:
        previous, current = parsed[start - 1], parsed[start]
        if (not previous or not previous[1]
                or current[0] != previous[0] + 1
                or _normalize(previous[1]) != body):
            break
        start -= 1
    copies = end - start + 1
    if (copies < minimum
            or (bare and bare_number != parsed[end][0] + 1)):
        return {"answer": answer, "changed": False, "complete_copies": 0,
                "first_number": None, "last_number": None,
                "removed_bare_number": False, "matched_body_sha256": None}
    repeated_start = spans[start + 1].start()
    cleaned = answer[:repeated_start].rstrip()
    if not cleaned or not answer.startswith(cleaned):
        raise FinalPublicError("E44 numbered-body trim violated suffix-only cleanup.")
    return {
        "answer": cleaned, "changed": True, "complete_copies": copies,
        "first_number": parsed[start][0], "last_number": parsed[end][0],
        "removed_bare_number": bare,
        "matched_body_sha256": hashlib.sha256(body.encode("utf-8")).hexdigest(),
    }


def clean_answer(answer: str, config: Config) -> tuple[str, dict[str, Any]]:
    policy = config.raw["postprocess"]
    value = answer
    passes: list[dict[str, Any]] = []
    for pass_index in range(1, policy["maximum_fixed_point_passes"] + 1):
        before = value
        numbered = trim_numbered_identical_body_suffix(value, config)
        partial = trim_structured_short_partial(numbered["answer"], config)
        line = trim_repeated_tail(partial["answer"])
        long = trim_long_token_suffix(
            line["answer"],
            minimum_block_tokens=policy["minimum_long_block_tokens"],
            maximum_block_tokens=policy["maximum_long_block_tokens"],
        )
        value = long["answer"]
        if (not value.strip() or len(value) > len(before)
                or not before.startswith(value)):
            raise FinalPublicError("E44 cleanup is not a monotonic suffix removal.")
        changed = value != before
        if changed:
            passes.append({
                "pass": pass_index,
                "numbered_body": {key: numbered[key] for key in (
                    "changed", "complete_copies", "first_number", "last_number",
                    "removed_bare_number", "matched_body_sha256")},
                "short_partial": {key: partial[key] for key in (
                    "changed", "partial_tokens", "prior_copies",
                    "matched_complete_sha256")},
                "line_sentence": {key: line[key] for key in (
                    "changed", "removed_characters", "removed_line_blocks",
                    "removed_sentence_blocks")},
                "long_token": {key: long[key] for key in (
                    "changed", "removed_characters", "removed_whitespace_tokens",
                    "removed_token_blocks", "matched_block_sizes")},
                "removed_characters": len(before) - len(value),
            })
        if not changed:
            return value, {
                "changed": value != answer, "passes": passes,
                "fixed_point_passes": pass_index,
                "short_partial_changed": any(
                    item["short_partial"]["changed"] for item in passes),
                "numbered_body_changed": any(
                    item["numbered_body"]["changed"] for item in passes),
                "line_sentence_changed": any(
                    item["line_sentence"]["changed"] for item in passes),
                "long_token_changed": any(
                    item["long_token"]["changed"] for item in passes),
                "removed_characters": len(answer) - len(value),
            }
    raise FinalPublicError("E44 cleanup did not reach its reviewed fixed point.")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def validate_e43_source(source: Path, public: Path, config: Config) -> tuple[
        list[dict[str, Any]], list[str], dict[str, str]]:
    report_path = source / "report.json"
    results_path = source / "derivation/results.jsonl"
    review_path = source / "review_changed.jsonl"
    submission_path = source / "submission.json"
    archive_path = source / "submission.zip"
    paths = (report_path, results_path, review_path, submission_path, archive_path)
    if not all(path.is_file() for path in paths):
        raise FinalPublicError("Add the complete saved corrected E43 output.")
    pin = config.raw["source_public"]
    expected = {
        "derivation/results.jsonl": pin["results_sha256"],
        "review_changed.jsonl": pin["review_sha256"],
        "submission.json": pin["submission_sha256"],
        "submission.zip": pin["zip_sha256"],
    }
    report = json.loads(report_path.read_text(encoding="utf-8"))
    diagnostics = report.get("diagnostics", {})
    if (report.get("experiment_id") != e43.EXPERIMENT
            or report.get("sample_size") != 1000
            or report.get("evidence", {}).get("config_sha256") != E43_CONFIG_SHA
            or report.get("evidence", {}).get("code_sha256") != E43_CODE_SHA
            or report.get("new_generation_performed") is not False
            or report.get("retrieval_performed") is not False
            or report.get("official_public_reference_answers_read") is not False
            or report.get("private_untouched") is not True
            or report.get("automatic_promotion") is not False
            or diagnostics.get("changed_questions") != 27
            or diagnostics.get("total_removed_characters_vs_e42") != 26450
            or diagnostics.get("candidate_duplicate_line_rate") != 0.176
            or diagnostics.get("candidate_duplicate_sentence_rate") != 0.241):
        raise FinalPublicError("Saved E43 report is not the scored corrected source.")
    for name, digest in expected.items():
        path = source / name
        details = report.get("files", {}).get(name, {})
        if (file_sha256(path) != digest or details.get("sha256") != digest
                or details.get("bytes") != path.stat().st_size):
            raise FinalPublicError(f"Saved E43 source file changed: {name}")
    _, ids = load_public_questions(public, config.e43.e42.e41.e40.e33.public)
    payload = submission_path.read_bytes()
    if payload.startswith(b"\xef\xbb\xbf"):
        raise FinalPublicError("Saved E43 submission has a UTF-8 BOM.")
    _validate_submission_payload(payload, ids)
    with zipfile.ZipFile(archive_path) as archive:
        if archive.namelist() != ["submission.json"] or archive.read("submission.json") != payload:
            raise FinalPublicError("Saved E43 ZIP is not the validated submission.")
    submission = json.loads(payload)
    rows = _read_jsonl(results_path)
    if len(rows) != 1000:
        raise FinalPublicError("Saved E43 source must contain exactly 1,000 rows.")
    for index, qid in enumerate(ids):
        row = rows[index]
        if (row.get("question_id") != qid or row.get("sample_index") != index
                or row.get("record_sha256") != _json_sha256(
                    {key: value for key, value in row.items() if key != "record_sha256"})
                or not isinstance(row.get("answer"), str) or not row["answer"].strip()
                or submission[qid]["answer"] != row["answer"]):
            raise FinalPublicError(f"Saved E43 row changed: {index}")
    return rows, ids, {
        "source_report_sha256": file_sha256(report_path),
        "source_results_sha256": file_sha256(results_path),
        "source_review_sha256": file_sha256(review_path),
        "source_submission_sha256": file_sha256(submission_path),
        "source_zip_sha256": file_sha256(archive_path),
    }


def run(*, root: Path, source: Path, public: Path, output: Path,
        config: Config) -> dict[str, Any]:
    baseline_rows, ids, source_evidence = validate_e43_source(source, public, config)
    report_path = output / "report.json"
    current_evidence = {"config_sha256": config.sha, "code_sha256": code_sha(root),
                        **source_evidence}
    if report_path.is_file():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if (report.get("experiment_id") != EXPERIMENT
                or any(report.get("evidence", {}).get(key) != value
                       for key, value in current_evidence.items())):
            raise FinalPublicError("Existing E44 output belongs to another run.")
        for name, details in report.get("files", {}).items():
            path = output / name
            if not path.is_file() or file_sha256(path) != details.get("sha256"):
                raise FinalPublicError("Existing E44 output is incomplete or changed.")
        return report
    output.mkdir(parents=True, exist_ok=True)
    derived, review, submission = [], [], {}
    for index, qid in enumerate(ids):
        baseline = baseline_rows[index]["answer"]
        answer, diagnostics = clean_answer(baseline, config)
        if len(answer) > len(baseline) or not baseline.startswith(answer):
            raise FinalPublicError(f"E44 changed a non-suffix byte: {index}")
        changed = answer != baseline
        record = {
            "question_id": qid, "sample_index": index, "variant": VARIANT,
            "source_record_sha256": baseline_rows[index]["record_sha256"],
            "source_answer_sha256": hashlib.sha256(baseline.encode("utf-8")).hexdigest(),
            "answer": answer, "changed_from_e43": changed,
            "postprocess": diagnostics,
        }
        record["record_sha256"] = _json_sha256(record)
        derived.append(record); submission[qid] = {"answer": answer}
        if changed:
            review.append({
                "question_id": qid, "sample_index": index,
                "e43_answer": baseline, "e44_answer": answer,
                "removed_characters": len(baseline) - len(answer),
                "postprocess": diagnostics,
            })
    _atomic_jsonl(output / "derivation/results.jsonl", derived)
    _atomic_jsonl(output / "review_changed.jsonl", review)
    payload = json.dumps(submission, ensure_ascii=False, indent=2).encode("utf-8") + b"\n"
    _validate_submission_payload(payload, ids)
    _atomic_bytes(output / "submission.json", payload)
    _write_submission_zip(output / "submission.zip", "submission.json", payload)
    with zipfile.ZipFile(output / "submission.zip") as archive:
        if archive.namelist() != ["submission.json"] or archive.read("submission.json") != payload:
            raise FinalPublicError("E44 ZIP does not contain the exact validated JSON.")
    before_diag = [answer_diagnostics(row["answer"]) for row in baseline_rows]
    after_diag = [answer_diagnostics(row["answer"]) for row in derived]
    report = {
        "schema_version": "1.0", "experiment_id": EXPERIMENT,
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "sample_size": 1000,
        "source_public_experiment_id": e43.EXPERIMENT,
        "source_public_meteor_operator_reported": 0.5817,
        "rollback_public_meteor": 0.5817,
        "new_generation_performed": False, "retrieval_performed": False,
        "postprocessing": config.raw["postprocess"]["pipeline"],
        "diagnostics": {
            "changed_questions": len(review), "unchanged_questions": 1000 - len(review),
            "short_partial_changed_questions": sum(
                row["postprocess"]["short_partial_changed"] for row in derived),
            "numbered_body_changed_questions": sum(
                row["postprocess"]["numbered_body_changed"] for row in derived),
            "line_sentence_changed_questions": sum(
                row["postprocess"]["line_sentence_changed"] for row in derived),
            "long_token_changed_questions": sum(
                row["postprocess"]["long_token_changed"] for row in derived),
            "total_removed_characters_vs_e43": sum(
                len(before["answer"]) - len(after["answer"])
                for before, after in zip(baseline_rows, derived)),
            "baseline_duplicate_line_rate": fmean(x["duplicate_line"] for x in before_diag),
            "candidate_duplicate_line_rate": fmean(x["duplicate_line"] for x in after_diag),
            "baseline_duplicate_sentence_rate": fmean(
                x["duplicate_sentence"] for x in before_diag),
            "candidate_duplicate_sentence_rate": fmean(
                x["duplicate_sentence"] for x in after_diag),
        },
        "validation": {"all_question_ids_present": True,
                       "all_answers_non_empty": True, "utf8_without_bom": True,
                       "archive_members": ["submission.json"],
                       "official_mapping_schema": {"question_id": {"answer": "string"}},
                       "all_changed_answers_are_strict_source_prefixes": True},
        "files": {name: {"sha256": file_sha256(output / name),
                           "bytes": (output / name).stat().st_size}
                  for name in ("derivation/results.jsonl", "review_changed.jsonl",
                               "submission.json", "submission.zip")},
        "evidence": {**current_evidence,
                     "public_sha256": config.e43.e42.e41.e40.e33.public.section("public")["sha256"],
                     "sample_ids_sha256": config.e43.e42.e41.e40.raw[
                         "source_e33"]["sample_ids_sha256"]},
        "saved_e43_predictions_read": True,
        "official_public_reference_answers_read": False,
        "private_untouched": True, "automatic_promotion": False,
        "warning": "Residual suffix-only public trial; retain E43 METEOR 0.5817 as rollback.",
    }
    _atomic_json(report_path, report)
    return report
