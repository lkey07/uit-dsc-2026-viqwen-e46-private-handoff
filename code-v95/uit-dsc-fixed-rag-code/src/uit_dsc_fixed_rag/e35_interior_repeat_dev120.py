"""Offline, dev-only audit of exact interior repetition in saved E32 answers."""
from __future__ import annotations

import hashlib
import json
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean
from typing import Any

from .corpus import file_sha256
from .e02_answer import _atomic_json, _atomic_jsonl
from .e03_rrf_grid import _json_sha256, _read_jsonl
from .e10_repetition_grid import answer_diagnostics
from .e13_e08b_max768_tailtrim import _line_spans, _sentence_spans
from .evaluation.official import ensure_nltk_resources, nltk_meteor_score, rouge_l_fmeasure

EXPERIMENT = "E35-exact-interior-repeat-dev120-v1"
SOURCE_VARIANT = "e32_parent_max1024_tailtrim_longtoken"
VARIANT = "e35_parent_max1024_two_trims_interior_exact"


class E35Error(RuntimeError):
    """Raised when the frozen E35 contract or saved E32 evidence changes."""


@dataclass(frozen=True)
class Config:
    raw: dict[str, Any]
    path: Path

    @property
    def sha(self) -> str:
        return file_sha256(self.path)


def load_config(path: Path) -> Config:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if (set(raw) != {"schema_version", "experiment_id", "source_e32", "official_splits",
                     "interior_rule", "run_contract"}
            or raw["schema_version"] != "1.0" or raw["experiment_id"] != EXPERIMENT):
        raise E35Error("E35 configuration changed.")
    if raw["source_e32"] != {
        "experiment_id": "E32-final-dev120-two-caps-tailtrim-v1",
        "config_sha256": "edc55d0776a86fcf3ced46184a0067aee7701978ae2481fbe6ce0e16932a5c17",
        "code_sha256": "c92c4bc01d516ef99d3fe7019ea03d2e1eadb3ac5c3281fac9ac5d6cd5ad6ab3",
        "sample_ids_sha256": "ffc390de97c137c571a768e1cc1e59b10be1245ed29529da39725528d05bd3e5",
        "candidate_results_sha256": "3a42bc0f71b7b8edaa04b5b4bf537189c6ee81307b7d9262d2e4d8c2787484c5",
        "candidate_meteor": 0.5627316657089687,
        "candidate_rouge_l": 0.561259109088349,
    } or raw["official_splits"] != {
        "train_sha256": "53db63c15f779babe99c3fcfcf4f4ecf864a7eb0e738637b9e717e1fe357d921",
        "dev_sha256": "2209b5e066ee354aae00f3b2b1aa69dbf71097c2ccd354f18b2f60daab128dcb",
    }:
        raise E35Error("Pinned E32 or official split identity changed.")
    if raw["interior_rule"] != {
        "line_and_sentence_blocks": [3, 2, 1],
        "minimum_block_characters": 120,
        "minimum_block_words": 16,
        "minimum_following_characters": 40,
        "minimum_following_words": 8,
        "maximum_deletions_per_answer": 1,
        "matching": "exact-text-after-whitespace-normalization-case-sensitive",
        "only_interior_consecutive_duplicates": True,
        "preserve_first_copy": True,
    }:
        raise E35Error("E35 interior rule changed.")
    if raw["run_contract"] != {
        "dev_only": True, "no_generation_or_retrieval": True,
        "reference_answers_used_only_after_postprocessing": True,
        "no_public_or_private_answers_read": True,
        "no_submission_created": True, "automatic_promotion_allowed": False,
    }:
        raise E35Error("E35 run contract changed.")
    return Config(raw, path)


def _ids_sha(ids: list[str]) -> str:
    return hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()


def code_sha(root: Path) -> str:
    paths = [root / "src/uit_dsc_fixed_rag/e35_interior_repeat_dev120.py",
             root / "scripts/run_e35_interior_repeat_kaggle.py"]
    return _json_sha256({path.relative_to(root).as_posix(): file_sha256(path) for path in paths})


def _normalized(value: str) -> str:
    return " ".join(value.split())


def _question_group(value: str) -> str:
    return " ".join(unicodedata.normalize("NFC", value).casefold().split())


def trim_interior_exact(answer: str, config: Config) -> dict[str, Any]:
    """Remove at most one substantive, adjacent duplicate before a substantive tail."""
    rule = config.raw["interior_rule"]
    candidates: list[dict[str, Any]] = []
    for kind, span_factory in (("line", _line_spans), ("sentence", _sentence_spans)):
        spans = span_factory(answer)
        for size in rule["line_and_sentence_blocks"]:
            for index in range(len(spans) - 2 * size + 1):
                first_start = spans[index][0]
                repeat_start = spans[index + size][0]
                repeat_end = spans[index + 2 * size - 1][1]
                first = _normalized(answer[first_start:repeat_start])
                repeated = _normalized(answer[repeat_start:repeat_end])
                following = answer[repeat_end:].strip()
                if (first != repeated
                        or len(first) < rule["minimum_block_characters"]
                        or len(first.split()) < rule["minimum_block_words"]
                        or len(following) < rule["minimum_following_characters"]
                        or len(following.split()) < rule["minimum_following_words"]):
                    continue
                candidates.append({
                    "kind": kind, "block_size": size,
                    "first_start": first_start, "repeat_start": repeat_start,
                    "repeat_end": repeat_end, "matched_characters": len(first),
                })
    if not candidates:
        return {"answer": answer, "changed": False, "removed_characters": 0,
                "eligible_matches": 0, "match": None}
    # A single deletion is deliberate: do not cascade through legal text.
    candidates.sort(key=lambda item: (-item["matched_characters"], item["repeat_start"], item["kind"]))
    chosen = candidates[0]
    start, end = chosen["repeat_start"], chosen["repeat_end"]
    before, after = answer[:start].rstrip(), answer[end:]
    result = before + (" " if after and not after[0].isspace() else "") + after
    if not result.strip() or len(result) >= len(answer):
        raise E35Error("Interior trim failed to remove one repeated block safely.")
    return {"answer": result, "changed": True,
            "removed_characters": len(answer) - len(result),
            "eligible_matches": len(candidates), "match": chosen}


def validate_source(source: Path, train: Path, dev: Path, config: Config):
    pin = config.raw["source_e32"]
    splits = config.raw["official_splits"]
    if (file_sha256(train) != splits["train_sha256"]
            or file_sha256(dev) != splits["dev_sha256"]):
        raise E35Error("Official train/dev split changed.")
    report_path = source / "report.json"
    result_path = source / f"evaluation/{SOURCE_VARIANT}/results.jsonl"
    if not report_path.is_file() or not result_path.is_file():
        raise E35Error("Add complete saved E32 output/dataset with report and selected results.")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if (report.get("experiment_id") != pin["experiment_id"]
            or report.get("clean_scored_size") != 120
            or report.get("sample_ids_sha256") != pin["sample_ids_sha256"]
            or report.get("evidence", {}).get("config_sha256") != pin["config_sha256"]
            or report.get("evidence", {}).get("code_sha256") != pin["code_sha256"]
            or report.get("evidence", {}).get("candidate_results_sha256", {}).get(SOURCE_VARIANT)
            != pin["candidate_results_sha256"]
            or report.get("metrics", {}).get(SOURCE_VARIANT, {}).get("meteor") != pin["candidate_meteor"]
            or report.get("metrics", {}).get(SOURCE_VARIANT, {}).get("rouge_l") != pin["candidate_rouge_l"]
            or report.get("public_read") is not False
            or report.get("private_untouched") is not True
            or file_sha256(result_path) != pin["candidate_results_sha256"]):
        raise E35Error("Saved E32 report/results do not match the selected dev-120 source.")
    rows = _read_jsonl(result_path)
    ids = [str(row.get("question_id")) for row in rows]
    if len(ids) != 120 or len(set(ids)) != 120 or _ids_sha(ids) != pin["sample_ids_sha256"]:
        raise E35Error("Saved E32 dev-120 IDs changed.")
    for index, (qid, row) in enumerate(zip(ids, rows)):
        digest = row.get("record_sha256")
        if (row.get("sample_index") != index or row.get("variant") != SOURCE_VARIANT
                or not isinstance(row.get("answer"), str) or not row["answer"].strip()
                or digest != _json_sha256({key: value for key, value in row.items() if key != "record_sha256"})):
            raise E35Error(f"Saved E32 record changed: {qid}")
    return rows, ids, {"report_sha256": file_sha256(report_path),
                       "results_sha256": file_sha256(result_path),
                       "train_sha256": file_sha256(train), "dev_sha256": file_sha256(dev)}


def run(*, root: Path, source: Path, train: Path, dev: Path,
        output: Path, config: Config) -> dict[str, Any]:
    rows, ids, evidence = validate_source(source, train, dev, config)
    current_evidence = {"config_sha256": config.sha, "code_sha256": code_sha(root), **evidence}
    existing = output / "report.json"
    if existing.is_file():
        old = json.loads(existing.read_text(encoding="utf-8"))
        if old.get("experiment_id") != EXPERIMENT or old.get("evidence") != current_evidence:
            raise E35Error("Existing E35 output belongs to another run.")
        required = {"evaluation/results.jsonl", "per_question_scores.jsonl", "review_changed.jsonl"}
        if set(old.get("files", {})) != required:
            raise E35Error("Existing E35 output file inventory changed.")
        for name, digest in old["files"].items():
            if not (output / name).is_file() or file_sha256(output / name) != digest:
                raise E35Error("Existing E35 output is incomplete or changed.")
        return old
    # Do not touch dev answers until every candidate has been frozen.
    derived = []
    for index, source_row in enumerate(rows):
        result = trim_interior_exact(source_row["answer"], config)
        derived.append({"question_id": ids[index], "sample_index": index,
                        "variant": VARIANT, "source_variant": SOURCE_VARIANT,
                        "source_record_sha256": source_row["record_sha256"], **result})
    output.mkdir(parents=True, exist_ok=True)
    _atomic_jsonl(output / "evaluation/results.jsonl", derived)
    # Candidate answers are now frozen and persisted; only scoring may inspect references.
    references = json.loads(dev.read_text(encoding="utf-8"))
    training = json.loads(train.read_text(encoding="utf-8"))
    train_groups = {_question_group(item["question"]) for item in training.values()}
    groups = [_question_group(references[qid]["question"]) for qid in ids]
    if len(set(groups)) != 120 or train_groups.intersection(groups):
        raise E35Error("Saved dev questions overlap training by normalized question.")
    ensure_nltk_resources(download=False)
    import nltk
    if nltk.__version__ != "3.7":
        raise E35Error("Official scorer parity requires nltk==3.7.")
    per_question = []
    review = []
    for qid, source_row, candidate in zip(ids, rows, derived):
        reference = references[qid]["answer"]
        original_score = {"meteor": nltk_meteor_score(reference, source_row["answer"]),
                          "rouge_l": rouge_l_fmeasure(reference, source_row["answer"])}
        candidate_score = {"meteor": nltk_meteor_score(reference, candidate["answer"]),
                           "rouge_l": rouge_l_fmeasure(reference, candidate["answer"])}
        entry = {"question_id": qid, "changed": candidate["changed"],
                 "finish_reason": source_row.get("finish_reason"),
                 "baseline": original_score, "candidate": candidate_score,
                 "delta": {key: candidate_score[key] - original_score[key] for key in original_score}}
        per_question.append(entry)
        if candidate["changed"]:
            review.append({**entry, "question": references[qid]["question"],
                           "reference_answer": reference, "baseline_answer": source_row["answer"],
                           "candidate_answer": candidate["answer"], "match": candidate["match"]})
    _atomic_jsonl(output / "per_question_scores.jsonl", per_question)
    _atomic_jsonl(output / "review_changed.jsonl", review)
    baseline = {key: fmean(row["baseline"][key] for row in per_question) for key in ("meteor", "rouge_l")}
    candidate = {key: fmean(row["candidate"][key] for row in per_question) for key in ("meteor", "rouge_l")}
    baseline_diagnostics = [answer_diagnostics(row["answer"]) for row in rows]
    candidate_diagnostics = [answer_diagnostics(row["answer"]) for row in derived]
    if abs(baseline["meteor"] - config.raw["source_e32"]["candidate_meteor"]) > 1e-10 or abs(baseline["rouge_l"] - config.raw["source_e32"]["candidate_rouge_l"]) > 1e-10:
        raise E35Error("Official scorer does not reproduce the pinned E32 baseline.")
    report = {
        "schema_version": "1.0", "experiment_id": EXPERIMENT,
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "sample_size": 120,
        "source_variant": SOURCE_VARIANT, "candidate_variant": VARIANT,
        "baseline": baseline, "candidate": candidate,
        "delta": {key: candidate[key] - baseline[key] for key in baseline},
        "diagnostics": {"changed_questions": len(review), "unchanged_questions": 120 - len(review),
                        "removed_characters": sum(row["removed_characters"] for row in derived),
                        "eligible_matches_total": sum(row["eligible_matches"] for row in derived),
                        "changed_length_finished_questions": sum(
                            source_row.get("finish_reason") == "length" and row["changed"]
                            for source_row, row in zip(rows, derived)),
                        "baseline_duplicate_line_rate": fmean(row["duplicate_line"] for row in baseline_diagnostics),
                        "candidate_duplicate_line_rate": fmean(row["duplicate_line"] for row in candidate_diagnostics),
                        "baseline_duplicate_sentence_rate": fmean(row["duplicate_sentence"] for row in baseline_diagnostics),
                        "candidate_duplicate_sentence_rate": fmean(row["duplicate_sentence"] for row in candidate_diagnostics),
                        "improved_meteor": sum(row["delta"]["meteor"] > 0 for row in per_question),
                        "worsened_meteor": sum(row["delta"]["meteor"] < 0 for row in per_question)},
        "evidence": current_evidence,
        "files": {name: file_sha256(output / name) for name in (
            "evaluation/results.jsonl", "per_question_scores.jsonl", "review_changed.jsonl")},
        "new_generation_performed": False, "retrieval_performed": False,
        "public_read": False, "private_untouched": True,
        "automatic_promotion_allowed": False,
        "warning": "Previously used dev-120: exploratory evidence only; manually review every changed legal answer.",
    }
    _atomic_json(output / "report.json", report)
    return report
