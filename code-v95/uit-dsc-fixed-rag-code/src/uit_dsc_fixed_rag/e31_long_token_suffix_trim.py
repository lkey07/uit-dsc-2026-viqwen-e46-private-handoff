"""E31 deterministic long-token suffix-loop trimming over frozen E30 answers."""
from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean
from typing import Any

from . import e30_max1280_vs_tailtrim as e30
from .corpus import file_sha256
from .e02_answer import _atomic_json, _atomic_jsonl
from .e03_rrf_grid import _bootstrap_ci, _json_sha256, _read_jsonl
from .e10_repetition_grid import answer_diagnostics
from .e18_source_metadata import save_once
from .evaluation.official import ensure_nltk_resources, nltk_meteor_score, rouge_l_fmeasure


EXPERIMENT = "E31-long-token-suffix-trim-dev200-v1"
SOURCES = (
    "e21_parent_greedy_max1024_tailtrim",
    "e21_parent_greedy_max1280_tailtrim",
)
DERIVED = (
    "e21_parent_greedy_max1024_tailtrim_longtoken",
    "e21_parent_greedy_max1280_tailtrim_longtoken",
)
CODE_VERSION = "0.66.0"
TOKEN = re.compile(r"\S+")


class E31Error(RuntimeError):
    """Raised when frozen E31 inputs or its conservative contract change."""


@dataclass(frozen=True)
class Config:
    raw: dict[str, Any]
    path: Path
    e30: Any

    @property
    def sha(self) -> str:
        return file_sha256(self.path)


def load_config(root: Path, path: Path) -> Config:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if set(raw) != {
        "schema_version", "experiment_id", "source_e30_config_path",
        "source_e30_config_sha256", "sources", "derived_variants",
        "long_token_suffix_policy", "evaluation", "parameter_budget", "run_contract",
    }:
        raise E31Error("Unexpected E31 config keys.")
    if (
        raw["schema_version"] != "1.0"
        or raw["experiment_id"] != EXPERIMENT
        or raw["source_e30_config_path"]
        != "configs/e30-max1280-vs-max1024-tailtrim-dev200-v1.json"
        or raw["source_e30_config_sha256"]
        != "89811542520bd9173f60544ba528ebaa331471fbc58cd402b6e869bde7363bf3"
    ):
        raise E31Error("E31 source identity changed.")
    e30_path = root / raw["source_e30_config_path"]
    if file_sha256(e30_path) != raw["source_e30_config_sha256"]:
        raise E31Error("Pinned E30 config bytes changed.")
    source_expected = [
        {
            "variant": SOURCES[0],
            "results_sha256": "8eb73af28a575f7a78d7845d9d2db92ae8b86d2aeb425e815ca6de811ab2d903",
            "meteor": 0.5669609272600331,
            "max_new_tokens": 1024,
        },
        {
            "variant": SOURCES[1],
            "results_sha256": "406cd40a0b1c6b2f3a7db8c24168f4315bb8024ffd978266edc41edb53444520",
            "meteor": 0.5663080049649737,
            "max_new_tokens": 1280,
        },
    ]
    if raw["sources"] != source_expected:
        raise E31Error("Pinned E30 answer sources changed.")
    if raw["derived_variants"] != [
        {"key": DERIVED[0], "source_variant": SOURCES[0]},
        {"key": DERIVED[1], "source_variant": SOURCES[1]},
    ]:
        raise E31Error("E31 variants changed.")
    if raw["long_token_suffix_policy"] != {
        "method": "exact-consecutive-long-token-suffix-trim-v1",
        "minimum_block_tokens": 32,
        "maximum_block_tokens": 256,
        "minimum_consecutive_copies": 2,
        "normalization": "unicode-nfkc-casefold-whitespace-token",
        "suffix_only": True,
        "remove_only_later_copies": True,
        "no_middle_or_nonconsecutive_dedupe": True,
        "never_rewrite_when_no_exact_suffix_repeat": True,
    }:
        raise E31Error("E31 long-token suffix policy changed.")
    if raw["evaluation"] != {
        "sample": "reuse-e19-e21-dev400-600",
        "sample_size": 200,
        "sample_ids_sha256": "dc17d6c9af9e941868c03cfc61fd17cdd1dc55aa5834beb3e75ba0fde32a5c73",
        "promotion_allowed": False,
    }:
        raise E31Error("E31 evaluation sample changed.")
    budget = raw["parameter_budget"]
    if budget != {
        "exclusive_limit": 4_000_000_000,
        "frozen_generator": 2_274_069_824,
        "adapter_parameter_cap": 50_000_000,
        "postprocessor_parameters": 0,
        "maximum_stack_total": 2_324_069_824,
    } or budget["maximum_stack_total"] >= budget["exclusive_limit"]:
        raise E31Error("E31 parameter budget failed.")
    if raw["run_contract"] != {
        "e30_answers_reused_byte_exactly": True,
        "same_policy_applied_to_max1024_and_max1280": True,
        "no_question_specific_routing": True,
        "same_policy_for_public_and_private": True,
        "reference_answers_scoring_only": True,
        "no_external_or_synthetic_data": True,
        "no_api_or_auxiliary_model": True,
        "fixed_non_agentic_rag": True,
        "holdout_untouched": True,
        "public_not_read": True,
        "offline_no_generation": True,
    }:
        raise E31Error("E31 run contract changed.")
    return Config(raw=raw, path=path, e30=e30.load_config(root, e30_path))


def code_sha(root: Path) -> str:
    paths = sorted((root / "src/uit_dsc_fixed_rag").rglob("*.py"))
    paths.append(root / "scripts/run_e31_long_token_suffix_trim_kaggle.py")
    return _json_sha256({
        path.relative_to(root).as_posix(): file_sha256(path)
        for path in paths if path.is_file()
    })


def sample(train: Path, dev: Path, config: Config):
    return e30.sample(train, dev, config.e30)


def _ids_sha(ids: list[str]) -> str:
    return hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()


def _normalized_tokens(value: str) -> list[tuple[int, int, str]]:
    return [
        (match.start(), match.end(), unicodedata.normalize("NFKC", match.group(0)).casefold())
        for match in TOKEN.finditer(value)
    ]


def trim_long_token_suffix(
    answer: str, *, minimum_block_tokens: int = 32, maximum_block_tokens: int = 256,
) -> dict[str, Any]:
    """Keep the first copy of exact consecutive long token blocks at the suffix."""

    if minimum_block_tokens < 2 or maximum_block_tokens < minimum_block_tokens:
        raise E31Error("Invalid long-token block bounds.")
    original = answer
    value = answer
    removed_blocks = 0
    matched_sizes: list[int] = []
    while True:
        spans = _normalized_tokens(value)
        normalized = [item[2] for item in spans]
        upper = min(maximum_block_tokens, len(spans) // 2)
        matched_size = 0
        for size in range(upper, minimum_block_tokens - 1, -1):
            if normalized[-2 * size:-size] == normalized[-size:]:
                matched_size = size
                break
        if not matched_size:
            break
        repeated_start = spans[-matched_size][0]
        candidate = value[:repeated_start].rstrip()
        if not candidate:
            raise E31Error("Long-token trim unexpectedly removed the complete answer.")
        value = candidate
        removed_blocks += 1
        matched_sizes.append(matched_size)
    return {
        "answer": value,
        "changed": value != original,
        "removed_characters": len(original) - len(value),
        "removed_whitespace_tokens": len(_normalized_tokens(original)) - len(_normalized_tokens(value)),
        "removed_token_blocks": removed_blocks,
        "matched_block_sizes": matched_sizes,
    }


def load_e30(directory: Path, ids: list[str], config: Config):
    report_path = directory / "report.json"
    if not report_path.is_file():
        raise E31Error("Add the complete E30 notebook output/dataset.")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if (
        report.get("experiment_id") != "E30-max1280-vs-max1024-tailtrim-dev200-v1"
        or report.get("sample_size") != 200
        or report.get("evidence", {}).get("config_sha256")
        != config.raw["source_e30_config_sha256"]
    ):
        raise E31Error("E30 report identity changed.")
    loaded = {}
    for source in config.raw["sources"]:
        variant = source["variant"]
        path = directory / f"evaluation/{variant}/results.jsonl"
        evidence_key = "tailtrim1024_results_sha256" if source["max_new_tokens"] == 1024 else "tailtrim1280_results_sha256"
        if (
            not path.is_file()
            or file_sha256(path) != source["results_sha256"]
            or report.get("evidence", {}).get(evidence_key) != source["results_sha256"]
            or report.get("metrics", {}).get(variant, {}).get("meteor") != source["meteor"]
        ):
            raise E31Error(f"E30 source results changed: {variant}")
        rows = _read_jsonl(path)
        if len(rows) != 200:
            raise E31Error(f"E30 source row count changed: {variant}")
        for index, (question_id, row) in enumerate(zip(ids, rows)):
            if (
                row.get("question_id") != question_id
                or row.get("sample_index") != index
                or row.get("variant") != variant
                or not isinstance(row.get("answer"), str)
                or not row["answer"].strip()
                or row.get("record_sha256")
                != _json_sha256({key: value for key, value in row.items() if key != "record_sha256"})
            ):
                raise E31Error(f"Changed E30 source record: {variant}/{index}")
        loaded[variant] = rows
    return loaded


def validate_preflight(
    *, root: Path, e30_artifact: Path, train: Path, dev: Path,
    output: Path, config: Config,
):
    _, _, ids = sample(train, dev, config)
    sources = load_e30(e30_artifact, ids, config)
    if _ids_sha(ids) != config.raw["evaluation"]["sample_ids_sha256"]:
        raise E31Error("E31 sample identity changed.")
    payload = {
        "schema_version": "1.0",
        "experiment_id": EXPERIMENT,
        "code_sha256": code_sha(root),
        "config_sha256": config.sha,
        "sample_ids_sha256": _ids_sha(ids),
        "sample_size": 200,
        "source_results_sha256": {
            item["variant"]: item["results_sha256"] for item in config.raw["sources"]
        },
        "source_row_counts": {variant: len(rows) for variant, rows in sources.items()},
        "offline_no_generation": True,
        "answers_used_during_postprocessing": False,
    }
    save_once(output / "preflight.json", payload)
    return payload


def check_preflight(root: Path, output: Path, config: Config):
    path = output / "preflight.json"
    if not path.is_file():
        raise E31Error("Run E31 preflight first.")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("experiment_id") != EXPERIMENT
        or payload.get("config_sha256") != config.sha
        or payload.get("code_sha256") != code_sha(root)
        or payload.get("offline_no_generation") is not True
    ):
        raise E31Error("E31 preflight code/config changed.")
    return payload


def _metrics(rows, scores):
    diagnostics = [answer_diagnostics(row["answer"]) for row in rows]
    result = {
        "meteor": fmean(item["meteor"] for item in scores),
        "rouge_l": fmean(item["rouge_l"] for item in scores),
        "mean_answer_characters": fmean(len(row["answer"]) for row in rows),
        "mean_whitespace_tokens": fmean(len(_normalized_tokens(row["answer"])) for row in rows),
        "length_finish_rate": fmean(
            row.get("source_finish_reason", row.get("finish_reason")) == "length" for row in rows
        ),
        "duplicate_line_rate": fmean(item["duplicate_line"] for item in diagnostics),
        "duplicate_sentence_rate": fmean(item["duplicate_sentence"] for item in diagnostics),
        "non_sentence_ending_rate": fmean(item["non_sentence_ending"] for item in diagnostics),
    }
    if all("long_token_trim_changed" in row for row in rows):
        result.update({
            "long_token_trim_changed_rate": fmean(row["long_token_trim_changed"] for row in rows),
            "changed_questions": sum(row["long_token_trim_changed"] for row in rows),
            "removed_characters_total": sum(row["long_token_removed_characters"] for row in rows),
            "removed_whitespace_tokens_total": sum(row["long_token_removed_whitespace_tokens"] for row in rows),
        })
    return result


def _paired(scores, left, right):
    meteor = [a["meteor"] - b["meteor"] for a, b in zip(scores[left], scores[right])]
    rouge = [a["rouge_l"] - b["rouge_l"] for a, b in zip(scores[left], scores[right])]
    return {
        "meteor_mean": fmean(meteor),
        "meteor_bootstrap_95_ci": _bootstrap_ci(meteor, seed=f"e31-{left}-{right}-meteor", iterations=10000),
        "rouge_l_mean": fmean(rouge),
        "rouge_l_bootstrap_95_ci": _bootstrap_ci(rouge, seed=f"e31-{left}-{right}-rouge", iterations=10000),
        "improved": sum(value > 0 for value in meteor),
        "worsened": sum(value < 0 for value in meteor),
        "tied": sum(value == 0 for value in meteor),
    }


def finalize(
    *, root: Path, e30_artifact: Path, train: Path, dev: Path,
    output: Path, config: Config,
):
    check_preflight(root, output, config)
    ensure_nltk_resources(download=False)
    questions, _, ids = sample(train, dev, config)
    by_variant = load_e30(e30_artifact, ids, config)
    derived_rows = {}
    for source_variant, derived_variant in zip(SOURCES, DERIVED):
        rows = []
        for source in by_variant[source_variant]:
            trimmed = trim_long_token_suffix(source["answer"])
            row = {
                **source,
                "variant": derived_variant,
                "answer": trimmed["answer"],
                "source_variant": source_variant,
                "long_token_trim_changed": trimmed["changed"],
                "long_token_removed_characters": trimmed["removed_characters"],
                "long_token_removed_whitespace_tokens": trimmed["removed_whitespace_tokens"],
                "long_token_removed_blocks": trimmed["removed_token_blocks"],
                "long_token_matched_block_sizes": trimmed["matched_block_sizes"],
            }
            row.pop("record_sha256", None)
            row["record_sha256"] = _json_sha256(row)
            rows.append(row)
        path = output / f"evaluation/{derived_variant}/results.jsonl"
        _atomic_jsonl(path, rows)
        derived_rows[derived_variant] = rows
    by_variant.update(derived_rows)
    scores = {
        variant: [
            {
                "meteor": nltk_meteor_score(questions[question_id]["answer"], row["answer"]),
                "rouge_l": rouge_l_fmeasure(questions[question_id]["answer"], row["answer"]),
            }
            for question_id, row in zip(ids, rows)
        ]
        for variant, rows in by_variant.items()
    }
    _atomic_jsonl(output / "per_question_scores.jsonl", [
        {
            "question_id": question_id,
            "scores": {variant: scores[variant][index] for variant in scores},
            "long_token_trim": {
                variant: {
                    "changed": derived_rows[variant][index]["long_token_trim_changed"],
                    "removed_characters": derived_rows[variant][index]["long_token_removed_characters"],
                    "removed_whitespace_tokens": derived_rows[variant][index]["long_token_removed_whitespace_tokens"],
                    "matched_block_sizes": derived_rows[variant][index]["long_token_matched_block_sizes"],
                }
                for variant in DERIVED
            },
        }
        for index, question_id in enumerate(ids)
    ])
    metrics = {variant: _metrics(rows, scores[variant]) for variant, rows in by_variant.items()}
    report = {
        "schema_version": "1.0",
        "experiment_id": EXPERIMENT,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "sample_size": 200,
        "sample_scope": "same-repeated-e19-e21-dev400-600",
        "sources": list(SOURCES),
        "derived_variants": list(DERIVED),
        "policy": config.raw["long_token_suffix_policy"],
        "metrics": metrics,
        "paired_longtoken1024_minus_tailtrim1024": _paired(scores, DERIVED[0], SOURCES[0]),
        "paired_longtoken1280_minus_tailtrim1280": _paired(scores, DERIVED[1], SOURCES[1]),
        "paired_longtoken1280_minus_longtoken1024": _paired(scores, DERIVED[1], DERIVED[0]),
        "smoke_leader": max(metrics, key=lambda variant: metrics[variant]["meteor"]),
        "promotion_allowed": False,
        "offline_no_generation": True,
        "answers_used_during_postprocessing": False,
        "public_read": False,
        "holdout_untouched": True,
        "evidence": {
            "config_sha256": config.sha,
            "code_sha256": code_sha(root),
            "source_results_sha256": {
                item["variant"]: item["results_sha256"] for item in config.raw["sources"]
            },
            "derived_results_sha256": {
                variant: file_sha256(output / f"evaluation/{variant}/results.jsonl")
                for variant in DERIVED
            },
        },
        "warning": "Repeated dev-200 smoke evidence; do not treat it as unseen validation.",
    }
    _atomic_json(output / "report.json", report)
    return report
