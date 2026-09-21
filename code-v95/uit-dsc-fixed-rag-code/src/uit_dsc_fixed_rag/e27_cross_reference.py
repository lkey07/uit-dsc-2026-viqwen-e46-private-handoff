"""E27 deterministic cross-reference expansion over the exact E21 parent prompt."""
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
    _write_worker_state,
)
from .e18_source_metadata import save_once, scan_selected, source_number
from .e19_metadata_lora import (
    _metrics,
    load_candidate_generator,
    validate_candidate_adapter,
)
from .evaluation.official import ensure_nltk_resources, nltk_meteor_score, rouge_l_fmeasure
from .sparse_normalization import fold_vietnamese_accents


EXPERIMENT = "E27-cross-reference-expansion-dev200-v1"
CONTROL = "parent_expanded_max704"
VARIANTS = (
    "e21_parent_plus_one_cross_reference_max704",
    "e21_parent_plus_two_cross_references_max704",
)
CODE_VERSION = "0.61.0"
LOG = logging.getLogger(__name__)

_REFERENCE = re.compile(
    r"(?i)(?:\bkhoản\s+(?P<clause>[0-9]+(?:[a-zđ])?)\s*(?:,\s*)?(?:của\s+)?(?=điều\b))?"
    r"\bđiều\s+(?P<article>[0-9]+(?:[a-zđ])?)\b"
)
_DOCUMENT_NUMBER = re.compile(r"\b[0-9]{1,4}/[0-9]{4}/[a-z0-9]+(?:-[a-z0-9]+)*\b")
_DOCUMENT_TYPE = re.compile(r"\b(?:bo luat|luat|nghi dinh|thong tu|quyet dinh|nghi quyet|phap lenh)\b")
_THIS_DOCUMENT = re.compile(
    r"\b(?:van ban|bo luat|luat|nghi dinh|thong tu|quyet dinh|nghi quyet|phap lenh)\s+nay\b"
)


class E27Error(RuntimeError):
    """Raised when reviewed E27 evidence or runtime identity changes."""


@dataclass(frozen=True)
class Config:
    raw: dict[str, Any]
    path: Path
    e21: Any

    @property
    def sha(self) -> str:
        return file_sha256(self.path)

    @property
    def e19(self) -> Any:
        return self.e21.source.source


def load_config(root: Path, path: Path) -> Config:
    raw = json.loads(path.read_text(encoding="utf-8"))
    expected_keys = {
        "schema_version", "experiment_id", "source_e21_config_path",
        "source_e21_config_sha256", "control", "metadata_source",
        "reference_policy", "variants", "inference", "evaluation",
        "parameter_budget", "run_contract",
    }
    if (
        set(raw) != expected_keys
        or raw.get("schema_version") != "1.0"
        or raw.get("experiment_id") != EXPERIMENT
        or raw.get("source_e21_config_path") != "configs/e21-parent-context-dev200-v1.json"
        or raw.get("source_e21_config_sha256")
        != "fe8554d5f222e03a8f60ee181f8cfc4d1570719efaa6c56e2f7559fad1672ade"
    ):
        raise E27Error("E27 config identity changed.")
    e21_path = root / raw["source_e21_config_path"]
    if file_sha256(e21_path) != raw["source_e21_config_sha256"]:
        raise E27Error("Pinned E21 config bytes changed.")
    e21 = parent.load_config(root, e21_path)
    if raw["control"] != {
        "experiment_id": "E21-parent-context-dev200-v1",
        "variant": CONTROL,
        "results_sha256": "cbe3d5e163e1983f031a06c767596fcebd91177e3b90625998a1deed28f6afe4",
        "code_sha256": "e5574861eb34ae3fdee1220d5260109ddd9f84feacea9c1ec46a78df8b5527c8",
        "adapter_sha256": "bfb9d8120337c4013b5dcb0b59c6d03beccfe9f9748e5a01ef62e1de64fb5edd",
        "meteor": 0.5633893466813007,
        "rouge_l": 0.5864053775086489,
        "sample_size": 200,
    }:
        raise E27Error("Pinned E21 control changed.")
    if raw["metadata_source"] != {
        "artifact_version": "e00-v2",
        "manifest_sha256": "04efd3905ad6d2758461587ca68d7d70fa2c568855b78f0c762c7a37ba547b2e",
        "chunks_sha256": "1e48c7762765ac2dd169045e9f5327c5311db3f1da8a6007ef70fff58718e367",
        "documents_sha256": "f2968724e8a25124034b9ff2144427f8853ed37359443bd149ec3832f4c1fed7",
        "policy": "exact-e00-v2-original-spans",
    }:
        raise E27Error("Pinned E00 metadata source changed.")
    if raw["reference_policy"] != {
        "sources": ["question-with-exact-document-number", "actually-selected-e21-parent-spans"],
        "reference_pattern": "optional-khoan-then-dieu",
        "same_document_resolution": "only-when-no-external-document-cue",
        "external_document_resolution": "exact-unique-header-document-number-only",
        "ambiguous_or_weak_reference": "skip",
        "existing_e21_article": "skip",
        "target_span_order": ["whole_article", "both_neighbors", "previous_neighbor", "next_neighbor", "seed"],
        "maximum_reference_tokens": 1200,
        "maximum_reference_characters": 6000,
        "maximum_prepared_candidates_per_question": 32,
        "never_evict_e21_evidence": True,
        "answer_used": False,
    }:
        raise E27Error("E27 reference policy changed.")
    if raw["variants"] != [
        {"key": VARIANTS[0], "worker_rank": 0, "maximum_added_references": 1},
        {"key": VARIANTS[1], "worker_rank": 1, "maximum_added_references": 2},
    ]:
        raise E27Error("E27 variants changed.")
    if raw["inference"] != {
        "base_context": "exact-e21-parent-expanded",
        "max_input_tokens": 8192,
        "max_new_tokens": 704,
        "generator": "e19_metadata_trained_rank8",
        "do_sample": False,
        "num_beams": 1,
        "enable_thinking": False,
        "use_cache": True,
    }:
        raise E27Error("E27 inference changed.")
    if raw["evaluation"] != {
        "sample": "reuse-e19-e21-dev400-600",
        "sample_size": 200,
        "sample_ids_sha256": "dc17d6c9af9e941868c03cfc61fd17cdd1dc55aa5834beb3e75ba0fde32a5c73",
        "questions_per_variant": 200,
        "promotion_allowed": False,
    }:
        raise E27Error("E27 evaluation sample changed.")
    budget = raw["parameter_budget"]
    if (
        budget != {
            "exclusive_limit": 4_000_000_000,
            "generator": 2_274_069_824,
            "adapter_parameter_cap": 50_000_000,
            "maximum_stack_total": 2_324_069_824,
        }
        or budget["maximum_stack_total"] != budget["generator"] + budget["adapter_parameter_cap"]
        or budget["maximum_stack_total"] >= budget["exclusive_limit"]
    ):
        raise E27Error("E27 parameter budget failed.")
    if raw["run_contract"] != {
        "top12_ranking_unchanged": True,
        "e21_parent_expansion_unchanged": True,
        "references_from_question_or_selected_evidence_only": True,
        "dev_answers_scoring_only": True,
        "control_answers_reused_byte_exactly": True,
        "same_e19_adapter_prompt_and_max704": True,
        "no_external_or_synthetic_data": True,
        "no_api_model": True,
        "fixed_non_agentic_rag": True,
        "holdout_untouched": True,
        "public_not_read": True,
        "checkpoint_after_each_question_id": True,
        "resume_fail_closed": True,
    }:
        raise E27Error("E27 run contract changed.")
    return Config(raw=raw, path=path, e21=e21)


def code_sha(root: Path) -> str:
    paths = sorted((root / "src/uit_dsc_fixed_rag").rglob("*.py"))
    paths.append(root / "scripts/run_e27_cross_reference_kaggle.py")
    return _json_sha256({
        path.relative_to(root).as_posix(): file_sha256(path)
        for path in paths if path.is_file()
    })


def sample(train: Path, dev: Path, config: Config):
    return parent.sample(train, dev, config.e21)


def _ids_sha(ids: list[str]) -> str:
    return hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()


def _normalize_number(value: str | None) -> str | None:
    if not value:
        return None
    return fold_vietnamese_accents(value).strip().rstrip(".-")


def _normalize_article(value: str | None) -> str | None:
    if not value:
        return None
    return fold_vietnamese_accents(value).strip().rstrip(".:")


def extract_references(
    text: str,
    *,
    origin: str,
    source_document_id: str | None = None,
    absolute_start: int = 0,
) -> list[dict[str, Any]]:
    """Extract conservative legal references without consulting an answer."""

    if origin not in {"question", "selected_evidence"}:
        raise E27Error("Unknown reference origin.")
    references = []
    global_document_numbers = list(dict.fromkeys(
        _DOCUMENT_NUMBER.findall(fold_vietnamese_accents(text))
    ))
    for match in _REFERENCE.finditer(text):
        window_start = max(0, match.start() - 120)
        window_end = min(len(text), match.end() + 180)
        folded_window = fold_vietnamese_accents(text[window_start:window_end])
        document_numbers = list(dict.fromkeys(_DOCUMENT_NUMBER.findall(folded_window)))
        if origin == "question" and not document_numbers:
            document_numbers = global_document_numbers
        if len(document_numbers) > 1:
            resolution = "ambiguous_document_numbers"
            document_number = None
        else:
            document_number = document_numbers[0] if document_numbers else None
            after = fold_vietnamese_accents(text[match.end():min(len(text), match.end() + 100)])
            has_external_cue = bool(_DOCUMENT_TYPE.search(after)) and not bool(_THIS_DOCUMENT.search(after))
            if document_number:
                resolution = "exact_document_number"
            elif origin == "selected_evidence" and source_document_id and not has_external_cue:
                resolution = "same_document"
            elif has_external_cue:
                resolution = "unresolved_external_document"
            else:
                resolution = "question_without_exact_document_number"
        trigger = {
            "origin": origin,
            "source_document_id": source_document_id,
            "start": absolute_start + match.start(),
            "end": absolute_start + match.end(),
            "text": match.group(0),
        }
        references.append({
            "article_number": _normalize_article(match.group("article")),
            "clause_number": _normalize_article(match.group("clause")),
            "document_number": _normalize_number(document_number),
            "resolution": resolution,
            "trigger": trigger,
        })
    return references


def load_e21(directory: Path, ids: list[str], config: Config):
    result_path = directory / "evaluation" / CONTROL / "results.jsonl"
    state_path = result_path.parent / "state.json"
    if (
        not result_path.is_file()
        or not state_path.is_file()
        or file_sha256(result_path) != config.raw["control"]["results_sha256"]
    ):
        raise E27Error("Add the full byte-exact E21 output.")
    rows = _read_jsonl(result_path)
    state = json.loads(state_path.read_text(encoding="utf-8"))
    identity = state.get("run_identity", {})
    if (
        len(rows) != 200
        or state.get("complete") is not True
        or state.get("completed_count") != 200
        or state.get("assigned_count") != 200
        or identity.get("worker_rank") != 1
        or identity.get("variant") != CONTROL
        or identity.get("device") != "cuda:1"
        or identity.get("config_sha256") != config.raw["source_e21_config_sha256"]
        or identity.get("code_sha256") != config.raw["control"]["code_sha256"]
        or identity.get("adapter_sha256") != config.raw["control"]["adapter_sha256"]
        or identity.get("identity_sha256")
        != _json_sha256({key: value for key, value in identity.items() if key != "identity_sha256"})
    ):
        raise E27Error("E21 control state changed.")
    for index, question_id in enumerate(ids):
        parent.validate_record(rows[index], question_id, index, 1, identity)
    prepared = parent.load_prepared(directory, ids, config.e21)
    prepared_path = directory / "prepared/results.jsonl"
    if identity.get("prepared_sha256") != file_sha256(prepared_path):
        raise E27Error("E21 prepared contexts differ from the control worker identity.")
    return rows, identity, prepared


def validate_preflight(
    *, root: Path, e00: Path, e21: Path, training: Path,
    train: Path, dev: Path, output: Path, config: Config,
):
    _, _, ids = sample(train, dev, config)
    controls, control_identity, prepared = load_e21(e21, ids, config)
    if (
        len(controls) != 200
        or len(prepared) != 200
        or _ids_sha(ids) != config.raw["evaluation"]["sample_ids_sha256"]
    ):
        raise E27Error("E27 sample identity changed.")
    for name, key in (
        ("manifest.json", "manifest_sha256"),
        ("chunks.jsonl", "chunks_sha256"),
        ("documents.jsonl", "documents_sha256"),
    ):
        path = e00 / name
        if not path.is_file() or file_sha256(path) != config.raw["metadata_source"][key]:
            raise E27Error(f"Pinned E00 artifact changed: {name}")
    adapter_sha, complete = validate_candidate_adapter(training, config.e19)
    if adapter_sha != config.raw["control"]["adapter_sha256"]:
        raise E27Error("E27 requires the exact E19 adapter used by E21.")
    payload = {
        "schema_version": "1.0",
        "experiment_id": EXPERIMENT,
        "code_sha256": code_sha(root),
        "config_sha256": config.sha,
        "sample_ids_sha256": _ids_sha(ids),
        "sample_size": 200,
        "e21_control_identity_sha256": control_identity["identity_sha256"],
        "e21_control_results_sha256": config.raw["control"]["results_sha256"],
        "e21_prepared_sha256": control_identity["prepared_sha256"],
        "adapter_sha256": adapter_sha,
        "adapter_identity_sha256": complete["identity_sha256"],
        "metadata_source": config.raw["metadata_source"],
        "variants": list(VARIANTS),
        "maximum_stack_parameters": config.raw["parameter_budget"]["maximum_stack_total"],
        "answers_used_for_reference_selection": False,
    }
    save_once(output / "preflight.json", payload)
    return payload


def check_preflight(root: Path, output: Path, config: Config):
    path = output / "preflight.json"
    if not path.is_file():
        raise E27Error("Run E27 preflight first.")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("experiment_id") != EXPERIMENT
        or payload.get("config_sha256") != config.sha
        or payload.get("code_sha256") != code_sha(root)
    ):
        raise E27Error("E27 preflight code/config changed.")
    return payload


def _raw_references(question: str, prepared: dict[str, Any]) -> list[dict[str, Any]]:
    values = extract_references(question, origin="question")
    seen_spans = set()
    for unit in prepared["units"]:
        for span in [unit["seed"], *unit["expansions"]]:
            key = (unit["document_id"], span["start"], span["end"])
            if key in seen_spans:
                continue
            seen_spans.add(key)
            values.extend(extract_references(
                span["text"], origin="selected_evidence",
                source_document_id=unit["document_id"], absolute_start=span["start"],
            ))
    return values


def _scan_document_numbers(path: Path, wanted_numbers: set[str], expected_sha: str):
    number_to_ids = {number: [] for number in wanted_numbers}
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for line in stream:
            digest.update(line)
            if not line.strip():
                raise E27Error("Blank E00 document row.")
            document = json.loads(line)
            number = _normalize_number(source_number(document["cleaned_text"]))
            if number in number_to_ids:
                number_to_ids[number].append(document["document_id"])
    if digest.hexdigest() != expected_sha:
        raise E27Error("E00 documents changed during number resolution.")
    return number_to_ids


def _scan_chunks_by_document(path: Path, wanted_documents: set[str], expected_sha: str):
    by_document = {document_id: [] for document_id in wanted_documents}
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for line in stream:
            digest.update(line)
            if not line.strip():
                raise E27Error("Blank E00 chunk row.")
            chunk = json.loads(line)
            if chunk["document_id"] in by_document:
                by_document[chunk["document_id"]].append(chunk)
    if digest.hexdigest() != expected_sha or any(not rows for rows in by_document.values()):
        raise E27Error("Missing or changed E00 chunks for referenced documents.")
    return by_document


def _article_index(chunks_by_document: dict[str, list[dict[str, Any]]], documents: dict[str, dict[str, Any]]):
    index: dict[str, dict[str, list[list[dict[str, Any]]]]] = {}
    for document_id, chunks in chunks_by_document.items():
        articles: dict[str, list[list[dict[str, Any]]]] = {}
        for block in parent.article_blocks(chunks, documents[document_id]):
            number = _normalize_article(block[0].get("article_number"))
            if number is not None:
                articles.setdefault(number, []).append(block)
        index[document_id] = articles
    return index


def _seed_for_reference(block: list[dict[str, Any]], clause_number: str | None):
    if clause_number is not None:
        matches = [
            chunk for chunk in block
            if clause_number in {_normalize_article(value) for value in chunk.get("clause_numbers", [])}
        ]
        if matches:
            return min(matches, key=lambda chunk: (chunk["chunk_index"], chunk["chunk_id"]))
    return block[0]


def prepare_references(
    *, root: Path, e00: Path, e21: Path, train: Path, dev: Path,
    output: Path, config: Config,
):
    checked = check_preflight(root, output, config)
    questions, _, ids = sample(train, dev, config)
    _, _, base_prepared = load_e21(e21, ids, config)
    raw_by_question = [
        _raw_references(questions[question_id]["question"], base_prepared[index])
        for index, question_id in enumerate(ids)
    ]
    wanted_numbers = {
        reference["document_number"]
        for references in raw_by_question for reference in references
        if reference["resolution"] == "exact_document_number" and reference["document_number"]
    }
    number_to_ids = _scan_document_numbers(
        e00 / "documents.jsonl", wanted_numbers,
        config.raw["metadata_source"]["documents_sha256"],
    )
    original_documents = {
        unit["document_id"] for row in base_prepared for unit in row["units"]
    }
    exact_target_documents = {
        document_ids[0]
        for document_ids in number_to_ids.values() if len(document_ids) == 1
    }
    wanted_documents = original_documents | exact_target_documents
    documents = scan_selected(
        e00 / "documents.jsonl", "document_id", wanted_documents,
        config.raw["metadata_source"]["documents_sha256"],
    )
    chunks_by_document = _scan_chunks_by_document(
        e00 / "chunks.jsonl", wanted_documents,
        config.raw["metadata_source"]["chunks_sha256"],
    )
    articles = _article_index(chunks_by_document, documents)
    maximum = config.raw["reference_policy"]["maximum_prepared_candidates_per_question"]
    rows = []
    counts = []
    for index, question_id in enumerate(ids):
        represented = {
            (unit["document_id"], unit["block_id"])
            for unit in base_prepared[index]["units"]
        }
        candidates: list[dict[str, Any]] = []
        candidate_by_key: dict[tuple[str, str], dict[str, Any]] = {}
        skipped = {
            "ambiguous_or_unresolved": 0,
            "missing_or_ambiguous_article": 0,
            "already_in_e21": 0,
        }
        for reference in raw_by_question[index]:
            if reference["resolution"] == "same_document":
                target_document = reference["trigger"]["source_document_id"]
                resolution = "same_document"
            elif reference["resolution"] == "exact_document_number":
                document_ids = number_to_ids.get(reference["document_number"], [])
                if len(document_ids) != 1:
                    skipped["ambiguous_or_unresolved"] += 1
                    continue
                target_document = document_ids[0]
                resolution = "exact_document_number"
            else:
                skipped["ambiguous_or_unresolved"] += 1
                continue
            matching_blocks = articles.get(target_document, {}).get(reference["article_number"], [])
            if len(matching_blocks) != 1:
                skipped["missing_or_ambiguous_article"] += 1
                continue
            block = matching_blocks[0]
            target_key = (target_document, block[0]["chunk_id"])
            if target_key in represented:
                skipped["already_in_e21"] += 1
                continue
            if target_key in candidate_by_key:
                candidate_by_key[target_key]["triggers"].append(reference["trigger"])
                continue
            seed = _seed_for_reference(block, reference["clause_number"])
            unit = parent.seed_unit(
                seed, 12 + len(candidates), block, documents[target_document], config.e21.policy,
            )
            candidate = {
                "target_document_id": target_document,
                "target_block_id": target_key[1],
                "target_article_number": reference["article_number"],
                "target_clause_number": reference["clause_number"],
                "resolution": resolution,
                "document_number": source_number(documents[target_document]["cleaned_text"]),
                "triggers": [reference["trigger"]],
                "unit": unit,
            }
            candidates.append(candidate)
            candidate_by_key[target_key] = candidate
            if len(candidates) >= maximum:
                break
        row = {
            "question_id": question_id,
            "sample_index": index,
            "answers_used": False,
            "reference_candidates": candidates,
            "raw_reference_count": len(raw_by_question[index]),
            "skipped": skipped,
        }
        rows.append(row)
        counts.append(len(candidates))
    path = output / "prepared/results.jsonl"
    save_once(path, rows, jsonl=True)
    summary = {
        "schema_version": "1.0",
        "experiment_id": EXPERIMENT,
        "sample_size": 200,
        "sample_ids_sha256": _ids_sha(ids),
        "config_sha256": config.sha,
        "e21_prepared_sha256": checked["e21_prepared_sha256"],
        "results_sha256": file_sha256(path),
        "questions_with_candidates": sum(value > 0 for value in counts),
        "candidate_count_total": sum(counts),
        "mean_candidates": fmean(counts),
        "maximum_candidates": max(counts),
        "answers_used": False,
        "variants": list(VARIANTS),
    }
    save_once(output / "prepared/summary.json", summary)
    return summary


def load_prepared(output: Path, ids: list[str], config: Config):
    path = output / "prepared/results.jsonl"
    summary_path = output / "prepared/summary.json"
    if not path.is_file() or not summary_path.is_file():
        raise E27Error("Run E27 prepare-references first.")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    rows = _read_jsonl(path)
    if (
        summary.get("experiment_id") != EXPERIMENT
        or summary.get("config_sha256") != config.sha
        or summary.get("results_sha256") != file_sha256(path)
        or summary.get("answers_used") is not False
        or [row.get("question_id") for row in rows] != ids
        or any(
            row.get("sample_index") != index
            or row.get("answers_used") is not False
            or len(row.get("reference_candidates", [])) > 32
            for index, row in enumerate(rows)
        )
    ):
        raise E27Error("Prepared E27 references changed.")
    return rows, summary


def _eligible(candidate: dict[str, Any], base_spans: list[dict[str, Any]]) -> bool:
    for trigger in candidate["triggers"]:
        if trigger["origin"] == "question":
            return True
        if trigger["origin"] != "selected_evidence":
            raise E27Error("Unknown prepared trigger origin.")
        if any(
            span["document_id"] == trigger["source_document_id"]
            and span["start"] <= trigger["start"] < trigger["end"] <= span["end"]
            for span in base_spans
        ):
            return True
    return False


def _flat_span(unit: dict[str, Any], span: dict[str, Any]) -> dict[str, Any]:
    value = {key: item for key, item in unit.items() if key not in {"seed", "expansions"}}
    value.update({key: item for key, item in span.items() if key != "kind"})
    return value


def pack_cross_references(
    question: str,
    base_prepared: dict[str, Any],
    reference_prepared: dict[str, Any],
    config: Config,
    message_counter,
    text_counter,
    maximum_added: int,
):
    """Preserve E21 spans and append only exact, eligible, fitting references."""

    base_messages, base_spans, base_diagnostics = parent.pack(
        question, base_prepared, config.e21, message_counter, text_counter, parent.VARIANTS[1],
    )
    selected = list(base_spans)
    added = []
    skipped = {"not_in_selected_evidence": 0, "span_too_large": 0, "input_budget": 0}
    policy = config.raw["reference_policy"]
    for candidate_index, candidate in enumerate(reference_prepared["reference_candidates"]):
        if len(added) >= maximum_added:
            break
        if not _eligible(candidate, base_spans):
            skipped["not_in_selected_evidence"] += 1
            continue
        unit = candidate["unit"]
        expansion_by_kind = {value["kind"]: value for value in unit["expansions"]}
        options = [
            unit["seed"] if kind == "seed" else expansion_by_kind.get(kind)
            for kind in policy["target_span_order"]
        ]
        options = [value for value in options if value is not None]
        size_eligible = [
            value for value in options
            if len(value["text"]) <= policy["maximum_reference_characters"]
            and text_counter(value["text"]) <= policy["maximum_reference_tokens"]
        ]
        if not size_eligible:
            skipped["span_too_large"] += 1
            continue
        chosen = None
        for value in size_eligible:
            addition = _flat_span(unit, value)
            trial_messages, trial_spans = parent.render(
                question, [*selected, addition], parent.inference_packing(config.e19),
            )
            if message_counter(trial_messages) <= config.raw["inference"]["max_input_tokens"]:
                if sum(len(span["text"]) for span in trial_spans) <= sum(len(span["text"]) for span in selected):
                    raise E27Error("Cross-reference addition did not add new source text.")
                selected = trial_spans
                chosen = value
                added.append({
                    "candidate_index": candidate_index,
                    "target_document_id": candidate["target_document_id"],
                    "target_block_id": candidate["target_block_id"],
                    "target_article_number": candidate["target_article_number"],
                    "target_clause_number": candidate["target_clause_number"],
                    "resolution": candidate["resolution"],
                    "span_kind": value.get("kind", "seed"),
                    "start": value["start"],
                    "end": value["end"],
                    "characters": len(value["text"]),
                    "tokens": text_counter(value["text"]),
                })
                base_messages = trial_messages
                break
        if chosen is None:
            skipped["input_budget"] += 1
    diagnostics = {
        "base": base_diagnostics,
        "base_span_count": len(base_spans),
        "prepared_candidate_count": len(reference_prepared["reference_candidates"]),
        "eligible_candidate_count": sum(
            _eligible(candidate, base_spans)
            for candidate in reference_prepared["reference_candidates"]
        ),
        "maximum_added_references": maximum_added,
        "added_references": added,
        "skipped": skipped,
        "selected_chunk_ids": sorted({chunk_id for span in selected for chunk_id in span["chunk_ids"]}),
        "body_characters": sum(len(span["text"]) for span in selected),
    }
    return base_messages, selected, diagnostics


def validate_generation_record(row, question_id, index, rank, identity):
    if (
        row.get("question_id") != question_id
        or row.get("sample_index") != index
        or row.get("worker_rank") != rank
        or row.get("variant") != VARIANTS[rank]
        or row.get("worker_identity_sha256") != identity["identity_sha256"]
        or not isinstance(row.get("answer"), str)
        or not row["answer"].strip()
        or row.get("record_sha256")
        != _json_sha256({key: value for key, value in row.items() if key != "record_sha256"})
    ):
        raise E27Error(f"Changed E27 generation record: {rank}/{index}")


def run_worker(
    *, root: Path, e21: Path, training: Path, train: Path, dev: Path,
    output: Path, config: Config, rank: int, device: str,
):
    import torch

    if rank not in (0, 1) or device != f"cuda:{rank}":
        raise E27Error("GPU0=max1 all200; GPU1=max2 all200.")
    checked = check_preflight(root, output, config)
    questions, _, ids = sample(train, dev, config)
    controls, control_identity, base_prepared = load_e21(e21, ids, config)
    references, prepared_summary = load_prepared(output, ids, config)
    runtime = {name: importlib.metadata.version(name) for name in control_identity["runtime"]}
    if runtime != control_identity["runtime"]:
        raise E27Error("Use the exact E21 generation runtime versions.")
    model, tokenizer, placement, parameters = load_candidate_generator(
        config=config.e19, training_directory=training, device=device,
    )
    adapter_sha, _ = validate_candidate_adapter(training, config.e19)
    generation_sha = _json_sha256(model.generation_config.to_dict())
    if adapter_sha != checked["adapter_sha256"] or generation_sha != control_identity["generation_config_sha256"]:
        raise E27Error("E19 adapter or generation defaults differ from E21.")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    def render(messages):
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )

    def count(text):
        return len(tokenizer(text, add_special_tokens=False)["input_ids"])

    maximum_added = config.raw["variants"][rank]["maximum_added_references"]
    identity = {
        "code_sha256": code_sha(root),
        "config_sha256": config.sha,
        "prepared_sha256": prepared_summary["results_sha256"],
        "e21_prepared_sha256": checked["e21_prepared_sha256"],
        "control_identity_sha256": control_identity["identity_sha256"],
        "adapter_sha256": adapter_sha,
        "generation_config_sha256": generation_sha,
        "runtime": runtime,
        "variant": VARIANTS[rank],
        "maximum_added_references": maximum_added,
        "worker_rank": rank,
        "device": device,
        "assigned_indices": list(range(200)),
        "device_map": placement,
        "adapter_parameters": parameters,
        "sample_ids_sha256": _ids_sha(ids),
    }
    identity["identity_sha256"] = _json_sha256(identity)
    folder = output / "evaluation" / VARIANTS[rank]
    records, state = folder / "records", folder / "state.json"
    records.mkdir(parents=True, exist_ok=True)
    assigned = list(range(200))
    done = _load_worker_progress(
        records=records, state_path=state, identity=identity,
        assigned_indices=assigned, sample_ids=ids,
    )
    for index in assigned[:done]:
        validate_generation_record(
            json.loads((records / f"{index:04d}.json").read_text(encoding="utf-8")),
            ids[index], index, rank, identity,
        )
    for index in assigned[done:]:
        question_id = ids[index]
        messages, spans, diagnostics = pack_cross_references(
            questions[question_id]["question"], base_prepared[index], references[index], config,
            lambda value: count(render(value)), count, maximum_added,
        )
        prompt = render(messages)
        base_messages, _, _ = parent.pack(
            questions[question_id]["question"], base_prepared[index], config.e21,
            lambda value: count(render(value)), count, parent.VARIANTS[1],
        )
        base_prompt_sha = hashlib.sha256(render(base_messages).encode("utf-8")).hexdigest()
        if base_prompt_sha != controls[index]["prompt_sha256"]:
            raise E27Error(f"E27 failed to reproduce the exact E21 base prompt: {question_id}")
        inputs = {
            key: value.to(device)
            for key, value in tokenizer(prompt, add_special_tokens=False, return_tensors="pt").items()
        }
        started = time.perf_counter()
        with torch.inference_mode():
            generated = model.generate(
                **inputs, do_sample=False, num_beams=1, max_new_tokens=704, use_cache=True,
            )
        latency = (time.perf_counter() - started) * 1000
        new_ids = generated[0, inputs["input_ids"].shape[1]:]
        answer = tokenizer.decode(
            new_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False,
        ).strip()
        if not answer:
            raise E27Error(f"Empty E27 answer: {question_id}")
        eos = tokenizer.eos_token_id
        eos_ids = set(eos if isinstance(eos, list) else [eos])
        finish = (
            "eos" if len(new_ids) and int(new_ids[-1]) in eos_ids
            else "length" if len(new_ids) >= 704 else "other"
        )
        row = {
            "question_id": question_id,
            "sample_index": index,
            "worker_rank": rank,
            "variant": VARIANTS[rank],
            "worker_identity_sha256": identity["identity_sha256"],
            "answer": answer,
            "input_tokens": count(prompt),
            "output_tokens": count(answer),
            "generated_tokens_including_special": len(new_ids),
            "finish_reason": finish,
            "generation_latency_ms": latency,
            "selected_context_count": len(spans),
            "selected_chunk_ids": diagnostics["selected_chunk_ids"],
            "packing": diagnostics,
            "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "base_prompt_sha256": base_prompt_sha,
            "evidence_spans": [
                {key: value for key, value in span.items() if key != "text"}
                for span in spans
            ],
        }
        row["record_sha256"] = _json_sha256(row)
        _atomic_json(records / f"{index:04d}.json", row)
        _write_worker_state(state, identity, index + 1, 200)
        LOG.info(
            "e27_generation variant=%s device=%s completed=%d total=200 question_id=%s added=%d finish=%s",
            VARIANTS[rank], device, index + 1, question_id,
            len(diagnostics["added_references"]), finish,
        )
    return {"variant": VARIANTS[rank], "completed": 200}


def finalize(
    *, root: Path, e21: Path, training: Path, train: Path, dev: Path,
    output: Path, config: Config,
):
    checked = check_preflight(root, output, config)
    ensure_nltk_resources(download=False)
    questions, _, ids = sample(train, dev, config)
    controls, control_identity, _ = load_e21(e21, ids, config)
    _, prepared_summary = load_prepared(output, ids, config)
    by_variant = {CONTROL: controls}
    for rank, variant in enumerate(VARIANTS):
        folder = output / "evaluation" / variant
        state = json.loads((folder / "state.json").read_text(encoding="utf-8"))
        identity = state.get("run_identity", {})
        if (
            state.get("complete") is not True
            or state.get("completed_count") != 200
            or state.get("assigned_count") != 200
            or identity.get("worker_rank") != rank
            or identity.get("device") != f"cuda:{rank}"
            or identity.get("variant") != variant
            or identity.get("maximum_added_references")
            != config.raw["variants"][rank]["maximum_added_references"]
            or identity.get("code_sha256") != code_sha(root)
            or identity.get("config_sha256") != config.sha
            or identity.get("prepared_sha256") != prepared_summary["results_sha256"]
            or identity.get("e21_prepared_sha256") != checked["e21_prepared_sha256"]
            or identity.get("control_identity_sha256") != control_identity["identity_sha256"]
            or identity.get("adapter_sha256") != checked["adapter_sha256"]
            or identity.get("runtime") != control_identity["runtime"]
            or identity.get("generation_config_sha256") != control_identity["generation_config_sha256"]
            or identity.get("identity_sha256")
            != _json_sha256({key: value for key, value in identity.items() if key != "identity_sha256"})
        ):
            raise E27Error("Incomplete or changed E27 generation state.")
        rows = []
        for index, question_id in enumerate(ids):
            row = json.loads((folder / f"records/{index:04d}.json").read_text(encoding="utf-8"))
            validate_generation_record(row, question_id, index, rank, identity)
            rows.append(row)
        _atomic_jsonl(folder / "results.jsonl", rows)
        by_variant[variant] = rows
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

    def paired(left, right):
        delta = [a["meteor"] - b["meteor"] for a, b in zip(scores[left], scores[right])]
        return {
            "meteor_mean": fmean(delta),
            "meteor_bootstrap_95_ci": _bootstrap_ci(
                delta, seed=f"e27-{left}-{right}", iterations=10000,
            ),
            "improved": sum(value > 0 for value in delta),
            "worsened": sum(value < 0 for value in delta),
            "tied": sum(value == 0 for value in delta),
        }

    _atomic_jsonl(output / "per_question_scores.jsonl", [
        {
            "question_id": question_id,
            "scores": {variant: scores[variant][index] for variant in scores},
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
        "control_variant": CONTROL,
        "metrics": metrics,
        "paired_max1_minus_e21": paired(VARIANTS[0], CONTROL),
        "paired_max2_minus_e21": paired(VARIANTS[1], CONTROL),
        "paired_max2_minus_max1": paired(VARIANTS[1], VARIANTS[0]),
        "reference_diagnostics": {
            "questions_with_prepared_candidates": prepared_summary["questions_with_candidates"],
            "prepared_candidate_count_total": prepared_summary["candidate_count_total"],
            "questions_with_added_references": {
                variant: sum(bool(row["packing"]["added_references"]) for row in by_variant[variant])
                for variant in VARIANTS
            },
            "added_reference_count": {
                variant: sum(len(row["packing"]["added_references"]) for row in by_variant[variant])
                for variant in VARIANTS
            },
            "answers_used": False,
        },
        "smoke_leader": max(metrics, key=lambda variant: metrics[variant]["meteor"]),
        "promotion_allowed": False,
        "public_read": False,
        "holdout_untouched": True,
        "evidence": {
            "config_sha256": config.sha,
            "code_sha256": code_sha(root),
            "control_results_sha256": config.raw["control"]["results_sha256"],
            "prepared_results_sha256": prepared_summary["results_sha256"],
            "candidate_results_sha256": {
                variant: file_sha256(output / f"evaluation/{variant}/results.jsonl")
                for variant in VARIANTS
            },
        },
        "warning": "Repeated dev-200; freeze a material winner before one final untouched-dev121 check.",
    }
    _atomic_json(output / "report.json", report)
    return report
