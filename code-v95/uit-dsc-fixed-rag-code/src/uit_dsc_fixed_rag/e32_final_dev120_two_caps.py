"""Final clean dev-120 paired test of E31 max1024/max1280 two-stage tail-trim."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import logging
import time
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean
from typing import Any

from . import e21_parent_context as parent
from . import e30_max1280_vs_tailtrim as e30
from . import e31_long_token_suffix_trim as e31
from .corpus import file_sha256
from .e02_answer import _atomic_json, _atomic_jsonl
from .e03_rrf_grid import _bootstrap_ci, _json_sha256, _load_worker_progress, _read_jsonl, _write_worker_state
from .e08b_context_lora import _load_splits, _validate_context_rows
from .e13_e08b_max768_tailtrim import trim_repeated_tail
from .e18_source_metadata import enrich_context, save_once, scan_selected
from .e19_metadata_lora import _metrics, load_candidate_generator, validate_candidate_adapter
from .evaluation.official import ensure_nltk_resources, nltk_meteor_score, rouge_l_fmeasure


EXPERIMENT = "E32-final-dev120-two-caps-tailtrim-v1"
VARIANTS = (
    {"key": "e32_parent_max1024_tailtrim_longtoken", "raw_key": "e32_parent_max1024_raw", "worker_rank": 0, "max_new_tokens": 1024},
    {"key": "e32_parent_max1280_tailtrim_longtoken", "raw_key": "e32_parent_max1280_raw", "worker_rank": 1, "max_new_tokens": 1280},
)
LOG = logging.getLogger(__name__)


class E32Error(RuntimeError):
    """Raised on altered inputs, leaked split groups or mixed checkpoints."""


@dataclass(frozen=True)
class Config:
    raw: dict[str, Any]
    path: Path
    e31: Any

    @property
    def sha(self) -> str:
        return file_sha256(self.path)

    @property
    def e30(self) -> Any:
        return self.e31.e30

    @property
    def e21(self) -> Any:
        return self.e30.e21

    @property
    def e19(self) -> Any:
        return self.e30.e19


def load_config(root: Path, path: Path) -> Config:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if set(raw) != {
        "schema_version", "experiment_id", "source_e31_config_path", "source_e31_config_sha256",
        "source_e21_control", "variants", "evaluation", "inference", "postprocessing",
        "parameter_budget", "run_contract",
    }:
        raise E32Error("Unexpected E32 config keys.")
    if (
        raw["schema_version"] != "1.0"
        or raw["experiment_id"] != EXPERIMENT
        or raw["source_e31_config_path"] != "configs/e31-long-token-suffix-trim-dev200-v1.json"
        or raw["source_e31_config_sha256"] != "7200e115586fe7f631279c0cd14d2789bda1165316c2ddb12ab2f57db0b3fef7"
    ):
        raise E32Error("E32 source identity changed.")
    source_path = root / raw["source_e31_config_path"]
    if file_sha256(source_path) != raw["source_e31_config_sha256"]:
        raise E32Error("Pinned E31 config bytes changed.")
    e31_config = e31.load_config(root, source_path)
    if raw["source_e21_control"] != {
        "experiment_id": "E21-parent-context-dev200-v1",
        "results_sha256": "cbe3d5e163e1983f031a06c767596fcebd91177e3b90625998a1deed28f6afe4",
        "prepared_sha256": "834afe9ff390db69b0460e0c1bc5896d5a2f6b0969121f0a02b0272e826328b6",
        "adapter_sha256": "bfb9d8120337c4013b5dcb0b59c6d03beccfe9f9748e5a01ef62e1de64fb5edd",
    }:
        raise E32Error("E32 calibration source changed.")
    if raw["variants"] != list(VARIANTS):
        raise E32Error("E32 variant policy changed.")
    if raw["evaluation"] != {
        "sample": "remaining-dev521-indices-400-520-minus-normalized-duplicate",
        "source_sample_size": 121,
        "source_sample_ids_sha256": "3c10ea650d1359000756f3cd47db6fb3ffd34885c770f156221d4650525fae85",
        "excluded_question_id": "142213",
        "excluded_matches_previous_dev_ids": ["25227", "112319"],
        "sample_size": 120,
        "sample_ids_sha256": "ffc390de97c137c571a768e1cc1e59b10be1245ed29529da39725528d05bd3e5",
        "old_dev200_excluded": True,
        "previous_dev200_excluded": True,
        "normalized_question_groups_disjoint_from_train_and_previous_dev": True,
        "questions_per_worker": 120,
        "automatic_promotion_allowed": False,
    }:
        raise E32Error("E32 clean evaluation split changed.")
    if raw["inference"] != {
        "base_context": "exact-e21-parent-expanded", "seed_contexts": 12,
        "max_input_tokens": 8192, "generator": "e19_metadata_trained_rank8",
        "do_sample": False, "num_beams": 1, "repetition_penalty": 1.0,
        "no_repeat_ngram_size": 0, "enable_thinking": False, "use_cache": True,
    }:
        raise E32Error("E32 inference policy changed.")
    if raw["postprocessing"] != {
        "first": "conservative-consecutive-tail-block-trim-v1",
        "second": "exact-consecutive-long-token-suffix-trim-v1",
        "long_token_minimum_block_tokens": 32,
        "long_token_maximum_block_tokens": 256,
        "apply_identically_to_both_variants": True,
    }:
        raise E32Error("E32 postprocessing policy changed.")
    if raw["parameter_budget"] != {
        "exclusive_limit": 4_000_000_000,
        "embedding": 567_754_752,
        "generator": 2_274_069_824,
        "adapter_parameter_cap": 50_000_000,
        "postprocessor_parameters": 0,
        "maximum_stack_total": 2_891_824_576,
    } or raw["parameter_budget"]["maximum_stack_total"] >= 4_000_000_000:
        raise E32Error("E32 parameter budget changed.")
    if raw["run_contract"] != {
        "fixed_non_agentic_rag": True,
        "e08a_ranked_top12_reused_without_search": True,
        "e21_parent_policy_reproduced_and_calibrated": True,
        "e19_adapter_unchanged": True,
        "both_variants_run_on_same_120_clean_ids": True,
        "one_prior_dev_duplicate_excluded_before_generation_and_scoring": True,
        "same_two_tailtrim_rules_for_both_variants": True,
        "no_question_specific_routing": True,
        "same_policy_available_for_public_and_private": True,
        "answers_used_only_during_final_scoring": True,
        "no_external_or_synthetic_data": True,
        "no_api_model": True,
        "public_not_read": True,
        "private_untouched": True,
        "checkpoint_after_each_question_id": True,
        "resume_fail_closed": True,
    }:
        raise E32Error("E32 run contract changed.")
    return Config(raw=raw, path=path, e31=e31_config)


def code_sha(root: Path) -> str:
    paths = sorted((root / "src/uit_dsc_fixed_rag").rglob("*.py"))
    paths.append(root / "scripts/run_e32_final_dev120_kaggle.py")
    return _json_sha256({
        path.relative_to(root).as_posix(): file_sha256(path) for path in paths if path.is_file()
    })


def _ids_sha(ids: list[str]) -> str:
    return hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()


def _normalize_question(question: str) -> str:
    return " ".join(unicodedata.normalize("NFC", question).casefold().split())


def sample(train: Path, dev: Path, config: Config):
    training, train_ids, questions, dev521 = _load_splits(train, dev, config.e19)
    spec = config.raw["evaluation"]
    source121 = dev521[400:521]
    if len(source121) != 121 or _ids_sha(source121) != spec["source_sample_ids_sha256"]:
        raise E32Error("The 121 reserved dev IDs changed.")
    excluded = spec["excluded_question_id"]
    if excluded not in source121:
        raise E32Error("Expected prior-dev duplicate was not found.")
    previous = {
        _normalize_question(row["question"]): [] for question_id, row in questions.items()
        if question_id not in source121
    }
    for question_id, row in questions.items():
        if question_id not in source121:
            previous[_normalize_question(row["question"])].append(question_id)
    collisions = {
        question_id: previous[_normalize_question(questions[question_id]["question"])]
        for question_id in source121
        if _normalize_question(questions[question_id]["question"]) in previous
    }
    if collisions != {excluded: spec["excluded_matches_previous_dev_ids"]}:
        raise E32Error(f"Unexpected normalized-question overlap with previous dev: {collisions}")
    ids = [question_id for question_id in source121 if question_id != excluded]
    groups = [_normalize_question(questions[question_id]["question"]) for question_id in ids]
    training_groups = {_normalize_question(training[question_id]["question"]) for question_id in train_ids}
    if (
        len(ids) != 120 or _ids_sha(ids) != spec["sample_ids_sha256"]
        or len(set(groups)) != 120 or training_groups.intersection(groups)
    ):
        raise E32Error("E32 clean dev-120 group split changed.")
    return questions, dev521, ids, collisions


def validate_sources(
    *, e08a: Path, e00: Path, training: Path, e21: Path,
    train: Path, dev: Path, config: Config,
):
    questions, dev521, ids, collisions = sample(train, dev, config)
    source = config.e19.section("source_e08a")
    report_path = e08a / "report.json"
    contexts_path = e08a / source["dev_results_path"]
    if (
        not report_path.is_file()
        or file_sha256(report_path) != source["report_sha256"]
        or not contexts_path.is_file()
        or contexts_path.stat().st_size != source["dev_results_bytes"]
        or file_sha256(contexts_path) != source["dev_results_sha256"]
    ):
        raise E32Error("Add the exact complete E08A retrieval output/dataset.")
    metadata = config.e19.contract["metadata_source"]
    for name, key in (("manifest.json", "manifest_sha256"),
                      ("chunks.jsonl", "chunks_sha256"),
                      ("documents.jsonl", "documents_sha256")):
        path = e00 / name
        if not path.is_file() or file_sha256(path) != metadata[key]:
            raise E32Error(f"Pinned E00 source missing or changed: {name}")
    adapter_sha, complete = validate_candidate_adapter(training, config.e19)
    if adapter_sha != config.raw["source_e21_control"]["adapter_sha256"]:
        raise E32Error("E19 adapter differs from the E21 calibration.")
    _, _, old_ids, _ = sample_old200(train, dev, config)
    _, control_identity, old_prepared = e30.load_e21(e21, old_ids, config.e30)
    if control_identity["prepared_sha256"] != config.raw["source_e21_control"]["prepared_sha256"]:
        raise E32Error("E21 calibration contexts changed.")
    return {
        "questions": questions, "dev521": dev521, "ids": ids,
        "collisions": collisions, "source": source, "metadata": metadata,
        "contexts_path": contexts_path, "adapter_sha": adapter_sha,
        "adapter_complete": complete, "control_identity": control_identity,
        "old_ids": old_ids, "old_prepared": old_prepared,
    }


def sample_old200(train: Path, dev: Path, config: Config):
    questions, dev521, _, collisions = sample(train, dev, config)
    ids = dev521[200:400]
    if len(ids) != 200 or _ids_sha(ids) != config.e30.raw["evaluation"]["sample_ids_sha256"]:
        raise E32Error("Old E21 dev-200 calibration IDs changed.")
    return questions, dev521, ids, collisions


def validate_preflight(
    *, root: Path, e08a: Path, e00: Path, training: Path, e21: Path,
    train: Path, dev: Path, output: Path, config: Config,
):
    checked = validate_sources(
        e08a=e08a, e00=e00, training=training, e21=e21,
        train=train, dev=dev, config=config,
    )
    payload = {
        "schema_version": "1.0", "experiment_id": EXPERIMENT,
        "code_sha256": code_sha(root), "config_sha256": config.sha,
        "sample_ids_sha256": _ids_sha(checked["ids"]), "sample_size": 120,
        "excluded_prior_dev_duplicate": checked["collisions"],
        "e08a_dev_results_sha256": checked["source"]["dev_results_sha256"],
        "e00_chunks_sha256": checked["metadata"]["chunks_sha256"],
        "e00_documents_sha256": checked["metadata"]["documents_sha256"],
        "e21_prepared_sha256": checked["control_identity"]["prepared_sha256"],
        "e21_control_identity_sha256": checked["control_identity"]["identity_sha256"],
        "e19_adapter_sha256": checked["adapter_sha"],
        "e19_adapter_identity_sha256": checked["adapter_complete"]["identity_sha256"],
        "variants": list(VARIANTS), "answers_used_before_scoring": False,
    }
    save_once(output / "preflight.json", payload)
    return payload


def check_preflight(root: Path, output: Path, config: Config):
    path = output / "preflight.json"
    if not path.is_file():
        raise E32Error("Run E32 preflight first.")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("experiment_id") != EXPERIMENT
        or payload.get("code_sha256") != code_sha(root)
        or payload.get("config_sha256") != config.sha
        or payload.get("sample_ids_sha256") != config.raw["evaluation"]["sample_ids_sha256"]
        or payload.get("variants") != list(VARIANTS)
    ):
        raise E32Error("E32 preflight code/config changed.")
    return payload


def _prepare_rows(rows, ids, seeds, documents, blocks, policy):
    prepared = []
    for index, (question_id, row) in enumerate(zip(ids, rows)):
        units = []
        for rank, context in enumerate(row["contexts"]):
            seed = seeds[context["chunk_id"]]
            enrich_context(context, seed, documents[seed["document_id"]])
            units.append(parent.seed_unit(
                seed, rank, blocks[context["chunk_id"]],
                documents[seed["document_id"]], policy,
            ))
        prepared.append({
            "question_id": question_id, "sample_index": index,
            "answers_used": False, "units": units,
        })
    return prepared


def prepare(
    *, root: Path, e08a: Path, e00: Path, training: Path, e21: Path,
    train: Path, dev: Path, output: Path, config: Config,
):
    checked = check_preflight(root, output, config)
    sources = validate_sources(
        e08a=e08a, e00=e00, training=training, e21=e21,
        train=train, dev=dev, config=config,
    )
    all_rows = _validate_context_rows(
        sources["contexts_path"], sources["dev521"], "dev521", sources["source"],
    )
    old_rows = all_rows[200:400]
    selected_by_id = {row["question_id"]: row for row in all_rows[400:521]}
    clean_rows = [selected_by_id[question_id] for question_id in sources["ids"]]
    needed = {context["chunk_id"] for row in old_rows + clean_rows for context in row["contexts"]}
    metadata = sources["metadata"]
    seeds = scan_selected(e00 / "chunks.jsonl", "chunk_id", needed, metadata["chunks_sha256"])
    doc_ids = {seed["document_id"] for seed in seeds.values()}
    documents = scan_selected(e00 / "documents.jsonl", "document_id", doc_ids, metadata["documents_sha256"])
    by_doc = {document_id: [] for document_id in doc_ids}
    digest = hashlib.sha256()
    with (e00 / "chunks.jsonl").open("rb") as stream:
        for line in stream:
            digest.update(line)
            chunk = json.loads(line)
            if chunk["document_id"] in by_doc:
                by_doc[chunk["document_id"]].append(chunk)
    if digest.hexdigest() != metadata["chunks_sha256"]:
        raise E32Error("E00 chunks changed during parent scan.")
    blocks = {}
    for document_id, chunks in by_doc.items():
        for block in parent.article_blocks(chunks, documents[document_id]):
            for chunk in block:
                if chunk["chunk_id"] in needed:
                    blocks[chunk["chunk_id"]] = block
    if set(blocks) != needed:
        raise E32Error("Not all selected E08A chunks resolve to an E00 article block.")
    calibration = _prepare_rows(
        old_rows, sources["old_ids"], seeds, documents, blocks, config.e21.policy,
    )
    if calibration != sources["old_prepared"]:
        raise E32Error("E32 parent-context builder does not reproduce exact E21 dev-200 contexts.")
    prepared = _prepare_rows(
        clean_rows, sources["ids"], seeds, documents, blocks, config.e21.policy,
    )
    destination = output / "prepared/results.jsonl"
    save_once(destination, prepared, jsonl=True)
    summary = {
        "schema_version": "1.0", "experiment_id": EXPERIMENT,
        "sample_size": 120, "sample_ids_sha256": _ids_sha(sources["ids"]),
        "results_sha256": file_sha256(destination),
        "source_contexts_sha256": file_sha256(sources["contexts_path"]),
        "e21_calibration_prepared_sha256": checked["e21_prepared_sha256"],
        "e21_calibration_exact_match": True,
        "excluded_question_id": config.raw["evaluation"]["excluded_question_id"],
        "expandable_questions": sum(any(unit["expansions"] for unit in row["units"]) for row in prepared),
        "answers_used": False,
    }
    save_once(output / "prepared/summary.json", summary)
    return summary


def load_prepared(output: Path, ids: list[str], config: Config):
    path = output / "prepared/results.jsonl"
    summary_path = output / "prepared/summary.json"
    if not path.is_file() or not summary_path.is_file():
        raise E32Error("Run E32 prepare-contexts first.")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    rows = _read_jsonl(path)
    if (
        summary.get("experiment_id") != EXPERIMENT
        or summary.get("sample_ids_sha256") != _ids_sha(ids)
        or summary.get("results_sha256") != file_sha256(path)
        or summary.get("e21_calibration_exact_match") is not True
        or summary.get("answers_used") is not False
        or [row.get("question_id") for row in rows] != ids
        or any(row.get("sample_index") != index or row.get("answers_used") is not False
               or [unit.get("rank") for unit in row.get("units", [])] != list(range(12))
               for index, row in enumerate(rows))
    ):
        raise E32Error("Prepared E32 contexts changed.")
    return rows


def validate_record(row, question_id, index, rank, identity):
    variant = VARIANTS[rank]
    if (
        row.get("question_id") != question_id
        or row.get("sample_index") != index
        or row.get("worker_rank") != rank
        or row.get("variant") != variant["raw_key"]
        or row.get("worker_identity_sha256") != identity["identity_sha256"]
        or not isinstance(row.get("answer"), str)
        or not row["answer"].strip()
        or row.get("record_sha256") != _json_sha256({key: value for key, value in row.items() if key != "record_sha256"})
    ):
        raise E32Error(f"Changed E32 generation record: {rank}/{index}")


def run_worker(
    *, root: Path, e08a: Path, e00: Path, training: Path, e21: Path,
    train: Path, dev: Path, output: Path, config: Config, rank: int, device: str,
):
    import torch

    if rank not in (0, 1) or device != f"cuda:{rank}":
        raise E32Error("GPU0=max1024 all120, GPU1=max1280 all120.")
    checked = check_preflight(root, output, config)
    questions, _, ids, _ = sample(train, dev, config)
    prepared = load_prepared(output, ids, config)
    _, control_identity, _ = e30.load_e21(e21, sample_old200(train, dev, config)[2], config.e30)
    runtime = {name: importlib.metadata.version(name) for name in control_identity["runtime"]}
    if runtime != control_identity["runtime"]:
        raise E32Error(f"Use exact E21 runtime: expected={control_identity['runtime']} current={runtime}")
    model, tokenizer, placement, parameters = load_candidate_generator(
        config=config.e19, training_directory=training, device=device,
    )
    adapter_sha, _ = validate_candidate_adapter(training, config.e19)
    generation_sha = _json_sha256(model.generation_config.to_dict())
    if (
        adapter_sha != checked["e19_adapter_sha256"]
        or generation_sha != control_identity["generation_config_sha256"]
    ):
        raise E32Error("Adapter or generation defaults differ from E21.")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    def render(messages):
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )

    def count(text):
        return len(tokenizer(text, add_special_tokens=False)["input_ids"])

    variant = VARIANTS[rank]
    indices = list(range(120))
    identity = {
        "code_sha256": code_sha(root), "config_sha256": config.sha,
        "prepared_sha256": file_sha256(output / "prepared/results.jsonl"),
        "sample_ids_sha256": _ids_sha(ids),
        "e21_control_identity_sha256": control_identity["identity_sha256"],
        "adapter_sha256": adapter_sha,
        "generation_config_sha256": generation_sha,
        "runtime": runtime,
        "variant": variant["raw_key"],
        "max_new_tokens": variant["max_new_tokens"],
        "no_repeat_ngram_size": 0,
        "worker_rank": rank, "device": device, "device_map": placement,
        "adapter_parameters": parameters, "assigned_indices": indices,
    }
    identity["identity_sha256"] = _json_sha256(identity)
    folder = output / "evaluation" / variant["raw_key"]
    records = folder / "records"
    state = folder / "state.json"
    records.mkdir(parents=True, exist_ok=True)
    done = _load_worker_progress(
        records=records, state_path=state, identity=identity,
        assigned_indices=indices, sample_ids=ids,
    )
    for index in indices[:done]:
        validate_record(
            json.loads((records / f"{index:04d}.json").read_text(encoding="utf-8")),
            ids[index], index, rank, identity,
        )
    for index in indices[done:]:
        question_id = ids[index]
        messages, spans, diagnostics = parent.pack(
            questions[question_id]["question"], prepared[index], config.e21,
            lambda items: count(render(items)), count, parent.VARIANTS[1],
        )
        prompt = render(messages)
        inputs = {
            key: value.to(device)
            for key, value in tokenizer(prompt, add_special_tokens=False, return_tensors="pt").items()
        }
        cap = variant["max_new_tokens"]
        started = time.perf_counter()
        with torch.inference_mode():
            generated = model.generate(
                **inputs, do_sample=False, num_beams=1, repetition_penalty=1.0,
                no_repeat_ngram_size=0, max_new_tokens=cap, use_cache=True,
            )
        latency = (time.perf_counter() - started) * 1000
        new_ids = generated[0, inputs["input_ids"].shape[1]:]
        answer = tokenizer.decode(
            new_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False,
        ).strip()
        if not answer:
            raise E32Error(f"Empty E32 answer: {question_id}")
        eos = model.generation_config.eos_token_id
        eos_ids = set(eos if isinstance(eos, list) else [eos])
        finish = (
            "eos" if len(new_ids) and int(new_ids[-1]) in eos_ids
            else "length" if len(new_ids) >= cap else "other"
        )
        row = {
            "question_id": question_id, "sample_index": index,
            "worker_rank": rank, "variant": variant["raw_key"],
            "worker_identity_sha256": identity["identity_sha256"],
            "answer": answer,
            "input_tokens": count(prompt), "output_tokens": count(answer),
            "generated_tokens_including_special": len(new_ids),
            "finish_reason": finish, "generation_latency_ms": latency,
            "selected_context_count": len(spans),
            "selected_chunk_ids": diagnostics["selected_chunk_ids"],
            "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        }
        row["record_sha256"] = _json_sha256(row)
        _atomic_json(records / f"{index:04d}.json", row)
        _write_worker_state(state, identity, index + 1, 120)
        LOG.info(
            "e32_generation variant=%s device=%s completed=%d/120 question_id=%s finish=%s",
            variant["raw_key"], device, index + 1, question_id, finish,
        )
    return {"variant": variant["raw_key"], "completed": 120}


def _paired(scores, left, right):
    meteor = [a["meteor"] - b["meteor"] for a, b in zip(scores[left], scores[right])]
    rouge = [a["rouge_l"] - b["rouge_l"] for a, b in zip(scores[left], scores[right])]
    return {
        "meteor_mean": fmean(meteor),
        "meteor_bootstrap_95_ci": _bootstrap_ci(
            meteor, seed=f"e32-{left}-{right}-meteor", iterations=10000,
        ),
        "rouge_l_mean": fmean(rouge),
        "rouge_l_bootstrap_95_ci": _bootstrap_ci(
            rouge, seed=f"e32-{left}-{right}-rouge", iterations=10000,
        ),
        "improved": sum(value > 0 for value in meteor),
        "worsened": sum(value < 0 for value in meteor),
        "tied": sum(value == 0 for value in meteor),
    }


def finalize(
    *, root: Path, e08a: Path, e00: Path, training: Path, e21: Path,
    train: Path, dev: Path, output: Path, config: Config,
):
    checked = check_preflight(root, output, config)
    ensure_nltk_resources(download=False)
    questions, _, ids, collisions = sample(train, dev, config)
    prepared = load_prepared(output, ids, config)
    raw_rows = {}
    identities = []
    for variant in VARIANTS:
        rank = variant["worker_rank"]
        folder = output / "evaluation" / variant["raw_key"]
        state = json.loads((folder / "state.json").read_text(encoding="utf-8"))
        identity = state.get("run_identity", {})
        if (
            state.get("complete") is not True
            or state.get("completed_count") != 120
            or state.get("assigned_count") != 120
            or identity.get("variant") != variant["raw_key"]
            or identity.get("worker_rank") != rank
            or identity.get("device") != f"cuda:{rank}"
            or identity.get("max_new_tokens") != variant["max_new_tokens"]
            or identity.get("no_repeat_ngram_size") != 0
            or identity.get("assigned_indices") != list(range(120))
            or identity.get("prepared_sha256") != file_sha256(output / "prepared/results.jsonl")
            or identity.get("e21_control_identity_sha256") != checked["e21_control_identity_sha256"]
            or identity.get("adapter_sha256") != checked["e19_adapter_sha256"]
            or identity.get("code_sha256") != code_sha(root)
            or identity.get("config_sha256") != config.sha
            or identity.get("identity_sha256")
            != _json_sha256({key: value for key, value in identity.items() if key != "identity_sha256"})
        ):
            raise E32Error("Incomplete or changed E32 worker state.")
        identities.append(identity)
        rows = []
        for index, question_id in enumerate(ids):
            row = json.loads((folder / f"records/{index:04d}.json").read_text(encoding="utf-8"))
            validate_record(row, question_id, index, rank, identity)
            rows.append(row)
        _atomic_jsonl(folder / "results.jsonl", rows)
        raw_rows[variant["raw_key"]] = rows
    if identities[0]["runtime"] != identities[1]["runtime"] or identities[0]["generation_config_sha256"] != identities[1]["generation_config_sha256"]:
        raise E32Error("E32 GPU runtimes or generation defaults differ.")
    for index in range(120):
        if raw_rows[VARIANTS[0]["raw_key"]][index]["prompt_sha256"] != raw_rows[VARIANTS[1]["raw_key"]][index]["prompt_sha256"]:
            raise E32Error(f"A/B prompts differ at index {index}.")

    try:
        from transformers import AutoTokenizer
    except ImportError as exc:  # pragma: no cover
        raise E32Error("Transformers is required for E32 finalization.") from exc
    tokenizer = AutoTokenizer.from_pretrained(
        config.e19.generator_model_id,
        revision=config.e19.generator_revision,
        trust_remote_code=False,
    )
    by_variant = dict(raw_rows)
    trim_diagnostics = {}
    for variant in VARIANTS:
        derived = []
        for source in raw_rows[variant["raw_key"]]:
            first = trim_repeated_tail(source["answer"])
            second = e31.trim_long_token_suffix(
                first["answer"], minimum_block_tokens=32, maximum_block_tokens=256,
            )
            row = {
                **source,
                "variant": variant["key"],
                "source_variant": variant["raw_key"],
                "answer": second["answer"],
                "output_tokens": len(tokenizer(second["answer"], add_special_tokens=False)["input_ids"]),
                "line_sentence_tailtrim_changed": first["changed"],
                "line_sentence_removed_characters": first["removed_characters"],
                "long_token_tailtrim_changed": second["changed"],
                "long_token_removed_characters": second["removed_characters"],
                "long_token_removed_whitespace_tokens": second["removed_whitespace_tokens"],
                "long_token_matched_block_sizes": second["matched_block_sizes"],
            }
            row.pop("record_sha256", None)
            row["record_sha256"] = _json_sha256(row)
            derived.append(row)
        path = output / f"evaluation/{variant['key']}/results.jsonl"
        _atomic_jsonl(path, derived)
        by_variant[variant["key"]] = derived
        trim_diagnostics[variant["key"]] = {
            "line_sentence_changed_questions": sum(row["line_sentence_tailtrim_changed"] for row in derived),
            "long_token_changed_questions": sum(row["long_token_tailtrim_changed"] for row in derived),
            "both_unchanged_questions": sum(
                not row["line_sentence_tailtrim_changed"] and not row["long_token_tailtrim_changed"]
                for row in derived
            ),
            "total_removed_characters": sum(
                row["line_sentence_removed_characters"] + row["long_token_removed_characters"]
                for row in derived
            ),
        }
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
            "postprocess_changed": {
                variant["key"]: (
                    by_variant[variant["key"]][index]["line_sentence_tailtrim_changed"]
                    or by_variant[variant["key"]][index]["long_token_tailtrim_changed"]
                )
                for variant in VARIANTS
            },
        }
        for index, question_id in enumerate(ids)
    ])
    metrics = {variant: _metrics(rows, scores[variant]) for variant, rows in by_variant.items()}
    report = {
        "schema_version": "1.0", "experiment_id": EXPERIMENT,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_reserved_dev_size": 121,
        "clean_scored_size": 120,
        "excluded_prior_dev_duplicate": collisions,
        "sample_ids_sha256": _ids_sha(ids),
        "metrics": metrics,
        "paired_final1280_minus_final1024": _paired(scores, VARIANTS[1]["key"], VARIANTS[0]["key"]),
        "paired_final1024_minus_raw1024": _paired(scores, VARIANTS[0]["key"], VARIANTS[0]["raw_key"]),
        "paired_final1280_minus_raw1280": _paired(scores, VARIANTS[1]["key"], VARIANTS[1]["raw_key"]),
        "trim_diagnostics": trim_diagnostics,
        "dev121_leader_on_clean120": max(
            (variant["key"] for variant in VARIANTS),
            key=lambda key: metrics[key]["meteor"],
        ),
        "automatic_promotion_allowed": False,
        "public_read": False, "private_untouched": True,
        "evidence": {
            "config_sha256": config.sha, "code_sha256": code_sha(root),
            "e08a_dev_results_sha256": checked["e08a_dev_results_sha256"],
            "e21_calibration_prepared_sha256": checked["e21_prepared_sha256"],
            "prepared_results_sha256": file_sha256(output / "prepared/results.jsonl"),
            "adapter_sha256": checked["e19_adapter_sha256"],
            "candidate_results_sha256": {
                variant["key"]: file_sha256(output / f"evaluation/{variant['key']}/results.jsonl")
                for variant in VARIANTS
            },
        },
        "warning": "Use this clean dev-120 once for operator selection; private performance is not guaranteed.",
    }
    _atomic_json(output / "report.json", report)
    return report
