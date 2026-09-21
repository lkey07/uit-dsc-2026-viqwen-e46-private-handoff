"""E19 metadata-aware QLoRA training and paired fresh-dev evaluation."""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import time
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean
from typing import Any

from .corpus import file_sha256
from .e02_answer import _atomic_json, _atomic_jsonl, pack_contexts
from .e03_rrf_grid import (
    _bootstrap_ci, _json_sha256, _load_worker_progress, _read_jsonl,
    _validate_worker_placement, _write_worker_state,
)
from .e08b_context_lora import (
    E08BConfig, _load_splits, _validate_context_rows, inference_packing,
    load_context_lora_generator, load_e08b_config, validate_e08b_preflight,
)
from .e10_repetition_grid import answer_diagnostics
from .e18_source_metadata import enrich_context, scan_selected
from .evaluation.official import ensure_nltk_resources, nltk_meteor_score, rouge_l_fmeasure
from .retrieval_diagnostics import select_dev_sample


EXPERIMENT = "E19-metadata-aware-lora-train5636-eval200-v1"
VARIANTS = ("e08b_rank8_metadata_max704", "e19_metadata_trained_rank8_max704")
CODE_VERSION = "0.51.0"
LOGGER = logging.getLogger(__name__)


class E19Error(RuntimeError):
    pass


def code_sha(project_root: Path) -> str:
    paths = sorted((project_root / "src/uit_dsc_fixed_rag").rglob("*.py"))
    paths += [
        project_root / "scripts/run_e19_metadata_lora_prepare_kaggle.py",
        project_root / "scripts/run_e19_metadata_lora_train_kaggle.py",
        project_root / "scripts/run_e19_metadata_lora_eval_kaggle.py",
    ]
    existing = [path for path in paths if path.is_file()]
    return _json_sha256({
        path.relative_to(project_root).as_posix(): file_sha256(path)
        for path in existing
    })


@dataclass(frozen=True)
class E19Config:
    raw: dict[str, Any]
    path: Path
    contract: dict[str, Any]
    base: E08BConfig

    @property
    def config_sha256(self) -> str:
        return file_sha256(self.path)

    def section(self, key: str) -> dict[str, Any]:
        value = self.raw.get(key)
        if not isinstance(value, dict):
            raise ValueError(f"E19 derived section must be an object: {key}")
        return value

    @property
    def generator_model_id(self) -> str:
        return self.base.generator_model_id

    @property
    def generator_revision(self) -> str:
        return self.base.generator_revision


def load_config(project_root: Path, path: Path) -> E19Config:
    contract = json.loads(path.read_text(encoding="utf-8"))
    if set(contract) != {
        "schema_version", "experiment_id", "base_training_config_path",
        "base_training_config_sha256", "source_e18_config_path",
        "source_e18_config_sha256", "metadata_source", "training",
        "control_source", "evaluation", "run_contract",
    }:
        raise E19Error("E19 config root changed.")
    if contract.get("schema_version") != "1.0" or contract.get("experiment_id") != EXPERIMENT:
        raise E19Error("E19 experiment identity changed.")
    base_path = project_root / contract["base_training_config_path"]
    e18_path = project_root / contract["source_e18_config_path"]
    if file_sha256(base_path) != contract["base_training_config_sha256"]:
        raise E19Error("Pinned E08B training config changed.")
    if file_sha256(e18_path) != contract["source_e18_config_sha256"]:
        raise E19Error("Pinned E18 metadata config changed.")
    base = load_e08b_config(base_path)
    metadata = contract["metadata_source"]
    e18 = json.loads(e18_path.read_text(encoding="utf-8"))
    if metadata != {
        **e18["metadata_source"], "policy": e18["metadata_policy"]
    }:
        raise E19Error("E19 metadata policy differs from E18.")
    if contract["training"] != {
        "sample_size": 5636,
        "supervision": "official-answer-only",
        "input": "question-plus-frozen-ranked-top12-contexts-with-e00-source-metadata",
        "maximum_sequence_tokens": 3072,
        "epochs": 1.0,
        "fresh_from_base": True,
        "lora_rank": 8,
        "lora_alpha": 16,
        "same_seed_and_optimization_as_e08b": True,
    }:
        raise E19Error("E19 training contract changed.")
    if contract["control_source"] != {
        "experiment_id": "E08B-context-aware-lora-train5636-dev521-v3",
        "training_config_sha256": contract["base_training_config_sha256"],
        "adapter_sha256": "14316bdb1a999e67c9baae90620c785977856e68cd540d46dbc57f1b2176499d",
        "fresh_from_base": True,
        "lora_rank": 8,
        "lora_alpha": 16,
    }:
        raise E19Error("E19 control adapter contract changed.")
    if contract["evaluation"] != {
        "dev521_offset": 200,
        "sample_size": 200,
        "sample_ids_sha256": "dc17d6c9af9e941868c03cfc61fd17cdd1dc55aa5834beb3e75ba0fde32a5c73",
        "control_variant": VARIANTS[0],
        "candidate_variant": VARIANTS[1],
        "context_count": 12,
        "max_input_tokens": 8192,
        "max_new_tokens": 704,
        "workers": 2,
        "questions_per_worker": 200,
        "promotion_allowed": False,
    }:
        raise E19Error("E19 fresh evaluation contract changed.")
    if not contract["run_contract"] or not all(contract["run_contract"].values()):
        raise E19Error("E19 run contract lost an invariant.")
    raw = copy.deepcopy(base.raw)
    raw["experiment_id"] = EXPERIMENT
    raw["train"]["input"] = contract["training"]["input"]
    raw["lora"]["initialization"] = "fresh-from-pinned-base-not-e08b-adapter"
    raw["inference"]["minimum_contexts"] = 12
    raw["inference"]["max_new_tokens"] = 704
    raw["scoring"]["bootstrap_seed"] = "uit-dsc-2026-e19-metadata-aware-lora-v1"
    raw["run_contract"] = copy.deepcopy(contract["run_contract"])
    return E19Config(raw=raw, path=path, contract=contract, base=base)


def _ids_sha(ids: list[str]) -> str:
    return hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()


def validate_e00(e00_directory: Path, config: E19Config) -> None:
    metadata = config.contract["metadata_source"]
    for filename, key in (
        ("manifest.json", "manifest_sha256"),
        ("chunks.jsonl", "chunks_sha256"),
        ("documents.jsonl", "documents_sha256"),
    ):
        path = e00_directory / filename
        if not path.is_file() or file_sha256(path) != metadata[key]:
            raise E19Error(f"Pinned E00-v2 source changed: {filename}")


def validate_training_preflight(
    *, project_root: Path, e08a_directory: Path, e00_directory: Path,
    train_path: Path, dev_path: Path, config: E19Config,
) -> dict[str, Any]:
    evidence = validate_e08b_preflight(
        project_root=project_root, e08a_directory=e08a_directory,
        train_path=train_path, dev_path=dev_path, config=config,
    )
    validate_e00(e00_directory, config)
    return {
        **evidence,
        "code_sha256": code_sha(project_root),
        "metadata_source": config.contract["metadata_source"],
        "training_input": config.contract["training"]["input"],
        "fresh_from_base": True,
        "lora_rank": 8,
        "lora_alpha": 16,
    }


def prepare_metadata_train_records(
    *, e08a_directory: Path, e00_directory: Path, train_path: Path,
    dev_path: Path, output_directory: Path, config: E19Config,
    preflight: dict[str, Any],
) -> dict[str, Any]:
    train, train_ids, _, _ = _load_splits(train_path, dev_path, config)
    source = config.section("source_e08a")
    context_path = e08a_directory / source["train_results_path"]
    if (
        file_sha256(context_path) != preflight.get("train_contexts_sha256")
        or file_sha256(train_path) != preflight.get("train_sha256")
    ):
        raise E19Error("E19 inputs changed after preflight.")
    context_rows = _validate_context_rows(context_path, train_ids, "train5636", source)
    wanted = {c["chunk_id"] for row in context_rows for c in row["contexts"]}
    metadata = config.contract["metadata_source"]
    chunks = scan_selected(
        e00_directory / "chunks.jsonl", "chunk_id", wanted,
        metadata["chunks_sha256"],
    )
    documents = scan_selected(
        e00_directory / "documents.jsonl", "document_id",
        {c["document_id"] for c in chunks.values()}, metadata["documents_sha256"],
    )
    rows: list[dict[str, Any]] = []
    coverage: list[dict[str, Any]] = []
    for index, (question_id, context_row) in enumerate(zip(train_ids, context_rows)):
        enriched_contexts = []
        for context in context_row["contexts"]:
            enriched, item = enrich_context(
                context, chunks[context["chunk_id"]], documents[context["document_id"]]
            )
            enriched_contexts.append(enriched)
            coverage.append(item)
        if [c["chunk_id"] for c in enriched_contexts] != [c["chunk_id"] for c in context_row["contexts"]]:
            raise E19Error("Metadata changed training retrieval order.")
        official = train[question_id]
        rows.append({
            "sample_index": index,
            "question_id": question_id,
            "question": official["question"],
            "answer": official["answer"],
            "contexts": enriched_contexts,
            "supervision": "official-answer-only",
            "answers_are_retrieval_labels": False,
            "metadata_policy": metadata["policy"],
        })
    root = output_directory / "training-data"
    root.mkdir(parents=True, exist_ok=True)
    records_path = root / "records.jsonl"
    _atomic_jsonl(records_path, rows)
    summary = {
        "schema_version": "1.0",
        "experiment_id": EXPERIMENT,
        "record_count": len(rows),
        "records_sha256": file_sha256(records_path),
        "sample_ids_sha256": _ids_sha(train_ids),
        "raw_source_contexts_sha256": source["train_results_sha256"],
        "context_candidates_per_record": 12,
        "context_instances": len(coverage),
        "with_source_title": sum(bool(x["source_title"]) for x in coverage),
        "with_document_number": sum(bool(x["document_number"]) for x in coverage),
        "with_article_title": sum(bool(x["article_title"]) for x in coverage),
        "unchanged_contexts": sum(not x["prefix"] for x in coverage),
        "same_context_bodies_and_order": True,
        "metadata_policy": metadata["policy"],
        "answers_are_retrieval_labels": False,
        "contains_dev_holdout_or_public": False,
    }
    _atomic_json(root / "summary.json", summary)
    return summary


def validate_candidate_adapter(
    training_directory: Path, config: E19Config
) -> tuple[str, dict[str, Any]]:
    final = training_directory / "adapter-final"
    adapter = final / "adapter_model.safetensors"
    complete_path = final / "complete.json"
    if not adapter.is_file() or not complete_path.is_file():
        raise E19Error("Completed E19 adapter is missing.")
    complete = json.loads(complete_path.read_text(encoding="utf-8"))
    observed = file_sha256(adapter)
    if (
        complete.get("experiment_id") != EXPERIMENT
        or complete.get("config_sha256") != config.config_sha256
        or complete.get("adapter_sha256") != observed
        or complete.get("fresh_from_base") is not True
        or complete.get("e08b_adapter_loaded") is not False
        or complete.get("metadata_policy") != config.contract["metadata_source"]["policy"]
        or complete.get("lora_rank") != 8
        or complete.get("lora_alpha") != 16
    ):
        raise E19Error("Completed E19 adapter evidence changed.")
    return observed, complete


def validate_control_adapter(
    training_directory: Path, config: E19Config
) -> tuple[str, dict[str, Any]]:
    """Validate the exact E08B rank-8 adapter used by the E18 public leader."""
    final = training_directory / "adapter-final"
    adapter = final / "adapter_model.safetensors"
    complete_path = final / "complete.json"
    if not adapter.is_file() or not complete_path.is_file():
        raise E19Error("Completed E08B control adapter is missing.")
    complete = json.loads(complete_path.read_text(encoding="utf-8"))
    expected = config.contract["control_source"]
    observed = file_sha256(adapter)
    if (
        observed != expected["adapter_sha256"]
        or complete.get("adapter_sha256") != observed
        or complete.get("experiment_id") != expected["experiment_id"]
        or complete.get("config_sha256") != expected["training_config_sha256"]
        or complete.get("fresh_from_base") is not expected["fresh_from_base"]
        or complete.get("e07_adapter_loaded") is not False
    ):
        raise E19Error("Completed E08B control adapter evidence changed.")
    return observed, complete


def load_candidate_generator(
    *, config: E19Config, training_directory: Path, device: str
) -> tuple[Any, Any, dict[str, Any], int]:
    try:
        import torch
        from peft import PeftModel
        from transformers import AutoTokenizer, Qwen3_5ForCausalLM
    except ImportError as exc:  # pragma: no cover
        raise E19Error("Install Transformers and PEFT for E19 inference.") from exc
    if device not in {"cuda:0", "cuda:1"} or torch.cuda.device_count() != 2:
        raise E19Error("E19 evaluation requires GPU T4 x2.")
    validate_candidate_adapter(training_directory, config)
    torch.cuda.set_device(int(device[-1]))
    tokenizer = AutoTokenizer.from_pretrained(
        config.generator_model_id, revision=config.generator_revision,
        trust_remote_code=False,
    )
    base = Qwen3_5ForCausalLM.from_pretrained(
        config.generator_model_id, revision=config.generator_revision,
        dtype=torch.float16, device_map={"": device}, low_cpu_mem_usage=True,
        trust_remote_code=False,
    )
    model = PeftModel.from_pretrained(
        base, training_directory / "adapter-final", is_trainable=False
    )
    model.eval()
    adapter_parameters = sum(p.numel() for n, p in model.named_parameters() if "lora_" in n)
    if not 0 < adapter_parameters <= config.section("lora")["trainable_parameter_cap"]:
        raise E19Error("E19 adapter parameter count violates cap.")
    return model, tokenizer, _validate_worker_placement(model, device), adapter_parameters


def load_control_generator(
    *, project_root: Path, config: E19Config, training_directory: Path, device: str
) -> tuple[Any, Any, dict[str, Any], int]:
    validate_control_adapter(training_directory, config)
    base = load_e08b_config(project_root / "configs/e08b-context-aware-lora-v3.json")
    return load_context_lora_generator(
        config=base, training_directory=training_directory, device=device
    )


def eval_sample(
    train_path: Path, dev_path: Path, config: E19Config
) -> tuple[dict[str, Any], list[str], list[str]]:
    _, _, dev, dev521_ids = _load_splits(train_path, dev_path, config)
    spec = config.contract["evaluation"]
    start = spec["dev521_offset"]
    ids = dev521_ids[start : start + spec["sample_size"]]
    if len(ids) != 200 or _ids_sha(ids) != spec["sample_ids_sha256"]:
        raise E19Error("E19 fresh dev200 identity changed.")
    full_ids = select_dev_sample(
        dev, seed=config.section("dev")["sample_seed"],
        size=config.section("dev")["full_size"],
    )
    normalize = lambda value: " ".join(
        unicodedata.normalize("NFC", value).casefold().split()
    )
    previous_groups = {
        normalize(dev[question_id]["question"])
        for question_id in full_ids[:400]
    }
    selected_groups = [normalize(dev[question_id]["question"]) for question_id in ids]
    if len(set(selected_groups)) != len(ids) or previous_groups.intersection(selected_groups):
        raise E19Error("E19 evaluation repeats a previously used normalized question group.")
    return dev, dev521_ids, ids


def prepare_eval_contexts(
    *, e08a_directory: Path, e00_directory: Path, train_path: Path,
    dev_path: Path, output_directory: Path, config: E19Config,
) -> dict[str, Any]:
    _, dev521_ids, ids = eval_sample(train_path, dev_path, config)
    source = config.section("source_e08a")
    all_rows = _validate_context_rows(
        e08a_directory / source["dev_results_path"], dev521_ids, "dev521", source
    )
    start = config.contract["evaluation"]["dev521_offset"]
    selected_rows = all_rows[start : start + len(ids)]
    wanted = {c["chunk_id"] for row in selected_rows for c in row["contexts"]}
    metadata = config.contract["metadata_source"]
    chunks = scan_selected(
        e00_directory / "chunks.jsonl", "chunk_id", wanted,
        metadata["chunks_sha256"],
    )
    documents = scan_selected(
        e00_directory / "documents.jsonl", "document_id",
        {c["document_id"] for c in chunks.values()}, metadata["documents_sha256"],
    )
    prepared, coverage = [], []
    for index, (question_id, row) in enumerate(zip(ids, selected_rows)):
        contexts = []
        for context in row["contexts"]:
            enriched, item = enrich_context(
                context, chunks[context["chunk_id"]], documents[context["document_id"]]
            )
            contexts.append(enriched)
            coverage.append(item)
        prepared.append({
            "question_id": question_id, "sample_index": index,
            "answer_included": False, "contexts": contexts,
        })
    root = output_directory / "prepared"
    root.mkdir(parents=True, exist_ok=True)
    results = root / "results.jsonl"
    _atomic_jsonl(results, prepared)
    summary = {
        "experiment_id": EXPERIMENT,
        "sample_size": len(ids),
        "sample_ids_sha256": _ids_sha(ids),
        "results_sha256": file_sha256(results),
        "raw_dev_contexts_sha256": source["dev_results_sha256"],
        "context_instances": len(coverage),
        "contexts_per_question": 12,
        "with_source_title": sum(bool(x["source_title"]) for x in coverage),
        "with_document_number": sum(bool(x["document_number"]) for x in coverage),
        "with_article_title": sum(bool(x["article_title"]) for x in coverage),
        "same_context_bodies_and_order": True,
        "answers_used": False,
        "metadata_policy": metadata["policy"],
    }
    _atomic_json(root / "summary.json", summary)
    return summary


def run_eval_worker(
    *, worker_rank: int, device: str, model: Any, tokenizer: Any,
    adapter_hash: str, adapter_parameters: int, device_map: dict[str, Any],
    train_path: Path, dev_path: Path, output_directory: Path, config: E19Config,
) -> dict[str, Any]:
    if worker_rank not in (0, 1) or device != f"cuda:{worker_rank}":
        raise E19Error("E19 worker/device mapping changed.")
    variant = VARIANTS[worker_rank]
    dev, _, ids = eval_sample(train_path, dev_path, config)
    prepared_path = output_directory / "prepared/results.jsonl"
    prepared = _read_jsonl(prepared_path)
    summary = json.loads(
        (output_directory / "prepared/summary.json").read_text(encoding="utf-8")
    )
    if (
        [r.get("question_id") for r in prepared] != ids
        or summary.get("experiment_id") != EXPERIMENT
        or summary.get("sample_ids_sha256") != _ids_sha(ids)
        or summary.get("results_sha256") != file_sha256(prepared_path)
        or summary.get("contexts_per_question") != 12
        or summary.get("same_context_bodies_and_order") is not True
        or summary.get("answers_used") is not False
    ):
        raise E19Error("Prepared E19 contexts changed.")
    assigned = list(range(len(ids)))
    identity = {
        "code_version": CODE_VERSION,
        "config_sha256": config.config_sha256,
        "prepared_sha256": file_sha256(prepared_path),
        "sample_ids_sha256": _ids_sha(ids),
        "variant": variant,
        "adapter_sha256": adapter_hash,
        "adapter_parameters": adapter_parameters,
        "worker_rank": worker_rank,
        "device": device,
        "device_map": device_map,
    }
    identity["identity_sha256"] = _json_sha256(identity)
    root = output_directory / "evaluation" / variant
    records = root / "records"
    records.mkdir(parents=True, exist_ok=True)
    state = root / "state.json"
    completed = _load_worker_progress(
        records=records, state_path=state, identity=identity,
        assigned_indices=assigned, sample_ids=ids,
    )
    for position in range(completed):
        row = json.loads((records / f"{position:04d}.json").read_text(encoding="utf-8"))
        if (
            row.get("variant") != variant
            or row.get("worker_rank") != worker_rank
            or row.get("record_sha256")
            != _json_sha256({k: v for k, v in row.items() if k != "record_sha256"})
        ):
            raise E19Error(f"Changed E19 checkpoint record: {variant}/{position}")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    packing = inference_packing(config)

    def count(messages):
        rendered = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
        return len(tokenizer(rendered, add_special_tokens=False)["input_ids"])

    import torch
    for position in range(completed, len(ids)):
        question_id = ids[position]
        selected, messages, input_tokens = pack_contexts(
            question=dev[question_id]["question"],
            contexts=prepared[position]["contexts"], config=packing,
            token_counter=count,
        )
        if len(selected) != 12:
            raise E19Error(f"E19 paired prompt did not fit all contexts: {question_id}")
        rendered = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = tokenizer(rendered, add_special_tokens=False, return_tensors="pt")
        inputs = {key: value.to(device) for key, value in inputs.items()}
        started = time.perf_counter()
        with torch.inference_mode():
            generated = model.generate(
                **inputs, do_sample=False, num_beams=1, max_new_tokens=704,
                use_cache=True,
            )
        latency = (time.perf_counter() - started) * 1000
        new_ids = generated[0, inputs["input_ids"].shape[1] :]
        answer = tokenizer.decode(
            new_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        ).strip()
        if not answer:
            raise E19Error(f"Empty E19 answer: {question_id}")
        eos = tokenizer.eos_token_id
        eos_ids = set(eos if isinstance(eos, list) else [eos])
        generated_count = int(new_ids.shape[0])
        finish = "eos" if generated_count and int(new_ids[-1]) in eos_ids else "length" if generated_count >= 704 else "other"
        record = {
            "question_id": question_id, "sample_index": position,
            "worker_rank": worker_rank, "variant": variant,
            "worker_identity_sha256": identity["identity_sha256"],
            "answer": answer, "selected_chunk_ids": [c["chunk_id"] for c in selected],
            "selected_context_count": 12, "input_tokens": input_tokens,
            "output_tokens": len(tokenizer(answer, add_special_tokens=False)["input_ids"]),
            "generated_tokens_including_special": generated_count,
            "finish_reason": finish, "generation_latency_ms": latency,
        }
        record["record_sha256"] = _json_sha256(record)
        _atomic_json(records / f"{position:04d}.json", record)
        _write_worker_state(state, identity, position + 1, len(ids))
        LOGGER.info(
            "e19_generation_progress variant=%s device=%s completed=%d total=200 question_id=%s finish=%s",
            variant, device, position + 1, question_id, finish,
        )
    _write_worker_state(state, identity, len(ids), len(ids))
    return {"variant": variant, "device": device, "completed": len(ids)}


def _metrics(rows: list[dict[str, Any]], scores: list[dict[str, float]]) -> dict[str, Any]:
    diagnostics = [answer_diagnostics(row["answer"]) for row in rows]
    return {
        "meteor": fmean(x["meteor"] for x in scores),
        "rouge_l": fmean(x["rouge_l"] for x in scores),
        "mean_output_tokens": fmean(x["output_tokens"] for x in rows),
        "mean_answer_characters": fmean(len(x["answer"]) for x in rows),
        "mean_selected_contexts": fmean(x["selected_context_count"] for x in rows),
        "length_finish_rate": fmean(x["finish_reason"] == "length" for x in rows),
        "duplicate_line_rate": fmean(x["duplicate_line"] for x in diagnostics),
        "duplicate_sentence_rate": fmean(x["duplicate_sentence"] for x in diagnostics),
        "non_sentence_ending_rate": fmean(x["non_sentence_ending"] for x in diagnostics),
        "mean_generation_latency_ms": fmean(x["generation_latency_ms"] for x in rows),
    }


def finalize_eval(
    *, train_path: Path, dev_path: Path, output_directory: Path,
    old_training_directory: Path, new_training_directory: Path,
    config: E19Config,
) -> dict[str, Any]:
    ensure_nltk_resources(download=False)
    dev, _, ids = eval_sample(train_path, dev_path, config)
    adapter_hashes = {
        VARIANTS[0]: validate_control_adapter(old_training_directory, config)[0],
        VARIANTS[1]: validate_candidate_adapter(new_training_directory, config)[0],
    }
    by_variant, scores = {}, {v: [] for v in VARIANTS}
    for rank, variant in enumerate(VARIANTS):
        root = output_directory / "evaluation" / variant
        state = json.loads((root / "state.json").read_text(encoding="utf-8"))
        if state.get("complete") is not True or state.get("completed_count") != 200:
            raise E19Error(f"Incomplete E19 worker: {variant}")
        rows = []
        for index, question_id in enumerate(ids):
            row = json.loads((root / "records" / f"{index:04d}.json").read_text(encoding="utf-8"))
            if (
                row.get("question_id") != question_id
                or row.get("sample_index") != index
                or row.get("variant") != variant
                or row.get("selected_context_count") != 12
                or not isinstance(row.get("answer"), str) or not row["answer"].strip()
                or row.get("record_sha256")
                != _json_sha256({k: v for k, v in row.items() if k != "record_sha256"})
            ):
                raise E19Error(f"Invalid E19 result: {variant}/{index}")
            rows.append(row)
            scores[variant].append({
                "meteor": nltk_meteor_score(dev[question_id]["answer"], row["answer"]),
                "rouge_l": rouge_l_fmeasure(dev[question_id]["answer"], row["answer"]),
            })
        by_variant[variant] = rows
        _atomic_jsonl(root / "results.jsonl", rows)
    per_question, deltas = [], []
    for index, question_id in enumerate(ids):
        values = {v: scores[v][index] for v in VARIANTS}
        delta = values[VARIANTS[1]]["meteor"] - values[VARIANTS[0]]["meteor"]
        deltas.append(delta)
        per_question.append({
            "question_id": question_id, "sample_index": index,
            "scores": values, "meteor_delta": delta,
        })
    _atomic_jsonl(output_directory / "per_question_scores.jsonl", per_question)
    metrics = {v: _metrics(by_variant[v], scores[v]) for v in VARIANTS}
    report = {
        "schema_version": "1.0", "experiment_id": EXPERIMENT,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "sample_size": 200, "sample_scope": "next-200-after-old-dev200-and-e18-dev200",
        "control_variant": VARIANTS[0], "candidate_variant": VARIANTS[1],
        "only_changed_factor": "answer-only-context-LoRA versus metadata-aware-context-LoRA training",
        "metrics": metrics,
        "paired_delta_candidate_minus_control": {
            "meteor_mean": fmean(deltas),
            "meteor_bootstrap_95_ci": _bootstrap_ci(
                deltas, seed="e19-metadata-training-meteor-v1", iterations=10000
            ),
            "improved_questions": sum(x > 0 for x in deltas),
            "worsened_questions": sum(x < 0 for x in deltas),
            "tied_questions": sum(x == 0 for x in deltas),
        },
        "smoke_leader": max(VARIANTS, key=lambda v: metrics[v]["meteor"]),
        "promotion_allowed": False,
        "evidence": {
            "config_sha256": config.config_sha256,
            "sample_ids_sha256": _ids_sha(ids),
            "prepared_contexts_sha256": file_sha256(output_directory / "prepared/results.jsonl"),
            "control_adapter_sha256": adapter_hashes[VARIANTS[0]],
            "candidate_adapter_sha256": adapter_hashes[VARIANTS[1]],
            "control_results_sha256": file_sha256(output_directory / "evaluation" / VARIANTS[0] / "results.jsonl"),
            "candidate_results_sha256": file_sha256(output_directory / "evaluation" / VARIANTS[1] / "results.jsonl"),
        },
        "public_read": False, "holdout_untouched": True,
        "warning": "E19 evaluates on a new dev200 and cannot automatically promote a public stack.",
    }
    _atomic_json(output_directory / "report.json", report)
    return report


__all__ = [
    "CODE_VERSION", "E19Config", "E19Error", "EXPERIMENT", "VARIANTS", "code_sha",
    "eval_sample", "finalize_eval", "load_candidate_generator", "load_config",
    "load_control_generator", "prepare_eval_contexts", "prepare_metadata_train_records",
    "run_eval_worker", "validate_candidate_adapter", "validate_control_adapter", "validate_e00",
    "validate_training_preflight",
]
