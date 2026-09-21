"""E25 parent-aware QLoRA training and paired E21 dev evaluation."""
from __future__ import annotations

import copy
import hashlib
import importlib.metadata
import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean
from typing import Any

from . import e21_parent_context as parent
from .corpus import file_sha256
from .e02_answer import _atomic_json, _atomic_jsonl, make_messages
from .e03_rrf_grid import (
    _bootstrap_ci,
    _json_sha256,
    _load_worker_progress,
    _read_jsonl,
    _validate_worker_placement,
    _write_worker_state,
)
from .e08b_context_lora import _load_splits, _validate_context_rows, inference_packing
from .e18_source_metadata import enrich_context, save_once, scan_selected
from .e19_metadata_lora import (
    E19Config,
    _metrics,
    load_config as load_e19_config,
    validate_training_preflight as validate_e19_training_preflight,
)
from .evaluation.official import ensure_nltk_resources, nltk_meteor_score, rouge_l_fmeasure


EXPERIMENT = "E25-parent-aware-lora-train5636-dev200-v1"
CONTROL = "e21_e19_parent_expanded_max704"
CANDIDATE = "e25_parent_trained_parent_expanded_max704"
CODE_VERSION = "0.59.0"
LOG = logging.getLogger(__name__)


class E25Error(RuntimeError):
    pass


@dataclass(frozen=True)
class Config:
    raw: dict[str, Any]
    contract: dict[str, Any]
    path: Path
    e19: E19Config
    e21: Any

    @property
    def sha(self) -> str:
        return file_sha256(self.path)

    @property
    def config_sha256(self) -> str:
        return self.sha

    @property
    def generator_model_id(self) -> str:
        return self.e19.generator_model_id

    @property
    def generator_revision(self) -> str:
        return self.e19.generator_revision

    def section(self, key: str) -> dict[str, Any]:
        value = self.raw.get(key)
        if not isinstance(value, dict):
            raise E25Error(f"E25 derived section must be an object: {key}")
        return value


def load_config(root: Path, path: Path) -> Config:
    contract = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "schema_version", "experiment_id", "source_e19_config_path",
        "source_e19_config_sha256", "source_e21_config_path",
        "source_e21_config_sha256", "source_e21_control", "training",
        "evaluation", "run_contract",
    }
    if set(contract) != expected or contract.get("schema_version") != "1.0" or contract.get("experiment_id") != EXPERIMENT:
        raise E25Error("E25 config identity changed.")
    if (
        contract["source_e19_config_path"] != "configs/e19-metadata-aware-lora-train5636-eval200-v1.json"
        or contract["source_e19_config_sha256"] != "ce7d391f40475d69e0505c17e1644783d65c56e31d5f7a6b8169c2420fb5e480"
        or contract["source_e21_config_path"] != "configs/e21-parent-context-dev200-v1.json"
        or contract["source_e21_config_sha256"] != "fe8554d5f222e03a8f60ee181f8cfc4d1570719efaa6c56e2f7559fad1672ade"
    ):
        raise E25Error("Pinned E19/E21 configs changed.")
    e19_path = root / contract["source_e19_config_path"]
    e21_path = root / contract["source_e21_config_path"]
    if file_sha256(e19_path) != contract["source_e19_config_sha256"] or file_sha256(e21_path) != contract["source_e21_config_sha256"]:
        raise E25Error("Pinned E19/E21 config bytes changed.")
    e19 = load_e19_config(root, e19_path)
    e21 = parent.load_config(root, e21_path)
    if contract["source_e21_control"] != {
        "experiment_id": "E21-parent-context-dev200-v1",
        "variant": "parent_expanded_max704",
        "results_sha256": "cbe3d5e163e1983f031a06c767596fcebd91177e3b90625998a1deed28f6afe4",
        "code_sha256": "e5574861eb34ae3fdee1220d5260109ddd9f84feacea9c1ec46a78df8b5527c8",
        "adapter_sha256": "bfb9d8120337c4013b5dcb0b59c6d03beccfe9f9748e5a01ef62e1de64fb5edd",
        "meteor": 0.5633893466813007,
        "rouge_l": 0.5864053775086489,
        "sample_size": 200,
    }:
        raise E25Error("Pinned E21 control evidence changed.")
    if contract["training"] != {
        "sample_size": 5636,
        "supervision": "official-answer-only",
        "input": "question-plus-e21-parent-expanded-top12-contexts-with-e00-source-metadata",
        "parent_assembly": "exact-e21-policy-before-3072-token-training-pack",
        "maximum_sequence_tokens": 3072,
        "epochs": 1.0,
        "fresh_from_base": True,
        "lora_rank": 8,
        "lora_alpha": 16,
        "same_seed_and_optimization_as_e19": True,
    }:
        raise E25Error("E25 training contract changed.")
    if contract["evaluation"] != {
        "sample": "reuse-e19-e21-dev400-600",
        "sample_size": 200,
        "sample_ids_sha256": "dc17d6c9af9e941868c03cfc61fd17cdd1dc55aa5834beb3e75ba0fde32a5c73",
        "control_variant": CONTROL,
        "candidate_variant": CANDIDATE,
        "contexts": "byte-identical-e21-parent-expanded-prompts",
        "max_input_tokens": 8192,
        "max_new_tokens": 704,
        "workers": 2,
        "questions_per_worker": 100,
        "worker_partition": "contiguous-100-100",
        "promotion_allowed": False,
    }:
        raise E25Error("E25 evaluation contract changed.")
    if not contract["run_contract"] or not all(value is True for value in contract["run_contract"].values()):
        raise E25Error("E25 run contract lost an invariant.")
    raw = copy.deepcopy(e19.raw)
    raw["experiment_id"] = EXPERIMENT
    raw["train"]["input"] = contract["training"]["input"]
    raw["train"]["context_packing"] = contract["training"]["parent_assembly"]
    raw["lora"]["initialization"] = "fresh-from-pinned-base-not-e19-adapter"
    raw["inference"]["max_new_tokens"] = 704
    raw["scoring"]["bootstrap_seed"] = "uit-dsc-2026-e25-parent-aware-lora-v1"
    raw["run_contract"] = copy.deepcopy(contract["run_contract"])
    return Config(raw=raw, contract=contract, path=path, e19=e19, e21=e21)


def code_sha(root: Path) -> str:
    paths = sorted((root / "src/uit_dsc_fixed_rag").rglob("*.py"))
    paths += [root / "scripts" / name for name in (
        "run_e25_parent_aware_prepare_kaggle.py",
        "run_e25_parent_aware_train_kaggle.py",
        "run_e25_parent_aware_eval_kaggle.py",
    )]
    return _json_sha256({path.relative_to(root).as_posix(): file_sha256(path) for path in paths if path.is_file()})


def _ids_sha(ids: list[str]) -> str:
    return hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()


def validate_training_preflight(
    *, root: Path, e08a: Path, e00: Path, train: Path, dev: Path, config: Config,
) -> dict[str, Any]:
    evidence = validate_e19_training_preflight(
        project_root=root, e08a_directory=e08a, e00_directory=e00,
        train_path=train, dev_path=dev, config=config.e19,
    )
    budget = config.section("parameter_budget")
    total = budget["embedding"] + budget["generator"] + budget["adapter_parameter_cap"]
    if total != budget["maximum_stack_total"] or total >= budget["exclusive_limit"]:
        raise E25Error("E25 parameter budget violation.")
    return {
        **evidence,
        "source_e19_config_sha256": config.contract["source_e19_config_sha256"],
        "source_e21_config_sha256": config.contract["source_e21_config_sha256"],
        "config_sha256": config.sha,
        "code_sha256": code_sha(root),
        "parent_policy": config.e21.policy,
        "training_input": config.contract["training"]["input"],
        "maximum_sequence_tokens": 3072,
        "fresh_from_base": True,
        "lora_rank": 8,
        "lora_alpha": 16,
        "maximum_stack_parameters": total,
    }


def contexts_from_spans(spans: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Render merged E21 spans into the exact document-grouped context format."""
    merged = parent.merge_spans(spans)
    documents: dict[str, list[dict[str, Any]]] = {}
    for item in merged:
        documents.setdefault(item["document_id"], []).append(item)
    contexts = []
    for parts in documents.values():
        first = parts[0]
        lines = []
        for label, value in (("Tên nguồn", first.get("source_title")), ("Số văn bản", first.get("document_number"))):
            if value:
                lines.append(f"{label}: {value.strip()}")
        for item in parts:
            label = f"Điều {item['article_number']}" if item.get("article_number") else "Trích đoạn"
            if item.get("article_title"):
                label += f": {item['article_title']}"
            lines.extend([label, "Nội dung:", item["text"]])
        contexts.append({"text": "\n".join(lines), "article_number": None})
    return contexts


def _build_parent_lookup(e00: Path, seeds: dict[str, dict[str, Any]], documents: dict[str, dict[str, Any]], chunks_sha: str):
    by_doc = {document_id: [] for document_id in documents}
    digest = hashlib.sha256()
    with (e00 / "chunks.jsonl").open("rb") as stream:
        for line in stream:
            digest.update(line)
            item = json.loads(line)
            if item["document_id"] in by_doc:
                by_doc[item["document_id"]].append(item)
    if digest.hexdigest() != chunks_sha:
        raise E25Error("E00 chunks changed during parent scan.")
    lookup = {}
    for document_id, chunks in by_doc.items():
        for block in parent.article_blocks(chunks, documents[document_id]):
            for item in block:
                if item["chunk_id"] in seeds:
                    lookup[item["chunk_id"]] = block
    if set(lookup) != set(seeds):
        missing = sorted(set(seeds) - set(lookup))[:5]
        raise E25Error(f"Some training seeds have no parent block: {missing}")
    return lookup


def prepare_train_records(
    *, e08a: Path, e00: Path, train: Path, dev: Path, output: Path,
    config: Config, preflight: dict[str, Any], tokenizer: Any,
) -> dict[str, Any]:
    records, train_ids, _, _ = _load_splits(train, dev, config.e19)
    source = config.e19.section("source_e08a")
    source_path = e08a / source["train_results_path"]
    if file_sha256(source_path) != source["train_results_sha256"] or file_sha256(train) != preflight.get("train_sha256"):
        raise E25Error("E25 training inputs changed after preflight.")
    context_rows = _validate_context_rows(source_path, train_ids, "train5636", source)
    metadata = config.e19.contract["metadata_source"]
    wanted = {context["chunk_id"] for row in context_rows for context in row["contexts"]}
    seeds = scan_selected(e00 / "chunks.jsonl", "chunk_id", wanted, metadata["chunks_sha256"])
    documents = scan_selected(
        e00 / "documents.jsonl", "document_id",
        {item["document_id"] for item in seeds.values()}, metadata["documents_sha256"],
    )
    lookup = _build_parent_lookup(e00, seeds, documents, metadata["chunks_sha256"])
    packing = inference_packing(config.e19)

    def render_messages(messages):
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )

    def count_text(text):
        return len(tokenizer(text, add_special_tokens=False)["input_ids"])

    def count_messages(messages):
        return count_text(render_messages(messages))

    prepared_rows = []
    expansion_counts, input_lengths, body_lengths, context_counts = [], [], [], []
    for index, (question_id, context_row) in enumerate(zip(train_ids, context_rows)):
        units = []
        for rank, context in enumerate(context_row["contexts"]):
            seed = seeds[context["chunk_id"]]
            enrich_context(context, seed, documents[seed["document_id"]])
            units.append(parent.seed_unit(
                seed, rank, lookup[seed["chunk_id"]], documents[seed["document_id"]], config.e21.policy,
            ))
        prepared = {"question_id": question_id, "sample_index": index, "answers_used": False, "units": units}
        question = records[question_id]["question"]
        messages, spans, diagnostics = parent.pack(
            question, prepared, config.e21, count_messages, count_text, parent.VARIANTS[1],
        )
        contexts = contexts_from_spans(spans)
        if messages != make_messages(question=question, contexts=contexts, config=packing):
            raise E25Error("Parent render differs from E21 prompt format.")
        if diagnostics["skipped_seed_ranks"] or diagnostics["seed_ranks"] != list(range(12)):
            raise E25Error("E25 parent assembly dropped a frozen retrieval seed.")
        prepared_rows.append({
            "sample_index": index,
            "question_id": question_id,
            "question": question,
            "answer": records[question_id]["answer"],
            "contexts": contexts,
            "parent_packing": diagnostics,
            "full_parent_input_tokens": count_messages(messages),
            "supervision": "official-answer-only",
            "answers_are_retrieval_labels": False,
            "parent_policy": "exact-e21-parent-expanded-v1",
        })
        expansion_counts.append(len(diagnostics["expansions"]))
        input_lengths.append(count_messages(messages))
        body_lengths.append(diagnostics["body_characters"])
        context_counts.append(len(contexts))
    root = output / "training-data"
    root.mkdir(parents=True, exist_ok=True)
    result_path = root / "records.jsonl"
    _atomic_jsonl(result_path, prepared_rows)
    summary = {
        "schema_version": "1.0",
        "experiment_id": EXPERIMENT,
        "record_count": len(prepared_rows),
        "records_sha256": file_sha256(result_path),
        "sample_ids_sha256": _ids_sha(train_ids),
        "raw_source_contexts_sha256": source["train_results_sha256"],
        "metadata_policy": config.e19.contract["metadata_source"]["policy"],
        "parent_policy": config.e21.policy,
        "parent_policy_source_config_sha256": config.contract["source_e21_config_sha256"],
        "questions_with_expansion": sum(value > 0 for value in expansion_counts),
        "mean_expansion_actions": fmean(expansion_counts),
        "mean_full_parent_input_tokens": fmean(input_lengths),
        "maximum_full_parent_input_tokens": max(input_lengths),
        "mean_full_parent_body_characters": fmean(body_lengths),
        "minimum_rendered_contexts": min(context_counts),
        "maximum_rendered_contexts": max(context_counts),
        "mean_rendered_contexts": fmean(context_counts),
        "all_12_seeds_preserved_before_training_pack": True,
        "answers_are_retrieval_labels": False,
        "contains_dev_holdout_or_public": False,
        "training_sequence_cap": 3072,
    }
    _atomic_json(root / "summary.json", summary)
    return summary


def validate_adapter(training: Path, config: Config) -> tuple[str, dict[str, Any]]:
    final = training / "adapter-final"
    adapter = final / "adapter_model.safetensors"
    complete_path = final / "complete.json"
    if not adapter.is_file() or not complete_path.is_file():
        raise E25Error("Completed E25 adapter is missing.")
    complete = json.loads(complete_path.read_text(encoding="utf-8"))
    observed = file_sha256(adapter)
    if (
        complete.get("experiment_id") != EXPERIMENT
        or complete.get("config_sha256") != config.sha
        or complete.get("adapter_sha256") != observed
        or complete.get("fresh_from_base") is not True
        or complete.get("e19_adapter_loaded") is not False
        or complete.get("lora_rank") != 8
        or complete.get("lora_alpha") != 16
        or complete.get("training_sequence_cap") != 3072
        or complete.get("parent_policy_source_config_sha256") != config.contract["source_e21_config_sha256"]
    ):
        raise E25Error("Completed E25 adapter evidence changed.")
    return observed, complete


def load_generator(*, config: Config, training: Path, device: str):
    try:
        import torch
        from peft import PeftModel
        from transformers import AutoTokenizer, Qwen3_5ForCausalLM
    except ImportError as exc:  # pragma: no cover
        raise E25Error("Install Transformers and PEFT for E25 inference.") from exc
    if device not in {"cuda:0", "cuda:1"} or torch.cuda.device_count() != 2:
        raise E25Error("E25 evaluation requires GPU T4 x2.")
    validate_adapter(training, config)
    torch.cuda.set_device(int(device[-1]))
    tokenizer = AutoTokenizer.from_pretrained(
        config.generator_model_id, revision=config.generator_revision, trust_remote_code=False,
    )
    base = Qwen3_5ForCausalLM.from_pretrained(
        config.generator_model_id, revision=config.generator_revision,
        dtype=torch.float16, device_map={"": device}, low_cpu_mem_usage=True,
        trust_remote_code=False,
    )
    model = PeftModel.from_pretrained(base, training / "adapter-final", is_trainable=False)
    model.eval()
    parameters = sum(value.numel() for name, value in model.named_parameters() if "lora_" in name)
    if not 0 < parameters <= config.section("lora")["trainable_parameter_cap"]:
        raise E25Error("E25 adapter parameter count violates cap.")
    return model, tokenizer, _validate_worker_placement(model, device), parameters


def eval_sample(train: Path, dev: Path, config: Config):
    return parent.sample(train, dev, config.e21)


def load_e21_source(directory: Path, ids: list[str], config: Config):
    spec = config.contract["source_e21_control"]
    result_path = directory / "evaluation" / "parent_expanded_max704" / "results.jsonl"
    prepared_path = directory / "prepared" / "results.jsonl"
    state_path = result_path.parent / "state.json"
    if not result_path.is_file() or not prepared_path.is_file() or not state_path.is_file() or file_sha256(result_path) != spec["results_sha256"]:
        raise E25Error("Add the full byte-exact E21 output.")
    rows = _read_jsonl(result_path)
    state = json.loads(state_path.read_text(encoding="utf-8"))
    identity = state.get("run_identity", {})
    if (
        len(rows) != 200
        or state.get("complete") is not True
        or state.get("completed_count") != 200
        or state.get("assigned_count") != 200
        or identity.get("worker_rank") != 1
        or identity.get("variant") != "parent_expanded_max704"
        or identity.get("device") != "cuda:1"
        or identity.get("config_sha256") != config.contract["source_e21_config_sha256"]
        or identity.get("code_sha256") != spec["code_sha256"]
        or identity.get("adapter_sha256") != spec["adapter_sha256"]
        or identity.get("prepared_sha256") != file_sha256(prepared_path)
        or identity.get("identity_sha256") != _json_sha256({key: value for key, value in identity.items() if key != "identity_sha256"})
    ):
        raise E25Error("E21 control state or provenance changed.")
    for index, question_id in enumerate(ids):
        parent.validate_record(rows[index], question_id, index, 1, identity)
    prepared = parent.load_prepared(directory, ids, config.e21)
    return prepared, rows, identity


def eval_preflight(*, root: Path, source: Path, training: Path, train: Path, dev: Path, output: Path, config: Config):
    _, _, ids = eval_sample(train, dev, config)
    _, _, source_identity = load_e21_source(source, ids, config)
    adapter_sha, complete = validate_adapter(training, config)
    scorer = config.section("scoring")
    if file_sha256(root / scorer["official_scorer_path"]) != scorer["official_scorer_sha256"]:
        raise E25Error("Pinned scorer changed.")
    payload = {
        "experiment_id": EXPERIMENT,
        "code_sha256": code_sha(root),
        "config_sha256": config.sha,
        "source_identity_sha256": source_identity["identity_sha256"],
        "control_results_sha256": config.contract["source_e21_control"]["results_sha256"],
        "candidate_adapter_sha256": adapter_sha,
        "candidate_training_identity_sha256": complete["identity_sha256"],
        "sample_size": 200,
        "partitions": [100, 100],
        "max_new_tokens": 704,
        "same_e21_prompts": True,
        "public_read": False,
    }
    save_once(output / "preflight.json", payload)
    return payload


def check_eval_preflight(root: Path, output: Path, config: Config):
    path = output / "preflight.json"
    if not path.is_file():
        raise E25Error("Run E25 evaluation preflight first.")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("experiment_id") != EXPERIMENT or payload.get("code_sha256") != code_sha(root) or payload.get("config_sha256") != config.sha:
        raise E25Error("E25 evaluation preflight identity changed.")
    return payload


def assigned_indices(rank: int) -> list[int]:
    if rank not in (0, 1):
        raise E25Error("Worker rank must be 0 or 1.")
    return list(range(rank * 100, (rank + 1) * 100))


def validate_generation_record(row, question_id, index, rank, identity):
    if (
        row.get("question_id") != question_id
        or row.get("sample_index") != index
        or row.get("worker_rank") != rank
        or row.get("variant") != CANDIDATE
        or row.get("worker_identity_sha256") != identity["identity_sha256"]
        or not isinstance(row.get("answer"), str)
        or not row["answer"].strip()
        or row.get("record_sha256") != _json_sha256({key: value for key, value in row.items() if key != "record_sha256"})
    ):
        raise E25Error(f"Changed E25 answer record: {rank}/{index}")


def run_eval_worker(*, root: Path, source: Path, training: Path, train: Path, dev: Path, output: Path, config: Config, rank: int, device: str):
    import torch

    assigned = assigned_indices(rank)
    if device != f"cuda:{rank}":
        raise E25Error("E25 worker/GPU mismatch.")
    checked = check_eval_preflight(root, output, config)
    questions, _, ids = eval_sample(train, dev, config)
    prepared, controls, source_identity = load_e21_source(source, ids, config)
    if checked["source_identity_sha256"] != source_identity["identity_sha256"]:
        raise E25Error("E21 source changed after preflight.")
    runtime = {name: importlib.metadata.version(name) for name in source_identity["runtime"]}
    if runtime != source_identity["runtime"]:
        raise E25Error("Use the exact E21 runtime versions printed by Notebook 2.")
    model, tokenizer, placement, parameters = load_generator(config=config, training=training, device=device)
    adapter_sha, _ = validate_adapter(training, config)
    generation_sha = _json_sha256(model.generation_config.to_dict())
    if adapter_sha != checked["candidate_adapter_sha256"] or generation_sha != source_identity["generation_config_sha256"]:
        raise E25Error("E25 adapter or generation defaults changed.")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    def render(messages):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)

    def count(text):
        return len(tokenizer(text, add_special_tokens=False)["input_ids"])

    identity = {
        "code_sha256": code_sha(root),
        "config_sha256": config.sha,
        "source_identity_sha256": source_identity["identity_sha256"],
        "candidate_adapter_sha256": adapter_sha,
        "candidate_training_identity_sha256": checked["candidate_training_identity_sha256"],
        "generation_config_sha256": generation_sha,
        "runtime": runtime,
        "worker_rank": rank,
        "device": device,
        "variant": CANDIDATE,
        "assigned_indices": assigned,
        "device_map": placement,
        "adapter_parameters": parameters,
    }
    identity["identity_sha256"] = _json_sha256(identity)
    folder = output / "generation" / f"worker-{rank}"
    records_dir, state_path = folder / "records", folder / "state.json"
    records_dir.mkdir(parents=True, exist_ok=True)
    done = _load_worker_progress(
        records=records_dir, state_path=state_path, identity=identity,
        assigned_indices=assigned, sample_ids=ids,
    )
    for index in assigned[:done]:
        row = json.loads((records_dir / f"{index:04d}.json").read_text(encoding="utf-8"))
        validate_generation_record(row, ids[index], index, rank, identity)
    for completed, index in enumerate(assigned[done:], start=done + 1):
        question_id = ids[index]
        messages, spans, diagnostics = parent.pack(
            questions[question_id]["question"], prepared[index], config.e21,
            lambda value: count(render(value)), count, parent.VARIANTS[1],
        )
        prompt = render(messages)
        prompt_sha = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        if prompt_sha != controls[index].get("prompt_sha256"):
            raise E25Error(f"Candidate prompt differs from byte-exact E21 control: {question_id}")
        inputs = {key: value.to(device) for key, value in tokenizer(prompt, add_special_tokens=False, return_tensors="pt").items()}
        started = time.perf_counter()
        with torch.inference_mode():
            generated = model.generate(
                **inputs, do_sample=False, num_beams=1, max_new_tokens=704, use_cache=True,
            )
        latency = (time.perf_counter() - started) * 1000
        new_ids = generated[0, inputs["input_ids"].shape[1]:]
        answer = tokenizer.decode(new_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False).strip()
        if not answer:
            raise E25Error(f"Empty E25 answer: {question_id}")
        eos = tokenizer.eos_token_id
        eos_ids = set(eos if isinstance(eos, list) else [eos])
        finish = "eos" if len(new_ids) and int(new_ids[-1]) in eos_ids else "length" if len(new_ids) >= 704 else "other"
        row = {
            "question_id": question_id,
            "sample_index": index,
            "worker_rank": rank,
            "variant": CANDIDATE,
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
            "prompt_sha256": prompt_sha,
            "evidence_spans": [{key: value for key, value in item.items() if key != "text"} for item in spans],
        }
        row["record_sha256"] = _json_sha256(row)
        _atomic_json(records_dir / f"{index:04d}.json", row)
        _write_worker_state(state_path, identity, completed, 100)
        LOG.info("e25_generation worker=%d device=%s completed=%d total=100 question_id=%s finish=%s", rank, device, completed, question_id, finish)
    return {"worker": rank, "completed": 100}


def finalize_eval(*, root: Path, source: Path, training: Path, train: Path, dev: Path, output: Path, config: Config):
    checked = check_eval_preflight(root, output, config)
    ensure_nltk_resources(download=False)
    questions, _, ids = eval_sample(train, dev, config)
    _, controls, source_identity = load_e21_source(source, ids, config)
    if checked["source_identity_sha256"] != source_identity["identity_sha256"]:
        raise E25Error("E21 source changed before final scoring.")
    candidates = []
    for rank in range(2):
        folder = output / "generation" / f"worker-{rank}"
        state = json.loads((folder / "state.json").read_text(encoding="utf-8"))
        identity = state.get("run_identity", {})
        if (
            state.get("complete") is not True
            or state.get("completed_count") != 100
            or state.get("assigned_count") != 100
            or identity.get("assigned_indices") != assigned_indices(rank)
            or identity.get("worker_rank") != rank
            or identity.get("device") != f"cuda:{rank}"
            or identity.get("variant") != CANDIDATE
            or identity.get("config_sha256") != config.sha
            or identity.get("code_sha256") != code_sha(root)
            or identity.get("source_identity_sha256") != source_identity["identity_sha256"]
            or identity.get("candidate_adapter_sha256") != checked["candidate_adapter_sha256"]
            or identity.get("runtime") != source_identity["runtime"]
            or identity.get("generation_config_sha256") != source_identity["generation_config_sha256"]
            or identity.get("identity_sha256") != _json_sha256({key: value for key, value in identity.items() if key != "identity_sha256"})
        ):
            raise E25Error("Incomplete or changed E25 worker state.")
        for index in assigned_indices(rank):
            row = json.loads((folder / "records" / f"{index:04d}.json").read_text(encoding="utf-8"))
            validate_generation_record(row, ids[index], index, rank, identity)
            if row["prompt_sha256"] != controls[index]["prompt_sha256"]:
                raise E25Error("E25/control prompts differ during final scoring.")
            candidates.append(row)
    by_variant = {CONTROL: controls, CANDIDATE: candidates}
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
    delta = [candidate["meteor"] - control["meteor"] for candidate, control in zip(scores[CANDIDATE], scores[CONTROL])]
    result_path = output / "results.jsonl"
    _atomic_jsonl(result_path, candidates)
    _atomic_jsonl(output / "per_question_scores.jsonl", [
        {"question_id": question_id, "scores": {variant: scores[variant][index] for variant in scores}}
        for index, question_id in enumerate(ids)
    ])
    metrics = {variant: _metrics(rows, scores[variant]) for variant, rows in by_variant.items()}
    report = {
        "schema_version": "1.0",
        "experiment_id": EXPERIMENT,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "sample_size": 200,
        "sample_scope": "same-repeated-e19-e21-dev400-600",
        "worker_partitions": [100, 100],
        "max_new_tokens": 704,
        "control_variant": CONTROL,
        "candidate_variant": CANDIDATE,
        "only_changed_factor": "E19 metadata-trained LoRA versus E25 metadata-plus-parent-trained LoRA",
        "same_e21_prompts": True,
        "metrics": metrics,
        "paired_candidate_minus_control": {
            "meteor_mean": fmean(delta),
            "meteor_bootstrap_95_ci": _bootstrap_ci(delta, seed="e25-parent-aware-v1", iterations=10000),
            "improved": sum(value > 0 for value in delta),
            "worsened": sum(value < 0 for value in delta),
            "tied": sum(value == 0 for value in delta),
        },
        "smoke_leader": max(metrics, key=lambda variant: metrics[variant]["meteor"]),
        "promotion_allowed": False,
        "public_read": False,
        "holdout_untouched": True,
        "evidence": {
            "config_sha256": config.sha,
            "code_sha256": code_sha(root),
            "source_identity_sha256": source_identity["identity_sha256"],
            "control_results_sha256": config.contract["source_e21_control"]["results_sha256"],
            "candidate_adapter_sha256": checked["candidate_adapter_sha256"],
            "candidate_results_sha256": file_sha256(result_path),
        },
        "warning": "Repeated dev-200; this is model selection evidence, not a private-test guarantee.",
    }
    _atomic_json(output / "report.json", report)
    return report
