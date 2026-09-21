"""Fresh Vi-Qwen2-3B-RAG QLoRA on E19's exact official metadata training records."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from typing import Any

from .corpus import file_sha256
from .e02_answer import _atomic_json, make_messages
from .e03_rrf_grid import _bootstrap_ci, _json_sha256, _load_worker_progress, _read_jsonl, _write_worker_state
from .e07_generator_ab import _as_text_chat_messages
from .e08b_context_lora import inference_packing
from .e13_e08b_max768_tailtrim import trim_repeated_tail
from .e19_metadata_lora import load_config as load_e19_config, validate_candidate_adapter
from .e21_parent_context import render as render_parent
from .e31_long_token_suffix_trim import trim_long_token_suffix
from .e36_failure_audit import load_config as load_e36_config
from .e18_source_metadata import save_once
from .evaluation.official import ensure_nltk_resources, nltk_meteor_score, rouge_l_fmeasure

EXPERIMENT = "E38-viqwen-metadata-lora-train5636-dev120-v1"
LOG = logging.getLogger(__name__)


class E38Error(RuntimeError):
    pass


@dataclass(frozen=True)
class Config:
    raw: dict[str, Any]
    path: Path
    e19: Any
    e36: Any

    @property
    def sha(self) -> str:
        return file_sha256(self.path)


def load_config(root: Path, path: Path) -> Config:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if (set(raw) != {"schema_version", "experiment_id", "source_e19_config_path",
                     "source_e19_config_sha256", "source_e36_config_path",
                     "source_e36_config_sha256", "organizer_csv_sha256", "model",
                     "training", "evaluation", "runtime"}
            or raw["schema_version"] != "1.0" or raw["experiment_id"] != EXPERIMENT
            or raw["source_e19_config_path"] != "configs/e19-metadata-aware-lora-train5636-eval200-v1.json"
            or raw["source_e19_config_sha256"] != "ce7d391f40475d69e0505c17e1644783d65c56e31d5f7a6b8169c2420fb5e480"
            or raw["source_e36_config_path"] != "configs/e36-e32-failure-audit-dev120-v1.json"
            or raw["source_e36_config_sha256"] != "c98e779efd2be83c71402dffa9627e60bc5b9fce12b780dabce98ff1d566d0e7"
            or raw["organizer_csv_sha256"] != "0b1de15db2869990739a4e6dd57186db14b09a5a2607ad834c927d0bab604e95"
            or raw["model"] != {
                "id": "AITeamVN/Vi-Qwen2-3B-RAG", "revision": "eaf427c24d86066a2b35828c499b7db3af321227",
                "architecture": "Qwen2ForCausalLM", "checkpoint_parameters": 3085938688,
                "embedding_parameters": 567754752, "adapter_parameter_cap": 50000000,
                "strict_stack_limit": 4000000000,
            }
            or raw["training"] != {
                "source_e19_experiment_id": "E19-metadata-aware-lora-train5636-eval200-v1",
                "source_e19_adapter_sha256": "bfb9d8120337c4013b5dcb0b59c6d03beccfe9f9748e5a01ef62e1de64fb5edd",
                "source_training_records_sha256": "1d7c5b97e539cfe33ce036a66de4595c959d30b0a39b10e10c2fba415ce3d46b",
                "records": 5636, "retrieved_contexts_per_record": 12, "maximum_sequence_tokens": 3072,
                "epochs": 1.0, "rank": 8, "alpha": 16, "dropout": 0.05,
                "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                "seed": 20260830, "learning_rate": 0.0001,
                "per_device_train_batch_size": 1, "gradient_accumulation_steps": 4,
                "warmup_steps": 0.03, "lr_scheduler": "cosine", "weight_decay": 0.0,
                "save_steps": 32, "quantization": "nf4-double-quant-float16",
                "fresh_from_pinned_checkpoint": True,
                "official_answers_supervise_generation_only": True,
                "metadata_and_top12_exactly_reused_from_e19": True,
            }
            or raw["evaluation"] != {
                "source_e36_experiment_id": "E36-e32-failure-audit-dev120-v1",
                "sample_size": 120,
                "sample_ids_sha256": "ffc390de97c137c571a768e1cc1e59b10be1245ed29529da39725528d05bd3e5",
                "control_variant": "e32_parent_max1024_tailtrim_longtoken",
                "control_meteor": 0.5627316657089687, "control_rouge_l": 0.561259109088349,
                "same_saved_e21_spans": True, "max_input_tokens": 8192,
                "max_new_tokens": 1024, "greedy": True,
                "postprocessing": ["conservative-consecutive-tail-block-trim-v1",
                                   "exact-consecutive-long-token-suffix-trim-v1"],
                "repeated_dev_smoke_only": True, "no_public_or_private_read": True,
            }
            or raw["runtime"] != {"torch": "2.10.0+cu128", "transformers": "5.16.1",
                                  "peft": "0.19.1", "accelerate": "1.13.0", "nltk": "3.7"}):
        raise E38Error("E38 reviewed experiment contract changed.")
    if sum(raw["model"][key] for key in ("checkpoint_parameters", "embedding_parameters", "adapter_parameter_cap")) >= raw["model"]["strict_stack_limit"]:
        raise E38Error("Vi-Qwen stack exceeds the BTC parameter cap.")
    e19_path = root / raw["source_e19_config_path"]
    e36_path = root / raw["source_e36_config_path"]
    if (file_sha256(e19_path) != raw["source_e19_config_sha256"]
            or file_sha256(e36_path) != raw["source_e36_config_sha256"]):
        raise E38Error("Pinned E19/E36 source configuration changed.")
    return Config(raw, path, load_e19_config(root, e19_path), load_e36_config(root, e36_path))


def code_sha(root: Path) -> str:
    paths = [root / "src/uit_dsc_fixed_rag/e38_viqwen_metadata_lora.py",
             root / "scripts/run_e38_viqwen_train_kaggle.py",
             root / "scripts/run_e38_viqwen_eval_kaggle.py"]
    return _json_sha256({p.relative_to(root).as_posix(): file_sha256(p) for p in paths})


def _ids_sha(ids: list[str]) -> str:
    return hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()


def _training_source(e19: Path, config: Config) -> tuple[dict[str, Any], Path]:
    adapter_sha, complete = validate_candidate_adapter(e19 / "training", config.e19)
    summary_path = e19 / "training-data/summary.json"
    records_path = e19 / "training-data/records.jsonl"
    if not summary_path.is_file() or not records_path.is_file():
        raise E38Error("Add complete E19 Notebook 1 output, including training-data.")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    source = config.raw["training"]
    if (adapter_sha != source["source_e19_adapter_sha256"]
            or complete.get("training_records_sha256") != source["source_training_records_sha256"]
            or complete.get("metadata_policy") != config.e19.contract["metadata_source"]["policy"]
            or summary.get("experiment_id") != source["source_e19_experiment_id"]
            or summary.get("records_sha256") != source["source_training_records_sha256"]
            or file_sha256(records_path) != source["source_training_records_sha256"]
            or summary.get("record_count") != source["records"]
            or summary.get("context_candidates_per_record") != 12
            or summary.get("same_context_bodies_and_order") is not True
            or summary.get("answers_are_retrieval_labels") is not False
            or summary.get("contains_dev_holdout_or_public") is not False
            or summary.get("metadata_policy") != config.e19.contract["metadata_source"]["policy"]):
        raise E38Error("E19 official metadata training source changed.")
    return summary, records_path


def _organizer_csv(root: Path, config: Config) -> Path:
    hits = list((root / "data/raw").glob("*Sheet1.csv"))
    if len(hits) != 1 or file_sha256(hits[0]) != config.raw["organizer_csv_sha256"]:
        raise E38Error("Missing or changed BTC model list.")
    if "https://huggingface.co/" + config.raw["model"]["id"] not in hits[0].read_text(encoding="utf-8-sig"):
        raise E38Error("Vi-Qwen checkpoint is absent from the BTC model list.")
    return hits[0]


def training_preflight(*, root: Path, e19: Path, output: Path, config: Config) -> dict[str, Any]:
    summary, _ = _training_source(e19, config)
    csv = _organizer_csv(root, config)
    payload = {
        "experiment_id": EXPERIMENT, "code_sha256": code_sha(root),
        "config_sha256": config.sha, "organizer_csv_sha256": file_sha256(csv),
        "training_records_sha256": summary["records_sha256"],
        "training_sample_ids_sha256": summary["sample_ids_sha256"],
        "metadata_policy": summary["metadata_policy"],
        "model_id": config.raw["model"]["id"],
        "model_revision": config.raw["model"]["revision"],
        "checkpoint_parameters": config.raw["model"]["checkpoint_parameters"],
        "maximum_stack_parameters": sum(config.raw["model"][key] for key in
                                        ("checkpoint_parameters", "embedding_parameters", "adapter_parameter_cap")),
        "dev_public_private_answers_loaded": False,
    }
    save_once(output / "preflight.json", payload)
    return payload


def check_training_preflight(root: Path, e19: Path, output: Path, config: Config) -> dict[str, Any]:
    path = output / "preflight.json"
    if not path.is_file():
        raise E38Error("Run E38 training preflight first.")
    saved = json.loads(path.read_text(encoding="utf-8"))
    summary, _ = _training_source(e19, config)
    if (saved.get("experiment_id") != EXPERIMENT
            or saved.get("code_sha256") != code_sha(root)
            or saved.get("config_sha256") != config.sha
            or saved.get("training_records_sha256") != summary["records_sha256"]
            or saved.get("training_sample_ids_sha256") != summary["sample_ids_sha256"]
            or saved.get("model_revision") != config.raw["model"]["revision"]):
        raise E38Error("E38 training preflight identity changed.")
    return saved


def count_checkpoint_tensors(model_cache: Path, config: Config) -> int:
    from safetensors import safe_open

    config_path = model_cache / "config.json"
    if not config_path.is_file():
        raise E38Error("Pinned Vi-Qwen model snapshot is incomplete.")
    model_config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.raw["model"]["architecture"] not in model_config.get("architectures", []):
        raise E38Error("Vi-Qwen model architecture changed.")
    single = model_cache / "model.safetensors"
    weights = [single] if single.is_file() else sorted(model_cache.glob("model-*-of-*.safetensors"))
    if not weights:
        raise E38Error("Pinned Vi-Qwen safetensors are missing.")
    names: set[str] = set()
    total = 0
    for path in weights:
        with safe_open(str(path), framework="pt", device="cpu") as reader:
            for name in reader.keys():
                if name in names:
                    raise E38Error("Duplicate Vi-Qwen tensor across shards.")
                names.add(name)
                shape = reader.get_slice(name).get_shape()
                size = 1
                for dimension in shape:
                    size *= dimension
                total += size
    if total != config.raw["model"]["checkpoint_parameters"]:
        raise E38Error(f"Vi-Qwen checkpoint parameter count changed: {total}.")
    return total


def render_viqwen(tokenizer: Any, messages: list[dict[str, Any]]) -> str:
    return tokenizer.apply_chat_template(_as_text_chat_messages(messages),
                                         tokenize=False, add_generation_prompt=True)


class TokenizedDataset:
    """Answer-only supervision; same E19 contexts/metadata, Vi-Qwen tokenizer."""

    def __init__(self, rows: list[dict[str, Any]], tokenizer: Any, config: Config):
        self.items: list[dict[str, list[int]]] = []
        self.context_counts: list[int] = []
        self.sequence_lengths: list[int] = []
        self.truncated_chars: list[int] = []
        maximum = config.raw["training"]["maximum_sequence_tokens"]
        packing = inference_packing(config.e19)
        for row in rows:
            contexts = row.get("contexts")
            if (not isinstance(contexts, list) or len(contexts) != 12
                    or row.get("supervision") != "official-answer-only"
                    or row.get("answers_are_retrieval_labels") is not False
                    or row.get("metadata_policy") != config.e19.contract["metadata_source"]["policy"]
                    or not isinstance(row.get("answer"), str) or not row["answer"].strip()):
                raise E38Error(f"Changed E19 training record: {row.get('question_id')}")
            answer_ids = tokenizer(row["answer"], add_special_tokens=False)["input_ids"] + [tokenizer.eos_token_id]
            if len(answer_ids) >= maximum:
                raise E38Error(f"Official answer cannot fit without truncation: {row['question_id']}")
            chosen: list[dict[str, Any]] = []
            prompt_ids: list[int] | None = None
            for context in contexts:
                messages = make_messages(question=row["question"], contexts=[*chosen, context], config=packing)
                ids = tokenizer(render_viqwen(tokenizer, messages), add_special_tokens=False)["input_ids"]
                if len(ids) + len(answer_ids) <= maximum:
                    chosen.append(context)
                    prompt_ids = ids
            trimmed = 0
            if not chosen:
                first = contexts[0]
                original = str(first["text"]).strip()
                low = config.e19.section("train")["minimum_truncated_context_characters"]
                high = len(original)
                best: tuple[dict[str, Any], list[int], int] | None = None
                while low <= high:
                    midpoint = (low + high) // 2
                    trial = {**first, "text": original[:midpoint].rstrip()}
                    messages = make_messages(question=row["question"], contexts=[trial], config=packing)
                    ids = tokenizer(render_viqwen(tokenizer, messages), add_special_tokens=False)["input_ids"]
                    if len(ids) + len(answer_ids) <= maximum:
                        best = (trial, ids, midpoint)
                        low = midpoint + 1
                    else:
                        high = midpoint - 1
                if best is not None:
                    chosen, prompt_ids, trimmed = [best[0]], best[1], len(original) - best[2]
            if not chosen or prompt_ids is None:
                raise E38Error(f"No non-truncated Vi-Qwen training example fits: {row['question_id']}")
            ids = prompt_ids + answer_ids
            self.items.append({"input_ids": ids, "attention_mask": [1] * len(ids),
                               "labels": [-100] * len(prompt_ids) + answer_ids})
            self.context_counts.append(len(chosen))
            self.sequence_lengths.append(len(ids))
            self.truncated_chars.append(trimmed)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        return self.items[index]

    def summary(self) -> dict[str, Any]:
        return {"records": len(self.items), "mean_selected_contexts": fmean(self.context_counts),
                "minimum_selected_contexts": min(self.context_counts),
                "records_with_all_12_contexts": sum(x == 12 for x in self.context_counts),
                "maximum_sequence_tokens": max(self.sequence_lengths),
                "mean_sequence_tokens": fmean(self.sequence_lengths),
                "context_tail_truncation_count": sum(x > 0 for x in self.truncated_chars),
                "answer_truncation_count": 0}


class CausalCollator:
    def __init__(self, tokenizer: Any):
        self.pad = tokenizer.pad_token_id

    def __call__(self, features: list[dict[str, list[int]]]) -> dict[str, Any]:
        import torch

        width = max(len(item["input_ids"]) for item in features)
        batch = {"input_ids": [], "attention_mask": [], "labels": []}
        for item in features:
            missing = width - len(item["input_ids"])
            batch["input_ids"].append(item["input_ids"] + [self.pad] * missing)
            batch["attention_mask"].append(item["attention_mask"] + [0] * missing)
            batch["labels"].append(item["labels"] + [-100] * missing)
        return {key: torch.tensor(value, dtype=torch.long) for key, value in batch.items()}


def validate_e38_adapter(training: Path, config: Config) -> tuple[str, dict[str, Any]]:
    final = training / "adapter-final"
    adapter = final / "adapter_model.safetensors"
    complete_path = final / "complete.json"
    if not adapter.is_file() or not complete_path.is_file():
        raise E38Error("Add the complete E38 Notebook 1 output.")
    complete = json.loads(complete_path.read_text(encoding="utf-8"))
    observed = file_sha256(adapter)
    if (complete.get("experiment_id") != EXPERIMENT
            or complete.get("config_sha256") != config.sha
            or complete.get("model_id") != config.raw["model"]["id"]
            or complete.get("model_revision") != config.raw["model"]["revision"]
            or complete.get("model_parameters") != config.raw["model"]["checkpoint_parameters"]
            or complete.get("training_records_sha256") != config.raw["training"]["source_training_records_sha256"]
            or complete.get("adapter_sha256") != observed
            or complete.get("fresh_from_pinned_checkpoint") is not True
            or complete.get("e19_adapter_loaded") is not False
            or complete.get("lora_rank") != 8 or complete.get("lora_alpha") != 16
            or not 0 < complete.get("trainable_parameters", 0) <= config.raw["model"]["adapter_parameter_cap"]):
        raise E38Error("Completed E38 adapter identity changed.")
    return observed, complete


def _dev_source(e36: Path, config: Config) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    report_path = e36 / "report.json"
    rows_path = e36 / "review_all120.jsonl"
    if not report_path.is_file() or not rows_path.is_file():
        raise E38Error("Add complete E36 audit output including review_all120.jsonl.")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    pin = config.raw["evaluation"]
    if (report.get("experiment_id") != pin["source_e36_experiment_id"]
            or report.get("sample_size") != 120 or report.get("exact_prompts_verified") != 120
            or report.get("baseline_meteor") != pin["control_meteor"]
            or report.get("baseline_rouge_l") != pin["control_rouge_l"]
            or report.get("evidence", {}).get("sample_ids_sha256") != pin["sample_ids_sha256"]
            or report.get("files", {}).get("review_all120.jsonl") != file_sha256(rows_path)
            or report.get("public_read") is not False or report.get("private_untouched") is not True):
        raise E38Error("E36 source audit changed.")
    rows = _read_jsonl(rows_path)
    ids = [str(r.get("question_id")) for r in rows]
    if (len(rows) != 120 or _ids_sha(ids) != pin["sample_ids_sha256"]
            or any(r.get("sample_index") != i or r.get("prompt_verified") is not True
                   or not isinstance(r.get("question"), str)
                   or not isinstance(r.get("packed_spans"), list) or not r["packed_spans"]
                   or not isinstance(r.get("reference_answer"), str)
                   or not isinstance(r.get("model_answer"), str)
                   or r.get("record_sha256") != _json_sha256({k: v for k, v in r.items() if k != "record_sha256"})
                   for i, r in enumerate(rows))):
        raise E38Error("E36 question order or record hashes changed.")
    return report, rows


def eval_preflight(*, root: Path, e36: Path, training: Path,
                   output: Path, config: Config) -> dict[str, Any]:
    report, rows = _dev_source(e36, config)
    adapter_sha, complete = validate_e38_adapter(training, config)
    prompt_rows = [{"question_id": row["question_id"], "sample_index": row["sample_index"],
                    "question": row["question"], "packed_spans": row["packed_spans"]}
                   for row in rows]
    prompt_path = output / "prompt-only/records.jsonl"
    save_once(prompt_path, prompt_rows, jsonl=True)
    payload = {
        "experiment_id": EXPERIMENT, "stage": "dev120-evaluation",
        "code_sha256": code_sha(root), "config_sha256": config.sha,
        "e36_report_sha256": file_sha256(e36 / "report.json"),
        "e36_rows_sha256": report["files"]["review_all120.jsonl"],
        "prompt_only_rows_sha256": file_sha256(prompt_path),
        "adapter_sha256": adapter_sha,
        "training_identity_sha256": complete["identity_sha256"],
        "sample_ids_sha256": config.raw["evaluation"]["sample_ids_sha256"],
        "model_revision": config.raw["model"]["revision"],
        "reference_answers_used_during_generation": False,
    }
    save_once(output / "preflight.json", payload)
    return payload


def _eval_check(root: Path, e36: Path, training: Path, output: Path,
                config: Config) -> dict[str, Any]:
    path = output / "preflight.json"
    if not path.is_file():
        raise E38Error("Run E38 evaluation preflight first.")
    saved = json.loads(path.read_text(encoding="utf-8"))
    adapter_sha, complete = validate_e38_adapter(training, config)
    if (saved.get("experiment_id") != EXPERIMENT or saved.get("stage") != "dev120-evaluation"
            or saved.get("code_sha256") != code_sha(root)
            or saved.get("config_sha256") != config.sha
            or saved.get("e36_report_sha256") != file_sha256(e36 / "report.json")
            or saved.get("e36_rows_sha256") != file_sha256(e36 / "review_all120.jsonl")
            or saved.get("prompt_only_rows_sha256") != file_sha256(output / "prompt-only/records.jsonl")
            or saved.get("adapter_sha256") != adapter_sha
            or saved.get("training_identity_sha256") != complete["identity_sha256"]):
        raise E38Error("E38 evaluation inputs changed after preflight.")
    return saved


def _eval_prompt(row: dict[str, Any], tokenizer: Any, config: Config) -> tuple[str, int]:
    spans = row["packed_spans"]
    packing = inference_packing(config.e19)
    for kept in range(len(spans), 0, -1):
        messages, _ = render_parent(row["question"], spans[:kept], packing)
        prompt = render_viqwen(tokenizer, messages)
        tokens = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
        if tokens <= config.raw["evaluation"]["max_input_tokens"]:
            return prompt, kept
    raise E38Error(f"Even first E21 span cannot fit Vi-Qwen: {row['question_id']}")


def run_eval_worker(*, root: Path, e36: Path, training: Path, output: Path,
                    config: Config, model_cache: Path, rank: int, device: str) -> dict[str, Any]:
    import torch
    from peft import PeftModel
    from transformers import AutoTokenizer, Qwen2ForCausalLM

    if rank not in (0, 1) or device != f"cuda:{rank}" or torch.cuda.device_count() != 2:
        raise E38Error("Evaluate E38 with T4 x2, rank 0/cuda:0 or rank 1/cuda:1.")
    saved = _eval_check(root, e36, training, output, config)
    if model_cache.resolve().name != config.raw["model"]["revision"]:
        raise E38Error("Wrong Vi-Qwen model cache revision.")
    count_checkpoint_tensors(model_cache, config)
    rows = _read_jsonl(output / "prompt-only/records.jsonl")
    if (len(rows) != 120 or _ids_sha([str(r.get("question_id")) for r in rows])
            != config.raw["evaluation"]["sample_ids_sha256"]
            or any(set(r) != {"question_id", "sample_index", "question", "packed_spans"}
                   or r["sample_index"] != i for i, r in enumerate(rows))):
        raise E38Error("E38 prompt-only dev rows changed.")
    runtime = {name: importlib.metadata.version(name)
               for name in ("torch", "transformers", "peft", "accelerate")}
    if runtime != {name: config.raw["runtime"][name] for name in runtime}:
        raise E38Error(f"Wrong E38 generation runtime: {runtime}")
    torch.cuda.set_device(rank)
    tokenizer = AutoTokenizer.from_pretrained(str(model_cache), trust_remote_code=False)
    base = Qwen2ForCausalLM.from_pretrained(str(model_cache), dtype=torch.float16,
                                            device_map={"": device}, low_cpu_mem_usage=True,
                                            trust_remote_code=False)
    model = PeftModel.from_pretrained(base, training / "adapter-final", is_trainable=False)
    model.eval()
    generation_sha = _json_sha256(model.generation_config.to_dict())
    indices = list(range(rank * 60, (rank + 1) * 60))
    identity = {
        "code_sha256": code_sha(root), "config_sha256": config.sha,
        "e36_rows_sha256": saved["e36_rows_sha256"],
        "adapter_sha256": saved["adapter_sha256"],
        "model_revision": config.raw["model"]["revision"],
        "generation_config_sha256": generation_sha,
        "sample_ids_sha256": config.raw["evaluation"]["sample_ids_sha256"],
        "runtime": runtime, "worker_rank": rank, "device": device,
        "assigned_indices": indices, "max_new_tokens": 1024,
    }
    identity["identity_sha256"] = _json_sha256(identity)
    folder = output / f"generation/worker-{rank}"
    records, state = folder / "records", folder / "state.json"
    records.mkdir(parents=True, exist_ok=True)
    ids = [r["question_id"] for r in rows]
    done = _load_worker_progress(records=records, state_path=state, identity=identity,
                                 assigned_indices=indices, sample_ids=ids)
    if {p.name for p in records.glob("*.json")} != {f"{i:04d}.json" for i in indices[:done]}:
        raise E38Error("Unexpected or incomplete resumed E38 records.")
    for index in indices[:done]:
        old = json.loads((records / f"{index:04d}.json").read_text(encoding="utf-8"))
        if old.get("record_sha256") != _json_sha256({k: v for k, v in old.items() if k != "record_sha256"}):
            raise E38Error(f"Changed resumed E38 record: {index}")
    for offset, index in enumerate(indices[done:], start=done + 1):
        row = rows[index]
        prompt, kept = _eval_prompt(row, tokenizer, config)
        prompt_sha = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        inputs = {key: value.to(device) for key, value in tokenizer(
            prompt, add_special_tokens=False, return_tensors="pt").items()}
        started = time.perf_counter()
        with torch.inference_mode():
            generated = model.generate(**inputs, do_sample=False, num_beams=1,
                                       repetition_penalty=1.0, no_repeat_ngram_size=0,
                                       max_new_tokens=1024, use_cache=True)
        latency = (time.perf_counter() - started) * 1000
        new_ids = generated[0, inputs["input_ids"].shape[1]:]
        answer = tokenizer.decode(new_ids, skip_special_tokens=True,
                                  clean_up_tokenization_spaces=False).strip()
        if not answer:
            raise E38Error(f"Empty E38 answer: {row['question_id']}")
        eos = model.generation_config.eos_token_id
        eos_ids = set(eos if isinstance(eos, list) else [eos])
        finish = "eos" if len(new_ids) and int(new_ids[-1]) in eos_ids else (
            "length" if len(new_ids) >= 1024 else "other")
        record = {
            "question_id": row["question_id"], "sample_index": index,
            "worker_rank": rank, "worker_identity_sha256": identity["identity_sha256"],
            "prompt_sha256": prompt_sha, "input_tokens": inputs["input_ids"].shape[1],
            "kept_e21_spans": kept, "original_e21_spans": len(row["packed_spans"]),
            "answer": answer, "generated_tokens_including_special": len(new_ids),
            "finish_reason": finish, "generation_latency_ms": latency,
        }
        record["record_sha256"] = _json_sha256(record)
        _atomic_json(records / f"{index:04d}.json", record)
        _write_worker_state(state, identity, offset, len(indices))
        LOG.info("E38 rank=%d completed=%d/%d question_id=%s finish=%s kept=%d",
                 rank, offset, len(indices), row["question_id"], finish, kept)
    return {"worker_rank": rank, "completed": len(indices), "generated": len(indices)}


def finalize_eval(*, root: Path, e36: Path, training: Path,
                  output: Path, config: Config) -> dict[str, Any]:
    saved = _eval_check(root, e36, training, output, config)
    ensure_nltk_resources(download=False)
    _, source = _dev_source(e36, config)
    generated: dict[int, dict[str, Any]] = {}
    for rank in (0, 1):
        folder = output / f"generation/worker-{rank}"
        state = json.loads((folder / "state.json").read_text(encoding="utf-8"))
        identity = state.get("run_identity", {})
        indices = list(range(rank * 60, (rank + 1) * 60))
        if (state.get("complete") is not True or state.get("completed_count") != 60
                or state.get("assigned_count") != 60
                or identity.get("assigned_indices") != indices
                or identity.get("code_sha256") != code_sha(root)
                or identity.get("config_sha256") != config.sha
                or identity.get("e36_rows_sha256") != saved["e36_rows_sha256"]
                or identity.get("adapter_sha256") != saved["adapter_sha256"]
                or identity.get("identity_sha256") != _json_sha256({k: v for k, v in identity.items() if k != "identity_sha256"})):
            raise E38Error("Incomplete or changed E38 generation worker.")
        for index in indices:
            row = json.loads((folder / f"records/{index:04d}.json").read_text(encoding="utf-8"))
            if (row.get("sample_index") != index or row.get("question_id") != source[index]["question_id"]
                    or row.get("worker_identity_sha256") != identity["identity_sha256"]
                    or row.get("record_sha256") != _json_sha256({k: v for k, v in row.items() if k != "record_sha256"})):
                raise E38Error(f"Changed E38 generated record: {index}")
            generated[index] = row
    if len(generated) != 120:
        raise E38Error("E38 generation is incomplete.")
    if (json.loads((output / "generation/worker-0/state.json").read_text(encoding="utf-8"))["run_identity"]["generation_config_sha256"]
            != json.loads((output / "generation/worker-1/state.json").read_text(encoding="utf-8"))["run_identity"]["generation_config_sha256"]):
        raise E38Error("E38 GPU generation defaults differ.")
    scored: list[dict[str, Any]] = []
    deltas: list[float] = []
    for index, baseline in enumerate(source):
        raw = generated[index]
        first = trim_repeated_tail(raw["answer"])
        answer = trim_long_token_suffix(first["answer"], minimum_block_tokens=32,
                                        maximum_block_tokens=256)["answer"]
        reference = baseline["reference_answer"]
        control_meteor = nltk_meteor_score(reference, baseline["model_answer"])
        control_rouge = rouge_l_fmeasure(reference, baseline["model_answer"])
        if (abs(control_meteor - baseline["scores"]["meteor"]) > 1e-10
                or abs(control_rouge - baseline["scores"]["rouge_l"]) > 1e-10):
            raise E38Error(f"Official E32 control score changed: {baseline['question_id']}")
        candidate_meteor = nltk_meteor_score(reference, answer)
        candidate_rouge = rouge_l_fmeasure(reference, answer)
        deltas.append(candidate_meteor - control_meteor)
        scored.append({
            "question_id": baseline["question_id"], "sample_index": index,
            "question": baseline["question"], "reference_answer": reference,
            "control_answer": baseline["model_answer"], "candidate_answer": answer,
            "control_meteor": control_meteor, "candidate_meteor": candidate_meteor,
            "control_rouge_l": control_rouge, "candidate_rouge_l": candidate_rouge,
            "finish_reason": raw["finish_reason"],
            "input_tokens": raw["input_tokens"], "kept_e21_spans": raw["kept_e21_spans"],
            "original_e21_spans": raw["original_e21_spans"],
        })
    metrics = {
        "control_meteor": fmean(r["control_meteor"] for r in scored),
        "candidate_meteor": fmean(r["candidate_meteor"] for r in scored),
        "control_rouge_l": fmean(r["control_rouge_l"] for r in scored),
        "candidate_rouge_l": fmean(r["candidate_rouge_l"] for r in scored),
        "paired_meteor_delta": fmean(deltas),
        "paired_meteor_95_ci": _bootstrap_ci(deltas, seed="e38-viqwen-vs-e32-dev120", iterations=10000),
        "improved": sum(d > 0 for d in deltas), "worsened": sum(d < 0 for d in deltas),
        "tied": sum(d == 0 for d in deltas),
    }
    if (abs(metrics["control_meteor"] - config.raw["evaluation"]["control_meteor"]) > 1e-10
            or abs(metrics["control_rouge_l"] - config.raw["evaluation"]["control_rouge_l"]) > 1e-10):
        raise E38Error("E32 control average differs from pinned audit.")
    save_once(output / "evaluation/results.jsonl", scored, jsonl=True)
    review = sorted(scored, key=lambda r: r["candidate_meteor"] - r["control_meteor"])
    save_once(output / "review_low40.jsonl", review[:40], jsonl=True)
    report = {
        "schema_version": "1.0", "experiment_id": EXPERIMENT,
        "stage": "dev120-evaluation", "sample_size": 120,
        "selected_generator": "AITeamVN/Vi-Qwen2-3B-RAG-fresh-metadata-rank8",
        "metrics": metrics,
        "packing": {"mean_input_tokens": fmean(r["input_tokens"] for r in scored),
                    "mean_kept_e21_spans": fmean(r["kept_e21_spans"] for r in scored),
                    "questions_with_dropped_spans": sum(r["kept_e21_spans"] < r["original_e21_spans"] for r in scored),
                    "length_finish_rate": sum(r["finish_reason"] == "length" for r in scored) / 120},
        "evidence": {"code_sha256": code_sha(root), "config_sha256": config.sha,
                     "adapter_sha256": saved["adapter_sha256"],
                     "e36_rows_sha256": saved["e36_rows_sha256"],
                     "results_sha256": file_sha256(output / "evaluation/results.jsonl")},
        "references_used_only_after_generation": True,
        "public_read": False, "private_untouched": True,
        "automatic_promotion_allowed": False,
        "warning": "This dev-120 has been used repeatedly. Smoke evidence only; do not submit automatically.",
    }
    save_once(output / "report.json", report)
    return report
