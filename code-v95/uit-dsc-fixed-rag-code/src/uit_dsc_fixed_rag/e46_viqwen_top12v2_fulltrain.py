"""Fresh Vi-Qwen2-3B-RAG QLoRA on all 7,000 official top12-v2 records."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from typing import Any

from .corpus import file_sha256
from .e02_answer import make_messages
from .e03_rrf_grid import _json_sha256, _read_jsonl
from .e07_generator_ab import _as_text_chat_messages
from .e08b_context_lora import inference_packing
from .e19_metadata_lora import load_config as load_e19_config
from .e45_top12v2_fulltrain import EXPERIMENT as E45_EXPERIMENT
from .e45_top12v2_fulltrain import load_config as load_e45_config


EXPERIMENT = "E46-viqwen-top12v2-fulltrain7000-v1"


class E46Error(RuntimeError):
    pass


@dataclass(frozen=True)
class Config:
    raw: dict[str, Any]
    path: Path
    e45: Any
    e19: Any

    @property
    def sha(self) -> str:
        return file_sha256(self.path)


def load_config(root: Path, path: Path) -> Config:
    raw = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "schema_version", "experiment_id", "source_e45_config_path", "source_e19_config_path",
        "organizer_csv_sha256", "model", "training", "runtime", "run_contract",
    }
    if set(raw) != required or raw.get("schema_version") != "1.0" or raw.get("experiment_id") != EXPERIMENT:
        raise E46Error("E46 config root changed.")
    if raw["source_e45_config_path"] != "configs/e45-top12v2-fulltrain7000-v1.json":
        raise E46Error("E46 source E45 config changed.")
    if raw["source_e19_config_path"] != "configs/e19-metadata-aware-lora-train5636-eval200-v1.json":
        raise E46Error("E46 prompt lineage changed.")
    if raw["model"] != {
        "id": "AITeamVN/Vi-Qwen2-3B-RAG",
        "revision": "eaf427c24d86066a2b35828c499b7db3af321227",
        "architecture": "Qwen2ForCausalLM", "checkpoint_parameters": 3085938688,
        "embedding_parameters": 567754752, "adapter_parameter_cap": 50000000,
        "strict_stack_limit": 4000000000,
    }:
        raise E46Error("E46 Vi-Qwen model contract changed.")
    expected_training = {
        "records": 7000, "retrieved_contexts_per_record": 12,
        "selector": "rrf-top6-then-legal-priority-v2", "maximum_sequence_tokens": 4096,
        "epochs": 1.0, "rank": 8, "alpha": 16, "dropout": 0.05,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "seed": 20260830, "learning_rate": 0.0001, "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 4, "warmup_ratio": 0.03,
        "lr_scheduler": "cosine", "weight_decay": 0.0, "save_steps": 32,
        "quantization": "nf4-double-quant-float16", "fresh_from_pinned_checkpoint": True,
        "official_answers_supervise_generation_only": True, "answer_truncation_allowed": False,
    }
    if raw["training"] != expected_training:
        raise E46Error("E46 training contract changed.")
    if raw["runtime"] != {
        "torch": "2.10.0+cu128", "transformers": "5.16.1",
        "peft": "0.19.1", "accelerate": "1.13.0",
    }:
        raise E46Error("E46 runtime contract changed.")
    if not raw["run_contract"] or not all(value is True for value in raw["run_contract"].values()):
        raise E46Error("E46 run contract lost an invariant.")
    maximum = sum(raw["model"][key] for key in
                  ("checkpoint_parameters", "embedding_parameters", "adapter_parameter_cap"))
    if maximum >= raw["model"]["strict_stack_limit"]:
        raise E46Error("E46 model stack exceeds the BTC parameter limit.")
    e45_path, e19_path = root / raw["source_e45_config_path"], root / raw["source_e19_config_path"]
    return Config(raw, path, load_e45_config(e45_path), load_e19_config(root, e19_path))


def code_sha(root: Path) -> str:
    paths = [
        root / "src/uit_dsc_fixed_rag/e46_viqwen_top12v2_fulltrain.py",
        root / "scripts/run_e46_viqwen_top12v2_fulltrain_kaggle.py",
    ]
    return _json_sha256({p.relative_to(root).as_posix(): file_sha256(p) for p in paths})


def _ids_sha(ids: list[str]) -> str:
    return hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()


def training_source(e45: Path, config: Config) -> tuple[dict[str, Any], list[dict[str, Any]], Path]:
    report_path = e45 / "report.json"
    summary_path = e45 / "training-data/summary.json"
    records_path = e45 / "training-data/records.jsonl"
    if not all(path.is_file() for path in (report_path, summary_path, records_path)):
        raise E46Error("Add the complete E45 output including training-data.")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if (report.get("experiment_id") != E45_EXPERIMENT or report.get("sample_size") != 7000
            or report.get("answers_used_by_retrieval_or_selector") is not False
            or report.get("public_read") is not False or report.get("private_read") is not False
            or summary.get("experiment_id") != E45_EXPERIMENT or summary.get("record_count") != 7000
            or summary.get("context_candidates_per_record") != 12
            or summary.get("selector") != config.raw["training"]["selector"]
            or summary.get("answers_are_retrieval_labels") is not False
            or summary.get("official_answers_used_for_generation_supervision_only") is not True
            or summary.get("contains_public_or_private") is not False
            or summary.get("records_sha256") != file_sha256(records_path)
            or report.get("files", {}).get("training-data/records.jsonl", {}).get("sha256") != file_sha256(records_path)
            or report.get("files", {}).get("training-data/summary.json", {}).get("sha256") != file_sha256(summary_path)):
        raise E46Error("E45 training source identity changed.")
    rows = _read_jsonl(records_path)
    ids = [str(row.get("question_id")) for row in rows]
    if (len(rows) != 7000 or len(set(ids)) != 7000 or _ids_sha(ids) != summary.get("sample_ids_sha256")
            or any(row.get("sample_index") != index
                   or row.get("selector") != config.raw["training"]["selector"]
                   or row.get("supervision") != "official-answer-only"
                   or row.get("answers_are_retrieval_labels") is not False
                   or not isinstance(row.get("question"), str) or not row["question"].strip()
                   or not isinstance(row.get("answer"), str) or not row["answer"].strip()
                   or not isinstance(row.get("contexts"), list) or len(row["contexts"]) != 12
                   for index, row in enumerate(rows))):
        raise E46Error("E45 record order/schema changed.")
    return summary, rows, records_path


def organizer_csv(root: Path, config: Config) -> Path:
    hits = list((root / "data/raw").glob("*Sheet1.csv"))
    if len(hits) != 1 or file_sha256(hits[0]) != config.raw["organizer_csv_sha256"]:
        raise E46Error("Missing or changed BTC model inventory CSV.")
    if "https://huggingface.co/" + config.raw["model"]["id"] not in hits[0].read_text(encoding="utf-8-sig"):
        raise E46Error("Vi-Qwen checkpoint is absent from the BTC model inventory.")
    return hits[0]


def preflight(*, root: Path, e45: Path, output: Path, config: Config) -> dict[str, Any]:
    summary, _, records_path = training_source(e45, config)
    csv = organizer_csv(root, config)
    payload = {
        "experiment_id": EXPERIMENT, "code_sha256": code_sha(root), "config_sha256": config.sha,
        "organizer_csv_sha256": file_sha256(csv), "source_e45_report_sha256": file_sha256(e45 / "report.json"),
        "training_records_sha256": file_sha256(records_path),
        "training_sample_ids_sha256": summary["sample_ids_sha256"],
        "training_records": summary["record_count"], "selector": summary["selector"],
        "model_id": config.raw["model"]["id"], "model_revision": config.raw["model"]["revision"],
        "checkpoint_parameters": config.raw["model"]["checkpoint_parameters"],
        "maximum_stack_parameters": sum(config.raw["model"][key] for key in
                                        ("checkpoint_parameters", "embedding_parameters", "adapter_parameter_cap")),
        "e38_adapter_loaded": False, "e19_adapter_loaded": False,
        "dev_public_private_answers_loaded": False,
    }
    from .e02_answer import _atomic_json
    _atomic_json(output / "preflight.json", payload)
    return payload


def check_preflight(*, root: Path, e45: Path, output: Path, config: Config) -> dict[str, Any]:
    path = output / "preflight.json"
    if not path.is_file():
        raise E46Error("Run E46 preflight first.")
    saved = json.loads(path.read_text(encoding="utf-8"))
    summary, _, records_path = training_source(e45, config)
    expected = {
        "experiment_id": EXPERIMENT, "code_sha256": code_sha(root), "config_sha256": config.sha,
        "source_e45_report_sha256": file_sha256(e45 / "report.json"),
        "training_records_sha256": file_sha256(records_path),
        "training_sample_ids_sha256": summary["sample_ids_sha256"],
        "model_revision": config.raw["model"]["revision"],
    }
    if any(saved.get(key) != value for key, value in expected.items()):
        raise E46Error("E46 preflight/checkpoint identity changed.")
    return saved


def count_checkpoint_parameters(model_cache: Path, config: Config) -> int:
    from safetensors import safe_open

    model_config_path = model_cache / "config.json"
    if not model_config_path.is_file():
        raise E46Error("Pinned Vi-Qwen snapshot is incomplete.")
    model_config = json.loads(model_config_path.read_text(encoding="utf-8"))
    if config.raw["model"]["architecture"] not in model_config.get("architectures", []):
        raise E46Error("Vi-Qwen architecture changed.")
    single = model_cache / "model.safetensors"
    weights = [single] if single.is_file() else sorted(model_cache.glob("model-*-of-*.safetensors"))
    if not weights:
        raise E46Error("Pinned Vi-Qwen safetensors are missing.")
    names: set[str] = set(); total = 0
    for path in weights:
        with safe_open(str(path), framework="pt", device="cpu") as reader:
            for name in reader.keys():
                if name in names:
                    raise E46Error("Duplicate Vi-Qwen tensor across shards.")
                names.add(name); size = 1
                for dimension in reader.get_slice(name).get_shape():
                    size *= dimension
                total += size
    if total != config.raw["model"]["checkpoint_parameters"]:
        raise E46Error(f"Vi-Qwen checkpoint parameter count changed: {total}")
    return total


def render_viqwen(tokenizer: Any, messages: list[dict[str, Any]]) -> str:
    return tokenizer.apply_chat_template(_as_text_chat_messages(messages), tokenize=False,
                                         add_generation_prompt=True)


class TokenizedDataset:
    """Answer-only supervision with full answers and E45 top12-v2 candidates."""

    def __init__(self, rows: list[dict[str, Any]], tokenizer: Any, config: Config):
        self.items: list[dict[str, list[int]]] = []
        self.context_counts: list[int] = []
        self.sequence_lengths: list[int] = []
        self.truncated_chars: list[int] = []
        maximum = config.raw["training"]["maximum_sequence_tokens"]
        packing = inference_packing(config.e19)
        for row in rows:
            answer_ids = tokenizer(row["answer"], add_special_tokens=False)["input_ids"] + [tokenizer.eos_token_id]
            if len(answer_ids) >= maximum:
                raise E46Error(f"Official answer cannot fit without truncation: {row['question_id']}")
            chosen: list[dict[str, Any]] = []
            prompt_ids: list[int] | None = None
            for context in row["contexts"]:
                messages = make_messages(question=row["question"], contexts=[*chosen, context], config=packing)
                ids = tokenizer(render_viqwen(tokenizer, messages), add_special_tokens=False)["input_ids"]
                if len(ids) + len(answer_ids) <= maximum:
                    chosen.append(context); prompt_ids = ids
            trimmed = 0
            if not chosen:
                first = row["contexts"][0]
                original = str(first["text"]).strip()
                low, high = config.e19.section("train")["minimum_truncated_context_characters"], len(original)
                best: tuple[dict[str, Any], list[int], int] | None = None
                while low <= high:
                    midpoint = (low + high) // 2
                    trial = {**first, "text": original[:midpoint].rstrip()}
                    messages = make_messages(question=row["question"], contexts=[trial], config=packing)
                    ids = tokenizer(render_viqwen(tokenizer, messages), add_special_tokens=False)["input_ids"]
                    if len(ids) + len(answer_ids) <= maximum:
                        best = (trial, ids, midpoint); low = midpoint + 1
                    else:
                        high = midpoint - 1
                if best is not None:
                    chosen, prompt_ids, trimmed = [best[0]], best[1], len(original) - best[2]
            if not chosen or prompt_ids is None:
                raise E46Error(f"No full-answer E46 training example fits: {row['question_id']}")
            ids = prompt_ids + answer_ids
            self.items.append({"input_ids": ids, "attention_mask": [1] * len(ids),
                               "labels": [-100] * len(prompt_ids) + answer_ids})
            self.context_counts.append(len(chosen)); self.sequence_lengths.append(len(ids))
            self.truncated_chars.append(trimmed)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        return self.items[index]

    def summary(self) -> dict[str, Any]:
        return {
            "records": len(self.items), "mean_selected_contexts": fmean(self.context_counts),
            "minimum_selected_contexts": min(self.context_counts),
            "records_with_all_12_contexts": sum(value == 12 for value in self.context_counts),
            "maximum_sequence_tokens": max(self.sequence_lengths),
            "mean_sequence_tokens": fmean(self.sequence_lengths),
            "context_tail_truncation_count": sum(value > 0 for value in self.truncated_chars),
            "answer_truncation_count": 0,
        }


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


__all__ = [
    "CausalCollator", "Config", "E46Error", "EXPERIMENT", "TokenizedDataset",
    "check_preflight", "code_sha", "count_checkpoint_parameters", "load_config",
    "preflight", "render_viqwen", "training_source",
]
