"""Contracts for fresh rank-16 context-aware E14 QLoRA training."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from uit_dsc_fixed_rag.corpus import file_sha256
from uit_dsc_fixed_rag.e08b_context_lora import (
    inference_packing,
    prepare_context_train_records,
    validate_e08b_preflight,
)


CODE_VERSION = "0.42.0"


class E14Error(RuntimeError):
    """Raised when E14 training evidence or identity changes."""


@dataclass(frozen=True)
class E14Config:
    raw: dict[str, Any]
    path: Path

    @property
    def config_sha256(self) -> str:
        return file_sha256(self.path)

    def section(self, key: str) -> dict[str, Any]:
        value = self.raw.get(key)
        if not isinstance(value, dict):
            raise ValueError(f"E14 section must be an object: {key}")
        return value

    @property
    def generator_model_id(self) -> str:
        return str(self.section("generator")["model_id"])

    @property
    def generator_revision(self) -> str:
        return str(self.section("generator")["revision"])


def load_config(path: Path) -> E14Config:
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "schema_version", "experiment_id", "source_e08a", "train", "dev",
        "prompt", "generator", "lora", "optimization", "inference",
        "parameter_budget", "execution", "scoring", "run_contract",
    }
    if (
        set(payload) != required
        or payload.get("schema_version") != "1.0"
        or payload.get("experiment_id")
        != "E14-context-aware-lora-rank16-train5636-v1"
    ):
        raise ValueError("E14 training config root is incompatible.")
    config = E14Config(payload, path)
    source = config.section("source_e08a")
    if (
        source.get("experiment_id") != "E08A-context-retrieval-train5636-dev521-v1"
        or source.get("train_results_sha256")
        != "8bb590020510bc512dbcf738165480cd164059826849a001d117c2374e8e0786"
        or source.get("selected_contexts") != 12
        or source.get("reranker") is not None
    ):
        raise ValueError("E14 E08A source contract changed.")
    train = config.section("train")
    if (
        train.get("record_count") != 5636
        or train.get("sample_size") != 5636
        or train.get("supervision") != "official-answer-only"
        or train.get("context_candidate_limit") != 12
        or train.get("maximum_sequence_tokens") != 3072
        or train.get("answer_truncation_allowed") is not False
        or train.get("assistant_loss_only") is not True
    ):
        raise ValueError("E14 training-data contract changed.")
    lora = config.section("lora")
    if (
        lora.get("initialization") != "fresh-from-pinned-base-not-e08b-adapter"
        or lora.get("rank") != 16
        or lora.get("alpha") != 32
        or lora.get("dropout") != 0.05
        or lora.get("quantization") != "nf4-double-quant"
        or lora.get("compute_dtype") != "float16"
        or lora.get("trainable_parameter_cap") != 50000000
    ):
        raise ValueError("E14 rank-16 LoRA contract changed.")
    optimization = config.section("optimization")
    if (
        optimization.get("epochs") != 1.0
        or optimization.get("learning_rate") != 0.0001
        or optimization.get("per_device_batch_size") != 1
        or optimization.get("gradient_accumulation_steps") != 4
        or optimization.get("effective_global_batch_size") != 8
        or optimization.get("gradient_checkpointing") is not True
    ):
        raise ValueError("E14 optimization contract changed.")
    inference = config.section("inference")
    if (
        inference.get("max_input_tokens") != 8192
        or inference.get("max_new_tokens") != 704
        or inference.get("do_sample") is not False
        or inference.get("num_beams") != 1
        or inference.get("enable_thinking") is not False
        or inference.get("repetition_penalty") != 1.0
        or inference.get("no_repeat_ngram_size") != 0
    ):
        raise ValueError("E14 max704 inference contract changed.")
    budget = config.section("parameter_budget")
    if (
        budget.get("maximum_stack_total")
        != budget.get("embedding") + budget.get("generator")
        + budget.get("adapter_parameter_cap")
        or budget["maximum_stack_total"] >= budget["exclusive_limit"]
    ):
        raise ValueError("E14 parameter budget failed.")
    contract = config.section("run_contract")
    if not contract or not all(value is True for value in contract.values()):
        raise ValueError("E14 run contract lost an invariant.")
    if config.section("scoring").get("promotion_allowed") is not False:
        raise ValueError("E14 cannot auto-promote.")
    return config


__all__ = [
    "CODE_VERSION", "E14Config", "E14Error", "inference_packing",
    "load_config", "prepare_context_train_records", "validate_e08b_preflight",
]
