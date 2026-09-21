"""E11 deterministic output-length refinement on saved E09 max640."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from uit_dsc_fixed_rag.corpus import file_sha256
from uit_dsc_fixed_rag.e10_repetition_grid import (
    _sample_ids,
    answer_diagnostics,
    finalize_score,
    generation_kwargs,
    load_generator,
    run_variant_worker,
    validate_preflight,
)


CODE_VERSION = "0.27.0"


class E11Error(RuntimeError):
    """Raised when E11 loses its frozen experiment identity."""


@dataclass(frozen=True)
class E11Config:
    raw: dict[str, Any]
    path: Path

    @property
    def config_sha256(self) -> str:
        return file_sha256(self.path)

    @property
    def code_version(self) -> str:
        return CODE_VERSION

    def section(self, key: str) -> dict[str, Any]:
        value = self.raw.get(key)
        if not isinstance(value, dict):
            raise ValueError(f"E11 section must be an object: {key}")
        return value

    @property
    def variants(self) -> list[dict[str, Any]]:
        value = self.section("inference").get("variants")
        if not isinstance(value, list):
            raise ValueError("E11 variants must be a list.")
        return value


def load_e11_config(path: Path) -> E11Config:
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "schema_version", "experiment_id", "source_contexts", "source_control",
        "source_max640", "dev", "generator", "lora", "inference",
        "parameter_budget", "execution", "scoring", "run_contract",
    }
    if (
        set(payload) != required
        or payload.get("schema_version") != "1.0"
        or payload.get("experiment_id")
        != "E11-output-length-refinement-qwen35-lora-dev200-v1"
    ):
        raise ValueError("E11 config root is incompatible.")
    config = E11Config(payload, path)
    source = config.section("source_max640")
    if (
        source.get("experiment_id")
        != "E09-output-length-grid-qwen35-lora-dev200-v2"
        or source.get("path") != "results.jsonl"
        or source.get("sha256")
        != "e7e7f319bee4ac10e2aeffea70fffa2e5c4b728e5cc275d84e76b166438e1fa8"
        or source.get("variant") != "max640"
        or source.get("max_new_tokens") != 640
        or source.get("sample_size") != 200
    ):
        raise ValueError("E11 max640 control identity changed.")
    expected_variants = [
        {
            "key": "max704", "max_new_tokens": 704,
            "repetition_penalty": 1.0, "no_repeat_ngram_size": 0,
            "worker_rank": 0, "device": "cuda:0",
        },
        {
            "key": "max768", "max_new_tokens": 768,
            "repetition_penalty": 1.0, "no_repeat_ngram_size": 0,
            "worker_rank": 1, "device": "cuda:1",
        },
    ]
    inference = config.section("inference")
    if (
        inference.get("max_input_tokens") != 8192
        or inference.get("minimum_contexts") != 1
        or inference.get("control_max_new_tokens") != 640
        or inference.get("variants") != expected_variants
        or inference.get("do_sample") is not False
        or inference.get("num_beams") != 1
        or inference.get("enable_thinking") is not False
        or inference.get("use_cache") is not True
    ):
        raise ValueError("E11 output-length grid changed.")
    generator = config.section("generator")
    if (
        generator.get("model_id") != "Qwen/Qwen3.5-2B"
        or generator.get("revision")
        != "15852e8c16360a2fea060d615a32b45270f8a8fc"
        or generator.get("published_parameter_count") != 2_274_069_824
        or generator.get("accepted_runtime_unique_parameter_counts")
        != [1_881_825_088, 2_213_241_664]
    ):
        raise ValueError("E11 generator identity changed.")
    execution = config.section("execution")
    if execution.get("workers") != 2 or execution.get("checkpoint_after_questions") != 1:
        raise ValueError("E11 dual-GPU execution changed.")
    budget = config.section("parameter_budget")
    if (
        budget.get("maximum_stack_total")
        != budget.get("embedding") + budget.get("generator")
        + budget.get("adapter_parameter_cap")
        or budget["maximum_stack_total"] >= budget["exclusive_limit"]
    ):
        raise ValueError("E11 parameter budget failed.")
    contract = config.section("run_contract")
    if not contract or not all(value is True for value in contract.values()):
        raise ValueError("E11 run contract lost an invariant.")
    if config.section("scoring").get("promotion_allowed") is not False:
        raise ValueError("E11 dev-200 grid cannot auto-promote.")
    return config


__all__ = [
    "E11Config", "E11Error", "answer_diagnostics", "finalize_score",
    "generation_kwargs", "load_e11_config", "load_generator",
    "run_variant_worker", "validate_preflight",
]
