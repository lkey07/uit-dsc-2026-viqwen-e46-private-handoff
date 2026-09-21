"""E29 max1024 generation split 100/100 plus conservative suffix tail-trim."""
from __future__ import annotations

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
from . import e27_cross_reference as e27
from .corpus import file_sha256
from .e02_answer import _atomic_json, _atomic_jsonl
from .e03_rrf_grid import (
    _bootstrap_ci, _json_sha256, _load_worker_progress, _read_jsonl,
    _write_worker_state,
)
from .e13_e08b_max768_tailtrim import trim_repeated_tail
from .e18_source_metadata import save_once
from .e19_metadata_lora import _metrics, load_candidate_generator, validate_candidate_adapter
from .evaluation.official import ensure_nltk_resources, nltk_meteor_score, rouge_l_fmeasure


EXPERIMENT = "E29-max1024-tailtrim-dev200-v1"
CONTROL = "parent_expanded_max704"
RAW = "e21_parent_greedy_max1024"
TRIMMED = "e21_parent_greedy_max1024_tailtrim"
CODE_VERSION = "0.63.0"
LOG = logging.getLogger(__name__)


class E29Error(RuntimeError):
    """Raised when reviewed E29 evidence or checkpoint identity changes."""


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
    if set(raw) != {
        "schema_version", "experiment_id", "source_e21_config_path",
        "source_e21_config_sha256", "control", "raw_variant", "derived_variant",
        "inference", "evaluation", "parameter_budget", "run_contract",
    }:
        raise E29Error("Unexpected E29 config keys.")
    if (
        raw["schema_version"] != "1.0"
        or raw["experiment_id"] != EXPERIMENT
        or raw["source_e21_config_path"] != "configs/e21-parent-context-dev200-v1.json"
        or raw["source_e21_config_sha256"]
        != "fe8554d5f222e03a8f60ee181f8cfc4d1570719efaa6c56e2f7559fad1672ade"
    ):
        raise E29Error("E29 source identity changed.")
    e21_path = root / raw["source_e21_config_path"]
    if file_sha256(e21_path) != raw["source_e21_config_sha256"]:
        raise E29Error("Pinned E21 config bytes changed.")
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
        "max_new_tokens": 704,
    }:
        raise E29Error("Pinned E21 control changed.")
    if raw["raw_variant"] != {
        "key": RAW, "max_new_tokens": 1024, "workers": 2,
        "partition": "contiguous-100-100",
    }:
        raise E29Error("E29 raw variant changed.")
    if raw["derived_variant"] != {
        "key": TRIMMED,
        "source_variant": RAW,
        "method": "conservative-consecutive-tail-block-trim-v1",
        "line_block_sizes": [3, 2, 1],
        "sentence_block_sizes": [3, 2, 1],
        "normalization": "unicode-casefold-collapse-whitespace",
        "only_exact_consecutive_suffix": True,
        "never_rewrite_when_no_suffix_repeat": True,
    }:
        raise E29Error("E29 tail-trim policy changed.")
    if raw["inference"] != {
        "base_context": "exact-e21-parent-expanded",
        "max_input_tokens": 8192,
        "generator": "e19_metadata_trained_rank8",
        "do_sample": False,
        "num_beams": 1,
        "repetition_penalty": 1.0,
        "no_repeat_ngram_size": 0,
        "enable_thinking": False,
        "use_cache": True,
        "apply_policy_to_every_question": True,
    }:
        raise E29Error("E29 inference policy changed.")
    if raw["evaluation"] != {
        "sample": "reuse-e19-e21-dev400-600",
        "sample_size": 200,
        "sample_ids_sha256": "dc17d6c9af9e941868c03cfc61fd17cdd1dc55aa5834beb3e75ba0fde32a5c73",
        "questions_per_worker": 100,
        "promotion_allowed": False,
    }:
        raise E29Error("E29 evaluation contract changed.")
    budget = raw["parameter_budget"]
    if budget != {
        "exclusive_limit": 4_000_000_000,
        "generator": 2_274_069_824,
        "adapter_parameter_cap": 50_000_000,
        "maximum_stack_total": 2_324_069_824,
    } or budget["maximum_stack_total"] >= budget["exclusive_limit"]:
        raise E29Error("E29 parameter budget failed.")
    expected_run = {
        "top12_ranking_unchanged": True,
        "e21_parent_expansion_unchanged": True,
        "all_questions_use_same_generation_policy": True,
        "no_question_specific_routing": True,
        "same_policy_for_public_and_private": True,
        "tailtrim_derived_without_generation": True,
        "tailtrim_changes_only_exact_consecutive_suffix_repeats": True,
        "dev_answers_scoring_only": True,
        "same_e19_adapter_and_prompt": True,
        "no_external_or_synthetic_data": True,
        "no_api_model": True,
        "fixed_non_agentic_rag": True,
        "holdout_untouched": True,
        "public_not_read": True,
        "checkpoint_after_each_question_id": True,
        "resume_fail_closed": True,
    }
    if raw["run_contract"] != expected_run:
        raise E29Error("E29 run contract changed.")
    return Config(raw=raw, path=path, e21=e21)


def code_sha(root: Path) -> str:
    paths = sorted((root / "src/uit_dsc_fixed_rag").rglob("*.py"))
    paths.append(root / "scripts/run_e29_max1024_tailtrim_kaggle.py")
    return _json_sha256({
        path.relative_to(root).as_posix(): file_sha256(path)
        for path in paths if path.is_file()
    })


def sample(train: Path, dev: Path, config: Config):
    return parent.sample(train, dev, config.e21)


def _ids_sha(ids: list[str]) -> str:
    return hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()


def assigned_indices(rank: int) -> list[int]:
    if rank not in (0, 1):
        raise E29Error("E29 worker rank must be 0 or 1.")
    return list(range(rank * 100, (rank + 1) * 100))


def load_e21(directory: Path, ids: list[str], config: Config):
    return e27.load_e21(directory, ids, config)


def validate_preflight(
    *, root: Path, e21: Path, training: Path, train: Path, dev: Path,
    output: Path, config: Config,
):
    _, _, ids = sample(train, dev, config)
    controls, control_identity, _ = load_e21(e21, ids, config)
    if len(controls) != 200 or _ids_sha(ids) != config.raw["evaluation"]["sample_ids_sha256"]:
        raise E29Error("E29 sample identity changed.")
    adapter_sha, complete = validate_candidate_adapter(training, config.e19)
    if adapter_sha != config.raw["control"]["adapter_sha256"]:
        raise E29Error("E29 requires the exact E19 adapter used by E21.")
    payload = {
        "schema_version": "1.0",
        "experiment_id": EXPERIMENT,
        "code_sha256": code_sha(root),
        "config_sha256": config.sha,
        "sample_ids_sha256": _ids_sha(ids),
        "sample_size": 200,
        "control_identity_sha256": control_identity["identity_sha256"],
        "control_results_sha256": config.raw["control"]["results_sha256"],
        "e21_prepared_sha256": control_identity["prepared_sha256"],
        "adapter_sha256": adapter_sha,
        "adapter_identity_sha256": complete["identity_sha256"],
        "worker_partitions": [assigned_indices(0), assigned_indices(1)],
        "raw_variant": RAW,
        "derived_variant": TRIMMED,
        "answers_used_during_generation_or_tailtrim": False,
    }
    save_once(output / "preflight.json", payload)
    return payload


def check_preflight(root: Path, output: Path, config: Config):
    path = output / "preflight.json"
    if not path.is_file():
        raise E29Error("Run E29 preflight first.")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("experiment_id") != EXPERIMENT
        or payload.get("config_sha256") != config.sha
        or payload.get("code_sha256") != code_sha(root)
        or payload.get("worker_partitions") != [assigned_indices(0), assigned_indices(1)]
    ):
        raise E29Error("E29 preflight code/config changed.")
    return payload


def validate_record(row, question_id, index, rank, identity):
    if (
        row.get("question_id") != question_id
        or row.get("sample_index") != index
        or row.get("worker_rank") != rank
        or row.get("variant") != RAW
        or row.get("worker_identity_sha256") != identity["identity_sha256"]
        or not isinstance(row.get("answer"), str)
        or not row["answer"].strip()
        or row.get("record_sha256")
        != _json_sha256({key: value for key, value in row.items() if key != "record_sha256"})
    ):
        raise E29Error(f"Changed E29 generation record: {rank}/{index}")


def run_worker(
    *, root: Path, e21: Path, training: Path, train: Path, dev: Path,
    output: Path, config: Config, rank: int, device: str,
):
    import torch

    if rank not in (0, 1) or device != f"cuda:{rank}":
        raise E29Error("GPU0=indices 0-99; GPU1=indices 100-199.")
    checked = check_preflight(root, output, config)
    questions, _, ids = sample(train, dev, config)
    controls, control_identity, prepared = load_e21(e21, ids, config)
    runtime = {name: importlib.metadata.version(name) for name in control_identity["runtime"]}
    if runtime != control_identity["runtime"]:
        raise E29Error(
            f"Use the exact E21 runtime. expected={control_identity['runtime']} current={runtime}"
        )
    model, tokenizer, placement, parameters = load_candidate_generator(
        config=config.e19, training_directory=training, device=device,
    )
    adapter_sha, _ = validate_candidate_adapter(training, config.e19)
    generation_sha = _json_sha256(model.generation_config.to_dict())
    if adapter_sha != checked["adapter_sha256"] or generation_sha != control_identity["generation_config_sha256"]:
        raise E29Error("E19 adapter or generation defaults differ from E21.")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    def render(messages):
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )

    def count(text):
        return len(tokenizer(text, add_special_tokens=False)["input_ids"])

    assigned = assigned_indices(rank)
    identity = {
        "code_sha256": code_sha(root),
        "config_sha256": config.sha,
        "e21_prepared_sha256": checked["e21_prepared_sha256"],
        "control_identity_sha256": control_identity["identity_sha256"],
        "adapter_sha256": adapter_sha,
        "generation_config_sha256": generation_sha,
        "runtime": runtime,
        "variant": RAW,
        "max_new_tokens": 1024,
        "worker_rank": rank,
        "device": device,
        "assigned_indices": assigned,
        "device_map": placement,
        "adapter_parameters": parameters,
        "sample_ids_sha256": _ids_sha(ids),
    }
    identity["identity_sha256"] = _json_sha256(identity)
    folder = output / "evaluation" / RAW
    records = folder / "records"
    state = folder / f"worker-{rank}-state.json"
    records.mkdir(parents=True, exist_ok=True)
    done = _load_worker_progress(
        records=records, state_path=state, identity=identity,
        assigned_indices=assigned, sample_ids=ids,
    )
    for index in assigned[:done]:
        validate_record(
            json.loads((records / f"{index:04d}.json").read_text(encoding="utf-8")),
            ids[index], index, rank, identity,
        )
    for progress, index in enumerate(assigned[done:], start=done + 1):
        question_id = ids[index]
        messages, spans, diagnostics = parent.pack(
            questions[question_id]["question"], prepared[index], config.e21,
            lambda value: count(render(value)), count, parent.VARIANTS[1],
        )
        prompt = render(messages)
        prompt_sha = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        if prompt_sha != controls[index]["prompt_sha256"]:
            raise E29Error(f"E29 failed to reproduce E21 prompt: {question_id}")
        inputs = {
            key: value.to(device)
            for key, value in tokenizer(prompt, add_special_tokens=False, return_tensors="pt").items()
        }
        started = time.perf_counter()
        with torch.inference_mode():
            generated = model.generate(
                **inputs, do_sample=False, num_beams=1, repetition_penalty=1.0,
                no_repeat_ngram_size=0, max_new_tokens=1024, use_cache=True,
            )
        latency = (time.perf_counter() - started) * 1000
        new_ids = generated[0, inputs["input_ids"].shape[1]:]
        answer = tokenizer.decode(
            new_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False,
        ).strip()
        if not answer:
            raise E29Error(f"Empty E29 answer: {question_id}")
        eos = model.generation_config.eos_token_id
        eos_ids = set(eos if isinstance(eos, list) else [eos])
        finish = (
            "eos" if len(new_ids) and int(new_ids[-1]) in eos_ids
            else "length" if len(new_ids) >= 1024 else "other"
        )
        row = {
            "question_id": question_id,
            "sample_index": index,
            "worker_rank": rank,
            "variant": RAW,
            "worker_identity_sha256": identity["identity_sha256"],
            "answer": answer,
            "input_tokens": count(prompt),
            "output_tokens": count(answer),
            "generated_tokens_including_special": len(new_ids),
            "finish_reason": finish,
            "generation_latency_ms": latency,
            "selected_context_count": len(spans),
            "selected_chunk_ids": diagnostics["selected_chunk_ids"],
            "prompt_sha256": prompt_sha,
        }
        row["record_sha256"] = _json_sha256(row)
        _atomic_json(records / f"{index:04d}.json", row)
        _write_worker_state(state, identity, progress, 100)
        LOG.info(
            "e29_generation worker=%d device=%s completed=%d total=100 global_at_least=%d/200 question_id=%s finish=%s",
            rank, device, progress, progress * 2, question_id, finish,
        )
    return {"worker_rank": rank, "completed": 100}


def _paired(scores, left, right):
    meteor = [a["meteor"] - b["meteor"] for a, b in zip(scores[left], scores[right])]
    rouge = [a["rouge_l"] - b["rouge_l"] for a, b in zip(scores[left], scores[right])]
    return {
        "meteor_mean": fmean(meteor),
        "meteor_bootstrap_95_ci": _bootstrap_ci(
            meteor, seed=f"e29-{left}-{right}-meteor", iterations=10000,
        ),
        "rouge_l_mean": fmean(rouge),
        "rouge_l_bootstrap_95_ci": _bootstrap_ci(
            rouge, seed=f"e29-{left}-{right}-rouge", iterations=10000,
        ),
        "improved": sum(value > 0 for value in meteor),
        "worsened": sum(value < 0 for value in meteor),
        "tied": sum(value == 0 for value in meteor),
    }


def finalize(
    *, root: Path, e21: Path, training: Path, train: Path, dev: Path,
    output: Path, config: Config,
):
    checked = check_preflight(root, output, config)
    ensure_nltk_resources(download=False)
    questions, _, ids = sample(train, dev, config)
    controls, control_identity, _ = load_e21(e21, ids, config)
    raw_folder = output / "evaluation" / RAW
    identities = []
    for rank in (0, 1):
        state = json.loads((raw_folder / f"worker-{rank}-state.json").read_text(encoding="utf-8"))
        identity = state.get("run_identity", {})
        if (
            state.get("complete") is not True
            or state.get("completed_count") != 100
            or state.get("assigned_count") != 100
            or identity.get("worker_rank") != rank
            or identity.get("device") != f"cuda:{rank}"
            or identity.get("variant") != RAW
            or identity.get("max_new_tokens") != 1024
            or identity.get("assigned_indices") != assigned_indices(rank)
            or identity.get("code_sha256") != code_sha(root)
            or identity.get("config_sha256") != config.sha
            or identity.get("e21_prepared_sha256") != checked["e21_prepared_sha256"]
            or identity.get("control_identity_sha256") != control_identity["identity_sha256"]
            or identity.get("adapter_sha256") != checked["adapter_sha256"]
            or identity.get("runtime") != control_identity["runtime"]
            or identity.get("generation_config_sha256") != control_identity["generation_config_sha256"]
            or identity.get("identity_sha256")
            != _json_sha256({key: value for key, value in identity.items() if key != "identity_sha256"})
        ):
            raise E29Error("Incomplete or changed E29 worker state.")
        identities.append(identity)
    if identities[0]["runtime"] != identities[1]["runtime"]:
        raise E29Error("E29 worker runtimes differ.")
    raw_rows = []
    for index, question_id in enumerate(ids):
        rank = 0 if index < 100 else 1
        row = json.loads((raw_folder / f"records/{index:04d}.json").read_text(encoding="utf-8"))
        validate_record(row, question_id, index, rank, identities[rank])
        raw_rows.append(row)
    _atomic_jsonl(raw_folder / "results.jsonl", raw_rows)

    try:
        from transformers import AutoTokenizer
    except ImportError as exc:  # pragma: no cover
        raise E29Error("Transformers is required for E29 finalization.") from exc
    tokenizer = AutoTokenizer.from_pretrained(
        config.e19.generator_model_id, revision=config.e19.generator_revision,
        trust_remote_code=False,
    )
    trimmed_rows = []
    for row in raw_rows:
        trimmed = trim_repeated_tail(row["answer"])
        derived = {
            **row,
            "variant": TRIMMED,
            "answer": trimmed["answer"],
            "output_tokens": len(tokenizer(
                trimmed["answer"], add_special_tokens=False,
            )["input_ids"]),
            "source_finish_reason": row["finish_reason"],
            "postprocess_changed": trimmed["changed"],
            "removed_characters": trimmed["removed_characters"],
            "removed_line_blocks": trimmed["removed_line_blocks"],
            "removed_sentence_blocks": trimmed["removed_sentence_blocks"],
        }
        derived.pop("record_sha256", None)
        derived["record_sha256"] = _json_sha256(derived)
        trimmed_rows.append(derived)
    trimmed_path = output / f"evaluation/{TRIMMED}/results.jsonl"
    _atomic_jsonl(trimmed_path, trimmed_rows)

    by_variant = {CONTROL: controls, RAW: raw_rows, TRIMMED: trimmed_rows}
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
            "tailtrim_changed": trimmed_rows[index]["postprocess_changed"],
        }
        for index, question_id in enumerate(ids)
    ])
    metrics = {variant: _metrics(rows, scores[variant]) for variant, rows in by_variant.items()}
    metrics[TRIMMED].update({
        "postprocess_changed_rate": fmean(row["postprocess_changed"] for row in trimmed_rows),
        "changed_questions": sum(row["postprocess_changed"] for row in trimmed_rows),
        "mean_removed_characters": fmean(row["removed_characters"] for row in trimmed_rows),
    })
    report = {
        "schema_version": "1.0",
        "experiment_id": EXPERIMENT,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "sample_size": 200,
        "sample_scope": "same-repeated-e19-e21-dev400-600",
        "control_variant": CONTROL,
        "raw_variant": RAW,
        "derived_variant": TRIMMED,
        "metrics": metrics,
        "paired_raw_minus_control": _paired(scores, RAW, CONTROL),
        "paired_tailtrim_minus_control": _paired(scores, TRIMMED, CONTROL),
        "paired_tailtrim_minus_raw": _paired(scores, TRIMMED, RAW),
        "tailtrim_diagnostics": {
            "changed_questions": sum(row["postprocess_changed"] for row in trimmed_rows),
            "unchanged_questions": sum(not row["postprocess_changed"] for row in trimmed_rows),
            "removed_characters_total": sum(row["removed_characters"] for row in trimmed_rows),
            "removed_line_blocks": sum(row["removed_line_blocks"] for row in trimmed_rows),
            "removed_sentence_blocks": sum(row["removed_sentence_blocks"] for row in trimmed_rows),
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
            "raw_results_sha256": file_sha256(raw_folder / "results.jsonl"),
            "tailtrim_results_sha256": file_sha256(trimmed_path),
        },
        "warning": "Repeated dev-200 smoke evidence; do not treat it as unseen validation.",
    }
    _atomic_json(output / "report.json", report)
    return report

