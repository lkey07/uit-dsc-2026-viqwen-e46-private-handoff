"""E30 identical suffix tail-trim on new max1280 and byte-reused E28 max1024."""
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
from .e03_rrf_grid import _bootstrap_ci, _json_sha256, _load_worker_progress, _read_jsonl, _write_worker_state
from .e13_e08b_max768_tailtrim import trim_repeated_tail
from .e18_source_metadata import save_once
from .e19_metadata_lora import _metrics, load_candidate_generator, validate_candidate_adapter
from .evaluation.official import ensure_nltk_resources, nltk_meteor_score, rouge_l_fmeasure


EXPERIMENT = "E30-max1280-vs-max1024-tailtrim-dev200-v1"
CONTROL = "parent_expanded_max704"
RAW1024 = "e21_parent_greedy_max1024"
TRIM1024 = "e21_parent_greedy_max1024_tailtrim"
RAW1280 = "e21_parent_greedy_max1280"
TRIM1280 = "e21_parent_greedy_max1280_tailtrim"
CODE_VERSION = "0.65.0"
LOG = logging.getLogger(__name__)


class E30Error(RuntimeError):
    """Raised when frozen E30 evidence or resumable state changes."""


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
        "source_e21_config_sha256", "control", "reused_max1024",
        "tailtrim_variant", "max1280_variant", "inference", "evaluation",
        "parameter_budget", "run_contract",
    }
    if set(raw) != expected_keys:
        raise E30Error("Unexpected E30 config keys.")
    if (
        raw["schema_version"] != "1.0"
        or raw["experiment_id"] != EXPERIMENT
        or raw["source_e21_config_path"] != "configs/e21-parent-context-dev200-v1.json"
        or raw["source_e21_config_sha256"]
        != "fe8554d5f222e03a8f60ee181f8cfc4d1570719efaa6c56e2f7559fad1672ade"
    ):
        raise E30Error("E30 source identity changed.")
    e21_path = root / raw["source_e21_config_path"]
    if file_sha256(e21_path) != raw["source_e21_config_sha256"]:
        raise E30Error("Pinned E21 config bytes changed.")
    e21 = parent.load_config(root, e21_path)
    if raw["control"] != {
        "experiment_id": "E21-parent-context-dev200-v1",
        "variant": CONTROL,
        "results_sha256": "cbe3d5e163e1983f031a06c767596fcebd91177e3b90625998a1deed28f6afe4",
        "code_sha256": "e5574861eb34ae3fdee1220d5260109ddd9f84feacea9c1ec46a78df8b5527c8",
        "adapter_sha256": "bfb9d8120337c4013b5dcb0b59c6d03beccfe9f9748e5a01ef62e1de64fb5edd",
        "sample_size": 200,
        "max_new_tokens": 704,
    }:
        raise E30Error("Pinned E21 control changed.")
    if raw["reused_max1024"] != {
        "experiment_id": "E28-long-output-antiloop-dev200-v1",
        "variant": RAW1024,
        "results_sha256": "061ef2bec0d15928a607fcca8b6b1116d3ef2543eb75ab79bbee35cc7255dc3a",
        "config_sha256": "2424d8bdc5432b7acbee75be6b9314d18b06e178c279cc7d8eee117be08daa8c",
        "code_sha256": "76a9579180f2a571b0bd7fa6c2575a29eeef14fad1f0e77d58fa5c381d0e0035",
        "max_new_tokens": 1024,
        "meteor": 0.565830270172808,
    }:
        raise E30Error("Pinned E28 max1024 evidence changed.")
    if raw["tailtrim_variant"] != {
        "key": TRIM1024,
        "source_variant": RAW1024,
        "max1280_key": TRIM1280,
        "max1280_source_variant": RAW1280,
        "method": "conservative-consecutive-tail-block-trim-v1",
        "line_block_sizes": [3, 2, 1],
        "sentence_block_sizes": [3, 2, 1],
        "normalization": "unicode-casefold-collapse-whitespace",
        "only_exact_consecutive_suffix": True,
        "never_rewrite_when_no_suffix_repeat": True,
    }:
        raise E30Error("E30 tail-trim policy changed.")
    if raw["max1280_variant"] != {
        "key": RAW1280, "max_new_tokens": 1280, "workers": 2,
        "partition": "contiguous-100-100", "no_repeat_ngram_size": 0,
    }:
        raise E30Error("E30 max1280 variant changed.")
    if raw["inference"] != {
        "base_context": "exact-e21-parent-expanded",
        "max_input_tokens": 8192,
        "generator": "e19_metadata_trained_rank8",
        "do_sample": False,
        "num_beams": 1,
        "repetition_penalty": 1.0,
        "enable_thinking": False,
        "use_cache": True,
    }:
        raise E30Error("E30 inference policy changed.")
    if raw["evaluation"] != {
        "sample": "reuse-e19-e21-dev400-600",
        "sample_size": 200,
        "sample_ids_sha256": "dc17d6c9af9e941868c03cfc61fd17cdd1dc55aa5834beb3e75ba0fde32a5c73",
        "questions_per_worker": 100,
        "promotion_allowed": False,
    }:
        raise E30Error("E30 evaluation contract changed.")
    budget = raw["parameter_budget"]
    if budget != {
        "exclusive_limit": 4_000_000_000,
        "generator": 2_274_069_824,
        "adapter_parameter_cap": 50_000_000,
        "maximum_stack_total": 2_324_069_824,
    } or budget["maximum_stack_total"] >= budget["exclusive_limit"]:
        raise E30Error("E30 parameter budget failed.")
    expected_run = {
        "top12_ranking_unchanged": True,
        "e21_parent_expansion_unchanged": True,
        "max1024_reused_byte_exactly": True,
        "max1280_applied_to_every_question": True,
        "no_question_specific_routing": True,
        "same_policy_for_public_and_private": True,
        "tailtrim_derived_without_generation": True,
        "same_tailtrim_applied_to_max1024_and_max1280": True,
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
        raise E30Error("E30 run contract changed.")
    return Config(raw=raw, path=path, e21=e21)


def code_sha(root: Path) -> str:
    paths = sorted((root / "src/uit_dsc_fixed_rag").rglob("*.py"))
    paths.append(root / "scripts/run_e30_max1280_vs_tailtrim_kaggle.py")
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
        raise E30Error("E30 worker rank must be 0 or 1.")
    return list(range(rank * 100, (rank + 1) * 100))


def load_e21(directory: Path, ids: list[str], config: Config):
    return e27.load_e21(directory, ids, config)


def load_e28_raw(directory: Path, ids: list[str], config: Config):
    report_path = directory / "report.json"
    result_path = directory / f"evaluation/{RAW1024}/results.jsonl"
    state_path = result_path.parent / "state.json"
    if not all(path.is_file() for path in (report_path, result_path, state_path)):
        raise E30Error("Add the complete E28 notebook output/dataset.")
    pinned = config.raw["reused_max1024"]
    if file_sha256(result_path) != pinned["results_sha256"]:
        raise E30Error("E28 raw max1024 results are not byte-exact.")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if (
        report.get("experiment_id") != pinned["experiment_id"]
        or report.get("sample_size") != 200
        or report.get("evidence", {}).get("config_sha256") != pinned["config_sha256"]
        or report.get("evidence", {}).get("code_sha256") != pinned["code_sha256"]
        or report.get("evidence", {}).get("candidate_results_sha256", {}).get(RAW1024)
        != pinned["results_sha256"]
        or report.get("metrics", {}).get(RAW1024, {}).get("meteor") != pinned["meteor"]
    ):
        raise E30Error("E28 report identity changed.")
    rows = _read_jsonl(result_path)
    state = json.loads(state_path.read_text(encoding="utf-8"))
    identity = state.get("run_identity", {})
    if (
        len(rows) != 200
        or state.get("complete") is not True
        or state.get("completed_count") != 200
        or identity.get("variant") != RAW1024
        or identity.get("max_new_tokens") != 1024
        or identity.get("worker_rank") != 0
        or identity.get("code_sha256") != pinned["code_sha256"]
        or identity.get("config_sha256") != pinned["config_sha256"]
        or identity.get("identity_sha256")
        != _json_sha256({key: value for key, value in identity.items() if key != "identity_sha256"})
    ):
        raise E30Error("E28 max1024 worker state changed.")
    for index, question_id in enumerate(ids):
        row = rows[index]
        if (
            row.get("question_id") != question_id
            or row.get("sample_index") != index
            or row.get("variant") != RAW1024
            or row.get("worker_identity_sha256") != identity["identity_sha256"]
            or not isinstance(row.get("answer"), str)
            or not row["answer"].strip()
            or row.get("record_sha256")
            != _json_sha256({key: value for key, value in row.items() if key != "record_sha256"})
        ):
            raise E30Error(f"Changed E28 max1024 record: {index}")
    return rows, identity


def validate_preflight(
    *, root: Path, e21: Path, e28: Path, training: Path, train: Path,
    dev: Path, output: Path, config: Config,
):
    _, _, ids = sample(train, dev, config)
    controls, control_identity, _ = load_e21(e21, ids, config)
    raw1024, e28_identity = load_e28_raw(e28, ids, config)
    if (
        len(controls) != 200
        or len(raw1024) != 200
        or _ids_sha(ids) != config.raw["evaluation"]["sample_ids_sha256"]
    ):
        raise E30Error("E30 sample identity changed.")
    adapter_sha, complete = validate_candidate_adapter(training, config.e19)
    if adapter_sha != config.raw["control"]["adapter_sha256"]:
        raise E30Error("E30 requires the exact E19 adapter used by E21/E28.")
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
        "e28_raw1024_results_sha256": config.raw["reused_max1024"]["results_sha256"],
        "e28_raw1024_identity_sha256": e28_identity["identity_sha256"],
        "adapter_sha256": adapter_sha,
        "adapter_identity_sha256": complete["identity_sha256"],
        "worker_partitions": [assigned_indices(0), assigned_indices(1)],
        "generated_variant": RAW1280,
        "derived_variants": [TRIM1024, TRIM1280],
        "answers_used_during_generation_or_tailtrim": False,
    }
    save_once(output / "preflight.json", payload)
    return payload


def check_preflight(root: Path, output: Path, config: Config):
    path = output / "preflight.json"
    if not path.is_file():
        raise E30Error("Run E30 preflight first.")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("experiment_id") != EXPERIMENT
        or payload.get("config_sha256") != config.sha
        or payload.get("code_sha256") != code_sha(root)
        or payload.get("worker_partitions") != [assigned_indices(0), assigned_indices(1)]
    ):
        raise E30Error("E30 preflight code/config changed.")
    return payload


def validate_record(row, question_id, index, rank, identity):
    if (
        row.get("question_id") != question_id
        or row.get("sample_index") != index
        or row.get("worker_rank") != rank
        or row.get("variant") != RAW1280
        or row.get("worker_identity_sha256") != identity["identity_sha256"]
        or not isinstance(row.get("answer"), str)
        or not row["answer"].strip()
        or row.get("record_sha256")
        != _json_sha256({key: value for key, value in row.items() if key != "record_sha256"})
    ):
        raise E30Error(f"Changed E30 generation record: {rank}/{index}")


def run_worker(
    *, root: Path, e21: Path, e28: Path, training: Path, train: Path,
    dev: Path, output: Path, config: Config, rank: int, device: str,
):
    import torch

    if rank not in (0, 1) or device != f"cuda:{rank}":
        raise E30Error("GPU0=indices 0-99; GPU1=indices 100-199.")
    checked = check_preflight(root, output, config)
    questions, _, ids = sample(train, dev, config)
    controls, control_identity, prepared = load_e21(e21, ids, config)
    load_e28_raw(e28, ids, config)
    runtime = {name: importlib.metadata.version(name) for name in control_identity["runtime"]}
    if runtime != control_identity["runtime"]:
        raise E30Error(f"Use exact E21 runtime. expected={control_identity['runtime']} current={runtime}")
    model, tokenizer, placement, parameters = load_candidate_generator(
        config=config.e19, training_directory=training, device=device,
    )
    adapter_sha, _ = validate_candidate_adapter(training, config.e19)
    generation_sha = _json_sha256(model.generation_config.to_dict())
    if adapter_sha != checked["adapter_sha256"] or generation_sha != control_identity["generation_config_sha256"]:
        raise E30Error("E19 adapter or generation defaults differ from E21.")
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
        "e28_raw1024_results_sha256": checked["e28_raw1024_results_sha256"],
        "adapter_sha256": adapter_sha,
        "generation_config_sha256": generation_sha,
        "runtime": runtime,
        "variant": RAW1280,
        "max_new_tokens": 1280,
        "no_repeat_ngram_size": 0,
        "worker_rank": rank,
        "device": device,
        "assigned_indices": assigned,
        "device_map": placement,
        "adapter_parameters": parameters,
        "sample_ids_sha256": _ids_sha(ids),
    }
    identity["identity_sha256"] = _json_sha256(identity)
    folder = output / "evaluation" / RAW1280
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
            raise E30Error(f"E30 failed to reproduce E21 prompt: {question_id}")
        inputs = {
            key: value.to(device)
            for key, value in tokenizer(prompt, add_special_tokens=False, return_tensors="pt").items()
        }
        started = time.perf_counter()
        with torch.inference_mode():
            generated = model.generate(
                **inputs, do_sample=False, num_beams=1, repetition_penalty=1.0,
                no_repeat_ngram_size=0, max_new_tokens=1280, use_cache=True,
            )
        latency = (time.perf_counter() - started) * 1000
        new_ids = generated[0, inputs["input_ids"].shape[1]:]
        answer = tokenizer.decode(
            new_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False,
        ).strip()
        if not answer:
            raise E30Error(f"Empty E30 answer: {question_id}")
        eos = model.generation_config.eos_token_id
        eos_ids = set(eos if isinstance(eos, list) else [eos])
        finish = (
            "eos" if len(new_ids) and int(new_ids[-1]) in eos_ids
            else "length" if len(new_ids) >= 1280 else "other"
        )
        row = {
            "question_id": question_id,
            "sample_index": index,
            "worker_rank": rank,
            "variant": RAW1280,
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
            "e30_generation worker=%d device=%s completed=%d total=100 global_at_least=%d/200 question_id=%s finish=%s",
            rank, device, progress, progress * 2, question_id, finish,
        )
    return {"worker_rank": rank, "completed": 100}


def _paired(scores, left, right):
    meteor = [a["meteor"] - b["meteor"] for a, b in zip(scores[left], scores[right])]
    rouge = [a["rouge_l"] - b["rouge_l"] for a, b in zip(scores[left], scores[right])]
    return {
        "meteor_mean": fmean(meteor),
        "meteor_bootstrap_95_ci": _bootstrap_ci(meteor, seed=f"e30-{left}-{right}-meteor", iterations=10000),
        "rouge_l_mean": fmean(rouge),
        "rouge_l_bootstrap_95_ci": _bootstrap_ci(rouge, seed=f"e30-{left}-{right}-rouge", iterations=10000),
        "improved": sum(value > 0 for value in meteor),
        "worsened": sum(value < 0 for value in meteor),
        "tied": sum(value == 0 for value in meteor),
    }


def finalize(
    *, root: Path, e21: Path, e28: Path, training: Path, train: Path,
    dev: Path, output: Path, config: Config,
):
    checked = check_preflight(root, output, config)
    ensure_nltk_resources(download=False)
    questions, _, ids = sample(train, dev, config)
    controls, control_identity, _ = load_e21(e21, ids, config)
    raw1024, _ = load_e28_raw(e28, ids, config)
    folder = output / "evaluation" / RAW1280
    identities = []
    for rank in (0, 1):
        state = json.loads((folder / f"worker-{rank}-state.json").read_text(encoding="utf-8"))
        identity = state.get("run_identity", {})
        if (
            state.get("complete") is not True
            or state.get("completed_count") != 100
            or state.get("assigned_count") != 100
            or identity.get("worker_rank") != rank
            or identity.get("device") != f"cuda:{rank}"
            or identity.get("variant") != RAW1280
            or identity.get("max_new_tokens") != 1280
            or identity.get("no_repeat_ngram_size") != 0
            or identity.get("assigned_indices") != assigned_indices(rank)
            or identity.get("code_sha256") != code_sha(root)
            or identity.get("config_sha256") != config.sha
            or identity.get("e21_prepared_sha256") != checked["e21_prepared_sha256"]
            or identity.get("control_identity_sha256") != control_identity["identity_sha256"]
            or identity.get("e28_raw1024_results_sha256") != checked["e28_raw1024_results_sha256"]
            or identity.get("adapter_sha256") != checked["adapter_sha256"]
            or identity.get("runtime") != control_identity["runtime"]
            or identity.get("generation_config_sha256") != control_identity["generation_config_sha256"]
            or identity.get("identity_sha256")
            != _json_sha256({key: value for key, value in identity.items() if key != "identity_sha256"})
        ):
            raise E30Error("Incomplete or changed E30 worker state.")
        identities.append(identity)
    raw1280 = []
    for index, question_id in enumerate(ids):
        rank = 0 if index < 100 else 1
        row = json.loads((folder / f"records/{index:04d}.json").read_text(encoding="utf-8"))
        validate_record(row, question_id, index, rank, identities[rank])
        raw1280.append(row)
    _atomic_jsonl(folder / "results.jsonl", raw1280)

    try:
        from transformers import AutoTokenizer
    except ImportError as exc:  # pragma: no cover
        raise E30Error("Transformers is required for E30 finalization.") from exc
    tokenizer = AutoTokenizer.from_pretrained(
        config.e19.generator_model_id, revision=config.e19.generator_revision,
        trust_remote_code=False,
    )
    def derive_tailtrim(rows, variant):
        derived_rows = []
        for row in rows:
            trimmed = trim_repeated_tail(row["answer"])
            derived = {
                **row,
                "variant": variant,
                "answer": trimmed["answer"],
                "output_tokens": len(tokenizer(trimmed["answer"], add_special_tokens=False)["input_ids"]),
                "source_finish_reason": row["finish_reason"],
                "postprocess_changed": trimmed["changed"],
                "removed_characters": trimmed["removed_characters"],
                "removed_line_blocks": trimmed["removed_line_blocks"],
                "removed_sentence_blocks": trimmed["removed_sentence_blocks"],
            }
            derived.pop("record_sha256", None)
            derived["record_sha256"] = _json_sha256(derived)
            derived_rows.append(derived)
        return derived_rows

    trim1024 = derive_tailtrim(raw1024, TRIM1024)
    trim1280 = derive_tailtrim(raw1280, TRIM1280)
    trim_path = output / f"evaluation/{TRIM1024}/results.jsonl"
    trim1280_path = output / f"evaluation/{TRIM1280}/results.jsonl"
    _atomic_jsonl(trim_path, trim1024)
    _atomic_jsonl(trim1280_path, trim1280)

    by_variant = {
        CONTROL: controls,
        RAW1024: raw1024,
        TRIM1024: trim1024,
        RAW1280: raw1280,
        TRIM1280: trim1280,
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
            "tailtrim_changed": trim1024[index]["postprocess_changed"],
            "tailtrim1280_changed": trim1280[index]["postprocess_changed"],
        }
        for index, question_id in enumerate(ids)
    ])
    metrics = {variant: _metrics(rows, scores[variant]) for variant, rows in by_variant.items()}
    for variant, rows in ((TRIM1024, trim1024), (TRIM1280, trim1280)):
        metrics[variant].update({
            "postprocess_changed_rate": fmean(row["postprocess_changed"] for row in rows),
            "changed_questions": sum(row["postprocess_changed"] for row in rows),
            "mean_removed_characters": fmean(row["removed_characters"] for row in rows),
        })
    report = {
        "schema_version": "1.0",
        "experiment_id": EXPERIMENT,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "sample_size": 200,
        "sample_scope": "same-repeated-e19-e21-dev400-600",
        "control_variant": CONTROL,
        "reused_variant": RAW1024,
        "derived_variants": [TRIM1024, TRIM1280],
        "generated_variant": RAW1280,
        "metrics": metrics,
        "primary_comparison": "max1280_tailtrim_minus_max1024_tailtrim",
        "paired_tailtrim1280_minus_tailtrim1024": _paired(scores, TRIM1280, TRIM1024),
        "paired_tailtrim1024_minus_raw1024": _paired(scores, TRIM1024, RAW1024),
        "paired_tailtrim1280_minus_raw1280": _paired(scores, TRIM1280, RAW1280),
        "paired_raw1280_minus_raw1024": _paired(scores, RAW1280, RAW1024),
        "paired_raw1280_minus_tailtrim1024": _paired(scores, RAW1280, TRIM1024),
        "paired_tailtrim1024_minus_control": _paired(scores, TRIM1024, CONTROL),
        "paired_raw1280_minus_control": _paired(scores, RAW1280, CONTROL),
        "tailtrim_diagnostics": {
            TRIM1024: {
                "changed_questions": sum(row["postprocess_changed"] for row in trim1024),
                "unchanged_questions": sum(not row["postprocess_changed"] for row in trim1024),
                "removed_characters_total": sum(row["removed_characters"] for row in trim1024),
                "removed_line_blocks": sum(row["removed_line_blocks"] for row in trim1024),
                "removed_sentence_blocks": sum(row["removed_sentence_blocks"] for row in trim1024),
            },
            TRIM1280: {
                "changed_questions": sum(row["postprocess_changed"] for row in trim1280),
                "unchanged_questions": sum(not row["postprocess_changed"] for row in trim1280),
                "removed_characters_total": sum(row["removed_characters"] for row in trim1280),
                "removed_line_blocks": sum(row["removed_line_blocks"] for row in trim1280),
                "removed_sentence_blocks": sum(row["removed_sentence_blocks"] for row in trim1280),
            },
            "answers_used": False,
        },
        "length_finish_count": {
            RAW1024: sum(row["finish_reason"] == "length" for row in raw1024),
            RAW1280: sum(row["finish_reason"] == "length" for row in raw1280),
        },
        "smoke_leader": max(metrics, key=lambda variant: metrics[variant]["meteor"]),
        "promotion_allowed": False,
        "public_read": False,
        "holdout_untouched": True,
        "evidence": {
            "config_sha256": config.sha,
            "code_sha256": code_sha(root),
            "control_results_sha256": config.raw["control"]["results_sha256"],
            "reused_raw1024_results_sha256": config.raw["reused_max1024"]["results_sha256"],
            "tailtrim1024_results_sha256": file_sha256(trim_path),
            "raw1280_results_sha256": file_sha256(folder / "results.jsonl"),
            "tailtrim1280_results_sha256": file_sha256(trim1280_path),
        },
        "warning": "Repeated dev-200 smoke evidence; do not treat it as unseen validation.",
    }
    _atomic_json(output / "report.json", report)
    return report
