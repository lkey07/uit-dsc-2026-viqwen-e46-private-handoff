"""E28 general max1024 and deterministic exact-16-token anti-loop comparison."""
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
    _bootstrap_ci,
    _json_sha256,
    _load_worker_progress,
    _read_jsonl,
    _write_worker_state,
)
from .e18_source_metadata import save_once
from .e19_metadata_lora import _metrics, load_candidate_generator, validate_candidate_adapter
from .evaluation.official import ensure_nltk_resources, nltk_meteor_score, rouge_l_fmeasure


EXPERIMENT = "E28-long-output-antiloop-dev200-v1"
CONTROL = "parent_expanded_max704"
VARIANTS = (
    "e21_parent_greedy_max1024",
    "e21_parent_exact16_antiloop_max1024",
)
CODE_VERSION = "0.62.0"
LOG = logging.getLogger(__name__)


class E28Error(RuntimeError):
    """Raised when E28 reviewed evidence or a checkpoint identity changes."""


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
        "source_e21_config_sha256", "control", "variants", "inference",
        "evaluation", "parameter_budget", "run_contract",
    }:
        raise E28Error("Unexpected E28 config keys.")
    if (
        raw["schema_version"] != "1.0"
        or raw["experiment_id"] != EXPERIMENT
        or raw["source_e21_config_path"] != "configs/e21-parent-context-dev200-v1.json"
        or raw["source_e21_config_sha256"]
        != "fe8554d5f222e03a8f60ee181f8cfc4d1570719efaa6c56e2f7559fad1672ade"
    ):
        raise E28Error("E28 source identity changed.")
    e21_path = root / raw["source_e21_config_path"]
    if file_sha256(e21_path) != raw["source_e21_config_sha256"]:
        raise E28Error("Pinned E21 config bytes changed.")
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
        raise E28Error("Pinned E21 control changed.")
    if raw["variants"] != [
        {
            "key": VARIANTS[0], "worker_rank": 0,
            "max_new_tokens": 1024, "no_repeat_ngram_size": 0,
        },
        {
            "key": VARIANTS[1], "worker_rank": 1,
            "max_new_tokens": 1024, "no_repeat_ngram_size": 16,
        },
    ]:
        raise E28Error("E28 variants changed.")
    if raw["inference"] != {
        "base_context": "exact-e21-parent-expanded",
        "max_input_tokens": 8192,
        "generator": "e19_metadata_trained_rank8",
        "do_sample": False,
        "num_beams": 1,
        "repetition_penalty": 1.0,
        "enable_thinking": False,
        "use_cache": True,
        "apply_policy_to_every_question": True,
    }:
        raise E28Error("E28 inference policy changed.")
    if raw["evaluation"] != {
        "sample": "reuse-e19-e21-dev400-600",
        "sample_size": 200,
        "sample_ids_sha256": "dc17d6c9af9e941868c03cfc61fd17cdd1dc55aa5834beb3e75ba0fde32a5c73",
        "questions_per_variant": 200,
        "promotion_allowed": False,
    }:
        raise E28Error("E28 evaluation sample changed.")
    budget = raw["parameter_budget"]
    if (
        budget != {
            "exclusive_limit": 4_000_000_000,
            "generator": 2_274_069_824,
            "adapter_parameter_cap": 50_000_000,
            "maximum_stack_total": 2_324_069_824,
        }
        or budget["maximum_stack_total"] >= budget["exclusive_limit"]
    ):
        raise E28Error("E28 parameter budget failed.")
    if raw["run_contract"] != {
        "top12_ranking_unchanged": True,
        "e21_parent_expansion_unchanged": True,
        "all_questions_regenerated": True,
        "no_dev_specific_routing": True,
        "same_policy_for_public_and_private": True,
        "dev_answers_scoring_only": True,
        "same_e19_adapter_and_prompt": True,
        "no_external_or_synthetic_data": True,
        "no_api_model": True,
        "fixed_non_agentic_rag": True,
        "holdout_untouched": True,
        "public_not_read": True,
        "checkpoint_after_each_question_id": True,
        "resume_fail_closed": True,
    }:
        raise E28Error("E28 run contract changed.")
    return Config(raw=raw, path=path, e21=e21)


def code_sha(root: Path) -> str:
    paths = sorted((root / "src/uit_dsc_fixed_rag").rglob("*.py"))
    paths.append(root / "scripts/run_e28_long_output_antiloop_kaggle.py")
    return _json_sha256({
        path.relative_to(root).as_posix(): file_sha256(path)
        for path in paths if path.is_file()
    })


def sample(train: Path, dev: Path, config: Config):
    return parent.sample(train, dev, config.e21)


def _ids_sha(ids: list[str]) -> str:
    return hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()


def load_e21(directory: Path, ids: list[str], config: Config):
    return e27.load_e21(directory, ids, config)


def validate_preflight(
    *, root: Path, e21: Path, training: Path, train: Path, dev: Path,
    output: Path, config: Config,
):
    _, _, ids = sample(train, dev, config)
    controls, control_identity, _ = load_e21(e21, ids, config)
    if len(controls) != 200 or _ids_sha(ids) != config.raw["evaluation"]["sample_ids_sha256"]:
        raise E28Error("E28 sample identity changed.")
    adapter_sha, complete = validate_candidate_adapter(training, config.e19)
    if adapter_sha != config.raw["control"]["adapter_sha256"]:
        raise E28Error("E28 requires the exact E19 adapter used by E21.")
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
        "variants": list(VARIANTS),
        "all_questions_regenerated": True,
        "dev_specific_routing": False,
        "answers_used_during_generation": False,
    }
    save_once(output / "preflight.json", payload)
    return payload


def check_preflight(root: Path, output: Path, config: Config):
    path = output / "preflight.json"
    if not path.is_file():
        raise E28Error("Run E28 preflight first.")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("experiment_id") != EXPERIMENT
        or payload.get("config_sha256") != config.sha
        or payload.get("code_sha256") != code_sha(root)
        or payload.get("all_questions_regenerated") is not True
        or payload.get("dev_specific_routing") is not False
    ):
        raise E28Error("E28 preflight code/config changed.")
    return payload


def validate_record(row, question_id, index, rank, identity):
    if (
        row.get("question_id") != question_id
        or row.get("sample_index") != index
        or row.get("worker_rank") != rank
        or row.get("variant") != VARIANTS[rank]
        or row.get("worker_identity_sha256") != identity["identity_sha256"]
        or row.get("regenerated") is not True
        or not isinstance(row.get("answer"), str)
        or not row["answer"].strip()
        or row.get("record_sha256")
        != _json_sha256({key: value for key, value in row.items() if key != "record_sha256"})
    ):
        raise E28Error(f"Changed E28 generation record: {rank}/{index}")


def run_worker(
    *, root: Path, e21: Path, training: Path, train: Path, dev: Path,
    output: Path, config: Config, rank: int, device: str,
):
    import torch

    if rank not in (0, 1) or device != f"cuda:{rank}":
        raise E28Error("GPU0=raw max1024 all200; GPU1=exact16 anti-loop max1024 all200.")
    checked = check_preflight(root, output, config)
    questions, _, ids = sample(train, dev, config)
    controls, control_identity, prepared = load_e21(e21, ids, config)
    runtime = {name: importlib.metadata.version(name) for name in control_identity["runtime"]}
    if runtime != control_identity["runtime"]:
        raise E28Error(
            f"Use the exact E21 runtime. expected={control_identity['runtime']} current={runtime}"
        )
    model, tokenizer, placement, parameters = load_candidate_generator(
        config=config.e19, training_directory=training, device=device,
    )
    adapter_sha, _ = validate_candidate_adapter(training, config.e19)
    generation_sha = _json_sha256(model.generation_config.to_dict())
    if adapter_sha != checked["adapter_sha256"] or generation_sha != control_identity["generation_config_sha256"]:
        raise E28Error("E19 adapter or generation defaults differ from E21.")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    def render(messages):
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )

    def count(text):
        return len(tokenizer(text, add_special_tokens=False)["input_ids"])

    variant = config.raw["variants"][rank]
    identity = {
        "code_sha256": code_sha(root),
        "config_sha256": config.sha,
        "e21_prepared_sha256": checked["e21_prepared_sha256"],
        "control_identity_sha256": control_identity["identity_sha256"],
        "adapter_sha256": adapter_sha,
        "generation_config_sha256": generation_sha,
        "runtime": runtime,
        "variant": VARIANTS[rank],
        "max_new_tokens": variant["max_new_tokens"],
        "no_repeat_ngram_size": variant["no_repeat_ngram_size"],
        "all_questions_regenerated": True,
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
        validate_record(
            json.loads((records / f"{index:04d}.json").read_text(encoding="utf-8")),
            ids[index], index, rank, identity,
        )
    for index in assigned[done:]:
        question_id = ids[index]
        messages, spans, diagnostics = parent.pack(
            questions[question_id]["question"], prepared[index], config.e21,
            lambda value: count(render(value)), count, parent.VARIANTS[1],
        )
        prompt = render(messages)
        prompt_sha = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        if prompt_sha != controls[index]["prompt_sha256"]:
            raise E28Error(f"E28 failed to reproduce E21 prompt: {question_id}")
        inputs = {
            key: value.to(device)
            for key, value in tokenizer(prompt, add_special_tokens=False, return_tensors="pt").items()
        }
        started = time.perf_counter()
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                do_sample=False,
                num_beams=1,
                repetition_penalty=1.0,
                no_repeat_ngram_size=variant["no_repeat_ngram_size"],
                max_new_tokens=variant["max_new_tokens"],
                use_cache=True,
            )
        latency = (time.perf_counter() - started) * 1000
        new_ids = generated[0, inputs["input_ids"].shape[1]:]
        answer = tokenizer.decode(
            new_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False,
        ).strip()
        if not answer:
            raise E28Error(f"Empty E28 answer: {question_id}")
        eos = model.generation_config.eos_token_id
        eos_ids = set(eos if isinstance(eos, list) else [eos])
        finish = (
            "eos" if len(new_ids) and int(new_ids[-1]) in eos_ids
            else "length" if len(new_ids) >= variant["max_new_tokens"] else "other"
        )
        row = {
            "question_id": question_id,
            "sample_index": index,
            "worker_rank": rank,
            "variant": VARIANTS[rank],
            "worker_identity_sha256": identity["identity_sha256"],
            "regenerated": True,
            "answer": answer,
            "input_tokens": count(prompt),
            "output_tokens": count(answer),
            "generated_tokens_including_special": len(new_ids),
            "finish_reason": finish,
            "generation_latency_ms": latency,
            "selected_context_count": len(spans),
            "selected_chunk_ids": diagnostics["selected_chunk_ids"],
            "prompt_sha256": prompt_sha,
            "no_repeat_ngram_size": variant["no_repeat_ngram_size"],
        }
        row["record_sha256"] = _json_sha256(row)
        _atomic_json(records / f"{index:04d}.json", row)
        _write_worker_state(state, identity, index + 1, 200)
        LOG.info(
            "e28_generation variant=%s device=%s completed=%d total=200 question_id=%s finish=%s",
            VARIANTS[rank], device, index + 1, question_id, finish,
        )
    return {"variant": VARIANTS[rank], "completed": 200}


def _paired(scores, left, right):
    delta = [a["meteor"] - b["meteor"] for a, b in zip(scores[left], scores[right])]
    return {
        "meteor_mean": fmean(delta),
        "meteor_bootstrap_95_ci": _bootstrap_ci(
            delta, seed=f"e28-{left}-{right}", iterations=10000,
        ),
        "improved": sum(value > 0 for value in delta),
        "worsened": sum(value < 0 for value in delta),
        "tied": sum(value == 0 for value in delta),
    }


def finalize(
    *, root: Path, e21: Path, training: Path, train: Path, dev: Path,
    output: Path, config: Config,
):
    checked = check_preflight(root, output, config)
    ensure_nltk_resources(download=False)
    questions, _, ids = sample(train, dev, config)
    controls, control_identity, _ = load_e21(e21, ids, config)
    by_variant = {CONTROL: controls}
    for rank, variant in enumerate(VARIANTS):
        folder = output / "evaluation" / variant
        state = json.loads((folder / "state.json").read_text(encoding="utf-8"))
        identity = state.get("run_identity", {})
        expected_variant = config.raw["variants"][rank]
        if (
            state.get("complete") is not True
            or state.get("completed_count") != 200
            or state.get("assigned_count") != 200
            or identity.get("worker_rank") != rank
            or identity.get("device") != f"cuda:{rank}"
            or identity.get("variant") != variant
            or identity.get("max_new_tokens") != 1024
            or identity.get("no_repeat_ngram_size") != expected_variant["no_repeat_ngram_size"]
            or identity.get("all_questions_regenerated") is not True
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
            raise E28Error("Incomplete or changed E28 generation state.")
        rows = []
        for index, question_id in enumerate(ids):
            row = json.loads((folder / f"records/{index:04d}.json").read_text(encoding="utf-8"))
            validate_record(row, question_id, index, rank, identity)
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
        "paired_raw1024_minus_e21": _paired(scores, VARIANTS[0], CONTROL),
        "paired_antiloop1024_minus_e21": _paired(scores, VARIANTS[1], CONTROL),
        "paired_antiloop_minus_raw1024": _paired(scores, VARIANTS[1], VARIANTS[0]),
        "generation_diagnostics": {
            "all_questions_regenerated": True,
            "dev_specific_routing": False,
            "policy_public_private_compatible": True,
            "length_finish_count": {
                variant: sum(row["finish_reason"] == "length" for row in by_variant[variant])
                for variant in VARIANTS
            },
            "answers_used_during_generation": False,
        },
        "smoke_leader": max(metrics, key=lambda variant: metrics[variant]["meteor"]),
        "promotion_allowed": False,
        "public_read": False,
        "holdout_untouched": True,
        "evidence": {
            "config_sha256": config.sha,
            "code_sha256": code_sha(root),
            "control_results_sha256": config.raw["control"]["results_sha256"],
            "candidate_results_sha256": {
                variant: file_sha256(output / f"evaluation/{variant}/results.jsonl")
                for variant in VARIANTS
            },
        },
        "warning": "Repeated dev-200; freeze a material winner before one final untouched-dev121 check.",
    }
    _atomic_json(output / "report.json", report)
    return report

