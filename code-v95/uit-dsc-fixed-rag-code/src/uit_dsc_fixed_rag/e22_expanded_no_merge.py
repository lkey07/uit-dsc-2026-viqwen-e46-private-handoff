"""E19-style expanded contexts without merging: one dev200 pass, split 100/100."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from statistics import fmean

from . import e21_parent_context as parent
from .corpus import file_sha256
from .e02_answer import _atomic_json, _atomic_jsonl, make_messages
from .e03_rrf_grid import _json_sha256, _read_jsonl, _load_worker_progress, _write_worker_state, _bootstrap_ci
from .e08b_context_lora import inference_packing
from .e18_source_metadata import save_once
from .e19_metadata_lora import validate_candidate_adapter, load_candidate_generator, _metrics
from .evaluation.official import ensure_nltk_resources, nltk_meteor_score, rouge_l_fmeasure

EXPERIMENT = "E22-expanded-no-merge-dev200-v1"
CANDIDATE = "expanded_no_merge_max704"
CONTROL = "parent_expanded_max704"
POLICY = dict(format="e19-per-chunk-metadata-original-seed-order-no-merge",
              expansion="same-e21-options-greedy-under-unmerged-prompt-budget",
              max_input_tokens=8192, max_new_tokens=704, keep_all_seeds=True,
              sample_size=200, partition="contiguous-100-100", runtime="match-e21-control")
LOG = logging.getLogger(__name__)


class E22Error(RuntimeError):
    pass


@dataclass(frozen=True)
class Config:
    raw: dict
    path: object
    source: object

    @property
    def sha(self):
        return file_sha256(self.path)

    @property
    def e19(self):
        return self.source.source.source


def load_config(root, path):
    raw = json.loads(path.read_text(encoding="utf-8"))
    if (set(raw) != {"schema_version", "experiment_id", "source_config", "source_config_sha256",
                     "source_code_sha256", "control_results_sha256", "policy", "promotion_allowed"}
            or raw["schema_version"] != "1.0" or raw["experiment_id"] != EXPERIMENT
            or raw["policy"] != POLICY or raw["promotion_allowed"] is not False
            or raw["source_config"] != "configs/e21-parent-context-dev200-v1.json"
            or raw["source_config_sha256"] != "fe8554d5f222e03a8f60ee181f8cfc4d1570719efaa6c56e2f7559fad1672ade"
            or raw["source_code_sha256"] != "e5574861eb34ae3fdee1220d5260109ddd9f84feacea9c1ec46a78df8b5527c8"
            or raw["control_results_sha256"] != "cbe3d5e163e1983f031a06c767596fcebd91177e3b90625998a1deed28f6afe4"):
        raise E22Error("Reviewed E22 contract changed.")
    source = parent.load_config(root, root / raw["source_config"])
    if source.sha != raw["source_config_sha256"]:
        raise E22Error("Pinned E21 configuration changed.")
    return Config(raw, path, source)


def code_sha(root):
    paths = sorted((root / "src/uit_dsc_fixed_rag").rglob("*.py"))
    paths.append(root / "scripts/run_e22_expanded_no_merge_kaggle.py")
    return _json_sha256({p.relative_to(root).as_posix(): file_sha256(p) for p in paths})


def sample(train, dev, config):
    return parent.sample(train, dev, config.source)


def load_source(directory, ids, config):
    """Anchor prepared inputs to the worker identity inside byte-pinned E21 answers."""
    result_path = directory / "evaluation" / CONTROL / "results.jsonl"
    if file_sha256(result_path) != config.raw["control_results_sha256"]:
        raise E22Error("Use the exact completed E21 GPU1 results.")
    rows = _read_jsonl(result_path)
    state = json.loads((result_path.parent / "state.json").read_text(encoding="utf-8"))
    identity = state["run_identity"]
    adapter_sha = config.source.source.section("source_control")["adapter_sha256"]
    if (len(rows) != 200 or state.get("complete") is not True or state.get("completed_count") != 200
            or state.get("assigned_count") != 200 or identity.get("worker_rank") != 1
            or identity.get("variant") != CONTROL or identity.get("device") != "cuda:1"
            or identity.get("config_sha256") != config.raw["source_config_sha256"]
            or identity.get("code_sha256") != config.raw["source_code_sha256"]
            or identity.get("adapter_sha256") != adapter_sha
            or identity.get("sample_ids_sha256") != config.source.source.section("evaluation")["sample_ids_sha256"]
            or identity.get("prepared_sha256") != file_sha256(directory / "prepared/results.jsonl")
            or identity.get("identity_sha256") != _json_sha256({k: v for k, v in identity.items() if k != "identity_sha256"})):
        raise E22Error("E21 state/prepared-input provenance is incomplete or changed.")
    for i, qid in enumerate(ids):
        parent.validate_record(rows[i], qid, i, 1, identity)
    prepared = parent.load_prepared(directory, ids, config.source)
    return prepared, rows, identity


def preflight(root, output, source, training, train, dev, config):
    _, _, ids = sample(train, dev, config)
    _, _, identity = load_source(source, ids, config)
    adapter_sha, _ = validate_candidate_adapter(training, config.e19)
    if adapter_sha != identity["adapter_sha256"]:
        raise E22Error("Wrong E19 adapter.")
    budget = config.source.source.section("parameter_budget")
    total = budget["embedding"] + budget["generator"] + budget["adapter_parameter_cap"]
    if total != budget["maximum_stack_total"] or total >= 4_000_000_000:
        raise E22Error("Parameter budget violation.")
    scorer = config.source.source.section("scoring")
    if file_sha256(root / scorer["official_scorer_path"]) != scorer["official_scorer_sha256"]:
        raise E22Error("Pinned scorer changed.")
    payload = dict(experiment_id=EXPERIMENT, code_sha256=code_sha(root), config_sha256=config.sha,
                   source_identity_sha256=identity["identity_sha256"], adapter_sha256=adapter_sha,
                   sample_size=200, partitions=[100, 100], max_new_tokens=704,
                   maximum_stack_parameters=total, reference_answers_used_for_packing=False)
    save_once(output / "preflight.json", payload)
    return payload


def check_preflight(root, output, config):
    p = json.loads((output / "preflight.json").read_text(encoding="utf-8"))
    if (p.get("experiment_id") != EXPERIMENT or p.get("config_sha256") != config.sha
            or p.get("code_sha256") != code_sha(root)):
        raise E22Error("E22 checkpoint config/code changed.")
    return p


def render(question, chosen, config):
    contexts = []
    for u in chosen:
        lines = []
        for label, value in (("Tên nguồn", u["source_title"]), ("Số văn bản", u["document_number"]),
                             ("Tiêu đề Điều", u["article_title"])):
            if value and value.strip():
                lines.append(f"{label}: {value.strip()}")
        prefix = "\n".join(lines) + "\nNội dung:\n" if lines else ""
        contexts.append({"article_number": u["article_number"], "text": prefix + u["text"]})
    return make_messages(question=question, contexts=contexts, config=inference_packing(config.e19))


def coverage(spans):
    return sorted((s["document_id"], s["block_id"], s["start"], s["end"]) for s in spans)


def pack(question, prepared, config, message_counter, text_counter, control):
    units = prepared["units"]
    chosen = [{**{k: v for k, v in u.items() if k not in ("seed", "expansions")}, **u["seed"]} for u in units]
    cap = config.raw["policy"]["max_input_tokens"]
    if len(chosen) != 12 or message_counter(render(question, chosen, config)) > cap:
        raise E22Error("All 12 E19-style seeds must fit; refusing to discard original evidence.")
    actions = []
    for i, seed in enumerate(list(chosen)):
        for extension in units[i]["expansions"]:
            if (len(extension["text"]) > config.source.policy["max_parent_characters"]
                    or text_counter(extension["text"]) > config.source.policy["max_parent_tokens"]):
                continue
            replacement = {**seed, **{k: v for k, v in extension.items() if k != "kind"}}
            trial = chosen[:i] + [replacement] + chosen[i + 1:]
            # Merging is used ONLY to measure unique evidence, never to construct the prompt.
            if sum(len(s["text"]) for s in parent.merge_spans(trial)) <= sum(len(s["text"]) for s in parent.merge_spans(chosen)):
                continue
            if message_counter(render(question, trial, config)) <= cap:
                chosen = trial
                actions.append({"seed_rank": i, "kind": extension["kind"]})
                break
    unique = parent.merge_spans(chosen)
    diagnostics = dict(seed_ranks=list(range(12)), skipped_seed_ranks=[], expansions=actions,
                       unique_body_characters=sum(len(s["text"]) for s in unique),
                       rendered_body_characters=sum(len(s["text"]) for s in chosen),
                       same_unique_evidence_as_e21=coverage(unique) == coverage(control["evidence_spans"]),
                       same_expansion_actions_as_e21=actions == control["packing"]["expansions"])
    return render(question, chosen, config), chosen, diagnostics


def assigned_indices(rank):
    if rank not in (0, 1):
        raise E22Error("Worker rank must be 0 or 1.")
    return list(range(rank * 100, (rank + 1) * 100))


def validate_record(row, qid, index, rank, identity):
    if (row.get("question_id") != qid or row.get("sample_index") != index
            or row.get("worker_rank") != rank or row.get("variant") != CANDIDATE
            or row.get("worker_identity_sha256") != identity["identity_sha256"]
            or not isinstance(row.get("answer"), str) or not row["answer"].strip()
            or row.get("record_sha256") != _json_sha256({k: v for k, v in row.items() if k != "record_sha256"})):
        raise E22Error(f"Changed answer record: {rank}/{index}")


def run_worker(root, output, source, training, train, dev, config, rank, device):
    import torch
    assigned = assigned_indices(rank)
    if device != f"cuda:{rank}":
        raise E22Error("Worker/GPU mismatch.")
    checked = check_preflight(root, output, config)
    questions, _, ids = sample(train, dev, config)
    prepared, controls, source_identity = load_source(source, ids, config)
    if checked["source_identity_sha256"] != source_identity["identity_sha256"]:
        raise E22Error("Source changed after preflight.")
    runtime = {key: importlib.metadata.version(key) for key in source_identity["runtime"]}
    if runtime != source_identity["runtime"]:
        raise E22Error("Use the E21 runtime versions printed by the notebook; do not mix environments.")
    model, tokenizer, placement, params = load_candidate_generator(config=config.e19, training_directory=training, device=device)
    observed_adapter, _ = validate_candidate_adapter(training, config.e19)
    generation_sha = _json_sha256(model.generation_config.to_dict())
    if observed_adapter != checked["adapter_sha256"] or generation_sha != source_identity["generation_config_sha256"]:
        raise E22Error("Adapter/generation defaults differ from E21 control.")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    def rendered(messages):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    def text_count(text):
        return len(tokenizer(text, add_special_tokens=False)["input_ids"])
    identity = dict(code_sha256=code_sha(root), config_sha256=config.sha,
                    source_identity_sha256=source_identity["identity_sha256"], adapter_sha256=observed_adapter,
                    generation_config_sha256=generation_sha, runtime=runtime, worker_rank=rank, device=device,
                    variant=CANDIDATE, assigned_indices=assigned, device_map=placement, adapter_parameters=params)
    identity["identity_sha256"] = _json_sha256(identity)
    folder = output / "generation" / f"worker-{rank}"
    records, state = folder / "records", folder / "state.json"
    records.mkdir(parents=True, exist_ok=True)
    done = _load_worker_progress(records=records, state_path=state, identity=identity, assigned_indices=assigned, sample_ids=ids)
    for i in assigned[:done]:
        validate_record(json.loads((records / f"{i:04d}.json").read_text(encoding="utf-8")), ids[i], i, rank, identity)
    for completed, i in enumerate(assigned[done:], start=done + 1):
        messages, spans, diagnostics = pack(questions[ids[i]]["question"], prepared[i], config,
                                             lambda m: text_count(rendered(m)), text_count, controls[i])
        prompt = rendered(messages)
        inputs = {k: v.to(device) for k, v in tokenizer(prompt, add_special_tokens=False, return_tensors="pt").items()}
        started = time.perf_counter()
        with torch.inference_mode():
            generated = model.generate(**inputs, do_sample=False, num_beams=1, max_new_tokens=704, use_cache=True)
        latency = (time.perf_counter() - started) * 1000
        new_ids = generated[0, inputs["input_ids"].shape[1]:]
        answer = tokenizer.decode(new_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False).strip()
        if not answer:
            raise E22Error(f"Empty answer: {ids[i]}")
        eos = tokenizer.eos_token_id
        eos_ids = set(eos if isinstance(eos, list) else [eos])
        finish = "eos" if len(new_ids) and int(new_ids[-1]) in eos_ids else "length" if len(new_ids) >= 704 else "other"
        row = dict(question_id=ids[i], sample_index=i, worker_rank=rank, variant=CANDIDATE,
                   worker_identity_sha256=identity["identity_sha256"], answer=answer,
                   input_tokens=text_count(prompt), output_tokens=text_count(answer),
                   generated_tokens_including_special=len(new_ids), finish_reason=finish,
                   generation_latency_ms=latency, selected_context_count=len(spans), packing=diagnostics,
                   prompt_sha256=hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                   evidence_spans=[{k: v for k, v in s.items() if k != "text"} for s in spans])
        row["record_sha256"] = _json_sha256(row)
        _atomic_json(records / f"{i:04d}.json", row)
        _write_worker_state(state, identity, completed, 100)
        LOG.info("e22_generation worker=%d device=%s completed=%d total=100 question_id=%s finish=%s", rank, device, completed, ids[i], finish)
    return dict(worker=rank, completed=100)


def finalize(root, output, source, train, dev, config):
    checked = check_preflight(root, output, config)
    ensure_nltk_resources(download=False)
    questions, _, ids = sample(train, dev, config)
    _, controls, source_identity = load_source(source, ids, config)
    if checked["source_identity_sha256"] != source_identity["identity_sha256"]:
        raise E22Error("Source changed after preflight.")
    candidates = []
    for rank in range(2):
        folder = output / "generation" / f"worker-{rank}"
        state = json.loads((folder / "state.json").read_text(encoding="utf-8"))
        identity = state["run_identity"]
        if (state.get("complete") is not True or state.get("completed_count") != 100 or state.get("assigned_count") != 100
                or identity.get("assigned_indices") != assigned_indices(rank) or identity.get("worker_rank") != rank
                or identity.get("device") != f"cuda:{rank}" or identity.get("variant") != CANDIDATE
                or identity.get("config_sha256") != config.sha or identity.get("code_sha256") != code_sha(root)
                or identity.get("source_identity_sha256") != source_identity["identity_sha256"]
                or identity.get("adapter_sha256") != checked["adapter_sha256"]
                or identity.get("runtime") != source_identity["runtime"]
                or identity.get("generation_config_sha256") != source_identity["generation_config_sha256"]
                or identity.get("identity_sha256") != _json_sha256({k: v for k, v in identity.items() if k != "identity_sha256"})):
            raise E22Error("Incomplete or changed E22 worker state.")
        for i in assigned_indices(rank):
            row = json.loads((folder / "records" / f"{i:04d}.json").read_text(encoding="utf-8"))
            validate_record(row, ids[i], i, rank, identity)
            candidates.append(row)
    by_variant = {CONTROL: controls, CANDIDATE: candidates}
    scores = {v: [dict(meteor=nltk_meteor_score(questions[q]["answer"], r["answer"]),
                       rouge_l=rouge_l_fmeasure(questions[q]["answer"], r["answer"])) for q, r in zip(ids, rows)]
              for v, rows in by_variant.items()}
    delta = [x["meteor"] - y["meteor"] for x, y in zip(scores[CANDIDATE], scores[CONTROL])]
    equal = [i for i, r in enumerate(candidates) if r["packing"]["same_unique_evidence_as_e21"]]
    changed = [i for i in range(200) if i not in equal]
    _atomic_jsonl(output / "results.jsonl", candidates)
    _atomic_jsonl(output / "per_question_scores.jsonl", [dict(question_id=q, scores={v: scores[v][i] for v in scores},
                       same_unique_evidence_as_e21=i in equal) for i, q in enumerate(ids)])
    report = dict(schema_version="1.0", experiment_id=EXPERIMENT, created_at_utc=datetime.now(timezone.utc).isoformat(),
                  sample_size=200, worker_partitions=[100, 100], max_new_tokens=704,
                  control_variant=CONTROL, candidate_variant=CANDIDATE,
                  metrics={v: _metrics(rows, scores[v]) for v, rows in by_variant.items()},
                  paired_candidate_minus_e21=dict(meteor_mean=fmean(delta),
                      meteor_bootstrap_95_ci=_bootstrap_ci(delta, seed="e22-no-merge-v1", iterations=10000),
                      improved=sum(x > 0 for x in delta), worsened=sum(x < 0 for x in delta), tied=sum(x == 0 for x in delta)),
                  packing=dict(same_unique_evidence_questions=len(equal), changed_unique_evidence_questions=len(changed),
                      same_evidence_meteor_delta=fmean(delta[i] for i in equal) if equal else None,
                      changed_evidence_meteor_delta=fmean(delta[i] for i in changed) if changed else None,
                      mean_input_tokens=fmean(r["input_tokens"] for r in candidates),
                      mean_unique_body_characters=fmean(r["packing"]["unique_body_characters"] for r in candidates),
                      mean_rendered_body_characters=fmean(r["packing"]["rendered_body_characters"] for r in candidates)),
                  evidence=dict(config_sha256=config.sha, code_sha256=code_sha(root),
                      control_results_sha256=config.raw["control_results_sha256"],
                      source_identity_sha256=source_identity["identity_sha256"], results_sha256=file_sha256(output / "results.jsonl")),
                  promotion_allowed=False, public_read=False, holdout_untouched=True,
                  warning="Same used E19/E21 dev200. Unmerged formatting can change feasible expansions under 8192 tokens; not a pure fixed-evidence ablation for those questions.")
    _atomic_json(output / "report.json", report)
    return report
