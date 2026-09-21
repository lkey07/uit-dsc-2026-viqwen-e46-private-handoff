"""E26 question-only hierarchical article retrieval and E21-style generation."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import logging
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean
from typing import Any

from . import e21_parent_context as parent
from .bm25 import SqliteBm25Index
from .corpus import file_sha256
from .e02_answer import _atomic_json, _atomic_jsonl
from .e02_compare import _TorchGpuFlatIpIndex, _load_mapping
from .e03_rrf_grid import (
    _bootstrap_ci, _json_sha256, _load_worker_progress, _read_jsonl,
    _write_worker_state,
)
from .e08_context_retrieval import load_e08_context_config, load_embedding
from .e18_source_metadata import save_once, scan_selected
from .e19_metadata_lora import (
    _metrics, load_candidate_generator, validate_candidate_adapter,
)
from .evaluation.official import ensure_nltk_resources, nltk_meteor_score, rouge_l_fmeasure


EXPERIMENT = "E26-hierarchical-article-retrieval-dev200-v1"
CONTROL = "parent_expanded_max704"
VARIANTS = (
    "hierarchical_top8_articles_parent_max704",
    "hierarchical_top10_articles_parent_max704",
)
CODE_VERSION = "0.60.0"
LOG = logging.getLogger(__name__)


class E26Error(RuntimeError):
    pass


@dataclass(frozen=True)
class Config:
    raw: dict[str, Any]
    path: Path
    e08a: Any
    e21: Any

    @property
    def sha(self) -> str:
        return file_sha256(self.path)

    @property
    def e19(self) -> Any:
        return self.e21.source.source


def load_config(root: Path, path: Path) -> Config:
    raw = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "schema_version", "experiment_id", "source_e08a_config_path",
        "source_e08a_config_sha256", "source_e21_config_path",
        "source_e21_config_sha256", "control", "retrieval", "variants",
        "inference", "evaluation", "parameter_budget", "run_contract",
    }
    if set(raw) != expected or raw.get("schema_version") != "1.0" or raw.get("experiment_id") != EXPERIMENT:
        raise E26Error("E26 config identity changed.")
    if (
        raw["source_e08a_config_path"] != "configs/e08a-context-retrieval-v1.json"
        or raw["source_e08a_config_sha256"] != "a13103c6a7ad7432792c658f7a1242aa8e1187ab5a251c7dde6101dc9ff644f7"
        or raw["source_e21_config_path"] != "configs/e21-parent-context-dev200-v1.json"
        or raw["source_e21_config_sha256"] != "fe8554d5f222e03a8f60ee181f8cfc4d1570719efaa6c56e2f7559fad1672ade"
    ):
        raise E26Error("Pinned E08A/E21 configs changed.")
    e08a_path, e21_path = root / raw["source_e08a_config_path"], root / raw["source_e21_config_path"]
    if file_sha256(e08a_path) != raw["source_e08a_config_sha256"] or file_sha256(e21_path) != raw["source_e21_config_sha256"]:
        raise E26Error("Pinned E08A/E21 config bytes changed.")
    e08a, e21 = load_e08_context_config(e08a_path), parent.load_config(root, e21_path)
    if raw["control"] != {
        "experiment_id": "E21-parent-context-dev200-v1",
        "variant": CONTROL,
        "results_sha256": "cbe3d5e163e1983f031a06c767596fcebd91177e3b90625998a1deed28f6afe4",
        "code_sha256": "e5574861eb34ae3fdee1220d5260109ddd9f84feacea9c1ec46a78df8b5527c8",
        "adapter_sha256": "bfb9d8120337c4013b5dcb0b59c6d03beccfe9f9748e5a01ef62e1de64fb5edd",
        "meteor": 0.5633893466813007,
        "rouge_l": 0.5864053775086489,
        "sample_size": 200,
    }:
        raise E26Error("Pinned E21 control changed.")
    if raw["retrieval"] != {
        "candidate_k_per_branch": 100,
        "branches": ["bm25", "dense_aiteam"],
        "sparse_weight": 0.5,
        "dense_weight": 0.5,
        "rrf_constant": 60,
        "grouping": "exact-document-contiguous-article-run",
        "maximum_hits_per_branch_per_article": 3,
        "article_score": "sum-weighted-reciprocal-ranks-of-up-to-three-best-unique-chunks-per-branch",
        "representative_seed": "highest-individual-weighted-rrf-chunk",
        "answer_used": False,
    }:
        raise E26Error("E26 retrieval policy changed.")
    if raw["variants"] != [
        {"key": VARIANTS[0], "worker_rank": 0, "article_count": 8},
        {"key": VARIANTS[1], "worker_rank": 1, "article_count": 10},
    ]:
        raise E26Error("E26 variants changed.")
    if raw["inference"] != {
        "parent_policy": "exact-e21-bounded-same-article-expansion",
        "max_input_tokens": 8192, "max_new_tokens": 704,
        "generator": "e19_metadata_trained_rank8", "do_sample": False,
        "num_beams": 1, "enable_thinking": False, "use_cache": True,
    }:
        raise E26Error("E26 inference changed.")
    if raw["evaluation"] != {
        "sample": "reuse-e19-e21-dev400-600",
        "sample_size": 200,
        "sample_ids_sha256": "dc17d6c9af9e941868c03cfc61fd17cdd1dc55aa5834beb3e75ba0fde32a5c73",
        "questions_per_variant": 200,
        "promotion_allowed": False,
    }:
        raise E26Error("E26 evaluation sample changed.")
    budget = raw["parameter_budget"]
    if (
        budget != {
            "exclusive_limit": 4_000_000_000,
            "embedding": 567_754_752,
            "generator": 2_274_069_824,
            "adapter_parameter_cap": 50_000_000,
            "maximum_stack_total": 2_891_824_576,
        }
        or budget["maximum_stack_total"]
        != budget["embedding"] + budget["generator"] + budget["adapter_parameter_cap"]
        or budget["maximum_stack_total"] >= budget["exclusive_limit"]
    ):
        raise E26Error("E26 parameter budget failed.")
    if raw["run_contract"] != {
        "retrieval_uses_question_only": True,
        "dev_answers_scoring_only": True,
        "control_answers_reused_byte_exactly": True,
        "same_e19_adapter_prompt_parent_policy_and_max704": True,
        "embedding_and_generator_loaded_in_separate_processes": True,
        "no_external_or_synthetic_data": True,
        "no_api_model": True,
        "fixed_non_agentic_rag": True,
        "holdout_untouched": True,
        "public_not_read": True,
        "checkpoint_after_each_question_id": True,
        "resume_fail_closed": True,
    }:
        raise E26Error("E26 run contract lost an invariant.")
    return Config(raw=raw, path=path, e08a=e08a, e21=e21)


def code_sha(root: Path) -> str:
    paths = sorted((root / "src/uit_dsc_fixed_rag").rglob("*.py"))
    paths.append(root / "scripts/run_e26_hierarchical_article_kaggle.py")
    return _json_sha256({path.relative_to(root).as_posix(): file_sha256(path) for path in paths if path.is_file()})


def sample(train: Path, dev: Path, config: Config):
    return parent.sample(train, dev, config.e21)


def _ids_sha(ids: list[str]) -> str:
    return hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()


def load_control(directory: Path, ids: list[str], config: Config):
    result_path = directory / "evaluation" / CONTROL / "results.jsonl"
    state_path = result_path.parent / "state.json"
    if not result_path.is_file() or not state_path.is_file() or file_sha256(result_path) != config.raw["control"]["results_sha256"]:
        raise E26Error("Add the full byte-exact E21 output.")
    rows = _read_jsonl(result_path)
    state = json.loads(state_path.read_text(encoding="utf-8"))
    identity = state.get("run_identity", {})
    if (
        len(rows) != 200 or state.get("complete") is not True
        or state.get("completed_count") != 200 or state.get("assigned_count") != 200
        or identity.get("worker_rank") != 1 or identity.get("variant") != CONTROL
        or identity.get("device") != "cuda:1"
        or identity.get("config_sha256") != config.raw["source_e21_config_sha256"]
        or identity.get("code_sha256") != config.raw["control"]["code_sha256"]
        or identity.get("adapter_sha256") != config.raw["control"]["adapter_sha256"]
        or identity.get("identity_sha256") != _json_sha256({key: value for key, value in identity.items() if key != "identity_sha256"})
    ):
        raise E26Error("E21 control state changed.")
    for index, question_id in enumerate(ids):
        parent.validate_record(rows[index], question_id, index, 1, identity)
    return rows, identity


def validate_preflight(*, root: Path, e00: Path, dense: Path, control: Path, training: Path, train: Path, dev: Path, output: Path, config: Config):
    _, _, ids = sample(train, dev, config)
    controls, control_identity = load_control(control, ids, config)
    if len(controls) != 200 or _ids_sha(ids) != config.raw["evaluation"]["sample_ids_sha256"]:
        raise E26Error("E26 sample identity changed.")
    e00_source = config.e08a.section("source_e00")
    metadata = config.e19.contract["metadata_source"]
    expected_e00 = {
        "manifest.json": metadata["manifest_sha256"],
        "chunks.jsonl": metadata["chunks_sha256"],
        "documents.jsonl": metadata["documents_sha256"],
        "bm25.sqlite3": e00_source["bm25_sha256"],
    }
    for name, expected in expected_e00.items():
        path = e00 / name
        if not path.is_file() or file_sha256(path) != expected:
            raise E26Error(f"Pinned E00 artifact changed: {name}")
    dense_cfg = config.e08a.section("source_dense")
    manifest_path = dense / "manifest.json"
    if not manifest_path.is_file():
        raise E26Error("Pinned AITeam dense manifest is missing.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_candidate = {
        "key": "embedding_aiteam",
        "model_id": dense_cfg["model_id"],
        "revision": dense_cfg["revision"],
        "parameter_count": dense_cfg["parameter_count"],
        "output_dimension": dense_cfg["dimension"],
    }
    if (
        manifest.get("artifact_type") != "e02-dense-flat-ip"
        or manifest.get("artifact_version") != "e02-v2"
        or manifest.get("record_count") != dense_cfg["record_count"]
        or manifest.get("dimension") != dense_cfg["dimension"]
        or manifest.get("normalized") is not True
        or manifest.get("source_chunks_sha256") != metadata["chunks_sha256"]
        or manifest.get("candidate") != expected_candidate
    ):
        raise E26Error("Pinned AITeam dense manifest changed.")
    for name, key in (("dense.faiss", "dense_faiss_sha256"), ("chunk_ids.jsonl", "chunk_ids_sha256")):
        path = dense / name
        if not path.is_file() or file_sha256(path) != dense_cfg[key]:
            raise E26Error(f"Pinned AITeam dense artifact changed: {name}")
    adapter_sha, complete = validate_candidate_adapter(training, config.e19)
    if adapter_sha != config.raw["control"]["adapter_sha256"]:
        raise E26Error("E26 requires the exact E19 adapter used by E21.")
    payload = {
        "experiment_id": EXPERIMENT,
        "code_sha256": code_sha(root), "config_sha256": config.sha,
        "sample_ids_sha256": _ids_sha(ids), "sample_size": 200,
        "e00": expected_e00,
        "dense_faiss_sha256": dense_cfg["dense_faiss_sha256"],
        "dense_mapping_sha256": dense_cfg["chunk_ids_sha256"],
        "control_identity_sha256": control_identity["identity_sha256"],
        "control_results_sha256": config.raw["control"]["results_sha256"],
        "adapter_sha256": adapter_sha,
        "adapter_identity_sha256": complete["identity_sha256"],
        "candidate_k_per_branch": 100, "variants": list(VARIANTS),
        "maximum_stack_parameters": config.raw["parameter_budget"]["maximum_stack_total"],
        "answers_used_for_retrieval": False,
    }
    save_once(output / "preflight.json", payload)
    return payload


def check_preflight(root: Path, output: Path, config: Config):
    path = output / "preflight.json"
    if not path.is_file():
        raise E26Error("Run E26 preflight first.")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("experiment_id") != EXPERIMENT or payload.get("config_sha256") != config.sha or payload.get("code_sha256") != code_sha(root):
        raise E26Error("E26 preflight code/config changed.")
    return payload


def prepare_bm25(*, root: Path, e00: Path, runtime: Path, output: Path, config: Config):
    checked = check_preflight(root, output, config)
    source = e00 / "bm25.sqlite3"
    target = runtime / "bm25.sqlite3"
    runtime.mkdir(parents=True, exist_ok=True)
    if target.is_file():
        if file_sha256(target) != checked["e00"]["bm25.sqlite3"]:
            raise E26Error("Existing local BM25 copy is incompatible.")
    else:
        temporary = runtime / ".bm25.sqlite3.tmp"
        shutil.copy2(source, temporary)
        if file_sha256(temporary) != checked["e00"]["bm25.sqlite3"]:
            raise E26Error("Local BM25 copy failed checksum verification.")
        temporary.replace(target)
    return {"local_bm25": str(target), "sha256": file_sha256(target)}


def _validate_retrieval_record(row, question_id, index, branch, identity):
    key = f"{branch}_top_ids"
    if (
        row.get("question_id") != question_id or row.get("sample_index") != index
        or row.get("branch") != branch or row.get("worker_identity_sha256") != identity["identity_sha256"]
        or not isinstance(row.get(key), list) or not 1 <= len(row[key]) <= 100
        or len(row[key]) != len(set(row[key]))
        or row.get("record_sha256") != _json_sha256({name: value for name, value in row.items() if name != "record_sha256"})
    ):
        raise E26Error(f"Changed E26 {branch} record: {index}")


def run_sparse(*, root: Path, runtime: Path, train: Path, dev: Path, output: Path, config: Config):
    checked = check_preflight(root, output, config)
    questions, _, ids = sample(train, dev, config)
    assigned = list(range(200))
    identity = {
        "code_sha256": code_sha(root), "config_sha256": config.sha,
        "branch": "sparse", "sample_ids_sha256": _ids_sha(ids),
        "bm25_sha256": checked["e00"]["bm25.sqlite3"], "assigned_indices": assigned,
    }
    identity["identity_sha256"] = _json_sha256(identity)
    folder = output / "retrieval/sparse"
    records, state = folder / "records", folder / "state.json"
    records.mkdir(parents=True, exist_ok=True)
    done = _load_worker_progress(records=records, state_path=state, identity=identity, assigned_indices=assigned, sample_ids=ids)
    for index in assigned[:done]:
        _validate_retrieval_record(json.loads((records / f"{index:04d}.json").read_text(encoding="utf-8")), ids[index], index, "sparse", identity)
    pending = assigned[done:]
    local = threading.local()

    def search(index):
        engine = getattr(local, "engine", None)
        if engine is None:
            engine = SqliteBm25Index(runtime / "bm25.sqlite3")
            local.engine = engine
        started = time.perf_counter()
        hits = engine.search(questions[ids[index]]["question"], top_k=100)
        return [item.chunk_id for item in hits], [float(item.score) for item in hits], (time.perf_counter() - started) * 1000

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(search, index) for index in pending]
        for completed, (index, future) in enumerate(zip(pending, futures), start=done + 1):
            top_ids, scores, latency = future.result()
            row = {
                "question_id": ids[index], "sample_index": index, "branch": "sparse",
                "worker_identity_sha256": identity["identity_sha256"],
                "sparse_top_ids": top_ids, "scores": scores, "latency_ms": latency,
            }
            row["record_sha256"] = _json_sha256(row)
            _atomic_json(records / f"{index:04d}.json", row)
            _write_worker_state(state, identity, completed, 200)
            LOG.info("e26_sparse completed=%d total=200 question_id=%s", completed, ids[index])
    return {"branch": "sparse", "completed": 200}


def dense_indices(rank: int) -> list[int]:
    if rank not in (0, 1):
        raise E26Error("Dense worker rank must be 0 or 1.")
    return list(range(rank * 100, (rank + 1) * 100))


def run_dense_worker(*, root: Path, dense: Path, train: Path, dev: Path, output: Path, config: Config, rank: int, device: str):
    import numpy as np

    if device != f"cuda:{rank}":
        raise E26Error("E26 dense worker/GPU mismatch.")
    checked = check_preflight(root, output, config)
    questions, _, ids = sample(train, dev, config)
    assigned = dense_indices(rank)
    dense_cfg = config.e08a.section("source_dense")
    model = load_embedding(config.e08a, device)
    mapping = _load_mapping(dense / "chunk_ids.jsonl", dense_cfg["record_count"])
    search = _TorchGpuFlatIpIndex(
        dense / "dense.faiss", device=device, expected_count=dense_cfg["record_count"],
        expected_dimension=dense_cfg["dimension"],
    )
    runtime_versions = {name: importlib.metadata.version(name) for name in ("torch", "transformers", "sentence-transformers")}
    identity = {
        "code_sha256": code_sha(root), "config_sha256": config.sha,
        "branch": "dense", "worker_rank": rank, "device": device,
        "assigned_indices": assigned, "sample_ids_sha256": _ids_sha(ids),
        "dense_faiss_sha256": checked["dense_faiss_sha256"],
        "dense_mapping_sha256": checked["dense_mapping_sha256"],
        "embedding_model_id": dense_cfg["model_id"], "embedding_revision": dense_cfg["revision"],
        "runtime": runtime_versions,
    }
    identity["identity_sha256"] = _json_sha256(identity)
    folder = output / f"retrieval/dense/worker-{rank}"
    records, state = folder / "records", folder / "state.json"
    records.mkdir(parents=True, exist_ok=True)
    done = _load_worker_progress(records=records, state_path=state, identity=identity, assigned_indices=assigned, sample_ids=ids)
    for index in assigned[:done]:
        _validate_retrieval_record(json.loads((records / f"{index:04d}.json").read_text(encoding="utf-8")), ids[index], index, "dense", identity)
    try:
        batch_size = 32
        for start in range(done, len(assigned), batch_size):
            batch_indices = assigned[start:min(start + batch_size, len(assigned))]
            vectors = model.encode(
                [dense_cfg["query_prefix"] + questions[ids[index]]["question"] for index in batch_indices],
                batch_size=len(batch_indices), show_progress_bar=False,
                convert_to_numpy=True, normalize_embeddings=True,
            )
            vectors = np.asarray(vectors, dtype=np.float32)
            if vectors.shape != (len(batch_indices), dense_cfg["dimension"]) or not np.isfinite(vectors).all():
                raise E26Error("Invalid E26 query embeddings.")
            started = time.perf_counter()
            scores, rows = search.search(vectors, 100)
            latency = (time.perf_counter() - started) * 1000 / len(batch_indices)
            for offset, index in enumerate(batch_indices):
                top_ids = [mapping[int(value)] for value in rows[offset] if int(value) >= 0]
                row = {
                    "question_id": ids[index], "sample_index": index, "branch": "dense",
                    "worker_identity_sha256": identity["identity_sha256"],
                    "dense_top_ids": top_ids,
                    "scores": [float(value) for value in scores[offset][:len(top_ids)]],
                    "latency_ms": latency,
                }
                row["record_sha256"] = _json_sha256(row)
                _atomic_json(records / f"{index:04d}.json", row)
                completed = start + offset + 1
                _write_worker_state(state, identity, completed, 100)
                LOG.info("e26_dense worker=%d completed=%d total=100 question_id=%s", rank, completed, ids[index])
    finally:
        search.close()
    return {"branch": "dense", "worker": rank, "completed": 100}


def aggregate_articles(sparse_ids: list[str], dense_ids: list[str], chunks: dict[str, dict[str, Any]], lookup: dict[str, list[dict[str, Any]]], config: Config):
    retrieval = config.raw["retrieval"]
    branches = {"sparse": sparse_ids, "dense": dense_ids}
    weights = {"sparse": retrieval["sparse_weight"], "dense": retrieval["dense_weight"]}
    groups: dict[tuple[str, str], dict[str, Any]] = {}
    individual: dict[str, float] = {}
    individual_best_rank: dict[str, int] = {}
    for branch, ids in branches.items():
        for rank, chunk_id in enumerate(ids, start=1):
            block = lookup[chunk_id]
            key = (chunks[chunk_id]["document_id"], block[0]["chunk_id"])
            group = groups.setdefault(key, {"document_id": key[0], "block_id": key[1], "branch_hits": {"sparse": [], "dense": []}, "chunk_ids": set()})
            group["branch_hits"][branch].append({"chunk_id": chunk_id, "rank": rank})
            group["chunk_ids"].add(chunk_id)
            individual[chunk_id] = individual.get(chunk_id, 0.0) + weights[branch] / (retrieval["rrf_constant"] + rank)
            individual_best_rank[chunk_id] = min(individual_best_rank.get(chunk_id, rank), rank)
    ranked = []
    cap = retrieval["maximum_hits_per_branch_per_article"]
    for group in groups.values():
        contributions = {}
        total = 0.0
        for branch in branches:
            ranks = sorted(item["rank"] for item in group["branch_hits"][branch])[:cap]
            contributions[branch] = sum(weights[branch] / (retrieval["rrf_constant"] + rank) for rank in ranks)
            total += contributions[branch]
        representative = min(group["chunk_ids"], key=lambda chunk_id: (-individual[chunk_id], individual_best_rank[chunk_id], chunk_id))
        ranked.append({
            "document_id": group["document_id"], "block_id": group["block_id"],
            "article_score": total, "branch_contributions": contributions,
            "branch_hits": group["branch_hits"], "representative_chunk_id": representative,
            "representative_chunk_score": individual[representative],
            "candidate_chunk_ids": sorted(group["chunk_ids"]),
        })
    return sorted(ranked, key=lambda item: (-item["article_score"], -item["representative_chunk_score"], item["document_id"], item["block_id"]))


def _load_branch_rows(output: Path, ids: list[str]):
    sparse_state = json.loads((output / "retrieval/sparse/state.json").read_text(encoding="utf-8"))
    sparse_identity = sparse_state["run_identity"]
    if sparse_state.get("complete") is not True or sparse_state.get("completed_count") != 200:
        raise E26Error("Sparse retrieval is incomplete.")
    sparse = []
    for index, question_id in enumerate(ids):
        row = json.loads((output / f"retrieval/sparse/records/{index:04d}.json").read_text(encoding="utf-8"))
        _validate_retrieval_record(row, question_id, index, "sparse", sparse_identity)
        sparse.append(row)
    dense_rows = [None] * 200
    dense_identities = []
    for rank in range(2):
        state = json.loads((output / f"retrieval/dense/worker-{rank}/state.json").read_text(encoding="utf-8"))
        identity = state["run_identity"]
        if state.get("complete") is not True or state.get("completed_count") != 100 or identity.get("assigned_indices") != dense_indices(rank):
            raise E26Error("Dense retrieval is incomplete.")
        dense_identities.append(identity)
        for index in dense_indices(rank):
            row = json.loads((output / f"retrieval/dense/worker-{rank}/records/{index:04d}.json").read_text(encoding="utf-8"))
            _validate_retrieval_record(row, ids[index], index, "dense", identity)
            dense_rows[index] = row
    return sparse, dense_rows, sparse_identity, dense_identities


def prepare_articles(*, root: Path, e00: Path, train: Path, dev: Path, output: Path, config: Config):
    check_preflight(root, output, config)
    _, _, ids = sample(train, dev, config)
    sparse, dense, sparse_identity, dense_identities = _load_branch_rows(output, ids)
    wanted = {chunk_id for row in sparse for chunk_id in row["sparse_top_ids"]}
    wanted.update(chunk_id for row in dense for chunk_id in row["dense_top_ids"])
    metadata = config.e19.contract["metadata_source"]
    chunks = scan_selected(e00 / "chunks.jsonl", "chunk_id", wanted, metadata["chunks_sha256"])
    documents = scan_selected(e00 / "documents.jsonl", "document_id", {item["document_id"] for item in chunks.values()}, metadata["documents_sha256"])
    by_doc = {document_id: [] for document_id in documents}
    digest = hashlib.sha256()
    with (e00 / "chunks.jsonl").open("rb") as stream:
        for line in stream:
            digest.update(line)
            item = json.loads(line)
            if item["document_id"] in by_doc:
                by_doc[item["document_id"]].append(item)
    if digest.hexdigest() != metadata["chunks_sha256"]:
        raise E26Error("E00 changed during article grouping.")
    lookup = {}
    for document_id, values in by_doc.items():
        for block in parent.article_blocks(values, documents[document_id]):
            for item in block:
                if item["chunk_id"] in wanted:
                    lookup[item["chunk_id"]] = block
    if set(lookup) != wanted:
        raise E26Error("Some top-100 candidates have no exact article group.")
    prepared = []
    article_counts, union_counts = [], []
    for index, question_id in enumerate(ids):
        ranking = aggregate_articles(sparse[index]["sparse_top_ids"], dense[index]["dense_top_ids"], chunks, lookup, config)
        if len(ranking) < 10:
            raise E26Error(f"Fewer than ten article candidates: {question_id}")
        units = []
        for article_rank, article in enumerate(ranking[:10]):
            seed = chunks[article["representative_chunk_id"]]
            units.append(parent.seed_unit(seed, article_rank, lookup[seed["chunk_id"]], documents[seed["document_id"]], config.e21.policy))
        prepared.append({
            "question_id": question_id, "sample_index": index, "answers_used": False,
            "candidate_union_count": len(set(sparse[index]["sparse_top_ids"]) | set(dense[index]["dense_top_ids"])),
            "article_candidate_count": len(ranking), "ranking_top10": ranking[:10], "units": units,
        })
        article_counts.append(len(ranking))
        union_counts.append(prepared[-1]["candidate_union_count"])
    path = output / "prepared/results.jsonl"
    _atomic_jsonl(path, prepared)
    summary = {
        "schema_version": "1.0", "experiment_id": EXPERIMENT,
        "sample_size": 200, "sample_ids_sha256": _ids_sha(ids),
        "config_sha256": config.sha, "results_sha256": file_sha256(path),
        "candidate_k_per_branch": 100,
        "mean_candidate_union_count": fmean(union_counts),
        "mean_article_candidate_count": fmean(article_counts),
        "minimum_article_candidate_count": min(article_counts),
        "maximum_article_candidate_count": max(article_counts),
        "sparse_identity_sha256": sparse_identity["identity_sha256"],
        "dense_identity_sha256": [item["identity_sha256"] for item in dense_identities],
        "answers_used": False, "variants": list(VARIANTS),
    }
    _atomic_json(output / "prepared/summary.json", summary)
    return summary


def load_prepared(output: Path, ids: list[str], config: Config):
    path = output / "prepared/results.jsonl"
    summary = json.loads((output / "prepared/summary.json").read_text(encoding="utf-8"))
    rows = _read_jsonl(path)
    if (
        summary.get("experiment_id") != EXPERIMENT or summary.get("config_sha256") != config.sha
        or summary.get("results_sha256") != file_sha256(path) or summary.get("answers_used") is not False
        or [row.get("question_id") for row in rows] != ids
        or any(row.get("sample_index") != index or row.get("answers_used") is not False or [unit.get("rank") for unit in row.get("units", [])] != list(range(10)) for index, row in enumerate(rows))
    ):
        raise E26Error("Prepared hierarchical articles changed.")
    return rows, summary


def validate_generation_record(row, question_id, index, rank, identity):
    if (
        row.get("question_id") != question_id or row.get("sample_index") != index
        or row.get("worker_rank") != rank or row.get("variant") != VARIANTS[rank]
        or row.get("worker_identity_sha256") != identity["identity_sha256"]
        or not isinstance(row.get("answer"), str) or not row["answer"].strip()
        or row.get("record_sha256") != _json_sha256({key: value for key, value in row.items() if key != "record_sha256"})
    ):
        raise E26Error(f"Changed E26 generation record: {rank}/{index}")


def run_variant_worker(*, root: Path, control: Path, training: Path, train: Path, dev: Path, output: Path, config: Config, rank: int, device: str):
    import torch

    if rank not in (0, 1) or device != f"cuda:{rank}":
        raise E26Error("GPU0=top8 all200; GPU1=top10 all200.")
    checked = check_preflight(root, output, config)
    questions, _, ids = sample(train, dev, config)
    _, control_identity = load_control(control, ids, config)
    prepared, prepared_summary = load_prepared(output, ids, config)
    runtime = {name: importlib.metadata.version(name) for name in control_identity["runtime"]}
    if runtime != control_identity["runtime"]:
        raise E26Error("Use exact E21 generation runtime versions.")
    model, tokenizer, placement, parameters = load_candidate_generator(config=config.e19, training_directory=training, device=device)
    adapter_sha, _ = validate_candidate_adapter(training, config.e19)
    generation_sha = _json_sha256(model.generation_config.to_dict())
    if adapter_sha != checked["adapter_sha256"] or generation_sha != control_identity["generation_config_sha256"]:
        raise E26Error("E19 adapter or generation defaults differ from E21.")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    def render(messages):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)

    def count(text):
        return len(tokenizer(text, add_special_tokens=False)["input_ids"])

    article_count = config.raw["variants"][rank]["article_count"]
    identity = {
        "code_sha256": code_sha(root), "config_sha256": config.sha,
        "prepared_sha256": prepared_summary["results_sha256"],
        "control_identity_sha256": control_identity["identity_sha256"],
        "adapter_sha256": adapter_sha, "generation_config_sha256": generation_sha,
        "runtime": runtime, "variant": VARIANTS[rank], "article_count": article_count,
        "worker_rank": rank, "device": device, "assigned_indices": list(range(200)),
        "device_map": placement, "adapter_parameters": parameters,
    }
    identity["identity_sha256"] = _json_sha256(identity)
    folder = output / "evaluation" / VARIANTS[rank]
    records, state = folder / "records", folder / "state.json"
    records.mkdir(parents=True, exist_ok=True)
    assigned = list(range(200))
    done = _load_worker_progress(records=records, state_path=state, identity=identity, assigned_indices=assigned, sample_ids=ids)
    for index in assigned[:done]:
        validate_generation_record(json.loads((records / f"{index:04d}.json").read_text(encoding="utf-8")), ids[index], index, rank, identity)
    for index in assigned[done:]:
        selected = {**prepared[index], "units": prepared[index]["units"][:article_count]}
        messages, spans, diagnostics = parent.pack(
            questions[ids[index]]["question"], selected, config.e21,
            lambda value: count(render(value)), count, parent.VARIANTS[1],
        )
        prompt = render(messages)
        inputs = {key: value.to(device) for key, value in tokenizer(prompt, add_special_tokens=False, return_tensors="pt").items()}
        started = time.perf_counter()
        with torch.inference_mode():
            generated = model.generate(**inputs, do_sample=False, num_beams=1, max_new_tokens=704, use_cache=True)
        latency = (time.perf_counter() - started) * 1000
        new_ids = generated[0, inputs["input_ids"].shape[1]:]
        answer = tokenizer.decode(new_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False).strip()
        if not answer:
            raise E26Error(f"Empty E26 answer: {ids[index]}")
        eos = tokenizer.eos_token_id
        eos_ids = set(eos if isinstance(eos, list) else [eos])
        finish = "eos" if len(new_ids) and int(new_ids[-1]) in eos_ids else "length" if len(new_ids) >= 704 else "other"
        row = {
            "question_id": ids[index], "sample_index": index, "worker_rank": rank,
            "variant": VARIANTS[rank], "worker_identity_sha256": identity["identity_sha256"],
            "answer": answer, "input_tokens": count(prompt), "output_tokens": count(answer),
            "generated_tokens_including_special": len(new_ids), "finish_reason": finish,
            "generation_latency_ms": latency, "selected_context_count": len(spans),
            "selected_article_count": article_count,
            "selected_article_keys": [[unit["document_id"], unit["block_id"]] for unit in selected["units"]],
            "selected_chunk_ids": diagnostics["selected_chunk_ids"], "packing": diagnostics,
            "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "evidence_spans": [{key: value for key, value in item.items() if key != "text"} for item in spans],
        }
        row["record_sha256"] = _json_sha256(row)
        _atomic_json(records / f"{index:04d}.json", row)
        _write_worker_state(state, identity, index + 1, 200)
        LOG.info("e26_generation variant=%s device=%s completed=%d total=200 question_id=%s finish=%s", VARIANTS[rank], device, index + 1, ids[index], finish)
    return {"variant": VARIANTS[rank], "completed": 200}


def finalize(*, root: Path, control: Path, training: Path, train: Path, dev: Path, output: Path, config: Config):
    checked = check_preflight(root, output, config)
    ensure_nltk_resources(download=False)
    questions, _, ids = sample(train, dev, config)
    controls, control_identity = load_control(control, ids, config)
    _, prepared_summary = load_prepared(output, ids, config)
    by_variant = {CONTROL: controls}
    for rank, variant in enumerate(VARIANTS):
        folder = output / "evaluation" / variant
        state = json.loads((folder / "state.json").read_text(encoding="utf-8"))
        identity = state.get("run_identity", {})
        if (
            state.get("complete") is not True or state.get("completed_count") != 200 or state.get("assigned_count") != 200
            or identity.get("worker_rank") != rank or identity.get("device") != f"cuda:{rank}"
            or identity.get("variant") != variant or identity.get("article_count") != config.raw["variants"][rank]["article_count"]
            or identity.get("code_sha256") != code_sha(root) or identity.get("config_sha256") != config.sha
            or identity.get("prepared_sha256") != prepared_summary["results_sha256"]
            or identity.get("control_identity_sha256") != control_identity["identity_sha256"]
            or identity.get("adapter_sha256") != checked["adapter_sha256"]
            or identity.get("runtime") != control_identity["runtime"]
            or identity.get("generation_config_sha256") != control_identity["generation_config_sha256"]
            or identity.get("identity_sha256") != _json_sha256({key: value for key, value in identity.items() if key != "identity_sha256"})
        ):
            raise E26Error("Incomplete or changed E26 generation state.")
        rows = []
        for index, question_id in enumerate(ids):
            row = json.loads((folder / f"records/{index:04d}.json").read_text(encoding="utf-8"))
            validate_generation_record(row, question_id, index, rank, identity)
            rows.append(row)
        _atomic_jsonl(folder / "results.jsonl", rows)
        by_variant[variant] = rows
    scores = {
        variant: [
            {"meteor": nltk_meteor_score(questions[question_id]["answer"], row["answer"]),
             "rouge_l": rouge_l_fmeasure(questions[question_id]["answer"], row["answer"])}
            for question_id, row in zip(ids, rows)
        ] for variant, rows in by_variant.items()
    }

    def paired(left, right):
        delta = [a["meteor"] - b["meteor"] for a, b in zip(scores[left], scores[right])]
        return {
            "meteor_mean": fmean(delta),
            "meteor_bootstrap_95_ci": _bootstrap_ci(delta, seed=f"e26-{left}-{right}", iterations=10000),
            "improved": sum(value > 0 for value in delta),
            "worsened": sum(value < 0 for value in delta),
            "tied": sum(value == 0 for value in delta),
        }

    _atomic_jsonl(output / "per_question_scores.jsonl", [
        {"question_id": question_id, "scores": {variant: scores[variant][index] for variant in scores}}
        for index, question_id in enumerate(ids)
    ])
    metrics = {variant: _metrics(rows, scores[variant]) for variant, rows in by_variant.items()}
    report = {
        "schema_version": "1.0", "experiment_id": EXPERIMENT,
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "sample_size": 200,
        "sample_scope": "same-repeated-e19-e21-dev400-600",
        "control_variant": CONTROL, "metrics": metrics,
        "paired_top8_minus_e21": paired(VARIANTS[0], CONTROL),
        "paired_top10_minus_e21": paired(VARIANTS[1], CONTROL),
        "paired_top10_minus_top8": paired(VARIANTS[1], VARIANTS[0]),
        "retrieval": {
            "candidate_k_per_branch": 100,
            "mean_candidate_union_count": prepared_summary["mean_candidate_union_count"],
            "mean_article_candidate_count": prepared_summary["mean_article_candidate_count"],
            "article_hit_cap_per_branch": 3, "answers_used": False,
        },
        "smoke_leader": max(metrics, key=lambda variant: metrics[variant]["meteor"]),
        "promotion_allowed": False, "public_read": False, "holdout_untouched": True,
        "evidence": {
            "config_sha256": config.sha, "code_sha256": code_sha(root),
            "control_results_sha256": config.raw["control"]["results_sha256"],
            "prepared_results_sha256": prepared_summary["results_sha256"],
            "candidate_results_sha256": {variant: file_sha256(output / f"evaluation/{variant}/results.jsonl") for variant in VARIANTS},
        },
        "warning": "Repeated dev-200; freeze any winner before one final untouched-dev121 check.",
    }
    _atomic_json(output / "report.json", report)
    return report
