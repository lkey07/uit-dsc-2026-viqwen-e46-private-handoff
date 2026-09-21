"""Leakage-safe deterministic splitting for official answer supervision."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from .normalization import normalize_question

JsonMapping = dict[str, dict[str, Any]]


class SplitValidationError(ValueError):
    """Raised when official data or split configuration violates its contract."""


@dataclass(frozen=True)
class SplitPolicy:
    """Validated deterministic split policy."""

    seed: str
    bucket_count: int
    train_buckets: int
    dev_buckets: int
    holdout_buckets: int

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "SplitPolicy":
        """Build and validate a policy from a JSON-compatible mapping."""

        policy = cls(
            seed=str(payload["seed"]),
            bucket_count=int(payload["bucket_count"]),
            train_buckets=int(payload["train_buckets"]),
            dev_buckets=int(payload["dev_buckets"]),
            holdout_buckets=int(payload["holdout_buckets"]),
        )
        if not policy.seed:
            raise SplitValidationError("split seed must not be empty")
        if policy.bucket_count <= 0:
            raise SplitValidationError("bucket_count must be positive")
        total = policy.train_buckets + policy.dev_buckets + policy.holdout_buckets
        if total != policy.bucket_count:
            raise SplitValidationError(
                "train/dev/holdout buckets must sum to bucket_count"
            )
        if min(policy.train_buckets, policy.dev_buckets, policy.holdout_buckets) <= 0:
            raise SplitValidationError("every split must receive at least one bucket")
        return policy

    def assign(self, normalized_question: str) -> str:
        """Assign a normalized question to train, dev, or holdout."""

        digest = hashlib.sha256(
            f"{self.seed}\0{normalized_question}".encode("utf-8")
        ).digest()
        bucket = int.from_bytes(digest[:8], "big") % self.bucket_count
        if bucket < self.train_buckets:
            return "train"
        if bucket < self.train_buckets + self.dev_buckets:
            return "dev"
        return "holdout"


def sha256_file(path: Path) -> str:
    """Return the lowercase SHA-256 digest of a file."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_qa_mapping(path: Path, *, require_answer: bool) -> JsonMapping:
    """Load and validate an official question-ID mapping."""

    with path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict) or not payload:
        raise SplitValidationError(f"{path.name} must be a non-empty JSON object")

    validated: JsonMapping = {}
    for raw_id, raw_record in payload.items():
        question_id = str(raw_id)
        if not isinstance(raw_record, dict):
            raise SplitValidationError(f"record {question_id} must be an object")
        question = raw_record.get("question")
        if not isinstance(question, str) or not question.strip():
            raise SplitValidationError(f"record {question_id} has invalid question")
        answer = raw_record.get("answer")
        if require_answer and not isinstance(answer, str):
            raise SplitValidationError(f"record {question_id} has invalid answer")
        if not require_answer and answer is not None:
            raise SplitValidationError(
                f"public record {question_id} unexpectedly contains an answer"
            )
        validated[question_id] = {"question": question, "answer": answer}
    return validated


def _normalized_index(records: Mapping[str, Mapping[str, Any]]) -> dict[str, list[str]]:
    index: dict[str, list[str]] = {}
    for question_id, record in records.items():
        key = normalize_question(str(record["question"]))
        index.setdefault(key, []).append(question_id)
    return index


def _answer_key(answer: str) -> str:
    return normalize_question(answer)


def _id_sort_key(question_id: str) -> tuple[int, int | str]:
    if question_id.isdecimal():
        return (0, int(question_id))
    return (1, question_id)


def _sorted_mapping(records: Mapping[str, dict[str, Any]]) -> JsonMapping:
    return {key: records[key] for key in sorted(records, key=_id_sort_key)}


def build_leakage_safe_splits(
    train_records: JsonMapping,
    warmup_records: JsonMapping,
    public_records: JsonMapping,
    policy: SplitPolicy,
) -> dict[str, Any]:
    """Create split payloads while quarantining public-question overlap.

    All original train IDs are preserved except records whose normalized
    question appears in public. Warm-up questions identical to train are
    recorded as aliases and excluded to avoid duplicate weighting. Unique
    warm-up records are eligible for the deterministic split.
    """

    train_index = _normalized_index(train_records)
    warmup_index = _normalized_index(warmup_records)
    public_index = _normalized_index(public_records)
    public_questions = set(public_index)

    split_records: dict[str, JsonMapping] = {
        "train": {},
        "dev": {},
        "holdout": {},
    }
    lineage: dict[str, dict[str, str]] = {}
    quarantined: list[dict[str, Any]] = []
    warmup_aliases: list[dict[str, Any]] = []

    def add_record(source: str, question_id: str, record: dict[str, Any]) -> None:
        normalized = normalize_question(str(record["question"]))
        if normalized in public_questions:
            quarantined.append(
                {
                    "source": source,
                    "question_id": question_id,
                    "public_ids": sorted(public_index[normalized], key=_id_sort_key),
                    "reason": "normalized_question_overlaps_public",
                }
            )
            return
        split_name = policy.assign(normalized)
        if question_id in lineage:
            raise SplitValidationError(
                f"eligible question ID collision for {question_id}: "
                f"{lineage[question_id]['source']} and {source}"
            )
        split_records[split_name][question_id] = {
            "question": record["question"],
            "answer": record["answer"],
        }
        lineage[question_id] = {
            "source": source,
            "split": split_name,
            "normalized_question_sha256": hashlib.sha256(
                normalized.encode("utf-8")
            ).hexdigest(),
        }

    for question_id in sorted(train_records, key=_id_sort_key):
        add_record("train", question_id, train_records[question_id])

    for question_id in sorted(warmup_records, key=_id_sort_key):
        record = warmup_records[question_id]
        normalized = normalize_question(str(record["question"]))
        if normalized in public_questions:
            add_record("warmup", question_id, record)
            continue
        if normalized in train_index:
            train_ids = sorted(train_index[normalized], key=_id_sort_key)
            train_answer_keys = {
                _answer_key(str(train_records[train_id]["answer"]))
                for train_id in train_ids
            }
            answer_matches = _answer_key(str(record["answer"])) in train_answer_keys
            if not answer_matches:
                raise SplitValidationError(
                    "warmup/train duplicate question has conflicting answers: "
                    f"warmup ID {question_id}, train IDs {train_ids}"
                )
            warmup_aliases.append(
                {
                    "warmup_id": question_id,
                    "train_ids": train_ids,
                    "answer_match": True,
                    "reason": "exact_normalized_question_and_answer_duplicate",
                }
            )
            continue
        add_record("warmup", question_id, record)

    normalized_split_owner: dict[str, str] = {}
    for split_name, records in split_records.items():
        for record in records.values():
            normalized = normalize_question(str(record["question"]))
            previous = normalized_split_owner.setdefault(normalized, split_name)
            if previous != split_name:
                raise SplitValidationError(
                    f"normalized question leaked between {previous} and {split_name}"
                )

    sorted_splits = {
        split_name: _sorted_mapping(records)
        for split_name, records in split_records.items()
    }
    duplicate_train_groups = [
        question_ids for question_ids in train_index.values() if len(question_ids) > 1
    ]
    conflicting_train_groups = [
        question_ids
        for question_ids in duplicate_train_groups
        if len(
            {
                _answer_key(str(train_records[question_id]["answer"]))
                for question_id in question_ids
            }
        )
        > 1
    ]
    return {
        "splits": sorted_splits,
        "lineage": {key: lineage[key] for key in sorted(lineage, key=_id_sort_key)},
        "quarantined_public_overlap": quarantined,
        "warmup_train_aliases": warmup_aliases,
        "audit": {
            "train_record_count": len(train_records),
            "train_normalized_group_count": len(train_index),
            "train_duplicate_group_count": sum(
                len(question_ids) > 1 for question_ids in train_index.values()
            ),
            "train_duplicate_extra_record_count": sum(
                len(question_ids) - 1 for question_ids in duplicate_train_groups
            ),
            "train_duplicate_answer_conflict_group_count": len(
                conflicting_train_groups
            ),
            "warmup_record_count": len(warmup_records),
            "warmup_normalized_group_count": len(warmup_index),
            "public_record_count": len(public_records),
            "public_normalized_group_count": len(public_index),
            "train_warmup_overlap_group_count": len(set(train_index) & set(warmup_index)),
            "train_public_overlap_group_count": len(set(train_index) & set(public_index)),
            "warmup_public_overlap_group_count": len(set(warmup_index) & set(public_index)),
            "warmup_unique_group_count": len(
                set(warmup_index) - set(train_index) - set(public_index)
            ),
        },
    }


def _write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    os.replace(temporary, path)


def write_split_artifacts(
    output_dir: Path,
    result: Mapping[str, Any],
    *,
    policy_payload: Mapping[str, Any],
    input_paths: Mapping[str, Path],
    force: bool = False,
) -> dict[str, Any]:
    """Persist split artifacts and return their manifest."""

    filenames = {
        "train": "train.json",
        "dev": "dev.json",
        "holdout": "holdout.json",
        "lineage": "lineage.json",
        "quarantined_public_overlap": "quarantined-public-overlap.json",
        "warmup_train_aliases": "warmup-train-aliases.json",
    }
    existing = [output_dir / filename for filename in filenames.values()]
    existing.append(output_dir / "manifest.json")
    conflicts = [str(path) for path in existing if path.exists()]
    if conflicts and not force:
        raise FileExistsError(
            "split artifacts already exist; pass --force to replace: " + ", ".join(conflicts)
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    for split_name, records in result["splits"].items():
        _write_json_atomic(output_dir / filenames[split_name], records)
    _write_json_atomic(output_dir / filenames["lineage"], result["lineage"])
    _write_json_atomic(
        output_dir / filenames["quarantined_public_overlap"],
        result["quarantined_public_overlap"],
    )
    _write_json_atomic(
        output_dir / filenames["warmup_train_aliases"],
        result["warmup_train_aliases"],
    )

    output_files = {
        name: {
            "filename": filename,
            "record_count": len(
                result["splits"].get(name, result.get(name, result.get("lineage", {})))
            ),
            "sha256": sha256_file(output_dir / filename),
        }
        for name, filename in filenames.items()
    }
    manifest = {
        "schema_version": "1.0.0",
        "artifact_type": "leakage_safe_answer_supervision_split",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "normalization": "NFC+casefold+whitespace",
        "policy": dict(policy_payload),
        "inputs": {
            name: {
                "filename": path.name,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for name, path in input_paths.items()
        },
        "audit": result["audit"],
        "split_record_counts": {
            name: len(records) for name, records in result["splits"].items()
        },
        "split_group_counts": {
            name: len(
                {
                    normalize_question(str(record["question"]))
                    for record in records.values()
                }
            )
            for name, records in result["splits"].items()
        },
        "quarantined_public_overlap_count": len(
            result["quarantined_public_overlap"]
        ),
        "warmup_train_alias_count": len(result["warmup_train_aliases"]),
        "outputs": output_files,
    }
    _write_json_atomic(output_dir / "manifest.json", manifest)
    return manifest
