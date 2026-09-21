"""Offline validation for the competition model inventory."""

from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path
from typing import Any


class ModelInventoryError(ValueError):
    """Raised when a model inventory violates a competition constraint."""


_REVISION_PATTERN = re.compile(r"^[0-9a-f]{40}$")
_REQUIRED_ROLES = {"embedding", "reranker", "generator"}


def file_sha256(path: Path) -> str:
    """Return the lowercase SHA-256 digest for a file."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_inventory(path: Path) -> dict[str, Any]:
    """Load a UTF-8 JSON inventory and require an object root."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ModelInventoryError("Inventory root must be a JSON object.")
    return payload


def registered_model_urls(csv_path: Path) -> set[str]:
    """Read model URLs from the organizer-provided registration reference."""

    with csv_path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        link_field = next(
            (field for field in (reader.fieldnames or []) if field.casefold() == "link"),
            None,
        )
        if link_field is None:
            raise ModelInventoryError("Registration reference has no Link column.")
        return {
            row[link_field].strip().rstrip("/")
            for row in reader
            if row.get(link_field, "").strip()
        }


def validate_inventory(
    inventory: dict[str, Any],
    registration_csv: Path,
) -> list[str]:
    """Validate identity, registration evidence and aggregate parameter budgets."""

    errors: list[str] = []
    limit = inventory.get("parameter_limit_exclusive")
    if not isinstance(limit, int) or limit <= 0:
        errors.append("parameter_limit_exclusive must be a positive integer")
        limit = 0

    reference = inventory.get("registration_reference", {})
    expected_hash = reference.get("sha256")
    actual_hash = file_sha256(registration_csv)
    if expected_hash != actual_hash:
        errors.append(
            f"registration CSV checksum mismatch: expected {expected_hash}, got {actual_hash}"
        )

    try:
        allowed_urls = registered_model_urls(registration_csv)
    except ModelInventoryError as exc:
        errors.append(str(exc))
        allowed_urls = set()

    models = inventory.get("models")
    if not isinstance(models, list) or not models:
        return errors + ["models must be a non-empty list"]

    by_key: dict[str, dict[str, Any]] = {}
    for index, raw_model in enumerate(models):
        if not isinstance(raw_model, dict):
            errors.append(f"models[{index}] must be an object")
            continue
        key = raw_model.get("key")
        if not isinstance(key, str) or not key:
            errors.append(f"models[{index}] has no key")
            continue
        if key in by_key:
            errors.append(f"duplicate model key: {key}")
        by_key[key] = raw_model

        role = raw_model.get("role")
        if role not in _REQUIRED_ROLES:
            errors.append(f"{key}: unsupported role {role!r}")
        revision = raw_model.get("revision")
        if not isinstance(revision, str) or not _REVISION_PATTERN.fullmatch(revision):
            errors.append(f"{key}: revision must be a lowercase 40-character commit hash")
        count = raw_model.get("parameter_count")
        if not isinstance(count, int) or count <= 0:
            errors.append(f"{key}: parameter_count must be a positive integer")
        if not raw_model.get("license"):
            errors.append(f"{key}: license is required")
        url = raw_model.get("url")
        if not isinstance(url, str) or url.rstrip("/") not in allowed_urls:
            errors.append(f"{key}: URL is absent from the registration reference")

    stacks = inventory.get("candidate_stacks")
    if not isinstance(stacks, list) or not stacks:
        return errors + ["candidate_stacks must be a non-empty list"]

    seen_stack_ids: set[str] = set()
    for index, raw_stack in enumerate(stacks):
        if not isinstance(raw_stack, dict):
            errors.append(f"candidate_stacks[{index}] must be an object")
            continue
        stack_id = raw_stack.get("stack_id")
        if not isinstance(stack_id, str) or not stack_id:
            errors.append(f"candidate_stacks[{index}] has no stack_id")
            continue
        if stack_id in seen_stack_ids:
            errors.append(f"duplicate stack_id: {stack_id}")
        seen_stack_ids.add(stack_id)

        keys = raw_stack.get("model_keys")
        if not isinstance(keys, list) or not all(isinstance(key, str) for key in keys):
            errors.append(f"{stack_id}: model_keys must be a list of strings")
            continue
        if len(keys) != len(set(keys)):
            errors.append(f"{stack_id}: model_keys must be unique")
            continue
        missing = [key for key in keys if key not in by_key]
        if missing:
            errors.append(f"{stack_id}: unknown model keys {missing}")
            continue
        roles = [by_key[key].get("role") for key in keys]
        if set(roles) != _REQUIRED_ROLES or len(roles) != len(_REQUIRED_ROLES):
            errors.append(f"{stack_id}: must contain exactly one model for each required role")
        total = sum(
            count if isinstance(count := by_key[key].get("parameter_count"), int) else 0
            for key in keys
        )
        if raw_stack.get("total_parameter_count") != total:
            errors.append(f"{stack_id}: stored total does not equal computed total {total}")
        headroom = limit - total
        if raw_stack.get("parameter_headroom") != headroom:
            errors.append(f"{stack_id}: stored headroom does not equal computed value {headroom}")
        if limit and total >= limit:
            errors.append(f"{stack_id}: {total} parameters violates exclusive limit {limit}")

    return errors


def require_valid_inventory(
    inventory_path: Path,
    registration_csv: Path,
) -> dict[str, Any]:
    """Load and validate an inventory, raising one actionable error on failure."""

    inventory = load_inventory(inventory_path)
    errors = validate_inventory(inventory, registration_csv)
    if errors:
        formatted = "\n".join(f"- {error}" for error in errors)
        raise ModelInventoryError(f"Model inventory is invalid:\n{formatted}")
    return inventory
