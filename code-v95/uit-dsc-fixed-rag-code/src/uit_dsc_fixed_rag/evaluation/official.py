"""Parity implementation of the official UIT DSC 2026 LegalQA scorer."""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from statistics import fmean
from typing import Any

_NON_ALPHANUM_RE = re.compile(r"[^a-z0-9]+")
_SPACES_RE = re.compile(r"\s+")
_VALID_TOKEN_RE = re.compile(r"^[a-z0-9]+$")


class ScoringContractError(ValueError):
    """Raised when a prediction or reference violates scorer schema."""


def official_rouge_tokens(text: str) -> list[str]:
    """Tokenize exactly like the scorer's vendored ROUGE implementation."""

    normalized = _NON_ALPHANUM_RE.sub(" ", str(text).lower())
    return [token for token in _SPACES_RE.split(normalized) if _VALID_TOKEN_RE.match(token)]


def rouge_l_fmeasure(reference: str, prediction: str) -> float:
    """Compute the official ASCII-tokenized ROUGE-L F-measure."""

    target = official_rouge_tokens(reference)
    candidate = official_rouge_tokens(prediction)
    if not target or not candidate:
        return 0.0

    previous = [0] * (len(candidate) + 1)
    for target_token in target:
        current = [0]
        for index, candidate_token in enumerate(candidate, start=1):
            if target_token == candidate_token:
                current.append(previous[index - 1] + 1)
            else:
                current.append(max(previous[index], current[-1]))
        previous = current
    lcs_length = previous[-1]
    precision = lcs_length / len(candidate)
    recall = lcs_length / len(target)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def nltk_meteor_score(reference: str, prediction: str) -> float:
    """Compute METEOR using the exact official NLTK whitespace-token call."""

    try:
        from nltk.translate.meteor_score import meteor_score
    except ImportError as exc:
        raise RuntimeError(
            "NLTK 3.7 is required for official METEOR; install "
            "requirements-evaluation.txt"
        ) from exc
    return float(meteor_score([str(reference).split()], str(prediction).split()))


def ensure_nltk_resources(*, download: bool = False) -> None:
    """Validate WordNet/OMW resources, optionally downloading official resources."""

    try:
        import nltk
    except ImportError as exc:
        raise RuntimeError("NLTK 3.7 is not installed") from exc

    missing: list[str] = []
    for resource in ("wordnet", "omw-1.4"):
        candidates = (f"corpora/{resource}", f"corpora/{resource}.zip")
        if not any(_nltk_resource_exists(nltk, candidate) for candidate in candidates):
            missing.append(resource)
    if missing and download:
        for resource in missing:
            if not nltk.download(resource, quiet=False):
                raise RuntimeError(f"failed to download NLTK resource {resource}")
        missing = []
        for resource in ("wordnet", "omw-1.4"):
            candidates = (f"corpora/{resource}", f"corpora/{resource}.zip")
            if not any(_nltk_resource_exists(nltk, candidate) for candidate in candidates):
                missing.append(resource)
    if missing:
        raise RuntimeError(
            "missing official NLTK resources: "
            + ", ".join(missing)
            + "; rerun with --download-nltk-data"
        )


def _nltk_resource_exists(nltk_module: Any, resource_path: str) -> bool:
    try:
        nltk_module.data.find(resource_path)
    except LookupError:
        return False
    return True


def _reference_answers(references: Mapping[str, Any]) -> dict[str, str]:
    answers: dict[str, str] = {}
    for raw_id, value in references.items():
        question_id = str(raw_id)
        answer = value.get("answer") if isinstance(value, Mapping) else value
        if not isinstance(answer, str):
            raise ScoringContractError(
                f"reference {question_id} must contain a string answer"
            )
        answers[question_id] = answer
    return answers


def _prediction_answers(predictions: Mapping[str, Any]) -> dict[str, str]:
    answers: dict[str, str] = {}
    for raw_id, value in predictions.items():
        question_id = str(raw_id)
        if not isinstance(value, Mapping) or set(value) != {"answer"}:
            raise ScoringContractError(
                f"prediction {question_id} must be exactly {{'answer': string}}"
            )
        answer = value["answer"]
        if not isinstance(answer, str):
            raise ScoringContractError(
                f"prediction {question_id} answer must be a string"
            )
        answers[question_id] = answer
    return answers


def score_prediction_mapping(
    predictions: Mapping[str, Any],
    references: Mapping[str, Any],
    *,
    meteor_function: Callable[[str, str], float] | None = None,
) -> dict[str, float]:
    """Score a complete prediction mapping with official macro aggregation."""

    predicted_answers = _prediction_answers(predictions)
    reference_answers = _reference_answers(references)
    predicted_ids = set(predicted_answers)
    reference_ids = set(reference_answers)
    if predicted_ids != reference_ids:
        missing = sorted(reference_ids - predicted_ids)
        extra = sorted(predicted_ids - reference_ids)
        raise ScoringContractError(
            f"prediction IDs must exactly match references; missing={missing[:5]}, "
            f"extra={extra[:5]}"
        )
    if not reference_ids:
        raise ScoringContractError("cannot score an empty reference mapping")

    meteor = meteor_function or nltk_meteor_score
    ordered_ids = list(predicted_answers)
    rouge_scores = [
        rouge_l_fmeasure(reference_answers[qid], predicted_answers[qid])
        for qid in ordered_ids
    ]
    meteor_scores = [
        meteor(reference_answers[qid], predicted_answers[qid]) for qid in ordered_ids
    ]
    return {"rouge": fmean(rouge_scores), "meteor": fmean(meteor_scores)}

