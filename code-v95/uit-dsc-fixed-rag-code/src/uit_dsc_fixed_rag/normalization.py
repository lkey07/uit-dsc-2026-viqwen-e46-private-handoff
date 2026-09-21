"""Text normalization used exclusively for leakage grouping."""

from __future__ import annotations

import re
import unicodedata

_WHITESPACE_RE = re.compile(r"\s+")


def normalize_question(question: str) -> str:
    """Return the deterministic key used to group equivalent questions.

    This normalization is intentionally conservative. It does not remove
    Vietnamese accents, punctuation, digits, negation, or legal references.
    """

    if not isinstance(question, str):
        raise TypeError("question must be a string")
    normalized = unicodedata.normalize("NFC", question).casefold()
    return _WHITESPACE_RE.sub(" ", normalized).strip()

