"""Search-only Vietnamese accent folding for the E01 secondary branches."""

from __future__ import annotations

import re
import unicodedata


_WHITESPACE = re.compile(r"\s+")
_WORD = re.compile(r"[^\W_]+", re.UNICODE)


def fold_vietnamese_accents(text: str) -> str:
    """Casefold accents and stroked-d while preserving digits and negation text."""

    if not isinstance(text, str):
        raise TypeError("Accent-fold input must be text.")
    normalized = unicodedata.normalize("NFC", text).casefold()
    normalized = normalized.replace("đ", "d")
    decomposed = unicodedata.normalize("NFD", normalized)
    without_marks = "".join(
        character
        for character in decomposed
        if unicodedata.category(character) != "Mn"
    )
    return _WHITESPACE.sub(" ", without_marks).strip()


def folded_query_tokens(text: str) -> list[str]:
    """Return unique folded Unicode word tokens in source order."""

    folded = fold_vietnamese_accents(text)
    seen: set[str] = set()
    result: list[str] = []
    for match in _WORD.finditer(folded):
        token = match.group(0)
        if token not in seen:
            seen.add(token)
            result.append(token)
    return result


def select_trigram_query_terms(
    text: str,
    *,
    minimum_characters: int,
    maximum_terms: int,
) -> list[str]:
    """Select the longest distinctive query terms without a stopword rewrite."""

    if minimum_characters < 3:
        raise ValueError("Trigram query terms must contain at least three characters.")
    if maximum_terms <= 0:
        raise ValueError("maximum_terms must be positive.")
    eligible = [
        (index, token)
        for index, token in enumerate(folded_query_tokens(text))
        if len(token) >= minimum_characters
    ]
    selected = sorted(eligible, key=lambda item: (-len(item[1]), item[0]))[:maximum_terms]
    return [token for _, token in sorted(selected, key=lambda item: item[0])]
