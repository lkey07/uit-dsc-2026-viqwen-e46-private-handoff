"""Conservative text cleaning for official Vietnamese legal passages."""

from __future__ import annotations

import html
import re
import unicodedata
from dataclasses import dataclass


_TAG_HINT = re.compile(
    r"<\s*/?\s*(?:br|p|div|span|table|thead|tbody|tr|td|th|strong|em|b|i|img|script|style)\b",
    re.IGNORECASE,
)
_SCRIPT_STYLE = re.compile(
    r"<\s*(?P<tag>script|style)\b[^>]*>.*?<\s*/\s*(?P=tag)\s*>",
    re.IGNORECASE | re.DOTALL,
)
_IMAGE_TAG = re.compile(r"<\s*img\b(?P<attrs>[^>]*)/?>", re.IGNORECASE)
_ALT_ATTRIBUTE = re.compile(
    r"\balt\s*=\s*(?:\"(?P<double>[^\"]*)\"|'(?P<single>[^']*)'|(?P<bare>[^\s>]+))",
    re.IGNORECASE,
)
_BLOCK_TAG = re.compile(
    r"<\s*/?\s*(?:br|p|div|li|table|thead|tbody|tr|h[1-6])\b[^>]*>",
    re.IGNORECASE,
)
_INLINE_TAG = re.compile(
    r"<\s*/?\s*(?:span|td|th|strong|em|b|i|u|font|a)\b[^>]*>",
    re.IGNORECASE,
)
_NOTICE = re.compile(
    r"Bạn\s+phải\s+đăng\s+nhập\s+hoặc\s+đăng\s+ký\s+Thành\s+Viên\s+TVPL\s+Pro\s+"
    r"để\s+sử\s+dụng\s+được\s+đầy\s+đủ\s+các\s+tiện\s+ích\s+gia\s+tăng\s+liên\s+quan\s+"
    r"đến\s+nội\s+dung\s+TCVN\.\s*Mọi\s+chi\s+tiết\s+xin\s+liên\s+hệ:\s*"
    r"ĐT:\s*\(028\)\s*3930\s*3279\s*DĐ:\s*0906\s*22\s*99\s*66",
    re.IGNORECASE,
)
_BLANK_LINES = re.compile(r"\n{3,}")
_INLINE_SPACES = re.compile(r"[ \t\v\f]+")
_UNSAFE_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_ZERO_WIDTH = dict.fromkeys(map(ord, "\ufeff\u200b\u200c\u200d\u2060"), None)


@dataclass(frozen=True)
class CleaningResult:
    """Cleaned text and auditable transformation counts."""

    text: str
    html_transform_count: int
    notice_removal_count: int
    non_nfc_input: bool
    crlf_input: bool


class ConservativePassageCleaner:
    """Normalize presentation noise while preserving visible legal content."""

    version = "fixed-rag-conservative-v1"

    def clean(self, text: str) -> CleaningResult:
        """Return deterministic NFC text without legal-content rewriting."""

        if not isinstance(text, str):
            raise TypeError("Passage text must be a string.")
        crlf_input = "\r" in text
        non_nfc_input = not unicodedata.is_normalized("NFC", text)
        normalized = text.replace("\r\n", "\n").replace("\r", "\n")
        normalized = unicodedata.normalize("NFC", normalized)
        normalized = normalized.translate(_ZERO_WIDTH).replace("\u00a0", " ")
        normalized = _UNSAFE_CONTROL.sub(" ", normalized)

        html_count = 0
        if _TAG_HINT.search(normalized):
            normalized, count = _SCRIPT_STYLE.subn("", normalized)
            html_count += count
            normalized, count = _IMAGE_TAG.subn(_visible_image_text, normalized)
            html_count += count
            normalized, count = _BLOCK_TAG.subn("\n", normalized)
            html_count += count
            normalized, count = _INLINE_TAG.subn("", normalized)
            html_count += count

        normalized, notice_count = _NOTICE.subn("\n", normalized)
        lines = [_INLINE_SPACES.sub(" ", line).strip() for line in normalized.split("\n")]
        normalized = "\n".join(lines)
        normalized = _BLANK_LINES.sub("\n\n", normalized).strip()
        normalized = html.unescape(normalized)
        return CleaningResult(
            text=normalized,
            html_transform_count=html_count,
            notice_removal_count=notice_count,
            non_nfc_input=non_nfc_input,
            crlf_input=crlf_input,
        )


def _visible_image_text(match: re.Match[str]) -> str:
    """Keep only organizer-provided visible alt text from presentation images."""

    alt = _ALT_ATTRIBUTE.search(match.group("attrs"))
    if alt is None:
        return ""
    return alt.group("double") or alt.group("single") or alt.group("bare") or ""
