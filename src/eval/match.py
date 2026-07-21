"""Chunking-independent match rule for retrieval scoring.

Ground truth is `source_url` + verbatim `passage_text` - never a chunk ID,
because chunk boundaries change in Phase 3. We judge whether a retrieved chunk
"contains" a supporting passage after normalising BOTH sides, so cosmetic
differences (whitespace, punctuation, HTML entities, the embedding tokenizer's
lowercasing/spacing) don't produce false misses that look like retrieval bugs.

The threshold is a named constant and is recorded in every results row.
"""
from __future__ import annotations

import html
import re

# A chunk "contains" a passage if the normalised passage is a substring of the
# normalised chunk, OR at least this fraction of the passage's tokens appear in
# the chunk (tolerates a passage split across a chunk boundary).
MATCH_TOKEN_OVERLAP_THRESHOLD = 0.9

_PUNCT_RE = re.compile(r"[^\w\s]", flags=re.UNICODE)
_WS_RE = re.compile(r"\s+")


def normalize(text: str) -> str:
    """Lowercase, decode HTML entities, strip punctuation, collapse whitespace."""
    text = html.unescape(text)
    text = text.lower()
    text = _PUNCT_RE.sub(" ", text)
    text = _WS_RE.sub(" ", text)
    return text.strip()


def chunk_contains_passage(chunk_text: str, passage_text: str) -> bool:
    """True if the chunk supports the passage under the normalised match rule."""
    norm_chunk = normalize(chunk_text)
    norm_passage = normalize(passage_text)
    if not norm_passage:
        return False
    if norm_passage in norm_chunk:
        return True
    passage_tokens = norm_passage.split()
    chunk_tokens = set(norm_chunk.split())
    present = sum(1 for t in passage_tokens if t in chunk_tokens)
    return (present / len(passage_tokens)) >= MATCH_TOKEN_OVERLAP_THRESHOLD


def chunk_contains_any(chunk_text: str, passages: list[dict]) -> bool:
    """True if the chunk contains any of the supporting passages."""
    return any(
        chunk_contains_passage(chunk_text, p["passage_text"]) for p in passages
    )
