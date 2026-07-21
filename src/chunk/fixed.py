"""Fixed-size token chunking - the deliberately naive Phase 1 splitter.

We concatenate each article's title + summary + body, tokenize with the SAME
tokenizer the embedding model uses, then slide a fixed window (size_tokens) with
a fixed overlap (overlap_tokens), ignoring the article's structure. This can cut
mid-sentence - that's the point: it's the baseline that structure-aware chunking
(Phase 3) must beat, measured.

Flat chunks, metadata = source_url only (per the Phase 1 spec). Output:
data/processed/chunks.jsonl with {chunk_id, source_url, text}.
"""
from __future__ import annotations

import json
from pathlib import Path

from transformers import AutoTokenizer

from src.config import CHUNKS_PATH, Config
from src.ingest.clean import read_articles


def _window(token_ids: list[int], size: int, overlap: int):
    """Yield (start, slice) windows of `size` ids stepping by size-overlap."""
    stride = max(1, size - overlap)
    start = 0
    n = len(token_ids)
    while start < n:
        yield token_ids[start : start + size]
        if start + size >= n:  # last window reached the end
            break
        start += stride


def chunk_all(cfg: Config) -> list[dict]:
    articles = read_articles()
    tokenizer = AutoTokenizer.from_pretrained(cfg.embedding.model_name)
    size = cfg.chunk.size_tokens
    overlap = cfg.chunk.overlap_tokens

    chunks: list[dict] = []
    for art in articles:
        full_text = "\n".join(
            p for p in (art["title"], art["summary"], art["body_text"]) if p
        )
        ids = tokenizer.encode(full_text, add_special_tokens=False)
        for i, win in enumerate(_window(ids, size, overlap)):
            text = tokenizer.decode(win, skip_special_tokens=True).strip()
            if not text:
                continue
            chunks.append(
                {
                    "chunk_id": f"{art['slug']}::{i}",
                    "source_url": art["source_url"],
                    "text": text,
                }
            )

    CHUNKS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
        for ch in chunks:
            f.write(json.dumps(ch, ensure_ascii=False) + "\n")
    return chunks


def read_chunks(path: Path = CHUNKS_PATH) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
