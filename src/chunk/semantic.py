"""Semantic chunking - Phase 3 Experiment 2c (the logged future-work chunker).

THE IDEA (why this exists)
--------------------------
Fixed chunking (fixed.py) cuts every 512 tokens regardless of meaning, so a chunk
boundary can land in the middle of a sentence or a clinical statement. Semantic
chunking instead puts boundaries *where the topic actually changes*: it measures how
similar each sentence is to the next, and cuts at the points where that similarity
DROPS - i.e. where the text stops talking about one thing and starts talking about
another. The goal is chunks that are each internally coherent (one topic per chunk),
which in theory gives cleaner embeddings and better retrieval.

This is the "percentile breakpoint" method (popularised by Greg Kamradt / LlamaIndex).
On THIS corpus it is a *hypothesis*, not a guaranteed win - our two other structural
chunkers went 1-1 (header-aware lost, small-to-big won), so we MEASURE it against the
frozen baseline like everything else.

THE ALGORITHM, in five steps
----------------------------
  1. Split each article into sentences.
  2. Embed each sentence (with a small "buffer" of neighbours - see _buffered - so a
     single short sentence doesn't create a noisy, unstable vector).
  3. Compute the cosine DISTANCE (1 - cosine similarity) between each pair of
     consecutive sentences. A big distance = the topic shifted between them.
  4. Pick a threshold = the Nth percentile of those distances (default 90th). Every
     gap whose distance exceeds the threshold is a "breakpoint" - cut there. Using a
     percentile (not an absolute distance) auto-adapts to each article's own spread.
  5. Group the sentences between breakpoints into chunks - then enforce a HARD token
     cap (size_tokens) so a very long coherent run can't exceed the embedder's
     512-token capacity (that's the one guardrail fixed.py's window gave us for free).

Design choices kept honest as a one-variable experiment vs the fixed baseline:
  * Same content (title + summary + body) and same embedding model/tokenizer as
    fixed.py - only the SPLITTING RULE changes.
  * Flat output {chunk_id, source_url, text} to CHUNKS_PATH, exactly like fixed.py, so
    the indexer, retriever, and eval need ZERO changes (this is a "flat" strategy, not
    a parent/child one like small_to_big).
  * The size cap reuses fixed.py's _window with the baseline overlap, so any forced
    sub-split still carries overlap protection at its boundaries.
"""
from __future__ import annotations

import json
import re

import numpy as np
from transformers import AutoTokenizer

from src.config import CHUNKS_PATH, Config
from src.chunk.fixed import _window  # reuse the baseline windower for the size cap
from src.index.build import get_model  # reuse the cached BGE embedder
from src.ingest.clean import read_articles

# A deliberately simple sentence splitter: break after . ! ? or a newline, when
# followed by whitespace + a capital/opening char. It is NOT linguistically perfect
# (it can mis-split abbreviations like "e.g."), but the downstream size cap makes the
# pipeline robust to occasional over-/under-splits, and keeping it dependency-free
# avoids pulling in nltk/spacy just to chunk ~143 short articles. Documented limitation.
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\"'])|\n+")

# Buffer = how many neighbours on EACH side to concatenate with a sentence before
# embedding it. Embedding "sentence i alone" is noisy for short sentences; embedding
# "sentence i-1 + i + i+1" gives a more stable sense of what position i is *about*,
# so the distance signal reflects real topic shifts, not sentence-length quirks.
_BUFFER_SIZE = 1


def _split_sentences(text: str) -> list[str]:
    """Split article text into non-empty, stripped sentences."""
    parts = _SENTENCE_SPLIT.split(text)
    return [s.strip() for s in parts if s and s.strip()]


def _buffered(sentences: list[str], buffer: int) -> list[str]:
    """Return, for each sentence, that sentence joined with `buffer` neighbours each side.

    Example (buffer=1): position 3's buffered text = sentences[2] + [3] + [4].
    This is what we EMBED; the original sentences are what we later concatenate into
    chunk text. Buffering only smooths the *distance signal*, it never changes output.
    """
    out: list[str] = []
    n = len(sentences)
    for i in range(n):
        lo = max(0, i - buffer)
        hi = min(n, i + buffer + 1)
        out.append(" ".join(sentences[lo:hi]))
    return out


def _emit_with_size_cap(text: str, tokenizer, cfg: Config) -> list[str]:
    """Yield chunk texts for one semantic group, splitting if it exceeds the token cap.

    A semantically-coherent run can still be longer than the embedder can encode
    (512 tokens). If so we fall back to the baseline sliding window (with overlap) to
    cut it into capacity-safe pieces - semantic boundaries first, size safety second.
    """
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) <= cfg.chunk.size_tokens:
        return [text]
    pieces: list[str] = []
    for win in _window(ids, cfg.chunk.size_tokens, cfg.chunk.overlap_tokens):
        piece = tokenizer.decode(win, skip_special_tokens=True).strip()
        if piece:
            pieces.append(piece)
    return pieces


def chunk_all(cfg: Config) -> list[dict]:
    articles = read_articles()
    tokenizer = AutoTokenizer.from_pretrained(cfg.embedding.model_name)
    model = get_model(cfg.embedding.model_name)
    percentile = cfg.chunk.semantic_breakpoint_percentile

    chunks: list[dict] = []
    for art in articles:
        slug = art["slug"]
        # Identical content to the baseline: title + summary + body.
        full_text = "\n".join(
            p for p in (art["title"], art["summary"], art["body_text"]) if p
        )
        sentences = _split_sentences(full_text)

        # Degenerate cases: 0 or 1 sentence has no "gap" to measure -> one group.
        # (The size cap still protects a single very long sentence.)
        if len(sentences) <= 1:
            groups = [sentences] if sentences else []
        else:
            # 1) Embed buffered sentences, L2-normalised so dot product == cosine.
            vecs = model.encode(
                _buffered(sentences, _BUFFER_SIZE),
                normalize_embeddings=True,
                convert_to_numpy=True,
            )
            # 2) Cosine distance between consecutive sentences: 1 - (v_i . v_{i+1}).
            sims = np.sum(vecs[:-1] * vecs[1:], axis=1)   # cosine sim of each adjacent pair
            distances = 1.0 - sims                         # high distance = topic shift

            # 3) Threshold = Nth percentile of THIS article's distances. Cutting at
            #    gaps above it means "cut at the most-dissimilar boundaries only".
            #    Per-article (not global) so each article adapts to its own texture.
            threshold = np.percentile(distances, percentile)
            breakpoints = [i for i, d in enumerate(distances) if d > threshold]

            # 4) Slice sentences into groups at each breakpoint. A breakpoint at index
            #    i means "cut BETWEEN sentence i and i+1", so the group ends at i.
            groups = []
            start = 0
            for bp in breakpoints:
                groups.append(sentences[start : bp + 1])
                start = bp + 1
            groups.append(sentences[start:])  # trailing group after the last breakpoint

        # 5) Emit each group as chunk text, applying the token size cap.
        idx = 0
        for group in groups:
            group_text = " ".join(group).strip()
            if not group_text:
                continue
            for piece in _emit_with_size_cap(group_text, tokenizer, cfg):
                chunks.append(
                    {
                        "chunk_id": f"{slug}::{idx}",
                        "source_url": art["source_url"],
                        "text": piece,
                    }
                )
                idx += 1

    CHUNKS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
        for ch in chunks:
            f.write(json.dumps(ch, ensure_ascii=False) + "\n")

    n_art = len(articles)
    avg = (len(chunks) / n_art) if n_art else 0.0
    print(
        f"semantic chunking: {len(chunks)} chunks over {n_art} articles "
        f"(avg {avg:.1f} chunks/article)  [breakpoint pctile={percentile}, "
        f"buffer={_BUFFER_SIZE}, size_cap={cfg.chunk.size_tokens}]"
    )
    return chunks
