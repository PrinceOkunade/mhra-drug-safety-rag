"""Hybrid retrieval - Phase 3 Experiment 3: dense (semantic) + BM25 (lexical).

Dense retrieval matches on MEANING (good for paraphrase) but can miss exact rare
tokens - a specific drug name ("glatiramer") or a batch code embeds into a fuzzy
semantic neighbourhood, so the one right chunk can rank low. BM25 is the opposite:
it scores exact term overlap, nailing rare keywords but blind to paraphrase.
Hybrid runs BOTH and fuses their rankings, aiming for semantic recall AND
exact-keyword precision - the target being the q032-type "right fact, wrong
ranking because dense didn't latch onto the drug name" failures.

Fusion = Reciprocal Rank Fusion (RRF), NOT weighted score-mixing. Dense cosine
(0-1) and BM25 scores (unbounded, corpus-dependent) live on different scales, so
normalising them to combine is brittle. RRF ignores the raw scores and fuses on
RANK POSITION only:  fused(d) = Σ_retrievers 1 / (rrf_k + rank_r(d)).  A document
near the top of either list scores well; appearing in both compounds. rrf_k=60 is
the standard constant (dampens the weight of very-top ranks so mid-list agreement
still counts).

BM25 is built lazily in-memory over the chunk texts (a few hundred chunks -> instant),
so there is no new persisted index. Returns the same {chunk_id, source_url, text,
score} records as dense retrieval, so eval + generation are unchanged.
"""
from __future__ import annotations

import re

from rank_bm25 import BM25Okapi

_bm25: BM25Okapi | None = None


def _tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokens - a plain lexical analyzer for BM25.

    Deliberately simple (no stemming/stopwords): drug names and codes are exactly
    the rare tokens we want BM25 to key on, and stemming risks mangling them.
    """
    return re.findall(r"[a-z0-9]+", text.lower())


def _get_bm25(chunks: list[dict]) -> BM25Okapi:
    global _bm25
    if _bm25 is None:
        _bm25 = BM25Okapi([_tokenize(c["text"]) for c in chunks])
    return _bm25


def _reset_cache() -> None:
    """Drop the cached BM25 index (call if the underlying chunks change)."""
    global _bm25
    _bm25 = None


def hybrid_retrieve(query: str, qvec, index, chunks: list[dict], cfg, k: int) -> list[dict]:
    """Fuse a dense FAISS ranking and a BM25 ranking via RRF; return top-k chunks.

    `qvec` is the already-embedded (bge-prefixed, normalized) query vector for the
    dense side; `query` is the RAW question text for BM25 (BM25 gets no bge prefix).
    """
    pool = min(max(k * 10, 50), len(chunks))

    # Dense ranked list (best-first) of chunk indices.
    _, d_idxs = index.search(qvec, pool)
    dense_ranked = [int(i) for i in d_idxs[0] if i >= 0]

    # BM25 ranked list over the same chunks.
    bm25 = _get_bm25(chunks)
    bm_scores = bm25.get_scores(_tokenize(query))
    bm25_ranked = sorted(range(len(chunks)), key=lambda i: bm_scores[i], reverse=True)[:pool]

    # Reciprocal Rank Fusion.
    rrf_k = cfg.retriever.rrf_k
    fused: dict[int, float] = {}
    for ranked in (dense_ranked, bm25_ranked):
        for rank, idx in enumerate(ranked, start=1):
            fused[idx] = fused.get(idx, 0.0) + 1.0 / (rrf_k + rank)

    top = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)[:k]
    hits: list[dict] = []
    for idx, score in top:
        rec = dict(chunks[idx])
        rec["score"] = float(score)  # the fused RRF score
        hits.append(rec)
    return hits
