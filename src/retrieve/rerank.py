"""Cross-encoder reranking - Phase 3 Experiment 5 (the ML component).

A bi-encoder (our BGE embedder) encodes the query and each passage SEPARATELY,
so the query never attends to the passage - fast, but it leaves rank-1 precision
on the table. A cross-encoder feeds [query, passage] through the model TOGETHER,
so every query token attends to every passage token: much sharper relevance
judgement, too slow for the whole corpus, ideal for reordering a small candidate
set. Its exact target is the diagnosed headroom - passages sitting at rank 2-3.

Step 1 (this file, off-the-shelf): `cross-encoder/ms-marco-MiniLM-L-6-v2`, trained
on MS MARCO passage ranking - the standard, generic control. Step 2 fine-tunes a
cross-encoder on corpus-specific synthetic pairs and measures the lift OVER this
control (that delta is what the fine-tuning bought).

Returns the same hit records, reordered, with an added `rerank_score`.
"""
from __future__ import annotations

from sentence_transformers import CrossEncoder

_reranker: CrossEncoder | None = None
_reranker_name: str | None = None


def get_reranker(model_name: str) -> CrossEncoder:
    """Cache the cross-encoder (reload only if the configured model changes)."""
    global _reranker, _reranker_name
    if _reranker is None or _reranker_name != model_name:
        _reranker = CrossEncoder(model_name)
        _reranker_name = model_name
    return _reranker


def rerank(query: str, hits: list[dict], cfg, k: int | None = None) -> list[dict]:
    """Re-score (query, passage) pairs with the cross-encoder and reorder, best-first.

    `hits` are the retriever's candidate records; `query` is the raw question (no
    bge prefix - the cross-encoder is not the bi-encoder). Truncates to k if given.
    """
    if not hits:
        return hits
    ce = get_reranker(cfg.retriever.reranker_model)
    scores = ce.predict([[query, h["text"]] for h in hits])
    order = sorted(range(len(hits)), key=lambda i: scores[i], reverse=True)
    out: list[dict] = []
    for i in order:
        rec = dict(hits[i])
        rec["rerank_score"] = float(scores[i])
        out.append(rec)
    return out[:k] if k is not None else out
