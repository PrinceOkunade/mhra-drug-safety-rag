"""Dense (semantic) retrieval over the FAISS index.

Embed the question (with bge's query-side instruction prefix), normalize, and ask
FAISS for the top-k nearest chunks by cosine similarity.
"""
from __future__ import annotations

import json
import pickle

import faiss

from src.config import CHUNKS_PKL, FAISS_PATH, PARENTS_PATH, Config
from src.index.build import get_model

_index = None
_chunks: list[dict] | None = None
_parents: dict[str, dict] | None = None


def _load():
    global _index, _chunks
    if _index is None:
        if not FAISS_PATH.exists():
            raise FileNotFoundError("index not built - run `build` first")
        _index = faiss.read_index(str(FAISS_PATH))
        with open(CHUNKS_PKL, "rb") as f:
            _chunks = pickle.load(f)
    return _index, _chunks


def _load_parents() -> dict[str, dict]:
    """Load the parent-window map for small-to-big retrieval (parent_id -> record)."""
    global _parents
    if _parents is None:
        if not PARENTS_PATH.exists():
            raise FileNotFoundError(
                "parents.jsonl not found - run `chunk` with strategy small_to_big"
            )
        _parents = {}
        with open(PARENTS_PATH, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    p = json.loads(line)
                    _parents[p["parent_id"]] = p
    return _parents


def _retrieve_small_to_big(index, children, qvec, k: int) -> list[dict]:
    """Match small children, but return their (de-duplicated) big parents.

    We search a larger CHILD pool, then walk hits best-first, mapping each child
    to its parent and keeping the FIRST (best-scoring) child per parent, until we
    have k distinct parents - so recall@k is measured over k parents, comparable
    to the baseline's k chunks.
    """
    parents = _load_parents()
    pool = min(max(k * 6, 50), len(children))
    scores, idxs = index.search(qvec, pool)

    hits: list[dict] = []
    seen: set[str] = set()
    for score, i in zip(scores[0], idxs[0]):
        if i < 0:
            continue
        child = children[i]
        pid = child["parent_id"]
        if pid in seen:
            continue
        seen.add(pid)
        parent = parents.get(pid)
        if parent is None:  # safety: stale index vs parents.jsonl
            continue
        hits.append(
            {
                "chunk_id": pid,
                "source_url": parent["source_url"],
                "text": parent["text"],          # the BIG parent is what gets read + scored
                "score": float(score),           # best child score for this parent
                "matched_child_id": child["chunk_id"],
            }
        )
        if len(hits) >= k:
            break
    return hits


def retrieve(query: str, cfg: Config, k: int | None = None) -> list[dict]:
    """Return the top-k chunk records with similarity scores (best first).

    `k` defaults to the configured top-k (the value the generation pipeline
    uses). Eval passes an explicit larger k (e.g. 10) to compute recall@10
    without changing the pipeline's behaviour.
    """
    index, chunks = _load()
    model = get_model(cfg.embedding.model_name)

    q = cfg.embedding.query_prefix + query
    qvec = model.encode([q], normalize_embeddings=True, convert_to_numpy=True)

    k = k or cfg.retriever.top_k

    # The reranker wraps ANY base retriever: pull a larger candidate pool of `n`
    # (rerank_top_n), cross-encode it, keep top-k. Without the reranker, the base
    # retriever returns k directly. This is what lets the CHAMPION stack
    # hybrid -> reranker (Exp 3 + Exp 5) as one system.
    reranker_on = cfg.retriever.reranker_enabled
    n = cfg.retriever.rerank_top_n if reranker_on else k

    if cfg.retriever.type == "hybrid":
        from src.retrieve.hybrid import hybrid_retrieve
        base = hybrid_retrieve(query, qvec, index, chunks, cfg, n)
    elif cfg.chunk.strategy == "small_to_big":
        base = _retrieve_small_to_big(index, chunks, qvec, n)
    else:
        nn = min(n, len(chunks))
        scores, idxs = index.search(qvec, nn)
        base = []
        for score, i in zip(scores[0], idxs[0]):
            if i < 0:
                continue
            rec = dict(chunks[i])
            rec["score"] = float(score)
            base.append(rec)

    if reranker_on:
        from src.retrieve.rerank import rerank
        return rerank(query, base, cfg, k=k)
    return base[:k]
