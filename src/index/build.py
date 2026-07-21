"""Embed chunks and build a local FAISS index.

We embed each chunk with the local sentence-transformers model, L2-normalize the
vectors, and store them in a flat inner-product index (IndexFlatIP). Because the
vectors are normalized, inner product == cosine similarity, and "flat" means
exact brute-force search - perfect at our 20-30 article scale.

Persists:
  data/index/faiss.index  - the vectors
  data/index/chunks.pkl   - the chunk records, aligned to index row order
"""
from __future__ import annotations

import pickle

import faiss
from sentence_transformers import SentenceTransformer

from src.config import CHUNKS_PKL, FAISS_PATH, INDEX_DIR, Config
from src.chunk.fixed import read_chunks

# Cache the model so query-time retrieval reuses the same loaded instance.
_model: SentenceTransformer | None = None


def get_model(model_name: str) -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(model_name)
    return _model


def build_index(cfg: Config) -> int:
    chunks = read_chunks()
    if not chunks:
        raise RuntimeError("no chunks to index - run chunk first")

    model = get_model(cfg.embedding.model_name)
    texts = [c["text"] for c in chunks]
    # Passages get NO query prefix (bge applies its instruction to queries only).
    vectors = model.encode(
        texts,
        normalize_embeddings=True,   # -> inner product == cosine
        convert_to_numpy=True,
        show_progress_bar=True,
    )

    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(FAISS_PATH))
    with open(CHUNKS_PKL, "wb") as f:
        pickle.dump(chunks, f)

    print(f"  indexed {len(chunks)} chunks (dim={dim})")
    return len(chunks)
