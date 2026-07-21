"""Pre-download the serving models at image-build time (Phase 4).

Run inside the Dockerfile so the BGE embedder and the cross-encoder reranker are
baked into the image's HF cache - cold starts on Cloud Run then don't reach out to
Hugging Face (faster, and no runtime network dependency on HF).
"""
from __future__ import annotations

from src.config import load_config
from src.index.build import get_model
from src.retrieve.rerank import get_reranker


def main() -> None:
    cfg = load_config()
    print(f"pre-downloading embedder: {cfg.embedding.model_name}")
    get_model(cfg.embedding.model_name)
    if cfg.retriever.reranker_enabled:
        print(f"pre-downloading reranker: {cfg.retriever.reranker_model}")
        get_reranker(cfg.retriever.reranker_model)
    print("models cached.")


if __name__ == "__main__":
    main()
