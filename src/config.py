"""Load config.yaml into a frozen dataclass that every stage reads.

One config object, no hard-coded knobs anywhere downstream. Phase 3's eval
harness will sweep these by loading alternate config files.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import yaml

# Repo root = parent of this file's parent (src/ -> repo/).
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RAW_DIR = DATA / "raw"
PROCESSED_DIR = DATA / "processed"
INDEX_DIR = DATA / "index"

ARTICLES_PATH = PROCESSED_DIR / "articles.jsonl"
CHUNKS_PATH = PROCESSED_DIR / "chunks.jsonl"
# small-to-big (Exp 2): CHUNKS_PATH holds the CHILD chunks that get embedded;
# PARENTS_PATH holds the parent windows that retrieval swaps in for reading.
PARENTS_PATH = PROCESSED_DIR / "parents.jsonl"
FAISS_PATH = INDEX_DIR / "faiss.index"
CHUNKS_PKL = INDEX_DIR / "chunks.pkl"

# Phase 2 evaluation artefacts (data lives in eval/, code in src/eval/).
EVAL_DIR = ROOT / "eval"
GOLD_PATH = EVAL_DIR / "gold_questions.jsonl"
RESULTS_DIR = EVAL_DIR / "results"
DRAFTS_DIR = EVAL_DIR / "drafts"


@dataclass(frozen=True)
class CorpusConfig:
    search_api: str
    base_url: str
    published_after: str
    published_before: str
    max_articles: int


@dataclass(frozen=True)
class ChunkConfig:
    size_tokens: int
    overlap_tokens: int
    strategy: str = "fixed"  # "fixed" (baseline) | "header_aware" | "small_to_big"
    # small-to-big children (only used when strategy == "small_to_big"): the small
    # units we MATCH on. Parents reuse size_tokens (non-overlapping) as the unit we READ.
    child_size_tokens: int = 128
    child_overlap_tokens: int = 16
    # header_aware Exp 1-corrected: merge any section below this many tokens into a
    # neighbour before chunking (kills pathological stub sections). 0 = v1 (no floor).
    header_min_section_tokens: int = 0
    # semantic (Exp 2c): cut a chunk boundary at any adjacent-sentence gap whose cosine
    # distance exceeds this percentile of the article's gaps. Higher = fewer, bigger
    # chunks; lower = more, smaller chunks. The tuning knob for semantic chunking.
    semantic_breakpoint_percentile: float = 90.0


@dataclass(frozen=True)
class EmbeddingConfig:
    model_name: str
    query_prefix: str


@dataclass(frozen=True)
class RetrieverConfig:
    type: str
    top_k: int
    reranker_enabled: bool
    rrf_k: int = 60  # Reciprocal Rank Fusion constant (Exp 3 hybrid); 60 is the standard default
    # Exp 5 reranker: retrieve rerank_top_n candidates, cross-encode, keep top_k.
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rerank_top_n: int = 50


@dataclass(frozen=True)
class LLMConfig:
    provider: str
    model: str
    temperature: float
    max_tokens: int
    judge_model: str
    # Exp 4: when True, show per-passage register + date to the generator and add
    # register-attribution + recency-preference rules (metadata USE, not just storage).
    metadata_enabled: bool = False


@dataclass(frozen=True)
class Config:
    corpus: CorpusConfig
    chunk: ChunkConfig
    embedding: EmbeddingConfig
    retriever: RetrieverConfig
    llm: LLMConfig


def load_config(path: Path | str | None = None) -> Config:
    """Read config.yaml (defaults to repo-root config.yaml) into a Config."""
    cfg_path = Path(path) if path else ROOT / "config.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return Config(
        corpus=CorpusConfig(**raw["corpus"]),
        chunk=ChunkConfig(**raw["chunk"]),
        embedding=EmbeddingConfig(**raw["embedding"]),
        retriever=RetrieverConfig(**raw["retriever"]),
        llm=LLMConfig(**raw["llm"]),
    )


def ensure_dirs() -> None:
    """Create the data directories the pipeline writes to."""
    for d in (RAW_DIR, PROCESSED_DIR, INDEX_DIR):
        d.mkdir(parents=True, exist_ok=True)


def raw_path_for(slug: str) -> Path:
    """Stable raw-HTML path for a slug.

    DSU slugs can exceed Windows' ~260-char path limit, so we cap the filename:
    first 80 chars of the slug + a short hash of the full slug (keeps it unique).
    """
    short = slug[:80]
    digest = hashlib.md5(slug.encode("utf-8")).hexdigest()[:8]
    return RAW_DIR / f"{short}-{digest}.html"
