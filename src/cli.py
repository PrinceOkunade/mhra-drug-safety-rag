"""Command-line interface for the Phase 1 MHRA Drug Safety RAG baseline.

Subcommands:
  build       full pipeline: enumerate -> fetch -> clean -> chunk -> index
  ingest      enumerate + fetch raw HTML only
  clean       clean fetched HTML -> articles.jsonl
  chunk       fixed-size token chunking -> chunks.jsonl
  index       embed chunks + build FAISS index
  diff-check  confirm the corpus start date via a structure diff
  ask "Q"     retrieve + generate a grounded, cited answer
  smoke-test  run a few hand-written questions and print results (qualitative)
"""
from __future__ import annotations

import argparse

from src.config import ensure_dirs, load_config

# Hand-written smoke questions matched to the 2024 corpus. The last one is
# deliberately out of domain to exercise the refusal path.
SMOKE_QUESTIONS = [
    "What is the reclassification status of codeine linctus and codeine oral solutions?",
    "What cardiac risk is associated with omega-3 acid ethyl ester medicines?",
    "When should fluoroquinolone antibiotics now be prescribed?",
    "What rare neurological risk has been linked to pseudoephedrine?",
    "What new safety measures apply to valproate in people under 55?",
    "What is the recommended ibuprofen dose for a tension headache?",  # expect refusal
]


def _chunker_for(cfg):
    """Pick the chunk_all matching cfg.chunk.strategy (config-driven, per rule 17)."""
    strategy = cfg.chunk.strategy
    if strategy == "fixed":
        from src.chunk.fixed import chunk_all
    elif strategy == "header_aware":
        from src.chunk.header_aware import chunk_all
    elif strategy == "small_to_big":
        from src.chunk.small_to_big import chunk_all
    elif strategy == "semantic":
        from src.chunk.semantic import chunk_all
    else:
        raise ValueError(
            f"unknown chunk.strategy {strategy!r} "
            f"(expected 'fixed', 'header_aware', 'small_to_big', or 'semantic')"
        )
    return chunk_all


def _print_answer(result: dict, show_chunks: bool = False) -> None:
    print(f"\nQ: {result['question']}")
    print("-" * 70)
    if show_chunks:
        for i, h in enumerate(result["hits"], 1):
            preview = h["text"].replace("\n", " ")[:160]
            print(f"  [{i}] score={h['score']:.3f} {h['source_url']}")
            print(f"      {preview}...")
        print("-" * 70)
    print(result["answer"])


def cmd_build(cfg) -> None:
    from src.ingest.feed import enumerate_articles, write_manifest
    from src.ingest.fetch import fetch_all
    from src.ingest.clean import clean_all
    from src.index.build import build_index

    chunk_all = _chunker_for(cfg)
    ensure_dirs()
    print("1/5 enumerating articles via gov.uk Search API...")
    records = enumerate_articles(cfg)
    write_manifest(records)
    print(f"    {len(records)} articles in window "
          f"{cfg.corpus.published_after}..{cfg.corpus.published_before}")
    print("2/5 fetching raw HTML...")
    fetch_all(records)
    print("3/5 cleaning HTML -> articles.jsonl...")
    arts = clean_all(records)
    print(f"    cleaned {len(arts)} articles (no boilerplate survived)")
    print("4/5 chunking -> chunks.jsonl...")
    chunks = chunk_all(cfg)
    print(f"    produced {len(chunks)} chunks")
    print("5/5 embedding + building FAISS index...")
    build_index(cfg)
    print("\nBuild complete. Try: python -m src.cli ask \"<question>\"")


def cmd_ingest(cfg) -> None:
    from src.ingest.feed import enumerate_articles, write_manifest
    from src.ingest.fetch import fetch_all

    ensure_dirs()
    records = enumerate_articles(cfg)
    write_manifest(records)
    fetch_all(records)
    print(f"ingested {len(records)} articles")


def cmd_clean(cfg) -> None:
    from src.ingest.feed import read_manifest
    from src.ingest.clean import clean_all

    arts = clean_all(read_manifest())
    print(f"cleaned {len(arts)} articles")


def cmd_chunk(cfg) -> None:
    chunk_all = _chunker_for(cfg)
    chunks = chunk_all(cfg)
    print(f"produced {len(chunks)} chunks")


def cmd_index(cfg) -> None:
    from src.index.build import build_index

    build_index(cfg)


def cmd_diff_check(cfg) -> None:
    from src.ingest.diff_check import run_diff_check

    run_diff_check(cfg)


def cmd_ask(cfg, question: str) -> None:
    from src.generate.answer import answer_question

    _print_answer(answer_question(question, cfg), show_chunks=True)


def cmd_smoke_test(cfg) -> None:
    from src.generate.answer import answer_question

    print("Running smoke test (qualitative - NOT the Phase 2 eval harness)\n")
    for q in SMOKE_QUESTIONS:
        _print_answer(answer_question(q, cfg), show_chunks=True)
        print("=" * 70)


def main() -> None:
    parser = argparse.ArgumentParser(description="MHRA Drug Safety RAG (Phase 1)")
    parser.add_argument("--config", default=None, help="path to config.yaml")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("build", help="run the full pipeline")
    sub.add_parser("ingest", help="enumerate + fetch raw HTML")
    sub.add_parser("clean", help="clean HTML -> articles.jsonl")
    sub.add_parser("chunk", help="chunk -> chunks.jsonl")
    sub.add_parser("index", help="embed + build FAISS index")
    sub.add_parser("diff-check", help="confirm corpus start date")
    p_ask = sub.add_parser("ask", help="ask a question")
    p_ask.add_argument("question", help="the question to ask")
    sub.add_parser("smoke-test", help="run hand-written smoke questions")

    args = parser.parse_args()
    cfg = load_config(args.config)

    if args.command == "build":
        cmd_build(cfg)
    elif args.command == "ingest":
        cmd_ingest(cfg)
    elif args.command == "clean":
        cmd_clean(cfg)
    elif args.command == "chunk":
        cmd_chunk(cfg)
    elif args.command == "index":
        cmd_index(cfg)
    elif args.command == "diff-check":
        cmd_diff_check(cfg)
    elif args.command == "ask":
        cmd_ask(cfg, args.question)
    elif args.command == "smoke-test":
        cmd_smoke_test(cfg)


if __name__ == "__main__":
    main()
