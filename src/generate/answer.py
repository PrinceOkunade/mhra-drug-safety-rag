"""Orchestrate one question end-to-end: retrieve -> prompt -> generate.

Returns the answer text plus the retrieved hits (so the CLI can show which
passages/URLs fed the answer).
"""
from __future__ import annotations

from src.config import Config
from src.generate.llm import generate
from src.generate.prompt import get_system_prompt, build_user_message
from src.retrieve.dense import retrieve


def answer_question(question: str, cfg: Config) -> dict:
    hits = retrieve(question, cfg)
    user_message = build_user_message(question, hits, cfg)
    answer = generate(get_system_prompt(cfg), user_message, cfg)
    return {
        "question": question,
        "answer": answer,
        "hits": hits,
        "sources": sorted({h["source_url"] for h in hits}),
    }
