"""Thin wrapper around the Anthropic Messages API (Claude Haiku 4.5).

Reads ANTHROPIC_API_KEY from .env. Plain call - no thinking/effort params (Haiku
4.5 does not take them). The model is set in config, so swapping providers later
is a one-file change.
"""
from __future__ import annotations

import anthropic
from dotenv import load_dotenv

from src.config import Config

load_dotenv()  # pulls ANTHROPIC_API_KEY from .env into the environment

_client: anthropic.Anthropic | None = None


def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env
    return _client


def generate(system: str, user_message: str, cfg: Config) -> str:
    """Send one prompt to Claude and return the text response."""
    client = _get_client()
    resp = client.messages.create(
        model=cfg.llm.model,
        max_tokens=cfg.llm.max_tokens,
        temperature=cfg.llm.temperature,
        system=system,
        messages=[{"role": "user", "content": user_message}],
    )
    return "".join(block.text for block in resp.content if block.type == "text")
