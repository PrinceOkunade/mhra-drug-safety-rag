"""Clean gov.uk article HTML down to title + summary + body text.

This is the real work of Phase 1. A gov.uk page is mostly chrome (cookie banner,
nav, "Is this page useful", footer, related links). We keep ONLY:
  - title:   the <h1>
  - summary: the .gem-c-lead-paragraph lead
  - body:    the [data-module="govspeak"] article container (which holds the
             "Advice for healthcare professionals / patients" sections)

A verification pass then scans the extracted text for known boilerplate and
fails loudly if any survives - polluted chunks are the #1 cause of a baseline
looking broken, so we prove cleanliness before embedding.
"""
from __future__ import annotations

import json
from pathlib import Path

from bs4 import BeautifulSoup

from src.config import ARTICLES_PATH, raw_path_for

# If any of these appear in the cleaned text, extraction grabbed page chrome.
BOILERPLATE_MARKERS = [
    "cookies on gov.uk",
    "is this page useful",
    "skip to main content",
    "tell us whether you accept cookies",
    "hide this message",
    "your browser does not",
    "related content",
    "is this page useful?",
]


def _extract(html: str) -> tuple[str, str, str]:
    """Return (title, summary, body_text) from one article's HTML."""
    soup = BeautifulSoup(html, "lxml")

    h1 = soup.find("h1")
    title = h1.get_text(strip=True) if h1 else ""

    lead = soup.select_one(".gem-c-lead-paragraph")
    summary = lead.get_text(strip=True) if lead else ""

    body_el = soup.select_one('[data-module="govspeak"]') or soup.select_one(
        ".govuk-govspeak"
    )
    if body_el is None:
        raise ValueError("no govspeak body container found")
    # Newline-separated keeps section headings on their own lines, which helps
    # the register separation (HCP vs patient advice) read cleanly downstream.
    body_text = body_el.get_text(separator="\n", strip=True)

    return title, summary, body_text


def _verify_clean(text: str, slug: str) -> None:
    low = text.lower()
    hits = [m for m in BOILERPLATE_MARKERS if m in low]
    if hits:
        raise AssertionError(f"boilerplate survived in {slug}: {hits}")


def clean_all(records: list[dict]) -> list[dict]:
    """Clean every fetched article and write data/processed/articles.jsonl."""
    articles: list[dict] = []
    for rec in records:
        raw_path = raw_path_for(rec["slug"])
        html = raw_path.read_text(encoding="utf-8")
        title, summary, body = _extract(html)

        combined = "\n".join([title, summary, body])
        _verify_clean(combined, rec["slug"])
        if not body.strip():
            raise AssertionError(f"empty body after cleaning: {rec['slug']}")

        articles.append(
            {
                "source_url": rec["url"],
                "slug": rec["slug"],
                "title": title or rec.get("title", ""),
                "summary": summary,
                "body_text": body,
                "published_date": rec.get("published_date", ""),
            }
        )

    ARTICLES_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(ARTICLES_PATH, "w", encoding="utf-8") as f:
        for art in articles:
            f.write(json.dumps(art, ensure_ascii=False) + "\n")
    return articles


def read_articles(path: Path = ARTICLES_PATH) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
