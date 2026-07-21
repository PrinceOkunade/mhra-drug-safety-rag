"""Enumerate Drug Safety Update articles via the gov.uk Search API.

We use the Search API (not the HTML listing page) because it's the only source
that honours a real date filter and returns clean JSON. We filter to
`content_store_document_type == drug_safety_update` and bound by
`public_timestamp`, oldest first, capped at `max_articles`.

Output: a manifest list of {url, slug, title, published_date}, also written to
data/processed/manifest.jsonl so fetch/clean can run independently.
"""
from __future__ import annotations

import json
from pathlib import Path

import requests

from src.config import Config, PROCESSED_DIR

USER_AGENT = "MHRA-RAG-portfolio/0.1 (educational; contact via gov.uk)"
MANIFEST_PATH = PROCESSED_DIR / "manifest.jsonl"


def enumerate_articles(cfg: Config) -> list[dict]:
    """Query the Search API and return article records (oldest first)."""
    c = cfg.corpus
    params = {
        "filter_content_store_document_type": "drug_safety_update",
        "filter_public_timestamp": f"from:{c.published_after},to:{c.published_before}",
        "order": "public_timestamp",          # oldest first -> stable slice
        "count": str(c.max_articles),
        "fields": "title,link,public_timestamp",
    }
    resp = requests.get(
        c.search_api, params=params, headers={"User-Agent": USER_AGENT}, timeout=30
    )
    resp.raise_for_status()
    results = resp.json().get("results", [])

    records: list[dict] = []
    for r in results:
        link = r.get("link", "")
        # Defensive: only keep canonical DSU article paths (rule: gov.uk only).
        if not link.startswith("/drug-safety-update/"):
            continue
        slug = link.rsplit("/", 1)[-1]
        records.append(
            {
                "url": c.base_url + link,
                "slug": slug,
                "title": r.get("title", "").strip(),
                "published_date": r.get("public_timestamp", "")[:10],  # YYYY-MM-DD
            }
        )
    return records[: c.max_articles]


def write_manifest(records: list[dict], path: Path = MANIFEST_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def read_manifest(path: Path = MANIFEST_PATH) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
