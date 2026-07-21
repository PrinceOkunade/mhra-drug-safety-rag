"""Download article HTML to data/raw/<slug>.html.

Polite: a descriptive User-Agent and a small delay between requests. Skips files
already downloaded so re-runs are cheap (the raw HTML is the cache).
"""
from __future__ import annotations

import time
from pathlib import Path

import requests

from src.config import RAW_DIR, raw_path_for
from src.ingest.feed import USER_AGENT

REQUEST_DELAY_S = 1.0


def fetch_all(records: list[dict], force: bool = False) -> list[Path]:
    """Download each record's URL into data/raw/. Returns the saved paths."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    for rec in records:
        out = raw_path_for(rec["slug"])
        if out.exists() and not force:
            saved.append(out)
            continue
        resp = requests.get(
            rec["url"], headers={"User-Agent": USER_AGENT}, timeout=30
        )
        resp.raise_for_status()
        out.write_text(resp.text, encoding="utf-8")
        saved.append(out)
        print(f"  fetched {rec['slug']}")
        time.sleep(REQUEST_DELAY_S)
    return saved
