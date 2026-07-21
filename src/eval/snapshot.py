"""Define and pin the Phase 2 corpus snapshot.

The gold set is certified relative to a SPECIFIC ingested corpus. We record
exactly which articles are in it (source_urls + dates) plus a content id, so:
  - unanswerable questions are unanswerable *relative to this snapshot*;
  - if the corpus changes (Phase 3), the snapshot id changes and the gold set
    must be re-certified;
  - every results row can record which snapshot it ran against.

Run:  python -m src.eval.snapshot   (writes eval/corpus_snapshot.json)
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json

from src.config import EVAL_DIR
from src.ingest.clean import read_articles

SNAPSHOT_PATH = EVAL_DIR / "corpus_snapshot.json"


def build_snapshot() -> dict:
    arts = read_articles()
    rows = sorted(
        ({"slug": a["slug"], "source_url": a["source_url"],
          "published_date": a.get("published_date", "")} for a in arts),
        key=lambda r: (r["published_date"], r["slug"]),
    )
    slugs = [r["slug"] for r in rows]
    # Stable id from the sorted slug list: changes iff the article SET changes.
    snapshot_id = hashlib.sha256("\n".join(sorted(slugs)).encode()).hexdigest()[:12]
    dates = [r["published_date"] for r in rows if r["published_date"]]
    return {
        "snapshot_id": snapshot_id,
        "created_at": _dt.date.today().isoformat(),
        "n_articles": len(rows),
        "date_min": min(dates) if dates else None,
        "date_max": max(dates) if dates else None,
        "articles": rows,
    }


def write_snapshot() -> dict:
    snap = build_snapshot()
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    with open(SNAPSHOT_PATH, "w", encoding="utf-8") as f:
        json.dump(snap, f, indent=2, ensure_ascii=False)
    return snap


def load_snapshot() -> dict:
    with open(SNAPSHOT_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    s = write_snapshot()
    print(f"corpus_snapshot.json written: id={s['snapshot_id']} "
          f"n={s['n_articles']} range={s['date_min']}..{s['date_max']}")
