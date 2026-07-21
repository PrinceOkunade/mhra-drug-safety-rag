"""Gold question set: loader, schema validation, and version stamp.

Ground truth is chunking-independent: each supporting passage is (source_url,
verbatim passage_text). NEVER reference chunk IDs here.

GOLD_SET_VERSION is recorded in every results row so numbers run against
different question sets are never compared by accident. Bump it whenever
certified questions are added or edited.
"""
from __future__ import annotations

import json
from pathlib import Path

from src.config import GOLD_PATH

GOLD_SET_VERSION = "v0.2"  # bump on every certified append/edit
# v0.2: corpus scaled to 2022-01-01..present (snapshot cd315f17d0d9); q005 amended
# (low-birth-weight risk), q018/q038 removed (flipped/ambiguous on bigger corpus);
# batch_04 (q061-q082) appended -> 80 questions (64 answerable + 16 unanswerable).


def gold_set_stamp() -> str:
    """Combined version recorded in results rows: gold version + corpus snapshot.

    Numbers are only comparable within the same gold version AND the same corpus
    snapshot, so both are pinned together.
    """
    try:
        from src.eval.snapshot import load_snapshot
        sid = load_snapshot()["snapshot_id"]
    except Exception:
        sid = "unknown"
    return f"{GOLD_SET_VERSION}+corpus:{sid}"

_TYPES = {"single_passage", "multi_passage", "unanswerable"}
_REGISTERS = {"HCP", "patient", "either"}
_PHRASINGS = {"verbatim", "paraphrased"}
_REQUIRED = {
    "id", "question", "answer_key", "supporting_passages",
    "register", "answerable", "type", "phrasing",
}


def validate_item(item: dict) -> None:
    """Raise AssertionError if a gold item violates the schema/invariants."""
    missing = _REQUIRED - item.keys()
    assert not missing, f"{item.get('id', '?')}: missing fields {missing}"
    assert item["type"] in _TYPES, f"{item['id']}: bad type {item['type']}"
    assert item["register"] in _REGISTERS, f"{item['id']}: bad register"
    assert item["phrasing"] in _PHRASINGS, f"{item['id']}: bad phrasing"
    assert isinstance(item["answerable"], bool), f"{item['id']}: answerable not bool"

    if item["answerable"]:
        assert item["type"] != "unanswerable", \
            f"{item['id']}: answerable item typed unanswerable"
        assert item["supporting_passages"], \
            f"{item['id']}: answerable item has no supporting passages"
        for p in item["supporting_passages"]:
            assert p.get("source_url") and p.get("passage_text"), \
                f"{item['id']}: passage missing source_url/passage_text"
    else:
        assert item["type"] == "unanswerable", \
            f"{item['id']}: unanswerable item must have type 'unanswerable'"
        assert not item["supporting_passages"], \
            f"{item['id']}: unanswerable item must have no supporting passages"


def load_gold(path: Path = GOLD_PATH, enforce_snapshot: bool = True) -> list[dict]:
    """Load and validate the gold set.

    With `enforce_snapshot`, also runs check-0: every cited source_url must be an
    article present in the pinned corpus snapshot (otherwise the question is an
    unfixable retrieval failure unrelated to the system).
    """
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found - certify and append at least one batch first."
        )
    snap_urls = None
    if enforce_snapshot:
        try:
            from src.eval.snapshot import load_snapshot
            snap_urls = {a["source_url"] for a in load_snapshot()["articles"]}
        except Exception:
            snap_urls = None  # snapshot not built yet; skip check-0

    items = []
    seen_ids = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            validate_item(item)
            assert item["id"] not in seen_ids, f"duplicate id {item['id']}"
            seen_ids.add(item["id"])
            if snap_urls is not None:
                for p in item["supporting_passages"]:
                    assert p["source_url"] in snap_urls, (
                        f"{item['id']}: source not in corpus snapshot "
                        f"({p['source_url']}) - drop the question or ingest the article"
                    )
            items.append(item)
    return items


def answerable(items: list[dict]) -> list[dict]:
    return [i for i in items if i["answerable"]]


def unanswerable(items: list[dict]) -> list[dict]:
    return [i for i in items if not i["answerable"]]
