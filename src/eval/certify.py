"""Certification harness - runs the automatable gold-set checks against the
INGESTED corpus (data/processed), not live gov.uk URLs.

For every drafted item it reports:
  check-0  existence-of-article : each cited source_url is in the corpus snapshot
  check-a  passage existence     : each passage is contained in >=1 CHUNK (the
                                    unit retrieval actually returns), under the
                                    normalised match rule. If a passage is in the
                                    article body but NOT containable in any single
                                    chunk, it is flagged SPLIT (an unwinnable
                                    recall miss - shorten the passage).
  currency : flags items drawn from a topic with >1 article in the snapshot, and
             names the most-recent article in that topic to check against.

checks b (support) and c (coverage) are SEMANTIC and remain human judgements -
this tool reports them as "manual" rather than auto-passing them. The agent must
not self-certify.

Run:  python -m src.eval.certify eval/drafts/batch_01.jsonl
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from src.eval.match import chunk_contains_passage
from src.chunk.fixed import read_chunks
from src.ingest.clean import read_articles
from src.eval.snapshot import load_snapshot

# Topic keys that are NOT superseding guidance (monthly bulletins, awareness).
_CURRENCY_EXCLUDE = {"letters", "medsafetyweek"}


def _topic(slug: str) -> str:
    return slug.split("-")[0]


def _load_drafts(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def certify(path: Path) -> None:
    items = _load_drafts(path)
    snap = load_snapshot()
    snap_urls = {a["source_url"] for a in snap["articles"]}
    url_to_slug = {a["source_url"]: a["slug"] for a in snap["articles"]}

    # Topic groups (for currency / supersession), newest first within each topic.
    topics: dict[str, list[dict]] = {}
    for a in snap["articles"]:
        topics.setdefault(_topic(a["slug"]), []).append(a)
    for grp in topics.values():
        grp.sort(key=lambda a: a["published_date"], reverse=True)

    chunks = read_chunks()
    bodies = {a["source_url"]: a["body_text"] for a in read_articles()}

    print(f"Corpus snapshot: id={snap['snapshot_id']} n={snap['n_articles']} "
          f"range={snap['date_min']}..{snap['date_max']}")
    print(f"Certifying {len(items)} items from {path.name}\n" + "=" * 78)

    for it in items:
        print(f"\n{it['id']}  [{it['type']} | {it['register']} | {it['phrasing']}]")
        print(f"  Q: {it['question']}")

        if not it["answerable"]:
            print("  (unanswerable) check-0/a: N/A - certified vs THIS snapshot only.")
            print("  MANUAL: confirm no snapshot article answers this question.")
            continue

        currency_topics: set[str] = set()
        for i, p in enumerate(it["supporting_passages"], 1):
            url = p["source_url"]
            in_snapshot = url in snap_urls
            slug = url_to_slug.get(url, url.rsplit("/", 1)[-1])

            # check-0
            c0 = "OK" if in_snapshot else "FAIL (article not in corpus)"
            # check-a: containable in a chunk?
            in_chunk = any(chunk_contains_passage(c["text"], p["passage_text"])
                           for c in chunks)
            in_body = (url in bodies) and chunk_contains_passage(
                bodies[url], p["passage_text"])
            if in_chunk:
                ca = "OK (in chunk)"
            elif in_body:
                ca = "SPLIT - in article body but no single chunk contains it (shorten passage)"
            else:
                ca = "MISSING - not found in corpus (verbatim?)"

            print(f"  passage {i} [{slug}]")
            print(f"    check-0 article-in-corpus : {c0}")
            print(f"    check-a passage-existence : {ca}")

            if in_snapshot:
                currency_topics.add(_topic(slug))

        # currency / supersession
        flags = []
        for t in currency_topics:
            grp = [a for a in topics.get(t, []) if _topic(a["slug"]) not in _CURRENCY_EXCLUDE]
            if t in _CURRENCY_EXCLUDE:
                continue
            if len(grp) > 1:
                newest = grp[0]
                used = {url_to_slug.get(p["source_url"]) for p in it["supporting_passages"]}
                is_newest = newest["slug"] in used
                flags.append(
                    f"topic '{t}' has {len(grp)} articles; newest = "
                    f"{newest['slug']} ({newest['published_date']})"
                    + ("" if is_newest else "  <-- NEWEST NOT CITED: review currency")
                )
        print("  check-d currency : " + ("OK (single-article topic)" if not flags
              else "REVIEW - " + "; ".join(flags)))
        print("  MANUAL: (b) does each passage support the answer_key? "
              "(c) is every answer_key claim covered by a passage?")

    print("\n" + "=" * 78)
    print("Automated checks done. (b) support, (c) coverage, and final (d) currency "
          "are human judgements - certify those by hand.")


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    if not target:
        print("usage: python -m src.eval.certify <drafts.jsonl>")
        raise SystemExit(1)
    certify(target)
