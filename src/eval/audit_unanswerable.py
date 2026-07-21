"""Audit: did any 'unanswerable' gold question flip to answerable on the scaled
corpus? Retrieve each unanswerable question against the current index and show
the top hits so a human can judge whether the corpus now genuinely answers it.

  python -m src.eval.audit_unanswerable
"""
from __future__ import annotations

from src.config import load_config
from src.eval.gold import load_gold, unanswerable
from src.retrieve.dense import retrieve


def run() -> None:
    cfg = load_config()
    for q in unanswerable(load_gold(enforce_snapshot=False)):
        hits = retrieve(q["question"], cfg, k=3)
        print("\n" + "=" * 70)
        print(f"{q['id']}: {q['question']}")
        for h in hits:
            snippet = h["text"].replace("\n", " ")[:140]
            print(f"  score={h['score']:.3f}  {h['source_url'].rsplit('/',1)[-1][:55]}")
            print(f"     {snippet}")


if __name__ == "__main__":
    run()
