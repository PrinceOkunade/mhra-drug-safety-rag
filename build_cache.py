"""Pre-compute demo responses so the Streamlit app never calls Claude on example clicks.

WHY THIS EXISTS
---------------
A public portfolio demo must be free and abuse-proof: if every example button hit the
Claude API, a reviewer (or a bot) could run up the bill, and the API key would have to
live in the deployment. Instead we run each curated example ONCE here, through the real
pipeline (`answer_question` - the exact function the eval harness scored), and freeze the
answer + retrieved chunks to `demo_cache.json`. The app then serves examples instantly
from that file, with no key and no network.

This reuses the existing retrieval + generation code - it does not reimplement anything.

Run (needs ANTHROPIC_API_KEY in .env, and the champion index built):
    python build_cache.py
"""
from __future__ import annotations

import json

from src.config import ARTICLES_PATH, ROOT, load_config

CACHE_PATH = ROOT / "demo_cache.json"

# Curated from the real 64+16 gold set, so each behaves exactly as measured. `adversarial`
# marks the unanswerable ones - clicking them demonstrates the refusal path live.
EXAMPLES: list[dict] = [
    {"question": "For omega-3-acid ethyl ester medicines used to treat high triglycerides, which heart-rhythm disorder is now a common adverse reaction, and at which dose is the risk greatest?", "adversarial": False},
    {"question": "What rare eye condition has been associated with semaglutide (Wegovy/Ozempic/Rybelsus)?", "adversarial": False},
    {"question": "What are the most common side effects of GLP-1 receptor agonists, and roughly how often do they occur?", "adversarial": False},
    {"question": "How can metformin affect vitamin B12, and what raises that risk?", "adversarial": False},
    {"question": "How has the legal status of codeine linctus changed, and why?", "adversarial": False},
    {"question": "Which side effects of finasteride can continue even after a man stops taking it?", "adversarial": False},
    {"question": "Why are pholcodine-containing cough and cold medicines being withdrawn from the UK market?", "adversarial": False},
    # Adversarial: plausible-sounding but NOT answerable from drug-safety advisories -> refusal.
    {"question": "What is the recommended starting dose of metformin for type 2 diabetes?", "adversarial": True},
    {"question": "Is it safe to take sertraline during pregnancy?", "adversarial": True},
    {"question": "Is it safe to drink alcohol while taking ibuprofen?", "adversarial": True},
]


def _article_lookup() -> dict[str, dict]:
    """source_url -> {title, published_date}, so retrieved chunks show provenance."""
    lut: dict[str, dict] = {}
    with open(ARTICLES_PATH, "r", encoding="utf-8") as f:
        for line in f:
            a = json.loads(line)
            lut[a["source_url"]] = {"title": a.get("title", ""), "date": a.get("published_date", "")}
    return lut


def main() -> None:
    cfg = load_config()
    articles = _article_lookup()
    # Import here (not at top) so `python build_cache.py --help`-style usage stays light;
    # this line triggers model/index loading.
    from src.generate.answer import answer_question

    out = []
    for i, ex in enumerate(EXAMPLES, 1):
        q = ex["question"]
        print(f"[{i}/{len(EXAMPLES)}] {'(adversarial) ' if ex['adversarial'] else ''}{q[:70]}...", flush=True)
        res = answer_question(q, cfg)
        chunks = [
            {
                "title": articles.get(h["source_url"], {}).get("title", "(unknown article)"),
                "date": articles.get(h["source_url"], {}).get("date", ""),
                "score": round(float(h.get("score", 0.0)), 4),
                "source_url": h["source_url"],
                "text": h["text"],
            }
            for h in res["hits"]
        ]
        out.append({
            "question": q,
            "adversarial": ex["adversarial"],
            "answer": res["answer"],
            "sources": res["sources"],
            "chunks": chunks,
        })

    with open(CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump({"config": {"retriever": cfg.retriever.type, "reranker": cfg.retriever.reranker_enabled,
                              "embedder": cfg.embedding.model_name, "generator": cfg.llm.model},
                   "examples": out}, f, indent=2, ensure_ascii=False)
    print(f"\nWrote {len(out)} cached responses -> {CACHE_PATH}")


if __name__ == "__main__":
    main()
