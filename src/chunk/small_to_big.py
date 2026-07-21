"""Small-to-big (parent-document) chunking - Phase 3 Experiment 2.

Motivated directly by Experiment 1's forensics: header-aware's regression was a
within-article *shuffle* - the right ARTICLE
ranked first, but a short sibling section (often a 40-token title / "Download
document" stub) out-embedded the substantive answer section and stole rank 1,
because a small chunk embeds generically to its keywords while the surrounding
context in a bigger chunk is what sharpens the match.

Small-to-big resolves that tension by DECOUPLING the unit we MATCH on from the
unit we READ:

  * MATCH on small CHILD chunks - precise, keyword-sharp; these are what get
    embedded and indexed (written to CHUNKS_PATH). Small = good recall@1.
  * READ the big PARENT window - at retrieval (retrieve/dense.py) each matched
    child is swapped for its parent, and THAT is handed to the LLM + scored by
    the eval. Because we return the parent, "which child ranked first" stops
    mattering - every within-article shuffle from Exp 1 is neutralised by
    construction, and the buried-fragment failure (q049) is fixed because the
    parent carries the surrounding context.

Design - kept a clean one-variable change vs the fixed baseline:
  * Parent = a `size_tokens` (512) window, the SAME size as the baseline chunk,
    but NON-OVERLAPPING (overlap 0) so every child maps to exactly ONE parent -
    no duplicate children across parents, no double-returned parents. Overlap 0
    vs the baseline's 50 is the one deliberate secondary difference; noted here
    honestly (it is small relative to the match-granularity change).
  * Child = a small window (`child_size_tokens` / `child_overlap_tokens`) carved
    from within each parent's own tokens, so a child never crosses a parent
    boundary.
  * Same tokenizer, same content (title + summary + body, exactly as fixed.py).
  * Child rows are the usual {chunk_id, source_url, text} PLUS a `parent_id`
    linking child -> parent. Parents go to PARENTS_PATH as
    {parent_id, source_url, text}.
"""
from __future__ import annotations

import json

from transformers import AutoTokenizer

from src.config import CHUNKS_PATH, PARENTS_PATH, Config
from src.chunk.fixed import _window  # reuse the baseline windower for parents AND children
from src.ingest.clean import read_articles


def chunk_all(cfg: Config) -> list[dict]:
    articles = read_articles()
    tokenizer = AutoTokenizer.from_pretrained(cfg.embedding.model_name)
    parent_size = cfg.chunk.size_tokens
    child_size = cfg.chunk.child_size_tokens
    child_overlap = cfg.chunk.child_overlap_tokens

    children: list[dict] = []
    parents: list[dict] = []

    for art in articles:
        slug = art["slug"]
        # Identical content to the baseline: title + summary + body.
        full_text = "\n".join(
            p for p in (art["title"], art["summary"], art["body_text"]) if p
        )
        ids = tokenizer.encode(full_text, add_special_tokens=False)

        # NON-OVERLAPPING parents (overlap 0) -> clean 1:1 child->parent mapping.
        for pj, pwin in enumerate(_window(ids, parent_size, 0)):
            parent_text = tokenizer.decode(pwin, skip_special_tokens=True).strip()
            if not parent_text:
                continue
            parent_id = f"{slug}::p{pj}"
            parents.append(
                {
                    "parent_id": parent_id,
                    "source_url": art["source_url"],
                    "text": parent_text,
                }
            )
            # Children are sub-windows of THIS parent's tokens (never cross it).
            for ci, cwin in enumerate(_window(pwin, child_size, child_overlap)):
                child_text = tokenizer.decode(cwin, skip_special_tokens=True).strip()
                if not child_text:
                    continue
                children.append(
                    {
                        "chunk_id": f"{parent_id}::c{ci}",
                        "source_url": art["source_url"],
                        "text": child_text,
                        "parent_id": parent_id,
                    }
                )

    # CHUNKS_PATH = the CHILDREN (what the indexer embeds).
    CHUNKS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
        for ch in children:
            f.write(json.dumps(ch, ensure_ascii=False) + "\n")
    # PARENTS_PATH = the parent windows (what retrieve.py swaps in for reading).
    with open(PARENTS_PATH, "w", encoding="utf-8") as f:
        for p in parents:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    n_par, n_ch = len(parents), len(children)
    avg = (n_ch / n_par) if n_par else 0.0
    print(
        f"small-to-big chunking: {n_ch} children over {n_par} parents "
        f"(avg {avg:.1f} children/parent)  "
        f"[child={child_size}/{child_overlap}, parent={parent_size}/0]"
    )
    return children
