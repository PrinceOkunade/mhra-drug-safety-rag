"""Header-aware chunking - the Phase 3 structure-aware splitter.

Experiment 1: beat the fixed-size baseline by splitting each article on its
section structure instead of a blind token window.

Design decisions that keep this a CLEAN one-variable change vs fixed.py, so any
retrieval delta is attributable to the split boundaries and nothing else:

  * Same content as baseline. fixed.py chunks `title + summary + body`. We keep
    exactly that text; title + summary are ATTACHED TO THE FIRST section (not a
    separate chunk), so only the boundaries differ, not what text exists.
  * Same tokenizer + same size/overlap. Over-long sections fall back to fixed's
    own `_window` at the identical cap, so the fallback path IS baseline
    behaviour - no new config knob.
  * Same output rows: {chunk_id, source_url, text}, same CHUNKS_PATH. The
    indexer downstream sees a structurally identical file.

Why raw HTML, not body_text: topic headings ("Review of pathological gambling",
"Reports of acute pulmonary damage") are arbitrary text, indistinguishable from
a body line once flattened. Only the raw <h2> tags reliably mark structure, so
we re-parse the raw HTML (as clean.py does) rather than pattern-match body_text.

v1 strategy: split on EVERY <h2>. All three subtypes (standalone advice
articles, letters/recalls bulletins, roundups) use h2s, so one splitter handles
all of them. Bulletins have only ~3 coarse h2s, so their big list-sections will
usually exceed the cap and degrade to _window - i.e. header-awareness mostly
helps STANDALONE articles. That is expected; per-item bulletin splitting is a
separate later experiment. We log per-subtype so the delta is not misread as
uniform.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from bs4 import BeautifulSoup
from transformers import AutoTokenizer

from src.config import CHUNKS_PATH, PROCESSED_DIR, Config, raw_path_for
from src.chunk.fixed import _window  # reuse the baseline splitter as the fallback
from src.ingest.clean import read_articles

# Sidecar debug/stats file. NEVER read by the indexer - keeps chunk rows clean.
DEBUG_PATH = PROCESSED_DIR / "chunk_debug_header_aware.jsonl"


def _subtype(slug: str) -> str:
    """Classify for LOGGING only - v1 uses one splitter for all subtypes."""
    if slug.startswith("letters-and-medicine-recalls"):
        return "bulletin"
    if slug.startswith("covid-19") or "medsafetyweek" in slug or slug.startswith("safety-roundup"):
        return "roundup"
    return "standalone"


def _content_root(body_el):
    """Unwrap single-tag wrappers until <h2>s sit at the child level."""
    node = body_el
    while True:
        tag_children = [c for c in node.children if getattr(c, "name", None)]
        has_h2 = any(getattr(c, "name", None) == "h2" for c in node.children)
        if (
            len(tag_children) == 1
            and tag_children[0].name in ("div", "article", "section")
            and not has_h2
        ):
            node = tag_children[0]
        else:
            return node


def _sections_from_html(html: str, title: str, summary: str) -> list[str]:
    """Return the article as a list of section texts.

    Section 0 = title + summary + any pre-first-<h2> body + first section body.
    Each subsequent <h2> starts a new section; <h3> and content accumulate into
    the current section. The <h2>/<h3> heading text is kept in the section (it
    is informative for retrieval, e.g. 'Advice for healthcare professionals:').
    """
    soup = BeautifulSoup(html, "lxml")
    body_el = soup.select_one('[data-module="govspeak"]') or soup.select_one(
        ".govuk-govspeak"
    )
    if body_el is None:
        raise ValueError("no govspeak body container found")
    root = _content_root(body_el)

    preamble: list[str] = [t for t in (title, summary) if t]
    sections: list[list[str]] = []
    current: list[str] | None = None

    for child in root.children:
        name = getattr(child, "name", None)
        if name is None:  # NavigableString (usually whitespace)
            text = str(child).strip()
            if not text:
                continue
        else:
            text = child.get_text(" ", strip=True)
            if not text:
                continue

        if name == "h2":
            if current is not None:
                sections.append(current)
            current = [text]
        else:
            if current is None:
                # content before the first <h2> -> belongs with the preamble
                preamble.append(text)
            else:
                current.append(text)

    if current is not None:
        sections.append(current)

    # No <h2> at all: whole body is one section (still valid; will size-cap).
    if not sections:
        sections = [[]]

    # Attach preamble (title+summary+pre-h2 text) to the FIRST section.
    sections[0] = preamble + sections[0]
    return ["\n".join(parts) for parts in sections if any(p.strip() for p in parts)]


def _merge_small_sections(sections: list[str], tokenizer, floor: int) -> list[str]:
    """Exp 1-corrected: fold any section below `floor` tokens into a neighbour.

    Rule (per PHASE3_SPEC): a below-floor section merges into the PREVIOUS section;
    if it is section 0 (no previous), it is carried forward and prepended to the
    NEXT section. This kills the pathological stubs the Exp 1 forensics found -
    40-token drug-name / "Download document" fragments (q071, q075) that spuriously
    won on cosine, and isolated small answer fragments (q049) that got buried.
    The floor is set ONCE on a structural principle (a real clinical advice
    statement doesn't fit in <~64 tokens), never swept against the gold set.
    """
    out: list[str] = []
    carry = ""  # a too-small section 0 waiting to attach to the next section
    for sec in sections:
        if carry:
            sec = carry + "\n" + sec
            carry = ""
        n = len(tokenizer.encode(sec, add_special_tokens=False))
        if n < floor:
            if out:
                out[-1] = out[-1] + "\n" + sec      # merge into previous
            else:
                carry = sec                          # section 0 too small -> next
        else:
            out.append(sec)
    if carry:  # every section was tiny (or a trailing small section 0)
        if out:
            out[-1] = out[-1] + "\n" + carry
        else:
            out.append(carry)
    return out


def chunk_all(cfg: Config) -> list[dict]:
    articles = read_articles()

    # Fail loudly if any article lacks raw HTML on disk - otherwise those
    # articles would silently vanish and the two strategies would be chunking
    # DIFFERENT corpora, invalidating the comparison.
    missing = [a["slug"] for a in articles if not raw_path_for(a["slug"]).exists()]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} article(s) have no raw HTML at raw_path_for(); "
            f"header-aware would drop them. First few: {missing[:5]}"
        )

    tokenizer = AutoTokenizer.from_pretrained(cfg.embedding.model_name)
    cap = cfg.chunk.size_tokens
    overlap = cfg.chunk.overlap_tokens
    floor = cfg.chunk.header_min_section_tokens

    chunks: list[dict] = []
    debug_rows: list[dict] = []
    # per-subtype tallies for the delta-by-subtype readout
    stats: dict[str, dict[str, int]] = defaultdict(
        lambda: {"articles": 0, "sections": 0, "chunks": 0, "capped_sections": 0}
    )

    for art in articles:
        slug = art["slug"]
        sub = _subtype(slug)
        html = raw_path_for(slug).read_text(encoding="utf-8")
        sections = _sections_from_html(html, art["title"], art["summary"])
        if floor > 0:
            sections = _merge_small_sections(sections, tokenizer, floor)

        i = 0
        capped = 0
        for sec_text in sections:
            ids = tokenizer.encode(sec_text, add_special_tokens=False)
            if len(ids) <= cap:
                # section fits -> one header-aware chunk
                text = sec_text.strip()
                if text:
                    chunks.append(
                        {
                            "chunk_id": f"{slug}::{i}",
                            "source_url": art["source_url"],
                            "text": text,
                        }
                    )
                    i += 1
            else:
                # over-long section -> degrade to EXACT baseline windowing
                capped += 1
                for win in _window(ids, cap, overlap):
                    text = tokenizer.decode(win, skip_special_tokens=True).strip()
                    if not text:
                        continue
                    chunks.append(
                        {
                            "chunk_id": f"{slug}::{i}",
                            "source_url": art["source_url"],
                            "text": text,
                        }
                    )
                    i += 1

        s = stats[sub]
        s["articles"] += 1
        s["sections"] += len(sections)
        s["chunks"] += i
        s["capped_sections"] += capped
        debug_rows.append(
            {
                "slug": slug,
                "subtype": sub,
                "n_sections": len(sections),
                "n_chunks": i,
                "capped_sections": capped,  # sections that fell back to _window
            }
        )

    # write chunks (same file/format the indexer reads)
    CHUNKS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
        for ch in chunks:
            f.write(json.dumps(ch, ensure_ascii=False) + "\n")

    # write sidecar debug (NOT read by the indexer)
    with open(DEBUG_PATH, "w", encoding="utf-8") as f:
        for row in debug_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # readable summary - this is what tells you WHERE header-awareness acted
    print("header-aware chunking - per subtype:")
    for sub in ("standalone", "bulletin", "roundup"):
        if sub in stats:
            s = stats[sub]
            print(
                f"  {sub:11} articles={s['articles']:3}  sections={s['sections']:4}  "
                f"chunks={s['chunks']:4}  capped(sections->window)={s['capped_sections']:3}"
            )
    print(f"  TOTAL chunks: {len(chunks)}  (debug: {DEBUG_PATH})")
    return chunks
