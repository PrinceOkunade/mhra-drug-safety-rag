"""Chunk metadata schema + derivation - Phase 3 Experiment 4.

Each chunk is tagged {drug_or_device, register, date, subtype, source_url}. Fields
are DERIVED (date/subtype/drug from the article the chunk came from; register from
the chunk text itself) and then USED at generation time (see prompt.py): the model
is shown each passage's register + date so it can (a) attribute HCP vs patient
advice correctly and (b) prefer the most recent advice when sources conflict.

The spec's rule: metadata that is only *stored* moves no metric - the whole point
is the prompt USE, which we measure with the adversarial probe set.
"""
from __future__ import annotations

import functools
import re

from src.ingest.clean import read_articles


def _subtype(slug: str) -> str:
    if slug.startswith("letters-and-medicine-recalls"):
        return "bulletin"
    if slug.startswith("covid-19") or "medsafetyweek" in slug or slug.startswith("safety-roundup"):
        return "roundup"
    return "standalone"


@functools.lru_cache(maxsize=1)
def _article_index() -> dict[str, dict]:
    """source_url -> {date, subtype, drug_or_device} from articles.jsonl."""
    idx: dict[str, dict] = {}
    for a in read_articles():
        idx[a["source_url"]] = {
            "date": a.get("published_date", ""),
            "subtype": _subtype(a["slug"]),
            "drug_or_device": a["slug"].split("-")[0],
        }
    return idx


_HCP_MARKER = "advice for healthcare professionals"
_PATIENT_MARKER = re.compile(
    r"advice[^.]{0,40}patient|give to patients|for patients and caregivers|advice for patients"
)


def chunk_register(text: str) -> str:
    """HCP / patient / HCP+patient / general - derived from the chunk's own text."""
    t = text.lower()
    hcp = _HCP_MARKER in t
    pat = bool(_PATIENT_MARKER.search(t))
    if hcp and pat:
        return "HCP+patient"
    if hcp:
        return "HCP"
    if pat:
        return "patient"
    return "general"


def enrich(hit: dict) -> dict:
    """Return the metadata dict for one retrieved hit."""
    meta = dict(_article_index().get(hit["source_url"], {"date": "", "subtype": "", "drug_or_device": ""}))
    meta["register"] = chunk_register(hit["text"])
    return meta
