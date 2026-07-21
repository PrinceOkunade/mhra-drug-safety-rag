"""Two-article HTML-structure diff to confirm the corpus start date.

The cleaner is built for the modern gov.uk template. This check fetches the
oldest article in our window and the newest DSU article, compares their
structural skeletons, and prints a verdict. If they match, the chosen start date
(config: published_after) sits safely inside the uniform-HTML era.
"""
from __future__ import annotations

import requests
from bs4 import BeautifulSoup

from src.config import Config
from src.ingest.feed import USER_AGENT

BLOCK_TAGS = {"h2", "h3", "h4", "p", "ul", "ol", "li", "table", "div"}


def _fetch_one(cfg: Config, order: str) -> dict | None:
    """Fetch one article record from the Search API by sort order."""
    c = cfg.corpus
    params = {
        "filter_content_store_document_type": "drug_safety_update",
        "filter_public_timestamp": f"from:{c.published_after},to:{c.published_before}",
        "order": order,
        "count": "1",
        "fields": "title,link,public_timestamp",
    }
    if order.startswith("-"):
        # Newest overall - drop the upper date bound.
        params["filter_public_timestamp"] = f"from:{c.published_after}"
    r = requests.get(c.search_api, params=params,
                     headers={"User-Agent": USER_AGENT}, timeout=30)
    r.raise_for_status()
    results = r.json().get("results", [])
    if not results:
        return None
    res = results[0]
    return {
        "url": c.base_url + res["link"],
        "published_date": res.get("public_timestamp", "")[:10],
    }


def _skeleton(html: str) -> dict:
    soup = BeautifulSoup(html, "lxml")
    body = soup.select_one('[data-module="govspeak"]') or soup.select_one(
        ".govuk-govspeak"
    )
    tag_profile = sorted({t.name for t in body.find_all(BLOCK_TAGS)}) if body else []
    return {
        "has_h1": soup.find("h1") is not None,
        "has_lead": soup.select_one(".gem-c-lead-paragraph") is not None,
        "has_govspeak": body is not None,
        "body_block_tags": tag_profile,
    }


def run_diff_check(cfg: Config) -> bool:
    oldest = _fetch_one(cfg, "public_timestamp")
    newest = _fetch_one(cfg, "-public_timestamp")
    if not oldest or not newest:
        print("diff-check: could not fetch comparison articles.")
        return False

    headers = {"User-Agent": USER_AGENT}
    sk_old = _skeleton(requests.get(oldest["url"], headers=headers, timeout=30).text)
    sk_new = _skeleton(requests.get(newest["url"], headers=headers, timeout=30).text)

    core_match = (
        sk_old["has_h1"] == sk_new["has_h1"]
        and sk_old["has_lead"] == sk_new["has_lead"]
        and sk_old["has_govspeak"] == sk_new["has_govspeak"]
        and sk_old["has_govspeak"]
    )

    print(f"oldest in window : {oldest['published_date']}  {oldest['url']}")
    print(f"  skeleton: {sk_old}")
    print(f"newest article   : {newest['published_date']}  {newest['url']}")
    print(f"  skeleton: {sk_new}")
    if core_match:
        print(
            f"\nVERDICT: structure matches -> start date "
            f"{cfg.corpus.published_after} is inside the uniform-HTML era. OK."
        )
    else:
        print(
            "\nVERDICT: structure differs -> move the start date forward and "
            "re-run before trusting the cleaner on older pages."
        )
    return core_match
