"""Retrieval metrics (no LLM needed) + small-sample-honest count helpers.

Retrieval metrics are deliberately runnable without any generation calls, so
Phase 3 can iterate on retrieval cheaply. Generation metrics (correctness,
faithfulness, refusal) are computed in run.py using the judge.

Every metric is reported as a raw count alongside the percentage, because some
denominators are small (e.g. ~15 unanswerable questions) and a single flipped
verdict can move a percentage by several points.
"""
from __future__ import annotations

from dataclasses import dataclass

from src.eval.match import chunk_contains_any


def first_relevant_rank(hits: list[dict], passages: list[dict]) -> int | None:
    """1-based rank of the first retrieved chunk that contains a supporting
    passage, or None if none of the hits do."""
    for rank, hit in enumerate(hits, start=1):
        if chunk_contains_any(hit["text"], passages):
            return rank
    return None


@dataclass
class Count:
    """A metric as numerator/denominator, rendered as 'n/d (pp%)'."""
    n: int
    d: int

    @property
    def pct(self) -> float:
        return (self.n / self.d * 100.0) if self.d else 0.0

    def as_dict(self) -> dict:
        return {"n": self.n, "d": self.d, "pct": round(self.pct, 1)}

    def __str__(self) -> str:
        return f"{self.n}/{self.d} ({self.pct:.0f}%)"


def aggregate_retrieval(ranks: list[int | None], ks=(1, 3, 5, 10)) -> dict:
    """Given the first-relevant-rank per answerable question, compute
    recall@k (as Counts) and MRR. Denominator = number of answerable questions."""
    d = len(ranks)
    recall = {
        f"recall@{k}": Count(sum(1 for r in ranks if r is not None and r <= k), d)
        for k in ks
    }
    mrr = (sum((1.0 / r) for r in ranks if r is not None) / d) if d else 0.0
    return {
        "recall": {k: v.as_dict() for k, v in recall.items()},
        "recall_str": {k: str(v) for k, v in recall.items()},
        "mrr": round(mrr, 4),
        "n_answerable": d,
    }
