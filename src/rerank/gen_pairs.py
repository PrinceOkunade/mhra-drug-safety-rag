"""Synthetic training-pair generation for the fine-tuned reranker (Exp 5 Step 2).

We can't train a cross-encoder on our 80 gold questions - that's the eval, and
using it would leak. Instead we SYNTHESISE training data from the corpus: for each
chunk, ask the generator LLM (Haiku) to write questions that the chunk directly
answers -> (question, positive-passage) pairs. For each positive we mine a HARD
NEGATIVE: the top BM25 chunk for that question that is NOT the positive - lexically
similar but wrong, which is exactly what teaches the reranker to discriminate.

Leakage discipline (PHASE3_SPEC, non-negotiable): every chunk that contains a gold
supporting passage is EXCLUDED from training entirely (as positive AND as negative
source), and the 80 gold questions never appear. So the eval stays clean.

Output: eval/reranker_train_pairs.jsonl  {question, passage, label}  (1=pos, 0=neg).
"""
from __future__ import annotations

import argparse
import json
import random
import re

from rank_bm25 import BM25Okapi

from src.config import load_config, GOLD_PATH, EVAL_DIR
from src.chunk.fixed import read_chunks
from src.generate.llm import generate
from src.eval.match import chunk_contains_any

PAIRS_PATH = EVAL_DIR / "reranker_train_pairs.jsonl"

SYSTEM = (
    "You generate retrieval questions for a medical drug-safety information system. "
    "Given one passage, you output specific questions that the passage itself directly "
    "and completely answers - the kind a healthcare professional or patient would ask. "
    "No preamble, no numbering, one question per line."
)


def _tok(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _load_gold_passages() -> list[dict]:
    passages: list[dict] = []
    with open(GOLD_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            g = json.loads(line)
            for p in g.get("supporting_passages", []):
                passages.append({"passage_text": p["passage_text"]})
    return passages


def _gen_questions(chunk_text: str, n: int, cfg) -> list[str]:
    user = (
        f"Passage:\n\"\"\"\n{chunk_text}\n\"\"\"\n\n"
        f"Write {n} distinct, specific questions that THIS passage directly and fully "
        f"answers. One question per line."
    )
    txt = generate(SYSTEM, user, cfg)
    out = []
    for line in txt.splitlines():
        q = line.strip().lstrip("0123456789.-) ").strip()
        if len(q) > 12 and q.endswith("?"):
            out.append(q)
    return out[:n]


def main(target: int, per_chunk: int) -> None:
    cfg = load_config()
    chunks = read_chunks()
    gold_passages = _load_gold_passages()

    # Exclude every chunk that carries a gold passage - no eval leakage.
    eligible = [c for c in chunks if not chunk_contains_any(c["text"], gold_passages)]
    print(f"{len(chunks)} chunks; {len(eligible)} eligible after removing gold-passage chunks")

    # BM25 over ALL chunks, for hard-negative mining.
    bm25 = BM25Okapi([_tok(c["text"]) for c in chunks])

    random.seed(0)
    random.shuffle(eligible)

    pairs: list[dict] = []
    n_pos = 0
    for c in eligible:
        if n_pos >= target:
            break
        for q in _gen_questions(c["text"], per_chunk, cfg):
            pairs.append({"question": q, "passage": c["text"], "label": 1})
            n_pos += 1
            # hard negative: best BM25 chunk that is neither this chunk nor a gold chunk
            scores = bm25.get_scores(_tok(q))
            for i in sorted(range(len(chunks)), key=lambda j: scores[j], reverse=True):
                if chunks[i]["chunk_id"] == c["chunk_id"]:
                    continue
                if chunk_contains_any(chunks[i]["text"], gold_passages):
                    continue
                pairs.append({"question": q, "passage": chunks[i]["text"], "label": 0})
                break
        if n_pos % 100 < per_chunk:
            print(f"  ...{n_pos} positives")

    PAIRS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(PAIRS_PATH, "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"wrote {len(pairs)} pairs ({n_pos} pos + {len(pairs)-n_pos} neg) -> {PAIRS_PATH}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=1000, help="number of positive pairs")
    ap.add_argument("--per-chunk", type=int, default=2, help="questions per chunk")
    args = ap.parse_args()
    main(args.target, args.per_chunk)
