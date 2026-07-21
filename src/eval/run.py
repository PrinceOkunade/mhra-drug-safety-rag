"""Evaluation harness - runs the current pipeline over the gold set and writes
one committed results row to eval/results/.

Retrieval metrics need no LLM and can be run alone (`--retrieval-only`) so Phase 3
can iterate on retrieval cheaply. Generation metrics layer on top via the judge.

Determinism: generation temperature = 0 (from config), recorded in the row.

Usage:
  python -m src.eval.run --retrieval-only        # recall@k + MRR, no API calls
  python -m src.eval.run                          # full eval (generation + judge)
  python -m src.eval.run --limit 10               # small sample (judge dev)
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json

from src.config import Config, RESULTS_DIR, load_config
from src.eval.gold import load_gold, gold_set_stamp, answerable, unanswerable
from src.eval.match import MATCH_TOKEN_OVERLAP_THRESHOLD, normalize
from src.eval.metrics import Count, aggregate_retrieval, first_relevant_rank
from src.generate.prompt import REFUSAL_TEXT
from src.retrieve.dense import retrieve

EVAL_RETRIEVAL_K = 10  # depth for recall@10; generation still uses config top_k
_REFUSAL_SIG = normalize(REFUSAL_TEXT).split("information")[0][:60]  # stable prefix


def is_refusal(answer: str) -> bool:
    return _REFUSAL_SIG in normalize(answer)


def _config_snapshot(cfg: Config) -> dict:
    return {
        # Read the ACTUAL strategy from config - must not be hard-coded, or a
        # non-fixed experiment (header_aware / small_to_big / semantic) writes a row
        # that mislabels itself as "fixed" and corrupts the committed record.
        "chunk_strategy": cfg.chunk.strategy,
        "chunk_size_tokens": cfg.chunk.size_tokens,
        "chunk_overlap_tokens": cfg.chunk.overlap_tokens,
        "embedding_model": cfg.embedding.model_name,
        "retriever_type": cfg.retriever.type,
        "top_k": cfg.retriever.top_k,
        "reranker_enabled": cfg.retriever.reranker_enabled,
        "generation_model": cfg.llm.model,
        "judge_model": cfg.llm.judge_model,
        "generation_temperature": cfg.llm.temperature,
        "match_token_overlap_threshold": MATCH_TOKEN_OVERLAP_THRESHOLD,
        "eval_retrieval_k": EVAL_RETRIEVAL_K,
    }


def run(cfg: Config, retrieval_only: bool = False, limit: int | None = None) -> dict:
    gold = load_gold()
    ans = answerable(gold)
    unans = unanswerable(gold)
    if limit:
        ans = ans[:limit]
        unans = unans[: max(1, limit // 4)]

    # Per-item log (keyed by question id) so we can diff exactly which questions
    # change between runs - essential for reading Phase 3 deltas and for telling
    # a real win from generation-variance noise.
    per_item: dict[str, dict] = {}

    # ---- Retrieval metrics (no LLM) ----
    print(f"Retrieval over {len(ans)} answerable questions (k={EVAL_RETRIEVAL_K})...")
    ranks = []
    for q in ans:
        hits = retrieve(q["question"], cfg, k=EVAL_RETRIEVAL_K)
        rank = first_relevant_rank(hits, q["supporting_passages"])
        ranks.append(rank)
        per_item[q["id"]] = {
            "id": q["id"], "answerable": True, "type": q["type"],
            "register": q["register"], "phrasing": q["phrasing"],
            "first_relevant_rank": rank,
        }
    retrieval = aggregate_retrieval(ranks)
    print("  recall:", retrieval["recall_str"], "| MRR:", retrieval["mrr"])

    row = {
        "timestamp": _dt.datetime.now().isoformat(timespec="seconds"),
        "gold_set_version": gold_set_stamp(),
        "config": _config_snapshot(cfg),
        "retrieval": retrieval,
    }

    if not retrieval_only:
        from src.generate.answer import answer_question
        from src.eval.judge import (
            judge_correctness, judge_faithfulness,
            FAITHFULNESS_PROMPT_VERSION, CORRECTNESS_PROMPT_VERSION,
        )

        print(f"Generation + judge over {len(ans)} answerable, {len(unans)} unanswerable...")
        n_faithful = 0
        answerable_not_refused = 0
        # Three-way correctness label over ALL answerable questions (a refusal on
        # an answerable item is 'incomplete' per spec §2 - it supplied no facts).
        labels = {"correct": 0, "incomplete": 0, "wrong": 0}
        for q in ans:
            res = answer_question(q["question"], cfg)
            refused = is_refusal(res["answer"])
            rec = per_item[q["id"]]
            rec["refused"] = refused
            if refused:
                labels["incomplete"] += 1  # false refusal = supplied none of the facts
                rec["correctness_label"] = "incomplete"
                rec["correctness_note"] = "false refusal (no facts supplied)"
                rec["faithful"] = None  # not scored - a refusal asserts no claims
                continue
            answerable_not_refused += 1
            corr = judge_correctness(q["question"], q["answer_key"], res["answer"], cfg)
            faith = judge_faithfulness(res["answer"], res["hits"], cfg)
            labels[corr["label"]] += 1
            n_faithful += 1 if faith["faithful"] else 0
            rec["correctness_label"] = corr["label"]
            rec["faithful"] = faith["faithful"]
            if corr["label"] != "correct":
                rec["missing_facts"] = corr.get("missing_facts", [])
                rec["contradictions"] = corr.get("contradictions", [])
            if not faith["faithful"]:
                rec["unsupported_claims"] = faith.get("unsupported_claims", [])

        n_correct = labels["correct"]

        unanswerable_refused = 0
        for q in unans:
            res = answer_question(q["question"], cfg)
            refused = is_refusal(res["answer"])
            if refused:
                unanswerable_refused += 1
            per_item[q["id"]] = {
                "id": q["id"], "answerable": False, "type": q["type"],
                "refused": refused,
                "refusal_correct": refused,  # should refuse
            }

        row["generation"] = {
            "answer_correctness": Count(n_correct, len(ans)).as_dict(),
            "correctness_labels": labels,  # correct | incomplete | wrong (diagnosis)
            "faithfulness": Count(n_faithful, answerable_not_refused).as_dict(),
            "refusal": {
                "unanswerable_refused": Count(unanswerable_refused, len(unans)).as_dict(),
                "answerable_not_refused": Count(answerable_not_refused, len(ans)).as_dict(),
                "false_refusals": len(ans) - answerable_not_refused,
                "false_answers": len(unans) - unanswerable_refused,
            },
            "judge_prompt_versions": {
                "faithfulness": FAITHFULNESS_PROMPT_VERSION,
                "correctness": CORRECTNESS_PROMPT_VERSION,
            },
        }
        print("  correctness:", row["generation"]["answer_correctness"], labels)
        print("  faithfulness:", row["generation"]["faithfulness"])
        print("  refusal:", row["generation"]["refusal"])

    # Per-item detail, sorted by id, so two rows can be diffed question-by-question.
    row["per_item"] = [per_item[k] for k in sorted(per_item)]

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    tag = "retrieval" if retrieval_only else "full"
    out = RESULTS_DIR / f"{row['timestamp'].replace(':', '')}_{tag}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(row, f, indent=2, ensure_ascii=False)
    print(f"\nResults row written: {out}")
    return row


def main() -> None:
    p = argparse.ArgumentParser(description="MHRA RAG evaluation harness")
    p.add_argument("--config", default=None)
    p.add_argument("--retrieval-only", action="store_true",
                   help="compute retrieval metrics only (no LLM/judge calls)")
    p.add_argument("--limit", type=int, default=None,
                   help="evaluate only the first N answerable (judge dev)")
    args = p.parse_args()
    run(load_config(args.config), retrieval_only=args.retrieval_only, limit=args.limit)


if __name__ == "__main__":
    main()
