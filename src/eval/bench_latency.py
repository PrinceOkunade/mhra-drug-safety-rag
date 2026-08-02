"""Latency benchmark - the system-metrics column the harness was missing.

Phase 3 measured retrieval QUALITY (recall@k, MRR) and generation QUALITY
(correctness, faithfulness, refusal). Neither says anything about how long a user
waits. Profiling the Phase 4 container showed the cross-encoder rerank costs
45-70s PER REQUEST on a 4-CPU box, dominating end-to-end latency - so the
quality/latency tradeoff between the champion (hybrid -> reranker) and
hybrid-alone needs to be MEASURED, not estimated by subtraction.

Design decisions:
  * Retrieval and generation are timed SEPARATELY. Retrieval is the part the
    config changes; generation is a network call to Claude and is config-independent,
    so timing it once and reusing it keeps the comparison clean (and cheap).
  * Reports MEDIAN and p95, never the mean: tail latency is what users feel, and a
    single slow cold call would drag a mean around.
  * The reranked config is deliberately given FEWER samples (it costs ~50s each);
    sample counts are reported alongside so the numbers are not over-claimed.
  * Config is varied via dataclasses.replace - config.yaml (the committed
    system-of-record) is never mutated.

Run:  python -m src.eval.bench_latency [--n-fast 20] [--n-slow 8] [--gen-calls 3]
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import replace

from src.config import GOLD_PATH, load_config


def _questions(limit: int) -> list[str]:
    qs: list[str] = []
    with open(GOLD_PATH, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            # answerable questions only - unanswerable ones exercise the same
            # retrieval path, but keeping the set uniform makes the numbers comparable.
            if rec.get("supporting_passages"):
                qs.append(rec["question"])
    return qs[:limit]


def _summarize(name: str, samples: list[float]) -> dict:
    s = sorted(samples)
    p95 = s[min(len(s) - 1, int(round(0.95 * (len(s) - 1))))]
    return {
        "config": name,
        "n": len(s),
        "median_s": round(statistics.median(s), 2),
        "p95_s": round(p95, 2),
        "min_s": round(s[0], 2),
        "max_s": round(s[-1], 2),
    }


def time_retrieval(cfg, questions: list[str], label: str) -> dict:
    from src.retrieve.dense import retrieve

    retrieve("warmup", cfg, k=cfg.retriever.top_k)  # exclude one-off lazy init
    samples = []
    for i, q in enumerate(questions, 1):
        t = time.perf_counter()
        retrieve(q, cfg, k=cfg.retriever.top_k)
        dt = time.perf_counter() - t
        samples.append(dt)
        print(f"  [{label}] {i}/{len(questions)}  {dt:6.2f}s", flush=True)
    return _summarize(label, samples)


def time_generation(cfg, questions: list[str]) -> dict:
    """Time ONLY the Claude call, with context already retrieved."""
    from src.generate.llm import generate
    from src.generate.prompt import build_user_message, get_system_prompt
    from src.retrieve.dense import retrieve

    samples = []
    for i, q in enumerate(questions, 1):
        hits = retrieve(q, cfg, k=cfg.retriever.top_k)
        t = time.perf_counter()
        generate(get_system_prompt(cfg), build_user_message(q, hits, cfg), cfg)
        dt = time.perf_counter() - t
        samples.append(dt)
        print(f"  [generation] {i}/{len(questions)}  {dt:6.2f}s", flush=True)
    return _summarize("generation (Claude call)", samples)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-fast", type=int, default=20, help="samples for hybrid-alone")
    ap.add_argument("--n-slow", type=int, default=8, help="samples for the reranked champion")
    ap.add_argument("--gen-calls", type=int, default=3, help="Claude calls to time")
    ap.add_argument("--skip-generation", action="store_true")
    args = ap.parse_args()

    base = load_config()
    champion = replace(base, retriever=replace(base.retriever, reranker_enabled=True))
    hybrid_only = replace(base, retriever=replace(base.retriever, reranker_enabled=False))

    rows = []
    rows.append(time_retrieval(hybrid_only, _questions(args.n_fast), "hybrid alone (retrieval)"))
    rows.append(time_retrieval(champion, _questions(args.n_slow), "hybrid + reranker (retrieval)"))

    gen = None
    if not args.skip_generation:
        gen = time_generation(hybrid_only, _questions(args.gen_calls))
        rows.append(gen)

    print("\n=== LATENCY (seconds) ===")
    print(f"{'config':34s} {'n':>3s} {'median':>8s} {'p95':>8s} {'min':>7s} {'max':>7s}")
    for r in rows:
        print(f"{r['config']:34s} {r['n']:3d} {r['median_s']:8.2f} {r['p95_s']:8.2f} "
              f"{r['min_s']:7.2f} {r['max_s']:7.2f}")

    if gen:
        print("\n=== END-TO-END (retrieval median + generation median) ===")
        for r in rows[:2]:
            print(f"{r['config']:34s} {r['median_s'] + gen['median_s']:8.2f}s")

    print("\n" + json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
