"""Judge-validation sample.

Generates real baseline answers for a diverse 15-item sample, runs the Sonnet
judge (correctness + faithfulness), and writes a hand-grading sheet so the human
can compare their own verdicts to the judge's and we can record the agreement
rate. Run BEFORE trusting any faithfulness number.

  python -m src.eval.validate_judge
"""
from __future__ import annotations

import datetime as _dt

from src.config import RESULTS_DIR, load_config
from src.eval.gold import load_gold
from src.eval.judge import judge_correctness, judge_faithfulness
from src.eval.run import is_refusal
from src.generate.answer import answer_question

# Diverse spread: multi-passage, single, HCP/patient, across all 3 batches,
# plus 3 unanswerable to test the refusal/faithfulness edge.
SAMPLE_IDS = [
    "q001", "q004", "q005", "q009", "q021", "q027", "q033",
    "q036", "q041", "q047", "q052", "q055",      # 12 answerable
    "q017", "q038", "q057",                        # 3 unanswerable
]


def run() -> None:
    cfg = load_config()
    gold = {q["id"]: q for q in load_gold()}
    lines = ["# Judge-validation sheet\n",
             f"_generated {_dt.datetime.now().isoformat(timespec='seconds')} | "
             f"judge = {cfg.llm.judge_model} | generator = {cfg.llm.model}_\n",
             "For each item, fill in YOUR verdict and compare to the JUDGE's.\n"]

    for qid in SAMPLE_IDS:
        q = gold[qid]
        res = answer_question(q["question"], cfg)
        refused = is_refusal(res["answer"])
        lines.append("\n" + "=" * 70)
        lines.append(f"## {qid}  ({'answerable' if q['answerable'] else 'UNANSWERABLE'})")
        lines.append(f"**Q:** {q['question']}")
        lines.append(f"**answer_key:** {q['answer_key']}")
        lines.append(f"**generated answer:**\n{res['answer']}")
        lines.append(f"**retrieved sources:** " +
                     ", ".join(s.rsplit('/', 1)[-1][:40] for s in res["sources"]))
        lines.append(f"**refused?** {refused}")

        if q["answerable"]:
            corr = judge_correctness(q["question"], q["answer_key"], res["answer"], cfg)
            faith = judge_faithfulness(res["answer"], res["hits"], cfg)
            lines.append(f"**JUDGE correctness:** {corr['label']} - {corr['reason']}")
            if corr.get("missing_facts"):
                lines.append(f"   missing: {corr['missing_facts']}")
            lines.append(f"**JUDGE faithfulness:** {faith['faithful']} - {faith['reason']}")
            if faith.get("unsupported_claims"):
                lines.append(f"   unsupported: {faith['unsupported_claims']}")
            lines.append("**YOUR correctness [ ]   YOUR faithfulness [ ]**")
        else:
            lines.append(f"**JUDGE (expected refusal):** refused = {refused}")
            lines.append("**YOUR verdict - correct refusal? [ ]**")

        # console one-liner
        verdict = (f"label={corr['label']} faithful={faith['faithful']}"
                   if q["answerable"] else f"refused={refused}")
        print(f"{qid}: {verdict}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"judge_validation_{_dt.datetime.now():%Y%m%dT%H%M%S}.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nHand-grading sheet written: {out}")


if __name__ == "__main__":
    run()
