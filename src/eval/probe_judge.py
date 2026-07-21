"""Negative-control probes for the judge + re-check of the tightened prompt.

The all-positive validation sample only proved the judge doesn't FALSE-FLAG good
answers. This proves it CATCHES bad ones (faithfulness is the safety metric):
  P1 unsupported claim   -> faithfulness should be FALSE
  P2 factually wrong     -> correctness should be FALSE
  P3 confident answer to an unanswerable Q -> faithfulness should be FALSE
Plus a re-check that the tightened correctness-v2 prompt now flags q005's
omission as INCORRECT (the one human/judge disagreement).

  python -m src.eval.probe_judge
"""
from __future__ import annotations

from src.config import load_config
from src.eval.judge import judge_correctness, judge_faithfulness


def main() -> None:
    cfg = load_config()
    print(f"judge = {cfg.llm.judge_model}\n" + "=" * 60)

    # P1 - unsupported claim injected; context does NOT mention 90% recovery.
    chunks = [{"text": "PRES and RCVS are recognised very rare side effects with "
                       "pseudoephedrine-containing medicines."}]
    ans = ("Pseudoephedrine can cause PRES and RCVS. Studies show 90% of patients "
           "recover fully within two weeks and the drug also lowers cholesterol.")
    f = judge_faithfulness(ans, chunks, cfg)
    print(f"P1 unsupported-claim   -> faithful={f['faithful']}  "
          f"(want False) {'PASS' if not f['faithful'] else 'FAIL'}")
    print(f"   reason: {f['reason']}")

    # P2 - factually wrong vs the key.
    c = judge_correctness(
        "When must systemic fluoroquinolones now be prescribed?",
        "Only when other commonly recommended antibiotics are inappropriate.",
        "Fluoroquinolones are now the recommended first-line antibiotic for all "
        "common infections.", cfg)
    print(f"\nP2 factually-wrong     -> correct={c['correct']}  "
          f"(want False) {'PASS' if not c['correct'] else 'FAIL'}")
    print(f"   reason: {c['reason']}")

    # P3 - confident answer to an unanswerable question, with empty context.
    f3 = judge_faithfulness(
        "The recommended starting dose of metformin is 500 mg once daily with "
        "the evening meal.", [], cfg)
    print(f"\nP3 confident-unanswerable -> faithful={f3['faithful']}  "
          f"(want False) {'PASS' if not f3['faithful'] else 'FAIL'}")
    print(f"   reason: {f3['reason']}")

    # Re-check q005 under correctness-v2 (should now be INCORRECT - omits 11% +
    # infertility). Uses the same answer the generator produced in validation.
    q005_answer = (
        "Valproate carries a known up to 30-40% risk of neurodevelopmental "
        "disorders in children born to mothers taking it during pregnancy. For "
        "fathers, the cumulative risk of neurodevelopmental disorders ranged from "
        "4.0% to 5.6% versus 2.3% to 3.2% in the comparator group.")
    q005_key = (
        "In pregnancy: physical birth defects in about 11% of babies and "
        "neurodevelopmental disorders in up to 30-40% of children. For men: a "
        "possible increased risk of neurodevelopmental disorders in children "
        "fathered around conception (about 4.0-5.6% with valproate vs 2.3-3.2% "
        "with lamotrigine/levetiracetam), and reports of infertility in men "
        "taking valproate (mechanism unclear).")
    c5 = judge_correctness(
        "What reproductive risks does valproate carry for the baby in pregnancy "
        "and for men around conception?", q005_key, q005_answer, cfg)
    print(f"\nq005 re-check (v2)     -> correct={c5['correct']}  "
          f"(want False, matching human) {'PASS' if not c5['correct'] else 'FAIL'}")
    print(f"   missing: {c5.get('missing_facts')}")
    print(f"   reason: {c5['reason']}")


if __name__ == "__main__":
    main()
