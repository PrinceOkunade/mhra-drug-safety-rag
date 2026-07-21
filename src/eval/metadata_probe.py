"""Exp 4 measurement: run the metadata probes with metadata OFF vs ON and judge.

Each probe is answered twice through the SAME retriever (metadata is generation-only
in this design, so retrieval is identical) - once with metadata_enabled=False and
once True - then a targeted judge scores whether the answer satisfies the probe's
property (register attribution, or reflecting the current/superseded position).
The OFF->ON delta is what the metadata USE bought.
"""
from __future__ import annotations

import json
import re
from dataclasses import replace
from datetime import datetime, timezone

from src.config import load_config, EVAL_DIR, RESULTS_DIR
from src.generate.answer import answer_question
from src.generate.llm import generate

PROBES_PATH = EVAL_DIR / "metadata_probes.jsonl"

REGISTER_JUDGE = """You evaluate whether an ANSWER correctly serves PATIENT-directed advice.
The person asked AS A PATIENT. A correct answer gives patient-facing advice (what to
watch for, what to do, when to seek help) and must NOT present healthcare-professional-
only clinical instructions (e.g. "test serum levels", "monitor levels", prescribing or
dosing decisions, authorisation/indication details) as if they were advice to the patient.
Given QUESTION, PATIENT-ADVICE REFERENCE, and ANSWER, reply with ONLY a JSON object:
{"verdict": "PASS" or "FAIL", "reason": "<one sentence>"}."""

SUPERSESSION_JUDGE = """You evaluate whether an ANSWER reflects the CURRENT MHRA position.
Given QUESTION, CURRENT-POSITION REFERENCE (the most recent article's advice), and ANSWER,
reply with ONLY a JSON object: {"verdict": "PASS" or "FAIL", "reason": "<one sentence>"}.
PASS if the answer conveys that the measures were changed/updated (the current position),
not only older superseded advice or a flat refusal."""


def _judge(cfg, system: str, question: str, reference: str, answer: str) -> dict:
    judge_cfg = replace(cfg, llm=replace(cfg.llm, model=cfg.llm.judge_model, temperature=0.0))
    user = f"QUESTION:\n{question}\n\nREFERENCE:\n{reference}\n\nANSWER:\n{answer}"
    raw = generate(system, user, judge_cfg)
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    try:
        obj = json.loads(m.group(0)) if m else {}
    except json.JSONDecodeError:
        obj = {}
    verdict = str(obj.get("verdict", "")).upper()
    return {"pass": verdict == "PASS", "reason": obj.get("reason", raw[:120])}


def run() -> None:
    cfg = load_config()
    cfg_off = replace(cfg, llm=replace(cfg.llm, metadata_enabled=False))
    cfg_on = replace(cfg, llm=replace(cfg.llm, metadata_enabled=True))

    probes = [json.loads(l) for l in open(PROBES_PATH, encoding="utf-8") if l.strip()]
    per_item = []
    off_pass = on_pass = 0

    for p in probes:
        system = REGISTER_JUDGE if p["property"] == "register" else SUPERSESSION_JUDGE
        ref = p["supporting_passages"][0]["passage_text"]

        ans_off = answer_question(p["question"], cfg_off)["answer"]
        ans_on = answer_question(p["question"], cfg_on)["answer"]
        j_off = _judge(cfg, system, p["question"], ref, ans_off)
        j_on = _judge(cfg, system, p["question"], ref, ans_on)
        off_pass += j_off["pass"]
        on_pass += j_on["pass"]

        flip = "OFF✗->ON✓" if (not j_off["pass"] and j_on["pass"]) else \
               ("OFF✓->ON✗" if (j_off["pass"] and not j_on["pass"]) else "same")
        print(f"{p['id']} [{p['property']:12}] OFF={'PASS' if j_off['pass'] else 'FAIL'} "
              f"ON={'PASS' if j_on['pass'] else 'FAIL'}  {flip}")
        print(f"   ON reason: {j_on['reason']}")
        per_item.append({"id": p["id"], "property": p["property"],
                         "off_pass": j_off["pass"], "on_pass": j_on["pass"],
                         "off_reason": j_off["reason"], "on_reason": j_on["reason"]})

    n = len(probes)
    print(f"\nmetadata OFF: {off_pass}/{n}   ON: {on_pass}/{n}   delta: {on_pass-off_pass:+d}")
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%S")
    out = RESULTS_DIR / f"{stamp}_metadata_probe.json"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"n": n, "off_pass": off_pass, "on_pass": on_pass,
                   "delta": on_pass - off_pass, "per_item": per_item}, f, indent=2)
    print(f"row -> {out}")


if __name__ == "__main__":
    run()
