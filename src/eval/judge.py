"""LLM-as-judge (Claude Sonnet 4.6) for the generation metrics.

The judge MUST be a different, stronger model than the generator (Haiku) - a
model grading its own output biases the eval.

Two judgements:
  - faithfulness: given ONLY the answer + the retrieved chunks shown to the
    generator, is every claim supported by those chunks? Per-claim, FAIL CLOSED
    (default to unfaithful when unsure - this is a safety metric).
  - answer-correctness: does the answer match the reference answer_key?

These prompts are v1 drafts. Per the spec they must be tightened on ~10 real
baseline outputs and validated against a human-graded sample before the numbers
are trusted; the agreement rate is recorded in eval/results/.
"""
from __future__ import annotations

import json
import re

import anthropic
from dotenv import load_dotenv

from src.config import Config

load_dotenv()

_client: anthropic.Anthropic | None = None

FAITHFULNESS_PROMPT_VERSION = "faithfulness-v1"
CORRECTNESS_PROMPT_VERSION = "correctness-v3"

FAITHFULNESS_SYSTEM = """You grade the FAITHFULNESS of an answer to its evidence.

You are given (1) an ANSWER and (2) the CONTEXT passages that were shown to the \
system that wrote it. You are NOT given the "correct" answer - do not use outside \
or medical knowledge of your own.

Definition: an answer is FAITHFUL if every factual claim it makes is directly \
stated in, or directly entailed by, the CONTEXT. It is UNFAITHFUL if any claim \
relies on outside/parametric knowledge, adds specifics not present in the context, \
or generalises beyond it.

Procedure:
1. Break the answer into its distinct factual claims (ignore the "Sources:" list \
and generic safety boilerplate).
2. For each claim, find a context sentence that supports it, or mark it unsupported.
3. If ANY claim is unsupported, the answer is unfaithful.
4. If you are unsure whether a claim is supported, treat it as UNSUPPORTED \
(fail closed).

A correct refusal ("I can't answer that from the provided MHRA sources...") is \
FAITHFUL by definition - it asserts no facts.

Respond with ONLY a JSON object:
{"faithful": true|false, "unsupported_claims": ["..."], "reason": "one sentence"}"""

CORRECTNESS_SYSTEM = """You grade whether a candidate ANSWER covers what the \
question asked, judged against a reference ANSWER KEY (ground truth). You judge \
ONLY against the ANSWER KEY - not against any source text. Do not use outside \
knowledge.

Definition - an answer is CORRECT only if BOTH hold:
(a) COVERAGE: every central fact in the ANSWER KEY appears in the answer, stated \
or unambiguously entailed; and
(b) NON-CONTRADICTION: nothing in the answer contradicts the ANSWER KEY.

Treat each distinct claim in the ANSWER KEY as central unless it is explicitly \
tagged "(secondary)".

Procedure (do this in order, and show it):
1. List the ANSWER KEY's central facts.
2. For each, mark present or MISSING in the candidate answer.
3. Assign exactly one label:
   - "correct"    - (a) and (b) both hold (all central facts present, no contradiction).
   - "incomplete" - no contradiction, but at least one central fact is MISSING.
   - "wrong"      - the answer contradicts the answer key.
   (If both a contradiction and a missing fact are present, label "wrong".)

Rules:
- Additional detail beyond the ANSWER KEY does NOT affect correctness, as long as \
it does not contradict the key. (Its accuracy is judged elsewhere, not here.) \
Adding correct extra detail is never an omission - never label such an answer \
"incomplete" for containing more than the key.
- A refusal ("I can't answer that...") on an answerable question is "incomplete" \
(it supplied none of the required facts).
- When unsure whether a missing item is central, default to CENTRAL -> fail \
(fail-closed: in a safety domain an omission is a failure until shown harmless).

Respond with ONLY a JSON object, so the verdict is auditable:
{"label": "correct"|"incomplete"|"wrong", "central_facts": ["..."], \
"missing_facts": ["..."], "contradictions": ["..."], "reason": "one sentence"}"""


def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic()
    return _client


def _ask_json(system: str, user: str, cfg: Config) -> dict:
    """Call the judge model and parse the JSON object from its reply."""
    resp = _get_client().messages.create(
        model=cfg.llm.judge_model,
        max_tokens=600,
        temperature=0.0,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    text = "".join(b.text for b in resp.content if b.type == "text")
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return {}


def judge_faithfulness(answer: str, chunks: list[dict], cfg: Config) -> dict:
    """Faithful iff every claim is supported by the shown chunks. Fail closed."""
    context = "\n\n".join(
        f"[{i}] {c['text']}" for i, c in enumerate(chunks, 1)
    ) or "(no passages)"
    user = f"CONTEXT:\n{context}\n\nANSWER:\n{answer}"
    out = _ask_json(FAITHFULNESS_SYSTEM, user, cfg)
    # Fail closed: anything other than an explicit true is unfaithful.
    return {
        "faithful": out.get("faithful") is True,
        "unsupported_claims": out.get("unsupported_claims", []),
        "reason": out.get("reason", "(judge returned no/!parseable verdict - failing closed)"),
    }


def judge_correctness(question: str, answer_key: str, answer: str, cfg: Config) -> dict:
    user = (
        f"QUESTION:\n{question}\n\nANSWER KEY (ground truth):\n{answer_key}\n\n"
        f"CANDIDATE ANSWER:\n{answer}"
    )
    out = _ask_json(CORRECTNESS_SYSTEM, user, cfg)
    label = out.get("label")
    if label not in ("correct", "incomplete", "wrong"):
        label = "wrong"  # fail closed on an unparseable/absent verdict
    return {
        "label": label,                       # correct | incomplete | wrong
        "correct": label == "correct",        # headline pass/fail
        "missing_facts": out.get("missing_facts", []),
        "contradictions": out.get("contradictions", []),
        "reason": out.get("reason", "(judge returned no/!parseable verdict - failing closed)"),
    }
