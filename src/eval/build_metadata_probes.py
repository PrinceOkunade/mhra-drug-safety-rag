"""Build the Exp 4 metadata probe set (eval/metadata_probes.jsonl).

6 adversarial probes (5 register + 1 supersession), certified by the user. Each is
grounded in a real article: we resolve the exact source_url and pull a verbatim
grounding passage from the article body, so the probe is not free-floating text.
"""
from __future__ import annotations

import json
import re

from src.config import EVAL_DIR
from src.ingest.clean import read_articles

PROBES_PATH = EVAL_DIR / "metadata_probes.jsonl"

# hand-written, user-certified; each references an article by slug prefix.
PROBES = [
    dict(id="R1", property="register", register_expected="patient",
         prefix="metformin-and-reduced-vitamin-b12",
         question="I'm a patient taking metformin - what should I watch for regarding vitamin B12?",
         answer_key="Watch for symptoms of low vitamin B12: extreme tiredness, a sore and red tongue, pins and needles, and pale or yellow skin; you may need blood tests."),
    dict(id="R3", property="register", register_expected="patient",
         prefix="hydroxychloroquine-chloroquine",
         question="I take hydroxychloroquine - is there a risk if I also take antibiotics?",
         answer_key="Taking macrolide antibiotics at the same time can increase the risk of heart-related side effects; seek urgent medical help if you notice signs."),
    dict(id="R4", property="register", register_expected="patient",
         prefix="cladribine-mavenclad",
         question="I've started cladribine (Mavenclad) for MS - what liver risk should I know about as a patient?",
         answer_key="There is a risk of serious liver injury (uncommon, most often within 8 weeks of starting the first treatment); you should have blood tests to check your liver function."),
    dict(id="R5", property="register", register_expected="patient",
         prefix="dupilumab-dupixent",
         question="I'm on dupilumab (Dupixent) - what should I know about eye side effects?",
         answer_key="Dupilumab can cause eye side effects (especially in atopic eczema); do not try to self-manage new or worsening eye problems - seek medical advice."),
    dict(id="R6", property="register", register_expected="patient",
         prefix="brolucizumab",
         question="After my brolucizumab eye injection, what symptoms should make me seek urgent help?",
         answer_key="Seek advice from your eye-care team straight away if you have decreased or changed vision, eye pain, worsening eye redness, or sensitivity to light after the injection."),
    dict(id="S1", property="supersession", register_expected=None,
         prefix="isotretinoin-changes-to-prescribing",
         question="Have the isotretinoin prescribing / risk-minimisation measures changed?",
         answer_key="Yes - in 2026 the Commission on Human Medicines endorsed changes to the isotretinoin risk-minimisation measures, following a review of the measures introduced in 2023; the current position updates the 2023 measures."),
]


def _patient_passage(body: str) -> str:
    lines = body.split("\n")
    out, grab = [], False
    for ln in lines:
        low = ln.lower().strip()
        if not grab and re.search(r"to give to patients|advice for patients|patients or parents|patients and caregiver", low):
            grab = True
            continue
        if grab:
            if "advice for healthcare professional" in low or low.startswith("report ") or ("yellow card" in low and len(ln) < 70):
                break
            if ln.strip():
                out.append(ln.strip())
            if len(out) >= 3:
                break
    return " ".join(out)[:400]


def main() -> None:
    by_prefix = {}
    for a in read_articles():
        by_prefix.setdefault(a["slug"], a)

    def find(prefix):
        for slug, a in by_prefix.items():
            if slug.startswith(prefix):
                return a
        return None

    rows = []
    for p in PROBES:
        a = find(p["prefix"])
        if a is None:
            raise SystemExit(f"probe {p['id']}: no article for prefix {p['prefix']}")
        passage = _patient_passage(a["body_text"]) if p["property"] == "register" else a["summary"]
        rows.append({
            "id": p["id"],
            "property": p["property"],
            "register_expected": p["register_expected"],
            "question": p["question"],
            "answer_key": p["answer_key"],
            "supporting_passages": [{"source_url": a["source_url"], "passage_text": passage}],
            "article_date": a.get("published_date", ""),
        })

    with open(PROBES_PATH, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} probes -> {PROBES_PATH}")
    for r in rows:
        print(f"  {r['id']} [{r['property']}] {r['supporting_passages'][0]['source_url'].split('/')[-1][:45]}")


if __name__ == "__main__":
    main()
