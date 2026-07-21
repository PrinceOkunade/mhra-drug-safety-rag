"""Prompt assembly - encodes the Phase 1 safety contract.

The system prompt enforces the non-negotiable rules: answer only from the
provided context, cite source URLs, refuse when unsupported, and keep
healthcare-professional advice separate from patient advice.
"""
from __future__ import annotations

REFUSAL_TEXT = (
    "I can't answer that from the provided MHRA sources. "
    "The retrieved context does not contain supporting information."
)

SYSTEM_PROMPT = f"""You are an information-retrieval assistant for UK MHRA Drug \
Safety Updates. You are NOT a medical adviser and must not give medical advice.

Rules you must follow exactly:
1. Answer ONLY using the numbered CONTEXT passages provided in the user message. \
Do not use any outside or prior knowledge.
2. If the context does not contain enough information to answer, respond with \
exactly this and nothing else: "{REFUSAL_TEXT}"
3. Cite your sources: after the answer, list the source URL(s) of the passages \
you actually used, under a "Sources:" heading.
4. Keep "Advice for healthcare professionals" and "Advice for patients" clearly \
separated and labelled when both are relevant. Never merge them.
5. Be concise and factual. Do not speculate or add reassurance beyond the source.
"""

# Exp 4: extra rules that only make sense when per-passage metadata is shown.
_METADATA_RULES = """
6. Each passage is tagged with [register] and [date]. When the register is \
"HCP" the advice is for healthcare professionals and when it is "patient" it is \
for patients - attribute advice to the correct audience and never present \
HCP-only advice as patient advice.
7. When passages about the same topic carry different [date] values and their \
advice conflicts, follow the MOST RECENT passage (later date supersedes earlier) \
and say so.
"""


def get_system_prompt(cfg) -> str:
    """Base safety prompt, plus metadata-use rules when metadata is enabled."""
    if getattr(cfg.llm, "metadata_enabled", False):
        return SYSTEM_PROMPT + _METADATA_RULES
    return SYSTEM_PROMPT


def build_user_message(question: str, chunks: list[dict], cfg=None) -> str:
    """Format retrieved chunks + question into the user turn.

    With metadata enabled, each passage header carries its derived register/date/
    drug so the model can attribute audience and prefer recent advice.
    """
    use_meta = cfg is not None and getattr(cfg.llm, "metadata_enabled", False)
    if use_meta:
        from src.generate.metadata import enrich

    blocks = []
    for i, ch in enumerate(chunks, 1):
        if use_meta:
            m = enrich(ch)
            header = (
                f"[{i}] source_url: {ch['source_url']} | [register]: {m['register']} | "
                f"[date]: {m['date']} | [drug_or_device]: {m['drug_or_device']}"
            )
        else:
            header = f"[{i}] source_url: {ch['source_url']}"
        blocks.append(f"{header}\n{ch['text']}")

    context = "\n\n".join(blocks) if blocks else "(no passages retrieved)"
    return (
        "CONTEXT PASSAGES:\n"
        f"{context}\n\n"
        "QUESTION:\n"
        f"{question}\n\n"
        "Answer using only the context above, following all the rules."
    )
