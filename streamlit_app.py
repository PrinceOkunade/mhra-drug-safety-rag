"""Streamlit demo for the Grounded Medical RAG system - provenance-first, not a chatbot.

Layout, top to bottom:
  1. Header (flat, solid - no gradient).
  2. ASK BOX at the top: type a question and run it live (no scrolling to find the input).
  3. Two framed columns below: example questions on the LEFT; the selected/answered
     question's grounded answer and retrieved provenance on the RIGHT, each in its own
     bordered frame so the three areas are clearly separated.
  4. Methodology strip (measured results).

Example answers come from demo_cache.json (instant, free, no key). A typed question runs
the real pipeline - on the deployment's key if configured (Streamlit secret), else the
visitor's own key.

Run:  streamlit run streamlit_app.py
"""
from __future__ import annotations

import json
import os
import time

import streamlit as st

try:
    from dotenv import load_dotenv
    load_dotenv()  # local dev: pick up ANTHROPIC_API_KEY from .env
except Exception:
    pass

from src.config import ROOT

CACHE_PATH = ROOT / "demo_cache.json"


@st.cache_data
def load_cache() -> dict:
    if not CACHE_PATH.exists():
        return {}
    with open(CACHE_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_articles() -> dict:
    from src.config import ARTICLES_PATH
    lut = {}
    with open(ARTICLES_PATH, "r", encoding="utf-8") as f:
        for line in f:
            a = json.loads(line)
            lut[a["source_url"]] = {"title": a.get("title", ""), "date": a.get("published_date", "")}
    return lut


def inject_style() -> None:
    """Inter font + flat header + bordered frames. Keep the Material icon font intact
    (a blanket font override breaks Streamlit's expander chevrons)."""
    st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');
html, body, .stApp, [data-testid="stAppViewContainer"], h1,h2,h3,h4,h5,h6,
p, li, a, label, button, input, textarea, select, .stMarkdown,
[data-testid="stMarkdownContainer"], [data-testid="stMetricValue"], [data-testid="stMetricLabel"] {
  font-family: 'Inter', -apple-system, 'Segoe UI', system-ui, sans-serif !important;
}
[data-testid="stIconMaterial"], [data-testid="stExpanderIcon"], .material-symbols-rounded {
  font-family: 'Material Symbols Rounded','Material Symbols Outlined' !important;
}
.block-container { padding-top: 1.4rem; max-width: 1280px; }

/* Header - flat, solid, clear hierarchy (no gradient) */
.badge{ display:inline-block; font-size:.7rem; font-weight:700; letter-spacing:.14em;
  color:#2563EB; background:rgba(37,99,235,.10); border:1px solid rgba(37,99,235,.28);
  padding:.28rem .65rem; border-radius:6px; margin-bottom:.7rem; }
h1.title{ font-size:2.3rem; font-weight:900; letter-spacing:-.025em; margin:0 0 .3rem; color:#111927; }
.title .accent{ color:#2563EB; }
.sub{ color:#556072; font-size:1rem; max-width:760px; margin:0 0 .9rem; }
.pills{ display:flex; flex-wrap:wrap; gap:.45rem; }
.pill{ font-size:.78rem; font-weight:600; padding:.32rem .7rem; border-radius:6px;
  background:#EEF2F7; border:1px solid rgba(15,30,55,.10); color:#475569; }
.pill.key{ background:rgba(37,99,235,.10); border-color:rgba(37,99,235,.30); color:#2563EB; }

/* Section labels - the small uppercase eyebrow that titles each region */
.eyebrow{ font-size:.72rem; font-weight:800; letter-spacing:.13em; color:#64748B;
  text-transform:uppercase; margin:0 0 .5rem; }
.cardh{ font-size:.98rem; font-weight:800; margin:0 0 .55rem; display:flex; align-items:center; gap:.4rem; }
.cardh.ans{ color:#2563EB; } .cardh.prov{ color:#1E40AF; } .cardh.ask{ color:#111927; }

/* Bordered FRAMES for the three areas (Streamlit container border=True) */
[data-testid="stVerticalBlockBorderWrapper"]{
  background:#FFFFFF; border:1px solid #E3E8EF !important; border-radius:14px;
  padding:1rem 1.15rem; box-shadow:0 10px 26px -22px rgba(15,30,55,.28);
}
/* The ask frame stands out: subtle blue edge */
.askwrap [data-testid="stVerticalBlockBorderWrapper"]{ border-color:#BFD3F5 !important;
  background:#FAFCFF; box-shadow:0 12px 26px -20px rgba(37,99,235,.35); }

/* Question list buttons: left-aligned, roomy, clear hover/active */
.stButton button{ text-align:left; justify-content:flex-start; white-space:normal; height:auto;
  padding:.58rem .8rem; border-radius:9px; font-weight:500; font-size:.9rem; line-height:1.35;
  border:1px solid #E3E8EF; background:#FFFFFF; }
.stButton button:hover{ border-color:#2563EB; color:#2563EB; background:#EFF4FF; }

.qtitle{ font-size:1.15rem; font-weight:800; letter-spacing:-.01em; margin:.1rem 0 .5rem; color:#111927; }
.answer h3{ font-size:1.02rem; margin:.8rem 0 .35rem; } .answer h4{ font-size:.94rem; margin:.7rem 0 .25rem; }
[data-testid="stExpander"]{ border-radius:9px; border-color:#E3E8EF; }
a{ color:#2563EB; }
hr{ margin:1.3rem 0; }
</style>
""", unsafe_allow_html=True)


def render_answer(entry: dict) -> None:
    if entry.get("adversarial"):
        st.warning("Not answerable from the corpus - the system should **refuse**, not guess.", icon="🛡️")
    st.markdown(entry["answer"])
    if entry.get("sources"):
        st.markdown("**Cited sources**")
        for u in entry["sources"]:
            st.markdown(f"- [{u.rsplit('/', 1)[-1][:70]}...]({u})")


def render_chunks(entry: dict) -> None:
    chunks = entry.get("chunks", [])
    st.caption(f"{len(chunks)} chunks retrieved | hybrid dense + BM25, top-k")
    for i, c in enumerate(chunks, 1):
        date = f" | {c['date']}" if c.get("date") else ""
        with st.expander(f"#{i} | score {c['score']}{date} | {c['title'][:44]}", expanded=(i == 1)):
            st.caption(f"[source article]({c['source_url']})")
            txt = c["text"]
            st.write(txt[:800] + ("..." if len(txt) > 800 else ""))


def render_detail(entry: dict, latency: float | None) -> None:
    """RIGHT side: question title, latency (if live), then answer + provenance frames."""
    st.markdown(f'<div class="qtitle">{entry["question"]}</div>', unsafe_allow_html=True)
    if latency is not None:
        a, b = st.columns(2)
        a.metric("Response time", f"{latency:.1f} s")
        b.metric("Chunks retrieved", len(entry.get("chunks", [])))
    # Stacked (not side by side): answer frame first, then provenance frame below it.
    with st.container(border=True):
        st.markdown('<div class="cardh ans">💬 Grounded answer</div>', unsafe_allow_html=True)
        render_answer(entry)
    with st.container(border=True):
        st.markdown('<div class="cardh prov">🔍 Retrieved provenance</div>', unsafe_allow_html=True)
        render_chunks(entry)


def run_live(question: str, api_key: str) -> tuple[dict, float]:
    """Run the real pipeline for a typed question; return (entry, latency_seconds)."""
    import src.generate.llm as llm
    from src.config import load_config
    from src.generate.answer import answer_question

    os.environ["ANTHROPIC_API_KEY"] = api_key.strip()
    llm._client = None
    articles = load_articles()

    with st.spinner("Retrieving + generating (first run loads the models, ~1-2 min)..."):
        cfg = load_config()
        t0 = time.perf_counter()
        res = answer_question(question, cfg)
        dt = time.perf_counter() - t0
    entry = {
        "question": question, "adversarial": False,
        "answer": res["answer"], "sources": res["sources"],
        "chunks": [{"title": articles.get(h["source_url"], {}).get("title", ""),
                    "date": articles.get(h["source_url"], {}).get("date", ""),
                    "score": round(float(h.get("score", 0.0)), 4),
                    "source_url": h["source_url"], "text": h["text"]} for h in res["hits"]],
    }
    return entry, dt


# ================================ PAGE ==========================================
st.set_page_config(page_title="Grounded Medical RAG", page_icon="💊", layout="wide")
inject_style()

cache = load_cache()
examples = cache.get("examples", [])
_owner_key = None
try:
    _owner_key = st.secrets.get("ANTHROPIC_API_KEY")
except Exception:
    _owner_key = None
_owner_key = _owner_key or os.getenv("ANTHROPIC_API_KEY")

# ---- 1. Header (flat) ----
st.markdown(
    '<span class="badge">RETRIEVAL-AUGMENTED | MHRA DRUG SAFETY</span>'
    '<h1 class="title">Grounded <span class="accent">Medical RAG</span></h1>'
    '<p class="sub">Ask about UK drug-safety advice and see the exact source passages behind every '
    'answer - grounded, cited, and refusing to guess when the evidence isn\'t there.</p>'
    '<div class="pills"><span class="pill">143 MHRA articles</span>'
    '<span class="pill">81% top-1 retrieval</span>'
    '<span class="pill key">100% faithful (64/64)</span>'
    '<span class="pill key">16/16 adversarial refused</span></div>',
    unsafe_allow_html=True,
)
st.divider()

if not examples:
    st.error("No demo cache found. Run `python build_cache.py` first to generate `demo_cache.json`.")
    st.stop()
if "current" not in st.session_state:
    st.session_state.current = examples[0]
    st.session_state.latency = None

# ---- 2. ASK BOX at the very top ----
st.markdown('<div class="askwrap">', unsafe_allow_html=True)
with st.container(border=True):
    st.markdown('<div class="cardh ask">🔎 Ask your own question</div>', unsafe_allow_html=True)
    with st.form("askform", clear_on_submit=False):
        if _owner_key:
            q_in = st.text_input("Question", placeholder="e.g. What cardiac risk is linked to omega-3 medicines?",
                                 label_visibility="collapsed")
            key_in = _owner_key
        else:
            key_in = st.text_input("Anthropic API key (sk-ant-...)", type="password")
            q_in = st.text_input("Question", placeholder="Type a UK drug-safety question...",
                                 label_visibility="collapsed")
        submitted = st.form_submit_button("Ask", type="primary")
    st.caption("Runs the real pipeline (hybrid retrieval + Claude), not a cache. Or pick a curated "
               "example below - those are pre-computed, instant, and free.")
st.markdown('</div>', unsafe_allow_html=True)

if submitted and q_in and key_in:
    try:
        entry, dt = run_live(q_in, key_in)
        st.session_state.current = entry
        st.session_state.latency = dt
    except Exception as exc:
        st.error(f"Live query failed: {exc}")
elif submitted and not key_in:
    st.warning("Add an Anthropic API key to run a live question, or click a curated example below.")

st.divider()

# ---- 3. Two framed columns: questions LEFT, answer + provenance RIGHT ----
left, right = st.columns([1, 2.5], gap="large")
with left:
    with st.container(border=True):
        st.markdown('<div class="eyebrow">Curated examples</div>', unsafe_allow_html=True)
        for i, ex in enumerate(examples):
            label = ("🛡️ " if ex["adversarial"] else "") + ex["question"]
            if st.button(label, key=f"q{i}", use_container_width=True):
                st.session_state.current = ex
                st.session_state.latency = None
with right:
    render_detail(st.session_state.current, st.session_state.latency)

# ---- 4. Methodology ----
st.divider()
st.markdown('<div class="eyebrow">How it is measured</div>', unsafe_allow_html=True)
st.caption("Self-authored 64-question gold set (+16 adversarial), each verified against a source passage, "
           "judged by a stronger model (Claude Sonnet) than the generator.")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Answer correctness", "63 / 64")
m2.metric("Faithfulness", "64 / 64")
m3.metric("Refusal on adversarial", "16 / 16", help="0 false refusals, 0 false answers")
m4.metric("Top-1 retrieval (recall@1)", "81%")
st.caption("Honest negatives (measured, then rejected): header-aware chunking underperformed the baseline; "
           "a cross-encoder reranker added ~15x latency with no gain, so it was cut.")
if cache.get("config"):
    c = cache["config"]
    st.caption(f"Champion | {c['retriever']} retrieval | {c['embedder'].split('/')[-1]} | {c['generator']} | "
               "not medical advice.")
