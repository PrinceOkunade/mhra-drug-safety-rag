"""Streamlit demo for the Grounded Medical RAG system - provenance-first, not a chatbot.

THE POINT OF THIS DEMO
----------------------
A reviewer spends ~60 seconds. The star is RETRIEVAL PROVENANCE: for each answer, show
the exact source chunks that produced it (article title, date, retrieval score), side by
side with the answer. Two columns:  LEFT = grounded answer + citations,  RIGHT = the
chunks it came from.

Example questions are served from `demo_cache.json` (pre-computed by build_cache.py) -
instant, free, no API key. An optional "live query" lets a visitor run their OWN question
on their OWN Anthropic key.

LOOK & FEEL: dark mode + Inter, a gradient hero, coloured section headers and card depth
(palette in .streamlit/config.toml, typography/hero/cards via the CSS block below).

STREAMLIT IN ONE SENTENCE: the script re-runs top-to-bottom on every interaction, so
"state" (which example is selected) lives in `st.session_state`, not in local variables.

Run:  streamlit run streamlit_app.py
"""
from __future__ import annotations

import json
import os

import streamlit as st

from src.config import ROOT

CACHE_PATH = ROOT / "demo_cache.json"


@st.cache_data  # load the JSON once, not on every rerun
def load_cache() -> dict:
    if not CACHE_PATH.exists():
        return {}
    with open(CACHE_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------- Look & feel (CSS) -----------------------------------
def inject_style() -> None:
    """Load Inter + layer a hero, gradient headings, and card depth over the dark theme."""
    st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

/* Typography - Inter on TEXT elements only. NOTE: do not use a broad [class*="st-"]
   selector here - it clobbers Streamlit's Material Symbols icon font and turns expander
   chevrons into raw ligature text ("arrow_right"). Target text tags, then restore the
   icon font explicitly below so icons always win. */
html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stSidebar"],
h1, h2, h3, h4, h5, h6, p, li, a, label, button, input, textarea, select,
.stMarkdown, [data-testid="stMarkdownContainer"], [data-testid="stMetricValue"],
[data-testid="stMetricLabel"], [data-testid="stExpander"] summary, [data-testid="stExpander"] p {
    font-family: 'Inter', -apple-system, 'Segoe UI', system-ui, sans-serif !important;
}
/* Restore Streamlit's Material icon font (expander chevrons, sidebar toggle, etc.).
   Verified testids for Streamlit 1.60: stIconMaterial, stExpanderIcon; font: Material Symbols Rounded. */
[data-testid="stIconMaterial"], [data-testid="stExpanderIcon"],
span[data-testid="stIconMaterial"], .material-symbols-rounded, .material-symbols-outlined {
    font-family: 'Material Symbols Rounded', 'Material Symbols Outlined' !important;
}
h1, h2, h3 { font-weight: 800 !important; letter-spacing: -0.02em; }
.block-container { padding-top: 1.4rem; max-width: 1240px; }

/* ---- Hero section ---- */
.hero {
    position: relative; margin: 0 0 1.4rem 0; padding: 2.6rem 2.4rem 2.2rem;
    border-radius: 22px; overflow: hidden;
    border: 1px solid rgba(255,255,255,.08);
    background:
        radial-gradient(1100px 380px at 12% -20%, rgba(45,212,191,.20), transparent 60%),
        radial-gradient(900px 420px at 92% 0%, rgba(129,140,248,.20), transparent 60%),
        linear-gradient(135deg, #101B33 0%, #0B1120 70%);
    box-shadow: 0 30px 70px -40px rgba(45,212,191,.55);
}
.hero-badge {
    display: inline-block; font-size: .72rem; font-weight: 700; letter-spacing: .14em;
    color: #5EEAD4; background: rgba(45,212,191,.10); border: 1px solid rgba(45,212,191,.30);
    padding: .3rem .7rem; border-radius: 999px; margin-bottom: 1rem;
}
.hero-title {
    font-size: 3rem; font-weight: 900; line-height: 1.05; letter-spacing: -0.03em; margin: 0;
    background: linear-gradient(92deg, #5EEAD4 0%, #38BDF8 45%, #A78BFA 100%);
    -webkit-background-clip: text; background-clip: text; color: transparent;
}
.hero-sub { color: #AEB9CC; font-size: 1.06rem; max-width: 720px; margin: .8rem 0 1.3rem; line-height: 1.55; }
.hero-pills { display: flex; flex-wrap: wrap; gap: .55rem; margin-bottom: 1rem; }
.pill {
    font-size: .82rem; font-weight: 600; padding: .38rem .8rem; border-radius: 999px;
    background: rgba(255,255,255,.05); border: 1px solid rgba(255,255,255,.12); color: #CBD5E1;
}
.pill-green { background: rgba(45,212,191,.12); border-color: rgba(45,212,191,.35); color: #5EEAD4; }
.pill-violet { background: rgba(167,139,250,.12); border-color: rgba(167,139,250,.35); color: #C4B5FD; }
.hero-note { color: #8A96AB; font-size: .86rem; }

/* ---- Coloured panel headers ---- */
.panel-h { font-size: 1.05rem; font-weight: 800; margin: .2rem 0 .6rem; letter-spacing: -0.01em; }
.panel-answer { color: #5EEAD4; }
.panel-prov   { color: #93C5FD; }

/* ---- Card depth for the two panels (st.container(border=True)) ---- */
[data-testid="stVerticalBlockBorderWrapper"] {
    background: linear-gradient(180deg, rgba(255,255,255,.02), rgba(255,255,255,0));
    border: 1px solid rgba(255,255,255,.08) !important; border-radius: 16px;
    padding: 1.1rem 1.2rem; box-shadow: 0 18px 40px -30px rgba(0,0,0,.8);
}

/* ---- Example-question buttons: left-aligned list items with a teal hover glow ---- */
.stButton button {
    text-align: left; justify-content: flex-start; white-space: normal; height: auto;
    padding: .7rem .95rem; border-radius: 12px; font-weight: 500; line-height: 1.35;
    border: 1px solid rgba(255,255,255,.10); background: #131C2E; color: #DCE4F0;
    transition: all .15s ease;
}
.stButton button:hover {
    border-color: #2DD4BF; color: #5EEAD4; background: #16233A;
    box-shadow: 0 0 0 1px rgba(45,212,191,.3), 0 10px 30px -18px rgba(45,212,191,.7);
}

/* ---- Section eyebrow + misc ---- */
.eyebrow { font-size: .78rem; font-weight: 700; letter-spacing: .12em; color: #7CE0D3; text-transform: uppercase; }
[data-testid="stExpander"] { border-radius: 12px; border-color: rgba(255,255,255,.08); }
a { color: #67E8F9; text-decoration: none; } a:hover { text-decoration: underline; }
[data-testid="stSidebar"] { border-right: 1px solid rgba(255,255,255,.06); }
</style>
""", unsafe_allow_html=True)


def render_hero() -> None:
    st.markdown("""
<div class="hero">
  <span class="hero-badge">RETRIEVAL-AUGMENTED | MHRA DRUG SAFETY</span>
  <h1 class="hero-title">Grounded Medical RAG</h1>
  <p class="hero-sub">Ask about UK drug-safety advice and see the exact source passages behind
  every answer - grounded, cited, and refusing to guess when the evidence isn't there.</p>
  <div class="hero-pills">
    <span class="pill">143 MHRA articles</span>
    <span class="pill">64-question gold set</span>
    <span class="pill pill-green">100% faithful (64/64)</span>
    <span class="pill pill-green">0 hallucinations</span>
    <span class="pill pill-violet">16/16 adversarial refused</span>
  </div>
  <div class="hero-note">⚠️ Information-retrieval demonstrator - <b>NOT medical advice</b>. Always consult the original source and a healthcare professional.</div>
</div>
""", unsafe_allow_html=True)


# ---------------------------- Result rendering ------------------------------------
def render_answer(entry: dict) -> None:
    """LEFT column: the grounded answer + its cited source URLs."""
    if entry.get("adversarial"):
        st.warning("Adversarial question - not answerable from the corpus. Expect a **refusal**, not a guess.", icon="🛡️")
    st.markdown(entry["answer"])
    if entry.get("sources"):
        st.markdown("**Cited sources**")
        for u in entry["sources"]:
            st.markdown(f"- [{u.rsplit('/', 1)[-1][:80]}...]({u})")


def render_chunks(entry: dict) -> None:
    """RIGHT column: the retrieved chunks (the provenance) that fed the answer."""
    chunks = entry.get("chunks", [])
    st.caption(f"{len(chunks)} chunks retrieved | hybrid dense + BM25, top-k")
    for i, c in enumerate(chunks, 1):
        date = f" | {c['date']}" if c.get("date") else ""
        with st.expander(f"#{i} | score {c['score']}{date} | {c['title'][:52]}", expanded=(i == 1)):
            st.caption(f"[source article]({c['source_url']})")
            txt = c["text"]
            st.write(txt[:800] + ("..." if len(txt) > 800 else ""))


def show_result(entry: dict) -> None:
    """The two-panel view: answer on the left, the chunks that produced it on the right."""
    st.markdown(f'<div class="eyebrow">Question</div>', unsafe_allow_html=True)
    st.subheader(entry["question"])
    left, right = st.columns(2, gap="large")
    with left:
        with st.container(border=True):
            st.markdown('<div class="panel-h panel-answer">💬 Grounded answer</div>', unsafe_allow_html=True)
            render_answer(entry)
    with right:
        with st.container(border=True):
            st.markdown('<div class="panel-h panel-prov">🔍 Retrieved provenance</div>', unsafe_allow_html=True)
            render_chunks(entry)


def live_query(question: str, api_key: str) -> None:
    """Optional: run the visitor's own question on their OWN key, through the real pipeline."""
    import src.generate.llm as llm
    from src.config import ARTICLES_PATH, load_config
    from src.generate.answer import answer_question

    os.environ["ANTHROPIC_API_KEY"] = api_key.strip()
    llm._client = None  # reset cached client so it uses the visitor's key

    articles = {}
    with open(ARTICLES_PATH, "r", encoding="utf-8") as f:
        for line in f:
            a = json.loads(line)
            articles[a["source_url"]] = {"title": a.get("title", ""), "date": a.get("published_date", "")}

    with st.spinner("Retrieving + generating (first run loads the models, ~1-2 min)..."):
        cfg = load_config()
        res = answer_question(question, cfg)
    show_result({
        "question": question, "adversarial": False,
        "answer": res["answer"], "sources": res["sources"],
        "chunks": [{"title": articles.get(h["source_url"], {}).get("title", ""),
                    "date": articles.get(h["source_url"], {}).get("date", ""),
                    "score": round(float(h.get("score", 0.0)), 4),
                    "source_url": h["source_url"], "text": h["text"]} for h in res["hits"]],
    })


# ================================ PAGE ==========================================
st.set_page_config(page_title="Grounded Medical RAG", page_icon="💊", layout="wide")
inject_style()
render_hero()

cache = load_cache()
examples = cache.get("examples", [])

# ---- Sidebar: the methodology panel (measured results, stated with n) ------------
with st.sidebar:
    st.markdown('<div class="eyebrow">How it\'s measured</div>', unsafe_allow_html=True)
    st.markdown(
        "Evaluated on a **hand-built 64-question gold set** (+16 adversarial), judged by a "
        "**stronger** model (Claude Sonnet) than the generator:"
    )
    st.metric("Answer correctness", "63 / 64")
    st.metric("Faithfulness - grounded, no hallucination", "64 / 64")
    st.metric("Refusal on adversarial", "16 / 16", help="0 false refusals, 0 false answers")
    st.divider()
    st.markdown("**Honest negatives** (measured, then rejected):")
    st.markdown(
        "- Header-aware chunking *underperformed* the fixed baseline.\n"
        "- A cross-encoder reranker added **~15x latency** with no end-to-end gain - so it "
        "was cut from the deployed system."
    )
    if cache.get("config"):
        c = cache["config"]
        st.divider()
        st.caption(f"Champion | {c['retriever']} retrieval | {c['embedder'].split('/')[-1]} | {c['generator']}")

# ---- Main: example buttons (primary), then the selected result -------------------
if not examples:
    st.error("No demo cache found. Run `python build_cache.py` first to generate `demo_cache.json`.")
    st.stop()

st.markdown('<div class="eyebrow">Try an example - pre-computed, so it\'s instant &amp; free</div>', unsafe_allow_html=True)
st.write("")
if "selected" not in st.session_state:
    st.session_state.selected = 0

cols = st.columns(2)
for idx, ex in enumerate(examples):
    label = ("🛡️ " if ex["adversarial"] else "") + ex["question"]
    if cols[idx % 2].button(label, key=f"ex{idx}", use_container_width=True):
        st.session_state.selected = idx

st.divider()
show_result(examples[st.session_state.selected])

# ---- Optional live query (bring your own key) ------------------------------------
st.divider()
with st.expander("🔑 Run your own question (bring your own Anthropic API key)"):
    st.caption("Your key is used only for this request, in this session - never stored or logged. "
               "The example answers above need no key.")
    key = st.text_input("Anthropic API key (sk-ant-...)", type="password")
    q = st.text_input("Your question about a UK drug-safety update")
    if st.button("Run live query", disabled=not (key and q)):
        try:
            live_query(q, key)
        except Exception as exc:
            st.error(f"Live query failed: {exc}")
