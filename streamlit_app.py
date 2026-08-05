"""Streamlit demo for the Grounded Medical RAG system - provenance-first, not a chatbot.

Master-detail layout (matches the deployed static demo): the example questions are a
clickable list on the LEFT; the selected question's grounded answer and the exact
retrieved chunks that produced it appear on the RIGHT (answer beside provenance). Light
theme. Example answers are served from demo_cache.json (instant, free, no key); an
optional live query runs the visitor's own question on their own Anthropic key.

Run:  streamlit run streamlit_app.py
"""
from __future__ import annotations

import json
import os
import time

import streamlit as st

try:
    from dotenv import load_dotenv
    load_dotenv()  # local dev: pick up ANTHROPIC_API_KEY from .env for the live-query box
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


def inject_style() -> None:
    """Inter font + light hero/cards/typography over the native light theme.

    NOTE: the font rule is scoped to text elements only, and the Material icon font is
    restored explicitly - a blanket font override breaks Streamlit's expander chevrons
    (they fall back to raw ligature text like 'arrow_right').
    """
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
h1,h2,h3 { font-weight: 800 !important; letter-spacing: -0.02em; }
.block-container { padding-top: 1.6rem; max-width: 1320px; }

/* Hero */
.hero{ position:relative; padding:2.2rem 2.3rem 1.9rem; border-radius:22px; overflow:hidden;
  border:1px solid rgba(15,30,55,.10); margin-bottom:.6rem;
  background:
    radial-gradient(1100px 380px at 12% -20%, rgba(13,148,136,.12), transparent 60%),
    radial-gradient(900px 420px at 92% 0%, rgba(124,58,237,.10), transparent 60%),
    linear-gradient(135deg,#EFFBFA 0%, #FFFFFF 70%);
  box-shadow:0 22px 50px -34px rgba(15,118,110,.35); }
.hero-badge{ display:inline-block; font-size:.72rem; font-weight:700; letter-spacing:.14em;
  color:#0F766E; background:rgba(13,148,136,.10); border:1px solid rgba(13,148,136,.28);
  padding:.3rem .7rem; border-radius:999px; margin-bottom:.9rem; }
.hero-title{ font-size:2.6rem; font-weight:900; line-height:1.05; letter-spacing:-.03em; margin:0;
  background:linear-gradient(92deg,#0D9488 0%, #2563EB 48%, #7C3AED 100%);
  -webkit-background-clip:text; background-clip:text; color:transparent; }
.hero-sub{ color:#4C596B; font-size:1.02rem; max-width:760px; margin:.7rem 0 1.1rem; }
.pills{ display:flex; flex-wrap:wrap; gap:.5rem; }
.pill{ font-size:.82rem; font-weight:600; padding:.36rem .78rem; border-radius:999px;
  background:#F1F5F9; border:1px solid rgba(15,30,55,.12); color:#475569; }
.pill.green{ background:rgba(13,148,136,.10); border-color:rgba(13,148,136,.30); color:#0F766E; }
.pill.violet{ background:rgba(124,58,237,.10); border-color:rgba(124,58,237,.30); color:#6D28D9; }

.eyebrow{ font-size:.75rem; font-weight:700; letter-spacing:.12em; color:#0F766E; text-transform:uppercase; margin-bottom:.4rem; }
.panel-h{ font-size:1.02rem; font-weight:800; margin:.1rem 0 .5rem; }
.panel-h.ans{ color:#0F766E; } .panel-h.prov{ color:#2563EB; }
.demo-note{ background:rgba(37,99,235,.06); border:1px solid rgba(37,99,235,.20); border-radius:12px;
  padding:.7rem 1rem; font-size:.86rem; color:#4C596B; margin:.4rem 0 1rem; line-height:1.5; }
.demo-note b{ color:#16202E; }

/* Question list buttons (left column): left-aligned, roomy, teal hover */
.stButton button{ text-align:left; justify-content:flex-start; white-space:normal; height:auto;
  padding:.62rem .85rem; border-radius:12px; font-weight:500; line-height:1.35;
  border:1px solid rgba(15,30,55,.14); }
.stButton button:hover{ border-color:#0F766E; color:#0F766E; background:#F0FBFA; }

[data-testid="stExpander"]{ border-radius:10px; border-color:rgba(15,30,55,.12); }
a{ color:#0F766E; }
</style>
""", unsafe_allow_html=True)


def render_hero() -> None:
    st.markdown("""
<div class="hero">
  <span class="hero-badge">RETRIEVAL-AUGMENTED | MHRA DRUG SAFETY</span>
  <div class="hero-title">Grounded Medical RAG</div>
  <p class="hero-sub">Ask about UK drug-safety advice and see the exact source passages behind every
  answer - grounded, cited, and refusing to guess when the evidence isn't there.</p>
  <div class="pills">
    <span class="pill">143 MHRA articles</span>
    <span class="pill">64-question gold set</span>
    <span class="pill green">100% faithful (64/64)</span>
    <span class="pill green">0 hallucinations</span>
    <span class="pill violet">16/16 adversarial refused</span>
  </div>
</div>
""", unsafe_allow_html=True)


def render_answer(entry: dict) -> None:
    if entry.get("adversarial"):
        st.warning("Adversarial question - not answerable from the corpus. Expect a **refusal**, not a guess.", icon="🛡️")
    st.markdown(entry["answer"])
    if entry.get("sources"):
        st.markdown("**Cited sources**")
        for u in entry["sources"]:
            st.markdown(f"- [{u.rsplit('/', 1)[-1][:80]}...]({u})")


def render_chunks(entry: dict) -> None:
    chunks = entry.get("chunks", [])
    st.caption(f"{len(chunks)} chunks retrieved | hybrid dense + BM25, top-k")
    for i, c in enumerate(chunks, 1):
        date = f" | {c['date']}" if c.get("date") else ""
        with st.expander(f"#{i} | score {c['score']}{date} | {c['title'][:48]}", expanded=(i == 1)):
            st.caption(f"[source article]({c['source_url']})")
            txt = c["text"]
            st.write(txt[:800] + ("..." if len(txt) > 800 else ""))


def show_result(entry: dict) -> None:
    """RIGHT side: question title, then answer beside provenance."""
    st.markdown(f"### {entry['question']}")
    ans, prov = st.columns(2, gap="medium")
    with ans:
        st.markdown('<div class="panel-h ans">💬 Grounded answer</div>', unsafe_allow_html=True)
        render_answer(entry)
    with prov:
        st.markdown('<div class="panel-h prov">🔍 Retrieved provenance</div>', unsafe_allow_html=True)
        render_chunks(entry)


def live_query(question: str, api_key: str) -> None:
    """Optional: run the visitor's own question on their OWN key, through the real pipeline."""
    import src.generate.llm as llm
    from src.config import ARTICLES_PATH, load_config
    from src.generate.answer import answer_question

    os.environ["ANTHROPIC_API_KEY"] = api_key.strip()
    llm._client = None

    articles = {}
    with open(ARTICLES_PATH, "r", encoding="utf-8") as f:
        for line in f:
            a = json.loads(line)
            articles[a["source_url"]] = {"title": a.get("title", ""), "date": a.get("published_date", "")}

    with st.spinner("Retrieving + generating (first run loads the models, ~1-2 min)..."):
        cfg = load_config()
        t0 = time.perf_counter()
        res = answer_question(question, cfg)
        dt = time.perf_counter() - t0
    # Per-query latency + retrieval size, so you can see how it performs live.
    m1, m2 = st.columns(2)
    m1.metric("Response time", f"{dt:.1f} s")
    m2.metric("Chunks retrieved", len(res["hits"]))
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

st.markdown(
    '<div class="demo-note">📋 These are <b>curated examples with cached answers</b>, so this preview '
    'loads instantly and needs no API key. The <b>full system</b> - live retrieval + generation and the '
    '<b>evaluation harness</b> that measured it - runs from the '
    '<a href="https://github.com/PrinceOkunade/mhra-drug-safety-rag" target="_blank">GitHub repo</a>. '
    '&#9888; Information-retrieval demonstrator - <b>NOT medical advice</b>.</div>',
    unsafe_allow_html=True,
)

if not examples:
    st.error("No demo cache found. Run `python build_cache.py` first to generate `demo_cache.json`.")
    st.stop()
if "selected" not in st.session_state:
    st.session_state.selected = 0

# ---- Master-detail: questions LEFT, answer + provenance RIGHT ----
left, right = st.columns([1, 2.4], gap="large")
with left:
    st.markdown('<div class="eyebrow">Pick a question</div>', unsafe_allow_html=True)
    for i, ex in enumerate(examples):
        label = ("🛡️ " if ex["adversarial"] else "") + ex["question"]
        if st.button(label, key=f"q{i}", use_container_width=True):
            st.session_state.selected = i
with right:
    show_result(examples[st.session_state.selected])

# ---- Methodology strip (bottom) ----
st.divider()
st.markdown('<div class="eyebrow">How it\'s measured</div>', unsafe_allow_html=True)
st.caption("Evaluated on a self-authored 64-question gold set (+16 adversarial), each verified against a "
           "source passage, and judged by a stronger model (Claude Sonnet) than the generator.")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Answer correctness", "63 / 64")
m2.metric("Faithfulness", "64 / 64")
m3.metric("Refusal on adversarial", "16 / 16", help="0 false refusals, 0 false answers")
m4.metric("Top-1 retrieval (recall@1)", "81%")
st.caption("Honest negatives (measured, then rejected): header-aware chunking underperformed the fixed "
           "baseline; a cross-encoder reranker added ~15x latency with no end-to-end gain, so it was cut.")
if cache.get("config"):
    c = cache["config"]
    st.caption(f"Champion | {c['retriever']} retrieval | {c['embedder'].split('/')[-1]} | {c['generator']}")

# ---- Live query: type your own question ----
# If a key is configured on the deployment (Streamlit secret or env), anyone can type a
# question and it runs on that key. Otherwise, visitors bring their own key.
st.divider()
st.markdown('<div class="eyebrow">Ask your own question (live)</div>', unsafe_allow_html=True)

_owner_key = None
try:
    _owner_key = st.secrets.get("ANTHROPIC_API_KEY")  # set in Streamlit Cloud -> Settings -> Secrets
except Exception:
    _owner_key = None
_owner_key = _owner_key or os.getenv("ANTHROPIC_API_KEY")

if _owner_key:
    st.caption("Type a UK drug-safety question and see the live answer, its cited sources, and the "
               "response time. This runs the real pipeline (hybrid retrieval + Claude), not a cache.")
    q = st.text_input("Your question", key="liveq", placeholder="e.g. What is the cardiac risk with omega-3 medicines?")
    if st.button("Run live query", type="primary", disabled=not q):
        try:
            live_query(q, _owner_key)
        except Exception as exc:
            st.error(f"Live query failed: {exc}")
else:
    with st.expander("🔑 Run your own question (bring your own Anthropic API key)"):
        st.caption("No key is configured on this deployment, so bring your own. It is used only for this "
                   "request, in this session - never stored or logged. The example answers above need no key.")
        key = st.text_input("Anthropic API key (sk-ant-...)", type="password")
        q = st.text_input("Your question about a UK drug-safety update")
        if st.button("Run live query", disabled=not (key and q)):
            try:
                live_query(q, key)
            except Exception as exc:
                st.error(f"Live query failed: {exc}")
