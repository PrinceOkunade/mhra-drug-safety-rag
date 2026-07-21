# Grounded Medical RAG - Retrieval-Augmented QA over UK Drug-Safety Updates

*Python | sentence-transformers (BGE) | FAISS | BM25 | cross-encoder reranker | Claude | LLM-as-judge evaluation*

**> Live demo:** https://huggingface.co/spaces/PrinceOkunade/MHRA-Drug-Safety-RAG - click a question to see the grounded answer beside the exact source passages it came from.

A retrieval-augmented question-answering system over **UK MHRA Drug Safety
Updates**. The portfolio point is not "a chatbot" - it is **evaluated retrieval
engineering**: a deliberately naive baseline that improves *one measured change at
a time*, where every change (including the ones that failed) is gated by a
reproducible measurement against a human-certified benchmark.

> **This is an information-retrieval demonstrator over MHRA source text. It is NOT
> medical advice.** Every answer is grounded in and cited to gov.uk source
> passages; when the sources don't support an answer, the system refuses rather
> than guessing. Always consult the original source and a healthcare professional.


---

## The headline: the discipline, not a green table

The single most important result in this project is a **negative one**, because
it shows the method works:

> **Experiment 1 (header-aware chunking) - ran it, it regressed, I diagnosed it,
> fixed the diagnosed bug, re-ran it clean, and it *still* lost.**
> Splitting articles on their `<h2>` structure *dropped* recall@1 from 70% to 66%.
> Rather than discard it, I broke down which questions moved: 7 of 8 rank-1 losses
> went to a **shorter** chunk, and 6 of 8 to a **sibling section of the same
> article** - two questions lost to a 40-token stub containing only a drug name and
> "Download document". That pinned the mechanism: pathological tiny chunks win
> spuriously on cosine similarity. I added a **min-section-token floor** (set once
> on a structural principle - *a real clinical statement doesn't fit in <64 tokens*
> - never tuned against the test set), which merged the stubs and returned q071/q075
> to rank 1 - **confirming the diagnosis**. Yet the corrected version *still* didn't
> beat fixed chunking (recall@1 69%), because substantial sibling sections compete
> for the query. So it was **rejected on the evidence.**

A green results table can be produced by trying things until something works. This
project is built to show the opposite: *measured, reproducible, honestly-reported
change* - including two techniques rejected and one null.

---

## Results - baseline -> every experiment (same gold set, raw counts + %)

Retrieval metrics over the **64 answerable** gold questions. **Win-threshold,
declared before each run:** a win moves **>=2-3 questions on recall@1 or >=0.02 MRR**.
+/-1 question is noise (two identical runs differed by 1); recall@3+ is near-saturated
and not judged.

| experiment | recall@1 | recall@3 | MRR | median latency | verdict |
|---|---|---|---|---|---|
| Phase 1 baseline (fixed chunking) | 70% (45/64) | 94% (60/64) | 0.826 | - | denominator |
| Exp 1 - header-aware + min-section floor | 69% (44/64) | 89% (57/64) | 0.803 | - | **rejected** - diagnosed negative |
| Exp 2 - small-to-big (parent-document) | 73% (47/64) | 97% (62/64) | 0.853 | - | adopted |
| Exp 3 - hybrid search (dense + BM25, RRF) | 81% (52/64) | 97% (62/64) | 0.890 | **0.19 s** (n=20) | best retrieval |
| Exp 4 - metadata schema (register / date) | - | - | - | - | **null** - 6/6 = 6/6 on adversarial probes |
| Exp 5a - off-the-shelf cross-encoder rerank | 80% (51/64) | 100% (64/64) | 0.891 | 39.01 s (n=8) | **rejected for deploy** - 15x latency, no quality gain |
| Exp 5b - **fine-tuned** cross-encoder rerank | 72% (46/64) | 97% (62/64) | 0.838 | - | **rejected** - overfit / dist. mismatch |
| **CHAMPION - hybrid search (deployed)** | **81% (52/64)** | **97% (62/64)** | **0.890** | **0.19 s** (n=20) | **system-of-record** |

Latency is **retrieval-only median** on a 4-CPU box (`python -m src.eval.bench_latency`),
measured only for the two configurations that are actual serving candidates - the
rejected experiments were never latency-profiled, so those cells are honestly blank
rather than back-filled with guesses.

### The quality/latency tradeoff (measured, not estimated)

Profiling the deployed container surfaced a cost the quality metrics are blind to:
the cross-encoder rerank is paid **on every request**, not once at startup.

Both serving candidates now have a **full, committed generation-eval row** (not just
retrieval): the reranked candidate at `2026-07-20T223305_full.json`, the deployed
hybrid-alone champion at `2026-07-21T233340_full.json`.

| serving candidate | recall@1 | recall@3 | MRR | correctness | faithful | refusal | retrieval | **end-to-end** |
|---|---|---|---|---|---|---|---|---|
| **hybrid alone (deployed champion)** | **81%** | 97% | 0.890 | 98.4% | 100% | 16/16, 0 false | 0.19 s | **2.72 s** |
| hybrid -> reranker (rejected for deploy) | 80% | **100%** | 0.891 | 98.4% | 100% | 16/16, 0 false | 39.01 s | **41.54 s** |

The reranker is **99.5% of its own retrieval time** and buys **no measurable
quality** - retrieval sits inside the declared noise band (+1 recall@1, -2 recall@3,
-0.0008 MRR), and **generation is identical**: same correctness (98.4%, same single
incomplete question q005), same 100% faithfulness, same 0 false refusals / 0 false
answers, and **zero correctness-label changes across all 64 answerable questions**.
The recall@3 the reranker preserves (100% vs 97%) never reaches the generator - at
`top_k=5` those passages already enter the context either way.

So the honest, now fully-measured statement is: **the reranker costs 15x end-to-end
latency and delivers no measurable difference in answer quality or safety.** On the
evidence, hybrid-alone is the better *deployment* choice; the reranker earns its place
in the project as the honestly-reported ML component whose measured verdict is "not
worth it here" - the same discipline as the Exp 1 negative.

**Two more honest results worth calling out:**
- **Exp 5b (the ML component):** fine-tuning a cross-encoder on LLM-synthesised
  question-passage pairs *underperformed* the off-the-shelf model - diagnosed as
  overfitting + distribution mismatch + catastrophic forgetting. On a small
  specialised corpus, broad pretraining beats narrow synthetic fine-tuning.
- **Reranker, adopted then dropped:** stacking the two winners (hybrid + reranker)
  does **not** beat hybrid alone - both target the same rank-1 headroom, so the
  system hits an ~80% recall@1 ceiling. Phase 4 latency profiling then settled it:
  the reranker costs **15x end-to-end latency (41.54 s vs 2.72 s)** for **identical**
  generation quality and **zero** answer changes across all 64 questions, so it was
  **cut from the deployed system** and kept as a documented, measured negative.

## Deployed system - full committed row

The champion (**hybrid retrieval, no reranker**), scored end-to-end on the full
80-question gold set (`eval/results/2026-07-21T233340_full.json`):

| retrieval | generation (safety + quality) |
|---|---|
| recall@1 **81%**, recall@3 **97%**, MRR **0.890** | correctness **98.4%** (63/64) | faithfulness **100%** (64/64) | refusal **16/16** unanswerable | **0** false answers | **0** false refusals | end-to-end **~2.7 s** |

---

## How it works (the champion pipeline)

```
enumerate (gov.uk Search API)  ->  fetch HTML  ->  clean (strip page chrome)
  ->  fixed-size token chunks   ->  BGE embeddings  ->  FAISS (cosine)
  ->  HYBRID retrieve: dense + BM25, fused by Reciprocal Rank Fusion  ->  top-k
  ->  [optional] CROSS-ENCODER rerank - built and measured (Exp 5a), but
       reranker_enabled: false in the deployed config (15x latency, no quality gain)
  ->  prompt (answer only from context; cite sources; refuse if unsupported;
             keep HCP vs patient advice separated)  ->  Claude Haiku 4.5
  ->  grounded, cited answer (or an explicit refusal)
```

Every knob (chunking strategy, retriever type, RRF constant, reranker model,
top-k, LLM) lives in `config.yaml`, so the evaluation harness can sweep configs
without code changes. The reranker wraps **any** base retriever (dense, hybrid, or
small-to-big) - that generality is what let Exp 5a stack hybrid -> reranker and
measure it cleanly before it was cut; flipping `reranker_enabled: true` re-enables it.

## The evaluation harness (the measuring instrument)

- **80-question gold set** (64 answerable + 16 unanswerable), **human-certified**
  in batches against a **pinned corpus snapshot**, with **chunking-independent
  ground truth** (source URL + verbatim passage, never chunk IDs) so retrieval
  changes are compared fairly.
- **LLM-as-judge** (Claude Sonnet 4.6 - a *different, stronger* model than the
  Haiku generator, to avoid self-grading bias) scoring answer-correctness
  (correct / incomplete / wrong), faithfulness, and refusal-correctness. The judge
  was **validated against 27 of my own manual labels** - it initially disagreed on
  one (graded a fact-omitting answer "correct"); I tightened its prompt, re-checked,
  and added adversarial negative-control probes before trusting it.
- **Metrics:** recall@k and MRR (retrieval, no LLM, deterministic) + correctness /
  faithfulness / refusal (generation, judged).

---

## Setup (Windows / PowerShell)

```powershell
cd "$env:USERPROFILE\Desktop\claude\RAG Project"
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env     # then paste your ANTHROPIC_API_KEY into .env
```

Generation calls **Claude Haiku 4.5** (needs an `ANTHROPIC_API_KEY`, pay-as-you-go;
a full eval run is cents). Embeddings, BM25, and the reranker run locally on CPU.

## Usage

```powershell
python -m src.cli build         # enumerate -> fetch -> clean -> chunk -> index
python -m src.cli ask "What cardiac risk is linked to omega-3 ethyl ester medicines?"
python -m src.cli smoke-test    # qualitative check over hand-written questions
python -m src.eval.run          # full evaluation harness -> eval/results/<row>.json
python -m src.eval.run --retrieval-only   # recall@k + MRR only (free, no LLM)
pytest -q                       # unit tests (cleaning + chunking + matching)
```

**As an API (Phase 4):** the champion pipeline is served by FastAPI -
`POST /ask`, `GET /health`, `GET /`. Models + index warm once at startup.

```powershell
uvicorn src.serve.app:app --port 8080
# POST {"question": "..."} to /ask -> {answer, sources, disclaimer}
```

Containerized for **Google Cloud Run** (`Dockerfile` bakes the prebuilt index +
models; the `ANTHROPIC_API_KEY` is a runtime secret, never baked). Build + deploy
in **[`DEPLOY.md`](DEPLOY.md)**.

## Demo UI (Streamlit)

A thin, **provenance-first** demo: each answer is shown next to the exact retrieved
chunks that produced it (article title, date, retrieval score). Example questions are
**pre-computed** into `demo_cache.json`, so the public demo runs **instantly, free, and
with no API key** - clicking an example never calls Claude. An optional *live query*
lets a visitor run their own question on their **own** Anthropic key.

```powershell
python build_cache.py                # one-time: freeze the 10 example responses -> demo_cache.json
streamlit run streamlit_app.py       # opens http://localhost:8501
```

`build_cache.py` runs each curated example (7 answerable + 3 adversarial, all drawn from
the gold set) through the real `answer_question` pipeline once and caches the answer +
retrieved chunks. The app itself imports no key and, for the examples, loads no models.

### Deploy to Hugging Face Spaces (CPU free tier)

The demo is built to run on the **free CPU tier without exposing a key** - the cached
examples are static JSON and load no models, so startup is fast and there is no bill.

**1. Files the Space needs.** For the cached examples (the demo itself):
`streamlit_app.py`, `demo_cache.json`, `.streamlit/config.toml`, `requirements.txt`,
and `src/` (for `src.config`). To also enable the optional live query, add
`data/index/` and `data/processed/articles.jsonl` - both are `.gitignore`d by default
(rebuildable), so `git add -f` them into the Space. Examples work fine without them.

**2. Add this front-matter to the top of the Space's `README.md`** (Spaces reads it as
config):

```yaml
---
title: Grounded Medical RAG
emoji: 💊
colorFrom: teal
colorTo: indigo
sdk: streamlit
app_file: streamlit_app.py
pinned: false
license: mit
---
```

**3. Leave `ANTHROPIC_API_KEY` unset.** The example answers need no key; only the opt-in
*live query* does, and it uses the **visitor's own** key (used for that request only,
never stored). Never add your key as a secret on a public Space.

**Free-tier notes:**
- `requirements.txt` pins CPU-only torch (`--extra-index-url .../whl/cpu`), so the build
  stays light. First build takes a few minutes; examples then load instantly.
- The Space sleeps when idle; waking it just reloads the cached JSON (fast) - no model
  load, because examples are pre-computed.
- Live query on free CPU loads `bge-small` + the index on first use (~1-2 min) - fine as
  an optional extra, but the cached examples are the demo.

**Streamlit Community Cloud** works the same way: point it at `streamlit_app.py`, commit
the same files, leave the key unset.

## Data, licensing & safety

- **Corpus:** gov.uk Drug Safety Update articles (`content_store_document_type =
  drug_safety_update`), 2022-2026, HTML only - no third-party links, no legacy PDF
  bulletins.
- **Licence:** Contains public sector information licensed under the **Open
  Government Licence v3.0**. © Crown copyright / MHRA. Reused with attribution.
- **Safety properties (enforced + measured):** citation grounding, refusal on
  unsupported context (100% faithfulness, 0 false answers on 16 adversarial
  questions), and separation of healthcare-professional vs patient advice.

## How it was built - phased, with gates

Phase 1 (naive baseline) -> Phase 2 (evaluation harness) -> Phase 3 (measured
improvements, above) -> **Phase 4 (deployment: FastAPI service + Docker + Cloud Run
- see [`DEPLOY.md`](DEPLOY.md))**. Each phase stopped at a gate for review; the
naive Phase-1 baseline was never "improved" in place, because it is the denominator
every later number depends on.

## Repo layout

```
src/ingest    enumerate + fetch + clean + structure diff-check
src/chunk     fixed (baseline) | header_aware | small_to_big chunkers
src/index     embed + FAISS build
src/retrieve  dense | hybrid (BM25+RRF) | cross-encoder rerank
src/generate  prompt (safety contract) + metadata + Claude call + orchestration
src/rerank    synthetic-pair generation + cross-encoder fine-tuning (Exp 5)
src/eval      gold set, judge, metrics, snapshot, certification, probes, run
config.yaml   single source of truth for every knob
eval/results  one committed JSON row per experiment
```
