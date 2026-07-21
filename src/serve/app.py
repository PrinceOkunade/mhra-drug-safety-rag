"""FastAPI service wrapping the champion RAG pipeline - Phase 4 (deployment).

Exposes the exact `answer_question` pipeline the evaluation harness measured
(hybrid retrieval -> cross-encoder reranker -> grounded, cited, refusing Claude
answer). The heavy artefacts (BGE embedder, cross-encoder, FAISS index, chunks)
are loaded ONCE at startup and cached, so requests only pay for retrieval +
generation, not model loading.

Run locally:   uvicorn src.serve.app:app --port 8080
Container:     the Dockerfile sets this as the entrypoint on $PORT.
"""
from __future__ import annotations

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.config import load_config

DISCLAIMER = (
    "Information-retrieval demonstrator over UK MHRA Drug Safety Updates. "
    "NOT medical advice. Answers are grounded in and cited to gov.uk source "
    "passages; unsupported questions are refused. Always consult the original "
    "source and a healthcare professional."
)

_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load config + warm the retriever/models once, before serving traffic."""
    # Local dev: pick up ANTHROPIC_API_KEY from .env. In Cloud Run the real env var
    # is already set and load_dotenv() does not override it.
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except Exception:
        pass
    cfg = load_config()
    _state["cfg"] = cfg
    # Warm the FAISS index, embedder and cross-encoder with a throwaway query so
    # the first real request is fast (and a broken index fails fast at boot).
    try:
        from src.retrieve.dense import retrieve
        retrieve("warmup", cfg, k=1)
        _state["ready"] = True
    except Exception as exc:  # index not built yet, etc. - stay up for /health
        _state["ready"] = False
        _state["warm_error"] = str(exc)
    yield
    _state.clear()


app = FastAPI(
    title="Grounded Medical RAG",
    description=DISCLAIMER,
    version="1.0.0",
    lifespan=lifespan,
)


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=500,
                          examples=["What cardiac risk is linked to omega-3 ethyl ester medicines?"])


class AskResponse(BaseModel):
    question: str
    answer: str
    sources: list[str]
    disclaimer: str = DISCLAIMER


@app.get("/")
def root() -> dict:
    return {"service": "Grounded Medical RAG", "disclaimer": DISCLAIMER,
            "endpoints": {"POST /ask": "ask a question", "GET /health": "liveness"}}


@app.get("/health")
def health() -> dict:
    """Cloud Run liveness probe. Reports whether the index warmed successfully."""
    return {"status": "ok", "retriever_ready": _state.get("ready", False),
            "api_key_present": bool(os.getenv("ANTHROPIC_API_KEY"))}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    cfg = _state.get("cfg")
    if cfg is None or not _state.get("ready"):
        raise HTTPException(status_code=503,
                            detail=f"retriever not ready: {_state.get('warm_error', 'starting up')}")
    if not os.getenv("ANTHROPIC_API_KEY"):
        raise HTTPException(status_code=503, detail="ANTHROPIC_API_KEY not configured")
    from src.generate.answer import answer_question
    result = answer_question(req.question, cfg)
    return AskResponse(question=result["question"], answer=result["answer"], sources=result["sources"])
