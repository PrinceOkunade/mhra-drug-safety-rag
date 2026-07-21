# Phase 4 - container for the champion Grounded Medical RAG service.
FROM python:3.12-slim

WORKDIR /app

# CPU-only torch FIRST (the default PyPI torch wheel bundles CUDA and is huge);
# sentence-transformers then sees torch already satisfied.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application code + the single source of truth.
COPY src ./src
COPY config.yaml .

# Bake the prebuilt retrieval artefacts (champion = fixed chunks + FAISS index).
# chunks.pkl + faiss.index are what the retriever loads; articles.jsonl backs the
# (optional) metadata module. Raw HTML and the rejected fine-tuned model are NOT
# copied (see .dockerignore).
COPY data/index ./data/index
COPY data/processed/articles.jsonl ./data/processed/articles.jsonl

# Pre-download the BGE embedder + off-the-shelf cross-encoder into the image's HF
# cache so cold starts don't reach out to Hugging Face.
RUN python -m src.serve.prewarm

# Hard-prove the baked cache is self-sufficient: after prewarm, forbid ALL Hugging
# Face network access. If anything were still resolving over the wire at startup,
# the container now fails loudly here instead of silently in a Cloud Run
# environment with restricted egress.
ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

# Cloud Run provides $PORT (default 8080). Bind 0.0.0.0.
ENV PORT=8080
EXPOSE 8080
# JSON (exec) form so uvicorn is PID 1 and receives SIGTERM directly - Cloud Run
# sends SIGTERM to drain an instance, and shell-form CMD can swallow it. The
# explicit `sh -c` keeps ${PORT} expansion, which JSON form alone would not do.
CMD ["sh", "-c", "exec uvicorn src.serve.app:app --host 0.0.0.0 --port ${PORT:-8080}"]
