# Phase 4 - Deployment (Docker + Google Cloud Run)

Ships the champion pipeline (hybrid retrieval -> cross-encoder reranker -> grounded
Claude answer) as an HTTP API. Endpoints: `POST /ask`, `GET /health`, `GET /`.

## Prerequisites
- A prebuilt index on disk (`data/index/faiss.index` + `chunks.pkl`) - run
  `python -m src.cli build` once if missing.
- Docker, and (for Cloud Run) the `gcloud` CLI authenticated to a project with
  Cloud Run + Artifact Registry + Secret Manager enabled.
- An `ANTHROPIC_API_KEY` (provided at deploy time as a secret - never baked in).

## Run locally (Docker)

```bash
docker build -t grounded-medical-rag .
docker run --rm -p 8080:8080 -e ANTHROPIC_API_KEY=sk-ant-... grounded-medical-rag
# then:
curl localhost:8080/health
curl -X POST localhost:8080/ask -H 'Content-Type: application/json' \
  -d '{"question":"What cardiac risk is linked to omega-3 ethyl ester medicines?"}'
```

## Deploy to Cloud Run

```bash
PROJECT=your-gcp-project
REGION=europe-west2
REPO=rag
IMAGE=$REGION-docker.pkg.dev/$PROJECT/$REPO/grounded-medical-rag:latest

# 1. one-time: Artifact Registry repo + store the API key as a secret
gcloud artifacts repositories create $REPO --repository-format=docker --location=$REGION
printf 'sk-ant-...' | gcloud secrets create anthropic-api-key --data-file=-

# 2. build + push (Cloud Build reads this repo's Dockerfile)
gcloud builds submit --tag $IMAGE

# 3. deploy - models are large, so give it room and a generous cold-start timeout
gcloud run deploy grounded-medical-rag \
  --image $IMAGE --region $REGION --platform managed --allow-unauthenticated \
  --memory 4Gi --cpu 2 --timeout 300 --concurrency 8 \
  --set-secrets ANTHROPIC_API_KEY=anthropic-api-key:latest
```

`gcloud run deploy` prints the service URL; hit `<URL>/health` then `POST <URL>/ask`.

## Measured performance (4-CPU dev box, 2 torch threads)

Numbers below are measured inside the container, not estimated.

| stage | time | notes |
|---|---|---|
| `import torch` + sentence-transformers | ~87 s | dominates cold start |
| load BGE embedder (from baked cache) | ~5 s | no network |
| load cross-encoder (from baked cache) | ~4 s | no network |
| load FAISS + 484 chunks | 0.14 s | |
| build BM25 | **0.18 s** | negligible - do NOT pre-persist it |
| hybrid retrieve, no rerank (median, n=20) | **0.19 s** | p95 0.34 s |
| hybrid + rerank (median, n=8) | **39.01 s** | p95 47.09 s - **dominates every request** |
| generation, Claude call (median, n=3) | 2.53 s | p95 3.17 s |
| **end-to-end, hybrid alone** | **2.72 s** | |
| **end-to-end, champion (with rerank)** | **41.54 s** | |

- **Cold start is ~100-170 s**, not "a few seconds" - almost all of it the torch
  import. `--timeout 300` covers it; `--min-instances 1` avoids paying it repeatedly.
- **The reranker is the latency bottleneck and it is paid per request, not once.**
  This is a CPU-bound throughput problem, not a container problem: the host venv is
  equally slow. Three levers, in order of preference:
  1. **Serve hybrid-alone** (`reranker_enabled: false`). Exp 3 measured hybrid at
     recall@1 **81%** / MRR **0.890** vs the champion's **80%** / **0.891** - inside
     the declared noise band, i.e. *statistically indistinguishable*. Latency
     benchmark (`python -m src.eval.bench_latency`) puts end-to-end at **2.72 s vs
     41.54 s - a 15x speedup**. Cost: recall@3 drops 100% -> 97% (2 questions).
  2. **Lower `rerank_top_n`** (50 -> 10-20). Roughly linear speedup, but it changes a
     measured config and so requires an eval re-run before it can be claimed.
  3. **More CPU** (`--cpu 4` or higher). The rerank is CPU-bound; `--cpu 2` in the
     recipe above will be *slower* than these numbers.

  Options 1 and 2 change the deployed system-of-record and must be re-measured and
  re-committed to `eval/results/` before being adopted - they are not free wins.
- **Memory:** embedder + reranker + FAISS + chunks fit comfortably in 4 GiB.
- **Offline-proven:** the image sets `HF_HUB_OFFLINE=1` / `TRANSFORMERS_OFFLINE=1`
  *after* prewarm, so any residual network dependency fails loudly at startup rather
  than silently in a restricted-egress environment. Verified: container boots clean
  with both models loaded and no HF requests.
- **Secret, not baked:** the API key is mounted from Secret Manager at runtime;
  `GET /health` reports `api_key_present` without revealing it.
- **Champion config is fixed in `config.yaml`** (hybrid + off-the-shelf reranker);
  swap any knob there and rebuild to redeploy a different configuration.
