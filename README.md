# Project 1: Hybrid RAG with reranking

Tenant-scoped question answering over company documents. A question goes through two retrievers
in **real Qdrant** (dense vector + BM25-weighted sparse vector), each restricted to the caller's
tenant, fused server-side with reciprocal-rank fusion (RRF), reranked, and answered with
validated `[C1]` citations. When nothing passes the rerank threshold the API abstains without
calling the LLM.

Runs offline with no API keys. The LLM, the dense embedding model and the cross-encoder are
deterministic stand-ins (see [Real vs simulated](docs/system-design.md#8-real-vs-simulated)).

## Quickstart

```bash
docker compose up -d --build          # rag-qdrant + rag-api, seed corpus ingested at startup
curl -s http://127.0.0.1:8101/healthz # {"status":"ok"}
./scripts/demo.sh                     # end-to-end curl walkthrough
docker compose run --rm rag-api pytest -q             # unit + integration tests (live Qdrant)
docker compose run --rm rag-api python scripts/ablation.py   # retrieval ablation, tiny demo set
docker compose down -v                # stop and remove the named volume
```

Swagger UI: http://127.0.0.1:8101/docs

## Demo walkthrough

Tokens (CONVENTIONS section 5): `token-acme-alice` (tenant `acme`), `token-globex-bob` (tenant `globex`).
The tenant comes from the token only; a `tenant_id` in the body is ignored.

```bash
# Same question, two tenants, two answers (5 days vs 10 days)
curl -s -X POST http://127.0.0.1:8101/v1/rag/ask \
  -H 'Authorization: Bearer token-acme-alice' -H 'Content-Type: application/json' \
  -d '{"question":"How many PTO days can I carry over into next year?"}'

# Exact identifier: the sparse (BM25) branch ranks the right chunk first
curl -s -X POST http://127.0.0.1:8101/v1/rag/ask \
  -H 'Authorization: Bearer token-acme-alice' -H 'Content-Type: application/json' \
  -d '{"question":"What does error DEPLOY-4021 mean?"}'

# Ingest text (or a PDF with -F file=@doc.pdf -F doc_id=my-doc)
curl -s -X POST http://127.0.0.1:8101/v1/rag/ingest \
  -H 'Authorization: Bearer token-acme-alice' -H 'Content-Type: application/json' \
  -d '{"doc_id":"acme-parking","text":"Parking permits renew every 6 months."}'
```

The response `debug` block shows the dense, sparse, server-fused, locally recomputed RRF and
reranked candidate lists, the rerank threshold, whether the LLM was called, and any chunks
flagged as instruction-like (prompt injection).

### Gateway mode (Project 2)

```bash
RAG_LLM_MODE=gateway docker compose up -d   # calls $RAG_LLM_BASE_URL/v1/chat/completions, model "smart"
```

If the gateway is down the API returns HTTP 502 `llm_gateway_error`; it never silently falls back to the mock.

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| `RAG_LLM_MODE` | `mock` | `mock` or `gateway` |
| `RAG_LLM_BASE_URL` | `http://gw-api:8000` | Project 2 gateway |
| `RAG_LLM_API_KEY` | `sk-demo-team-a` | Gateway virtual key (the combined stack uses `sk-demo-rag-service`, 300 RPM) |
| `RAG_RERANK_THRESHOLD` | `0.35` | Minimum rerank score for evidence; below it, abstain |

## Folder map

```text
app/main.py       FastAPI routes, bearer auth, PDF/text ingest validation
app/pipeline.py   ingest (chunk -> upsert -> manifest) and ask (retrieve -> rerank -> abstain/generate)
app/store.py      Qdrant collections, idempotent upsert + stale-chunk removal, hybrid prefetch + RRF query
app/encoders.py   hashed n-gram dense embedder (384-d), BM25 sparse encoder
app/ranking.py    local RRF function, reranker interface + simulated cross-encoder
app/llm.py        prompt building, mock + gateway LLM, citation validation, injection flags
app/text.py       tokenizer, page splitting, chunker, deterministic chunk IDs
data/corpus/      seed documents (<tenant>/<doc_id>.md)
data/eval_questions.jsonl   12 labeled questions for the ablation (tiny demo set)
scripts/          demo.sh, ablation.py
tests/            test_unit.py (no services), test_integration.py (live Qdrant)
docs/system-design.md       design doc (Hello Interview format)
```
