"""Runtime settings and the static demo token table (stand-in for OIDC)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Principal:
    tenant_id: str
    user_id: str


# CONVENTIONS section 5. The tenant always comes from here, never from the request body.
TOKENS: dict[str, Principal] = {
    "token-acme-alice": Principal("acme", "alice"),
    "token-globex-bob": Principal("globex", "bob"),
}


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


@dataclass(frozen=True)
class Settings:
    qdrant_url: str = field(default_factory=lambda: _env("QDRANT_URL", "http://rag-qdrant:6333"))
    collection: str = field(default_factory=lambda: _env("RAG_COLLECTION", "rag_chunks"))
    manifest_collection: str = field(default_factory=lambda: _env("RAG_MANIFEST_COLLECTION", "rag_manifests"))
    corpus_dir: Path = field(default_factory=lambda: Path(_env("RAG_CORPUS_DIR", "/app/data/corpus")))
    llm_mode: str = field(default_factory=lambda: _env("RAG_LLM_MODE", "mock"))
    llm_base_url: str = field(default_factory=lambda: _env("RAG_LLM_BASE_URL", "http://gw-api:8000"))
    llm_api_key: str = field(default_factory=lambda: _env("RAG_LLM_API_KEY", "sk-demo-team-a"))
    rerank_threshold: float = field(default_factory=lambda: float(_env("RAG_RERANK_THRESHOLD", "0.35")))
    chunk_size: int = 900
    chunk_overlap: int = 140
    candidates_per_branch: int = 30   # top-N per retriever (prefetch limit)
    rerank_candidates: int = 30       # fused candidates sent to the reranker
    rrf_k: int = 60
    max_upload_bytes: int = 5 * 1024 * 1024
