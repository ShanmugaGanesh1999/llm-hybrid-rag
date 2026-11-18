"""Ingestion and question answering. The API layer (main.py) only handles HTTP and auth."""
from __future__ import annotations

import hashlib
import logging
import time
import uuid
from pathlib import Path

from app.config import Principal, Settings
from app.llm import (ABSTAIN_TOKEN, Evidence, GatewayLLM, MockLLM, build_messages,
                     looks_like_injection, validate_citations)
from app.ranking import Reranker, SimulatedCrossEncoder, rrf
from app.store import VectorStore
from app.text import chunk_id, chunk_text, extract_title, split_pages

log = logging.getLogger("rag")


class RagPipeline:
    def __init__(self, settings: Settings, reranker: Reranker | None = None, llm=None):
        self.s = settings
        self.store = VectorStore(settings.qdrant_url, settings.collection,
                                 settings.manifest_collection, settings.rrf_k)
        self.reranker = reranker or SimulatedCrossEncoder()
        if llm is not None:
            self.llm = llm
        elif settings.llm_mode == "gateway":
            self.llm = GatewayLLM(settings.llm_base_url, settings.llm_api_key)
        else:
            self.llm = MockLLM()

    # ---------- ingestion ----------
    def make_chunks(self, tenant_id: str, doc_id: str, pages: list[tuple[int, str]]) -> list[dict]:
        chunks = []
        for page, body in pages:
            for i, text in enumerate(chunk_text(body, self.s.chunk_size, self.s.chunk_overlap)):
                chunks.append({"chunk_id": chunk_id(tenant_id, doc_id, page, i),
                               "page": page, "chunk_index": i, "text": text})
        return chunks

    def ingest_pages(self, tenant_id: str, doc_id: str, title: str, source: str,
                     pages: list[tuple[int, str]]) -> dict:
        checksum = hashlib.sha256("\f".join(f"{n}:{t}" for n, t in pages).encode()).hexdigest()
        chunks = self.make_chunks(tenant_id, doc_id, pages)
        return self.store.upsert_document(tenant_id, doc_id, title, source, checksum, chunks)

    def ingest_text(self, tenant_id: str, doc_id: str, text: str, title: str | None = None,
                    source: str | None = None) -> dict:
        return self.ingest_pages(tenant_id, doc_id, title or extract_title(text, doc_id),
                                 source or f"{doc_id}.txt", split_pages(text))

    def ingest_corpus(self, corpus_dir: Path | None = None) -> list[dict]:
        """Seed corpus layout: <corpus_dir>/<tenant>/<doc_id>.md. Idempotent."""
        results = []
        for path in sorted((corpus_dir or self.s.corpus_dir).glob("*/*.md")):
            res = self.ingest_text(path.parent.name, path.stem, path.read_text(encoding="utf-8"),
                                   source=path.name)
            results.append({"tenant_id": path.parent.name, **res})
        return results

    # ---------- retrieval ----------
    def retrieve(self, tenant_id: str, question: str) -> dict:
        """Hybrid search, then rerank the fused candidates. Returns every stage for debug/ablation."""
        stages = self.store.search(tenant_id, question, self.s.candidates_per_branch)
        candidates = stages["fused"][: self.s.rerank_candidates]
        scores = self.reranker.predict([(question, h.payload["text"]) for h in candidates])
        reranked = sorted(zip(candidates, scores), key=lambda hs: (-hs[1], hs[0].chunk_id))
        # Our own RRF over the two branch rankings; should agree with Qdrant's server-side fusion.
        local = rrf([[h.chunk_id for h in stages["dense"]], [h.chunk_id for h in stages["sparse"]]],
                    k=self.s.rrf_k)
        return {**stages, "reranked": reranked, "rrf_local": local}

    # ---------- answering ----------
    def ask(self, principal: Principal, question: str, top_k: int = 5) -> dict:
        t0 = time.perf_counter()
        request_id = str(uuid.uuid4())
        r = self.retrieve(principal.tenant_id, question)
        passing = [(h, s) for h, s in r["reranked"] if s >= self.s.rerank_threshold][:top_k]
        evidence = [Evidence(f"C{i}", h.payload["text"], h, s) for i, (h, s) in enumerate(passing, 1)]
        debug = {
            "dense": [h.brief() for h in r["dense"][:10]],
            "sparse": [h.brief() for h in r["sparse"][:10]],
            "fused": [h.brief() for h in r["fused"][:10]],
            "rrf_local": [{"chunk_id": c, "score": round(s, 5)} for c, s in r["rrf_local"][:10]],
            "reranked": [{**h.brief(), "rerank_score": s} for h, s in r["reranked"][:10]],
            "rerank_threshold": self.s.rerank_threshold,
            "reranker": self.reranker.model_id,
            "suspicious_chunk_ids": [e.hit.chunk_id for e in evidence if looks_like_injection(e.text)],
            "llm_called": False,
        }
        answer, cited, abstain_reason = "", [], None
        if not evidence:
            abstain_reason = "no candidate passed the rerank threshold"
        else:
            raw = self.llm.complete(build_messages(question, evidence))  # LLMError propagates
            debug["llm_called"], debug["llm_model"] = True, getattr(self.llm, "model", "?")
            answer, cited, invalid = validate_citations(raw, evidence)
            debug["invalid_citations_dropped"] = invalid
            if ABSTAIN_TOKEN in raw:
                abstain_reason = "model reported insufficient evidence"
            elif not cited:
                abstain_reason = "model answer had no valid citations"
        if abstain_reason:
            answer, cited = "I don't have enough evidence in your documents to answer that.", []
            debug["abstain_reason"] = abstain_reason
        latency_ms = int((time.perf_counter() - t0) * 1000)
        log.info("ask request_id=%s tenant=%s candidates=%d evidence=%d abstained=%s latency_ms=%d",
                 request_id, principal.tenant_id, len(r["fused"]), len(evidence),
                 bool(abstain_reason), latency_ms)
        return {
            "request_id": request_id,
            "answer": answer,
            "abstained": bool(abstain_reason),
            "citations": [{"ref": e.ref, "chunk_id": e.hit.chunk_id, "doc_id": e.hit.payload["doc_id"],
                           "title": e.hit.payload["title"], "page": e.hit.payload["page"],
                           "rerank_score": e.rerank_score} for e in cited],
            "retrieved_doc_ids": list(dict.fromkeys(e.hit.payload["doc_id"] for e in evidence)),
            "debug": debug,
            "latency_ms": latency_ms,
        }
