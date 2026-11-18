"""Qdrant access: collection setup, idempotent document upsert with manifests, hybrid search.

Chunk collection points:   vectors {"dense": 384-d cosine, "sparse": BM25 weights (modifier=idf)}
                           payload {tenant_id, doc_id, title, source, page, chunk_index, text, ...}
Manifest collection points (no vectors): one per (tenant, doc) listing its chunk IDs + checksum.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timezone

from qdrant_client import QdrantClient, models

from app.encoders import DENSE_DIM, BM25SparseEncoder, HashingEmbedder
from app.text import manifest_id


@dataclass
class Hit:
    chunk_id: str
    score: float
    payload: dict

    def brief(self) -> dict:
        p = self.payload
        return {"chunk_id": self.chunk_id, "doc_id": p["doc_id"], "page": p["page"], "score": round(self.score, 4)}


def tenant_filter(tenant_id: str) -> models.Filter:
    return models.Filter(must=[models.FieldCondition(key="tenant_id", match=models.MatchValue(value=tenant_id))])


class VectorStore:
    def __init__(self, url: str, collection: str, manifest_collection: str, rrf_k: int = 60):
        self.client = QdrantClient(url=url, timeout=10)
        self.collection, self.manifests, self.rrf_k = collection, manifest_collection, rrf_k
        self.embedder, self.sparse = HashingEmbedder(), BM25SparseEncoder()

    # ---------- setup ----------
    def wait_ready(self, attempts: int = 30, delay: float = 1.0) -> None:
        for i in range(attempts):
            try:
                self.client.get_collections()
                return
            except Exception:
                if i == attempts - 1:
                    raise
                time.sleep(delay)

    def ensure_collections(self) -> None:
        if not self.client.collection_exists(self.collection):
            self.client.create_collection(
                self.collection,
                vectors_config={"dense": models.VectorParams(size=DENSE_DIM, distance=models.Distance.COSINE)},
                sparse_vectors_config={"sparse": models.SparseVectorParams(modifier=models.Modifier.IDF)},
            )
            # is_tenant=True tells Qdrant to co-locate each tenant's points (multitenancy).
            self.client.create_payload_index(
                self.collection, "tenant_id", models.KeywordIndexParams(type="keyword", is_tenant=True))
            self.client.create_payload_index(self.collection, "doc_id", models.PayloadSchemaType.KEYWORD)
        if not self.client.collection_exists(self.manifests):
            self.client.create_collection(self.manifests, vectors_config={})
            self.client.create_payload_index(self.manifests, "tenant_id", models.PayloadSchemaType.KEYWORD)

    def drop_collections(self) -> None:
        for name in (self.collection, self.manifests):
            if self.client.collection_exists(name):
                self.client.delete_collection(name)

    # ---------- documents ----------
    def get_manifest(self, tenant_id: str, doc_id: str) -> dict | None:
        found = self.client.retrieve(self.manifests, ids=[manifest_id(tenant_id, doc_id)], with_payload=True)
        return found[0].payload if found else None

    def list_manifests(self, tenant_id: str) -> list[dict]:
        points, _ = self.client.scroll(self.manifests, scroll_filter=tenant_filter(tenant_id),
                                       limit=1000, with_payload=True)
        return sorted((p.payload for p in points), key=lambda m: m["doc_id"])

    def upsert_document(self, tenant_id: str, doc_id: str, title: str, source: str, checksum: str,
                        chunks: list[dict]) -> dict:
        """Upsert chunks under deterministic IDs, delete chunk IDs the previous version had
        but this one lacks (stale), then write the manifest. Unchanged checksum: no-op."""
        old = self.get_manifest(tenant_id, doc_id)
        if old and old["checksum"] == checksum:
            return {"doc_id": doc_id, "status": "unchanged", "chunks": len(old["chunk_ids"]), "removed": 0}
        points = []
        for c in chunks:
            indices, values = self.sparse.encode_document(c["text"])
            payload = {"tenant_id": tenant_id, "doc_id": doc_id, "title": title, "source": source,
                       "page": c["page"], "chunk_index": c["chunk_index"], "text": c["text"],
                       "checksum": checksum, "embedding_model": self.embedder.model_id}
            points.append(models.PointStruct(
                id=c["chunk_id"], payload=payload,
                vector={"dense": self.embedder.embed(c["text"]),
                        "sparse": models.SparseVector(indices=indices, values=values)}))
        if points:
            self.client.upsert(self.collection, points=points, wait=True)
        new_ids = [c["chunk_id"] for c in chunks]
        stale = sorted(set(old["chunk_ids"]) - set(new_ids)) if old else []
        if stale:
            self.client.delete(self.collection, points_selector=models.PointIdsList(points=stale), wait=True)
        manifest = {"tenant_id": tenant_id, "doc_id": doc_id, "title": title, "source": source,
                    "checksum": checksum, "chunk_ids": new_ids,
                    "ingested_at": datetime.now(timezone.utc).isoformat()}
        self.client.upsert(self.manifests, wait=True, points=[
            models.PointStruct(id=manifest_id(tenant_id, doc_id), vector={}, payload=manifest)])
        return {"doc_id": doc_id, "status": "updated" if old else "created",
                "chunks": len(new_ids), "removed": len(stale)}

    def count_chunks(self, tenant_id: str, doc_id: str | None = None) -> int:
        flt = tenant_filter(tenant_id)
        if doc_id:
            flt.must.append(models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id)))
        return self.client.count(self.collection, count_filter=flt, exact=True).count

    # ---------- search ----------
    def search(self, tenant_id: str, question: str, limit: int) -> dict[str, list[Hit]]:
        """One round trip, three queries: dense-only, sparse-only (both for debug/ablation)
        and the production query: two prefetch branches, each with the SAME tenant filter,
        fused server-side with RRF. The filter is inside every branch, not applied after."""
        acl = tenant_filter(tenant_id)
        dense = self.embedder.embed(question)
        indices, values = self.sparse.encode_query(question)
        dense_req = models.QueryRequest(query=dense, using="dense", filter=acl, limit=limit, with_payload=True)
        requests = [dense_req]
        prefetch = [models.Prefetch(query=dense, using="dense", filter=acl, limit=limit)]
        if indices:  # an all-stopword question has no sparse terms
            sparse_vec = models.SparseVector(indices=indices, values=values)
            requests.append(models.QueryRequest(query=sparse_vec, using="sparse", filter=acl,
                                                limit=limit, with_payload=True))
            prefetch.append(models.Prefetch(query=sparse_vec, using="sparse", filter=acl, limit=limit))
        requests.append(models.QueryRequest(
            prefetch=prefetch, query=models.RrfQuery(rrf=models.Rrf(k=self.rrf_k)),
            filter=acl, limit=limit, with_payload=True))
        responses = self.client.query_batch_points(self.collection, requests=requests)
        as_hits = [[Hit(str(p.id), p.score, p.payload) for p in r.points] for r in responses]
        return {"dense": as_hits[0], "sparse": as_hits[1] if indices else [], "fused": as_hits[-1]}
