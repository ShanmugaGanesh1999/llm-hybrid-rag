"""Retrieval ablation on a TINY DEMO SET (12 hand-labeled questions). Not a benchmark.

Usage (inside compose):  docker compose run --rm rag-api python scripts/ablation.py
A question counts as a hit when a chunk from a labeled (doc_id, page) appears in the top k.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import Settings  # noqa: E402
from app.pipeline import RagPipeline  # noqa: E402

K = 5
MODES = ("dense", "sparse", "fused", "fused+rerank")


def first_relevant_rank(ranked: list[dict], relevant: set[tuple[str, int]]) -> int | None:
    for rank, payload in enumerate(ranked, start=1):
        if (payload["doc_id"], payload["page"]) in relevant:
            return rank
    return None


def main() -> None:
    pipeline = RagPipeline(Settings())
    pipeline.store.wait_ready()
    pipeline.store.ensure_collections()
    pipeline.ingest_corpus()  # idempotent
    path = Path(__file__).resolve().parents[1] / "data" / "eval_questions.jsonl"
    questions = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

    stats = {m: {"hits": 0, "rr": 0.0, "misses": []} for m in MODES}
    latencies = []
    for q in questions:
        t0 = time.perf_counter()
        r = pipeline.retrieve(q["tenant"], q["question"])
        latencies.append((time.perf_counter() - t0) * 1000)
        ranked = {
            "dense": [h.payload for h in r["dense"]],
            "sparse": [h.payload for h in r["sparse"]],
            "fused": [h.payload for h in r["fused"]],
            "fused+rerank": [h.payload for h, _ in r["reranked"]],
        }
        relevant = {tuple(x) for x in q["relevant"]}
        for mode in MODES:
            rank = first_relevant_rank(ranked[mode], relevant)
            if rank is not None and rank <= K:
                stats[mode]["hits"] += 1
            else:
                stats[mode]["misses"].append(q["id"])
            if rank is not None and rank <= 10:
                stats[mode]["rr"] += 1.0 / rank

    n = len(questions)
    print(f"Tiny demo set: {n} labeled questions (NOT a benchmark; numbers are illustrative only)")
    print(f"{'mode':<14}{'Recall@5':>10}{'MRR@10':>10}   misses@5")
    for mode in MODES:
        s = stats[mode]
        print(f"{mode:<14}{s['hits'] / n:>10.2f}{s['rr'] / n:>10.3f}   {', '.join(s['misses']) or '-'}")
    latencies.sort()
    print(f"retrieve+rerank latency (ms): p50={latencies[n // 2]:.1f} max={latencies[-1]:.1f}")


if __name__ == "__main__":
    main()
