"""Reciprocal-rank fusion and the reranker interface."""
from __future__ import annotations

from typing import Protocol, Sequence

from app.text import concept_terms, is_identifier, tokenize


def rrf(rankings: Sequence[Sequence[str]], k: int = 60) -> list[tuple[str, float]]:
    """score(d) = sum over rankings of 1 / (k + rank_i(d)), rank starting at 1.
    Raw retriever scores are ignored on purpose: cosine and BM25 are not comparable.
    Ties break by id so the output is deterministic."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))


class Reranker(Protocol):
    model_id: str

    def predict(self, pairs: Sequence[tuple[str, str]]) -> list[float]: ...


class SimulatedCrossEncoder:
    """SIMULATION of `cross-encoder/ms-marco-MiniLM-L-6-v2`.

    A real cross-encoder reads (query, passage) jointly and outputs a relevance logit.
    This deterministic stand-in scores each pair from:
      * 0.7 x weighted query-term coverage (longer words weigh more; identifiers most),
      * 0.3 x query-bigram (phrase) match ratio,
      * +0.2 if every identifier in the query (e.g. DEPLOY-4021) appears, -0.3 if any is missing,
      * x0.5 if the most specific query term (heaviest weight) is missing from the passage.
    Scores are clamped to [0, 1]. Like real cross-encoder scores they are ranking
    signals, not probabilities that an answer is correct. Swap in a real model by
    implementing `Reranker.predict`.
    """

    model_id = "simulated:cross-encoder/ms-marco-MiniLM-L-6-v2"

    def score(self, query: str, passage: str) -> float:
        q = concept_terms(tokenize(query))
        if not q:
            return 0.0
        p_seq = concept_terms(tokenize(passage))
        p_set = set(p_seq)
        terms = list(dict.fromkeys(q))
        weights = {t: 2.0 if is_identifier(t) else 1.0 + min(len(t), 10) / 10 for t in terms}
        coverage = sum(w for t, w in weights.items() if t in p_set) / sum(weights.values())
        q_bigrams = set(zip(q, q[1:]))
        phrase = len(q_bigrams & set(zip(p_seq, p_seq[1:]))) / len(q_bigrams) if q_bigrams else coverage
        idents = [t for t in terms if is_identifier(t)]
        boost = (0.2 if all(t in p_set for t in idents) else -0.3) if idents else 0.0
        score = 0.7 * coverage + 0.3 * phrase + boost
        # A real cross-encoder scores "parental leave policy" vs a sick-leave passage low even
        # though two of three words match: the passage misses what the question is about.
        # Approximate that: halve the score when the most specific query term (heaviest
        # weight, i.e. identifier or longest word) is absent from the passage.
        anchor = max(terms, key=lambda t: (weights[t], len(t)))
        if len(terms) >= 2 and anchor not in p_set:
            score *= 0.5
        return round(max(0.0, min(1.0, score)), 4)

    def predict(self, pairs: Sequence[tuple[str, str]]) -> list[float]:
        return [self.score(q, p) for q, p in pairs]
