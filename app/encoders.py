"""Dense and sparse encoders. No torch: both are deterministic and run in microseconds.

* `HashingEmbedder` stands in for all-MiniLM-L6-v2 (384 dims, cosine). It hashes word,
  word-bigram and character-trigram features into 384 signed buckets and L2-normalizes.
  Character trigrams give typo/morphology tolerance; the concept map in `text.py`
  simulates a little semantic generalization. It is NOT a neural embedding.
* `BM25SparseEncoder` produces BM25 document-side term weights (TF saturation and length
  normalization). Qdrant multiplies them by IDF at query time (`modifier: idf`), so the
  dot product of a query (weight 1 per term) and a document equals the BM25 score.
"""
from __future__ import annotations

import math
from collections import Counter
from hashlib import blake2b

from app.text import concept_terms, tokenize

DENSE_DIM = 384


def _hash64(feature: str) -> int:
    return int.from_bytes(blake2b(feature.encode(), digest_size=8).digest(), "little")


class HashingEmbedder:
    model_id = "hash-ngram-384-v1"
    dim = DENSE_DIM

    def embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        words = concept_terms(tokenize(text))

        def add(feature: str, weight: float) -> None:
            h = _hash64(feature)
            vec[h % self.dim] += weight if (h >> 40) & 1 else -weight

        for w in words:
            add("w:" + w, 1.0)
            padded = f"^{w}$"
            grams = [padded[i : i + 3] for i in range(len(padded) - 2)]
            for g in grams:
                add("c:" + g, 0.6 / math.sqrt(len(grams)))
        for a, b in zip(words, words[1:]):
            add(f"b:{a}|{b}", 0.5)
        norm = math.sqrt(sum(x * x for x in vec)) or 1.0
        return [x / norm for x in vec]


class BM25SparseEncoder:
    model_id = "bm25-hashed-v1"

    def __init__(self, k1: float = 1.2, b: float = 0.75, avgdl: float = 70.0):
        # avgdl is a fixed corpus-wide estimate (tokens per ~900-char chunk) so stored
        # weights never need recomputing when other documents arrive.
        self.k1, self.b, self.avgdl = k1, b, avgdl

    @staticmethod
    def term_index(term: str) -> int:
        return int.from_bytes(blake2b(term.encode(), digest_size=4).digest(), "little")

    def _to_sparse(self, weights: dict[str, float]) -> tuple[list[int], list[float]]:
        merged: dict[int, float] = {}
        for term, w in weights.items():  # merge the (rare) hash collisions
            idx = self.term_index(term)
            merged[idx] = merged.get(idx, 0.0) + w
        items = sorted(merged.items())
        return [i for i, _ in items], [v for _, v in items]

    def encode_document(self, text: str) -> tuple[list[int], list[float]]:
        tokens = tokenize(text)
        tf = Counter(tokens)
        norm = self.k1 * (1 - self.b + self.b * len(tokens) / self.avgdl)
        return self._to_sparse({t: f * (self.k1 + 1) / (f + norm) for t, f in tf.items()})

    def encode_query(self, text: str) -> tuple[list[int], list[float]]:
        return self._to_sparse({t: 1.0 for t in tokenize(text)})
