"""
Retrieval stages, applied in this order by the pipeline:

  1. candidates   dense vectors, or BM25 + dense fused with Reciprocal Rank Fusion
  2. rerank       a cross-encoder rescores (question, chunk) pairs jointly
  3. recency      streaming prior: newer chunks get a bonus that halves every N minutes
"""
import math
import time
from functools import lru_cache
from typing import Optional

from app.config import get_settings
from app.models import Chunk
from app.store import Index


def reciprocal_rank_fusion(ranked_lists: list[list[Chunk]], k: int = 60) -> list[Chunk]:
    """RRF (Cormack et al., 2009): score(d) = sum over lists of 1 / (k + rank).

    It only uses ranks, so BM25 scores (unbounded) and cosine scores (0..1)
    never need to be put on the same scale.
    """
    scores: dict[str, float] = {}
    docs: dict[str, Chunk] = {}
    for ranked in ranked_lists:
        for rank, doc in enumerate(ranked, start=1):
            scores[doc.chunk_id] = scores.get(doc.chunk_id, 0.0) + 1.0 / (k + rank)
            docs.setdefault(doc.chunk_id, doc)
    if not scores:
        return []
    best = max(scores.values())
    return [docs[cid].model_copy(update={"rrf_score": scores[cid], "score": scores[cid] / best})
            for cid in sorted(scores, key=scores.get, reverse=True)]


def retrieve(index: Index, query: str, method: str, k: int,
             time_range: Optional[tuple[int, int]] = None) -> list[Chunk]:
    if method == "vector":
        return index.vector_search(query, k, time_range)
    s = get_settings()
    n = max(k, s.candidates_k)
    fused = reciprocal_rank_fusion(
        [index.vector_search(query, n, time_range), index.keyword_search(query, n, time_range)],
        k=s.rrf_constant,
    )
    return fused[:k]


@lru_cache
def _cross_encoder():
    from sentence_transformers import CrossEncoder
    return CrossEncoder(get_settings().reranker_model)


def rerank(query: str, docs: list[Chunk]) -> list[Chunk]:
    """Re-order all candidates by cross-encoder relevance (sigmoid of the logit)."""
    if not docs:
        return []
    logits = _cross_encoder().predict([(query, d.text) for d in docs])
    scored = [d.model_copy(update={"rerank_score": float(z), "score": 1 / (1 + math.exp(-z))})
              for d, z in zip(docs, logits)]
    return sorted(scored, key=lambda d: d.score, reverse=True)


def apply_recency_boost(docs: list[Chunk], now: Optional[float] = None) -> list[Chunk]:
    """score += max_boost * 0.5 ** (age / half_life). Chunks without timestamps get no boost."""
    s = get_settings()
    now = now or time.time()
    boosted = []
    for d in docs:
        boost = 0.0
        if end := d.metadata.get("unix_end"):
            age_min = max(0.0, now - end) / 60
            boost = s.recency_max_boost * 0.5 ** (age_min / s.recency_half_life_minutes)
        boosted.append(d.model_copy(update={"recency_boost": round(boost, 4), "score": d.score + boost}))
    return sorted(boosted, key=lambda d: d.score, reverse=True)
