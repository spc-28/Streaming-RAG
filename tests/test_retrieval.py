"""Fusion, recency, ranking metrics and Pareto dominance: pure functions, no models needed."""
import time

from app.metrics import hit_rate_at_k, mrr, ndcg_at_k, pareto_front, precision_at_k, recall_at_k
from app.models import Chunk
from app.retrieval import apply_recency_boost, reciprocal_rank_fusion


def chunk(cid: str, **meta) -> Chunk:
    return Chunk(chunk_id=cid, text=f"text {cid}", metadata=meta)


def test_rrf_merges_and_ranks_overlap_first():
    fused = reciprocal_rank_fusion([[chunk("a"), chunk("b")], [chunk("c"), chunk("a")]])
    assert [d.chunk_id for d in fused][0] == "a"          # in both lists
    assert {d.chunk_id for d in fused} == {"a", "b", "c"}
    assert fused[0].score == 1.0                           # normalised to the best


def test_rrf_scores_are_monotonic_and_empty_is_safe():
    fused = reciprocal_rank_fusion([[chunk(f"d{i}") for i in range(5)]])
    scores = [d.rrf_score for d in fused]
    assert scores == sorted(scores, reverse=True)
    assert reciprocal_rank_fusion([[]]) == []


def test_recency_boost_halves_every_half_life():
    now = time.time()
    fresh = chunk("fresh", unix_end=now).model_copy(update={"score": 0.5})
    old = chunk("old", unix_end=now - 30 * 60).model_copy(update={"score": 0.5})  # one half-life
    undated = chunk("undated").model_copy(update={"score": 0.5})
    ranked = apply_recency_boost([undated, old, fresh], now=now)
    assert [d.chunk_id for d in ranked] == ["fresh", "old", "undated"]
    boosts = {d.chunk_id: d.recency_boost for d in ranked}
    assert abs(boosts["old"] - boosts["fresh"] / 2) < 1e-3
    assert boosts["undated"] == 0.0


def test_ranking_metrics():
    retrieved, relevant = ["a", "b", "c", "d"], {"a", "c"}
    assert precision_at_k(retrieved, relevant, 4) == 0.5
    assert precision_at_k([], relevant, 5) == 0.0
    assert abs(recall_at_k(retrieved, {"a", "c", "e"}, 4) - 2 / 3) < 1e-9
    assert mrr(["x", "y", "c"], {"c"}) == 1 / 3
    assert mrr(["x"], {"c"}) == 0.0
    assert hit_rate_at_k(["a", "b", "c"], {"c"}, 2) == 0.0
    assert ndcg_at_k(["a", "b"], {"a", "b"}, 2) == 1.0
    assert ndcg_at_k(["x", "y"], {"a"}, 2) == 0.0


def test_pareto_front():
    points = {
        "fast_cheap": (0.70, 100, 0.001),
        "accurate": (0.90, 900, 0.002),
        "dominated": (0.65, 500, 0.002),   # worse than fast_cheap on every axis
        "twin": (0.70, 100, 0.001),        # identical points do not dominate each other
    }
    assert set(pareto_front(points)) == {"fast_cheap", "accurate", "twin"}
