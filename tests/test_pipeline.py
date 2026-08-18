"""Query routing, tracing, prompt building, cost estimates and the freshness-aware cache."""
import numpy as np
import pytest

from app.cache import SemanticCache
from app.llm import Generation
from app.models import Chunk, QueryResponse
from app.pipeline import ROUTES, Trace, build_prompt, classify_query


@pytest.mark.parametrize("question, expected", [
    ("What's the latest news?", "temporal"),
    ("What happened today?", "temporal"),
    ("Tell me about breaking developments", "temporal"),
    ("Summarize the day", "summary"),
    ("Give me an overview", "summary"),
    ("What are the main topics discussed?", "summary"),
    ("Compare the two approaches", "comparison"),
    ("What's the difference between X and Y?", "comparison"),
    ("Who is the president of France?", "factoid"),
    ("How many people attended?", "factoid"),
])
def test_classify_query(question, expected):
    assert classify_query(question) == expected
    assert expected in ROUTES


def test_temporal_route_uses_recency():
    assert ROUTES["temporal"].recency and not ROUTES["factoid"].recency


def test_trace_records_spans_and_cost():
    trace = Trace()
    with trace.span("retrieval") as s:
        s["candidates"] = 3
    with trace.span("generation") as s:
        s["cost_usd"] = 0.002
    d = trace.to_dict()
    assert [s["name"] for s in d["spans"]] == ["retrieval", "generation"]
    assert d["total_cost_usd"] == 0.002 and all("duration_ms" in s for s in d["spans"])


def test_generation_cost():
    assert Generation("x", "gemini-2.0-flash", 1_000_000, 1_000_000).cost_usd == 0.5
    assert Generation("x", "unknown-model", 100, 100).cost_usd == 0.0


def test_prompt_includes_time_and_story():
    doc = Chunk(chunk_id="c1", text="Rates held.", metadata={"start_time_utc": "2026-01-15T09:12:00+00:00",
                                                             "event_id": 3})
    prompt = build_prompt("What happened?", [doc])
    assert "[09:12 UTC · c1 · story 3]" in prompt and "QUESTION: What happened?" in prompt


def _unit(*xs) -> np.ndarray:
    v = np.array(xs, dtype=float)
    return v / np.linalg.norm(v)


def test_cache_hits_paraphrases_and_evicts_only_relevant_entries():
    cache = SemanticCache()
    ecb, mars = _unit(1, 0, 0), _unit(0, 1, 0)
    key = ("full", None, True)
    cache.put(ecb, key, QueryResponse(trace_id="t1", question="ECB?", answer="held", pipeline="full"))
    cache.put(mars, key, QueryResponse(trace_id="t2", question="Mars?", answer="water", pipeline="full"))

    assert cache.get(_unit(1, 0.05, 0), key)[0].answer == "held"   # paraphrase → hit
    assert cache.get(_unit(1, 0.05, 0), ("baseline", None, True)) is None  # different settings → miss

    assert cache.invalidate(_unit(0.9, 0, 0.1)) == 1                # new ECB audio arrives
    assert cache.get(ecb, key) is None and cache.get(mars, key)[0].answer == "water"
