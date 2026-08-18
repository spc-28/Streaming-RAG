"""
RAG pipelines. Each pipeline runs the same stages with different settings:

    retrieve (vector | hybrid) → rerank? → recency boost? → generate → verify faithfulness

  baseline    vector search only
  hybrid      BM25 + vector fused with RRF
  full        hybrid + cross-encoder rerank (default)
  reflective  run `full`; if the answer is poorly grounded, retry with a broader strategy
  adaptive    classify the question, then use the strategy that suits that type
"""
import re
import time
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from typing import Optional

from app.config import get_settings
from app.faithfulness import check_faithfulness
from app.llm import LLMError, generate
from app.models import Chunk, QueryRequest, QueryResponse
from app.retrieval import apply_recency_boost, rerank, retrieve
from app.store import Index


@dataclass(frozen=True)
class Strategy:
    retriever: str = "hybrid"    # "vector" | "hybrid"
    rerank: bool = True
    recency: bool = False
    top_k: int = 5


PRESETS = {
    "baseline": Strategy("vector", rerank=False),
    "hybrid": Strategy("hybrid", rerank=False),
    "full": Strategy(),
}

# Reflective: each retry is a genuinely different strategy, not a re-run of the same one.
ESCALATION = [Strategy(top_k=5), Strategy(top_k=10), Strategy(top_k=10, recency=True)]

# Adaptive: precision for facts, recall for summaries and comparisons, freshness for "latest".
ROUTES = {
    "factoid": Strategy(top_k=3),
    "summary": Strategy(top_k=8),
    "comparison": Strategy(top_k=8),
    "temporal": Strategy(recency=True),
}

_ROUTING_RULES = [  # checked in order; the first match wins
    ("temporal", r"\b(latest|recent|recently|just|now|today|yesterday|currently|breaking|live|"
                 r"this (morning|evening)|tonight|in the (last|past)|ago)\b|what'?s happening"),
    ("summary", r"\b(summari[sz]e|summary|overview|main (topics|points|stories)|key points)\b|"
                r"what (was|were) (discussed|covered)"),
    ("comparison", r"\b(compare|comparison|versus|vs\.?|difference between|(better|worse) than)\b"),
]


def classify_query(question: str) -> str:
    q = question.lower()
    return next((qtype for qtype, pattern in _ROUTING_RULES if re.search(pattern, q)), "factoid")


SYSTEM_PROMPT = """You answer questions about live news transcripts.
Use only information in the provided transcript segments; no background knowledge or speculation.
Cite the segment time when you use it, e.g. "In the 09:12 UTC segment, ...".
Segments that share a story number are the same developing story; say how it evolved.
If the transcripts do not contain the answer, reply exactly:
"I don't have enough information in the available transcripts." """


def build_prompt(question: str, docs: list[Chunk]) -> str:
    blocks = []
    for d in docs:
        when = d.metadata.get("start_time_utc", "")[11:16] or "??:??"
        story = f" · story {d.metadata['event_id']}" if "event_id" in d.metadata else ""
        blocks.append(f"[{when} UTC · {d.chunk_id}{story}]\n{d.text}")
    return "TRANSCRIPTS:\n\n" + "\n\n---\n\n".join(blocks) + f"\n\nQUESTION: {question}"


class Trace:
    """Per-request timing and cost of every stage, returned with the answer."""

    def __init__(self):
        self.trace_id = uuid.uuid4().hex[:12]
        self.spans: list[dict] = []
        self._start = time.perf_counter()

    @contextmanager
    def span(self, name: str, **info):
        span = {"name": name, **info}
        t0 = time.perf_counter()
        try:
            yield span
        finally:
            span["duration_ms"] = round((time.perf_counter() - t0) * 1000, 1)
            self.spans.append(span)

    def to_dict(self) -> dict:
        return {
            "trace_id": self.trace_id,
            "total_duration_ms": round((time.perf_counter() - self._start) * 1000, 1),
            "total_cost_usd": round(sum(s.get("cost_usd", 0) for s in self.spans), 6),
            "spans": self.spans,
        }


def _time_range(req: QueryRequest) -> Optional[tuple[int, int]]:
    if req.minutes_ago is None:
        return None
    target = datetime.now(timezone.utc).timestamp() - req.minutes_ago * 60
    half_window = req.time_window_minutes * 60 / 2
    return int(target - half_window), int(target + half_window)


def _run(req: QueryRequest, strategy: Strategy, index: Index, trace: Trace, name: str) -> QueryResponse:
    with trace.span("retrieval", retriever=strategy.retriever) as span:
        docs = retrieve(index, req.question, strategy.retriever, strategy.top_k * 3, _time_range(req))
        span["candidates"] = len(docs)
    if strategy.rerank and docs:
        with trace.span("rerank"):
            docs = rerank(req.question, docs)
    if strategy.recency and docs:
        with trace.span("recency_boost"):
            docs = apply_recency_boost(docs)
    docs = docs[:strategy.top_k]

    answer, model, faith = "No relevant transcripts found for this query.", None, None
    if docs:
        with trace.span("generation") as span:
            try:
                gen = generate(build_prompt(req.question, docs), system=SYSTEM_PROMPT)
                answer, model = gen.text, gen.model
                span.update(model=gen.model, input_tokens=gen.input_tokens,
                            output_tokens=gen.output_tokens, cost_usd=gen.cost_usd)
            except LLMError as e:
                answer = f"Generation error: {e}"
        if req.check_faithfulness and model:
            with trace.span("faithfulness") as span:
                faith = check_faithfulness(answer, docs)
                span["score"] = faith.faithfulness_score

    return QueryResponse(
        trace_id=trace.trace_id, question=req.question, answer=answer, pipeline=name,
        sources=docs, faithfulness=faith, config={**asdict(strategy), "model": model},
        trace=trace.to_dict(),
    )


def _faith(resp: QueryResponse) -> float:
    return resp.faithfulness.faithfulness_score if resp.faithfulness else 0.0


def _reflective(req: QueryRequest, index: Index, trace: Trace) -> QueryResponse:
    threshold = get_settings().reflective_faithfulness_threshold
    req = req.model_copy(update={"check_faithfulness": True})  # the retry decision depends on it
    best = None
    for attempt, strategy in enumerate(ESCALATION):
        resp = _run(req, strategy, index, trace, "reflective")
        if best is None or _faith(resp) > _faith(best):
            best = resp
        if _faith(resp) >= threshold:
            break
    else:
        best.answer += (f"\n\n(Low confidence: faithfulness {_faith(best):.2f} stayed below "
                        f"{threshold:.2f} after {len(ESCALATION)} attempts.)")
    best.retry_count = attempt
    best.trace = trace.to_dict()
    return best


def _adaptive(req: QueryRequest, index: Index, trace: Trace) -> QueryResponse:
    with trace.span("routing") as span:
        query_type = span["query_type"] = classify_query(req.question)
    resp = _run(req, ROUTES[query_type], index, trace, "adaptive")
    resp.query_type = query_type
    return resp


def answer(req: QueryRequest, index: Index) -> QueryResponse:
    trace = Trace()
    if req.pipeline == "reflective":
        return _reflective(req, index, trace)
    if req.pipeline == "adaptive":
        return _adaptive(req, index, trace)
    strategy = PRESETS[req.pipeline]
    if req.top_k:
        strategy = replace(strategy, top_k=req.top_k)
    return _run(req, strategy, index, trace, req.pipeline)
