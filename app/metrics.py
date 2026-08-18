"""Evaluation metrics: retrieval ranking quality, LLM-as-judge, and Pareto dominance."""
import math
import re
from typing import Optional

from app.llm import LLMError, generate


# ---------- Retrieval (ids of retrieved chunks vs. the set of relevant ones) ----------
def precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    top = retrieved[:k]
    return sum(d in relevant for d in top) / len(top) if top else 0.0


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    return sum(d in relevant for d in retrieved[:k]) / len(relevant) if relevant else 0.0


def hit_rate_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    return float(any(d in relevant for d in retrieved[:k]))


def mrr(retrieved: list[str], relevant: set[str]) -> float:
    """1 / rank of the first relevant result."""
    return next((1 / rank for rank, d in enumerate(retrieved, start=1) if d in relevant), 0.0)


def ndcg_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    dcg = sum(1 / math.log2(i + 2) for i, d in enumerate(retrieved[:k]) if d in relevant)
    ideal = sum(1 / math.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / ideal if ideal else 0.0


def retrieval_metrics(retrieved: list[str], relevant: set[str]) -> dict[str, float]:
    return {
        "precision_at_3": round(precision_at_k(retrieved, relevant, 3), 4),
        "recall_at_5": round(recall_at_k(retrieved, relevant, 5), 4),
        "mrr": round(mrr(retrieved, relevant), 4),
        "ndcg_at_5": round(ndcg_at_k(retrieved, relevant, 5), 4),
        "hit_rate_at_5": hit_rate_at_k(retrieved, relevant, 5),
    }


# ---------- LLM-as-judge ----------
JUDGE_DIMENSIONS = ("relevance", "faithfulness", "completeness", "conciseness")

JUDGE_PROMPT = """You are grading an AI answer about news transcripts.

SOURCES:
{sources}

QUESTION: {question}
AI ANSWER: {answer}
GROUND TRUTH: {ground_truth}

Score each dimension from 1 (bad) to 5 (excellent):
RELEVANCE: does it address the question?
FAITHFULNESS: is every claim grounded in the sources?
COMPLETENESS: does it cover the key facts in the ground truth?
CONCISENESS: is it free of padding?

Reply with exactly four lines, e.g. "RELEVANCE: 4"."""


def llm_judge(question: str, answer: str, ground_truth: str, sources: list[str]) -> Optional[dict]:
    prompt = JUDGE_PROMPT.format(sources="\n---\n".join(sources) or "(none)", question=question,
                                 answer=answer, ground_truth=ground_truth)
    try:
        reply = generate(prompt).text
    except LLMError:
        return None
    scores = {}
    for dim in JUDGE_DIMENSIONS:
        if m := re.search(rf"{dim}\s*:\s*(\d)", reply, re.IGNORECASE):
            scores[f"judge_{dim}"] = min(5, max(1, int(m.group(1))))
    return scores or None


# ---------- Pareto frontier ----------
def pareto_front(points: dict[str, tuple[float, float, float]]) -> list[str]:
    """Ids of points that no other point beats on (quality ↑, latency ↓, cost ↓) all at once.

    b dominates a if it is at least as good on every axis and not identical,
    i.e. strictly better on at least one.
    """
    def dominates(b, a):
        return b[0] >= a[0] and b[1] <= a[1] and b[2] <= a[2] and b != a

    return [pid for pid, p in points.items() if not any(dominates(q, p) for q in points.values())]
