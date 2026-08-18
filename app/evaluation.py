"""
Evaluation workflows.

  run_eval      one pipeline on one dataset → per-question and summary metrics, saved to SQLite
  benchmark     several pipelines on the same dataset → comparison table + Pareto-optimal set
  compare_runs  paired Wilcoxon signed-rank test per metric: is a difference real or noise?
  pareto        across saved runs: which are not beaten on quality, latency and cost at once
  synthetic     the LLM writes QA pairs from live chunks, so the eval set grows with the stream

Datasets are indexed into their own Chroma collection, so evaluation never
touches the live index.
"""
import json
import logging
import random
import re
import sqlite3
import statistics
import uuid
from contextlib import closing
from datetime import datetime, timezone
from typing import Optional

from app.config import get_settings
from app.ingest import Ingestor
from app.llm import LLMError, generate
from app.metrics import JUDGE_DIMENSIONS, llm_judge, pareto_front, retrieval_metrics
from app.models import EvalDataset, QAPair, QueryRequest
from app.pipeline import answer
from app.store import Index, get_index

logger = logging.getLogger(__name__)

QUALITY_METRICS = ["precision_at_3", "recall_at_5", "mrr", "ndcg_at_5", "hit_rate_at_5",
                   "faithfulness", "numeric_accuracy", "citation_accuracy"]
JUDGE_METRICS = [f"judge_{d}" for d in JUDGE_DIMENSIONS]

TABLE_LAYOUT = (
    [("Retrieval", m) for m in QUALITY_METRICS[:5]]
    + [("Generation", m) for m in QUALITY_METRICS[5:]]
    + [("LLM judge (1-5)", m) for m in JUDGE_METRICS]
    + [("Performance", m) for m in ("p50_latency_ms", "p95_latency_ms", "avg_cost_usd")]
)


# ---------- Datasets (JSON files in datasets/) ----------
def _dataset_path(dataset_id: str):
    if not re.fullmatch(r"[\w.-]+", dataset_id):
        raise ValueError(f"Invalid dataset id: {dataset_id!r}")
    return get_settings().datasets_dir / f"{dataset_id}.json"


def load_dataset(dataset_id: str) -> EvalDataset:
    path = _dataset_path(dataset_id)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_id}")
    return EvalDataset.model_validate_json(path.read_text(encoding="utf-8"))


def save_dataset(dataset: EvalDataset) -> None:
    path = _dataset_path(dataset.dataset_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dataset.model_dump_json(indent=2), encoding="utf-8")


def list_datasets() -> list[dict]:
    datasets = [load_dataset(p.stem) for p in sorted(get_settings().datasets_dir.glob("*.json"))]
    return [{"dataset_id": d.dataset_id, "description": d.description,
             "num_qa_pairs": len(d.qa_pairs), "synthetic": d.synthetic} for d in datasets]


# ---------- Run storage (SQLite, one row per run) ----------
def _sql(query: str, params: tuple = ()) -> list[tuple]:
    with closing(sqlite3.connect(get_settings().data_dir / "experiments.db")) as conn, conn:
        conn.execute("CREATE TABLE IF NOT EXISTS runs (run_id TEXT PRIMARY KEY, created_at TEXT, "
                     "dataset_id TEXT, pipeline TEXT, summary TEXT, results TEXT)")
        return conn.execute(query, params).fetchall()


def _to_run(row: tuple, with_results: bool = True) -> dict:
    run = dict(zip(("run_id", "created_at", "dataset_id", "pipeline"), row[:4]), summary=json.loads(row[4]))
    if with_results:
        run["results"] = json.loads(row[5])
    return run


def get_run(run_id: str) -> dict:
    rows = _sql("SELECT * FROM runs WHERE run_id = ?", (run_id,))
    if not rows:
        raise FileNotFoundError(f"Run not found: {run_id}")
    return _to_run(rows[0])


def list_runs(dataset_id: Optional[str] = None, limit: int = 100) -> list[dict]:
    rows = _sql("SELECT * FROM runs WHERE ? IS NULL OR dataset_id = ? ORDER BY created_at DESC LIMIT ?",
                (dataset_id, dataset_id, limit))
    return [_to_run(r, with_results=False) for r in rows]


# ---------- Running evaluations ----------
def _eval_index(dataset: EvalDataset, reindex: bool) -> Index:
    index = get_index(f"eval_{dataset.dataset_id}")
    if reindex or index.count() == 0:
        index.clear()
        ingestor = Ingestor(index, watch=False)
        for t in dataset.transcripts:
            ts = t.get("timestamp")
            start = datetime.fromisoformat(ts.replace("Z", "+00:00")) if ts else None
            ingestor.ingest_text(t["text"], source=t.get("source", "eval"), base_id=t["id"], start=start)
    return index


def _summarize(results: list[dict]) -> dict:
    summary = {}
    for key in QUALITY_METRICS + JUDGE_METRICS:
        values = [r[key] for r in results if key in r]
        if values:
            summary[key] = round(statistics.mean(values), 4)
    if results:
        latencies = sorted(r["latency_ms"] for r in results)
        summary.update(
            p50_latency_ms=latencies[len(latencies) // 2],
            p95_latency_ms=latencies[min(len(latencies) - 1, int(0.95 * len(latencies)))],
            avg_latency_ms=round(statistics.mean(latencies), 1),
            avg_cost_usd=round(statistics.mean(r["cost_usd"] for r in results), 6),
            total_cost_usd=round(sum(r["cost_usd"] for r in results), 6),
        )
    return summary


def run_eval(dataset: EvalDataset, pipeline: str, judge: bool = False, reindex: bool = True) -> dict:
    index = _eval_index(dataset, reindex)
    chunk_ids = [d.chunk_id for d in index.all()]
    results = []
    for qa in dataset.qa_pairs:
        try:
            resp = answer(QueryRequest(question=qa.question, pipeline=pipeline), index)
        except Exception as e:
            logger.error(f"{pipeline} failed on {qa.qa_id}: {e}")
            continue
        # Datasets label whole transcripts; a transcript "t1" is indexed as chunks "t1_c0", "t1_c1", ...
        relevant = {cid for cid in chunk_ids for t in qa.relevant_chunk_ids
                    if cid == t or cid.startswith(f"{t}_c")}
        retrieved = [s.chunk_id for s in resp.sources]
        row = {"qa_id": qa.qa_id, "question": qa.question, "answer": resp.answer,
               "ground_truth": qa.ground_truth_answer, "retrieved": retrieved,
               "latency_ms": resp.trace["total_duration_ms"], "cost_usd": resp.trace["total_cost_usd"]}
        if relevant:  # unanswerable questions have no retrieval ground truth
            row.update(retrieval_metrics(retrieved, relevant))
        if f := resp.faithfulness:
            row.update(faithfulness=f.faithfulness_score, numeric_accuracy=f.numeric_accuracy,
                       citation_accuracy=f.citation_accuracy)
        if judge:
            row.update(llm_judge(qa.question, resp.answer, qa.ground_truth_answer,
                                 [s.text for s in resp.sources]) or {})
        results.append(row)
        logger.info(f"[{pipeline}] {qa.qa_id}: faith={row.get('faithfulness', '-')} "
                    f"mrr={row.get('mrr', '-')} {row['latency_ms']:.0f}ms")

    run = {"run_id": uuid.uuid4().hex[:12], "created_at": datetime.now(timezone.utc).isoformat(),
           "dataset_id": dataset.dataset_id, "pipeline": pipeline,
           "summary": _summarize(results), "results": results}
    _sql("INSERT INTO runs VALUES (?, ?, ?, ?, ?, ?)",
         (run["run_id"], run["created_at"], run["dataset_id"], pipeline,
          json.dumps(run["summary"]), json.dumps(results)))
    return run


def _axes(summary: dict, metric: str = "faithfulness") -> tuple[float, float, float]:
    return summary.get(metric, 0.0), summary.get("avg_latency_ms", 0.0), summary.get("avg_cost_usd", 0.0)


def benchmark(dataset: EvalDataset, pipelines: Optional[list[str]] = None, judge: bool = False) -> dict:
    pipelines = pipelines or ["baseline", "hybrid", "full", "adaptive"]
    runs = {p: run_eval(dataset, p, judge, reindex=(i == 0)) for i, p in enumerate(pipelines)}
    table = [{"category": category, "metric": metric, **{p: runs[p]["summary"].get(metric) for p in pipelines}}
             for category, metric in TABLE_LAYOUT
             if any(metric in runs[p]["summary"] for p in pipelines)]
    return {
        "dataset_id": dataset.dataset_id,
        "pipelines": pipelines,
        "run_ids": {p: run["run_id"] for p, run in runs.items()},
        "comparison_table": table,
        "pareto_optimal": pareto_front({p: _axes(run["summary"]) for p, run in runs.items()}),
    }


def compare_runs(baseline_id: str, candidate_id: str, alpha: float = 0.05) -> dict:
    """Per metric, a paired Wilcoxon signed-rank test over the questions both runs answered."""
    from scipy.stats import wilcoxon

    base, cand = get_run(baseline_id), get_run(candidate_id)
    b = {r["qa_id"]: r for r in base["results"]}
    c = {r["qa_id"]: r for r in cand["results"]}
    shared = sorted(b.keys() & c.keys())

    rows = []
    for metric in QUALITY_METRICS + JUDGE_METRICS:
        pairs = [(b[q][metric], c[q][metric]) for q in shared if metric in b[q] and metric in c[q]]
        if not pairs:
            continue
        diffs = [y - x for x, y in pairs]
        try:
            p_value = float(wilcoxon(diffs).pvalue) if any(diffs) else 1.0
        except ValueError:
            p_value = 1.0
        delta = statistics.mean(diffs)
        verdict = "no significant change" if p_value >= alpha else ("improved" if delta > 0 else "regressed")
        rows.append({"metric": metric, "baseline": round(statistics.mean(x for x, _ in pairs), 4),
                     "candidate": round(statistics.mean(y for _, y in pairs), 4),
                     "delta": round(delta, 4), "p_value": round(p_value, 4), "n": len(pairs), "verdict": verdict})

    regressed = [r["metric"] for r in rows if r["verdict"] == "regressed"]
    improved = [r["metric"] for r in rows if r["verdict"] == "improved"]
    return {
        "baseline": {"run_id": baseline_id, "pipeline": base["pipeline"]},
        "candidate": {"run_id": candidate_id, "pipeline": cand["pipeline"]},
        "n_shared_questions": len(shared), "alpha": alpha, "rows": rows,
        "has_regression": bool(regressed),
        "summary": (f"Regression on: {', '.join(regressed)}" if regressed
                    else f"Significant improvement on: {', '.join(improved)}" if improved
                    else "No statistically significant changes."),
    }


def pareto(metric: str = "faithfulness", quality_floor: float = 0.7,
           dataset_id: Optional[str] = None) -> dict:
    """Pareto frontier over saved runs, plus the fastest frontier run that meets the quality floor."""
    runs = list_runs(dataset_id, limit=500)
    front = set(pareto_front({r["run_id"]: _axes(r["summary"], metric) for r in runs}))
    points = []
    for r in runs:
        quality, latency, cost = _axes(r["summary"], metric)
        points.append({"run_id": r["run_id"], "pipeline": r["pipeline"], "dataset_id": r["dataset_id"],
                       "quality": quality, "latency_ms": latency, "cost_usd": cost,
                       "pareto_optimal": r["run_id"] in front})
    eligible = [p for p in points if p["pareto_optimal"] and p["quality"] >= quality_floor]
    return {"metric": metric, "quality_floor": quality_floor, "points": points,
            "recommended": min(eligible, key=lambda p: (p["latency_ms"], p["cost_usd"]), default=None)}


# ---------- Synthetic QA from live data ----------
SYNTH_PROMPT = """Write {n} question-answer pairs that can be answered from this news transcript alone.
Mix factoid (who/what/where), numeric (how many, percentages) and summary questions.

TRANSCRIPT:
{text}

Use exactly this format for each pair:
Q: <question>
A: <answer in 1-2 sentences>
TYPE: <factoid|numeric|summary>
---"""


def generate_synthetic(n_chunks: int = 10, questions_per_chunk: int = 3,
                       dataset_id: Optional[str] = None) -> EvalDataset:
    docs = get_index().all()
    if not docs:
        raise ValueError("Nothing is indexed yet. Ingest some content first.")
    sample = random.sample(docs, min(n_chunks, len(docs)))
    qa_pairs = []
    for d in sample:
        try:
            reply = generate(SYNTH_PROMPT.format(n=questions_per_chunk, text=d.text)).text
        except LLMError as e:
            logger.warning(f"Synthetic generation failed for {d.chunk_id}: {e}")
            continue
        for block in re.split(r"\n\s*-{3,}\s*", reply):
            q, a = re.search(r"Q:\s*(.+)", block), re.search(r"A:\s*(.+)", block)
            if q and a:
                qtype = re.search(r"TYPE:\s*(\w+)", block)
                qa_pairs.append(QAPair(qa_id=f"synth_{len(qa_pairs) + 1}", question=q.group(1).strip(),
                                       ground_truth_answer=a.group(1).strip(), relevant_chunk_ids=[d.chunk_id],
                                       type=qtype.group(1).lower() if qtype else "factoid"))
    dataset = EvalDataset(
        dataset_id=dataset_id or f"synthetic_{uuid.uuid4().hex[:8]}",
        description=f"Generated from {len(sample)} live chunks",
        transcripts=[{"id": d.chunk_id, "text": d.text, "source": d.metadata.get("source", "live"),
                      "timestamp": d.metadata.get("start_time_utc")} for d in sample],
        qa_pairs=qa_pairs, synthetic=True,
    )
    save_dataset(dataset)
    return dataset
