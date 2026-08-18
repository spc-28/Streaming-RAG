"""HTTP API, mounted at /api/v1. Handlers stay thin and call into the modules that do the work."""
import time
from typing import Optional

from fastapi import APIRouter
from pydantic import BaseModel, Field

from app import __version__
from app import evaluation as ev
from app.cache import answer_cache
from app.config import get_settings
from app.ingest import get_ingestor, get_live_capture, list_events
from app.models import PipelineName, QueryRequest, QueryResponse
from app.pipeline import answer, classify_query
from app.store import embed, get_index

router = APIRouter()


# ---------- Status ----------
@router.get("/health")
def health() -> dict:
    return {"status": "ok", "version": __version__, "indexed_chunks": get_index().count(),
            "llm_provider": get_settings().llm_provider}


@router.get("/stats")
def stats() -> dict:
    return {**get_index().stats(), "cache": answer_cache.stats()}


# ---------- Query ----------
@router.post("/query", response_model=QueryResponse)
def query(req: QueryRequest) -> QueryResponse:
    # "Latest news" questions and relative time windows change meaning as the
    # stream moves on, so they are never served from the cache.
    cacheable = req.use_cache and req.minutes_ago is None and classify_query(req.question) != "temporal"
    if cacheable:
        started = time.perf_counter()
        question_emb, key = embed(req.question), (req.pipeline, req.top_k, req.check_faithfulness)
        if hit := answer_cache.get(question_emb, key):
            cached, similarity = hit
            span = {"name": "cache_hit", "similarity": round(similarity, 3), "original_question": cached.question,
                    "duration_ms": round((time.perf_counter() - started) * 1000, 1)}
            trace = {"trace_id": cached.trace_id, "total_duration_ms": span["duration_ms"],
                     "total_cost_usd": 0.0, "spans": [span]}
            return cached.model_copy(update={"cached": True, "question": req.question, "trace": trace})

    response = answer(req, get_index())
    if cacheable and response.config.get("model"):  # only cache real LLM answers, not errors
        answer_cache.put(question_emb, key, response)
    return response


# ---------- Ingestion ----------
class TextIngest(BaseModel):
    text: str = Field(..., min_length=1)
    source: str = "manual"


@router.post("/ingest/text")
def ingest_text(req: TextIngest) -> dict:
    return get_ingestor().ingest_text(req.text, source=req.source)


@router.get("/ingest/status")
def capture_status() -> dict:
    return get_live_capture().status()


@router.post("/ingest/start")
def capture_start() -> dict:
    get_live_capture().start()
    return get_live_capture().status()


@router.post("/ingest/stop")
def capture_stop() -> dict:
    get_live_capture().stop()
    return get_live_capture().status()


@router.get("/events")
def events() -> dict:
    return {"events": list_events(get_index())}


# ---------- Standing queries ----------
class StandingQueryIn(BaseModel):
    query: str = Field(..., min_length=1)
    threshold: Optional[float] = None


@router.get("/standing-queries")
def standing_queries() -> dict:
    return {"queries": list(get_ingestor().standing.queries.values())}


@router.post("/standing-queries")
def add_standing_query(req: StandingQueryIn) -> dict:
    return get_ingestor().standing.add(req.query, req.threshold)


@router.delete("/standing-queries/{query_id}")
def remove_standing_query(query_id: str) -> dict:
    return {"removed": get_ingestor().standing.remove(query_id)}


@router.get("/standing-queries/alerts")
def alerts() -> dict:
    return {"alerts": list(get_ingestor().standing.alerts)}


# ---------- Evaluation ----------
class EvalRequest(BaseModel):
    dataset_id: str = "sample"
    pipeline: PipelineName = "full"
    judge: bool = False


class BenchmarkRequest(BaseModel):
    dataset_id: str = "sample"
    pipelines: Optional[list[PipelineName]] = None
    judge: bool = False


class SyntheticRequest(BaseModel):
    n_chunks: int = 10
    questions_per_chunk: int = 3
    dataset_id: Optional[str] = None


class CompareRequest(BaseModel):
    baseline_id: str
    candidate_id: str
    alpha: float = 0.05


@router.get("/datasets")
def datasets() -> dict:
    return {"datasets": ev.list_datasets()}


@router.post("/eval/run")
def run_eval(req: EvalRequest) -> dict:
    return ev.run_eval(ev.load_dataset(req.dataset_id), req.pipeline, req.judge)


@router.post("/eval/benchmark")
def benchmark(req: BenchmarkRequest) -> dict:
    return ev.benchmark(ev.load_dataset(req.dataset_id), req.pipelines, req.judge)


@router.post("/eval/synthetic")
def synthetic(req: SyntheticRequest) -> dict:
    dataset = ev.generate_synthetic(req.n_chunks, req.questions_per_chunk, req.dataset_id)
    return {"dataset_id": dataset.dataset_id, "num_qa_pairs": len(dataset.qa_pairs)}


@router.post("/eval/compare")
def compare(req: CompareRequest) -> dict:
    return ev.compare_runs(req.baseline_id, req.candidate_id, req.alpha)


@router.get("/experiments")
def experiments(dataset_id: Optional[str] = None) -> dict:
    return {"runs": ev.list_runs(dataset_id)}


@router.get("/experiments/pareto")
def pareto(metric: str = "faithfulness", quality_floor: float = 0.7, dataset_id: Optional[str] = None) -> dict:
    return ev.pareto(metric, quality_floor, dataset_id)


@router.get("/experiments/{run_id}")
def experiment(run_id: str) -> dict:
    return ev.get_run(run_id)
