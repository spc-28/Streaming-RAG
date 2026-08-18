"""Pydantic schemas shared by the API, the pipelines and evaluation."""
from typing import Literal, Optional

from pydantic import BaseModel, Field

PipelineName = Literal["baseline", "hybrid", "full", "reflective", "adaptive"]


class Chunk(BaseModel):
    """An indexed transcript chunk plus the scores each retrieval stage attached to it.

    `score` is always the current stage's relevance in roughly [0, 1], so later
    stages (like the recency boost) can build on it.
    """
    chunk_id: str
    text: str
    metadata: dict = {}
    score: float = 0.0
    rrf_score: Optional[float] = None
    rerank_score: Optional[float] = None
    recency_boost: Optional[float] = None


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1)
    pipeline: PipelineName = "full"
    top_k: Optional[int] = None          # baseline/hybrid/full only; reflective & adaptive choose their own
    check_faithfulness: bool = True
    minutes_ago: Optional[int] = None    # only search audio from around N minutes ago
    time_window_minutes: int = 30
    use_cache: bool = True


class ClaimVerdict(BaseModel):
    claim: str
    verdict: Literal["supported", "partially_supported", "unsupported", "contradicted"]
    entailment_score: float


class FaithfulnessReport(BaseModel):
    faithfulness_score: float
    total_claims: int = 0
    supported: int = 0
    partially_supported: int = 0
    unsupported: int = 0
    contradicted: int = 0
    numeric_accuracy: float = 1.0
    citation_accuracy: float = 1.0
    claim_results: list[ClaimVerdict] = []


class QueryResponse(BaseModel):
    trace_id: str
    question: str
    answer: str
    pipeline: str
    sources: list[Chunk] = []
    faithfulness: Optional[FaithfulnessReport] = None
    query_type: Optional[str] = None     # set by the adaptive pipeline
    retry_count: int = 0                 # set by the reflective pipeline
    cached: bool = False
    config: dict = {}
    trace: dict = {}


class QAPair(BaseModel):
    qa_id: str
    question: str
    ground_truth_answer: str
    relevant_chunk_ids: list[str] = []   # transcript ids; empty = unanswerable
    type: str = "factoid"


class EvalDataset(BaseModel):
    dataset_id: str
    description: str = ""
    transcripts: list[dict] = []         # {id, text, source?, timestamp?}
    qa_pairs: list[QAPair] = []
    synthetic: bool = False
