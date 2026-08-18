"""All settings. Override any of them with an environment variable or a .env file."""
import logging
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    data_dir: Path = Path("data")
    datasets_dir: Path = Path("datasets")

    # Live audio source + speech-to-text (faster-whisper, runs locally)
    stream_url: str = "http://stream.live.vc.bbcmedia.co.uk/bbc_world_service"
    chunk_duration_seconds: int = 60
    whisper_model: str = "base"
    whisper_device: str = "cpu"
    whisper_compute_type: str = "int8"

    # Local HuggingFace models
    embedding_model: str = "all-MiniLM-L6-v2"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    nli_model: str = "cross-encoder/nli-deberta-v3-small"

    # LLM (both providers have a free tier)
    llm_provider: Literal["gemini", "groq"] = "gemini"
    gemini_api_key: str = ""
    gemini_model: str = "gemini-3.5-flash-lite"
    groq_api_key: str = ""
    groq_model: str = "llama-3.3-70b-versatile"
    llm_temperature: float = 0.2

    # Retrieval
    candidates_k: int = 15            # candidates per retriever before fusion
    rrf_constant: int = 60
    recency_half_life_minutes: float = 30.0
    recency_max_boost: float = 0.3

    # Streaming features (all thresholds are cosine similarities)
    topic_split_threshold: float = 0.35       # chunk boundary when similarity drops below
    event_similarity_threshold: float = 0.5   # consecutive chunks above this = same story
    standing_query_threshold: float = 0.55
    cache_hit_threshold: float = 0.92         # question ≈ cached question
    cache_invalidate_threshold: float = 0.4   # new chunk is relevant to a cached question

    reflective_faithfulness_threshold: float = 0.5
    log_level: str = "INFO"


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    return settings


def setup_logging() -> None:
    logging.basicConfig(level=get_settings().log_level, datefmt="%H:%M:%S",
                        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s")
    for noisy in ("httpx", "httpcore", "chromadb", "sentence_transformers"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
