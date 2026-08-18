"""
The index: every chunk is stored in ChromaDB (dense vectors) and in an
in-memory BM25 index (keywords). Chroma is the source of truth; BM25 is
rebuilt from it on startup.
"""
import re
import threading
from datetime import datetime, timezone
from functools import lru_cache
from typing import Optional

import numpy as np

from app.config import get_settings
from app.models import Chunk

LIVE_INDEX = "rag_chunks"


@lru_cache
def _embedder():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(get_settings().embedding_model)


def embed(texts: str | list[str]) -> np.ndarray:
    """Unit-normalised embeddings, so a dot product is a cosine similarity."""
    return _embedder().encode(texts, normalize_embeddings=True, show_progress_bar=False)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def _tokens(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def _in_window(meta: dict, time_range: Optional[tuple[int, int]]) -> bool:
    if not time_range:
        return True
    return meta.get("unix_start", 0) <= time_range[1] and meta.get("unix_end", 0) >= time_range[0]


@lru_cache
def _chroma():
    import chromadb
    return chromadb.PersistentClient(path=str(get_settings().data_dir / "chroma"))


class Index:
    """Hybrid index over one Chroma collection."""

    def __init__(self, name: str):
        self.name = name
        self._lock = threading.Lock()
        self._collection = _chroma().get_or_create_collection(name, metadata={"hnsw:space": "cosine"})
        data = self._collection.get(include=["documents", "metadatas"])
        self._docs = [Chunk(chunk_id=i, text=t, metadata=m or {})
                      for i, t, m in zip(data["ids"], data["documents"], data["metadatas"])]
        self._rebuild_bm25()

    def _rebuild_bm25(self) -> None:
        # Full rebuild on every add: O(n), but simple and fine for a few thousand chunks.
        from rank_bm25 import BM25Okapi
        self._bm25 = BM25Okapi([_tokens(d.text) for d in self._docs]) if self._docs else None

    def add(self, chunk_id: str, text: str, metadata: dict, embedding: np.ndarray) -> None:
        meta = {k: v for k, v in metadata.items() if isinstance(v, (str, int, float, bool))}
        self._collection.upsert(ids=[chunk_id], documents=[text],
                                embeddings=[embedding.tolist()], metadatas=[meta])
        with self._lock:
            self._docs = [d for d in self._docs if d.chunk_id != chunk_id]
            self._docs.append(Chunk(chunk_id=chunk_id, text=text, metadata=meta))
            self._rebuild_bm25()

    def vector_search(self, query: str, k: int, time_range=None) -> list[Chunk]:
        if not self._docs:
            return []
        where = None
        if time_range:
            where = {"$and": [{"unix_start": {"$lte": time_range[1]}},
                              {"unix_end": {"$gte": time_range[0]}}]}
        res = self._collection.query(
            query_embeddings=[embed(query).tolist()], n_results=min(k, len(self._docs)),
            where=where, include=["documents", "metadatas", "distances"],
        )
        return [Chunk(chunk_id=i, text=t, metadata=m, score=1 - dist)
                for i, t, m, dist in zip(res["ids"][0], res["documents"][0],
                                         res["metadatas"][0], res["distances"][0])]

    def keyword_search(self, query: str, k: int, time_range=None) -> list[Chunk]:
        with self._lock:
            if not self._bm25:
                return []
            scores = self._bm25.get_scores(_tokens(query))
            ranked = [self._docs[i].model_copy(update={"score": float(scores[i])})
                      for i in np.argsort(scores)[::-1]
                      if scores[i] > 0 and _in_window(self._docs[i].metadata, time_range)]
        return ranked[:k]

    def all(self) -> list[Chunk]:
        return list(self._docs)

    def count(self) -> int:
        return len(self._docs)

    def clear(self) -> None:
        _chroma().delete_collection(self.name)
        self._collection = _chroma().get_or_create_collection(self.name, metadata={"hnsw:space": "cosine"})
        with self._lock:
            self._docs, self._bm25 = [], None

    def stats(self) -> dict:
        metas = [d.metadata for d in self._docs]
        starts = [m["unix_start"] for m in metas if m.get("unix_start")]
        ends = [m["unix_end"] for m in metas if m.get("unix_end")]
        words = sum(m.get("word_count", 0) for m in metas)

        def iso(ts):
            return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat() if ts else None

        return {
            "total_docs": len(metas),
            "total_words": words,
            "avg_words": words // len(metas) if metas else 0,
            "oldest_time": iso(min(starts, default=None)),
            "newest_time": iso(max(ends, default=None)),
            "span_minutes": (max(ends) - min(starts)) / 60 if starts and ends else 0,
        }


@lru_cache
def get_index(name: str = LIVE_INDEX) -> Index:
    return Index(name)
