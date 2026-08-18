"""
Freshness-aware semantic answer cache.

A semantic cache returns a stored answer when a new question is a paraphrase
of a cached one ("What did the ECB do?" ≈ "What did the ECB decide?"). On a
live stream that alone is unsafe: an answer goes stale as soon as relevant
audio arrives. So the ingestor shows every new chunk to the cache, and only the
entries whose *question* that chunk is relevant to are evicted. This is the
same question-vs-chunk matching that standing queries use. Unrelated news
leaves the rest of the cache warm.
"""
import threading

import numpy as np

from app.config import get_settings
from app.models import QueryResponse


class SemanticCache:
    def __init__(self, max_entries: int = 256):
        self.max_entries = max_entries
        self._entries: list[tuple[np.ndarray, tuple, QueryResponse]] = []  # (question embedding, key, answer)
        self._lock = threading.Lock()
        self.hits = self.misses = self.invalidated = 0

    def get(self, question_emb: np.ndarray, key: tuple) -> tuple[QueryResponse, float] | None:
        """Most similar cached answer for the same pipeline settings, if similar enough."""
        with self._lock:
            candidates = [(float(emb @ question_emb), resp) for emb, k, resp in self._entries if k == key]
            sim, resp = max(candidates, key=lambda c: c[0], default=(0.0, None))
            if resp is not None and sim >= get_settings().cache_hit_threshold:
                self.hits += 1
                return resp, sim
            self.misses += 1
            return None

    def put(self, question_emb: np.ndarray, key: tuple, response: QueryResponse) -> None:
        with self._lock:
            self._entries.append((question_emb, key, response))
            del self._entries[:-self.max_entries]

    def invalidate(self, chunk_emb: np.ndarray) -> int:
        """Evict answers to questions this new chunk is relevant to. Returns how many."""
        threshold = get_settings().cache_invalidate_threshold
        with self._lock:
            before = len(self._entries)
            self._entries = [e for e in self._entries if float(e[0] @ chunk_emb) < threshold]
            evicted = before - len(self._entries)
            self.invalidated += evicted
            return evicted

    def stats(self) -> dict:
        return {"entries": len(self._entries), "hits": self.hits,
                "misses": self.misses, "invalidated": self.invalidated}


answer_cache = SemanticCache()
