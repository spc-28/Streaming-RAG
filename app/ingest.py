"""
Ingestion: live audio → text → topic chunks → index.

Every new chunk on the live index is also shown to three "watchers":
  - story tracking    consecutive similar chunks share an `event_id` (one developing story)
  - standing queries  saved questions that raise an alert when a matching chunk arrives
  - answer cache      cached answers that this chunk makes stale are evicted
"""
import logging
import subprocess
import tempfile
import threading
import uuid
import wave
from collections import deque
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np

from app.cache import answer_cache
from app.config import get_settings
from app.faithfulness import split_sentences
from app.store import Index, cosine, embed, get_index

logger = logging.getLogger(__name__)


def _now() -> datetime:
    return datetime.now(timezone.utc)


# ---------- Audio → text ----------
def capture_audio(seconds: int) -> Path:
    """Record `seconds` of the live stream to a 16 kHz mono WAV with ffmpeg."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        path = Path(f.name)
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", get_settings().stream_url,
           "-t", str(seconds), "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", str(path)]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=seconds + 30)
    except FileNotFoundError:
        raise RuntimeError("ffmpeg is not installed (brew install ffmpeg / apt install ffmpeg)")
    except subprocess.CalledProcessError as e:
        path.unlink(missing_ok=True)
        raise RuntimeError(f"ffmpeg failed: {e.stderr[:200]}")
    except subprocess.TimeoutExpired:
        path.unlink(missing_ok=True)
        raise RuntimeError("ffmpeg timed out reading the stream")
    return path


@lru_cache
def _whisper():
    from faster_whisper import WhisperModel
    s = get_settings()
    return WhisperModel(s.whisper_model, device=s.whisper_device, compute_type=s.whisper_compute_type)


def transcribe(path: Path) -> str:
    # ffmpeg already wrote 16 kHz mono 16-bit PCM, so read it directly instead of
    # letting faster-whisper decode the file with PyAV.
    with wave.open(str(path)) as wav:
        pcm = np.frombuffer(wav.readframes(wav.getnframes()), dtype=np.int16)
    segments, _ = _whisper().transcribe(pcm.astype(np.float32) / 32768, language="en",
                                        beam_size=5, vad_filter=True)
    return " ".join(seg.text.strip() for seg in segments)


# ---------- Text → chunks ----------
def chunk_by_topic(text: str, min_sentences: int = 2, max_sentences: int = 15) -> list[str]:
    """Cut between sentences where the topic shifts.

    At each gap, compare the mean embedding of the 2 sentences before it with the
    2 after it; a cosine similarity below the threshold marks a new topic.
    """
    sentences = split_sentences(text)
    if len(sentences) < 2 * min_sentences:
        return [" ".join(sentences) or text.strip()]
    emb = embed(sentences)
    threshold = get_settings().topic_split_threshold
    chunks, start = [], 0
    for i in range(min_sentences, len(sentences) - min_sentences + 1):
        if i - start < min_sentences:
            continue
        topic_shift = cosine(emb[i - 2:i].mean(0), emb[i:i + 2].mean(0)) < threshold
        if topic_shift or i - start >= max_sentences:
            chunks.append(" ".join(sentences[start:i]))
            start = i
    chunks.append(" ".join(sentences[start:]))
    return chunks


# ---------- Standing queries ----------
class StandingQueries:
    """Saved questions checked against every new chunk (idea from StreamingRAG, Sankaradas et al. 2024)."""

    def __init__(self):
        self.queries: dict[str, dict] = {}
        self.alerts: deque[dict] = deque(maxlen=200)
        self._embeddings: dict[str, np.ndarray] = {}
        self._count = 0
        self._lock = threading.Lock()

    def add(self, text: str, threshold: Optional[float] = None) -> dict:
        emb = embed(text)
        with self._lock:
            self._count += 1
            qid = f"sq_{self._count:03d}"
            self._embeddings[qid] = emb
            self.queries[qid] = {"id": qid, "query": text, "total_alerts": 0, "created_at": _now().isoformat(),
                                 "threshold": threshold or get_settings().standing_query_threshold}
            return self.queries[qid]

    def remove(self, qid: str) -> bool:
        with self._lock:
            self._embeddings.pop(qid, None)
            return self.queries.pop(qid, None) is not None

    def check(self, chunk_id: str, text: str, emb: np.ndarray, time: str) -> list[dict]:
        fired = []
        with self._lock:
            for qid, q in self.queries.items():
                sim = float(self._embeddings[qid] @ emb)
                if sim >= q["threshold"]:
                    q["total_alerts"] += 1
                    alert = {"query_id": qid, "query_text": q["query"], "chunk_id": chunk_id,
                             "similarity": round(sim, 3), "preview": text[:200], "time": time}
                    self.alerts.appendleft(alert)
                    fired.append(alert)
        return fired


# ---------- Ingestor ----------
class Ingestor:
    """Chunks text and indexes it. With `watch=True` (the live index) it also runs the watchers."""

    def __init__(self, index: Index, watch: bool = True):
        self.index = index
        self.watch = watch
        self.standing = StandingQueries()
        self._lock = threading.Lock()
        self._last_emb: Optional[np.ndarray] = None
        self._event_id = max((d.metadata.get("event_id", 0) for d in index.all()), default=0)

    def _track_story(self, emb: np.ndarray) -> tuple[int, bool]:
        """Same story as the previous chunk if their embeddings are similar enough."""
        with self._lock:
            is_new = self._last_emb is None or \
                cosine(self._last_emb, emb) < get_settings().event_similarity_threshold
            if is_new:
                self._event_id += 1
            self._last_emb = emb
            return self._event_id, is_new

    def ingest_text(self, text: str, source: str = "manual", base_id: Optional[str] = None,
                    start: Optional[datetime] = None) -> dict:
        start = start or _now()
        end = start + timedelta(seconds=get_settings().chunk_duration_seconds)
        base_id = base_id or f"text_{uuid.uuid4().hex[:8]}"
        chunks = chunk_by_topic(text)
        out = {"chunk_id": base_id, "word_count": len(text.split()), "preview": text[:200],
               "indexed_ids": [], "new_events": [], "alerts": [], "cache_evictions": 0}

        for i, (chunk, emb) in enumerate(zip(chunks, embed(chunks))):
            cid = f"{base_id}_c{i}"
            meta = {"source": source, "parent_id": base_id, "word_count": len(chunk.split()),
                    "start_time_utc": start.isoformat(), "end_time_utc": end.isoformat(),
                    "unix_start": int(start.timestamp()), "unix_end": int(end.timestamp())}
            if self.watch:
                meta["event_id"], is_new = self._track_story(emb)
                if is_new:
                    out["new_events"].append(meta["event_id"])
                out["alerts"] += self.standing.check(cid, chunk, emb, meta["start_time_utc"])
                out["cache_evictions"] += answer_cache.invalidate(emb)
            self.index.add(cid, chunk, meta, emb)
            out["indexed_ids"].append(cid)
        return out

    def ingest_live(self) -> dict:
        """Capture one chunk of the live stream, transcribe it, index it."""
        s = get_settings()
        start = _now()
        path = capture_audio(s.chunk_duration_seconds)
        try:
            text = transcribe(path)
        finally:
            path.unlink(missing_ok=True)
        if not text.strip():
            raise RuntimeError("No speech detected in this chunk")
        return self.ingest_text(text, source=s.stream_url, base_id=f"live_{start:%Y%m%dT%H%M%S}", start=start)


def list_events(index: Index) -> list[dict]:
    """Group chunks into stories by their `event_id`, newest story first."""
    stories: dict[int, list] = {}
    for d in sorted(index.all(), key=lambda d: d.metadata.get("unix_start", 0)):
        if (eid := d.metadata.get("event_id")) is not None:
            stories.setdefault(eid, []).append(d)
    return [{
        "event_id": eid,
        "num_chunks": len(docs),
        "chunk_ids": [d.chunk_id for d in docs],
        "start_time": docs[0].metadata.get("start_time_utc"),
        "end_time": docs[-1].metadata.get("end_time_utc"),
        "duration_minutes": round((docs[-1].metadata.get("unix_end", 0)
                                   - docs[0].metadata.get("unix_start", 0)) / 60, 1),
        "preview": docs[0].text[:200],
    } for eid, docs in sorted(stories.items(), reverse=True)]


# ---------- Background capture ----------
class LiveCapture:
    """Background thread: capture → transcribe → ingest, until stopped."""

    def __init__(self, ingestor: Ingestor):
        self.ingestor = ingestor
        self.recent: deque[dict] = deque(maxlen=20)
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def status(self) -> dict:
        alive = self._thread is not None and self._thread.is_alive()
        return {"running": alive and not self._stop.is_set(),
                "stopping": alive and self._stop.is_set(),  # finishing the chunk in progress
                "recent": list(self.recent)}

    def start(self) -> None:
        self._stop.clear()  # if a stopping thread is still alive, this just lets it carry on
        if not (self._thread and self._thread.is_alive()):
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self.recent.appendleft({**self.ingestor.ingest_live(), "time": _now().isoformat()})
            except Exception as e:
                logger.warning(f"Live capture failed: {e}")
                self.recent.appendleft({"error": str(e)[:200], "time": _now().isoformat()})
                self._stop.wait(10)  # back off instead of spinning on a persistent failure


@lru_cache
def get_ingestor() -> Ingestor:
    return Ingestor(get_index())


@lru_cache
def get_live_capture() -> LiveCapture:
    return LiveCapture(get_ingestor())
