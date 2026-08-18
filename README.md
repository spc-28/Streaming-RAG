# Streaming RAG

Ask questions about a **live radio stream** (BBC World Service by default) and get answers that are
checked against the transcripts they came from.

Audio is recorded in 60-second chunks, transcribed locally with Whisper, split by topic and indexed
for hybrid search. Questions go through a configurable RAG pipeline, and every answer gets a
multi-level faithfulness check. An evaluation harness compares pipelines with significance tests and
a cost/latency/quality Pareto frontier.

## How it works

```
INGEST (every 60 s)                                       QUERY
radio ──ffmpeg──▶ WAV ──Whisper──▶ text                   question
                                    │                        │
                         split where topic shifts            ├─▶ semantic cache hit? ──▶ answer
                                    │                        ▼
                     embed each chunk (MiniLM)            retrieve: BM25 + vector, fused with RRF
                                    │                        ▼
        ┌───────────────┬───────────┼──────────────┐      rerank: cross-encoder
        ▼               ▼           ▼              ▼         ▼
  story tracking   standing     evict stale     index:    recency boost (for "latest" questions)
  (event_id)       queries      cached answers  Chroma       ▼
                   (alerts)                     + BM25    generate: Gemini / Groq, cite HH:MM UTC
                                                             ▼
                                                          verify: claims (NLI) · numbers · citations
```

## The interesting parts

| Idea | Why | Where |
|---|---|---|
| **Hybrid retrieval + RRF** | BM25 catches exact names and figures that embeddings blur; vectors catch paraphrases. Reciprocal Rank Fusion merges the two by rank, so their incompatible score scales never need normalising. | `retrieval.py` |
| **Cross-encoder rerank** | A bi-encoder embeds the question and the chunk separately; a cross-encoder reads them together, which is slower but more precise. So it only rescores the top 3×k candidates. | `retrieval.py` |
| **Multi-level faithfulness** | NLI checks each claim against the source sentences (what RAGAS does). It also checks that every **number** in the answer appears in the sources, keeping the scale ("50 million" ≠ "50 billion"), and that every cited **"09:12 UTC"** belongs to a retrieved chunk. Wrong figures and invented citations are the worst errors in a news answer, and claim-level NLI misses them. | `faithfulness.py` |
| **Reflective retry** | If faithfulness is below 0.5, it doesn't re-run the same thing. It escalates: top-5 → top-10 → top-10 + recency. If all three fail, it returns the best answer, labelled low-confidence. | `pipeline.py` |
| **Adaptive routing** | Regex rules classify the question as factoid, summary, comparison or temporal, then pick the strategy: tight top-k for facts, wide for summaries, recency for "latest". | `pipeline.py` |
| **Recency boost** | For streaming news, newer is usually more relevant. After reranking, a chunk gets `0.3 × 0.5^(age / 30 min)` added to its relevance score. | `retrieval.py` |
| **Story tracking + standing queries** | Consecutive chunks with similar embeddings share an `event_id`, and the LLM is told which segments belong to the same developing story. Standing queries fire alerts when matching audio arrives (idea from *StreamingRAG*, Sankaradas et al. 2024). | `ingest.py` |
| **Freshness-aware semantic cache** | Paraphrased questions reuse a cached answer (cosine ≥ 0.92). On a live stream that answer can go stale, so every new chunk is compared with the cached *questions*, and only the ones it is relevant to are evicted. Unrelated news leaves the cache warm. Questions about "the latest" are never cached. | `cache.py` |
| **Evaluation you can trust** | Runs are compared per question with a paired **Wilcoxon signed-rank test** ("is this real or noise?"). A **Pareto frontier** over (faithfulness ↑, latency ↓, cost ↓) shows which configurations are worth considering. A synthetic QA generator turns live chunks into new test sets. | `evaluation.py`, `metrics.py` |

Every response carries a **trace** with per-stage latency, tokens and cost, which the console shows.

## Code map (~1,800 lines of Python in 15 files)

| File | Responsibility |
|---|---|
| `app/config.py` | All settings (env / `.env` overridable) |
| `app/models.py` | Pydantic schemas |
| `app/store.py` | `Index`: Chroma (dense) + BM25 (sparse) over the same chunks; `embed()` |
| `app/ingest.py` | ffmpeg capture, Whisper, topic chunking, story tracking, standing queries, background capture loop |
| `app/retrieval.py` | RRF fusion, cross-encoder rerank, recency boost |
| `app/llm.py` | `generate()` for Gemini / Groq, token cost |
| `app/faithfulness.py` | Claim NLI + numeric + citation checks |
| `app/pipeline.py` | Strategies, the five pipelines, query router, prompt, trace |
| `app/cache.py` | Freshness-aware semantic answer cache |
| `app/metrics.py` | P@k, recall, MRR, nDCG, LLM-as-judge, Pareto dominance |
| `app/evaluation.py` | Datasets, eval runs (SQLite), benchmark, Wilcoxon comparison, synthetic QA |
| `app/api.py`, `app/main.py` | FastAPI routes; serves the React console in `web/` |
| `app/cli.py` | Evaluation from the command line |

## Quick start

```bash
brew install ffmpeg                 # or: sudo apt install ffmpeg
pip install -r requirements.txt     # first run downloads ~500 MB of HuggingFace models
cd web && npm install && npm run build && cd ..
cp .env.example .env                # add GEMINI_API_KEY (free: https://aistudio.google.com/apikey)
make run                            # http://localhost:8000 (console), /docs (API)
```

Or with Docker: `docker compose up --build`.

Try it without audio:

```bash
curl -X POST localhost:8000/api/v1/ingest/text -H 'Content-Type: application/json' \
  -d '{"text": "The ECB held rates at 4.5 percent. Lagarde said cuts could begin in spring."}'
curl -X POST localhost:8000/api/v1/query -H 'Content-Type: application/json' \
  -d '{"question": "What did the ECB decide?", "pipeline": "full"}'
```

## Evaluation

```bash
python -m app.cli benchmark --dataset sample            # baseline vs hybrid vs full vs adaptive
python -m app.cli eval --dataset sample --pipeline reflective --judge
python -m app.cli compare <baseline_run_id> <candidate_run_id>   # exit code 1 on a significant regression
python -m app.cli pareto --floor 0.7
python -m app.cli synthetic --n-chunks 10               # build a dataset from live captured audio
make test                                               # unit tests, no models or API key needed
```

Evaluation indexes each dataset into its own Chroma collection, so it never touches live data.
`datasets/sample.json` is a 10-question smoke test. With only 3 short transcripts, every pipeline
retrieves the right one, so use `synthetic` on real captured audio to see the pipelines separate.

## API (`/api/v1`)

| Endpoint | Purpose |
|---|---|
| `POST /query` | Ask a question (`pipeline`: baseline, hybrid, full, reflective, adaptive) |
| `POST /ingest/text` | Index text directly |
| `POST /ingest/start`, `/ingest/stop`, `GET /ingest/status` | Background live capture |
| `GET /events` | Detected stories |
| `GET/POST /standing-queries`, `DELETE /standing-queries/{id}`, `GET /standing-queries/alerts` | Alerts |
| `GET /datasets`, `POST /eval/run`, `/eval/benchmark`, `/eval/synthetic`, `/eval/compare` | Evaluation |
| `GET /experiments`, `/experiments/pareto`, `/experiments/{run_id}` | Saved runs |
| `GET /health`, `/stats` | Status, index and cache statistics |

## Trade-offs and limitations

- **BM25 is rebuilt on every insert.** It's O(n), but simple and fine for thousands of chunks. At larger scale, use a search engine with incremental indexing (OpenSearch, or Postgres full-text search).
- **State is per process.** Standing queries and the cache live in memory. Story ids are stored in chunk metadata, so they survive restarts.
- **Thresholds are hand-set** for `all-MiniLM-L6-v2` (topic split 0.35, story 0.5, cache 0.92 / 0.4). A different embedding model needs re-tuning; the evaluation harness is the tool for that.
- **NLI faithfulness is sentence-level.** A claim that combines facts from two source sentences can be marked "unsupported" even though it is correct.
- **The query router is regex rules.** That's cheap and transparent, and the same interface could take an LLM classifier.
