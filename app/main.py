"""
FastAPI app: the API at /api/v1, a radio-stream proxy, and the React console at /.

Run: uvicorn app.main:app --reload
"""
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from app import __version__
from app.api import router
from app.config import get_settings, setup_logging
from app.ingest import get_ingestor
from app.llm import LLMError

setup_logging()
UI_DIR = Path(__file__).resolve().parent.parent / "web" / "dist"


@asynccontextmanager
async def lifespan(app: FastAPI):
    get_ingestor()  # load the embedding model and build the BM25 index before the first request
    yield


app = FastAPI(title="Streaming RAG", version=__version__, lifespan=lifespan)
app.include_router(router, prefix="/api/v1")

for error, status in [(FileNotFoundError, 404), (ValueError, 400), (LLMError, 502)]:
    app.add_exception_handler(
        error, lambda request, exc, status=status: JSONResponse({"detail": str(exc)}, status_code=status))


@app.get("/stream-proxy", include_in_schema=False)
async def stream_proxy():
    """Relay the live radio stream so the browser can play it without CORS problems."""
    async def relay():
        async with httpx.AsyncClient(timeout=None, follow_redirects=True) as client:
            async with client.stream("GET", get_settings().stream_url) as resp:
                async for chunk in resp.aiter_bytes():
                    yield chunk

    return StreamingResponse(relay(), media_type="audio/mpeg")


if UI_DIR.exists():
    app.mount("/", StaticFiles(directory=UI_DIR, html=True), name="ui")
else:
    @app.get("/", include_in_schema=False)
    def ui_not_built() -> dict:
        return {"message": "UI not built: cd web && npm install && npm run build", "api_docs": "/docs"}
