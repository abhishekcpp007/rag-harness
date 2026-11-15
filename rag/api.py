"""HTTP surface. Seven routes, no magic.

What is here beyond the routes is the small amount of machinery a service needs before anyone
else points traffic at it, and each piece is here because something measurable went wrong first.

The index is built during startup rather than on the first request. It used to be lazy, and
`bench/load.py` priced that at 16 concurrent clients, 64 requests, against a cold process:

    lazy, no lock      p50 86.2ms   p95 2718.7ms   max 2729.1ms
    lazy, locked       p50 35.0ms   p95  400.5ms   max  431.2ms
    built at startup   p50 35.2ms   p95   57.3ms   max   77.2ms

The first row is the version that shipped. Sixteen threads arrived, none of them found a
retriever, and each one loaded the encoder for itself. The lock in `_build` collapses that into
one build, which is most of the fix, and warming at startup moves the remaining 400ms off the
request path entirely. It also makes the readiness probe honest: the port does not answer until
the index is loaded.

One middleware gives every request an id, writes its log line, records its latency and applies
the limiter, rather than that being sprinkled through the handlers.
"""
from __future__ import annotations

import threading
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field

from .answer import answer_question
from .config import SETTINGS, Settings
from .embedding import get_embedder
from .ingest import ingest
from .llm import get_llm
from .observability import Metrics, TokenBucket, log_request
from .retrieve import Retriever
from .store import Store

_state: dict = {}
_build_lock = threading.Lock()
METRICS = Metrics()
# A readiness probe and a metrics scrape are not traffic, and throttling them means an orchestrator
# starts killing a container that is only busy.
UNLIMITED_ROUTES = ("/health", "/metrics")


def _settings() -> Settings:
    return _state.get("settings", SETTINGS)


def _limiter() -> TokenBucket:
    if "limiter" not in _state:
        s = _settings()
        _state["limiter"] = TokenBucket(s.rate_limit_rps, s.rate_limit_burst)
    return _state["limiter"]


def _build() -> Retriever | None:
    """Build the retriever once, whichever thread gets here first.

    The lock is the point. Without it every thread that arrives before the first build finishes
    starts its own, which means loading the encoder once per concurrent request.
    """
    with _build_lock:
        if "retriever" in _state:
            return _state["retriever"]
        s = _settings()
        store = Store(s.db_path)
        if store.counts()["chunks"] == 0:
            return None
        _state["retriever"] = Retriever(store=store, embedder=get_embedder(s), settings=s)
        return _state["retriever"]


def _retriever() -> Retriever:
    r = _state.get("retriever") or _build()
    if r is None:
        raise HTTPException(409, "index is empty, POST /ingest first")
    return r


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Warm the index before the port starts answering. An empty index is not an error here:
    the container may well be started so that something can POST /ingest into it."""
    _build()
    yield
    _state.clear()


app = FastAPI(title="rag-harness", version="0.1.0", lifespan=lifespan)


@app.middleware("http")
async def observe(request: Request, call_next):
    route = request.url.path
    rid = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
    client = request.client.host if request.client else "unknown"
    exempt = route in UNLIMITED_ROUTES
    allowed, wait_s = (True, 0.0) if exempt else _limiter().take(client)
    started = time.perf_counter()
    if not allowed:
        METRICS.count_limited()
        response = PlainTextResponse("rate limited\n", status_code=429)
        response.headers["retry-after"] = str(max(1, int(wait_s + 0.999)))
    else:
        response = await call_next(request)
    ms = (time.perf_counter() - started) * 1000
    response.headers["x-request-id"] = rid
    METRICS.observe(route, response.status_code, ms)
    log_request(
        {
            "request_id": rid,
            "route": route,
            "method": request.method,
            "status": response.status_code,
            "ms": round(ms, 2),
            "query": (request.query_params.get("q") or "")[:120],
        },
        _settings().log_format,
    )
    return response


class AskBody(BaseModel):
    question: str = Field(min_length=3, max_length=500)
    k: int | None = Field(default=None, ge=1, le=20)


@app.get("/health")
def health() -> dict:
    s = _settings()
    store = Store(s.db_path)
    counts = store.counts()
    return {
        "ok": True,
        "ready": "retriever" in _state,
        "index": counts,
        "embedder": store.get_meta("embedder", "none"),
        "llm": s.llm,
    }


@app.get("/metrics", response_class=PlainTextResponse)
def metrics() -> str:
    return METRICS.render(Store(_settings().db_path).counts())


@app.post("/ingest")
def do_ingest() -> dict:
    with _build_lock:
        _state.pop("retriever", None)
    stats = ingest(_settings())
    _build()
    return stats


@app.get("/search")
def search(q: str = Query(min_length=2), k: int = Query(default=5, ge=1, le=20)) -> dict:
    r = _retriever()
    trace: dict = {}
    hits = r.search(q, k, trace=trace)
    return {
        "query": q,
        "timing_ms": trace,
        "hits": [
            {
                "chunk_id": h.chunk_id,
                "doc_id": h.doc_id,
                "heading_path": h.heading_path,
                "score": round(h.score, 5),
                "signals": h.signals,
                "text": h.text,
            }
            for h in hits
        ],
    }


@app.post("/answer")
def answer(body: AskBody) -> dict:
    s = _settings()
    r = _retriever()
    if body.k:
        s = s.with_(final_k=body.k)
    return answer_question(body.question, r, get_llm(s.llm), s).to_dict()


@app.get("/explain")
def explain(q: str = Query(min_length=2), chunk_id: str = Query()) -> dict:
    """Per-term BM25 contribution for one chunk. The first thing to reach for when a
    ranking looks wrong."""
    r = _retriever()
    try:
        terms = r.bm25.explain(q, chunk_id)
    except ValueError:
        raise HTTPException(404, f"no chunk {chunk_id!r}") from None
    return {
        "query": q,
        "chunk_id": chunk_id,
        "terms": [{"term": t, "score": round(s, 4)} for t, s in terms],
    }
