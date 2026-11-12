"""HTTP surface. Six routes, no magic."""
from __future__ import annotations

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from .answer import answer_question
from .config import SETTINGS, Settings
from .embedding import get_embedder
from .ingest import ingest
from .llm import get_llm
from .retrieve import Retriever
from .store import Store

app = FastAPI(title="rag-harness", version="0.1.0")
_state: dict = {}


def _settings() -> Settings:
    return _state.get("settings", SETTINGS)


def _retriever() -> Retriever:
    if "retriever" not in _state:
        s = _settings()
        store = Store(s.db_path)
        if store.counts()["chunks"] == 0:
            raise HTTPException(409, "index is empty, POST /ingest first")
        _state["retriever"] = Retriever(store=store, embedder=get_embedder(s), settings=s)
    return _state["retriever"]


class AskBody(BaseModel):
    question: str = Field(min_length=3, max_length=500)
    k: int | None = Field(default=None, ge=1, le=20)


@app.get("/health")
def health() -> dict:
    s = _settings()
    store = Store(s.db_path)
    return {
        "ok": True,
        "index": store.counts(),
        "embedder": store.get_meta("embedder", "none"),
        "llm": s.llm,
    }


@app.post("/ingest")
def do_ingest() -> dict:
    _state.pop("retriever", None)
    return ingest(_settings())


@app.get("/search")
def search(q: str = Query(min_length=2), k: int = Query(default=5, ge=1, le=20)) -> dict:
    r = _retriever()
    hits = r.search(q, k)
    return {
        "query": q,
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
        raise HTTPException(404, f"no chunk {chunk_id!r}")
    return {"query": q, "chunk_id": chunk_id, "terms": [{"term": t, "score": round(s, 4)} for t, s in terms]}
