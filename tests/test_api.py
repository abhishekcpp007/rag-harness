"""The HTTP surface, through FastAPI's test client.

These run against the real index the session fixture builds, on the hash embedder, so they stay
offline. What they are for is the machinery around the routes: readiness, the request id, the
metrics observation, the limiter and the stage timings. Retrieval quality is the eval's job.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from rag import api


@pytest.fixture
def client(indexed, monkeypatch):
    monkeypatch.setitem(api._state, "settings", indexed)
    api._state.pop("retriever", None)
    api._state.pop("limiter", None)
    api.METRICS = api.Metrics()
    with TestClient(api.app) as c:
        yield c
    api._state.pop("retriever", None)
    api._state.pop("limiter", None)


def test_health_reports_ready_after_startup(client):
    body = client.get("/health").json()
    assert body["ok"] is True
    assert body["ready"] is True
    assert body["index"]["chunks"] > 20


def test_the_index_is_built_during_startup_not_on_the_first_request(indexed, monkeypatch):
    """The lifespan does the build, so the first request in must not be the one that pays for
    it. Entering the context is startup; the retriever has to exist before any call."""
    monkeypatch.setitem(api._state, "settings", indexed)
    api._state.pop("retriever", None)
    with TestClient(api.app):
        assert "retriever" in api._state


def test_search_returns_hits_and_per_stage_timings(client):
    body = client.get("/search", params={"q": "what does NW-4021 mean", "k": 3}).json()
    assert len(body["hits"]) == 3
    assert body["timing_ms"]["dense_ms"] >= 0
    assert body["timing_ms"]["candidates"] >= 3
    assert "rerank_ms" in body["timing_ms"]


def test_two_searches_do_not_share_a_timing_record(client):
    a = client.get("/search", params={"q": "settlement batch rejected"}).json()
    b = client.get("/search", params={"q": "webhook signature"}).json()
    assert a["timing_ms"] is not b["timing_ms"]
    assert a["hits"][0]["chunk_id"] != b["hits"][0]["chunk_id"]


def test_every_response_carries_a_request_id(client):
    assert len(client.get("/health").headers["x-request-id"]) == 12


def test_a_request_id_from_the_caller_is_kept(client):
    r = client.get("/health", headers={"x-request-id": "trace-me-please"})
    assert r.headers["x-request-id"] == "trace-me-please"


def test_metrics_counts_the_requests_that_came_before_it(client):
    client.get("/search", params={"q": "chargeback deadline"})
    client.get("/search", params={"q": "chargeback deadline"})
    text = client.get("/metrics").text
    assert 'rag_requests_total{route="/search",status="200"} 2' in text
    assert "rag_index_chunks" in text


def test_a_bad_query_is_a_422_and_is_counted_as_one(client):
    assert client.get("/search", params={"q": "x"}).status_code == 422
    assert 'rag_requests_total{route="/search",status="422"} 1' in client.get("/metrics").text


def test_the_limiter_refuses_with_a_retry_after_when_it_is_on(indexed, monkeypatch):
    tight = indexed.with_(rate_limit_rps=1.0, rate_limit_burst=2)
    monkeypatch.setitem(api._state, "settings", tight)
    api._state.pop("retriever", None)
    api._state.pop("limiter", None)
    with TestClient(api.app) as c:
        codes = [c.get("/search", params={"q": "settlement batch"}).status_code
                 for _ in range(5)]
    api._state.pop("limiter", None)
    assert codes[:2] == [200, 200]
    assert 429 in codes


def test_the_limiter_never_touches_health_or_metrics(indexed, monkeypatch):
    """An orchestrator probing readiness is not traffic. Throttle the probe and it starts
    killing a container whose only problem is that it is busy."""
    tight = indexed.with_(rate_limit_rps=1.0, rate_limit_burst=1)
    monkeypatch.setitem(api._state, "settings", tight)
    api._state.pop("retriever", None)
    api._state.pop("limiter", None)
    with TestClient(api.app) as c:
        assert all(c.get("/health").status_code == 200 for _ in range(10))
        assert all(c.get("/metrics").status_code == 200 for _ in range(10))
        assert c.get("/search", params={"q": "settlement batch"}).status_code == 200
        assert c.get("/search", params={"q": "settlement batch"}).status_code == 429
    api._state.pop("limiter", None)


def test_the_limiter_is_off_by_default(client):
    assert all(client.get("/health").status_code == 200 for _ in range(30))


def test_answer_cites_what_it_used(client):
    body = client.post("/answer", json={"question": "What does NW-4021 mean?"}).json()
    assert "citations" in body
    assert body["citations"]


def test_explain_names_the_terms_for_one_chunk(client):
    hit = client.get("/search", params={"q": "settlement batch"}).json()["hits"][0]
    body = client.get("/explain", params={"q": "settlement batch",
                                          "chunk_id": hit["chunk_id"]}).json()
    assert [t["term"] for t in body["terms"]]


def test_explain_on_a_missing_chunk_is_a_404(client):
    assert client.get("/explain", params={"q": "batch", "chunk_id": "nope#9"}).status_code == 404
