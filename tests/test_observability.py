"""Metrics and the rate limiter. No HTTP here, so a failure points at one of the two."""
from __future__ import annotations

import threading

from rag.observability import BUCKETS_MS, Metrics, TokenBucket


def test_a_request_lands_in_the_first_bucket_it_fits():
    m = Metrics()
    m.observe("/search", 200, 3.0)
    text = m.render({"chunks": 54, "documents": 12})
    assert 'rag_request_duration_ms_bucket{route="/search",le="1"} 0' in text
    assert 'rag_request_duration_ms_bucket{route="/search",le="5"} 1' in text
    assert 'rag_request_duration_ms_count{route="/search"} 1' in text


def test_a_slow_request_still_counts_in_the_total():
    m = Metrics()
    m.observe("/answer", 200, 99_999.0)
    text = m.render({})
    assert 'rag_request_duration_ms_bucket{route="/answer",le="+Inf"} 1' in text
    assert f'le="{BUCKETS_MS[-1]}"}} 0' in text


def test_buckets_are_cumulative():
    m = Metrics()
    for ms in (0.5, 3.0, 30.0):
        m.observe("/search", 200, ms)
    text = m.render({})
    counts = {}
    for line in text.splitlines():
        if line.startswith("rag_request_duration_ms_bucket"):
            edge = line.split('le="')[1].split('"')[0]
            counts[edge] = int(line.rsplit(" ", 1)[1])
    assert counts["1"] == 1
    assert counts["5"] == 2
    assert counts["50"] == 3
    assert counts["+Inf"] == 3


def test_statuses_are_counted_apart():
    m = Metrics()
    m.observe("/search", 200, 1.0)
    m.observe("/search", 500, 1.0)
    text = m.render({})
    assert 'rag_requests_total{route="/search",status="200"} 1' in text
    assert 'rag_requests_total{route="/search",status="500"} 1' in text


def test_concurrent_observations_are_not_lost():
    m = Metrics()
    def hammer():
        for _ in range(500):
            m.observe("/search", 200, 1.0)
    threads = [threading.Thread(target=hammer) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert m.requests[("/search", 200)] == 4000


def test_index_size_is_a_gauge_not_a_counter():
    text = Metrics().render({"chunks": 54, "documents": 12})
    assert "rag_index_chunks 54" in text
    assert "rag_index_documents 12" in text


def test_limiter_is_off_when_the_rate_is_zero():
    b = TokenBucket(rps=0.0, burst=1)
    assert all(b.take("1.2.3.4")[0] for _ in range(100))


def test_burst_is_allowed_then_refused():
    b = TokenBucket(rps=1.0, burst=3)
    assert [b.take("a")[0] for _ in range(5)] == [True, True, True, False, False]


def test_retry_after_grows_as_the_bucket_empties():
    b = TokenBucket(rps=2.0, burst=1)
    b.take("a")
    _, wait = b.take("a")
    assert 0.0 < wait <= 0.5


def test_clients_get_their_own_bucket():
    b = TokenBucket(rps=1.0, burst=1)
    assert b.take("a")[0]
    assert not b.take("a")[0]
    assert b.take("b")[0]


def test_the_bucket_refills_with_the_clock(monkeypatch):
    now = [1000.0]
    monkeypatch.setattr("rag.observability.time.monotonic", lambda: now[0])
    b = TokenBucket(rps=10.0, burst=2)
    assert b.take("a")[0] and b.take("a")[0]
    assert not b.take("a")[0]
    now[0] += 0.2
    assert b.take("a")[0]
    assert b.take("a")[0]


def test_idle_clients_are_dropped_instead_of_accumulating(monkeypatch):
    now = [1000.0]
    monkeypatch.setattr("rag.observability.time.monotonic", lambda: now[0])
    b = TokenBucket(rps=1000.0, burst=1)
    monkeypatch.setattr(TokenBucket, "MAX_KEYS", 50)
    for i in range(60):
        b.take(f"client-{i}")
    now[0] += 10.0
    b.take("fresh")
    assert len(b._state) < 60
