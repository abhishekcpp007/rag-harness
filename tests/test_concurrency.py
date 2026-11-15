"""The failures that only exist with more than one thread.

FastAPI runs a plain `def` route in a worker threadpool, so everything the service touches on a
request is touched concurrently. Both of the bugs pinned here shipped and passed a green suite,
and `bench/load.py` is what found them. These are the cheap version of that load test, so the
next person does not need a running service to catch a regression.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from rag.embedding import get_embedder
from rag.retrieve import Retriever
from rag.store import Store

THREADS = 8
ROUNDS = 40


def test_the_store_is_usable_from_another_thread(indexed):
    """One `sqlite3` connection shared across threads raises ProgrammingError on the second
    thread that touches it. That is what 176 of 200 concurrent requests died on."""
    store = Store(indexed.db_path)
    assert store.counts()["chunks"] > 20
    with ThreadPoolExecutor(max_workers=THREADS) as pool:
        counts = list(pool.map(lambda _: store.counts()["chunks"], range(THREADS * 2)))
    assert len(set(counts)) == 1


def test_the_matrix_is_shared_rather_than_loaded_per_thread(indexed):
    store = Store(indexed.db_path)
    with ThreadPoolExecutor(max_workers=THREADS) as pool:
        mats = list(pool.map(lambda _: store.matrix, range(THREADS)))
    first = mats[0]
    assert all(m is first for m in mats)


def test_concurrent_searches_return_what_a_single_thread_returns(indexed):
    """The BM25 score accumulator is reused between queries instead of reallocated. Sharing one
    buffer across threads has two searches adding into the same slots, and the damage shows up
    as a ranking that is subtly wrong rather than as an exception."""
    r = Retriever(store=Store(indexed.db_path), embedder=get_embedder(indexed),
                  settings=indexed.with_(use_sparse=True))
    questions = [
        "what does NW-4021 mean",
        "settlement batch rejected by the acquirer",
        "how do I rotate a webhook signing secret",
        "which service owns the balance",
        "chargeback deadline",
        "idempotency key validity",
        "NW-5001 retry",
        "payout limit for IMPS",
    ]
    expected = {q: [h.chunk_id for h in r.search(q, 5)] for q in questions}

    def hammer(i):
        q = questions[i % len(questions)]
        return q, [h.chunk_id for h in r.search(q, 5)]

    with ThreadPoolExecutor(max_workers=THREADS) as pool:
        results = list(pool.map(hammer, range(THREADS * ROUNDS)))

    wrong = [(q, got) for q, got in results if got != expected[q]]
    assert not wrong, wrong[:3]


def test_concurrent_bm25_scores_are_stable(indexed):
    r = Retriever(store=Store(indexed.db_path), embedder=get_embedder(indexed), settings=indexed)
    once = r.bm25.search("settlement batch acquirer", 10)

    with ThreadPoolExecutor(max_workers=THREADS) as pool:
        runs = list(pool.map(lambda _: r.bm25.search("settlement batch acquirer", 10),
                             range(THREADS * ROUNDS)))
    assert all(run == once for run in runs)
