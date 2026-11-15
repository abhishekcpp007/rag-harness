"""Pseudo-relevance feedback. Runs on the hash embedder, so no download and the same numbers
everywhere.

These pin the mechanics: that the feedback vector moves the query, that beta scales how far, and
that switching it off leaves the first pass untouched. Whether the move helps is the eval's
question, not a unit test's.
"""
from __future__ import annotations

import numpy as np

from rag.embedding import get_embedder
from rag.retrieve import Retriever
from rag.store import Store


def _retriever(settings):
    return Retriever(store=Store(settings.db_path), embedder=get_embedder(settings),
                     settings=settings)


def test_expansion_off_returns_the_query_vector_unchanged(indexed):
    s = indexed.with_(expansion="none")
    r = _retriever(s)
    q = r.embedder.encode(["settlement delay"])[0]
    assert r.dense("settlement delay", 5) == _retriever(s).dense("settlement delay", 5)
    # with expansion off nothing calls _expand, so the vector the search uses is still q
    assert np.allclose(q, r.embedder.encode(["settlement delay"])[0])


def test_expansion_moves_the_query_toward_the_first_pass(indexed):
    s = indexed.with_(expansion="prf", prf_docs=3, prf_beta=0.4)
    r = _retriever(s)
    mat = r.store.matrix
    q = r.embedder.encode(["settlement delay"])[0]
    qp = r._expand(q, mat)
    assert not np.allclose(q, qp)
    assert abs(float(np.linalg.norm(qp)) - 1.0) < 1e-5
    # it moves toward the neighbourhood, never away from it
    before = float(np.max(mat @ q))
    after = float(np.max(mat @ qp))
    assert after >= before - 1e-6


def test_beta_zero_is_a_no_op(indexed):
    s = indexed.with_(expansion="prf", prf_docs=3, prf_beta=0.0)
    r = _retriever(s)
    q = r.embedder.encode(["webhook signature"])[0]
    assert np.allclose(q, r._expand(q, r.store.matrix), atol=1e-6)


def test_bigger_beta_moves_the_query_further(indexed):
    small = _retriever(indexed.with_(expansion="prf", prf_docs=3, prf_beta=0.2))
    large = _retriever(indexed.with_(expansion="prf", prf_docs=3, prf_beta=0.9))
    q = small.embedder.encode(["chargeback deadline"])[0]
    mat = small.store.matrix
    d_small = float(np.linalg.norm(small._expand(q, mat) - q))
    d_large = float(np.linalg.norm(large._expand(q, mat) - q))
    assert d_large > d_small


def test_expansion_still_returns_k_hits(indexed):
    r = _retriever(indexed.with_(expansion="prf"))
    assert len(r.dense("how do payouts work", 5)) == 5
