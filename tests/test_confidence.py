"""The regression test for a bug the eval found and a ranking table would have hidden.

`confidence` once read its BM25 term off `hit.signals["sparse"]`, which only exists when sparse
retrieval is switched on. Turning sparse off therefore drove confidence to roughly half its real
value, and the service abstained on 77 percent of the questions it was retrieving correctly. The
retrieval metrics looked better than ever at the same time.
"""
from rag.confidence import confidence
from rag.embedding import get_embedder
from rag.retrieve import Retriever
from rag.store import Store

ANSWERABLE = "What time does the settlement sweep run in India?"
UNANSWERABLE = "How do I issue a virtual card to an employee?"


def _conf(settings, question):
    r = Retriever(store=Store(settings.db_path), embedder=get_embedder(settings), settings=settings)
    hits = r.search(question, settings.final_k)
    return confidence(question, hits, settings, best_bm25=r.best_bm25(question))


def test_an_answerable_question_scores_above_the_threshold(indexed):
    assert _conf(indexed, ANSWERABLE) >= indexed.abstain_confidence


def test_an_unanswerable_question_scores_below_the_threshold(indexed):
    assert _conf(indexed, UNANSWERABLE) < indexed.abstain_confidence


def test_confidence_does_not_move_when_sparse_retrieval_is_switched_off(indexed):
    with_sparse = _conf(indexed.with_(use_sparse=True), ANSWERABLE)
    without = _conf(indexed.with_(use_sparse=False), ANSWERABLE)
    assert abs(with_sparse - without) < 0.2, (with_sparse, without)
    assert without >= indexed.abstain_confidence


def test_no_hits_means_no_confidence(indexed):
    assert confidence("anything", [], indexed, best_bm25=0.0) == 0.0


def test_switching_both_retrievers_off_fails_loudly(indexed):
    """It used to return zeros and look like a result."""
    import pytest

    with pytest.raises(ValueError, match="both retrievers"):
        Retriever(
            store=Store(indexed.db_path),
            embedder=get_embedder(indexed),
            settings=indexed.with_(use_dense=False, use_sparse=False),
        )
