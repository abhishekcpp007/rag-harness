"""Reranker tests. No model download: the cross encoder takes its scores from a stub.

What these pin is the ordering contract, not the quality of any particular model. Quality is the
eval's job and its numbers live in the README.
"""
from __future__ import annotations

import pytest

from rag.config import SETTINGS
from rag.rerank import (
    CrossEncoderReranker,
    LexicalReranker,
    NoReranker,
    RRFReranker,
    get_reranker,
)
from rag.store import Hit


def hit(chunk_id: str, text: str, score: float) -> Hit:
    return Hit(chunk_id=chunk_id, doc_id="d", heading_path="Payments > Errors",
               text=text, score=score, signals={})


def ids(hits):
    return [h.chunk_id for h in hits]


def test_no_reranker_leaves_the_order_alone():
    hits = [hit("a", "one", 0.9), hit("b", "two", 0.1)]
    assert ids(NoReranker().rerank("anything", hits)) == ["a", "b"]


def test_lexical_promotes_the_chunk_that_shares_the_identifier():
    hits = [
        hit("prose", "the acquirer may time out during authorisation", 0.50),
        hit("table", "NW-5001 acquirer timeout, do not retry the charge", 0.49),
    ]
    assert ids(LexicalReranker().rerank("what is NW-5001", hits)) == ["table", "prose"]


def test_lexical_cannot_promote_a_chunk_fusion_scored_near_zero():
    """The bonus is multiplicative, so term overlap alone is not enough to win."""
    hits = [
        hit("strong", "acquirer timeout handling", 0.90),
        hit("weak", "NW-5001 NW-5001 NW-5001", 0.01),
    ]
    assert ids(LexicalReranker().rerank("NW-5001", hits))[0] == "strong"


def test_cross_encoder_reorders_by_its_own_score():
    r = CrossEncoderReranker(score_fn=lambda q, docs: [-2.0, 3.0])
    hits = [hit("a", "one", 0.9), hit("b", "two", 0.1)]
    out = r.rerank("q", hits)
    assert ids(out) == ["b", "a"]
    # squashed into (0, 1) so a later multiplicative pass cannot flip the sign
    assert 0.0 < out[0].score < 1.0
    assert out[1].signals["rerank"]["cross"] == -2.0


def test_cross_encoder_refuses_a_mismatched_score_count():
    r = CrossEncoderReranker(score_fn=lambda q, docs: [1.0])
    with pytest.raises(ValueError, match="would pair the wrong score"):
        r.rerank("q", [hit("a", "one", 0.9), hit("b", "two", 0.1)])


def test_rrf_lets_a_stage_minority_opinion_survive():
    """The chunk each stage ranks second beats the one that is first for one and last for
    the other. That is the whole reason this is rank fusion and not a chained rescore."""
    cross = CrossEncoderReranker(score_fn=lambda q, docs: [9.0, 1.0, 0.0])
    hits = [
        hit("only-cross-likes-it", "semantic prose about settlement delays", 0.10),
        hit("both-like-it", "settlement NW-5001 delay", 0.50),
        hit("only-lexical-likes-it", "NW-5001", 0.49),
    ]
    out = ids(RRFReranker(cross, LexicalReranker(), k=1).rerank("NW-5001 settlement", hits))
    assert out[0] == "both-like-it"


def test_rrf_does_not_let_the_first_stage_scores_leak_into_the_second():
    scored = []

    class Spy:
        name = "spy"

        def rerank(self, query, hits):
            scored.append([h.score for h in hits])
            return hits

    cross = CrossEncoderReranker(score_fn=lambda q, docs: [5.0, 5.0])
    hits = [hit("a", "one", 0.90), hit("b", "two", 0.10)]
    RRFReranker(cross, Spy(), k=10).rerank("q", hits)
    assert scored == [[0.90, 0.10]]


def test_get_reranker_rejects_an_unknown_name():
    with pytest.raises(ValueError, match="unknown reranker"):
        get_reranker(SETTINGS.with_(rerank="magic"))


@pytest.mark.parametrize("kind", ["none", "lexical"])
def test_get_reranker_builds_the_offline_ones(kind):
    assert get_reranker(SETTINGS.with_(rerank=kind)) is not None
