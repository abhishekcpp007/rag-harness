import os
from collections import Counter
from pathlib import Path

import pytest
import yaml

from rag.bm25 import BM25
from rag.store import Store
from rag.tokenize import tokenize

GOLDEN = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) / "eval" / "golden.yaml"
TEXTS: dict[str, str] = {}


def build():
    b = BM25()
    b.add("a", "Error NW-4021 means the settlement batch was rejected by the acquirer.")
    b.add("b", "Settlement batches are submitted nightly at 23:40 UTC to the acquirer.")
    b.add("c", "Refunds settle on the next batch and reach the customer in five days.")
    b.finalize()
    return b


def test_exact_identifier_wins():
    assert build().search("NW-4021", 3)[0][0] == "a"


def test_bare_number_still_finds_the_identifier():
    assert build().search("error 4021", 3)[0][0] == "a"


def test_unknown_terms_return_nothing():
    assert build().search("kubernetes helm chart", 3) == []


def test_explain_names_the_terms_that_scored():
    terms = dict(build().explain("settlement batch", "a"))
    assert terms["settlement"] > 0
    assert terms["batch"] > 0


def brute_force(index, query, k):
    """The scorer this index replaced, written out longhand.

    Every document, every query term, pure Python. It is the reference the inverted index has to
    agree with, which is the only reason a rewrite for speed is allowed to touch a ranking file.
    """
    terms = [t for t in tokenize(query) if index.df.get(t)]
    if not terms:
        return []
    out = []
    for i, doc_id in enumerate(index.ids):
        tf = Counter(tokenize(TEXTS[doc_id]))
        dl = index.len[i] or 1
        s = 0.0
        for term in terms:
            f = tf.get(term, 0)
            if not f:
                continue
            denom = f + index.k1 * (1 - index.b + index.b * dl / (index.avg_len or 1))
            s += index._idf(term) * f * (index.k1 + 1) / denom
        if s > 0:
            out.append((doc_id, s))
    out.sort(key=lambda x: -x[1])
    return out[:k]


@pytest.fixture(scope="module")
def corpus_index(indexed):
    store = Store(indexed.db_path)
    b = BM25()
    for row in store.all_chunks():
        head = row["heading_path"]
        text = f"{head} {head} {row['text']}"
        TEXTS[row["chunk_id"]] = text
        b.add(row["chunk_id"], text)
    b.finalize()
    return b


QUERIES = [q["question"] for q in yaml.safe_load(GOLDEN.read_text())]


@pytest.mark.parametrize("query", QUERIES)
def test_inverted_index_scores_match_the_brute_force_scorer(corpus_index, query):
    fast = corpus_index.search(query, 20)
    slow = brute_force(corpus_index, query, 20)
    assert [c for c, _ in fast] == [c for c, _ in slow]
    for (_, a), (_, b) in zip(fast, slow):
        assert a == pytest.approx(b, rel=1e-12)


def test_scoring_buffer_is_cleared_between_queries(corpus_index):
    """The score accumulator is allocated once and reused, so a query that forgot to reset it
    would inherit the previous query's scores. This runs the same query either side of another
    one and the numbers have to be identical."""
    first = corpus_index.search("settlement batch rejected", 10)
    corpus_index.search("webhook signature timeout retry", 10)
    assert corpus_index.search("settlement batch rejected", 10) == first


def test_a_term_repeated_in_the_query_counts_twice(corpus_index):
    once = corpus_index.search("chargeback", 1)[0][1]
    twice = corpus_index.search("chargeback chargeback", 1)[0][1]
    assert twice == pytest.approx(2 * once, rel=1e-12)


def test_adding_after_finalize_reindexes_instead_of_losing_the_term(corpus_index):
    b = BM25()
    b.add("a", "settlement batch")
    b.finalize()
    b.add("b", "kubernetes helm chart")
    b.finalize()
    assert [c for c, _ in b.search("helm", 5)] == ["b"]
    assert [c for c, _ in b.search("settlement", 5)] == ["a"]
