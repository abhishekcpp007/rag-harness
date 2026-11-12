from rag.bm25 import BM25


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
