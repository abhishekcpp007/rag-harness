from rag.tokenize import tokenize


def test_identifier_is_kept_whole_and_split():
    toks = tokenize("What causes NW-4021?")
    assert "nw-4021" in toks
    assert "nw" in toks
    assert "4021" in toks


def test_camel_case_is_split():
    assert "settlement" in tokenize("settlementBatch")
    assert "batch" in tokenize("settlementBatch")


def test_stopwords_go_away():
    assert tokenize("what is the batch") == ["batch"]


def test_stopwords_can_be_kept():
    assert "the" in tokenize("what is the batch", drop_stop=False)
