from rag.tokenize import _expand, tokenize


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


def test_the_word_cache_does_not_mix_up_the_two_stopword_modes():
    """Expansion is cached per raw word, and `drop_stop` changes what a word expands to. If it
    were left out of the cache key, whichever mode ran first would answer for both."""
    _expand.cache_clear()
    assert tokenize("what is the batch", drop_stop=False)[0] == "what"
    assert tokenize("what is the batch") == ["batch"]
    assert tokenize("what is the batch", drop_stop=False)[0] == "what"


def test_caching_a_word_returns_a_fresh_list_each_call():
    """The cache hands back the same tuple every time, so a caller that mutated the returned
    list would corrupt the next caller's tokens."""
    first = tokenize("settlement batch")
    first.append("injected")
    assert tokenize("settlement batch") == ["settlement", "batch"]
