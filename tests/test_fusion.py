from rag.retrieve import rrf

RUNS = {
    "dense": [("x", 0.9), ("p", 0.8), ("q", 0.7), ("r", 0.6)],
    "sparse": [("p", 9.0), ("q", 8.0), ("r", 7.0), ("y", 6.0)],
}


def test_at_k60_agreement_beats_one_confident_signal():
    """Two mid ranks outscore a lone first place. 2/64 is more than 1/61."""
    ranked = [cid for cid, _, _ in rrf(RUNS, k=60)]
    assert ranked[0] == "p"
    assert ranked.index("r") < ranked.index("x")


def test_at_k0_the_lone_first_place_climbs_back():
    ranked = [cid for cid, _, _ in rrf(RUNS, k=0)]
    assert ranked.index("x") < ranked.index("q")


def test_agreement_still_counts():
    runs = {"dense": [("a", 1.0), ("b", 0.9)], "sparse": [("a", 5.0), ("c", 4.0)]}
    out = {cid: score for cid, score, _ in rrf(runs, k=60)}
    assert out["a"] > out["b"] > 0


def test_signals_are_carried_for_debugging():
    runs = {"dense": [("a", 1.0)], "sparse": [("a", 5.0)]}
    _, _, signals = rrf(runs, k=60)[0]
    assert signals["dense"]["rank"] == 1
    assert signals["sparse"]["score"] == 5.0


def test_empty_runs_are_fine():
    assert rrf({}, k=60) == []
