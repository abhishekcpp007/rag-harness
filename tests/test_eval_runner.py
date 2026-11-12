"""The harness grading itself.

Both of these come from running the repo inside Docker. The first fix is for a flag that had
never worked, the second for a run that swapped its encoder and still wrote the file under the
old label.
"""
import pytest

from eval.run_eval import assert_embedder, main
from eval.table import check_backend


def test_a_fallback_to_the_hash_backend_is_an_error():
    """get_embedder falls back when fastembed is missing. The service may. The eval may not."""
    with pytest.raises(ValueError, match="asked for the 'fastembed' embedder"):
        assert_embedder("fastembed", "hash")


def test_the_backend_that_ran_satisfies_the_one_that_was_asked_for():
    assert_embedder("fastembed", "fastembed:BAAI/bge-small-en-v1.5")
    assert_embedder("hash", "hash")


def test_the_abstain_threshold_flag_reaches_the_settings(tmp_path, monkeypatch):
    """It used to be --abstain-score, and it set a field by that name. No such field exists, so
    the flag raised TypeError on every use and nothing noticed, because nothing called it."""
    seen = {}

    def fake_run(settings, golden, *, name):
        seen["threshold"] = settings.abstain_confidence
        return {"name": name, "config": {}, "index": {}, "summary": {
            "recall@1": 0, "recall@3": 0, "mrr": 0, "chunk_recall@1": 0, "chunk_recall@3": 0,
            "chunk_mrr": 0, "answer_ok": 0, "citation_support": 0,
            "abstained_on_unanswerable": 0, "abstained_on_answerable": 0, "ms_per_question": 0,
        }, "by_kind": {}, "rows": []}

    monkeypatch.setattr("eval.run_eval.run", fake_run)
    rc = main(["--name", "probe", "--abstain-confidence", "0.71", "--out", str(tmp_path)])
    assert rc == 0
    assert seen["threshold"] == 0.71


def test_the_readme_table_refuses_a_row_written_by_the_wrong_encoder():
    hash_run = {"config": {"embedder_actual": "hash"}, "index": {"embedder": "hash"}}
    with pytest.raises(ValueError, match="labels it 'fastembed'"):
        check_backend("default", hash_run)


def test_the_readme_table_accepts_a_row_that_matches_its_label():
    check_backend("default", {"config": {"embedder_actual": "fastembed:BAAI/bge-small-en-v1.5"}})
    check_backend("hash-only", {"config": {"embedder_actual": "hash"}})
