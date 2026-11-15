"""Run the golden set against a configuration and write a result file.

Usage:
    python -m eval.run_eval --name hybrid
    python -m eval.run_eval --name hybrid --with-sparse
    python -m eval.run_eval --name sparse-only --no-dense --with-sparse
    python -m eval.run_eval --name flat-chunks --no-headings

Every run reindexes, so a result file is reproducible from its own config block.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone

import yaml

from eval.metrics import citation_support, contains_all, mean, mrr, ndcg_at_k, recall_at_k
from rag.answer import answer_question
from rag.config import SETTINGS
from rag.embedding import get_embedder
from rag.ingest import ingest
from rag.llm import get_llm
from rag.retrieve import Retriever
from rag.store import Store

HERE = os.path.dirname(__file__)


def assert_embedder(requested: str, actual: str) -> None:
    """Fail when the backend that ran is not the backend that was asked for.

    `get_embedder` falls back to the hash backend when fastembed is missing, prints one line and
    carries on. For the service that is a kindness, because a fresh clone still answers. For the
    eval it is poison: the run writes hash numbers into a file whose config block says bge-small,
    and the README table reads that file. Caught the first time this ran inside Docker, where an
    `ENV RAG_EMBEDDER=hash` left over from the test stage quietly turned every ablation into the
    hash ablation. The numbers were right. The label was a lie.
    """
    if actual.split(":")[0] != requested:
        raise ValueError(
            f"asked for the {requested!r} embedder and got {actual!r}. An eval run that silently "
            f"swaps its encoder writes a number nobody can trust: install the extra with "
            f"pip install '.[embed]', or pass --embedder {actual.split(':')[0]} and mean it"
        )


def load_golden(path: str | None = None) -> list[dict]:
    with open(path or os.path.join(HERE, "golden.yaml"), encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def run(settings, golden: list[dict], *, name: str) -> dict:
    index_stats = ingest(settings)
    assert_embedder(settings.embedder, index_stats["embedder"])
    store = Store(settings.db_path)
    retriever = Retriever(store=store, embedder=get_embedder(settings), settings=settings)
    llm = get_llm(settings.llm)

    rows = []
    t0 = time.time()
    for item in golden:
        gold = item.get("gold_docs") or []
        answerable = item["kind"] != "unanswerable"

        gold_chunks = item.get("gold_chunks") or []
        hits = retriever.search(item["question"], max(settings.final_k, 5))
        chunk_ids = [h.chunk_id for h in hits]
        docs, seen = [], set()
        for h in hits:
            if h.doc_id not in seen:
                seen.add(h.doc_id)
                docs.append(h.doc_id)

        a = answer_question(item["question"], retriever, llm, settings)

        rows.append(
            {
                "id": item["id"],
                "kind": item["kind"],
                "question": item["question"],
                "gold_docs": gold,
                "retrieved_docs": docs,
                "confidence": a.confidence,
                "retrieved_chunks": chunk_ids,
                "chunk_recall@1": recall_at_k(chunk_ids, gold_chunks, 1),
                "chunk_recall@3": recall_at_k(chunk_ids, gold_chunks, 3),
                "chunk_recall@5": recall_at_k(chunk_ids, gold_chunks, 5),
                "chunk_mrr": mrr(chunk_ids, gold_chunks),
                "recall@1": recall_at_k(docs, gold, 1),
                "recall@3": recall_at_k(docs, gold, 3),
                "recall@5": recall_at_k(docs, gold, 5),
                "mrr": mrr(docs, gold),
                "ndcg@5": ndcg_at_k(docs, gold, 5),
                "abstained": a.abstained,
                "abstain_correct": a.abstained == (not answerable),
                "citation_support": citation_support(a.citations, gold),
                "answer_ok": (not a.abstained)
                and contains_all(a.text, item.get("must_contain", []))
                if answerable
                else None,
                "answer": a.text,
                "citations": [c["doc_id"] for c in a.citations],
            }
        )
    elapsed = time.time() - t0

    ans = [r for r in rows if r["kind"] != "unanswerable"]
    una = [r for r in rows if r["kind"] == "unanswerable"]

    summary = {
        "answerable_n": len(ans),
        "unanswerable_n": len(una),
        "recall@1": mean([r["recall@1"] for r in ans]),
        "recall@3": mean([r["recall@3"] for r in ans]),
        "recall@5": mean([r["recall@5"] for r in ans]),
        "mrr": mean([r["mrr"] for r in ans]),
        "ndcg@5": mean([r["ndcg@5"] for r in ans]),
        "chunk_recall@1": mean([r["chunk_recall@1"] for r in ans]),
        "chunk_recall@3": mean([r["chunk_recall@3"] for r in ans]),
        "chunk_recall@5": mean([r["chunk_recall@5"] for r in ans]),
        "chunk_mrr": mean([r["chunk_mrr"] for r in ans]),
        "citation_support": mean([r["citation_support"] for r in ans]),
        "answer_ok": mean([1.0 if r["answer_ok"] else 0.0 for r in ans]),
        "abstained_on_answerable": mean([1.0 if r["abstained"] else 0.0 for r in ans]),
        "abstained_on_unanswerable": mean([1.0 if r["abstained"] else 0.0 for r in una]),
        "ms_per_question": round(1000 * elapsed / max(len(rows), 1), 1),
    }

    by_kind = {}
    for kind in sorted({r["kind"] for r in rows}):
        sel = [r for r in rows if r["kind"] == kind]
        if kind == "unanswerable":
            by_kind[kind] = {"n": len(sel), "abstained": mean([1.0 if r["abstained"] else 0.0 for r in sel])}
        else:
            by_kind[kind] = {
                "n": len(sel),
                "recall@3": mean([r["recall@3"] for r in sel]),
                "mrr": mean([r["mrr"] for r in sel]),
                "chunk_recall@3": mean([r["chunk_recall@3"] for r in sel]),
                "chunk_mrr": mean([r["chunk_mrr"] for r in sel]),
                "answer_ok": mean([1.0 if r["answer_ok"] else 0.0 for r in sel]),
            }

    return {
        "name": name,
        "ran_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config": {
            "embedder": settings.embedder,
            "embedder_actual": index_stats["embedder"],
            "embed_model": settings.embed_model if settings.embedder != "hash" else None,
            "use_dense": settings.use_dense,
            "use_sparse": settings.use_sparse,
            "rerank": settings.rerank,
            "expansion": settings.expansion,
            "prf_beta": settings.prf_beta if settings.expansion == "prf" else None,
            "prf_docs": settings.prf_docs if settings.expansion == "prf" else None,
            "rerank_model": settings.rerank_model if settings.rerank.startswith("cross") else None,
            "rerank_candidates": settings.rerank_candidates if settings.rerank.startswith("cross") else None,
            "carry_heading_path": settings.carry_heading_path,
            "chunk_words": settings.chunk_words,
            "chunk_overlap": settings.chunk_overlap,
            "final_k": settings.final_k,
            "abstain_confidence": settings.abstain_confidence,
            "llm": settings.llm,
        },
        "index": index_stats,
        "summary": summary,
        "by_kind": by_kind,
        "rows": rows,
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--no-dense", action="store_true")
    p.add_argument("--no-sparse", action="store_true")
    p.add_argument("--with-sparse", action="store_true", help="BM25 is off by default, see README")
    p.add_argument("--no-headings", action="store_true")
    p.add_argument("--no-rerank", action="store_true")
    p.add_argument("--rerank", default=None, choices=["none", "lexical", "cross", "cross+lexical"],
                   help="which final pass to run, see rag/rerank.py")
    p.add_argument("--expansion", default=None, choices=["none", "prf"],
                   help="pull the query vector toward the first pass centroid before retrieving")
    p.add_argument("--rerank-model", default=None,
                   help="only read when --rerank starts with cross")
    p.add_argument("--embedder", default=None)
    p.add_argument("--chunk-words", type=int, default=None)
    p.add_argument("--abstain-confidence", type=float, default=None,
                   help="override the refusal threshold for this run")
    p.add_argument("--out", default=os.path.join(HERE, "results"))
    args = p.parse_args(argv)

    s = SETTINGS
    if args.no_dense:
        s = s.with_(use_dense=False)
    if args.no_sparse:
        s = s.with_(use_sparse=False)
    if args.with_sparse:
        s = s.with_(use_sparse=True)
    if args.no_headings:
        s = s.with_(carry_heading_path=False)
    if args.no_rerank:
        s = s.with_(rerank="none")
    if args.rerank:
        s = s.with_(rerank=args.rerank)
    if args.rerank_model:
        s = s.with_(rerank_model=args.rerank_model)
    if args.expansion:
        s = s.with_(expansion=args.expansion)
    if args.embedder:
        s = s.with_(embedder=args.embedder)
    if args.chunk_words:
        s = s.with_(chunk_words=args.chunk_words)
    if args.abstain_confidence is not None:
        s = s.with_(abstain_confidence=args.abstain_confidence)
    s = s.with_(db_path=os.path.join("data", f"eval-{args.name}.sqlite3"))

    result = run(s, load_golden(), name=args.name)
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, f"{args.name}.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)

    sm = result["summary"]
    print(f"\n{args.name}")
    print(f"  doc   recall@1 {sm['recall@1']}  recall@3 {sm['recall@3']}  mrr {sm['mrr']}")
    print(f"  chunk recall@1 {sm['chunk_recall@1']}  recall@3 {sm['chunk_recall@3']}  mrr {sm['chunk_mrr']}")
    print(f"  answer_ok {sm['answer_ok']}   citation_support {sm['citation_support']}")
    print(f"  abstained on unanswerable {sm['abstained_on_unanswerable']}"
          f"   on answerable {sm['abstained_on_answerable']}")
    print(f"  {sm['ms_per_question']} ms per question  ->  {path}")
    return 0


def sweep_abstain(settings, golden: list[dict], steps: int = 50) -> dict:
    """Walk the abstain threshold and record what each setting costs.

    Published so the knee is visible. The threshold in `rag/config.py` was read off this curve,
    on this same golden set, which makes the abstain numbers an upper bound rather than a
    prediction about a fresh corpus.
    """
    from rag.confidence import confidence

    assert_embedder(settings.embedder, ingest(settings)["embedder"])
    store = Store(settings.db_path)
    retriever = Retriever(store=store, embedder=get_embedder(settings), settings=settings)

    confs = []
    for item in golden:
        hits = retriever.search(item["question"], settings.final_k)
        confs.append((item["kind"] != "unanswerable",
                      confidence(item["question"], hits, settings,
                                 best_bm25=retriever.best_bm25(item["question"]))))

    n_ans = sum(1 for a, _ in confs if a)
    n_una = sum(1 for a, _ in confs if not a)
    curve = []
    for i in range(steps + 1):
        t = i / steps
        caught = sum(1 for a, c in confs if not a and c < t) / max(n_una, 1)
        lost = sum(1 for a, c in confs if a and c < t) / max(n_ans, 1)
        curve.append({"threshold": round(t, 3), "abstained_on_unanswerable": round(caught, 3),
                      "abstained_on_answerable": round(lost, 3), "margin": round(caught - lost, 3)})
    best = max(curve, key=lambda r: r["margin"])
    # The operating point actually shipped. Refusing a question the handbook can answer is the
    # failure a user sees and complains about, so it gets a budget: at most one in eight. Under
    # that budget, take the threshold that catches the most unanswerable questions.
    budget = 0.125
    allowed = [r for r in curve if r["abstained_on_answerable"] <= budget]
    # Tie-break downward, not upward. The first version of this line read
    # `max(..., key=(catch, threshold))`, which on a tie took the *highest* threshold that still
    # caught the most unanswerable questions. Two settings caught 71 percent here, 0.34 and 0.36,
    # and the rule picked 0.36, which refuses 11 percent of answerable questions where 0.34
    # refuses 9 percent. Same benefit, more harm, chosen on purpose by a sort key nobody reread.
    best_catch = max((r["abstained_on_unanswerable"] for r in allowed), default=0.0)
    chosen = min(
        (r for r in allowed if r["abstained_on_unanswerable"] == best_catch),
        key=lambda r: (r["abstained_on_answerable"], r["threshold"]),
    )
    return {"embedder": settings.embedder, "n_answerable": n_ans, "n_unanswerable": n_una,
            "wrong_abstain_budget": budget, "best_margin": best, "chosen": chosen,
            "curve": curve}


if __name__ == "__main__":
    raise SystemExit(main())
