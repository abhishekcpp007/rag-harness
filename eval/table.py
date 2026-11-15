"""Render the committed result files as the markdown tables that go in the README."""
from __future__ import annotations

import json
import os

HERE = os.path.dirname(__file__)
ORDER = ["default", "no-expansion", "no-rerank", "cross-rerank", "cross-lex", "no-headings",
         "hybrid", "sparse-only", "hash-only", "hash-hybrid"]
LABEL = {
    "default": "default (bge-small, dense, PRF expansion, lexical rerank)",
    "no-expansion": "default minus the query expansion",
    "cross-rerank": "default with a cross encoder instead of the lexical rerank",
    "cross-lex": "default with the cross encoder rank-fused into the lexical one",
    "no-rerank": "default minus the lexical rerank",
    "no-headings": "default minus the heading path",
    "hybrid": "default plus BM25 fusion",
    "sparse-only": "BM25 alone",
    "hash-only": "hash embedder, dense only",
    "hash-hybrid": "hash embedder plus BM25 fusion",
}
KINDS = ["identifier", "semantic", "procedural", "factual"]
# What each row claims its encoder was. A result file that disagrees does not get rendered.
EXPECT = {
    "default": "fastembed", "no-expansion": "fastembed", "cross-rerank": "fastembed",
    "cross-lex": "fastembed", "no-rerank": "fastembed", "no-headings": "fastembed",
    "hybrid": "fastembed", "sparse-only": "fastembed",
    "hash-only": "hash", "hash-hybrid": "hash",
}


def check_backend(name: str, result: dict) -> None:
    """The label says bge-small, so the file had better have been written by bge-small.

    These tables get pasted into the README and then into a conversation with someone deciding
    whether to hire me. A stale file written by the fallback encoder would sit in that table
    looking exactly like the real thing, and the only clue would be a number that moved.
    """
    actual = (result.get("config", {}).get("embedder_actual")
              or result.get("index", {}).get("embedder", "unknown"))
    want = EXPECT.get(name)
    if want and actual.split(":")[0] != want:
        raise ValueError(
            f"{name}.json says it ran on {actual!r}, but this table labels it {want!r}. "
            f"Rerun it with make eval rather than publishing the wrong encoder."
        )


def load(name):
    with open(os.path.join(HERE, "results", f"{name}.json"), encoding="utf-8") as fh:
        return json.load(fh)


def main() -> None:
    rows = [(n, load(n)) for n in ORDER if os.path.exists(os.path.join(HERE, "results", f"{n}.json"))]
    for n, d in rows:
        check_backend(n, d)

    print("| run | chunk recall@1 | chunk recall@3 | chunk MRR | answer ok | citations on target | ms |")
    print("|---|---|---|---|---|---|---|")
    for n, d in rows:
        s = d["summary"]
        print(f"| {LABEL.get(n, n)} | {s['chunk_recall@1']:.2f} | {s['chunk_recall@3']:.2f} "
              f"| {s['chunk_mrr']:.3f} | {s['answer_ok']:.2f} | {s['citation_support']:.2f} "
              f"| {s['ms_per_question']:.0f} |")

    print()
    print("| run | " + " | ".join(KINDS) + " |")
    print("|---" * (len(KINDS) + 1) + "|")
    for n, d in rows:
        cells = " | ".join(f"{d['by_kind'][k]['chunk_mrr']:.2f}" for k in KINDS)
        print(f"| {LABEL.get(n, n)} | {cells} |")

    print()
    print("| run | refused, cannot be answered | refused, could have been answered |")
    print("|---|---|---|")
    for n, d in rows:
        s = d["summary"]
        print(f"| {LABEL.get(n, n)} | {s['abstained_on_unanswerable']:.2f} "
              f"| {s['abstained_on_answerable']:.2f} |")


if __name__ == "__main__":
    main()
