"""Per-query latency for each retrieval configuration, measured properly.

This used to be a column in the eval table, which was wrong twice over. The eval runs each
question once, so its millisecond figure is a single sample with the answer generator and the
metric bookkeeping folded in, and the ablations that do identical amounts of retrieval work came
out 15ms and 54ms apart on a contended container. A number that noisy next to a ranking number
that is stable to four digits makes the stable one look worse than it is.

So latency lives here. One retriever per configuration, built and warmed before the clock starts,
then every query timed on its own, with the median and the 95th over the repeats. The answer path
is measured separately, because it is a different question from how fast retrieval is.

Run:
    python -m bench.latency                  # all of them, slow: the cross encoders are in here
    python -m bench.latency --skip-cross     # the ones that answer in milliseconds
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.answer import answer_question
from rag.config import SETTINGS
from rag.embedding import get_embedder
from rag.llm import get_llm
from rag.retrieve import Retriever
from rag.store import Store

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GOLDEN = os.path.join(HERE, "eval", "golden.yaml")
WARMUP = 2
# Repeats per configuration. The millisecond ones get five passes because they are free; a
# reranker priced at seconds per query gets two, and says so in the output rather than pretending
# to the same precision.
REPEATS = 5
CROSS_REPEATS = 2

CONFIGS = [
    ("default (lexical rerank, PRF)", {}, REPEATS),
    ("no query expansion", {"expansion": "none"}, REPEATS),
    ("no rerank", {"rerank": "none"}, REPEATS),
    ("plus BM25 fusion", {"use_sparse": True}, REPEATS),
    ("BM25 alone", {"use_dense": False, "use_sparse": True}, REPEATS),
    ("hash embedder", {"embedder": "hash"}, REPEATS),
]
CROSS_CONFIGS = [
    ("cross encoder (ms-marco MiniLM)", {"rerank": "cross"}, CROSS_REPEATS),
    ("cross encoder rank-fused with lexical", {"rerank": "cross+lexical"}, CROSS_REPEATS),
    ("cross encoder (bge-reranker-base)",
     {"rerank": "cross", "rerank_model": "BAAI/bge-reranker-base"}, CROSS_REPEATS),
]


def questions(limit: int) -> list[str]:
    with open(GOLDEN) as f:
        return [q["question"] for q in yaml.safe_load(f)][:limit]


def measure(label: str, overrides: dict, qs: list[str], answer: bool, repeats: int) -> dict:
    s = SETTINGS.with_(**overrides)
    if s.embedder != SETTINGS.embedder:
        s = s.with_(db_path=os.path.join("data", f"eval-{s.embedder}-only.sqlite3"))
    r = Retriever(store=Store(s.db_path), embedder=get_embedder(s), settings=s)
    llm = get_llm(s.llm)
    for q in qs[:WARMUP]:
        r.search(q, s.final_k)

    samples = []
    for _ in range(repeats):
        for q in qs:
            t = time.perf_counter()
            r.search(q, s.final_k)
            samples.append((time.perf_counter() - t) * 1000)
    samples.sort()
    out = {
        "config": label,
        "queries": len(qs),
        "repeats": repeats,
        "search_p50_ms": round(statistics.median(samples), 3),
        "search_p95_ms": round(samples[min(len(samples) - 1, int(0.95 * len(samples)))], 3),
    }
    if answer:
        full = []
        for q in qs:
            t = time.perf_counter()
            answer_question(q, r, llm, s)
            full.append((time.perf_counter() - t) * 1000)
        out["answer_p50_ms"] = round(statistics.median(sorted(full)), 3)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-cross", action="store_true",
                    help="leave out the three cross encoder rows, which take minutes")
    ap.add_argument("--queries", type=int, default=12)
    ap.add_argument("--out", default="bench/results/latency.json")
    args = ap.parse_args()

    qs = questions(args.queries)
    configs = CONFIGS if args.skip_cross else CONFIGS + CROSS_CONFIGS
    rows = []
    for label, overrides, repeats in configs:
        row = measure(label, overrides, qs, answer="cross" not in label, repeats=repeats)
        rows.append(row)
        extra = f"   answer {row['answer_p50_ms']:8.2f}ms" if "answer_p50_ms" in row else ""
        print(f"{label:<40} search p50 {row['search_p50_ms']:9.3f}ms  "
              f"p95 {row['search_p95_ms']:9.3f}ms{extra}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"queries": len(qs), "rows": rows}, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
