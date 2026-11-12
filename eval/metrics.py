"""Retrieval and answer metrics. Nothing here calls a model, so it is cheap to unit test."""
from __future__ import annotations

import math


def recall_at_k(retrieved_docs: list[str], gold_docs: list[str], k: int) -> float:
    """Did any gold document show up in the top k. Binary, because every question in the set has
    its answer in one place and partial credit would hide a miss."""
    if not gold_docs:
        return float("nan")
    return 1.0 if set(retrieved_docs[:k]) & set(gold_docs) else 0.0


def mrr(retrieved_docs: list[str], gold_docs: list[str]) -> float:
    if not gold_docs:
        return float("nan")
    for i, d in enumerate(retrieved_docs, start=1):
        if d in gold_docs:
            return 1.0 / i
    return 0.0


def ndcg_at_k(retrieved_docs: list[str], gold_docs: list[str], k: int) -> float:
    if not gold_docs:
        return float("nan")
    gains = [1.0 if d in gold_docs else 0.0 for d in retrieved_docs[:k]]
    dcg = sum(g / math.log2(i + 2) for i, g in enumerate(gains))
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(len(gold_docs), k)))
    return dcg / ideal if ideal else 0.0


def citation_support(citations: list[dict], gold_docs: list[str]) -> float:
    """Fraction of the answer's citations that point at a gold document. An answer can be right
    and still cite the wrong place, and that is the failure a reader cannot see."""
    if not citations:
        return float("nan")
    if not gold_docs:
        return float("nan")
    good = sum(1 for c in citations if c["doc_id"] in gold_docs)
    return good / len(citations)


def contains_all(text: str, needles: list[str]) -> bool:
    low = text.lower()
    return all(n.lower() in low for n in needles)


def mean(xs: list[float]) -> float:
    xs = [x for x in xs if not math.isnan(x)]
    return round(sum(xs) / len(xs), 4) if xs else float("nan")
