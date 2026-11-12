"""How sure the retriever is, on a 0 to 1 scale, so the service can refuse to answer.

The obvious candidate was the fused RRF score, and it does not work. RRF scores a chunk purely on
its rank, so the top hit of a query about something the corpus has never heard of scores exactly
the same as the top hit of a query the corpus answers perfectly. Measured on the golden set, the
answerable and unanswerable distributions sat on top of each other: medians 0.0377 and 0.0348,
with the best unanswerable query scoring above the median answerable one.

What does separate them is a blend of the best BM25 score and how much of the question's
vocabulary appears in the retrieved text. A question about Apple Pay matches no rare term in the
corpus, so both halves collapse. See `eval/results/abstain-sweep.json` for the curve.
"""
from __future__ import annotations

from .config import Settings
from .store import Hit
from .tokenize import tokenize


def confidence(
    query: str, hits: list[Hit], settings: Settings, *, best_bm25: float | None = None
) -> float:
    """`best_bm25` comes from the BM25 index directly, never from the hit signals.

    The first version read it off `hit.signals["sparse"]`, which quietly tied the abstain decision
    to whether sparse retrieval was switched on. The dense-only ablation then abstained on 77
    percent of the questions it had actually answered correctly, and the retrieval metrics looked
    better than ever while the service said "I could not find that" to three questions in four.
    The eval caught it. Reading a ranking table would not have.
    """
    if not hits:
        return 0.0
    q = set(tokenize(query))
    if not q:
        return 0.0

    best_bm25 = float(best_bm25 or 0.0)
    best_overlap = 0.0
    for h in hits:
        toks = set(tokenize(f"{h.heading_path} {h.text}"))
        best_overlap = max(best_overlap, len(q & toks) / len(q))

    w = settings.conf_overlap_weight
    cap = settings.conf_bm25_cap or 1.0
    return round(w * best_overlap + (1 - w) * min(best_bm25 / cap, 1.0), 4)
