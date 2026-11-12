"""Dense plus BM25, fused with reciprocal rank.

Why both. Dense retrieval is good at "what happens when the acquirer times out" and bad at
"NW-5001", because an identifier carries almost no semantic signal and the encoder has never seen
it. BM25 is the opposite. Running them separately and fusing the ranks gets both, and it needs no
score calibration, which is the part that usually rots.

Reciprocal rank fusion scores a chunk as the sum over retrievers of 1 / (k + rank). The constant
k decides how much agreement is worth. At k=60 agreement dominates: a chunk ranked fourth by both
retrievers scores 2/64 = 0.0313 and beats a chunk ranked first by one and missing from the other,
which scores 1/61 = 0.0164. At k=0 that flips, and the lone top hit wins. The unit test in
`tests/test_fusion.py` pins both ends, because the first version of this docstring claimed the
opposite and nothing in the code disagreed with it.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .bm25 import BM25
from .config import Settings
from .embedding import Embedder
from .store import Hit, Store
from .tokenize import tokenize


@dataclass
class Retriever:
    store: Store
    embedder: Embedder
    settings: Settings
    bm25: BM25 | None = None

    def __post_init__(self) -> None:
        if not (self.settings.use_dense or self.settings.use_sparse):
            # An eval run once landed here and reported 0.0 for every metric without failing.
            # Silence is the worst thing a harness can do, so this is loud now.
            raise ValueError(
                "both retrievers are switched off, so nothing can be retrieved: "
                "set RAG_USE_DENSE=1 or RAG_USE_SPARSE=1"
            )
        if self.bm25 is None:
            self.bm25 = self._build_bm25()

    def _build_bm25(self) -> BM25:
        b = BM25()
        for row in self.store.all_chunks():
            head = row["heading_path"]
            b.add(row["chunk_id"], f"{head} {head} {row['text']}")
        b.finalize()
        return b

    # ---------- individual signals ----------

    def dense(self, query: str, k: int) -> list[tuple[str, float]]:
        mat = self.store.matrix
        if mat is None or not len(mat):
            return []
        q = self.embedder.encode([query])[0]
        sims = mat @ q
        k = min(k, len(sims))
        idx = np.argpartition(-sims, k - 1)[:k]
        idx = idx[np.argsort(-sims[idx])]
        ids = self.store.vector_ids
        return [(ids[i], float(sims[i])) for i in idx]

    def sparse(self, query: str, k: int) -> list[tuple[str, float]]:
        return self.bm25.search(query, k)

    def best_bm25(self, query: str) -> float:
        """Top BM25 score, whether or not sparse retrieval is part of the fusion. The abstain
        decision needs a relevance number and must not change when an ablation flag flips."""
        top = self.bm25.search(query, 1)
        return top[0][1] if top else 0.0

    # ---------- fusion ----------

    def search(self, query: str, k: int | None = None) -> list[Hit]:
        s = self.settings
        k = k or s.final_k
        runs: dict[str, list[tuple[str, float]]] = {}
        if s.use_dense:
            runs["dense"] = self.dense(query, s.dense_k)
        if s.use_sparse:
            runs["sparse"] = self.sparse(query, s.sparse_k)

        fused = rrf(runs, k=s.rrf_k)
        hits = []
        for chunk_id, score, signals in fused[: max(k * 3, k)]:
            row = self.store.chunk(chunk_id)
            if row is None:
                continue
            hits.append(
                Hit(
                    chunk_id=chunk_id,
                    doc_id=row["doc_id"],
                    heading_path=row["heading_path"],
                    text=row["text"],
                    score=score,
                    signals=signals,
                )
            )

        if s.rerank == "lexical":
            hits = lexical_rerank(query, hits)
        return hits[:k]


def rrf(runs: dict[str, list[tuple[str, float]]], k: int = 60):
    """Reciprocal rank fusion. Returns (chunk_id, score, per-signal ranks) sorted by score."""
    acc: dict[str, float] = {}
    where: dict[str, dict] = {}
    for name, run in runs.items():
        for rank, (chunk_id, raw) in enumerate(run, start=1):
            acc[chunk_id] = acc.get(chunk_id, 0.0) + 1.0 / (k + rank)
            where.setdefault(chunk_id, {})[name] = {"rank": rank, "score": round(float(raw), 4)}
    out = [(cid, score, where[cid]) for cid, score in acc.items()]
    out.sort(key=lambda x: (-x[1], x[0]))
    return out


def lexical_rerank(query: str, hits: list[Hit]) -> list[Hit]:
    """A cheap final pass that pushes exact term and identifier matches up.

    A cross encoder would do this better and costs a model load plus roughly 40ms per query. On a
    corpus this size the overlap bonus recovered most of the same ranking, so the heavier model is
    left out until the eval says it is needed.
    """
    q = set(tokenize(query))
    if not q:
        return hits
    rescored = []
    for h in hits:
        toks = set(tokenize(f"{h.heading_path} {h.text}"))
        overlap = len(q & toks) / len(q)
        has_id = any(c.isdigit() for t in (q & toks) for c in t)
        bonus = 0.25 * overlap + (0.15 if has_id else 0.0)
        h.signals = dict(h.signals, rerank={"overlap": round(overlap, 3), "bonus": round(bonus, 3)})
        rescored.append((h.score * (1 + bonus), h))
    rescored.sort(key=lambda x: -x[0])
    for score, h in rescored:
        h.score = score
    return [h for _, h in rescored]
