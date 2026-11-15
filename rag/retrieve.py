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
from .rerank import Reranker, get_reranker
from .store import Hit, Store


@dataclass
class Retriever:
    store: Store
    embedder: Embedder
    settings: Settings
    bm25: BM25 | None = None
    reranker: Reranker | None = None

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
        if self.reranker is None:
            self.reranker = get_reranker(self.settings)

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
        if self.settings.expansion == "prf":
            q = self._expand(q, mat)
        sims = mat @ q
        k = min(k, len(sims))
        idx = np.argpartition(-sims, k - 1)[:k]
        idx = idx[np.argsort(-sims[idx])]
        ids = self.store.vector_ids
        return [(ids[i], float(sims[i])) for i in idx]

    def _expand(self, q: np.ndarray, mat: np.ndarray) -> np.ndarray:
        """Pseudo-relevance feedback, the Rocchio move, done in the embedding space.

        The golden set's worst category is semantic, and reading its failures says why: someone
        asks what stops "a double tap on the pay button" from charging twice and the handbook
        says "idempotency key". No encoder bridges that on its own, because the two share no
        vocabulary. So retrieve once, assume the top few chunks are roughly on topic, and pull
        the query vector toward their centroid before retrieving for real.

        It costs one extra matrix multiply against an index that is already in memory, and it
        can also drag a query that started off topic further off. Which way it goes is a number,
        not an opinion, and the number is in the README table.
        """
        s = self.settings
        sims = mat @ q
        m = min(s.prf_docs, len(sims))
        if m <= 0:
            return q
        idx = np.argpartition(-sims, m - 1)[:m]
        centroid = mat[idx].mean(axis=0)
        qp = q + s.prf_beta * centroid
        n = float(np.linalg.norm(qp))
        return qp / n if n else q

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
        # The reranker can only fix an ordering it can see, so the pool it gets is wider than
        # the k we return. For the cross encoder that pool is also the per-query cost.
        pool = max(k * 3, k, s.rerank_candidates if s.rerank.startswith("cross") else 0)
        hits = []
        for chunk_id, score, signals in fused[:pool]:
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

        return self.reranker.rerank(query, hits)[:k]


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
