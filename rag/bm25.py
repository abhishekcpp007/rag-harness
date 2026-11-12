"""Okapi BM25. Small enough to read, which is the point: when a ranking looks wrong you can
print the term scores and see why."""
from __future__ import annotations

import math
from collections import Counter

from .tokenize import tokenize

K1 = 1.4
B = 0.72


class BM25:
    def __init__(self, k1: float = K1, b: float = B) -> None:
        self.k1, self.b = k1, b
        self.ids: list[str] = []
        self.tf: list[Counter[str]] = []
        self.len: list[int] = []
        self.df: Counter[str] = Counter()
        self.avg_len: float = 0.0

    def add(self, doc_id: str, text: str) -> None:
        toks = tokenize(text)
        tf = Counter(toks)
        self.ids.append(doc_id)
        self.tf.append(tf)
        self.len.append(len(toks))
        self.df.update(tf.keys())

    def finalize(self) -> None:
        self.avg_len = (sum(self.len) / len(self.len)) if self.len else 0.0

    def _idf(self, term: str) -> float:
        n = len(self.ids)
        df = self.df.get(term, 0)
        if df == 0:
            return 0.0
        return math.log(1.0 + (n - df + 0.5) / (df + 0.5))

    def search(self, query: str, k: int = 20) -> list[tuple[str, float]]:
        if not self.ids:
            return []
        q = [t for t in tokenize(query) if self.df.get(t)]
        if not q:
            return []
        scores: list[tuple[str, float]] = []
        for i, tf in enumerate(self.tf):
            s = 0.0
            dl = self.len[i] or 1
            for term in q:
                f = tf.get(term, 0)
                if not f:
                    continue
                denom = f + self.k1 * (1 - self.b + self.b * dl / (self.avg_len or 1))
                s += self._idf(term) * f * (self.k1 + 1) / denom
            if s > 0:
                scores.append((self.ids[i], s))
        scores.sort(key=lambda x: -x[1])
        return scores[:k]

    def explain(self, query: str, doc_id: str) -> list[tuple[str, float]]:
        """Per-term contribution for one document. Used when a result looks wrong."""
        i = self.ids.index(doc_id)
        tf, dl = self.tf[i], self.len[i] or 1
        out = []
        for term in tokenize(query):
            f = tf.get(term, 0)
            if not f:
                continue
            denom = f + self.k1 * (1 - self.b + self.b * dl / (self.avg_len or 1))
            out.append((term, self._idf(term) * f * (self.k1 + 1) / denom))
        out.sort(key=lambda x: -x[1])
        return out
