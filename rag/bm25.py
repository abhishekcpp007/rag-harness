"""Okapi BM25 over an inverted index.

The first version of this file scored every document in the corpus for every query, with a
Python loop over the query terms inside a Python loop over the documents. At 54 chunks that is
1.2ms and nobody notices. `bench/scale.py` is where somebody noticed: 18ms at ten thousand
chunks, 92ms at fifty thousand, growing in a straight line, while the dense side of the same
search was still finishing in 2ms. The vector index everyone worries about was never the
bottleneck. This loop was.

So the index is inverted. A term maps to the documents that contain it and the count in each,
scoring touches only the documents a query term actually appears in, and the arithmetic runs in
numpy instead of the interpreter. The formula is unchanged, and `tests/test_bm25.py` pins the
scores against a brute-force scorer written out longhand, so the speedup cannot quietly become a
ranking change.

The per-document length normalizer does not depend on the query, so it is computed once in
`finalize` rather than once per term per document.

The score accumulator is per thread. Reusing one array across queries avoids allocating a vector
the length of the corpus on every search, and sharing that array between concurrent searches
would have two requests adding into the same slots. `bench/load.py` exists to catch that class of
thing, after it caught the shared SQLite connection next door in `store.py`.
"""
from __future__ import annotations

import math
import threading
from collections import Counter

import numpy as np

from .tokenize import tokenize

K1 = 1.4
B = 0.72


class BM25:
    def __init__(self, k1: float = K1, b: float = B) -> None:
        self.k1, self.b = k1, b
        self.ids: list[str] = []
        self.len: list[int] = []
        self.df: Counter[str] = Counter()
        self.avg_len: float = 0.0
        self._docs: dict[str, list[int]] = {}
        self._freqs: dict[str, list[int]] = {}
        self._post: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self._idfs: dict[str, float] = {}
        self._pos: dict[str, int] = {}
        self._norm: np.ndarray | None = None
        self._local = threading.local()
        self._dirty = True

    def add(self, doc_id: str, text: str) -> None:
        if self._post and not self._docs:
            self._rehydrate()
        toks = tokenize(text)
        tf = Counter(toks)
        i = len(self.ids)
        self.ids.append(doc_id)
        self.len.append(len(toks))
        self.df.update(tf.keys())
        for term, f in tf.items():
            self._docs.setdefault(term, []).append(i)
            self._freqs.setdefault(term, []).append(f)
        self._dirty = True

    def finalize(self) -> None:
        n = len(self.ids)
        self.avg_len = (sum(self.len) / n) if n else 0.0
        self._post = {
            term: (np.asarray(docs, dtype=np.int64),
                   np.asarray(self._freqs[term], dtype=np.float64))
            for term, docs in self._docs.items()
        }
        self._idfs = {term: self._idf(term) for term in self._post}
        lens = np.asarray(self.len, dtype=np.float64)
        lens[lens == 0] = 1.0
        avg = self.avg_len or 1.0
        self._norm = self.k1 * (1 - self.b + self.b * lens / avg)
        self._pos = {doc_id: i for i, doc_id in enumerate(self.ids)}
        self._local = threading.local()
        self._docs, self._freqs = {}, {}
        self._dirty = False

    def _rehydrate(self) -> None:
        """Turn the arrays back into append-able lists, for the rare case of indexing more
        documents into an index that has already been finalized. Keeping both shapes alive the
        whole time doubled the memory the index holds, and `bench/scale.py` reports that number."""
        for term, (idx, freqs) in self._post.items():
            self._docs[term] = idx.tolist()
            self._freqs[term] = [int(f) for f in freqs]

    def _scratch(self) -> np.ndarray:
        buf = getattr(self._local, "scores", None)
        if buf is None or len(buf) != len(self.ids):
            buf = np.zeros(len(self.ids), dtype=np.float64)
            self._local.scores = buf
        return buf

    def _idf(self, term: str) -> float:
        n = len(self.ids)
        df = self.df.get(term, 0)
        if df == 0:
            return 0.0
        return math.log(1.0 + (n - df + 0.5) / (df + 0.5))

    def search(self, query: str, k: int = 20) -> list[tuple[str, float]]:
        if not self.ids:
            return []
        if self._dirty:
            self.finalize()
        terms = [t for t in tokenize(query) if t in self._post]
        if not terms:
            return []
        scores, norm = self._scratch(), self._norm
        touched = []
        for term in terms:
            idx, f = self._post[term]
            scores[idx] += self._idfs[term] * f * (self.k1 + 1) / (f + norm[idx])
            touched.append(idx)
        cand = np.unique(np.concatenate(touched))
        vals = scores[cand]
        scores[cand] = 0.0
        keep = vals > 0
        cand, vals = cand[keep], vals[keep]
        if not len(cand):
            return []
        k = min(k, len(cand))
        top = np.argpartition(-vals, k - 1)[:k]
        top = top[np.lexsort((cand[top], -vals[top]))]
        return [(self.ids[int(cand[i])], float(vals[i])) for i in top]

    def explain(self, query: str, doc_id: str) -> list[tuple[str, float]]:
        """Per-term contribution for one document. Used when a result looks wrong."""
        if self._dirty:
            self.finalize()
        if doc_id not in self._pos:
            raise ValueError(f"no document {doc_id!r} in this index")
        i = self._pos[doc_id]
        out = []
        for term in tokenize(query):
            post = self._post.get(term)
            if post is None:
                continue
            idx, freqs = post
            at = int(np.searchsorted(idx, i))
            if at >= len(idx) or idx[at] != i:
                continue
            f = float(freqs[at])
            out.append((term, float(self._idfs[term] * f * (self.k1 + 1) / (f + self._norm[i]))))
        out.sort(key=lambda x: -x[1])
        return out
