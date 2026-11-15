"""Where this design stops working, measured instead of guessed.

The README says a flat float32 matrix is the right index for a few hundred chunks and that a
vector database earns its place somewhere past a million. That was a reasonable thing to say and
nobody had checked it, so this file checks it, and the answer is not the one the sentence implied.

Two indexes get swept over corpus size. The dense side is one BLAS call against a normalized
matrix. The sparse side is BM25. The interesting result is which one falls over first, because
the vector index is the part everybody plans to replace.

Run:
    python -m bench.scale                      # the default sweep, writes bench/results/scale.json
    python -m bench.scale --sizes 1000,10000   # a narrower one
    python -m bench.scale --dense-max 500000   # push the dense side on its own

Synthetic text is resampled from the real corpus, so term frequencies and document lengths are
the shape the service actually sees rather than uniform noise.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import statistics
import sys
import time
import tracemalloc
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.bm25 import BM25
from rag.config import SETTINGS
from rag.store import Store
from rag.tokenize import tokenize

DIMS = 384
QUERIES = 25
TOP_K = 10
HNSW_M = 16
HNSW_EF_CONSTRUCTION = 200
HNSW_EF_SWEEP = (32, 64, 128, 256)
SEED = 11

# A graph index is only as good as the structure it can walk, so the synthetic vectors have to
# have the structure the real ones do. Measured off the 54 real bge-small chunks in this repo:
# two chunks of the same document sit at cosine 0.710 on average, two chunks of different
# documents at 0.620. Nothing is at zero. The model puts everything inside a narrow cone, which
# is the opposite of the isotropic noise a benchmark reaches for by default and a good deal
# harder on a nearest-neighbour graph, because every candidate looks about equally close.
#
# So a vector is a global cone direction, plus a per-document direction, plus noise, weighted to
# land on those two numbers. The weights come out of
#   between-document sim = g^2 / (g^2 + c^2 + n^2),  within-document sim = (g^2 + c^2) / (...)
# and the bench reports the similarities it actually achieved, so the fit is checked rather than
# asserted.
MEASURED_BETWEEN_DOC_SIM = 0.620
MEASURED_WITHIN_DOC_SIM = 0.710
VECTORS_PER_CLUSTER = 400


class BruteForceBM25(BM25):
    """The scorer the inverted index replaced, kept here so the speedup is a measurement.

    Every document scored for every query, in the interpreter. `tests/test_bm25.py` already pins
    that the two agree on the ranking; this one only times them.
    """

    def __init__(self, *a, **kw) -> None:
        super().__init__(*a, **kw)
        self.tf: list[Counter[str]] = []

    def add(self, doc_id: str, text: str) -> None:
        super().add(doc_id, text)
        self.tf.append(Counter(tokenize(text)))

    def search(self, query: str, k: int = 20) -> list[tuple[str, float]]:
        if not self.ids:
            return []
        q = [t for t in tokenize(query) if self.df.get(t)]
        if not q:
            return []
        scores = []
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


def corpus_sentences() -> list[str]:
    """Real sentences out of the indexed corpus. Falls back to nothing useful if the index is
    missing, which is worth failing loudly over rather than benchmarking uniform noise."""
    store = Store(SETTINGS.db_path)
    rows = store.all_chunks()
    if not rows:
        raise SystemExit(
            f"no chunks in {SETTINGS.db_path}. Run `make index` first: the synthetic corpus is "
            f"resampled from the real one so the term distribution is not made up."
        )
    out = []
    for row in rows:
        for sentence in row["text"].replace("\n", " ").split(". "):
            s = sentence.strip()
            if len(s.split()) > 4:
                out.append(s)
    return out


def synth_docs(n: int, sentences: list[str], rng: random.Random) -> list[str]:
    docs = []
    for i in range(n):
        picked = rng.choices(sentences, k=rng.randint(4, 9))
        docs.append(f"Section {i // 20} > Topic {i % 20} NW-{4000 + i % 2000} " + ". ".join(picked))
    return docs


def synth_queries(sentences: list[str], rng: random.Random) -> list[str]:
    qs = []
    for _ in range(QUERIES):
        words = rng.choice(sentences).split()
        start = rng.randrange(max(1, len(words) - 6))
        qs.append(" ".join(words[start:start + 6]))
    return qs


def timed(fn, reps: int) -> dict:
    samples = []
    for _ in range(reps):
        t = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t) * 1000)
    samples.sort()
    return {
        "p50_ms": round(statistics.median(samples), 3),
        "p95_ms": round(samples[min(len(samples) - 1, int(0.95 * len(samples)))], 3),
        "mean_ms": round(statistics.fmean(samples), 3),
    }


def measure_sparse(cls, docs: list[str], queries: list[str]) -> dict:
    tracemalloc.start()
    t = time.perf_counter()
    index = cls()
    for i, d in enumerate(docs):
        index.add(f"c{i}", d)
    index.finalize()
    build_s = time.perf_counter() - t
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    per_query = timed(lambda: [index.search(q, TOP_K) for q in queries], 3)
    return {
        "build_s": round(build_s, 2),
        "peak_mb": round(peak / 1e6, 1),
        **{k: round(v / len(queries), 3) for k, v in per_query.items()},
    }


def sparse_row(n: int, sentences: list[str], rng: random.Random, brute: bool) -> dict:
    docs = synth_docs(n, sentences, rng)
    queries = synth_queries(sentences, rng)
    row: dict = {"chunks": n, "inverted": measure_sparse(BM25, docs, queries)}
    if brute:
        row["brute_force"] = measure_sparse(BruteForceBM25, docs, queries)
        row["speedup"] = round(row["brute_force"]["p50_ms"] / row["inverted"]["p50_ms"], 1)
    return row


def synth_vectors(n: int, geometry: str, rng) -> tuple[np.ndarray, dict]:
    if geometry == "isotropic":
        mat = rng.standard_normal((n, DIMS), dtype=np.float32)
        pick = np.arange(n)
    elif geometry == "measured":
        g = math.sqrt(MEASURED_BETWEEN_DOC_SIM)
        c = math.sqrt(MEASURED_WITHIN_DOC_SIM - MEASURED_BETWEEN_DOC_SIM)
        noise_w = math.sqrt(1.0 - MEASURED_WITHIN_DOC_SIM)
        k = max(4, n // VECTORS_PER_CLUSTER)
        axis = rng.standard_normal(DIMS).astype(np.float32)
        axis /= np.linalg.norm(axis)
        centroids = rng.standard_normal((k, DIMS)).astype(np.float32)
        centroids -= np.outer(centroids @ axis, axis)
        centroids /= np.linalg.norm(centroids, axis=1, keepdims=True)
        pick = rng.integers(0, k, size=n)
        noise = rng.standard_normal((n, DIMS)).astype(np.float32) / math.sqrt(DIMS)
        mat = g * axis + c * centroids[pick] + noise_w * noise
    else:
        raise SystemExit(f"unknown geometry {geometry!r}, expected measured or isotropic")
    mat /= np.linalg.norm(mat, axis=1, keepdims=True)
    return mat, achieved_sims(mat, pick, rng)


def achieved_sims(mat: np.ndarray, pick: np.ndarray, rng) -> dict:
    """What the generator actually produced, sampled rather than computed over every pair."""
    n = len(mat)
    a = rng.choice(n, size=min(n, 4000))
    b = rng.choice(n, size=min(n, 4000))
    keep = a != b
    a, b = a[keep], b[keep]
    sims = np.einsum("ij,ij->i", mat[a], mat[b])
    same = pick[a] == pick[b]
    out = {"between_doc_sim": round(float(sims[~same].mean()), 3)}
    if same.any():
        out["within_doc_sim"] = round(float(sims[same].mean()), 3)
    return out


def dense_row(n: int, hnsw: bool, geometry: str) -> dict:
    rng = np.random.default_rng(SEED)
    mat, sims = synth_vectors(n, geometry, rng)
    qs = mat[rng.choice(n, size=QUERIES, replace=False)] + 0.3 * rng.standard_normal(
        (QUERIES, DIMS), dtype=np.float32
    ) / math.sqrt(DIMS)
    qs /= np.linalg.norm(qs, axis=1, keepdims=True)

    def exact(q):
        sims_ = mat @ q
        idx = np.argpartition(-sims_, TOP_K - 1)[:TOP_K]
        return idx[np.argsort(-sims_[idx])]

    row = {
        "chunks": n,
        "geometry": geometry,
        "achieved": sims,
        "matrix_mb": round(mat.nbytes / 1e6, 1),
        "exact": timed(lambda: [exact(q) for q in qs], 5),
    }
    for key in ("p50_ms", "p95_ms", "mean_ms"):
        row["exact"][key] = round(row["exact"][key] / QUERIES, 3)

    if hnsw:
        import hnswlib

        truth = [set(exact(q).tolist()) for q in qs]
        tracemalloc.start()
        t = time.perf_counter()
        index = hnswlib.Index(space="ip", dim=DIMS)
        index.init_index(max_elements=n, ef_construction=HNSW_EF_CONSTRUCTION, M=HNSW_M)
        index.add_items(mat, np.arange(n))
        build_s = time.perf_counter() - t
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        row["hnsw"] = {"build_s": round(build_s, 2), "peak_mb": round(peak / 1e6, 1), "ef": {}}
        for ef in HNSW_EF_SWEEP:
            index.set_ef(ef)
            hits = 0
            for q, want in zip(qs, truth):
                got, _ = index.knn_query(q, k=TOP_K)
                hits += len(want & set(got[0].tolist()))
            timings = timed(lambda: [index.knn_query(q, k=TOP_K) for q in qs], 5)
            row["hnsw"]["ef"][str(ef)] = {
                "recall_at_10": round(hits / (QUERIES * TOP_K), 3),
                "p50_ms": round(timings["p50_ms"] / QUERIES, 3),
                "p95_ms": round(timings["p95_ms"] / QUERIES, 3),
            }
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="1000,10000,50000")
    ap.add_argument("--dense-max", type=int, default=0,
                    help="extra dense-only sizes up to this many chunks")
    ap.add_argument("--no-brute", action="store_true",
                    help="skip the scorer that was replaced, which is the slow part of this sweep")
    ap.add_argument("--no-hnsw", action="store_true")
    ap.add_argument("--geometry", default="measured,isotropic",
                    help="dense vector geometry to sweep: measured, isotropic, or both")
    ap.add_argument("--out", default="bench/results/scale.json")
    args = ap.parse_args()

    sizes = [int(s) for s in args.sizes.split(",")]
    sentences = corpus_sentences()
    rng = random.Random(SEED)
    print(f"{len(sentences)} real sentences resampled into each synthetic corpus\n")

    sparse, dense = [], []
    for n in sizes:
        row = sparse_row(n, sentences, rng, brute=not args.no_brute)
        sparse.append(row)
        inv = row["inverted"]
        line = f"sparse n={n:>7}  inverted {inv['p50_ms']:8.3f}ms  build {inv['build_s']:6.2f}s"
        if "brute_force" in row:
            line += f"   brute {row['brute_force']['p50_ms']:9.3f}ms   {row['speedup']}x"
        print(line)

    dense_sizes = list(sizes)
    if args.dense_max:
        dense_sizes += [s for s in (100_000, 250_000, 500_000, 1_000_000)
                        if s > max(sizes) and s <= args.dense_max]
    for geometry in args.geometry.split(","):
        for n in dense_sizes:
            row = dense_row(n, hnsw=not args.no_hnsw, geometry=geometry)
            dense.append(row)
            line = (f"dense  {geometry:<9} n={n:>7}  exact {row['exact']['p50_ms']:8.3f}ms  "
                    f"{row['matrix_mb']:7.1f}MB  sim {row['achieved']}")
            print(line)
            if "hnsw" in row:
                print(f"       hnsw build {row['hnsw']['build_s']:6.2f}s", end="")
                for ef, m in row["hnsw"]["ef"].items():
                    print(f"   ef={ef} {m['p50_ms']:6.3f}ms r@10 {m['recall_at_10']:.3f}", end="")
                print()

    out = {
        "dims": DIMS,
        "top_k": TOP_K,
        "queries_per_point": QUERIES,
        "hnsw": {"M": HNSW_M, "ef_construction": HNSW_EF_CONSTRUCTION, "ef_sweep": HNSW_EF_SWEEP},
        "sparse": sparse,
        "dense": dense,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
