"""Three embedding backends behind one interface.

`hash` needs no download and no key, and it is deterministic, so CI and the unit tests run the
same everywhere. It is a signed hashing trick over word unigrams and bigrams, which is a real
vector space but a weak one. The eval table in the README reports it separately so nobody mistakes
it for a sentence encoder.

`fastembed` is the one the committed numbers use. `openai` is there for the path where the corpus
is large enough that a hosted model is worth paying for.
"""
from __future__ import annotations

import hashlib
import itertools
from typing import Protocol

import numpy as np

from .config import Settings
from .tokenize import tokenize


class Embedder(Protocol):
    name: str
    dims: int

    def encode(self, texts: list[str]) -> np.ndarray: ...


def _l2(m: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(m, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return m / n


class HashEmbedder:
    """Signed feature hashing over unigrams and bigrams."""

    def __init__(self, dims: int = 512) -> None:
        self.name = "hash"
        self.dims = dims

    def _bucket(self, token: str) -> tuple[int, float]:
        h = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        v = int.from_bytes(h, "big")
        return v % self.dims, 1.0 if (v >> 63) & 1 else -1.0

    def encode(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dims), dtype=np.float32)
        for row, text in enumerate(texts):
            toks = tokenize(text)
            grams = toks + [f"{a}_{b}" for a, b in itertools.pairwise(toks)]
            for t in grams:
                i, sign = self._bucket(t)
                out[row, i] += sign
        # sublinear damping, otherwise one repeated word dominates the vector
        out = np.sign(out) * np.log1p(np.abs(out))
        return _l2(out)


class FastEmbedEmbedder:
    def __init__(self, model: str = "BAAI/bge-small-en-v1.5") -> None:
        from fastembed import TextEmbedding  # imported lazily on purpose

        self._m = TextEmbedding(model_name=model)
        self.name = f"fastembed:{model}"
        self.dims = len(next(iter(self._m.embed(["probe"]))))

    def encode(self, texts: list[str]) -> np.ndarray:
        vecs = np.array(list(self._m.embed(texts)), dtype=np.float32)
        return _l2(vecs)


class OpenAIEmbedder:
    def __init__(self, model: str = "text-embedding-3-small") -> None:
        from openai import OpenAI

        self._c = OpenAI()
        self._model = model
        self.name = f"openai:{model}"
        self.dims = 1536

    def encode(self, texts: list[str]) -> np.ndarray:
        out = []
        for i in range(0, len(texts), 128):
            batch = texts[i : i + 128]
            resp = self._c.embeddings.create(model=self._model, input=batch)
            out.extend(d.embedding for d in resp.data)
        return _l2(np.array(out, dtype=np.float32))


def get_embedder(settings: Settings) -> Embedder:
    kind = settings.embedder
    if kind == "hash":
        return HashEmbedder(settings.hash_dims)
    if kind == "fastembed":
        try:
            return FastEmbedEmbedder(settings.embed_model)
        except ImportError:
            # Falling back rather than crashing, so a fresh clone runs. The quality drop is
            # real and measured, see the ablation table in the README.
            print("fastembed is not installed, falling back to the hash embedder "
                  "(pip install 'rag-harness[embed]' for the real one)")
            return HashEmbedder(settings.hash_dims)
    if kind == "openai":
        return OpenAIEmbedder(settings.embed_model)
    raise ValueError(f"unknown embedder {kind!r}")
