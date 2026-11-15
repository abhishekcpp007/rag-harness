"""Final-pass rerankers, and the measurement that decides which one ships.

Fusion and the dense index both rank a chunk without ever looking at the query and the chunk
together. A cross encoder does look at them together, one pair at a time, which is why it is
slower and why it usually wins. The README used to say a cross encoder "would do this better"
and then not run one, which is exactly the kind of claim this repo exists to stop making.

So both live here behind one interface and the ablation table reports them side by side. The
lexical pass stays the default until the numbers say otherwise.
"""
from __future__ import annotations

import math
from functools import lru_cache
from typing import Protocol

from .store import Hit
from .tokenize import tokenize


class Reranker(Protocol):
    name: str

    def rerank(self, query: str, hits: list[Hit]) -> list[Hit]: ...


class NoReranker:
    name = "none"

    def rerank(self, query: str, hits: list[Hit]) -> list[Hit]:
        return hits


class LexicalReranker:
    """A cheap final pass that pushes exact term and identifier matches up.

    The bonus is multiplicative on the fused score rather than additive, so a chunk nothing
    retrieved well cannot be promoted by term overlap alone.
    """

    name = "lexical"

    def rerank(self, query: str, hits: list[Hit]) -> list[Hit]:
        q = set(tokenize(query))
        if not q:
            return hits
        rescored = []
        for h in hits:
            toks = set(tokenize(f"{h.heading_path} {h.text}"))
            overlap = len(q & toks) / len(q)
            has_id = any(c.isdigit() for t in (q & toks) for c in t)
            bonus = 0.25 * overlap + (0.15 if has_id else 0.0)
            h.signals = dict(
                h.signals, rerank={"overlap": round(overlap, 3), "bonus": round(bonus, 3)}
            )
            rescored.append((h.score * (1 + bonus), h))
        rescored.sort(key=lambda x: -x[0])
        for score, h in rescored:
            h.score = score
        return [h for _, h in rescored]


class CrossEncoderReranker:
    """Scores every (query, chunk) pair with a small cross encoder and reorders by that score.

    The score replaces the fused score instead of scaling it. That is deliberate: once a model
    has read the pair, the retriever's rank is the weaker signal, and blending the two just
    smuggles the retriever's mistakes back in. The fused rank is kept in `signals` so a bad
    reorder is still traceable.

    `score_fn` exists so the unit tests can drive the reordering without downloading 80MB of
    ONNX. Nothing else passes it.
    """

    def __init__(self, model: str = "Xenova/ms-marco-MiniLM-L-6-v2", score_fn=None) -> None:
        self.name = f"cross:{model}"
        self._model = model
        self._score_fn = score_fn

    @staticmethod
    def _squash(x: float) -> float:
        """ms-marco heads emit logits, which run negative. A reranker downstream of this one
        multiplies by a bonus, and multiplying a negative score by 1.15 moves it the wrong way,
        so the score leaves here in (0, 1)."""
        if x >= 0:
            return 1.0 / (1.0 + math.exp(-x))
        e = math.exp(x)
        return e / (1.0 + e)

    def _scores(self, query: str, docs: list[str]) -> list[float]:
        if self._score_fn is not None:
            return list(self._score_fn(query, docs))
        return list(_encoder(self._model).rerank(query, docs))

    def rerank(self, query: str, hits: list[Hit]) -> list[Hit]:
        if not hits:
            return hits
        docs = [f"{h.heading_path}\n{h.text}" for h in hits]
        scores = self._scores(query, docs)
        if len(scores) != len(hits):
            raise ValueError(
                f"the cross encoder returned {len(scores)} scores for {len(hits)} chunks, "
                f"so the reorder would pair the wrong score with the wrong chunk"
            )
        rescored = []
        for h, s in zip(hits, scores):
            p = self._squash(float(s))
            h.signals = dict(h.signals, rerank={"cross": round(float(s), 4), "cross_p": round(p, 4),
                                                "fused_score": round(float(h.score), 6)})
            rescored.append((p, h))
        rescored.sort(key=lambda x: -x[0])
        for score, h in rescored:
            h.score = score
        return [h for _, h in rescored]


class RRFReranker:
    """Fuses the orderings of several rerankers by reciprocal rank.

    The reason this exists is in the ablation table. A cross encoder beats the lexical pass on
    semantic questions and loses to it on identifiers, because an error code carries no meaning
    for a model trained on web passages while the lexical pass just matches the string. Picking
    one of the two means picking which half of the golden set to be bad at.

    Chaining them does not fix it either, and the first version of this class tried. A sigmoid
    over cross-encoder logits saturates, so the gap between the top two probabilities is usually
    wider than a 40 percent lexical bonus can close, and the chained order came out byte for byte
    identical to the cross-encoder order. Fusing ranks instead of scaling scores sidesteps the
    whole calibration problem, which is the same argument `retrieve.py` makes for RRF upstream.

    Each stage scores a fresh copy of the ranking, because the stages mutate `score` and the
    second one must not inherit the first one's numbers.
    """

    def __init__(self, *stages: Reranker, k: int = 10) -> None:
        self.stages = stages
        self.k = k
        self.name = "rrf(" + ",".join(s.name.split(":")[0] for s in stages) + ")"

    def rerank(self, query: str, hits: list[Hit]) -> list[Hit]:
        if not hits:
            return hits
        by_id = {h.chunk_id: h for h in hits}
        acc: dict[str, float] = {cid: 0.0 for cid in by_id}
        ranks: dict[str, dict[str, int]] = {cid: {} for cid in by_id}
        for stage in self.stages:
            keep = {h.chunk_id: h.score for h in hits}
            ordered = stage.rerank(query, list(hits))
            for rank, h in enumerate(ordered, start=1):
                acc[h.chunk_id] += 1.0 / (self.k + rank)
                ranks[h.chunk_id][stage.name.split(":")[0]] = rank
            for h in hits:
                h.score = keep[h.chunk_id]
        out = sorted(by_id.values(), key=lambda h: (-acc[h.chunk_id], h.chunk_id))
        for h in out:
            h.score = acc[h.chunk_id]
            h.signals = dict(h.signals, rerank_fusion=ranks[h.chunk_id])
        return out


@lru_cache(maxsize=4)
def _encoder(model: str):
    """One model per process. Reloading it per query turned a 40ms pass into a 3s one."""
    from fastembed.rerank.cross_encoder import TextCrossEncoder

    return TextCrossEncoder(model_name=model)


def get_reranker(settings) -> Reranker:
    kind = settings.rerank
    if kind == "none":
        return NoReranker()
    if kind == "lexical":
        return LexicalReranker()
    if kind == "cross":
        return CrossEncoderReranker(settings.rerank_model)
    if kind == "cross+lexical":
        return RRFReranker(CrossEncoderReranker(settings.rerank_model), LexicalReranker(),
                           k=settings.rerank_rrf_k)
    raise ValueError(
        f"unknown reranker {kind!r}, expected none, lexical, cross or cross+lexical"
    )
