"""Assemble context, generate, then check the generation before it leaves the process.

Two guards sit here and both of them matter more than the prompt.

Abstain. If retrieval confidence is under the threshold, nothing is sent to the model at all. A
model handed five irrelevant chunks will still write a confident paragraph, and that paragraph is
the thing that gets screenshotted. The confidence signal and its threshold come from a sweep over
the golden set, not from a guess. See `rag/confidence.py` for what had to be thrown away first.

Citation stripping. Every sentence has to carry a [n] that points at a block that was actually in
the context. A sentence that cites nothing, or cites a block number that does not exist, is
dropped before the answer is returned. If dropping leaves nothing, the answer becomes an abstain.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass

from .confidence import confidence
from .config import Settings
from .llm import LLM, _cited
from .retrieve import Retriever
from .store import Hit

ABSTAIN = "I could not find that in the handbook."
SENT_SPLIT = re.compile(r"(?<=\])\s+")


@dataclass
class Answer:
    question: str
    text: str
    abstained: bool
    citations: list[dict]
    hits: list[dict]
    backend: str
    confidence: float
    dropped_sentences: int

    def to_dict(self) -> dict:
        return asdict(self)


def answer_question(
    question: str,
    retriever: Retriever,
    llm: LLM,
    settings: Settings | None = None,
) -> Answer:
    s = settings or retriever.settings
    hits: list[Hit] = retriever.search(question, s.final_k)
    conf = confidence(question, hits, s, best_bm25=retriever.best_bm25(question))

    if not hits or conf < s.abstain_confidence:
        return Answer(question, ABSTAIN, True, [], [_hit(h) for h in hits], "abstain", conf, 0)

    blocks = [f"{h.heading_path} :: {h.text}" for h in hits[: s.max_context_chunks]]
    gen = llm.generate(question, blocks)

    if gen.text.strip() == "NOT_IN_CORPUS":
        return Answer(question, ABSTAIN, True, [], [_hit(h) for h in hits], gen.backend, conf, 0)

    kept, dropped = _enforce_citations(gen.text, len(blocks))
    if not kept:
        return Answer(question, ABSTAIN, True, [], [_hit(h) for h in hits], gen.backend, conf, dropped)

    used = _cited(kept)
    citations = [
        {
            "n": n,
            "chunk_id": hits[n - 1].chunk_id,
            "doc_id": hits[n - 1].doc_id,
            "heading_path": hits[n - 1].heading_path,
        }
        for n in used
        if 1 <= n <= len(hits)
    ]
    return Answer(question, kept, False, citations, [_hit(h) for h in hits], gen.backend, conf, dropped)


def _enforce_citations(text: str, n_blocks: int) -> tuple[str, int]:
    sentences = [s for s in SENT_SPLIT.split(text.strip()) if s.strip()]
    kept, dropped = [], 0
    for sent in sentences:
        nums = _cited(sent)
        if nums and all(1 <= n <= n_blocks for n in nums):
            kept.append(sent.strip())
        else:
            dropped += 1
    return " ".join(kept), dropped


def _hit(h: Hit) -> dict:
    return {
        "chunk_id": h.chunk_id,
        "doc_id": h.doc_id,
        "heading_path": h.heading_path,
        "score": round(h.score, 5),
        "signals": h.signals,
        "preview": h.text[:160].replace("\n", " "),
    }
