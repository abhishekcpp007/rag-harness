"""One tokenizer, shared by the sparse index and the reranker.

The only interesting rule is what happens to an identifier. `NW-4021` becomes three tokens:
`nw-4021`, `nw` and `4021`. Somebody who pastes the whole code still hits the exact token, and
somebody who types "error 4021" hits the number. Without that split, dense retrieval has to carry
exact-match traffic on its own, and it is bad at it.
"""
from __future__ import annotations

import re

WORD = re.compile(r"[A-Za-z][A-Za-z0-9_\-]*|\d[\d_\-\.]*")
SPLIT = re.compile(r"[-_\.]")
CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")

STOP = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "can", "do", "does", "for", "from",
    "has", "have", "how", "i", "if", "in", "is", "it", "its", "of", "on", "or", "our", "that",
    "the", "then", "there", "they", "this", "to", "was", "we", "what", "when", "where", "which",
    "who", "why", "will", "with", "you", "your",
}


SIBILANT = ("s", "x", "z", "ch", "sh")


def _stem(w: str) -> str:
    """The smallest plural rule that is safe in both directions.

    Query and document go through the same function, so the only thing that matters is that
    `runs` and `run` land on the same string, and that `batches` and `batch` do too. Anything
    cleverer than this started turning `codes` into `cod`.
    """
    if len(w) > 4 and w.endswith("es") and w[:-2].endswith(SIBILANT):
        return w[:-2]
    if len(w) > 3 and w.endswith("s") and not w.endswith(("ss", "us", "is")):
        return w[:-1]
    return w


def tokenize(text: str, *, drop_stop: bool = True) -> list[str]:
    out: list[str] = []
    for raw in WORD.findall(text):
        low = raw.lower()
        out.append(low)
        parts = [p for p in SPLIT.split(low) if p]
        if len(parts) > 1:
            out.extend(parts)
        if CAMEL.search(raw):
            out.extend(p.lower() for p in CAMEL.split(raw) if p)
    if drop_stop:
        out = [t for t in out if t not in STOP]
    return [_stem(t) for t in out]
