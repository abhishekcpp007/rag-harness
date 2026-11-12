"""Answer generators behind one call.

`extractive` is the default and it is not a toy. It picks sentences out of the retrieved chunks by
term overlap and attaches the citation of the chunk each sentence came from. It cannot hallucinate,
because it never writes a word that is not already in the corpus, so it gives the eval a stable
floor to compare a real model against. It also means the test suite runs with no key and no
network.

`anthropic` and `openai` take the same context and the same instruction.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Protocol

from .tokenize import tokenize

SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z`|])")

SYSTEM = """You answer questions about an internal engineering handbook.

Rules you do not break:
- Use only the numbered context below. If the answer is not there, reply exactly: NOT_IN_CORPUS
- Put a citation like [2] at the end of every sentence, naming the context block it came from.
- Two sentences at most unless the question asks for steps.
- No preamble. Start with the answer."""


@dataclass
class Generation:
    text: str
    used: list[int]
    backend: str


class LLM(Protocol):
    name: str

    def generate(self, question: str, blocks: list[str]) -> Generation: ...


class ExtractiveLLM:
    """Picks sentences out of the retrieved blocks and cites the block each one came from.

    It cannot hallucinate, because it never writes a word that is not already in the corpus. That
    makes it a stable floor to measure a real model against, and it keeps the test suite off the
    network.

    The scorer is term overlap with the question plus a density term that prefers a short sentence
    carrying the same matches. I tried replacing it with sentence level BM25, so that a rare token
    like `23:40` would outweigh a common one like `time`, and added positional priors on top. That
    version scored 0.571 on answer quality against 0.600 for this one, so it is not here.
    """

    name = "extractive"

    def generate(self, question: str, blocks: list[str]) -> Generation:
        q = set(tokenize(question))
        scored: list[tuple[float, int, str]] = []
        for i, block in enumerate(blocks, start=1):
            for sent in _sentences(block):
                toks = set(tokenize(sent))
                if not toks:
                    continue
                overlap = len(q & toks) / (len(q) or 1)
                density = len(q & toks) / (len(toks) ** 0.5)
                scored.append((overlap + 0.3 * density, i, sent))
        scored.sort(key=lambda x: -x[0])
        picked, seen = [], set()
        for score, i, sent in scored[:6]:
            key = sent.strip()[:60]
            if key in seen or score <= 0:
                continue
            seen.add(key)
            picked.append((i, sent.strip()))
            if len(picked) == 2:
                break
        if not picked:
            return Generation("NOT_IN_CORPUS", [], self.name)
        text = " ".join(f"{s} [{i}]" for i, s in picked)
        return Generation(text, sorted({i for i, _ in picked}), self.name)


class AnthropicLLM:
    def __init__(self, model: str = "claude-sonnet-5-5") -> None:
        import anthropic

        self._c = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        self._model = model
        self.name = f"anthropic:{model}"

    def generate(self, question: str, blocks: list[str]) -> Generation:
        ctx = "\n\n".join(f"[{i}] {b}" for i, b in enumerate(blocks, start=1))
        msg = self._c.messages.create(
            model=self._model,
            max_tokens=400,
            system=SYSTEM,
            messages=[{"role": "user", "content": f"{ctx}\n\nQuestion: {question}"}],
        )
        text = "".join(p.text for p in msg.content if p.type == "text").strip()
        return Generation(text, _cited(text), self.name)


class OpenAILLM:
    def __init__(self, model: str = "gpt-4.1-mini") -> None:
        from openai import OpenAI

        self._c = OpenAI()
        self._model = model
        self.name = f"openai:{model}"

    def generate(self, question: str, blocks: list[str]) -> Generation:
        ctx = "\n\n".join(f"[{i}] {b}" for i, b in enumerate(blocks, start=1))
        resp = self._c.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": f"{ctx}\n\nQuestion: {question}"},
            ],
            max_tokens=400,
        )
        text = (resp.choices[0].message.content or "").strip()
        return Generation(text, _cited(text), self.name)


def _sentences(block: str) -> list[str]:
    """Unwrap first, then split.

    The corpus is hard wrapped at about 100 columns, so a sentence routinely spans two source
    lines. Splitting per line produced answers like "refused every time it was asked." with the
    subject left behind on the line above. Paragraphs are joined back into one line before the
    sentence splitter sees them.
    """
    body = block.split(" :: ", 1)[-1]
    out = []
    for para in body.split("\n\n"):
        lines = [ln.strip() for ln in para.split("\n") if ln.strip()]
        if not lines or lines[0].startswith("```"):
            continue
        if any(ln.startswith(("|", "-", "*", "1.", "2.", "3.", "4.")) for ln in lines):
            out.extend(lines)  # table rows and list items stand on their own
            continue
        joined = " ".join(lines)
        out.extend(s for s in SENTENCE.split(joined) if s.strip())
    return out


def _cited(text: str) -> list[int]:
    return sorted({int(n) for n in re.findall(r"\[(\d+)\]", text)})


def get_llm(name: str) -> LLM:
    if name == "extractive":
        return ExtractiveLLM()
    if name == "anthropic":
        return AnthropicLLM()
    if name == "openai":
        return OpenAILLM()
    raise ValueError(f"unknown llm {name!r}")
