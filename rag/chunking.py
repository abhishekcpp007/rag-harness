"""Markdown chunker that keeps a chunk inside one section and never cuts a code fence.

Two things here matter more than the window size.

The heading path travels with the text. A paragraph that says "retry twice, then page the
on-call" means nothing on its own. Embedded as "Runbook: Settlement stall > Recovery steps ::
retry twice, then page the on-call" it lands near the questions people actually ask.

Fenced code blocks and table rows are atomic. Splitting a fence in half produces a chunk that
looks like config but parses as prose, and the retriever starts matching on stray backticks.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

HEADING = re.compile(r"^(#{1,6})\s+(.*)$")
FENCE = re.compile(r"^\s*(```|~~~)")


@dataclass
class Chunk:
    doc_id: str
    chunk_id: str
    ordinal: int
    heading_path: str
    text: str
    start_line: int
    end_line: int

    @property
    def embed_text(self) -> str:
        if self.heading_path:
            return f"{self.heading_path} :: {self.text}"
        return self.text


@dataclass
class _Block:
    """A paragraph, a list, a table or a fenced code block. Never split further."""

    lines: list[str] = field(default_factory=list)
    start_line: int = 0
    atomic: bool = False

    @property
    def text(self) -> str:
        return "\n".join(self.lines).strip()

    @property
    def words(self) -> int:
        return len(self.text.split())


def _blocks(lines: list[str]) -> list[tuple[str, _Block]]:
    """Walk the file once, emitting (heading_path, block) pairs."""
    out: list[tuple[str, _Block]] = []
    stack: list[tuple[int, str]] = []
    cur = _Block(start_line=1)
    in_fence = False

    def flush():
        nonlocal cur
        if cur.text:
            out.append((_path(stack), cur))
        cur = _Block()

    for i, raw in enumerate(lines, start=1):
        line = raw.rstrip("\n")

        if FENCE.match(line):
            if not in_fence:
                flush()
                cur = _Block(start_line=i, atomic=True)
            cur.lines.append(line)
            if in_fence:
                flush()
            in_fence = not in_fence
            continue

        if in_fence:
            cur.lines.append(line)
            continue

        m = HEADING.match(line)
        if m:
            flush()
            level, title = len(m.group(1)), m.group(2).strip()
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, title))
            continue

        if not line.strip():
            flush()
            continue

        if not cur.lines:
            cur.start_line = i
        cur.lines.append(line)

    flush()
    return out


def _path(stack: list[tuple[int, str]]) -> str:
    return " > ".join(t for _, t in stack)


def chunk_markdown(doc_id: str, text: str, *, words: int = 180, overlap: int = 40) -> list[Chunk]:
    lines = text.splitlines()
    blocks = _blocks(lines)

    chunks: list[Chunk] = []
    buf: list[_Block] = []
    buf_path = ""
    ordinal = 0

    def emit():
        nonlocal buf, ordinal
        if not buf:
            return
        body = "\n\n".join(b.text for b in buf).strip()
        if not body:
            buf = []
            return
        chunks.append(
            Chunk(
                doc_id=doc_id,
                chunk_id=f"{doc_id}#{ordinal}",
                ordinal=ordinal,
                heading_path=buf_path,
                text=body,
                start_line=buf[0].start_line,
                end_line=buf[-1].start_line + len(buf[-1].lines) - 1,
            )
        )
        ordinal += 1
        buf = []

    for path, block in blocks:
        if path != buf_path:
            emit()
            buf_path = path

        projected = sum(b.words for b in buf) + block.words
        if buf and projected > words:
            tail = _overlap_tail(buf, overlap)
            emit()
            buf = tail
        buf.append(block)

    emit()
    return [c for c in chunks if c.text.strip()]


def _overlap_tail(buf: list[_Block], overlap: int) -> list[_Block]:
    """Carry the last few non-atomic words forward so a sentence split across a boundary
    still has its lead-in. Code fences are never duplicated."""
    if overlap <= 0 or not buf:
        return []
    last = buf[-1]
    if last.atomic:
        return []
    words = last.text.split()
    if len(words) <= overlap:
        return [last]
    tail = _Block(lines=[" ".join(words[-overlap:])], start_line=last.start_line)
    return [tail]
