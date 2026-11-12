"""Load the corpus, chunk it, embed it, write it. Idempotent on a rerun."""
from __future__ import annotations

import hashlib
import os
import time

from .chunking import chunk_markdown
from .config import Settings
from .embedding import get_embedder
from .store import Store


def _title(text: str, fallback: str) -> str:
    for line in text.splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return fallback


def ingest(settings: Settings, *, verbose: bool = False) -> dict:
    t0 = time.time()
    store = Store(settings.db_path)
    store.reset()
    embedder = get_embedder(settings)

    paths = sorted(
        os.path.join(settings.corpus_dir, f)
        for f in os.listdir(settings.corpus_dir)
        if f.endswith((".md", ".markdown"))
    )
    if not paths:
        raise SystemExit(f"no markdown found under {settings.corpus_dir!r}")

    total_chunks = 0
    for path in paths:
        with open(path, encoding="utf-8") as fh:
            raw = fh.read()
        doc_id = os.path.splitext(os.path.basename(path))[0]
        sha = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
        store.add_document(doc_id, path, _title(raw, doc_id), sha)

        chunks = chunk_markdown(
            doc_id, raw, words=settings.chunk_words, overlap=settings.chunk_overlap
        )
        texts = [c.embed_text if settings.carry_heading_path else c.text for c in chunks]
        vecs = embedder.encode(texts)
        store.add_chunks(chunks, vecs)
        total_chunks += len(chunks)
        if verbose:
            print(f"  {doc_id:<30} {len(chunks):>3} chunks")

    store.set_meta(
        embedder=embedder.name,
        dims=embedder.dims,
        chunk_words=settings.chunk_words,
        chunk_overlap=settings.chunk_overlap,
        carry_heading_path=settings.carry_heading_path,
    )
    store.commit()
    store.load_matrix()

    out = {
        "documents": len(paths),
        "chunks": total_chunks,
        "embedder": embedder.name,
        "dims": embedder.dims,
        "seconds": round(time.time() - t0, 2),
    }
    if verbose:
        print(f"indexed {out['documents']} docs, {out['chunks']} chunks in {out['seconds']}s")
    return out
