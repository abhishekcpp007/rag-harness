"""SQLite for the text, one float32 matrix in memory for the vectors.

A dedicated vector database earns its place somewhere past a million chunks. This corpus has a few
hundred, and a normalized matrix times a query vector is one BLAS call, so the whole index fits in
a file you can copy around and diff against. The honest limit is stated in the README.
"""
from __future__ import annotations

import json
import os
import sqlite3
from dataclasses import dataclass

import numpy as np

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    doc_id TEXT PRIMARY KEY,
    path   TEXT NOT NULL,
    title  TEXT NOT NULL,
    sha    TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id     TEXT PRIMARY KEY,
    doc_id       TEXT NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
    ordinal      INTEGER NOT NULL,
    heading_path TEXT NOT NULL,
    text         TEXT NOT NULL,
    start_line   INTEGER NOT NULL,
    end_line     INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS chunks_doc ON chunks(doc_id);
CREATE TABLE IF NOT EXISTS vectors (
    chunk_id TEXT PRIMARY KEY REFERENCES chunks(chunk_id) ON DELETE CASCADE,
    vec      BLOB NOT NULL
);
CREATE TABLE IF NOT EXISTS meta (k TEXT PRIMARY KEY, v TEXT NOT NULL);
"""


@dataclass
class Hit:
    chunk_id: str
    doc_id: str
    heading_path: str
    text: str
    score: float
    signals: dict


class Store:
    def __init__(self, path: str) -> None:
        self.path = path
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.row_factory = sqlite3.Row
        self.db.executescript(SCHEMA)
        self._ids: list[str] = []
        self._mat: np.ndarray | None = None

    # ---------- writing ----------

    def reset(self) -> None:
        self.db.executescript(
            "DELETE FROM vectors; DELETE FROM chunks; DELETE FROM documents; DELETE FROM meta;"
        )
        self.db.commit()
        self._ids, self._mat = [], None

    def add_document(self, doc_id: str, path: str, title: str, sha: str) -> None:
        self.db.execute(
            "INSERT OR REPLACE INTO documents(doc_id,path,title,sha) VALUES (?,?,?,?)",
            (doc_id, path, title, sha),
        )

    def add_chunks(self, chunks, vectors: np.ndarray) -> None:
        rows = [
            (c.chunk_id, c.doc_id, c.ordinal, c.heading_path, c.text, c.start_line, c.end_line)
            for c in chunks
        ]
        self.db.executemany(
            "INSERT OR REPLACE INTO chunks VALUES (?,?,?,?,?,?,?)",
            rows,
        )
        self.db.executemany(
            "INSERT OR REPLACE INTO vectors VALUES (?,?)",
            [
                (c.chunk_id, np.asarray(v, dtype=np.float32).tobytes())
                for c, v in zip(chunks, vectors)
            ],
        )

    def set_meta(self, **kw) -> None:
        self.db.executemany(
            "INSERT OR REPLACE INTO meta VALUES (?,?)",
            [(k, json.dumps(v)) for k, v in kw.items()],
        )
        self.db.commit()

    def get_meta(self, key: str, default=None):
        row = self.db.execute("SELECT v FROM meta WHERE k=?", (key,)).fetchone()
        return json.loads(row["v"]) if row else default

    def commit(self) -> None:
        self.db.commit()

    # ---------- reading ----------

    def load_matrix(self) -> None:
        rows = self.db.execute(
            "SELECT v.chunk_id, v.vec FROM vectors v JOIN chunks c USING(chunk_id) "
            "ORDER BY c.doc_id, c.ordinal"
        ).fetchall()
        self._ids = [r["chunk_id"] for r in rows]
        if rows:
            self._mat = np.vstack([np.frombuffer(r["vec"], dtype=np.float32) for r in rows])
        else:
            self._mat = None

    @property
    def matrix(self) -> np.ndarray | None:
        if self._mat is None:
            self.load_matrix()
        return self._mat

    @property
    def vector_ids(self) -> list[str]:
        if self._mat is None:
            self.load_matrix()
        return self._ids

    def all_chunks(self) -> list[sqlite3.Row]:
        return self.db.execute("SELECT * FROM chunks ORDER BY doc_id, ordinal").fetchall()

    def chunk(self, chunk_id: str) -> sqlite3.Row | None:
        return self.db.execute("SELECT * FROM chunks WHERE chunk_id=?", (chunk_id,)).fetchone()

    def counts(self) -> dict:
        d = self.db.execute("SELECT COUNT(*) n FROM documents").fetchone()["n"]
        c = self.db.execute("SELECT COUNT(*) n FROM chunks").fetchone()["n"]
        return {"documents": d, "chunks": c}
