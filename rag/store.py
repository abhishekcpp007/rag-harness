"""SQLite for the text, one float32 matrix in memory for the vectors.

A dedicated vector database earns its place somewhere past a million chunks, and `bench/scale.py`
now says where: a flat matrix answers a query over a quarter of a million chunks in 12ms, while
an approximate graph index over the same vectors gives up part of its top 10 to beat that. This
corpus has 54 chunks. The matrix stays, and the file is still one you can copy around and diff.

The connection is per thread. It did not used to be, and `bench/load.py` is how that was found:
FastAPI runs a plain `def` route in a worker threadpool, every request shared one `sqlite3`
connection, and 176 of 200 concurrent requests died on `SQLite objects created in a thread can
only be used in that same thread`. A single-threaded test suite cannot see that, and neither can
one request at a time by hand.

The vectors are different. They are written once at ingest and only read afterwards, so every
thread shares the one matrix rather than each loading a copy of it, with a lock around the first
load so two threads racing into a cold index do not both pay for it.
"""
from __future__ import annotations

import json
import os
import sqlite3
import threading
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
        self._local = threading.local()
        self._load_lock = threading.Lock()
        self._ids: list[str] = []
        self._mat: np.ndarray | None = None
        self._loaded = False
        self.db.executescript(SCHEMA)
        self.db.commit()

    @property
    def db(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self.path)
            conn.row_factory = sqlite3.Row
            self._local.conn = conn
        return conn

    # ---------- writing ----------

    def reset(self) -> None:
        self.db.executescript(
            "DELETE FROM vectors; DELETE FROM chunks; DELETE FROM documents; DELETE FROM meta;"
        )
        self.db.commit()
        self._ids, self._mat, self._loaded = [], None, False

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
        self._loaded = True

    def _load_once(self) -> None:
        if self._loaded:
            return
        with self._load_lock:
            if not self._loaded:
                self.load_matrix()

    @property
    def matrix(self) -> np.ndarray | None:
        self._load_once()
        return self._mat

    @property
    def vector_ids(self) -> list[str]:
        self._load_once()
        return self._ids

    def all_chunks(self) -> list[sqlite3.Row]:
        return self.db.execute("SELECT * FROM chunks ORDER BY doc_id, ordinal").fetchall()

    def chunk(self, chunk_id: str) -> sqlite3.Row | None:
        return self.db.execute("SELECT * FROM chunks WHERE chunk_id=?", (chunk_id,)).fetchone()

    def chunks_by_id(self, chunk_ids) -> dict[str, sqlite3.Row]:
        """One statement for the whole candidate pool rather than one per hit.

        Fusion hands back twenty ids and the old code looked each one up on its own, so a search
        paid twenty round trips to SQLite to build five results. It is the same rows either way.
        """
        ids = list(chunk_ids)
        if not ids:
            return {}
        marks = ",".join("?" * len(ids))
        rows = self.db.execute(
            f"SELECT * FROM chunks WHERE chunk_id IN ({marks})", ids
        ).fetchall()
        return {r["chunk_id"]: r for r in rows}

    def counts(self) -> dict:
        d = self.db.execute("SELECT COUNT(*) n FROM documents").fetchone()["n"]
        c = self.db.execute("SELECT COUNT(*) n FROM chunks").fetchone()["n"]
        return {"documents": d, "chunks": c}
