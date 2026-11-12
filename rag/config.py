"""Every knob lives here so an experiment is one environment variable away."""
from __future__ import annotations

import os
from dataclasses import dataclass, replace


def _int(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


def _float(name: str, default: float) -> float:
    return float(os.getenv(name, str(default)))


@dataclass(frozen=True)
class Settings:
    corpus_dir: str = os.getenv("RAG_CORPUS_DIR", "corpus")
    db_path: str = os.getenv("RAG_DB_PATH", "data/index.sqlite3")

    # chunking
    chunk_words: int = _int("RAG_CHUNK_WORDS", 180)
    chunk_overlap: int = _int("RAG_CHUNK_OVERLAP", 40)
    carry_heading_path: bool = os.getenv("RAG_CARRY_HEADINGS", "1") == "1"

    # embedding
    embedder: str = os.getenv("RAG_EMBEDDER", "fastembed")
    embed_model: str = os.getenv("RAG_EMBED_MODEL", "BAAI/bge-small-en-v1.5")
    hash_dims: int = _int("RAG_HASH_DIMS", 512)

    # retrieval
    dense_k: int = _int("RAG_DENSE_K", 20)
    sparse_k: int = _int("RAG_SPARSE_K", 20)
    rrf_k: int = _int("RAG_RRF_K", 60)
    final_k: int = _int("RAG_FINAL_K", 5)
    use_dense: bool = os.getenv("RAG_USE_DENSE", "1") == "1"
    use_sparse: bool = os.getenv("RAG_USE_SPARSE", "0") == "1"
    rerank: str = os.getenv("RAG_RERANK", "lexical")  # none | lexical

    # answering
    llm: str = os.getenv("RAG_LLM", "extractive")
    abstain_confidence: float = _float("RAG_ABSTAIN_CONFIDENCE", 0.34)
    conf_overlap_weight: float = _float("RAG_CONF_OVERLAP_W", 0.5)
    conf_bm25_cap: float = _float("RAG_CONF_BM25_CAP", 12.0)
    max_context_chunks: int = _int("RAG_MAX_CONTEXT", 5)

    def with_(self, **kw) -> Settings:
        return replace(self, **kw)


SETTINGS = Settings()
