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
    rerank: str = os.getenv("RAG_RERANK", "lexical")  # none | lexical | cross | cross+lexical
    rerank_model: str = os.getenv("RAG_RERANK_MODEL", "Xenova/ms-marco-MiniLM-L-6-v2")
    # How many fused candidates the reranker gets to look at. The lexical pass is free so it
    # sees whatever fusion returned; the cross encoder is priced per pair, so it gets a budget.
    rerank_candidates: int = _int("RAG_RERANK_CANDIDATES", 20)
    # RRF constant for cross+lexical. Smaller than the retrieval-side 60 on purpose: with only
    # two rankers over 20 candidates, k=60 flattens them into near agreement with everything.
    rerank_rrf_k: int = _int("RAG_RERANK_RRF_K", 10)

    # query expansion: none | prf (pseudo-relevance feedback, see rag/retrieve.py)
    expansion: str = os.getenv("RAG_EXPANSION", "prf")
    prf_docs: int = _int("RAG_PRF_DOCS", 3)
    prf_beta: float = _float("RAG_PRF_BETA", 0.4)

    # service
    log_format: str = os.getenv("RAG_LOG_FORMAT", "json")  # json | text
    # Off by default. A rate limiter that is on by default means bench/load.py measures the
    # limiter rather than the service, and a single tenant service behind a gateway has one
    # already. Set the rps to switch it on.
    rate_limit_rps: float = _float("RAG_RATE_LIMIT_RPS", 0.0)
    rate_limit_burst: int = _int("RAG_RATE_LIMIT_BURST", 20)

    # answering
    llm: str = os.getenv("RAG_LLM", "extractive")
    abstain_confidence: float = _float("RAG_ABSTAIN_CONFIDENCE", 0.34)
    conf_overlap_weight: float = _float("RAG_CONF_OVERLAP_W", 0.5)
    conf_bm25_cap: float = _float("RAG_CONF_BM25_CAP", 12.0)
    max_context_chunks: int = _int("RAG_MAX_CONTEXT", 5)

    def with_(self, **kw) -> Settings:
        return replace(self, **kw)


SETTINGS = Settings()
