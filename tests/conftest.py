import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# The whole suite runs on the hash embedder. No download, no network, same numbers on every
# machine. The fastembed path is covered by the eval runs, not by unit tests.
os.environ["RAG_EMBEDDER"] = "hash"

import pytest

from rag.config import SETTINGS
from rag.embedding import get_embedder
from rag.ingest import ingest
from rag.retrieve import Retriever
from rag.store import Store

CORPUS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "corpus")


@pytest.fixture(scope="session")
def settings(tmp_path_factory):
    db = tmp_path_factory.mktemp("idx") / "test.sqlite3"
    return SETTINGS.with_(embedder="hash", corpus_dir=CORPUS, db_path=str(db))


@pytest.fixture(scope="session")
def indexed(settings):
    stats = ingest(settings)
    assert stats["chunks"] > 20
    return settings


@pytest.fixture(scope="session")
def retriever(indexed):
    return Retriever(
        store=Store(indexed.db_path), embedder=get_embedder(indexed), settings=indexed
    )
