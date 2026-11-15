FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    RAG_DB_PATH=/data/index.sqlite3 \
    RAG_CORPUS_DIR=/app/corpus

WORKDIR /app

COPY pyproject.toml README.md ./
COPY rag ./rag
RUN pip install --no-cache-dir -e ".[embed]"

COPY corpus ./corpus
COPY eval ./eval

# The test stage. Dev deps and the suite land here and nowhere else, so the image you ship stays
# small. Build it with: docker build --target test -t rag-harness:test .
FROM base AS test

RUN pip install --no-cache-dir -e ".[dev]"
COPY tests ./tests
# The benchmarks are linted in here but never run: the scale sweep wants minutes and the load
# test wants a running service. hnswlib is imported inside the function that needs it, so the
# test image does not carry a compiler toolchain for it.
COPY bench ./bench

# Hash embedder, so the suite needs no model download and no network at all.
ENV RAG_EMBEDDER=hash \
    RAG_DB_PATH=/tmp/test-index.sqlite3

CMD ["sh", "-c", "ruff check rag eval tests bench && pytest -q"]

FROM base AS runtime

# Index at build time so the container starts answering immediately. The model weights land in
# the image too, which keeps the first request off the network.
RUN mkdir -p /data && python -m rag.cli ingest

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)"

CMD ["uvicorn", "rag.api:app", "--host", "0.0.0.0", "--port", "8000"]
