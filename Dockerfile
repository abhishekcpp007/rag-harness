FROM python:3.12-slim

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

# Index at build time so the container starts answering immediately. The model weights land in
# the image too, which keeps the first request off the network.
RUN mkdir -p /data && python -m rag.cli ingest

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)"

CMD ["uvicorn", "rag.api:app", "--host", "0.0.0.0", "--port", "8000"]
