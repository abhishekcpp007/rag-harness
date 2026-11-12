.PHONY: install index test lint eval sweep tables serve docker clean

install:
	uv venv --python 3.12 .venv || python3 -m venv .venv
	.venv/bin/pip install -e ".[dev,embed]"

index:
	.venv/bin/python -m rag.cli ingest

test:
	.venv/bin/python -m pytest -q

lint:
	.venv/bin/ruff check rag eval tests

eval:
	.venv/bin/python -m eval.run_eval --name default
	.venv/bin/python -m eval.run_eval --name no-rerank --no-rerank
	.venv/bin/python -m eval.run_eval --name no-headings --no-headings
	.venv/bin/python -m eval.run_eval --name hybrid --with-sparse
	.venv/bin/python -m eval.run_eval --name sparse-only --no-dense --with-sparse
	.venv/bin/python -m eval.run_eval --name hash-only --embedder hash
	.venv/bin/python -m eval.run_eval --name hash-hybrid --embedder hash --with-sparse

sweep:
	.venv/bin/python -m eval.sweep

tables:
	.venv/bin/python -m eval.table

serve:
	.venv/bin/python -m rag.cli serve

docker:
	docker compose up --build

clean:
	rm -rf data/*.sqlite3 .pytest_cache .ruff_cache
