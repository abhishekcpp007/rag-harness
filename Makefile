.PHONY: install index test lint eval sweep tables significance serve bench latency load docker docker-test clean

install:
	uv venv --python 3.12 .venv || python3 -m venv .venv
	.venv/bin/pip install -e ".[dev,embed,bench]"

index:
	.venv/bin/python -m rag.cli ingest

test:
	.venv/bin/python -m pytest -q

lint:
	.venv/bin/ruff check rag eval tests bench

eval:
	PY=.venv/bin/python sh eval/run_all.sh

sweep:
	.venv/bin/python -m eval.sweep

tables:
	.venv/bin/python -m eval.table

# Bootstrap intervals and the paired tests. This is the file that decides which differences in
# the README are allowed to be called findings.
significance:
	.venv/bin/python -m eval.significance

serve:
	.venv/bin/python -m rag.cli serve

# Where each index stops working. Takes a few minutes: it builds an approximate index at every
# size on both vector geometries so the recall it gives up is a measurement too.
bench:
	.venv/bin/python -m bench.scale --sizes 1000,10000,50000 --dense-max 250000

# Per-query latency for each retrieval configuration. Slow, because the cross encoders are in it.
latency:
	.venv/bin/python -m bench.latency

# Needs the service up in another shell. Exits non-zero if any request failed, which is the
# point: this is what found the shared SQLite connection.
load:
	.venv/bin/python -m bench.load -c 16 -n 400 --out bench/results/load-search.json
	.venv/bin/python -m bench.load --route answer -c 16 -n 200 --out bench/results/load-answer.json

docker:
	docker compose up --build

# The same lint and suite the CI docker job runs. No network, so a pass here means the tests
# really do not touch one.
docker-test:
	docker build --target test -t rag-harness:test .
	docker run --rm --network none rag-harness:test ruff check rag eval tests bench
	docker run --rm --network none rag-harness:test pytest -q

clean:
	rm -rf data/*.sqlite3 .pytest_cache .ruff_cache
