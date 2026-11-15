"""Hit the running service with concurrent requests and report what it did.

A single-threaded request never tells you whether a service is correct under load. FastAPI runs
a plain `def` route in a worker threadpool, so two requests can land in different threads inside
the same process, touching the same objects. Everything in `rag/` was written and tested one
call at a time, which is exactly the shape of code that passes its suite and then falls over in
production.

Run:
    make serve                                 # in one shell
    python -m bench.load --concurrency 16      # in another
    python -m bench.load --route answer -n 200

The exit status is non-zero if any request failed, so this works as a gate rather than a
curiosity. Failures are grouped by message, because one bug produces hundreds of identical ones.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import httpx
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

GOLDEN = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "eval", "golden.yaml")
TIMEOUT_S = 30.0


def questions() -> list[str]:
    with open(GOLDEN) as f:
        return [q["question"] for q in yaml.safe_load(f)]


def one(client: httpx.Client, base: str, route: str, question: str) -> tuple[float, int, str]:
    t = time.perf_counter()
    try:
        if route == "answer":
            r = client.post(f"{base}/answer", json={"question": question})
        else:
            r = client.get(f"{base}/search", params={"q": question, "k": 5})
        ms = (time.perf_counter() - t) * 1000
        if r.status_code != 200:
            return ms, r.status_code, r.text[:300]
        body = r.json()
        key = "hits" if route == "search" else "citations"
        if key not in body:
            return ms, r.status_code, f"200 with no {key} in the body"
        return ms, r.status_code, ""
    except (httpx.HTTPError, OSError, ValueError) as e:
        return (time.perf_counter() - t) * 1000, 0, f"{type(e).__name__}: {e}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--route", default="search", choices=["search", "answer"])
    ap.add_argument("--concurrency", "-c", type=int, default=16)
    ap.add_argument("--requests", "-n", type=int, default=400)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    qs = questions()
    work = [qs[i % len(qs)] for i in range(args.requests)]

    with httpx.Client(timeout=TIMEOUT_S) as client:
        client.get(f"{args.url}/health")  # warm the index before the clock starts
        started = time.perf_counter()
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            results = list(pool.map(lambda q: one(client, args.url, args.route, q), work))
        wall = time.perf_counter() - started

    latencies = sorted(r[0] for r in results)
    failures = collections.Counter(r[2] for r in results if r[2])
    codes = collections.Counter(r[1] for r in results)

    def pct(p: float) -> float:
        return round(latencies[min(len(latencies) - 1, int(p * len(latencies)))], 1)

    report = {
        "route": args.route,
        "concurrency": args.concurrency,
        "requests": args.requests,
        "wall_s": round(wall, 2),
        "rps": round(args.requests / wall, 1),
        "p50_ms": round(statistics.median(latencies), 1),
        "p95_ms": pct(0.95),
        "p99_ms": pct(0.99),
        "max_ms": round(latencies[-1], 1),
        "status_codes": dict(sorted(codes.items())),
        "failed": sum(failures.values()),
        "failures": failures.most_common(5),
    }
    print(json.dumps(report, indent=2))
    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2)
    if report["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
