"""What a service needs before anyone else points traffic at it, in one file. No dependencies.

The log line is JSON on stdout. A container's log driver wants one object per line, and a human
reading `docker logs` wants the fields in a stable order, so both get the same line. The request
id is echoed back in a header, which is the only way a user's bug report and a log line ever meet.

Metrics are a counter per route and status plus a latency histogram, rendered in the Prometheus
text format. The buckets are fixed and chosen off `bench/load.py`: a search answers in single
digit milliseconds warm, so buckets that start at 100ms would put every request in the first one
and say nothing.

The rate limiter is a token bucket per client, and it is off by default. A single tenant service
behind a gateway does not need one, and switching it on by default would mean the load test is
measuring the limiter instead of the service. `RAG_RATE_LIMIT_RPS` turns it on.
"""
from __future__ import annotations

import json
import sys
import threading
import time
from collections import defaultdict

BUCKETS_MS = (1, 2, 5, 10, 25, 50, 100, 250, 500, 1000, 2500)


class Metrics:
    """Counters and a histogram, guarded by one lock.

    Every request touches this, so the lock is held for a dictionary update and nothing else. No
    work happens inside it and no I/O.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.requests: dict[tuple[str, int], int] = defaultdict(int)
        self.latency: dict[str, list[int]] = defaultdict(lambda: [0] * (len(BUCKETS_MS) + 1))
        self.latency_sum_ms: dict[str, float] = defaultdict(float)
        self.limited: int = 0
        self.started = time.time()

    def observe(self, route: str, status: int, ms: float) -> None:
        slot = len(BUCKETS_MS)
        for i, edge in enumerate(BUCKETS_MS):
            if ms <= edge:
                slot = i
                break
        with self._lock:
            self.requests[(route, status)] += 1
            self.latency[route][slot] += 1
            self.latency_sum_ms[route] += ms

    def count_limited(self) -> None:
        with self._lock:
            self.limited += 1

    def render(self, index: dict) -> str:
        with self._lock:
            requests = dict(self.requests)
            latency = {r: list(b) for r, b in self.latency.items()}
            sums = dict(self.latency_sum_ms)
            limited = self.limited
        out = [
            "# HELP rag_requests_total Requests by route and status.",
            "# TYPE rag_requests_total counter",
        ]
        for (route, status), n in sorted(requests.items()):
            out.append(f'rag_requests_total{{route="{route}",status="{status}"}} {n}')
        out += [
            "# HELP rag_request_duration_ms Request latency by route.",
            "# TYPE rag_request_duration_ms histogram",
        ]
        for route, buckets in sorted(latency.items()):
            running = 0
            for edge, n in zip(BUCKETS_MS, buckets):
                running += n
                out.append(f'rag_request_duration_ms_bucket{{route="{route}",le="{edge}"}} {running}')
            running += buckets[-1]
            out.append(f'rag_request_duration_ms_bucket{{route="{route}",le="+Inf"}} {running}')
            out.append(f'rag_request_duration_ms_sum{{route="{route}"}} {sums[route]:.1f}')
            out.append(f'rag_request_duration_ms_count{{route="{route}"}} {running}')
        out += [
            "# HELP rag_rate_limited_total Requests refused by the rate limiter.",
            "# TYPE rag_rate_limited_total counter",
            f"rag_rate_limited_total {limited}",
            "# HELP rag_index_chunks Chunks currently indexed.",
            "# TYPE rag_index_chunks gauge",
            f"rag_index_chunks {index.get('chunks', 0)}",
            "# HELP rag_index_documents Documents currently indexed.",
            "# TYPE rag_index_documents gauge",
            f"rag_index_documents {index.get('documents', 0)}",
            "# HELP rag_uptime_seconds Seconds since the process started.",
            "# TYPE rag_uptime_seconds gauge",
            f"rag_uptime_seconds {time.time() - self.started:.1f}",
        ]
        return "\n".join(out) + "\n"


class TokenBucket:
    """One bucket per client key, refilled by the clock rather than by a background task.

    Idle keys are dropped once the map gets large, because a per-IP dictionary that only ever
    grows is a slow memory leak wearing a rate limiter's clothes.
    """

    MAX_KEYS = 10_000

    def __init__(self, rps: float, burst: int) -> None:
        self.rps = rps
        self.burst = float(burst)
        self._lock = threading.Lock()
        self._state: dict[str, tuple[float, float]] = {}

    def take(self, key: str) -> tuple[bool, float]:
        """Returns whether the request may proceed, and how long to wait if it may not."""
        if self.rps <= 0:
            return True, 0.0
        now = time.monotonic()
        with self._lock:
            tokens, last = self._state.get(key, (self.burst, now))
            tokens = min(self.burst, tokens + (now - last) * self.rps)
            if tokens < 1.0:
                self._state[key] = (tokens, now)
                return False, (1.0 - tokens) / self.rps
            self._state[key] = (tokens - 1.0, now)
            if len(self._state) > self.MAX_KEYS:
                cutoff = now - self.burst / self.rps
                self._state = {k: v for k, v in self._state.items() if v[1] > cutoff}
            return True, 0.0


def log_request(fields: dict, fmt: str = "json") -> None:
    if fmt == "json":
        sys.stdout.write(json.dumps(fields, separators=(",", ":")) + "\n")
    else:
        body = " ".join(f"{k}={v}" for k, v in fields.items())
        sys.stdout.write(body + "\n")
    sys.stdout.flush()
