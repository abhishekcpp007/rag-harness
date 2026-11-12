# Rate Limits

Limits are per API key, not per merchant, and they are enforced at `gateway` with a sliding window
of 60 seconds.

| Tier | Writes per minute | Reads per minute | Burst |
|---|---|---|---|
| sandbox | 60 | 600 | 20 |
| standard | 300 | 3000 | 60 |
| scale | 1200 | 12000 | 200 |

Every response carries the budget.

```
Northwind-RateLimit-Limit: 300
Northwind-RateLimit-Remaining: 271
Northwind-RateLimit-Reset: 34
```

Exceeding the limit returns HTTP 429 with code `NW-4029` and a `Retry-After` header in seconds.

## Backoff

Retry with full jitter. Sleep a random amount between zero and `min(cap, base * 2 ** attempt)`,
with `base` at 0.5 seconds and `cap` at 30 seconds. Fixed backoff from many clients re-synchronises
them into the same second, which is how a limit breach turns into a thundering herd.

Do not retry a 429 on a write without the original `Idempotency-Key`. A retry with a new key is a
second charge.

## Getting a higher tier

Tier changes are a support request with a reason and a traffic estimate. They are applied within
one business day. A tier is not raised during an incident, because the limit is usually the only
thing keeping the queue drained.

## Bulk reads

Listing endpoints cost one unit per page, not per row, so a `limit=100` page is the same budget as
a `limit=10` page. Pull the full page.
