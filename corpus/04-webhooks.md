# Webhooks

`notify` delivers events over HTTPS POST. Register an endpoint per environment in the dashboard.

## Delivery

Each delivery carries two headers.

```
Northwind-Signature: t=1763000000,v1=5d41402abc4b2a76b9719d911017c592
Northwind-Event-Id: evt_01JC...
```

Verify by computing HMAC SHA-256 over `"{t}.{raw_body}"` with your endpoint's signing secret and
comparing against `v1` in constant time. Reject anything where `t` is more than 5 minutes old,
otherwise a captured request can be replayed at leisure.

Do not parse the body before verifying the signature.

## Retries

A delivery is retried if your endpoint does not answer 2xx within 10 seconds. The schedule is
10 seconds, 1 minute, 10 minutes, 1 hour, then every 6 hours up to 3 days. After 3 days the event
is dropped and the endpoint is marked unhealthy.

Deliveries are at least once. The same `Northwind-Event-Id` can arrive twice, so store the id and
make the handler idempotent.

## Ordering

There is no ordering guarantee. `charge.captured` can arrive before `charge.authorised`. Every
event carries the full object and an `object_version` integer. Drop an event whose
`object_version` is lower than the version you already stored.

## Event types

`charge.authorised`, `charge.captured`, `charge.failed`, `charge.refunded`, `dispute.opened`,
`dispute.won`, `dispute.lost`, `payout.paid`, `payout.returned`, `settlement.completed`.

## Why the endpoint went unhealthy

The most common cause is answering 2xx only after doing the work. Acknowledge inside 10 seconds by
writing the event to your own queue and returning 200, then do the work off the queue.
