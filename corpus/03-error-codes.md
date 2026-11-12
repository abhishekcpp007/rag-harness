# Error Codes

Every error response carries a stable `code`, a human `message` and a `request_id`. Match on
`code`. The message text changes without notice.

## Request errors, 400 range

| Code | Meaning | What to do |
|---|---|---|
| NW-4001 | Missing required field | Check the field named in `details.field` |
| NW-4002 | Amount is not an integer in the smallest unit | Send 125000, not 1250.00 |
| NW-4003 | Unsupported currency for this merchant | Ask support to enable the currency |
| NW-4011 | API key missing or malformed | Send `Authorization: Bearer nw_live_...` |
| NW-4012 | API key revoked | Rotate the key in the dashboard |
| NW-4013 | Scope not granted to this key | Key needs `charges:write` |
| NW-4021 | Settlement batch rejected by the acquirer | See the settlement stall runbook |
| NW-4029 | Rate limit exceeded | Back off, read `Retry-After` |
| NW-4090 | Idempotency key replayed with a different body | Use a fresh key |
| NW-4091 | Charge already captured | Nothing to do, the first capture won |

## Platform errors, 500 range

| Code | Meaning | What to do |
|---|---|---|
| NW-5001 | Acquirer timeout | Do not retry the charge, poll its state |
| NW-5002 | Ledger write failed, charge rolled back | Safe to retry with the same idempotency key |
| NW-5003 | Region failover in progress | Retry after 60 seconds |

## The two that get mishandled

**NW-5001 is not a failure.** An acquirer timeout means Northwind does not yet know the outcome.
Retrying creates a second authorisation on the customer's card. Poll `GET /v2/charges/{id}` until
the state leaves `requires_action`, which takes under 90 seconds in practice.

**NW-4021 is never the merchant's fault.** It means a whole settlement batch bounced, so every
charge in that batch is stuck, not just the one being looked at. Opening a ticket per charge makes
the queue worse. Open one ticket for the batch id.
