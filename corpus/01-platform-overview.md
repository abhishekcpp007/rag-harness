# Northwind Payments Platform Overview

Northwind is a card acquiring and payouts platform for marketplaces in India and Singapore. A
merchant integrates once and gets card collection, UPI collection, split settlement to sub-sellers
and outbound payouts.

## Services

The platform is seven services behind one gateway.

| Service | Owns | Language |
|---|---|---|
| `gateway` | auth, rate limits, request routing | Go |
| `authz` | API keys, scopes, merchant tenancy | Go |
| `payments` | charge lifecycle, refunds, chargebacks | Java |
| `settlement` | batching, acquirer files, reconciliation | Java |
| `payouts` | beneficiary records, outbound transfers | Python |
| `ledger` | double entry book of record | Rust |
| `notify` | webhooks, email, merchant alerts | Python |

`ledger` is the only service allowed to write a balance. Everything else asks it to post a journal
entry and waits for the entry id. A service that computes a balance locally is a bug, and the
reconciliation job will flag it the next morning.

## Request path

A charge arrives at `gateway`, which checks the API key with `authz` and applies the merchant rate
limit. It forwards to `payments`, which writes an intent row, calls the acquirer, and posts two
journal entries to `ledger`. `notify` picks the event off the outbox table and delivers the
webhook.

Nothing calls the acquirer except `payments`. If you need acquirer data somewhere else, read it
from the charge record, not from the acquirer.

## Environments

There are three. `sandbox` talks to acquirer simulators and never moves money. `staging` talks to
the acquirer's own test endpoints and moves test money. `production` is production.

Sandbox data is wiped every Sunday at 02:00 IST. Do not store anything in sandbox you cannot
recreate from a script.

## Regions

India traffic runs out of `ap-south-1` and Singapore traffic out of `ap-southeast-1`. The two
regions share no database. A merchant is pinned to one region at onboarding and cannot be moved
without a support ticket and a 48 hour freeze on their account.
