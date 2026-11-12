# Charges API

Base URL is `https://api.northwind.dev/v2` in sandbox and `https://api.northwind.com/v2` in
production.

## Create a charge

```
POST /v2/charges
Authorization: Bearer nw_live_...
Idempotency-Key: <uuid>

{
  "amount": 125000,
  "currency": "INR",
  "method": "card",
  "customer_id": "cus_8f21",
  "capture": false,
  "metadata": {"order_id": "A-5512"}
}
```

`amount` is always in the smallest unit. 125000 means 1250.00 INR. Sending a decimal is rejected
with `NW-4002`.

The `Idempotency-Key` header is required on every write. Keys live for 24 hours. Replaying a key
with a different body returns `NW-4090` and does not create a second charge.

## Capture

A charge created with `capture: false` is authorised but not captured. Capture it within 7 days or
the authorisation expires and the hold is released by the issuer.

```
POST /v2/charges/{id}/capture
{ "amount": 125000 }
```

Partial capture is allowed down to 1 unit. You cannot capture more than the authorised amount.

## Charge states

`requires_action` to `authorised` to `captured` to `settled`. A charge can move to `failed` from
any state before `captured`, and to `refunded` or `disputed` after `captured`.

The state machine only moves forward. There is no API that moves a charge backwards, and support
cannot do it either. A mistake is corrected with a refund, not by editing state.

## Listing

```
GET /v2/charges?created_after=2026-11-01T00:00:00Z&limit=100&cursor=...
```

Cursor pagination only. There is no offset parameter, because offsets over a live table skip rows
when new charges land between pages. Maximum `limit` is 100.
