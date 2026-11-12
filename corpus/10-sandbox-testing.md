# Sandbox and Testing

Sandbox keys start with `nw_test_`. They cannot reach production and a production key cannot reach
sandbox. Mixing them up returns `NW-4011`.

## Test cards

| Number | Behaviour |
|---|---|
| 4111 1111 1111 1111 | Authorises and captures |
| 4000 0000 0000 0002 | Declined by issuer |
| 4000 0000 0000 0069 | Expired card |
| 4000 0000 0000 0119 | Acquirer timeout, returns NW-5001 |
| 4000 0000 0000 9995 | Insufficient funds |

Any future expiry and any 3 digit CVV work. The amount does not change the outcome, the card
number does.

## Forcing a state

Sandbox exposes one endpoint that production does not.

```
POST /v2/_sandbox/charges/{id}/advance
{ "to": "disputed" }
```

Use it to drive a charge into `disputed` or `settled` without waiting for a batch. The endpoint is
missing in production, so guard it behind an environment check in your own test helpers rather
than discovering it at deploy time.

## Replaying webhooks

```
nwctl webhook replay --event evt_01JC... --endpoint https://localhost:4000/hooks
```

Replay sends the original payload with a fresh signature and a fresh timestamp, so a handler that
rejects old timestamps still accepts it.

## What sandbox does not simulate

Settlement timing is compressed. A sandbox batch sweeps every 10 minutes instead of nightly, so
sandbox will never reproduce a T+2 timing bug. Rate limits are real but the tier is fixed at the
sandbox row and cannot be raised.
