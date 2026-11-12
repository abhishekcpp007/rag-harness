# Refunds and Disputes

## Refunds

```
POST /v2/charges/{id}/refunds
{ "amount": 50000, "reason": "requested_by_customer" }
```

A refund can be partial, and several partial refunds are allowed up to the captured amount. A
refund on an uncaptured charge is rejected, because there is nothing to return. Void the
authorisation instead with `POST /v2/charges/{id}/void`.

Refunds settle on the next batch. The customer sees the money in 5 to 7 business days for cards
and within 24 hours for UPI. Northwind cannot make a card refund arrive faster, the timing belongs
to the issuer.

The Northwind processing fee on the original charge is **not** returned on a refund. The
interchange portion is returned by the network, and that lands as a separate credit on the monthly
invoice rather than against the charge.

## Disputes

A dispute opens when the cardholder's bank raises a chargeback. The flow is `opened`,
`evidence_submitted`, then `won` or `lost`.

The evidence window is 7 calendar days from `dispute.opened`, not business days. Miss it and the
dispute is lost by default with no appeal.

```
POST /v2/disputes/{id}/evidence
{ "files": ["file_01J..."], "narrative": "..." }
```

Useful evidence is the delivery proof, the customer's own written acknowledgement and the AVS
result on the original authorisation. A narrative with no attached document loses.

## Money movement during a dispute

The disputed amount is debited from the merchant balance the moment the dispute opens, not when it
is lost. Winning returns it on the next settlement. A merchant whose balance cannot cover the debit
goes into negative balance, and payouts are paused until the balance is positive again.
