# Payouts

Outbound transfers to a beneficiary bank account or UPI handle. Payouts are the only part of the
platform that can move money out, so the controls are heavier than on the collection side.

## Beneficiaries

A beneficiary is created once and reused.

```
POST /v2/beneficiaries
{ "name": "Lata Rane", "account_number": "0012...", "ifsc": "HDFC0000123", "type": "bank" }
```

A new beneficiary is in `pending_verification` for 30 minutes while a penny drop confirms the name
match. Payouts to a pending beneficiary are rejected. A name mismatch above the fuzzy threshold
puts the beneficiary in `rejected` and it cannot be retried with the same account number for 24
hours.

## Creating a payout

```
POST /v2/payouts
{ "beneficiary_id": "ben_71", "amount": 450000, "mode": "IMPS", "idempotency_key": "..." }
```

Modes are `IMPS` up to 5 lakh INR, `NEFT` with no cap but settled in half hourly cycles, `RTGS`
above 2 lakh INR, and `UPI` up to 1 lakh INR.

## States

`created`, `processing`, then `paid` or `returned`. A `returned` payout means the beneficiary bank
sent the money back, usually a closed account or a wrong IFSC. The amount is credited back to the
merchant balance within 2 business days and a `payout.returned` webhook fires.

A payout stuck in `processing` for over 4 hours on IMPS is an incident. NEFT can sit for a full
cycle and that is normal.

## Negative balance

Payouts are blocked whenever the merchant balance is negative, including a balance pushed negative
by a dispute debit. The block is automatic and support cannot override it. The merchant tops up by
bank transfer to the virtual account on their dashboard.
