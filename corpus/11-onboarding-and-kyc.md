# Merchant Onboarding and KYC

A merchant goes live in four stages: `registered`, `documents_submitted`, `under_review`,
`activated`.

## Documents

For a private limited company: certificate of incorporation, PAN, GSTIN, a cancelled cheque or
bank statement no older than 3 months, and the director's identity proof. A proprietorship
substitutes the owner's PAN for the company PAN.

Documents are uploaded through the dashboard, never over email. The review SLA is 2 business days
from the point where the last required document is accepted, and a rejected document resets that
clock.

## Limits before activation

A merchant in `under_review` can take live charges up to a cumulative 2 lakh INR, but settlement is
held until activation. This is deliberate. It lets an integration be proven against real cards
without exposing Northwind to a merchant who never clears review.

## Sub-sellers on a marketplace

A marketplace onboards its own sub-sellers through `POST /v2/accounts`. The marketplace is
responsible for collecting sub-seller KYC and Northwind re-verifies the PAN and the bank account
only. A sub-seller whose PAN fails verification can still receive a split, but the split is held in
the marketplace's balance rather than paid out.

## Deactivation

An activated merchant can be deactivated for a failed periodic review, for a dispute rate above
1 percent of monthly volume, or for a chargeback ratio breach raised by the network. Deactivation
stops new charges immediately. Existing captured charges still settle, because the money already
belongs to the merchant.
