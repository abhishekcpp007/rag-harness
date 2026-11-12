# Settlement

Captured charges are swept into batches and sent to the acquirer. Money reaches the merchant bank
account on T+2 for India and T+1 for Singapore, counting business days only.

## Batch cutoff

The sweep runs at 23:40 IST for `ap-south-1` and 23:40 SGT for `ap-southeast-1`. A charge captured
at 23:41 waits for the next day's batch. There is no manual sweep, and asking for one has been
refused every time it was asked.

## Batch states

`building` to `submitted` to `acknowledged` to `completed`. A batch that the acquirer rejects goes
to `rejected` and every charge in it returns to `captured`, so the charges are not lost, they are
simply unsettled until the next batch picks them up.

## Split settlement

A marketplace charge can be split across sub-sellers at capture time.

```
"split": [
  {"account_id": "acct_11", "amount": 90000},
  {"account_id": "acct_42", "amount": 30000}
]
```

Splits must sum to the captured amount exactly. A split that is short by even 1 unit is rejected
with `NW-4001` naming `split`. Northwind's own fee is taken from the platform account, not from
the split, so the split is gross and not net.

## Reconciliation

Every morning at 06:00 local the recon job compares the acquirer file against `ledger`. A mismatch
raises a `recon.break` alert with the batch id and the rupee difference. A break over 1000 INR
pages the on-call. Anything smaller goes to the finance queue and is cleared by hand within two
business days.

The recon job reads the acquirer file from `s3://nw-acquirer-files/{region}/{date}/`. Files land
by 04:30. A missing file is itself a break, and the job raises `recon.file_missing` rather than
reporting a clean run on no data.
