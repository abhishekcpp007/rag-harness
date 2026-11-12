# Runbook: Settlement stall

Fires when a batch has been in `submitted` for more than 90 minutes, or when charges rejected with
`NW-4021` cross 50 in an hour.

## Confirm the blast radius

```
nwctl batch list --state submitted --older-than 90m --region ap-south-1
nwctl batch show <batch_id> --counts
```

Read the charge count. One stuck batch is routine. Three stuck batches in a row means the acquirer
connection is the problem, not the batch.

## Recovery steps

1. Check the acquirer status page and the shared channel `#acq-icici` before touching anything.
2. If the acquirer is up, re-submit once with `nwctl batch resubmit <batch_id>`. Re-submission is
   idempotent on the batch id and will not double pay.
3. If the re-submission is rejected again, mark the batch `held` with
   `nwctl batch hold <batch_id> --reason "<ticket>"`. Holding releases the charges back to
   `captured` so the next sweep picks them up.
4. Page the on-call for `settlement` if two re-submissions fail, or if the held amount is over
   50 lakh INR.

Never edit a batch row directly. The batch id appears in the acquirer file and in `ledger`, and a
hand edit puts the two out of sync in a way recon cannot repair on its own.

## What to tell merchants

One message on the status page naming the batch and the expected settlement date. Do not promise a
time until the acquirer has acknowledged the re-submission, because the acknowledgement is the only
signal that the file was accepted.

## After the incident

Attach the batch id, the acquirer's reject reason and the recon output to the incident doc. The
reject reason code from the acquirer is not the same as the Northwind code, and the mapping lives
in `settlement/src/main/resources/acq-reject-map.yaml`.
