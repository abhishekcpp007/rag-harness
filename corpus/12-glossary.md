# Glossary

**Acquirer** The bank that holds Northwind's merchant account and routes card traffic to the
networks. Northwind uses ICICI in India and DBS in Singapore.

**Authorisation** The issuer's promise to hold funds. It is not money. An authorisation expires
after 7 days if it is not captured.

**Capture** Turning an authorisation into a claim on real money. Capture is what starts settlement.

**Chargeback** The cardholder's bank reversing a settled charge on the cardholder's behalf. Inside
Northwind this is modelled as a dispute.

**Interchange** The portion of a card fee that goes to the issuing bank. Returned to the merchant
on a refund, but on the monthly invoice rather than against the charge.

**Idempotency key** A client supplied uuid that makes a write safe to retry. Lives 24 hours.

**Journal entry** A balanced pair of postings in `ledger`. The only way a balance changes.

**Penny drop** A one rupee transfer used to confirm that a beneficiary account exists and the name
matches.

**Recon break** A difference between the acquirer file and `ledger` for a given batch.

**Settlement batch** The nightly bundle of captured charges sent to the acquirer.

**Split** The division of one marketplace charge across sub-seller accounts at capture time.

**Sweep** The job that moves captured charges into a batch at the nightly cutoff.

**T+2** Two business days after the batch, which is when Indian merchants see money in the bank.

**Virtual account** A per merchant bank account number used for top ups when the balance goes
negative.
