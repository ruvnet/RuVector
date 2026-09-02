# How RuVector proves which key signed a retrieval result

## The short version

RuVector retrieval receipts could already prove that a result had not
changed after it was issued. They could not prove that a particular
signing key approved that receipt.

This change adds Ed25519 signatures to receipt roots. One signature can
cover one receipt or a batch of receipts. At a batch size of 128, the
measured signing cost per query fell to 5.8 to 7.7 percent of the single
receipt cost. Every one of 1,500 injected signing and proof tamper trials
was rejected.

The practical takeaway is simple: use one signature per query when the
proof must exist immediately. Use a batch when throughput matters and a
short wait for the batch to close is acceptable.

## What was fixed before merge

Signing only a raw 32 byte root was too ambiguous. A valid signature
could be copied into a context the signer never intended. The first API
also let callers check Merkle inclusion without first checking the batch
signature.

The merged contract signs a complete statement containing:

1. Protocol version

2. Receipt or batch purpose

3. SHA256 identifier of the public key

4. Deployment or index scope

5. Issuance time

6. Receipt or batch root

Verification uses the strict Ed25519 path. A successful signature check
returns a trusted root token. Batch inclusion accepts only that token, so
the compiler makes the required authentication step difficult to skip.

Empty batches and invalid proof indexes now return typed errors instead
of terminating the process. The canonical statement encoder uses a
fixed 140 byte buffer, so the stronger contract adds no heap allocation
to the signing or verification path.

## Measured result

Command:

```text
cargo run --release -p ruvector-retrieval-receipt --bin benchmark -- 5000 128 10 200
```

Environment: 12 logical CPUs, Rust 1.94.1, release profile, 128 warmup
sign and verify operations, then three independent runs.

Mean results:

1. Batch 1: 15,598 ns signing per query, 36,411 ns uncached verify, 170
bytes of portable evidence

2. Batch 8: 2,940 ns signing per query, 34,948 ns uncached verify, 266
bytes of portable evidence

3. Batch 32: 1,337 ns signing per query, 35,394 ns uncached verify, 330
bytes of portable evidence

4. Batch 128: 1,090 ns signing per query, 41,219 ns uncached verify, 394
bytes of portable evidence

The batch 128 signing cost averaged about 14 times lower than batch 1.
Uncached verification did not improve with batch size, which confirms
that the benchmark measured real signature amortization instead of
hiding verification work.

## What this proves

It proves that the signed statement came from the private key matching a
supplied public key and that the statement fields were not modified.

It does not prove that the result was correct. A compromised signer can
sign false data. It also does not prove that a key belongs to a named
company or engine. That requires an external key registry, rotation
policy, revocation history, and durable audit record.

This distinction matters. A cryptographic primitive can authenticate a
key. Organizational identity is a governance system built around that
primitive.

## Deployment decision

This remains experimental and is not connected to the default query
path. Production promotion needs two additional measurements:

1. Batch fill latency under the target query arrival rate

2. Key custody and rotation using the target HSM or KMS

The largest uncertainty is batch fill time. A batch of 128 is useful only
if the traffic rate closes it inside the receipt availability service
level. The fix path is a time bounded batcher that signs when either the
size limit or the time limit is reached.

## Acceptance test

Run the command above. Promotion requires both acceptance sections to
print `ACCEPT`, all 23 tests to pass, and every tamper count to equal its
trial count.

Design record: `docs/adr/ADR-340-signed-retrieval-receipt-anchoring.md`

Evidence: `docs/research/nightly/2026-08-31-signed-retrieval-receipts/README.md`
