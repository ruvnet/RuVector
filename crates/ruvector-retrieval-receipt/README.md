# ruvector-retrieval-receipt

**Witness-chained provenance for ANN retrieval results** — cryptographic receipts that commit a
query's top-k results (together with copies of each vector's ingestion `WriteReceipt`) so that a
receipt/result pair, once issued, cannot be silently mutated in transit or in storage. Part of
the [ruvector](https://github.com/ruvnet/ruvector) ecosystem.

> `ruvector-proof-gate` proves what was *written*. This crate makes the record of what a query
> *returned* tamper-evident after issuance — a read-side provenance primitive no major vector
> database (Qdrant, Milvus, Weaviate, LanceDB, FAISS, pgvector, Chroma, Vespa, Pinecone)
> documents today.

## What it gives you

Search a `RetrievalIndex` (a brute-force cosine index whose ingestion path is a real
`ruvector_proof_gate::HashChainGate`), wrap the result set in a `RetrievalReceipt`, and a later
holder of the receipt can check — offline, without talking to the query engine — that the
results they hold are the ones the engine committed to at query time.

**Threat model, stated plainly:** receipts are unsigned commitments produced by the query
engine itself. They detect *post-issuance mutation* of a receipt/result pair. They do **not**
protect against a dishonest query engine (leaves are engine-chosen; nothing binds a score to an
actual cosine computation or the committed set to the true top-k), and they do **not** prove
write-chain membership — leaves commit to *copies* of `WriteReceipt` fields, verification never
consults the write gate, so mutating the ingestion history after issuance leaves existing
receipts verifying. Anchoring leaves to `MerkleGate`'s MMR membership proofs is the named
future-work item. See ADR-304's Threat Model section.

## Variants

| Variant | Generation | Verify 1-of-k (worst case, k=10) | Proof size (worst case, k=10) | Guarantee |
|---|---|---|---|---|
| `NoReceipt` | ~0 | N/A | 0 bytes | none (baseline) |
| `PerResultReceipt` | O(k) hashes | O(idx) work | O(idx) bytes | sequential tamper-evidence |
| `MerkleReceipt` | O(k) hashes | O(log k) work | O(log k) bytes | membership-proof tamper-evidence |

## Usage

```rust
use ruvector_retrieval_receipt::{
    query_hash, ReceiptVariant, RetrievalIndex, RetrievalReceipt,
};

let index = RetrievalIndex::ingest(5_000, 128, 0xC0FF_EE01);
let query = vec![0.1; 128];
let results = index.search(&query, 10);

let qh = query_hash(&query);
let root = index.index_state_root();
let receipt = RetrievalReceipt::build(ReceiptVariant::Merkle, qh, root, &results);

assert!(receipt.verify_full(qh, root, &results));
```

## Performance

Measured (n=5,000, dims=128, k=10, release build): `MerkleReceipt` generation ≈ 19.6 µs
(1.8% of a 1.1 ms brute-force search), single-result verification ≈ 3.8 µs, worst-case proof
size 160 bytes, vs 320 bytes / 8.2 µs for `PerResultReceipt` — where the per-result figure is
defined as the genesis-anchored chain replay (O(idx)); the durable comparison is the
asymptotic O(log k) vs O(k) proof size, not the specific constant at k=10. Both variants
rejected all 200/200 injected tamper trials — expected from SHA-256 by construction, a
regression check rather than an empirical detection rate. Full methodology and raw output:
[`docs/research/nightly/2026-08-13-retrieval-receipts/README.md`](../../docs/research/nightly/2026-08-13-retrieval-receipts/README.md).

See [`ADR-304`](../../docs/adr/ADR-304-retrieval-receipts.md) for the design rationale,
documented limitations (Merkle padding malleability), and rejection criteria for production
promotion.

## Signed anchoring (origin authentication)

The variants above are unsigned: they detect tamper but do not authenticate an issuing key.
The `signing` module adds typed and scoped Ed25519 root signing, either per query or
amortized across a batch. Organizational identity still requires an external key registry,
rotation policy, and revocation history.

```rust
use ruvector_retrieval_receipt::{
    query_hash, verify_root, AnchorContext, AnchorPurpose, BatchAnchor, Issuer,
    ReceiptVariant, RetrievalIndex, RetrievalReceipt,
};

let issuer = Issuer::generate();
let index = RetrievalIndex::ingest(5_000, 128, 0xC0FF_EE01);
let query = vec![0.1; 128];
let results = index.search(&query, 10);
let qh = query_hash(&query);
let receipt = RetrievalReceipt::build(ReceiptVariant::Merkle, qh, index.index_state_root(), &results);
let root = receipt.root().unwrap();
let issued_at_unix_ms = 1_788_134_400_000;

// Bind signatures to this index state so they cannot be replayed in another scope.
let receipt_context = AnchorContext::new(AnchorPurpose::Receipt, index.index_state_root());
let signed = issuer.sign_root(receipt_context, root, issued_at_unix_ms);
assert!(verify_root(&issuer.verifying_key, receipt_context, &signed).is_some());

// Batched: authenticate once, then cache the verified root for B proofs.
let anchor = BatchAnchor::build(&[root]).unwrap();
let batch_context = AnchorContext::new(AnchorPurpose::Batch, index.index_state_root());
let signed_batch = issuer.sign_root(batch_context, anchor.root(), issued_at_unix_ms);
let verified_batch = verify_root(&issuer.verifying_key, batch_context, &signed_batch).unwrap();
let proof = anchor.proof_for(0).unwrap();
assert!(BatchAnchor::verify_inclusion(root, &proof, &verified_batch));
```

Measured (n=5,000, dims=128, k=10, release build, mean of 3 warmed runs): per-query signing
costs ≈15.6 µs amortized; batching to 128 queries per signature drops that to ≈1.1 µs/query
(≈14x, 5.8–7.7% of per-query cost). An *uncaching* verifier — one that re-checks the batch
signature on every query instead of once per batch — sees none of that benefit (mean naive
verify cost stays in a ~35,000–41,000ns band across batch sizes), which is the deliberately
checked "does batching game the benchmark" condition. 100% tamper detection
(root/signature/inclusion-proof tamper) across all batch sizes in all 3 runs. Full methodology,
raw output, and the
benchmark-harness bug this run caught and fixed:
[`docs/research/nightly/2026-08-31-signed-retrieval-receipts/README.md`](../../docs/research/nightly/2026-08-31-signed-retrieval-receipts/README.md).

See [`ADR-340`](../../docs/adr/ADR-340-signed-retrieval-receipt-anchoring.md) for the design
rationale, threat model, and rejection criteria.

The signed statement covers a protocol version, purpose, SHA256 public key ID, scope hash,
issuance time, and root. Verification uses strict Ed25519 checks. Batch construction and proof
lookup return recoverable errors for empty or invalid input. A batch inclusion proof cannot be
checked without the authenticated token returned by `verify_root`.
