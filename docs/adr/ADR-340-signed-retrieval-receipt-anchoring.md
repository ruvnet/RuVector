# ADR-340: Signed Retrieval-Receipt Anchoring — Ed25519 Roots, Per-Query and Batched

## Status

Proposed. Experimental crate extension (`ruvector-retrieval-receipt::signing`),
not wired into the default query path of any production index. Layered on
top of ADR-304's unsigned receipts without modifying them.

## Context

ADR-304 (`ruvector-retrieval-receipt`) gave RuVector a tamper-evident
*retrieval* path: `PerResultReceipt` and `MerkleReceipt` commit a query's
top-k result set to a root, so a holder can detect whether the
receipt/result pair they were handed was silently mutated after issuance.
That ADR's own Threat Model, restated in the 2026-08-13 nightly research
README's "Next Research" section, named the gap explicitly:

> Design and benchmark root/head signing (Ed25519 over `index_state_root`
> + receipt root) to close the origin-authentication gap this crate leaves
> open.

An unsigned receipt proves *what* was returned did not change after
issuance. It proves nothing about *who* issued it — any party holding the
same leaves can reproduce the same root, so a receipt alone cannot be
shown to a third party as evidence that a signing key vouched for it. That
is the origin-authentication gap this ADR closes. Binding that key to a
specific engine or organization requires an external key registry,
rotation policy, and revocation history.

This connects three existing ecosystem primitives without inventing a new
one: `ruvector-proof-gate`'s write-side hash chains (ADR-227), ADR-304's
read-side receipts, and the workspace's existing Ed25519 signing pattern
(`ed25519-dalek = "2.1"` with `rand::rngs::OsRng`, already used by
`cognitum-gate-tilezero::permit`, `rvm-checkpoint`, `rvf-crypto`, and
several others) — reused verbatim here rather than introduced fresh.

## Hypothesis

```text
Given a MerkleReceipt root produced for each of a stream of queries
against a RetrievalIndex,

when the root is signed with Ed25519 either (A) individually per query, or
(B) batched — B receipt roots folded into a second Merkle tree whose root
is signed once —

then batched signing (B) should reduce the amortized per-query signing
cost by roughly the batch factor relative to per-query signing (A),

subject to: every tamper of a signed root, a batch signature byte, or an
inclusion-proof sibling remains detected, and the reduction must not be
achieved by silently weakening what an *uncaching* verifier (one that
checks the batch signature fresh on every query, as (A) inherently does)
pays — that per-query cost must stay dominated by one Ed25519 verify
regardless of batch size, or the batching would be gaming the benchmark
rather than amortizing real work.
```

## Decision

Add a `signing` module to the existing `ruvector-retrieval-receipt` crate
(no new crate — the surface area is small and it has no reason to exist
independently of the receipts it signs):

- `Issuer` — an Ed25519 keypair standing in for a query engine's signing
  key. `sign_root(context, root, issued_at_unix_ms)` signs a canonical
  statement containing version, purpose, public-key ID, scope, time, and
  root.
- `verify_root(vk, expected_context, signed) -> Option<VerifiedRoot>` —
  strict, fail-closed signature and context check. The opaque success
  token prevents unauthenticated roots from reaching batch inclusion.
- `BatchAnchor` — a second-level Merkle tree over a batch of `B` receipt
  roots (domain-separated from ADR-304's per-result tree:
  `ruvector:retrieval:batch:leaf:` / `...:node:`, distinct from
  `ruvector:retrieval:leaf:` / `...:node:`), signed once via `Issuer`.
  `proof_for(idx)` / `verify_inclusion(...)` give each query an O(log B)
  membership proof against the signed batch root. `B = 1` degenerates to
  per-query signing (one extra domain-separated hash, negligible next to
  the signature itself).
- `RetrievalReceipt::root() -> Option<[u8; 32]>` — exposes the signable
  root (`chain_head` for `PerResult`, `root` for `Merkle`; `None` for the
  unsigned no-op variant, which has nothing to vouch for).

Both strategies sign a *root the caller already computed* — signing does
not replace the unsigned receipt or its tamper-evidence property; it adds
an origin proof on top, opt-in, at a separately measured cost.

## Threat Model

What signed anchoring adds to ADR-304's threat model, and what it still
does not close:

- **Adds:** origin authentication under a supplied public key. The signed
  statement also prevents cross-purpose and cross-scope replay. It does
  not establish legal non-repudiation or organizational identity without
  durable external key ownership and revocation evidence.
- **Does not add:** issuer honesty. A malicious or compromised issuer can
  sign a false root exactly as validly as a true one; signing binds
  *origin*, not *correctness* of the underlying leaves.
- **Does not add:** real-time availability of a signed anchor. A batch
  signature does not exist — and so cannot be checked — until the batch
  closes. This ADR's benchmark measures only in-process CPU cost; it does
  not model the wall-clock delay of waiting for a batch to fill, which is
  a deployment-specific, separately-measured concern (see Limitations in
  the nightly research README).
- **Does not add:** protection if a verifier does not actually cache the
  batch-signature check. The benchmark's `verify_naive` path shows this
  explicitly: an uncaching verifier's per-query cost stays flat across
  batch sizes, because it re-pays the ~1 Ed25519-verify cost every time
  regardless of `B`.

## Evidence

Measured, `cargo run --release -p ruvector-retrieval-receipt --bin
benchmark` (n=5,000 vectors, dims=128, k=10, 200 queries, 3 repeated
runs after 128 sign/verify warmups; hardware: 12 logical CPUs, rustc
1.94.1, `release` profile):

| batch_size (B) | sign amortized (ns/query) | verify naive (ns/query) | verify cached (ns/query) | sig-verify-once (ns/batch) | proof bytes (sig + worst-case inclusion) | tamper detect |
|---:|---:|---:|---:|---:|---:|---:|
| 1   | ~15,600 | ~36,400 | ~360  | ~34,500 | 170 | 300/300 |
| 8   | ~2,940  | ~34,900 | ~1,430 | ~34,400 | 266 | 450/450 |
| 32  | ~1,340  | ~35,400 | ~2,200 | ~37,800 | 330 | 450/450 |
| 128 | ~1,090  | ~41,200 | ~2,650 | ~42,100 | 394 | 450/450 |

(Means across the 3 repeated runs referenced in the nightly research
README; per-run raw output preserved there.)

- Amortized signing cost at B=128 is ~5.8–7.7% of the B=1 (per-query) cost
  across all 3 runs — comfortably under the 10% threshold fixed before
  the run.
- `verify_naive` means stay within a roughly 35,000–41,000ns band across
  batch sizes (dominated by the O(1) Ed25519 verify on this
  hardware) — confirms batching does not help an uncaching verifier, as
  the hypothesis's Subject-to clause requires.
- `verify_cached` grows from ~0.36µs (B=1, no inclusion proof exists) to
  ~2.65µs (B=128, a 7-level inclusion proof) — small in absolute terms, but
  real: caching the signature check does not make batch membership free.
- Every injected tamper (root-byte flip, signature-byte flip,
  inclusion-proof-sibling flip) was rejected at every batch size across 3
  repeated runs: 100% detection, matching the pre-fixed acceptance
  threshold. (One benchmark-harness bug was found and fixed during this
  run: at B=1 the original root-swap tamper swapped an index with itself,
  a no-op that produced 50/150 false "not detected" results before being
  replaced with a direct byte flip — recorded here because a fabricated
  or unexamined non-100% number would otherwise have blocked promotion
  for the wrong reason.)
- All 23 unit tests pass (15 receipt tests + 8 focused signing tests), `cargo
  clippy --all-targets --release` clean, `cargo fmt --check` clean.

Existing ADR-304 unsigned-receipt numbers (MerkleReceipt generation ≈
18–19µs, 1.7–1.9% of a ~1.02–1.06ms brute-force search) are unchanged by
this ADR — this benchmark run re-confirms them as a regression check, not
a new claim.

## Consequences

**Positive:**

- Closes the specific gap the prior nightly run named, with a measured
  cost rather than an assumed one.
- Batched signing is a genuine, quantified systems tradeoff (throughput
  vs. real-time availability of the anchor), not a strictly-better
  replacement for per-query signing — both remain valid choices depending
  on whether a deployment can tolerate batch-fill latency.
- No changes to `receipt.rs` or `index.rs`; ADR-304's unsigned receipts,
  their tests, and their documented threat model are untouched.

**Negative / costs:**

- Adds `ed25519-dalek` + `rand` as dependencies (both already used
  elsewhere in the workspace at the same versions — no new dependency
  *family*, but this crate previously had none beyond `sha2`).
- A batch signature is a single point of trust for every query in the
  batch: if the issuer's key is compromised mid-batch, all B queries in
  flight are affected, vs. B independent exposures for per-query signing.
  This is a real, not merely theoretical, tradeoff of batching and should
  be weighed against the batch size chosen in any real deployment.
- Still does not solve write-chain membership (ADR-304's named
  future-work item is untouched by this ADR).

## Alternatives Considered

- **Sign every result leaf individually** (not just the root): rejected —
  turns O(1) signing into O(k) per query, with no benefit over signing
  the already-aggregating Merkle root, since the root already
  cryptographically commits every leaf.
- **BLS aggregate signatures** instead of a batch Merkle tree: would allow
  combining B individual per-query signatures into one short aggregate
  signature after the fact (rather than requiring a pre-formed batch),
  avoiding the batch-fill latency this ADR's batching still incurs. Not
  implemented here — it requires a pairing-friendly curve library not
  currently in the workspace dependency set, a larger dependency-surface
  decision this ADR does not make unilaterally. Left as future work (see
  Open Questions).
- **Sign `index_state_root` directly** instead of (or in addition to) the
  receipt root, as literally suggested by the prior nightly run: this ADR
  signs the receipt root, which already binds `index_state_root` as an
  input to every leaf hash (`receipt::result_leaf`) — signing the receipt
  root therefore transitively authenticates the cited index state without
  a second signature. Signing `index_state_root` as an independent,
  periodically-refreshed anchor (decoupled from any specific query) is a
  different and complementary mechanism, not implemented here.

## Implementation Plan

Already implemented in this branch:

1. `crates/ruvector-retrieval-receipt/src/signing.rs` — `Issuer`,
   typed signed statements, strict verification, `BatchAnchor`, and
   security regression tests.
2. `crates/ruvector-retrieval-receipt/src/lib.rs` — `RetrievalReceipt::root()`,
   `pub mod signing`, re-exports.
3. `crates/ruvector-retrieval-receipt/src/bin/benchmark.rs` — signed
   anchoring benchmark section (batch sizes 1/8/32/128), reusing the
   existing `Xorshift64` and percentile helpers.
4. `crates/ruvector-retrieval-receipt/Cargo.toml` — `ed25519-dalek =
   "2.1"` (`rand_core` feature) + `rand = "0.8"`, matching the exact
   versions already used elsewhere in the workspace.

Not implemented (explicitly out of scope for this ADR, tracked as future
work): wiring signing into `RetrievalIndex::search`'s return path,
async/queued batch accumulation with a real wall-clock fill timer, key
rotation, and BLS aggregation.

## API Shape

```rust
use ruvector_retrieval_receipt::{
    query_hash, verify_root, AnchorContext, AnchorPurpose, BatchAnchor, Issuer,
    ReceiptVariant, RetrievalIndex, RetrievalReceipt,
};

let issuer = Issuer::generate();
let index = RetrievalIndex::ingest(5_000, 128, 0xC0FF_EE01);
let root_state = index.index_state_root();

// Per-query (batch of 1):
let query = vec![0.1; 128];
let results = index.search(&query, 10);
let qh = query_hash(&query);
let receipt = RetrievalReceipt::build(ReceiptVariant::Merkle, qh, root_state, &results);
let root = receipt.root().unwrap();
let issued_at_unix_ms = 1_788_134_400_000;
let receipt_context = AnchorContext::new(AnchorPurpose::Receipt, root_state);
let signed = issuer.sign_root(receipt_context, root, issued_at_unix_ms);
assert!(verify_root(&issuer.verifying_key, receipt_context, &signed).is_some());

// Batched (B receipt roots, one signature):
let roots: Vec<[u8; 32]> = vec![root /* , ... more query roots ... */];
let anchor = BatchAnchor::build(&roots).unwrap();
let batch_context = AnchorContext::new(AnchorPurpose::Batch, root_state);
let signed_batch = issuer.sign_root(batch_context, anchor.root(), issued_at_unix_ms);
let verified_batch = verify_root(&issuer.verifying_key, batch_context, &signed_batch).unwrap();
let proof = anchor.proof_for(0).unwrap();
assert!(BatchAnchor::verify_inclusion(roots[0], &proof, &verified_batch));
```

## Feature Flags

None. `signing` is a public module of the existing crate, compiled
unconditionally — the crate is already experimental and unwired from any
production path (per ADR-304's Status), so there is no default-path
regression risk to gate behind a flag.

## Benchmark Evidence

See Evidence above and the full methodology, raw 3-run output, and
Limitations in
[`docs/research/nightly/2026-08-31-signed-retrieval-receipts/README.md`](../research/nightly/2026-08-31-signed-retrieval-receipts/README.md).

## Security

- Uses `ed25519-dalek 2.1` (a maintained, widely-used implementation
  already present in this workspace at the same pinned version) rather
  than a hand-rolled signature scheme.
- Domain separation: batch-tree hashing (`ruvector:retrieval:batch:*`) is
  namespaced apart from ADR-304's per-result tree (`ruvector:retrieval:*`)
  so a batch leaf and a per-result leaf can never collide even if a
  32-byte value happened to appear in both trees.
- Signed statements domain-separate and bind version, purpose, SHA256
  public-key ID, deployment scope, issuance time, and root. Verification
  uses `ed25519-dalek::VerifyingKey::verify_strict`.
- Fail-closed verification throughout: `verify_root` returns `None` on
  any mismatch. Batch construction and proof lookup return recoverable
  errors for empty or out-of-range input. `verify_inclusion` accepts only
  the opaque token produced by a successful signature check.
- Key management (generation, storage, rotation) is out of scope for this
  crate, as for every other signing primitive in the workspace — `Issuer`
  is a thin wrapper for benchmarking and API demonstration, not a
  production KMS integration.

## Governance

Same governance posture as ADR-304: experimental, not on any default
query path, requires an explicit promotion decision (with its own
benchmark evidence against a target deployment) before any production
index adopts it. This ADR does not, by itself, authorize wiring signing
into a live query path.

## Failure Modes

- **Issuer key compromise:** every root signed under the compromised key
  is no longer trustworthy for origin authentication, retroactively, for as
  long as the key was in use. Batching increases blast radius per
  incident (one key covers B queries per signature) but does not change
  the total exposure across the key's lifetime.
- **Batch never closes:** in a real streaming deployment (not modeled by
  this in-process benchmark), a batch that never fills never produces a
  signed anchor — a liveness failure mode a production implementation
  must handle with a fill-timeout, not addressed here.
- **Verifier does not cache:** as shown by `verify_naive`, an
  implementation that re-verifies the batch signature per query gets none
  of batching's throughput benefit and should use per-query signing
  (B=1) instead — a real operational footgun this ADR surfaces rather
  than hides.

## Migration

None — this is a new, additive, opt-in module on an already-experimental,
unwired crate. No existing caller's behavior changes.

## Rollback

Remove the `signing` module and its two `Cargo.toml` dependencies; no
other code in the workspace depends on it as of this ADR.

## Rejection Criteria

This ADR's mechanism is falsified — and should not be promoted toward
production — if any of:

- Amortized signing cost at a target production batch size does not
  clear the amortization threshold on production-representative hardware
  (this ADR's 10% threshold at B=128 held on 4-core commodity hardware;
  a slower or more contended verifier host could differ).
- A caching verifier's amortized per-query authentication cost (sign +
  verify combined) exceeds the cost of simply re-running the query on
  trusted infrastructure — at which point the receipt provides no
  practical efficiency advantage over re-computation, only an audit
  advantage.
- Real deployment measurement shows batch-fill latency (not modeled here)
  dominates end-to-end receipt-availability time at the batch sizes that
  clear the amortization threshold, making the throughput win illusory in
  practice.

## Open Questions

- Does a production deployment's actual query arrival rate make
  wall-clock batch-fill latency acceptable at B=32–128, or does it force
  smaller batches that erode the amortization win measured here?
- Is BLS aggregation (Alternatives Considered) worth the added
  pairing-curve dependency for deployments that need per-query signing
  latency *and* per-query signature-verify amortization simultaneously?
- Should `index_state_root` get its own independent, periodically-signed
  anchor (decoupled from any query), complementary to per-receipt
  signing, for auditors who want to verify index state without holding
  any specific query's receipt?
