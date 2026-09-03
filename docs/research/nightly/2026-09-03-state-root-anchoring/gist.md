# Periodic Index-State-Root Anchoring: An O(1) Audit Checkpoint Decoupled From Any Query

## Problem

RuVector's `ruvector-retrieval-receipt` crate gives an auditor two ways to
authenticate a vector index's state today. First, ADR-340's signed receipt
roots: an Ed25519 signature over a specific query's result-commitment root,
which transitively authenticates the `index_state_root` cited by that
query. Second, `ruvector-proof-gate`'s `HashChainGate::verify_integrity`:
replay the entire write history from genesis and confirm every commitment
re-derives correctly.

Both have a real gap. The first only helps an auditor who happens to be
holding a specific query's receipt — no receipt, no authentication. The
second works with no receipt, but costs O(n) in the number of writes and
requires access to the full write history, which an external auditor often
doesn't have.

ADR-340's own Open Questions named the missing third option explicitly:
sign `index_state_root` itself, independently, on a schedule decoupled
from any query.

## Hypothesis

```text
Given a HashChainGate-backed index accumulating N writes, whose full-history
integrity check costs O(N),

when index_state_root is signed independently of any query, either (A) on
every write or (B) periodically every W writes,

then policy B should reduce signing operations by roughly the factor W,
giving an auditor an O(1)-verifiable checkpoint without full replay,

subject to: every tamper stays detected at every W, staleness never
exceeds W-1 (measured exactly), and the O(1) checkpoint is never presented
as a replacement for O(N) full-history verification.
```

## Technical Design

`crates/ruvector-retrieval-receipt/src/state_anchor.rs` adds:

- `StateAnchorPolicy::new(interval_writes)` — a `Result`-returning
  constructor (fails closed on `interval_writes == 0`, matching the crate's
  existing panic-free-on-untrusted-input convention).
- `StateAnchorLog::observe_write(issuer, scope, root, write_count, ts)` —
  called after every write; signs and records a `StateAnchor` only when
  `write_count` lands on an interval boundary.
- `StateAnchorLog::staleness_at(write_count)` — writes since the nearest
  anchor, the number a monitor would alert on if it grows past the policy
  bound.
- `verify_state_anchor(pubkey, scope, claimed_root, anchor)` — O(1)
  verification, no receipt or write history required.

It reuses ADR-340's signing primitives unchanged via a new third purpose,
`AnchorPurpose::StateAnchor`, domain-bound into the signed statement exactly
like the existing `Receipt` and `Batch` purposes — so a signature produced
for one purpose can never be replayed as another.

```rust
let policy = StateAnchorPolicy::new(32)?; // anchor every 32 writes
let mut log = StateAnchorLog::new(policy);

gate.admit(&payload)?;
let anchor = log.observe_write(
    &issuer, scope_hash, gate.chain_root(), gate.len() as u64, now_ms,
); // Some(StateAnchor) only on a boundary

// An auditor with no receipt, no write history — just (pubkey, scope, anchor):
assert!(verify_state_anchor(&issuer.verifying_key, scope_hash, claimed_root, &anchor).is_some());
```

Operating directly over `[u8; 32]` roots and write counts (not tied to
`RetrievalIndex`) keeps the module's decoupling honest: this is a
write-path primitive over whatever `ruvector_proof_gate::WriteGate`
produces, independent of any retrieval index or query.

## Benchmark Evidence

`cargo run --release -p ruvector-retrieval-receipt --bin benchmark -- 5000
128 10 200`, 3 repeated runs, 4 logical CPUs, rustc 1.94.1, release
profile. Representative run:

| interval_writes | anchors_taken | expected | sign_amortized_ns | max_staleness | anchor_verify_ns | tamper (2×40) |
|---|---|---|---|---|---|---|
| 1   | 5,000 | 5,000 | 17,445.5 | 0   | 46,575 | 80/80 |
| 8   | 625   | 625   | 2,126.8  | 7   | 45,317 | 80/80 |
| 32  | 156   | 156   | 562.5    | 31  | 45,953 | 80/80 |
| 128 | 39    | 39    | 141.1    | 127 | 43,996 | 80/80 |
| 512 | 9     | 9     | 32.7     | 511 | 46,756 | 80/80 |

Five acceptance criteria, all fixed before the run, all passed in all 3
runs: anchor count matches `⌊N/W⌋` **exactly**; max staleness matches
`W-1` **exactly**; 100% tamper detection (400 trials/run); anchor-verify
cost within a 2x band across intervals (observed 1.04–1.07x); amortized
signing cost at `W=512` under 10% of the `W=1` cost (observed 0.2%).
**ACCEPT** in all 3 runs.

The part worth being honest about: this does not replace full-history
integrity checking. At n=10,000, `verify_integrity`'s O(n) full
re-derivation cost was 28–31x more expensive than one O(1)
`verify_state_anchor` call across the 3 runs — real, and the gap grows
with n — but an auditor who needs zero-staleness proof over the *entire*
history, not just a periodic checkpoint, still pays that O(n) cost. This
benchmark reports that cost explicitly (as a descriptive, non-gated table)
rather than letting the O(1) number stand in for it.

## Limitations

- Uniform synthetic write rate — no bursty traffic tested. A time-based
  (or hybrid) anchoring policy is plausible future work, not implemented:
  a wall-clock interval's staleness bound depends on an assumed write-rate
  ceiling, a workload-dependent guarantee weaker than the write-count-based
  exact bound measured here.
- `StateAnchorLog` is in-process, non-durable, and not thread-safe in this
  experimental crate — a production deployment needs a persistence design
  before this is more than a research prototype.
- No wall-clock anchor-fill latency measurement (an anchor at `W=512`
  doesn't exist until write 512 lands) — same disclosed limitation as
  ADR-340's own batch-fill-latency gap.
- No WASM binary-size or on-device measurement.

## Production Relevance

An auditor — a compliance reviewer, another agent, a backup-integrity
check — that wants to confirm "was this index ever attested to be in state
R" no longer has to choose between holding a specific query's receipt or
paying for a full write-history replay. A durably persisted anchor log (not
yet implemented) plus a periodic anchoring job would give that answer in
O(1), at a signing cost that amortizes by roughly the chosen interval and
a staleness bound that's exact and disclosed up front — not a free
capability, a real, now-measured tradeoff a deployment can choose.

## RuVector Ecosystem Implications

This is the fourth run in an unbroken thread: ADR-304 (unsigned receipts)
→ ADR-340 (signed receipt roots) → this run, ADR-341 (independent periodic
state anchoring) — each nightly run finishing the previous one's named
open question rather than starting a new island. It connects
`ruvector-proof-gate` (the anchored root's source), `ruvector-retrieval-
receipt` (the crate housing all three anchor types now), and the
workspace's shared Ed25519 signing pattern, unchanged. The concrete next
integration point is a ruFlo periodic-anchoring-plus-staleness-alert
workflow — not implemented here, since no ruFlo workflow-definition surface
for this repository was found during capability verification, but the
mechanism this run built is exactly what such a workflow would call.

## Future Direction

1. Durable `StateAnchorLog` persistence and its own benchmark.
2. Time-based/hybrid anchoring policy under a bursty write-rate workload.
3. "Anchor of anchors" — fold `StateAnchor`s into their own Merkle
   structure so a verifier holding only the latest anchor can verify
   inclusion of an earlier one, parallel to ADR-340's `BatchAnchor`.
4. Automatic staleness-alerting, implemented and benchmarked for detection
   latency.
5. WASM binary-size and on-device latency measurement.

## References

- `crates/ruvector-retrieval-receipt/src/state_anchor.rs` (this repo, new).
- ADR-304 (`docs/adr/ADR-304-retrieval-receipts.md`).
- ADR-340 (`docs/adr/ADR-340-signed-retrieval-receipt-anchoring.md`).
- ADR-341 (`docs/adr/ADR-341-periodic-state-root-anchoring.md`), this
  run's full design record.
- Full research README and raw benchmark output:
  `docs/research/nightly/2026-09-03-state-root-anchoring/`.
