# ADR-341: Independent, Periodic `index_state_root` Anchoring

## Status

Proposed. Experimental crate extension
(`ruvector-retrieval-receipt::state_anchor`), not wired into the default
write or query path of any production index. Layered on top of ADR-340's
signed receipt roots without modifying them, and reuses ADR-340's
`Issuer`/`AnchorContext`/`verify_root` machinery via a new
`AnchorPurpose::StateAnchor`.

## Context

ADR-304 gave RuVector tamper-evident retrieval receipts. ADR-340 added
Ed25519 signing of receipt roots, closing the origin-authentication gap —
but every signature ADR-340 produces is still tied to a specific query.
ADR-340's own Open Questions and this repository's 2026-08-31 nightly
research README named the remaining gap explicitly:

> Should `index_state_root` get its own independent, periodically-signed
> anchor (decoupled from any query), complementary to per-receipt signing,
> for auditors who want to verify index state without holding any specific
> query's receipt?

An auditor who wants to confirm "the index was in state `R` at some
attested point" today has exactly two options: hold a specific signed
receipt that happens to cite `R` (ADR-340), or replay the entire write
history via `HashChainGate::verify_integrity` — O(n) in the number of
writes, and only possible with access to that full history. Neither serves
an auditor who has *no* receipt and does not want to pay for full replay.
This ADR adds a third option: an independently, periodically signed
checkpoint of `index_state_root` itself.

## Hypothesis

```text
Given a HashChainGate-backed index accumulating N writes, whose full-history
integrity check (verify_integrity) costs O(N) hash re-derivations,

when the index_state_root is anchored (signed) independently of any query,
either (A) on every write (interval_writes = 1) or (B) periodically every W
writes (interval_writes = W),

then the number of signing operations required to cover N writes drops by
approximately the factor W under policy B relative to policy A, enabling an
external auditor who holds only the anchor log (no full write history, no
query receipts) to authenticate the index's state at a bounded number of
checkpoints in O(1) per checkpoint,

subject to: every tampered anchor (claimed-root corruption, signature-byte
flip) remains detected at every interval W; the maximum staleness (writes
since the last anchor an auditor can pin the state to) never exceeds W-1,
measured exactly rather than merely asserted; and O(1) anchor verification
must not silently substitute for O(N) full-history integrity checking — this
ADR's benchmark also measures verify_integrity's O(N) cost so the tradeoff
is disclosed, not hidden.
```

Acceptance thresholds, fixed before this run:

1. Anchor count at every tested interval must equal `⌊N / interval_writes⌋`
   exactly — a structural correctness property, not a fuzzy threshold.
2. Maximum observed staleness at every interval `W > 1` must equal exactly
   `W - 1`.
3. Every injected tamper (claimed-root byte flip, signature-byte flip) must
   be detected at every interval: **100%**.
4. O(1) anchor-verify cost must stay within a **2x** band across all tested
   intervals — a large drop would mean the benchmark accidentally amortizes
   something the design claims it does not.
5. Amortized signing cost at the largest tested interval (`W=512`) must drop
   below **10%** of the `W=1` (per-write) cost.

## Decision

Add `crates/ruvector-retrieval-receipt/src/state_anchor.rs`:

- `StateAnchorPolicy::new(interval_writes)` — fails closed (`Result`, not
  panic) on a zero interval, since interval is caller-supplied input to this
  crate's public API.
- `StateAnchorLog` — an append-only, in-process log. `observe_write` is
  called after every admitted write with the write gate's current
  `chain_root()` and `len()`; it signs and appends a `StateAnchor` only when
  `write_count` lands on an interval boundary. `latest_at_or_before` and
  `staleness_at` answer the auditor-facing "how stale is my view" query.
- `verify_state_anchor` — O(1) verification of one `StateAnchor` against a
  claimed root, a public key, and a scope, with no dependency on any query
  receipt or the write history.
- A new `AnchorPurpose::StateAnchor = 3` variant, reusing ADR-340's typed,
  domain-separated signing machinery unchanged — no new signature format, no
  new dependency.

`StateAnchorLog` operates directly over `[u8; 32]` roots and write counts,
not over `RetrievalIndex`: state anchoring is a write-path concept (the
`index_state_root` a `ruvector_proof_gate::WriteGate` produces), independent
of any retrieval index or query, matching the module's decoupling thesis.

## Evidence

- **Command:** `cargo run --release -p ruvector-retrieval-receipt --bin
  benchmark -- 5000 128 10 200` (n=5,000 writes for the interval sweep;
  n∈{625, 1,250, 2,500, 5,000, 10,000} for the descriptive full-replay-cost
  comparison).
- **Hardware:** 4 logical CPUs, rustc 1.94.1 / cargo 1.94.1, release
  profile.
- **Repetitions:** 3 full process runs; raw output preserved unedited in
  `docs/research/nightly/2026-09-03-state-root-anchoring/raw-runs.txt`.
- **Result (representative run):**

  | interval_writes | anchors_taken | expected | sign_amort_ns | max_staleness | anchor_verify_ns | tamper (2 kinds × 40) |
  |---|---|---|---|---|---|---|
  | 1   | 5,000 | 5,000 | 17,445.5 | 0   | 46,575 | 80/80 |
  | 8   | 625   | 625   | 2,126.8  | 7   | 45,317 | 80/80 |
  | 32  | 156   | 156   | 562.5    | 31  | 45,953 | 80/80 |
  | 128 | 39    | 39    | 141.1    | 127 | 43,996 | 80/80 |
  | 512 | 9     | 9     | 32.7     | 511 | 46,756 | 80/80 |

  All five acceptance thresholds passed in every one of 3 runs. Anchor
  count and max staleness matched the theoretical formula **exactly** in
  every run (not merely within tolerance) — these are structural
  correctness checks, not statistical ones. Amortized signing cost at
  `W=512` was 0.2% of the `W=1` cost (threshold: <10%). Anchor-verify cost
  stayed within a 1.04–1.07x band across intervals in every run (threshold:
  <2x).
- **Descriptive comparison (not gated):** at n=10,000,
  `HashChainGate::verify_integrity` (full re-derivation) cost ~1.4ms versus
  a flat ~46–50μs for `verify_state_anchor` — **28–31x** cheaper at this
  scale across the 3 runs, and the gap widens with n since `verify_integrity`
  is O(n) while anchor verification is O(1). See the nightly research
  README for the full table.

## Consequences

- An auditor who trusts the signer's key can now check "was the index ever
  attested to be in state `R`" in O(1), without holding any query receipt
  and without the full write history — a capability that did not exist
  after ADR-340 alone.
- This does not replace `verify_integrity`: an auditor who needs
  zero-staleness, full-history integrity (not just periodic checkpoints)
  still pays the O(n) cost. `interval_writes` is a policy choice a
  deployment must make explicitly; this ADR does not pick a default.
- One more signing key-management surface: `StateAnchor`s must be signed by
  a key an auditor is willing to trust, same caveat as ADR-340's `Issuer`.
- `StateAnchorLog` is in-process and non-durable by design in this
  experimental crate; a production deployment must persist each
  `StateAnchor` as it is produced (see ruFlo Integration below) or the
  anchor history is lost on restart.

## Alternatives Considered

- **Sign every write (`interval_writes = 1`) unconditionally.** Zero
  staleness, but the amortization benefit measured here (98–99.8% signing
  cost reduction at `W ≥ 8`) is exactly what this ADR exists to make
  available as a policy choice, not to rule out.
- **Time-based anchoring interval** (e.g. "anchor every 60 seconds")
  instead of write-count-based. Write-count-based was chosen because it
  gives an exact, verifiable staleness bound (`W - 1` writes) independent of
  write-arrival rate; a time-based policy's staleness bound would depend on
  an assumed write-rate ceiling, which is a different and weaker guarantee.
  A wall-clock hybrid (anchor at `min(W writes, T elapsed)`) is plausible
  future work, not implemented here.
- **Derive the state anchor from `MerkleGate`'s MMR instead of
  `HashChainGate`.** `HashChainGate::chain_root()` is what
  `RetrievalIndex::index_state_root()` already uses (see `index.rs`);
  reusing it keeps this ADR's benchmark comparable to ADR-304/ADR-340's
  existing numbers rather than introducing a second write-chain variant into
  the comparison.

## Implementation Plan

Already implemented in this branch:

1. `crates/ruvector-retrieval-receipt/src/signing.rs` — `AnchorPurpose::StateAnchor`
   variant, `AnchorError::InvalidInterval`.
2. `crates/ruvector-retrieval-receipt/src/state_anchor.rs` — `StateAnchorPolicy`,
   `StateAnchor`, `StateAnchorLog`, `verify_state_anchor`, 6 unit tests.
3. `crates/ruvector-retrieval-receipt/src/lib.rs` — `pub mod state_anchor`, re-exports.
4. `crates/ruvector-retrieval-receipt/src/bin/benchmark.rs` — interval-sweep
   benchmark section plus a descriptive, non-gated `verify_integrity`
   scaling comparison.

## API Shape

```rust
let policy = StateAnchorPolicy::new(32)?; // anchor every 32 writes
let mut log = StateAnchorLog::new(policy);
let mut gate = HashChainGate::new();

// After every admitted write:
gate.admit(&payload)?;
let anchor = log.observe_write(
    &issuer, scope_hash, gate.chain_root(), gate.len() as u64, now_unix_ms,
); // Some(StateAnchor) only on an interval boundary

// An auditor holding only (public key, scope, one StateAnchor):
let verified = verify_state_anchor(&issuer.verifying_key, scope_hash, claimed_root, &anchor);
assert!(verified.is_some());

// How stale is my view relative to the latest anchor?
let staleness = log.staleness_at(gate.len() as u64); // bounded by interval_writes - 1
```

## Feature Flags

None. Additive module in an already-experimental crate; no default-path
wiring.

## Benchmark Evidence

See Evidence above and `docs/research/nightly/2026-09-03-state-root-anchoring/`
(README, gist, raw-runs.txt) for the full methodology, complete tables, and
all 3 raw runs.

## Security

- Reuses ADR-340's typed statement (`RootStatement`) and domain-separated
  signing unchanged — no new signature format. `AnchorPurpose::StateAnchor`
  is bound into the signed statement, so a receipt- or batch-purpose
  signature cannot be replayed as a state anchor and vice versa (tested:
  `state_anchor_purpose_is_isolated_from_receipt_and_batch`).
- **Adds:** an O(1)-verifiable checkpoint that `index_state_root` was
  attested at a specific write count, independent of any query.
- **Does not add:** issuer honesty (identical caveat to ADR-340) — a
  malicious issuer signs a false root exactly as validly as a true one.
- **Does not add:** protection against a stalled or dishonest anchoring
  job — if the periodic job stops running, `staleness_at` for writes past
  the last real anchor grows unbounded; nothing in this crate detects that
  automatically (see Failure Modes in the nightly README and Next Research
  below).
- **Does not add:** durability. `StateAnchorLog` is in-process; see
  Consequences.

## Governance

Experimental, matching ADR-304 and ADR-340's posture: not on any default
write path, no production index adopts it as a result of this ADR alone. A
promotion decision requires a durable-storage design for `StateAnchor`s and
benchmark evidence against a target deployment's actual write-rate
characteristics, not just this synthetic workload.

## Failure Modes

- A stalled anchoring job silently grows staleness past the declared bound;
  this ADR does not add monitoring for that (see Next Research).
- `StateAnchorLog` is not thread-safe and not persisted; a crash between
  `observe_write` calls loses in-flight anchor state (the underlying
  `HashChainGate` state is unaffected — only unpersisted anchors are lost).
- A verifier who does not independently know `scope_hash` cannot detect a
  cross-deployment replay of a validly signed anchor from a different scope
  under the same key; scope must be established out of band, same as
  ADR-340.

## Migration

None — purely additive. No existing type's field or method signature
changed.

## Rollback

Delete `state_anchor.rs`, its `lib.rs` wiring, and the benchmark section;
revert the `AnchorPurpose`/`AnchorError` additions in `signing.rs`. No
other crate depends on this module.

## Rejection Criteria

Would have been rejected (per this ADR's own fixed thresholds) had any of:
anchor count not matching `⌊N/W⌋` exactly, max staleness exceeding `W - 1`,
any tamper trial going undetected, or anchor-verify cost varying by more
than 2x across intervals. None occurred in any of 3 runs.

## Open Questions

- Should a `StateAnchorLog` implementation detect and surface a stalled
  anchoring job itself (e.g. an explicit "no anchor within the last X
  writes" check), or is that purely an external monitoring concern?
- Is a write-count-based interval the right primitive for a production
  deployment with bursty or intermittent write traffic, or does the
  time-based hybrid noted under Alternatives Considered matter enough to
  implement and benchmark?
- Should `StateAnchor`s be foldable into a Merkle structure themselves (an
  "anchor of anchors"), so an auditor holding only the *latest* anchor can
  verify inclusion of any earlier one without trusting a durable log
  externally? This would parallel ADR-340's `BatchAnchor` but over anchors
  instead of receipt roots.

## References

- ADR-304 (`docs/adr/ADR-304-retrieval-receipts.md`).
- ADR-340 (`docs/adr/ADR-340-signed-retrieval-receipt-anchoring.md`), whose
  Open Questions and the 2026-08-31 nightly research README's Next Research
  item #4 are the direct origin of this ADR's hypothesis.
- `ruvector-proof-gate` source (`HashChainGate::verify_integrity`,
  `chain_root`), in-repo.
- `crates/ruvector-retrieval-receipt/src/state_anchor.rs`, this ADR's
  implementation.
