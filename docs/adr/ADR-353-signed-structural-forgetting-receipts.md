# ADR-353: Signed Structural Forgetting Receipts

## Status

Accepted as an opt-in capability, at the same experimental tier as its two
dependencies. New module (`ruvector-agent-memory::structural_witness`,
feature-gated behind `mincut-forget`, off by default) plus a non-breaking
refactor of `graph_forget::MincutGatedForgetting` (extraction of a new
public `decide()` method; `CompactionPolicy::select_survivors`'s behavior is
unchanged). **This ADR does not alter ADR-345's status.**
`MincutGatedForgetting` itself remains "rejected for production use as
designed" per ADR-345's own measured evidence (0.0pp bridge-survival gap at
the tested corpus scale); this ADR only makes whatever decision that policy
produces auditable and tamper-evident, and does not re-evaluate the
decision's quality.

## Context

Three prior, independently-evidenced pieces of `ruvector-agent-memory` have
never been composed into one payload worth signing:

1. **ADR-345** (`graph_forget::MincutGatedForgetting`) computes a
   mincut-boundary signal — which evicted memories were structurally
   load-bearing — but discards it: `select_survivors` uses the signal
   internally and returns only a plain index list.
2. **ADR-345** (`witnessed_compaction::compact_witnessed`) certifies *which*
   ids were evicted with a chained `LedgerWitnessRecord`, but its `aux`
   field (an existing, already-hashed 64-bit slot in the ADR-134 record
   shape) is always `0` — the eviction witness carries no explanation.
3. **ADR-347** (`witness_signing::SignedWitnessSink`) can Ed25519-sign any
   `WitnessSink` payload, including eviction witnesses (composition is
   already free — any `WitnessSink` implementor works), but nothing has
   signed a *structural* payload specifically; the 2026-09-16 nightly's Next
   Research item 6 named this composition as an open follow-up.

Composing (1)-(3) as a no-op (sign the existing all-zero `aux`) would
satisfy that follow-up's letter but add nothing worth signing. This ADR
instead defines the payload first.

## Hypothesis

```text
Given MincutGatedForgetting's eviction witness records carry no structural
justification (aux = 0) today, and SignedWitnessSink can sign any
WitnessSink payload but has never signed a structural one,

when evictions are annotated via a new compact_witnessed_structural
function that packs the mincut-boundary flag, protection mode, and scalar
importance score into aux -- computing the mincut partition exactly once
per decision via a new MincutGatedForgetting::decide method, identical cost
to today's plain compact_witnessed on the same policy -- and optionally
routed through the existing, unmodified SignedWitnessSink,

then: (1) an auditor running audit_structural_evictions over only the
resulting witness records (no vectors, no MemoryStore) recovers the exact
ground-truth boundary-eviction set in every trial; (2) the structural-
encoding overhead versus plain compact_witnessed on the identical
MincutGatedForgetting decision stays within 10% mean wall-clock; (3) a
naive (non-recomputing) tamper of the boundary bit is caught by the
existing unsigned MemoryWitnessLog::verify_chain in every trial; (4) a
diligent (fully forward-recomputed) tamper of the same bit is NOT caught by
the unsigned walk but IS rejected by verify_signed_chain in every trial,

subject to: zero false negatives/positives in gates 1, 3, and 4 across the
full trial matrix.
```

## Decision

1. Add `MincutGatedForgetting::decide(entries, target_size, context) ->
   ForgetDecision { survivors, boundary, scalar }`, extracting
   `select_survivors`'s existing body unchanged; `select_survivors` becomes
   `self.decide(..).survivors`. This guarantees `survivors` and `boundary`
   come from the *same* `partition()` call — calling a hypothetical separate
   boundary accessor a second time would risk disagreeing with the first
   call, since `partition()` is documented (2026-09-05/09-11 nightlies) as
   non-deterministic across repeated calls on identical input.
2. Add `structural_witness::{encode_structural_aux, decode_structural_aux}`:
   bit 0 = mincut-boundary flag, bit 1 = `ForgetMode::Hard` flag, bits 8-39 =
   `f32::to_bits` of the evicted entry's scalar `weighted_importance` score,
   bits 2-7 and 40-63 reserved (zero).
3. Add `structural_witness::compact_witnessed_structural(store, decision:
   &ForgetDecision, hard_mode: bool, actor_id, now_ns, chain, sink)`,
   field-for-field identical to `witnessed_compaction::compact_witnessed`
   (ADR-345) except `aux` carries the encoding from (2) instead of `0`. Same
   witness-first ordering, same `EvictionWitnessChain` cursor type, same
   `WitnessSink` trait object parameter (so `SignedWitnessSink` composes
   with zero additional code, as it already did with `compact_witnessed`).
   **It takes the decision, not a policy** — it never calls `decide()`
   itself. An earlier version of this design took `policy:
   &MincutGatedForgetting` and called `decide()` internally; that version's
   own unit test, asserting a hand-guessed ground-truth boundary set
   against a *separately* computed `decide()` call, failed during this run
   against `partition()`'s documented non-determinism (two independently
   "correct" minimum cuts for the test's constructed graph disagreed on
   which gateway vertex sat on the boundary). The fix was this API shape,
   not a looser assertion: requiring the caller to compute `decide()`
   exactly once and pass the result to every function that needs it (both
   production code and tests) removes the second call entirely, rather than
   tolerating its variance.
4. Add `structural_witness::is_hard_mode(mode: ForgetMode) -> bool`, a
   trivial helper for callers that have a `ForgetMode` but not a full
   policy value.
5. Add `structural_witness::audit_structural_evictions(records) ->
   StructuralAuditReport { evicted, boundary_evicted, hard_mode_evicted }` —
   a pure function over `&[LedgerWitnessRecord]`, no vectors or
   `MemoryStore` required. Performs no verification itself; callers who need
   trustworthiness, not just internal consistency, must verify the records
   first (`verify_signed_chain` or at minimum `verify_chain`).
6. No new Cargo dependency, no modification to `witness_signing.rs`,
   `witnessed_compaction.rs`, or `ruvector-mincut`. Not feature-gated beyond
   inheriting `mincut-forget` (it depends on `graph_forget`, which is
   already gated).

## Evidence

One `cargo run --release -p ruvector-agent-memory --example
structural_witness_bench --features mincut-forget` run (84-memory corpus, 6
clusters x 12 + 12 bridges, 32-dim, compacted to 42; identical to the
2026-09-05 nightly's corpus and `mincut_trials = 1`):

```
baseline(compact_witnessed)                     71347.5us mean  (1427.686ms / 20 trials)
candidate_A(structural, unsigned)               58844.4us mean  (1177.396ms / 20 trials)
candidate_B(structural, signed BatchTail16)     65374.4us mean  (1308.099ms / 20 trials)

Structural-encoding overhead (candidate_A vs baseline): -17.52% (gate: <= 10%) -- see caveat below
Audit-accuracy: 20/20 trials exactly recovered ground truth
Naive-tamper detection: 10/10 caught by the unsigned chain walk
Diligent-forgery rejection: 10/10 (unsigned walk fooled AND signed chain rejects)
```

**Overhead-measurement caveat.** The -17.52% figure means candidate_A's
mean was *lower* than baseline's in this run, not that annotating records
is faster. Both variants independently pay for one
`RuVectorGraphAnalyzer::partition()` call per trial, whose own cost
(documented elsewhere as tens of milliseconds and non-deterministic
call-to-call) dwarfs and swamps the actual bit-packing cost this gate
targets. The gate is reported as PASS because the true marginal cost is
almost certainly negligible relative to this noise floor, not because this
measurement precisely isolated it — a tighter isolation is listed as Next
Research rather than attempted here.

`cargo test -p ruvector-agent-memory --features mincut-forget --lib`
(full crate suite, all features): **56 passed; 0 failed; 0 ignored**
(353.15s). The filtered `--lib structural_witness` run (this module only):
**8 passed; 0 failed; 0 ignored** (211.73s, debug build —
`RuVectorGraphAnalyzer::partition()` dominates wall-clock here).

Full methodology, raw output, and the audit-accuracy / forgery-detection
trial matrices: `docs/research/nightly/2026-10-12_signed-structural-forgetting-receipts/README.md`.

**Verdict: ACCEPT.** All four pre-registered mandatory gates pass on real,
`--release`-measured evidence; the overhead gate's caveat is stated
plainly rather than presented as a precise measurement.

## Consequences

- Eviction witnesses produced by `MincutGatedForgetting` can now carry a
  real, auditable, signable justification at effectively zero marginal
  cost over the mincut computation the policy already pays for.
- Purely additive: `compact_witnessed`, `MemoryWitnessLog`,
  `SignedWitnessSink`, and every other existing caller/type are untouched.
  `compact_witnessed_structural` is a new, separate function; nothing is
  removed or renamed.
- Does **not** change `MincutGatedForgetting`'s production recommendation
  (still "experimental, not promoted," per ADR-345).

## Alternatives Considered

- **A new `action_kind` or wider record format for structural evictions.**
  Rejected: breaks ADR-134's fixed 64-byte shape for no benefit the
  existing `aux` field doesn't already provide.
- **Reuse the existing private `boundary_indices` from a second call site
  instead of adding `decide()`.** Rejected: pays the expensive,
  non-deterministic `partition()` cost twice per decision and risks the two
  calls disagreeing with each other.
- **A separate signed field for the raw `f32` score instead of packing via
  `to_bits` into `aux`.** Rejected: would need an ADR-134 shape change;
  `to_bits`/`from_bits` round-trips exactly for the finite scores this
  record shape ever contains.
- **Store a full crossing-edge list instead of one boundary bit.** Deferred:
  does not fit in the remaining `aux` bits for any non-trivial corpus; a
  single bit answers the audit question this ADR targets.

## API Shape

```rust
// graph_forget.rs (refactor; feature "mincut-forget")
pub struct ForgetDecision {
    pub survivors: Vec<usize>,
    pub boundary: HashSet<usize>,
    pub scalar: Vec<f32>,
}
impl MincutGatedForgetting {
    pub fn decide(&self, entries: &[MemoryEntry], target_size: usize,
        context: &[Vec<f32>]) -> ForgetDecision;
}

// structural_witness.rs (new; feature "mincut-forget")
pub const AUX_BOUNDARY_BIT: u64 = 1 << 0;
pub const AUX_HARD_MODE_BIT: u64 = 1 << 1;
pub const AUX_SCORE_SHIFT: u32 = 8;
pub fn encode_structural_aux(boundary: bool, hard_mode: bool, score: f32) -> u64;
pub fn decode_structural_aux(aux: u64) -> (bool, bool, f32);

pub fn is_hard_mode(mode: ForgetMode) -> bool;

pub fn compact_witnessed_structural(
    store: &mut MemoryStore,
    decision: &ForgetDecision,
    hard_mode: bool,
    actor_id: &str,
    now_ns: u64,
    chain: &mut EvictionWitnessChain,
    sink: &mut dyn WitnessSink,
) -> Result<Vec<LedgerWitnessRecord>, LedgerError>;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct StructuralAuditReport {
    pub evicted: usize,
    pub boundary_evicted: usize,
    pub hard_mode_evicted: usize,
}
pub fn audit_structural_evictions(records: &[LedgerWitnessRecord]) -> StructuralAuditReport;
```

## Feature Flags

- `mincut-forget` (existing, unchanged): already gates `graph_forget`;
  `structural_witness` is gated identically since it depends on
  `MincutGatedForgetting` directly. Off by default.

## Benchmark Evidence

See "Evidence" above and the linked nightly README.

## Security

- No new cryptographic primitive; reuses `SignedWitnessSink`/
  `ed25519-dalek` exactly as ADR-347 shipped it.
- `aux` is covered by `LedgerWitnessRecord::chain_hash` (all 64 bytes) but
  excluded from `compute_record_hash` (first 48 bytes) — a tamper is caught
  by the chain walk (`prev_hash`/head-commitment propagation), not by a
  record re-hashing itself in isolation. This is pre-existing ADR-134
  behavior, not introduced here; this ADR's tests demonstrate it against
  structural content specifically.
- Naive-vs-diligent tamper distinction measured identically in kind to
  ADR-347's finding, now against this payload: a non-recomputing tamper is
  caught by the unsigned walk; a fully forward-recomputed one is caught
  only by `verify_signed_chain`.
- `audit_structural_evictions` performs no verification; documented
  explicitly as trusting its input.
- Key management unaddressed, inherited from ADR-347.

## Governance

None beyond "no witness, no mutation," preserved identically to
`compact_witnessed`.

## Rollback

Remove `structural_witness.rs`, its tests, the `lib.rs` wiring, and the
`decide()` extraction from `graph_forget.rs` (restore `select_survivors`'s
inlined body). No other caller in the repository depends on `decide()` or
`structural_witness`; `compact_witnessed` and `witness_signing` are
untouched either way.

## Rejection Criteria

This ADR's hypothesis would be rejected if any of:

- Any audit-accuracy trial's recovered boundary set did not exactly match
  ground truth.
- Any naive-tamper trial was not caught by the unsigned chain walk.
- Any diligent-forgery trial was either caught by the unsigned walk (a
  broken forgery construction) or missed by `verify_signed_chain`.
- Structural-encoding overhead exceeded 10% mean latency versus the
  identical-policy baseline.

See the nightly README's "Acceptance Result" for the measured values
against each gate.

## Open Questions

1. Should `StructuralAuditReport` feed an RVF provenance export alongside
   ADR-347's signed admission spans?
2. Multi-issuer verification for federated structural receipts (inherited
   from ADR-347's open question 3, now with a second payload type).
3. Does re-evaluating ADR-345's rejected bridge-survival hypothesis at a
   different corpus scale, using this ADR's audit instrumentation to
   collect the data cheaply, change that earlier verdict? (Explicitly not
   answered by this ADR.)
