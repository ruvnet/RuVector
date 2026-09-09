# ADR-346: Signed Eviction-Witness Anchoring for Agent Memory

## Status

Accepted (for the module as implemented). Enabled by default
(`ruvector-agent-memory::eviction_witness_signing`, no feature flag — see
[API Shape](#api-shape)). Not yet wired into `compact_witnessed`'s call
sites in this repository; that integration is explicitly deferred (see
[Migration](#migration)).

## Context

`ruvector-agent-memory`'s eviction-witness chain
(`witnessed_compaction::compact_witnessed`, ADR-345, 2026-09-05 nightly)
certifies every evicted entry with a chained `LedgerWitnessRecord`
(ADR-134 schema, keyless FNV-1a linking). Both `crate::ops`'s module docs
and ADR-134 §3/§9 name the same limitation explicitly: FNV-1a chaining is
tamper-EVIDENT against accidental corruption only. An adversary with write
access to the persisted log can mutate any record and recompute every
downstream `record_hash`/`prev_hash` (and, if they also control any head
commitment stored alongside the log, that too) in one O(n) pass;
`MemoryWitnessLog::verify_chain()` on the result still reports `true`.
`crate::ops` flags this as "an explicit follow-up gate that MUST land
before WP8 cross-repo anchoring makes this log load-bearing for acceptance
decisions." The 2026-09-05 nightly run's "Next Research" item 4 named the
same fix: "wire an `Ed25519` `WitnessSigner`... so eviction receipts are
signed, not just hash-chained."

This crate already depends on `rvf-types` with the `ed25519` feature
(unconditionally, not optional) for `AtomicObservation` per-observation
signatures (ADR-320). `ruvector-retrieval-receipt::state_anchor` (ADR-342)
separately proved out a periodic head-anchoring policy — interval vs.
staleness vs. signing-cost tradeoff — for a *different* chain (retrieval
receipts). This ADR asks: does the same anchoring shape, applied to the
eviction-witness chain using the crate's own already-present Ed25519
primitive, actually close the adversarial gap `crate::ops` names, and at
what cost?

## Hypothesis

```text
Given an eviction-witness chain of N = 4,096 FNV-1a-linked
LedgerWitnessRecords (real compact_witnessed output, not synthetic
records),

when the chain head is signed with Ed25519 (rvf_types::ed25519_sign, no
new signature scheme) either after every record (interval_records = 1,
"candidate_a") or periodically every K records (interval_records = K > 1,
"candidate_b"), and compared against baseline (FNV-1a chaining only, no
signing),

then a real relink attack (mutate one record, recompute every downstream
record_hash/prev_hash AND the log's head commitment) is caught 100% of the
time by any signed anchor whose sequence is at or after the tampered
record, while it is caught 0% of the time by baseline's verify_chain()
alone,

subject to: (a) zero false positives — an anchor whose sequence predates
the tamper must still verify against its own (unaffected) chain segment;
(b) max staleness (records since the last anchor) never exceeds
interval_records - 1, measured exactly; (c) amortized signing cost must
drop materially as interval_records grows (Pareto tradeoff against
staleness, not a free lunch).
```

Acceptance thresholds, fixed in the benchmark source before it was run
(see `examples/signed_eviction_witness_bench.rs`'s printed `ACCEPTANCE`
block): 100% detection once a covering anchor exists, AND 0 false
positives on anchors that predate the tamper. Both are binary, not
tunable after the fact.

Verdict: **ACCEPT**, 4 independent runs of the final code (see [Benchmark Evidence](#benchmark-evidence)).

## Decision

1. Add `ruvector-agent-memory::eviction_witness_signing`: `sign_anchor`,
   `verify_anchor`, `verify_anchor_against_chain`, `EvictionAnchorPolicy`,
   `EvictionAnchorLog`, and the attack-simulation utility
   `relink_tampered_suffix` (used by both the module's own tests and the
   benchmark to produce a *real* relinked-chain attack rather than assert
   the vulnerability exists).
2. No new signature scheme: signing goes through `rvf_types::ed25519_sign`
   / `ed25519_verify` / `Ed25519Keypair`, already an unconditional
   dependency of this crate.
3. No new feature flag: unlike `mincut-forget` and `proof-gate`, this
   module has no optional dependency to gate and is compiled in by
   default.
4. Recommend `EvictionAnchorPolicy::new(16)` or similar (single-digit
   percent memory overhead, sub-16-record staleness) as the production
   default over `interval_records = 1` (137.5% memory overhead relative to
   the raw chain — see [Benchmark Evidence](#benchmark-evidence)) — this is
   evidence-based guidance, not itself an integration.
5. Do **not** wire this into `compact_witnessed`'s existing call sites in
   this change. That is a deliberate scope boundary (see [Migration](#migration)).

## Evidence

Real code, real signs/verifies, real relink attack — see
[Benchmark Evidence](#benchmark-evidence) below and
`docs/research/nightly/2026-09-09-signed-eviction-witness-anchoring/README.md`
for full methodology, four raw runs, and honest limitations.

## Consequences

### Positive

- Closes a security gap that two independent parts of this crate already
  named as blocking ("MUST land before WP8 cross-repo anchoring"), with
  measured evidence the fix actually works against a real attack, not an
  assumed one.
- Reuses existing primitives end to end: `rvf_types` Ed25519 (already a
  dependency, ADR-320 precedent), and the interval/staleness anchor shape
  already proven by ADR-342 for a sibling crate's chain. No new
  cryptographic surface, no new crate.
- Quantifies, rather than asserts, the interval/cost/memory Pareto
  frontier for this specific chain (12 points, `K` from 1 to 4,096),
  giving a concrete, evidence-backed default (`K≈16`) instead of a guess.

### Negative

- Software Ed25519 signing here measures ~77-89 µs/op — two to three
  orders of magnitude slower than ADR-134 §9's <10 µs TEE-backed
  `WitnessSigner` budget, because `rvf_types::ed25519_sign` re-derives a
  `SigningKey` from raw secret bytes on every call rather than caching an
  expanded key. This is measured, not designed around; flagged in
  [Next Research](#open-questions) rather than hidden.
- `interval_records = 1` (candidate_a, the design ADR-134 §9's trait shape
  points to) costs more in anchor bytes (137.5%) than the raw 64-byte
  FNV-1a chain it protects — a real, not hypothetical, argument against
  per-record signing as the default here.
- An anchor authenticates only its own `(sequence, chain_head)` pair. Any
  tamper strictly after the most recent anchor an auditor holds is
  invisible until the next anchor lands — the staleness window is real and
  is exactly `interval_records - 1`, not asymptotically small.
- Anchors are not yet persisted, transported, or wired to any
  `WitnessSink` call site — this ADR ships the mechanism and its measured
  cost/security tradeoff, not an end-to-end deployed capability.

## Alternatives

| Alternative | Reason for rejection this run |
|-------------|-------------------------------|
| Sign every record inline via the ADR-134 §9 `WitnessSigner` trait (candidate_a as the *only* option) | Measured: 137.5% memory overhead vs. the chain itself at N=4,096; no worse detection than periodic anchoring, so periodic anchoring Pareto-dominates it for this workload. |
| A new signature scheme or crate for eviction anchoring | Rejected outright per ADR-320's own security gate ("no new hash or signature scheme") and this crate's existing unconditional `rvf-types` dependency. |
| TEE-backed signing (ADR-134 §9's stated preference for the <10 µs budget) | Out of scope: this environment has no TEE; software Ed25519 is what `rvf_types` provides today. Flagged as a real gap, not silently substituted. |
| Wiring anchoring directly into `compact_witnessed`'s existing call sites now | Deferred — see [Migration](#migration); keeping the mechanism decoupled from the call site lets it be adopted (or not) without touching ADR-345's existing, independently-tested API. |

## Implementation Plan

Already implemented in this change:
`crates/ruvector-agent-memory/src/eviction_witness_signing.rs` (module +
7 unit tests) and
`crates/ruvector-agent-memory/examples/signed_eviction_witness_bench.rs`
(benchmark). Follow-up (not this change): a `WitnessSink` adapter that
calls `EvictionAnchorLog::note_record` from within `compact_witnessed`
itself, once a key-management story (see [Open Questions](#open-questions))
exists.

## API Shape

```rust
pub struct EvictionAnchorPolicy { /* interval_records: u64 */ }
impl EvictionAnchorPolicy {
    pub fn new(interval_records: u64) -> Result<Self, AnchorError>;
}

pub struct EvictionAnchorLog { pub anchors: Vec<SignedEvictionAnchor>, /* .. */ }
impl EvictionAnchorLog {
    pub fn new(policy: EvictionAnchorPolicy) -> Self;
    pub fn note_record(&mut self, keypair: &Ed25519Keypair,
        record: &LedgerWitnessRecord, issued_at_ns: u64)
        -> Option<&SignedEvictionAnchor>;
    pub fn staleness_at(&self, at_record_count: u64) -> u64;
}

pub fn verify_anchor_against_chain(public_key: &[u8; 32],
    anchor: &SignedEvictionAnchor, chain_head_at_sequence: u64) -> bool;
```

## Feature Flags

None. Compiled in unconditionally, matching `rvf-types`'s own
unconditional (non-optional) dependency status in this crate's
`Cargo.toml`.

## Benchmark Evidence

Command (release build, real `compact_witnessed` output, no synthetic
records, no mocked crypto):

```bash
cargo run --release -p ruvector-agent-memory --example signed_eviction_witness_bench
```

Environment: `rustc 1.94.1 (e408947bf 2026-03-25)`, `cargo 1.94.1`, Linux
x86_64, commit `edaffffb3b85768eb1f3ec1f683b7f46f0506af4` (branch
`claude/focused-darwin-qrtfle`). N = 4,096 evicted records, single
`compact_witnessed` call, LRU policy, target_size = 0.

Signing cost (candidate_a, `K=1`, 4 independent runs, ~5% run-to-run
variance): 77,536 / 78,399 / 81,694 / 80,325 ns/anchor. Verify cost
(same 4 runs, paired): 48,936 / 48,958 / 48,970 / 48,556 ns/op.

Memory overhead by interval `K` (anchor struct = 88 bytes; chain = 4,096 ×
64 = 262,144 bytes): K=1 → 137.500%, K=2 → 68.750%, K=4 → 34.375%, K=8 →
17.188%, K=16 → 8.594%, K=32 → 4.297%, K=64 → 2.148%, K=128 → 1.074%,
K=256 → 0.537%, K=512 → 0.269%, K=1024 → 0.134%, K=4096 → 0.034%.

Adversarial detection (K=16, 111 sampled tamper positions, real
`relink_tampered_suffix` attack): 111/111 (100.0%) detected once a
covering anchor exists; 0 false positives on anchors predating the
tamper. Baseline (FNV-1a-only, no signing): relinked tamper at record
2,048 still reports `verify_chain() == true`.

`ACCEPTANCE` block printed by the benchmark on every one of 4 runs:
`100% detection once covered: true | 0 false positives: true => ACCEPT`.

Full raw output (all 4 runs) is reproduced in
`docs/research/nightly/2026-09-09-signed-eviction-witness-anchoring/README.md`.

## Security

- Threat model: a log-writing adversary who can mutate the persisted
  eviction-witness log (records and any co-located head commitment) but
  does **not** hold the Ed25519 secret key backing an externally-held
  anchor. This module's guarantee holds exactly under that model and no
  further — an adversary who also compromises the signing key is out of
  scope (key management is a deployment concern, not addressed here; see
  [Open Questions](#open-questions)).
- Domain separation: `EVICTION_ANCHOR_DOMAIN` prevents an anchor signed
  for this chain from being replayed as valid for a different signed
  artifact (mirrors `ruvector-retrieval-receipt::signing`'s
  `SIGNED_ROOT_DOMAIN` pattern).
- Deterministic Ed25519 (RFC 8032): identical `(sequence, chain_head,
  issued_at_ns)` always produces the same signature; no signature
  malleability concern for this use.
- Anchors must be held by a party other than the log's writer (e.g., an
  external auditor, a separate anchor store) to provide any guarantee
  beyond what `verify_chain()` already gives; this module does not
  itself provide that separation, only the primitive.

## Governance

No production data path currently depends on this module (see
[Migration](#migration)); no governance gate is bypassed or introduced by
this change. Recommended default (`K≈16`) is guidance for a future
integration decision, not a policy this ADR enforces.

## Failure Modes

- Signing failure: `rvf_types::ed25519_sign` has no fallible path in the
  current API (always returns a signature); a future `WitnessSink`
  integration must still decide whether a *missing* anchor blocks the
  eviction ("no witness, no mutation" — ADR-134's own invariant) or is
  logged as a gap. Left to the integration ADR.
- Clock skew / non-monotonic `issued_at_ns`: not checked by this module;
  callers must supply a trustworthy timestamp source, exactly as
  `ruvector-retrieval-receipt::state_anchor` already requires of its
  callers.
- Verification against a chain that was never anchored at all
  (`interval_records` chosen larger than the actual run length) correctly
  produces zero anchors — not an error, just no protection, matching the
  staleness math.

## Migration

Not applicable — no existing call site is modified. Adopting this module
in `compact_witnessed` (or any other `WitnessSink` consumer) is future
work requiring a key-management decision this ADR deliberately does not
make.

## Rollback

Trivial: the module is additive, has no feature flag to disable, and is
not called by any existing code path. Deleting
`eviction_witness_signing.rs` and its two `pub use` lines in `lib.rs`
fully reverts this change with no other effects.

## Rejection Criteria

This ADR's Decision would be rejected if a future run showed: (a) the
relink-attack detection rate falls below 100% once a covering anchor
exists (would mean the domain-separated signature construction has a
flaw); (b) any false positive on an anchor that predates its tamper
(would mean `verify_anchor_against_chain`'s sequence indexing is wrong);
or (c) amortized signing cost fails to drop with increasing
`interval_records` (would mean the periodic-anchor design provides no
real cost benefit over per-record signing, undermining the case for
`candidate_b` over `candidate_a`). None of these occurred in any of the 4 runs.

## Open Questions

1. **Key management is unaddressed.** Where does the signing key live,
   who rotates it, and how does an auditor obtain the corresponding
   public key out-of-band? This ADR provides the primitive, not the
   deployment story — same gap `ruvector-retrieval-receipt::signing`'s own
   module docs flag for its `Issuer`.
2. **Per-call `SigningKey` re-derivation dominates signing cost.**
   `rvf_types::ed25519_sign(secret: &[u8;32], ...)` re-derives the
   expanded signing key on every call. Caching an expanded key across
   calls in a future `rvf-types` API could plausibly close most of the
   measured 77-89 µs vs. ADR-134 §9's <10 µs budget gap — untested this
   run, named as immediate next research.
3. **Should `compact_witnessed` itself grow an optional signing hook?**
   Deferred by design (see [Migration](#migration)); the answer likely
   depends on (1).
4. **RVF/RVM applicability.** A signed eviction anchor is a natural
   candidate for RVF's "witness portability" and "signed lineage" fields
   (an RVF package could carry the anchor log alongside the memory
   snapshot it certifies) and for RVM's "proof-gated mutation" pattern
   (an anchor could itself be the artifact a P2/P3 proof references, per
   ADR-134 §10) — plausible, not measured or implemented this run.
