# ADR-357: TileZero-Gated Agent Memory Admission (`TileZeroGate`)

## Status

**Accepted** (opt-in, feature-flagged off by default). Nightly research
2026-11-10,
`docs/research/nightly/2026-11-10-tilezero-gated-agent-memory-admission/`.

## Context

`crates/ruvector-agent-memory/src/ledger.rs`'s `TransactionalLedger<S, G>`
promotes a `Pending` entry to `Accepted` only through a `G: ProofGate`:

```rust
pub trait ProofGate {
    fn admit(&mut self, entry_id: u64, content: &[u8]) -> Result<AcceptanceReceipt, LedgerError>;
}
```

Two implementations exist before this ADR:

- `AlwaysAdmitGate` — unconditional admission, denies 0% by construction.
- `WriteGateAdapter<G: ruvector_proof_gate::WriteGate>` (feature
  `proof-gate`) — hash-chain/Merkle tamper-evidence over the *receipt
  chain*. It never inspects the candidate's `content` bytes at all.

Neither gate can refuse an acceptance because of what the content actually
*says*. Separately, `crates/cognitum-gate-tilezero` implements a fully
tested, three-filter admission decision
(`ThreeFilterDecision::evaluate` — structural min-cut, shift/drift, and
evidence filters over a `ReducedGraph` summary) for TileZero, the central
arbiter of a 256-tile WASM fabric. That decision logic had never been wired
to any content store. This ADR closes both gaps at once: it gives
`ruvector-agent-memory` its first content-inspecting `ProofGate`, and gives
`cognitum-gate-tilezero`'s decision logic its first real integration
outside the 256-tile fabric it was built for.

The ledger's "no witness, no mutation" invariant (all witness records for
an operation are emitted to the `WitnessSink` as one atomic batch *before*
any state mutation; a sink failure aborts with no partial application) is
unchanged by this ADR and is re-verified with `TileZeroGate` installed (see
Evidence below).

## Hypothesis

See the nightly research doc's verbatim, pre-registered hypothesis and
acceptance thresholds
(`docs/research/nightly/2026-11-10-tilezero-gated-agent-memory-admission/README.md#hypothesis`).
In short: routing acceptance through a new `TileZeroGate` (which builds a
per-admission `ReducedGraph` from the candidate's content against a sliding
window of recently-admitted content) should Deny/Defer a measurably higher
fraction of synthetically "incoherent" candidates than the two existing
gates, while Permitting coherent candidates at a comparable rate, with
latency measured and reported (no latency ceiling for acceptance).

## Decision

Add `TileZeroGate` (`crates/ruvector-agent-memory/src/tilezero_gate.rs`),
feature-gated behind `tilezero-gate` (off by default), implementing
`ProofGate` by:

1. Deriving a fixed-64-dimension pseudo-embedding from the candidate's raw
   content bytes via byte-trigram feature hashing (no real embedding
   model — see Security/Limitations).
2. Comparing it (mean cosine similarity) against a sliding window
   (`VecDeque`, default capacity 64) of the embeddings of the most
   recently *admitted* entries.
3. Mapping that single similarity scalar onto `cognitum-gate-tilezero`'s
   `ReducedGraph`'s three scalar fields (`global_cut`,
   `aggregate_shift_pressure`, `aggregate_evidence`) via its own public
   setters, and evaluating with `ThreeFilterDecision::evaluate`.
4. `Permit` admits (and the embedding joins the window); `Deny` and
   `Defer` both collapse to `Err(LedgerError::GateDenied(_))`, since
   `ProofGate::admit` has no third outcome.

`cognitum-gate-tilezero`'s optional `mincut` feature (pinned to a stale
`ruvector-mincut = "0.1.30"` from crates.io) is **not** enabled.
`ReducedGraph` does not need it — confirmed by reading
`crates/cognitum-gate-tilezero/src/supergraph.rs`: it is a plain scalar
summary with public setters (`set_global_cut`, `set_shift_pressure`,
`set_evidence`), independent of any mincut crate.

## Evidence

Commands actually run (exit codes from this session; see the nightly
doc's "Build/Test Hygiene" section for the full transcript):

| Command | Exit |
|---|---|
| `cargo fmt -p ruvector-agent-memory -- --check` | 0 |
| `cargo build --release -p ruvector-agent-memory --features tilezero-gate` | 0 |
| `cargo build --release -p cognitum-gate-tilezero` | 0 |
| `cargo test -p ruvector-agent-memory --features tilezero-gate` | 0 (92 tests: 52 lib + 1 main-bin + 10 arbitration + 8 atomic_observation_fusion + 13 tarl_ledger + 8 tilezero_gate_tests, + 0 doctests) |
| `cargo test -p ruvector-agent-memory` (no feature) | 0 (48 lib tests — the 4 new unit tests correctly compile out; `tilezero_gate_tests.rs` compiles to 0 tests, correctly gated off) |
| `cargo clippy -p ruvector-agent-memory --all-targets --features tilezero-gate` | 0, zero warnings attributable to any file this ADR touches (one `clippy::assertions_on_constants` warning was found and fixed during this run; see the nightly doc) |

Benchmark (`cargo run --release -p ruvector-agent-memory --features
tilezero-gate --example tilezero_gate_bench`, seed 42, 500 trials/stream,
raw stdout in
`docs/research/nightly/2026-11-10-tilezero-gated-agent-memory-admission/benchmark_raw_output.txt`):

| Gate | Stream | Permit | Refused | Mean accept() latency |
|---|---|---|---|---|
| `AlwaysAdmitGate` | coherent | 500/500 | 0 | 0.548us |
| `AlwaysAdmitGate` | incoherent | 500/500 | 0 | 0.508us |
| `TileZeroGate` (default) | coherent | 496/500 (99.2%) | 4 | 4.124us |
| `TileZeroGate` (default) | incoherent | 0/500 (0.0%) | 500 (100.0%) | 4.859us |

**Pre-registered acceptance, default `GateThresholds`:**

1. Deny+Defer delta on incoherent stream: 100.0% − 0.0% = **100.0pp** (need ≥ 50.0pp) → **PASS**.
2. Coherent Permit rate: **99.2%** (need ≥ 85.0%) → **PASS**.
3. All tests pass, release build clean, clippy clean → **PASS** (table above).
4. Witness-before-mutation invariant with `TileZeroGate` installed:
   `tilezero_gate_failing_sink_no_witness_no_mutation` in
   `crates/ruvector-agent-memory/tests/tilezero_gate_tests.rs` → **PASS**.
5. Added latency: **~3.96us/admission** mean (`TileZeroGate.accept()` vs
   `AlwaysAdmitGate.accept()`, pooled over both streams) — reported, no
   ceiling to meet.

**Verdict: ACCEPT.** See the nightly doc for the full computation and all
honesty caveats (byte-trigram-hashing scope limitation, the single-scalar
three-filter collapse, the tau_permit>100 cold-start trap discovered during
the Darwin-lite sweep).

**Darwin-lite sweep** (13 `GateThresholds` configs: default + 3
generations × 4 candidates): default fitness 0.5988; best candidate
(`gen3_cand4`, `tau_permit≈92.0`) fitness 0.6151 — a real but modest
improvement (coherent permit 100.0% vs 99.2%, incoherent permit 0.4% vs
0.0%). **Promoted** as the new recommended default for `TileZeroGate`
going forward (see nightly doc for the full 13-row lineage table and the
tau_permit>100 trap this sweep discovered and the correctness hard-gate
extension it required).

## Consequences

- **Positive**: `ruvector-agent-memory` gains its first content-inspecting
  admission gate, fully opt-in and additive — no change to any existing
  code path or default behavior.
- **Negative**: ~8x higher per-admission latency than `AlwaysAdmitGate`
  (microseconds, not milliseconds — see Benchmark). Pulls `tokio` in as a
  transitive compile dependency (feature-gated; unused at runtime by this
  integration, see Implementation plan).
- **Risk, explicitly accepted**: the embedding (byte-trigram feature
  hashing) and the three-filter→single-scalar collapse are both ad hoc
  simplifications, not a claim about real semantic poisoning detection
  (see nightly doc's Security section).

## Alternatives

See the nightly doc's "Rejected Alternatives" section (enabling
`cognitum-gate-tilezero`'s `mincut` feature; depending on
`ruvector-retrieval-receipt`'s signing machinery instead of a fresh gate;
building genuinely independent structural/shift/evidence signals from a
synthetic multi-tile report instead of one scalar).

## Implementation plan

1. `crates/ruvector-agent-memory/Cargo.toml`: optional path dependency
   `cognitum-gate-tilezero`, new feature `tilezero-gate =
   ["dep:cognitum-gate-tilezero"]`, new example target
   `tilezero_gate_bench`.
2. `crates/ruvector-agent-memory/src/tilezero_gate.rs` (new,
   `#[cfg(feature = "tilezero-gate")]`): `TileZeroGate`, `embed_content`,
   `TILEZERO_EMBED_DIM`, `TILEZERO_DEFAULT_WINDOW`; wired into `lib.rs`.
3. `crates/ruvector-agent-memory/tests/tilezero_gate_tests.rs` (new): 8
   tests covering classification, invariant preservation, compilation
   sanity.
4. `crates/ruvector-agent-memory/examples/tilezero_gate_bench.rs` (new):
   the benchmark and Darwin-lite sweep described above.
5. This ADR plus the nightly research doc and gist.

`cognitum-gate-tilezero` pulls in `tokio` (`sync`, `time` features)
unconditionally in its own `Cargo.toml`. `ThreeFilterDecision` and
`ReducedGraph` (the only types `TileZeroGate` uses) are plain synchronous
functions — no `async`/`.await`, no runtime needed at the call site. The
straightforward path (depend on the crate as-is, do not touch its
Cargo.toml) worked without issue; `tokio` only adds compile time to the
dependency graph, not runtime behavior. See the nightly doc for whether
that compile-time cost was separately measured (it was not, beyond overall
build times already reported above).

## API shape

```rust
pub struct TileZeroGate { /* private */ }

impl TileZeroGate {
    pub fn new(thresholds: GateThresholds) -> Self;
    pub fn with_window_size(thresholds: GateThresholds, window_size: usize) -> Self;
    pub fn window_len(&self) -> usize;
    pub fn window_size(&self) -> usize;
    pub fn last_outcome(&self) -> Option<&DecisionOutcome>;
    pub fn thresholds(&self) -> &GateThresholds;
}

impl ProofGate for TileZeroGate {
    fn admit(&mut self, entry_id: u64, content: &[u8]) -> Result<AcceptanceReceipt, LedgerError>;
}

pub fn embed_content(content: &[u8]) -> Vec<f32>;
pub const TILEZERO_EMBED_DIM: usize = 64;
pub const TILEZERO_DEFAULT_WINDOW: usize = 64;
```

## Feature flags

`tilezero-gate = ["dep:cognitum-gate-tilezero"]` — off by default. No
existing behavior changes when this feature is not enabled (verified: 48
pre-existing lib tests pass identically with and without it, and the new
test file compiles to zero tests when the feature is off).

## Benchmark evidence

See Evidence above and the nightly doc's full benchmark section + raw log
file for the complete table, the 13-config Darwin-lite lineage, and the
reproducibility spot-check.

## Security

A content-inspecting admission gate is a new attack surface worth naming
explicitly:

- **Adversarial input to the embedding itself.** `embed_content`'s
  byte-trigram feature hashing is a known, fixed, public transform — an
  adversary who knows it (and this is open-source) could in principle
  craft content whose trigrams deliberately hash into the same buckets as
  recent legitimate history with matching signs, inflating its similarity
  score without being semantically related at all. This is not evaluated
  in this run; it is a real, named residual risk of any fixed, public
  hashing scheme used as a security-relevant similarity signal (see
  nightly doc Security section for more).
- **Gate-bypass risk is unchanged from the existing `ProofGate` trait.**
  `TileZeroGate` is consulted the same way `AlwaysAdmitGate`/
  `WriteGateAdapter` are — before staging, before the witness batch. A
  caller who constructs the ledger with `AlwaysAdmitGate` instead is not
  "bypassing" `TileZeroGate`; it simply was never installed. There is no
  new way to bypass a gate that *is* installed.
- **The witness chain's tamper-evidence is independent of, and unaffected
  by, this gate** — `MemoryWitnessLog::verify_chain` and (if wrapped)
  `SignedWitnessSink` do not consult `ProofGate` at all.

## Governance

Feature-flagged off by default; nothing changes for existing deployments.
`GateThresholds` is currently a plain, caller-constructed struct with no
runtime reconfiguration API — whoever constructs the `TransactionalLedger`
controls it, i.e. governance is identical to ordinary code-review-gated
configuration today. No production auto-tuning exists; the Darwin-lite
sweep in this run is an offline research exercise, not a live
reconfiguration path (see nightly doc's ruFlo Implications for a concrete
proposal to make it one).

## Failure modes

See the nightly doc's Failure Modes section in full, including the
`tau_permit > 100.0` cold-start trap discovered by the Darwin-lite sweep
(a misconfiguration can make the gate permanently defer every admission,
cheaply, by never warming its window) and the `GateDecision::Defer`
collapse into the same `Err` as `Deny` at this trait's binary boundary.

## Migration

Additive only. Existing callers using `AlwaysAdmitGate` or
`WriteGateAdapter` are completely unaffected. To adopt: construct
`TransactionalLedger::new(sink, TileZeroGate::new(thresholds))` instead of
the existing gate.

## Rollback

Remove the `tilezero-gate` feature from the enabling crate's feature set,
or simply construct the ledger with a different `ProofGate`. No persisted
state or schema is introduced by this gate (it holds only an in-memory
sliding window, never serialized).

## Rejection criteria

Would have been REJECT if either pre-registered acceptance criterion 1 or
2 failed on the default-config measurement, or if any pre-existing test
regressed, or if the witness-before-mutation invariant test failed. None
of those happened — see Evidence.

## Open questions

1. Whether a real upstream embedding (not byte-trigram hashing) changes
   the separation numbers, and by how much.
2. Whether genuinely independent structural/shift/evidence signals (from
   a synthetic multi-worker-tile report, exercising `merge.rs`'s
   `ReportMerger` instead of the single-scalar collapse this run uses)
   would change the picture.
3. Adversarial-embedding-crafting risk against the fixed hashing scheme —
   not evaluated here.
4. Whether `cognitum-gate-tilezero`'s `tokio` dependency should eventually
   be made optional/feature-gated upstream, now that a second consumer
   (this one, sync-only) exists.

See the nightly research doc for the full discussion, all numbers, and
the complete list of practical/long-horizon applications.
