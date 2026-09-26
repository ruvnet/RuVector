# ADR-305: Witnessed Evolution — Hash-Chained Provenance for Evolutionary ANN Parameter Search

## Status

Proposed. Experimental crate (`ruvector-witnessed-evolution`), not wired
into any production tuning path.

## Context

`ruvector-sona` already runs an unwitnessed `(1+1)`-evolution strategy
(`examples/darwin_autotuner.rs`, `src/auto_tuner.rs`) to tune configs
against a live, drifting stream. `ruvector-proof-gate` (ADR-227) gives
RuVector a tamper-evident *write* path via `HashChainGate`/`MerkleGate`.
`ruvector-retrieval-receipt` (ADR-304) extends that guarantee to the *read*
path. No crate combines an evolutionary search with a witness chain over
its own mutation/fitness/promotion history — the process that decides
*which* parameters end up serving reads and writes has never had the same
tamper-evidence its own outputs already get.

This gap matters specifically for this repository's own nightly research
process: the harness's Darwin promotion gate lists `witness_valid` and
`reward_hack_free` as mandatory preconditions, and states "a failed Darwin
candidate must remain part of the lineage so future runs do not rediscover
it blindly." Neither claim is enforceable today — a Darwin lineage is
whatever prose a nightly run happened to write down, with no cryptographic
guarantee it wasn't edited after the fact.

Grep across the repository confirms no existing crate provides this:
`ruvector-sona`'s auto-tuner has zero `witness`/`merkle`/`hash_chain`/`proof`
references. `ruvector-retrieval-receipt` witnesses query *results*, not the
*process that chose the parameters* those queries ran with.

## Hypothesis

```text
Given a fixed, seeded ruvector-coherence-hnsw workload and a fixed
(1+1)-ES over its [coherence_threshold, ef] genome,

when every generation's genome, fitness, and accept/reject decision is
committed to a ruvector-proof-gate HashChainGate as it is produced,

then the witnessed run's final genome and fitness are bit-identical to an
unwitnessed run of the same algorithm and seed, an independent replayer
verifies 100% of honest lineages, and a single forged fitness value is
caught at the exact generation it was forged, 100% of the time,

subject to witnessing wall-clock overhead staying under 15%, the witnessed
search beating a fixed-default baseline, and build/tests remaining green.
```

Full derivation, benchmark methodology, and raw results:
[`docs/research/nightly/2026-08-19-witnessed-evolution-ann-tuning/README.md`](../research/nightly/2026-08-19-witnessed-evolution-ann-tuning/README.md).

## Decision

Add `crates/ruvector-witnessed-evolution`, a small crate that:

1. Defines a two-parameter genome (`threshold`, `ef`) over
   `ruvector-coherence-hnsw`'s `CoherenceGatedSearch`, deliberately
   excluding graph-topology parameters (out of scope: O(N²) rebuild cost
   per generation) and wall-clock latency from the fitness function
   (out of scope for a different reason: timer noise would break
   determinism).
2. Runs a `(1+1)`-ES in two variants — `run_unwitnessed` and
   `run_witnessed` — sharing identical mutation/acceptance logic so their
   trajectories are provably comparable.
3. Commits every generation through `ruvector_proof_gate::HashChainGate`,
   reusing `WritePayload.vector` for the genome and `WritePayload.metadata`
   for packed fitness/decision bytes — no new hashing primitive.
4. Provides `WitnessedLineage::replay_verify`, an independent auditor that
   recomputes payload hashes, re-derives the chain, re-evaluates fitness
   from the raw genome against the workload, and re-derives every
   accept/reject decision under the fixed promotion policy (accept iff
   composite fitness strictly improves on the running incumbent;
   generation 0 always accepted).
5. Provides `tamper_composite`, an adversarial test hook that mutates a
   committed record's evidence without recomputing its receipt — modeling
   an attacker who can edit a log file but not forge a SHA-256 preimage.

## Evidence

- 11 unit/integration tests, all passing (`cargo test -p
  ruvector-witnessed-evolution`).
- `cargo clippy --all-targets` and `cargo fmt --check`: clean.
- Release benchmark run 3 times independently
  (`cargo run --release -p ruvector-witnessed-evolution --bin benchmark`):
  identical final genome (`threshold=0.101, ef=41, composite=0.9274`),
  identical chain root across all 3 runs; 3.2% composite-fitness
  improvement over the fixed default (`composite=0.8988`); honest lineage
  replay-verified on all 3 runs; forged-fitness tamper detected at the
  exact tampered generation on all 3 runs; wall-clock "overhead" measured
  negative in all 3 runs (noise — witnessing cost is ~8µs against a
  ~300–460ms search budget, below the measurement noise floor).
- Full raw output and methodology in the linked research doc.

## Consequences

**Positive:**

- A `WitnessedLineage` is a working, tested, ~154 bytes/generation
  provenance format any future nightly Darwin run tuning ANN parameters
  can reuse directly, rather than re-deriving one.
- Demonstrates the witnessing overhead for this class of search is
  negligible — this pattern can be adopted elsewhere in the ecosystem
  (`ruvector-sona`'s auto-tuner, future adaptive-cache or quantization
  tuners) without a meaningful performance argument against it.

**Negative / accepted limitations:**

- The witness chain is unsigned — it detects post-issuance mutation of
  evidence, not dishonesty in the search process that produced it in the
  first place (same threat model `ruvector-retrieval-receipt` already
  accepts for its own receipts).
- `replay_verify` trusts the workload it is handed; a lineage's evidentiary
  value depends on shipping the workload seed alongside it.
- Two-parameter genome only; does not generalize to graph-topology
  parameters without further work (see Migration).

## Alternatives Considered

- **Fold latency into the fitness function.** Rejected: breaks the
  bit-identical determinism claim between witnessed and unwitnessed runs,
  and breaks `replay_verify`'s exact-match check. Latency is measured and
  reported separately instead.
- **Evolve graph-build parameters (`m`, `m_longjump`).** Rejected for this
  PoC: O(N²) rebuild per generation, 40× more expensive, and a materially
  different (index-time vs. query-time) tuning workflow.
- **`MerkleGate` instead of `HashChainGate`.** Rejected: an MMR's
  advantage (O(log n) single-leaf proofs) matters for large, frequently
  spot-checked logs; a 40-generation lineage's O(n) full re-derivation in
  `replay_verify` is already sub-millisecond. Matches the tradeoff
  reasoning `ruvector-retrieval-receipt` already made for its own variant
  choice.
- **Witness a random-search variant instead of the same ES algorithm
  witnessed/unwitnessed.** Rejected: would conflate "does witnessing cost
  anything" with "is ES better than random search," a different and
  already-answered question.

## Implementation Plan

Already implemented as described in Decision. No further phased rollout —
this is a standalone, self-contained crate with no dependents yet.

## API Shape

```rust
// ruvector_witnessed_evolution
pub struct Genome { pub threshold: f32, pub ef: f32 }
pub struct Workload { /* fixed dataset/graph/queries/ground-truth */ }
pub struct WorkloadConfig { /* build parameters, grouped to bound arity */ }
pub struct Fitness { pub recall_mean: f32, pub avg_expansions: f32, pub composite: f32 }

pub fn run_unwitnessed(workload: &Workload, generations: usize, seed: u64) -> EsOutcome;
pub fn run_witnessed(workload: &Workload, generations: usize, seed: u64) -> (EsOutcome, WitnessedLineage);

impl WitnessedLineage {
    pub fn record(&mut self, generation: u64, genome: Genome, fitness: Fitness, accepted: bool) -> WriteReceipt;
    pub fn replay_verify(&self, workload: &Workload) -> ReplayReport;
    pub fn tamper_composite(&mut self, generation_idx: usize, forged_composite: f32) -> bool; // test-only hook
    pub fn chain_root(&self) -> [u8; 32];
    pub fn records(&self) -> &[GenerationRecord];
}
```

## Feature Flags

None. The crate has no feature gates; it depends unconditionally on
`ruvector-proof-gate` and `ruvector-coherence-hnsw`, both already in the
default workspace member set.

## Benchmark Evidence

See "Benchmark Methodology" and "Benchmark Results" in the linked research
doc for the full raw transcript, hardware/toolchain versions, and the
three-run overhead table.

## Security

No new cryptographic primitive: reuses `ruvector-proof-gate`'s existing
SHA-256 `HashChainGate`. No secrets, credentials, or PII pass through
`WitnessedLineage` — only search-parameter floats and derived fitness
scalars. `tamper_composite` is a test-only hook exposed on the public API
for adversarial testing; it is not gated behind a feature flag because it
is the mechanism the crate's own tests use to validate tamper-evidence, and
misuse of it in a real pipeline would only corrupt an operator's own
evidence (not forge a passing verification — see next point).

**A tampered lineage cannot be made to pass `replay_verify` by also calling
`tamper_composite` "correctly":** any change to a committed generation's
plaintext fields changes the payload hash `replay_verify` recomputes,
which is checked against the receipt's `payload_hash` captured at
admission time — a value `tamper_composite` does not (and structurally
cannot, without breaking encapsulation) touch.

## Governance

Experimental only. Any future decision to gate real parameter promotions
on `replay_verify().verified` needs its own ADR describing the promotion
pipeline, storage location for lineages, and retention policy.

## Failure Modes

See "Failure Modes" in the linked research doc: fitness-determinism
dependency, replay's dependency on being given the correct workload, and
the unsigned-chain limitation.

## Migration

None required — new, standalone crate. Future migration path if adopted
for a production tuner: (1) extend the genome type per-domain (e.g. cache
thresholds, quantization bit budgets) while reusing `WitnessedLineage`
unchanged; (2) wire into a ruFlo scheduled workflow; (3) add signing.

## Rollback

Remove the crate from workspace `members` and delete
`crates/ruvector-witnessed-evolution`. No other crate depends on it.

## Rejection Criteria

Would have been rejected had any of: witnessed/unwitnessed runs diverged,
the ES failed to beat the fixed baseline, `replay_verify` failed on an
honest lineage, a tampered lineage passed verification, or overhead
exceeded 15%. None occurred across 3 independent runs (see Evidence).

## Open Questions

1. Does the witnessing overhead remain negligible at production query
   volumes and higher-frequency online re-tuning (per `sona::auto_tuner`'s
   staleness-weighted online model), rather than this PoC's one-shot batch
   search?
2. What is the right lineage storage/retention format for a ruFlo-scheduled
   production deployment — this ADR does not specify one?
3. Is a signed variant (tying `chain_root` to an agent identity key)
   worth the added key-management complexity for the threat models that
   actually matter here?
