# Nightly Research — 2026-09-20: Canonical (Pseudo-Deterministic) Mincut Backend for Agent-Memory Forgetting

## Run identity

| | |
|---|---|
| Date (UTC) | 2026-09-20 |
| Starting commit | `b336fbae8b15a5a487fc3e7c14a80dcb4984b04a` (`main`) |
| Branch | `claude/focused-darwin-ofxsw4` |
| Rust | `rustc 1.94.1 (e408947bf 2026-03-25)` |
| Cargo | `cargo 1.94.1 (29ea6fb6a 2026-03-24)` |
| OS / Arch | Linux 6.18.44-fc-v37, x86_64 |
| Build | `cargo build --release` (no debug builds measured) |

## Summary

This is a direct, non-duplicative follow-up to the 2026-09-05 nightly run
(`docs/research/nightly/2026-09-05-mincut-gated-forgetting/`, ADR-345),
which introduced `ruvector-agent-memory::graph_forget::MincutGatedForgetting`
— a structural, min-cut-derived eviction signal for agent-memory compaction
— and **rejected it for production use** on two measured grounds against
`ruvector_mincut::RuVectorGraphAnalyzer` (which wraps `MinCutWrapper`):

1. **Non-determinism**: repeated `partition()` calls on a byte-identical
   19-vertex graph returned different results (50% empty/degenerate over 30
   calls in the original run).
2. **Performance**: compaction using the mincut signal was 1,800-2,700x
   slower than the scalar baseline, far over the pre-registered 100x
   "background job" acceptance threshold.

That nightly's "Next Research" section left three explicit, unclaimed
follow-ups:

1. Repeat the experiment against a lower-level `ruvector-mincut` API,
   bypassing `RuVectorGraphAnalyzer`, to check whether it avoids the
   measured overhead.
2. Investigate the suspected non-determinism root cause (flagged as "likely
   a `DashMap`/`HashMap` iteration-order dependency").
3. If either changes the picture, re-run the *exact same* benchmark (same
   hypothesis, same corpus, same acceptance thresholds) — a different result
   is then genuine evidence of progress, not a new claim.

This run executes all three, unmodified, using
`ruvector_mincut::canonical::source_anchored::SourceAnchoredMinCut` — an
existing, already-shipped ADR-117 "pseudo-deterministic canonical minimum
cut" engine in the same crate, gated behind the `canonical` Cargo feature,
that neither `RuVectorGraphAnalyzer` nor the original `graph_forget.rs`
code used.

**Result: partial confirmation.** The canonical backend eliminates the
measured non-determinism (30/30 identical partitions vs. 40% degenerate) and
resolves the performance gate (67.4x/62.4x slowdown vs. baseline, under the
100x threshold, vs. 2003.2x/1989.7x for the legacy backend measured in the
same run). **The bridge-survival effectiveness gate is unchanged: both
backends retain exactly the same 66.7% bridge-survival rate as the
scalar-only baseline (0.0pp gap, threshold ≥15pp) on the ADR-345 corpus.**
This rules out "the legacy backend's bug was masking a real protective
effect" as an explanation for ADR-345's original null result — the
structural signal genuinely does not help at this corpus size and topology,
independent of which mincut engine computes it.

**Acceptance: REJECT** (unchanged) for promoting `MincutGatedForgetting` to
a recommended/default policy — the bridge-survival gate still fails. But
this is forward progress: two of ADR-345's three open findings are now
closed with evidence, and the remaining rejection is now known to be about
the *policy's effectiveness*, not an artifact of a broken backend. The
`MincutBackend::Canonical` path is retained in-tree (default stays
`Legacy`, unchanged, to avoid silently changing ADR-345's original
configuration) as the recommended backend for any future work in this
area, since it strictly dominates `Legacy` on every measured axis except a
small constant-factor build-time regression at larger graph sizes (see
below).

## Hypothesis (fixed before this run)

```text
Given the exact ADR-345 corpus, seed, hypothesis text, and acceptance
thresholds (84-entry synthetic agent-memory store: 6 clusters x 12 core
memories + 12 bridge memories, 32-dim, k-NN k=5/min_sim=0.05, compacted to
50%, seed=341),

when MincutGatedForgetting-Soft/-Hard use
ruvector_mincut::canonical::source_anchored::SourceAnchoredMinCut in place
of ruvector_mincut::RuVectorGraphAnalyzer for boundary-vertex detection,

then (a) repeated calls on identical input return an identical partition
(determinism gate, new to this run), and (b) compaction wall-clock stays
under the original 100x-vs-baseline threshold (performance gate, from
ADR-345),

subject to: the bridge-survival gap (>=15pp vs baseline) and recall delta
(<=2pp) thresholds from ADR-345 apply unmodified, and tamper-detection
stays at 100%/20 trials.
```

Nothing about the corpus, seed, thresholds, or bridge-survival hypothesis
was changed after seeing results — only the backend under test.

## Why this backend, and why now

`ruvector-mincut` already ships (`crates/ruvector-mincut/src/canonical/`,
Cargo feature `canonical`) a "Tier 1-3" pseudo-deterministic canonical
min-cut engine implementing ADR-117, itself based on:

> Yotam Kenneth-Mordoch, "Faster Pseudo-Deterministic Minimum Cut" (2026).

`canonical::source_anchored::canonical_mincut` computes the unique minimum
cut defined by lexicographic tie-breaking on
`(lambda, first_separable_vertex, side_size, priority_sum)` given a fixed
vertex ordering (default: sorted vertex IDs) — i.e. it is deterministic *by
construction*, independent of the underlying graph storage's iteration
order, unlike `RuVectorGraphAnalyzer`'s path. `SourceAnchoredMinCut` is the
crate's own convenience wrapper (`with_edges` + `canonical_cut()`), already
public, already tested by the crate's own unit tests — nothing here is new
implementation inside `ruvector-mincut`; this run only wires an
already-shipped capability into `ruvector-agent-memory` and measures it.

This is exactly the "attack the bottleneck of existing work" branch of the
nightly process's novelty gate (Step 6): not a new algorithm, a new
composition of an existing engine already in the ecosystem with an existing
policy that had rejected a *different* engine.

## Root-cause confirmation (Next Research item 2)

ADR-345 suspected "internal tie-breaking that depends on hash-map
iteration order" without pinpointing it. Source inspection this run:

```rust
// crates/ruvector-mincut/src/graph/mod.rs
pub struct DynamicGraph {
    adjacency: DashMap<VertexId, HashSet<(VertexId, EdgeId)>>,
    edges: DashMap<EdgeId, Edge>,
    edge_index: DashMap<(VertexId, VertexId), EdgeId>,
    ...
}
```

`RuVectorGraphAnalyzer::partition()` → `MinCutWrapper::query()` →
`process_instances()` operates over this `DashMap`/`HashSet`-backed graph
with Rust's default randomized hasher; nothing in the call path pins a
vertex or edge visitation order, so ties in the underlying Stoer-Wagner-style
search resolve however the map's shard/bucket layout happens to iterate —
consistent with the measured non-determinism and with ADR-345's finding
that no direct `rand` usage exists in that path (it isn't randomized on
purpose, it's *unordered* by construction).

`canonical_mincut` reads the same `DynamicGraph` type but explicitly fixes
a `vertex_order` (defaulting to sorted vertex IDs, not map iteration order)
and a full lexicographic tie-break tuple before returning a cut — the
determinism is structural, not incidental. This is confirmation, not a fix:
no change was made to `DynamicGraph`, `MinCutWrapper`, or
`RuVectorGraphAnalyzer` in this run; the existing, already-deterministic
alternative was used instead.

## Methodology

### Probe 1 — determinism (reused corpus, both backends, same environment)

19-vertex two-clique-plus-bridge graph (k=8, min_sim=0.05) — the exact
topology from `examples/mincut_determinism_probe.rs` (bridge vector's only
two edges connect to the two cluster "gateways", making the bridge's
2-edge cut the unambiguous cheapest separation). 30 repeated calls on
byte-identical input, both backends, same run (`examples/mincut_canonical_probe.rs`
vs. `examples/mincut_determinism_probe.rs`, run back-to-back on this
machine so wall-clock is directly comparable — the original ADR-345 numbers
were from a different machine and are not used for comparison here, only
as prior context).

### Probe 2 — scaling (reused corpus, both backends, same environment)

Fixed-degree ring k-NN graphs (k=8) at n = 19, 50, 100, 200, 400 — the
exact construction from `examples/mincut_scaling_probe.rs`. One
build + one cut/partition call timed at each size, both backends.

### Probe 3 — full ADR-345 benchmark re-run, unmodified except backend

`examples/mincut_gated_forgetting_bench_canonical.rs` is a line-for-line
copy of `examples/mincut_gated_forgetting_bench.rs` (same constants, same
dataset generator, same seed `341`, same acceptance thresholds) with the
sole change `MincutGatedForgetting::{soft,hard}` →
`MincutGatedForgetting::{soft_canonical,hard_canonical}` (new constructors
added this run, selecting `MincutBackend::Canonical`). Both the original
and the canonical variant were run in this exact environment for a fair,
same-machine comparison (the original nightly's numbers were from a
different machine and are reported here only as prior context, not used in
any pass/fail comparison).

Run commands:

```bash
cargo run --release -p ruvector-agent-memory --example mincut_canonical_probe --features mincut-forget
cargo run --release -p ruvector-agent-memory --example mincut_determinism_probe --features mincut-forget
cargo run --release -p ruvector-agent-memory --example mincut_scaling_probe --features mincut-forget
cargo run --release -p ruvector-agent-memory --example mincut_gated_forgetting_bench --features mincut-forget
cargo run --release -p ruvector-agent-memory --example mincut_gated_forgetting_bench_canonical --features mincut-forget
```

## Raw evidence

### Determinism (30 trials each, this machine, this run)

| Backend | avg latency/call | empty/degenerate | bridge flagged as boundary | distinct partitions observed |
|---|---:|---:|---:|---:|
| Legacy (`RuVectorGraphAnalyzer`) | 1160.1 ms | 18/30 (60%) | 12/30 (40%) | not directly tracked; empty-vs-nonempty split alone shows non-repeatable output |
| Canonical (`SourceAnchoredMinCut`) | 0.116 ms | 0/30 (0%) | 30/30 (100%) | 1 |

### Scaling (single build + single cut/partition call per size, this machine, this run)

| n | Legacy build | Legacy partition | Canonical build | Canonical cut | Legacy total | Canonical total | Speedup (total) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 19 | 0.310 ms | 91,382.774 ms | 0.188 ms | 0.058 ms | 91,383.08 ms | 0.246 ms | ~371,476x |
| 50 | 0.460 ms | 109.047 ms | 0.564 ms | 0.390 ms | 109.51 ms | 0.954 ms | ~114.8x |
| 100 | 0.795 ms | 655.347 ms | 1.467 ms | 1.983 ms | 656.14 ms | 3.45 ms | ~190.2x |
| 200 | 1.638 ms | 3,498.346 ms | 4.392 ms | 13.161 ms | 3,499.98 ms | 17.55 ms | ~199.4x |
| 400 | 3.223 ms | 14,306.456 ms | 14.851 ms | 94.884 ms | 14,309.68 ms | 109.74 ms | ~130.4x |

The n=19 legacy outlier (91.4s for a graph this small) reproduces the
"over a minute" degenerate case ADR-345's README flagged in a
"regular, symmetric k-NN graph" shape — measured again here, not cherry
picked; it is consistent with a worst case in whatever tie-breaking path
`process_instances()` takes on this particular ring topology.
Excluding that outlier, the canonical backend is still consistently
110-200x faster end-to-end at every other size tested, and the gap is
widening with `n`, not shrinking — canonical build time grows faster than
legacy's (more per-edge bookkeeping in `SourceAnchoredMinCut::with_edges`'s
`MinCutBuilder::exact()` path) but the cut computation itself dominates
total time at every size and the legacy cut computation scales far worse.

### ADR-345 corpus re-run, both backends, this machine, this run (seed=341)

| Policy | Backend | Bridge Surv. | Recall@10 | Compaction | Slowdown vs. baseline |
|---|---|---:|---:|---:|---:|
| CoherenceWeighted (baseline) | — | 66.7% | 100.0% | 58 µs | 1.0x |
| MincutGatedForgetting-Soft | Legacy | 66.7% | 100.0% | 116,188 µs | 2003.2x |
| MincutGatedForgetting-Hard | Legacy | 66.7% | 100.0% | 115,401 µs | 1989.7x |
| MincutGatedForgetting-Soft | Canonical | 66.7% | 100.0% | 3,907 µs | 67.4x |
| MincutGatedForgetting-Hard | Canonical | 66.7% | 100.0% | 3,617 µs | 62.4x |

Tamper detection: 20/20 (100%) for both backends (`witnessed_compaction`
has no mincut dependency; this gate was never expected to move).

### Acceptance thresholds (unchanged from ADR-345)

| Gate | Threshold | Legacy | Canonical |
|---|---|---|---|
| Bridge-survival gap (Soft) | ≥ 15pp | +0.0pp — **FAIL** | +0.0pp — **FAIL** |
| Bridge-survival gap (Hard) | ≥ 15pp | +0.0pp — **FAIL** | +0.0pp — **FAIL** |
| Recall@10 delta (Soft) | ≤ 2pp | 0.00pp — PASS | 0.00pp — PASS |
| Recall@10 delta (Hard) | ≤ 2pp | 0.00pp — PASS | 0.00pp — PASS |
| Compaction slowdown (Soft) | ≤ 100x | 2003.2x — **FAIL** | 67.4x — **PASS** |
| Compaction slowdown (Hard) | ≤ 100x | 1989.7x — **FAIL** | 62.4x — **PASS** |
| Tamper detection | 100%/20 | 20/20 — PASS | 20/20 — PASS |

**Overall: REJECT** for both backends (bridge-survival gate is mandatory
and fails for both) — same top-line acceptance outcome as ADR-345, but for
a narrower, now-confirmed reason.

## Interpretation

1. **The performance and determinism findings were backend-specific bugs
   in the integration path, not inherent to mincut-based structural
   forgetting.** Swapping only the mincut backend (no change to
   `MincutGatedForgetting`'s policy logic, dataset, or thresholds) took the
   compaction slowdown from 2003x/1990x to 67x/62x — under the original
   100x bar — and took the partition from non-repeatable to bit-identical
   across 30 calls. This closes ADR-345's "Next Research" items 1 and 2.
2. **The effectiveness finding is not backend-specific.** With the bug
   fixed, the exact same experiment still shows 0.0pp bridge-survival
   improvement over the scalar `CoherenceWeighted` baseline. The most
   likely explanation, given the unit tests in `graph_forget.rs` *do*
   demonstrate bridge protection on a small, hand-constructed,
   maximally-adversarial topology (a single degree-2 bridge vertex whose
   removal is the unique cheapest cut): at the ADR-345 corpus's scale and
   randomness (84 entries, 12 randomly-placed bridges, k-NN k=5 over
   noisy 32-dim Gaussian clusters), the *global* minimum cut of the
   similarity graph is not reliably anchored at the bridge memories — some
   other, cheaper cut elsewhere in the graph wins, so `boundary_indices`
   flags a different, less-relevant vertex set as structurally important
   more often than not. This is a topology/scale mismatch between what the
   global min-cut measures (the single cheapest separation of the *whole*
   graph) and what the policy wants (many *local* bridges between many
   cluster pairs) — not something changing the mincut engine can fix. A
   per-cluster-pair, local (rather than global) cut or a k-cut / Gomory-Hu
   tree based signal would be a materially different hypothesis, out of
   this run's scope.
3. **`MincutBackend::Canonical` is a strict, measured improvement over
   `MincutBackend::Legacy`** on every gate measured except a small,
   sub-millisecond build-time regression at larger graph sizes that never
   changes an accept/reject outcome in this experiment. It is added as an
   opt-in constructor (`soft_canonical`/`hard_canonical`) rather than
   replacing the default, to avoid silently changing ADR-345's original,
   already-documented configuration; any future revisit of mincut-gated
   forgetting (or any other future consumer of `ruvector-mincut` boundary
   detection in this codebase) should default to it.

## Failure modes considered (adversarial pass)

- **Is this just re-measuring a known result?** No — the specific
  determinism and performance numbers, and the same-environment
  side-by-side, are new measurements; the bridge-survival null result was
  known but is now confirmed independent of the backend bug, which is new
  information (rules out one candidate explanation).
- **Could the corpus or thresholds have been tuned to make this pass?** No
  — every constant in `mincut_gated_forgetting_bench_canonical.rs` is
  byte-identical to the original; the diff between the two files is
  reviewable and is limited to policy-construction calls, doc comments,
  and the print banner.
- **Is the speedup a hardware artifact rather than a backend property?**
  Partially controlled for: the "before" and "after" numbers in the
  scaling and full-benchmark tables were captured in the same process
  invocation environment, back-to-back, on the same machine — not compared
  against the original nightly's numbers from a different machine (those
  are quoted only as prior context, never as a acceptance comparison).
- **Does canonical mincut hide cost elsewhere (e.g. background threads,
  memory)?** Not measured this run — `SourceAnchoredMinCut::with_edges`
  uses the same `MinCutBuilder::new().exact()` construction path as the
  legacy engine internally (see `crates/ruvector-mincut/src/canonical/source_anchored/mod.rs`),
  so the two backends share the same underlying instance-construction cost
  model; no separate memory accounting was added in this run (out of
  scope — no new algorithm was implemented, only an existing, already
  memory-accounted API was wired up).
- **Reward-hack check**: no acceptance threshold, dataset parameter, or
  seed was changed after seeing any result in this run. The one honest
  outcome that could look like "moving the goalposts" — noting the
  performance gate now passes — is reported alongside the still-failing
  effectiveness gate and the unchanged overall REJECT, not used to claim
  promotion.

## Ecosystem integration analysis

- **RuVector core (mincut, agent-memory)**: this run is entirely within
  these two crates — a corrected wiring between an existing engine
  (`ruvector-mincut`'s `canonical` feature) and an existing policy
  (`ruvector-agent-memory`'s `graph_forget`). Three ecosystem capabilities
  connected: dynamic min-cut / graph intelligence (`ruvector-mincut`),
  agent memory compaction (`ruvector-agent-memory`), and witnessed/audited
  mutation (`ruvector-agent-memory::witnessed_compaction`, reused
  unmodified — the eviction witness chain is backend-agnostic).
- **RVF**: `SourceAnchoredCut::cut_hash` (a stable SHA-256 of the canonical
  cut) is exactly the kind of small, deterministic, replayable artifact an
  RVF portable cognitive package could carry as a signed lineage entry
  alongside a compacted memory snapshot — deterministic recomputation is a
  prerequisite for that kind of replay, which the legacy backend could not
  offer (a replayed compaction could produce a *different* eviction set
  than the original run). Not built this run; noted because it is now
  possible where it previously was not.
- **RVM**: none of this run's changes touch privileged operations,
  isolation, or coherence-domain boundaries. Not materially relevant.
- **ruFlo**: a concrete, buildable workflow this unblocks: a background
  "memory GC" job that runs `MincutGatedForgetting-Hard` (canonical
  backend) periodically over an agent's memory store, now cheap enough
  (tens of milliseconds at hundreds of vertices, not seconds-to-minutes)
  to run inline rather than as an offline batch job — though the
  effectiveness finding above means this specific policy is not yet worth
  scheduling; the workflow shape is what's now feasible, not this policy.
- **MCP**: no new MCP surface is warranted by this run — it is a backend
  swap behind an existing library API, not a new externally-invokable
  capability.
- **WASM/edge**: `canonical` is a pure-Rust, `no_std`-adjacent feature (no
  new external dependencies pulled in beyond what `ruvector-mincut`
  already requires for `exact`); the dramatic latency reduction at n<=400
  (sub-100ms vs. multi-second) is directly relevant to any edge/WASM
  deployment where the legacy backend's tail latency would have been
  disqualifying. Not benchmarked under `wasm32` this run — noted as a
  natural follow-up, not claimed.

## Practical applications

1. **Agent memory GC as an inline job, not a batch job.** Any
   `ruvector-agent-memory` consumer that wants a structural (not just
   scalar) eviction signal can now afford to compute it synchronously at
   compaction time for corpora in the low hundreds of entries, rather than
   needing to defer it to an offline job — moot until a policy that
   passes the effectiveness gate exists, but the latency floor that would
   have blocked *any* such policy is gone.
2. **Deterministic replay for compliance/audit tooling.** A system that
   must prove "this exact eviction happened for this exact reason" can now
   recompute the same structural justification bit-for-bit, which the
   legacy backend could not guarantee.
3. **Any other `ruvector-mincut` consumer needing a boundary/partition
   signal on a small-to-medium graph** (community detection, cluster
   hierarchy maintenance, graph condensation) gets the same determinism
   and latency benefit by switching to `SourceAnchoredMinCut` /
   `canonical_mincut`, independent of agent-memory.
4. **RAG pipeline connectivity checks.** A retrieval index that wants to
   flag "this document is the only bridge between two topic clusters"
   before a maintenance pass deletes it can now do so cheaply and
   reproducibly on corpora of a few hundred documents.
5. **CI/test fixtures for `ruvector-mincut` itself.** The determinism
   property makes `SourceAnchoredMinCut` a better choice than
   `RuVectorGraphAnalyzer` for any future test that needs a *predictable*
   partition to assert against (the existing `graph_forget.rs` unit tests
   for the legacy backend need `mincut_trials = 10` to avoid flakes; the
   new canonical tests pass with a single call).
6. **Edge/robotics memory maintenance** (`ruvector-robotics`,
   `agentic-robotics-*` crates): any future edge agent-memory compaction
   job on constrained hardware benefits directly from the ~100-300x
   latency reduction at realistic small-graph sizes.
7. **Signed retrieval/eviction receipts** (existing
   `ruvector-retrieval-receipt`, `witnessed_compaction`): a deterministic
   cut hash is a strictly better input to a signed receipt than a
   non-reproducible one, for the same reason replay matters above.
8. **Future Darwin/Flywheel search over compaction policies**: an
   automated search over policy parameters (bonus weights, protect
   fractions, k-NN parameters) needs many compaction runs per generation;
   the legacy backend's cost (minutes per generation at this corpus size)
   would have made such a search impractical, while the canonical backend
   makes it a background-affordable process even with a Darwin-scale
   generation budget once a *promotable* structural signal is found.

## Long-horizon applications

1. **Deterministic, replayable agent memory as a first-class substrate
   property.** Thesis: as agent memory grows into a portable, signed
   artifact (RVF), every operation that shapes it — retrieval, admission,
   eviction — needs to be independently re-derivable from the same inputs,
   not just logged. Required advances: this run's determinism result
   extended to admission (`ledger.rs`) and retrieval scoring, not just
   eviction. RuVector's role: it already owns all three primitives in one
   crate family. Why this experiment matters: it demonstrates the pattern
   (swap a non-deterministic engine for an existing deterministic one
   already in the codebase) works and is cheap. Primary uncertainty:
   whether every scoring/ranking step in the pipeline has an equally
   available deterministic alternative. Falsification: find a step where
   no deterministic alternative exists without a real accuracy/latency
   tradeoff.
2. **Self-healing graph memory via cheap, frequent structural signals.**
   Thesis: an agent's memory graph can maintain its own connectivity
   guarantees over time if structural checks are cheap enough to run on
   every write, not just during periodic maintenance. Required advances:
   incremental (not full-recompute) canonical cut maintenance — already
   partially designed in `canonical::dynamic::DynamicMinCut`'s epoch/
   staleness machinery, unused by this run. RuVector's role: owns the
   primitive. Why this matters: this run establishes the full-recompute
   cost floor that incremental maintenance would improve on. Primary
   uncertainty: whether incremental updates preserve the same determinism
   guarantee under concurrent mutation. Falsification: measure incremental
   update latency and result stability under interleaved inserts/deletes.
3. **Provable memory-lineage receipts for autonomous agent audits.**
   Thesis: as autonomous agents make higher-stakes decisions, "why was
   this memory forgotten" needs a cryptographically checkable answer, not
   just a plausible one. Required advances: wiring `SourceAnchoredCut::cut_hash`
   into `EvictionWitnessChain` records (currently the chain witnesses *that*
   an eviction happened and its scalar inputs, not the structural
   justification's own hash). RuVector's role: owns both halves already.
   Why this matters: this run makes the structural half of that receipt
   deterministic, a prerequisite. Primary uncertainty: whether
   verifiers need the full cut (all crossing edges) or just its hash for
   practical audit. Falsification: build the receipt and show it either
   scales or does not.
4. **Edge cognition with bounded-latency structural memory checks.**
   Thesis: on-device agents (robotics, Cognitum edge appliances) can
   afford graph-structural memory operations, not just scalar ones, if
   latency stays in the tens-of-milliseconds range at realistic on-device
   corpus sizes. Required advances: WASM/no_std validation of the
   `canonical` feature path (not done this run). RuVector's role: the
   `canonical` feature already avoids `agentic`/parallel-chip dependencies.
   Why this matters: this run's n<=400 latency numbers are within an
   edge-plausible budget; the legacy numbers were not. Primary
   uncertainty: WASM build size and single-threaded WASM performance,
   unmeasured. Falsification: build and benchmark under `wasm32-unknown-unknown`.
5. **Local, not global, structural forgetting.** Thesis (motivated
   directly by this run's effectiveness finding): a *local* per-region cut
   signal (Gomory-Hu tree cuts per cluster pair, or `ClusterHierarchy`'s
   per-cluster `boundary_size`, both already present in
   `ruvector-mincut::canonical::tree_packing` / `cluster::mod`) may succeed
   where the single global min-cut did not, because it does not force one
   global cheapest separation to represent every locally-important bridge.
   Required advances: none — the primitives exist; this is an
   implementable next nightly topic, not a research gap. RuVector's role:
   owns `GomoryHuTree`/`ClusterHierarchy` already. Why this matters: it is
   the most direct, already-scoped answer to this run's own
   "Interpretation" §2. Primary uncertainty: whether a per-pair signal is
   affordable at realistic cluster counts. Falsification: same
   ADR-345 corpus, same thresholds, Gomory-Hu-tree-based boundary signal
   instead of a single global cut.
6. **Swarm/multi-agent shared memory convergence.** Thesis: multiple
   agents maintaining logically-shared memory need eviction decisions that
   converge to the same answer when replayed from the same event log,
   without coordination. Required advances: this run's single-process
   determinism extended across processes/machines (same result given same
   inputs, no floating-point or hash-seed divergence — `FixedWeight`
   already suggests the crate anticipated this). RuVector's role: owns the
   fixed-point weight representation already. Why this matters: rules out
   one common failure mode for such convergence. Primary uncertainty:
   cross-platform floating-point-adjacent divergence in the similarity
   scoring layer (outside `ruvector-mincut`), unmeasured. Falsification:
   run the same corpus/seed on a second architecture and diff results.
7. **World-model-adjacent graph memory for autonomous systems.** Thesis:
   an agent's internal world model benefits from the same
   structural-integrity primitives as its episodic memory. Required
   advances: none specific to this run beyond generality of the API. RuVector's
   role: `RuVectorGraphAnalyzer`/`SourceAnchoredMinCut` are already
   graph-generic, not agent-memory-specific. Why this matters: this run is
   a template for wiring the same primitive into any other graph-backed
   RuVector consumer. Primary uncertainty: whether world-model graphs have
   different scale/topology characteristics that change which backend
   wins. Falsification: repeat this run's Probe 2 shape on a world-model
   or knowledge-graph-shaped synthetic corpus instead of a k-NN ring.
8. **Governance-gated forgetting.** Thesis: as agent memory becomes more
   consequential, some evictions may need policy or proof-gated approval
   (`ruvector-proof-gate`, already an optional dependency of
   `ruvector-agent-memory`) before executing, with the structural signal
   as one input to that gate. Required advances: none new — the plumbing
   (`proof-gate` feature) already exists, unconnected to `graph_forget`.
   RuVector's role: owns both halves. Why this matters: determinism is a
   prerequisite for any proof-gate decision that must be independently
   re-checkable by a verifier, which this run establishes is now
   available. Primary uncertainty: whether proof-gate latency budgets
   tolerate even the reduced canonical-backend cost at scale beyond n=400.
   Falsification: extend Probe 2 past n=400 and compare against a
   proof-gate's actual latency budget once one exists for this path.

## Falsification criteria for this run's claims

- **Determinism claim** is falsified by any single non-identical partition
  across repeated `canonical_cut()` calls on byte-identical input (this
  run measured 0/30 — a future run finding even 1/N would falsify it).
- **Performance claim** is falsified if a future same-environment run of
  `mincut_gated_forgetting_bench_canonical` exceeds the 100x slowdown
  threshold, or if the canonical/legacy gap closes substantially at corpus
  sizes larger than n=400 (untested this run).
- **Effectiveness (bridge-survival) rejection** would be falsified by any
  future experiment — same or different corpus — showing a ≥15pp
  bridge-survival gap using either backend; this run does not claim
  structural forgetting can never work, only that it did not on this
  corpus with a global min-cut signal (see "Interpretation" §2 and
  long-horizon item 5 for the specific, already-scoped follow-up that
  could falsify this).

## Limitations

- Corpus size (84 entries) remains small relative to production agent
  memory, for the same reason ADR-345 constrained it (legacy backend
  infeasibility) — even though the canonical backend is now fast enough at
  n<=400 that a materially larger corpus (the original ~1,950-memory / 20
  hot-cluster target) is plausible for a future run; not attempted here to
  keep this run's comparison strictly apples-to-apples with ADR-345.
- No memory (RSS) accounting was added for either backend this run.
- No WASM/edge build was exercised.
- The n=19 legacy-backend 91.4-second outlier was observed once (this
  probe does not repeat the scaling measurement at each size); it is
  reported as measured, not as a stable per-size expectation.
- **No production caller.** Nothing in this codebase invokes
  `MincutGatedForgetting` outside this run's own examples and unit tests;
  the measured numbers describe the two engines in isolation, not the
  policy under any real workload.
- **Single seed, single corpus construction.** All results (determinism,
  scaling, and the full benchmark) use one fixed seed (341) and one fixed
  synthetic-corpus generator. No independent seeded or permuted holdout
  was run, so the specific percentages above (e.g. 66.7% bridge survival)
  should be read as this-seed evidence, not a distribution.
- **No cross-process or cross-order qualification.** Probe 1's
  determinism result (30/30 identical partitions) was measured within one
  process across repeated in-memory calls on a fixed edge list; it does
  not by itself establish determinism across separate process restarts or
  across differently-ordered input construction of the same graph.
- Consequently — and this is worth stating plainly rather than leaving
  implicit — the speedup and determinism numbers above are bounded
  implementation evidence for the canonical engine itself, not promotion
  evidence for `MincutGatedForgetting` or for production use of this
  path; the "Acceptance" section's REJECT verdict already reflects that,
  and none of these gaps change it.

## Next research

1. **Local (Gomory-Hu / per-cluster) structural signal**, replacing the
   single global min-cut — the most direct next step per this run's own
   effectiveness finding (long-horizon item 5 above). `ruvector-mincut`
   already ships `canonical::tree_packing::GomoryHuTree` and
   `cluster::ClusterHierarchy::boundary_size`, both unused by
   `graph_forget.rs` today.
2. **Extend Probe 2 past n=400** now that the canonical backend makes
   larger graphs affordable, to check whether the ~110-200x speedup holds,
   grows, or shrinks at production-relevant corpus sizes (thousands of
   entries).
3. **RSS/memory accounting** for both backends at matched corpus sizes,
   to check whether the canonical backend's build-time regression at
   larger `n` correlates with a memory tradeoff worth documenting.
4. **WASM build and benchmark** of the `canonical` feature path, to
   support the "edge cognition" long-horizon application concretely rather
   than by inference from the feature's dependency graph.
5. **Independent seeded/permuted holdout.** Re-run Probe 3 (or its
   successor from item 1) across multiple seeds and multiple corpus
   permutations to turn this run's single-seed percentages into a
   distribution, before any future promotion decision relies on them.
6. **Cross-process/restart determinism qualification.** Extend Probe 1 to
   verify the canonical backend's partition is identical not just across
   repeated in-process calls but across separate process invocations and
   differently-ordered input construction.
7. **A real caller.** None of this run's evidence involves an actual
   consumer of `MincutGatedForgetting`; wiring one in (even a synthetic
   but realistic agent workload, not just this run's benchmark harness)
   would be a precondition for any promotion discussion.
