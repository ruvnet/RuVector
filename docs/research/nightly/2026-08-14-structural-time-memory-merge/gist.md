# Resolving Concurrent Memory Writes Without a Shared Clock

## Problem

Any system where more than one autonomous process can write to the same piece of shared state
eventually has to answer: two writes conflict, which one wins? The standard answer — last-write-wins
(LWW) by wall-clock timestamp — quietly assumes synchronized clocks and throws away all information
about *what* was written, keeping only *when*.

For a Rust-native "agent memory" substrate like RuVector, this stops being a theoretical concern the
moment more than one agent (a ruFlo worker, an edge device, an independent MCP client) can update the
same memory key. Unsynchronized edge devices routinely drift by hundreds of milliseconds to seconds;
under that drift, naive LWW doesn't just occasionally pick the "wrong" write by some fuzzy quality
measure — it can pick a write that is *provably, causally* earlier than the one it discards.

## Hypothesis

Two independent RuVector primitives, never previously connected, look like a plausible fix:

- **`emergent-time`'s structural proper time** — a clock that measures how much a system's state
  actually changed (embedding movement, entropy, graph topology, coherence, prediction error),
  instead of counting wall-clock ticks.
- **Coherence scoring** — how well a piece of content matches the current context, already used in
  `ruvector-agent-memory` to rank a single agent's own memories.

The hypothesis: for writes that are *genuinely concurrent* (no causal relationship a vector clock
can establish), preferring the write with the larger structural shift, weighted by its coherence
with the current shared context, should recover more of the "actually more valuable" write than
either wall-clock LWW or a vector-clock LWW with an arbitrary tiebreak — while never overriding
real causal order, which the LWW-by-wall-clock approach can do under skew.

## Technical Design

`crates/ruvector-structural-memory-merge` implements three `MergePolicy` variants:

```rust
pub trait MergePolicy {
    fn resolve(&self, a: &MemoryWrite, b: &MemoryWrite, context: &[Vec<f64>]) -> Decision;
}
```

- `LwwWallClock` — winner is whichever write has the larger `wall_ts_ms`. Never checks the vector
  clock.
- `LwwVectorClock` — checks the vector clock first; if one write causally precedes the other, the
  later one wins unconditionally. Only for genuinely concurrent writes does it fall back to a
  tiebreak (here: higher `agent_id`) — a rule that's deterministic but carries no information about
  content.
- `StructuralCoherenceMerge` — same causal-order gate as `LwwVectorClock`, but the concurrent-case
  tiebreak is `τ · (0.5 + 0.5·coherence)`, where `τ` is `StructuralProperTime::tick(prev, cur)` on
  each write's five-channel state snapshot, and `coherence` is cosine similarity between the new
  write's embedding and the current shared context window.

The causal-order gate is the load-bearing design choice: structural time is used *only* to break
ties among writes a vector clock cannot order, never to override a real happens-before relationship.

## Honest Evaluation Design

The hardest part of this kind of benchmark is avoiding circularity — if "correct" is defined in
terms of the same signal the algorithm uses, the algorithm cannot lose. This PoC avoids that by
generating a hidden per-write `alpha` (a structural-shift magnitude towards a drifting, unobserved
"ideal" target) that drives three *independently noised* observable channels: the structural
snapshot, a separately-sampled coherence context window, and a skewed wall-clock timestamp.
`true_quality = alpha + independent noise` is the evaluation label; no `MergePolicy` implementation
ever sees `alpha` or `true_quality`, only the noisy, imperfect observables. A separate 500-pair
causally-ordered control set, constructed so the correct winner is deterministic by vector-clock
construction (independent of any quality signal), isolates "does this policy ever break real causal
order" from "does this policy pick the better concurrent write" — two different failure modes,
reported separately.

## Actual Implementation and Evidence

936 lines across four files (none over 500), one dependency (`emergent-time`, itself dependency-free). Full test
suite: 10/10 passing; `cargo clippy --all-targets`: clean.

Measured with `cargo run --release -p ruvector-structural-memory-merge` (x86-64 Linux, `rustc
1.94.1`, release profile, seed `0xA6E17`, 2000 concurrent conflicts + 500 causal-control pairs, 6
agents):

| Skew | Policy | Correct-resolution rate | Causal violations (of 500) |
|---|---|---|---|
| 400ms | LwwWallClock | 48.5% | 19 |
| 400ms | LwwVectorClock | 50.6% | 0 |
| 400ms | **StructuralCoherenceMerge** | **86.8%** | **0** |

Both LWW baselines sit at roughly chance (50%) on correct-resolution rate at every skew level
tested (0/400/2000ms) — expected, since they are structurally incapable of reading write content.
`StructuralCoherenceMerge` beats both by 36–38 percentage points while never violating causal order,
at 2.8× the per-resolution cost of either baseline (still >4.2M resolutions/sec). Wall-clock LWW's
causal-order violation count scales directly with injected skew: 0 at 0ms, 19/500 at 400ms, 171/500
at 2000ms — a direct, measured demonstration of the clock-skew failure mode this design targets.

Acceptance was pre-registered before the run (`src/main.rs` doc comment): beat both baselines by
≥10pp, zero causal violations, throughput within 5× — all four conditions held. **VERDICT: ACCEPT.**

## Limitations

The evaluation corpus is synthetic, not drawn from a real multi-agent memory trace. No ablation was
run to separate how much of the win comes from `τ` versus coherence individually. The design assumes
honest agents — an agent that fabricates its own `StateSnapshot` to inflate `τ` is not defended
against here; that requires RVM-style attestation, not addressed in this cycle. No WASM build was
measured, only argued to be plausible from the dependency graph. None of this is fixed in this PR;
all of it is listed as explicit follow-up work in ADR-305.

## Production Relevance

This does not ship as a default in `ruvector-agent-memory` — it ships as a standalone, opt-in crate
with a clear promotion path (ablation → real-trace validation → Byzantine hardening → feature-flagged
integration) documented in ADR-305. The most direct next step is a ruFlo workflow that calls
`MergePolicy::resolve` whenever a shared-memory namespace detects a write conflict, replacing
whatever ad hoc LWW the underlying store does today.

## RuVector Ecosystem Implications

This is the second real (non-benchmark-only) use of `emergent-time`'s structural time inside the
workspace, and the first to connect it to the agent-memory subsystem. It gives RuVector a measured,
non-hand-wavy answer to "how does shared agent memory behave when more than one agent can write to
it" — a gap none of the prior agent-memory nightlies addressed, and one every multi-agent RuVector
deployment (ruFlo swarms, edge fleets) will eventually need an answer to.

## Future Direction

1. Ablate the `τ` and coherence signal components independently.
2. Validate against a real multi-agent memory trace instead of synthetic ground truth.
3. Add RVM-attested snapshots before any untrusted-multi-tenant deployment.
4. Wire a feature-flagged multi-writer mode into `ruvector-agent-memory`.
5. Route `Decision` audit records through `ruvector-proof-gate` for tamper-evidence.

## References

- `crates/emergent-time/src/structural_clock.rs`
- `crates/ruvector-agent-memory/src/scoring.rs`
- Lamport, L. "Time, Clocks, and the Ordering of Events in a Distributed System." *Communications
  of the ACM*, 1978.
- `docs/adr/ADR-305-structural-time-memory-merge.md` (full evidence and acceptance record)
