# ADR-305: Structural-Time Conflict Resolution for Concurrent Multi-Agent Memory

**Date**: 2026-08-14
**Status**: Accepted — PoC merged as an experimental crate; production integration requires the follow-ups in "Migration"
**Deciders**: Nightly research agent
**Tags**: agent-memory, emergent-time, coherence, crdt, multi-agent, ruvector-structural-memory-merge

---

## Context

RuVector positions itself as a substrate for autonomous, multi-agent systems (`docs/research/nightly/2026-06-13-temporal-coherence-agent-memory`, `2026-06-14-agent-memory-compaction`). Those two nightlies both address **single-agent** memory: how one agent ranks or compacts its own memory over time.

Neither addresses what happens when **multiple agents share a memory namespace and write concurrently** — e.g. a swarm of edge agents (Cognitum Seed devices, ruFlo workers) updating a shared belief about the same entity without a synchronized wall clock or a single sequencer. RuVector has no answer to: *when two agents' writes to the same memory key conflict, which one survives?*

The default answer in eventually-consistent systems is **last-write-wins (LWW) by wall-clock timestamp**. It has two known problems, both directly relevant to RuVector's edge/agent-OS ambitions:

1. **Clock skew.** Unsynchronized edge devices do not share NTP-grade clocks. A device with a persistent forward clock bias wins every conflict it participates in, regardless of causal order or content.
2. **Semantic blindness.** LWW (and its safer cousin, vector-clock LWW with an arbitrary tiebreak) has no notion of *which write is more valuable* — it can only ever look at metadata, never content.

RuVector already has `crates/emergent-time`, which defines **structural proper time**: a clock that measures how much a system's state actually moved (`τ = f(Δv, ΔS, ΔG, ΔC, ΔE)` over embedding, entropy, graph, coherence, and prediction-error channels), rather than counting wall-clock ticks. It also has `crates/ruvector-agent-memory::scoring`, which scores memories by coherence against an active context window. Neither had been connected to the multi-agent conflict-resolution problem before this nightly.

---

## Hypothesis

> Given 2000 genuinely concurrent (causally-unordered, per vector clock) synthetic memory-write conflicts across 6 agents, each write carrying a hidden ground-truth quality label never exposed to any policy,
>
> when conflicts are resolved by `StructuralCoherenceMerge` (respects causal order exactly; for truly concurrent writes, prefers the one with larger structural-proper-time magnitude weighted by coherence against the current context) instead of `LwwWallClock` (400ms per-agent clock skew) or `LwwVectorClock` (agent-id tiebreak on concurrent writes),
>
> then its correct-resolution rate (picks the write with the higher hidden quality) should exceed both baselines by ≥ 10 percentage points,
>
> subject to zero causal-order violations on a 500-case causally-ordered control set (all three policies to the extent they are logically capable of it) and merge throughput within 5× of the wall-clock baseline.

---

## Decision

**Accept.** Implement `crates/ruvector-structural-memory-merge` as a new, small (4 files, 936 lines total, none over 500) workspace crate. It depends only on `emergent-time` (already zero-dependency) and the Rust standard library.

Three `MergePolicy` implementations were built and measured against each other on identical synthetic data:

| Policy | Causal order | Concurrent tiebreak |
|---|---|---|
| `LwwWallClock` (baseline) | **Not respected** — picks the later `wall_ts_ms`, which clock skew can invert | later `wall_ts_ms` |
| `LwwVectorClock` (variant A) | Always respected | higher `agent_id` (content-blind) |
| `StructuralCoherenceMerge` (variant B, candidate) | Always respected | higher `τ · (0.5 + 0.5·coherence)` |

`τ` is `emergent_time::structural_clock::StructuralProperTime::tick(prev_snapshot, snapshot)` — reused directly from the existing crate, not reimplemented. `coherence` is a local f64 cosine-similarity-to-context function, structurally identical to `ruvector-agent-memory::scoring::coherence_score` but operating on the `StateSnapshot` embeddings `StructuralProperTime` already consumes.

---

## Evidence

**Command**: `cargo run --release -p ruvector-structural-memory-merge --bin structural-memory-merge-bench`
**Hardware**: x86-64 Linux 6.18.5-fc-v20, `rustc 1.94.1`, release profile (`opt-level=3, lto=fat, codegen-units=1, strip=true`).
**Seed**: `0xA6E17` (deterministic; `scenario::tests::deterministic_for_fixed_seed` pins this).
**Corpus**: 2000 concurrent conflicts + 500 causally-ordered control pairs, 6 agents, dim 16.

| Skew | Policy | Correct-resolution rate | Mean quality regret | Causal violations (of 500) | Merges/sec |
|---|---|---|---|---|---|
| 0ms | LwwWallClock | 50.3% | 0.1752 | 0 | 13,980,367 |
| 0ms | LwwVectorClock | 50.6% | 0.1716 | 0 | 13,623,369 |
| 0ms | **StructuralCoherenceMerge** | **86.8%** | **0.0162** | 0 | 4,324,119 |
| 400ms | LwwWallClock | 48.5% | 0.1784 | **19** | 12,098,078 |
| 400ms | LwwVectorClock | 50.6% | 0.1716 | 0 | 12,153,589 |
| 400ms | **StructuralCoherenceMerge** | **86.8%** | **0.0162** | 0 | 4,338,379 |
| 2000ms | LwwWallClock | 48.6% | 0.1796 | **171** | 12,348,027 |
| 2000ms | LwwVectorClock | 50.6% | 0.1716 | 0 | 12,218,791 |
| 2000ms | **StructuralCoherenceMerge** | **86.8%** | **0.0162** | 0 | 4,285,081 |

Acceptance check at the pre-registered 400ms realistic-skew condition, printed by the binary itself:

```
- beats LwwWallClock by >=10pp: true (+38.3pp)
- beats LwwVectorClock by >=10pp: true (+36.2pp)
- zero causal-order violations (vclock & structural): true (vclock=0, structural=0)
- throughput within 5x of wall-clock baseline: true (4338379 vs 12098078 merges/sec)

VERDICT: ACCEPT
```

**Honest caveat on the margin** (this is load-bearing, not decoration): both LWW baselines are *structurally incapable* of reading write content, so they sit at ≈50% (chance) by construction on a two-way choice — that is not a weak baseline being beaten unfairly, it is the actual behavior of the two most common real-world defaults (naive LWW and vector-clock LWW), but it does mean the reported +36–38pp margin should be read as "content-aware beats content-blind under this noise model," not as evidence `StructuralCoherenceMerge` is close to any information-theoretic ceiling. No ablation against coherence-only or τ-only variants was run this cycle (see "Open Questions").

**Wall-clock skew directly causes causal-order violations**, and the effect scales with skew magnitude: 0 at 0ms, 19/500 (3.8%) at 400ms, 171/500 (34.2%) at 2000ms. `LwwVectorClock` and `StructuralCoherenceMerge` have **zero** violations at every skew level tested, by construction (they consult the vector clock before ever considering wall time or structural score).

**Build/test status**: `cargo build --release -p ruvector-structural-memory-merge` clean; `cargo test --release -p ruvector-structural-memory-merge` — 10/10 passed; `cargo clippy --release -p ruvector-structural-memory-merge --all-targets` — clean (no warnings after two style fixes: `derive(Default)`, collapsed `if`/`else if`). Release binary size: 390,672 bytes (stripped).

---

## Consequences

**Positive**:
- RuVector gains a second real integration point for `emergent-time` beyond its own crate (the first, self-contained, use was the drift-to-failure early-warning benchmark in `emergent-time` itself) — evidence that structural time is a reusable primitive, not a one-off.
- Establishes an honest experimental pattern (hidden ground truth, independent noise channels, a causally-ordered control group) for testing any future "smarter conflict resolution" claim in this codebase, reducing the risk of circular benchmarks.
- Directly serves the "swarm memory" and "agent operating system" long-horizon theses in `CLAUDE.md`'s ecosystem map.

**Negative / costs**:
- `StructuralCoherenceMerge` is ~2.8–3.2× slower per resolution than either LWW baseline (still >4.2M resolutions/sec in this microbenchmark, so not a practical bottleneck at plausible RuVector agent-memory write rates, but non-zero).
- Requires every write to carry a `StateSnapshot` (5 extra scalars + an embedding) and a `VectorClock` (one `u64` per known agent) — more metadata than a bare timestamp.
- Vector clocks grow with the number of distinct agents that have ever written to a namespace; this PoC does not implement pruning/compaction of stale agent entries (a known, standard vector-clock cost, not unique to this design).

---

## Alternatives

1. **Pure vector-clock LWW with a smarter tiebreak** (e.g. random, round-robin) instead of agent-id — rejected for evaluation here because any metadata-only tiebreak is, by the nature of "genuinely concurrent," blind to content and will sit at ≈50% correct-resolution on this benchmark's two-way choice; agent-id was chosen as a representative, deterministic instance of that whole family rather than re-testing several equivalent-in-expectation variants.
2. **CRDT-style merge (keep both writes, let the reader reconcile)** — avoids picking a "loser" entirely. Legitimate for some data types (e.g. counters, sets) but not for a single scalar memory slot where the RuVector agent-memory model expects one active value per key; flagged as a real alternative worth its own nightly if RuVector's memory model moves towards multi-value registers.
3. **Coherence-only (no τ) or τ-only (no coherence) tiebreak** — not built this cycle; see "Open Questions."
4. **Full CRDT libraries (e.g. `automerge`-style)** — rejected as out of scope: RuVector's memory model is single-writer-per-key by convention, and pulling in a general CRDT engine is a much larger dependency and design commitment than this PoC's question warrants.

---

## Implementation Plan / API Shape

Already implemented in this PR (`crates/ruvector-structural-memory-merge`):

```rust
pub trait MergePolicy {
    fn name(&self) -> &'static str;
    fn resolve(&self, a: &MemoryWrite, b: &MemoryWrite, context: &[Vec<f64>]) -> Decision;
}

pub struct MemoryWrite {
    pub agent_id: AgentId,
    pub key: MemoryKey,
    pub wall_ts_ms: f64,
    pub vclock: VectorClock,
    pub prev_snapshot: StateSnapshot, // emergent_time::structural_clock::StateSnapshot
    pub snapshot: StateSnapshot,
}

pub struct Decision {
    pub winner: Winner,
    pub reason: &'static str,
    pub tau_a: f64,
    pub tau_b: f64,
    pub causal_order: CausalOrder,
}
```

No feature flags were needed — the crate has no optional functionality and no unsafe code.

---

## Security / Governance

- Pure computation over caller-supplied data; no I/O, no network, no filesystem access, no unsafe blocks.
- `Decision` records `reason` and both `τ` values, which is enough to *audit* a merge after the fact, but this PoC does **not** implement tamper-evidence (no hash chaining). Production hardening should route `Decision` records through `ruvector-proof-gate` (writes) and/or `ruvector-retrieval-receipt` (reads) rather than reimplementing witness chaining here — this ADR explicitly does not claim tamper-evidence.
- `VectorClock` growth is unbounded in agent count; a production integration must define a pruning/GC policy (standard vector-clock operational concern) before this ships on a long-lived namespace.

---

## Failure Modes

- If an agent's `StateSnapshot` is fabricated or manipulated by a malicious participant, `StructuralCoherenceMerge` can be steered to prefer that agent's writes — this PoC assumes honest-but-uncoordinated agents, not Byzantine ones. A Byzantine-robust variant is out of scope here.
- If `context` (the coherence comparison window) is empty or degenerate, `StructuralCoherenceMerge` falls back to pure `τ` comparison (`coherence_score` returns `0.0`), which is a graceful but less-informed degradation, not a crash.
- Two writes with identical `τ` and `coherence` scores fall through to whichever compares `>=` first (`Winner::A`) — a silent, deterministic tiebreak, not a panic.

---

## Migration / Rollback

- Additive only: new crate, new workspace member line, no changes to any existing crate's public API or behavior. Rollback is `git revert` of this PR with no other side effects.
- Not wired into `ruvector-agent-memory`, MCP, or ruFlo in this PR — this is a PoC-level integration, not a production migration. See "Practical Applications" in the nightly README for the concrete next step (an `ruvector-agent-memory` feature flag exposing `StructuralCoherenceMerge` as an optional multi-writer mode).

---

## Rejection Criteria (for this ADR, going forward)

This design should be reconsidered / reverted if a future nightly or production integration shows:
- The correct-resolution advantage collapses (e.g. below LwwVectorClock) on a *realistic* (not synthetic) multi-agent memory corpus.
- Vector-clock metadata growth becomes a measured production cost RuVector cannot amortize.
- A Byzantine or adversarial multi-agent setting is in scope, where this design's honest-agent assumption is violated.

---

## Open Questions

1. **Ablation**: does `τ` alone, or `coherence` alone, capture most of the win, or is the product genuinely better than either signal? Not measured this cycle.
2. **Real corpus**: this PoC's ground truth is synthetic-by-construction (a hidden `alpha` driving correlated noisy observables). A follow-up should validate against a real multi-agent memory trace (e.g. from an actual ruFlo swarm run) if one becomes available.
3. **Byzantine robustness**: what does `StructuralCoherenceMerge` do under an adversarial agent deliberately inflating its own `τ`/`coherence`? Not addressed; flagged as a hard constraint before any production use outside a trusted-agent-set.
