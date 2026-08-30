# Wall-clock recency breaks agent memory compaction during idle gaps — here's a fix using "proper time"

## Problem

Agent memory systems that compact/evict old entries almost universally use
wall-clock recency: keep what was accessed most recently, evict what
wasn't. This works fine when activity is roughly continuous. It breaks when
activity is **bursty** — dense sessions separated by long idle gaps (an
agent waiting on a human, a slow tool call, a scheduled pause, an overnight
gap).

Concretely: if a compaction policy normalizes recency as
`(t - min) / (max - min)` across the whole memory store, a single idle gap
that's orders of magnitude longer than the time actually spent writing
memories will dominate that range. Every memory written *before* the gap
gets crushed toward `recency ≈ 0` — indistinguishable from genuinely old,
irrelevant memories — no matter how close to the gap (and how relevant) it
actually was.

## Hypothesis

Swap the recency signal for something that only advances when the agent's
actual state changes, not when wall-clock time passes. `emergent-time`, an
existing RuVector crate, implements exactly this: **Structural Proper
Time** — internal time as the arc length of a system's trajectory through
its own state space, generalizing the physical notion of proper time. An
idle gap with zero new memories contributes zero structural time.

Pre-registered hypothesis: on a bursty-idle memory workload, a compaction
policy using structural-time recency should out-recall a wall-clock-recency
policy by a meaningful margin, without regressing on ordinary
(non-bursty) workloads.

## Technical design

`ruvector-agent-memory` already has a `CoherencePolicy`:

```text
importance = α·recency + β·frequency + γ·coherence
```

where `recency` comes from `last_accessed_at`. Two new policies keep this
exact formula and swap only the recency term:

```rust
pub struct StructuralTimePolicy {
    pub weights: CoherenceWeights,
    pub metric: StructuralMetric, // only w_embedding is nonzero
}
```

Recency per entry is computed by treating the memory stream (in write
order) as a trajectory of `StateSnapshot`s (embedding only — this crate
has no honest entropy/graph/prediction-error signal per memory, so those
`StructuralMetric` channels are left at zero rather than fabricated), and
taking `StructuralProperTime::cumulative()` over it — the L2 arc length
between consecutive embeddings, summed. `GatedStructuralTimePolicy` adds a
jitter gate: embedding movement below a threshold contributes no
structural time.

Everything else — the weighting formula, the coherence term, the
compaction machinery — is unmodified, reused code from a prior nightly
result. This is a new *composition*, not a new algorithm.

## Actual implementation

New files in `crates/ruvector-agent-memory/`:
`src/temporal_compaction.rs` (246 lines, 3 unit tests),
`examples/temporal_compaction_bench.rs` (the benchmark, 3 embedded
acceptance/robustness tests), plus a small `MemoryStore::advance_clock()`
addition to `src/memory.rs` to simulate an idle gap, and one new dependency:
`emergent-time` (path dependency, zero transitive deps — it's a
dependency-free crate itself).

## Actual benchmark evidence

Synthetic dataset: 2 000 memories, 20 topic clusters × 100 entries, 64-dim.
15 clusters ("phase 1") written densely, then either a 500 000-tick idle
gap (bursty-idle workload) or nothing (steady control), then 5 more
clusters ("phase 2", the agent's current activity). Held-out queries probe
the last 3 phase-1 clusters — what the agent was working on right before
the gap. The compaction context window is drawn only from phase-2 topics,
so the coherence term cannot leak the evaluation answer.

```
[bursty-idle]                     Recall@10
LRU                                   66.7%
CoherenceWeighted (baseline)          27.2%
StructuralTime (candidate A)          59.0%   (+31.8pp vs baseline)
GatedStructuralTime (candidate B)     59.0%   (+31.8pp vs baseline)

[steady control]                  Recall@10
CoherenceWeighted (baseline)          59.0%
StructuralTime (candidate A)          59.0%   (+0.0pp — no regression)
```

Verdict: **ACCEPT** (pre-registered thresholds: ≥3.0pp win on bursty-idle,
≤1.0pp regression tolerance on steady). Direction reproduces across 8
independent random seeds. FNV-1a witness hash over
`(seed, params, rounded recall values)`: `e0d3b9cf5b37176e` — re-running
`cargo run --release -p ruvector-agent-memory --example
temporal_compaction_bench` reproduces it exactly.

One disclosed, honest wrinkle: plain `LruPolicy` (no coherence term at all)
also outperforms `CoherenceWeighted` on the bursty-idle workload (66.7% vs
27.2%). That's not the paper's claim — LRU isn't a like-for-like
comparison, it just happens to keep a clean "most recent N" cut that
covers 2 of 3 recall clusters by construction. It's reported because it's
informative: when the coherence term's context window is unrelated to
what's later queried (the realistic case — an agent's *current* topic
isn't necessarily what it'll be asked to recall), it doesn't just fail to
help under an idle gap, it actively degrades an otherwise-cleaner recency
cutoff by tying together already wall-clock-crushed candidates into an
unstable tiebreak.

## Limitations

Synthetic dataset only, one gap magnitude, one dataset shape (single gap,
two topic-disjoint phases), no honest multi-channel structural metric (only
embedding), and `GatedStructuralTimePolicy` is not shown to add anything
over the ungated version on this dataset (the gate never binds at this
noise level). None of these are claimed to be solved by this run.

## Production relevance

Ships as an additional, independently-selectable `CompactionPolicy` — no
feature flag, no default-path change, no migration required to adopt it
experimentally. Not recommended for default-on promotion without
validation against real (not synthetic) agent-memory traces first.

## RuVector ecosystem implications

Connects `ruvector-agent-memory` (compaction), `emergent-time` (the clock),
vector coherence scoring, and witness/provenance (reusing `emergent-time`'s
own FNV-1a hash for the benchmark's evidence seal) — four existing pieces,
zero new primitives. A plausible RVF extension: serialize per-entry
structural-time values into a portable memory snapshot so a restored agent
recomputes compaction consistently across a suspend/resume boundary without
needing wall-clock alignment with its prior instance. No RVM fit identified
(this is a scoring-function swap, not an isolation/capability boundary). A
concrete ruFlo workflow: trigger structural-time-based compaction
specifically when an idle gap is detected (via `emergent-time`'s own
change-point detectors), and ordinary `CoherencePolicy` otherwise.

## Future direction

Validate on real multi-day agent traces; wire a second honest structural
channel (e.g. a mincut-derived `ΔG`); characterize the effect as a function
of gap-to-activity ratio; test multi-gap traces.

## References

Page & Wootters, "Evolution without evolution" (1983); DeWitt (1967);
Connes & Rovelli, thermal time (1994) — via `emergent-time`
(`crates/emergent-time/README.md`, `docs/adr/ADR-251-agentic-time.md`).
Full report: `docs/research/nightly/2026-08-29-structural-time-memory-compaction/README.md`.
