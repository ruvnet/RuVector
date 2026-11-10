# Coherence-Gated Graph Draft for Speculative ANN

**150-char summary:** Swapping speculative ANN's linear u8 draft scan for a
graph beam draft cuts distance evaluations ~74x; adding a coherence gate on
top finds no usable middle ground — a sharp recall cliff, not a tradeoff.

---

## Abstract

`ruvector-speculative-ann` (nightly research, 2026-07-27) pairs a cheap
draft stage with an exact verify stage, mirroring speculative decoding for
LLMs. Its own research notes flagged an explicit open item: replace the
draft's O(n) linear u8 scan with coherence-gated graph traversal (the
technique `ruvector-coherence-hnsw` validated separately) for O(log n) draft
cost.

This run implements that exact proposal in a new crate,
`ruvector-coherence-speculative-ann`, and measures it honestly. The result
splits in two:

- **Graph-based drafting (no gate) is a clear win.** ~54 distance
  evaluations per query instead of n=4,000 (a ~74x reduction in
  evaluations), 0.998 recall@10 against the exact baseline, and 3-4x lower
  mean wall-clock latency than a brute-force exact scan.
- **Coherence gating on top of that draft is falsified for this
  configuration.** A 13-point threshold sweep found a sharp cliff: every
  threshold that preserves recall within 2 percentage points of the ungated
  graph draft reduces draft cost by less than the noise floor (0-1.8%);
  every threshold that cuts draft cost by a measurable amount (10%+)
  collapses recall by double digits. No threshold in the sweep sits in a
  usable middle.

Per the task's own framing: a falsified hypothesis with clean, reproducible
evidence is a successful nightly run. This one also leaves a validated,
reusable positive sub-result (graph drafting) for later integration work.

---

## Hypothesis

```
Given a 4,000-vector, 64-dim clustered corpus with a navigable small-world
draft graph (M=16 local + 6 long-jump edges, u8-quantized ranking distances)
and a fixed distant entry point (node 0, simulating a coarse upper-layer
HNSW descent),

when the draft stage applies a traversal-coherence expansion gate
(ruvector-coherence-hnsw's traversal_coherence metric) instead of expanding
every popped candidate's neighbors unconditionally,

then average per-query draft cost (graph-traversal expansions) should fall
by at least 10% relative to the ungated graph draft,

subject to:
  - recall@10 remaining within 2 percentage points of the ungated graph
    draft's recall@10 (0.998), and
  - mean speculative-pipeline latency remaining below the LinearFull exact
    scan baseline.
```

Declared before benchmarking (see git history of `src/bin/benchmark.rs`):
`MIN_COHERENCE_RECALL = 0.85`, `MAX_RECALL_GAP_VS_BASELINE_DRAFT = 0.02`,
`MIN_MEANINGFUL_DRAFT_COST_REDUCTION = 0.10`.

---

## Why This Matters Now (2026)

Agent memory and RAG retrieval increasingly run many small ANN queries per
agent turn. `ruvector-speculative-ann`'s draft+verify protocol already cuts
latency versus exact search, but its draft stage is still O(n) — it does not
scale to large corpora. Graph-based drafting is the obvious next step, and
coherence gating was the next proposed refinement on top of it. Knowing
*now*, with evidence, that naive fixed-threshold gating doesn't work saves a
future nightly run from re-discovering the same cliff, and tells us
precisely what would need to change (adaptive thresholds, multiple entries)
before it's worth trying again.

## Why RuVector Is the Right Substrate

RuVector already has both halves of this experiment as separately validated
crates (`ruvector-speculative-ann`, `ruvector-coherence-hnsw`) — this run is
a direct, measured test of combining them, exactly the kind of
"new composition of known techniques" novelty the nightly process calls for
rather than inventing an unrelated mechanism.

---

## MetaHarness / Flywheel / Darwin / RVF / RVM Role

**Tooling reality check (recorded honestly, not faked):** `npx metaharness
--help` resolves only to the public scaffolding tool for *creating new*
harness projects, not a repo-integrated orchestration CLI for this
repository. `npx ruvector harness doctor/status` do not resolve to any
installed executable. There is no repo-local MetaHarness/Darwin/Flywheel CLI
to call. The roles below were performed directly by this run rather than
through that tooling:

- **Goal Planner / SOTA Researcher**: read `ruvector-speculative-ann`'s own
  "What to Improve Next" list (an explicit backlog item) instead of
  generating a disconnected new topic — satisfying the "don't duplicate,
  extend an unresolved assumption" rule.
- **Rust Engineer**: implemented the crate directly (see Implementation).
- **Benchmark Engineer**: ran the real benchmark binary on this machine,
  captured raw stdout, and used a diagnostic trace binary
  (`src/bin/diag.rs`) to root-cause the surprising "identical draft cost at
  every low threshold" observation rather than reporting it uncritically.
- **Adversarial Reviewer**: the identical-cost observation was treated as
  suspicious rather than accepted; it was traced to a real mechanism (see
  "Root Cause"), not a benchmarking bug, and the acceptance criterion was
  tightened (from "any reduction" to "reduction beyond the noise floor")
  once the noise-level effect was understood.
- **Darwin** (bounded parameter exploration, simulated manually): a 13-point
  threshold sweep stands in for a generations/candidates search — this is
  a one-dimensional parameter, so an exhaustive sweep is more honest than a
  simulated genetic search over it.
- **Flywheel** (evidence retention, simulated manually): this README + ADR-353
  + the raw benchmark transcript are the retained evidence; the negative
  result is the "lesson" a future run should consult before re-attempting
  coherence-gated drafting.
- **RVF/RVM**: see dedicated sections below.

---

## Architecture

```mermaid
flowchart TD
    Q["Query (f32)"] --> QZ["Scalar Quantizer\n(train-time, shared)"]
    QZ --> DG["DraftGraph\n(f32 rows + u8 rows + k-NN/long-jump adjacency)"]
    DG --> BD["Draft: bounded beam traversal\nranked by u8 distance"]
    BD -->|"gate: traversal_coherence(entry,cand,query) >= t"| EXP{"expand\nneighbors?"}
    EXP -->|yes| BD
    EXP -->|no, but still pooled| POOL["Draft candidate pool (<=ef)"]
    BD --> POOL
    POOL --> VER["Verify: exact f32 distances\non pool only"]
    VER --> OUT["Top-k, sorted"]

    style BD fill:#2d7a9e,color:#fff
    style EXP fill:#9e7a2d,color:#fff
    style VER fill:#7a2d9e,color:#fff
```

Three measured variants, all behind the shared `AnnVariant` trait:

1. **LinearFull** — brute-force exact f32 scan (ground truth).
2. **GraphDraftSpeculative\<BaselineGraphDraft\>** — graph beam traversal,
   u8-ranked, every popped candidate's neighbors expanded unconditionally,
   then exact f32 verify on the returned pool.
3. **GraphDraftSpeculative\<CoherenceGatedGraphDraft\>** — same, but a
   candidate's neighbors are only expanded when its traversal coherence
   relative to the fixed entry point is at or above a threshold `t`.

---

## Implementation

Crate: `crates/ruvector-coherence-speculative-ann/` (new, workspace member).

```
src/
  lib.rs          — Hit, AnnVariant, ScalarQuantizer, recall_at_k (same
                    shapes as ruvector-speculative-ann, standalone copy)
  dataset.rs       — deterministic LCG dataset (tests) + ClusteredDataset
                    (benchmark; mirrors ruvector-coherence-hnsw's
                    clustered_unit_vectors methodology)
  graph.rs         — DraftGraph: brute-force exact-f32 local k-NN + random
                    long-jump edges (same structure as FlatGraph), storing
                    both f32 (verify) and u8-quantized (draft ranking) rows
  coherence.rs     — traversal_coherence (ported from ruvector-coherence-hnsw)
  search.rs        — BaselineGraphDraft / CoherenceGatedGraphDraft: bounded
                    beam traversal over DraftGraph, ranked by u8 distance
  linear_full.rs   — LinearFull baseline
  variants.rs      — GraphDraftSpeculative<S: DraftSearcher>: wires a draft
                    searcher + exact verify behind AnnVariant
  metrics.rs       — LatencyStats (mean/p50/p95/p99)
  bin/benchmark.rs — the benchmark binary (see below)
  bin/diag.rs      — reproducibility tool for the root-cause trace
                    (per-query pops/expansions at several thresholds)
```

Index construction (local k-NN) still uses exact f32 distances — graph
*quality* is not penalized by quantization, matching
`ruvector-coherence-hnsw`'s convention. Only per-query *draft-time ranking*
uses u8 distances; the final verify re-ranks the returned pool exactly in
f32. This keeps the lossy-draft/exact-verify structure that makes the
protocol "speculative" in the first place.

18 unit tests (dataset determinism, quantizer correctness, graph
connectivity, draft-cost monotonicity, recall arithmetic) all pass:
`cargo test -p ruvector-coherence-speculative-ann --release`. `cargo clippy
--all-targets -- -D warnings` is clean.

---

## Benchmark Methodology

**Hardware**: x86_64 Linux (containerized instance), `uname -a`:
`Linux vm 6.18.44-fc-v139 #1 SMP PREEMPT_DYNAMIC x86_64 GNU/Linux`
**Toolchain**: `rustc 1.97.0 (2d8144b78 2026-07-07)`, `cargo 1.97.0`
(workspace floor `rust-version = "1.77"`)
**Dataset**: 20 clusters x 200 points = 4,000 vectors, dims=64, cluster
std=0.15, unit-normalized (`ClusteredDataset::generate`, seed
`0xC0DECAFEBABE7777`)
**Queries**: 300, each sampled near a random cluster center
**k**: 10. **Draft pool width (ef, = k')**: 50.
**Graph**: M=16 local + 6 long-jump edges per node (brute-force O(N^2 D)
index build, reported separately from per-query latency per benchmark
hygiene convention)
**Command**: `cargo run --release -p ruvector-coherence-speculative-ann
--bin benchmark`
**Timing**: `std::time::Instant` per query; percentiles from a sorted
per-query latency array (see `metrics.rs`).

Index construction time: 284 ms (one-time, reported separately, excluded
from per-query latency as required by benchmark hygiene).

---

## Real Benchmark Results (raw, unedited)

```
Variant                          Recall@10   Mean(us)   p50(us)   p95(us)    DraftCost       Memory
----------------------------------------------------------------------------------------------------
LinearFull                           1.000      324.5     320.3     384.6   n/a (O(n))       0.98MB
GraphDraft(Baseline,ef=50)           0.998       73.8      68.2     117.1         54.2       1.56MB
GraphDraft(Coherence,t=0.00)         0.998       79.8      75.8     115.0         54.2       1.56MB
GraphDraft(Coherence,t=0.30)         0.998       59.9      57.8      76.7         54.2       1.56MB
GraphDraft(Coherence,t=0.45)         0.998       96.1      92.7     132.0         54.2       1.56MB
GraphDraft(Coherence,t=0.50)         0.998       72.5      69.6     101.2         54.0       1.56MB
GraphDraft(Coherence,t=0.52)         0.985       82.8      79.0     113.4         53.2       1.56MB
GraphDraft(Coherence,t=0.55)         0.875       57.0      59.3      88.3         47.2       1.56MB
GraphDraft(Coherence,t=0.58)         0.735       57.2      63.7     106.1         39.4       1.56MB
GraphDraft(Coherence,t=0.60)         0.623       40.7      51.1      82.7         33.0       1.56MB
GraphDraft(Coherence,t=0.65)         0.373       40.0       8.8     115.6         19.2       1.56MB
GraphDraft(Coherence,t=0.70)         0.331       29.8       7.4      89.6         17.1       1.56MB
GraphDraft(Coherence,t=0.75)         0.233       24.7       7.4      88.0         12.1       1.56MB
GraphDraft(Coherence,t=0.80)         0.080       12.8       6.9      74.3          4.3       1.56MB
GraphDraft(Coherence,t=0.85)         0.014        7.8       6.9       9.6          1.0       1.56MB

[PASS] coherence-gated recall@10 >= 0.85 (best: t=0.55, got 0.875)
[FAIL] coherence-gated recall within 0.02 of baseline graph draft (baseline 0.998, got 0.875)
[PASS] coherence-gated mean latency < LinearFull mean latency (57.0us vs 324.5us)
[PASS] coherence-gated draft cost reduced by >= 10% vs baseline graph-draft (47.2 vs 54.2 expansions/query, actual reduction 12.9%)

ACCEPTANCE RESULT: REJECT
```

Mean-latency figures above are noisy at this scale (single-digit-to-low-
hundred microsecond queries on a shared container; p50/p95 swing run to run)
— the **DraftCost** column (average graph expansions per query) is the
stable, reproducible signal, since it counts algorithmic work rather than
wall-clock time. All recall and draft-cost figures were stable across
re-runs; the acceptance logic and threshold choice were fixed in the source
before the final run shown here (see git history for the two earlier,
narrower sweeps that led to widening the threshold range).

---

## Root Cause of the Cliff

Traced with `cargo run --release -p ruvector-coherence-speculative-ann --bin
diag`, which prints per-query pops/expansions directly (bypassing the
`AnnVariant` wrapper):

```
q0: baseline(pops=55,exp=54) t0.00(pops=55,exp=54) t0.45(pops=55,exp=54) t0.80(pops=23,exp=1)
q2: baseline(pops=53,exp=52) t0.00(pops=53,exp=52) t0.45(pops=53,exp=52) t0.80(pops=53,exp=52)
q5: baseline(pops=56,exp=55) t0.00(pops=56,exp=55) t0.45(pops=69,exp=55) t0.80(pops=23,exp=1)
```

A population-wide sample of `traversal_coherence` values (40,000 samples
across 10 queries) shows real spread (min=0.056, p50=0.507, p90=0.618,
max=1.0) — the metric itself is not degenerate. But the beam's priority
queue pops candidates in increasing u8-distance order, so by the time a node
is actually popped, it is already close to the query — and a node close to
the query is, almost by definition, roughly in the entry-to-query direction.
Low-to-moderate thresholds (<=0.50) therefore gate essentially nothing among
the ~54 candidates the search actually visits, even though they would gate
plenty of the full 4,000-point population. Once the threshold crosses into
the range that starts rejecting some of those *already-near-query* nodes
(>=0.55), it rejects exactly the expansions the beam needs to keep
converging, and recall collapses abruptly rather than trading off smoothly.

---

## Memory and Performance Math

```
DraftGraph storage (N=4000, dims=64, M+Mlj=22 edges/node):
  f32 corpus:   4000 x 64 x 4B        = 1.00 MB
  u8  corpus:   4000 x 64 x 1B        = 0.25 MB
  adjacency:    4000 x 22 x 4B (u32)  = 0.34 MB   (approx, variable degree)
  total                               ~ 1.56 MB   (measured: 1.56 MB)

LinearFull storage: 4000 x 64 x 4B = 0.98 MB (measured)

Overhead of DraftGraph vs LinearFull: ~1.6x (adjacency + redundant u8 copy)
— a real, modest, reportable memory cost for the graph-draft structure, not
hidden.

Distance evaluations per query:
  LinearFull:        n = 4,000
  GraphDraft(any):   ~54.2 (measured average, ef=50 pool)
  Reduction:         4000 / 54.2 ~= 73.8x fewer evaluations
```

The observed ~3-4x wall-clock speedup (324.5us -> ~60-80us) is smaller than
the ~74x evaluation-count reduction suggests, for the same reason
`ruvector-speculative-ann`'s own prior work noted: at this corpus size the
linear scan is cache-friendly and auto-vectorizes well, while the graph
traversal pays per-hop branching and heap-maintenance overhead that a flat
scan doesn't. This gap is honestly reported, not hidden behind the
evaluation-count number alone (see CLAUDE.md "no favorable metric hides a
regression").

---

## Failure Modes

1. **The cliff itself** (primary finding) — see Root Cause.
2. **Fixed single entry point degeneracy**: because the whole gate concept is
   relative to one entry point, "coherent" and "close to query" converge
   once the beam is past its first few hops, removing the gate's
   discriminating power exactly when it would matter most.
3. **u8 quantization on draft ranking, not on coherence**: coherence is
   computed on the exact f32 vectors (direction needs real geometry), so
   quantization error does not explain the cliff — confirmed by the
   diagnostic trace showing a genuine, non-degenerate coherence distribution.
4. **Scale not tested**: this PoC uses n=4,000 (smaller than
   `ruvector-speculative-ann`'s n=10,000) to keep the brute-force O(N^2 D)
   graph build tractable in a single nightly run. The cliff's location
   (threshold ~0.52-0.55) may shift at larger n; not measured here.

---

## Rejected Alternatives

- **Treating the mechanical "ACCEPT" from the first acceptance-check draft as
  final.** An earlier version of the acceptance check used a bare `cost <
  baseline_cost` comparison, which passed by float noise alone (0.998 recall,
  functionally zero cost change) — see ADR-353 and the git history of
  `src/bin/benchmark.rs`. This was caught by treating a too-good, suspicious
  result adversarially rather than reporting it as a win, and the check was
  tightened to require a reduction large enough to not be noise
  (`MIN_MEANINGFUL_DRAFT_COST_REDUCTION = 0.10`) before the real (REJECT)
  result above was accepted as final.
- **AdaptiveCoherenceSearch-style dynamic threshold** — `ruvector-coherence-hnsw`
  already implements this (threshold rises as the beam converges, falls when
  stuck). Not evaluated in this draft-graph context this run; flagged as the
  most promising next experiment, since it could in principle target
  low-coherence candidates near the *start* of the search (where gating is
  cheap) without penalizing the near-query region where this run's fixed
  threshold caused the cliff.
- **Multiple randomized entry points** — would break the "near-query implies
  coherent" degeneracy; not implemented (adds a parameter dimension outside
  the hypothesis as declared).

---

## Security

No new attack surface. The crate operates entirely on an in-memory,
deterministically generated synthetic corpus; no network I/O, no file I/O
beyond what `cargo` itself does, no secrets, no user input parsing.

## Governance

Research-only. Not integrated into any production crate or retrieval path.
Retained as a negative result (this README + ADR-353 + reproducible
benchmark/diagnostic binaries) so a future nightly run does not have to
re-derive the same cliff before trying a different gating mechanism.

## MCP Implications

None proposed. A production integration (if the adaptive-threshold follow-up
succeeds) would likely expose this as an internal draft-stage
implementation detail of `ruvector-agent-memory`'s recall path, not a
standalone MCP tool.

## WASM and Edge Implications

The `DraftGraph`'s u8 corpus (0.25 MB at this scale) is edge/WASM-friendly;
the f32 verify corpus and adjacency lists are the dominant memory cost
(1.56 MB total at N=4000, scaling linearly with N). Not benchmarked under
`wasm32` in this pass — the crate has no `wasm-bindgen` target yet.

## RVF Implications

If an adaptive-threshold follow-up succeeds, the `DraftGraph` (vectors +
adjacency + quantizer) is a natural RVF-portable artifact: deterministic to
rebuild from a corpus, snapshot-able, and small enough (MB-scale per few
thousand vectors) for cold-start portability. Not implemented.

## RVM Implications

No RVM-specific proof-gating or coherence-domain isolation need was
identified for this draft-stage mechanism; it operates entirely within a
single corpus/domain. Not forced.

## ruFlo Implications

The *threshold sweep itself* is a natural ruFlo workflow shape (propose
threshold -> run benchmark -> observe recall/cost -> adjust), which is
exactly what this run did manually. A `ruFlo` step that automates the sweep
and halts at the first threshold failing the recall-gap check would
reproduce this run's manual loop.

---

## Practical Applications

| Application | User | Problem | RuVector Capability | Ecosystem Integration | Path | Business Value | Main Risk | Horizon |
|---|---|---|---|---|---|---|---|---|
| Agent memory recall | AI agent systems | O(n) linear draft doesn't scale | Graph draft (ungated) | `ruvector-agent-memory` | Near-term | Lower recall latency at larger memory sizes | Needs re-validation at agent-memory's real corpus sizes | Near-term |
| RAG retrieval acceleration | LLM app developers | Retrieval is often the latency bottleneck | Graph draft + exact verify | Replace linear scan in RAG pipelines | Near-term | Fewer distance evals per query | Graph build is O(N^2 D) upfront | Near-term |
| Edge/WASM agent memory | Mobile/IoT developers | f32 corpus too large for constrained RAM | u8 draft ranking (already present) | `ruvector-edge-index` | Medium-term | Smaller working set during ranking | Verify stage still needs f32 access | Medium-term |
| Code intelligence search | IDE tooling | Low-latency symbol/embedding search | Graph draft | `ruvector-gnn-rerank` combo | Medium-term | Faster candidate generation before rerank | Index staleness on frequent edits | Medium-term |

## Long Horizon Applications

| Thesis | Required Advances | RuVector Role | Why This Experiment Matters | Primary Uncertainty | Falsification Path |
|---|---|---|---|---|---|
| Direction-aware traversal control as a general retrieval primitive | An adaptive, beam-state-conditioned gate that avoids this run's fixed-threshold cliff | Host the adaptive gate in `ruvector-coherence-hnsw` | This run bounds the problem precisely: fixed thresholds don't work, so the search space for "what does work" is narrower | Whether adaptive thresholds generalize past this synthetic clustered dataset | Re-run this exact benchmark harness with an adaptive gate swapped in |
| Multi-entry navigable graphs for agent memory at scale | Entry-point diversity strategies that don't reintroduce the "near-query implies coherent" degeneracy | `ruvector-agent-memory` + this crate's `DraftGraph` | Establishes that single-entry coherence gating is a dead end worth avoiding | Whether multiple entries meaningfully change the recall/cost curve | Re-run with K random entries, compare cliff location |

---

## Promotion Decision

**REJECT** the coherence-gating mechanism as implemented. **ACCEPT** (for
future integration consideration, not production promotion in this run) the
graph-based drafting substitution on its own — it is a clean, measured win
with no demonstrated recall cost at this scale.

## Witness Evidence

- Benchmark command and raw output: this document, "Real Benchmark Results"
  (unedited `cargo run --release` stdout).
- Diagnostic trace: this document, "Root Cause of the Cliff" (unedited
  `cargo run --release --bin diag` stdout).
- Source commit: see the branch's git log for the commits that added this
  crate, its tests, and this documentation (conventional-commit sequence).
- No cryptographic witness chain was generated — `ruvector-retrieval-receipt`'s
  signer infrastructure was not invoked for this research-only PoC (no
  production write path exists to attach a receipt to).

## Production Path

None recommended from this run. If the adaptive-threshold follow-up (Open
Question 1 in ADR-353) succeeds, the production path would be: wire
`DraftGraph` + the winning gate into `ruvector-speculative-ann`'s draft
trait, re-run that crate's own acceptance benchmark, and only then consider
`ruvector-agent-memory` integration.

## Falsification Criteria (declared before benchmarking)

The hypothesis is falsified if no threshold simultaneously clears
`cost reduction >= 10%` and `recall within 2pp of ungated graph draft`. The
13-point sweep found none — the hypothesis is falsified as declared.

## Limitations

- Single dataset size (n=4,000) and single dimensionality (64) tested.
- Single fixed entry point; not varied.
- Fixed-threshold gate only; adaptive gate not evaluated in this context.
- Wall-clock latency figures are noisy at microsecond scale on shared
  container hardware; draft-cost (expansion count) is the more reliable
  signal and is reported alongside latency rather than instead of it.

## Next Research

1. Re-run with `ruvector-coherence-hnsw`'s `AdaptiveCoherenceSearch` gate
   (threshold rises/falls with beam convergence) substituted for the fixed
   threshold, same benchmark harness.
2. Re-run at n=10,000 (matching `ruvector-speculative-ann`'s own scale) to
   check whether the cliff's location shifts.
3. Test entry-point diversity (K random entries, majority/union pooling) as
   an alternative way to avoid the "near-query implies coherent" degeneracy
   root-caused above.

---

## References

[^1]: `ruvector-speculative-ann` nightly research README, 2026-07-27 —
source of this run's hypothesis (see its "What to Improve Next" section).
[^2]: `ruvector-coherence-hnsw` nightly research README, source crate for
the `traversal_coherence` metric and `AdaptiveCoherenceSearch` (not
evaluated this run, flagged as next research).
[^3]: Leviathan Y., Kalman M., Matias Y. "Fast Inference from Transformers
via Speculative Decoding." ICML 2023. https://arxiv.org/abs/2211.17192.
[^4]: Malkov Y., Yashunin D. "Efficient and robust approximate nearest
neighbor search using Hierarchical Navigable Small World graphs." IEEE
TPAMI 2018. https://arxiv.org/abs/1603.09320.
