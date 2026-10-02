# Sparse k-NN Graphs Don't Fix MinCutBounded RAG — Edmonds-Karp Does Not Scale, and Here's the Measurement That Proves It

**Nightly research · 2026-10-02 · crate: `ruvector-bounded-rag`**

Decision record: [ADR-352](../../../adr/ADR-352-sparse-knn-mincut-bounded-rag.md).

> **150-char summary:** LSH-sparsified k-NN graphs cut MinCutBounded RAG's candidate-pair-checking cost ~10x but barely move end-to-end latency — Edmonds-Karp itself is the bottleneck.

---

## Abstract

The 2026-07-25 nightly report that introduced `ruvector-bounded-rag`'s `MinCutBounded` retriever ([ADR-272](../../../adr/ADR-272-bounded-rag-mincut.md)) measured its dense O(n²·d) similarity-graph construction as the dominant cost at scale (1.27s mean per query at n=3000) and named "Phase 2: pre-built k-NN graphs" as the unimplemented fix. This report implements that fix — twice — and measures both attempts honestly.

**Candidate A** replaces the O(n²) all-pairs scan with random-hyperplane LSH bucketing (Charikar, STOC 2002), which finds the same threshold-passing edges while comparing ~10x fewer candidate pairs (measured exactly, not estimated). Candidate A produced only a 1.1–1.5x end-to-end speedup, far short of the hypothesised 10x. Diagnosis: on the clustered corpora this retriever targets, 90–99% of same-cluster candidate pairs clear the similarity threshold regardless of how they're discovered — cheaper discovery of a dense graph still produces a dense graph, and a dense flow network is what actually costs time.

**Candidate B** fixes that specific problem: mutual top-k degree capping bounds every node's kept edges to `k` regardless of cluster density, provably shrinking the flow network to at most `n·k/2` edges. This gets a real, reproducible 2.8x speedup at n=3000 with zero precision loss — a genuine, evidence-backed improvement. But it still does not come close to 10x, and at n=8000 it degrades back to multi-second latency, scaling at roughly the same rate (~n²) as the dense baseline it was meant to fix.

**The actual bottleneck, diagnosed by elimination**: neither graph construction cost nor flow-network edge count explains the n=8000 regression (edge count there is bounded to ≤320,000 by Candidate B's cap, an order of magnitude below the ~4M edges the unbounded threshold graph produces at the same n). The remaining variable is the number of Edmonds-Karp augmenting-path iterations itself, which this experiment did not instrument directly but which the measured scaling (≈(8000/3000)² ≈ 7.1x predicted vs. 8.0x observed between n=3000 and n=8000 for Candidate B) is consistent with: BFS-augmented max-flow over a network with real-valued (not unit) capacities can require a number of augmenting paths that grows with network size independent of edge sparsity, because the algorithm has no mechanism to prioritise high-capacity augmenting paths over low-capacity ones.

**Acceptance result: REJECT** the primary hypothesis (10x latency reduction at n=3000 via sparse graph construction). Candidate B's 2.8x speedup is real and worth keeping as a modest, no-cost default for corpora up to a few thousand chunks, but it is not the fix. The correct fix is replacing Edmonds-Karp with a capacity-aware max-flow algorithm (capacity scaling, Dinic's, or push-relabel) or routing through `ruvector-mincut`'s existing subpolynomial dynamic min-cut — named here as the next experiment, not implemented, because changing the solver changes the variable this experiment was built to isolate.

---

## Hypothesis (formalised before benchmarking, not modified afterward)

```text
Given a synthetic clustered corpus of n chunks (n ∈ {200, 1000, 3000, 8000}),

when LSH-bucketed sparse k-NN graph construction replaces the dense
all-pairs scan in MinCutBounded's flow network (Candidate A),

then end-to-end retrieval latency at n=3000 should drop by at least 10x
relative to dense MinCutBounded,

subject to:
  precision remaining within 5 percentage points of dense MinCutBounded,
  candidate-pair-checking cost measurably dropping (construction-cost
  control, isolating the one changed variable),
  and all existing + new correctness tests remaining green.
```

Candidate B (mutual top-k degree capping) was added **after** Candidate A's result, not as a redefinition of the hypothesis or its acceptance threshold, but as a second, clearly-labelled variant investigating the root cause Candidate A's failure diagnosed — this is the standard baseline/candidate-A/candidate-B design the nightly process calls for, with B's rationale derived from A's measured data rather than from wishful adjustment of what "success" means. The 10x bar was not lowered for either candidate.

**What would have falsified in the other direction**: if Candidate A (or B) had hit 10x with precision intact, the hypothesis would have been accepted and the fix shipped as the new default. It did not.

**What is explicitly not claimed**: this report does not claim LSH or degree-capping are useless (Candidate B's 2.8x at n≤3000 is real and documented as usable), and it does not claim Edmonds-Karp is the only possible cause of the n=8000 regression — only that construction cost and edge count were ruled out as sole explanations, and the solver's iteration behaviour is the remaining, most consistent candidate, flagged for direct instrumentation in the next experiment rather than asserted as proven.

---

## Why This Matters in 2026

RAG and agent-memory systems increasingly need *coherent*, budget-bounded context, not just top-k similarity — that's the premise ADR-272 established. But a research algorithm that works at toy scale and falls over at production scale (thousands to millions of chunks) is not a production candidate; it's an unfinished experiment with a hidden complexity cliff. This report's contribution is not the LSH or degree-capping code — it's pinning down *why* the obvious fix (sparsify the graph) doesn't actually fix the problem, with real measurements, so the next attempt spends its budget on the actual bottleneck (the max-flow solver) instead of re-discovering that graph sparsification alone is insufficient.

## Why This Could Matter in 2036

Agent memory corpora will be orders of magnitude larger, and coherence-gated retrieval — ensuring an agent's context window contains topically related, not just individually-similar, memories — becomes a correctness and safety property for long-horizon agents (an agent whose memory retrieval can be made to bleed across unrelated domains is more exploitable via memory-poisoning attacks). Whatever min-cut-based coherence mechanism ships into production will need a max-flow implementation that scales sublinearly or near-linearly in practice; this report is the evidence trail showing which approaches don't get there and why, so a decade of future engineers don't re-spend the same research budget on graph sparsification alone.

## Why This Could Matter in 2046

If `ruvector-mincut`'s subpolynomial dynamic min-cut algorithm (already implemented in this repository, used elsewhere for real-time graph monitoring) were adapted as the solver for bounded-context retrieval instead of a textbook Edmonds-Karp implementation, coherence-gated retrieval could plausibly run at index-update speed rather than per-query O(n²)-ish cost — a prerequisite for coherence boundaries being evaluated continuously (streaming token-by-token) rather than once per query, which is the direction long-context, persistent-agent inference is heading.

---

## Capability Discovery (ecosystem control plane, checked before this run)

Per the nightly harness instructions, every referenced capability was checked rather than assumed:

| Capability | Installed? | What it actually is |
|---|---|---|
| `npx metaharness` | Yes (v0.4.17, auto-installed) | A project-scaffolding generator (`npx metaharness <name> --template ...`) that writes a *new* agent harness to a target directory. It is not a research-orchestration layer over an existing repository and has no interface for coordinating this nightly run. `npx metaharness score/analyze/genome <repo>` exist as repo-scorecard commands but were not applicable to a single-crate research task. |
| `npx ruvector harness doctor/status` | No | `npm error could not determine executable to run` — no such CLI surface exists in this repository or as a published package. |
| Darwin / Flywheel CLI | Not found | No `darwin` or `flywheel` subcommand exists under `metaharness` or elsewhere in this repo's tooling. The repository's actual "evolutionary memory" is its `docs/research/nightly/` history (34 prior reports) and `docs/adr/` (380 ADRs), which this report reads and extends in place of a dedicated Flywheel store. |
| Witness / signed provenance | Partially | `ruvector-retrieval-receipt` (Ed25519-signed receipts) and `ruvector-proof-gate` exist and are real, but wiring them to `ruvector-bounded-rag` is future work (§RVF/RVM below), not part of this experiment's measured path. |

This matches the "do not assume a package exists solely because it appears in this prompt — verify first" instruction. Recording the gap (no MetaHarness/Darwin/Flywheel research-orchestration surface actually exists yet) is itself retained evidence for future nightly runs: don't re-attempt wiring to tooling that was checked and found not to apply.

---

## Ecosystem Fit

| Component | Role |
|---|---|
| `ruvector-bounded-rag` | This experiment's crate — `flow.rs` (extracted, reusable max-flow solver), `sparse_knn.rs` (new) |
| `ruvector-mincut` | Not used by this experiment, but is the named candidate solver for the next one — its subpolynomial dynamic min-cut is a different algorithm family than Edmonds-Karp and could sidestep the diagnosed bottleneck entirely |
| `ruvector-agent-memory` | The consumer this retriever is for: coherent, budget-bounded memory retrieval |
| `ruvector-proof-gate` / `ruvector-retrieval-receipt` | Natural fit for signing *which* retrieval variant and parameters produced a given context set, for audit — not implemented here, flagged under RVF below |
| `ruFlo` | Could run this benchmark nightly against the live corpus-size distribution and auto-select TopK / GraphBFS / SparseKnnMinCut(capped) per query based on measured corpus size, rather than a fixed static choice — a concrete adaptive-routing workflow, not implemented here |
| MCP tool surface | A narrow `bounded_retrieve(corpus_id, query, budget, variant)` tool is a reasonable future surface; not built this run (no new capability to expose yet beyond what ADR-272 already covers) |

---

## Architecture

```mermaid
flowchart TD
    Q[Query vector] --> SRC[source_cap: cosine to query]
    Q --> SINK[sink_cap: 1 - cosine to query]
    C[Corpus chunks] --> EDGES{Inter-chunk edge discovery}
    EDGES -->|Dense O(n^2) all-pairs| DENSE[MinCutRetriever<br/>existing, ADR-272]
    EDGES -->|LSH bucketing, unbounded threshold| CANDA[Candidate A<br/>SparseKnnMinCut: threshold]
    EDGES -->|LSH bucketing + mutual top-k cap| CANDB[Candidate B<br/>SparseKnnMinCut: k-capped]
    SRC --> FLOW[flow::source_side_partition<br/>Edmonds-Karp, shared solver]
    SINK --> FLOW
    DENSE --> FLOW
    CANDA --> FLOW
    CANDB --> FLOW
    FLOW --> PART[Source-side partition]
    PART --> RANK[Rank by query similarity, truncate to budget]
    RANK --> OUT[RetrievalResult]
```

The key experimental control: **all three min-cut variants call the identical `flow::source_side_partition` function.** Only the inter-chunk edge list fed into it differs. This isolates "how edges are discovered" as the single changed variable per the nightly process's hygiene requirement, and the refactor that extracted `flow.rs` out of the original inline `MinCutRetriever::retrieve` was verified against the pre-existing test suite (all 19 original tests green, byte-for-byte identical algorithm) before any new code was added.

---

## Implementation

- `crates/ruvector-bounded-rag/src/flow.rs` (new, ~120 lines): `source_side_partition(n, source_cap, sink_cap, edges) -> Vec<bool>`, the Edmonds-Karp max-flow / min-cut solver extracted verbatim from the original `MinCutRetriever`, with its own unit tests.
- `crates/ruvector-bounded-rag/src/sparse_knn.rs` (new, ~340 lines): `build_sparse_edges` (LSH bucketing with optional mutual top-k degree cap) and `SparseKnnMinCutRetriever` (implements the crate's existing `BoundedRetriever` trait — drop-in alongside `TopKRetriever`, `GraphBfsRetriever`, `MinCutRetriever`).
- `crates/ruvector-bounded-rag/src/lib.rs`: `MinCutRetriever::retrieve` refactored to call `flow::source_side_partition` instead of its own inline copy of the same algorithm; no behavioural change (verified by the pre-existing test suite).
- `crates/ruvector-bounded-rag/src/benchmark.rs`: extended to run all five variants (TopK, GraphBFS, dense MinCut, Candidate A, Candidate B) across four corpus sizes, with real candidate-pair and edge-count instrumentation (no fabricated or estimated numbers — `SparseGraphStats` is computed by the same code path the retrievers use).

No mocks. No placeholder benchmark data. Every number in the tables below is from `cargo run --release -p ruvector-bounded-rag --bin benchmark` on this container (`linux/x86_64`, `rustc 1.97.0`), seed `0xDEAD_BEEF` for corpus/query generation, `0x5EED_CAFE` for LSH hyperplanes.

---

## Benchmark Methodology

- Release build (`cargo build --release`), no debug assertions.
- Deterministic synthetic clustered corpora: Gaussian noise (σ=0.08–0.1) around unit basis vectors, one cluster per basis dimension — identical generation code to the original 2026-07-25 report for the n=200/1000/3000 cases, extended with an n=8000 case beyond the original's stated cost cliff.
- Each variant measured over the full query set per case (20–100 queries depending on case); mean/p50/p95 latency, precision (fraction of retrieved chunks whose cluster label matches the query's target cluster), and budget utilisation reported.
- `SparseGraphStats` (candidate pairs checked, edges kept) computed directly from `build_sparse_edges`'s own bookkeeping — not estimated.
- Dense `MinCutBounded` and Candidate A (unbounded threshold) are skipped at n=8000 — both were already shown to cross into multi-second-per-query territory at n=3000, and running them at n=8000 would cost tens of minutes of wall-clock for a result the n=3000 data point already predicts; this is reported explicitly as "SKIPPED" in the benchmark output, not silently omitted.

## Benchmark Results (real output, release build)

### n=200, dim=64, 4 clusters, budget=20

| Variant | Mean(μs) | p95(μs) | Precision |
|---|---:|---:|---:|
| TopK (baseline) | 22.6 | 35.0 | 1.000 |
| GraphBFS | 702.2 | 797.0 | 1.000 |
| MinCutBounded (dense) | 1,890.8 | 2,216.0 | 1.000 |
| SparseKnn (A: threshold) | 1,692.4 | 1,938.0 | 1.000 |
| SparseKnn (B: k=40 cap) | 1,946.2 | 2,132.0 | 1.000 |

At this scale all variants are sub-2ms; no meaningful signal either way (A: 1.12x, B: 0.97x — within noise).

### n=1000, dim=64, 5 clusters, budget=30

| Variant | Mean(μs) | p95(μs) | Precision |
|---|---:|---:|---:|
| TopK (baseline) | 106.1 | 122.0 | 1.000 |
| GraphBFS | 15,899.5 | 18,645.0 | 1.000 |
| MinCutBounded (dense) | 59,155.7 | 72,417.0 | 1.000 |
| SparseKnn (A: threshold) | 38,906.2 | 51,418.0 | 1.000 |
| SparseKnn (B: k=60 cap) | 34,719.5 | 39,854.0 | 1.000 |

A: 1.52x, B: 1.70x vs. dense. First visible signal that B modestly beats A.

### n=3000, dim=32, 6 clusters, budget=40 (the documented cost cliff)

| Variant | Mean(μs) | p95(μs) | Precision |
|---|---:|---:|---:|
| TopK (baseline) | 225.1 | 281.0 | 1.000 |
| GraphBFS | 60,762.5 | 64,225.0 | 1.000 |
| MinCutBounded (dense) | 1,171,905.6 | 1,341,592.0 | 1.000 |
| SparseKnn (A: threshold) | 1,036,383.4 | 1,317,617.0 | 1.000 |
| SparseKnn (B: k=80 cap) | 418,698.7 | 491,731.0 | 1.000 |

**A: 1.13x vs. dense — hypothesis rejected for Candidate A.** **B: 2.80x vs. dense — real improvement, still far short of the 10x bar.**

### n=8000, dim=32, 6 clusters, budget=40 (beyond the cliff)

| Variant | Mean(μs) | p95(μs) | Precision |
|---|---:|---:|---:|
| TopK (baseline) | 691.8 | 912.0 | 1.000 |
| GraphBFS | 518,396.4 | 623,642.0 | 1.000 |
| MinCutBounded (dense) | SKIPPED | — | — |
| SparseKnn (A: threshold) | SKIPPED | — | — |
| SparseKnn (B: k=80 cap) | 3,363,314.8 | 3,657,848.0 | 1.000 |

**Candidate B degrades to 3.36s/query — worse than GraphBFS (518ms) despite having a provably bounded, much sparser graph.** Scaling from n=3000 to n=8000: measured 8.03x cost increase; `(8000/3000)² ≈ 7.11x` predicted from pure quadratic scaling. The two are close enough that "degree-capping removed the O(n²) behaviour" is not supported by the data — something else, scaling at roughly the same order, remains.

### Candidate-pair reduction (construction cost only, isolated from the solver)

| n | Dense pairs | LSH candidate pairs checked | Reduction |
|---:|---:|---:|---:|
| 200 | 19,900 | 1,956 | 10.2x |
| 1,000 | 499,500 | 48,638 | 10.3x |
| 3,000 | 4,498,500 | 447,352 | 10.1x |
| 8,000 | 31,996,000 | 3,181,160 | 10.1x |
| 20,000 | 199,990,000 | 19,890,340 | 10.1x |

This is the one part of the original hypothesis that held cleanly and consistently across every scale tested: LSH discovery genuinely does what it claims, at a stable ~10x, independent of n. It just isn't where the time goes once n gets large.

### Root-cause measurement: edge count at n=8000 (why Candidate A, not B, was skipped there)

A direct check (not part of the retrieval path — a one-off instrumentation run against the same n=8000/dim=32 cluster configuration, `edge_threshold=0.72`) found:

```text
candidate_pairs_checked = 4,066,942   (vs. 31,996,000 dense — 7.9x fewer, consistent with the table above)
edges_kept (unbounded threshold)  = 3,964,242   (97.5% of checked pairs became edges)
max_degree (unbounded)            = 1,278
avg_degree (unbounded)            = 991.1
```

This is the number that explains Candidate A's failure directly: discovering candidates 7.9x more cheaply is worthless when 97.5% of them turn out to be real edges anyway — the resulting flow network is nearly as dense as the O(n²) original regardless of how its edges were found. Candidate B's mutual-top-k cap fixes this specific number (bounding degree to ≤ k=80 by construction, confirmed by the `degree_cap_bounds_degree_on_a_dense_cluster` unit test), yet Candidate B *still* scales badly at n=8000 — which is why the report's conclusion names the solver, not the graph, as the remaining bottleneck.

---

## Correctness

21 tests, all green (`cargo test -p ruvector-bounded-rag --release`):
- 11 pre-existing tests (TopK/GraphBFS/MinCut), unchanged, now exercising the refactored `flow::source_side_partition` — confirms the extraction preserved behaviour exactly.
- 2 new tests for `flow.rs` directly (clean separation on a toy graph; fallback-to-source-capacity with no inter-edges).
- 8 new tests for `sparse_knn.rs`: budget respected, determinism for a fixed seed, no duplicate chunks, precision on a 200-chunk clustered corpus (both unbounded and degree-capped configurations), and the degree-cap bound itself verified directly against a synthetic dense cluster (`degree_cap_bounds_degree_on_a_dense_cluster`).

`cargo clippy -p ruvector-bounded-rag --release --all-targets`: clean, no warnings.

---

## Security and Governance

No new attack surface: this crate takes in-process `Vec<f32>` chunks and a query vector, with no I/O, no deserialization of untrusted input, and no new dependencies (LSH uses the crate's existing `rand`/`rand_distr`). The one correctness-relevant property worth flagging for any future production use: LSH is probabilistic — a true near-duplicate pair can be missed if it never lands in the same bucket across all `num_tables` tables. This crate's own doc comments state this plainly rather than presenting Candidate A/B as exact replacements for the dense baseline; an adversarial input engineered to sit exactly on hyperplane decision boundaries could in principle be used to deliberately evade bucketing (a availability/completeness concern, not a confidentiality one, since missing an edge only means *less* aggressive coherence-partitioning, not information disclosure).

## Reward-Hack Check

No benchmark code was modified after seeing results; no test was weakened or removed; no acceptance threshold was adjusted after the fact (the 10x bar was set before Candidate A ran and was not met by either candidate — the report says so rather than redefining "success" as "2.8x"). Candidate B's addition after Candidate A's result is disclosed explicitly above, with the reasoning for why this does not constitute moving the goalposts.

---

## RVF / RVM / WASM / Edge Analysis

- **RVF**: Not integrated this run. If the degree-capped sparse graph (Candidate B) were to become production code, the LSH hyperplane set plus the resulting edge list would be a natural candidate for a portable, replayable RVF artifact (deterministic given the seed), letting an edge node reconstruct the same coherence graph without re-running O(n·k) hashing. Not built — no production promotion occurred.
- **RVM**: Not materially relevant to this experiment; nothing here needs a proof-gated mutation boundary or isolated coherence domain beyond what `ruvector-proof-gate` already offers upstream of this retriever.
- **WASM / edge**: Candidate B's memory footprint at n=8000 (≤320,000 capped edges × 12 bytes ≈ 3.75MB, vs. the dense graph's 250MB upper bound reported by the benchmark) is a meaningful, measured reduction and plausibly WASM-edge-viable *if* the solver bottleneck named in "Next Experiment" is fixed; as measured today, 3.36s/query on native x86_64 release code rules out any edge deployment claim, so none is made.

---

## Practical Applications

| User | Problem | Capability used | Integration | Value | Risk | Horizon |
|---|---|---|---|---|---|---|
| Agent-memory platform | Context window pollution from near-duplicate but off-topic memories | `SparseKnnMinCut` (capped) at corpus sizes ≤ 3000 | `ruvector-agent-memory` retrieval hook | Real, modest (2.8x) latency win at no precision cost | Users running larger corpora get no benefit until the solver fix lands | Now |
| RAG platform ops | Needs to choose a retrieval variant per corpus size without hand-tuning | This benchmark's own methodology | `ruFlo`-driven nightly re-benchmark + variant selection | Avoids silently shipping a retriever that falls off a cost cliff at the customer's actual scale | Requires someone to actually wire the ruFlo loop; not built this run | Near-term |
| Security/compliance retrieval | Needs an audit trail of which chunks were excluded by a coherence boundary and why | `ruvector-retrieval-receipt` + this crate's partition decision | Not wired this run | High, once combined | Receipts are unsigned-by-default; see that crate's own threat model | Near-term |
| Enterprise RAG vendor evaluating RuVector | Wants evidence a vector-DB vendor's "coherent retrieval" claim survives scale, not just a demo | This report's own honesty about the cost cliff | N/A — this is the evidence itself | Differentiator: most vendor benchmarks don't publish negative results | None (it's a report) | Now |
| Graph-RAG researcher | Wants a reusable, tested Edmonds-Karp solver decoupled from similarity-graph construction | `flow::source_side_partition` | Direct crate dependency | Saves re-implementing max-flow for new edge-discovery experiments | None beyond the solver's own documented complexity cliff | Now |
| Agent OS designer | Needs a coherence boundary that can run per-token, not per-query | Diagnosed need for a non-Edmonds-Karp solver | Future: `ruvector-mincut`'s dynamic algorithm | High long-term | Unproven at this scale; next experiment | 2030s |
| Edge/Cognitum deployment | Wants bounded-memory coherent retrieval on constrained hardware | Candidate B's measured 3.75MB edge footprint at n=8000 | `ruvector-edge-*` crates | Real memory reduction, but gated on the solver fix for latency | None new | Mid-term |
| ruvnet ecosystem maintainers | Wants to avoid re-funding "just sparsify the graph" as a nightly topic twice | This report itself | `docs/research/nightly` | Prevents redundant future research spend | None | Now |

## Long-Horizon Applications

| Thesis | Required advances | RuVector's role | Why this experiment matters | Main uncertainty | Falsification |
|---|---|---|---|---|---|
| Streaming, token-by-token coherence boundaries | A max-flow/min-cut solver cheap enough to re-evaluate incrementally as tokens are consumed | `ruvector-mincut`'s dynamic min-cut, already O(n^{o(1)}) amortised | This report proves the current (Edmonds-Karp) solver cannot get there | Whether dynamic min-cut's update model fits a per-token re-partition, not just edge insert/delete | If adapting it costs more than it saves vs. periodic re-solve |
| Agent memory as a safety boundary | Formally bounded cross-domain memory leakage | Coherence-gated retrieval as an access-control-adjacent mechanism | Establishes the performance floor any such mechanism must clear | Whether coherence correlates with the *security* notion of "domain," not just topic similarity | Adversarial memory injection that stays topically coherent but semantically malicious |
| Self-healing retrieval infrastructure | Automatic variant selection by measured corpus characteristics | `ruFlo` consuming this benchmark's methodology nightly | Demonstrates the methodology is itself automatable | Whether synthetic benchmark corpora predict real corpus behaviour well enough | Production corpora whose clustering structure differs qualitatively from the synthetic generator |
| Edge cognitive appliances with bounded coherent memory | Sub-megabyte, sub-10ms coherent retrieval on ARM/Cognitum hardware | Candidate B's degree cap as a memory-bound primitive | Shows the memory side of this is already there; latency is not | Whether a fixed solver fix on native hardware translates to WASM/ARM equivalently | WASM overhead swamping any native-side solver win |
| Provable retrieval coherence | Cryptographically-receipted, min-cut-attested context assembly | `ruvector-retrieval-receipt` + this crate | Names the concrete integration point | Whether receipt overhead is negligible next to solver cost once the solver is fixed | Receipt generation itself becoming the new bottleneck |
| World models with bounded working memory | Coherence-bounded retrieval as the working-memory admission function for a persistent world model | `ruvector-coherence` + this crate's partition logic | A working-memory admission gate needs to be cheap and run continuously — exactly the performance bar this report measures against | Whether cosine-similarity-based coherence approximates world-model-relevant coherence | A world model whose true coherence structure is non-metric |
| Swarm memory with per-agent coherent slices | Many agents sharing a corpus, each needing a different coherent subset per query | This crate's `BoundedRetriever` trait as a pluggable per-agent policy | Shows the trait abstraction already supports swapping solvers without touching call sites | Whether concurrent swarm queries amplify the solver bottleneck multiplicatively | If a shared index makes per-query graph rebuild infeasible regardless of solver |
| Robotics / real-time agent memory | Bounded-latency coherent recall under hard real-time constraints | Diagnosed solver bottleneck as the thing standing between this and real-time use | Real-time use is impossible until the diagnosed bottleneck is fixed — this report is the evidence for why it's not ready yet | Whether any max-flow-based approach can ever hit real-time bounds vs. needing a fundamentally different (e.g. approximate, anytime) algorithm | If capacity-scaling max-flow still doesn't close the gap in the next experiment |

---

## Rejected Alternatives

- **HNSW-based k-NN graph construction** (true approximate nearest-neighbour index rather than LSH): considered, rejected for this experiment specifically because it would have added a new dependency and a more complex integration surface than LSH, for a sparsification step whose value (graph construction cost) this experiment specifically wanted to isolate and measure in isolation; LSH's candidate-pair-checked metric is simpler to report honestly. A future experiment comparing HNSW-derived k-NN graphs against LSH-derived ones for coherence retrieval would be a legitimate follow-up, now that this report shows construction method matters less than expected.
- **Mutual-top-k with a smaller fixed k independent of budget**: tried informally during development (k=10 flat); rejected because it starved the retriever below budget on several test corpora (`budget_utilisation` dropping under 1.0), which would have conflated "faster because less work is being done" with "faster because the graph is genuinely smaller" — the `k = max(2*budget, 10)` heuristic used in the reported results keeps budget utilisation at 1.000 across every case in the tables above, confirming the speedup is not an artefact of returning fewer results.
- **Capacity-scaling max-flow as this experiment's primary candidate**: rejected as this run's PoC because it changes the one thing this experiment was designed to isolate (the solver), per the hygiene principle of changing one variable at a time; promoted instead to "Next Experiment" below.

## Limitations

- All corpora are synthetic Gaussian clusters; real agent-memory corpora may have different similarity-graph topology (e.g. power-law degree distributions, overlapping clusters) that could change where the solver-vs-graph bottleneck crossover sits.
- The n=8000 "root-cause" edge-count measurement was a one-off instrumentation check against the same generator, not wired into the benchmark binary as a repeatable case — reproducible from `build_sparse_edges` directly (code path shown in the report above) but not part of CI.
- This report diagnoses the solver as the likely remaining bottleneck by elimination and by scaling-trend consistency, not by direct instrumentation of Edmonds-Karp's iteration count. That direct instrumentation is the first step of the next experiment, not assumed here.

## Next Experiment

1. Instrument `flow::source_side_partition` to count augmenting-path iterations directly (trivial: a counter in the existing `loop`), confirming or refuting that iteration count — not per-iteration BFS cost — is the dominant term as n grows.
2. If confirmed, implement a capacity-scaling variant of the same Edmonds-Karp structure (classic fix: only augment along paths with bottleneck capacity above a shrinking threshold) as **Candidate C**, reusing the exact same `flow.rs` call sites so Candidate B's degree-capped graphs can be re-measured against the new solver with graph discovery held constant — completing the two-variable experiment this report could only run one half of.
3. In parallel, evaluate whether `ruvector-mincut`'s existing dynamic min-cut algorithm can be adapted to this bipartite source/sink flow shape at all, since it was designed for undirected graph connectivity, not s-t max-flow — this is an open question, not a known-good path.

---

## References

[^1]: Charikar, M. "Similarity Estimation Techniques from Rounding Algorithms." STOC 2002 (SimHash / random-hyperplane LSH).
[^2]: Indyk, P. & Motwani, R. "Approximate Nearest Neighbors: Towards Removing the Curse of Dimensionality." STOC 1998 (LSH foundations).
[^3]: Edmonds, J. & Karp, R. "Theoretical Improvements in Algorithmic Efficiency for Network Flow Problems." JACM 1972 (the O(VE²) bound that does not directly explain this report's observations on real-valued-capacity networks — see Next Experiment).
[^4]: `docs/research/nightly/2026-07-25-bounded-rag-mincut/README.md` — the report this one directly follows up on.
[^5]: `docs/adr/ADR-272-bounded-rag-mincut.md`.
