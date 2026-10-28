# Nightly Research — 2026-10-28

## Adaptive Local Coherence Gating for GNN Candidate Reranking under Metadata-Filtered ANN Retrieval

**Result: REJECT.** The hypothesized recall advantage did not materialize. This document
records the full research loop and the measured evidence — a falsified hypothesis with a
clear, honest explanation is the success condition this nightly process is designed to
produce when the underlying idea doesn't pan out.

## 1. Summary / abstract

`ruvector-gnn-rerank` already reranks noisy first-stage ANN candidates by diffusing scores
over a candidate similarity graph, with `GnnMincutReranker` gating diffusion edges behind a
single global cosine-similarity threshold. We hypothesized that this global threshold would
degrade under upstream metadata filtering (tenant/ACL/freshness filters applied before
reranking), because filtering sparsifies the candidate pool's local similarity structure
unevenly, potentially leaving many nodes with no edges clearing a fixed global bar. We
implemented `AdaptiveCoherenceReranker`, which gates each node's diffusion edges relative to
*that node's own* local similarity ranking (keep its own top fraction of edges) rather than
one global cutoff, and benchmarked both rerankers across five filter selectivity levels on a
seeded synthetic workload. The predicted large recall gain at low selectivity did not appear:
the measured mean gain was +0.18 percentage points against a required +3.0 pp threshold, with
a 95% confidence interval ([-1.6, +2.0] pp) that excludes the hypothesized effect. The new
reranker is additive, safe (no regression, no latency cost), and kept in the crate as an
available but *not recommended* variant.

## 2. Formal hypothesis

```
Given a synthetic clustered-vector candidate pool (seeded, deterministic) subjected to a
metadata-style filter at selectivity s ∈ {1.00, 0.50, 0.25, 0.10, 0.05} (filter independent
of semantic cluster membership, thinning each cluster proportionally),

when candidates are reranked with AdaptiveCoherenceReranker (per-node top-fraction edge
gating) instead of the existing GnnMincutReranker (global fixed-threshold edge gating),

then mean recall@10 (vs. ExactL2Reranker oracle ground truth on the same filtered pool)
at low selectivity (s <= 0.10) should exceed GnnMincutReranker's by >= 3 percentage
points, averaged over >= 200 trials with varying seeds,

subject to:
  - recall@10 at s = 1.00 (unfiltered) not regressing by more than 0.5 percentage points
    vs. GnnMincutReranker,
  - mean per-query latency not exceeding 2x GnnMincutReranker's at matched candidate-set
    size,
  - cargo test -p ruvector-gnn-rerank and cargo clippy -p ruvector-gnn-rerank -- -D warnings
    staying green.
```

ACCEPT required all four conditions. REJECT applies if the primary `then` clause fails,
regardless of the subject-to conditions. That is what happened here.

## 3. Why it matters now (2026) and over 10–20 years

Retrieval-augmented generation and agent-memory systems increasingly operate behind
access-control, tenancy, and freshness filters — a RAG system serving multiple tenants, an
agent querying only its own memory namespace, a security-scoped search over a document
corpus. First-stage ANN indexes are built for the *unfiltered* corpus; the filter is applied
either before or after retrieval, and in either case the reranking stage sees a pool whose
structure differs from the unfiltered case the reranker's parameters were likely tuned
against. As agent memory and multi-tenant retrieval infrastructure scale (RuVector's
`ruvector-agent-memory`, `ruvector-edge-tenancy`, `ruvector-edge-authz`), this gap between
"tuned for the common case" and "running under a restrictive filter" only grows — whether
the mechanism investigated here works or not, the *question* (does coherence-gated reranking
degrade under filtering, and how should it adapt) stays relevant for as long as filtered
retrieval remains a core pattern — plausibly a multi-decade concern for any system that does
semantic search with access control, not just RuVector.

## 4. 2026 SOTA context

- GNRR (arXiv 2406.11720) and Maniscope (arXiv 2602.15860) — graph-based reranking to recover
  recall lost to quantization/first-stage noise, the direct ancestors of this crate's existing
  diffusion rerankers.
- AQR-HNSW (arXiv 2602.21600) — adaptive-quality reranking referenced by the crate's existing
  `GnnMincutReranker` design.
- Filtered ANN more broadly (Milvus/Qdrant/Weaviate pre-filtering vs. post-filtering debates,
  Pinecone's metadata filtering docs) consistently treat the filter as orthogonal to the
  index's internal similarity structure — none of the public documentation surveyed addresses
  *reranker* parameter sensitivity to filter selectivity specifically, which is the gap this
  experiment targeted.
- We did not find a directly comparable academic result on adaptive vs. global coherence
  thresholds for post-filter graph reranking; this appears to be a genuinely untried
  composition rather than a renamed existing technique — the rejection is informative, not
  because the idea was unoriginal.

## 5. RuVector ecosystem fit

- **RuVector**: direct — extends an existing production-path crate (`ruvector-gnn-rerank`)
  with no breaking change.
- **ruFlo**: if any reranker variant *had* shown a selectivity-dependent advantage, a natural
  ruFlo workflow would be "observe the filter's realized selectivity per query and switch
  reranker variant adaptively." Since no variant shows a measurable selectivity-dependent
  advantage here, there is nothing for ruFlo to switch between yet — this is the honest
  implication of a rejected hypothesis, not a reason to build the switch anyway.
- **MetaHarness**: `npx metaharness --help` was probed (Step 3/Capability discovery below);
  it is an agent-harness *scaffolding* tool, not a research orchestrator for in-repo Rust
  experiments, so it was not used beyond that discovery step.
- **Flywheel**: no dedicated Flywheel CLI (`npx ruvector harness flywheel`) exists in this
  environment — this document, the raw benchmark output, and ADR-355 together **are** the
  flywheel record for this run (hypothesis, evidence, decision, rejection, all in one place).
- **Darwin**: no dedicated Darwin CLI exists either; a manual, bounded one-generation,
  four-candidate parameter sweep (§10) stood in for it, following the spec's fitness formula.
- **MCP**: a narrow, read-only `rerank_candidates(reranker: enum, candidates, k) -> ranked`
  tool would be a reasonable MCP surface for *any* of these rerankers, but nothing here
  specifically motivates adding it yet.
- **RVF / RVM**: not materially relevant — reranking is a stateless, in-memory transform over
  an already-fetched candidate set; there is no portable state or coherence-domain boundary
  to capture.

## 6. Capability discovery (honest findings)

| Capability | Installed | Notes |
|---|---|---|
| `npx metaharness` | Yes (v0.4.19 auto-installed) | Scaffolding tool for *new* agent harnesses (score/analyze/genome/learn/weight-eft/redblue/flywheel/turn-credit/avo/proxy subcommands). Not a Rust-experiment orchestrator. |
| `npx ruvector harness doctor/status` | **No** | `npm error: could not determine executable to run`. No such CLI exists in this checkout. |
| pi.ruv.io brain MCP tools (`brain_*`) | **No** | Not present in this session's tool list; not probed further, honestly recorded as unavailable rather than silently skipped. |
| Darwin / Flywheel dedicated CLI | **No** | No binding found for this experiment type; see §5 for the manual stand-ins used instead. |

Given the above, this run proceeded via direct Rust engineering plus Claude Code tooling,
exactly as the nightly spec's failure-handling guidance permits when assumed infrastructure
does not exist in the checkout.

## 7. Architecture

```mermaid
flowchart LR
    Q[Query] --> FS[First-stage ANN retrieval<br/>noisy candidate pool, size≈300]
    FS --> F{Metadata filter<br/>selectivity s}
    F -->|kept fraction s| Pool[Filtered candidate pool]
    Pool --> NS[NoisyScoreReranker<br/>baseline]
    Pool --> GM[GnnMincutReranker<br/>global threshold gate]
    Pool --> AC[AdaptiveCoherenceReranker<br/>per-node top-fraction gate]
    Pool --> EX[ExactL2Reranker<br/>oracle / ground truth]
    NS --> R[recall@10 vs oracle]
    GM --> R
    AC --> R
    EX --> R
```

## 8. Implementation

- `crates/ruvector-gnn-rerank/src/reranker.rs`: new `AdaptiveCoherenceReranker` struct +
  `CandidateReranker` impl. Builds the same `CandidateGraph` as the existing variants, then
  for each node keeps `max(min_keep, ceil(keep_fraction * degree))` of its own strongest
  edges (already sorted descending by `CandidateGraph::build`) before doing the same
  weighted-mean diffusion update as `GnnMincutReranker`. Zero-degree nodes are left
  unchanged. Single hop, matching `GnnMincutReranker`'s structure for a fair comparison.
- `crates/ruvector-gnn-rerank/src/lib.rs`: re-export + unit test.
- `crates/ruvector-gnn-rerank/src/bin/filtered_rerank_bench.rs`: the benchmark (methodology
  below).
- 4 new unit tests total (3 in `reranker.rs`, 1 in `lib.rs`): k-results-returned, zero-degree
  tolerance (lone candidate is left unchanged), and a `min_keep`-floor test that
  deterministically proves the floor changes the diffusion result (constructed with
  `alpha=0.0` so the result is exactly the kept-neighbour weighted mean).

## 9. Benchmark methodology

Run: `cargo run --release -p ruvector-gnn-rerank --bin filtered_rerank_bench`

- Synthetic corpus: N=5,000 vectors, dim=128, 20 clusters, cluster sigma=0.5,
  `corpus_seed=42` (deterministic).
- Per trial `t` (seed = `base_seed(777) + t`): a query jittered near a random corpus point;
  a noisy first-stage pool of `RETRIEVAL_K=300` candidates (true top-300 by L2, each scored
  with independent `N(0, 0.40)` noise — identical noise model to the crate's existing
  `src/main.rs` benchmark); one filter-admission draw `u_i ~ Uniform(0,1)` per pool member.
- A selectivity level `s` keeps exactly the candidates with `u_i < s`. Because the same
  admission draws are reused across every `s` within a trial, the filtered pool at a lower
  `s` is a strict *subset* of the pool at a higher `s` for that trial — a tighter control
  than redrawing independent filters per selectivity level, since it isolates the effect of
  `s` from trial-to-trial query/noise variance.
- Ground truth for recall@10 is `ExactL2Reranker`'s top-10 **on that same filtered pool**
  (not the corpus-global top-10), since a restrictive filter can legitimately remove the
  global top-k entirely; the fair question is how close each cheap reranker gets to the best
  achievable ordering of what the filter actually left behind.
- A trial is skipped for a given `s` if the filtered pool has fewer than `k=10` members
  (can happen by chance at `s=0.05`); the benchmark keeps drawing trial seeds until 200
  succeed per `s` and reports the exact count used (200/200 achieved at every level in the
  run below).
- `k_graph=8`, `k=10`, defaults for all rerankers (`GnnMincutReranker`: α=0.60,
  coherence_threshold=0.50; `AdaptiveCoherenceReranker`: α=0.60, keep_fraction=0.50,
  min_keep=1).
- Latency measured with `std::time::Instant` around each `rerank()` call in a release build;
  no new dependency was needed.

## 10. Results (raw, preserved)

Raw stdout: [`raw_benchmark_output.txt`](./raw_benchmark_output.txt).

| s (selectivity) | Variant | Mean recall@10 | ±stderr | Mean latency (µs) | p95 latency (µs) | n trials |
|---|---|---|---|---|---|---|
| 1.00 | NoisyScore | 29.80% | 0.78% | 30.52 | 51.75 | 200 |
| 1.00 | GnnMincut | 38.55% | 0.95% | 5943.06 | 7242.31 | 200 |
| 1.00 | AdaptiveCoherence | 40.05% | 1.02% | 5913.06 | 7208.54 | 200 |
| 0.50 | NoisyScore | 33.00% | 0.93% | 17.07 | 26.34 | 200 |
| 0.50 | GnnMincut | 38.75% | 1.10% | 1450.59 | 1931.53 | 200 |
| 0.50 | AdaptiveCoherence | 38.85% | 1.11% | 1450.97 | 1997.53 | 200 |
| 0.25 | NoisyScore | 40.40% | 0.96% | 6.77 | 7.89 | 200 |
| 0.25 | GnnMincut | 43.90% | 1.05% | 324.01 | 435.72 | 200 |
| 0.25 | AdaptiveCoherence | 42.85% | 1.04% | 311.64 | 419.07 | 200 |
| 0.10 | NoisyScore | 60.25% | 0.91% | 2.72 | 3.51 | 200 |
| 0.10 | GnnMincut | 61.00% | 0.93% | 55.05 | 94.56 | 200 |
| 0.10 | AdaptiveCoherence | 60.80% | 0.93% | 49.58 | 88.01 | 200 |
| 0.05 | NoisyScore | 84.75% | 0.92% | 1.49 | 2.08 | 200 |
| 0.05 | GnnMincut | 84.60% | 0.89% | 15.73 | 30.12 | 200 |
| 0.05 | AdaptiveCoherence | 85.15% | 0.90% | 14.61 | 26.67 | 200 |

**Acceptance check (measured):**

1. Unfiltered (s=1.00) regression vs GnnMincut: **+1.50 pp** (Adaptive is better) → budget
   ≤0.50 pp regression → **PASS**.
2. Mean low-selectivity (s≤0.10) recall@10 gain: **+0.18 pp** → threshold ≥3.00 pp →
   **FAIL**. 95% CI ≈ [−1.6, +2.0] pp — excludes the hypothesized +3 pp effect; this is a
   confident rejection, not a "can't tell" result, even though the two variants'
   *individual* recall estimates overlap substantially at every selectivity level (the
   difference is small and precisely estimated to be small, not imprecisely estimated).
3. Latency ratio (Adaptive / Mincut) ≤2.0x at every selectivity → observed 0.90x–1.00x
   (Adaptive is never slower) → **PASS**.
4. `cargo test -p ruvector-gnn-rerank` (24 tests, including 4 new) and
   `cargo clippy -p ruvector-gnn-rerank --all-targets -- -D warnings` → **PASS** (both green).

**Primary hypothesis: REJECTED.** Condition 2 is the hypothesis's own `then` clause; its
failure determines the outcome regardless of the other three conditions passing.

## 11. Memory / performance math

Per query, for a filtered pool of size `n`: `CandidateGraph::build` is O(n² × dim) (cosine
similarity between all pairs, computed once per upper triangle), and the diffusion pass is
O(n × k_graph). At `n=300, dim=128`: ~11.5M multiply-adds for graph construction, which
dominates wall-clock time at high selectivity (explains why `GnnMincutReranker` and
`AdaptiveCoherenceReranker` both cost ~5.9ms at s=1.00 vs ~15µs at s=0.05, where `n≈15`:
graph construction scales quadratically with the *filtered* pool size, not the unfiltered
retrieval_k). `AdaptiveCoherenceReranker`'s extra per-node sort-and-truncate step is O(k_graph
log k_graph) and negligible next to graph construction — consistent with the measured ≤1.00x
latency ratio.

## 12. Failure modes

- Zero-degree nodes (e.g., a filtered pool of size 1): both `GnnMincutReranker` and
  `AdaptiveCoherenceReranker` fall back to the unmodified `noisy_score`. Tested explicitly.
- At very low `n` (near `k`), `CandidateGraph::build`'s `k.min(n-1)` already caps degree;
  `AdaptiveCoherenceReranker`'s `min(deg)` clamp prevents `keep` from exceeding available
  edges.

## 13. Rejected alternatives

Per-node top-fraction gating (the implemented design) was the first and only gating
formulation implemented and tested — chosen a priori because it is the most direct way to
make a global absolute threshold "adapt" to local sparsity without introducing new
hyperparameters beyond a fraction and a floor. No other formulation (e.g., per-node z-score
thresholds, dynamic global-threshold recalibration from the realized pool's similarity
distribution) was implemented; they remain candidate next experiments (§20).

## 14. Security / governance

No new attack surface. Reranking is a read-only, in-memory transform over an already-fetched
candidate set; the shared `validate()` helper (unchanged) still rejects non-finite scores/
vectors and dimension mismatches for all four variants before any reranker-specific logic
runs.

## 15. MCP / WASM / edge / RVF / RVM / ruFlo implications

See §5 for RuVector-ecosystem fit; in brief: MCP — a narrow read-only `rerank_candidates`
tool is plausible for any variant but not specifically motivated by this result; WASM/edge —
the crate has no WASM target today and this change doesn't add one; RVF/RVM — not materially
relevant (no portable or coherence-domain state involved); ruFlo — nothing to adaptively
switch to, given no variant shows a selectivity-dependent advantage.

## 16. Practical applications

| User | Problem | RuVector capability | Ecosystem integration | Implementation path | Business value | Main risk | Horizon |
|---|---|---|---|---|---|---|---|
| Multi-tenant SaaS RAG operator | Per-tenant filtered search degrades rerank quality | `ruvector-gnn-rerank` + tenant filter | `ruvector-edge-tenancy` | Drop-in reranker swap | Retained recall under scoping | None validated here — do not deploy on this evidence | Now |
| Enterprise document search | ACL-scoped queries | Filtered candidate reranking | `ruvector-filter` | Same crate | Compliance-safe retrieval quality | False confidence if this rejected hypothesis is mistaken for a win | Now |
| Agent memory system | Namespace-scoped recall | Agent memory + reranking | `ruvector-agent-memory` | Benchmark harness reusable for namespace filters | Faster agent recall debugging | Needs a cluster-correlated filter model to be realistic | 1-3y |
| Security retrieval (SIEM/log search) | Time/severity-filtered search | Filtered ANN + rerank | `ruvector-edge-analytics` | Reuse benchmark methodology | Faster incident triage | Different filter correlation structure than tested | 1-3y |
| Code intelligence search | Repo/branch-scoped code search | Filtered rerank | `ruvector-filter` | Direct reuse | Better scoped code recall | Same caveat as above | Now |
| Edge anomaly detection | Device/fleet-scoped vector search | Lightweight rerank on edge | `ruvector-edge-index` | Needs latency re-validation on constrained hardware | Lower false negatives | Latency model here is server-class, not edge-validated | 3-5y |
| Scientific literature search | Field/date-filtered semantic search | Filtered rerank | `ruvector-filter` | Direct reuse | Faster literature triage | Same filter-correlation caveat | Now |
| Local-first personal assistant | Private, on-device filtered recall | Rerank on local memory | `ruvector-agent-memory` | Needs on-device latency validation | Privacy-preserving relevance | Benchmark here is not on-device hardware | 3-5y |

## 17. Long-horizon applications

| Thesis | Required advances | RuVector role | Why this experiment matters | Primary uncertainty | Falsification path |
|---|---|---|---|---|---|
| Self-healing graph memory under access-control churn | Dynamic re-gating as ACLs change | Coherence-gated reranking substrate | Establishes the measurement methodology for filter-sensitivity, even though this specific mechanism was rejected | Whether *any* local-gating scheme helps under realistic (correlated) filters | Re-run with cluster-correlated filters; if still no gain, close the whole direction |
| Agent operating systems with per-agent memory scoping | Fast, scoped recall at agent-swarm scale | `ruvector-agent-memory` + reranking | Filtered recall is a core OS-level primitive | Scaling behavior at >>5,000-vector corpora | Benchmark at 10-100x corpus scale |
| Synthetic nervous systems with gated signal propagation | Generalizing graph diffusion beyond ANN rerank | `ruvector-gnn`, `ruvector-mincut` | The per-node-adaptive-vs-global-threshold question recurs in any gated propagation system | Does the rejection here generalize to non-ANN propagation graphs? | Apply same experiment design to a non-retrieval propagation task |
| Proof-gated autonomous infrastructure | Combining witness/provenance with reranking decisions | `ruvector-proof-gate`, `ruvector-retrieval-receipt` | Establishes that reranker changes need measured evidence, not just plausibility | Governance overhead vs. benefit | Track whether any future reranker candidate clears this bar |
| RVM coherence domains | Per-domain reranking policy | `rvm` | Not materially exercised by this experiment | Whether reranking ever needs domain isolation | Revisit if a future reranker has state, unlike these |
| Robotics / embodied agent memory | Real-time filtered recall under motion/sensor constraints | `ruvector-robotics`, `ruvector-mmwave` | Latency-sensitivity measurement method transfers | Hard real-time constraints not modeled here | Re-run with hard latency budgets |
| Scientific autonomous systems | Filtered semantic search over evolving corpora | `ruvector-sota-bench` | Methodology for honestly rejecting a plausible-sounding idea | Generalization to non-synthetic corpora | Repeat on a real embedding dataset |
| Swarm memory / world models | Shared, filtered memory across many agents | `ruvector-collections`, `ruvector-mincut` | Filter-selectivity sensitivity is a shared concern across swarm memory designs | Multi-agent contention effects not modeled | Multi-agent concurrent-filter benchmark |

## 18. Promotion decision

**REJECT.** `AdaptiveCoherenceReranker` is implemented, tested, and benchmarked, but is
**not** recommended as a replacement for or complement to `GnnMincutReranker` based on this
evidence. It is kept in the crate as an additive, low-risk alternative with no demonstrated
advantage for the originally hypothesized scenario.

## 19. Witness evidence

- Starting commit: `687b58c4fad7561b105738b009b88865f72914c6` (branch
  `claude/focused-darwin-kux36n`, identical to `origin/main`'s tip at run start).
- Commands run (exact): `cargo test -p ruvector-gnn-rerank`,
  `cargo clippy -p ruvector-gnn-rerank --all-targets -- -D warnings`, `cargo fmt -p
  ruvector-gnn-rerank`, `cargo build --release -p ruvector-gnn-rerank --bin
  filtered_rerank_bench`, `./target/release/filtered_rerank_bench` (output captured verbatim
  to `raw_benchmark_output.txt`), plus a throwaway (not committed) Darwin-sweep example run
  via `cargo run --release -p ruvector-gnn-rerank --example _scratch_sweep`.
- Toolchain: `rustc 1.97.0 (2d8144b78 2026-07-07)`, `cargo 1.97.0 (c980f4866 2026-06-30)`.
- Hardware/OS: `Linux vm 6.18.44-fc-v180 x86_64`, 4 logical CPUs (`nproc`).
- No signed witness/provenance system was available in this environment (no
  `ruvector harness` CLI, no brain MCP) — this document plus the raw output file is the
  complete evidence trail for this run.

## 20. Production path

Not applicable — the hypothesis was rejected. If a future run retests with a
cluster-correlated filter model and finds a real effect, the production path would be:
feature-flag-free opt-in (the type already exists), A/B validate against production traffic
selectivity distributions, then consider as a default only with production telemetry
matching or exceeding the synthetic benchmark's claimed effect size.

## 21. Falsification criteria

Already applied: the hypothesis specified a concrete, falsifiable threshold (+3 pp mean
recall gain at low selectivity, 95% CI reasoning) and was falsified by the measured data
exactly as specified in advance.

## 22. Limitations

- Synthetic Gaussian-cluster corpus only; no real embedding dataset was used.
- Filter model is uniform-random per candidate, independent of cluster membership — likely
  the single biggest limitation, since it may not stress the global-threshold mechanism the
  way a real ACL/tenant filter (which is often correlated with semantic clusters, e.g. "this
  tenant's documents form their own sub-clusters") would.
- Single-threaded, server-class hardware latency only; no edge/WASM measurement.
- `k_graph=8` fixed throughout; not varied as an independent factor.

## 23. Next research

Re-run the same methodology with a filter model where filter-admissibility is correlated
with cluster membership (e.g., each cluster independently assigned "admissible" or not,
rather than per-candidate uniform draws) — this is a materially different and more realistic
sparsification pattern for tenant/ACL-style filters, and is the most direct way to test
whether the original intuition (global threshold collapses unevenly across sparsified local
neighbourhoods) holds under a filter model that actually produces that unevenness.

## 24. References

- GNRR, arXiv:2406.11720
- Maniscope, arXiv:2602.15860
- AQR-HNSW, arXiv:2602.21600
- `crates/ruvector-gnn-rerank` (ADR-194, nightly 2026-05-21) — prior art this experiment
  extends.
- `docs/adr/ADR-355-adaptive-coherence-filtered-rerank.md` — companion ADR for this run.
