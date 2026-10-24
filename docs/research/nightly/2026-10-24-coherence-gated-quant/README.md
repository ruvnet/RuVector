# Coherence-Gated Mixed-Precision Quantization — Nightly Research (2026-10-24)

## 1. Title

Coherence-Gated Mixed-Precision Quantization for Agent Memory (ADR-353)

## 2. Summary

We tested whether `ruvector-attn-mincut`'s dynamic min-cut gate — built for
attention sparsity — can double as a per-vector *local coherence* signal
that drives mixed-precision allocation between `ruvector-turboquant`'s
Turbo4 (4-bit) and 1-bit codes. **Result: REJECT.** The gate is a binary
"does a cheap separating cut exist" detector; on a single unimodal
Gaussian cluster (the realistic shape of one agent-memory topic) no cheap
cut exists regardless of edge-weight choice, so the signal collapses to a
constant and precision allocation degenerates into a near-random split.
Measured recall@10 fell from 0.657 (uniform Turbo4 baseline) to 0.301 at
30% high-precision allocation — well outside the acceptance threshold.

## 3. Abstract

Agent memory stores accumulate embeddings with very uneven local
redundancy: near-duplicate observations cluster tightly, while boundary or
novel observations sit in low-density regions. Uniform quantization spends
the same bit budget on both. We hypothesized that `ruvector-attn-mincut`'s
existing min-cut machinery — already shipped in this repo for attention
gating — could cheaply supply a per-vector "how structurally redundant is
this point" score to drive Turbo4-vs-1-bit allocation in
`ruvector-turboquant`, without inventing a new coherence metric. We built
the full pipeline (synthetic clustered corpus, block-wise coherence
scoring reusing the real `dynamic_min_cut` API, three encoding variants,
honest brute-force-verified recall/memory/latency measurement) and found
the specific reuse path does not work on realistic data, with a clear,
reproducible root cause.

## 4. Hypothesis

> Given a clustered agent-memory corpus (40 clusters × 150 vectors, dim
> 128), when per-vector Turbo4-vs-1-bit precision is allocated by a
> mincut-derived local coherence score (candidate_A) instead of uniform
> Turbo4 (baseline), then recall@10-per-byte should improve by ≥15%
> relative to baseline, subject to recall@10 staying ≥0.9× baseline and
> mean query latency staying ≤2× baseline.

Falsification condition (fixed before running the benchmark): recall@10
drops more than 10% without a ≥15% recall-per-byte gain, or recall-per-byte
does not improve at all.

## 5. Why this matters now (2026)

Agent memory stores (RuVector's own `ruvector-agent-memory`,
`ruvector-memory-admission`) are exactly the regime where this would pay
off: long-lived stores accumulate far more near-duplicate context than
novel context, and bit budget is the dominant cost at scale. If a coherence
signal RuVector already computes for attention gating could be reused for
storage-tier precision allocation at zero extra algorithmic surface area,
that's free leverage across two previously separate subsystems.

## 6. Long horizon thesis

**2036**: edge agents with bounded flash/RAM budgets need memory stores
that spend bits where they matter — this is a storage-tier instance of the
broader "spend compute/bits proportional to information content" principle
that recurs across ANN, attention, and model compression research.
**2046**: if agent operating systems standardize on a coherence primitive
(RVM coherence domains being one candidate), every subsystem that currently
reinvents its own notion of "important vs. redundant" — attention gating,
memory admission, quantization, forgetting — could share one scored graph
instead of five bespoke heuristics. This experiment is a data point against
naively assuming today's attention-shaped coherence primitive is already
that shared abstraction.

## 7. RuVector ecosystem fit

Connects: `ruvector-attn-mincut` (coherence source), `ruvector-turboquant`
(Turbo4 + 1-bit codec), agent memory (the target workload shape),
RVF/RVM (discussed below as forward-looking fits), and witness/ADR
conventions already established by the mincut-gated-forgetting and
mincut-partition-determinism nightly topics (2026-09-05, 2026-09-11) this
one deliberately avoids duplicating by testing a *different* consumer of
the same primitive (storage precision, not admission/forgetting).

## 8. MetaHarness role

`npx metaharness --help` is installed (v0.4.17) but is a project *scaffold*
generator (`npx metaharness <name>`, vertical templates, `score`/`analyze`/
`genome` subcommands against an external repo target) — it does not expose
a research-orchestration API usable against this repository's existing
crates. `npx ruvector harness doctor/status` is **not installed** (`npm
error: could not determine executable to run`). No MetaHarness role was
exercised this run; the goal-decomposition / critic-separation / evidence
roles in this report were performed directly instead of through a tool
that does not exist yet in this checkout.

## 9. Flywheel role

No `ruvector harness flywheel` CLI is installed in this environment (see
above). This document and its ADR serve as the durable record in its
place: the **Flywheel update** (§21 below) captures what would otherwise be
written into that system.

## 10. Darwin role

Not run. Darwin evolution only makes sense once a candidate clears its
basic acceptance gate so there is a validated axis (e.g. `high_precision_frac`,
`residual_frac`) worth exploring; candidate_A failed that gate outright, so
spending a bounded evolutionary budget tuning its parameters would not
change the underlying root cause (the coherence signal itself carries no
information on this workload shape) and was skipped rather than run for
appearance's sake.

## 11. Architecture

```mermaid
flowchart LR
    subgraph Corpus
        A[Synthetic clustered corpus\n40 x 150, dim 128]
    end
    subgraph Coherence
        A --> B[Per-cluster self-similarity graph\ncosine, diagonal removed]
        B --> C["dynamic_min_cut (ruvector-attn-mincut)\ns = nearest-to-centroid, t = farthest"]
        C --> D[coherence(i) = kept_edges(i) / total_edges(i)]
    end
    subgraph Allocation
        D --> E{Variant}
        E -->|baseline| F[Turbo4 4-bit for all]
        E -->|candidate_A| G[bottom 30% least-coherent: Turbo4\nrest: 1-bit]
        E -->|candidate_B| H[same split + bottom 5%: + exact f32 residual]
    end
    subgraph Measurement
        F --> I[Brute-force ground truth\nrecall@10, bytes/vector, latency]
        G --> I
        H --> I
    end
```

## 12. Implementation

New crate: `crates/ruvector-coherence-quant` (workspace member, depends on
`ruvector-attn-mincut` and `ruvector-turboquant` as path deps — no new
external dependencies).

- `src/dataset.rs` — deterministic SplitMix64-seeded clustered Gaussian
  corpus + a realistic "perturbed existing memory" query workload.
- `src/coherence.rs` — builds the per-cluster self-similarity graph and
  calls the real, unmodified `ruvector_attn_mincut::dynamic_min_cut`
  (source = nearest point to centroid, sink = farthest, so the global cut
  is asked to separate "core" from "fringe"); derives
  `coherence(i) = kept_incident_edges(i) / total_incident_edges(i)`.
- `src/allocator.rs` — three variants (`Baseline`, `CandidateA`,
  `CandidateB`) using `Turbo4Codec::encode_dual` for Turbo4 + 1-bit codes
  from one rotation pass, plus exact-residual caching for `CandidateB`.
  Precision selection is **exact-rank** (`select_bottom_frac`), not a value
  threshold — an early implementation used a percentile *value* cutoff and
  silently selected 100% of vectors whenever coherence scores tied (which,
  per the root-cause finding below, is the common case); this was caught
  by `tests/correctness.rs` and fixed before the accepted benchmark ran.
- `src/metrics.rs` — recall@k and latency-percentile helpers, unit tested
  against known distributions.
- `examples/benchmark.rs` — the full, reproducible experiment. Run with:

  ```bash
  cargo run --release -p ruvector-coherence-quant --example benchmark
  ```

## 13. Benchmark methodology

- Hardware: this session's container (`x86_64`, see `uname -a` in run log).
- Rust 1.97.0 / Cargo 1.97.0, `cargo build --release`.
- Corpus: 40 Gaussian clusters × 150 points, dim 128, seed 42; cluster
  spread randomized 0.3–1.1× per cluster (not uniform difficulty).
- Queries: 200, seed 99, built by perturbing a random corpus point by
  small Gaussian noise (the realistic "retrieve the memory most like this
  new observation" pattern, not a copy of a corpus vector).
- Ground truth: exact brute-force L2 top-10 over the **full** corpus,
  f32, no quantization — never approximated.
- `high_precision_frac = 0.30`, `residual_frac = 0.05`, Turbo4 rotation
  seed 7, `MinCutConfig::default()` (`λ=0.5, τ=2, ε=0.01`) — all fixed
  before the run, not tuned after seeing results.
- Each variant measured on the same corpus/query/ground-truth triple; one
  pass each (deterministic — repeats are identical given fixed seeds, so
  variance reporting across repeats would not add information here, unlike
  a real wall-clock-timing benchmark on contended hardware).

## 14. Benchmark results (raw, from this run)

```
corpus: 40 clusters x 150 = 6000 vectors, dim=128, seed=42
queries: 200, seed=99, k=10
high_precision_frac=0.3, residual_frac=0.05, rotation_seed=7
coherence scoring (mincut over 40 blocks): 185.8 ms
mean coherence score: 1.0000
computing ground truth (exact brute-force L2 over full corpus)...
ground truth: 128.1 ms

variant                                     bytes/vec  t4_frac  b1_frac exact_frac  encode_ms    recall@10    mean_us     p95_us
baseline (uniform Turbo4)                        72.0     1.00     0.00      0.00       24.5        0.657      252.5      283.0
candidate_A (coherence-gated)                    38.4     0.30     0.70      0.00       24.5        0.301      487.9      545.0
candidate_B (coherence-gated + residual)         65.7     0.30     0.70      0.05       23.8        0.280      534.7      645.0

recall@10 per byte: baseline=0.009118 candidate_A=0.007839 (-14.0%) candidate_B=0.004261 (-53.3%)

acceptance checks (candidate_A vs baseline):
  recall@10 >= 0.9x baseline: false (0.301 vs 0.657)
  recall-per-byte gain >= 15%: false (-14.0%)
  mean query latency <= 2x baseline: true (487.9us vs 252.5us)

ACCEPTANCE RESULT: REJECT
```

Reproduce with the exact command in §12. `cargo test -p ruvector-coherence-quant`
runs the correctness/determinism tests (5 unit + 3 integration, all green).

## 15. Root cause (the actual finding)

`mean coherence score: 1.0000` is the signature of total signal collapse.
We diagnosed it with two controlled toy graphs run directly against the
unmodified `dynamic_min_cut`:

| Toy graph | True structure | `cut_cost` returned | Gate fired? |
|---|---|---|---|
| 6-node complete digraph, uniform weight 1.0 | no cheap separation exists | `0` (reported only when nothing is cut) | No |
| Two 3-cliques joined by one weight-0.02 bridge edge | one obvious cheap cut | `0.02` (exactly the bridge) | Yes, correctly |

The bridge case proves the mechanism is correct: it finds the true minimum
cut and reports its cost. The dense case proves the limitation: `cut_cost`
in `GatingResult` is *cost of the cut that was actually applied*, reported
as `0` whenever the λ-gate declines to cut at all (because the graph's
true min-cut exceeds `λ · mean_weight`). A single unimodal Gaussian
cluster — the shape every corpus cluster in this benchmark has, and the
realistic shape of one topic in an agent-memory store — is, as an affinity
graph, exactly the "dense, no cheap separation" case: we confirmed this
on the real corpus (one 150-point cluster, cosine weights: 22,350/22,500
possible edges present, mean weight 0.91, true min-cut cost exceeding the
λ=0.5 threshold of 0.45) and it did **not** change under a Gaussian-RBF
edge-weight re-parameterization or a λ sweep from 0.1 to 0.8 — because the
underlying graph genuinely has no community boundary to find. Min-cut
detects *between-community* boundaries; a unimodal density gradient
(center vs. fringe of one blob) is not a community boundary, it is a
continuous property no binary graph cut can expose.

**Conclusion: this is not a tuning problem.** `dynamic_min_cut`'s λ-gate
cannot supply a continuous local-coherence gradient over a unimodal
cluster under any edge-weight or λ choice we tested, because no such
gradient is representable as "a cheap cut exists or doesn't." The
hypothesis that this specific primitive, used as the field's own public
API intends it to be used, could drive storage-tier precision allocation
is **falsified** for this workload shape.

## 16. Memory math

Turbo4 code: `dim/2 + 8 = 72` bytes at dim 128. 1-bit code:
`ceil(128/64)*8 + 8 = 24` bytes. Exact f32 residual: `128*4 = 512` bytes,
additive on top of a Turbo4 code. At the measured allocation
(`t4_frac=0.30, b1_frac=0.70`): `0.30*72 + 0.70*24 = 38.4` bytes/vector —
matches candidate_A's measured 38.4 exactly, confirming the accounting is
correct; the recall collapse is a selection-quality problem, not a
bookkeeping one.

## 17. Performance math

Query cost scales with corpus size regardless of variant (every variant
scores the whole 6000-vector corpus per query here — no index is built;
this prototype measures quantization quality, not ANN traversal speed).
candidate_A/B are slower per query than baseline (488/535 µs vs 253 µs)
because mixed precision costs an extra `Bits1Query` preparation per query
and because the Bit1 scorer's popcount kernel, while bandwidth-light per
candidate, is evaluated here without the oversampled-cascade pattern
`ruvector-turboquant` actually ships with (`bits1.rs`'s own doc comment:
"the cheap candidate-generation plane", meant to prefilter before a
Turbo4 rescore of a small shortlist — using it to score 70% of a flat
corpus directly, as this prototype does, is not its intended fast path).
Still within the fixed ≤2× latency gate.

## 18. Failure modes

- Coherence signal collapse on unimodal clusters (root cause, above).
- Mixed-precision ranking bias: Turbo4 asymmetric distance, 1-bit
  popcount distance, and exact L2 are not guaranteed to be calibrated
  against each other, so ranking them jointly (as `allocator::estimate_distance`
  does) can itself distort recall independent of the coherence-selection
  problem. Not isolated in this run — a real limitation of the harness as
  built, noted for anyone reusing this code.
- Percentile-threshold-with-ties bug (found and fixed before the accepted
  run, documented in §12 and covered by `tests/correctness.rs`) — kept in
  this report because it is exactly the kind of error a degenerate
  coherence signal invites, and future reuse of similar scoring should
  test for it explicitly.

## 19. Rejected alternatives

- **Bypass the λ-gate via `DinicSolver` directly** to get the raw,
  unconditional min-cut instead of the gated one: `DinicSolver::new` is
  private in `ruvector-attn-mincut` (only `dynamic_min_cut` is a usable
  public entry point), so this is not possible without a change to that
  crate. Recorded as a concrete, actionable API suggestion (§24) rather
  than worked around with a hand-rolled max-flow copy, which would no
  longer be "reusing the ecosystem primitive" — the thing under test.
- **Tune λ instead of root-causing**: swept 0.1–0.8 against both cosine
  and RBF edge weights (§15); cut_cost stayed 0 throughout. Confirmed this
  is structural, not a parameter-search problem, before writing this up.
- **Multi-modal sub-clusters per "cluster"** (which would give min-cut a
  real community boundary to find) was considered but rejected for this
  run: it would validate a different, narrower hypothesis ("mincut finds
  coherence boundaries when they exist, which we already know from the
  bridge toy case") rather than the one that was formalized before
  benchmarking, which concerns realistic agent-memory topic clusters
  (unimodal). Noted as the natural next experiment (§25).

## 20. Security

No new attack surface: the crate only reads in-process vectors and runs
deterministic arithmetic; it holds no secrets, makes no network calls, and
is not wired into any serving path. The one security-relevant finding is
documentation, not code: anyone reusing `dynamic_min_cut`'s `cut_cost`
field as a magnitude (rather than "cost of the cut that was applied, 0 if
none") would silently mis-measure coherence the same way this prototype
initially did — worth a doc clarification upstream in
`ruvector-attn-mincut` regardless of this experiment's outcome.

## 21. Governance

This result should block (not merely caution against) any future proposal
to wire `dynamic_min_cut` into agent-memory admission/forgetting/compaction
decisions on unimodal-cluster-shaped data without first addressing the gate
collapse — those nightly topics (2026-09-02, 2026-09-05) use mincut for a
*different* purpose (gating low-weight edges entirely, not scoring a
continuous precision gradient) and are not contradicted by this finding,
but anyone extending them toward continuous scoring should read this ADR
first.

## 22. MCP implications

None proposed. The underlying primitive doesn't produce a usable signal
for this use case, so there's nothing production-worthy to expose yet.

## 23. WASM implications

Not evaluated — gated on the hypothesis first producing a working
candidate. `ruvector-turboquant` and `ruvector-attn-mincut` are both
already pure-Rust, dependency-light, and used elsewhere in WASM contexts
in this repo, so a future working version of this idea would likely port
cleanly; that's inherited, not demonstrated by this run.

## 24. RVF implications

If a working coherence-gated precision scheme is found later (§25), it is
a natural fit for an RVF snapshot: precision allocation is exactly the
kind of per-entry metadata that benefits from deterministic replay (same
corpus + same coherence pass ⇒ same allocation, as this prototype's
determinism tests already enforce) and copy-on-write update (re-scoring
only the block whose membership changed). Not integrated this run.

## 25. RVM implications

A coherence domain boundary is conceptually close to what this experiment
needed and did not get from `dynamic_min_cut`: RVM's isolation domains are
about *enforcing* a boundary once established, not *discovering* one in a
density gradient, so RVM is a consumer of a working coherence signal, not
a fix for today's finding.

## 26. ruFlo implications

A working version would be a natural ruFlo continuous-maintenance
workflow: re-score coherence and reallocate precision as an agent-memory
store accumulates new entries, analogous to index-repair workflows already
described for other nightly topics. Not implemented; gated on §25 above.

## 27. Practical applications

1. **Agent long-term memory compaction** — User: an agent runtime with a
   growing memory store. Problem: uniform quantization wastes bits on
   near-duplicate context. RuVector capability: `ruvector-agent-memory` +
   a *working* coherence signal (not this one). Integration: compaction
   pass. Path: needs §25's open question resolved first. Value: lower
   memory footprint at fixed recall. Risk: recall regression on boundary
   memories (exactly what this run measured). Horizon: near-term once
   fixed.
2. **Edge agent memory under a byte budget** — similar, with WASM/edge
   deployment as the integration point; same gating on a fixed signal.
3. **Enterprise RAG index tiering** — hot/cold precision tiers by access
   pattern rather than structural coherence; a different, likely more
   tractable, signal than this run tested.
4. **Code intelligence embeddings** — code embeddings cluster around
   near-duplicate boilerplate; worth retesting this hypothesis there
   specifically because near-duplicate boilerplate plus genuinely distinct
   call sites may be closer to the *bimodal* structure min-cut can detect,
   unlike this run's unimodal synthetic clusters.
5. **MCP memory servers** — same compaction idea, exposed over MCP once a
   signal works.
6. **Security/anomaly retrieval stores** — anomalies are naturally
   boundary/outlier points; if retested there, the workload may be closer
   to the bridge-toy-case shape (two populations, not one blob) where this
   run's own toy diagnostic shows the mechanism works correctly.
7. **Scientific embedding archives** — long-tail, rarely-requested entries
   as 1-bit candidates, frequently-clustered reference entries at full
   precision — same gating as #1.
8. **Local-first personal assistants** — small on-device memory budget;
   same gating as #1/#2, smaller scale.

## 28. Long horizon applications

1. **Self-healing graph memory** — requires a coherence signal that
   survives inserts/deletes; this run didn't test either, having failed
   earlier on the static case.
2. **Synthetic nervous systems / world models** — coherence-weighted
   attention and storage sharing one scored graph (§6); this run is
   evidence that today's primitive isn't yet that shared abstraction.
3. **Agent operating systems** — a kernel-level "importance" primitive
   used by scheduling, memory, and attention alike; same gap.
4. **Autonomous edge cognition** — byte-budget-aware memory under this
   primitive; gated on a fix.
5. **Swarm memory** — shared coherence scoring across agents; gated on a
   fix, and on a cross-agent consistency story not explored here.
6. **RVM coherence domains as the stable abstraction** — §25.
7. **Proof-gated autonomous infrastructure** — witness-backed coherence
   decisions; this run's witness evidence (§29) is itself a small step
   here (recording a negative result verifiably, not just asserting it).
8. **Robotics memory** — same byte-budget story under real-time
   constraints, a harder latency bar than this run's.

## 29. Evolution results

Not run (§10).

## 30. Promotion decision

**Not promoted.** `candidate_A` and `candidate_B` are both rejected on
their own measured numbers (recall@10 collapse). The parent (uniform
Turbo4, `ruvector-turboquant`'s existing production path) is retained
unchanged. This nightly run's deliverable is the falsified hypothesis plus
its root cause, not a code change to production paths.

## 31. Witness evidence

- Repo commit at run start: see PR base.
- Run identity: UTC date 2026-10-24; Rust 1.97.0 / Cargo 1.97.0 (see
  `rustc --version` in session log); `uname -a`: `Linux ... 6.18.44-fc-v153
  ... x86_64 GNU/Linux`.
- Exact benchmark command and raw stdout: §14 of this document (copied
  verbatim from `cargo run --release -p ruvector-coherence-quant --example
  benchmark`, not hand-edited).
- Deterministic seeds recorded in §13; re-running the same command on the
  same toolchain reproduces identical numbers (coherence scoring and
  dataset generation are both seeded, no wall-clock-dependent randomness
  in the measured metrics other than latency µs, which is hardware-timing
  noise, not seed-dependent).
- No signed witness chain or Flywheel ledger is wired into this checkout
  (§8–9); this document is the evidence record.

## 32. Production path

None — not promoted. Should a future run fix the coherence-signal problem
(§15, §25), the production path would be: land it as an opt-in encoding
mode in `ruvector-turboquant` (not a new always-on default), gated behind
its own feature flag, with the same brute-force-verified recall testing
this harness already does.

## 33. Falsification criteria

Already met: recall@10 fell well below the 0.9× floor (0.301 vs. required
≥0.591) and recall-per-byte regressed instead of improving by the required
15%. Per §10 of the harness instructions, this is recorded as REJECT, not
reinterpreted as inconclusive or silently re-thresholded.

## 34. Limitations

- Synthetic corpus only; real agent-memory embedding distributions may be
  less unimodal-per-topic than this generator (which would help), or may
  be *more* unimodal at fine topic granularity (which would hurt) — this
  run does not resolve which.
- Mixed-precision ranking calibration bias (§18) is unquantified.
- Single run per variant (deterministic given fixed seeds); no variance
  across random corpus instances was measured, which would matter more for
  a borderline result than for this clear a rejection.
- `ruvector-attn-mincut`'s `GatingResult.cut_cost` semantics (reports 0 iff
  the gate declined to cut, not "the true min-cut value") is undocumented
  in that crate; we inferred it from source and confirmed it with the toy
  diagnostics in §15, but this is worth a doc fix in that crate independent
  of this experiment.

## 35. Next research

1. Retest the identical harness on a **synthetic bimodal** corpus (two
   sub-populations per topic cluster) to confirm the mechanism produces a
   usable, non-degenerate signal exactly where the toy bridge case
   predicts it should — closing the loop on §15's root cause before trying
   it on any real embedding distribution.
2. If (1) succeeds, retest on a **real embedding corpus** (code embeddings
   or any embedding model already available in this repo) to check whether
   real topic clusters are closer to unimodal (this run's failure mode) or
   bimodal/multimodal (where the mechanism works).
3. Independently of mincut: test a **plain local-density** coherence score
   (e.g. k-NN average distance or a Gaussian-mixture responsibility) as a
   simpler, continuous-by-construction alternative signal for the same
   precision-allocation idea — decoupling "is coherence-gated quantization
   useful at all" from "is mincut the right coherence source."
4. File the `GatingResult.cut_cost` documentation gap found in §15/§34
   against `ruvector-attn-mincut` directly (doc-only fix, independent of
   whether (1)–(3) ever land).

## 36. References

- ADR-296 (Turbo4 4-bit quantized datatype), ADR-297 (1-bit candidate
  plane) — `crates/ruvector-turboquant`.
- `ruvector-attn-mincut` crate docs (`crates/ruvector-attn-mincut/src/lib.rs`,
  `mincut.rs`, `gating.rs`) — Dinic's algorithm, dynamic min-cut gating.
- Prior nightly topics reused for convention and explicitly not duplicated:
  `2026-09-02-mincut-streaming-memory-admission`,
  `2026-09-05-mincut-gated-forgetting`,
  `2026-09-11-mincut-partition-determinism`,
  `2026-06-20-pq-adc-search`.
- Dinic, E. A. (1970). Algorithm for solution of a problem of maximum flow
  in a network with power estimation. (The max-flow/min-cut algorithm this
  crate implements.)
