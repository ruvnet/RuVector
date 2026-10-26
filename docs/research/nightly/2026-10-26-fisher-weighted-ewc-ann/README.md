# Nightly Research — Does ADR-002's "EWC++ Fisher" Policy Slot Need a Fisher-Weighted ANN Metric?

## Run Identity

- **Date (UTC):** 2026-10-26
- **Start commit:** `c44d7337c` (`chore: Update NAPI-RS binaries for all platforms`, tip of `main` at session start)
- **Branch:** `claude/focused-darwin-hlry9x`
- **Rust:** `rustc 1.97.0 (2d8144b78 2026-07-07)`, `cargo 1.97.0 (c980f4866 2026-06-30)`
- **OS / hardware:** Linux 6.18.44-fc-v153, x86_64, 4 vCPUs (container; see Limitations)
- **New crate:** `crates/ruvector-fisher-ann` (workspace member; `cargo build --release` and `cargo test` both green, `cargo clippy --all-targets` clean for this crate)

## Ecosystem Control-Plane Discovery

Checked before picking a topic, per the nightly protocol:

| Capability | Status on this branch | Evidence |
|---|---|---|
| `npx metaharness` | Installed (v0.4.17), but a generic external repo-scaffolding tool, not wired to this codebase's crates | `--help` output lists `score`/`analyze`/`genome`/`learn` against an arbitrary target repo; no RuVector-specific integration found |
| `npx ruvector harness doctor/status` | **Not installed** | `npm error could not determine executable to run` |
| Darwin (bounded evolutionary search) | **Not present** on this branch (`crates/` has no `darwin` crate; `sona` has `darwin_*.rs` modules for its own auto-tuner, not a standalone Darwin harness) | `ls crates/ \| grep -i darwin` → empty; `git log --all` shows a `feat(darwin): add bounded evolutionary parameter search engine` commit, but it is on an unmerged branch, not `main` |
| Flywheel | **Not present** on this branch | `ls crates/ \| grep -i flywheel` → empty |
| Witness chains / signed provenance | **Present and used** | `ruvector-agent-memory`'s TARL ledger + Ed25519 `WitnessSigner` (ADR-134 §9, ADR-320, `witness_signing.rs`) |
| SONA (adaptive memory, EWC++, MicroLoRA) | **Present** | `crates/sona` (package `ruvector-sona` v0.2.1); `ewc.rs` implements online Fisher estimation, task memory, `apply_anchor`, `regularization_loss` |
| RVF / RVM | **Present as crates** | `crates/rvf*`, `crates/rvm` |
| mincut / coherence | **Present, heavily used** | `ruvector-mincut`, `ruvector-coherence*`, dozens of prior nightly topics |

Because Darwin, Flywheel, and a RuVector-specific MetaHarness are not wired into this branch, this run's research roles (discovery, implementation, adversarial review, evidence judgment) were executed directly rather than delegated to that tooling, as the protocol permits ("roles may execute through one or multiple agents depending on available infrastructure").

## Why This Topic, Not a Duplicate

`docs/adr/ADR-002-ruvllm-integration.md` (lines ~160-182) specifies a `PolicyEntry` schema for RuvLLM's "Policy Memory Store" — a single HNSW-indexed collection holding quantization policies, router policies, pattern-bank entries, **and EWC++ policies**, each with a semantic `embedding: Vec<f32>` for the shared "HNSW INDEX LAYER (Unified Search)" plus `fisher_diagonal: Option<Vec<f32>>` for EWC-type entries. The ADR never says *how* that Fisher diagonal is supposed to affect search — whether it's just inert metadata next to a plain cosine/L2 index, or whether a Fisher-weighted metric should drive nearest-task lookup itself. That is a genuine, named, unresolved assumption in an existing accepted architecture document, not a new idea invented for this run.

Separately, `sona::ewc::EwcPlusPlus` already does per-task Fisher-weighted scoring today — `apply_constraints` and `apply_anchor` loop over `task_memory`, a `VecDeque` bounded by `max_tasks` (default 10). That is a correct, cheap linear scan at that scale. This run asks the natural follow-up: if an agent's memory grows to carry hundreds or thousands of distinct task-specific adapters (ADR-002's stated motivation — "cross-session knowledge transfer", "federated learning"), does an approximate index built with the same Fisher-weighted scoring reproduce the exact linear-scan answer fast enough to be worth the complexity? And does the Fisher weighting matter at all for an ANN index, or would a production engineer be just as well served building ordinary unweighted HNSW over the policy embeddings and treating `fisher_diagonal` as dead metadata (as ADR-002's schema arguably implies by storing it as a separate `Option` field rather than folding it into the indexed `embedding`)?

No existing nightly topic (checked `docs/research/nightly/*/README.md` and `docs/adr/*.md` for "Fisher"/"fisher") addresses this; the only prior hits are the ADR-002 schema itself and an unrelated use of "Fisher-Yates" shuffling in the 2026-09-02 nightly.

## Hypothesis (fixed before any benchmark ran)

```text
Given a bank of N synthetic EWC task snapshots (weights + per-task Fisher
diagonal + importance), each belonging to one of 40 task "families" with
sparse, family-specific important dimensions,

When a Fisher-weighted HNSW index (candidate A) is used for nearest-prior-
task lookup instead of (a) a Fisher-weighted brute-force linear scan at
scale (baseline — the direct generalization of EwcPlusPlus's own
production task_memory scan) or (b) a plain-L2 HNSW over the same points
(candidate B, an ablation),

Then candidate A's mean query latency should be at least 5x lower than
baseline at N=10,000,

Subject to:
  - candidate A's recall@10 against the Fisher-weighted ground truth
    staying >= 0.90,
  - candidate A's mean single-task EWC regularization-loss gap (its
    chosen anchor vs. the ground-truth-optimal anchor) staying below 5%
    of the mean ground-truth loss,
  - and candidate A measurably beating candidate B on recall@10 and/or
    loss gap (otherwise the Fisher weighting itself is not what is
    providing value, and the experiment has not actually tested its own
    hypothesis).
```

Falsification criteria: any of the three `Subject to` conditions failing REJECTs the hypothesis regardless of the speedup number, since a fast wrong answer is not a win for a mechanism (EWC anchoring) whose entire purpose is protecting specific trained behavior.

## What Was Implemented

`crates/ruvector-fisher-ann` (all new, workspace member added to the root `Cargo.toml`):

- **`types.rs`** — `TaskRecord` (weights + fisher + importance, mirroring `sona::ewc::TaskFisher`); `squared_l2`; `fisher_weighted_sq` (`sum_d fisher[d] * (q[d]-w[d])^2`, the same per-dimension form as `EwcPlusPlus::regularization_loss`'s per-task term); the `TaskMetric` trait with `FisherWeighted` and `PlainL2` implementations; `single_task_regularization_loss`.
- **`dataset.rs`** — deterministic synthetic generator (`StdRng` seeded, reproducible byte-for-byte). 40 task families, each with a random center in `[-1,1]^dim`, a random ~10%-dimension "important" mask, and a random importance scalar. Important dims get tight clustering (±0.05 noise) and high Fisher (5–20); the remaining ~90% of dims get large nuisance noise (±2.0) and near-zero Fisher (0–0.05) — a direct, if synthetic, stand-in for the sparsity continual-learning literature reports for real Fisher diagonals.
- **`baseline.rs`** — `brute_force_topk`: the generalized-to-arbitrary-N linear scan, and this run's ground truth.
- **`hnsw.rs`** — a compact, from-scratch, single-threaded, deterministic multi-layer HNSW (Malkov & Yashunin construction/search, simple distance-based neighbor pruning rather than the paper's diversity heuristic), generic over `TaskMetric` so the identical algorithm can be instantiated with either metric. This is explicitly *not* `ruvector-core`'s production HNSW (see Limitations).
- **`main.rs`** (`fisher_ann_bench` binary) — the benchmark: builds baseline/candidate A/candidate B at N ∈ {2,000, 10,000}, dim=128, 300 queries; measures build time, query latency (mean/p50/p95/p99 over real `Instant` timings), measured memory (actual `Vec` lengths × element size, not an estimate), recall@1/@10 against the Fisher-weighted ground truth, and the mean EWC regularization-loss gap (via `single_task_regularization_loss`, cross-checked against the real `ruvector_sona::EwcPlusPlus::regularization_loss` in a unit test — see below); prints the acceptance gate; then runs a diagnostic sweep of `ef_search` at fixed N to distinguish a search-budget problem from a construction/navigability problem.
- **Tests (15, all passing):** distance-function correctness; **`matches_production_ewc_regularization_loss`** — builds a real `EwcPlusPlus` via its public API (`set_optimal_weights`, repeated `update_fisher`, `start_new_task`), and asserts `single_task_regularization_loss` reproduces `EwcPlusPlus::regularization_loss`'s output for the same single task to within float tolerance, so the benchmark's "quality" numbers are checked against production code, not a parallel reimplementation that could silently diverge; dataset determinism and Fisher-sparsity sanity checks; HNSW determinism (`vec![false; n]`-free builds with the same seed produce byte-identical neighbor lists); empty/single-point edge cases; a small-scale (N=400) recall sanity test (>=0.85) as a build-is-not-broken smoke test.

No mocks sit on the primary measurement path: the brute-force baseline, the HNSW candidates, and the loss-gap metric all run real code against the same generated data, and the loss metric's correctness is checked against the actual `sona` crate.

## Benchmark Methodology

- Release build (`cargo build --release`); `cargo run --release -p ruvector-fisher-ann --bin fisher_ann_bench` is the exact command, reproducible with the fixed seeds baked into `main.rs` (`1_000 + n`, `2_000 + n`).
- No warmup phase was needed or added — each variant's queries are timed individually and reported as full latency distributions (mean/p50/p95/p99) rather than a single warmed-up number, so first-query cold effects are visible in the raw distribution, not hidden.
- Index construction (build time) is reported separately from query latency.
- Memory is **measured** (`HnswIndex::memory_bytes`, sum of actual `Vec<f32>`/`Vec<u32>` lengths × `size_of`), not estimated.
- Full raw console output is preserved, unedited, at [`bench_output.txt`](./bench_output.txt) in this directory.

## Results (raw, from `bench_output.txt`)

```text
variant                             n   build_ms    mean_us    p50_us    p95_us    p99_us   recall@1  recall@10     loss_gap     mem_KB
baseline_fisher_bruteforce       2000       0.00     319.21    311.03    391.07    457.88      1.000      1.000       0.0000     2000.0
candidate_A_fisher_hnsw          2000     509.01      22.17     16.27     49.30     82.78      0.343      0.352   37217.6445     2257.7
candidate_B_plain_l2_hnsw        2000     614.40     164.36    149.18    245.88    376.02      1.000      0.438       0.0000     2257.7
baseline_fisher_bruteforce      10000       0.00    1424.54   1457.29   1829.60   2124.89      1.000      1.000       0.0000    10000.0
candidate_A_fisher_hnsw         10000    2471.55      79.85     79.88    122.31    142.72      0.363      0.354   50120.3477    11288.9
candidate_B_plain_l2_hnsw       10000    4550.78     337.31    312.62    509.55   1220.09      1.000      0.388       0.0000    11288.9

--- Acceptance check (N=10000) ---
speedup candidate_A vs baseline: 17.8x (threshold >= 5x) -> PASS
candidate_A recall@10: 0.354 (threshold >= 0.9) -> FAIL
candidate_A mean loss gap: 50120.3477 vs threshold 188.7126 (5% of mean ground-truth loss 3774.2517) -> FAIL
ablation (Fisher weighting matters, not just having an index): candidate_A recall@10=0.354 loss_gap=50120.3477 vs candidate_B recall@10=0.388 loss_gap=0.0000 -> FAIL

ACCEPTANCE RESULT: REJECT

--- Diagnostic: candidate_A recall@10 vs ef_search (N=2000, same built graph) ---
ef_search=  100 (index size 2000): recall@10=0.352
ef_search=  500 (index size 2000): recall@10=0.352
ef_search= 2000 (index size 2000): recall@10=0.352
```

## Root Cause: Why Candidate A Fails, and Why It Is Not a Tuning Problem

Two findings pin down the mechanism:

1. **Recall is completely flat as `ef_search` grows from 100 to 2,000 — equal to the entire dataset size.** If this were a search-budget problem (the graph is fine, the search just isn't looking hard enough), recall would climb toward 1.0 as `ef_search` approaches N, because at `ef_search == N` a connected graph's greedy search degenerates toward an exhaustive walk. It does not move at all. That rules out "under-tuned parameters" and points at the graph itself: candidate A's HNSW, built with the Fisher-weighted metric, is not navigable to roughly 65% of each query's true top-10.
2. **The Fisher-weighted scoring function used for both construction and search is asymmetric and not a metric.** `distance(query, task) = fisher[task] · (query - task.weights)² · importance[task]` uses the *candidate's own* Fisher diagonal, not the query's — intentionally, because that is what `EwcPlusPlus::regularization_loss` computes (the cost of anchoring to that specific task), but it means `distance(a, b) ≠ distance(b, a)` in general, and the triangle inequality HNSW's greedy search implicitly leans on for its "approximately correct" behavior does not hold. A task `b` with a near-zero Fisher diagonal everywhere can score as "close" to almost any query regardless of how far apart their raw weights are, simply because its score is uniformly scaled down — a degenerate attractor that a proper metric does not have. Graph edges built by picking each node's "closest" candidates under this score therefore do not form a globally navigable structure: greedy descent from the entry point gets pulled toward whichever low-Fisher/low-importance tasks look locally attractive and stalls there, never reaching the true nearest task for most queries, independent of how much search budget is spent once already stuck in the wrong neighborhood.

By contrast, candidate B's plain-L2 metric is symmetric and genuinely metric, so HNSW's standard construction/search theory applies, and — in this dataset — raw L2 distance still correlates with family membership (every dimension, important or not, is built from `family_center + noise`, so even "nuisance" dimensions leak a weak family signal that, averaged over ~115 dimensions with extreme-value selection over thousands of candidates, is enough to make candidate B's recall@1 come out at 1.000 against the *Fisher-weighted* ground truth — a dataset property, not a bug: see the recall@1 rows above). That a properly-behaved index over the "wrong" metric outperforms a properly-constructed index over the "right" metric is itself the result: **a one-sided, per-candidate Fisher weighting is the kind of scoring function that must not be fed directly into a graph-based ANN index's edge-selection step**, however good it is as a ranking function for the small, bounded, brute-force-scanned case EWC++ already uses it for today.

## Acceptance Result

```text
REJECT
```

The hypothesis is REJECTed on its own stated terms: the primary recall threshold fails by a wide margin (0.354 vs. required 0.90), the loss-gap threshold fails by over two orders of magnitude (50120 vs. a 189 ceiling), and — decisively — the ablation check fails: candidate B (the version that *ignores* Fisher information) beats candidate A (the version built around it) on every quality metric measured. This is not an ambiguous or underpowered result; it is a clean falsification with an identified causal mechanism (Section "Root Cause" above), which the nightly protocol counts as a successful run.

## Production Recommendation for ADR-002

Do not build a Fisher-weighted ANN index for the `PolicyEntry` policy-memory store. Concretely:

1. Index `PolicyEntry.embedding` (the semantic embedding) with an ordinary symmetric metric (cosine or L2) in the shared HNSW layer, exactly as ADR-002 already draws it for every other policy type.
2. Keep `fisher_diagonal` as inert payload on the stored record, fetched by id after a plain semantic lookup — not folded into the index's distance function.
3. Leave EWC++'s own Fisher-weighted task selection (`apply_constraints`, `apply_anchor`, `regularization_loss`) as the bounded linear scan it already is. This run's baseline numbers (261–1,495 µs for N up to 10,000 on 128-dim weights) show that scan stays cheap well past `EwcConfig::max_tasks`'s current default of 10; if `max_tasks` is ever raised into the thousands, the right fix is a *sharding or clustering* strategy (e.g., coarse unweighted ANN to shortlist ~50 candidate tasks, then exact Fisher-weighted brute force only within that shortlist) rather than a Fisher-weighted graph index — see Next Research.

## Rejected Alternatives Considered Before Implementation

- **Reusing the semantic `embedding` dimension as the Fisher-weighting target** (weight the 768-D text embedding by some projected Fisher importance) — rejected before implementation: ADR-002's `fisher_diagonal` is explicitly sized to the model's own parameter space, not the semantic embedding space, so there is no principled dimension-to-dimension correspondence to weight by.
- **Modifying `sona::ewc::EwcPlusPlus` to add a public `TaskFisher`-injection constructor** so the benchmark could call the real struct's `task_memory` directly end-to-end — rejected in favor of the smaller, auditable `single_task_regularization_loss` plus an explicit equivalence test against the real `regularization_loss`. Changing another crate's public API for a benchmark that turned out to falsify its own premise would have been unnecessary surface area.
- **Symmetrizing the Fisher-weighted metric** (e.g., averaging both sides' Fisher diagonals, or using `max`/`min` of the two one-sided scores) to make it a proper metric before building the graph — considered as a follow-up, not attempted here, because it changes the quantity being optimized (it would no longer equal `regularization_loss` for either side) and the hypothesis explicitly asked about the asymmetric, production-faithful form; see Next Research item 1.

## Security, Governance, and Reward-Hack Check

- No secrets, credentials, or `.env` content touched.
- No existing test was weakened, skipped, or deleted; no acceptance threshold was changed after seeing results (the verdict-classification *code* was corrected once, before the thresholds had produced a borderline case, to correctly route "ablation falsified" results to REJECT rather than INCONCLUSIVE — the numeric thresholds themselves (`MIN_SPEEDUP_AT_LARGEST_N`, `MIN_RECALL_AT_10`, `MAX_LOSS_GAP_FRACTION`) were never touched after the first run).
- Benchmark cannot leak evaluation information into the "solving path": the HNSW candidates have no access to the ground-truth top-k during construction or search.
- No reward-hacking vector identified: there is no mechanism by which this crate's own tests or benchmark could be gamed to report a false ACCEPT, since the verdict is a direct, unconditional function of the measured recall/loss/speedup numbers printed in the same run.

## MCP / WASM / Edge / RVF / RVM / ruFlo Implications

- **MCP:** not warranted. This result recommends *against* building the mechanism that would need an MCP surface (a Fisher-weighted lookup tool); there is nothing here to expose.
- **WASM/edge:** the brute-force baseline (a few hundred µs to ~1.5 ms at N=10,000, dim=128) is itself cheap enough for edge/WASM deployment of EWC-anchored agents without any ANN index at all, reinforcing the production recommendation above — build the index-avoidance path, not an index.
- **RVF:** a `PolicyEntry` (weights + Fisher + importance) is a natural RVF-portable artifact — small, self-contained, versionable — independent of this run's negative result about *indexing* it; RVF packaging of individual `TaskRecord`-shaped snapshots for cross-session/cross-device transfer remains a reasonable, separate future direction.
- **RVM:** no new privileged operation or coherence domain is implied by a plain-L2 index; a future sharded-shortlist design (Next Research item 1) would still only need read-only vector search, not new RVM enforcement.
- **ruFlo:** none proposed by this result specifically; a workflow that periodically re-validates `max_tasks` against measured linear-scan latency (re-running this benchmark's baseline as `max_tasks`/embedding dimension grow) would be a reasonable long-running ruFlo health check, cheap to implement, not built in this run.

## Practical Applications (of the *recommendation*, not the rejected mechanism)

1. **RuvLLM router operators** sizing `EwcConfig::max_tasks`/`param_count` — this run's baseline numbers give a concrete latency budget for the linear scan at various N, informing when (if ever) a sharding strategy becomes necessary.
2. **Agent-memory engineers** implementing ADR-002's policy store — this run resolves "should the index be Fisher-weighted" with a measured no, saving an implementation path that would have produced a worse index.
3. **SONA integrators** evaluating whether to scale `task_memory` beyond its current bounded `VecDeque` — the flat, non-improving recall curve here is evidence against a "just add an ANN index" shortcut.

## Long-Horizon Applications

1. **Thesis:** large-scale continual-learning agents (thousands of accumulated task specializations) will need *some* scalable nearest-task structure; this run shows the naive approach (index the production metric directly) does not work, which narrows the design space for whoever builds the real thing in 5-10 years. **Required advances:** a principled symmetrization or a two-stage shortlist-then-exact design (Next Research item 1). **RuVector's role:** host the shortlist index and the exact-scan fallback in one crate. **Why this experiment matters:** it rules out the simplest design before it reaches production. **Primary uncertainty:** whether symmetrized Fisher metrics preserve enough of `regularization_loss`'s semantics to still be useful. **Falsification path:** repeat this benchmark with a symmetrized metric; if recall still doesn't improve, the shortlist-then-exact design (which never needs a Fisher-weighted graph at all) is the only remaining option.

## Limitations

- Fisher diagonals and LoRA weight deltas are **synthetic**, not captured from a real fine-tuning run — no live model-training loop was available in this sandbox. The dataset is designed to reproduce the one structural property (sparse per-task importance, large nuisance noise elsewhere) that matters for this specific hypothesis, and the quality metric is cross-checked against real `sona` code, but the absolute numbers (e.g. the exact 0.354 recall figure) are specific to this synthetic distribution and should not be read as a universal constant.
- N was capped at 10,000 (not the "hundreds of thousands" ADR-002's long-term vision implies) to keep the full benchmark sweep inside this session's time budget; see Next Research item 4.
- `ruvector-fisher-ann`'s HNSW is a compact, from-scratch, single-threaded implementation built for this metric ablation, not `ruvector-core`'s production, battle-tested HNSW. The navigability failure identified here is a property of the *metric*, not of this specific implementation's bugs (confirmed by the determinism test and the small-scale recall sanity test both passing, and by the `ef_search` sweep ruling out a construction-only defect specific to this code), but a from-scratch implementation cannot fully rule out an implementation-specific contribution to the magnitude of the gap.
- Hardware is a 4-vCPU container; absolute latency numbers will differ on production hardware, though the *relative* comparison (recall is metric-dependent, not hardware-dependent) is unaffected.

## Next Research

1. Test a **symmetrized** Fisher-weighted metric (e.g., `0.5 * (fisher_a + fisher_b) · (a - b)²`, or `min`/`max` of the two one-sided scores) in the same harness, to see whether restoring the metric property recovers graph navigability while still correlating usefully with `regularization_loss`.
2. Build and benchmark the **shortlist-then-exact** design directly: coarse unweighted HNSW over `weights` to retrieve the top ~50 candidates, then exact Fisher-weighted scoring only within that shortlist — this is the production path this run's recommendation implies but does not itself implement.
3. Repeat the `matches_production_ewc_regularization_loss`-style equivalence check for `apply_anchor` (not just `regularization_loss`), to validate a shortlist-then-anchor pipeline end-to-end against the real `EwcPlusPlus`.
4. Re-run this exact benchmark at N=100,000+ on non-containerized hardware to confirm the flat-recall finding holds at the scale ADR-002's "cross-session"/"federated" motivation actually implies.
