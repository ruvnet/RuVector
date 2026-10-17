# 05 — Roadmap

Three phases. Each picks a coherent slice of the §02 candidate list,
specifies the files to touch, an acceptance test, and an LoC budget.
Each phase is sized to ~3–6 engineer-weeks. Phases are independent —
Phase 2 doesn't block on Phase 1 except where noted.

---

## Phase 1 — Low-hanging integrations (3 candidates, 4–5 weeks)

Pick the three Tier-A candidates from §02. They share three desirable
properties:

- All use Pattern 1 (direct embed) — no new infrastructure required.
- All can pin the same major version of `ruvector-rabitq` (`^2.2`).
- All have the consumer code already structured around vectors, so
  the integration is *adding a new index path*, not redesigning a
  hot loop.

### P1.A — `ruvector-diskann` RaBitQ backend

**Files to touch:**

- `crates/ruvector-diskann/Cargo.toml` — add `ruvector-rabitq = {
  path = "../ruvector-rabitq", version = "^2.2" }`.
- `crates/ruvector-diskann/src/index.rs` — add a `Backend` enum
  alongside the existing PQ path (`pq.rs:14`). New variant
  `Backend::Rabitq { plus: RabitqPlusIndex, seed: u64 }`. Constructor
  `DiskAnnIndex::new_rabitq(config, seed, rerank_factor)`.
- `crates/ruvector-diskann/src/index.rs:169` `search` — branch on
  backend; RaBitQ path calls `RabitqPlusIndex::search_with_rerank`.
- `crates/ruvector-diskann/src/index.rs:219,297` `save`/`load` —
  delegate to `ruvector_rabitq::persist::save_index/load_index`
  on the rabitq path.

**Acceptance test:** on the same dataset (Gaussian-clustered D=128
n=100k, the one in `crates/ruvector-rabitq/src/main.rs`), the rabitq
path achieves recall@10 ≥ 95% at QPS ≥ 2× the existing PQ path. New
file `crates/ruvector-diskann/tests/rabitq_backend_smoke.rs`.

**LoC budget:** ≤500 LoC source + ≤200 LoC tests.

### P1.B — `ruvector-graph` `VectorPropertyIndex`

**Files to touch:**

- `crates/ruvector-graph/Cargo.toml` — add the rabitq dep.
- `crates/ruvector-graph/src/index.rs` — new `VectorPropertyIndex`
  struct alongside `LabelIndex` (`:15`), `PropertyIndex` (`:79`),
  `EdgeTypeIndex` (`:180`), `AdjacencyIndex` (`:240`). Same
  lifecycle methods (`new`, `add_node`, `remove_node`, plus a new
  `knn(&self, property, query, k) -> Vec<NodeId>`).
- `crates/ruvector-graph/src/node.rs` — extend `Node` to carry
  optional vector-typed properties; or a side-table indexed by
  `NodeId`.
- New `crates/ruvector-graph/src/index.rs` regression: build a graph
  with 10k nodes carrying 128-dim embeddings, query top-k, assert
  recall ≥ 90% vs brute-force cosine.

**Acceptance test:** insert 10k node embeddings, run 100 queries,
recall@10 ≥ 90% vs an in-test brute-force cosine baseline; round-trip
the index to a `.rbpx` file via the new `save_property_index_rabitq`
and reload bit-identically.

**LoC budget:** ≤600 LoC source + ≤250 LoC tests.

### P1.C — `ruvector-gnn` `differentiable_search`

**Files to touch:**

- `crates/ruvector-gnn/Cargo.toml` — add the rabitq dep behind a
  default-on feature `rabitq` so the WASM build can opt out.
- `crates/ruvector-gnn/src/search.rs:56` `differentiable_search` —
  add a sibling `differentiable_search_rabitq(query, &
  RabitqPlusIndex, top_k, temperature)`. Top-k via
  `search_with_rerank`, softmax weights from the rerank f32 scores so
  gradients stay meaningful.
- `crates/ruvector-gnn/src/search.rs:105` `hierarchical_forward` —
  parameterise so callers can pass a per-layer
  `&RabitqPlusIndex` instead of a `&[Vec<f32>]`.

**Acceptance test:** on the existing `test_differentiable_search`
(`search.rs:204`), the rabitq path returns the same top-k ids and
softmax weights within 1e-3 vs the reference cosine path on D=128
n=10k. Also a new throughput micro-bench showing ≥ 2× QPS.

**LoC budget:** ≤300 LoC source + ≤150 LoC tests.

### Phase 1 acceptance gate

All three candidates merged with green tests, no regressions on
existing crate suites, no new clippy warnings. Total: 3 PRs, 4–5
engineer-weeks, ~1400 LoC source + 600 LoC tests.

### Phase 1 milestones

- **Week 1–2.** P1.A (DiskANN backend). Has the highest §02 strategic
  value and the sharpest existing call site for "where would 32×
  earn its keep" — ADR-154 already named it.
- **Week 2–3.** P1.B (graph VectorPropertyIndex). Independent of P1.A.
- **Week 3–4.** P1.C (GNN). Smallest, lands last.
- **Week 5.** Documentation pass: a §"Choosing a pattern" page added
  to each consumer's README citing §03; bench summary in
  `crates/ruvector-rabitq/BENCHMARK.md` extended with the three new
  call sites' numbers.

---

## Phase 2 — Make `VectorKernel` real (~4–6 weeks)

The trait at `crates/ruvector-rabitq/src/kernel.rs` is shipped but has
**zero non-test callers**. Phase 2 changes that by wiring two kernels
to two consumers — exactly the minimum to prove the dispatch policy
isn't paper.

### P2.A — Wire `VectorKernel` dispatch into ruLake

**Files to touch:**

- `crates/ruvector-rulake/src/lake.rs:590-630` — replace the doc-only
  `// plug-point` with a real `register_kernel(Arc<dyn VectorKernel>)`
  method, a `kernels: Vec<Arc<dyn VectorKernel>>` field, and the
  dispatch policy from ADR-157. Used inside `search_batch` (already
  has the right shape per its doc comment at `:595`).
- `crates/ruvector-rulake/src/cache.rs:833` — re-route the batch scan
  through the dispatcher.
- `crates/ruvector-rulake/tests/` — new `kernel_dispatch.rs` testing:
  (a) default kernel is `CpuKernel`; (b) registered determinism-false
  kernel is filtered on `Consistency::Frozen`; (c) batch_size <
  caps().min_batch is filtered.

**Acceptance test:** `RuLake::cache_stats()` exposes which kernel
served the last query (or last batch). Witness output is
bit-identical regardless of which deterministic kernel served.

### P2.B — Ship a portable SIMD `CpuSimdKernel`

**Files to touch:**

- `crates/ruvector-rabitq/src/kernel.rs` — add `CpuSimdKernel` behind
  feature flag `simd`. Implementation uses `std::simd` (when stable)
  or a `target_feature(enable = "avx2,popcnt")` portable path
  otherwise; falls back to scalar via the existing `CpuKernel` if
  detection fails.
- `crates/ruvector-rabitq/Cargo.toml` — add the `simd` feature.

**Acceptance test:** on the same Gaussian D=128 n=100k bench from
`crates/ruvector-rabitq/src/main.rs`, the SIMD kernel achieves ≥ 1.5×
QPS vs `CpuKernel` at bit-identical scan output (per the ADR-157 hard
gate).

### P2.C — Connect a second consumer

Pick **one** of B1 (`ruvector-attention` KV cache) or C3
(`ruvector-fpga-transformer`) for the second `VectorKernel` consumer.
This is the smaller-effort half — the dispatch is already real in
ruLake, the second consumer just adopts the same pattern.

Likely B1 because the kernel surface there is the closest match to the
existing rabitq hot path (asymmetric scan over a K-cache).

**Acceptance test:** B1's KV cache, with `RabitqAsymIndex` behind
`VectorKernel` dispatch, demonstrates the same bit-identical output on
CPU vs SIMD, and ships a benchmark showing the ratio.

### Phase 2 acceptance gate

Two consumers using `VectorKernel`, two kernels available
(`CpuKernel` + `CpuSimdKernel`). A first GPU kernel can land in
**Phase 2.5** as a separate `ruvector-rabitq-cuda` crate that passes
the ADR-157 acceptance gate (2× p95 OR 30% cost). Phase 2 itself does
not commit to GPU.

### Phase 2 LoC budget

~600 LoC ruLake dispatch + 800 LoC SIMD kernel + 400 LoC second-
consumer adoption + 500 LoC tests = ~2300 LoC across two crates and
one new feature.

---

## Phase 3 — Cross-cutting story (1–2 ADRs, no code commitment)

Phase 3 is a research-not-code phase. It commits the question
"should RaBitQ be the workspace's canonical vector compression
substrate?" and produces an ADR that either says yes (and lists the
consequences) or no (and lists the alternatives).

### P3.A — Draft ADR-160 "RaBitQ as the workspace's canonical 1-bit compression"

The ADR would say:

- All workspace crates that ship 1-bit binary vector compression use
  `ruvector-rabitq`. Re-implementations are PR-blocked (Anti-pattern
  A from §03).
- 4-bit, 8-bit, and PQ tiers are **not** subsumed — RaBitQ is the
  canonical *1-bit* path; ADR-001's tiered scheme stays for higher
  bitwidths.
- A migration plan for `ruvector-core::quantization::BinaryQuantized`
  (the original 15–20% recall path called out in ADR-154
  §"Measured gap"): deprecate, then delete, then point to RaBitQ.
- Cross-cutting impact on `ruvector-graph`, `ruvector-gnn`,
  `ruvector-attention`, `ruvector-temporal-tensor`, `ruvllm` — each
  named, each with its preferred §03 pattern, each with effort
  estimate.

### P3.B — Optionally, ADR-161 "Memory-substrate consolidation around ruLake"

Strictly downstream of ADR-156. If multiple new consumers (B2, B4,
plus future agent crates) end up sitting on `RuLake`, this ADR commits
that pattern: the agent-memory hierarchy, the ruvllm RAG cache, and
the rvAgent witness handoff are one substrate, not three.

### Phase 3 acceptance

ADR-160 in `docs/adr/` with status "Proposed", reviewed by maintainers
of the named consumer crates, and a one-page consequences section
rolled into the relevant crates' `Cargo.toml` comments. No code
changes — Phase 3 is the *decision* phase that drives Phases 4+ on the
quarterly roadmap.

### Phase 3 effort

~2 engineer-weeks split across writing + review. ADR-class work, not
implementation.

---

## Total roadmap effort

- **Phase 1:** 4–5 engineer-weeks, ~2000 LoC.
- **Phase 2:** 4–6 engineer-weeks, ~2300 LoC.
- **Phase 3:** ~2 engineer-weeks (docs).

**Total: ~10–13 engineer-weeks** to land three new consumers, make
the kernel trait load-bearing, and lock the workspace position. This
fits inside one quarter for a single engineer or 6 weeks for two.
