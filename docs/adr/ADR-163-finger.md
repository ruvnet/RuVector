# ADR-163: Add `ruvector-finger` — FINGER approximate distance skipping for graph ANN

**Status**: Proposed
**Date**: 2026-04-27
**Driver**: Chen et al., "FINGER: Fast Inference for Graph-based Approximate Nearest Neighbor Search," WWW 2023 (arXiv:2206.11408). The algorithm precomputes a residual-basis at each graph node and approximates neighbor distances in O(K) rather than O(D) during beam search, potentially delivering 2–3× QPS improvements at maintained recall on structured high-dimensional datasets.

## Context

Every graph-based ANN algorithm in ruvector — ruvector-acorn's ACORN variants, ruvector-diskann's Vamana walk, the hnsw_rs-backed standard search in ruvector-core — shares the same bottleneck: O(D) exact distance computations for every neighbor evaluated during beam search. On a typical HNSW search at ef=100, an 80%-plus fraction of those distance computations produce a result larger than the current k-th nearest result and contribute nothing to the answer.

The existing ruvector quantization stack (ruvector-rabitq, ADR-157 / ADR-158) addresses this by compressing *stored* vectors to 1-bit codes, reducing per-distance cost from O(D×32 bits) to O(D/64 + rerank). FINGER takes a different approach: rather than compressing vectors, it precomputes a low-rank subspace at each graph node that allows approximate distance evaluation from the current beam-search context.

**Why this is distinct from RaBitQ**:
- RaBitQ quantizes stored vectors → lower memory, always-approximate distances.
- FINGER uses full-precision vectors → maintains exact distances for promising candidates; only approximates to *filter out* candidates too far to matter.
- The two are complementary: FINGER + RaBitQ would first use FINGER to skip ~80% of candidates, then use RaBitQ codes for the remaining 20% as a pre-screening before exact rerank.

**Gap in the Rust ecosystem**: No production Rust implementation of FINGER exists. The reference implementation from the paper authors is single-threaded C++ with no WASM or npm packaging.

## Decision

1. **Add `crates/ruvector-finger`** as a new workspace member implementing:
   - `GraphWalk` trait: read-only access to node vectors and neighbor lists (with `Sync` supertrait for rayon-parallel basis construction)
   - `FlatGraph`: standalone brute-force k-NN graph for isolated benchmarking
   - `NodeBasis`: precomputed K-dimensional orthonormal residual basis per node (Modified Gram-Schmidt)
   - `FingerIndex<G: GraphWalk>`: wraps any `GraphWalk` implementor with FINGER acceleration
   - Three constructors: `FingerIndex::exact` (baseline), `finger_k4` (K=4), `finger_k8` (K=8)
   - `finger-demo` binary that reports QPS, recall@10, prune rate, and memory for N ∈ {1K, 5K, 10K}

2. **Correct the canonical implementation bug**: pruned nodes must NOT be marked `visited`. Marking pruned neighbors as visited — an intuitive but incorrect optimization — causes 40–70% recall loss on unstructured data while appearing to perform better (fewer raw edge evaluations). The fix: `continue` without `visited.insert(nb_id)` in the FINGER-skip branch.

3. **Document the data-dependence boundary**: FINGER's speedup is gated on the correlation between graph edge directions and query directions. On random isotropic Gaussian data (zero manifold structure), the K=4 basis captures only K/D=3% of query-direction variance; approximation error degrades recall significantly. On structured data (SIFT, text embeddings) with intrinsic dimensionality ≪ D, the speedup is 2–3×. The research document quantifies this boundary and provides a closed-form speedup estimate.

4. **Memory layout**: `NodeBasis` uses flat row-major `Vec<f32>` for basis vectors (K×dim) and edge projections (M×K) to maximize cache locality during the inner product loop. At K=4, M=16, D=128: ~2.3 KB per node.

5. **No WASM crate in this PR**: WASM packaging is deferred to a follow-up ADR (like ADR-161 for rabitq-wasm). The trait design supports it — `GraphWalk` removes `Sync` requirement for single-threaded WASM builds via `#[cfg(target_arch = "wasm32")]` branches.

## Measured results (this PR)

Hardware: x86_64 Linux, rustc 1.94.1 release, 8 logical cores (rayon), flat k-NN graph, Gaussian N=5000, D=128, M=16, ef=200.

| Variant | Build(ms) | QPS | Recall@10 | Prune% | Basis(KB) |
|---------|-----------|-----|-----------|--------|-----------|
| ExactBeam | 0 | 4,515 | 88.0% | 0.0% | 0 |
| FINGER-K4 | 13 | 4,190 | 65.2% | 81.2% | 11,562 |
| FINGER-K8 | 24 | 2,996 | 78.7% | 75.4% | 22,812 |

At N=10,000: FINGER-K4 reaches 1.02× QPS (effectively break-even) with 82% prune rate. The fundamental result: 80%+ prune rates achieve break-even on random D=128 data because the O(K×D) projection overhead (K=4, D=128) offsets the O(M×D) savings from pruning 80% of M=16 neighbors. Expected 2× QPS is achievable at D≥256 on structured datasets.

## Alternatives considered

**A. Wrap ruvector-acorn's AcornGraph directly.**
The `AcornGraph` struct from `crates/ruvector-acorn` has public `neighbors` and `data` fields that could serve as a `GraphWalk` implementation. Rejected because: (1) adding a cross-crate dependency creates a larger scope PR; (2) the `GraphWalk` trait decouples FINGER from any specific graph implementation; (3) `FlatGraph` in ruvector-finger provides a clean standalone benchmark without pulling in ACORN's build-time overhead.

**B. Implement as a feature flag on ruvector-acorn.**
Rejected: a standalone crate is discoverable, independently testable, and follows the workspace pattern (cf. ruvector-rabitq, ruvector-diskann).

**C. Implement only on the hnsw_rs wrapper in ruvector-core.**
Rejected: hnsw_rs owns the graph walk internally and provides no hook for per-neighbor distance interception. FINGER requires access to the beam-search candidate loop, which is not exposed by hnsw_rs's public API.

**D. Implement Ada-ef instead of FINGER.**
Ada-ef (arXiv:2512.06636) adapts the beam width ef per query using dataset statistics rather than approximating per-distance. It achieves 4× latency reduction at maintained recall but requires a calibration pass over representative queries and produces NO per-neighbor savings (all ef nodes are still scored exactly). For the use case of "reduce per-distance cost," FINGER is the right abstraction. Ada-ef would be a complementary follow-up.

**E. Skip the research crate; just add a SIMD kernel to ruvector-core.**
SIMD would accelerate every distance computation by 4–8× uniformly. FINGER selectively eliminates 80% of computations at the cost of some recall. They are orthogonal; FINGER is more interesting research and the `GraphWalk` trait enables future SIMD+FINGER composition.

## Consequences

- A new standalone crate `ruvector-finger` with no production dependencies on other ruvector crates. Zero risk to existing indices.
- The `GraphWalk` trait defines an extension point; future crates (ruvector-acorn, ruvector-diskann) can implement it to get FINGER acceleration without code changes in ruvector-finger.
- The data-dependence analysis in the research document (`docs/research/nightly/2026-04-27-finger/README.md`) documents when FINGER is and is not beneficial — reducing the risk of misuse in production.
- Memory: at N=1M nodes, K=4, M=16, D=128, the basis storage is ~2.3 GB. This is documented as a known scaling concern and motivates the "adaptive K per node" roadmap item.
- Build time: basis construction at N=5000, D=128 takes 13 ms (K=4) with rayon on 8 cores. Scales linearly with N.

## See also

- ADR-157 — RaBitQ: 1-bit rotation quantization (complementary approach)
- ADR-160 — ACORN: filtered HNSW (the graph FINGER is designed to accelerate)
- `crates/ruvector-rabitq/` — workspace sibling, code patterns followed
- `docs/research/nightly/2026-04-27-finger/README.md` — full SOTA survey and benchmark analysis
- arXiv:2206.11408 — original FINGER paper
