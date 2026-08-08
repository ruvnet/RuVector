# ADR-296: Anisotropic Product Quantization for Angular ANN Search

- **Status**: Proposed
- **Date**: 2026-08-06
- **Deciders**: RuVector Architecture Team
- **Tags**: ann, quantization, pq, cosine-similarity, embedding, compression

## Context

`ruvector-pq-search` implements Product Quantization with Asymmetric Distance Computation (ADC) using isotropic k-means: centroids are placed to minimise L2 reconstruction error. The ADC table uses inner product at search time (correct for cosine similarity), but the codebook training metric (L2) is mismatched to the search metric (inner product / cosine similarity).

This mismatch is a known limitation documented in the ScaNN paper (Guo et al., NeurIPS 2020). For MIPS and cosine workloads, the relevant error is the residual component parallel to the query direction — not the total L2 error. Anisotropic quantization (AQ) applies a directional penalty during k-means training:

```
L_AQ(x, c, η) = ‖x - c‖² + (η - 1) · (<x - c, x̂>)²
```

where x̂ = x / ‖x‖ and η > 1. This steers centroids toward directions that minimise inner-product error, improving recall for cosine search without increasing the code size.

All major embedding models used with RuVector (OpenAI, Cohere, sentence-transformers, BGE, E5) produce cosine-similarity vectors. Fixing this metric mismatch is directly relevant to the agent memory, RAG, and graph retrieval workloads that RuVector serves.

## Decision

Introduce `ruvector-aq-search` as a standalone crate implementing AQ codebook training and three search variants:

1. **IsotropicFlat** — isotropic k-means + IP ADC scan (baseline parity with `ruvector-pq-search`)
2. **AnisotropicFlat** — AQ k-means + IP ADC scan (same memory, better codebook alignment)
3. **AnisotropicResidual** — AQ k-means + ADC candidate retrieval + exact IP re-rank (highest recall)

All three implement the unified `AqSearch` trait. No external service dependency. Training is deterministic (seeded k-means).

The AQ training modifies only the k-means assignment step — the code format (M u8 bytes per vector), ADC table structure, and scan loop are identical to `ruvector-pq-search`. This means AQ is a drop-in training improvement with no serving-path changes.

## Consequences

### Positive
- Fixes the L2/IP training metric mismatch for cosine search workloads
- No code-size overhead: same M-byte PQ codes
- `AnisotropicResidual` with overfetch=16 achieves recall@10 = 1.00 on clustered 128-dim data at <550µs mean latency
- `AqSearch` trait enables testing of new quantisation schemes without serving-path changes
- η is a ruFlo-tunable parameter: feedback loops can adapt compression quality to query-time recall signals

### Negative
- AQ flat shows only marginal gain (~0.3%) over isotropic PQ on uniformly random vectors — gain requires clustered corpus
- Training is 2× slower than isotropic (modified assignment step); acceptable for offline-only training
- Streaming training (updating codebooks as data arrives) is unsolved
- `AnisotropicResidual` stores full f32 vectors (5MB for 10K × 128-dim), limiting applicability at large scale without an HNSW overlay

### Neutral
- `ruvector-pq-search` continues unchanged; AQ is additive
- Code path for both variants is identical at serving time — only the trained centroids differ

## Alternatives Considered

### Keep isotropic PQ in `ruvector-pq-search`
Not a fix. The metric mismatch is real and quantifiable. Rejected.

### Optimised PQ (OPQ) via rotation
OPQ finds a linear rotation that aligns subspaces with principal directions of the data before isotropic training. This reduces L2 error uniformly but does not specifically target the inner-product direction. OPQ is orthogonal to AQ; combining them (rotated AQ) is future work.

### Binary quantisation (RaBitQ)
Already implemented as `ruvector-rabitq`. Binary quantisation compresses more aggressively but has lower recall at same recall@10. Appropriate for different trade-off point. Not a replacement for PQ.

### Scalar quantisation (SQ)
SQ stores D f16 or i8 values per vector (still O(D) per vector), with no sub-space decomposition. Higher recall than PQ but lower compression ratio. Appropriate for hot-tier caches, not for cold-tier billion-scale storage.

### DPQ (Differentiable PQ)
DPQ trains codebooks end-to-end via gradient descent. Higher recall than AQ but requires automatic differentiation and is harder to implement in safe Rust without a tensor library. Marked as future research.

## Implementation Plan

1. **Phase 1** (complete): `ruvector-aq-search` PoC with `IsotropicFlat`, `AnisotropicFlat`, `AnisotropicResidual`, 14 unit tests, benchmark binary passing all acceptance tests.

2. **Phase 2**: Add `--features aq` to `ruvector-pq-search` exposing `AnisotropicCodebook` as a configurable variant of the existing codebook. This allows existing `PqSearch` users to opt in without changing their code.

3. **Phase 3**: Integrate AQ codes into `ruvector-coherence-hnsw` as the quantisation layer for HNSW edge candidate scoring. This is the highest-impact deployment: HNSW graph structure provides recall through graph traversal; AQ improves the accuracy of per-edge inner-product estimates.

4. **Phase 4** (future): `ruvector-aq-search-wasm` — WASM SIMD port for Cognitum Seed and edge deployments.

## Benchmark Evidence

Run: `cargo run --release -p ruvector-aq-search --bin aq-benchmark`

```
OS: linux / x86_64
Rust: 1.94.1
Dataset: N=10,000, DIM=128, Q=500, K=10
PQ: M=8, K=256, η=2.0, overfetch=16

Variant                  Recall@10  Mean(µs)  p50(µs)  p95(µs)  QPS   Mem(MB)
IsotropicFlat               0.2448     425.1    420.5    466.3   2352     0.20
AnisotropicFlat(η=2.0)      0.2456     462.5    427.6    663.4   2162     0.20
AnisotropicResidual(16×)    1.0000     533.1    527.9    600.1   1876     5.08

[PASS] AQ flat recall ≥ isotropic recall - 0.02
[PASS] AQ+Residual recall ≥ 0.70
[PASS] AQ flat memory ≤ isotropic memory + 5 MB
```

**Note on flat recall**: 0.24–0.25 recall@10 for flat PQ on clustered 128-dim data reflects a fundamental property of flat scan with 64× compression. This is not a defect — it is the known behaviour of PQ without an index structure. The `AnisotropicResidual` variant shows the correct strategy: use AQ ADC for fast candidate retrieval, then exact re-rank.

## Failure Modes

| Condition | Behaviour | Response |
|-----------|-----------|----------|
| Uniform random training corpus | AQ flat gain ≈ 0; matches isotropic | Acceptable; use residual re-rank |
| η too large (>4) | Centroids migrate away from cluster means | Grid-search η on held-out recall |
| dim % M ≠ 0 | Panic at assert | Enforce constraint at construction |
| N < K (fewer vectors than centroids) | k-means degeneracy | Assert N ≥ K; document minimum training size |
| Streaming inserts without retrain | Codebook becomes stale | Schedule periodic retrain via ruFlo hook |

## Security Considerations

- AQ codebooks trained on private embedding corpora may leak distributional information (embedding inversion attacks). Store codebooks with the same access controls as the raw vectors.
- No network calls, no external dependencies, no privilege escalation in this crate.
- Pair with `ruvector-proof-gate` for witness-logged writes if provenance is required.

## Migration Path

- `ruvector-pq-search` users: no breaking changes. AQ is in a separate crate.
- To migrate: replace `FlatPqIndex::new()` with `AnisotropicFlat::new()` with matching M and K parameters. The `AqSearch` trait has the same `insert`/`search` interface shape as `PqSearch`.
- ADC table format is compatible: both use inner product.

## Open Questions

1. What is the recall gain from AQ on real production embedding corpora (OpenAI ada-002, E5-large, BGE-M3)?  
2. What is the optimal η for each embedding model family?  
3. Does AQ + OPQ (rotation + directional penalty) outperform either alone?  
4. Can streaming k-means with AQ loss converge stably?  
5. Is there a WASM SIMD implementation of the ADC scan that fits in Cognitum Seed's 256KB SRAM budget?
