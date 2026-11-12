# Entropy-Adaptive Beam Search for ANN Graph Traversal

**Date**: 2026-08-13  
**Crate**: `ruvector-entropy-ann` (`crates/ruvector-entropy-ann`)  
**Status**: PoC complete — **negative result** (entropy signal does not work as hypothesised)  
**ADR**: [ADR-303](../../../adr/ADR-303-entropy-adaptive-ann.md)

---

## Summary of Outcome

The hypothesis — that the Shannon entropy of the candidate-heap distance distribution can serve as
a live, per-query beam-width control signal — is **refuted by measurement** on this PoC's data:

1. **No adaptivity**: `EntropyScaledEf` computes `ef_actual` = 122–124 for *every* query. A plain
   `FixedEf` baseline at the matched budget (ef=124) reproduces its recall **to four decimal
   places** on all three query sets. The recall gain over `FixedEf(50)` is entirely "2.5× the ef
   budget", not entropy.
2. **Wrong sign**: at temperatures where the softmin actually discriminates (T ≈ 0.001–0.01),
   hard queries have *lower* entropy than easy ones — softmin entropy over already-retrieved
   neighbour distances tracks local density, with the wrong sign for beam control. At the T=0.1
   used in the headline benchmark, the softmin is effectively at infinite temperature
   (H ≈ ln(n) for every query; spread ≈ 0.003 nats).
3. **The reactive gate never fires**: `EntropyThresholdBeam` is behaviourally identical to
   `FixedEf(50)` and only adds per-step entropy overhead.

The crate merges as an honestly-framed negative result plus a reusable benchmark harness with a
built-in matched-budget control.

---

## Motivation (original)

Standard HNSW search uses a fixed `ef_search` parameter chosen offline. This causes two symmetric
failures:

- **Easy queries** (near a tight cluster centroid): over-search — the result heap fills early but
  beam expansion continues, burning distance computations on neighbours already dominated.
- **Hard queries** (between clusters, out-of-distribution): under-search — the fixed budget runs
  out before the true nearest neighbours are reached.

Per-query adaptivity requires a *zero-calibration runtime signal* derived from the search itself.
The hypothesis was that heap-distance entropy provides that signal. It does not (see above).

---

## Prior Art

| Work | Venue | Adaptive signal | Unit |
|------|-------|----------------|------|
| Distance Adaptive Beam (arXiv:2505.15636) | NeurIPS 2025 | Scalar distance threshold | Scalar |
| EDEN (arXiv:2605.09745) | ICML 2026 | Entropy of beam tokens | LLM beam decode |
| Ada-ef (arXiv:2512.06636) | – | Predicted ef from query features | Offline regressor |
| Vespa adaptive HNSW | 2024 | Candidate list size heuristic | Scalar |
| **This work** | – | **Shannon entropy of heap distances** | **ANN graph traversal — refuted** |

Note that EDEN's claim is that entropy-based branching beats fixed branching **within the same
budget**. That is precisely the property this PoC fails to reproduce: the entropy variant only
wins by spending a larger budget.

---

## Entropy Semantics (hypothesis, refuted)

Given the current result heap, distances `{d_i}` are converted to a probability mass via softmin
at temperature *T*:

```
p_i = exp(-d_i / T) / Z,   Z = Σ exp(-d_j / T)
H   = -Σ p_i ln(p_i)
```

The hypothesised interpretation (high H → ambiguous → expand; low H → converged → stop) is not
what the measurements show. Softmin entropy over *retrieved-neighbour* distances measures how
locally dense the reached neighbourhood is relative to *T* — and on this data, hard queries land
in relatively denser local neighbourhoods, giving them *lower* entropy. The signal has the wrong
sign for beam control.

---

## Variants

Three variants are implemented in `src/search.rs`:

### 1. FixedEfSearch (baseline)

Standard beam search with fixed `ef_search` budget. The benchmark runs it at two budgets:
ef=50 (nominal) and ef=124 (matched to EntropyScaledEf's measured mean `ef_actual`) so the
adaptive claim stays falsifiable.

### 2. EntropyThresholdBeam

At each expansion step, if the result heap has ≥ k entries, compute H and stop when `H < h_stop`.
**Measured**: with brute-force entry the heap distribution is near-uniform for every query
(H ≈ ln(heap_size) ≈ 2.99 at T=0.1), so `h_stop=0.6` never fires. Behaviourally identical to
FixedEf(50), ~10–13 µs/query slower.

### 3. EntropyScaledEf

Probe the first `probe_depth=5` expansion steps at `base_ef`, then:

```
H_max     = ln(|results|)
scale     = clamp(1 + alpha * H / H_max, ef_min_factor, ef_max_factor)
ef_actual = base_ef * scale
```

**Measured**: H ≈ H_max for every query, so the clamp saturates at `ef_max_factor=2.5` and
`ef_actual` = 122–124 for every query. No per-query adaptivity; equivalent to FixedEf(124).

---

## Benchmark Results

**Environment**: macOS / aarch64, 12 CPU threads, release build (`opt-level=3 lto=thin`)  
**Dataset**: N=2000 vectors, D=16, 10 clusters, noise=0.2, k=10, ef=50, graph-k=16  
**Index size**: 375 KB total (125 KB vectors + 250 KB adjacency)  
**Methodology note**: ground-truth (brute-force) computation is performed *outside* the timed
closure; latencies below are ANN search only. (An earlier version of this table timed the
brute-force ground-truth scan inside the measured closure, which inflated all latencies and
masked the entropy variants' true overhead.)

| Variant | Queries | Recall@10 | Mean latency | p50 | p95 | QPS | Result |
|---------|---------|-----------|-------------|-----|-----|-----|--------|
| FixedEf(50) | easy | 0.870 | 32.9 µs | 31.3 µs | 41.2 µs | 30373 | PASS |
| FixedEf(50) | hard | 0.801 | 32.7 µs | 32.0 µs | 39.1 µs | 30551 | PASS |
| FixedEf(50) | mixed | 0.697 | 32.2 µs | 31.5 µs | 38.0 µs | 31032 | PASS |
| **FixedEf(124, matched)** | easy | **0.906** | 50.4 µs | 49.8 µs | 65.4 µs | 19825 | PASS |
| **FixedEf(124, matched)** | hard | **0.817** | 47.9 µs | 46.1 µs | 60.2 µs | 20877 | PASS |
| **FixedEf(124, matched)** | mixed | **0.736** | 47.9 µs | 45.8 µs | 64.2 µs | 20885 | PASS |
| EntropyThreshold | easy | 0.870 | 46.2 µs | 44.1 µs | 60.0 µs | 21616 | PASS |
| EntropyThreshold | hard | 0.801 | 43.7 µs | 42.9 µs | 50.0 µs | 22890 | PASS |
| EntropyThreshold | mixed | 0.697 | 42.6 µs | 41.7 µs | 50.2 µs | 23476 | PASS |
| EntropyScaledEf | easy | 0.906 | 49.7 µs | 49.2 µs | 66.0 µs | 20127 | PASS |
| EntropyScaledEf | hard | 0.817 | 49.9 µs | 48.7 µs | 62.2 µs | 20050 | PASS |
| EntropyScaledEf | mixed | 0.736 | 49.7 µs | 47.5 µs | 65.3 µs | 20115 | PASS |

**The decisive comparison** is EntropyScaledEf vs the matched-budget control:

| Query type | FixedEf(124, matched) | EntropyScaledEf | Δ recall |
|------------|----------------------|-----------------|----------|
| Easy | 0.906 | 0.906 | 0.0 |
| Hard | 0.817 | 0.817 | 0.0 |
| Mixed | 0.736 | 0.736 | 0.0 |

At the same effective budget, entropy contributes nothing. Meanwhile EntropyScaledEf costs
~50 µs/query vs ~33 µs for FixedEf(50) — a ~50% search-latency increase that buys exactly what a
larger fixed ef buys.

---

## Entropy Distribution Analysis

At T=0.1, entropy is ≈ ln(20) ≈ 2.996 for all query types:

| Query type | Mean H | p50 H | p95 H |
|------------|--------|-------|-------|
| Easy | 2.987 | 2.988 | 2.992 |
| Hard | 2.984 | 2.986 | 2.992 |
| Mixed | 2.984 | 2.987 | 2.993 |

**Why entropy does not discriminate at T=0.1**: relative to this dataset's distance scale, T=0.1
is effectively infinite temperature — softmin probabilities are near-uniform for every query, so
H ≈ ln(n) regardless of difficulty. In the usable range (T ≈ 0.001–0.01) the distributions do
differ, but the separation is **negative**: hard queries have lower entropy than easy ones. Note
the sign in the table above — hard-query mean H is already (marginally) *below* easy-query mean H.
The signal measures local neighbourhood density, not routing ambiguity.

Whether an approximate (multi-layer HNSW) entry point would produce a usable, correctly-signed
entropy signal is **untested conjecture**. Nothing in this PoC demonstrates it; it is listed under
Future Work as a hypothesis, not a mitigation.

---

## Limitations

1. **Flat single-layer graph**: no HNSW upper layers. Entry is brute-force O(n·dim).
2. **O(n²) graph construction**: acceptable for n=2000 but not production-scale.
3. **L2² metric only**: inner product and cosine require separate distance functions.
4. **No SIMD**: distance computation is scalar.
5. **Single-threaded**: beam search is sequential.
6. **Synthetic 16D data only**: none of the conclusions are validated on real embedding corpora.

---

## Future Work (all untested hypotheses)

1. **Frontier entropy instead of result entropy**: compute H over the *candidate frontier*
   (pre-retrieval) rather than retrieved results — may address the wrong-quantity problem.
2. **Multi-layer HNSW entry**: conjecture that approximate entries produce mixed-cluster early
   heaps where entropy discriminates. Must be tested against a matched-budget FixedEf control.
3. **Temperature calibration**: fit T to the dataset distance scale (T ≈ 0.001–0.01 here).
4. **Real datasets** (SIFT-1M, GIST-960): check whether the negative sign persists.

---

## Running

```bash
# All tests (15 assertions, no mocks)
cargo test -p ruvector-entropy-ann

# Release benchmark (12 variant × query-type rows, includes FixedEf(124) matched-budget control)
cargo run --release -p ruvector-entropy-ann --bin benchmark
```
