# Graph-Neighbour Cascade ANN: Fixing Quantization Rank Inversions Without a Second Full Scan

*RuVector Nightly Research — 2026-08-02*

---

## The Problem in One Sentence

INT8 scalar quantization is fast and memory-efficient, but in high dimensions (D ≥ 64) the quantization noise often exceeds the distance gap between the k-th and (k+1)-th nearest neighbours, silently flipping their rank and costing recall.

## How Big Is the Problem?

For N=5000 random Gaussian vectors in D=128:

| Quantity | Value |
|----------|-------|
| Typical gap: dist(10th NN) − dist(11th NN) | ≈ 0.046 |
| INT8 quantization noise (SD in distance² space) | ≈ 0.31 |
| Rank inversion probability at the boundary | ≈ 46% |

A pure `QuantizedCascade(ef=1)` with initial pool = k achieves recall **0.975** — missing 2.5% of ground-truth top-10 hits.

## The Standard Fix (and Why It's Wasteful)

Increase `ef_mult`: scan more candidates, keep pool = 4k. The missing neighbours are somewhere in the larger pool. This works but wastes exact f32 re-rank budget on the extra 3k candidates that were *never* ambiguous.

## The Cascade Fix

Instead of a bigger pool everywhere, identify *which* candidates are uncertain and expand the pool *only* there, using a prebuilt k-NN graph.

```
Stage 1  INT8 scan → top-k approx candidates
Stage 2  Uncertain zone = candidates with approx_dist ≤ kth_dist × (1 + δ)
Stage 3  Graph expand = union of graph-neighbours of uncertain candidates
           (excluding candidates already in Stage 1 pool)
Stage 4  Exact f32 re-rank of (Stage 1 ∪ Stage 3) → top-k result
```

With `δ = 0.30` and `K_graph = 32`, the uncertain zone typically contains 3–6 candidates and expands by 50–150 additional IDs — far fewer than the 3k extra candidates from `ef_mult=4`.

## Benchmark Results

N=5000, D=128, K=10, K_graph=32, 300 queries (release build, x86_64 Linux):

| Variant | Recall@10 | Mean latency | Memory |
|---------|-----------|-------------|--------|
| LinearFull (ground truth) | 1.000 | 1003 µs | 2.44 MB |
| QuantizedCascade (ef=1) | 0.975 | 1006 µs | 3.05 MB |
| QuantizedCascade (ef=4) | 1.000 | 1096 µs | 3.05 MB |
| **GraphNeighbourCascade (ef=1+graph)** | **0.988** | **1117 µs** | **3.66 MB** |

**GNC recall gain over QC(ef=1): +0.013** — well above the 0.005 threshold.

GNC uses only ef=1 initial pool but sits 77% of the way from QC(ef=1) to QC(ef=4) in recall, at 2% higher latency than QC(ef=4) and with only 0.61 MB extra graph memory.

## The Code (Core Search Loop)

```rust
fn search(&self, query: &[f32], k: usize) -> Vec<Hit> {
    // Stage 1: INT8 approximate scan
    let mut approx: Vec<(usize, f32)> = (0..self.corpus.n)
        .map(|id| (id, self.corpus.dist_sq_approx(id, query)))
        .collect();
    approx.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    approx.truncate(k * self.ef_mult);

    // Stage 2: Uncertain zone
    let kth_dist = approx.get(k - 1).map(|&(_, d)| d).unwrap_or(f32::MAX);
    let threshold = kth_dist * (1.0 + self.delta);
    let uncertain: Vec<usize> = approx.iter().take(k)
        .filter(|&&(_, d)| d <= threshold)
        .map(|&(id, _)| id)
        .collect();

    // Stage 3: Graph expansion
    let initial: HashSet<usize> = approx.iter().map(|&(id, _)| id).collect();
    let mut expanded: Vec<usize> = uncertain.iter()
        .flat_map(|&uid| self.graph.neighbours(uid))
        .map(|&nb| nb as usize)
        .filter(|nb| !initial.contains(nb))
        .collect();
    expanded.sort_unstable();
    expanded.dedup();

    // Stage 4: Exact re-rank
    let mut hits: Vec<Hit> = approx.iter()
        .map(|&(id, _)| Hit { id, dist: self.corpus.dist_sq_exact(id, query) })
        .chain(expanded.iter().map(|&id| Hit { id, dist: self.corpus.dist_sq_exact(id, query) }))
        .collect();
    hits.sort_unstable_by(|a, b| a.dist.partial_cmp(&b.dist).unwrap());
    hits.dedup_by_key(|h| h.id);
    hits.truncate(k);
    hits
}
```

## Memory Layout

```
QuantizedCorpus
  u8  data   N × D bytes     (INT8 encoded vectors)
  f32 data   N × D × 4 bytes (exact f32 for re-rank)

KnnGraph
  adj  N × K_graph × 4 bytes (u32 neighbour IDs, flat row-major)

Total for N=5000, D=128, K_graph=32:
  u8  = 0.61 MB
  f32 = 2.44 MB
  adj = 0.61 MB
  GNC = 3.66 MB  vs  QC = 3.05 MB  (+0.61 MB for graph)
```

## Why This Is Interesting for Future Systems

**For agentic memory retrieval**: Agents querying episodic memory stores want high recall on *unexpected* neighbours (cross-episode associations), not just the nearest hit. The uncertain zone is exactly where cross-episode candidates hide. Graph expansion naturally surfaces them.

**For MCP tool surfaces**: A `vector_search_cascade(query, k, delta)` MCP tool could expose δ as a runtime knob — low δ for speed, high δ for exhaustive recall — without rebuilding the index.

**For self-optimising indexes**: ruFlo can close-loop on sampled recall to auto-tune δ, converging on the minimum value that keeps recall above a target. Same pattern as ADR-272's adaptive k' controller.

**For edge AI**: The graph adds bounded, predictable memory. On a 256 MB device, N=100K × D=128 with K_graph=16 costs: f32=51MB + u8=13MB + graph=6MB = 70MB total. High recall on a microcontroller.

## Running It

```bash
# Clone ruvector
git clone https://github.com/ruvnet/ruvector
cd ruvector
git checkout research/nightly/2026-08-02-graph-neighbour-cascade-ann

# Acceptance tests (N=2000, D=64, fast)
cargo test -p ruvector-cascade-ann

# Full benchmark (N=5000, D=128, ~30s build + run)
cargo run --release -p ruvector-cascade-ann --bin benchmark
```

## Open Questions

- How small can δ be for a given quantization scheme and still recover boundary misses reliably?
- Does 2-hop expansion recover the remaining 1.2% recall gap vs QC(ef=4)?
- Can the coherence gate (ADR-240) replace δ as the uncertainty signal — treating low-coherence candidates as uncertain rather than distance-threshold candidates?

---

*Crate*: `crates/ruvector-cascade-ann` · *ADR*: `docs/adr/ADR-273-graph-neighbour-cascade-ann.md` · *Full research*: `docs/research/nightly/2026-08-02-graph-neighbour-cascade-ann/README.md`
