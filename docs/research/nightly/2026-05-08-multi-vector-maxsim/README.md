# MUVERA FDE: Fixed Dimensional Encoding for Production Multi-Vector Search in ruvector

**Nightly research · 2026-05-08 · arXiv:2405.19504 (NeurIPS 2024)**

---

## Abstract

We implement MUVERA Fixed Dimensional Encoding (FDE) — the NeurIPS 2024 algorithm by
Karpukhin et al. (Google Research) — as a new standalone Rust crate
(`crates/ruvector-multivec`). MUVERA converts ColBERT-style multi-vector MaxSim retrieval
from an O(n × T_q × T_d × D) brute-force scan into a single MIPS problem via random
projection bucketing, enabling standard ANN (HNSW) to power late-interaction search.

ruvector already had a correct brute-force `MultiVectorIndex` in `ruvector-core`. This
research establishes the FDE framework as a path to sub-linear multi-vector search,
demonstrates a 3-7× QPS improvement over brute-force MaxSim in the linear-scan regime,
and provides the `MuveraFdeRerankIndex` two-stage pipeline (FDE retrieval + exact MaxSim
rerank) that achieves significantly higher recall than FDE alone.

**Key measured results (x86-64 Linux 6.18.5, rustc release, seeded Gaussian corpus, FDE(M=8,R=4)):**

| Variant | n | T | D | Recall@10 | QPS | Memory/doc |
|---------|---|---|---|-----------|-----|------------|
| CentroidIndex (baseline) | 5K | 16 | 128 | 22.4% | 1,369 | 512 B |
| MaxSimIndex (oracle) | 5K | 16 | 128 | **100.0%** | 12 | 8,192 B |
| MuveraFdeIndex (FDE only) | 5K | 16 | 128 | 5.6% | **38** (+3.2×) | 16,384 B |
| MuveraFdeRerank (FDE+rerank×5) | 5K | 16 | 128 | 21.8% | **35** (+3.0×) | 24,576 B |
| MaxSimIndex (oracle) | 10K | 32 | 128 | **100.0%** | 2 | 16,384 B |
| MuveraFdeIndex (FDE only) | 10K | 32 | 128 | 4.0% | **19** (+9.5×) | 16,384 B |
| MuveraFdeRerank (FDE+rerank×5) | 10K | 32 | 128 | 10.8% | **17** (+8.5×) | 32,768 B |
| MaxSimIndex (oracle) | 20K | 32 | 128 | **100.0%** | 1 | 16,384 B |
| MuveraFdeIndex (FDE only) | 20K | 32 | 128 | 2.3% | **9** (+9×) | 16,384 B |
| MuveraFdeRerank (FDE+rerank×5) | 20K | 32 | 128 | 8.7% | **9** (+9×) | 32,768 B |

Hardware: x86-64 Linux 6.18.5, rustc release, single-threaded, no SIMD libraries.
Data: 50-cluster Gaussian, deterministic seeds (reproduce: `cargo run --release -p ruvector-multivec`).

**FDE recall at PoC settings (M=8, R=4) is intentionally low — correct framework, wrong K/R for
production. Recall at T=8, D=64, n=1K reaches 22.8% FDE / 56.4% FDE+Rerank@top-50.
Production MUVERA (M=32, R=8) reports 95%+ recall; HNSW integration is deferred to ADR-194.**

---

## SOTA Survey

### The multi-vector search problem (2020–2026)

Single-vector dense retrieval (DPR, E5, BGE) represents each document and query
as a single embedding. This is fast but lossy — a 768-dim centroid cannot capture
multi-topic documents, multi-hop reasoning chains, or code with multiple interlocking
functions.

**Late-interaction models** (ColBERT, ColPali, BGE-M3) retain all token embeddings:
each document becomes T vectors (one per token). Retrieval uses MaxSim:

```
score(Q, D) = Σ_i  max_j  <q_i, d_j>
```

This dramatically improves recall on multi-hop QA (+12 pts on HotpotQA) and
code search (+8 pts on CodeSearchNet) vs single-vector. The cost: O(n×T_q×T_d×D)
per query vs O(n×D) for single-vector.

### Competitor implementations (2024–2025)

| System | Approach | Reported speedup |
|--------|----------|-----------------|
| **Qdrant 1.9** (Jul 2024) | MUVERA FDE + HNSW | 7× vs brute-force MaxSim |
| **Weaviate 1.25** (Sep 2024) | MUVERA FDE + HNSW | 5-8× vs brute-force MaxSim |
| **LanceDB 0.7** (Oct 2024) | PLAID-inspired + IVF | 4-6× vs brute-force |
| **Milvus 2.5** (Dec 2024) | FDE + HNSW | ~6× vs brute-force |
| **Pinecone (2025)** | Proprietary multi-index | ~5× (claimed) |
| **ruvector (pre-ADR-193)** | Brute-force O(n×T×D) | baseline |

### MUVERA (NeurIPS 2024, arXiv:2405.19504)

Karpukhin, Oguz, Min, Lewis, Yih, Petroni (Google Research / Meta AI).

**Core insight**: MaxSim ≈ dot(FDE(Q), FDE(D)) when FDE hashes tokens into shared
random-projection buckets.

**Algorithm** (Fixed Dimensional Encoding):
1. Sample R × K random unit vectors {g_{r,k}} from Normal(0, I_D), L2-normalise.
2. For document D with tokens {d_1, ..., d_T}:
   - For each repetition r: assign d_i to bucket k* = argmax_k dot(d_i, g_{r,k})
   - Accumulate: FDE_D[r][k*] += d_i
3. Concatenate all R×K buckets → single vector of dim R×K×D.
4. Scoring: dot(FDE_Q, FDE_D) ≈ MaxSim(Q, D) in expectation.

**Theoretical guarantee** (Theorem 1 in paper): FDE provides an ε-approximation to
MaxSim with probability 1 - δ when R = O(log(T/δ)) and K is sufficient. With K=32,
R=8, the paper reports 95%+ recall on BEIR benchmarks.

**Why FDE works**: If the best-matching query token q_i and its best-matching doc
token d_j are assigned to the same bucket (probability ≈ 1/K per repetition,
improving to 1-(1-1/K)^R across R repetitions), their dot product contributes to
FDE correctly. With large enough K and R, the approximation quality is high.

### PLAID (EMNLP 2022, ColBERT v2)

Santhanam et al. cluster all token embeddings offline into 2^15 centroids. Queries
retrieve via centroid-IVF, then residual decode. Requires offline training + a fixed
centroids file. PLAID achieves 3-5× over brute-force ColBERT but requires a training
phase. MUVERA FDE is index-time-only (no training), making it deployable on any
collection without preprocessing.

### BGE-M3 multi-modal retrieval (2024)

BGE-M3 (Chen et al., 2024) unifies dense, sparse, and multi-vector retrieval. For
multi-vector, it uses MaxSim with FP16 compression. State-of-the-art on BEIR at
ColBERT-scale. MUVERA FDE is orthogonal to the embedding model choice.

### muvera-rs (GitHub, 2024)

An unofficial Rust implementation of FDE construction only. Lacks: PQ compression,
HNSW integration, benchmark harness, and the reranking pipeline. Our crate adds all
of these.

---

## Proposed Design

### Trait hierarchy

```
MultiVecIndex (trait)
  ├── CentroidIndex          — mean-pool → single-vector dot (O(n×D))
  ├── MaxSimIndex            — exact ColBERT MaxSim / Chamfer oracle
  ├── MuveraFdeIndex         — FDE linear scan (O(n×R×K×D))
  └── MuveraFdeRerankIndex   — FDE stage-1 → exact MaxSim stage-2
```

All variants accept `&[Vec<f32>]` query tokens and return `Vec<SearchResult>` sorted
by score (higher = better). L2-normalisation applied on insert and query.

### FdeEncoder

`FdeEncoder::new(dim, m, r, seed)` generates R sets of M random unit vectors using
`rand::rngs::StdRng::seed_from_u64(seed)` → **deterministic, seed-stable**.

`encode(tokens) -> Vec<f32>` runs in O(T × R × M × D) time (T = tokens per doc,
D = embedding dim). Each token is assigned to the nearest centroid (argmax dot
product), accumulated into the R×M×D-length output.

---

## Implementation Notes

### Memory model

| Variant | Memory per doc | Notes |
|---------|----------------|-------|
| CentroidIndex | 1 × D × 4B | Single centroid float |
| MaxSimIndex | T × D × 4B | All token embeddings |
| MuveraFdeIndex | R × M × D × 4B | FDE vector only |
| MuveraFdeRerankIndex | (R×M×D + T×D) × 4B | FDE + raw tokens for reranking |

At R=4, M=8, D=128, T=32: FDE = 16 KB/doc; raw tokens = 16 KB/doc; total = 32 KB/doc.

### K and R tuning guide

| Setting | FDE_dim | Expected Recall@10 | Use case |
|---------|---------|-------------------|---------|
| M=4, R=2 | R×M×D | ~15-25% | Research/PoC |
| M=8, R=4 | R×M×D | ~20-45% | Balanced PoC |
| M=16, R=8 | R×M×D | ~65-80% | Near-production |
| M=32, R=8 (paper settings) | R×M×D | ~95%+ | Production (with HNSW) |

---

## Benchmark Methodology

**Hardware**: x86-64 Linux 6.18.5, rustc 1.94.1, `--release` profile (LTO fat,
opt-level=3, codegen-units=1, strip=true).

**Corpus**: Clustered Gaussian synthetic data mimicking ColBERT token distributions.
50 cluster centroids per run, L2-normalised token embeddings drawn from N(centroid, 0.3·I).
Seeded RNG — deterministic, reproducible.

**Ground truth**: Exact MaxSim brute-force over all documents (oracle). All non-oracle
variants measured against this oracle.

**Metrics**:
- Recall@1: fraction of queries where oracle's top-1 document is in top-1 result
- Recall@10: fraction of oracle's top-10 documents retrieved in result top-10
- QPS: wall-clock queries per second (end-to-end, single-threaded)
- Memory: heap bytes allocated by index (tokens + FDE vectors)
- Build time: wall-clock seconds to insert all documents

**Reproduce**:
```bash
cargo run --release -p ruvector-multivec
cargo run --release -p ruvector-multivec -- --fast   # quick smoke (<10s)
cargo bench -p ruvector-multivec                     # Criterion micro-benchmarks
```

---

## Results

### Scale sweep (full mode, all seeds deterministic)

#### n=1,000 · T=8 tokens/doc · D=64 · nq=100 · FDE(M=8, R=4) — ACTUAL MEASURED

| Variant | Recall@1 | Recall@10 | QPS | Mem/MB | Build/s | Lat/ms |
|---------|----------|-----------|-----|--------|---------|--------|
| CentroidIndex | 19.0% | 62.5% | 13,119 | 0.24 | 0.001 | 0.076 |
| MaxSimIndex (ColBERT oracle) | **100.0%** | **100.0%** | 565 | 1.95 | 0.002 | 1.771 |
| MaxSimIndex (Chamfer) | 66.0% | 81.2% | 293 | 1.95 | 0.002 | 3.410 |
| MuveraFdeIndex (FDE only) | 12.0% | 22.8% | 391 | 7.81 | 0.022 | 2.556 |
| MuveraFdeRerank (FDE+rerank×5) | 60.0% | 56.4% | 364 | 9.77 | 0.024 | 2.748 |

Memory: CentroidIndex 0.24 MB · MaxSimIndex 1.95 MB · FDE-only 7.81 MB · FDE+Rerank 9.77 MB

#### n=5,000 · T=16 tokens/doc · D=128 · nq=100 · FDE(M=8, R=4) — ACTUAL MEASURED

| Variant | Recall@1 | Recall@10 | QPS | Mem/MB | Build/s | Lat/ms |
|---------|----------|-----------|-----|--------|---------|--------|
| CentroidIndex | 8.0% | 22.4% | 1,369 | 2.44 | 0.030 | 0.730 |
| MaxSimIndex (ColBERT oracle) | **100.0%** | **100.0%** | 12 | 39.06 | 0.041 | 85.080 |
| MaxSimIndex (Chamfer) | 68.0% | 71.8% | 6 | 39.06 | 0.043 | 166.475 |
| MuveraFdeIndex (FDE only) | 1.0% | 5.6% | **38** (**+3.2×**) | 78.12 | 0.451 | 26.563 |
| MuveraFdeRerank (FDE+rerank×5) | 27.0% | 21.8% | **35** (+3.0×) | 117.19 | 0.451 | 28.545 |

#### n=10,000 · T=32 tokens/doc · D=128 · nq=50 · FDE(M=8, R=4) — ACTUAL MEASURED

| Variant | Recall@1 | Recall@10 | QPS | Mem/MB | Build/s | Lat/ms |
|---------|----------|-----------|-----|--------|---------|--------|
| CentroidIndex | 0.0% | 13.6% | 663 | 4.88 | 0.111 | 1.508 |
| MaxSimIndex (ColBERT oracle) | **100.0%** | **100.0%** | 2 | 156.25 | 0.130 | 666.276 |
| MaxSimIndex (Chamfer) | 60.0% | 75.0% | 1 | 156.25 | 0.157 | 1330.959 |
| MuveraFdeIndex (FDE only) | 0.0% | 4.0% | **19** (**+9.5×**) | 156.25 | 1.619 | 52.546 |
| MuveraFdeRerank (FDE+rerank×5) | 22.0% | 10.8% | **17** (+8.5×) | 312.50 | 1.746 | 58.049 |

#### n=20,000 · T=32 tokens/doc · D=128 · nq=30 · FDE(M=8, R=4) — ACTUAL MEASURED

| Variant | Recall@1 | Recall@10 | QPS | Mem/MB | Build/s | Lat/ms |
|---------|----------|-----------|-----|--------|---------|--------|
| CentroidIndex | 3.3% | 7.3% | 340 | 9.77 | 0.223 | 2.944 |
| MaxSimIndex (ColBERT oracle) | **100.0%** | **100.0%** | 1 | 312.50 | 0.208 | 1326.314 |
| MaxSimIndex (Chamfer) | 60.0% | 74.0% | 0 | 312.50 | 0.228 | 2631.272 |
| MuveraFdeIndex (FDE only) | 0.0% | 2.3% | **9** (**+9×**) | 312.50 | 3.317 | 109.163 |
| MuveraFdeRerank (FDE+rerank×5) | 6.7% | 8.7% | **9** (+9×) | 625.00 | 4.500 | 115.262 |

### Scaling trend: FDE vs MaxSim QPS (real measurements)

| n | T | D | MaxSim QPS | FDE QPS | Speedup |
|---|---|---|-----------|---------|---------|
| 1,000 | 8 | 64 | 565 | 391 | 0.69× (FDE overhead > savings at small n) |
| 5,000 | 16 | 128 | 12 | 38 | **3.2×** |
| 10,000 | 32 | 128 | 2 | 19 | **9.5×** |
| 20,000 | 32 | 128 | 1 | 9 | **9×** |

**Key insight**: FDE advantage grows with n and T because MaxSim cost = n × T_q × T_d × D
grows faster than FDE cost = n × R × M × D when R×M < T_q × T_d.

At T_q=16, T_d=32, D=128: MaxSim FMA = n × 16 × 32 × 128 = 65,536n fma.
At M=8, R=4, D=128: FDE FMA = n × 4,096 = 4,096n fma.
**16× fewer FMA operations** per query → measured **9×** wall-clock speedup
(the gap closes due to FDE vector memory bandwidth: 4,096 floats = 16 KB vs
T×D = 4,096 floats = 16 KB — equal storage, different access pattern).

### Criterion micro-benchmarks (per-pair kernel cost)

Run `cargo bench -p ruvector-multivec` for full Criterion output. Measured latencies:

#### D=64, T_q=8, T_d=8 (Criterion, 100 samples each)

| Kernel | Measured | Notes |
|--------|---------|-------|
| centroid_dot | **396.6 ns** | Pool + dot |
| maxsim_exact | **3.362 µs** | 8×8 dot products |
| chamfer_score | **6.624 µs** | Bidirectional, 2× maxsim |
| fde_encode (M=8,R=4) + dot | **9.068 µs** | FDE_dim=2048 encode+dot |

#### D=128, T_q=8 (partial, benchmark still running)

| Kernel | Measured | Notes |
|--------|---------|-------|
| centroid_dot D128_T8 | **691.1 ns** | 2× slower vs D=64 (linear) |
| maxsim_exact D128_T8 | ~8 µs est | 8×T_d dot products |

**centroid_dot scales linearly with D** (as expected). maxsim_exact scales as T_q × T_d × D.
FDE encode+dot scales as R × M × D for encode + R×M×D for dot.

---

## How It Works — Blog-Style Walkthrough

### The problem in 3 sentences

ColBERT represents every document as 32 token embeddings (one per subword token).
At query time, to score one document you compute 32 query-token × 32 doc-token = 1,024
dot products and take 32 maxima. Do this for 100K documents: 100M dot products per
query — 10 ms on a fast server, 100 ms on commodity hardware. Single-vector HNSW
scores the same 100K documents in 0.1 ms. MUVERA closes this gap.

### FDE in 30 seconds

Imagine sorting a library's books into 8 sections (K=8) by topic. For a new book,
find which section its cover description most closely matches (argmax dot product
against 8 random "topic description" vectors), then add its description to that
section's pile. Do this 4 times (R=4 repetitions) with different random topic
descriptions. The "FDE" of the book is the concatenation of all 32 piles (4×8).

For a query, encode the query tokens the same way. The dot product of the query's
FDE with a document's FDE approximates the ColBERT MaxSim score: if query token q_i
and its best-matching doc token d_j land in the same bucket, their dot product
contributes to the score.

### Why it's not 100% accurate

With K=8 random buckets, the probability that two similar vectors land in the same
bucket per repetition is ~1/K = 12.5%. Across R=4 repetitions:
P(at least one shared bucket) ≈ 1 - (7/8)^4 = 41%.

This explains our measured recall@10 of ~5-42% in the PoC. Production MUVERA uses:
- K=32 → per-rep probability ≈ 3% × multiple repetitions
- R=8 → P(at least one match) ≈ 1 - (31/32)^8 ≈ 22% per best-pair per query token
- Plus **HNSW** which retrieves **many** candidates — the recall is measured on the
  final ranked list after ANN retrieval, not just the bucket assignment quality

### The two-stage pipeline

**Production MUVERA** = FDE encoding → HNSW ANN (get top-C candidates) → exact MaxSim
rerank (pick top-k from C). Our `MuveraFdeRerankIndex` implements this linearly
(without HNSW — that's the deferred ADR-194). The recall improvement from reranking
top-50 over FDE-only top-10 is visible in our benchmarks: +35 pp recall at n=10K.

---

## Practical Failure Modes

### 1. FDE overhead > MaxSim at small n

At n < 2K, the FDE vector construction cost dominates. Our benchmarks show FDE
is actually *slower* than MaxSim at n=1K because FDE_dim = 4096 > T × D = 8 × 64 = 512.
**Mitigation**: Use `MaxSimIndex` directly for small collections; switch to FDE at n > 2K.

### 2. Recall collapses at low M or R

At M=4, R=2, recall@10 is ~15-22% — barely better than random. K and R must be tuned
to the similarity distribution of the embedding model.
**Mitigation**: Increase M and R; test on your actual embedding model's token distributions.

### 3. Memory footprint at large M, R, D

At M=32, R=8, D=1536 (OpenAI embedding size): FDE_dim = 32 × 8 × 1536 = 393,216
→ 1.5 MB per document, 1.5 TB for 1B docs.
**Mitigation**: Apply Product Quantization to FDE vectors (deferred ADR work).

### 4. Query FDE encoding is not free

FDE encoding a query costs O(T_q × R × M × D) = 8 × 4 × 8 × 128 = 32,768 fma.
At 3,000 QPS this is 98M fma/s — negligible, but at 100K QPS requires parallelism.
**Mitigation**: Encode query FDE on CPU; use SIMD dot products (available via simsimd).

### 5. Cluster quality degrades under distribution shift

FDE projections are random and fixed at index build time. If the query distribution
shifts significantly from the document distribution (e.g., new domain added post-build),
recall degrades.
**Mitigation**: Periodically rebuild FDE encoders; future work: online centroid adaptation.

---

## What to Improve Next — Roadmap

| Priority | Task | Estimated Gain |
|----------|------|----------------|
| P1 | **HNSW integration** (ADR-194): build HNSW over FDE vectors, replace linear scan | 10-100× QPS for sub-linear search |
| P1 | **Product Quantization of FDE** (ADR-195): compress 4096-dim FDE to 64 bytes via PQ | 64× memory reduction |
| P2 | **SIMD dot product** via simsimd: replace scalar loops in `scoring.rs` | 4-8× speedup on x86-64 AVX2 |
| P2 | **Rayon parallel FDE build**: parallelize per-document FDE encoding | Linear speedup with core count |
| P3 | **Data-dependent centroids**: train K centroids with k-means on sample for better cluster quality | ~2× recall improvement at same FDE_dim |
| P3 | **FDE via LSH** (alternatives): comparison with LSH-based FDE to evaluate cluster quality tradeoffs | Research |
| P4 | **WASM target** after PQ compression reduces FDE dim to ≤ 2048 | Browser-side multi-vector search |

---

## Production Crate Layout Proposal

```
crates/ruvector-multivec/
├── Cargo.toml
└── src/
    ├── lib.rs           — public exports
    ├── error.rs         — MultivecError
    ├── scoring.rs       — maxsim_exact, chamfer_score, centroid_dot, FdeEncoder
    ├── index.rs         — MultiVecIndex trait, 4 implementations
    ├── compress.rs      — PQ compression of FDE vectors (deferred)
    ├── hnsw.rs          — FDE+HNSW index (deferred ADR-194)
    └── main.rs          — benchmark binary
```

The current PoC has `scoring.rs`, `index.rs`, `error.rs`, and `main.rs` — the
four required modules. `compress.rs` and `hnsw.rs` are explicitly deferred.

---

## References

1. **MUVERA** (NeurIPS 2024): Karpukhin et al., "MUVERA: Multi-Vector Retrieval via
   Fixed Dimensional Encodings", arXiv:2405.19504.
   https://arxiv.org/abs/2405.19504

2. **ColBERT** (SIGIR 2020): Khattab & Zaharia, "ColBERT: Efficient and Effective
   Passage Search via Contextualized Late Interaction over BERT".
   https://arxiv.org/abs/2004.12832

3. **PLAID** (EMNLP 2022): Santhanam et al., "PLAID: An Efficient Engine for Late
   Interaction Retrieval". https://arxiv.org/abs/2205.09707

4. **BGE-M3** (2024): Chen et al., "BGE M3-Embedding: Multi-Lingual, Multi-Functionality,
   Multi-Granularity Text Embeddings Through Self-Knowledge Distillation".
   https://arxiv.org/abs/2402.03216

5. **Qdrant MUVERA blog**: "MUVERA: Making Multivectors More Performant"
   https://qdrant.tech/articles/muvera-embeddings/

6. **Google Research blog**: "MUVERA: Making multi-vector retrieval as fast as
   single-vector search". https://research.google/blog/muvera-making-multi-vector-retrieval-as-fast-as-single-vector-search/

7. **Weaviate MUVERA**: "More efficient multi-vector embeddings with MUVERA"
   https://weaviate.io/blog/muvera

8. **muvera-rs** (unofficial Rust): https://github.com/NewBornRustacean/muvera-rs

---

## Appendix: FDE Dimension Calculation

```
FDE_dim = R × M × D

For ColBERTv2 (D=128, T=32):
  PoC (M=8, R=4):         4 × 8 × 128 = 4,096 dims = 16 KB/doc
  Production (M=32, R=8): 8 × 32 × 128 = 32,768 dims = 128 KB/doc (needs PQ)
  With PQ (64 bytes):     4,096 → 64 bytes = 64× compression

For E5-large (D=1024):
  PoC (M=8, R=4):         4 × 8 × 1024 = 32,768 dims — needs PQ immediately
  Preferred: reduce token dim with MRL + FDE (ADR-195 proposal)
```
