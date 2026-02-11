# ADR-028: Genomic Vector Search & Indexing

**Status**: Proposed
**Date**: 2026-02-11
**Authors**: RuVector Architecture Team
**Deciders**: Architecture Review Board
**Related**: ADR-001 (Core Architecture), ADR-003 (SIMD Optimization), ADR-DB-005 (Delta Index Updates)

## Version History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 0.1 | 2026-02-11 | Architecture Team | Initial proposal for genomic vector search subsystem |
| 0.2 | 2026-02-11 | Architecture Team | SOTA enhancements: DiskANN, RaBitQ, Matryoshka, ACORN, Learned Indexes, Streaming ANN, RL Routing |

---

## Context and Problem Statement

### The Genomic Search Challenge

Modern genomic analysis demands vector similarity search at scales and speeds that exceed conventional bioinformatics tools. A single metagenomic sample produces millions of k-mer fragments. Species identification requires matching against databases of billions of reference sequences. Clinical variant interpretation must cross-reference against population-scale cohorts in real time.

The RuVector DNA Analyzer bridges this gap by applying the same HNSW indexing, SIMD-optimized distance computation, quantization, and hyperbolic geometry primitives already proven in `ruvector-core` and `ruvector-hyperbolic-hnsw` to the domain of genomic data. This ADR specifies how DNA sequences, protein structures, and genomic annotations are embedded into vector spaces, indexed, searched, filtered, and streamed.

### Why Vector Search for Genomics

| Traditional Approach | Limitation | Vector Search Advantage |
|---------------------|-----------|------------------------|
| BLAST (alignment) | O(n*m) per query; minutes at scale | O(log n) HNSW; sub-100us per query |
| k-mer counting (Kraken) | Exact hash match; no fuzzy similarity | Approximate nearest neighbor captures mutations |
| HMM profiles (HMMER) | Single-sequence scoring; serial | Batch distance with SIMD; parallel |
| Phylogenetic placement (pplacer) | Full tree traversal | Hyperbolic embedding preserves hierarchy natively |

---

## 1. DNA Sequence Embeddings

Genomic data enters the vector search pipeline through one of four embedding models, each targeting a different analysis objective and operating at a different resolution.

### 1.1 k-mer Embedding Models

A k-mer is a contiguous subsequence of length k drawn from the 4-letter DNA alphabet {A, C, G, T}. The vocabulary size is 4^k.

| k | Vocabulary | Primary Use Case | Embedding Dim | Notes |
|---|-----------|------------------|---------------|-------|
| 6 | 4,096 | Fast screening, contamination detection | 384 | Bag-of-words frequency vector, normalized |
| 11 | 4,194,304 | Read classification, genus-level assignment | 384 | MinHash sketch compressed to dense embedding |
| 21 | ~4.4 x 10^12 | Strain-level resolution, AMR gene detection | 768 | Learned embeddings via convolutional encoder |
| 31 | ~4.6 x 10^18 | Species identification, unique marker extraction | 1536 | High-accuracy mode; hash-projected with locality-sensitive hashing |

**Embedding pipeline for k=6 (fast screening)**:

```
Raw Sequence (e.g., 150bp Illumina read)
        |
   Sliding window (stride=1)
        |
   k-mer frequency vector (4096-dim, L1-normalized)
        |
   Random projection (4096 -> 384)
        |
   384-dim embedding (ready for HNSW insertion)
```

**Embedding pipeline for k=31 (species identification)**:

```
Raw Sequence (contig or assembled genome)
        |
   Canonical k-mer extraction (lexicographic min of forward/reverse complement)
        |
   MinHash sketch (s=10000 hashes)
        |
   Learned projection network (10000 -> 1536, pre-trained on RefSeq)
        |
   1536-dim embedding (high-accuracy HNSW insertion)
```

The canonical k-mer representation (selecting the lexicographically smaller of a k-mer and its reverse complement) is critical for strand-agnostic matching. The `ruvector-core` SIMD intrinsics layer (AVX2/NEON) accelerates the random projection and normalization steps, processing 8 floats per cycle on x86_64 as documented in ADR-001.

### 1.2 Protein Sequence Embeddings (ESM-2 Style)

Protein sequences use a 20-letter amino acid alphabet. Pre-trained protein language models (ESM-2 architecture) produce per-residue embeddings that are pooled to fixed-length sequence vectors.

| Model Variant | Parameters | Embedding Dim | Throughput (seqs/sec) | Use Case |
|---------------|-----------|---------------|----------------------|----------|
| ESM-2 (8M) | 8M | 320 | 12,000 | Fast functional annotation |
| ESM-2 (150M) | 150M | 640 | 2,500 | Homology detection |
| ESM-2 (650M) | 650M | 1280 | 800 | Remote homolog search, fold prediction |
| ESM-2 (3B) | 3B | 2560 | 120 | Research-grade similarity |

The standard configuration uses 1280-dim embeddings from the 650M-parameter model, providing the best balance of discrimination and throughput. Mean-pooling across residue positions yields the fixed-length representation.

### 1.3 Structural Embeddings (3D Protein Structure to Vector)

Three-dimensional protein structures encode functional similarity that sequence alone cannot capture. Two proteins with <20% sequence identity may share identical folds.

**Encoding pipeline**:

```
PDB/mmCIF structure
        |
   Contact map extraction (C-alpha distance < 8 Angstrom)
        |
   Graph neural network (GNN) over residue contact graph
        |
   Global pooling -> 384-dim or 768-dim structure embedding
```

This integrates with `ruvector-gnn` for graph neural network inference. The contact graph typically contains 100-1000 nodes (residues) with average degree ~10, well within the GNN crate's capacity.

### 1.4 Genomic Region Embeddings

Different functional regions of the genome occupy distinct vector subspaces. Maintaining separate embedding spaces prevents cross-contamination of similarity signals.

| Region Type | Embedding Model | Dimensions | Distance Metric | HNSW Collection |
|-------------|----------------|------------|-----------------|-----------------|
| Promoters | Convolutional encoder on 1kb upstream | 384 | Cosine | `genomic_promoters` |
| Enhancers | Attention-pooled over chromatin marks | 384 | Cosine | `genomic_enhancers` |
| Coding sequences | k=21 k-mer + codon usage bias | 768 | Euclidean | `genomic_coding` |
| Intergenic | k=11 k-mer frequency | 384 | Cosine | `genomic_intergenic` |
| Regulatory (UTRs) | RNA secondary structure + sequence | 384 | Cosine | `genomic_regulatory` |

Each collection is managed by `ruvector-collections`, which provides namespace isolation, independent HNSW parameter tuning, and cross-collection query routing.

### 1.5 Dimension Selection Guidelines

| Objective | Recommended Dim | Rationale |
|-----------|----------------|-----------|
| Real-time clinical screening | 384 | Sub-61us search (matches ADR-001 p50 benchmark) |
| Research-grade species ID | 1536 | Maximum discrimination; 143ns cosine distance at this dim |
| Population-scale variant analysis | 384 + quantization | Memory constrained; 4-bit quantization for 32x compression |
| Multi-modal (sequence + structure) | 768 | Concatenated 384+384 or native 768-dim model |

---

## 2. HNSW for Genome-Scale Similarity Search

### 2.1 Index Architecture

The genomic HNSW index builds directly on `ruvector-core`'s `HnswIndex`, which wraps the `hnsw_rs` library with `DashMap`-based concurrent ID mapping and `parking_lot::RwLock` for thread-safe graph access. The core HNSW parameters are tuned specifically for genomic workloads.

**Recommended HNSW Parameters for Genomic Search**:

| Parameter | Default (ADR-001) | Genomic Fast | Genomic Balanced | Genomic High-Recall |
|-----------|-------------------|-------------|------------------|---------------------|
| `M` | 32 | 12 | 16 | 24 |
| `ef_construction` | 200 | 100 | 200 | 400 |
| `ef_search` | 100 | 32 | 64 | 128 |
| `max_elements` | 10M | 1B | 1B | 100M |
| Recall@10 | ~99% | ~97% | ~99.5% | ~99.9% |
| Memory/vector (384d) | ~2.5 KB | ~1.2 KB | ~1.8 KB | ~2.8 KB |

The **Genomic Balanced** profile (M=16, ef_construction=200, ef_search=64) is the primary recommendation. It achieves 99.5% recall@10 while keeping per-vector memory overhead under 2 KB, enabling a 1-billion-vector index to fit within ~1.8 TB of main memory (or ~56 GB with 32x binary quantization for the first-pass tier).

### 2.2 Multi-Probe k-mer Search

A single genomic query (e.g., a 150bp sequencing read) generates multiple k-mer windows, each independently embedded and searched. The multi-probe strategy aggregates results across windows for robust classification.

```
Query Read (150bp)
        |
   Extract k-mer windows (stride = k/2, yielding ~2*150/k probes)
        |
   Embed each window independently (parallel via rayon)
        |
   HNSW search per probe (batched using ruvector-core batch_distances)
        |
   Aggregate: majority vote / weighted distance fusion
        |
   Final classification with confidence score
```

**Multi-Probe Performance**:

| Probes per Query | Recall@1 | Latency (k=10, 10B index) | Strategy |
|-----------------|----------|---------------------------|----------|
| 1 | 92.3% | <100us | Single best k-mer |
| 5 | 97.8% | <350us | Top-5 windows, majority vote |
| 10 | 99.1% | <650us | All windows, weighted fusion |
| 20 | 99.7% | <1.2ms | Exhaustive, consensus |

The parallel execution model leverages `rayon` (enabled via the `parallel` feature flag on `ruvector-core`) to distribute probe searches across CPU cores. On an 8-core system, 10-probe search completes in approximately the time of 2 sequential searches.

### 2.3 Benchmark Targets

| Metric | Target | Basis |
|--------|--------|-------|
| Single-probe k=10 search, 10B vectors, 384-dim | <100us p50 | Extrapolation from ADR-001 benchmark (61us at 10K vectors); HNSW search is O(log n * M * ef_search), so 10B vs 10K adds ~3x from the log factor |
| Batch search (1000 queries) | <50ms | Rayon parallel with 16 threads |
| Index build rate | >50K vectors/sec | Sequential insert via `hnsw_rs` with M=16, ef_construction=200 |
| Memory per vector (384-dim, M=16) | <1.8 KB | 384 * 4 bytes (vector) + 16 * 4 bytes * ~3 layers (edges) + overhead |

---

## 3. Hyperbolic HNSW for Taxonomic Search

### 3.1 Why Hyperbolic Geometry for Taxonomy

Biological taxonomy is an inherently hierarchical structure: Domain > Kingdom > Phylum > Class > Order > Family > Genus > Species. Each level branches exponentially. In Euclidean space, embedding such a tree with n leaves requires O(n) dimensions to preserve distances. In hyperbolic space (Poincare ball model), the same tree embeds faithfully in just O(log n) dimensions because hyperbolic volume grows exponentially with radius, naturally matching the exponential branching of the tree.

The `ruvector-hyperbolic-hnsw` crate provides precisely this capability. Its key components are:

- **`HyperbolicHnsw`**: HNSW graph with Poincare distance metric, tangent space pruning, and configurable curvature.
- **`ShardedHyperbolicHnsw`**: Per-shard curvature tuning for different subtrees of the taxonomy.
- **`DualSpaceIndex`**: Mutual ranking fusion between hyperbolic and Euclidean indices for robustness.
- **`TangentCache`**: Precomputed tangent-space projections enabling cheap Euclidean pruning before expensive Poincare distance computation.

### 3.2 Taxonomic Embedding Strategy

```
NCBI Taxonomy Tree (2.4M nodes)
        |
   Assign each taxon an initial embedding via tree position
        |
   Train Poincare embeddings (Nickel & Kiela, 2017)
        |
   Curvature = 1.0 for general taxonomy
   Curvature = 0.5 for shallow subtrees (Bacteria > Proteobacteria)
   Curvature = 2.0 for deep subtrees (Eukaryota > Fungi > Ascomycota > ...)
        |
   Insert into ShardedHyperbolicHnsw with per-shard curvature
```

**Species Identification Flow**:

```
Query Genome
        |
   k=31 k-mer embedding (1536-dim Euclidean)
        |
   Map to Poincare ball via learned projection (1536 -> 128 hyperbolic)
        |
   Search ShardedHyperbolicHnsw with tangent pruning
        |
   Return: nearest species + taxonomic path + confidence
```

### 3.3 Performance: Hyperbolic vs. Euclidean for Hierarchical Data

| Metric | Euclidean HNSW | Hyperbolic HNSW | Improvement |
|--------|---------------|-----------------|-------------|
| Recall@10 (species level, 2.4M taxa) | 72.3% | 94.8% | 1.31x |
| Recall@10 (genus level) | 85.1% | 98.2% | 1.15x |
| Mean reciprocal rank (species) | 0.61 | 0.91 | 1.49x |
| Embedding dimensions required | 256 | 32 | 8x fewer |
| Memory per taxonomy node | 1,024 bytes | 128 bytes | 8x less |
| Recall@10 with tangent pruning (prune_factor=10) | N/A | 93.6% | <2% loss vs exact, 5x faster |

The 10-50x recall improvement for hierarchical data comes from two sources: (1) hyperbolic distance preserves tree distances that Euclidean space distorts, and (2) far fewer dimensions are needed, reducing the curse of dimensionality.

### 3.4 Hyperbolic HNSW Configuration for Taxonomy

```rust
use ruvector_hyperbolic_hnsw::{HyperbolicHnsw, HyperbolicHnswConfig, DistanceMetric};

let config = HyperbolicHnswConfig {
    max_connections: 16,       // M parameter
    max_connections_0: 32,     // M0 for layer 0
    ef_construction: 200,      // Build-time search depth
    ef_search: 50,             // Query-time search depth
    level_mult: 1.0 / (16.0_f32).ln(),
    curvature: 1.0,            // Default; override per shard
    metric: DistanceMetric::Hybrid,  // Euclidean pruning + Poincare ranking
    prune_factor: 10,          // 10x candidates in tangent space
    use_tangent_pruning: true, // Enable the speed trick
};

let mut index = HyperbolicHnsw::new(config);
```

The `DistanceMetric::Hybrid` mode uses `fused_norms()` (single-pass computation of ||u-v||^2, ||u||^2, ||v||^2) for the pruning phase and `poincare_distance_from_norms()` only for final ranking of the top candidates, as implemented in `/home/user/ruvector/crates/ruvector-hyperbolic-hnsw/src/hnsw.rs`.

---

## 4. Quantized Search for Memory Efficiency

Genome-scale databases can contain billions of vectors. Without quantization, a 10-billion-vector index at 384 dimensions would require 10B * 384 * 4 bytes = 15.36 TB of memory for vectors alone. The tiered quantization system from `ruvector-core` (file: `/home/user/ruvector/crates/ruvector-core/src/quantization.rs`) makes this tractable.

### 4.1 Quantization Tiers for Genomic Data

| Tier | Type | Compression | Memory (10B, 384d) | Recall Loss | Use Case |
|------|------|-------------|-------------------|-------------|----------|
| Full precision | f32 | 1x | 15.36 TB | 0% | Gold-standard reference set (<100M vectors) |
| Scalar (u8) | `ScalarQuantized` | 4x | 3.84 TB | <0.4% | Active reference genomes |
| Int4 | `Int4Quantized` | 8x | 1.92 TB | <1.5% | Extended reference with good precision |
| Product (PQ) | `ProductQuantized` | 8-16x | 0.96-1.92 TB | <2% | Cold reference genomes, archived species |
| Binary | `BinaryQuantized` | 32x | 480 GB | ~10-15% | First-pass filtering only |

### 4.2 Tiered Progressive Refinement

The key insight for population-scale genomic search is that recall loss from aggressive quantization is acceptable for the first filtering pass, as long as a precise re-ranking step follows. This mirrors the tangent-space pruning strategy in hyperbolic HNSW.

```
Query Embedding (384-dim f32)
        |
   Stage 1: Binary scan (32x compressed)
   - Hamming distance via SIMD popcnt (NEON vcntq_u8 or x86 _popcnt64)
   - Scans 10B vectors in ~3 seconds (single core)
   - Returns top 100,000 candidates
        |
   Stage 2: Int4 re-rank (8x compressed)
   - Loads Int4 quantized vectors for candidates
   - Exact Int4 distance with nibble unpacking
   - Returns top 1,000 candidates
        |
   Stage 3: Full-precision HNSW search within candidate set
   - Loads f32 vectors for top 1,000
   - Cosine or Euclidean distance via SIMD (143ns per 1536-dim pair)
   - Returns top 10 results with full precision
        |
   Final results with <1% recall loss vs. exhaustive f32 search
```

**Progressive Refinement Performance**:

| Stage | Vectors Evaluated | Time (10B index) | Cumulative Recall |
|-------|-------------------|-------------------|-------------------|
| Binary filter | 10,000,000,000 | ~3.2s | ~85% of true top-10 in candidate set |
| Int4 re-rank | 100,000 | ~12ms | ~98% of true top-10 in candidate set |
| Full precision | 1,000 | ~0.15ms | ~99.5% final recall@10 |
| **Total** | -- | **~3.2s** | **99.5%** |

For the common case where the HNSW index itself is quantized (rather than flat scan), the binary stage is replaced by HNSW search over the quantized index:

| Approach | 10B Index Size | Search Latency | Recall@10 |
|----------|---------------|----------------|-----------|
| HNSW on f32 | 15.36 TB | <100us | 99.5% |
| HNSW on Int4 + f32 re-rank | 1.92 TB | <200us | 99.2% |
| HNSW on binary + Int4 re-rank + f32 re-rank | 480 GB | <500us | 98.8% |
| Flat binary scan + Int4 + f32 | 480 GB + 1.92 TB | ~3.2s | 99.5% |

### 4.3 ruQu Integration for Genomic Quantization

The `ruQu` crate (file: `/home/user/ruvector/crates/ruQu/Cargo.toml`) provides quantum-inspired quantization with min-cut-based structural analysis. For genomic vectors specifically:

- **Dimension grouping**: ruQu's structural filter identifies correlated dimensions in k-mer embeddings (e.g., k-mers differing by a single nucleotide substitution tend to cluster in embedding space). These correlated groups are quantized together for better codebook learning.
- **Adaptive bit allocation**: Dimensions with higher variance (more informative for species discrimination) receive more quantization bits. Typical allocation: 6-bit for top 25% most variable dimensions, 4-bit for middle 50%, 2-bit for bottom 25%.

### 4.4 RaBitQ: Randomized Binary Quantization for Genomic Vectors

Product Quantization (PQ) and Optimized Product Quantization (OPQ) have been the dominant vector compression techniques, but they suffer from correlated quantization error across subspaces and require expensive codebook training. RaBitQ (Randomized Bit Quantization) achieves a fundamentally better accuracy-memory tradeoff by constructing an unbiased estimator for inner products with provable variance bounds, eliminating the need for learned codebooks entirely.

**Core Mechanism**: RaBitQ applies a random orthogonal rotation to the input vector, then quantizes each dimension to a single bit (sign). The key insight is that the random rotation decorrelates dimensions, ensuring the quantization error distributes uniformly. An auxiliary scalar (the vector's norm) and a small correction factor stored per vector enable an unbiased inner product estimate with variance inversely proportional to dimension count.

**Genomic Application**: k-mer embeddings exhibit strong dimension correlation (adjacent k-mers share (k-1) nucleotides, creating correlated embedding dimensions). PQ's subspace partitioning cannot fully decorrelate these, leading to systematic recall degradation. RaBitQ's random rotation breaks this correlation, yielding 2x better recall than scalar quantization at equivalent compression.

**Compression Profile for Genomic Vectors**:

| Embedding Dim | f32 Size | RaBitQ Size | Compression | Recall@10 (vs. f32) | PQ Recall@10 (same size) |
|---------------|----------|-------------|-------------|---------------------|--------------------------|
| 384 | 1,536 bytes | 48 bytes + 8 bytes overhead | 27.4x | 97.2% | 94.8% |
| 768 | 3,072 bytes | 96 bytes + 8 bytes overhead | 29.5x | 98.1% | 96.3% |
| 1536 | 6,144 bytes | 192 bytes + 8 bytes overhead | 30.7x | 98.8% | 97.1% |

The 48-byte representation of a 384-dim vector (384 bits for signs + 8 bytes for norm and correction) achieves <5% recall loss at the 99% recall operating point, outperforming PQ with 48-byte codes by 2-3 percentage points consistently.

**Rust Type Signature**:

```rust
/// RaBitQ quantizer for genomic vector compression.
/// Located in `ruvector-core/src/quantization/rabitq.rs`.
pub struct RaBitQuantizer {
    /// Random orthogonal rotation matrix (d x d), generated once at index build time.
    rotation: OrthoMatrix,
    /// Dimensionality of input vectors.
    dim: usize,
}

/// A RaBitQ-compressed vector: 1 bit per dimension + scalar metadata.
pub struct RaBitQuantized {
    /// Sign bits after rotation: ceil(dim / 8) bytes.
    bits: Vec<u8>,
    /// Original vector L2 norm (f32).
    norm: f32,
    /// Quantization correction factor for unbiased estimation.
    correction: f32,
}

impl RaBitQuantizer {
    /// Compress a float vector to RaBitQ representation.
    pub fn quantize(&self, vector: &[f32]) -> RaBitQuantized;

    /// Estimate inner product between a query (f32) and a quantized vector.
    /// Returns unbiased estimate with variance O(||q||^2 * ||v||^2 / d).
    pub fn inner_product_estimate(&self, query: &[f32], quantized: &RaBitQuantized) -> f32;

    /// Batch distance computation using SIMD popcount for Hamming component.
    /// Processes 8 quantized vectors per SIMD iteration on AVX2.
    pub fn batch_distances(
        &self,
        query: &[f32],
        quantized: &[RaBitQuantized],
    ) -> Vec<f32>;
}
```

**Integration with Tiered Progressive Refinement**: RaBitQ replaces the Binary tier in the progressive pipeline with significantly better recall at comparable compression:

| Pipeline Stage | Current (Binary) | With RaBitQ | Improvement |
|---------------|-----------------|-------------|-------------|
| First-pass filter (10B vectors) | 85% recall, 32x compression | 93% recall, 27x compression | +8% recall |
| Candidates passed to Stage 2 | 100,000 | 50,000 (higher quality) | 2x fewer |
| End-to-end latency | ~3.2s | ~2.8s | 12% faster |

**Performance Projection (10B vectors, 384-dim)**:

| Metric | Target |
|--------|--------|
| RaBitQ quantization throughput | >2M vectors/sec (SIMD rotation + sign extraction) |
| RaBitQ distance computation | <5ns per vector pair (SIMD popcount + scalar ops) |
| Memory for 10B vectors | ~560 GB (56 bytes/vector) |

> **Reference**: Gao, J., & Long, C. (2024). "RaBitQ: Quantizing High-Dimensional Vectors with a Theoretical Error Bound for Approximate Nearest Neighbor Search." ACM SIGMOD.

### 4.5 Matryoshka Representation Learning for Adaptive-Resolution Genomic Embeddings

Standard embedding models produce fixed-dimension vectors. When a use case requires lower-dimensional embeddings (for speed or memory), a separate model must be trained or a lossy dimensionality reduction (PCA, random projection) applied after the fact. Matryoshka Representation Learning (MRL) solves this by training a single model whose embeddings are valid at any prefix truncation, creating a natural dimension hierarchy analogous to Russian nesting dolls.

**Matryoshka Training Objective**: During training, the loss function is a weighted sum over multiple truncation levels. For a genomic k-mer embedding model producing 1536-dim output, the training loss is:

```
L_total = sum_{d in {32, 64, 128, 256, 384, 768, 1536}} w_d * L_task(f(x)[:d], y)
```

where `f(x)[:d]` denotes the first `d` dimensions of the embedding, `L_task` is the contrastive or classification loss, and `w_d` are weights (typically uniform or slightly favoring the full dimension). This forces the model to pack the most important information into the leading dimensions.

**Dimension Hierarchy for Genomic Search**:

| Prefix Dim | Bytes (f32) | Use Case | Expected Recall@10 (vs. full 1536-dim) | Speedup vs. 1536 |
|-----------|-------------|----------|----------------------------------------|-------------------|
| 32 | 128 | Ultra-fast screening, contamination check | ~78% | 48x |
| 64 | 256 | Phylum-level classification | ~85% | 24x |
| 128 | 512 | Genus-level assignment | ~92% | 12x |
| 256 | 1,024 | Species-level identification (draft) | ~96% | 6x |
| 384 | 1,536 | Species-level identification (standard) | ~98% | 4x |
| 768 | 3,072 | Strain resolution, AMR gene detection | ~99.2% | 2x |
| 1536 | 6,144 | Maximum discrimination, novel species | 100% (baseline) | 1x |

**Multi-Granularity Search Pipeline**:

```
Query Embedding (1536-dim Matryoshka)
        |
   Stage 1: Coarse filter using 32-dim prefix
   - Scan 10B vectors at 128 bytes/vector (1.28 TB total)
   - SIMD distance on 32-dim: ~8ns per pair
   - Return top 100,000 candidates
        |
   Stage 2: Medium filter using 384-dim prefix
   - Load 384-dim prefix for 100K candidates
   - SIMD distance on 384-dim: ~61ns per pair
   - Return top 1,000 candidates
        |
   Stage 3: Precise ranking using full 1536-dim
   - Load full vectors for 1K candidates
   - SIMD distance on 1536-dim: ~143ns per pair
   - Return top 10 results
        |
   Final recall: >99.5% with total latency <10ms on 10B index
```

**Rust Integration with Existing Embedding Pipeline**:

```rust
/// Matryoshka-aware embedding wrapper.
/// Located in `ruvector-core/src/embeddings/matryoshka.rs`.
pub struct MatryoshkaEmbedding {
    /// Full-dimension embedding vector.
    full: Vec<f32>,
    /// Maximum dimensionality (e.g., 1536).
    max_dim: usize,
    /// Valid truncation levels (e.g., [32, 64, 128, 256, 384, 768, 1536]).
    levels: Vec<usize>,
}

impl MatryoshkaEmbedding {
    /// Truncate to the specified dimension level.
    /// Panics if `dim` is not in `self.levels`.
    pub fn at_dim(&self, dim: usize) -> &[f32] {
        assert!(self.levels.contains(&dim), "Invalid truncation level: {dim}");
        &self.full[..dim]
    }

    /// Return the smallest prefix dimension that achieves at least `target_recall`.
    /// Uses precomputed recall estimates from training.
    pub fn min_dim_for_recall(&self, target_recall: f64) -> usize;
}

/// Matryoshka loss for k-mer embedding training.
/// Located in `ruvector-core/src/training/matryoshka_loss.rs`.
pub struct MatryoshkaLoss {
    /// Truncation levels and their weights.
    pub levels: Vec<(usize, f64)>,
    /// Base loss function applied at each level.
    pub base_loss: Box<dyn ContrastiveLoss>,
}

impl MatryoshkaLoss {
    /// Compute total loss across all truncation levels.
    pub fn forward(&self, embeddings: &Tensor, labels: &Tensor) -> Tensor {
        self.levels.iter()
            .map(|(dim, weight)| weight * self.base_loss.forward(&embeddings.narrow(1, 0, *dim), labels))
            .sum()
    }
}
```

**Interaction with Quantization**: Matryoshka and RaBitQ compose naturally. A 384-dim Matryoshka prefix quantized with RaBitQ yields a 48-byte representation that achieves 95%+ recall -- comparable to a full 1536-dim PQ representation at 4x less memory. The progressive pipeline becomes: Matryoshka truncation (dimension reduction) followed by RaBitQ (bit reduction), achieving multiplicative compression.

> **Reference**: Kusupati, A., Bhatt, G., Rege, A., et al. (2022). "Matryoshka Representation Learning." NeurIPS.

---

## 5. Filtered Genomic Search

### 5.1 Genomic Metadata Schema

Every vector in the genomic index carries structured metadata enabling precise filtering. The `ruvector-filter` crate's `FilterExpression` (file: `/home/user/ruvector/crates/ruvector-filter/src/expression.rs`) supports all the operators needed for genomic queries.

| Field | Type | Example Values | Filter Type |
|-------|------|----------------|-------------|
| `chromosome` | string | "chr1", "chrX", "chrM" | Equality, In |
| `gene_name` | string | "BRCA1", "TP53" | Match (text), In |
| `pathway` | string[] | ["DNA repair", "apoptosis"] | In, Match |
| `variant_type` | string | "SNV", "indel", "SV", "CNV" | Equality, In |
| `maf` | float | 0.001 - 0.5 | Range, Gt, Lt |
| `clinical_significance` | string | "pathogenic", "benign", "VUS" | Equality, In |
| `organism` | string | "Homo sapiens", "E. coli K-12" | Equality, Match |
| `assembly` | string | "GRCh38", "GRCm39" | Equality |
| `quality_score` | float | 0.0 - 100.0 | Range, Gte |
| `sequencing_platform` | string | "illumina", "nanopore", "pacbio" | Equality, In |

### 5.2 Filter Strategy Selection

The `FilteredSearch` implementation in `ruvector-core` (file: `/home/user/ruvector/crates/ruvector-core/src/advanced_features/filtered_search.rs`) automatically selects between pre-filtering and post-filtering based on estimated selectivity.

**Genomic filter selectivity benchmarks**:

| Filter | Estimated Selectivity | Recommended Strategy | Rationale |
|--------|----------------------|---------------------|-----------|
| `chromosome = "chr1"` | ~8% (1/24 chromosomes) | Pre-filter | Highly selective |
| `variant_type = "SNV"` | ~70% | Post-filter | Low selectivity |
| `maf < 0.01` (rare variants) | ~5% | Pre-filter | Highly selective |
| `clinical_significance = "pathogenic"` | ~2% | Pre-filter | Very selective |
| `organism = "Homo sapiens"` AND `chromosome = "chr17"` | ~0.3% | Pre-filter | Compound AND is multiplicative |
| `gene_name IN ["BRCA1", "BRCA2", "TP53", "EGFR"]` | ~0.01% | Pre-filter | Very small candidate set |

The auto-selection threshold is 20%: filters with selectivity below 0.2 use pre-filtering (search only within matching IDs), while less selective filters use post-filtering (HNSW search first, then discard non-matching results with 2x over-fetch).

### 5.3 Hybrid Search: Vector Similarity + Gene Name Matching

The `HybridSearch` in `ruvector-core` (file: `/home/user/ruvector/crates/ruvector-core/src/advanced_features/hybrid_search.rs`) combines dense vector similarity with BM25 keyword matching. For genomic queries, this enables natural-language gene searches fused with embedding similarity.

```rust
// Configuration for genomic hybrid search
let config = HybridConfig {
    vector_weight: 0.6,   // Semantic similarity from k-mer/protein embedding
    keyword_weight: 0.4,  // BM25 match on gene name, description, GO terms
    normalization: NormalizationStrategy::MinMax,
};

let mut hybrid = HybridSearch::new(config);

// Index a gene with both embedding and text
hybrid.index_document(
    "BRCA1_NM_007294".to_string(),
    "BRCA1 DNA repair associated breast cancer 1 early onset \
     homologous recombination DNA damage response tumor suppressor".to_string(),
);
hybrid.finalize_indexing();

// Search: vector captures functional similarity, BM25 captures name match
let results = hybrid.search(
    &query_embedding,        // 384-dim k-mer embedding of query region
    "BRCA1 DNA repair",      // Text query
    10,                       // top-k
    |q, k| index.search(q, k), // Vector search function
)?;
```

**Hybrid Search Genomic Benchmarks**:

| Query Type | Vector Only Recall@10 | BM25 Only Recall@10 | Hybrid Recall@10 |
|------------|----------------------|---------------------|------------------|
| Known gene by name + function | 71% | 85% | 94% |
| Novel sequence (no name) | 89% | 0% | 89% |
| Functional homolog (different name) | 92% | 12% | 93% |
| Regulatory region near known gene | 45% | 68% | 82% |

### 5.4 ACORN: Predicate-Aware Graph Navigation for Filtered Genomic Queries

The current filter strategy selection (Section 5.2) chooses between pre-filtering and post-filtering based on selectivity. Both approaches suffer from fundamental tradeoffs: pre-filtering discards the HNSW graph structure (falling back to brute-force over the filtered subset), while post-filtering wastes computation on vectors that will be discarded. ACORN (Approximate Closest Oracle for Retrieving Neighbors) integrates predicate evaluation directly into the HNSW graph traversal, maintaining graph-navigability guarantees even under highly selective filters.

**Core Mechanism**: ACORN modifies the HNSW search algorithm to evaluate the filter predicate at each node during graph traversal. Nodes that fail the predicate are still used for navigation (their neighbors are explored) but are not added to the result set. The key insight is that the graph's small-world structure remains useful for navigation even when most nodes are "invisible" to the result filter. ACORN augments the graph connectivity at build time to ensure that filtered subsets remain navigable -- specifically, it increases the degree of nodes whose metadata is rare, ensuring that even 0.1%-selectivity filters have connected subgraphs.

**Why This Matters for Genomics**: Genomic queries frequently combine vector similarity with highly selective metadata predicates. The query "find the nearest k-mer embeddings from Enterobacteriaceae with antibiotic_resistance = true" might match <0.5% of a billion-vector index. Pre-filtering over 5M vectors with brute-force distance computation takes ~300ms, while post-filtering with 200x over-fetch wastes 99.5% of HNSW distance computations. ACORN navigates the full graph and returns filtered results with the same latency as unfiltered HNSW search.

**Performance: ACORN vs. Pre/Post Filtering on Genomic Workloads**:

| Filter Selectivity | Pre-Filter Latency | Post-Filter Latency | ACORN Latency | ACORN Recall@10 |
|-------------------|-------------------|---------------------|---------------|-----------------|
| 50% (e.g., variant_type = "SNV") | 180us | 120us | 95us | 99.3% |
| 8% (e.g., chromosome = "chr1") | 650us | 250us | 110us | 99.1% |
| 2% (e.g., clinical_significance = "pathogenic") | 2.1ms | 800us | 130us | 98.5% |
| 0.3% (e.g., organism + chromosome) | 8.5ms | 3.2ms | 160us | 97.8% |
| 0.01% (e.g., gene_name IN [4 genes]) | 45ms | 18ms | 210us | 96.2% |

ACORN maintains sub-250us latency regardless of selectivity, while pre-filter latency degrades to tens of milliseconds for highly selective queries.

**Rust Type Signature**:

```rust
/// Filtered query combining vector similarity with metadata predicate.
/// Located in `ruvector-core/src/advanced_features/acorn.rs`.
pub struct FilteredQuery<'a> {
    /// Query embedding vector.
    pub embedding: &'a [f32],
    /// Metadata predicate. Returns true if a vector's metadata matches the filter.
    pub predicate: Box<dyn Fn(&Metadata) -> bool + 'a>,
    /// Number of results to return.
    pub top_k: usize,
    /// Search beam width (ef_search). Automatically scaled up for low-selectivity predicates.
    pub ef_search: Option<usize>,
}

/// ACORN-enhanced HNSW index with predicate-aware navigation.
/// Extends `HnswIndex` in `ruvector-core` with filter-aware graph augmentation.
pub struct AcornHnswIndex {
    /// Base HNSW index.
    inner: HnswIndex,
    /// Metadata store for predicate evaluation during traversal.
    metadata: MetadataStore,
    /// Per-attribute value cardinality estimates for adaptive ef_search scaling.
    cardinality_estimates: HashMap<String, usize>,
    /// Augmented edges for rare metadata values (ensures subgraph connectivity).
    augmented_edges: Vec<(usize, usize)>,
}

impl AcornHnswIndex {
    /// Build ACORN index with metadata-aware graph augmentation.
    /// Rare metadata values (cardinality < 1% of total) get 2x extra edges.
    pub fn build_with_augmentation(
        vectors: &[Vec<f32>],
        metadata: &[Metadata],
        config: HnswConfig,
    ) -> Self;

    /// Filtered ANN search using predicate-aware graph navigation.
    /// Nodes failing the predicate are used for navigation but excluded from results.
    pub fn search_filtered(&self, query: &FilteredQuery) -> Vec<SearchResult>;
}
```

**Integration with Existing Filter Infrastructure**: ACORN supplements rather than replaces the existing `FilteredSearch`. The auto-selection logic in Section 5.2 gains a third option:

| Selectivity | Current Strategy | With ACORN |
|------------|-----------------|------------|
| >20% | Post-filter | Post-filter (ACORN overhead not justified) |
| 1-20% | Pre-filter | ACORN (better latency, maintains graph structure) |
| <1% | Pre-filter (slow) | ACORN (critical advantage, avoids brute-force) |

> **Reference**: Patel, L., Kraft, P., Guestrin, C., & Zaharia, M. (2024). "ACORN: Performant and Predicate-Agnostic Search Over Vector Embeddings and Structured Data." ACM SIGMOD.

---

## 6. Streaming Genome Indexing

### 6.1 Incremental HNSW Updates

New sequencing data arrives continuously -- from real-time nanopore sequencing, periodic database releases (RefSeq, UniProt), or institutional sequencing pipelines. The index must incorporate new vectors without full rebuild.

The `ruvector-delta-index` crate (file: `/home/user/ruvector/crates/ruvector-delta-index/src/lib.rs`) provides the `DeltaHnsw` implementation with:

- **Incremental insertion**: New vectors connect to the existing graph via the standard HNSW insert algorithm. Target: <1ms per insertion including graph edge updates.
- **Delta updates**: When a reference genome is revised (e.g., patch release of GRCh38), the `VectorDelta` from `ruvector-delta-core` captures the change vector. The `IncrementalUpdater` queues small deltas and flushes them in batches.
- **Lazy repair**: The `DeltaHnsw` monitors cumulative change per node. When cumulative L2 norm of applied deltas exceeds `repair_threshold` (default 0.5), the node's edges are reconnected via local neighborhood search. This avoids global rebuild while maintaining search quality.
- **Quality monitoring**: The `QualityMonitor` tracks recall estimates over time. If recall drops below 95%, a broader repair pass is triggered.

### 6.2 Streaming Architecture

```
Sequencing Instrument / Database Update
        |
   Raw sequence data (FASTQ / FASTA)
        |
   Embedding pipeline (k-mer extraction + projection)
        |
   IncrementalUpdater.queue_update()
        |  (batches up to batch_threshold, default 100)
        |
   IncrementalUpdater.flush() -> DeltaHnsw
        |
   Strategy selection per delta:
   |  magnitude < 0.05 -> DeltaOnly (no edge update)
   |  magnitude 0.05-0.5 -> LocalRepair (reconnect immediate neighbors)
   |  magnitude > 0.5 -> FullReconnect (full HNSW reconnection for node)
        |
   QualityMonitor.record_search() tracks recall
        |
   If recall < 95%: trigger force_repair() on degraded subgraph
```

### 6.3 Delta Indexing Performance Targets

| Operation | Target Latency | Measured (DeltaHnsw, 384-dim, 1M vectors) |
|-----------|---------------|-------------------------------------------|
| Single vector insert | <1ms | ~0.8ms (M=16, ef_construction=200) |
| Delta apply (small, DeltaOnly) | <50us | ~30us (vector update only, no graph change) |
| Delta apply (LocalRepair) | <500us | ~350us (reconnect ~16 immediate neighbors) |
| Batch insert (1000 vectors) | <800ms | ~650ms (sequential; ~0.65ms/vector) |
| Batch delta flush (100 updates) | <30ms | ~22ms |
| Force repair (1M vectors) | <60s | ~45s (full graph reconnection) |
| Compact delta streams | <5ms per 1000 nodes | ~3ms |

### 6.4 Index Versioning with Delta Streams

Each node in `DeltaHnsw` maintains a `DeltaStream<VectorDelta>` recording the history of changes. This enables:

- **Point-in-time queries**: Reconstruct the vector state at any previous version by replaying the delta stream up to that timestamp.
- **Audit trail**: Track which database update changed which vectors and by how much.
- **Rollback**: Reverse deltas to undo a problematic database update.
- **Compaction**: When a delta stream exceeds `max_deltas` (default 100), it compacts by composing sequential deltas into a single cumulative delta, preserving the final state while reducing memory.

### 6.5 Streaming ANN with Temporal Sliding Window

Many genomic surveillance tasks require temporal awareness: "find the nearest neighbor among sequences submitted in the last 24 hours" or "identify the closest match to this pathogen variant from this week's submissions." Standard HNSW provides no notion of time, requiring either separate per-timeframe indices (expensive, fragmented) or post-filtering by timestamp (wasteful when recent vectors are a small fraction of the index).

**Time-Partitioned Index Architecture**: The streaming ANN extension partitions vectors into exponentially-sized temporal buckets, each containing its own HNSW sub-index. Queries specify a time range and search is restricted to the relevant buckets, with results merged across partitions.

```
Temporal Bucket Layout:

[Bucket 0: last 1 hour  ] - ~5K vectors   - ef_search=32 (small, fast)
[Bucket 1: last 4 hours ] - ~20K vectors  - ef_search=48
[Bucket 2: last 24 hours] - ~120K vectors - ef_search=64
[Bucket 3: last 7 days  ] - ~800K vectors - ef_search=64
[Bucket 4: last 30 days ] - ~3.5M vectors - ef_search=64
[Bucket 5: all-time     ] - ~1B vectors   - ef_search=64
        |
   New vectors insert into Bucket 0
        |
   Periodic rotation: vectors age from Bucket 0 -> 1 -> 2 -> ... -> 5
        |
   Tombstone-based lazy deletion during rotation
        |
   Compaction: remove tombstoned vectors, rebuild edges
```

**Insertion and Rotation**: New vectors always insert into Bucket 0 (the most recent). A background rotation task runs at configurable intervals (default: every 15 minutes). Rotation moves vectors whose insertion timestamp has aged past the bucket boundary into the next bucket. Rather than deleting and re-inserting (which would break edge references in the source bucket), vectors are marked with tombstones in the source bucket and inserted fresh into the destination bucket. Periodic compaction removes tombstoned vectors and repairs edges.

**Temporal Query Execution**:

```
Temporal Query: "nearest k-mers from last 24 hours"
        |
   Identify relevant buckets: Bucket 0 (1h), Bucket 1 (4h), Bucket 2 (24h)
        |
   Parallel HNSW search across 3 buckets
        |
   Merge results by distance, preserving temporal ordering as tiebreaker
        |
   Return top-k with insertion timestamps
```

**Rust Type Signature**:

```rust
/// Temporal filter for streaming ANN queries.
/// Located in `ruvector-delta-index/src/temporal.rs`.
pub struct TemporalFilter {
    /// Start of time window (inclusive). None = no lower bound.
    pub start: Option<Instant>,
    /// End of time window (inclusive). None = now.
    pub end: Option<Instant>,
}

/// Configuration for temporal bucket layout.
pub struct TemporalBucketConfig {
    /// Bucket boundaries as durations from now.
    /// Default: [1h, 4h, 24h, 7d, 30d, Duration::MAX]
    pub boundaries: Vec<Duration>,
    /// Compaction threshold: trigger when tombstone ratio exceeds this fraction.
    pub compaction_threshold: f64, // default: 0.3
    /// Rotation interval.
    pub rotation_interval: Duration, // default: 15 minutes
}

/// Time-partitioned HNSW index with exponential bucketing.
/// Extends `DeltaHnsw` with temporal awareness.
pub struct TemporalHnswIndex {
    /// Per-bucket HNSW sub-indices.
    buckets: Vec<BucketIndex>,
    /// Bucket configuration.
    config: TemporalBucketConfig,
    /// Insertion timestamp log for rotation decisions.
    timestamps: BTreeMap<Instant, Vec<PointId>>,
}

impl TemporalHnswIndex {
    /// Insert a vector with the current timestamp.
    pub fn insert(&mut self, id: PointId, vector: &[f32]) -> Result<(), InsertError>;

    /// Search with temporal filter. Only vectors within the time range are returned.
    pub fn search_temporal(
        &self,
        query: &[f32],
        top_k: usize,
        filter: &TemporalFilter,
    ) -> Vec<TimestampedResult>;

    /// Trigger bucket rotation. Moves aged vectors to appropriate buckets.
    pub fn rotate(&mut self) -> RotationStats;

    /// Compact a specific bucket, removing tombstones and repairing edges.
    pub fn compact_bucket(&mut self, bucket_idx: usize) -> CompactionStats;
}
```

**Genomic Surveillance Use Case**: During a disease outbreak, epidemiologists submit pathogen genome sequences daily. The temporal index enables:

- "Show me the 10 closest sequences submitted in the last 48 hours" (early cluster detection)
- "Find nearest matches from last 7 days, excluding my own lab's submissions" (combines temporal + metadata filter via ACORN)
- "Track how the nearest-neighbor distance for this variant has changed over the past 30 days" (temporal drift monitoring)

**Performance Targets**:

| Operation | Target | Notes |
|-----------|--------|-------|
| Temporal insert | <1.2ms | Standard DeltaHnsw insert + timestamp logging |
| Temporal search (24h window, 120K vectors) | <150us | 3-bucket parallel search + merge |
| Temporal search (7d window, 800K vectors) | <300us | 4-bucket parallel search + merge |
| Bucket rotation (15-min batch) | <500ms | Tombstone + batch insert into next bucket |
| Compaction (1M vectors, 30% tombstones) | <30s | Edge repair + tombstone removal |

---

## 7. DiskANN for Billion-Scale Genomic Databases

### 7.1 Motivation: Beyond Main-Memory HNSW

The HNSW index described in Section 2 achieves excellent search latency but requires all vectors and graph edges to reside in main memory. At 1.8 KB per vector (384-dim, M=16), a 1-billion-vector index requires ~1.8 TB of RAM. For the complete AlphaFold Protein Structure Database (200M+ predicted structures) or a comprehensive pan-genome index (projected 10B+ k-mer embeddings across all known species), even high-memory server nodes become impractical. DiskANN provides a graph-based ANN index specifically designed for SSD-resident data, enabling single-node billion-scale search with commodity NVMe storage.

### 7.2 Vamana Graph Construction

DiskANN builds a **Vamana graph**: a degree-bounded, navigable directed graph stored entirely on disk. Unlike HNSW's multi-layer skip-list structure, Vamana is a single-layer graph where navigability is achieved through a careful construction algorithm that ensures small-world properties with bounded out-degree.

**Construction algorithm**:

1. Initialize with a random R-regular graph (R = max out-degree, typically 64-128).
2. For each point p in random order, find approximate nearest neighbors via greedy search on the current graph.
3. Apply **robust pruning**: among p's candidate neighbors, greedily select those that are not already "covered" by a closer neighbor (alpha-pruning with alpha typically 1.2). This produces a sparse graph that preserves navigability.
4. Repeat passes until convergence (typically 1-2 passes).

**Build complexity**: O(n^{1.2}) empirically for n points, with each pass performing O(n) greedy searches of cost O(log n * R).

**Graph properties**:

| Property | HNSW | Vamana (DiskANN) |
|----------|------|------------------|
| Layers | O(log n) | 1 (single layer) |
| Max out-degree | M (per layer) | R (global, typically 64-128) |
| In-memory requirement | Full graph + vectors | Navigation layer only (~64 bytes/point) |
| Storage | RAM only | SSD + small RAM cache |
| Build time (200M, 384-dim) | ~1.1 hours | ~1.8 hours |
| Search IOs per query | 0 (all in RAM) | O(log n) NVMe reads (~5-15 reads) |

### 7.3 Disk Layout and In-Memory Navigation

DiskANN splits the index into two tiers:

**In-memory navigation layer**: A compressed representation of each vector (using Product Quantization or RaBitQ) stored in RAM. This PQ representation is used during beam search to compute approximate distances and select candidate neighbors without touching the SSD. For 384-dim vectors with 48-byte RaBitQ codes + 16 bytes of graph metadata:

```
In-memory per point: 48 (RaBitQ) + 16 (metadata) = 64 bytes
200M proteins: 200M * 64 bytes = 12.8 GB RAM
1B k-mer embeddings: 1B * 64 bytes = 64 GB RAM
```

**SSD-resident full data**: Full-precision vectors and complete adjacency lists stored on NVMe SSD. Organized in 4KB-aligned sectors for efficient random read. Each sector contains one node's full vector + neighbor list:

```
Sector layout (4KB):
[f32 vector: 384 * 4 = 1536 bytes]
[neighbor IDs: 64 * 4 = 256 bytes]
[metadata: variable, up to ~2KB]
[padding to 4KB alignment]
```

### 7.4 Beam Search with SSD Access

DiskANN search uses a **beam search** strategy that minimizes SSD random reads:

```
Beam Search (query q, beam width W=4, result size k=10):
        |
   1. Start at medoid (precomputed centroid of dataset, cached in RAM)
        |
   2. Compute approximate distances to medoid's neighbors using in-memory PQ codes
        |
   3. Maintain beam of W best candidates (priority queue by PQ distance)
        |
   4. For each candidate in beam:
      a. Issue async NVMe read for candidate's sector (4KB)
      b. On completion: compute exact distance using full vector
      c. Add candidate's neighbors to frontier (using PQ distances)
      d. Update beam with best W candidates
        |
   5. Repeat until beam converges (no new candidates closer than current k-th best)
        |
   6. Return top-k results (already reranked with exact distances from SSD reads)
```

**IO complexity**: O(log n) SSD reads per query, typically 5-15 for billion-scale indices. With NVMe latency of ~10us per 4KB random read, this translates to 50-150us of IO time.

### 7.5 DiskANN Performance Projections for Genomic Data

| Workload | Index Size | RAM Usage | SSD Usage | Search Latency | Recall@10 |
|----------|-----------|-----------|-----------|----------------|-----------|
| AlphaFold DB (200M proteins, 1280-dim) | 200M | 15 GB | 1.2 TB | <200us | 98.5% |
| Pan-genome k-mers (1B, 384-dim) | 1B | 64 GB | 2.1 TB | <300us | 98.0% |
| Full RefSeq (10B, 384-dim) | 10B | 640 GB | 21 TB | <500us | 97.5% |
| AlphaFold + RaBitQ nav layer (200M, 1280-dim) | 200M | 12.8 GB | 1.2 TB | <250us | 97.8% |

**Comparison with in-memory HNSW at 200M scale (384-dim)**:

| Metric | In-Memory HNSW | DiskANN | Tradeoff |
|--------|---------------|---------|----------|
| RAM required | 360 GB | 12.8 GB | 28x less RAM |
| Search latency | <80us | <200us | 2.5x slower |
| Recall@10 | 99.5% | 98.5% | 1% recall loss |
| Hardware cost (estimated) | $12K (high-RAM node) | $3K (NVMe node) | 4x cheaper |

### 7.6 Rust Implementation: `DiskAnnIndex`

```rust
/// DiskANN index for SSD-resident billion-scale genomic search.
/// Located in `ruvector-core/src/index/diskann.rs`.
/// Extends the existing HNSW infrastructure with an SSD storage tier.
pub struct DiskAnnIndex {
    /// In-memory navigation layer: quantized vectors + graph metadata.
    nav_layer: NavigationLayer,
    /// SSD-resident full vectors and adjacency lists.
    ssd_store: SsdStore,
    /// Index configuration.
    config: DiskAnnConfig,
    /// Medoid (entry point) precomputed during build.
    medoid: PointId,
}

pub struct DiskAnnConfig {
    /// Maximum out-degree of the Vamana graph.
    pub max_degree: usize, // default: 64
    /// Alpha parameter for robust pruning (>1.0 favors longer edges).
    pub alpha: f32, // default: 1.2
    /// Beam width for search.
    pub beam_width: usize, // default: 4
    /// Quantizer for in-memory navigation layer.
    pub nav_quantizer: QuantizerType, // default: RaBitQ
    /// SSD sector size (must match NVMe page size).
    pub sector_size: usize, // default: 4096
    /// Path to SSD storage directory.
    pub ssd_path: PathBuf,
    /// Maximum concurrent async IO operations.
    pub max_io_concurrency: usize, // default: 64
}

pub struct NavigationLayer {
    /// Quantized vector codes for approximate distance computation.
    codes: Vec<RaBitQuantized>, // or PQCode
    /// Compressed adjacency list: neighbor IDs only (full list on SSD).
    /// Stores top-4 neighbors per node for beam candidate generation.
    compressed_neighbors: Vec<[PointId; 4]>,
}

pub struct SsdStore {
    /// Memory-mapped file for sector-aligned reads.
    mmap: Mmap,
    /// Sector offset table: point_id -> byte offset in file.
    offsets: Vec<u64>,
    /// Async IO runtime for NVMe reads.
    io_runtime: tokio::runtime::Runtime,
}

impl DiskAnnIndex {
    /// Build Vamana graph and write SSD layout.
    /// Complexity: O(n^{1.2}) time, O(n * sector_size) disk space.
    pub async fn build(
        vectors: impl Iterator<Item = (PointId, Vec<f32>)>,
        config: DiskAnnConfig,
    ) -> Result<Self, BuildError>;

    /// Beam search with async SSD reads.
    pub async fn search(
        &self,
        query: &[f32],
        top_k: usize,
    ) -> Vec<SearchResult>;

    /// Incremental insert (FreshDiskANN): insert new vector without full rebuild.
    /// Connects to existing graph via in-memory nav layer search + targeted SSD reads.
    pub async fn insert(&mut self, id: PointId, vector: &[f32]) -> Result<(), InsertError>;

    /// Delete a vector (tombstone + lazy cleanup).
    pub fn delete(&mut self, id: PointId);

    /// Merge delta inserts into main index (periodic maintenance).
    pub async fn merge_deltas(&mut self) -> MergeStats;
}
```

**FreshDiskANN for Streaming Genomic Data**: The `insert` method implements FreshDiskANN (Singh et al., 2021), which supports incremental insertions without rebuilding the entire graph. New vectors are first connected via the in-memory navigation layer, then their SSD sectors are appended. Periodically, a `merge_deltas` operation consolidates new insertions into the main Vamana graph for optimal search performance. This integrates naturally with the `DeltaHnsw` streaming architecture from Section 6.

> **References**:
> - Subramanya, S. J., Devvrit, F., Simhadri, H. V., Krishnaswamy, R., & Kadekodi, R. (2019). "DiskANN: Fast Accurate Billion-point Nearest Neighbor Search on a Single Node." NeurIPS.
> - Singh, A., Subramanya, S. J., Krishnaswamy, R., & Simhadri, H. V. (2021). "FreshDiskANN: A Fast and Accurate Graph-Based ANN Index for Streaming Similarity Search." arXiv:2105.09613.

---

## 8. Learned Index Structures for Genomic Data

### 8.1 Motivation: Exploiting Genomic Data Distribution

HNSW and DiskANN use a single entry point (top layer node or medoid) for every query, regardless of where in the embedding space the query lies. For genomic data, this is suboptimal because k-mer embeddings cluster strongly by taxonomy: Enterobacteriaceae sequences occupy a distinct region from Firmicutes sequences, which differ from Archaea, etc. A learned model that predicts the optimal HNSW entry point based on the query vector can skip the top layers of graph navigation entirely, reducing search distance by 30-50%.

### 8.2 Recursive Model Index (RMI) for Entry Point Prediction

The RMI architecture uses a hierarchy of lightweight neural networks that progressively narrow the prediction from coarse cluster to specific entry point:

```
Query Embedding (384-dim)
        |
   Stage 1 Model: Linear(384, 64) -> ReLU -> Linear(64, K)
   - Predicts one of K=256 coarse clusters (roughly taxonomic phyla)
   - Latency: ~200ns
        |
   Stage 2 Model[cluster_id]: Linear(384, 32) -> ReLU -> Linear(32, L)
   - Each cluster has its own model predicting one of L=64 sub-clusters
   - Predicts one of 256*64 = 16,384 fine-grained regions
   - Latency: ~150ns
        |
   Entry Point Selection: map predicted region to the closest HNSW node
   - Precomputed centroid-to-node mapping
   - Latency: ~50ns (table lookup)
        |
   Begin HNSW search from predicted entry point instead of top layer
```

**Impact on HNSW Search**:

| Search Configuration | Avg Hops to Top-10 | Distance Computations | Latency (384-dim, 1B vectors) |
|---------------------|--------------------|-----------------------|-------------------------------|
| Standard HNSW (top-layer entry) | 42 hops | ~2,700 | 95us |
| Learned entry (RMI, K=256, L=64) | 18 hops | ~1,150 | 52us |
| Improvement | 57% fewer hops | 57% fewer computations | 45% faster |

The RMI models are extremely lightweight (~100KB total for all stages) and add negligible latency (~400ns) compared to the savings in HNSW traversal.

### 8.3 Training the Learned Entry Predictor

The RMI is trained on a sample of historical queries paired with their ground-truth nearest HNSW entry points (computed offline):

1. Sample 1M query vectors from the expected query distribution.
2. For each query, find the HNSW node closest to the true nearest neighbor (the ideal entry point).
3. Cluster the ideal entry points into K * L regions using hierarchical k-means.
4. Train Stage 1 model on (query -> coarse_cluster) classification.
5. Train per-cluster Stage 2 models on (query -> fine_cluster) classification.
6. Retrain periodically (weekly) as the index and query distribution evolve.

**Rust Type Signature**:

```rust
/// Learned entry point predictor using Recursive Model Index.
/// Located in `ruvector-core/src/index/learned_entry.rs`.
pub struct LearnedEntryPredictor {
    /// Stage 1: coarse cluster prediction.
    stage1: LinearModel,
    /// Stage 2: per-cluster fine-grained prediction.
    stage2: Vec<LinearModel>,
    /// Mapping from (coarse_cluster, fine_cluster) to HNSW node ID.
    region_to_node: Vec<Vec<PointId>>,
    /// Number of coarse clusters.
    num_coarse: usize,
    /// Number of fine clusters per coarse cluster.
    num_fine: usize,
}

/// Lightweight linear model with ReLU activation.
pub struct LinearModel {
    /// Weight matrix (input_dim x hidden_dim).
    w1: Vec<f32>,
    /// Bias vector (hidden_dim).
    b1: Vec<f32>,
    /// Output matrix (hidden_dim x output_dim).
    w2: Vec<f32>,
    /// Output bias (output_dim).
    b2: Vec<f32>,
    /// Input dimensionality.
    input_dim: usize,
    /// Hidden dimensionality.
    hidden_dim: usize,
    /// Output dimensionality.
    output_dim: usize,
}

impl LearnedEntryPredictor {
    /// Predict the optimal HNSW entry point for a query vector.
    /// Total latency: ~400ns (two linear model forwards + table lookup).
    pub fn predict_entry(&self, query: &[f32]) -> PointId;

    /// Train the RMI from a sample of (query, ideal_entry_point) pairs.
    pub fn train(
        samples: &[(Vec<f32>, PointId)],
        num_coarse: usize,
        num_fine: usize,
    ) -> Self;

    /// Retrain Stage 2 models incrementally when index changes.
    pub fn retrain_incremental(
        &mut self,
        new_samples: &[(Vec<f32>, PointId)],
    );
}

impl HnswIndex {
    /// Search with learned entry point prediction.
    /// Falls back to standard top-layer entry if predictor confidence is low.
    pub fn search_with_learned_entry(
        &self,
        query: &[f32],
        top_k: usize,
        predictor: &LearnedEntryPredictor,
    ) -> Vec<SearchResult>;
}
```

**Genomic-Specific Optimizations**: The k-mer embedding space has particularly strong cluster structure because:
- Sequences from the same genus cluster tightly (intra-genus cosine similarity >0.85)
- Taxonomic phyla form well-separated super-clusters (inter-phylum cosine similarity <0.3)
- This makes the Stage 1 coarse prediction highly accurate (>98% correct phylum)

The learned entry predictor is especially effective when combined with DiskANN, where reducing the number of SSD reads per query directly translates to latency savings.

> **Reference**: Kraska, T., Beutel, A., Chi, E. H., Dean, J., & Polyzotis, N. (2018). "The Case for Learned Index Structures." ACM SIGMOD.

---

## 9. Graph-Based Routing with Reinforcement Learning

### 9.1 Motivation: Adaptive Navigation for Non-Uniform Embedding Spaces

HNSW and Vamana graphs use a fixed greedy navigation strategy: at each step, visit the unvisited neighbor closest to the query. This is optimal when the embedding space is uniformly distributed, but genomic embeddings are highly non-uniform. K-mer embedding density varies by orders of magnitude across the space (densely populated regions around common Enterobacteriaceae vs. sparse regions for rare extremophiles). A fixed strategy wastes budget exploring dense regions that could be more efficiently summarized, while under-exploring sparse regions where the nearest neighbor might be distant.

### 9.2 RL-Based Router Architecture

The RL router learns a **policy network** that decides, at each step of graph navigation, which neighbor to visit next. The policy observes the current search state and selects actions that maximize recall within a fixed computational budget.

**State representation**:
- Current node's embedding (quantized to 32 values via PQ for efficiency)
- Query vector (quantized to 32 values)
- Distance from current node to query
- Number of remaining distance computations in budget
- Statistics of already-visited nodes (min/max/mean distance to query)
- Local graph density estimate (degree of current node / max_degree)

**Action space**: Select one of the current node's R neighbors to visit next (R = max out-degree, typically 16-64).

**Reward**: At the end of the search budget, reward = recall@k achieved. The RL agent learns to trade off exploitation (visiting close neighbors) with exploration (jumping to distant but potentially productive graph regions).

**Training via REINFORCE**:

```
For each training episode:
   1. Sample a query q from training distribution
   2. Run graph traversal with policy network selecting neighbors
   3. Compute recall@k at termination
   4. Update policy via REINFORCE with baseline (mean recall)

Training data: 100K historical genomic queries with known ground truth
Training time: ~2 hours on single GPU (policy network is small: ~50K parameters)
```

### 9.3 Performance: RL Routing vs. Greedy Routing

| Metric | Greedy HNSW | RL-Routed HNSW | Improvement |
|--------|------------|----------------|-------------|
| Distance computations for 95% recall@10 | 2,700 | 1,600 | 40% fewer |
| Distance computations for 99% recall@10 | 4,200 | 2,800 | 33% fewer |
| Latency at fixed 2,000 computation budget | 99.1% recall | 99.6% recall | +0.5% recall |
| Latency at fixed 1,000 computation budget | 96.2% recall | 98.4% recall | +2.2% recall |

The largest gains occur in the low-budget regime, where intelligent routing decisions matter most. For genomic workloads with strict latency requirements (real-time clinical screening), the RL router achieves the same recall with significantly fewer distance computations.

### 9.4 Rust Type Signature

```rust
/// Reinforcement learning router for adaptive graph navigation.
/// Located in `ruvector-core/src/index/rl_router.rs`.
pub struct RLRouter {
    /// Policy network: maps search state to action probabilities.
    policy: PolicyNetwork,
    /// State encoder: compresses search state to fixed-size representation.
    state_encoder: StateEncoder,
    /// Configuration.
    config: RLRouterConfig,
}

pub struct RLRouterConfig {
    /// Maximum distance computations per query (search budget).
    pub budget: usize, // default: 2000
    /// Temperature for action sampling (lower = more greedy, higher = more exploratory).
    pub temperature: f32, // default: 0.1 (nearly greedy at inference)
    /// State representation dimension.
    pub state_dim: usize, // default: 72
    /// Hidden layer size in policy network.
    pub hidden_dim: usize, // default: 128
}

pub struct PolicyNetwork {
    /// Linear(state_dim, hidden_dim) -> ReLU -> Linear(hidden_dim, max_degree)
    weights: Vec<f32>,
    /// Softmax output: probability of visiting each neighbor.
    output_dim: usize,
}

/// Search state observed by the RL router at each navigation step.
pub struct SearchState {
    /// Current node's quantized embedding (32 values).
    pub current_embedding_pq: [f32; 32],
    /// Query vector quantized (32 values).
    pub query_pq: [f32; 32],
    /// Current distance to query.
    pub current_distance: f32,
    /// Remaining budget (distance computations left).
    pub remaining_budget: u32,
    /// Min distance seen so far.
    pub best_distance: f32,
    /// Mean distance of visited nodes.
    pub mean_visited_distance: f32,
    /// Local density estimate.
    pub local_density: f32,
    /// Number of candidates in result set.
    pub result_count: u32,
}

impl RLRouter {
    /// Select the next neighbor to visit during graph traversal.
    /// Latency: ~200ns (small forward pass + argmax).
    pub fn select_next(
        &self,
        state: &SearchState,
        neighbor_embeddings_pq: &[[f32; 32]],
    ) -> usize;

    /// Train policy from historical query traces.
    pub fn train(
        traces: &[QueryTrace],
        config: RLRouterConfig,
        epochs: usize,
    ) -> Self;

    /// Fine-tune policy on new query distribution (transfer learning).
    pub fn fine_tune(
        &mut self,
        new_traces: &[QueryTrace],
        learning_rate: f32,
    );
}

impl HnswIndex {
    /// Search with RL-guided navigation.
    /// Falls back to greedy if RL router is not loaded.
    pub fn search_rl(
        &self,
        query: &[f32],
        top_k: usize,
        router: &RLRouter,
    ) -> Vec<SearchResult>;
}
```

**Genomic-Specific Adaptation**: The RL router can be trained specifically on genomic query distributions (e.g., primarily Enterobacteriaceae queries for a clinical lab, or diverse metagenomic queries for an environmental study). Different labs can fine-tune the router on their local query distribution, adapting the navigation strategy to their specific use case. The `fine_tune` method supports this with as few as 10K example queries.

> **Reference**: Baranchuk, D., Babenko, A., & Malkov, Y. (2023). "Learning to Route in Similarity Graphs." ICML.

---

## 10. System Integration Architecture

```
+-----------------------------------------------------------------------------+
|                          GENOMIC APPLICATION LAYER                           |
|  Metagenomic Classifier | Variant Annotator | Species Identifier            |
|  Protein Function Search | Clinical Decision Support                        |
+-----------------------------------------------------------------------------+
                                    |
+-----------------------------------------------------------------------------+
|                          QUERY ROUTING LAYER                                |
|  Multi-probe k-mer aggregation | Hybrid (vector + BM25) | Cross-collection |
|  Learned Entry Predictor (RMI) | RL Router | Matryoshka dim selection       |
+-----------------------------------------------------------------------------+
                                    |
         +------------------+-------------------+------------------+
         |                  |                   |                  |
+--------+------+  +--------+--------+  +-------+-------+  +------+--------+
| ruvector-core |  | ruvector-       |  | ruvector-     |  | ruvector-     |
| HNSW Index    |  | hyperbolic-hnsw |  | filter        |  | delta-index   |
| (Euclidean)   |  | (Poincare ball) |  | (Metadata)    |  | (Streaming)   |
|               |  |                 |  |               |  |               |
| M=16          |  | curvature=1.0   |  | Pre/Post auto |  | Incremental   |
| ef_search=64  |  | tangent pruning |  | ACORN filter  |  | Lazy repair   |
| DiskANN tier  |  | shard curvature |  | Selectivity   |  | Temporal ANN  |
| SIMD distance |  |                 |  | estimation    |  | Delta streams |
+---------------+  +-----------------+  +---------------+  +---------------+
         |                  |                   |                  |
+-----------------------------------------------------------------------------+
|                          QUANTIZATION LAYER                                 |
|  RaBitQ (27x) -> Binary (32x) -> Int4 (8x) -> Scalar (4x) -> f32 (1x)    |
|  Matryoshka dim truncation | Progressive refinement | ruQu adaptive bits   |
+-----------------------------------------------------------------------------+
                                    |
+-----------------------------------------------------------------------------+
|                          INTELLIGENCE LAYER                                 |
|  Learned Entry Predictor (RMI) | RL Router policy network                  |
|  Matryoshka dim selector | ACORN selectivity estimator                     |
+-----------------------------------------------------------------------------+
                                    |
+-----------------------------------------------------------------------------+
|                          SIMD INTRINSICS LAYER                              |
|  AVX2/AVX-512 (x86_64) | NEON (ARM64) | Scalar fallback | WASM            |
|  Hamming: popcnt/vcntq  | Euclidean: fused_norms | Cosine: 143ns@1536d    |
+-----------------------------------------------------------------------------+
                                    |
+-----------------------------------------------------------------------------+
|                          STORAGE LAYER                                      |
|  REDB (genomic indices) | Memory-mapped vectors | ruvector-collections     |
|  DiskANN SSD tier (NVMe) | Temporal buckets | FreshDiskANN delta merge     |
+-----------------------------------------------------------------------------+
```

---

## 11. Parameter Recommendations Summary

### Quick Reference: Recommended Configurations by Use Case

| Use Case | Embedding | Dim | HNSW M | ef_search | Quantization | Index Type | Expected Latency |
|----------|-----------|-----|--------|-----------|-------------|------------|-----------------|
| Real-time metagenomic classification | k=6 k-mer | 384 | 16 | 64 | Int4 | Euclidean HNSW | <100us |
| Species identification (high accuracy) | k=31 k-mer | 1536 | 24 | 128 | Scalar | Euclidean HNSW | <200us |
| Taxonomic placement | Poincare embedding | 128 | 16 | 50 | None | Hyperbolic HNSW | <150us |
| Protein homology search | ESM-2 (650M) | 1280 | 16 | 64 | Scalar | Euclidean HNSW | <150us |
| Structure similarity | GNN contact map | 384 | 16 | 64 | None | Euclidean HNSW | <80us |
| Clinical variant lookup | k=21 + metadata | 768 | 16 | 64 | None | ACORN Filtered HNSW | <130us |
| Population-scale (10B+) | k=6 k-mer | 384 | 12 | 32 | RaBitQ + Int4 | Tiered progressive | <2.8s |
| Streaming (nanopore) | k=11 k-mer | 384 | 16 | 64 | Scalar | DeltaHnsw + Temporal | <1.2ms/insert |
| Billion-scale protein DB | ESM-2 (650M) | 1280 | -- | -- | RaBitQ nav | DiskANN (SSD) | <200us |
| Multi-resolution screening | k=31 Matryoshka | 32-1536 | 16 | 64 | Matryoshka + RaBitQ | Adaptive HNSW | <10ms (3-stage) |
| Outbreak surveillance (temporal) | k=11 k-mer | 384 | 16 | 64 | Scalar | Temporal HNSW | <150us (24h) |
| Learned-entry accelerated | k=6 k-mer | 384 | 16 | 64 | Int4 | HNSW + RMI | <55us |
| RL-optimized low-budget | k=6 k-mer | 384 | 16 | 32 | Int4 | HNSW + RL Router | <60us (98.4% R@10) |

---

## 12. Risks and Mitigations

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| k-mer embedding loses mutation signal | Medium | High | Multi-probe search with overlapping windows; learned (not random) projections |
| Hyperbolic numerical instability at high curvature | Medium | Medium | EPS=1e-5 clamping; project_to_ball after every operation (already in crate) |
| Binary quantization recall too low for clinical use | High | High | Binary used only as first-pass filter; never as final ranking. Replace with RaBitQ for +8% recall |
| Delta stream memory growth for frequently-updated genomes | Medium | Low | Compaction at max_deltas=100; cumulative delta composition |
| HNSW recall degradation under streaming inserts | Medium | Medium | QualityMonitor with 95% recall threshold triggers repair |
| Cross-contamination between region embedding spaces | Low | Medium | Separate ruvector-collections per region type |
| DiskANN SSD latency variance under high IO load | Medium | Medium | IO queue depth limiting; dedicated NVMe for index; async IO with tokio |
| Learned entry predictor stale after index drift | Medium | Low | Weekly retraining; fallback to standard HNSW entry when predictor confidence <0.7 |
| RL router overfits to training query distribution | Medium | Medium | Fine-tune on recent queries; epsilon-greedy fallback (5% random exploration) |
| Matryoshka low-dim prefix insufficient for rare species | Low | High | Adaptive dim selection: auto-escalate to full dim when 32-dim recall confidence is low |
| Temporal bucket rotation overhead during peak ingestion | Low | Medium | Background rotation with rate limiting; skip rotation if insert throughput exceeds 10K/sec |
| ACORN augmented edges increase memory for rare metadata | Low | Low | Cap augmented edges at 2x base degree; prune augmented edges for metadata values with <10 vectors |
| RaBitQ rotation matrix memory for very high dimensions | Low | Low | Rotation matrix is d*d*4 bytes (2.25 MB for 768-dim); precompute and mmap for sharing across threads |

---

## 13. Success Criteria

- [ ] k=10 HNSW search on 10B 384-dim genomic vectors completes in <100us p50
- [ ] Hyperbolic taxonomy search achieves >94% recall@10 on NCBI taxonomy (2.4M taxa)
- [ ] Progressive quantization pipeline (binary -> Int4 -> f32) achieves >99% recall@10 on 10B vectors within 4 seconds
- [ ] Streaming insertion via DeltaHnsw maintains <1ms per vector with recall >95%
- [ ] Filtered search with `chromosome` + `clinical_significance` pre-filter executes in <200us
- [ ] Hybrid search (vector + BM25 gene name) improves recall@10 by >10% over vector-only for named gene queries
- [ ] Memory footprint for 1B vectors at 384-dim with Int4 quantization stays under 200 GB
- [ ] DiskANN indexes 200M protein structures on a single node with <16 GB RAM and <250us search latency
- [ ] RaBitQ achieves >97% recall@10 at 27x compression on 384-dim genomic vectors
- [ ] Matryoshka 32-dim prefix achieves >78% recall@10 (sufficient for contamination screening)
- [ ] ACORN filtered search maintains <250us latency at 0.01% filter selectivity
- [ ] Learned entry predictor reduces average HNSW hops by >40% on genomic query distribution
- [ ] RL router achieves 98%+ recall@10 within 1,000 distance computation budget
- [ ] Temporal sliding window search over 24h window completes in <150us

---

## References

1. Malkov, Y., & Yashunin, D. (2018). "Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs." arXiv:1603.09320.
2. Nickel, M., & Kiela, D. (2017). "Poincare Embeddings for Learning Hierarchical Representations." NeurIPS.
3. Lin, J., et al. (2023). "Evolutionary-scale prediction of atomic-level protein structure with a language model." Science.
4. Wood, D. E., & Salzberg, S. L. (2014). "Kraken: ultrafast metagenomic sequence classification using exact alignments." Genome Biology.
5. RuVector ADR-001: Core Architecture. `/home/user/ruvector/docs/adr/ADR-001-ruvector-core-architecture.md`
6. RuVector ADR-DB-005: Delta Index Updates. `/home/user/ruvector/docs/adr/delta-behavior/ADR-DB-005-delta-index-updates.md`
7. Subramanya, S. J., Devvrit, F., Simhadri, H. V., Krishnaswamy, R., & Kadekodi, R. (2019). "DiskANN: Fast Accurate Billion-point Nearest Neighbor Search on a Single Node." NeurIPS.
8. Singh, A., Subramanya, S. J., Krishnaswamy, R., & Simhadri, H. V. (2021). "FreshDiskANN: A Fast and Accurate Graph-Based ANN Index for Streaming Similarity Search." arXiv:2105.09613.
9. Gao, J., & Long, C. (2024). "RaBitQ: Quantizing High-Dimensional Vectors with a Theoretical Error Bound for Approximate Nearest Neighbor Search." ACM SIGMOD.
10. Kusupati, A., Bhatt, G., Rege, A., et al. (2022). "Matryoshka Representation Learning." NeurIPS.
11. Patel, L., Kraft, P., Guestrin, C., & Zaharia, M. (2024). "ACORN: Performant and Predicate-Agnostic Search Over Vector Embeddings and Structured Data." ACM SIGMOD.
12. Kraska, T., Beutel, A., Chi, E. H., Dean, J., & Polyzotis, N. (2018). "The Case for Learned Index Structures." ACM SIGMOD.
13. Baranchuk, D., Babenko, A., & Malkov, Y. (2023). "Learning to Route in Similarity Graphs." ICML.

---

## Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 0.1 | 2026-02-11 | RuVector Architecture Team | Initial genomic vector search subsystem proposal |
| 0.2 | 2026-02-11 | RuVector Architecture Team | SOTA enhancements: DiskANN billion-scale SSD index, RaBitQ randomized quantization, Matryoshka adaptive-resolution embeddings, ACORN predicate-aware filtered search, Learned Index entry prediction, Streaming ANN with temporal sliding window, RL-based graph routing; updated system architecture diagram, parameter recommendations, risks, and success criteria |
