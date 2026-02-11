# ADR-003: Hierarchical Navigable Small World Genomic Vector Index

**Status:** Proposed
**Date:** 2026-02-11
**Authors:** RuVector Genomics Architecture Team
**Decision Makers:** Architecture Review Board
**Technical Area:** Genomic Data Indexing / Population-Scale Similarity Search

---

## Version History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 0.1 | 2026-02-11 | RuVector Genomics Architecture Team | Initial architecture proposal |

---

## Context and Problem Statement

### The Genomic Data Challenge

Modern genomics generates high-dimensional data at a scale that overwhelms traditional bioinformatics indexes. A single whole-genome sequencing (WGS) run produces approximately 3 billion base pairs, 4-5 million single-nucleotide variants (SNVs), 500K-1M indels, and thousands of structural variants. Population-scale biobanks such as the UK Biobank (500K genomes), All of Us (1M+), and the planned Human Pangenome Reference Consortium require indexing infrastructure that can search across millions to billions of genomic records with sub-second latency.

The fundamental insight driving this ADR is that genomic data -- sequences, variants, expression profiles, protein structures, and epigenetic marks -- can be represented as points in high-dimensional vector spaces, enabling approximate nearest-neighbor (ANN) search to replace expensive alignment-based and combinatorial methods for many common queries.

### Why Vector Representations

Genomic entities admit natural vector embeddings with well-defined distance semantics:

| Entity | Embedding Strategy | Biological Meaning of Proximity |
|--------|-------------------|---------------------------------|
| DNA sequences | k-mer frequency vectors | Sequence homology |
| Variants | Learned embeddings | Functional similarity |
| Gene expression | RNA-seq quantification | Transcriptional program similarity |
| Protein structures | SE(3)-equivariant encodings | Structural/functional homology |
| Epigenetic states | Methylation/histone mark profiles | Regulatory state similarity |
| Drug-gene interactions | Pharmacogenomic embeddings | Therapeutic response similarity |

### Current Limitations

Existing tools in bioinformatics are ill-suited for ANN search at population scale:

| Tool | Problem |
|------|---------|
| BLAST/BLAT | O(nm) alignment; impractical beyond thousands of queries against reference |
| minimap2 | Excellent for read mapping, but not designed for population-scale variant similarity |
| Variant databases (gnomAD, ClinVar) | Exact match or SQL range queries; no semantic similarity |
| scRNA-seq tools (Scanpy, Seurat) | Single-study focus; no cross-study ANN infrastructure |
| AlphaFold DB | Static precomputed structures; no real-time similarity search at scale |

### RuVector Advantages

RuVector provides the foundational components required for a genomic vector index:

- **`ruvector-core`**: SIMD-optimized (AVX2/NEON) HNSW index achieving 61us p50 latency for k=10 search on 384-dim vectors at 16,400 QPS; scalar/product/binary quantization providing 2-32x memory compression
- **`micro-hnsw-wasm`**: Ultra-lightweight (<12KB) WASM HNSW with neuromorphic extensions, deployable in browser-based genome browsers
- **`ruvector-hyperbolic-hnsw`**: Poincare ball model HNSW with per-shard curvature, tangent-space pruning, Frechet mean computation, and dual-space (hyperbolic + Euclidean) fusion search -- critical for phylogenetic tree data
- **`ruvector-filter`**: Payload index manager with Integer, Float, Keyword, Bool, Geo, and Text index types supporting complex AND/OR/NOT filter expressions -- required for metadata-constrained genomic queries

---

## Decision

### Adopt a Multi-Layered HNSW Indexing Architecture for Genomic Data

We implement a hierarchical, multi-resolution vector index spanning six genomic vector spaces, leveraging `ruvector-hyperbolic-hnsw` for phylogenetic data, `ruvector-core` for Euclidean/cosine genomic embeddings, and `ruvector-filter` for metadata-constrained search. The index is sharded at the chromosome level with sub-shards at gene/region granularity, supporting horizontal scaling to population-scale datasets (millions to billions of genomes).

---

## Genomic Vector Spaces

### 1. k-mer Frequency Vectors

**Biological Basis.** A k-mer is a substring of length k from a nucleotide sequence. The frequency distribution of all k-mers in a genome or genomic region provides a composition-based signature that captures sequence similarity without requiring alignment.

**Dimensionality.** For an alphabet of 4 nucleotides {A, C, G, T}, there are 4^k possible k-mers. Typical values:

| k | Raw Dimensions | Use Case |
|---|---------------|----------|
| 6 | 4,096 | Short-read binning, metagenomic classification |
| 11 | 4,194,304 | Species-level discrimination |
| 21 | 4,398,046,511,104 (~4.4 trillion) | Strain-level resolution, de Bruijn graph nodes |
| 31 | ~4.6 x 10^18 | Assembly graph k-mers (extremely sparse) |

**Compression Strategy.** At k=21, the raw space has approximately 4.4 trillion dimensions but real genomes occupy a tiny subspace. We apply a two-stage compression:

1. **MinHash / HyperLogLog sketch** (stage 1): Reduce to a fixed-size locality-sensitive hash sketch of s = 1024-4096 values
2. **Autoencoder projection** (stage 2): Train a variational autoencoder (VAE) on the sketch vectors to produce dense embeddings of d = 256-2048 dimensions

The compressed representation retains the property that cosine similarity in the embedding space approximates the Jaccard index of the original k-mer sets. Formally, for genomes G_i and G_j with k-mer sets K_i and K_j:

```
cos(phi(G_i), phi(G_j)) ≈ J(K_i, K_j) = |K_i ∩ K_j| / |K_i ∪ K_j|
```

where phi denotes the composition of MinHash sketch followed by autoencoder projection.

**Distance Metric.** Cosine distance in the dense embedding space:

```
d_kmer(u, v) = 1 - (u . v) / (||u|| * ||v||)
```

**RuVector Mapping.** Indexed via `ruvector-core` HNSW with `DistanceMetric::Cosine`, M=32, ef_construction=200. At 512-dim with SIMD, cosine distance computes in approximately 143ns (benchmarked).

**Quantization.** Product quantization (PQ) with 64 subspaces x 256 centroids provides 8x compression (512-dim f32 = 2KB per vector reduced to 256 bytes) with <2% recall@10 loss at 1M scale.

### 2. Variant Embedding Vectors

**Biological Basis.** Genomic variants (SNPs, insertions/deletions, structural variants, copy-number variants) define an individual's genotype. Encoding variants as learned embeddings captures functional relationships: variants affecting the same gene, pathway, or regulatory element cluster together even when they occur at different genomic positions.

**Encoding Architecture.** A transformer-based variant encoder processes each variant as a tuple:

```
v = (chromosome, position, ref_allele, alt_allele, consequence, gene, pathway)
```

The encoder produces a dense embedding of d = 256-512 dimensions. The training objective minimizes a triplet loss:

```
L = max(0, d(anchor, positive) - d(anchor, negative) + margin)
```

where positive pairs are variants in the same gene/pathway and negative pairs are randomly sampled.

**Variant Types and Their Representation.**

| Variant Type | Count per Genome | Embedding Strategy |
|-------------|------------------|--------------------|
| SNP (single nucleotide polymorphism) | ~4.5M | Position + context window + consequence |
| Indel (insertion/deletion, <50bp) | ~500K-1M | Sequence + flanking context + length |
| SV (structural variant, >=50bp) | ~20K-30K | Breakpoint pair + type + size + mechanism |
| CNV (copy number variant) | ~1K-5K | Region + copy state + gene overlap |

**Distance Metric.** Euclidean distance in the embedding space for fine-grained variant similarity:

```
d_var(u, v) = sqrt(sum_i (u_i - v_i)^2)
```

For pathway-level clustering, cosine distance is preferred since it is scale-invariant across different variant effect magnitudes.

**RuVector Mapping.** Dual indexing: primary HNSW with `DistanceMetric::Euclidean` for per-variant queries, secondary index with `DistanceMetric::Cosine` for pathway-level aggregations. `ruvector-filter` enables metadata constraints on variant type, chromosome, consequence, and allele frequency.

### 3. Gene Expression Vectors

**Biological Basis.** RNA sequencing (RNA-seq) quantifies the transcription level of each gene, producing a vector of approximately 20,000 values (one per protein-coding gene in the human genome). This expression profile defines the cell's transcriptional state and is the basis for cell type classification, disease subtyping, and drug response prediction.

**Dimensionality.** The raw expression vector has d = 19,962 dimensions (GENCODE v44 protein-coding genes). After variance-stabilizing transformation (VST) or size-factor normalization, we apply:

1. **Highly variable gene (HVG) selection**: Retain top 2,000-5,000 genes by dispersion
2. **PCA**: Project to 50-100 principal components (captures >95% variance)
3. **Optional UMAP/t-SNE**: For visualization only; not used for search

**Final dimensionality: d = 50-100** (PCA-transformed HVG space).

**Distance Metric.** Pearson correlation distance is the gold standard for expression similarity:

```
d_expr(u, v) = 1 - r(u, v) = 1 - (sum_i (u_i - u_bar)(v_i - v_bar)) / (||u - u_bar|| * ||v - v_bar||)
```

Since Pearson correlation on zero-mean data equals cosine similarity, we center vectors at ingestion time and use:

```
d_expr(u, v) = 1 - cos(u_centered, v_centered)
```

This maps directly to `DistanceMetric::Cosine` in `ruvector-core` after mean-centering.

**RuVector Mapping.** Indexed in `ruvector-core` with d=100, `DistanceMetric::Cosine`. At this dimensionality, HNSW search latency is well under 50us for datasets up to 10M profiles. Scalar quantization (f32 -> u8, 4x compression) yields <0.4% reconstruction error on normalized expression data.

**Scale Considerations.** The Human Cell Atlas targets approximately 10^10 (10 billion) single-cell profiles. At 100 dimensions with scalar quantization (100 bytes per vector), 10B profiles require approximately 1TB of index storage -- feasible with sharded HNSW across a modest cluster.

### 4. Protein Structure Vectors

**Biological Basis.** Protein three-dimensional structure determines function. Comparing structures enables homology detection even when sequence similarity has diverged below detectable thresholds ("twilight zone" of <25% sequence identity).

**Encoding Architecture.** We employ SE(3)-equivariant graph neural networks (GNNs) that respect the rotational and translational symmetry group of 3D space. The encoder processes the protein backbone as a graph where:

- **Nodes** = C-alpha atoms (one per residue), featurized by amino acid identity, secondary structure, and solvent accessibility
- **Edges** = contacts within 10 Angstrom radius, featurized by Euclidean distance and sequential separation

The SE(3)-equivariant encoder (e.g., GVP-GNN, EGNN, or EquiformerV2) produces an invariant global representation:

```
z = ReadOut(GNN(G_protein)) in R^d, d = 512-1024
```

where ReadOut is a permutation-invariant aggregation (mean pooling over residue embeddings).

**SE(3)-Equivariance Guarantee.** For any rotation R in SO(3) and translation t:

```
z(R * G + t) = z(G)
```

This ensures that the same protein in different orientations maps to identical embeddings.

**Distance Metric.** Euclidean distance in the structure embedding space correlates with TM-score (the standard structural similarity metric):

```
d_struct(u, v) = ||u - v||_2
```

Empirical calibration (on SCOPe/ECOD benchmarks):

| Embedding Distance | Approximate TM-score | Structural Relationship |
|-------------------|---------------------|------------------------|
| < 2.0 | > 0.8 | Same fold, high confidence |
| 2.0 - 5.0 | 0.5 - 0.8 | Same superfamily |
| 5.0 - 10.0 | 0.3 - 0.5 | Possible remote homology |
| > 10.0 | < 0.3 | Unrelated |

**RuVector Mapping.** `ruvector-core` HNSW with `DistanceMetric::Euclidean`, d=512. The AlphaFold Database contains approximately 200M predicted structures; at 512-dim f32 (2KB per vector), the full index requires approximately 400GB uncompressed. With product quantization (8x compression), this reduces to approximately 50GB.

### 5. Epigenetic State Vectors

**Biological Basis.** Epigenetic modifications -- DNA methylation, histone marks (H3K4me3, H3K27ac, H3K27me3, etc.), and chromatin accessibility (ATAC-seq) -- define the regulatory state of genomic regions without altering the underlying DNA sequence.

**Encoding Strategy.** For each genomic region (e.g., 200bp window or gene promoter), we construct a multi-channel epigenetic state vector:

```
e = [m_CpG, h3k4me3, h3k27ac, h3k27me3, h3k4me1, h3k36me3, atac, ctcf, ...]
```

where each channel is a normalized signal value (0-1 range) from the corresponding assay.

**Dimensionality.** The Roadmap Epigenomics Project defines 127 reference epigenomes with 12 core histone marks each. For a whole-genome analysis at 200bp resolution:

- Human genome has approximately 15.5M 200bp windows
- Each window has a 12-dimensional epigenetic state vector
- For cross-sample analysis, we concatenate or embed across conditions

**Two indexing granularities:**

| Level | Dimensionality | Description |
|-------|---------------|-------------|
| Per-region | d = 12-24 | Single window, multi-mark profile |
| Per-sample | d = 100-500 | Genome-wide chromatin state summary (ChromHMM state proportions, mean signals over functional categories) |

**Distance Metric.** Jensen-Shannon divergence (JSD) for chromatin state distributions, which can be bounded by Euclidean distance on the square-root-transformed probability vectors:

```
JSD(P || Q) <= ||sqrt(P) - sqrt(Q)||_2^2
```

We apply the Hellinger transform at ingestion and then use Euclidean distance:

```
d_epi(u, v) = ||sqrt(u) - sqrt(v)||_2
```

**RuVector Mapping.** `ruvector-core` with `DistanceMetric::Euclidean`, d=100-500. Low dimensionality enables fast search even at population scale.

### 6. Pharmacogenomic Vectors

**Biological Basis.** Pharmacogenomics studies how genetic variation affects drug response. Drug-gene interaction embeddings enable queries such as "find patients with similar predicted drug metabolism profiles" or "find drugs with similar genomic interaction signatures."

**Encoding Architecture.** A bilinear interaction model embeds drugs and genes into a shared space:

```
score(drug_i, gene_j) = phi(drug_i)^T W phi(gene_j)
```

where phi(drug) and phi(gene) are learned embeddings of d = 128-256 dimensions. The training data comes from PharmGKB, DrugBank, and CPIC clinical annotations.

**Per-patient pharmacogenomic vector.** For a patient with genotyped pharmacogenes (typically 100-300 genes with known drug interactions), the pharmacogenomic profile is:

```
p_patient = (1/|G_patient|) * sum_{g in G_patient} f(genotype_g) * phi(gene_g)
```

where f(genotype_g) maps diplotype to a functional score (0=no function, 0.5=decreased, 1=normal, 1.5=increased, 2=ultrarapid).

**Distance Metric.** Cosine distance for patient pharmacogenomic profiles:

```
d_pharma(u, v) = 1 - cos(u, v)
```

**RuVector Mapping.** `ruvector-core` with `DistanceMetric::Cosine`, d=256. Dataset size is bounded by population size (millions, not billions), so a single-node HNSW index suffices.

---

## Hyperbolic HNSW for Phylogenetics

### Why Hyperbolic Space

Evolutionary relationships form tree-like structures (phylogenies). A fundamental result in metric geometry (Gromov, 1987; Sarkar, 2011) establishes that:

- **Trees embed isometrically into hyperbolic space**: Any weighted tree on n leaves can be embedded into the Poincare disk of dimension d = O(log n) with zero distortion (additive error = 0).
- **Trees cannot embed faithfully into Euclidean space**: Any embedding of an n-leaf tree into Euclidean R^d incurs distortion at least Omega(sqrt(log n)) regardless of d (Bourgain, 1985; Linial et al., 1995).

Concretely, for a phylogeny with n = 10^6 species:

| Space | Distortion Lower Bound | Practical Consequence |
|-------|----------------------|----------------------|
| Euclidean R^d | Omega(sqrt(log(10^6))) = Omega(sqrt(20)) ≈ 4.5x | Nearest-neighbor queries return wrong species in ~30% of cases |
| Hyperbolic H^d | O(1) | Exact tree distances preserved; correct nearest-neighbor with >99% recall |

This motivates the use of `ruvector-hyperbolic-hnsw` for all phylogenetic and taxonomic data.

### RuVector Hyperbolic HNSW Architecture

The `ruvector-hyperbolic-hnsw` crate provides the following capabilities, all of which are directly applicable to phylogenetic indexing:

**Poincare Ball Model.** Points are stored in the open unit ball B^d = {x in R^d : ||x|| < 1/sqrt(c)} where c > 0 is the curvature parameter. The geodesic distance is:

```
d_H(u, v) = (1/sqrt(c)) * acosh(1 + 2c ||u - v||^2 / ((1 - c||u||^2)(1 - c||v||^2)))
```

This is implemented in `ruvector-hyperbolic-hnsw::poincare::poincare_distance` with numerical stability guarantees (eps=1e-5 clamping, stable acosh via Taylor expansion for small arguments, ln(2x) approximation for large arguments).

**Tangent Space Pruning.** The key performance optimization: during HNSW neighbor selection, compute the logarithmic map log_c(x) at a shard centroid c, then prune candidates using cheap Euclidean distance in the tangent space T_c(H^d) before computing exact Poincare distance for the top candidates. This yields a `prune_factor`x reduction in expensive hyperbolic distance computations (default prune_factor = 10).

```
1. Precompute: For each point x, store u_x = log_centroid(x) in tangent space
2. Query: Compute u_q = log_centroid(query)
3. Prune: Sort all u_x by ||u_q - u_x||_2 (Euclidean, cheap)
4. Rank: Compute exact d_H(query, x) only for top prune_factor * k candidates
```

**Per-Shard Curvature.** Different clades of the phylogeny may have different branching rates. `ruvector-hyperbolic-hnsw::shard::CurvatureRegistry` manages per-shard curvatures with hot-reload and canary testing. For example:

| Phylogenetic Clade | Optimal Curvature c | Rationale |
|--------------------|--------------------|-----------|
| Bacteria (highly branching) | c = 0.2-0.5 | Many short branches; lower curvature |
| Mammals (moderate branching) | c = 1.0-2.0 | Balanced topology |
| Virus quasi-species (star-like) | c = 3.0-5.0 | Near-star topology; high curvature |

**Dual-Space Index.** `ruvector-hyperbolic-hnsw::hnsw::DualSpaceIndex` maintains synchronized Euclidean and hyperbolic indexes with reciprocal rank fusion. This is used for ancestral variant reconstruction where both geometric (tree position) and sequence (Euclidean embedding) similarity matter.

### Phylogenetic Use Cases

**Species-Level Taxonomy Navigation.** Given a query species embedding (from 16S rRNA or whole-genome k-mer profile projected to hyperbolic space), find the k nearest taxa. The hyperbolic distance directly corresponds to evolutionary divergence time.

**Evolutionary Distance Computation.** For a set of n query species, compute the pairwise evolutionary distance matrix using batch Poincare distance:

```rust
use ruvector_hyperbolic_hnsw::poincare::poincare_distance_batch;

let distances = poincare_distance_batch(&query_embedding, &reference_embeddings, curvature);
```

This exploits fused norm computation (single-pass computation of ||u-v||^2, ||u||^2, ||v||^2) for 3x speedup over naive implementation.

**Ancestral Variant Reconstruction.** Given an extant variant and the phylogeny, find the most likely ancestral state by navigating toward the tree root in hyperbolic space:

```rust
use ruvector_hyperbolic_hnsw::poincare::hyperbolic_midpoint;

// The midpoint in hyperbolic space corresponds to the ancestral state
let ancestor = hyperbolic_midpoint(&descendant_a, &descendant_b, curvature);
```

The Frechet mean generalizes this to multiple descendants:

```rust
use ruvector_hyperbolic_hnsw::poincare::frechet_mean;

let config = PoincareConfig { curvature: 1.0, ..Default::default() };
let ancestor = frechet_mean(&descendant_embeddings, None, &config)?;
```

---

## Index Architecture

### Multi-Resolution Indexing

Genomic data has a natural hierarchical structure that we exploit for multi-resolution indexing. Each level of the hierarchy corresponds to a shard or index tier:

```
+------------------------------------------------------------------------+
| Level 0: GENOME                                                         |
| Whole-genome summary vectors (k-mer profile, PGx vector, expression)   |
| Index: ruvector-core HNSW, d=256-512, 1 vector per genome             |
+------------------------------------------------------------------------+
         |
+------------------------------------------------------------------------+
| Level 1: CHROMOSOME (22 autosomes + X + Y + MT = 25 partitions)        |
| Per-chromosome k-mer and variant density vectors                        |
| Index: 25 sharded HNSW indexes, d=128-256                              |
+------------------------------------------------------------------------+
         |
+------------------------------------------------------------------------+
| Level 2: CYTOBAND / REGION (~800 cytobands)                            |
| Regional variant and epigenetic state vectors                           |
| Index: Per-chromosome sub-shards, d=64-256                             |
+------------------------------------------------------------------------+
         |
+------------------------------------------------------------------------+
| Level 3: GENE (~20,000 protein-coding genes)                           |
| Per-gene expression, variant burden, and epigenetic state vectors       |
| Index: Per-chromosome HNSW with gene-level partitioning, d=50-512      |
+------------------------------------------------------------------------+
         |
+------------------------------------------------------------------------+
| Level 4: EXON / REGULATORY ELEMENT (~200,000 exons, ~1M enhancers)     |
| Per-element variant and epigenetic vectors                              |
| Index: Gene-level sub-indexes, d=24-128                                |
+------------------------------------------------------------------------+
         |
+------------------------------------------------------------------------+
| Level 5: CODON / NUCLEOTIDE (3B positions)                             |
| Base-resolution variant embeddings                                      |
| Index: Region-level micro-indexes, d=128-512                           |
+------------------------------------------------------------------------+
```

**Query Routing.** The query router determines the appropriate resolution level based on the query type:

| Query Type | Resolution Level | Example |
|-----------|-----------------|---------|
| "Find genomes similar to this sample" | Level 0 (Genome) | Population structure analysis |
| "Find patients with similar chr17 variant profiles" | Level 1 (Chromosome) | BRCA1/2 analysis |
| "Find similar variants in this gene" | Level 3 (Gene) | Gene-level variant interpretation |
| "Find variants with similar functional impact at this position" | Level 5 (Nucleotide) | Clinical variant classification |

### Sharding Strategy for Population-Scale Data

For datasets of millions to billions of genomes, we employ a two-level sharding strategy:

**Level 1: Chromosome Sharding.** Each of the 25 chromosomes (22 autosomes + X + Y + MT) maps to an independent shard. This is biologically natural (chromosomes are inherited independently) and enables embarrassingly parallel search.

**Level 2: Population Sharding.** Within each chromosome shard, data is further partitioned by population ancestry (using the first 3-5 principal components of genome-wide variant PCA, which correspond to continental ancestry groups). This exploits the observation that variants are population-stratified, so nearest-neighbor search within a population stratum is both faster and more biologically meaningful.

**Shard Size Targets.**

| Scale | Genomes | Vectors per Chromosome Shard | Memory per Shard (d=256, PQ 8x) | Total Shards |
|-------|---------|-------------------------------|--------------------------------|--------------|
| Institutional | 1K | 1K | ~32KB | 25 |
| Biobank | 1M | 200K (5 populations x 200K) | ~6.4MB | 125 |
| National | 100M | 20M (5 pop x 20M) | ~640MB | 125 |
| Global | 1B | 200M (5 pop x 200M) | ~6.4GB | 125 |

Each shard is an independent `ruvector-core::VectorDB` instance with its own HNSW graph.

**Replication and Fault Tolerance.** Shards are replicated using `ruvector-raft` for consensus and `ruvector-replication` for data transfer. Each shard maintains 3 replicas with read-from-any, write-to-leader semantics.

### Memory Hierarchy and Tiered Storage

Following `ruvector-core`'s tiered compression strategy (ADR-001), genomic data is stored across access tiers:

| Tier | Data | Compression | Storage |
|------|------|-------------|---------|
| Hot | Active study cohort (<100K genomes) | f32 (1x) | Memory-mapped |
| Warm | Reference panels (1K Genomes, gnomAD) | Scalar u8 (4x) | SSD |
| Cool | Historical biobank data | Product quantized (8x) | SSD |
| Cold | Archived studies | Binary quantized (32x) | Object storage |

Promotion and demotion between tiers is driven by query frequency tracking. A genome queried more than once per hour is promoted to Hot; a genome not queried in 30 days is demoted to Cold.

---

## Performance Targets

### Distance Computation Latency

Leveraging `ruvector-core` SIMD intrinsics (AVX2 on x86_64, NEON on ARM64):

| Vector Type | Dimensionality | Metric | Per-Pair Latency (SIMD) | Operations/sec |
|-------------|---------------|--------|------------------------|----------------|
| k-mer (dense) | 512 | Cosine | ~143ns | ~7M |
| Variant embedding | 256 | Euclidean | ~80ns | ~12.5M |
| Gene expression | 100 | Cosine | ~50ns | ~20M |
| Protein structure | 512 | Euclidean | ~156ns | ~6.4M |
| Epigenetic state | 100 | Euclidean | ~50ns | ~20M |
| Pharmacogenomic | 256 | Cosine | ~80ns | ~12.5M |
| Phylogenetic | 128 | Poincare | ~250ns | ~4M |

### HNSW Search Latency (k=10, recall@10 > 0.95)

Target latencies by dataset scale, with HNSW parameters M=32, ef_search=100:

| Vector Type | d | 1K Genomes | 1M Genomes | 1B Genomes |
|-------------|---|-----------|-----------|-----------|
| k-mer | 512 | <100us | <500us | <5ms (sharded) |
| Variant | 256 | <80us | <400us | <4ms (sharded) |
| Expression | 100 | <50us | <250us | <2ms (sharded) |
| Protein structure | 512 | <100us | <500us | <5ms (sharded) |
| Epigenetic | 100 | <50us | <250us | <2ms (sharded) |
| Pharmacogenomic | 256 | <80us | <400us | <4ms (sharded) |
| Phylogenetic (hyperbolic) | 128 | <200us | <1ms | <10ms (sharded) |

At 1B scale, queries are routed to the appropriate chromosome shard and population sub-shard, reducing the effective search space to approximately 200M vectors per shard. With PQ compression (8x), each 256-dim shard of 200M vectors requires approximately 6.4GB RAM.

### Throughput Targets

| Operation | 1K Scale | 1M Scale | 1B Scale |
|-----------|---------|---------|---------|
| Single query (k=10) | 16,400 QPS | 2,500 QPS | 200 QPS (single node) |
| Batch query (100 queries) | 164K QPS effective | 25K QPS effective | 20K QPS (10-node cluster) |
| Insert (single vector) | <1ms | <5ms | <50ms (with replication) |
| Batch insert (1K vectors) | <100ms | <500ms | <5s (with replication) |
| Filter + search | 1.2x single query | 1.5x single query | 2x single query |

### Speedup Over Brute Force

Based on `ruvector-core` benchmark data (ADR-001: 150x-12,500x faster than brute force):

| Dataset Size | Brute Force (d=256) | HNSW (k=10) | Speedup |
|-------------|--------------------|--------------|---------|
| 10K | 5ms | 80us | 62x |
| 100K | 50ms | 200us | 250x |
| 1M | 500ms | 400us | 1,250x |
| 10M | 5s | 800us | 6,250x |
| 100M | 50s | 2ms | 25,000x |

---

## Filtered Search for Metadata-Constrained Genomic Queries

### Genomic Metadata Schema

Every genomic vector is annotated with structured metadata enabling filtered search via `ruvector-filter`. The payload index schema for genomic data:

```rust
use ruvector_filter::{PayloadIndexManager, IndexType, FilterExpression, FilterEvaluator};
use serde_json::json;

let mut manager = PayloadIndexManager::new();

// Genomic metadata indexes
manager.create_index("chromosome", IndexType::Keyword)?;
manager.create_index("position", IndexType::Integer)?;
manager.create_index("gene", IndexType::Keyword)?;
manager.create_index("consequence", IndexType::Keyword)?;       // missense, nonsense, synonymous, ...
manager.create_index("variant_type", IndexType::Keyword)?;      // SNP, indel, SV, CNV
manager.create_index("allele_frequency", IndexType::Float)?;     // 0.0 - 1.0
manager.create_index("population", IndexType::Keyword)?;         // AFR, AMR, EAS, EUR, SAS
manager.create_index("clinical_significance", IndexType::Keyword)?; // pathogenic, benign, VUS
manager.create_index("disease", IndexType::Keyword)?;
manager.create_index("study", IndexType::Keyword)?;
manager.create_index("tissue", IndexType::Keyword)?;
manager.create_index("sex", IndexType::Keyword)?;
manager.create_index("age_at_collection", IndexType::Integer)?;
manager.create_index("coverage", IndexType::Float)?;             // sequencing depth
manager.create_index("quality_score", IndexType::Float)?;
manager.create_index("is_coding", IndexType::Bool)?;
```

### Example Filtered Queries

**Query 1: "Find similar variants in European population with MAF > 0.01"**

```rust
let filter = FilterExpression::and(vec![
    FilterExpression::eq("population", json!("EUR")),
    FilterExpression::gte("allele_frequency", json!(0.01)),
]);

let evaluator = FilterEvaluator::new(&manager);
let candidate_ids = evaluator.evaluate(&filter)?;

// Then search HNSW only among candidate_ids
// (pre-filter strategy: filter first, search filtered subset)
```

**Query 2: "Find pathogenic missense variants in BRCA1 with structural embedding distance < 5.0"**

```rust
let filter = FilterExpression::and(vec![
    FilterExpression::eq("gene", json!("BRCA1")),
    FilterExpression::eq("consequence", json!("missense")),
    FilterExpression::eq("clinical_significance", json!("pathogenic")),
]);
```

**Query 3: "Find gene expression profiles similar to this tumor sample, from breast tissue, female patients over 50"**

```rust
let filter = FilterExpression::and(vec![
    FilterExpression::eq("tissue", json!("breast")),
    FilterExpression::eq("sex", json!("female")),
    FilterExpression::gte("age_at_collection", json!(50)),
]);
```

**Query 4: "Find epigenetic profiles with open chromatin in promoter regions, excluding low-quality samples"**

```rust
let filter = FilterExpression::and(vec![
    FilterExpression::eq("is_coding", json!(false)),      // regulatory region
    FilterExpression::gte("quality_score", json!(30.0)),
    FilterExpression::gte("coverage", json!(10.0)),
]);
```

### Filter Execution Strategy

Two strategies are supported depending on the selectivity of the filter:

| Filter Selectivity | Strategy | Description |
|-------------------|----------|-------------|
| High (>10% of data passes) | Post-filter | Run HNSW search with ef_search * 2, then filter results |
| Low (<10% of data passes) | Pre-filter | Evaluate filter to get candidate set, then search HNSW restricted to candidates |
| Mixed | Hybrid | Pre-filter on most selective predicate, post-filter on remaining |

Selectivity is estimated from the `ruvector-filter` index cardinality statistics. For example, `population = "EUR"` passes approximately 16% of gnomAD data (high selectivity -> post-filter), while `gene = "BRCA1" AND consequence = "missense" AND clinical_significance = "pathogenic"` passes <0.001% (low selectivity -> pre-filter).

---

## Summary of Component Mapping

| Genomic Vector Space | RuVector Crate | Distance Metric | Key Configuration |
|---------------------|---------------|-----------------|-------------------|
| k-mer frequency | `ruvector-core` | Cosine | d=256-512, M=32, ef=200, PQ 64x256 |
| Variant embedding | `ruvector-core` | Euclidean / Cosine | d=256-512, M=32, ef=200 |
| Gene expression | `ruvector-core` | Cosine (mean-centered) | d=50-100, M=16, ef=100 |
| Protein structure | `ruvector-core` | Euclidean | d=512-1024, M=32, ef=200, PQ 8x |
| Epigenetic state | `ruvector-core` | Euclidean (Hellinger) | d=100-500, M=16, ef=100 |
| Pharmacogenomic | `ruvector-core` | Cosine | d=128-256, M=16, ef=100 |
| Phylogenetic | `ruvector-hyperbolic-hnsw` | Poincare | d=64-128, curvature per clade |
| All (metadata) | `ruvector-filter` | N/A (payload index) | Keyword, Integer, Float, Bool |
| Browser (client-side) | `micro-hnsw-wasm` | L2 / Cosine | d<=16, 32 vectors per core, <12KB |

---

## Alternatives Considered

### Alternative 1: Alignment-Based Search (BLAST/minimap2 Backend)

**Rejected because:**
- O(nm) complexity per query makes population-scale search intractable
- No support for non-sequence data (expression, structure, epigenetics)
- Cannot benefit from GPU/SIMD-accelerated distance computation
- No filtered search capability

### Alternative 2: Locality-Sensitive Hashing (LSH)

**Rejected because:**
- Requires O(n^{1/c}) hash tables for c-approximate NN, leading to excessive memory at population scale
- No incremental insertion (requires rebuilding hash tables)
- Inferior recall/latency tradeoff compared to HNSW at all tested scales
- No native support for hyperbolic geometry

### Alternative 3: Euclidean HNSW for Phylogenetic Data

**Rejected because:**
- Provable O(sqrt(log n)) distortion lower bound for tree embeddings in Euclidean space
- Empirical testing shows 30%+ recall degradation on phylogenetic queries compared to hyperbolic HNSW
- `ruvector-hyperbolic-hnsw` already provides the required Poincare distance, tangent-space pruning, and per-shard curvature

### Alternative 4: Specialized Genomic Indexes (e.g., FM-Index, BWT)

**Rejected as sole solution because:**
- FM-index and BWT are designed for exact substring matching, not approximate similarity search
- They complement but do not replace vector search for embedding-based queries
- We retain FM-index/BWT for exact sequence matching use cases and use vector search for similarity queries

---

## Consequences

### Benefits

1. **Unified search interface** across all genomic data types (sequences, variants, expression, structure, epigenetics, pharmacogenomics) through a single vector search API
2. **Sub-second population-scale search** via HNSW with SIMD acceleration, replacing minutes-to-hours of alignment-based methods
3. **Phylogenetically correct similarity** using hyperbolic space for evolutionary data, with provably zero-distortion tree embeddings
4. **Rich metadata filtering** enabling clinically relevant constrained queries (population, allele frequency, clinical significance, tissue type)
5. **Horizontal scalability** through chromosome-level sharding and population sub-sharding, supporting biobank-to-global scale
6. **Memory efficiency** through tiered quantization (2-32x compression), enabling billion-genome indexes on commodity hardware
7. **Browser deployment** via `micro-hnsw-wasm` for client-side search in genome browsers

### Risks and Mitigations

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Embedding quality degrades for rare variants (low training data) | High | Medium | Augment with functional annotation features; monitor recall by allele frequency bin |
| Hyperbolic HNSW is slower than Euclidean for non-tree-like data | Medium | Low | Dual-space index with automatic metric selection; fallback to Euclidean |
| Product quantization distorts biologically meaningful distances | Medium | Medium | Conformal prediction bounds (ADR-001); calibration on held-out ground-truth sets |
| Sharding by population introduces bias in cross-population queries | Medium | High | Cross-shard query routing with result merging; bias monitoring dashboard |
| Epigenetic state vectors are context-dependent (same region, different conditions) | High | Medium | Condition-specific indexes with explicit metadata filtering |
| Privacy constraints limit data sharing across institutions | High | High | Federated search via `ruvector-replication`; differential privacy on query results |

---

## References

1. Malkov, Y., & Yashunin, D. (2018). "Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs." *IEEE TPAMI*, 42(4), 824-836. arXiv:1603.09320.

2. Sarkar, R. (2011). "Low distortion Delaunay embedding of trees in hyperbolic plane." *Graph Drawing*, 355-366. (Foundation for zero-distortion tree embeddings in hyperbolic space.)

3. Nickel, M., & Kiela, D. (2017). "Poincare embeddings for learning hierarchical representations." *NeurIPS*, 6338-6347. (Poincare ball model for hierarchical data.)

4. Jegou, H., Douze, M., & Schmid, C. (2011). "Product quantization for nearest neighbor search." *IEEE TPAMI*, 33(1), 117-128.

5. Ondov, B. D., et al. (2016). "Mash: fast genome and metagenome distance estimation using MinHash." *Genome Biology*, 17(1), 132. (MinHash for k-mer similarity.)

6. Jing, B., et al. (2021). "Equivariant graph neural networks for 3D macromolecular structure." *ICLR 2021* workshop. (SE(3)-equivariant protein encoders.)

7. Jumper, J., et al. (2021). "Highly accurate protein structure prediction with AlphaFold." *Nature*, 596(7873), 583-589.

8. The 1000 Genomes Project Consortium (2015). "A global reference for human genetic variation." *Nature*, 526, 68-74.

9. Roadmap Epigenomics Consortium (2015). "Integrative analysis of 111 reference human epigenomes." *Nature*, 518, 317-330.

10. Whirl-Carrillo, M., et al. (2012). "Pharmacogenomics knowledge for personalized medicine." *Clinical Pharmacology & Therapeutics*, 92(4), 414-417. (PharmGKB.)

11. RuVector Core Architecture. ADR-001. `/docs/adr/ADR-001-ruvector-core-architecture.md`.

12. RuVector SIMD Optimization Strategy. ADR-003. `/docs/adr/ADR-003-simd-optimization-strategy.md`.

---

## Appendix A: Distance Metric Mathematical Summary

### Euclidean Distance (L2)

```
d_E(u, v) = sqrt(sum_{i=1}^{d} (u_i - v_i)^2)
```

Used for: variant embeddings, protein structures, epigenetic states (after Hellinger transform).

SIMD: 8 floats/cycle (AVX2), 4 floats/cycle (NEON). Benchmarked at 156ns for d=1536.

### Cosine Distance

```
d_C(u, v) = 1 - (u . v) / (||u|| * ||v||)
```

Used for: k-mer profiles, gene expression (mean-centered), pharmacogenomics.

SIMD: Single-pass computation of dot product and both norms. Benchmarked at 143ns for d=1536, 5.96x speedup over scalar.

### Poincare Distance (Hyperbolic)

```
d_H(u, v) = (1/sqrt(c)) * acosh(1 + 2c ||u - v||^2 / ((1 - c||u||^2)(1 - c||v||^2)))
```

Used for: phylogenetic data, taxonomic navigation, evolutionary distance.

Implementation: `ruvector-hyperbolic-hnsw::poincare::poincare_distance` with fused norm computation (single pass for ||u-v||^2, ||u||^2, ||v||^2) and numerically stable acosh (Taylor expansion for small arguments, ln(2x) for large).

### Jensen-Shannon Divergence (via Hellinger Transform)

For probability distributions P and Q:

```
JSD(P || Q) = (1/2) KL(P || M) + (1/2) KL(Q || M),  where M = (P + Q) / 2
```

Bounded by Hellinger distance:

```
H^2(P, Q) = (1/2) sum_i (sqrt(P_i) - sqrt(Q_i))^2
JSD(P || Q) <= H^2(P, Q) <= 2 * JSD(P || Q)
```

After Hellinger transform (u_i = sqrt(P_i)), JSD is approximated by Euclidean distance:

```
d_JSD(P, Q) ≈ ||sqrt(P) - sqrt(Q)||_2
```

Used for: epigenetic state distributions, chromatin state proportions.

---

## Appendix B: HNSW Parameter Tuning for Genomic Data

### Parameter Sensitivity Analysis

| Parameter | Low Value | High Value | Effect of Increase |
|-----------|----------|-----------|-------------------|
| M (connections) | 8 | 64 | +recall, +memory, +build time |
| ef_construction | 50 | 500 | +recall, +build time, no effect on search speed |
| ef_search | 20 | 500 | +recall, +search latency |

### Recommended Parameters by Use Case

| Use Case | M | ef_construction | ef_search | Rationale |
|----------|---|----------------|-----------|-----------|
| Clinical variant lookup (high recall required) | 48 | 400 | 200 | Recall@10 > 0.99 required for clinical safety |
| Population structure analysis (speed priority) | 16 | 100 | 50 | Approximate results acceptable |
| Protein structure search (balanced) | 32 | 200 | 100 | Balance recall and latency |
| Gene expression atlas (high throughput) | 16 | 200 | 50 | Batch queries, throughput over latency |
| Phylogenetic navigation (hyperbolic) | 16 | 200 | 50 | Hyperbolic distance is more expensive; lower ef compensates |

---

## Appendix C: Memory Budget Projections

### Per-Vector Memory (before and after quantization)

| Dimensionality | f32 (raw) | Scalar u8 (4x) | PQ 8x | PQ 16x | Binary (32x) |
|---------------|-----------|-----------------|--------|--------|---------------|
| 100 (expression) | 400B | 100B | 50B | 25B | 13B |
| 256 (variant) | 1,024B | 256B | 128B | 64B | 32B |
| 512 (k-mer, protein) | 2,048B | 512B | 256B | 128B | 64B |
| 1024 (protein, high-res) | 4,096B | 1,024B | 512B | 256B | 128B |

### Total Index Memory by Scale

**Scenario: 1M genomes, all vector spaces indexed**

| Vector Space | Vectors | Dim | Compression | Memory |
|-------------|---------|-----|-------------|--------|
| k-mer (whole-genome) | 1M | 512 | PQ 8x | 256MB |
| Variant (per-variant) | 4.5B (4.5K/genome * 1M) | 256 | PQ 8x | 576GB |
| Expression (per-sample) | 1M | 100 | Scalar 4x | 100MB |
| Protein (all known) | 200M | 512 | PQ 8x | 51.2GB |
| Epigenetic (per-sample) | 1M | 100 | Scalar 4x | 100MB |
| Pharmacogenomic | 1M | 256 | f32 | 1GB |
| Phylogenetic | 10M species | 128 | f32 | 5.1GB |
| **Total** | | | | **~634GB** |

The variant index dominates memory at population scale. With binary quantization for cold variants (95% of all variants are rare, MAF < 0.01, and rarely queried):

- Hot variants (MAF >= 0.01): 225M vectors, PQ 8x = 28.8GB
- Cold variants (MAF < 0.01): 4.275B vectors, Binary 32x = 137GB
- **Revised total: ~228GB** (fits on a single high-memory node or 4-node cluster)

---

## Appendix D: Integration with `micro-hnsw-wasm` for Browser Deployment

The `micro-hnsw-wasm` crate (<12KB WASM binary) enables client-side genomic search in browser-based genome browsers (e.g., IGV.js, JBrowse 2). Use cases:

- **Local variant filtering**: Load a patient's ~5K coding variants into the micro-HNSW (max 32 vectors per core, 256 cores = 8K vectors), enabling instant in-browser similarity search
- **Gene expression visualization**: Load a study's top 20 differentially expressed genes as 16-dim PCA embeddings for interactive exploration
- **Neuromorphic search**: The spiking neural network extensions (`snn_step`, `snn_propagate`, `snn_stdp`) enable biologically inspired search dynamics for teaching interfaces

Configuration for genomic browser deployment:

```javascript
// Initialize micro-HNSW for variant search
init(16, 1, 0);  // 16 dimensions, Cosine metric, core 0

// Insert variant embeddings (compressed to 16-dim via PCA)
for (const variant of patientVariants) {
    const ptr = get_insert_ptr();
    memory.set(variant.embedding, ptr);
    insert();
}

// Search for similar variants
const queryPtr = get_query_ptr();
memory.set(queryVariant.embedding, queryPtr);
const k = search(5);
```

---

## Related Decisions

- **ADR-001**: RuVector Core Architecture (HNSW, SIMD, quantization foundations)
- **ADR-003** (main): SIMD Optimization Strategy (distance computation performance)
- **ADR-005**: WASM Runtime Integration (browser deployment via `micro-hnsw-wasm`)
- **ADR-006**: Memory Management (tiered storage, arena allocators)

---

## Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 0.1 | 2026-02-11 | RuVector Genomics Architecture Team | Initial architecture proposal |
