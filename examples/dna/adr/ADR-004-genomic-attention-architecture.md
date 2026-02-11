# ADR-004: Flash Attention Genomic Sequence Architecture

**Status**: Proposed
**Date**: 2026-02-11
**Authors**: ruv.io, RuVector Team
**Deciders**: Architecture Review Board
**Target Crates**: `ruvector-attention`, `ruvector-attention-unified-wasm`, `ruvector-fpga-transformer`, `ruvector-sparse-inference`

## Version History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 0.1 | 2026-02-11 | ruv.io | Initial genomic attention architecture proposal |

---

## Context

### The Genomic Sequence Analysis Problem

DNA sequences encode the complete developmental program of an organism through a four-letter alphabet {A, C, G, T} arranged in double-stranded helices. The human reference genome (GRCh38) contains approximately 3.2 billion base pairs (bp) organized across 22 autosomes, 2 sex chromosomes, and mitochondrial DNA. Functional interpretation of this sequence requires capturing interactions across multiple scales:

| Biological Scale | Typical Range | Interaction Type | Example |
|-----------------|---------------|-----------------|---------|
| **Motif** | 6-30 bp | Transcription factor binding | TATA box (TATAAA) at -25 to -30 relative to TSS |
| **Codon** | 3 bp triplets | Amino acid encoding | ATG start codon, UAA/UAG/UGA stop codons |
| **Exon** | 50-300 bp | Protein-coding segments | ~180,000 exons in human genome |
| **Gene** | 1-2,400 kbp | Regulatory unit | Median ~27 kbp, DMD gene spans 2.4 Mbp |
| **TAD** | 200 kbp - 2 Mbp | Topologically associated domain | ~2,200 TADs per cell type |
| **Chromosome** | 47-249 Mbp | Structural unit | Chr1 (249 Mbp) to Chr22 (51 Mbp) |
| **Genome** | 3.2 Gbp | Complete organism program | ~20,000 protein-coding genes |

Transformer architectures are natural candidates for genomic analysis because the self-attention mechanism directly models pairwise interactions between sequence positions -- precisely the biology of enhancer-promoter loops, splice site recognition, and chromatin contact maps. However, standard self-attention has O(n^2) time and memory complexity, which is intractable for genomic-scale sequences.

### Computational Intractability of Naive Attention

For a standard self-attention layer with sequence length n and head dimension d:

**Time complexity:**

```
T_attention = 2 * n^2 * d    (for Q*K^T and attn*V matmuls)
```

**Memory for the attention matrix:**

```
M_attention = n^2 * sizeof(float)
```

For the full human genome at nucleotide resolution (n = 3.2 x 10^9):

```
M_attention = (3.2 x 10^9)^2 * 4 bytes
            = 4.096 x 10^19 bytes
            = 40.96 exabytes
```

This exceeds the total global DRAM capacity. Even for a single chromosome (Chr1, n = 249 x 10^6):

```
M_attention_chr1 = (249 x 10^6)^2 * 4 bytes
                 = 2.48 x 10^17 bytes
                 = 248 petabytes
```

For comparison, the existing `FlashAttention` in `ruvector-attention` (see `crates/ruvector-attention/src/sparse/flash.rs`) reduces memory from O(n^2) to O(B) where B is the block size, but still requires O(n^2) FLOPs. The `LocalGlobalAttention` (see `crates/ruvector-attention/src/sparse/local_global.rs`) reduces complexity to O(n * (w + g)) but uses a fixed pattern that does not adapt to genomic structure.

### What Existing Genomic Models Do -- and Where They Fall Short

| Model | Max Sequence | Architecture | Limitation |
|-------|-------------|--------------|------------|
| DNABERT-2 | 512 bp | BERT + BPE tokenization | Cannot capture enhancer-promoter (10 kbp - 1 Mbp) |
| Nucleotide Transformer | 6,000 bp | Standard transformer | Misses gene-level regulatory interactions |
| HyenaDNA | 1M bp | Implicit convolution | No explicit pairwise attention for contact map learning |
| Enformer | 196,608 bp | Dilated convolutions + attention | Fixed receptive field, no dynamic sparsity |
| Evo | 131,072 bp | StripedHyena (SSM + attention) | Limited to ~131 kbp, no chromosome-scale |

None of these models can simultaneously: (a) resolve single-nucleotide variants (SNVs) at 1 bp resolution, (b) capture megabase-scale enhancer-promoter interactions, and (c) detect trans-chromosomal translocations. A hierarchical flash attention architecture is required.

### RuVector's Existing Primitives

The following crates provide the foundation for genomic attention:

1. **`ruvector-attention`** (v0.1.31): `FlashAttention` with tiled online softmax, `LocalGlobalAttention` for O(n*(w+g)) complexity, `AttentionMask` with sparse index sets, `MoEAttention` with `TopKRouting` for expert-based routing, and `SparseMaskBuilder` for programmatic pattern construction.

2. **`ruvector-fpga-transformer`** (v0.1.0): `Engine` with `TransformerBackend` trait, `FixedShape` for zero-allocation inference, `QuantSpec` for INT4/INT8 quantization, `CoherenceGate` for principled computation routing, and deterministic latency paths suitable for clinical deployment.

3. **`ruvector-sparse-inference`**: `SparseInferenceEngine` with `LowRankPredictor` for activation locality, `SparseFfn` for selective neuron computation, 3/5/7-bit precision lanes with `GraduationPolicy`, and `PiContext` for quantization calibration.

4. **`ruvector-attention-unified-wasm`** (v0.1.0): Unified WebAssembly bindings for 18+ attention mechanisms, optimized with `opt-level = "z"`, LTO, and single codegen unit for minimal binary size.

---

## Decision

### Adopt a Six-Level Hierarchical Flash Attention Architecture for Genomic Sequence Analysis

We introduce a hierarchical attention system where each level operates on a different biological scale, uses scale-appropriate attention patterns and sparsity, and communicates with adjacent levels through pooling and upsampling operations. Each level leverages the appropriate RuVector crate for its computational characteristics.

### Architecture Overview

```
+===========================================================================+
|                     GENOMIC HIERARCHICAL ATTENTION                        |
+===========================================================================+
|                                                                           |
|  Level 6: Genome-Level          Population GWAS cross-genome attention    |
|  n_eff ~ 500K-5M variants       ruvector-sparse-inference (MoE)          |
|       |                                                                   |
|       v  (variant summarization)                                          |
|  Level 5: Chromosome-Level      Trans-chromosomal sparse attention        |
|  n_eff ~ 50K-250K bins          ruvector-sparse-inference                 |
|       |                                                                   |
|       v  (bin pooling 10kbp)                                              |
|  Level 4: Gene-Level            Regulatory element sparse attention       |
|  n_eff ~ 2K-20K elements        ruvector-sparse-inference (Hi-C prior)    |
|       |                                                                   |
|       v  (element pooling)                                                |
|  Level 3: Exon-Level            Cross-exon attention for splicing         |
|  n_eff ~ 200-2000 exons         ruvector-attention (FlashAttention)       |
|       |                                                                   |
|       v  (exon boundary pooling)                                          |
|  Level 2: Codon-Level           Reading frame grouped attention           |
|  n_eff ~ 100-5000 codons        ruvector-attention (FlashAttention)       |
|       |                                                                   |
|       v  (3bp stride pooling)                                             |
|  Level 1: Nucleotide-Level      Local sliding window flash attention      |
|  n ~ 512 bp per window          ruvector-attention (FlashAttention)       |
|  (tiled across full sequence)   ruvector-fpga-transformer (clinical)      |
|                                                                           |
+===========================================================================+
```

---

## Genomic Attention Layers

### Level 1: Nucleotide-Level Attention

**Biological rationale.** Individual nucleotides participate in short-range interactions critical for transcription factor (TF) binding site recognition, DNA secondary structure (hairpins, cruciforms), and local epigenetic signals. The typical TF binding motif spans 6-20 bp, but cooperative binding and spacing constraints extend the effective interaction range to ~500 bp. The 512 bp window captures the vast majority of cis-element interactions within a single promoter region.

**Encoding.** Each nucleotide is encoded as a 4-bit one-hot vector {A=[1,0,0,0], C=[0,1,0,0], G=[0,0,1,0], T=[0,0,0,1]}, projected to a d_model-dimensional embedding. For FPGA deployment, this encoding maps directly to 4-bit quantized inputs in `ruvector-fpga-transformer`'s `QuantSpec`.

**Attention mechanism.** We use a sliding window of w = 512 bp with stride s = 256 bp (50% overlap), applying `FlashAttention` from `ruvector-attention` within each window. The tiled online softmax ensures O(B) memory per window.

**Formal definition.** For a window starting at position p, let X_p in R^{w x d} be the nucleotide embedding matrix. The attention output is:

```
Q_p = X_p * W_Q,   K_p = X_p * W_K,   V_p = X_p * W_V

    where W_Q, W_K, W_V in R^{d x d_k}

Attention_L1(X_p) = FlashAttn(Q_p, K_p, V_p, block_size=B_1)
```

The `FlashAttn` operation from `crates/ruvector-attention/src/sparse/flash.rs` processes Q*K^T in tiles of B_1 rows, maintaining running max and sum-of-exponentials for numerically stable online softmax without materializing the full w x w attention matrix.

**Cross-window communication.** Overlapping regions (256 bp) are averaged between adjacent windows. Additionally, the last g_1 = 16 positions of each window serve as "global sentinel" tokens visible to the next window, following the `LocalGlobalAttention` pattern from `ruvector-attention`.

**Computational analysis:**

| Metric | Formula | Value (w=512, d=128, B_1=64, h=8) |
|--------|---------|-----------------------------------|
| FLOPs per window | 2 * h * w^2 * d_k | 2 * 8 * 512^2 * 16 = 67.1 MFLOPs |
| Memory per window | h * B_1 * w * sizeof(f32) | 8 * 64 * 512 * 4 = 1.0 MB |
| Windows per chromosome (Chr1) | ceil(n / s) | ceil(249M / 256) = 972,657 |
| Total FLOPs per chromosome | FLOPs/window * num_windows | 65.3 TFLOPs |
| Total FLOPs whole genome | sum over all chromosomes | ~838 TFLOPs |
| Latency per window (FPGA @ 1 TFLOP/s) | FLOPs / throughput | 67.1 us |

**FPGA acceleration.** For clinical applications requiring deterministic latency, Level 1 maps directly onto `ruvector-fpga-transformer`'s `Engine`:

```rust
use ruvector_fpga_transformer::types::FixedShape;

// Nucleotide-level attention shape
const NUCLEOTIDE_SHAPE: FixedShape = FixedShape::new(
    512,   // seq_len: window size in bp
    128,   // d_model: nucleotide embedding dimension
    8,     // heads: multi-head attention
    16,    // d_head: dimension per head
    5,     // vocab: {A, C, G, T, N}
);
```

The `FixedShape` enables zero-allocation inference paths, and the 4-bit nucleotide encoding aligns with the FPGA's native INT4 quantization datapath.

---

### Level 2: Codon-Level Attention

**Biological rationale.** In protein-coding regions, the triplet reading frame imposes a fundamental 3 bp periodicity on sequence function. Each codon maps to one of 20 amino acids (plus stop signals) according to the genetic code. Codon usage bias, wobble base pairing, and synonymous codon selection affect mRNA stability, translational speed, and protein folding. Codon-level attention captures dependencies between codons within and across reading frames, enabling reading frame detection, codon usage optimization analysis, and frameshift mutation identification.

**Pooling from Level 1.** Level 1 nucleotide embeddings are pooled into codon representations using a strided average pool with kernel size 3 and stride 3, aligned to reading frame boundaries. For each potential reading frame f in {0, 1, 2}:

```
C_i^f = (1/3) * sum_{j=0}^{2} h_{3i+f+j}^{L1}

    where h_k^{L1} is the Level 1 output embedding at position k
```

This produces three parallel codon-level sequences, one per reading frame.

**Grouped query attention across reading frames.** We apply grouped attention where codons in one reading frame attend to codons in all three reading frames, capturing frame-dependent and frame-independent patterns:

```
Q_f = C^f * W_Q^{codon},   K_g = C^g * W_K^{codon},   V_g = C^g * W_V^{codon}

Attention_L2(C^f) = FlashAttn(Q_f, concat(K_0, K_1, K_2), concat(V_0, V_1, V_2), block_size=B_2)
```

where f is the query reading frame and g in {0, 1, 2} iterates over all reading frames.

**Effective sequence length.** For a coding region of length L bp:

```
n_codons = floor(L / 3)      per reading frame
n_total  = 3 * n_codons       across all frames
```

For the median human exon (~170 bp): n_codons ~ 56 per frame, n_total ~ 168.
For the longest human exon (TTN exon 363, ~17,106 bp): n_codons ~ 5,702 per frame, n_total ~ 17,106.

**Computational analysis:**

| Metric | Formula | Value (typical exon, d=128) |
|--------|---------|----------------------------|
| FLOPs per exon | 2 * h * n_total^2 * d_k | 2 * 8 * 168^2 * 16 = 7.2 MFLOPs |
| FLOPs (TTN exon 363) | 2 * h * 17106^2 * 16 | ~74.9 GFLOPs |
| Memory per exon (flash) | h * B_2 * n_total * sizeof(f32) | 8 * 32 * 168 * 4 = 172 KB |

For exons exceeding 5,000 codons, we apply sliding window flash attention (as in Level 1) within each reading frame with window w_2 = 1024 codons.

---

### Level 3: Exon-Level Attention

**Biological rationale.** Alternative splicing affects >95% of human multi-exon genes, generating transcript diversity through exon skipping, alternative 5'/3' splice sites, intron retention, and mutually exclusive exons. Splice site recognition depends on: (a) the consensus splice site dinucleotides (GT at 5' donor, AG at 3' acceptor), (b) the branch point sequence ~20-40 bp upstream of the 3' site, (c) exonic splicing enhancers/silencers (ESEs/ESSs) within exons, and (d) long-range exon definition through cross-exon interactions. Exon-level attention directly models the pairwise compatibility between exons -- which exon combinations are co-included or mutually exclusive.

**Pooling from Level 2.** Each exon is represented by a single vector obtained by attention-weighted pooling of its codon-level embeddings:

```
e_i = sum_j alpha_j * c_j^{L2}

    where alpha_j = softmax(w_pool^T * c_j^{L2}) for codons j within exon i
```

**Exon-exon attention.** For a gene with E exons, we apply full flash attention over the exon sequence:

```
Q_exon = E * W_Q^{exon},   K_exon = E * W_K^{exon},   V_exon = E * W_V^{exon}

Attention_L3(E) = FlashAttn(Q_exon, K_exon, V_exon, block_size=B_3)

    where E in R^{num_exons x d}
```

**Splice site position encoding.** We augment exon embeddings with positional features encoding:
- Distance to nearest splice site (donor/acceptor)
- Splice site strength score (from MaxEntScan or similar)
- Exon phase (0, 1, 2 -- which codon position the exon boundary falls on)
- Constitutive vs. alternatively spliced status (from RNA-seq evidence)

**Effective sequence length.** The median human gene contains ~8.8 exons (mean ~10.4). The most exon-rich gene (TTN) has 363 exons. Even TTN is tractable for full flash attention:

```
n_max_exons = 363   (TTN)
n_median_exons = 9
```

**Computational analysis:**

| Metric | Formula | Value (TTN, d=256) |
|--------|---------|---------------------|
| FLOPs (TTN, worst case) | 2 * h * 363^2 * d_k | 2 * 16 * 363^2 * 16 = 67.4 MFLOPs |
| FLOPs (median gene) | 2 * h * 9^2 * d_k | 2 * 16 * 81 * 16 = 41.5 KFLOPs |
| Memory (TTN, flash) | h * B_3 * 363 * sizeof(f32) | 16 * 16 * 363 * 4 = 373 KB |

Level 3 is compute-light because the number of exons per gene is small. The bottleneck is the number of genes to process, which is embarrassingly parallel across genes.

---

### Level 4: Gene-Level Attention (Regulatory Element Interactions)

**Biological rationale.** Gene expression is controlled by distal regulatory elements -- enhancers (activate), silencers (repress), insulators (boundary), and locus control regions (LCRs) -- that can reside tens of kilobases to megabases from their target promoters. These elements interact through 3D chromatin looping captured by Hi-C, Micro-C, and ChIA-PET experiments. The TAD (topologically associated domain) structure constrains most enhancer-promoter interactions to within-TAD contacts, though notable exceptions exist (e.g., the Shh limb enhancer ZRS is 1 Mbp from its target).

**Regulatory element catalog.** We define the attention vocabulary at this level from curated databases:

| Element Type | Count (Human) | Source |
|-------------|---------------|--------|
| Promoters | ~20,000 | FANTOM5/EPDnew |
| Enhancers | ~1,000,000 | ENCODE cCREs / VISTA |
| Silencers | ~30,000 | SilencerDB |
| Insulators | ~50,000 | CTCF ChIP-seq peaks |
| Super-enhancers | ~10,000 | dbSUPER |

**Sparse attention via Hi-C contact prior.** Rather than computing full pairwise attention between ~1.1M regulatory elements (which would require ~5 TB of memory for the attention matrix), we use `ruvector-sparse-inference` to implement a structurally-informed sparse attention pattern derived from Hi-C contact frequency maps.

Let H(i,j) be the Hi-C contact frequency between loci i and j (typically stored at 5-10 kbp resolution). We define the sparse attention mask:

```
M_HiC(i, j) = 1   if H(i,j) > tau_contact   OR   |i - j| < d_local
              0   otherwise

    where tau_contact is a contact frequency threshold (e.g., top 5% of contacts)
    and d_local is a local neighborhood radius (e.g., 50 kbp)
```

The sparsity ratio depends on the threshold:

```
sparsity = |{(i,j) : M_HiC(i,j) = 1}| / n^2
```

For typical Hi-C data at 10 kbp resolution with n ~ 300K bins:

| Threshold (percentile) | Non-zero entries | Sparsity ratio | Memory (sparse, f32) |
|------------------------|-----------------|----------------|---------------------|
| Top 1% | ~900M | 1.0% | 3.6 GB |
| Top 5% | ~4.5B | 5.0% | 18 GB |
| Top 10% | ~9.0B | 10.0% | 36 GB |
| Local only (50 kbp) | ~1.5B | 1.7% | 6 GB |
| Combined (1% + local) | ~2.1B | 2.3% | 8.4 GB |

We target the combined mask (top 1% Hi-C + local 50 kbp) with ~2.3% density.

**Implementation via `ruvector-sparse-inference`.** The `LowRankPredictor` efficiently selects which regulatory element pairs to attend to by learning a low-rank approximation of the Hi-C-informed attention pattern:

```rust
use ruvector_sparse_inference::{SparseInferenceEngine, SparsityConfig};

// Gene-level regulatory attention
let regulatory_engine = SparseInferenceEngine::new_sparse(
    256,    // input_dim: regulatory element embedding
    4096,   // hidden_dim: FFN intermediate
    0.023,  // sparsity_ratio: ~2.3% density from Hi-C
)?;
```

**Computational analysis:**

| Metric | Formula | Value (n=300K elements, 2.3% density) |
|--------|---------|--------------------------------------|
| FLOPs (sparse attention) | 2 * h * nnz * d_k | 2 * 16 * 2.1B * 16 = 1.08 PFLOPs |
| FLOPs (full attention) | 2 * h * n^2 * d_k | 2 * 16 * 9 x 10^10 * 16 = 46.1 PFLOPs |
| Speedup from sparsity | full / sparse | ~43x |
| Memory (sparse CSR) | nnz * (sizeof(f32) + sizeof(u32)) | 2.1B * 8 = 16.8 GB |
| Memory (full dense) | n^2 * sizeof(f32) | 9 x 10^10 * 4 = 360 GB |

---

### Level 5: Chromosome-Level Attention (Trans-Chromosomal Interactions)

**Biological rationale.** Chromosomes occupy distinct territories in the nucleus, but significant inter-chromosomal interactions occur: (a) balanced translocations (e.g., t(9;22) BCR-ABL in CML, t(8;14) MYC-IGH in Burkitt lymphoma, t(15;17) PML-RARA in APL), (b) trans-chromosomal enhancer hijacking in cancer, (c) coordinated gene regulation at nuclear speckles and transcription factories, and (d) homologous chromosome pairing in meiosis. Detecting these interactions requires genome-wide, cross-chromosomal attention.

**Binning strategy.** Each chromosome is divided into non-overlapping bins of b = 10 kbp. Bin representations are derived by max-pooling Level 4 regulatory element embeddings within each bin:

```
bin_k = MaxPool({r_i^{L4} : r_i falls within genomic interval [kb, (k+1)b)})
```

**Effective sequence lengths:**

| Chromosome | Length (bp) | Bins (10 kbp) |
|-----------|------------|---------------|
| Chr1 | 248,956,422 | 24,896 |
| Chr22 | 50,818,468 | 5,082 |
| ChrX | 156,040,895 | 15,604 |
| **Whole genome** | **3,088,286,401** | **308,829** |

**Ultra-sparse attention pattern.** Cross-chromosomal contacts are rare (typically <2% of Hi-C contacts are inter-chromosomal). We implement a two-tier sparse pattern:

```
Intra-chromosomal:  Local window (w_5 = 500 bins = 5 Mbp) + Hi-C top contacts
Inter-chromosomal:  Only known translocation breakpoint regions + nuclear compartment co-localization
```

The resulting sparsity:

```
Intra-chromosomal density:   ~0.5% (window + Hi-C)
Inter-chromosomal density:   ~0.01% (breakpoints + compartments)
Overall density:             ~0.1%
Total non-zero entries:      ~95M (out of ~95B total pairs)
```

**Translocation detection.** For clinical translocation detection, we define "sentinel" attention patterns at known breakpoint hotspots. For the ~600 recurrent translocation breakpoints cataloged in COSMIC and the Mitelman database, we ensure these positions always attend to their partner regions:

```
M_translocation(i, j) = 1   if (i, j) in breakpoint_pairs
                         0   otherwise

M_L5 = M_intra_HiC  UNION  M_translocation  UNION  M_compartment
```

**Computational analysis:**

| Metric | Formula | Value (n=308K bins, 0.1% density) |
|--------|---------|----------------------------------|
| FLOPs (sparse) | 2 * h * nnz * d_k | 2 * 16 * 95M * 32 = 97.3 GFLOPs |
| Memory (sparse CSR) | nnz * 8 | 95M * 8 = 760 MB |
| Memory (dense, hypothetical) | n^2 * 4 | 3.8 x 10^11 = 381 GB |
| Compression ratio | dense / sparse | ~501x |

---

### Level 6: Genome-Level Attention (Population-Scale GWAS)

**Biological rationale.** Genome-wide association studies (GWAS) compare genetic variants across thousands to millions of individuals to identify disease-associated loci. As of 2026, the GWAS Catalog contains >400,000 variant-trait associations from >6,000 studies. Population-scale analysis requires cross-genome attention where variant representations from one individual attend to the same locus across the cohort, enabling: (a) linkage disequilibrium (LD) pattern learning, (b) polygenic risk score (PRS) computation, (c) gene-gene interaction (epistasis) detection, and (d) population stratification correction.

**Variant representation.** Each variant is encoded as:

```
v_i = [allele_embedding || positional_encoding || MAF_encoding || LD_context]

    where:
    - allele_embedding in R^{d_allele}:  learned embedding for ref/alt alleles
    - positional_encoding in R^{d_pos}:   genomic coordinate encoding
    - MAF_encoding in R^{d_maf}:          minor allele frequency features
    - LD_context in R^{d_ld}:             local LD structure from Level 5
```

**Mixture-of-Experts routing for tissue-specific analysis.** Different tissues and diseases activate different regulatory programs. We use `ruvector-attention`'s `MoEAttention` with `TopKRouting` to route variants to tissue-specific expert networks:

```rust
use ruvector_attention::moe::{MoEAttention, MoEConfig, TopKRouting};

let gwas_moe_config = MoEConfig {
    num_experts: 53,       // One per GTEx tissue type
    top_k: 4,              // Activate top 4 tissues per variant
    capacity_factor: 1.25,
    jitter_noise: 0.01,
};
```

The expert gating function learns which tissues are relevant for each variant based on epigenomic annotations (chromatin accessibility, histone marks, eQTL evidence):

```
G(v_i) = TopK(softmax(W_gate * v_i), k=4)

Output_L6(v_i) = sum_{j in TopK} G(v_i)_j * Expert_j(v_i)
```

**Cross-genome attention.** For a cohort of P individuals and V variants:

| GWAS Scale | Individuals (P) | Variants (V) | Naive complexity |
|-----------|----------------|-------------|-----------------|
| Small study | 5,000 | 500,000 | 2.5 x 10^9 |
| Biobank | 500,000 | 1,000,000 | 5 x 10^11 |
| Global meta-analysis | 5,000,000 | 5,000,000 | 2.5 x 10^13 |

We apply sparse attention in two dimensions:

1. **Locus attention** (within each variant, across individuals): O(P) linear attention via `LinearAttention` from `ruvector-attention`, since individual genotypes at a locus are exchangeable (no positional ordering of individuals).

2. **Variant interaction attention** (within each individual, across variants): Sparse attention using LD blocks, where variants only attend to others within the same LD block or to sentinel variants in other blocks.

```
LD block sparsity:  ~200-500 variants per LD block
Number of LD blocks: ~1,700 (European population, hapmap3)
Cross-block sentinels: ~10 per block

Density:  ~(500 + 10 * 1700) / 1M = ~2.2%
```

**Computational analysis:**

| Metric | Formula | Value (V=1M, P=500K, 2.2% density) |
|--------|---------|--------------------------------------|
| Locus attention (linear) | V * P * d | 1M * 500K * 128 = 64 TFLOPs |
| Variant interaction (sparse) | P * 2 * h * nnz_per_person * d_k | 500K * 2 * 16 * 22K * 16 = 5.6 TFLOPs |
| MoE routing | V * P * (d * num_experts + top_k * d * d) | ~8.5 TFLOPs |
| **Total Level 6** | | **~78 TFLOPs** |

---

## FPGA Acceleration

### Using `ruvector-fpga-transformer` for Clinical Genomics

Clinical genomic analysis imposes requirements absent from research settings:

| Requirement | Specification | Rationale |
|------------|--------------|-----------|
| Deterministic latency | p99 < 2x p50 | CAP/CLIA laboratory certification |
| Reproducibility | Bit-exact across runs | Diagnostic consistency |
| Auditability | Full inference trace | Regulatory compliance (FDA 21 CFR Part 11) |
| Turnaround time | <4 hours whole genome | Clinical actionability |

The `ruvector-fpga-transformer` crate's architecture is uniquely suited to these constraints.

### Dedicated Attention Computation Units

**Custom precision for nucleotide encodings.** The 4-letter DNA alphabet requires only 2 bits per nucleotide, but we use 4-bit encoding to accommodate the ambiguity code (N, R, Y, S, W, K, M, B, D, H, V) from IUPAC nomenclature:

```
FPGA Quantization Pipeline:
  Nucleotide (4-bit) --> Embedding Lookup (INT4 weights) --> Q/K/V Projection (INT8 matmul)
  --> Attention Scores (INT8 accumulate) --> Softmax (FP16) --> Output (INT8)
```

This maps to `ruvector-fpga-transformer`'s `QuantSpec`:

```rust
use ruvector_fpga_transformer::types::QuantSpec;

const NUCLEOTIDE_QUANT: QuantSpec = QuantSpec {
    input_bits: 4,      // 4-bit nucleotide encoding
    weight_bits: 4,     // INT4 embedding weights
    accumulator_bits: 8, // INT8 for Q*K^T accumulation
    output_bits: 8,     // INT8 output activations
};
```

**Streaming sequence processing.** The FPGA processes nucleotide windows in a streaming fashion without materializing the full genome in memory. Each Level 1 window (512 bp) is processed as a self-contained unit:

```
Genome Stream --> Window Extractor (512bp, stride 256bp)
             --> FPGA Attention Unit (pipelined Q*K^T + softmax + attn*V)
             --> Cross-Window Merger (overlap averaging)
             --> Level 1 Output Buffer

Pipeline depth:  3 stages (Q*K^T | softmax | attn*V)
Throughput:      1 window per pipeline fill time
Latency:         3 * (window_compute_time)
```

**Deterministic latency guarantees.** The `FixedShape` specification in `ruvector-fpga-transformer` enables compile-time latency bounds:

| Level | Shape | FPGA Latency (target) | Jitter bound |
|-------|-------|----------------------|--------------|
| L1 (nucleotide) | 512 x 128, 8 heads | 67 us | +/- 2 us |
| L2 (codon) | 168 x 128, 8 heads | 12 us | +/- 1 us |
| L3 (exon) | 363 x 256, 16 heads | 45 us | +/- 3 us |

Levels 4-6 are too sparse and irregular for fixed-shape FPGA execution; they use `ruvector-sparse-inference` on CPU/GPU instead.

**Witness logging for auditability.** Every FPGA inference produces a `WitnessLog` record containing the input hash, quantization parameters, attention weights (compressed), and output hash. This enables bit-exact replay for regulatory audits:

```rust
use ruvector_fpga_transformer::types::WitnessLog;

// Each window inference produces an auditable witness
let witness: WitnessLog = engine.infer(request)?
    .witness
    .expect("witness logging enabled for clinical mode");

// Witness contains:
// - input_hash: SHA-256 of nucleotide window
// - quant_spec: exact quantization parameters used
// - attention_checksum: compressed attention weight hash
// - output_hash: SHA-256 of output embeddings
// - timestamp_ns: nanosecond-precision timestamp
// - fpga_bitstream_hash: hardware configuration hash
```

---

## Sparse Inference

### Using `ruvector-sparse-inference` for Genomic Sparsity

#### Pruned Attention from Hi-C Contact Maps

Hi-C experiments produce genome-wide chromatin contact frequency matrices. These contact maps directly inform which genomic loci interact in 3D space and therefore which attention connections are biologically meaningful.

**Hi-C to attention mask conversion:**

```
Input:   Hi-C contact matrix H in R^{n x n} at resolution r (e.g., 10 kbp)
Process: 1. Normalize: H_norm = ICE(H) or KR(H)  (iterative correction)
         2. Expected: E(d) = mean(H_norm[i,j] for |i-j| = d)  (distance decay)
         3. O/E ratio: R(i,j) = H_norm(i,j) / E(|i-j|)
         4. Threshold: M(i,j) = 1 if R(i,j) > tau, 0 otherwise
Output:  Sparse attention mask M with density dependent on tau
```

The `SparseInferenceEngine`'s predictor learns to approximate this mask using a low-rank decomposition P * Q^T that can be evaluated in O(n * k) time where k is the rank (typically 32-64), much faster than computing the full Hi-C-derived mask:

```
Predicted mask: M_hat(i,j) = sigma(P_i^T * Q_j) > 0.5

    where P, Q in R^{n x k} are learned low-rank factors
    and sigma is the sigmoid function
```

#### Mixture-of-Experts for Tissue-Specific Analysis

Different cell types and tissues have distinct chromatin architectures and therefore different attention patterns. The `MoEAttention` from `ruvector-attention` routes genomic regions to tissue-specific expert subnetworks:

| Expert | Tissue Class | Characteristic Pattern |
|--------|-------------|----------------------|
| Expert 1 | Blood/Immune | High super-enhancer density, active V(D)J recombination loci |
| Expert 2 | Neural | Long-range enhancer-promoter interactions, large TADs |
| Expert 3 | Epithelial | Compact chromatin, tissue-specific enhancers |
| Expert 4 | Muscle | Super-enhancer clusters at myogenic loci |
| ... | ... | ... |
| Expert 53 | (all GTEx tissues) | Tissue-specific epigenomic signatures |

The routing decision is informed by epigenomic features at each locus:

```
Epigenomic feature vector:
  f_i = [H3K4me3 || H3K27ac || ATAC-seq || DNase-seq || H3K27me3 || CTCF || ...]

Expert routing:
  G(f_i) = TopK(softmax(W_route * f_i + b_route), k=4)
```

#### Dynamic Sparsity Based on Sequence Complexity

Not all genomic regions require the same computational investment. Repetitive elements (~45% of the human genome) and heterochromatic regions have low sequence complexity and limited regulatory activity, while gene-dense, euchromatic regions are information-rich.

We define a complexity score:

```
Complexity(window) = -sum_{k in {A,C,G,T}} p_k * log2(p_k)    (Shannon entropy)

    where p_k is the frequency of nucleotide k in the window
    Range: [0, 2] bits (0 = homopolymer, 2 = equimolar ACGT)
```

Dynamic sparsity adjustment:

| Entropy Range | Region Type | Sparsity Ratio | Attention Pattern |
|--------------|-------------|----------------|-------------------|
| [0, 0.5] | Simple repeats (Alu, LINE, SINE) | 5% density | Ultra-sparse, skip most |
| (0.5, 1.0] | Low-complexity (AT-rich, centromeric) | 15% density | Local window only |
| (1.0, 1.5] | Moderate (intergenic, intronic) | 30% density | Local + top Hi-C contacts |
| (1.5, 2.0] | High complexity (exonic, regulatory) | 60% density | Full hierarchical attention |

This adaptive scheme reduces total computation by approximately 2.5x compared to uniform sparsity, because ~45% of the genome has entropy < 1.0.

**Implementation via precision lanes.** The `PrecisionLane` system in `ruvector-sparse-inference` maps naturally to this dynamic sparsity:

```rust
use ruvector_sparse_inference::precision::{PrecisionLane, LaneConfig};

// Low-complexity regions: aggressive quantization, minimal attention
let repeat_lane = LaneConfig {
    precision: PrecisionLane::Bit3,
    sparsity_ratio: 0.05,
    // ...
};

// High-complexity regulatory regions: full precision, dense attention
let regulatory_lane = LaneConfig {
    precision: PrecisionLane::Bit7,
    sparsity_ratio: 0.60,
    // ...
};
```

---

## Memory Optimization

### Tiling Strategies per Hierarchical Level

The fundamental tradeoff at each level is between **storing intermediate results** (fast but memory-intensive) and **recomputing from lower levels** (slow but memory-efficient). The `FlashAttention` tiling strategy from `ruvector-attention` addresses this within a single level; here we extend it across levels.

#### Level 1 (Nucleotide): Full Tiling

```
Strategy:   Tile-and-discard
Tile size:  B_1 = 64 nucleotides (within 512bp window)
Store:      Only running statistics (max, sum_exp, partial output)
Discard:    Attention scores after each tile
Recompute:  Never (streaming, one-pass)

Memory per window:
  Embeddings:          512 * 128 * 4 = 256 KB
  Q, K, V:             3 * 512 * 128 * 4 = 768 KB
  Tile workspace:      64 * 512 * 4 = 128 KB  (one tile of scores)
  Running statistics:  512 * 4 * 3 = 6 KB  (max, sum_exp, output per row)
  Total:               ~1.15 MB per window
```

#### Level 2 (Codon): Full Materialization for Small Exons, Tiling for Large

```
Threshold:  n_codons < 2048 --> full materialization
            n_codons >= 2048 --> tiled flash attention

Full materialization memory (median exon, 168 codons):
  Attention matrix:   168 * 168 * 4 = 110 KB
  Embeddings + QKV:   168 * 128 * 4 * 4 = 344 KB
  Total:              ~454 KB

Tiled memory (TTN exon 363, 17,106 codons, B_2=128):
  Tile workspace:     128 * 17106 * 4 = 8.5 MB
  Embeddings + QKV:   17106 * 128 * 4 * 4 = 35 MB
  Total:              ~43.5 MB
```

#### Level 3 (Exon): Always Full Materialization

```
Strategy:   Full materialization (max 363 exons per gene)
Memory:     363 * 363 * 4 = 527 KB (attention matrix)
            363 * 256 * 4 * 4 = 1.4 MB (embeddings + QKV)
            Total: ~2 MB per gene

For all ~20,000 genes in parallel:
  Total: ~40 GB (fully parallel) or ~2 MB (sequential)
  Recommended: batch of 1,000 genes = ~2 GB
```

#### Level 4 (Gene/Regulatory): Sparse CSR Storage

```
Strategy:       Sparse CSR (Compressed Sparse Row) format
Non-zeros:      ~2.1 billion (2.3% of 300K x 300K)
CSR storage:
  values:       2.1B * 4 bytes = 8.4 GB
  col_indices:  2.1B * 4 bytes = 8.4 GB
  row_ptrs:     300K * 4 bytes = 1.2 MB
  Total:        ~16.8 GB

Tiling for GPU memory:
  Tile size:    10K rows x full columns
  Tiles:        30 tiles
  Memory per tile: ~560 MB
```

#### Level 5 (Chromosome): Ultra-Sparse COO Storage

```
Strategy:       COO (Coordinate) format for ultra-sparse pattern
Non-zeros:      ~95 million (0.1% of 308K x 308K)
COO storage:
  row_indices:  95M * 4 bytes = 380 MB
  col_indices:  95M * 4 bytes = 380 MB
  values:       95M * 4 bytes = 380 MB
  Total:        ~1.14 GB
```

#### Level 6 (Genome/GWAS): Distributed Block-Sparse

```
Strategy:       Block-sparse by LD block, distributed across nodes
Block size:     ~500 variants (one LD block)
Blocks:         ~1,700
Per-block dense attention:  500 * 500 * 4 = 1 MB
Cross-block sparse:         17K sentinels, ~0.01% density
Total per individual:       ~1.7 GB
For 500K individuals (distributed):
  Per node (1,000 individuals): ~1.7 TB
  Total cluster:                ~850 TB
  Recomputation strategy:      Store only LD block summaries (~3.4 GB per individual)
```

### Recomputation vs. Storage Tradeoffs Summary

| Level | Store | Recompute | Recommended | Reasoning |
|-------|-------|-----------|-------------|-----------|
| L1 | 1.15 MB/window | N/A (streaming) | Store | Already minimal |
| L2 | 454 KB/exon | 67 us/exon | Store | Cheap to store |
| L3 | 2 MB/gene | 45 us/gene | Store | Cheap to store |
| L4 | 16.8 GB genome | 1.08 PFLOPs | Store (sparse) | Recompute is expensive |
| L5 | 1.14 GB genome | 97.3 GFLOPs | Store | Cheap in memory and compute |
| L6 | 1.7 GB/person | 78 TFLOPs | Recompute (store summaries) | Storage dominates at population scale |

### Gradient Checkpointing for Training

During training (backpropagation), intermediate activations must be stored or recomputed for the backward pass. We apply selective gradient checkpointing:

```
Checkpoint boundary:  Between each hierarchical level
Store:                Level outputs (pooled representations)
Recompute:            Within-level attention during backward pass
Memory savings:       ~60% reduction in activation memory
Compute overhead:     ~33% increase in training FLOPs (one extra forward pass per level)
```

---

## Performance

### IO Complexity Analysis

The flash attention algorithm's key insight is that IO operations (reading/writing from HBM to SRAM) are the bottleneck, not arithmetic. For each level, we analyze IO complexity following the Dao et al. (2022) framework.

Let M = SRAM size, N = sequence length, d = head dimension.

**Standard attention IO complexity:** O(N * d + N^2)

**Flash attention IO complexity:** O(N^2 * d^2 / M)

For the genomic hierarchy:

| Level | N | d | M (SRAM) | Standard IO | Flash IO | Reduction |
|-------|---|---|----------|-------------|----------|-----------|
| L1 | 512 | 16 | 192 KB | 270 KB | 44 KB | 6.1x |
| L2 | 168 | 16 | 192 KB | 115 KB | 2.4 KB | 48x |
| L3 | 363 | 16 | 192 KB | 530 KB | 11 KB | 48x |
| L4 (sparse) | 300K | 16 | 192 KB | N/A (sparse) | N/A (sparse) | Use CSR scatter |
| L5 (sparse) | 308K | 32 | 192 KB | N/A (sparse) | N/A (sparse) | Use COO gather |

For sparse levels (L4, L5, L6), the IO complexity is:

```
IO_sparse = nnz * (sizeof(index) + sizeof(value)) + N * d * sizeof(float) * 3   (for Q, K, V)
```

### FLOP Counts Summary

| Level | FLOPs per Unit | Units | Total FLOPs (whole genome) |
|-------|---------------|-------|----------------------------|
| L1 (nucleotide) | 67.1 MFLOPs/window | ~12.4M windows | 838 TFLOPs |
| L2 (codon) | 7.2 MFLOPs/exon (median) | ~180,000 exons | 1.3 TFLOPs |
| L3 (exon) | 67.4 MFLOPs/gene (worst) | ~20,000 genes | 1.35 TFLOPs |
| L4 (regulatory) | 1.08 PFLOPs (genome-wide) | 1 | 1,080 TFLOPs |
| L5 (chromosome) | 97.3 GFLOPs (genome-wide) | 1 | 0.097 TFLOPs |
| L6 (GWAS) | 78 TFLOPs (per cohort) | 1 | 78 TFLOPs |
| **Total** | | | **~2,000 TFLOPs** |

### Latency Targets

| Level | Target Latency | Hardware | Parallelism |
|-------|---------------|----------|-------------|
| L1 | 67 us per window | FPGA | Pipeline windows |
| L1 (whole genome) | 14 minutes | FPGA array (64 units) | 64-way window parallel |
| L2 | 12 us per exon | FPGA | Parallel across exons |
| L2 (whole genome) | 2.2 seconds | FPGA array (64 units) | Exon-parallel |
| L3 | 45 us per gene | FPGA | Parallel across genes |
| L3 (whole genome) | 14 seconds | FPGA array (64 units) | Gene-parallel |
| L4 | 18 minutes | 8x GPU (A100) | Sparse row partitioning |
| L5 | 12 seconds | 1x GPU (A100) | Single GPU sufficient |
| L6 | 1.3 hours | 128-node cluster | Individual-parallel |
| **End-to-end (L1-L5)** | **~33 minutes** | **FPGA + GPU** | **Hierarchical pipeline** |
| **End-to-end (L1-L6, GWAS)** | **~1.8 hours** | **FPGA + GPU + cluster** | **Full pipeline** |

### Performance Comparison with Existing Models

| Model | Max Sequence | Time per Genome | Memory | This Architecture |
|-------|-------------|----------------|--------|-------------------|
| Enformer | 196,608 bp | N/A (cannot) | N/A | 33 min, 18 GB |
| HyenaDNA | 1M bp | ~days (extrapolated) | >1 TB | 33 min, 18 GB |
| DNABERT-2 | 512 bp | ~weeks (extrapolated) | >100 TB | 33 min, 18 GB |
| Evo | 131,072 bp | ~days (extrapolated) | >500 GB | 33 min, 18 GB |

The hierarchical approach achieves orders-of-magnitude speedup because it avoids O(n^2) scaling by decomposing the problem into biologically-motivated hierarchical levels.

---

## WASM Deployment

### Browser-Based Genomic Attention for Point-of-Care Diagnostics

The `ruvector-attention-unified-wasm` crate enables deploying the genomic attention hierarchy in web browsers, targeting point-of-care (PoC) diagnostic applications where: (a) patient data must not leave the clinical site (data sovereignty), (b) internet connectivity may be unreliable (rural clinics, field hospitals), (c) specialized hardware is unavailable, and (d) results are needed in minutes, not hours.

**WASM deployment scope.** Not all six levels are suitable for browser execution. We define a PoC diagnostic tier that runs entirely in-browser:

| Level | WASM Feasible | Rationale |
|-------|--------------|-----------|
| L1 (nucleotide) | Yes | Small windows, deterministic compute |
| L2 (codon) | Yes | Small per-exon, embarrassingly parallel |
| L3 (exon) | Yes | <2 MB per gene |
| L4 (regulatory) | Partial | Requires pre-computed sparse mask (~17 GB) -- stream from CDN |
| L5 (chromosome) | No | Requires full genome context |
| L6 (GWAS) | No | Requires population data |

**Target applications for WASM deployment:**

| Application | Levels Used | Input Size | Target Latency |
|------------|------------|------------|----------------|
| Single gene panel (50 genes) | L1, L2, L3 | ~1.35 Mbp | <5 seconds |
| Exome analysis (180K exons) | L1, L2, L3 | ~30 Mbp | <2 minutes |
| Targeted variant interpretation | L1, L2 | ~10 kbp per variant | <1 second |
| Splice site prediction | L1, L2, L3 | Per gene | <500 ms |

**WASM binary size optimization.** Following `ruvector-attention-unified-wasm`'s release profile:

```toml
[profile.release]
opt-level = "z"       # Optimize for size
lto = true            # Link-time optimization
codegen-units = 1     # Single codegen unit for better optimization
panic = "abort"       # No unwinding overhead
strip = true          # Strip debug symbols
```

Estimated WASM binary sizes:

| Component | Estimated Size | Notes |
|-----------|---------------|-------|
| L1 attention engine | ~180 KB | FlashAttention + nucleotide encoding |
| L2 codon attention | ~120 KB | Grouped query attention |
| L3 exon attention | ~150 KB | Flash attention + pooling |
| Model weights (50-gene panel) | ~2 MB | INT4 quantized |
| Total WASM bundle | ~2.5 MB | Gzipped: ~800 KB |

**Browser memory constraints.** WebAssembly has a linear memory model with a practical limit of ~2-4 GB in current browsers. Memory budgeting for a 50-gene panel:

```
WASM linear memory budget (2 GB):
  Model weights:      2 MB (INT4 quantized, 50 genes)
  Input sequences:    1.35 MB (50 genes, average 27 kbp each)
  L1 workspace:       1.15 MB (per window, reused)
  L2 workspace:       454 KB (per exon, reused)
  L3 workspace:       2 MB (per gene, reused)
  Output buffer:      10 MB (gene-level predictions)
  Overhead:           5 MB (allocator, stack, tables)
  Total:              ~22 MB

Headroom:             ~1.97 GB available for larger panels
Maximum feasible:     ~3,000 genes in 2 GB WASM memory
```

**Web Worker parallelism.** Each Web Worker gets its own WASM instance and memory. For multi-core utilization:

```javascript
// Spawn Web Workers for parallel gene processing
const numWorkers = navigator.hardwareConcurrency || 4;
const genesPerWorker = Math.ceil(totalGenes / numWorkers);

for (let i = 0; i < numWorkers; i++) {
    const worker = new Worker('genomic-attention-worker.js');
    worker.postMessage({
        genes: genePanel.slice(i * genesPerWorker, (i + 1) * genesPerWorker),
        wasmModule: compiledModule,  // SharedArrayBuffer for module sharing
    });
}
```

**Offline capability.** The WASM bundle + model weights can be cached via Service Worker for fully offline operation:

```javascript
// Service Worker caching strategy
self.addEventListener('install', event => {
    event.waitUntil(
        caches.open('genomic-attention-v1').then(cache => {
            return cache.addAll([
                '/genomic-attention.wasm',
                '/models/50-gene-panel.bin',
                '/models/exome-panel.bin',
            ]);
        })
    );
});
```

---

## Mathematical Appendix

### A. Formal Hierarchical Attention Formulation

Let G = {g_1, ..., g_N} be the genome sequence where g_i in {A, C, G, T, N}. Define the hierarchical representation:

```
Level 1:  h_i^{(1)} = NucleotideAttn(Embed(g_{i-w/2:i+w/2}))          for i in 1..N, stride s
Level 2:  h_j^{(2)} = CodonAttn(Pool_3(h_{3j:3j+2}^{(1)}))            for j in 1..N/3
Level 3:  h_k^{(3)} = ExonAttn(AttnPool({h_j^{(2)} : j in exon_k}))   for k in 1..E
Level 4:  h_m^{(4)} = RegulatoryAttn_sparse(Pool({h_k^{(3)} : k in element_m}))
Level 5:  h_p^{(5)} = ChromosomeAttn_sparse(BinPool(h^{(4)}, bin_size=10kbp))
Level 6:  h_q^{(6)} = GWASAttn_MoE(VariantEmbed(v_q), population_context)
```

### B. Memory Requirement Derivation

Total peak memory for whole-genome inference (Levels 1-5, single sample):

```
M_total = M_L1 + M_L2 + M_L3 + M_L4 + M_L5

M_L1 = num_concurrent_windows * (w * d * 4 * 4 + B_1 * w * 4)
     = 64 * (512 * 128 * 16 + 64 * 512 * 4)
     = 64 * (1,048,576 + 131,072)
     = 75.5 MB

M_L2 = num_concurrent_exons * n_codons * d * 4 * 4
     = 1000 * 168 * 128 * 16
     = 3.44 GB  (batch of 1000 exons)

M_L3 = num_concurrent_genes * n_exons * d * 4 * 4
     = 1000 * 9 * 256 * 16
     = 36.9 MB  (batch of 1000 genes, median)

M_L4 = nnz * 8 + n * d * 4 * 3
     = 2.1B * 8 + 300K * 256 * 12
     = 16.8 GB + 0.9 GB
     = 17.7 GB

M_L5 = nnz * 12 + n * d * 4 * 3
     = 95M * 12 + 308K * 512 * 12
     = 1.14 GB + 1.89 GB
     = 3.03 GB

M_total_peak = 75.5 MB + 3.44 GB + 36.9 MB + 17.7 GB + 3.03 GB
             = ~24.3 GB

    (Note: Levels are pipelined, so peak is max of concurrent levels,
     not sum. With proper scheduling: peak ~ 17.7 GB from Level 4.)
```

### C. FLOP Derivation for Flash Attention with Tiling

For flash attention with block size B on sequence length N and head dimension d:

```
Forward pass FLOPs:
  Q * K^T computation:     2 * N * N * d        (matmul)
  Softmax:                 5 * N * N             (exp, sum, div, max, sub)
  Attention * V:           2 * N * N * d        (matmul)
  Total forward:           4 * N^2 * d + 5 * N^2
                         = N^2 * (4d + 5)

For h heads:
  Total forward:           h * N^2 * (4d + 5)

IO operations (HBM reads/writes):
  Standard:                O(N * d + N^2)
  Flash (tiled):           O(N^2 * d^2 / M)

  where M = SRAM size in elements

Flash attention achieves the IO-optimal bound when:
  B_r = ceil(M / (4d))     (row tile size)
  B_c = min(ceil(M / (4d)), d)   (column tile size)

Number of tiles:           ceil(N / B_r) * ceil(N / B_c)
IO per tile:               B_r * d + B_c * d + B_r * B_c
Total IO:                  ceil(N/B_r) * ceil(N/B_c) * (B_r*d + B_c*d + B_r*B_c)
                         = O(N^2 * d^2 / M)   when B_r, B_c = Theta(M/d)
```

### D. Genomic Position Encoding

Standard sinusoidal or learned positional encodings are inadequate for genomic sequences because:
1. Genomic coordinates span 10^0 to 10^9 bp
2. Biologically meaningful distances are logarithmic (10 bp vs 10 kbp vs 10 Mbp)
3. Strand orientation matters (5' to 3' vs 3' to 5')

We define a multi-scale genomic position encoding:

```
PE_genomic(pos, strand) = concat(
    PE_fine(pos mod 1000),           // Sub-kilobase resolution
    PE_mid(floor(pos / 1000)),       // Kilobase resolution
    PE_coarse(floor(pos / 1000000)), // Megabase resolution
    PE_chrom(chromosome_id),         // Chromosome identity
    PE_strand(strand)                // Strand: +1 or -1
)

where PE_fine, PE_mid, PE_coarse use sinusoidal encoding:
  PE(p, 2i)   = sin(p / 10000^{2i/d_pe})
  PE(p, 2i+1) = cos(p / 10000^{2i/d_pe})

and PE_chrom is a learned embedding for the 24 chromosomes (1-22, X, Y).
```

---

## Consequences

### Positive

1. **Full-genome attention in 33 minutes** -- orders of magnitude faster than extrapolating existing flat-attention models, enabled by biologically-informed hierarchical decomposition.

2. **Single-nucleotide resolution** -- Level 1 preserves individual base-pair information while higher levels capture progressively longer-range interactions, supporting both SNV interpretation and structural variant detection.

3. **Clinical deployment readiness** -- FPGA acceleration via `ruvector-fpga-transformer` provides deterministic latency and witness logging required for CAP/CLIA certification and FDA 21 CFR Part 11 compliance.

4. **Population-scale GWAS** -- Level 6's MoE architecture with tissue-specific experts scales to biobank-sized cohorts (500K+ individuals) while maintaining biological interpretability.

5. **Offline point-of-care** -- WASM deployment enables gene panel analysis in <5 seconds in a web browser with no server dependency, suitable for resource-limited clinical settings.

6. **Reuse of existing RuVector primitives** -- All six levels build on existing crate APIs (`FlashAttention`, `LocalGlobalAttention`, `MoEAttention`, `SparseInferenceEngine`, `Engine`, WASM bindings), minimizing new code surface.

### Negative

1. **Hi-C data dependency** -- Levels 4 and 5 require pre-computed chromatin contact maps. Hi-C data is cell-type-specific and not available for all tissues. Mitigation: use predicted Hi-C from sequence-based models (e.g., Akita, Orca) as fallback.

2. **Training complexity** -- Six-level hierarchical training requires careful curriculum learning: Level 1 must converge before Level 2 can train meaningfully, and so on. Mitigation: pre-train each level independently, then fine-tune end-to-end with frozen lower levels.

3. **Annotation dependency** -- Exon boundaries (Level 3) and regulatory element catalogs (Level 4) depend on reference genome annotations that are incomplete, especially for non-coding regions and non-model organisms. Mitigation: use annotation-free modes with uniform binning at Levels 3-4.

4. **FPGA resource requirements** -- The 64-unit FPGA array for 14-minute whole-genome Level 1 processing represents significant hardware investment (~$200K-500K for Xilinx Alveo U250 cluster). Mitigation: GPU fallback path with ~4x slower but lower-cost execution.

5. **WASM precision limitations** -- WebAssembly SIMD (128-bit) provides only f32 precision, which may introduce numerical differences compared to native f64 paths for softmax computation. Mitigation: validate WASM outputs against native reference within clinically acceptable tolerance (correlation > 0.999).

---

## References

1. Dao, T., Fu, D.Y., Ermon, S., Rudra, A., & Re, C. (2022). FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness. NeurIPS 2022.
2. Avsec, Z. et al. (2021). Effective gene expression prediction from sequence by integrating long-range interactions. Nature Methods 18, 1196-1203. (Enformer)
3. Nguyen, E. et al. (2024). Sequence Modeling and Design from Molecular to Genome Scale with Evo. Science 386, 6723.
4. Zhou, J. et al. (2023). DNABERT-2: Efficient Foundation Model and Benchmark for Multi-Species Genome. ICLR 2024.
5. Nguyen, E. et al. (2023). HyenaDNA: Long-Range Genomic Sequence Modeling at Single Nucleotide Resolution. NeurIPS 2023.
6. Rao, S.S.P. et al. (2014). A 3D map of the human genome at kilobase resolution reveals principles of chromatin looping. Cell 159(7), 1665-1680.
7. ENCODE Project Consortium (2020). Expanded encyclopaedias of DNA elements in the human and mouse genomes. Nature 583, 699-710.
8. Buniello, A. et al. (2019). The NHGRI-EBI GWAS Catalog of published genome-wide association studies. Nucleic Acids Research 47(D1), D1005-D1012.
9. GTEx Consortium (2020). The GTEx Consortium atlas of genetic regulatory effects across human tissues. Science 369(6509), 1318-1330.
10. Lieberman-Aiden, E. et al. (2009). Comprehensive mapping of long-range interactions reveals folding principles of the human genome. Science 326(5950), 289-293.
