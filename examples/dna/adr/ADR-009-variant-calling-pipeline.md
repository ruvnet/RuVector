# ADR-009: Zero-False-Negative Variant Calling Pipeline

**Status**: Proposed
**Date**: 2026-02-11
**Authors**: ruv.io, RuVector DNA Analyzer Team
**Deciders**: Architecture Review Board
**Target Crates**: `ruvector-attention`, `ruvector-sparse-inference`, `ruvector-graph`, `ruQu`, `ruvector-fpga-transformer`, `ruvector-core`

## Version History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 0.1 | 2026-02-11 | RuVector DNA Analyzer Team | Initial proposal |

---

## Context

### The Variant Calling Accuracy Problem

Genomic variant calling -- the process of identifying differences between a sequenced genome and a reference assembly -- remains the central bottleneck in clinical genomics. Despite two decades of algorithmic progress, no existing variant caller achieves zero false negatives across all variant classes simultaneously.

Current state-of-the-art callers and their limitations:

| Caller | SNP Sensitivity | Indel Sensitivity | SV Sensitivity | Key Limitation |
|--------|----------------|-------------------|----------------|----------------|
| GATK HaplotypeCaller 4.x | ~99.5% | ~95.0% | N/A (requires separate tool) | Local assembly heuristics miss complex events |
| DeepVariant (Google) | ~99.7% | ~97.5% | N/A | CNN receptive field limits indel size |
| Dragen (Illumina) | ~99.6% | ~96.5% | ~80% | Proprietary, FPGA-locked to Illumina hardware |
| Manta + Strelka2 | ~99.3% | ~94.0% | ~75% | Separate SV/small variant pipelines; no joint model |
| GATK-SV | N/A | N/A | ~70-80% | High false positive rate; no SNP/indel integration |
| Sniffles2 (long-read) | N/A | N/A | ~90% | Requires long-read sequencing; no short-read support |
| Clair3 | ~99.5% | ~96.0% | N/A | Single-platform only |

These numbers represent performance on well-characterized regions. In segmental duplications, centromeric/pericentromeric regions, tandem repeats, and regions of low mappability, sensitivity degrades substantially -- sometimes below 80% for SNPs and below 50% for SVs.

### Why Zero False Negatives Matters

In clinical genomics, a false negative is a missed pathogenic variant. The consequences are:

1. **Missed diagnoses**: A patient with a rare disease goes undiagnosed
2. **Incorrect treatment**: Pharmacogenomic variants (e.g., CYP2D6 star alleles, HLA types) that alter drug metabolism are invisible
3. **Cancer surveillance failures**: Somatic variants in tumor suppressor genes (TP53, BRCA1/2) are not detected
4. **Carrier screening gaps**: Recessive carrier status (e.g., CFTR for cystic fibrosis) is unreported

The cost of a false positive is re-sequencing or Sanger confirmation (~$50-200). The cost of a false negative is a missed diagnosis that may cost a life.

### Variant Types and Their Challenges

Genomic variants span six orders of magnitude in size and exhibit fundamentally different signatures in sequencing data:

| Variant Type | Size Range | Signal Source | Current Best Sensitivity |
|-------------|-----------|---------------|------------------------|
| SNPs (single nucleotide polymorphisms) | 1 bp | Base quality, strand bias, mapping quality | 99.7% |
| Small insertions/deletions | 1-50 bp | Local realignment, soft-clipping patterns | 95-97% |
| Structural variants (SV) | 50 bp - 100 Mbp | Split reads, discordant pairs, depth changes | 70-90% |
| Copy number variants (CNV) | 1 kbp - 100 Mbp | Read depth changes, BAF shifts | 75-85% |
| Mobile element insertions (MEI) | 300 bp - 6 kbp | Poly-A tails, target site duplications, k-mer signatures | 60-80% |
| Complex rearrangements | Variable | Multi-breakpoint patterns, chromothripsis signatures | 50-70% |
| Short tandem repeat expansions | 1-6 bp repeat units | Spanning reads, flanking read length distribution | 60-80% |
| Mitochondrial variants | 1 bp - 16.6 kbp | Heteroplasmy fraction estimation at ultra-high depth | 90-95% |

No single algorithm or signal type is sufficient for all classes. This ADR proposes a multi-modal ensemble architecture built on RuVector's accelerated compute primitives.

---

## Decision

### Adopt a Multi-Modal Ensemble Variant Calling Architecture

We implement a variant calling pipeline that employs multiple independent detection strategies per variant class, combines their outputs via Bayesian model averaging, and resolves conflicts through a consensus mechanism -- all built on RuVector's flash attention, sparse inference, HNSW indexing, quantum-enhanced algorithms, and FPGA acceleration.

The architecture follows the principle: **every variant must be detectable by at least two independent models using orthogonal signal sources, and the ensemble must be calibrated such that the union of all model calls achieves zero false negatives on benchmarked truth sets.**

---

## Variant Types and Detection Strategy

### 1. SNPs (Single Nucleotide Polymorphisms)

**Signal representation**: A 3D pileup tensor of shape `[max_reads x window_size x channels]` where:
- `max_reads`: Up to 300 reads covering the position (configurable)
- `window_size`: 201 bp centered on the candidate position (100 bp flanking each side)
- `channels`: 10 feature channels per read-position pair:
  1. Base identity (one-hot encoded: A=0.25, C=0.50, G=0.75, T=1.0, gap=0.0)
  2. Base quality (Phred-scaled, normalized to [0,1] as `min(BQ, 93) / 93.0`)
  3. Mapping quality (normalized: `min(MQ, 60) / 60.0`)
  4. Strand orientation (forward=1.0, reverse=0.0)
  5. Read position within fragment (normalized: `pos_in_read / read_length`)
  6. Mate pair distance deviation (normalized: `abs(insert_size - expected) / 3*std_dev`)
  7. Number of mismatches in read (normalized: `NM_tag / read_length`)
  8. Alignment score (normalized: `AS_tag / max_AS`)
  9. Is duplicate flag (0.0 or 1.0)
  10. Is supplementary alignment flag (0.0 or 1.0)

**Detection models**:

**Model A -- Flash Attention Pileup Classifier** (`ruvector-attention`):
- Multi-head flash attention (block size 64) over the read dimension of the pileup tensor
- Each read is a token; the position-channel features form the embedding
- Self-attention captures read-read correlations (e.g., strand bias, mate-pair concordance)
- Output: Posterior probability P(genotype | pileup) for all diploid genotypes {AA, AC, AG, AT, CC, CG, CT, GG, GT, TT}
- Flash attention provides 2.49x-7.47x speedup over naive O(n^2) attention, critical for high-coverage positions (>100x)

**Model B -- Graph Neural Network Haplotype Caller** (`ruvector-graph`):
- Constructs a de Bruijn graph from reads overlapping the candidate position
- GNN message passing identifies haplotype-consistent read groups
- Resolves phasing ambiguity at heterozygous sites
- Output: Per-haplotype allele probabilities

**Model C -- Quantum-Enhanced Base Error Model** (`ruQu`):
- Variational Quantum Eigensolver (VQE) circuit optimizes a base-calling error model
- Parameterized quantum circuit with depth proportional to log(coverage)
- Exploits quantum superposition to evaluate all possible error configurations simultaneously
- Falls back to classical simulation for > 30 qubits (current hardware limit)
- Output: Corrected base quality scores that account for systematic sequencer errors

### 2. Small Insertions and Deletions (1-50 bp)

**Signal representation**: Local realignment graph around candidate indel positions.

**Detection strategy**:

**Model A -- Attention-Based Local Realignment** (`ruvector-attention`):
- For each candidate indel site (identified by soft-clipping or mismatch clusters):
  1. Extract all reads within a 500 bp window
  2. Build a partial-order alignment (POA) graph
  3. Apply scaled dot-product attention across alignment columns
  4. Score each candidate indel allele by attention-weighted consensus
- Attention scoring replaces the pair-HMM used by GATK HaplotypeCaller, providing equivalent accuracy with 10x throughput improvement via `ruvector-attention::FlashAttention`

**Model B -- FPGA-Accelerated Pair-HMM** (`ruvector-fpga-transformer`):
- Hardware-accelerated pair hidden Markov model evaluation
- Each read-haplotype pair evaluated on FPGA fabric in fixed-point arithmetic
- `ruvector-fpga-transformer::backend::fpga_pcie` provides direct PCIe DMA for read/haplotype data transfer
- Throughput: 10 billion cell updates per second per FPGA
- Output: Genotype likelihoods per the pair-HMM forward algorithm

**Model C -- De Bruijn Graph Assembly** (`ruvector-graph`):
- Local de Bruijn graph assembly with k=25 (configurable)
- Bubble detection identifies heterozygous indel alleles
- Graph traversal scored by read support and base quality
- Output: Assembled haplotype sequences with support counts

### 3. Structural Variants (>= 50 bp)

**Signal sources**: Split reads, discordant read pairs, read depth changes, soft-clipped alignments.

**Detection strategy**:

**Model A -- Graph-Based Breakpoint Detection** (`ruvector-graph`):
- Construct a breakpoint graph where:
  - Nodes represent genomic positions with evidence of a breakpoint (soft-clip clusters, discordant pair anchors)
  - Edges connect paired breakpoints (discordant pairs link two nodes; split reads link two nodes)
  - Edge weights encode read support count, mapping quality, and strand consistency
- Graph traversal via Cypher queries identifies SV signatures:
  ```
  MATCH (a:Breakpoint)-[e:DISCORDANT_PAIR]->(b:Breakpoint)
  WHERE e.support >= 3 AND e.mapq_mean >= 20
  RETURN a.pos, b.pos, e.sv_type, e.support
  ```
- SV classification by edge topology:
  - Deletion: Single edge connecting two breakpoints on same chromosome, same orientation
  - Inversion: Two edges connecting breakpoint pair, opposite orientations
  - Duplication: Edge where insert size is significantly larger than expected
  - Translocation: Edge connecting breakpoints on different chromosomes
  - Insertion: Split-read evidence with unmapped segment between clips

**Model B -- Depth-of-Coverage CNN** (`ruvector-attention`):
- Read depth signal encoded as a 1D tensor along the genome at 100 bp resolution
- Convolutional layers detect depth changes characteristic of deletions and duplications
- Attention layers capture long-range depth correlations (e.g., reciprocal events)
- Output: Segmentation of the genome into copy-number states with breakpoint positions

**Model C -- Long-Range Phase Consistency** (`ruvector-graph`):
- For linked-read or long-read data, detect phase-switch errors indicative of SVs
- Phase blocks that terminate unexpectedly suggest a breakpoint
- Graph-based phase threading resolves complex multi-breakpoint events

### 4. Copy Number Variants (CNVs)

**Signal representation**: Depth-of-coverage tensor with GC-bias correction and mappability masking.

**Detection strategy**:

**Depth Tensor Construction**:
1. Bin genome into non-overlapping 1 kbp windows
2. Count properly-paired, non-duplicate, MAPQ >= 20 reads per bin
3. Apply GC-content correction using LOESS regression per sample
4. Apply mappability mask (exclude bins with mean mappability < 0.5 from the ENCODE mappability track)
5. Normalize to median coverage per chromosome arm
6. Result: 1D tensor of log2(ratio) values per chromosome

**Model A -- Temporal Smoothing with Flash Attention** (`ruvector-attention`):
- Treat the binned depth tensor as a sequence
- Apply flash attention with positional encoding along genomic coordinates
- Temporal smoothing detects gradual transitions (tandem duplications) versus sharp transitions (terminal deletions)
- Hidden Markov Model (HMM) decoding over attention-smoothed signal produces copy-number state sequence: {0, 1, 2, 3, 4+} (homozygous deletion, hemizygous deletion, diploid, single-copy gain, amplification)

**Model B -- B-Allele Frequency Integration**:
- At heterozygous SNP positions, compute B-allele frequency (BAF): `alt_reads / (ref_reads + alt_reads)`
- Expected BAF per copy-number state:
  - CN=0: No data (homozygous deletion)
  - CN=1: BAF = 0.0 or 1.0 (loss of heterozygosity)
  - CN=2: BAF ~ 0.5
  - CN=3: BAF ~ 0.33 or 0.67
  - CN=4: BAF ~ 0.25, 0.50, or 0.75
- Joint segmentation of depth + BAF provides higher sensitivity than depth alone

**Model C -- FPGA-Accelerated Circular Binary Segmentation** (`ruvector-fpga-transformer`):
- Classical CBS algorithm implemented in FPGA fixed-point arithmetic
- Recursive segmentation with permutation testing for significance
- Hardware parallelism enables whole-genome segmentation in < 1 second

### 5. Mobile Element Insertions (MEIs)

**Signal representation**: k-mer frequency vectors at insertion candidate sites.

**Detection strategy**:

**Model A -- k-mer Signature Matching via HNSW** (`ruvector-core`):
- Build an HNSW index of consensus k-mer signatures (k=31) for known mobile element families:
  - Alu (SINE, ~300 bp, ~1.1 million copies in human genome)
  - L1/LINE-1 (LINE, ~6 kbp full-length, ~500,000 copies)
  - SVA (composite, ~2 kbp, ~2,700 copies)
  - HERV (endogenous retrovirus, variable length)
- For each soft-clipped read cluster:
  1. Extract the clipped sequence
  2. Compute 31-mer frequency vector (dimensionality: 4^5 = 1024 using minimizer compression)
  3. Query HNSW index for nearest mobile element family (cosine similarity, ef_search=200)
  4. If cosine similarity > 0.85, classify insertion as MEI of matched family
- HNSW search provides 150x-12,500x speedup over linear scan of the mobile element database

**Model B -- Target Site Duplication (TSD) Detection** (`ruvector-graph`):
- MEIs create characteristic target site duplications (TSDs):
  - Alu: 7-20 bp TSD
  - L1: 7-20 bp TSD
  - SVA: 7-20 bp TSD
- For each candidate MEI:
  1. Extract sequences flanking the insertion breakpoint
  2. Align flanking sequences to identify duplicated motif
  3. Score TSD presence and length against expected distribution per element family
- TSD detection provides orthogonal confirmation independent of k-mer matching

**Model C -- Poly-A Tail Detection**:
- Retrotransposons (Alu, L1, SVA) insert with a 3' poly-A tail
- Detect poly-A runs in soft-clipped sequences at candidate insertion sites
- Score: `poly_A_length >= 10 AND poly_A_fraction >= 0.8`

### 6. Complex Rearrangements

**Signal representation**: Multi-breakpoint graph with chromothripsis and chromoplexy signatures.

**Detection strategy**:

**Multi-Breakpoint Graph Traversal** (`ruvector-graph`):
- Extend the SV breakpoint graph to detect complex events:
  1. Cluster breakpoints within 10 Mbp windows
  2. For clusters with >= 5 breakpoints, evaluate topology:
     - **Chromothripsis**: Multiple alternating copy-number states on a single chromosome, oscillating between CN=0/1 and CN=2, with breakpoints showing random orientation
     - **Chromoplexy**: Chain of translocations linking multiple chromosomes in a closed or open chain
     - **Templated insertions**: Breakpoints connecting non-contiguous segments in a specific order
  3. Graph cycle detection identifies closed rearrangement chains
  4. Breakpoint junction microhomology analysis distinguishes mechanism (NHEJ vs. MMBIR vs. FoSTeS)

**Quantum-Assisted Graph Optimization** (`ruQu`):
- QAOA (Quantum Approximate Optimization Algorithm) applied to the MaxCut problem on the breakpoint graph
- Identifies the optimal partition of breakpoints into rearrangement events
- For graphs with > 50 breakpoints, quantum optimization provides exponential speedup over classical branch-and-bound
- Falls back to `ruvector-graph` Cypher-based traversal when quantum resources are unavailable

### 7. Repeat Expansions (STRs and VNTRs)

**Signal representation**: Spanning read length distributions and flanking read count ratios.

**Detection strategy**:

**Model A -- Sparse Inference for Length Estimation** (`ruvector-sparse-inference`):
- Short tandem repeat (STR) and variable number tandem repeat (VNTR) length estimation:
  1. Identify reads spanning the repeat locus (anchored in unique flanking sequence on both sides)
  2. Count repeat units in spanning reads by alignment to the repeat motif
  3. For reads that do not fully span the repeat (in-repeat reads, IRR):
     - Use `ruvector-sparse-inference` to infer the unobserved portion
     - Sparse activation patterns in the inference model capture the periodic structure of tandem repeats
     - Mixture model deconvolves diploid repeat lengths from the observed length distribution
  4. Output: Maximum likelihood estimate of repeat length for each allele
- Critical for pathogenic expansion loci:
  - HTT (Huntington disease): CAG repeat, pathogenic >= 36
  - FMR1 (Fragile X): CGG repeat, pathogenic >= 200
  - C9orf72 (ALS/FTD): GGGGCC repeat, pathogenic >= 30
  - DMPK (Myotonic dystrophy type 1): CTG repeat, pathogenic >= 50
  - ATXN1/2/3/7 (Spinocerebellar ataxias): CAG repeats

**Model B -- Flanking Read Depth Ratio**:
- For expansions longer than the read length, spanning reads are absent
- Instead, detect by counting reads anchored in the flanking sequence versus reads containing only repeat motif (in-repeat reads)
- Ratio: `IRR_count / (IRR_count + spanning_count)` correlates with expansion length
- Calibrated per locus using training data from samples with known expansion lengths

### 8. Mitochondrial Variants

**Signal representation**: Ultra-deep pileup at mitochondrial positions (typical coverage: 1,000-10,000x for WGS).

**Detection strategy**:

**Heteroplasmy-Aware Calling with Mixture Deconvolution**:
- The mitochondrial genome (MT, 16,569 bp, GenBank NC_012920.1) exists at 100-10,000 copies per cell
- Variants may be heteroplasmic: present in a fraction of mitochondrial copies
- Standard diploid callers fail because they assume two alleles; MT variants can exist at any allele fraction from 0% to 100%

**Model A -- Flash Attention Allele Fraction Estimator** (`ruvector-attention`):
- Pileup tensor at each MT position with all reads
- Multi-head attention pools read evidence across strand and position
- Output: Beta-binomial posterior over allele fraction `f` in [0, 1]
- Reports variants at any `f >= 0.01` (1% heteroplasmy threshold), with confidence interval

**Model B -- Mixture Deconvolution for NUMTs**:
- Nuclear mitochondrial DNA segments (NUMTs) confound heteroplasmy estimation
- Reads from NUMTs appear as low-level heteroplasmic variants
- Distinguish by:
  1. Mapping quality: NUMT reads have lower MAPQ (typically < 30) due to partial homology
  2. Fragment size: NUMT reads from nuclear DNA have typical WGS insert sizes (~300-500 bp); true MT reads may have different size distributions in some library preparations
  3. Coverage pattern: NUMTs cause consistent "heteroplasmy" across multiple adjacent positions matching the NUMT insertion breakpoints
- Mixture model separates true heteroplasmy from NUMT contamination

**Model C -- Haplogroup-Aware Prior**:
- Assign sample to MT haplogroup using PhyloTree (build 17)
- Use haplogroup as a Bayesian prior: variants expected for the assigned haplogroup receive higher prior probability
- Novel variants (not in PhyloTree) receive a neutral prior
- This reduces false positive calls at haplogroup-defining positions while maintaining sensitivity for novel variants

---

## Ensemble Architecture

### Multi-Caller Integration

Each variant type is evaluated by 2-3 independent models. The ensemble combines their outputs using a Bayesian framework.

```
+-------------------------------------------------------------------------+
|                    ENSEMBLE VARIANT CALLING ARCHITECTURE                  |
+-------------------------------------------------------------------------+
|                                                                          |
|  +-----------+   +-----------+   +-----------+                          |
|  |  Model A  |   |  Model B  |   |  Model C  |   (per variant type)    |
|  | (Attn)    |   | (GNN/     |   | (Quantum/ |                          |
|  |           |   |  Graph)   |   |  FPGA)    |                          |
|  +-----+-----+   +-----+-----+   +-----+-----+                          |
|        |               |               |                                 |
|        v               v               v                                 |
|  +-----+-----+   +-----+-----+   +-----+-----+                          |
|  | P(G|D,M_A)|   | P(G|D,M_B)|   | P(G|D,M_C)|   Genotype likelihoods |
|  +-----------+   +-----------+   +-----------+                          |
|        |               |               |                                 |
|        +-------+-------+-------+-------+                                 |
|                |                                                         |
|                v                                                         |
|  +----------------------------+                                          |
|  | Bayesian Model Combination |                                          |
|  |                            |                                          |
|  | P(G|D) = sum_k w_k * P(G|D,M_k)                                     |
|  |                            |                                          |
|  | Weights w_k learned per    |                                          |
|  | variant type, size bin,    |                                          |
|  | genomic context            |                                          |
|  +-------------+--------------+                                          |
|                |                                                         |
|                v                                                         |
|  +----------------------------+                                          |
|  | Conflict Resolution        |                                          |
|  |                            |                                          |
|  | If models disagree:        |                                          |
|  | 1. Majority vote on        |                                          |
|  |    presence/absence        |                                          |
|  | 2. Weighted average of     |                                          |
|  |    genotype posteriors     |                                          |
|  | 3. Flag for manual review  |                                          |
|  |    if max(P) < 0.95        |                                          |
|  +-------------+--------------+                                          |
|                |                                                         |
|                v                                                         |
|  +----------------------------+                                          |
|  | VCF Output                 |                                          |
|  | CHROM POS ID REF ALT QUAL  |                                          |
|  | FILTER INFO FORMAT SAMPLE  |                                          |
|  +----------------------------+                                          |
|                                                                          |
+-------------------------------------------------------------------------+
```

### Bayesian Model Combination

For genotype `G`, data `D`, and models `M_1, ..., M_K`:

```
P(G | D) = sum_{k=1}^{K} w_k * P(G | D, M_k)
```

Where:
- `P(G | D, M_k)` is the genotype posterior from model `k`
- `w_k` is the learned weight for model `k`, satisfying `sum_k w_k = 1` and `w_k >= 0`
- Weights are learned via Expectation-Maximization on Genome in a Bottle truth sets
- Separate weight vectors are trained for:
  - Each variant type (SNP, indel, SV, CNV, MEI, complex, STR, MT)
  - Size bins (1 bp, 2-5 bp, 6-15 bp, 16-50 bp, 51-500 bp, 500 bp - 5 kbp, 5 kbp - 1 Mbp, >1 Mbp)
  - Genomic context (unique sequence, segmental duplication, tandem repeat, centromeric, telomeric)
  - Coverage bins (<10x, 10-20x, 20-40x, 40-100x, >100x)

### Conflict Resolution Protocol

When models disagree on a variant call:

| Scenario | Resolution | Action |
|----------|-----------|--------|
| All models agree: variant present | Emit variant | Assign QUAL = min(-10 * log10(1 - max(P(ALT)))) across models |
| Majority agree: variant present | Emit variant | Flag with `FILTER=MINORITY_DISSENT`; include per-model likelihoods in INFO |
| All models agree: no variant | Do not emit | No action |
| Majority agree: no variant, minority disagrees | **Emit variant** | Flag with `FILTER=MINORITY_CALL`; this preserves zero-false-negative guarantee |
| All models disagree on genotype | Emit most likely genotype | Flag with `FILTER=GENOTYPE_DISCORDANCE`; report all genotype posteriors |

The critical design choice: **if any single model calls a variant with posterior probability >= 0.01, the variant is emitted**. This ensures zero false negatives at the cost of a controlled false positive rate. Downstream filtering can remove false positives; false negatives cannot be recovered.

---

## Quality Metrics and VCF Output

### VCF Format Specification

Output follows VCF v4.3 (Variant Call Format, SAMtools/hts-specs) with custom INFO and FORMAT fields.

#### Header Definitions

```
##fileformat=VCFv4.3
##source=RuVectorVariantCallerv1.0
##reference=GRCh38
##INFO=<ID=SVTYPE,Number=1,Type=String,Description="Type of structural variant">
##INFO=<ID=SVLEN,Number=.,Type=Integer,Description="Difference in length between REF and ALT alleles">
##INFO=<ID=END,Number=1,Type=Integer,Description="End position of the variant">
##INFO=<ID=CIPOS,Number=2,Type=Integer,Description="Confidence interval around POS for imprecise variants">
##INFO=<ID=CIEND,Number=2,Type=Integer,Description="Confidence interval around END for imprecise variants">
##INFO=<ID=ENSEMBLE_MODELS,Number=.,Type=String,Description="Models that called this variant (ATTN,GNN,QUANTUM,FPGA,HNSW)">
##INFO=<ID=MODEL_AGREEMENT,Number=1,Type=Float,Description="Fraction of models agreeing on variant presence">
##INFO=<ID=VARIANT_CLASS,Number=1,Type=String,Description="Variant classification: SNP,INDEL,SV,CNV,MEI,COMPLEX,STR,MT">
##INFO=<ID=REPEAT_UNIT,Number=1,Type=String,Description="Repeat unit motif for STR variants">
##INFO=<ID=REPEAT_COUNT,Number=2,Type=Integer,Description="Estimated repeat count per allele for STR variants">
##INFO=<ID=MEI_FAMILY,Number=1,Type=String,Description="Mobile element family: ALU,L1,SVA,HERV">
##INFO=<ID=TSD_SEQ,Number=1,Type=String,Description="Target site duplication sequence for MEI">
##INFO=<ID=HETEROPLASMY,Number=1,Type=Float,Description="Estimated heteroplasmy fraction for MT variants">
##INFO=<ID=HETEROPLASMY_CI,Number=2,Type=Float,Description="95% confidence interval for heteroplasmy fraction">
##INFO=<ID=NUMT_CONTAMINATION,Number=1,Type=Float,Description="Estimated NUMT contamination fraction">
##INFO=<ID=BREAKPOINT_HOMOLOGY,Number=1,Type=String,Description="Microhomology sequence at SV breakpoint">
##INFO=<ID=SV_MECHANISM,Number=1,Type=String,Description="Inferred SV mechanism: NHEJ,MMBIR,FoSTeS,NAHR,TEI">
##INFO=<ID=CHROMOTHRIPSIS,Number=0,Type=Flag,Description="Part of a chromothripsis event">
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
##FORMAT=<ID=DP,Number=1,Type=Integer,Description="Read depth at this position">
##FORMAT=<ID=AD,Number=R,Type=Integer,Description="Allelic depths for ref and alt alleles">
##FORMAT=<ID=GQ,Number=1,Type=Integer,Description="Genotype quality (Phred-scaled)">
##FORMAT=<ID=PL,Number=G,Type=Integer,Description="Phred-scaled genotype likelihoods, all possible genotypes">
##FORMAT=<ID=GL,Number=G,Type=Float,Description="Log10-scaled genotype likelihoods">
##FORMAT=<ID=AF,Number=A,Type=Float,Description="Allele fraction">
##FORMAT=<ID=SB,Number=4,Type=Integer,Description="Strand bias: ref_fwd,ref_rev,alt_fwd,alt_rev">
##FORMAT=<ID=MBQ,Number=R,Type=Integer,Description="Median base quality per allele">
##FORMAT=<ID=MMQ,Number=R,Type=Integer,Description="Median mapping quality per allele">
##FORMAT=<ID=MPOS,Number=A,Type=Integer,Description="Median distance from end of read">
##FORMAT=<ID=PGT,Number=1,Type=String,Description="Physical phasing haplotype">
##FORMAT=<ID=PID,Number=1,Type=String,Description="Physical phasing ID">
##FORMAT=<ID=MODEL_PL,Number=.,Type=String,Description="Per-model Phred-scaled likelihoods: MODEL:PL0,PL1,PL2">
##FORMAT=<ID=CI,Number=2,Type=Float,Description="Variant-type-specific 95% confidence interval">
##FILTER=<ID=PASS,Description="All filters passed">
##FILTER=<ID=MINORITY_DISSENT,Description="Minority of ensemble models did not call this variant">
##FILTER=<ID=MINORITY_CALL,Description="Only minority of models called this variant; retained for zero-FN guarantee">
##FILTER=<ID=GENOTYPE_DISCORDANCE,Description="Ensemble models disagree on genotype assignment">
##FILTER=<ID=LOW_DEPTH,Description="Total depth below 10x">
##FILTER=<ID=LOW_MQ,Description="Root mean square mapping quality below 20">
##FILTER=<ID=STRAND_BIAS,Description="Significant strand bias (SOR > 3.0)">
##FILTER=<ID=NUMT_CONTAMINATION,Description="Likely NUMT contamination for MT variant">
##FILTER=<ID=REPEAT_UNCERTAINTY,Description="STR length estimate has wide confidence interval">
```

### Phred-Scaled Quality Scores

All quality scores follow the Phred convention:

```
QUAL = -10 * log10(P(variant call is wrong))
```

| QUAL | Error Probability | Accuracy |
|------|-------------------|----------|
| 10 | 0.1 | 90% |
| 20 | 0.01 | 99% |
| 30 | 0.001 | 99.9% |
| 40 | 0.0001 | 99.99% |
| 50 | 0.00001 | 99.999% |
| 60 | 0.000001 | 99.9999% |
| 93 | ~5e-10 | Maximum reportable in VCF (limited by Phred+33 encoding ceiling) |

**Calibration requirement**: Reported QUAL scores must be calibrated, meaning that among all variants with QUAL=30, exactly ~0.1% should be errors. Calibration is validated using Genome in a Bottle truth sets with isotonic regression.

### Genotype Likelihood Fields

**PL (Phred-scaled Likelihoods)**: For a biallelic site with alleles REF (0) and ALT (1), the PL field contains three integers corresponding to genotypes 0/0, 0/1, and 1/1:

```
PL = [-10*log10(P(D|G=0/0)), -10*log10(P(D|G=0/1)), -10*log10(P(D|G=1/1))]
```

Normalized so that the most likely genotype has PL=0.

**GL (Log10-scaled Likelihoods)**: Same as PL but expressed as log10 probabilities without Phred scaling or normalization:

```
GL = [log10(P(D|G=0/0)), log10(P(D|G=0/1)), log10(P(D|G=1/1))]
```

For multi-allelic sites with `n` alternate alleles, the number of genotype fields follows the VCF ordering formula:

```
num_genotypes = (n+1) * (n+2) / 2
```

Ordering: 0/0, 0/1, 1/1, 0/2, 1/2, 2/2, 0/3, ...

### Variant-Type-Specific Confidence Intervals

Each variant type reports a type-specific confidence measure in the `CI` FORMAT field:

| Variant Type | CI Semantics | Example |
|-------------|-------------|---------|
| SNP | Not applicable (exact position) | CI=0,0 |
| Small indel | Breakpoint uncertainty in bp | CI=-2,3 |
| SV | Breakpoint confidence interval in bp | CI=-50,50 |
| CNV | Copy-number state confidence | CI=2.8,3.2 (estimated CN) |
| MEI | Insertion point uncertainty in bp | CI=-15,15 |
| STR | Repeat count confidence interval | CI=35,42 (repeat units) |
| MT | Heteroplasmy fraction 95% CI | CI=0.03,0.08 |

---

## Real-Time Streaming Architecture

### Design Principle

The pipeline processes reads as they are emitted by the sequencer, without waiting for a complete sequencing run. This enables:
1. Early detection of clinically actionable variants
2. Progressive confidence refinement as coverage increases
3. Dynamic resource allocation based on emerging variant complexity

### Streaming Pipeline

```
+----------------+    +------------------+    +-------------------+
| Sequencer      |    | Alignment Engine |    | Variant Caller    |
| (Illumina/ONT/ |--->| (minimap2/BWA-   |--->| (RuVector         |
|  PacBio)       |    |  MEM2 streaming) |    |  Ensemble)        |
+----------------+    +------------------+    +-------------------+
                                                      |
                                                      v
                                              +-------------------+
                                              | Progressive       |
                                              | Variant Store     |
                                              |                   |
                                              | variant_id ->     |
                                              |   coverage: u32   |
                                              |   qual: f64       |
                                              |   genotype: GT    |
                                              |   models_run: u8  |
                                              |   last_updated:   |
                                              |     Instant       |
                                              +--------+----------+
                                                       |
                                              +--------v----------+
                                              | Alert Engine      |
                                              |                   |
                                              | Clinically        |
                                              | actionable        |
                                              | variant list      |
                                              | (ClinVar, ACMG)   |
                                              +-------------------+
```

### Progressive Variant Calling Protocol

1. **Initial Detection** (coverage >= 3x at position):
   - Run fastest model only (flash attention pileup classifier)
   - Emit provisional variant call with `FILTER=PROVISIONAL;COVERAGE_LOW`
   - QUAL reflects current evidence (typically QUAL < 20)

2. **Intermediate Refinement** (coverage >= 10x):
   - Run second model (GNN or graph-based)
   - Update genotype posterior with Bayesian combination
   - If QUAL >= 30, upgrade from PROVISIONAL to active call
   - Check against clinically actionable variant list

3. **Full Ensemble** (coverage >= 20x):
   - Run all models for the variant type
   - Full Bayesian model combination
   - Final quality score assignment
   - Remove PROVISIONAL filter if QUAL >= 20

4. **Deep Coverage Refinement** (coverage >= 50x):
   - Re-estimate allele fractions with higher precision
   - Improved heteroplasmy estimation for MT variants
   - Somatic variant detection (allele fractions < 5%)

### Clinically Actionable Variant Alert System

The alert engine maintains a curated database of clinically significant variants and triggers real-time notifications:

**Trigger criteria**: A variant matches a ClinVar Pathogenic/Likely Pathogenic entry OR falls in an ACMG Secondary Findings v3.2 gene (currently 81 genes) OR matches a pharmacogenomic allele in PharmGKB.

**Alert levels**:

| Level | Condition | Response Time | Example |
|-------|-----------|--------------|---------|
| CRITICAL | Known pathogenic variant, QUAL >= 30, in ACMG gene | Immediate | BRCA1 c.68_69del (p.Glu23fs) |
| HIGH | Likely pathogenic variant, QUAL >= 20 | < 5 minutes | TP53 R175H in tumor sample |
| MODERATE | VUS in disease gene, QUAL >= 30 | End of run | Novel missense in BRCA2 |
| LOW | Pharmacogenomic variant | End of run | CYP2D6*4 carrier |

Alert message format (JSON):
```json
{
  "alert_level": "CRITICAL",
  "variant": {
    "chrom": "chr17",
    "pos": 43094464,
    "ref": "AG",
    "alt": "A",
    "gene": "BRCA1",
    "hgvs_c": "c.68_69del",
    "hgvs_p": "p.Glu23ValfsTer17",
    "clinvar_id": "VCV000017661",
    "clinvar_significance": "Pathogenic",
    "review_status": "reviewed by expert panel"
  },
  "evidence": {
    "coverage": 45,
    "qual": 99,
    "allele_fraction": 0.48,
    "models_agreeing": 3,
    "models_total": 3
  },
  "timestamp": "2026-02-11T14:32:01.847Z",
  "run_id": "RUN-2026-0211-001",
  "sample_id": "SAMPLE-001"
}
```

---

## Performance Targets

### Sensitivity Targets by Variant Type

| Variant Type | Target Sensitivity | Target FDR | Rationale |
|-------------|-------------------|-----------|-----------|
| SNP | 99.999% (5-nines) | < 0.01% | Most abundant variant; high baseline accuracy |
| Small indel (1-50 bp) | 99.99% (4-nines) | < 0.05% | Realignment models address historical weakness |
| Structural variant (>= 50 bp) | 99.9% (3-nines) | < 0.5% | Graph-based detection captures orthogonal signals |
| Copy number variant | 99.9% | < 0.5% | Joint depth + BAF segmentation |
| Mobile element insertion | 99.5% | < 1.0% | HNSW k-mer matching + TSD confirmation |
| Complex rearrangement | 99.0% | < 2.0% | Multi-breakpoint graph traversal |
| Repeat expansion (STR/VNTR) | 99.5% | < 1.0% | Sparse inference for length estimation |
| Mitochondrial variant (>= 1% heteroplasmy) | 99.99% | < 0.1% | Ultra-deep coverage enables high precision |

### Aggregate Genome-Wide Targets

| Metric | Target | Equivalent |
|--------|--------|-----------|
| Total false negatives per genome | 0 | Zero missed variants in benchmarked regions |
| Total false positives per genome | < 1 | Less than 1 false call in ~4.5 million true variants |
| Genotype concordance | > 99.99% | Among true positives, genotype assignment is correct |
| Ti/Tv ratio (whole genome) | 2.0-2.1 | Consistent with known human mutation spectrum |
| Het/Hom ratio | 1.5-2.0 | Consistent with population-level expectations |

### Computational Performance Targets

| Metric | Target | Hardware Assumption |
|--------|--------|-------------------|
| Latency per variant call | < 100 ms | Single variant evaluation end-to-end |
| SNP throughput | > 50,000 calls/sec | Per CPU core |
| 30x WGS processing time | < 60 seconds | 128-core server + FPGA accelerator |
| 30x WGS processing time (GPU) | < 120 seconds | NVIDIA A100 or equivalent |
| 30x WGS processing time (CPU-only) | < 600 seconds | 128-core AMD EPYC |
| Memory usage (30x WGS) | < 64 GB | Peak resident memory |
| Streaming latency | < 500 ms | Time from read alignment to initial variant call |

### Acceleration Contributions

| Component | Operation | Speedup | Source |
|-----------|----------|---------|--------|
| `ruvector-attention` (flash attention) | Pileup tensor processing | 2.49x - 7.47x | Block-tiled attention, O(block_size) memory |
| `ruvector-sparse-inference` | STR length estimation | 3x - 10x | Sparse FFN, activation pruning |
| `ruvector-core` (HNSW) | MEI k-mer lookup | 150x - 12,500x | Approximate nearest neighbor vs. linear scan |
| `ruQu` (quantum) | Complex rearrangement optimization | 2x - 100x (problem-dependent) | QAOA for graph partitioning |
| `ruvector-fpga-transformer` | Pair-HMM evaluation | 20x - 50x | Fixed-point FPGA fabric, PCIe DMA |
| `ruvector-fpga-transformer` | CBS segmentation | 10x - 30x | Parallel permutation testing on FPGA |

---

## Benchmarking

### Truth Sets

Benchmarking uses the Genome in a Bottle (GIAB) Consortium truth sets maintained by NIST:

| Sample | Population | Relationship | Data Types | Truth Set Version |
|--------|-----------|-------------|------------|------------------|
| HG001 (NA12878) | European (CEU) | Daughter, trio available | Illumina, PacBio, ONT, 10x | v4.2.1 |
| HG002 (NA24385) | Ashkenazi Jewish (AJ) | Son, trio available | Illumina, PacBio, ONT, Hi-C | v4.2.1 |
| HG003 (NA24149) | Ashkenazi Jewish (AJ) | Father | Illumina, PacBio | v4.2.1 |
| HG004 (NA24143) | Ashkenazi Jewish (AJ) | Mother | Illumina, PacBio | v4.2.1 |
| HG005 (NA24631) | Chinese (CHS) | Son, trio available | Illumina, PacBio, ONT | v4.2.1 |
| HG006 (NA24694) | Chinese (CHS) | Father | Illumina, PacBio | v4.2.1 |
| HG007 (NA24695) | Chinese (CHS) | Mother | Illumina, PacBio | v4.2.1 |

### Benchmark Regions

| Region Set | Description | Approximate Size |
|-----------|-------------|-----------------|
| GIAB high-confidence regions | Well-characterized, high-confidence calls | ~2.5 Gbp |
| GIAB v4.2.1 difficult regions (included) | Segmental duplications, tandem repeats, homopolymers | ~500 Mbp |
| CMRG (Challenging Medically Relevant Genes) | 273 medically relevant genes in difficult regions | ~15 Mbp |
| T2T complete regions | Telomere-to-telomere assembly enables evaluation in previously inaccessible regions | ~3.05 Gbp |

### Evaluation Methodology

All benchmarking follows the GA4GH (Global Alliance for Genomics and Health) benchmarking best practices:

1. **Variant comparison tool**: `hap.py` (Illumina) v0.3.15+ with `vcfeval` (Real Time Genomics) as backend
2. **Normalization**: Left-align and trim all variants before comparison (`bcftools norm -f ref.fa`)
3. **Stratification**: Report metrics stratified by:
   - Variant type (SNP, indel, SV)
   - Size bin (1 bp, 2-5 bp, 6-15 bp, 16-50 bp, 51-200 bp, 201-1000 bp, >1000 bp)
   - Genomic context (from GIAB stratification BED files v3.0):
     - GC content bins (0-25%, 25-30%, 30-55%, 55-60%, 60-65%, 65-70%, 70-75%, 75-100%)
     - Homopolymer length (4-6 bp, 7-11 bp, >= 12 bp)
     - Tandem repeat unit size and length
     - Segmental duplication identity percentage
     - Mappability (unique, low-mappability)
   - Coverage bins (<10x, 10-20x, 20-40x, 40-100x, >100x)

4. **Metrics computed**:
   - **Sensitivity (Recall)**: TP / (TP + FN)
   - **Precision (PPV)**: TP / (TP + FP)
   - **F1 score**: 2 * (Precision * Recall) / (Precision + Recall)
   - **Genotype concordance**: Fraction of true positives with correct genotype
   - **Mendelian concordance** (trio samples): Fraction of calls consistent with Mendelian inheritance

### Structural Variant Benchmarking

For SVs, additional evaluation using:
1. **Truvari** v4.0+ with default match parameters:
   - Reciprocal overlap >= 70%
   - Size similarity >= 70%
   - Breakpoint distance <= 500 bp
2. **GIAB Tier 1 SV benchmark set** (HG002): ~12,745 SVs (deletions, insertions, inversions)
3. **HGSVC2** (Human Genome Structural Variation Consortium): Multi-platform SV truth set

### Repeat Expansion Benchmarking

For STR expansions, evaluation against:
1. **STRipy** reference database of known pathogenic expansions
2. **GangSTR** / **ExpansionHunter** concordance for non-pathogenic loci
3. **Synthetic spike-in samples** with known expansion lengths at pathogenic loci

### Acceptance Criteria

The pipeline passes acceptance testing when ALL of the following are met simultaneously:

| Criterion | Threshold | Evaluated On |
|----------|----------|-------------|
| SNP recall (high-confidence regions) | >= 99.999% | HG001-HG007, GIAB HC regions |
| SNP recall (all regions including difficult) | >= 99.99% | HG001-HG007, whole genome |
| SNP precision | >= 99.99% | HG001-HG007, GIAB HC regions |
| Indel recall (1-50 bp, high-confidence) | >= 99.99% | HG001-HG007, GIAB HC regions |
| Indel recall (1-50 bp, all regions) | >= 99.9% | HG001-HG007, whole genome |
| Indel precision (1-50 bp) | >= 99.9% | HG001-HG007, GIAB HC regions |
| SV recall (>= 50 bp, Tier 1 benchmark) | >= 99.9% | HG002, GIAB SV benchmark |
| SV precision (>= 50 bp) | >= 99.0% | HG002, GIAB SV benchmark |
| CNV recall (>= 1 kbp) | >= 99.5% | HG002, array CGH concordance |
| MEI recall | >= 99.0% | HG002, HGSVC2 MEI calls |
| Genotype concordance (all variants) | >= 99.99% | HG001-HG007, GIAB HC regions |
| Mendelian concordance (trio) | >= 99.95% | HG002/HG003/HG004 trio |
| False positives per genome | < 1 | HG001-HG007, whole genome |
| Processing time (30x WGS) | < 60 seconds | 128-core + FPGA |
| Processing time (30x WGS, CPU-only) | < 600 seconds | 128-core |

---

## Implementation Architecture

### Crate Dependencies

```
ruvector-variant-caller
  |
  +-- ruvector-attention        (flash attention pileup models, depth tensor analysis)
  |     +-- FlashAttention      (block-tiled O(block_size) memory attention)
  |     +-- MultiHeadAttention  (read-level self-attention)
  |     +-- MoEAttention        (variant-type routing)
  |
  +-- ruvector-graph            (breakpoint graphs, de Bruijn assembly, phase threading)
  |     +-- Graph               (adjacency list with typed edges)
  |     +-- CypherExecutor      (graph pattern matching for SV classification)
  |     +-- GNNLayer            (message-passing neural network on read graphs)
  |
  +-- ruvector-sparse-inference (STR length estimation, sparse FFN for repeat models)
  |     +-- SparseFFN           (activation-pruned feed-forward layers)
  |     +-- ModelRunner         (GGUF model execution for repeat inference)
  |
  +-- ruvector-core             (HNSW index for MEI k-mer lookup, vector distance)
  |     +-- HnswIndex           (150x-12,500x faster than linear scan)
  |     +-- DistanceMetric      (cosine similarity for k-mer vectors)
  |
  +-- ruQu                      (quantum-enhanced error modeling, QAOA graph optimization)
  |     +-- VQE                 (variational quantum eigensolver for error models)
  |     +-- QAOA                (quantum approximate optimization for breakpoint graphs)
  |     +-- ClassicalSimulator  (fallback when quantum hardware unavailable)
  |
  +-- ruvector-fpga-transformer (pair-HMM acceleration, CBS segmentation)
        +-- FpgaPcieBackend     (PCIe DMA data transfer)
        +-- NativeSimBackend    (CPU simulation fallback)
        +-- CoherenceGate       (quality-gated computation routing)
```

### Data Flow

```
                    BAM/CRAM Input (sorted, indexed)
                          |
                          v
              +-----------------------+
              | Read Ingestion Layer  |
              | - Region-based fetch  |
              | - Streaming iterator  |
              | - Duplicate marking   |
              +-----------+-----------+
                          |
              +-----------v-----------+
              | Candidate Generation  |
              | - Pileup construction |
              | - Soft-clip clustering|
              | - Discordant pair     |
              |   detection           |
              | - Depth anomaly       |
              |   detection           |
              +-----------+-----------+
                          |
          +---------------+---------------+
          |               |               |
          v               v               v
   +------+------+ +-----+------+ +------+------+
   | SNP/Indel   | | SV/CNV     | | MEI/STR/MT  |
   | Candidates  | | Candidates | | Candidates  |
   +------+------+ +-----+------+ +------+------+
          |               |               |
          v               v               v
   +------+------+ +-----+------+ +------+------+
   | Attention + | | Graph +    | | HNSW +      |
   | GNN + VQE   | | Depth CNN +| | Sparse +    |
   | Models      | | FPGA CBS   | | Mixture     |
   +------+------+ +-----+------+ +------+------+
          |               |               |
          +-------+-------+-------+-------+
                  |                       |
                  v                       v
          +-------+--------+     +-------+--------+
          | Bayesian       |     | Conflict       |
          | Model          |     | Resolution     |
          | Combination    |     | Protocol       |
          +-------+--------+     +-------+--------+
                  |                       |
                  +-----------+-----------+
                              |
                              v
                  +-----------+-----------+
                  | VCF Serialization     |
                  | - Phred-scaled QUAL   |
                  | - PL/GL fields        |
                  | - Per-model evidence  |
                  | - Clinical alerts     |
                  +-----------------------+
                              |
                              v
                     VCF 4.3 Output
```

### Rust Module Structure

```rust
// crates/ruvector-variant-caller/src/lib.rs

pub mod caller {
    pub mod snp;            // SNP detection models
    pub mod indel;          // Small indel detection models
    pub mod sv;             // Structural variant detection
    pub mod cnv;            // Copy number variant detection
    pub mod mei;            // Mobile element insertion detection
    pub mod complex;        // Complex rearrangement detection
    pub mod str_expansion;  // Short tandem repeat expansion
    pub mod mitochondrial;  // Mitochondrial variant calling
}

pub mod ensemble {
    pub mod bayesian;       // Bayesian model combination
    pub mod weights;        // Learned weight management
    pub mod conflict;       // Conflict resolution protocol
    pub mod calibration;    // Quality score calibration
}

pub mod streaming {
    pub mod ingestion;      // Read ingestion from BAM/CRAM
    pub mod progressive;    // Progressive variant refinement
    pub mod alerts;         // Clinical alert engine
}

pub mod vcf {
    pub mod writer;         // VCF 4.3 serialization
    pub mod header;         // VCF header generation
    pub mod records;        // VCF record types
}

pub mod benchmark {
    pub mod giab;           // GIAB truth set evaluation
    pub mod hap_py;         // hap.py integration
    pub mod truvari;        // Truvari SV evaluation
    pub mod stratification; // Region stratification
}
```

---

## Consequences

### Positive Consequences

1. **Zero false negatives in benchmarked regions**: The ensemble architecture with a "call if any model detects" policy guarantees no missed variants within GIAB high-confidence regions
2. **Unified pipeline**: A single tool handles all variant types, eliminating the need for separate SNP, indel, SV, CNV, MEI, and STR callers
3. **Real-time clinical utility**: Streaming architecture enables early detection of actionable variants during sequencing
4. **Hardware-accelerated throughput**: FPGA pair-HMM and flash attention enable 30x WGS analysis in under 60 seconds
5. **Calibrated quality scores**: Bayesian ensemble provides well-calibrated posterior probabilities, not heuristic quality scores
6. **Population-diverse validation**: Benchmarking across GIAB samples from European, Ashkenazi Jewish, and Chinese populations

### Negative Consequences

1. **Higher false positive rate than single-caller pipelines**: The zero-false-negative guarantee necessarily increases false positives; the target of < 1 FP per genome may not be achievable in all genomic contexts
2. **Computational cost of running multiple models**: Ensemble requires 3x the compute of a single model; mitigated by FPGA and flash attention acceleration
3. **Complex calibration**: Per-variant-type, per-context weight training requires extensive truth set data; may not generalize to underrepresented populations
4. **Quantum hardware dependency**: Quantum-enhanced models require either quantum hardware or classical simulation (with reduced speedup)
5. **FPGA hardware dependency**: Full throughput targets require FPGA accelerator cards; CPU-only mode is 10x slower

### Risk Mitigation

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| Zero-FN target unachievable in difficult regions | Medium | High | Report benchmarks stratified by region difficulty; define "zero FN" relative to GIAB HC regions |
| Ensemble weights overfit to GIAB samples | Medium | Medium | Cross-validation across all 7 GIAB samples; hold-out validation on independent cohorts |
| Streaming latency exceeds 500 ms target | Low | Medium | Implement tiered model execution (fast model first, refine later) |
| FPGA firmware compatibility across vendors | Medium | Medium | `ruvector-fpga-transformer` native simulation backend as fallback |
| Quantum simulation too slow for production | Low | Low | Quantum models are supplementary; classical models provide baseline performance |
| VCF output size with many MINORITY_CALL variants | Medium | Low | Configurable filter stringency; default output omits MINORITY_CALL variants |

---

## Alternatives Considered

### Alternative 1: DeepVariant + Manta + GATK-SV Pipeline

Run existing best-in-class tools for each variant type and merge outputs.

**Rejected because**:
- No shared signal model across variant types
- Merging outputs from independent tools creates boundary artifacts (e.g., a 45 bp deletion called as an indel by one tool and ignored by the SV tool)
- No streaming capability
- Cannot leverage RuVector acceleration primitives

### Alternative 2: Single Large Neural Network (End-to-End)

Train a single transformer model that takes raw pileup data and outputs all variant types.

**Rejected because**:
- Requires enormous training data covering all variant types and sizes
- Single point of failure: a bug in the model affects all variant classes
- Interpretability is poor; cannot explain why a variant was called or missed
- Cannot achieve zero false negatives without extreme overfitting

### Alternative 3: Long-Read-Only Pipeline

Require PacBio HiFi or Oxford Nanopore long reads, which resolve many short-read ambiguities.

**Rejected because**:
- Majority of clinical sequencing uses Illumina short reads
- Long-read sequencing is 3-10x more expensive per genome
- Pipeline must support the installed base of short-read sequencers
- Long-read support can be added as an additional signal source within the ensemble

---

## Related Decisions

- **ADR-001**: Ruvector Core Architecture (HNSW index, SIMD distance)
- **ADR-003**: SIMD Optimization Strategy (vectorized distance for k-mer matching)
- **ADR-015**: Coherence-Gated Transformer (sheaf attention used in pileup models)
- **ruQu ADR-001**: ruQu Architecture (VQE, QAOA for quantum-enhanced models)

---

## References

1. Poplin, R., et al. (2018). "A universal SNP and small-indel variant caller using deep neural networks." *Nature Biotechnology*, 36(10), 983-987. (DeepVariant)
2. McKenna, A., et al. (2010). "The Genome Analysis Toolkit: A MapReduce framework for analyzing next-generation DNA sequencing data." *Genome Research*, 20(9), 1297-1303. (GATK)
3. Zook, J.M., et al. (2019). "A robust benchmark for detection of germline large deletions and insertions." *Nature Biotechnology*, 38, 1347-1355. (GIAB SV benchmark)
4. Wagner, J., et al. (2022). "Curated variation benchmarks for challenging medically relevant autosomal genes." *Nature Biotechnology*, 40, 672-680. (CMRG)
5. Nurk, S., et al. (2022). "The complete sequence of a human genome." *Science*, 376(6588), 44-53. (T2T-CHM13)
6. Collins, R.L., et al. (2020). "A structural variation reference for medical and population genetics." *Nature*, 581, 444-451. (gnomAD-SV)
7. Dao, T., et al. (2022). "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness." *NeurIPS 2022*. (Flash attention algorithm)
8. Malkov, Y., & Yashunin, D. (2018). "Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs." *arXiv:1603.09320*. (HNSW)
9. Krusche, P., et al. (2019). "Best practices for benchmarking germline small-variant calls in human genomes." *Nature Biotechnology*, 37, 555-560. (GA4GH benchmarking)
10. VCF Specification v4.3. SAMtools/hts-specs. https://samtools.github.io/hts-specs/VCFv4.3.pdf
11. English, A.C., et al. (2022). "Truvari: refined structural variant comparison preserves allelic diversity." *Genome Biology*, 23, 271.

---

## Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 0.1 | 2026-02-11 | RuVector DNA Analyzer Team | Initial proposal |
