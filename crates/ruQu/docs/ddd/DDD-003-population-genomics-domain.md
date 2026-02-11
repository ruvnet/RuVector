# DDD-003: Population Genomics Domain Model

**Status**: Proposed
**Date**: 2026-02-11
**Authors**: ruv.io, RuVector Team
**Related ADR**: ADR-001-ruqu-architecture
**Related DDD**: DDD-001-coherence-gate-domain, DDD-002-syndrome-processing-domain

---

## Overview

This document defines the Domain-Driven Design model for the Population Genomics bounded context -- the large-scale analytical subsystem that computes ancestry decomposition, kinship matrices, genome-wide association studies (GWAS), and natural selection signals across millions of individuals. The design exploits RuVector's HNSW indexing, distributed clustering, sparse inference, and subpolynomial dynamic min-cut to achieve order-of-magnitude speedups over current state-of-the-art tooling (PLINK 2.0, GCTA, ADMIXTURE).

---

## Strategic Design

### Domain Vision Statement

> The Population Genomics domain provides real-time, population-scale inference on genetic variation data, enabling ancestry estimation in sub-100ms, kinship computation for 1M samples in under 2.4 hours, and full-genome association scans at 10x throughput improvement over existing tools -- all powered by RuVector's vector-native architecture.

### Core Domain

**Population Structure Analysis** is the core domain. This is what differentiates the RuVector DNA Analyzer from traditional bioinformatics pipelines:

- Not variant calling (that is upstream infrastructure)
- Not quality control or imputation (those are supporting domains)
- **The novel capability**: Treating every individual as a high-dimensional genotype vector, enabling vector-similarity operations (PCA, kinship, association) to run as native RuVector queries with HNSW-accelerated nearest-neighbor lookups and distributed sparse-matrix computation.

### Supporting Domains

| Domain | Role | Boundary |
|--------|------|----------|
| **Variant Ingestion** | Load VCF/BGEN/PLINK genotype data | Generic, infrastructure |
| **Quality Control** | Filter samples and variants by missingness, HWE, MAF | Generic, pipeline |
| **Imputation** | Fill missing genotypes from reference panels | External, existing tools |
| **Phenotype Registry** | Store and validate phenotype/covariate data | Supporting, CRUD |
| **Reference Panels** | 1000 Genomes, gnomAD allele frequency databases | External, read-only |

### Generic Subdomains

- Logging, observability, and audit trail
- Job scheduling and compute orchestration
- Authentication and access control (HIPAA/GDPR compliance)

---

## Ubiquitous Language

### Core Terms

| Term | Definition | Context |
|------|------------|---------|
| **Individual** | A single sequenced organism with a genotype vector embedding | Aggregate root |
| **Population** | A named group of individuals sharing demographic or geographic origin | Aggregate root |
| **Genotype Vector** | A 384-dimensional embedding of an individual's multi-locus genotype, stored in ruvector-core | Value object |
| **Allele Frequency** | The proportion of a given allele at a locus within a population | Value object |
| **Fst** | Wright's fixation index measuring population differentiation (0 = identical, 1 = fixed) | Value object |
| **Kinship Coefficient** | Probability that alleles drawn from two individuals are IBD | Value object |

### Analysis Terms

| Term | Definition | Context |
|------|------------|---------|
| **GWAS** | Genome-Wide Association Study: statistical scan testing each variant for phenotype association | Domain service |
| **Manhattan Plot** | Visualization of -log10(p-values) across chromosomal positions | Value object |
| **PCA** | Principal Component Analysis of the genotype matrix for population stratification | Domain service |
| **Ancestry Component** | Fractional membership in a reference population (e.g., 0.42 European, 0.38 East Asian) | Value object |
| **Selection Signal** | Evidence of natural selection acting on a genomic locus | Entity |
| **GRM** | Genetic Relationship Matrix (n x n kinship matrix) | Value object |

### Scalability Terms

| Term | Definition | Context |
|------|------------|---------|
| **Genotype Shard** | A partition of the genotype matrix distributed across ruvector-cluster nodes | Infrastructure |
| **Hypergraph Sketch** | Khanna et al. sparsification of population structure as hypergraph with O-tilde(n) edges | Algorithm |
| **Stratification Cut** | Dynamic min-cut partitioning of the population relatedness graph into subpopulations | Algorithm |

---

## Bounded Contexts

### Context Map

```
+-----------------------------------------------------------------------------+
|                     POPULATION GENOMICS CONTEXT                              |
|                           (Core Domain)                                      |
|  +-------------+  +-------------+  +-------------+  +-------------+         |
|  |  Ancestry   |  |   Kinship   |  |    GWAS     |  | Selection   |         |
|  |   Engine    |  |   Engine    |  |   Engine    |  |  Scanner    |         |
|  +-------------+  +-------------+  +-------------+  +-------------+         |
+-----------+--------------+--------------+--------------+--------------------+
            |              |              |              |
            | Upstream     | Upstream     | Upstream     | Downstream
            v              v              v              v
+-----------------+ +-----------------+ +-----------------+ +-----------------+
|    VARIANT      | |   PHENOTYPE     | |   REFERENCE     | |  REPORTING      |
|    INGESTION    | |   REGISTRY      | |   PANELS        | |  CONTEXT        |
|  (Supporting)   | |  (Supporting)   | |   (External)    | |  (Downstream)   |
+-----------------+ +-----------------+ +-----------------+ +-----------------+
```

### Population Genomics Context (Core)

**Responsibility**: Compute ancestry, kinship, association, and selection analyses at population scale.

**Key Aggregates**:
- Individual
- Population
- GWASResult
- SelectionSignal

**Anti-Corruption Layers**:
- Variant Adapter (translates VCF/BGEN to genotype vectors)
- Phenotype Adapter (translates clinical data to domain phenotype model)
- Reference Panel Adapter (translates 1000G/gnomAD frequencies to domain format)

### Variant Ingestion Context (Supporting)

**Responsibility**: Parse, validate, and vectorize raw genotype files.

**Key Aggregates**:
- GenotypeFile
- VariantCatalog
- IngestionJob

**Relationship**: Conformist -- conforms to Population Genomics Context's genotype vector schema.

### Phenotype Registry Context (Supporting)

**Responsibility**: Manage phenotype definitions, covariate data, and case/control assignments.

**Key Aggregates**:
- PhenotypeDefinition
- CovariateSet
- CohortAssignment

**Relationship**: Customer-Supplier with Population Genomics Context.

---

## Aggregates

### Individual (Root Aggregate)

The central entity representing a genotyped organism.

```
+---------------------------------------------------------------------+
|                         INDIVIDUAL                                    |
|                       (Aggregate Root)                               |
+---------------------------------------------------------------------+
|  individual_id: IndividualId (UUID)                                   |
|  sample_id: SampleId (external lab identifier)                        |
|  genotype_vector: GenotypeVector (384-dim f32, stored in HNSW)        |
|  ancestry_components: Vec<AncestryComponent>                          |
|  relatedness_edges: Vec<RelatednessEdge>                              |
|  metadata: IndividualMetadata                                         |
|  created_at: Timestamp                                                |
|  updated_at: Timestamp                                                |
+---------------------------------------------------------------------+
|  +---------------------------------------------------------------+  |
|  | GenotypeVector (Value Object)                                 |  |
|  |  dimensions: usize (384)                                      |  |
|  |  embedding: Vec<f32>                                          |  |
|  |  source_variants: usize (count of variants encoded)           |  |
|  |  missing_rate: f32 (fraction of missing calls)                |  |
|  |  encoding: GenotypeEncoding { Additive | Dominant | Recessive }|  |
|  +---------------------------------------------------------------+  |
|  +---------------------------------------------------------------+  |
|  | AncestryComponent (Value Object)                              |  |
|  |  population_label: PopulationLabel                            |  |
|  |  proportion: f64 (0.0 .. 1.0)                                |  |
|  |  confidence_interval: (f64, f64) (lower, upper 95% CI)       |  |
|  +---------------------------------------------------------------+  |
|  +---------------------------------------------------------------+  |
|  | RelatednessEdge (Value Object)                                |  |
|  |  other_id: IndividualId                                       |  |
|  |  kinship_coefficient: f64                                     |  |
|  |  relationship_degree: RelationshipDegree                      |  |
|  |  ibd_segments: usize                                          |  |
|  +---------------------------------------------------------------+  |
+---------------------------------------------------------------------+
|  Invariants:                                                          |
|  - genotype_vector dimensions must equal configured embedding size    |
|  - ancestry_components proportions must sum to 1.0 (+/- epsilon)     |
|  - kinship_coefficient in range [0.0, 0.5]                           |
|  - missing_rate in range [0.0, 1.0], must be below QC threshold      |
+---------------------------------------------------------------------+
```

### Population (Aggregate Root)

A defined group of individuals with computed summary statistics.

```
+---------------------------------------------------------------------+
|                         POPULATION                                    |
|                       (Aggregate Root)                               |
+---------------------------------------------------------------------+
|  population_id: PopulationId                                          |
|  label: PopulationLabel                                               |
|  description: String                                                  |
|  member_count: usize                                                  |
|  allele_frequencies: AlleleFrequencyMap                                |
|  fst_distances: FstMatrix                                             |
|  demographic_history: Option<DemographicHistory>                      |
|  centroid: GenotypeVector (mean embedding of members)                 |
+---------------------------------------------------------------------+
|  +---------------------------------------------------------------+  |
|  | AlleleFrequencyMap (Value Object)                             |  |
|  |  frequencies: HashMap<VariantId, AlleleFrequency>             |  |
|  |  variant_count: usize                                         |  |
|  |  effective_sample_size: f64                                   |  |
|  +---------------------------------------------------------------+  |
|  +---------------------------------------------------------------+  |
|  | FstMatrix (Value Object)                                      |  |
|  |  populations: Vec<PopulationId>                               |  |
|  |  values: Vec<f64> (upper-triangular packed)                   |  |
|  |  method: FstMethod { Weir | Hudson | Nei }                   |  |
|  +---------------------------------------------------------------+  |
|  +---------------------------------------------------------------+  |
|  | DemographicHistory (Value Object)                             |  |
|  |  effective_sizes: Vec<(Generation, f64)>                      |  |
|  |  migration_rates: Vec<(PopulationId, PopulationId, f64)>      |  |
|  |  divergence_times: Vec<(PopulationId, PopulationId, Generation)> |
|  +---------------------------------------------------------------+  |
+---------------------------------------------------------------------+
|  Invariants:                                                          |
|  - All allele frequencies in [0.0, 1.0]                               |
|  - Fst values in [0.0, 1.0]                                          |
|  - FstMatrix is symmetric                                             |
|  - member_count > 0                                                   |
|  - centroid dimensions match configured embedding size                |
+---------------------------------------------------------------------+
```

### GWASResult (Aggregate Root)

The output of a genome-wide association scan.

```
+---------------------------------------------------------------------+
|                        GWAS RESULT                                    |
|                       (Aggregate Root)                               |
+---------------------------------------------------------------------+
|  gwas_id: GwasId                                                      |
|  phenotype: PhenotypeId                                               |
|  model: AssociationModel { Linear | Logistic | Mixed }                |
|  sample_count: usize                                                  |
|  variant_count: usize                                                 |
|  genomic_inflation: f64 (lambda_GC)                                   |
|  significant_loci: Vec<SignificantLocus>                               |
|  manhattan_plot: ManhattanPlot                                        |
|  covariates_used: Vec<CovariateId>                                    |
|  pca_adjustment: usize (number of PCs used)                           |
|  created_at: Timestamp                                                |
+---------------------------------------------------------------------+
|  +---------------------------------------------------------------+  |
|  | SignificantLocus (Entity)                                     |  |
|  |  locus_id: LocusId                                            |  |
|  |  variant_id: VariantId                                        |  |
|  |  chromosome: Chromosome                                       |  |
|  |  position: u64                                                |  |
|  |  p_value: f64                                                 |  |
|  |  odds_ratio: Option<f64> (for binary traits)                  |  |
|  |  beta: Option<f64> (for continuous traits)                    |  |
|  |  standard_error: f64                                          |  |
|  |  effect_allele: Allele                                        |  |
|  |  allele_frequency: f64                                        |  |
|  +---------------------------------------------------------------+  |
|  +---------------------------------------------------------------+  |
|  | ManhattanPlot (Value Object)                                  |  |
|  |  points: Vec<PlotPoint { chr, pos, neg_log10_p }>             |  |
|  |  significance_threshold: f64 (default 5e-8)                   |  |
|  |  suggestive_threshold: f64 (default 1e-5)                     |  |
|  +---------------------------------------------------------------+  |
+---------------------------------------------------------------------+
|  Invariants:                                                          |
|  - genomic_inflation (lambda_GC) should be in [0.9, 1.1] for valid scan |
|  - p_values in (0.0, 1.0]                                            |
|  - significant_loci sorted by p_value ascending                       |
|  - sample_count >= minimum for specified model                        |
+---------------------------------------------------------------------+
```

### SelectionSignal (Aggregate Root)

Evidence of natural selection at a genomic locus.

```
+---------------------------------------------------------------------+
|                     SELECTION SIGNAL                                  |
|                       (Aggregate Root)                               |
+---------------------------------------------------------------------+
|  signal_id: SelectionSignalId                                         |
|  locus: GenomicRegion { chromosome, start, end }                      |
|  selection_type: SelectionType { Positive | Balancing | Purifying }    |
|  evidence_score: f64 (composite score, 0.0 - 1.0)                    |
|  statistics: Vec<SelectionStatistic>                                  |
|  populations_affected: Vec<PopulationId>                              |
|  candidate_gene: Option<GeneId>                                       |
+---------------------------------------------------------------------+
|  +---------------------------------------------------------------+  |
|  | SelectionStatistic (Value Object)                             |  |
|  |  test_name: String (iHS, XP-EHH, Tajima's D, Fst-outlier)    |  |
|  |  value: f64                                                   |  |
|  |  p_value: f64                                                 |  |
|  |  empirical_rank: f64 (percentile in genome-wide distribution) |  |
|  +---------------------------------------------------------------+  |
+---------------------------------------------------------------------+
|  Invariants:                                                          |
|  - evidence_score in [0.0, 1.0]                                       |
|  - At least one SelectionStatistic required                           |
|  - GenomicRegion.start < GenomicRegion.end                            |
|  - populations_affected non-empty                                     |
+---------------------------------------------------------------------+
```

---

## Value Objects

### GenotypeVector

384-dimensional embedding stored in ruvector-core HNSW index.

```rust
struct GenotypeVector {
    embedding: [f32; 384],
    source_variants: usize,
    missing_rate: f32,
    encoding: GenotypeEncoding,
}

impl GenotypeVector {
    fn cosine_similarity(&self, other: &GenotypeVector) -> f32;
    fn euclidean_distance(&self, other: &GenotypeVector) -> f32;
    fn is_valid(&self) -> bool {
        self.missing_rate <= MAX_MISSING_RATE
    }
}
```

### AlleleFrequency

Frequency of a specific allele in a population.

```rust
struct AlleleFrequency {
    variant_id: VariantId,
    ref_allele: Allele,
    alt_allele: Allele,
    frequency: f64,        // alt allele frequency
    sample_count: usize,   // number of non-missing samples
    hw_p_value: f64,       // Hardy-Weinberg equilibrium test
}

impl AlleleFrequency {
    fn is_polymorphic(&self) -> bool {
        self.frequency > 0.0 && self.frequency < 1.0
    }
    fn minor_allele_frequency(&self) -> f64 {
        self.frequency.min(1.0 - self.frequency)
    }
}
```

### KinshipCoefficient

Pairwise kinship with relationship inference.

```rust
struct KinshipCoefficient {
    value: f64,
    ibd0: f64,  // P(0 alleles IBD)
    ibd1: f64,  // P(1 allele IBD)
    ibd2: f64,  // P(2 alleles IBD)
}

impl KinshipCoefficient {
    fn relationship_degree(&self) -> RelationshipDegree {
        match self.value {
            v if v > 0.354 => RelationshipDegree::Duplicate,
            v if v > 0.177 => RelationshipDegree::First,   // parent-child, siblings
            v if v > 0.0884 => RelationshipDegree::Second,  // grandparent, uncle
            v if v > 0.0442 => RelationshipDegree::Third,
            _ => RelationshipDegree::Unrelated,
        }
    }
}
```

### GenomicRegion

A contiguous span on a chromosome.

```rust
struct GenomicRegion {
    chromosome: Chromosome,
    start: u64,
    end: u64,
}

impl GenomicRegion {
    fn length(&self) -> u64 { self.end - self.start }
    fn overlaps(&self, other: &GenomicRegion) -> bool {
        self.chromosome == other.chromosome
            && self.start < other.end
            && other.start < self.end
    }
}
```

---

## Domain Events

### Analysis Events

| Event | Trigger | Payload |
|-------|---------|---------|
| `IndividualGenotyped` | Genotype vector computed | individual_id, vector_hash, variant_count |
| `AncestryInferred` | Ancestry estimation complete | individual_id, components, method |
| `KinshipComputed` | Pairwise kinship calculated | individual_pair, coefficient, relationship |
| `GRMConstructed` | Full kinship matrix built | population_id, dimension, compute_time |
| `PCACompleted` | Principal components extracted | population_id, n_components, variance_explained |

### GWAS Events

| Event | Trigger | Payload |
|-------|---------|---------|
| `GWASScanStarted` | Scan initiated | gwas_id, phenotype, sample_count, variant_count |
| `GWASScanCompleted` | Full scan done | gwas_id, significant_count, lambda_gc, duration |
| `SignificantLocusFound` | p < threshold | gwas_id, locus_id, variant_id, p_value |
| `GenomicInflationWarning` | lambda_GC outside [0.9, 1.1] | gwas_id, lambda_gc |

### Population Events

| Event | Trigger | Payload |
|-------|---------|---------|
| `PopulationDefined` | New population created | population_id, label, member_count |
| `AlleleFrequenciesUpdated` | Frequencies recomputed | population_id, variant_count |
| `FstMatrixUpdated` | Differentiation recalculated | population_ids, mean_fst |
| `PopulationStratificationDetected` | Min-cut identifies substructure | population_id, cut_value, subgroups |
| `SelectionSignalDetected` | Selection test significant | signal_id, locus, selection_type, score |

---

## Domain Services

### AncestryInferenceService

Per-sample ancestry estimation using HNSW nearest-neighbor search.

```rust
trait AncestryInferenceService {
    /// Infer ancestry for a single individual in <100ms
    /// Uses ruvector-core HNSW to find k nearest reference panel individuals,
    /// then computes soft clustering via weighted nearest-neighbor assignment.
    async fn infer_ancestry(
        &self,
        individual: &Individual,
        reference_panel: &ReferencePanel,
        k: usize,
    ) -> Result<Vec<AncestryComponent>, AncestryError>;

    /// Batch ancestry for a cohort using ruvector-cluster distributed search
    async fn infer_batch(
        &self,
        individuals: &[IndividualId],
        reference_panel: &ReferencePanel,
    ) -> Result<Vec<(IndividualId, Vec<AncestryComponent>)>, AncestryError>;

    /// Update ancestry as new reference samples arrive (incremental)
    async fn update_references(
        &self,
        new_references: &[Individual],
    ) -> Result<(), AncestryError>;
}
```

**Performance bound**: Per-sample inference in O(log n) via HNSW where n = reference panel size. Target <100ms for panels up to 10M individuals.

### KinshipComputationService

Genetic relationship matrix construction using sparse-matrix operations.

```rust
trait KinshipComputationService {
    /// Construct the GRM for n individuals
    /// Uses ruvector-sparse-inference for sparse matrix multiply:
    /// GRM = (1/m) * Z * Z^T where Z is standardized genotype matrix
    ///
    /// Complexity: O(n^2 * m / p) with p cluster nodes via ruvector-cluster
    /// where m = variants, n = individuals
    async fn compute_grm(
        &self,
        population: &PopulationId,
        config: GrmConfig,
    ) -> Result<GeneticRelationshipMatrix, KinshipError>;

    /// Sparse kinship for specific pairs only (avoid full n^2)
    /// Uses HNSW pre-filtering: only compute kinship for pairs
    /// within distance threshold in embedding space
    async fn compute_sparse_kinship(
        &self,
        population: &PopulationId,
        distance_threshold: f32,
    ) -> Result<SparseKinshipMatrix, KinshipError>;

    /// Identify close relatives (kinship > threshold)
    async fn find_relatives(
        &self,
        individual: &IndividualId,
        threshold: f64,
    ) -> Result<Vec<RelatednessEdge>, KinshipError>;
}
```

**Performance bound**: 1M x 1M GRM in <2.4 hours. This exploits ruvector-sparse-inference for the Z*Z^T product with block-sparse structure (most pairs are unrelated, yielding kinship near zero). Distributed across ruvector-cluster nodes, the block-sparse multiply achieves O(n * k_eff * m / p) where k_eff is the effective number of related pairs per individual (typically O(sqrt(n)) due to population structure), yielding a 10x improvement over the naive O(n^2 * m) GCTA approach.

### GWASService

Genome-wide association scan using distributed computation.

```rust
trait GWASService {
    /// Run full GWAS scan
    /// Each variant test is independent -> embarrassingly parallel across
    /// ruvector-cluster nodes. Uses ruvector-sparse-inference for mixed
    /// model computations (GRAMMAR approximation with pre-computed GRM).
    async fn run_scan(
        &self,
        phenotype: &PhenotypeId,
        config: GwasConfig,
    ) -> Result<GWASResult, GwasError>;

    /// Incremental GWAS: test additional variants against existing model
    async fn extend_scan(
        &self,
        gwas_id: &GwasId,
        new_variants: &[VariantId],
    ) -> Result<Vec<SignificantLocus>, GwasError>;

    /// Conditional analysis: test loci adjusting for known signals
    async fn conditional_analysis(
        &self,
        gwas_id: &GwasId,
        conditioning_variants: &[VariantId],
    ) -> Result<GWASResult, GwasError>;
}
```

**Performance bound**: Full scan of 10M variants x 1M samples in <1 hour. Sharding strategy: variants are partitioned across ruvector-cluster nodes (each node tests m/p variants), with the pre-computed GRM inverse broadcast once. Per-variant test cost: O(n) for score test, yielding total O(n * m / p). With p=100 nodes and SIMD-vectorized score tests, the 1-hour target requires ~28K variant-tests/second/node, well within ruvector-sparse-inference's SIMD throughput.

### PopulationStratificationService

Dynamic population structure detection using min-cut.

```rust
trait PopulationStratificationService {
    /// Run PCA on the genotype matrix
    /// Uses randomized SVD via ruvector-sparse-inference:
    /// Complexity O(n * m * k) for top-k components where k << min(n, m)
    async fn compute_pca(
        &self,
        population: &PopulationId,
        n_components: usize,
    ) -> Result<PCAResult, StratificationError>;

    /// Detect population substructure using dynamic min-cut
    /// Models the relatedness graph as a weighted graph where edge weights
    /// are kinship coefficients. Uses ruvector-mincut subpolynomial algorithm
    /// for real-time stratification as new samples arrive.
    ///
    /// Complexity: O(n^{o(1)}) amortized per sample insertion
    /// (arXiv:2512.13105, El-Hayek/Henzinger/Li SODA 2025)
    async fn detect_substructure(
        &self,
        population: &PopulationId,
    ) -> Result<Vec<Subpopulation>, StratificationError>;

    /// Update stratification incrementally as new samples arrive
    /// Uses hypergraph sparsification (Khanna et al.) to maintain
    /// an O-tilde(n) sketch of the population structure hypergraph
    async fn update_incremental(
        &self,
        population: &PopulationId,
        new_individuals: &[IndividualId],
    ) -> Result<StratificationDelta, StratificationError>;
}
```

**Performance bound**: PCA of first 20 components for 1M samples in <30 minutes via randomized SVD with O(n * m * k) complexity. Dynamic stratification via subpolynomial min-cut yields O(n^{o(1)}) = 2^{O(log^{1-c} n)} amortized update time per new sample, per arXiv:2512.13105. The hypergraph sketch (Khanna et al.) reduces the population structure from O(n^2) pairwise edges to O-tilde(n) hyperedges while preserving all cuts within (1 +/- epsilon).

---

## Repositories

### IndividualRepository

```rust
trait IndividualRepository {
    /// Store individual with genotype vector indexed in HNSW
    async fn store(&self, individual: Individual) -> Result<(), StoreError>;

    /// Find by ID
    async fn find_by_id(&self, id: &IndividualId) -> Option<Individual>;

    /// Find nearest neighbors in genotype space (ancestry similarity)
    /// Delegates to ruvector-core HNSW search
    async fn find_nearest(
        &self,
        query: &GenotypeVector,
        k: usize,
    ) -> Result<Vec<(IndividualId, f32)>, SearchError>;

    /// Find all individuals in a population
    async fn find_by_population(&self, pop: &PopulationId) -> Vec<IndividualId>;

    /// Count total individuals
    async fn count(&self) -> usize;

    /// Batch insert (for initial data loading)
    async fn store_batch(
        &self,
        individuals: Vec<Individual>,
    ) -> Result<usize, StoreError>;
}
```

### PopulationRepository

```rust
trait PopulationRepository {
    async fn store(&self, population: Population) -> Result<(), StoreError>;
    async fn find_by_id(&self, id: &PopulationId) -> Option<Population>;
    async fn find_by_label(&self, label: &PopulationLabel) -> Option<Population>;
    async fn list_all(&self) -> Vec<Population>;
    async fn update_frequencies(
        &self,
        id: &PopulationId,
        frequencies: AlleleFrequencyMap,
    ) -> Result<(), StoreError>;
}
```

### GWASResultRepository

```rust
trait GWASResultRepository {
    async fn store(&self, result: GWASResult) -> Result<(), StoreError>;
    async fn find_by_id(&self, id: &GwasId) -> Option<GWASResult>;
    async fn find_by_phenotype(&self, phenotype: &PhenotypeId) -> Vec<GWASResult>;
    async fn find_significant_loci(
        &self,
        gwas_id: &GwasId,
        p_threshold: f64,
    ) -> Vec<SignificantLocus>;
}
```

### SelectionSignalRepository

```rust
trait SelectionSignalRepository {
    async fn store(&self, signal: SelectionSignal) -> Result<(), StoreError>;
    async fn find_by_region(&self, region: &GenomicRegion) -> Vec<SelectionSignal>;
    async fn find_by_type(&self, stype: SelectionType) -> Vec<SelectionSignal>;
    async fn find_top_signals(&self, n: usize) -> Vec<SelectionSignal>;
}
```

---

## RuVector Integration Architecture

### Crate Mapping

| Domain Operation | RuVector Crate | Mechanism |
|-----------------|----------------|-----------|
| Genotype vector storage + HNSW search | `ruvector-core` | 384-dim HNSW index, O(log n) ANN |
| Distributed GRM/GWAS computation | `ruvector-cluster` | Shard genotype matrix across nodes |
| Sparse GRM multiply (Z*Z^T) | `ruvector-sparse-inference` | Block-sparse SIMD matrix ops |
| Population structure hypergraph | `ruvector-core::advanced::hypergraph` | Hyperedge representation |
| Dynamic stratification min-cut | `ruvector-mincut::subpolynomial` | O(n^{o(1)}) amortized updates |
| Randomized SVD for PCA | `ruvector-sparse-inference` | Low-rank approximation |
| Selection scan parallelism | `ruvector-cluster` | Variant-parallel distributed scan |

### Performance Targets with Complexity Bounds

| Operation | Target | Complexity | Bound Source |
|-----------|--------|------------|--------------|
| Ancestry inference (per sample) | <100ms | O(log n) HNSW search | ruvector-core benchmarks |
| GRM construction (1M x 1M) | <2.4 hours | O(n * k_eff * m / p) sparse | ruvector-sparse-inference |
| PCA (20 components, 1M samples) | <30 minutes | O(n * m * k) randomized SVD | Halko et al. 2011 |
| GWAS (10M variants x 1M samples) | <1 hour | O(n * m / p) distributed | ruvector-cluster |
| Stratification update (per sample) | O(n^{o(1)}) | 2^{O(log^{1-c} n)} amortized | arXiv:2512.13105 |
| Hypergraph sketch maintenance | O-tilde(n) space | Sparsification preserving cuts | Khanna et al. |

---

## Invariants and Business Rules

### Individual Invariants

1. **Vector Dimensionality**: Genotype vector must be exactly 384 dimensions
2. **Ancestry Sum-to-One**: Ancestry component proportions must sum to 1.0 within floating-point tolerance (|sum - 1.0| < 1e-6)
3. **QC Thresholds**: Missing rate must be below configured maximum (default 0.05)
4. **Unique Sample ID**: sample_id must be unique across the system

### GWAS Invariants

1. **Minimum Sample Size**: GWAS requires minimum n=100 for linear, n=50 cases + 50 controls for logistic
2. **Inflation Control**: Lambda_GC must be logged and flagged if outside [0.9, 1.1]
3. **Multiple Testing**: Genome-wide significance requires p < 5e-8 (Bonferroni for ~1M independent tests)
4. **Covariate Completeness**: All covariates must be non-missing for included samples

### Population Invariants

1. **Non-Empty**: Populations must have at least one member
2. **Fst Symmetry**: Fst matrix must be symmetric with zero diagonal
3. **Frequency Range**: All allele frequencies must be in [0.0, 1.0]

---

## Anti-Corruption Layers

### Variant Ingestion ACL

Translates VCF/BGEN/PLINK binary formats to domain genotype vectors.

```rust
impl VariantIngestionAcl {
    /// Convert raw genotype calls to 384-dim embedding
    /// Uses additive encoding (0/1/2) -> PCA projection -> HNSW-compatible vector
    fn vectorize(&self, genotypes: &RawGenotypes) -> Result<GenotypeVector, AclError> {
        let additive = self.encode_additive(genotypes);
        let standardized = self.standardize(additive);
        let embedding = self.project_to_embedding(standardized);
        Ok(GenotypeVector {
            embedding,
            source_variants: genotypes.variant_count(),
            missing_rate: genotypes.missing_fraction(),
            encoding: GenotypeEncoding::Additive,
        })
    }
}
```

### Reference Panel ACL

Translates external reference databases to domain model.

```rust
impl ReferencePanelAcl {
    fn translate(&self, panel: ExternalPanel) -> ReferencePanel {
        ReferencePanel {
            individuals: panel.samples.iter()
                .map(|s| self.to_domain_individual(s))
                .collect(),
            populations: panel.populations.iter()
                .map(|p| self.to_domain_population(p))
                .collect(),
        }
    }
}
```

---

## Context Boundaries Summary

| Boundary | Upstream | Downstream | Integration Pattern |
|----------|----------|------------|---------------------|
| Variant -> PopGen | Variant Ingestion | Population Genomics | Published Language (GenotypeVector) |
| PopGen -> Reporting | Population Genomics | Reporting Context | Domain Events (GWASScanCompleted) |
| RefPanel -> PopGen | Reference Panels | Population Genomics | ACL (ReferencePanel) |
| Phenotype -> PopGen | Phenotype Registry | Population Genomics | Customer-Supplier (PhenotypeData) |

---

## References

- DDD-001: Coherence Gate Domain Model
- DDD-002: Syndrome Processing Domain Model
- arXiv:2512.13105 -- El-Hayek, Henzinger, Li. "Deterministic and Exact Fully-dynamic Minimum Cut of Superpolylogarithmic Size in Subpolynomial Time" (SODA 2025)
- Khanna et al. -- Hypergraph sparsification with O-tilde(n) sketch preserving all cuts
- Halko, Martinsson, Tropp. "Finding Structure with Randomness" (2011) -- randomized SVD bounds
- Evans, Eric. "Domain-Driven Design." Addison-Wesley, 2003.
- Vernon, Vaughn. "Implementing Domain-Driven Design." Addison-Wesley, 2013.
