# ADR-018: RuVector DNA Analyzer - Domain-Driven Design Architecture

**Status**: Proposed
**Date**: 2026-02-11
**Parent**: ADR-001 RuVector Core Architecture, ADR-016 Delta-Behavior DDD Architecture
**Author**: System Architecture Designer

## Abstract

This ADR defines a comprehensive Domain-Driven Design (DDD) architecture for the
"RuVector DNA Analyzer" -- a futuristic DNA analysis engine built on the RuVector
vector database ecosystem. The system encompasses ten bounded contexts spanning raw
signal ingestion through clinical pharmacogenomics, population-scale genomics, pathogen
surveillance, and CRISPR guide engineering. Each context is mapped to existing RuVector
crates and follows the same DDD rigor established by ADR-016 (Delta-Behavior system).

---

## 1. Executive Summary

The RuVector DNA Analyzer models genomic data as high-dimensional vector
representations traversing a pipeline of bounded contexts. By treating each
analysis stage as a distinct subdomain with explicit anti-corruption layers and
published language contracts, the system enables:

- **Streaming basecalling**: Raw nanopore/illumina signals become vector-embedded
  sequences via SONA-powered adaptive neural networks
- **Graph-aware alignment**: Sequences map to population-aware reference graphs,
  not just linear references, using ruvector-graph and ruvector-mincut
- **Incremental variant calling**: Delta-based updates propagate variant discoveries
  through the pipeline without full recomputation
- **Clinical-grade annotation**: Variants flow through annotation and pharmacogenomics
  contexts with traceable provenance chains
- **Pathogen surveillance**: Metagenomic classification leverages
  ruvector-hyperbolic-hnsw for taxonomic tree indexing at scale
- **CRISPR engineering**: Guide RNA design uses ruvector-attention for off-target
  prediction with gated transformer models

---

## 2. Domain Analysis

### 2.1 Strategic Domain Map

```
+===========================================================================+
|                      RUVECTOR DNA ANALYZER SYSTEM                         |
+===========================================================================+
|                                                                           |
|  +-------------------+     +---------------------+     +-----------------+|
|  | 1. Sequence       |     | 2. Alignment &      |     | 3. Variant      ||
|  |    Ingestion      |---->|    Mapping           |---->|    Calling       ||
|  |                   |     |                      |     |                 ||
|  | - Basecallers     |     | - Seed-indexers      |     | - Genotypers    ||
|  | - QC Filters      |     | - Chain-extenders    |     | - SV Callers    ||
|  | - Adaptor Trims   |     | - Graph Aligners     |     | - Phasing       ||
|  +-------------------+     +---------------------+     +--------+--------+|
|                                      |                          |         |
|                                      v                          v         |
|  +-------------------+     +---------------------+     +-----------------+|
|  | 4. Graph Genome   |     | 5. Annotation &     |<----| (Variants)      ||
|  |    Domain         |<----|    Interpretation    |     +-----------------+|
|  |                   |     |                      |                       |
|  | - Ref Graphs      |     | - ClinVar Lookup    |     +-----------------+|
|  | - Min-Cut Parts   |     | - Consequence Pred.  |---->| 6. Epigenomics  ||
|  | - Bubble Chains   |     | - ACMG Classify     |     |                 ||
|  +-------------------+     +---------------------+     | - Methylation   ||
|                                      |                  | - Chromatin     ||
|                                      v                  | - Hi-C / 3D    ||
|  +-------------------+     +---------------------+     +-----------------+|
|  | 7. Pharmaco-      |<----| (Clinical Sig.)     |                       |
|  |    genomics       |     +---------------------+     +-----------------+|
|  |                   |                                  | 9. Pathogen     ||
|  | - PGx Alleles     |     +---------------------+     |    Surveillance ||
|  | - Drug Response   |     | 8. Population        |     |                 ||
|  | - Dosing Models   |     |    Genomics          |     | - Metagenomics  ||
|  +-------------------+     |                      |     | - AMR Detection ||
|                             | - Ancestry           |     | - Outbreak      ||
|  +-------------------+     | - Relatedness        |     +-----------------+|
|  | 10. CRISPR        |     | - GWAS               |                       |
|  |     Engineering   |     +---------------------+                       |
|  |                   |                                                    |
|  | - Guide Design    |                                                    |
|  | - Off-Target Pred |                                                    |
|  | - Edit Scoring    |                                                    |
|  +-------------------+                                                    |
|                                                                           |
+===========================================================================+
```

### 2.2 Core Domain Concepts

| Domain Concept         | Definition                                                           |
|------------------------|----------------------------------------------------------------------|
| **ReadSignal**         | Raw electrical/optical signal from a sequencing instrument            |
| **BasecalledSequence** | Nucleotide string with per-base quality scores (PHRED)               |
| **Alignment**          | Mapping of a sequence to a reference coordinate system               |
| **Variant**            | Deviation from a reference genome (SNV, indel, SV, CNV)              |
| **GenomeGraph**        | Population-aware directed graph representing all known alleles       |
| **Annotation**         | Functional/clinical metadata attached to a variant                   |
| **Epigenome**          | Chromatin state, methylation, and 3D structure overlay               |
| **Pharmacotype**       | Genotype-derived drug metabolism phenotype                           |
| **PopulationAllele**   | Allele frequency and linkage data across cohorts                     |
| **PathogenSignature**  | Taxonomic classification vector for metagenomic reads                |
| **GuideRNA**           | CRISPR spacer sequence with off-target profile                       |

---

## 3. Bounded Context Definitions

### 3.1 Sequence Ingestion Domain

**Purpose**: Convert raw instrument signals into basecalled sequences with quality
metrics. This is the entry point for all genomic data.

#### Ubiquitous Language

| Term               | Definition                                                        |
|--------------------|-------------------------------------------------------------------|
| **ReadSignal**     | Raw analog/digital signal from sequencer (nanopore current, Illumina intensity) |
| **Basecaller**     | Neural network that translates signal to nucleotide sequence       |
| **QualityScore**   | PHRED-scaled confidence for each basecall (Q30 = 1:1000 error)    |
| **AdaptorTrimmer** | Component that removes synthetic adapter sequences                 |
| **ReadGroup**      | Batch of reads from a single library/run/lane                      |
| **FlowCell**       | Physical sequencing unit producing reads                           |
| **SignalChunk**    | Windowed segment of raw signal for streaming basecalling           |

#### Aggregate Root: SequencingRun

```rust
pub mod sequence_ingestion {
    /// Root aggregate for a sequencing run
    pub struct SequencingRun {
        pub id: RunId,
        pub instrument: InstrumentType,
        pub flow_cell: FlowCellId,
        pub read_groups: Vec<ReadGroup>,
        pub status: RunStatus,
        pub metrics: RunMetrics,
        pub started_at: Timestamp,
    }

    // --- Value Objects ---

    #[derive(Clone, Copy, PartialEq, Eq, Hash)]
    pub struct RunId(pub u128);

    #[derive(Clone, Copy, PartialEq, Eq, Hash)]
    pub struct FlowCellId(pub [u8; 16]);

    #[derive(Clone, Copy, PartialEq, Eq)]
    pub enum InstrumentType {
        NanoporePromethion,
        NanoporeMinion,
        IlluminaNovaSeq,
        IlluminaNextSeq,
        PacBioRevio,
        ElementAviti,
    }

    #[derive(Clone, Copy, PartialEq, Eq)]
    pub enum RunStatus {
        Initializing,
        Sequencing,
        Basecalling,
        Complete,
        Failed,
    }

    pub struct RunMetrics {
        pub total_reads: u64,
        pub total_bases: u64,
        pub mean_quality: f32,
        pub n50_length: u32,
        pub pass_rate: f32,
    }

    // --- Entities ---

    pub struct ReadGroup {
        pub id: ReadGroupId,
        pub sample_id: SampleId,
        pub library_id: LibraryId,
        pub reads: Vec<BasecalledRead>,
    }

    pub struct BasecalledRead {
        pub id: ReadId,
        pub sequence: Vec<u8>,          // ACGT as 0,1,2,3
        pub quality_scores: Vec<u8>,     // PHRED scores
        pub signal_embedding: Vec<f32>,  // SONA-produced embedding
        pub length: u32,
        pub mean_quality: f32,
        pub is_pass: bool,
    }

    #[derive(Clone, Copy, PartialEq, Eq, Hash)]
    pub struct ReadId(pub u128);
    pub struct ReadGroupId(pub String);
    pub struct SampleId(pub String);
    pub struct LibraryId(pub String);
    pub struct Timestamp(pub u64);

    // --- Invariants ---
    // 1. All reads in a ReadGroup share the same SampleId
    // 2. Quality scores length == sequence length
    // 3. signal_embedding dimensionality is fixed per InstrumentType
    // 4. mean_quality must equal the arithmetic mean of quality_scores
    // 5. is_pass is true iff mean_quality >= instrument pass threshold

    // --- Repository Interface ---
    pub trait SequencingRunRepository: Send + Sync {
        fn save(&self, run: &SequencingRun) -> Result<(), IngestionError>;
        fn find_by_id(&self, id: &RunId) -> Result<Option<SequencingRun>, IngestionError>;
        fn find_by_flow_cell(&self, fc: &FlowCellId) -> Result<Vec<SequencingRun>, IngestionError>;
        fn find_active(&self) -> Result<Vec<SequencingRun>, IngestionError>;
    }

    pub trait ReadRepository: Send + Sync {
        fn store_batch(&self, reads: &[BasecalledRead]) -> Result<u64, IngestionError>;
        fn find_by_id(&self, id: &ReadId) -> Result<Option<BasecalledRead>, IngestionError>;
        fn find_by_quality_range(&self, min_q: f32, max_q: f32) -> Result<Vec<ReadId>, IngestionError>;
        fn count_by_run(&self, run_id: &RunId) -> Result<u64, IngestionError>;
    }

    #[derive(Debug)]
    pub enum IngestionError {
        SignalCorrupted(String),
        BasecallFailed(String),
        QualityBelowThreshold { expected: f32, actual: f32 },
        StorageFull,
        DuplicateRead(ReadId),
    }
}
```

#### Domain Events

| Event                    | Payload                                    | Published When                        |
|--------------------------|--------------------------------------------|---------------------------------------|
| `RunStarted`             | run_id, instrument, flow_cell_id           | New sequencing run begins             |
| `SignalChunkReceived`    | run_id, chunk_index, signal_length         | Raw signal arrives from instrument    |
| `ReadBasecalled`         | read_id, run_id, length, mean_quality      | Single read basecalled                |
| `ReadGroupComplete`      | read_group_id, read_count, pass_rate       | All reads in group basecalled         |
| `RunComplete`            | run_id, total_reads, total_bases, n50      | Entire run finished                   |
| `QualityCheckFailed`     | read_id, reason                            | Read fails QC filters                 |

#### Domain Services

```rust
pub trait BasecallingService: Send + Sync {
    /// Process a raw signal chunk into basecalled reads
    fn basecall(&self, signal: &[f32], config: &BasecallConfig) -> Result<Vec<BasecalledRead>, IngestionError>;
}

pub trait QualityControlService: Send + Sync {
    /// Apply QC filters to reads, returning pass/fail partition
    fn filter(&self, reads: &[BasecalledRead], policy: &QcPolicy) -> (Vec<BasecalledRead>, Vec<BasecalledRead>);
}

pub trait AdaptorTrimmingService: Send + Sync {
    /// Remove adapter sequences from read ends
    fn trim(&self, read: &mut BasecalledRead, adapters: &[Vec<u8>]) -> TrimResult;
}
```

---

### 3.2 Alignment & Mapping Domain

**Purpose**: Map basecalled sequences to positions on a reference genome (linear or
graph-based), producing coordinate-sorted alignments.

#### Ubiquitous Language

| Term              | Definition                                                         |
|-------------------|--------------------------------------------------------------------|
| **Alignment**     | A read mapped to reference coordinates with CIGAR operations       |
| **SeedHit**       | Short exact match between read and reference used to anchor alignment |
| **ChainedAnchors**| Set of collinear seeds forming a candidate alignment region        |
| **CIGAR**         | Compact representation of alignment operations (M/I/D/S/H/N)      |
| **MappingQuality**| PHRED-scaled probability that the mapping position is wrong        |
| **SplitAlignment**| Read spanning a structural breakpoint mapped in multiple segments  |
| **SupplementaryAlignment** | Secondary location for a chimeric/split read              |

#### Aggregate Root: AlignmentBatch

```rust
pub mod alignment_mapping {
    pub struct AlignmentBatch {
        pub id: BatchId,
        pub reference_id: ReferenceId,
        pub alignments: Vec<Alignment>,
        pub unmapped_reads: Vec<ReadId>,
        pub metrics: AlignmentMetrics,
    }

    pub struct Alignment {
        pub read_id: ReadId,
        pub reference_id: ReferenceId,
        pub contig: ContigId,
        pub position: GenomicPosition,
        pub cigar: CigarString,
        pub mapping_quality: u8,
        pub alignment_score: i32,
        pub is_primary: bool,
        pub is_supplementary: bool,
        pub mate: Option<MateInfo>,
        pub tags: AlignmentTags,
    }

    // --- Value Objects ---

    #[derive(Clone, Copy, PartialEq, Eq, Hash)]
    pub struct BatchId(pub u128);

    #[derive(Clone, Copy, PartialEq, Eq, Hash)]
    pub struct ReferenceId(pub u64);

    #[derive(Clone, PartialEq, Eq, Hash)]
    pub struct ContigId(pub String);  // e.g., "chr1", "chrX"

    #[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
    pub struct GenomicPosition {
        pub contig_index: u32,
        pub offset: u64,         // 0-based position
        pub strand: Strand,
    }

    #[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
    pub enum Strand { Forward, Reverse }

    pub struct CigarString(pub Vec<CigarOp>);

    #[derive(Clone, Copy)]
    pub enum CigarOp {
        Match(u32),
        Insertion(u32),
        Deletion(u32),
        SoftClip(u32),
        HardClip(u32),
        RefSkip(u32),   // spliced alignment
        Mismatch(u32),
    }

    pub struct MateInfo {
        pub contig: ContigId,
        pub position: u64,
        pub insert_size: i64,
    }

    pub struct AlignmentMetrics {
        pub total_reads: u64,
        pub mapped_reads: u64,
        pub mapping_rate: f32,
        pub mean_mapq: f32,
        pub mean_coverage: f32,
        pub duplicate_rate: f32,
    }

    pub struct AlignmentTags(pub Vec<(String, TagValue)>);

    pub enum TagValue {
        Int(i64),
        Float(f32),
        String(String),
        ByteArray(Vec<u8>),
    }

    // --- Invariants ---
    // 1. position.offset + reference_consumed(cigar) <= contig_length
    // 2. mapping_quality in 0..=255
    // 3. Exactly one primary alignment per read per batch
    // 4. Supplementary alignments must reference the same read_id
    // 5. insert_size is defined only for paired-end reads

    // --- Repository Interface ---
    pub trait AlignmentRepository: Send + Sync {
        fn store_batch(&self, batch: &AlignmentBatch) -> Result<(), AlignmentError>;
        fn find_overlapping(&self, region: &GenomicRegion) -> Result<Vec<Alignment>, AlignmentError>;
        fn find_by_read(&self, read_id: &ReadId) -> Result<Vec<Alignment>, AlignmentError>;
        fn coverage_at(&self, position: &GenomicPosition) -> Result<u32, AlignmentError>;
    }

    #[derive(Clone)]
    pub struct GenomicRegion {
        pub contig: ContigId,
        pub start: u64,
        pub end: u64,
    }

    #[derive(Debug)]
    pub enum AlignmentError {
        ReferenceNotFound(ReferenceId),
        IndexCorrupted(String),
        PositionOutOfBounds { position: u64, contig_length: u64 },
        InvalidCigar(String),
    }
}
```

#### Domain Events

| Event                    | Payload                                       | Published When                      |
|--------------------------|-----------------------------------------------|-------------------------------------|
| `AlignmentCompleted`     | batch_id, ref_id, mapped_count, unmapped_count | Batch of reads aligned             |
| `SplitAlignmentDetected` | read_id, segment_count, breakpoints            | Read maps to disjoint regions      |
| `LowMappingQuality`      | read_id, mapq, threshold                      | Read below MAPQ threshold           |
| `CoverageThresholdReached`| region, coverage_depth                        | Region exceeds target coverage      |
| `ChimericReadDetected`   | read_id, contig_a, contig_b                   | Read spans two chromosomes          |

---

### 3.3 Variant Calling Domain

**Purpose**: Identify and genotype genomic variants (SNV, indel, structural variants,
copy number variants) from alignment pileups.

#### Ubiquitous Language

| Term                | Definition                                                       |
|---------------------|------------------------------------------------------------------|
| **Variant**         | Any deviation from the reference: SNV, indel, SV, or CNV         |
| **Genotype**        | Diploid allele assignment (0/0, 0/1, 1/1, etc.)                 |
| **Pileup**          | Stack of aligned reads at a specific position                    |
| **HaplotypeBLock**  | Phased segment of linked alleles on one chromosome copy          |
| **StructuralVariant** | Large-scale rearrangement (>50bp): deletion, duplication, inversion, translocation |
| **VariantQuality**  | PHRED-scaled confidence in the variant call                      |
| **AlleleDepth**     | Read support counts per allele at a site                         |

#### Aggregate Root: VariantCallSet

```rust
pub mod variant_calling {
    pub struct VariantCallSet {
        pub id: CallSetId,
        pub sample_id: SampleId,
        pub reference_id: ReferenceId,
        pub caller: CallerInfo,
        pub variants: Vec<Variant>,
        pub metrics: CallSetMetrics,
    }

    pub struct Variant {
        pub id: VariantId,
        pub position: GenomicPosition,
        pub reference_allele: Vec<u8>,
        pub alternate_alleles: Vec<Vec<u8>>,
        pub variant_type: VariantType,
        pub genotype: Genotype,
        pub quality: f32,
        pub filter: FilterStatus,
        pub allele_depths: Vec<u32>,
        pub total_depth: u32,
        pub strand_bias: f32,
        pub genotype_likelihood: Vec<f32>,  // PL field, PHRED-scaled
        pub effect_embedding: Vec<f32>,     // GNN-predicted effect vector
    }

    // --- Value Objects ---

    #[derive(Clone, Copy, PartialEq, Eq, Hash)]
    pub struct CallSetId(pub u128);

    #[derive(Clone, Copy, PartialEq, Eq, Hash)]
    pub struct VariantId(pub u128);

    #[derive(Clone, Copy, PartialEq, Eq)]
    pub enum VariantType {
        Snv,
        Insertion,
        Deletion,
        Mnv,                     // multi-nucleotide variant
        DeletionSv,              // structural deletion >50bp
        DuplicationSv,
        InversionSv,
        TranslocationSv,
        CopyNumberGain,
        CopyNumberLoss,
        ComplexSv,
    }

    #[derive(Clone, PartialEq, Eq)]
    pub struct Genotype {
        pub alleles: Vec<u8>,    // indices into ref + alt alleles
        pub phased: bool,        // true = | separator, false = / separator
    }

    #[derive(Clone, Copy, PartialEq, Eq)]
    pub enum FilterStatus {
        Pass,
        LowQuality,
        LowDepth,
        StrandBias,
        ExcessHeterozygosity,
        MapQualityBias,
        Custom(&'static str),
    }

    pub struct CallerInfo {
        pub name: String,        // e.g., "ruvector-deepvariant"
        pub version: String,
        pub parameters: Vec<(String, String)>,
    }

    pub struct CallSetMetrics {
        pub total_variants: u64,
        pub snv_count: u64,
        pub indel_count: u64,
        pub sv_count: u64,
        pub ti_tv_ratio: f32,    // transition/transversion
        pub het_hom_ratio: f32,
        pub mean_quality: f32,
    }

    // --- Invariants ---
    // 1. reference_allele.len() >= 1 (always anchored)
    // 2. At least one alternate_allele differs from reference_allele
    // 3. genotype.alleles indices are valid into [ref] + alternate_alleles
    // 4. allele_depths.len() == 1 + alternate_alleles.len()
    // 5. total_depth == allele_depths.iter().sum()
    // 6. Ti/Tv ratio for WGS should be ~2.0-2.1 (aggregate invariant)
    // 7. effect_embedding dimensionality matches GNN model config

    // --- Repository Interface ---
    pub trait VariantRepository: Send + Sync {
        fn store_callset(&self, callset: &VariantCallSet) -> Result<(), VariantError>;
        fn find_in_region(&self, region: &GenomicRegion) -> Result<Vec<Variant>, VariantError>;
        fn find_by_type(&self, vtype: VariantType) -> Result<Vec<Variant>, VariantError>;
        fn find_by_quality_range(&self, min_q: f32, max_q: f32) -> Result<Vec<Variant>, VariantError>;
        fn nearest_by_embedding(&self, embedding: &[f32], k: usize) -> Result<Vec<Variant>, VariantError>;
    }

    #[derive(Debug)]
    pub enum VariantError {
        InvalidGenotype(String),
        DepthMismatch { expected: usize, actual: usize },
        PositionOutOfBounds,
        DuplicateVariant(VariantId),
        EmbeddingDimensionMismatch { expected: usize, actual: usize },
    }
}
```

#### Domain Events

| Event                   | Payload                                         | Published When                    |
|-------------------------|-------------------------------------------------|-----------------------------------|
| `VariantCalled`         | variant_id, position, type, quality, genotype    | New variant discovered            |
| `GenotypeRefined`       | variant_id, old_gt, new_gt, new_quality          | Genotype updated with more data   |
| `StructuralVariantFound`| variant_id, sv_type, breakpoints, size           | SV detected from split reads      |
| `PhasingCompleted`      | sample_id, block_count, phase_rate               | Haplotype phasing done            |
| `CallSetFinalized`      | callset_id, variant_count, metrics               | Variant calling pipeline done     |

---

### 3.4 Graph Genome Domain

**Purpose**: Maintain population-aware reference graphs that represent all known
alleles as graph structures rather than linear references. Supports min-cut partitioning
for efficient graph traversal.

#### Ubiquitous Language

| Term              | Definition                                                        |
|-------------------|-------------------------------------------------------------------|
| **GenomeGraph**   | Directed graph where nodes are sequence segments, edges are adjacencies |
| **Bubble**        | Subgraph representing allelic variation between two anchor nodes    |
| **Superbubble**   | Nested bubble structure for complex variation                      |
| **Partition**     | Min-cut decomposition of the graph for parallel processing         |
| **PathHaplotype** | A walk through the graph representing one haplotype                |
| **AnchorNode**    | High-confidence invariant node shared by all haplotypes            |

#### Aggregate Root: GenomeGraph

```rust
pub mod graph_genome {
    pub struct GenomeGraph {
        pub id: GraphId,
        pub name: String,            // e.g., "GRCh38-pangenome-v2"
        pub contigs: Vec<ContigGraph>,
        pub population_sources: Vec<PopulationSource>,
        pub statistics: GraphStatistics,
    }

    pub struct ContigGraph {
        pub contig_id: ContigId,
        pub nodes: Vec<SequenceNode>,
        pub edges: Vec<GraphEdge>,
        pub bubbles: Vec<Bubble>,
        pub partitions: Vec<Partition>,
    }

    pub struct SequenceNode {
        pub id: NodeId,
        pub sequence: Vec<u8>,       // nucleotide content
        pub length: u32,
        pub is_reference: bool,      // true if on GRCh38 backbone
        pub allele_frequency: f32,   // population frequency
        pub embedding: Vec<f32>,     // vector representation for ANN search
    }

    pub struct GraphEdge {
        pub from: NodeId,
        pub to: NodeId,
        pub edge_type: EdgeType,
        pub weight: f32,             // traversal frequency
    }

    #[derive(Clone, Copy, PartialEq, Eq)]
    pub enum EdgeType {
        Reference,           // follows linear reference
        Variant,             // alternative allele path
        Structural,          // structural variant edge
    }

    pub struct Bubble {
        pub id: BubbleId,
        pub source_node: NodeId,     // entry anchor
        pub sink_node: NodeId,       // exit anchor
        pub paths: Vec<Vec<NodeId>>, // allelic paths through bubble
        pub complexity: BubbleComplexity,
    }

    #[derive(Clone, Copy)]
    pub enum BubbleComplexity {
        Simple,              // biallelic SNV/indel
        Multi,               // multiallelic
        Super,               // nested superbubble
    }

    pub struct Partition {
        pub id: PartitionId,
        pub node_set: Vec<NodeId>,
        pub cut_edges: Vec<GraphEdge>,
        pub boundary_nodes: Vec<NodeId>,
        pub min_cut_value: f32,
    }

    // --- Value Objects ---
    #[derive(Clone, Copy, PartialEq, Eq, Hash)]
    pub struct GraphId(pub u128);
    #[derive(Clone, Copy, PartialEq, Eq, Hash)]
    pub struct NodeId(pub u64);
    #[derive(Clone, Copy, PartialEq, Eq, Hash)]
    pub struct BubbleId(pub u64);
    #[derive(Clone, Copy, PartialEq, Eq, Hash)]
    pub struct PartitionId(pub u32);

    pub struct PopulationSource {
        pub name: String,       // e.g., "1000Genomes", "HPRC"
        pub sample_count: u32,
        pub ancestry_groups: Vec<String>,
    }

    pub struct GraphStatistics {
        pub total_nodes: u64,
        pub total_edges: u64,
        pub total_bubbles: u64,
        pub total_partitions: u32,
        pub mean_node_length: f32,
        pub graph_complexity: f32,  // edges/nodes ratio
    }

    // --- Invariants ---
    // 1. Graph is a DAG within each contig (no cycles)
    // 2. Every bubble has exactly one source and one sink
    // 3. All paths through a bubble connect source to sink
    // 4. Partitions are non-overlapping (except boundary nodes)
    // 5. Sum of partition node sets = total node set
    // 6. allele_frequency in [0.0, 1.0]
    // 7. At least one Reference edge path exists per contig (backbone)

    // --- Repository Interface ---
    pub trait GenomeGraphRepository: Send + Sync {
        fn save(&self, graph: &GenomeGraph) -> Result<(), GraphGenomeError>;
        fn find_by_id(&self, id: &GraphId) -> Result<Option<GenomeGraph>, GraphGenomeError>;
        fn subgraph(&self, graph_id: &GraphId, region: &GenomicRegion) -> Result<ContigGraph, GraphGenomeError>;
        fn find_bubbles_in_region(&self, graph_id: &GraphId, region: &GenomicRegion) -> Result<Vec<Bubble>, GraphGenomeError>;
        fn find_partition(&self, graph_id: &GraphId, partition_id: &PartitionId) -> Result<Partition, GraphGenomeError>;
        fn nearest_nodes_by_embedding(&self, embedding: &[f32], k: usize) -> Result<Vec<SequenceNode>, GraphGenomeError>;
    }

    #[derive(Debug)]
    pub enum GraphGenomeError {
        CycleDetected(ContigId),
        OrphanedNode(NodeId),
        InvalidBubble { bubble_id: BubbleId, reason: String },
        PartitionOverlap(PartitionId, PartitionId),
        NodeNotFound(NodeId),
    }
}
```

#### Domain Events

| Event                     | Payload                                      | Published When                     |
|---------------------------|----------------------------------------------|------------------------------------|
| `GraphConstructed`        | graph_id, node_count, edge_count             | New pangenome graph built          |
| `BubbleIdentified`        | bubble_id, source, sink, path_count          | Allelic variation site found       |
| `GraphPartitioned`        | graph_id, partition_count, max_cut_value     | Min-cut partitioning complete      |
| `GraphUpdated`            | graph_id, nodes_added, edges_added           | New population data incorporated   |
| `PathHaplotypeResolved`   | graph_id, sample_id, path_nodes              | Sample haplotype traced through graph |

---

### 3.5 Annotation & Interpretation Domain

**Purpose**: Annotate variants with functional consequences, clinical significance,
population frequencies, and ACMG/AMP classification.

#### Aggregate Root: AnnotatedVariant

```rust
pub mod annotation_interpretation {
    pub struct AnnotatedVariant {
        pub variant_id: VariantId,
        pub consequence: VariantConsequence,
        pub clinical: ClinicalAnnotation,
        pub population_freq: PopulationFrequency,
        pub predictions: InSilicoPredictions,
        pub acmg_classification: AcmgClassification,
        pub provenance: AnnotationProvenance,
    }

    pub struct VariantConsequence {
        pub gene: Option<GeneId>,
        pub transcript: Option<TranscriptId>,
        pub consequence_type: ConsequenceType,
        pub hgvs_coding: Option<String>,     // e.g., "c.123A>G"
        pub hgvs_protein: Option<String>,    // e.g., "p.Thr41Ala"
        pub exon_number: Option<u32>,
        pub codon_change: Option<(String, String)>,
    }

    #[derive(Clone, Copy, PartialEq, Eq)]
    pub enum ConsequenceType {
        Synonymous,
        Missense,
        Nonsense,              // stop gained
        Frameshift,
        SpliceDonor,
        SpliceAcceptor,
        FivePrimeUtr,
        ThreePrimeUtr,
        Intergenic,
        Intronic,
        StartLoss,
        StopLoss,
        InframeDeletion,
        InframeInsertion,
        RegulatoryRegion,
    }

    pub struct ClinicalAnnotation {
        pub clinvar_id: Option<String>,
        pub clinvar_significance: Option<ClinicalSignificance>,
        pub omim_ids: Vec<String>,
        pub disease_associations: Vec<DiseaseAssociation>,
        pub review_status: ClinVarReviewStatus,
    }

    #[derive(Clone, Copy, PartialEq, Eq)]
    pub enum ClinicalSignificance {
        Benign,
        LikelyBenign,
        Vus,                   // variant of uncertain significance
        LikelyPathogenic,
        Pathogenic,
        DrugResponse,
        RiskFactor,
        Conflicting,
    }

    #[derive(Clone, Copy, PartialEq, Eq)]
    pub enum ClinVarReviewStatus {
        NoAssertion,
        SingleSubmitter,
        MultipleSubmitters,
        ExpertPanel,
        PracticeGuideline,
    }

    pub struct DiseaseAssociation {
        pub disease_name: String,
        pub mondo_id: Option<String>,
        pub inheritance: InheritancePattern,
        pub penetrance: Penetrance,
    }

    #[derive(Clone, Copy, PartialEq, Eq)]
    pub enum InheritancePattern {
        AutosomalDominant,
        AutosomalRecessive,
        XLinked,
        Mitochondrial,
        Complex,
    }

    #[derive(Clone, Copy, PartialEq, Eq)]
    pub enum Penetrance { Complete, Incomplete, Unknown }

    pub struct PopulationFrequency {
        pub gnomad_af: Option<f32>,          // global allele frequency
        pub gnomad_af_by_pop: Vec<(String, f32)>,  // per-population AF
        pub topmed_af: Option<f32>,
        pub is_rare: bool,                    // AF < 0.01
    }

    pub struct InSilicoPredictions {
        pub sift_score: Option<f32>,
        pub polyphen_score: Option<f32>,
        pub cadd_phred: Option<f32>,
        pub revel_score: Option<f32>,
        pub alphamissense_score: Option<f32>,
        pub gnn_effect_vector: Vec<f32>,     // RuVector GNN prediction
    }

    #[derive(Clone, Copy, PartialEq, Eq)]
    pub enum AcmgClassification {
        Benign,
        LikelyBenign,
        Vus,
        LikelyPathogenic,
        Pathogenic,
    }

    pub struct AnnotationProvenance {
        pub databases_queried: Vec<(String, String)>,  // (name, version)
        pub annotation_date: u64,
        pub pipeline_version: String,
    }

    // --- Value Objects ---
    pub struct GeneId(pub String);       // e.g., "ENSG00000141510" (TP53)
    pub struct TranscriptId(pub String); // e.g., "ENST00000269305"

    // --- Repository Interface ---
    pub trait AnnotationRepository: Send + Sync {
        fn annotate(&self, variant_id: &VariantId) -> Result<AnnotatedVariant, AnnotationError>;
        fn find_pathogenic_in_gene(&self, gene: &GeneId) -> Result<Vec<AnnotatedVariant>, AnnotationError>;
        fn search_by_disease(&self, disease: &str) -> Result<Vec<AnnotatedVariant>, AnnotationError>;
        fn nearest_by_effect_vector(&self, vector: &[f32], k: usize) -> Result<Vec<AnnotatedVariant>, AnnotationError>;
    }

    #[derive(Debug)]
    pub enum AnnotationError {
        VariantNotFound(VariantId),
        DatabaseUnavailable(String),
        TranscriptMappingFailed(String),
        ConsequencePredictionFailed(String),
    }
}
```

#### Domain Events

| Event                     | Payload                                        | Published When                    |
|---------------------------|------------------------------------------------|-----------------------------------|
| `VariantAnnotated`        | variant_id, consequence, clinical_sig          | Annotation pipeline completes     |
| `PathogenicVariantFound`  | variant_id, gene, disease, acmg_class          | P/LP variant identified           |
| `NovelVariantDetected`    | variant_id, position, consequence              | Variant absent from all databases |
| `AcmgReclassified`       | variant_id, old_class, new_class, evidence     | Classification changed            |

---

### 3.6 Epigenomics Domain

**Purpose**: Model the epigenetic landscape including DNA methylation, histone
modifications, chromatin accessibility, and 3D genome structure (Hi-C/TADs).

#### Aggregate Root: EpigenomicProfile

```rust
pub mod epigenomics {
    pub struct EpigenomicProfile {
        pub id: ProfileId,
        pub sample_id: SampleId,
        pub cell_type: CellType,
        pub methylation_map: MethylationMap,
        pub chromatin_state: ChromatinStateMap,
        pub hi_c_contacts: Option<ContactMatrix>,
        pub tad_boundaries: Vec<TadBoundary>,
    }

    pub struct MethylationMap {
        pub cpg_sites: Vec<CpGSite>,
        pub global_methylation_level: f32,
        pub differentially_methylated_regions: Vec<DmrRegion>,
    }

    pub struct CpGSite {
        pub position: GenomicPosition,
        pub methylation_ratio: f32,    // 0.0 (unmethylated) to 1.0 (fully methylated)
        pub coverage: u32,
        pub context: MethylationContext,
    }

    #[derive(Clone, Copy, PartialEq, Eq)]
    pub enum MethylationContext { CpG, CHG, CHH }

    pub struct DmrRegion {
        pub region: GenomicRegion,
        pub mean_delta_methylation: f32,
        pub p_value: f64,
        pub associated_gene: Option<GeneId>,
    }

    pub struct ChromatinStateMap {
        pub states: Vec<ChromatinSegment>,
        pub model: ChromHmmModel,
    }

    pub struct ChromatinSegment {
        pub region: GenomicRegion,
        pub state: ChromatinState,
        pub posterior_probability: f32,
    }

    #[derive(Clone, Copy, PartialEq, Eq)]
    pub enum ChromatinState {
        ActivePromoter,
        StrongEnhancer,
        WeakEnhancer,
        Transcribed,
        Heterochromatin,
        PoisedPromoter,
        Repressed,
        Quiescent,
    }

    pub struct ContactMatrix {
        pub resolution: u32,           // bin size in bp
        pub matrix_embedding: Vec<f32>,// flattened + compressed via ruvector-core
        pub compartments: Vec<Compartment>,
    }

    pub struct Compartment {
        pub region: GenomicRegion,
        pub compartment_type: CompartmentType,
        pub eigenvector_value: f32,
    }

    #[derive(Clone, Copy, PartialEq, Eq)]
    pub enum CompartmentType { A, B }  // A = active, B = inactive

    pub struct TadBoundary {
        pub position: GenomicPosition,
        pub insulation_score: f32,
        pub boundary_strength: f32,
    }

    // --- Value Objects ---
    #[derive(Clone, Copy, PartialEq, Eq, Hash)]
    pub struct ProfileId(pub u128);
    pub struct CellType(pub String);
    pub struct ChromHmmModel(pub String);

    // --- Repository Interface ---
    pub trait EpigenomicRepository: Send + Sync {
        fn store_profile(&self, profile: &EpigenomicProfile) -> Result<(), EpigenomicError>;
        fn find_methylation_in_region(&self, sample: &SampleId, region: &GenomicRegion) -> Result<Vec<CpGSite>, EpigenomicError>;
        fn find_dmrs(&self, sample_a: &SampleId, sample_b: &SampleId) -> Result<Vec<DmrRegion>, EpigenomicError>;
        fn chromatin_state_at(&self, sample: &SampleId, position: &GenomicPosition) -> Result<ChromatinState, EpigenomicError>;
    }

    #[derive(Debug)]
    pub enum EpigenomicError {
        ProfileNotFound(ProfileId),
        ResolutionMismatch { expected: u32, actual: u32 },
        InsufficientCoverage { site: GenomicPosition, coverage: u32 },
    }
}
```

#### Domain Events

| Event                        | Payload                                   | Published When                   |
|------------------------------|-------------------------------------------|----------------------------------|
| `MethylationProfiled`       | profile_id, sample_id, cpg_count          | Methylation analysis complete    |
| `DmrIdentified`             | region, delta_methylation, gene           | Differentially methylated region |
| `TadBoundaryDisrupted`      | position, variant_id, insulation_change   | Variant disrupts TAD boundary    |
| `ChromatinStateChanged`     | region, old_state, new_state, cell_type   | State transition detected        |

---

### 3.7 Pharmacogenomics Domain

**Purpose**: Translate genotypes into drug response predictions, star-allele calls,
and clinical dosing recommendations.

#### Aggregate Root: PharmacogenomicProfile

```rust
pub mod pharmacogenomics {
    pub struct PharmacogenomicProfile {
        pub id: PgxProfileId,
        pub sample_id: SampleId,
        pub star_alleles: Vec<StarAlleleDiplotype>,
        pub drug_interactions: Vec<DrugGeneInteraction>,
        pub dosing_recommendations: Vec<DosingRecommendation>,
        pub metabolizer_phenotypes: Vec<MetabolizerPhenotype>,
    }

    pub struct StarAlleleDiplotype {
        pub gene: GeneId,
        pub gene_symbol: String,            // e.g., "CYP2D6"
        pub allele_1: StarAllele,
        pub allele_2: StarAllele,
        pub activity_score: f32,
        pub function: AlleleFunction,
    }

    pub struct StarAllele {
        pub name: String,                   // e.g., "*1", "*4", "*17"
        pub defining_variants: Vec<VariantId>,
        pub function: AlleleFunction,
    }

    #[derive(Clone, Copy, PartialEq, Eq)]
    pub enum AlleleFunction {
        NormalFunction,
        DecreasedFunction,
        NoFunction,
        IncreasedFunction,
        UncertainFunction,
    }

    pub struct MetabolizerPhenotype {
        pub gene_symbol: String,
        pub phenotype: MetabolizerStatus,
        pub activity_score: f32,
    }

    #[derive(Clone, Copy, PartialEq, Eq)]
    pub enum MetabolizerStatus {
        UltrarapidMetabolizer,
        RapidMetabolizer,
        NormalMetabolizer,
        IntermediateMetabolizer,
        PoorMetabolizer,
    }

    pub struct DrugGeneInteraction {
        pub drug_name: String,
        pub rxnorm_id: Option<String>,
        pub gene_symbol: String,
        pub evidence_level: EvidenceLevel,
        pub interaction_type: InteractionType,
        pub predicted_response_embedding: Vec<f32>,  // SONA-predicted
    }

    #[derive(Clone, Copy, PartialEq, Eq)]
    pub enum EvidenceLevel { Level1A, Level1B, Level2A, Level2B, Level3, Level4 }

    #[derive(Clone, Copy, PartialEq, Eq)]
    pub enum InteractionType { Dosing, Efficacy, Toxicity, Contraindication }

    pub struct DosingRecommendation {
        pub drug_name: String,
        pub gene_symbol: String,
        pub phenotype: MetabolizerStatus,
        pub recommendation: String,
        pub source: String,               // e.g., "CPIC", "DPWG"
        pub guideline_version: String,
    }

    // --- Value Objects ---
    #[derive(Clone, Copy, PartialEq, Eq, Hash)]
    pub struct PgxProfileId(pub u128);

    // --- Repository Interface ---
    pub trait PharmacogenomicRepository: Send + Sync {
        fn store_profile(&self, profile: &PharmacogenomicProfile) -> Result<(), PgxError>;
        fn find_by_sample(&self, sample: &SampleId) -> Result<Option<PharmacogenomicProfile>, PgxError>;
        fn find_interactions_for_drug(&self, drug: &str) -> Result<Vec<DrugGeneInteraction>, PgxError>;
        fn nearest_by_response_embedding(&self, embedding: &[f32], k: usize) -> Result<Vec<DrugGeneInteraction>, PgxError>;
    }

    #[derive(Debug)]
    pub enum PgxError {
        GeneNotInPanel(String),
        AlleleNotRecognized { gene: String, allele: String },
        InsufficientCoverage(String),
        GuidelineNotFound(String),
    }
}
```

#### Domain Events

| Event                       | Payload                                      | Published When                  |
|-----------------------------|----------------------------------------------|---------------------------------|
| `StarAllelesCalled`        | sample_id, gene, diplotype, activity_score    | PGx allele calling complete     |
| `DrugInteractionIdentified`| sample_id, drug, gene, interaction_type       | Clinically relevant interaction |
| `DosingAlertGenerated`     | sample_id, drug, recommendation, urgency      | Actionable dosing change        |
| `PoorMetabolizerDetected`  | sample_id, gene, phenotype                    | PM phenotype identified         |

---

### 3.8 Population Genomics Domain

**Purpose**: Analyze cohort-level genomic data for ancestry inference, kinship
estimation, allele frequency calculation, and genome-wide association studies.

#### Aggregate Root: PopulationStudy

```rust
pub mod population_genomics {
    pub struct PopulationStudy {
        pub id: StudyId,
        pub name: String,
        pub cohort: Cohort,
        pub allele_frequencies: AlleleFrequencyTable,
        pub pca_result: Option<PcaResult>,
        pub gwas_results: Vec<GwasResult>,
        pub kinship_matrix: Option<KinshipMatrix>,
    }

    pub struct Cohort {
        pub samples: Vec<SampleId>,
        pub ancestry_composition: Vec<AncestryAssignment>,
        pub sample_count: u32,
    }

    pub struct AncestryAssignment {
        pub sample_id: SampleId,
        pub ancestry_proportions: Vec<(AncestryGroup, f32)>,
        pub principal_components: Vec<f32>,    // top PCs as embedding
    }

    pub struct AncestryGroup(pub String);  // e.g., "EUR", "AFR", "EAS"

    pub struct AlleleFrequencyTable {
        pub variant_count: u64,
        pub entries: Vec<AlleleFrequencyEntry>,
    }

    pub struct AlleleFrequencyEntry {
        pub variant_id: VariantId,
        pub global_af: f32,
        pub population_afs: Vec<(AncestryGroup, f32)>,
        pub hardy_weinberg_p: f64,
    }

    pub struct PcaResult {
        pub eigenvalues: Vec<f64>,
        pub variance_explained: Vec<f64>,
        pub sample_projections: Vec<(SampleId, Vec<f64>)>,
    }

    pub struct GwasResult {
        pub trait_name: String,
        pub variant_id: VariantId,
        pub p_value: f64,
        pub odds_ratio: f64,
        pub beta: f64,
        pub standard_error: f64,
        pub effect_embedding: Vec<f32>,   // vector for similarity search
    }

    pub struct KinshipMatrix {
        pub sample_ids: Vec<SampleId>,
        pub coefficients: Vec<Vec<f32>>,  // symmetric matrix
    }

    // --- Value Objects ---
    #[derive(Clone, Copy, PartialEq, Eq, Hash)]
    pub struct StudyId(pub u128);

    // --- Invariants ---
    // 1. ancestry_proportions sum to 1.0 per sample
    // 2. allele_frequency in [0.0, 1.0]
    // 3. hardy_weinberg_p is valid probability
    // 4. kinship_matrix is symmetric and sample_ids.len() == matrix dimension
    // 5. GWAS p-values < 5e-8 are genome-wide significant

    // --- Repository Interface ---
    pub trait PopulationRepository: Send + Sync {
        fn store_study(&self, study: &PopulationStudy) -> Result<(), PopulationError>;
        fn find_allele_freq(&self, variant: &VariantId, population: &AncestryGroup) -> Result<f32, PopulationError>;
        fn find_gwas_hits(&self, trait_name: &str, p_threshold: f64) -> Result<Vec<GwasResult>, PopulationError>;
        fn find_related_samples(&self, sample: &SampleId, kinship_threshold: f32) -> Result<Vec<(SampleId, f32)>, PopulationError>;
        fn nearest_by_ancestry_embedding(&self, pcs: &[f32], k: usize) -> Result<Vec<AncestryAssignment>, PopulationError>;
    }

    #[derive(Debug)]
    pub enum PopulationError {
        SampleNotInCohort(SampleId),
        InsufficientSampleSize { required: u32, actual: u32 },
        HardyWeinbergViolation { variant: VariantId, p_value: f64 },
    }
}
```

#### Domain Events

| Event                      | Payload                                    | Published When                   |
|----------------------------|--------------------------------------------|----------------------------------|
| `AncestryInferred`        | sample_id, ancestry_proportions, pcs        | Ancestry assignment complete     |
| `GwasSignificantHit`      | trait, variant_id, p_value, odds_ratio      | Genome-wide significant signal   |
| `AlleleFrequencyUpdated`  | variant_id, old_af, new_af, population      | New samples shift frequency      |
| `KinshipDetected`         | sample_a, sample_b, coefficient             | Related individuals found        |
| `PopulationStructureShift`| study_id, pc_variance_change                | PCA reveals new clustering       |

---

### 3.9 Pathogen Surveillance Domain

**Purpose**: Classify metagenomic reads to taxonomy, detect antimicrobial resistance
genes, and support real-time pathogen outbreak surveillance.

#### Aggregate Root: SurveillanceSample

```rust
pub mod pathogen_surveillance {
    pub struct SurveillanceSample {
        pub id: SurveillanceSampleId,
        pub sample_id: SampleId,
        pub collection_metadata: CollectionMetadata,
        pub taxonomic_profile: TaxonomicProfile,
        pub amr_detections: Vec<AmrDetection>,
        pub virulence_factors: Vec<VirulenceFactor>,
        pub outbreak_links: Vec<OutbreakLink>,
    }

    pub struct CollectionMetadata {
        pub collection_date: u64,
        pub geographic_location: GeoLocation,
        pub host_species: String,
        pub sample_type: SampleType,
        pub sequencing_platform: String,
    }

    pub struct GeoLocation {
        pub latitude: f64,
        pub longitude: f64,
        pub country: String,
        pub region: Option<String>,
    }

    #[derive(Clone, Copy, PartialEq, Eq)]
    pub enum SampleType { Clinical, Environmental, Wastewater, Food, Surveillance }

    pub struct TaxonomicProfile {
        pub classifications: Vec<TaxonomicClassification>,
        pub diversity_index: f64,        // Shannon diversity
        pub dominant_species: Option<TaxonId>,
        pub read_classification_rate: f32,
    }

    pub struct TaxonomicClassification {
        pub taxon_id: TaxonId,
        pub taxon_name: String,
        pub rank: TaxonomicRank,
        pub abundance: f32,              // relative abundance [0.0, 1.0]
        pub read_count: u64,
        pub confidence: f32,
        pub taxonomy_embedding: Vec<f32>, // hyperbolic embedding in taxonomy tree
    }

    #[derive(Clone, Copy, PartialEq, Eq, Hash)]
    pub struct TaxonId(pub u64);         // NCBI taxonomy ID

    #[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
    pub enum TaxonomicRank {
        Superkingdom, Phylum, Class, Order, Family, Genus, Species, Strain,
    }

    pub struct AmrDetection {
        pub gene_name: String,           // e.g., "blaNDM-1", "mecA"
        pub gene_family: String,         // e.g., "carbapenemase", "PBP2a"
        pub drug_class: String,          // e.g., "carbapenems", "methicillin"
        pub identity_percent: f32,
        pub coverage_percent: f32,
        pub contig_id: Option<String>,
        pub mechanism: ResistanceMechanism,
        pub clinical_relevance: ClinicalRelevance,
    }

    #[derive(Clone, Copy, PartialEq, Eq)]
    pub enum ResistanceMechanism {
        EnzymaticInactivation,
        TargetModification,
        EffluxPump,
        TargetProtection,
        ReducedPermeability,
    }

    #[derive(Clone, Copy, PartialEq, Eq)]
    pub enum ClinicalRelevance { Critical, High, Moderate, Low }

    pub struct VirulenceFactor {
        pub gene_name: String,
        pub factor_type: String,         // e.g., "toxin", "adhesin", "capsule"
        pub identity_percent: f32,
        pub source_organism: TaxonId,
    }

    pub struct OutbreakLink {
        pub linked_sample: SurveillanceSampleId,
        pub snp_distance: u32,           // core genome SNPs apart
        pub cgmlst_distance: u32,        // cgMLST allelic differences
        pub cluster_id: Option<String>,
        pub link_confidence: f32,
    }

    // --- Value Objects ---
    #[derive(Clone, Copy, PartialEq, Eq, Hash)]
    pub struct SurveillanceSampleId(pub u128);

    // --- Repository Interface ---
    pub trait SurveillanceRepository: Send + Sync {
        fn store_sample(&self, sample: &SurveillanceSample) -> Result<(), SurveillanceError>;
        fn find_by_taxon(&self, taxon: &TaxonId) -> Result<Vec<SurveillanceSample>, SurveillanceError>;
        fn find_by_amr_gene(&self, gene: &str) -> Result<Vec<SurveillanceSample>, SurveillanceError>;
        fn find_outbreak_cluster(&self, sample: &SurveillanceSampleId, snp_threshold: u32) -> Result<Vec<OutbreakLink>, SurveillanceError>;
        fn nearest_by_taxonomy_embedding(&self, embedding: &[f32], k: usize) -> Result<Vec<TaxonomicClassification>, SurveillanceError>;
        fn search_by_geolocation(&self, center: &GeoLocation, radius_km: f64) -> Result<Vec<SurveillanceSample>, SurveillanceError>;
    }

    #[derive(Debug)]
    pub enum SurveillanceError {
        TaxonNotFound(TaxonId),
        ClassificationFailed(String),
        OutbreakLinkageTimeout,
        InsufficientGenomeCoverage { required: f32, actual: f32 },
    }
}
```

#### Domain Events

| Event                       | Payload                                        | Published When                 |
|-----------------------------|------------------------------------------------|--------------------------------|
| `PathogenDetected`         | sample_id, taxon_id, abundance, confidence      | Pathogen above threshold       |
| `AmrGeneDetected`          | sample_id, gene, drug_class, relevance          | Resistance gene found          |
| `OutbreakClusterExpanded`  | cluster_id, new_sample, total_samples           | New sample joins cluster       |
| `NovelResistancePattern`   | sample_id, genes, mechanism                     | Unknown AMR combination        |
| `SurveillanceAlert`        | alert_type, severity, affected_region           | Public health alert triggered  |

---

### 3.10 CRISPR Engineering Domain

**Purpose**: Design guide RNAs for CRISPR-Cas experiments, predict off-target sites,
and score editing efficiency.

#### Aggregate Root: CrisprExperiment

```rust
pub mod crispr_engineering {
    pub struct CrisprExperiment {
        pub id: ExperimentId,
        pub target_gene: GeneId,
        pub target_region: GenomicRegion,
        pub cas_system: CasSystem,
        pub guides: Vec<GuideRna>,
        pub off_target_analysis: OffTargetAnalysis,
        pub editing_predictions: Vec<EditingPrediction>,
    }

    pub struct GuideRna {
        pub id: GuideId,
        pub spacer_sequence: Vec<u8>,        // 20-24nt guide sequence
        pub pam_sequence: Vec<u8>,           // e.g., "NGG" for SpCas9
        pub target_strand: Strand,
        pub genomic_position: GenomicPosition,
        pub on_target_score: f32,            // predicted cutting efficiency
        pub specificity_score: f32,          // 1.0 = perfectly specific
        pub gc_content: f32,
        pub secondary_structure_dg: f32,     // free energy of folding
        pub sequence_embedding: Vec<f32>,    // attention-model embedding
    }

    #[derive(Clone, Copy, PartialEq, Eq)]
    pub enum CasSystem {
        SpCas9,          // S. pyogenes Cas9, PAM = NGG
        SaCas9,          // S. aureus Cas9, PAM = NNGRRT
        Cas12a,          // Cpf1, PAM = TTTV
        Cas13,           // RNA targeting
        BasEditor,       // CBE or ABE
        PrimeEditor,     // PE2/PE3
    }

    pub struct OffTargetAnalysis {
        pub guide_id: GuideId,
        pub off_target_sites: Vec<OffTargetSite>,
        pub aggregate_off_target_score: f32,
        pub search_parameters: OffTargetSearchParams,
    }

    pub struct OffTargetSite {
        pub position: GenomicPosition,
        pub sequence: Vec<u8>,
        pub mismatches: u8,
        pub mismatch_positions: Vec<u8>,
        pub bulges: u8,
        pub cutting_probability: f32,        // model-predicted
        pub in_gene: Option<GeneId>,
        pub in_exon: bool,
        pub site_embedding: Vec<f32>,        // for similarity clustering
    }

    pub struct OffTargetSearchParams {
        pub max_mismatches: u8,
        pub max_bulges: u8,
        pub include_non_canonical_pam: bool,
        pub genome_graph_id: Option<GraphId>,  // search against pangenome
    }

    pub struct EditingPrediction {
        pub guide_id: GuideId,
        pub edit_type: EditType,
        pub predicted_outcome: EditOutcome,
        pub efficiency: f32,
        pub precision: f32,                   // fraction of desired edit among all edits
    }

    #[derive(Clone, Copy, PartialEq, Eq)]
    pub enum EditType {
        Knockout,            // indel-mediated gene disruption
        KnockIn,             // HDR-mediated insertion
        BaseEdit,            // C>T or A>G without DSB
        PrimeEdit,           // precise edit without DSB or donor
        Deletion,            // defined deletion
        Activation,          // CRISPRa
        Repression,          // CRISPRi
    }

    pub struct EditOutcome {
        pub indel_distribution: Vec<(i32, f32)>,  // (size, probability) negative=del, positive=ins
        pub frameshift_probability: f32,
        pub desired_edit_probability: f32,
    }

    // --- Value Objects ---
    #[derive(Clone, Copy, PartialEq, Eq, Hash)]
    pub struct ExperimentId(pub u128);
    #[derive(Clone, Copy, PartialEq, Eq, Hash)]
    pub struct GuideId(pub u128);

    // --- Invariants ---
    // 1. spacer_sequence.len() matches CasSystem requirements (20 for SpCas9)
    // 2. pam_sequence matches CasSystem PAM motif
    // 3. on_target_score in [0.0, 1.0]
    // 4. specificity_score in [0.0, 1.0]
    // 5. gc_content in [0.0, 1.0] and computed from spacer_sequence
    // 6. off_target_sites sorted by cutting_probability descending
    // 7. indel_distribution probabilities sum to 1.0

    // --- Repository Interface ---
    pub trait CrisprRepository: Send + Sync {
        fn store_experiment(&self, exp: &CrisprExperiment) -> Result<(), CrisprError>;
        fn find_guides_for_gene(&self, gene: &GeneId) -> Result<Vec<GuideRna>, CrisprError>;
        fn find_off_targets_in_region(&self, region: &GenomicRegion) -> Result<Vec<OffTargetSite>, CrisprError>;
        fn rank_guides(&self, guides: &[GuideId], criteria: RankingCriteria) -> Result<Vec<(GuideId, f32)>, CrisprError>;
        fn nearest_by_guide_embedding(&self, embedding: &[f32], k: usize) -> Result<Vec<GuideRna>, CrisprError>;
    }

    pub struct RankingCriteria {
        pub on_target_weight: f32,
        pub specificity_weight: f32,
        pub gc_preference: (f32, f32),      // (min, max) preferred GC range
        pub avoid_genes: Vec<GeneId>,
    }

    #[derive(Debug)]
    pub enum CrisprError {
        NoPamSiteFound(GenomicRegion),
        OffTargetSearchTimeout,
        InvalidSpacerLength { expected: usize, actual: usize },
        GenomeGraphRequired,
    }
}
```

#### Domain Events

| Event                      | Payload                                     | Published When                 |
|----------------------------|---------------------------------------------|--------------------------------|
| `GuideDesigned`           | guide_id, gene, on_target_score, gc         | New guide RNA designed         |
| `OffTargetAnalysisComplete`| guide_id, off_target_count, specificity     | Off-target search finishes     |
| `HighRiskOffTarget`       | guide_id, off_target_position, gene, prob   | Off-target in critical gene    |
| `EditingPredicted`        | guide_id, edit_type, efficiency             | Outcome prediction complete    |
| `GuideRanked`             | experiment_id, top_guide, score             | Guide ranking finalized        |

---

## 4. Context Map: Relationships & Integration Patterns

```
+=====================================================================+
|                      CONTEXT MAP                                     |
+=====================================================================+
|                                                                      |
|  [Sequence Ingestion] ----(Published Language)----> [Alignment]      |
|         |                      FASTQ/CRAM                            |
|         |                                                            |
|  [Alignment] ---------(Published Language)--------> [Variant Calling]|
|         |                      BAM/CRAM                              |
|         |                                                            |
|  [Graph Genome] <===(Shared Kernel)===> [Alignment]                  |
|         |              GenomicCoordinates                            |
|         |              ContigId, GenomicPosition                     |
|         |                                                            |
|  [Variant Calling] ----(Published Language)----> [Annotation]        |
|         |                      VCF                                   |
|         |                                                            |
|  [Annotation] ---------(Conformist)-----------> [Pharmacogenomics]   |
|         |           ClinVar/PharmGKB schema                          |
|         |                                                            |
|  [Annotation] ----(Anti-Corruption Layer)-----> [Epigenomics]        |
|         |           EpigenomeAnnotationAdapter                       |
|         |                                                            |
|  [Variant Calling] --(Anti-Corruption Layer)--> [Population Genomics]|
|         |           PopulationVariantAdapter                         |
|         |                                                            |
|  [Sequence Ingestion] --(Published Language)--> [Pathogen Surveillance]
|         |                  FASTQ                                     |
|         |                                                            |
|  [Graph Genome] ----(Anti-Corruption Layer)---> [CRISPR Engineering] |
|         |           GenomeSearchAdapter                              |
|         |                                                            |
|  [Pathogen Surveillance] --(Customer/Supplier)--> [Population Genomics]
|                           Allele frequency data                      |
|                                                                      |
+=====================================================================+
```

### 4.1 Relationship Details

| Upstream Context        | Downstream Context        | Pattern                  | Shared Artifact              |
|-------------------------|---------------------------|--------------------------|------------------------------|
| Sequence Ingestion      | Alignment & Mapping       | Published Language       | `BasecalledRead` events via FASTQ-like contract |
| Alignment & Mapping     | Variant Calling           | Published Language       | `Alignment` stream via BAM-like contract |
| Graph Genome            | Alignment & Mapping       | Shared Kernel            | `GenomicPosition`, `ContigId`, `GenomicRegion` |
| Variant Calling         | Annotation                | Published Language       | `Variant` events via VCF-like contract |
| Annotation              | Pharmacogenomics          | Conformist               | PGx context conforms to ClinVar/PharmGKB schema |
| Annotation              | Epigenomics               | Anti-Corruption Layer    | `EpigenomeAnnotationAdapter` translates variant effects to chromatin context |
| Variant Calling         | Population Genomics       | Anti-Corruption Layer    | `PopulationVariantAdapter` aggregates individual calls to cohort frequencies |
| Sequence Ingestion      | Pathogen Surveillance     | Published Language       | Raw reads for metagenomic classification |
| Graph Genome            | CRISPR Engineering        | Anti-Corruption Layer    | `GenomeSearchAdapter` provides graph-aware PAM site search |
| Pathogen Surveillance   | Population Genomics       | Customer/Supplier        | Pathogen population frequencies feed allele tables |
| Epigenomics             | Annotation                | Anti-Corruption Layer    | Regulatory annotations enriching variant interpretation |

### 4.2 Anti-Corruption Layer Definitions

```rust
/// ACL: Epigenomics <-> Annotation
pub trait EpigenomeAnnotationAdapter: Send + Sync {
    /// Translate a variant position into its epigenomic context
    fn get_regulatory_context(
        &self,
        position: &GenomicPosition,
        cell_type: &CellType,
    ) -> Result<RegulatoryContext, AdapterError>;
}

pub struct RegulatoryContext {
    pub chromatin_state: ChromatinState,
    pub methylation_level: Option<f32>,
    pub in_enhancer: bool,
    pub in_promoter: bool,
    pub tad_boundary_distance: Option<u64>,
    pub compartment: CompartmentType,
}

/// ACL: Graph Genome <-> CRISPR Engineering
pub trait GenomeSearchAdapter: Send + Sync {
    /// Search for PAM sites across all haplotypes in the pangenome graph
    fn find_pam_sites_in_graph(
        &self,
        graph_id: &GraphId,
        region: &GenomicRegion,
        pam_pattern: &[u8],
    ) -> Result<Vec<PamSiteResult>, AdapterError>;
}

pub struct PamSiteResult {
    pub position: GenomicPosition,
    pub node_id: NodeId,
    pub haplotype_count: u32,        // how many haplotypes contain this site
    pub allele_frequency: f32,
}

/// ACL: Variant Calling <-> Population Genomics
pub trait PopulationVariantAdapter: Send + Sync {
    /// Aggregate individual variant calls into population-level frequencies
    fn aggregate_to_population(
        &self,
        variants: &[Variant],
        cohort: &Cohort,
    ) -> Result<Vec<AlleleFrequencyEntry>, AdapterError>;
}

#[derive(Debug)]
pub enum AdapterError {
    UpstreamUnavailable(String),
    TranslationFailed(String),
    SchemaVersionMismatch { expected: String, actual: String },
}
```

### 4.3 Anti-Corruption Layer Patterns (SOTA Enhancement)

Beyond the adapter-level ACLs defined in Section 4.2, the DNA Analyzer employs three
strategic anti-corruption patterns to protect bounded context integrity when
interfacing with external systems and cross-context data flows.

#### Translator Pattern

External bioinformatics file formats (VCF, FASTA, GFF3, BAM/CRAM) carry legacy
semantics that do not map cleanly to the internal domain model. A dedicated
translator layer converts external representations to domain value objects at the
system boundary, ensuring no external schema leaks into the core domain.

```rust
/// Anti-Corruption Layer: External format translators
pub mod acl_translators {
    use super::*;

    /// Translator for VCF (Variant Call Format) ingestion
    pub trait VcfTranslator: Send + Sync {
        /// Parse a VCF record into the domain Variant model, rejecting
        /// records that violate domain invariants (e.g., missing REF allele)
        fn translate_record(
            &self,
            vcf_record: &RawVcfRecord,
        ) -> Result<variant_calling::Variant, TranslationError>;

        /// Batch-translate a VCF file, collecting errors per record
        fn translate_file(
            &self,
            vcf_path: &std::path::Path,
        ) -> Result<TranslationBatch, TranslationError>;
    }

    /// Translator for FASTA/FASTQ sequence ingestion
    pub trait FastaTranslator: Send + Sync {
        fn translate_record(
            &self,
            fasta_record: &RawFastaRecord,
        ) -> Result<sequence_ingestion::BasecalledRead, TranslationError>;
    }

    /// Translator for GFF3 gene annotation format
    pub trait Gff3Translator: Send + Sync {
        fn translate_feature(
            &self,
            gff3_record: &RawGff3Record,
        ) -> Result<annotation_interpretation::VariantConsequence, TranslationError>;
    }

    /// Raw external record — opaque to the domain
    pub struct RawVcfRecord {
        pub line_number: u64,
        pub raw_fields: Vec<String>,
        pub info_map: Vec<(String, String)>,
        pub format_fields: Vec<String>,
        pub sample_values: Vec<Vec<String>>,
    }

    pub struct RawFastaRecord {
        pub header: String,
        pub sequence: Vec<u8>,
        pub quality: Option<Vec<u8>>,
    }

    pub struct RawGff3Record {
        pub seqid: String,
        pub source: String,
        pub feature_type: String,
        pub start: u64,
        pub end: u64,
        pub attributes: Vec<(String, String)>,
    }

    pub struct TranslationBatch {
        pub successful: Vec<variant_calling::Variant>,
        pub failed: Vec<(u64, TranslationError)>,  // (line_number, error)
        pub warnings: Vec<(u64, String)>,
    }

    #[derive(Debug)]
    pub enum TranslationError {
        MalformedRecord { line: u64, reason: String },
        MissingRequiredField(String),
        InvariantViolation(String),
        UnsupportedVersion { expected: String, actual: String },
        EncodingError(String),
    }
}
```

#### Published Language Contracts

Cross-context communication uses strongly-typed serialization contracts defined
with Protocol Buffers or Cap'n Proto. These contracts form the Published Language
that upstream and downstream contexts agree upon, versioned independently of the
domain model.

```rust
/// Published Language: Cross-context serialization contracts
pub mod published_language {
    use super::*;

    /// Contract for Variant Calling -> Annotation data flow
    /// Versioned independently of both contexts
    #[derive(Clone)]
    pub struct VariantContract {
        pub contract_version: ContractVersion,
        pub variant_id: [u8; 16],       // UUID bytes
        pub contig: String,
        pub position: u64,
        pub ref_allele: Vec<u8>,
        pub alt_alleles: Vec<Vec<u8>>,
        pub quality: f32,
        pub genotype_indices: Vec<u8>,
        pub genotype_phased: bool,
        pub effect_embedding: Vec<f32>,
    }

    /// Contract for Sequence Ingestion -> Alignment data flow
    #[derive(Clone)]
    pub struct ReadContract {
        pub contract_version: ContractVersion,
        pub read_id: [u8; 16],
        pub sequence: Vec<u8>,
        pub quality_scores: Vec<u8>,
        pub signal_embedding: Vec<f32>,
        pub read_group: String,
        pub sample_id: String,
    }

    /// Contract for Alignment -> Variant Calling data flow
    #[derive(Clone)]
    pub struct AlignmentContract {
        pub contract_version: ContractVersion,
        pub read_id: [u8; 16],
        pub contig: String,
        pub position: u64,
        pub cigar_ops: Vec<(u8, u32)>,  // (op_code, length)
        pub mapping_quality: u8,
        pub is_primary: bool,
    }

    #[derive(Clone, Copy, PartialEq, Eq)]
    pub struct ContractVersion {
        pub major: u16,
        pub minor: u16,
    }

    /// Serialization codec trait — implementations use Protocol Buffers
    /// or Cap'n Proto depending on latency requirements
    pub trait ContractCodec<T>: Send + Sync {
        fn serialize(&self, value: &T) -> Result<Vec<u8>, CodecError>;
        fn deserialize(&self, bytes: &[u8]) -> Result<T, CodecError>;
        fn version(&self) -> ContractVersion;
    }

    #[derive(Debug)]
    pub enum CodecError {
        SerializationFailed(String),
        DeserializationFailed(String),
        VersionMismatch { expected: ContractVersion, actual: ContractVersion },
        SchemaEvolutionRequired(String),
    }
}
```

#### Conformist Pattern

The Pharmacogenomics context operates as a Conformist to the upstream Annotation
context. Rather than maintaining its own translation layer, it directly adopts
the ClinVar and PharmGKB schemas published by the Annotation context, accepting
schema changes without modification. This is appropriate because CPIC/DPWG
clinical guidelines already define the authoritative schema.

```rust
/// Conformist Pattern: Pharmacogenomics conforms to Annotation schemas
pub mod conformist_pgx {
    use super::*;

    /// PGx context directly consumes Annotation's ClinicalSignificance
    /// without translation — it is a Conformist to the upstream schema.
    pub trait PharmacogenomicConformist: Send + Sync {
        /// Accept annotation data as-is from the upstream context
        fn accept_clinical_annotation(
            &self,
            annotation: &annotation_interpretation::ClinicalAnnotation,
        ) -> Result<(), ConformistError>;

        /// Subscribe to upstream schema changes — PGx must adapt
        fn on_upstream_schema_change(
            &self,
            old_version: &str,
            new_version: &str,
        ) -> Result<MigrationPlan, ConformistError>;
    }

    pub struct MigrationPlan {
        pub breaking_changes: Vec<String>,
        pub backward_compatible: bool,
        pub migration_steps: Vec<String>,
    }

    #[derive(Debug)]
    pub enum ConformistError {
        UpstreamSchemaRejected(String),
        MigrationFailed(String),
    }
}
```

---

## 5. Event Sourcing with Genomic Event Store

Every state change across all bounded contexts is captured as an immutable
`GenomicEvent`. This event-sourcing approach provides perfect reproducibility for
clinical-grade analysis pipelines, full auditability for FDA 21 CFR Part 11
compliance, and the ability to reconstruct pipeline state at any point in time
via temporal queries.

### 5.1 Event Store Architecture

The Genomic Event Store uses an append-only log with content-addressable hashing
(Merkle tree). Each event is identified by its content hash, forming a tamper-evident
chain. This maps to `ruvector-delta-core` for incremental event streaming and
`ruvector-raft` for distributed consensus on event ordering.

```rust
/// Event Sourcing: Genomic Event Store
/// Maps to: ruvector-delta-core (streaming), ruvector-raft (ordering)
pub mod genomic_event_store {
    use super::*;
    use std::collections::BTreeMap;

    // --- Core Event Types ---

    /// Content-addressable event identifier (SHA-256 of serialized event)
    #[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
    pub struct ContentHash(pub [u8; 32]);

    /// Unique event identifier backed by content hash for tamper evidence
    #[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
    pub struct EventId(pub ContentHash);

    /// Monotonically increasing sequence number within a stream
    #[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
    pub struct SequenceNumber(pub u64);

    /// Logical timestamp for causal ordering across contexts
    #[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
    pub struct LamportTimestamp(pub u64);

    /// Wall-clock timestamp in nanoseconds since epoch
    #[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
    pub struct WallClock(pub u64);

    // --- Domain Event Trait ---

    /// Marker trait for all domain events across bounded contexts
    pub trait DomainEvent: Send + Sync + 'static {
        /// The bounded context that produced this event
        fn source_context(&self) -> &'static str;

        /// Human-readable event type name
        fn event_type(&self) -> &'static str;

        /// Aggregate root ID that this event belongs to
        fn aggregate_id(&self) -> [u8; 16];

        /// Serialize to bytes for content-addressable hashing
        fn to_bytes(&self) -> Vec<u8>;
    }

    // --- Concrete Genomic Events ---

    /// A variant was discovered by a caller
    #[derive(Clone, Debug)]
    pub struct VariantDiscovered {
        pub variant_id: [u8; 16],
        pub callset_id: [u8; 16],
        pub contig: String,
        pub position: u64,
        pub ref_allele: Vec<u8>,
        pub alt_alleles: Vec<Vec<u8>>,
        pub quality: f32,
        pub caller_name: String,
        pub caller_version: String,
    }

    /// An annotation was added or updated on a variant
    #[derive(Clone, Debug)]
    pub struct AnnotationUpdated {
        pub variant_id: [u8; 16],
        pub annotation_source: String,      // e.g., "ClinVar", "gnomAD"
        pub annotation_version: String,
        pub previous_hash: Option<ContentHash>,  // hash of prior annotation
        pub clinical_significance: Option<String>,
        pub population_frequency: Option<f32>,
        pub consequence_type: Option<String>,
    }

    /// A phasing block was extended or refined
    #[derive(Clone, Debug)]
    pub struct PhaseBlockExtended {
        pub sample_id: String,
        pub contig: String,
        pub block_start: u64,
        pub block_end: u64,
        pub variant_count: u32,
        pub phase_quality: f32,
        pub extension_type: PhaseExtensionType,
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub enum PhaseExtensionType {
        NewBlock,
        Merged,          // two blocks merged via long-read evidence
        Extended,        // block boundary pushed outward
        Refined,         // internal phasing confidence improved
    }

    /// A quality gate was triggered (pass or fail)
    #[derive(Clone, Debug)]
    pub struct QualityGateTriggered {
        pub gate_name: String,
        pub context: String,             // bounded context name
        pub aggregate_id: [u8; 16],
        pub passed: bool,
        pub metric_name: String,
        pub metric_value: f64,
        pub threshold: f64,
        pub action_taken: QualityGateAction,
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub enum QualityGateAction {
        Accepted,
        Filtered,
        Quarantined,
        Escalated,       // sent for manual review
        Requeued,        // sent back for reprocessing
    }

    // --- Event Envelope ---

    /// Immutable envelope wrapping any domain event with metadata
    #[derive(Clone, Debug)]
    pub struct EventEnvelope<T: DomainEvent> {
        pub event_id: EventId,
        pub sequence_number: SequenceNumber,
        pub lamport_timestamp: LamportTimestamp,
        pub wall_clock: WallClock,
        pub correlation_id: [u8; 16],     // traces event across contexts
        pub causation_id: Option<EventId>, // the event that caused this one
        pub source_context: &'static str,
        pub aggregate_id: [u8; 16],
        pub payload: T,
        pub merkle_proof: MerkleProof,
    }

    // --- Merkle Tree for Tamper Evidence ---

    /// Proof that an event exists in the append-only log
    #[derive(Clone, Debug)]
    pub struct MerkleProof {
        pub root_hash: ContentHash,
        pub leaf_hash: ContentHash,
        pub path: Vec<MerklePathNode>,
    }

    #[derive(Clone, Debug)]
    pub struct MerklePathNode {
        pub hash: ContentHash,
        pub direction: MerkleDirection,
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub enum MerkleDirection { Left, Right }

    // --- Event Stream ---

    /// Typed event stream for a specific aggregate or projection
    pub struct EventStream<T: DomainEvent> {
        pub stream_id: String,
        pub events: Vec<EventEnvelope<T>>,
        pub last_sequence: SequenceNumber,
        pub snapshot: Option<StreamSnapshot>,
    }

    /// Periodic snapshot to avoid replaying entire history
    #[derive(Clone, Debug)]
    pub struct StreamSnapshot {
        pub sequence_number: SequenceNumber,
        pub state_hash: ContentHash,
        pub serialized_state: Vec<u8>,
        pub created_at: WallClock,
    }

    // --- Event Store Interface ---

    /// The Genomic Event Store: append-only, content-addressable, Merkle-verified
    /// Backed by ruvector-delta-core for streaming and ruvector-raft for ordering
    pub trait GenomicEventStore: Send + Sync {
        /// Append events to a stream (returns assigned sequence numbers)
        fn append<T: DomainEvent>(
            &self,
            stream_id: &str,
            events: &[T],
            expected_version: Option<SequenceNumber>,
        ) -> Result<Vec<SequenceNumber>, EventStoreError>;

        /// Read events from a stream starting at a sequence number
        fn read_stream<T: DomainEvent>(
            &self,
            stream_id: &str,
            from: SequenceNumber,
            max_count: usize,
        ) -> Result<EventStream<T>, EventStoreError>;

        /// Read all events across all streams (for projections)
        fn read_all(
            &self,
            from: SequenceNumber,
            max_count: usize,
        ) -> Result<Vec<EventEnvelope<Box<dyn DomainEvent>>>, EventStoreError>;

        /// Temporal query: reconstruct stream state at a point in time
        fn read_at_timestamp<T: DomainEvent>(
            &self,
            stream_id: &str,
            timestamp: WallClock,
        ) -> Result<EventStream<T>, EventStoreError>;

        /// Temporal query: reconstruct stream state at a sequence number
        fn read_at_version<T: DomainEvent>(
            &self,
            stream_id: &str,
            version: SequenceNumber,
        ) -> Result<EventStream<T>, EventStoreError>;

        /// Verify Merkle proof for an event
        fn verify_proof(&self, proof: &MerkleProof) -> Result<bool, EventStoreError>;

        /// Store a snapshot for efficient replay
        fn save_snapshot(
            &self,
            stream_id: &str,
            snapshot: &StreamSnapshot,
        ) -> Result<(), EventStoreError>;

        /// Get the latest snapshot for a stream
        fn load_snapshot(
            &self,
            stream_id: &str,
        ) -> Result<Option<StreamSnapshot>, EventStoreError>;

        /// Subscribe to real-time events (for projections and process managers)
        fn subscribe(
            &self,
            filter: EventFilter,
        ) -> Result<Box<dyn EventSubscription>, EventStoreError>;
    }

    pub struct EventFilter {
        pub stream_pattern: Option<String>,  // glob pattern for stream IDs
        pub event_types: Option<Vec<String>>,
        pub source_contexts: Option<Vec<String>>,
        pub from_sequence: Option<SequenceNumber>,
    }

    pub trait EventSubscription: Send {
        fn next_event(&mut self) -> Result<EventEnvelope<Box<dyn DomainEvent>>, EventStoreError>;
        fn unsubscribe(self);
    }

    #[derive(Debug)]
    pub enum EventStoreError {
        ConcurrencyConflict {
            stream_id: String,
            expected: SequenceNumber,
            actual: SequenceNumber,
        },
        StreamNotFound(String),
        CorruptedEvent { event_id: EventId, reason: String },
        MerkleVerificationFailed(ContentHash),
        SerializationError(String),
        StorageFull,
        RaftConsensusTimeout,
    }
}
```

### 5.2 Temporal Queries

The event store supports temporal queries that reconstruct pipeline state at any
historical point. This is critical for:

- **Reproducibility**: Rerunning a variant calling pipeline with the exact
  annotation database state from a specific date
- **Audit trails**: Demonstrating to regulators (FDA 21 CFR Part 11) exactly what
  data and models produced a clinical report
- **Debugging**: Pinpointing when a variant classification changed and what
  evidence triggered the change

Temporal queries operate on two axes:
1. **Wall-clock time**: "What was the state at 2026-01-15T10:30:00Z?"
2. **Logical version**: "What was the state at sequence number 42,000?"

### 5.3 FDA 21 CFR Part 11 Compliance

The Merkle tree structure provides:
- **Tamper evidence**: Any modification to historical events invalidates the hash chain
- **Non-repudiation**: Events carry cryptographic correlation IDs linking to operator sessions
- **Complete audit trail**: The append-only log is the single source of truth for all state changes
- **Electronic signatures**: Quality gate events record operator approval with verifiable identity

---

## 6. CQRS (Command Query Responsibility Segregation)

The DNA Analyzer separates write models (optimized for streaming pipeline throughput)
from read models (optimized for clinical interpretation queries). This enables each
side to scale independently and use storage structures tailored to their access
patterns.

### 6.1 Write Model: Variant Calling Pipeline

The write side is optimized for high-throughput append-only operations. Variant
callers, aligners, and basecallers emit domain events without maintaining query
indexes. This maps to `ruvector-delta-core` for streaming writes.

### 6.2 Read Model: Clinical Interpretation Views

The read side maintains pre-materialized views optimized for clinical queries:
variant-by-gene lookups, pathway impact summaries, drug interaction matrices, and
population frequency dashboards. These views are updated asynchronously via domain
event projections. This maps to `ruvector-core` (HNSW indexes) and
`ruvector-collections` (lookup tables).

### 6.3 CQRS Types

```rust
/// CQRS: Command Query Responsibility Segregation
/// Write model maps to: ruvector-delta-core (streaming appends)
/// Read model maps to: ruvector-core (HNSW), ruvector-collections (lookup tables)
pub mod cqrs {
    use super::*;

    // --- Command Side ---

    /// Command bus dispatches write operations to the appropriate handler
    pub struct CommandBus {
        pub handlers: Vec<Box<dyn CommandHandler>>,
        pub event_store: Box<dyn genomic_event_store::GenomicEventStore>,
        pub middleware: Vec<Box<dyn CommandMiddleware>>,
    }

    /// A command represents an intent to change state
    pub trait Command: Send + Sync + 'static {
        fn command_type(&self) -> &'static str;
        fn target_aggregate(&self) -> [u8; 16];
        fn validate(&self) -> Result<(), CommandValidationError>;
    }

    /// Handler processes a command and emits domain events
    pub trait CommandHandler: Send + Sync {
        fn handles(&self) -> &'static str;  // command type
        fn handle(
            &self,
            command: &dyn Command,
        ) -> Result<Vec<Box<dyn genomic_event_store::DomainEvent>>, CommandError>;
    }

    /// Middleware for cross-cutting concerns (auth, logging, rate limiting)
    pub trait CommandMiddleware: Send + Sync {
        fn before(&self, command: &dyn Command) -> Result<(), CommandError>;
        fn after(&self, command: &dyn Command, events: &[Box<dyn genomic_event_store::DomainEvent>]) -> Result<(), CommandError>;
    }

    // --- Concrete Commands ---

    pub struct CallVariantsCommand {
        pub sample_id: String,
        pub region: GenomicRegion,
        pub caller_config: Vec<(String, String)>,
    }

    pub struct AnnotateVariantCommand {
        pub variant_id: [u8; 16],
        pub databases: Vec<String>,
        pub force_refresh: bool,
    }

    pub struct ClassifyVariantCommand {
        pub variant_id: [u8; 16],
        pub evidence_codes: Vec<String>,
        pub reviewer_id: String,
    }

    // --- Query Side ---

    /// Query projection: a read-optimized materialized view
    /// Updated asynchronously from domain events
    pub struct QueryProjection<T: Send + Sync> {
        pub projection_name: String,
        pub last_processed_sequence: genomic_event_store::SequenceNumber,
        pub state: T,
        pub refresh_policy: RefreshPolicy,
    }

    #[derive(Clone, Copy, PartialEq, Eq)]
    pub enum RefreshPolicy {
        RealTime,          // updated on every event
        BatchInterval(u64), // updated every N seconds
        OnDemand,          // updated only when queried
    }

    /// Materialized view for clinical queries
    pub struct MaterializedView {
        pub name: String,
        pub query_type: ClinicalQueryType,
        pub last_updated: genomic_event_store::WallClock,
        pub row_count: u64,
    }

    #[derive(Clone, Copy, PartialEq, Eq)]
    pub enum ClinicalQueryType {
        VariantsByGene,           // all variants in a gene, sorted by pathogenicity
        PathwayImpact,            // variants grouped by biological pathway
        DrugInteractionMatrix,    // sample x drug interaction grid
        PopulationFrequencyDash,  // variant frequencies across populations
        PharmacogenomicReport,    // per-sample PGx summary
        QualityMetricsSummary,    // pipeline QC dashboard
    }

    // --- Concrete Projections ---

    /// Variants indexed by gene for clinical lookup
    pub struct VariantsByGeneProjection {
        pub gene_index: Vec<(String, Vec<VariantSummary>)>,
    }

    pub struct VariantSummary {
        pub variant_id: [u8; 16],
        pub hgvs: String,
        pub consequence: String,
        pub acmg_class: String,
        pub population_af: f32,
        pub last_updated: genomic_event_store::WallClock,
    }

    /// Drug interaction matrix for pharmacogenomics queries
    pub struct DrugInteractionProjection {
        pub sample_id: String,
        pub interactions: Vec<DrugInteractionEntry>,
    }

    pub struct DrugInteractionEntry {
        pub drug_name: String,
        pub gene_symbol: String,
        pub metabolizer_status: String,
        pub recommendation: String,
        pub evidence_level: String,
    }

    // --- Projection Engine ---

    /// Engine that subscribes to domain events and updates projections
    pub trait ProjectionEngine: Send + Sync {
        /// Register a projection to be kept up to date
        fn register_projection(
            &mut self,
            projection_name: &str,
            handler: Box<dyn ProjectionHandler>,
        ) -> Result<(), ProjectionError>;

        /// Start processing events for all registered projections
        fn start(&self) -> Result<(), ProjectionError>;

        /// Rebuild a projection from scratch (replay all events)
        fn rebuild(&self, projection_name: &str) -> Result<(), ProjectionError>;

        /// Get current lag (events behind) for a projection
        fn lag(&self, projection_name: &str) -> Result<u64, ProjectionError>;
    }

    pub trait ProjectionHandler: Send + Sync {
        fn handle_event(
            &mut self,
            event: &genomic_event_store::EventEnvelope<Box<dyn genomic_event_store::DomainEvent>>,
        ) -> Result<(), ProjectionError>;
    }

    // --- Errors ---

    #[derive(Debug)]
    pub enum CommandValidationError {
        MissingField(String),
        InvalidValue { field: String, reason: String },
        InvariantViolation(String),
    }

    #[derive(Debug)]
    pub enum CommandError {
        ValidationFailed(CommandValidationError),
        ConcurrencyConflict(String),
        AggregateNotFound([u8; 16]),
        Unauthorized(String),
        InternalError(String),
    }

    #[derive(Debug)]
    pub enum ProjectionError {
        ProjectionNotFound(String),
        EventDeserializationFailed(String),
        StateCorrupted(String),
        RebuildInProgress(String),
    }
}
```

### 6.4 Eventual Consistency Model

Write and read models are eventually consistent. The projection engine processes
domain events asynchronously, meaning clinical query views may lag behind the latest
variant calls by milliseconds to seconds. For safety-critical queries (e.g.,
pharmacogenomic dosing alerts), the system enforces a maximum staleness bound
configured per projection.

---

## 7. Saga Pattern for Cross-Context Transactions

Genomic analysis workflows span multiple bounded contexts. The Saga Pattern
coordinates these multi-context transactions using choreography-based sagas with
compensating actions for rollback. Each saga progresses through a well-defined
state machine with timeout and dead-letter handling for stalled steps.

### 7.1 Genomic Saga State Machine

A typical whole-genome analysis saga progresses through:

```
Initiated → VariantCalled → Annotated → ClinicallyClassified → Reported
    |              |              |                |
    v              v              v                v
 (timeout)    (compensate:   (compensate:    (compensate:
              retract call)  remove annot.)  retract class.)
```

### 7.2 Saga Types

```rust
/// Saga Pattern: Cross-context transaction coordination
/// Maps to: ruvector-raft (saga state persistence), ruvector-delta-core (event routing)
pub mod saga {
    use super::*;

    // --- Saga State Machine ---

    /// Marker trait for saga state types
    pub trait SagaState: Send + Sync + Clone + 'static {
        fn state_name(&self) -> &'static str;
        fn is_terminal(&self) -> bool;
        fn is_compensating(&self) -> bool;
    }

    /// A step in the saga with forward action and compensating action
    pub struct SagaStep<S: SagaState> {
        pub step_name: String,
        pub state: S,
        pub forward_action: Box<dyn SagaAction>,
        pub compensating_action: Box<dyn CompensatingAction>,
        pub timeout: std::time::Duration,
        pub retry_policy: RetryPolicy,
    }

    /// Forward action executed during normal saga flow
    pub trait SagaAction: Send + Sync {
        fn execute(
            &self,
            context: &SagaContext,
        ) -> Result<SagaActionResult, SagaError>;
    }

    /// Compensating action executed during rollback
    pub trait CompensatingAction: Send + Sync {
        fn compensate(
            &self,
            context: &SagaContext,
            original_result: &SagaActionResult,
        ) -> Result<(), SagaError>;
    }

    pub struct SagaActionResult {
        pub events_produced: Vec<Box<dyn genomic_event_store::DomainEvent>>,
        pub data: Vec<(String, Vec<u8>)>,  // key-value data for subsequent steps
    }

    pub struct SagaContext {
        pub saga_id: SagaId,
        pub correlation_id: [u8; 16],
        pub data: Vec<(String, Vec<u8>)>,  // accumulated data from prior steps
        pub started_at: genomic_event_store::WallClock,
    }

    // --- Genomic Saga ---

    /// A complete genomic analysis saga
    pub struct GenomicSaga<S: SagaState> {
        pub id: SagaId,
        pub name: String,
        pub current_state: S,
        pub steps: Vec<SagaStep<S>>,
        pub completed_steps: Vec<CompletedStep>,
        pub context: SagaContext,
        pub status: SagaStatus,
        pub created_at: genomic_event_store::WallClock,
        pub updated_at: genomic_event_store::WallClock,
    }

    #[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
    pub struct SagaId(pub [u8; 16]);

    #[derive(Clone, Copy, PartialEq, Eq, Debug)]
    pub enum SagaStatus {
        Running,
        Completed,
        Compensating,      // rolling back
        Failed,            // compensation failed
        TimedOut,
        DeadLettered,      // sent to dead-letter queue for manual review
    }

    pub struct CompletedStep {
        pub step_name: String,
        pub result: SagaActionResult,
        pub completed_at: genomic_event_store::WallClock,
        pub duration_ms: u64,
    }

    // --- Whole-Genome Analysis Saga States ---

    #[derive(Clone, Debug)]
    pub enum WholeGenomeSagaState {
        Initiated,
        VariantsCalled,
        Annotated,
        ClinicallyClassified,
        Reported,
        // Compensating states
        RetractingCalls,
        RemovingAnnotations,
        RetractingClassification,
        CompensationComplete,
    }

    impl SagaState for WholeGenomeSagaState {
        fn state_name(&self) -> &'static str {
            match self {
                Self::Initiated => "initiated",
                Self::VariantsCalled => "variants_called",
                Self::Annotated => "annotated",
                Self::ClinicallyClassified => "clinically_classified",
                Self::Reported => "reported",
                Self::RetractingCalls => "retracting_calls",
                Self::RemovingAnnotations => "removing_annotations",
                Self::RetractingClassification => "retracting_classification",
                Self::CompensationComplete => "compensation_complete",
            }
        }

        fn is_terminal(&self) -> bool {
            matches!(self, Self::Reported | Self::CompensationComplete)
        }

        fn is_compensating(&self) -> bool {
            matches!(
                self,
                Self::RetractingCalls
                    | Self::RemovingAnnotations
                    | Self::RetractingClassification
                    | Self::CompensationComplete
            )
        }
    }

    // --- Retry and Timeout ---

    #[derive(Clone, Debug)]
    pub struct RetryPolicy {
        pub max_retries: u32,
        pub base_delay_ms: u64,
        pub backoff_multiplier: f64,
        pub max_delay_ms: u64,
    }

    // --- Saga Orchestrator ---

    /// Orchestrates saga execution, compensation, and dead-lettering
    pub trait SagaOrchestrator: Send + Sync {
        /// Start a new saga
        fn start_saga<S: SagaState>(
            &self,
            saga: GenomicSaga<S>,
        ) -> Result<SagaId, SagaError>;

        /// Advance a saga to its next step
        fn advance(&self, saga_id: &SagaId) -> Result<SagaStatus, SagaError>;

        /// Trigger compensation (rollback) from the current step
        fn compensate(&self, saga_id: &SagaId, reason: &str) -> Result<(), SagaError>;

        /// Move a stalled saga to the dead-letter queue
        fn dead_letter(&self, saga_id: &SagaId, reason: &str) -> Result<(), SagaError>;

        /// List all sagas in a given status
        fn list_by_status(&self, status: SagaStatus) -> Result<Vec<SagaId>, SagaError>;

        /// Get current saga state
        fn get_state<S: SagaState>(
            &self,
            saga_id: &SagaId,
        ) -> Result<GenomicSaga<S>, SagaError>;
    }

    #[derive(Debug)]
    pub enum SagaError {
        SagaNotFound(SagaId),
        StepExecutionFailed { step: String, reason: String },
        CompensationFailed { step: String, reason: String },
        TimeoutExceeded { step: String, timeout_ms: u64 },
        InvalidStateTransition { from: String, to: String },
        DeadLettered(SagaId),
        PersistenceError(String),
    }
}
```

### 7.3 Compensating Transactions

Each saga step defines a compensating action for rollback:

| Step                    | Forward Action                     | Compensating Action                  |
|-------------------------|------------------------------------|--------------------------------------|
| VariantCalling          | Call variants from alignments      | Retract variant calls, emit `VariantRetracted` event |
| Annotation              | Annotate called variants           | Remove annotations, emit `AnnotationRetracted` event |
| ClinicalClassification  | Assign ACMG class with evidence    | Retract classification, emit `ClassificationRetracted` event |
| Reporting               | Generate clinical report           | Mark report as superseded, emit `ReportRetracted` event |

---

## 8. Hexagonal Architecture (Ports & Adapters)

Each bounded context exposes ports (trait interfaces) and adapters (concrete
implementations). This enables testing with real adapter swaps instead of mocks,
and allows infrastructure changes (e.g., switching from FPGA to GPU basecalling)
without modifying domain logic.

### 8.1 Port Definitions

```rust
/// Hexagonal Architecture: Ports & Adapters
/// Each bounded context defines primary ports (driving) and secondary ports (driven)
pub mod hexagonal {
    use super::*;

    // --- Primary Ports (Driving Side) ---
    // These are called BY external actors (CLI, API, UI, pipeline orchestrator)

    /// Primary port: Sequence ingestion entry point
    pub trait SequenceIngestionPort: Send + Sync {
        /// Ingest raw signal from a sequencing instrument
        fn ingest_signal(
            &self,
            signal: &[f32],
            instrument: InstrumentConfig,
        ) -> Result<Vec<sequence_ingestion::BasecalledRead>, PortError>;

        /// Ingest pre-basecalled reads (e.g., from FASTQ)
        fn ingest_reads(
            &self,
            reads: Vec<sequence_ingestion::BasecalledRead>,
        ) -> Result<u64, PortError>;

        /// Start a streaming ingestion session
        fn start_streaming_session(
            &self,
            config: StreamingConfig,
        ) -> Result<SessionHandle, PortError>;
    }

    /// Primary port: Variant calling entry point
    pub trait VariantCallingPort: Send + Sync {
        /// Call variants from aligned reads in a region
        fn call_variants(
            &self,
            sample_id: &str,
            region: &GenomicRegion,
            config: CallerConfig,
        ) -> Result<variant_calling::VariantCallSet, PortError>;

        /// Incrementally update calls with new alignment data
        fn update_calls(
            &self,
            callset_id: &[u8; 16],
            new_alignments: &[alignment_mapping::Alignment],
        ) -> Result<Vec<variant_calling::Variant>, PortError>;
    }

    /// Primary port: Annotation and interpretation entry point
    pub trait AnnotationPort: Send + Sync {
        /// Annotate a single variant
        fn annotate_variant(
            &self,
            variant_id: &[u8; 16],
        ) -> Result<annotation_interpretation::AnnotatedVariant, PortError>;

        /// Batch annotate a callset
        fn annotate_callset(
            &self,
            callset_id: &[u8; 16],
        ) -> Result<Vec<annotation_interpretation::AnnotatedVariant>, PortError>;

        /// Query clinically significant variants for a sample
        fn query_clinical_variants(
            &self,
            sample_id: &str,
            min_significance: &str,
        ) -> Result<Vec<annotation_interpretation::AnnotatedVariant>, PortError>;
    }

    // --- Secondary Ports (Driven Side) ---
    // These are called BY the domain to interact with infrastructure

    /// Secondary port: Genome storage (implemented by ruvector-core, filesystem, S3, etc.)
    pub trait GenomeStoragePort: Send + Sync {
        /// Store a genomic data blob (sequence, signal, alignment)
        fn store(&self, key: &StorageKey, data: &[u8]) -> Result<(), PortError>;

        /// Retrieve a genomic data blob
        fn retrieve(&self, key: &StorageKey) -> Result<Vec<u8>, PortError>;

        /// Delete a genomic data blob
        fn delete(&self, key: &StorageKey) -> Result<(), PortError>;

        /// Check if a key exists
        fn exists(&self, key: &StorageKey) -> Result<bool, PortError>;

        /// List keys matching a prefix
        fn list_prefix(&self, prefix: &str) -> Result<Vec<StorageKey>, PortError>;
    }

    /// Secondary port: Reference data access (genome references, annotation DBs)
    pub trait ReferenceDataPort: Send + Sync {
        /// Load a reference genome or graph
        fn load_reference(
            &self,
            reference_id: &str,
            region: Option<&GenomicRegion>,
        ) -> Result<ReferenceData, PortError>;

        /// Query an annotation database (ClinVar, gnomAD, etc.)
        fn query_database(
            &self,
            database_name: &str,
            query: &DatabaseQuery,
        ) -> Result<Vec<DatabaseRecord>, PortError>;

        /// Get database version/timestamp
        fn database_version(&self, database_name: &str) -> Result<String, PortError>;
    }

    /// Secondary port: Audit logging (for compliance and traceability)
    pub trait AuditLogPort: Send + Sync {
        /// Log a domain event for audit purposes
        fn log_event(
            &self,
            entry: &AuditEntry,
        ) -> Result<(), PortError>;

        /// Query audit log by time range and context
        fn query_log(
            &self,
            filter: &AuditFilter,
        ) -> Result<Vec<AuditEntry>, PortError>;

        /// Verify audit log integrity (Merkle chain)
        fn verify_integrity(
            &self,
            from: u64,
            to: u64,
        ) -> Result<bool, PortError>;
    }

    // --- Supporting Types ---

    pub struct InstrumentConfig {
        pub instrument_type: String,
        pub basecaller_model: String,
        pub quality_threshold: f32,
    }

    pub struct StreamingConfig {
        pub chunk_size: usize,
        pub max_buffer_size: usize,
        pub flush_interval_ms: u64,
    }

    pub struct SessionHandle {
        pub session_id: [u8; 16],
        pub created_at: u64,
    }

    pub struct CallerConfig {
        pub caller_name: String,
        pub parameters: Vec<(String, String)>,
    }

    #[derive(Clone)]
    pub struct StorageKey(pub String);

    pub struct ReferenceData {
        pub reference_id: String,
        pub sequence: Option<Vec<u8>>,
        pub graph: Option<graph_genome::ContigGraph>,
    }

    pub struct DatabaseQuery {
        pub query_type: String,
        pub parameters: Vec<(String, String)>,
        pub max_results: usize,
    }

    pub struct DatabaseRecord {
        pub fields: Vec<(String, String)>,
    }

    pub struct AuditEntry {
        pub timestamp: u64,
        pub context: String,
        pub action: String,
        pub actor: String,
        pub details: Vec<(String, String)>,
        pub event_hash: Option<genomic_event_store::ContentHash>,
    }

    pub struct AuditFilter {
        pub from_timestamp: Option<u64>,
        pub to_timestamp: Option<u64>,
        pub context: Option<String>,
        pub actor: Option<String>,
    }

    #[derive(Debug)]
    pub enum PortError {
        NotFound(String),
        StorageError(String),
        ConnectionError(String),
        AuthorizationError(String),
        TimeoutError(String),
        ValidationError(String),
    }
}
```

### 8.2 Adapter Implementations

```rust
/// Hexagonal Architecture: Concrete adapters
/// These implement the ports using specific RuVector crates
pub mod adapters {
    use super::*;

    /// Adapter: FPGA-accelerated basecalling (implements SequenceIngestionPort)
    /// Maps to: sona (neural basecalling with FPGA offload)
    pub struct FpgaBasecallingAdapter {
        pub device_id: u32,
        pub model_path: String,
        pub batch_size: usize,
        pub precision: ComputePrecision,
    }

    #[derive(Clone, Copy, PartialEq, Eq)]
    pub enum ComputePrecision { Fp32, Fp16, Int8 }

    /// Adapter: HNSW-backed vector search (implements nearest-neighbor queries)
    /// Maps to: ruvector-core (HNSW index)
    pub struct HnswSearchAdapter {
        pub index_name: String,
        pub ef_construction: usize,
        pub ef_search: usize,
        pub max_connections: usize,
        pub distance_metric: DistanceMetric,
    }

    #[derive(Clone, Copy, PartialEq, Eq)]
    pub enum DistanceMetric { Cosine, Euclidean, DotProduct, Hyperbolic }

    /// Adapter: Delta-based storage (implements GenomeStoragePort)
    /// Maps to: ruvector-delta-core (incremental storage)
    pub struct DeltaStorageAdapter {
        pub base_path: String,
        pub compression: CompressionType,
        pub tier_policy: TierPolicy,
    }

    #[derive(Clone, Copy, PartialEq, Eq)]
    pub enum CompressionType { None, Lz4, Zstd, Snappy }

    #[derive(Clone)]
    pub struct TierPolicy {
        pub hot_retention_hours: u32,
        pub warm_retention_days: u32,
        pub cold_storage_backend: String,
    }

    /// Adapter: Raft-replicated audit log (implements AuditLogPort)
    /// Maps to: ruvector-raft (distributed consensus)
    pub struct RaftAuditLogAdapter {
        pub cluster_nodes: Vec<String>,
        pub replication_factor: u32,
        pub flush_interval_ms: u64,
    }

    /// Adapter: GPU-accelerated attention model (for CRISPR off-target prediction)
    /// Maps to: ruvector-attention (transformer inference)
    pub struct GpuAttentionAdapter {
        pub device_id: u32,
        pub model_path: String,
        pub max_sequence_length: usize,
        pub attention_heads: usize,
    }
}
```

### 8.3 Adapter Swap for Testing

The hexagonal architecture enables testing by swapping adapters:

| Production Adapter            | Test Adapter                | Port                    |
|-------------------------------|-----------------------------|-------------------------|
| `FpgaBasecallingAdapter`      | `InMemoryBasecallingAdapter`| `SequenceIngestionPort` |
| `HnswSearchAdapter`           | `BruteForceSearchAdapter`   | Nearest-neighbor queries|
| `DeltaStorageAdapter`          | `InMemoryStorageAdapter`    | `GenomeStoragePort`     |
| `RaftAuditLogAdapter`          | `VecAuditLogAdapter`        | `AuditLogPort`          |
| `GpuAttentionAdapter`          | `CpuAttentionAdapter`       | Attention inference     |

---

## 9. Domain Event Choreography with Process Managers

Process managers coordinate long-running cross-context workflows by listening to
domain events and issuing commands to advance the workflow. Unlike sagas (which
handle compensation), process managers focus on orchestrating the happy path with
explicit transition guards and coherence gate checkpoints.

### 9.1 Whole-Genome Analysis Process Manager

The `WholeGenomeAnalysisProcessManager` orchestrates the complete pipeline from
raw sequencing signal to final clinical report. It listens for domain events from
each bounded context and issues commands to the next context when preconditions
(transition guards) are satisfied.

```rust
/// Process Managers: Long-running cross-context workflow coordination
/// Maps to: ruvector-raft (state persistence), ruvector-delta-core (event routing)
pub mod process_managers {
    use super::*;

    /// Process Manager for whole-genome analysis pipeline
    pub struct WholeGenomeAnalysisProcessManager {
        pub id: ProcessManagerId,
        pub sample_id: String,
        pub current_phase: AnalysisPhase,
        pub phase_results: Vec<PhaseResult>,
        pub coherence_gates: Vec<CoherenceGate>,
        pub started_at: genomic_event_store::WallClock,
        pub timeout: std::time::Duration,
    }

    #[derive(Clone, Copy, PartialEq, Eq, Debug)]
    pub enum AnalysisPhase {
        AwaitingSignal,
        Basecalling,
        QualityControl,
        Aligning,
        VariantCalling,
        Annotating,
        ClinicalClassification,
        PharmacogenomicProfiling,
        ReportGeneration,
        Complete,
        Failed,
    }

    /// Process Manager for multi-sample population study
    pub struct PopulationStudyProcessManager {
        pub id: ProcessManagerId,
        pub study_name: String,
        pub sample_count: u32,
        pub samples_processed: u32,
        pub current_phase: PopulationPhase,
        pub federated_nodes: Vec<String>,
        pub aggregation_state: AggregationState,
    }

    #[derive(Clone, Copy, PartialEq, Eq, Debug)]
    pub enum PopulationPhase {
        SampleCollection,
        IndividualAnalysis,
        AlleleFrequencyAggregation,
        PcaComputation,
        GwasAnalysis,
        ResultsConsolidation,
        Complete,
    }

    #[derive(Clone, Debug)]
    pub struct AggregationState {
        pub samples_received: u32,
        pub samples_expected: u32,
        pub partial_frequencies: Vec<(String, f32)>,  // (variant_key, running_af)
    }

    // --- Supporting Types ---

    #[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
    pub struct ProcessManagerId(pub [u8; 16]);

    pub struct PhaseResult {
        pub phase: AnalysisPhase,
        pub completed_at: genomic_event_store::WallClock,
        pub duration_ms: u64,
        pub metrics: Vec<(String, f64)>,
        pub events_produced: u32,
    }

    /// Coherence gate: a checkpoint that must pass before phase transition
    pub struct CoherenceGate {
        pub gate_name: String,
        pub phase_transition: (AnalysisPhase, AnalysisPhase),
        pub guard: Box<dyn TransitionGuard>,
        pub on_failure: GateFailureAction,
    }

    /// Guard that evaluates whether a phase transition is allowed
    pub trait TransitionGuard: Send + Sync {
        fn evaluate(&self, context: &ProcessManagerContext) -> Result<bool, ProcessManagerError>;
        fn description(&self) -> &str;
    }

    pub struct ProcessManagerContext {
        pub sample_id: String,
        pub phase_results: Vec<PhaseResult>,
        pub accumulated_metrics: Vec<(String, f64)>,
    }

    #[derive(Clone, Copy, PartialEq, Eq, Debug)]
    pub enum GateFailureAction {
        Block,             // halt the pipeline
        Retry,             // retry the previous phase
        Skip,              // skip and continue (with warning)
        Escalate,          // send for manual review
    }

    // --- Process Manager Trait ---

    /// Core process manager interface
    pub trait ProcessManager: Send + Sync {
        /// Handle an incoming domain event and decide next action
        fn handle_event(
            &mut self,
            event: &genomic_event_store::EventEnvelope<Box<dyn genomic_event_store::DomainEvent>>,
        ) -> Result<Vec<ProcessManagerAction>, ProcessManagerError>;

        /// Get current phase
        fn current_phase(&self) -> &str;

        /// Check if the process is complete
        fn is_complete(&self) -> bool;

        /// Get accumulated results
        fn results(&self) -> &[PhaseResult];
    }

    pub enum ProcessManagerAction {
        /// Issue a command to a bounded context
        IssueCommand(Box<dyn cqrs::Command>),
        /// Schedule a delayed action (timeout handling)
        ScheduleTimeout { delay_ms: u64, action: String },
        /// Emit a process-level event
        EmitEvent(Box<dyn genomic_event_store::DomainEvent>),
        /// No action needed for this event
        NoAction,
    }

    #[derive(Debug)]
    pub enum ProcessManagerError {
        InvalidPhaseTransition { from: String, to: String },
        CoherenceGateFailed { gate: String, reason: String },
        TimeoutExceeded { phase: String, timeout_ms: u64 },
        EventHandlingFailed(String),
        PersistenceError(String),
    }
}
```

### 9.2 Transition Guards

| Gate Name                 | Transition                           | Condition                                     |
|---------------------------|--------------------------------------|-----------------------------------------------|
| `MinCoverageGate`         | QualityControl -> Aligning           | Mean coverage >= 30x for WGS                  |
| `MappingRateGate`         | Aligning -> VariantCalling           | Mapping rate >= 95%                            |
| `TiTvRatioGate`           | VariantCalling -> Annotating         | Ti/Tv ratio in [1.8, 2.3] for WGS             |
| `AnnotationCompletenessGate` | Annotating -> ClinicalClassification | >= 99% of variants annotated                |
| `PgxCoverageGate`         | ClinicalClassification -> PgxProfiling | All CPIC Level 1A genes at >= 20x coverage  |
| `ReportSignoffGate`       | ReportGeneration -> Complete         | Clinical geneticist sign-off recorded          |

---

## 10. Aggregate Consistency Boundaries

Each aggregate root defines an explicit consistency boundary. Within a single
aggregate, all invariants are enforced transactionally. Cross-aggregate communication
uses eventual consistency via domain events. Optimistic concurrency with version
vectors prevents lost updates.

### 10.1 Consistency Rules

```rust
/// Aggregate Consistency: Version vectors and optimistic concurrency
/// Maps to: ruvector-delta-consensus (version tracking), ruvector-raft (distributed locking)
pub mod aggregate_consistency {
    use super::*;

    /// Version vector for optimistic concurrency control
    #[derive(Clone, Debug, PartialEq, Eq)]
    pub struct VersionVector {
        pub aggregate_id: [u8; 16],
        pub version: u64,
        pub last_event_id: Option<genomic_event_store::EventId>,
        pub last_modified_by: String,    // actor/context that last modified
        pub last_modified_at: genomic_event_store::WallClock,
    }

    /// Concurrency control for aggregate updates
    pub trait AggregateRoot: Send + Sync {
        /// Get the current version vector
        fn version(&self) -> &VersionVector;

        /// Apply an event, incrementing the version
        fn apply_event(
            &mut self,
            event: &dyn genomic_event_store::DomainEvent,
        ) -> Result<VersionVector, ConsistencyError>;

        /// Check invariants after applying an event
        fn check_invariants(&self) -> Result<(), ConsistencyError>;
    }

    /// Conflict resolution strategies for cross-aggregate eventual consistency
    #[derive(Clone, Copy, PartialEq, Eq, Debug)]
    pub enum ConflictResolution {
        /// Last writer wins (for annotations where latest data is authoritative)
        LastWriterWins,
        /// Merge concurrent changes (for variant calls where both may be valid)
        Merge,
        /// Reject conflicting updates (for clinical classifications requiring review)
        Reject,
        /// Custom resolution function
        Custom,
    }

    /// Configuration per aggregate type
    pub struct AggregateConsistencyConfig {
        pub aggregate_type: String,
        pub conflict_resolution: ConflictResolution,
        pub max_events_before_snapshot: u64,
        pub staleness_bound_ms: Option<u64>,  // for CQRS read models
    }

    /// Cross-aggregate consistency rules
    pub struct EventualConsistencyContract {
        pub source_aggregate: String,
        pub target_aggregate: String,
        pub propagation_event: String,
        pub max_propagation_delay_ms: u64,
        pub conflict_resolution: ConflictResolution,
        pub idempotency_key: String,
    }

    #[derive(Debug)]
    pub enum ConsistencyError {
        OptimisticConcurrencyViolation {
            aggregate_id: [u8; 16],
            expected_version: u64,
            actual_version: u64,
        },
        InvariantViolation(String),
        ConflictDetected {
            aggregate_id: [u8; 16],
            conflicting_events: Vec<genomic_event_store::EventId>,
        },
        StaleRead {
            aggregate_id: [u8; 16],
            staleness_ms: u64,
            bound_ms: u64,
        },
    }
}
```

### 10.2 Per-Context Consistency Boundaries

| Bounded Context         | Aggregate Root        | Transaction Scope            | Cross-Aggregate Strategy      |
|-------------------------|-----------------------|------------------------------|-------------------------------|
| Sequence Ingestion      | `SequencingRun`       | Single run + its read groups | Eventual: `RunComplete` event |
| Alignment & Mapping     | `AlignmentBatch`      | Single batch of alignments   | Eventual: `AlignmentCompleted` event |
| Variant Calling         | `VariantCallSet`      | Single sample's callset      | Merge: concurrent callers produce union |
| Graph Genome            | `GenomeGraph`         | Single contig graph          | Eventual: `GraphUpdated` event |
| Annotation              | `AnnotatedVariant`    | Single variant annotation    | Last-writer-wins: latest DB is authoritative |
| Epigenomics             | `EpigenomicProfile`   | Single sample profile        | Eventual: `MethylationProfiled` event |
| Pharmacogenomics        | `PharmacogenomicProfile` | Single sample PGx profile | Eventual: `StarAllelesCalled` event |
| Population Genomics     | `PopulationStudy`     | Single study cohort          | Merge: federated frequency aggregation |
| Pathogen Surveillance   | `SurveillanceSample`  | Single surveillance sample   | Eventual: `OutbreakClusterExpanded` event |
| CRISPR Engineering      | `CrisprExperiment`    | Single experiment            | Reject: guide designs require explicit review |

---

## 11. Domain Event Flow (Pipeline)

The complete event-driven pipeline flows as follows:

```
   +-----------------------+
   | Instrument Signal     |
   +-----------+-----------+
               |
               v
   +------------------------+     +------------------------+
   | SignalChunkReceived     |     | RunStarted             |
   +------------------------+     +------------------------+
               |
               v
   +------------------------+
   | ReadBasecalled          |---+
   +------------------------+   |
               |                |
               v                v
   +------------------------+  +-----------------------------+
   | AlignmentCompleted      |  | (Pathogen Surveillance)     |
   +------------------------+  | PathogenDetected            |
               |               | AmrGeneDetected             |
               v               | OutbreakClusterExpanded     |
   +------------------------+  +-----------------------------+
   | VariantCalled           |
   | StructuralVariantFound  |
   | PhasingCompleted        |
   +----------+-------------+
              |
     +--------+--------+-------------------+
     v                  v                   v
+----------------+ +------------------+ +---------------------+
| VariantAnnotated| | AlleleFrequency  | | MethylationProfiled |
| PathogenicFound | | Updated          | | TadBoundaryDisrupted|
| AcmgReclassified| | GwasSignificant  | +---------------------+
+--------+-------+ | Hit              |
         |         | AncestryInferred |
         v         +------------------+
+-------------------+
| StarAllelesCalled  |
| DrugInteraction    |
| Identified         |
| DosingAlert        |
| Generated          |
+-------------------+

CRISPR Engineering operates on-demand:
  GraphConstructed + VariantAnnotated --> GuideDesigned
  GuideDesigned --> OffTargetAnalysisComplete
  OffTargetAnalysisComplete --> EditingPredicted --> GuideRanked
```

---

## 12. Mapping to RuVector Crates

Each bounded context maps to specific RuVector infrastructure crates:

```
+===========================================================================+
|  BOUNDED CONTEXT             |  PRIMARY RUVECTOR CRATES                   |
+===========================================================================+
|                              |                                            |
|  1. Sequence Ingestion       |  sona              - adaptive basecalling  |
|                              |  ruvector-core      - read embedding store |
|                              |  ruvector-delta-*   - incremental updates  |
|                              |  ruvector-temporal-tensor - signal windows |
|                              |                                            |
|  2. Alignment & Mapping      |  ruvector-core      - seed index (HNSW)   |
|                              |  ruvector-graph     - graph alignment      |
|                              |  ruvector-mincut    - graph partitioning   |
|                              |  ruvector-dag       - alignment DAG        |
|                              |                                            |
|  3. Variant Calling          |  ruvector-gnn       - variant effect pred. |
|                              |  ruvector-core      - variant embeddings   |
|                              |  ruvector-delta-core- incremental calling  |
|                              |  ruvector-sparse-inference - genotyper     |
|                              |                                            |
|  4. Graph Genome             |  ruvector-graph     - genome graph store   |
|                              |  ruvector-mincut    - min-cut partitioning |
|                              |  ruvector-dag       - variant DAGs         |
|                              |  cognitum-gate-kernel - graph sharding     |
|                              |  ruvector-delta-graph - incremental graphs |
|                              |                                            |
|  5. Annotation               |  ruvector-gnn       - effect prediction   |
|                              |  ruvector-core      - annotation vectors  |
|                              |  ruvector-attention  - consequence pred.  |
|                              |  ruvector-collections - lookup tables     |
|                              |                                            |
|  6. Epigenomics              |  ruvector-temporal-tensor - time-series   |
|                              |  ruvector-core      - methylation vectors |
|                              |  ruvector-graph     - Hi-C contact graphs |
|                              |  ruvector-attention  - 3D structure pred. |
|                              |                                            |
|  7. Pharmacogenomics         |  sona               - drug response pred. |
|                              |  ruvector-core      - PGx embeddings      |
|                              |  ruvector-gnn       - interaction graphs  |
|                              |  ruvector-sparse-inference - allele call  |
|                              |                                            |
|  8. Population Genomics      |  ruvector-core      - PCA embeddings      |
|                              |  ruvector-cluster   - ancestry clustering |
|                              |  ruvector-math      - statistics/PCA      |
|                              |  ruvector-delta-consensus - cohort sync   |
|                              |                                            |
|  9. Pathogen Surveillance    |  ruvector-hyperbolic-hnsw - taxonomy tree |
|                              |  ruvector-core      - pathogen vectors    |
|                              |  ruvector-cluster   - outbreak clustering |
|                              |  ruvector-graph     - transmission graphs |
|                              |  ruvector-raft      - distributed sync    |
|                              |                                            |
|  10. CRISPR Engineering      |  ruvector-attention  - off-target model   |
|                              |  cognitum-gate-kernel - gated seq. attn.  |
|                              |  ruvector-graph     - pangenome search    |
|                              |  ruvector-core      - guide embeddings    |
|                              |  ruvector-mincut    - graph-aware search  |
|                              |                                            |
+===========================================================================+
```

### 12.1 Crate Mapping Rationale

**ruvector-core** serves as the foundational vector storage layer across all ten
contexts. Every entity with an embedding field (reads, variants, guides, taxonomy
nodes, drug responses) stores its vectors through ruvector-core's HNSW index, enabling
sub-millisecond approximate nearest neighbor queries. This is the universal
infrastructure crate.

**sona** (Self-Optimizing Neural Architecture) drives two key functions:
1. *Sequence Ingestion*: Adaptive basecalling with two-tier LoRA fine-tuning. The
   `MicroLoRA` layer adapts per-flowcell, while `BaseLoRA` captures instrument-level
   patterns. EWC++ prevents catastrophic forgetting across runs.
2. *Pharmacogenomics*: Drug response prediction using the `ReasoningBank` to accumulate
   pharmacological evidence and `TrajectoryBuffer` to track patient outcome trajectories.

**ruvector-mincut** powers two contexts:
1. *Graph Genome*: The `SubpolynomialMinCut` algorithm partitions pangenome graphs
   into balanced components for parallel alignment. `HierarchicalDecomposition`
   enables multi-resolution graph traversal.
2. *CRISPR Engineering*: Min-cut analysis identifies structural boundaries in the
   genome graph that affect guide specificity across haplotypes.

**ruvector-gnn** provides Graph Neural Network inference for:
1. *Variant Calling*: Predicting variant effect vectors from local graph topology
   around variant sites. The GNN operates on the alignment pileup graph.
2. *Annotation*: Predicting functional consequences and pathogenicity scores using
   gene interaction networks as input graphs.
3. *Pharmacogenomics*: Modeling drug-gene-variant interaction networks.

**ruvector-attention** with its `ScaledDotProductAttention`, MoE router, and sparse
attention masks serves:
1. *Annotation*: Transformer-based consequence prediction attending to protein sequence
   context windows around variant positions.
2. *Epigenomics*: Attention over Hi-C contact matrices for 3D genome structure
   prediction.
3. *CRISPR Engineering*: Gated attention models for off-target prediction, with the
   guide sequence as query attending to candidate genomic sites as keys.

**ruvector-hyperbolic-hnsw** is purpose-built for the *Pathogen Surveillance* context.
Taxonomic trees are naturally hierarchical, and hyperbolic space embeddings
(Poincare ball model) preserve tree distances with exponentially less distortion than
Euclidean space. The `ShardedHyperbolicHnsw` partitions the taxonomy across curvature
regions, and `DualSpaceIndex` enables both Euclidean sequence-similarity and hyperbolic
taxonomy-distance queries.

**cognitum-gate-kernel** provides the gated graph attention mechanism used in:
1. *Graph Genome*: The `CompactGraph` structure with `ShardEdge` and `VertexEntry`
   maps directly to genome graph shards. The `EvidenceAccumulator` tracks alignment
   evidence across graph bubbles.
2. *CRISPR Engineering*: Gated attention over sequence-PAM interactions.

**ruvector-dag** models dependency structures in:
1. *Alignment*: The `QueryDag` represents multi-seed alignment chains as DAGs.
   `TopologicalIterator` orders chain extensions.
2. *Variant Calling*: Variant dependency DAGs where structural variants may encompass
   smaller variants. `MinCutResult` identifies independent variant blocks.
3. *Graph Genome*: Bubble nesting hierarchies as DAGs.

**ruvector-delta-*** crates enable incremental processing:
1. *Sequence Ingestion*: `ruvector-delta-core` streams basecalling deltas as new
   signal chunks arrive, using `DeltaWindow` for batched processing.
2. *Graph Genome*: `ruvector-delta-graph` propagates graph updates when new population
   data is incorporated without full reconstruction.
3. *Population Genomics*: `ruvector-delta-consensus` synchronizes allele frequency
   updates across distributed cohort nodes via Raft consensus.

**ruvector-temporal-tensor** stores time-series data:
1. *Sequence Ingestion*: Raw signal windows with tiered storage (hot/warm/cold) via
   `TierPolicy` and `BlockMeta`.
2. *Epigenomics*: Temporal methylation profiles tracking changes across cell
   differentiation or treatment time courses.

**ruvector-sparse-inference** provides lightweight neural inference:
1. *Variant Calling*: The `SparseInferenceEngine` runs quantized genotyping models
   with `QuantizedWeights` and `NeuronCache` for efficient per-site inference.
2. *Pharmacogenomics*: Sparse star-allele calling models.

**ruvector-cluster** handles unsupervised grouping:
1. *Population Genomics*: Ancestry clustering from PCA embeddings.
2. *Pathogen Surveillance*: Outbreak cluster detection from SNP distance matrices.

**ruvector-graph** is the general-purpose property graph database used across six
contexts for storing genome graphs, Hi-C contact networks, drug interaction networks,
and transmission graphs. Its `TransactionManager` with `IsolationLevel` support
ensures ACID properties for concurrent pipeline stages.

---

## 13. Deployment Architecture

```
+===========================================================================+
|                   DNA ANALYZER DEPLOYMENT                                 |
+===========================================================================+
|                                                                           |
|  Tier 1: Streaming Layer (Hot Path)                                       |
|  +------+ +----------+ +-----------+ +------------+                       |
|  | Ingest| | Alignment| | Variant   | | Pathogen   |                      |
|  | Worker| | Worker   | | Caller    | | Classifier |                      |
|  +---+---+ +----+-----+ +-----+-----+ +-----+------+                     |
|      |          |              |              |                            |
|      v          v              v              v                            |
|  +-----------------------------------------------------+                 |
|  |  ruvector-delta-core Event Bus (at-least-once)       |                 |
|  +-----------------------------------------------------+                 |
|                                                                           |
|  Tier 2: Analytical Layer (Warm Path)                                     |
|  +----------+ +------------+ +----------+ +-----------+                   |
|  | Annotator| | Population | | Epigenome| | PGx Engine|                   |
|  | Service  | | Aggregator | | Profiler | |           |                   |
|  +----------+ +------------+ +----------+ +-----------+                   |
|                                                                           |
|  Tier 3: Engineering Layer (On-Demand)                                    |
|  +-------------------+                                                    |
|  | CRISPR Designer   |                                                    |
|  | (GPU-accelerated) |                                                    |
|  +-------------------+                                                    |
|                                                                           |
|  Infrastructure:                                                          |
|  +------------------+ +------------------+ +-------------------+          |
|  | ruvector-core    | | ruvector-raft    | | ruvector-postgres |          |
|  | (HNSW Indices)   | | (Consensus)      | | (Durable Store)  |          |
|  +------------------+ +------------------+ +-------------------+          |
|                                                                           |
+===========================================================================+
```

---

## 8. Scalability Considerations

| Concern                   | Strategy                                              | RuVector Crate           |
|---------------------------|-------------------------------------------------------|--------------------------|
| Read throughput           | Sharded ingestion workers, streaming delta windows     | ruvector-delta-core      |
| Alignment parallelism     | Min-cut graph partitions, independent per-partition    | ruvector-mincut          |
| Variant call fan-out      | DAG-based independent variant blocks                   | ruvector-dag             |
| Pangenome graph size      | Hierarchical decomposition, compact graph shards       | cognitum-gate-kernel     |
| Taxonomy search           | Hyperbolic HNSW with curvature-aware sharding          | ruvector-hyperbolic-hnsw |
| Cross-context sync        | Raft consensus for distributed cohort updates           | ruvector-raft            |
| Embedding index growth    | Tiered storage with temporal tensor compression         | ruvector-temporal-tensor |
| Neural inference latency  | Sparse quantized models with neuron caching             | ruvector-sparse-inference|
| Off-target search         | Attention mask sparsification, graph-partitioned search | ruvector-attention       |

---

## 9. Cross-Cutting Concerns

### 9.1 Shared Kernel: Genomic Coordinates

The following types constitute the Shared Kernel used by all bounded contexts:

```rust
/// Shared Kernel - used by ALL bounded contexts
pub mod genomic_coordinates {
    #[derive(Clone, PartialEq, Eq, Hash)]
    pub struct ContigId(pub String);

    #[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
    pub struct GenomicPosition {
        pub contig_index: u32,
        pub offset: u64,
        pub strand: Strand,
    }

    #[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
    pub enum Strand { Forward, Reverse }

    #[derive(Clone)]
    pub struct GenomicRegion {
        pub contig: ContigId,
        pub start: u64,
        pub end: u64,
    }

    pub struct SampleId(pub String);
    pub struct GeneId(pub String);
}
```

### 9.2 Observability

All domain events carry correlation IDs for distributed tracing:

```rust
pub struct EventEnvelope<T> {
    pub event_id: u128,
    pub correlation_id: u128,         // traces event across contexts
    pub source_context: &'static str, // e.g., "variant_calling"
    pub timestamp: u64,
    pub payload: T,
}
```

### 9.3 Security & Compliance

- All `SampleId` values are pseudonymized; a separate Identity Mapping Service
  (outside the domain model) handles PHI linkage
- Variant data at rest encrypted via ruvector-core's storage layer
- Audit log captures every domain event envelope for HIPAA compliance
- Access control is role-based per bounded context (clinician, researcher, bioinformatician)

---

## 10. Decision Log

| Decision                                                 | Rationale                                                     |
|----------------------------------------------------------|---------------------------------------------------------------|
| 10 bounded contexts (not fewer)                          | Genomics subdomains have genuinely distinct ubiquitous languages; collapsing them creates ambiguity |
| Shared Kernel for genomic coordinates only               | Position/region types are universal; all other types are context-specific to prevent coupling |
| Anti-corruption layers for cross-domain queries          | Epigenomics and Population Genomics have fundamentally different data models from Variant Calling |
| Conformist for Pharmacogenomics consuming Annotation     | PGx standards (CPIC/DPWG) already define the schema; conforming avoids translation overhead |
| Published Language (VCF/BAM-like) for pipeline stages    | Industry-standard formats reduce integration friction |
| ruvector-hyperbolic-hnsw for taxonomy (not flat HNSW)    | Taxonomy is hierarchical; hyperbolic space preserves tree distances with O(log n) dimensions |
| Delta-based incremental updates throughout               | Genomic pipelines process terabytes; full recomputation is prohibitive |
| GNN for variant effect prediction                        | Graph topology around variants carries structural information that MLP/CNN cannot capture |
| Gated attention for CRISPR off-target                    | Sequence-PAM interaction requires position-aware attention with gating for mismatch tolerance |
| SONA for adaptive basecalling                            | Instrument drift requires online adaptation; EWC++ prevents forgetting across runs |

---

## References

- ADR-001: RuVector Core Architecture
- ADR-016: Delta-Behavior System DDD Architecture
- Evans, Eric. *Domain-Driven Design: Tackling Complexity in the Heart of Software*. 2003.
- Poplin et al. "A universal SNP and small-indel variant caller using deep neural networks." *Nature Biotechnology* 36, 983-987 (2018).
- Garrison et al. "Variation graph toolkit improves read mapping by representing genetic variation in the reference." *Nature Biotechnology* 36, 875-879 (2018).
- Rautiainen et al. "Telomere-to-telomere assembly of a complete human genome." *Science* 376, 44-53 (2022).
- Nickel & Kiela. "Poincare Embeddings for Learning Hierarchical Representations." *NeurIPS* 2017.
