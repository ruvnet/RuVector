# DDD-004: CRISPR Engineering Domain Model

**Status**: Proposed
**Date**: 2026-02-11
**Authors**: ruv.io, RuVector Team
**Related ADR**: ADR-014-coherence-engine, ADR-015-coherence-gated-transformer
**Related DDD**: DDD-001-coherence-gate-domain, DDD-003-epigenomics-domain

---

## Overview

This document defines the Domain-Driven Design model for the CRISPR Engineering bounded context within the RuVector DNA Analyzer. The domain encompasses guide RNA design, off-target effect prediction, editing efficiency optimization, and editing outcome tracking. It leverages `ruvector-mincut` for off-target specificity analysis via Gomory-Hu trees and dynamic min-cut, `ruvector-attention` for sequence-to-activity scoring, `ruvector-gnn` for genome-context-aware guide ranking, and `ruvector-hyperbolic-hnsw` for guide library search in hierarchical specificity space.

---

## Strategic Design

### Domain Vision Statement

> The CRISPR Engineering domain provides end-to-end computational support for genome editing -- from guide RNA design with formal specificity guarantees through editing outcome prediction -- by modeling off-target interference as a graph-theoretic min-cut problem and leveraging attention mechanisms for activity prediction, capabilities that existing tools cannot match at scale.

### Core Domain

**Guide Specificity and Optimization** is the core domain. The differentiating capability is:

- Not sequence alignment (that is an upstream concern)
- Not cell culture protocols (that is a wet-lab concern)
- **The novel capability**: Computing formal all-pairs specificity for large guide libraries in subpolynomial time via Gomory-Hu trees, combined with dynamic min-cut for real-time guide re-optimization as new off-target data arrives from experiments.

### Supporting Domains

| Domain | Role | Boundary |
|--------|------|----------|
| **Reference Genome** | Genomic coordinates, PAM site enumeration | Generic, external |
| **Epigenomics** | Chromatin state affects editing efficiency | Separate bounded context (DDD-003) |
| **Variant Calling** | Genotype-aware off-target prediction | Separate bounded context |
| **Sequencing Pipeline** | GUIDE-seq, CIRCLE-seq, DISCOVER-seq data | Generic, infrastructure |
| **Cell Biology** | Cell type, delivery method metadata | External context |

### Generic Subdomains

- Logging and observability
- Sequence I/O (FASTA, FASTQ, VCF)
- Configuration and parameter management
- Statistical significance testing

---

## Ubiquitous Language

### Core Terms

| Term | Definition | Context |
|------|------------|---------|
| **Guide RNA (gRNA)** | A ~20-nucleotide RNA sequence that directs Cas protein to a genomic target | Core entity |
| **PAM** | Protospacer Adjacent Motif; short DNA sequence required adjacent to the target (e.g., NGG for SpCas9) | Targeting constraint |
| **On-Target Score** | Predicted cutting efficiency at the intended target site (0.0-1.0) | Efficiency metric |
| **Off-Target Site** | Unintended genomic locus where the guide may bind and cut | Safety concern |
| **Mismatch** | A position where the guide RNA does not complement the genomic DNA | Off-target feature |
| **Bulge** | An insertion or deletion in the RNA-DNA heteroduplex | Off-target feature |
| **Cutting Frequency Determination (CFD)** | Score quantifying off-target cutting probability based on mismatch pattern | Standard metric |
| **HDR** | Homology-Directed Repair; precise editing using a donor template | Outcome type |
| **NHEJ** | Non-Homologous End Joining; error-prone repair causing insertions/deletions | Outcome type |
| **Indel Spectrum** | Distribution of insertion/deletion sizes and sequences at an edit site | Outcome characterization |

### Graph-Theoretic Terms

| Term | Definition | Context |
|------|------------|---------|
| **Specificity Graph** | Graph where guides are nodes and edges connect guides that share off-target sites, weighted by cross-reactivity | Core model |
| **Gomory-Hu Tree** | Compact tree representing all-pairs max-flow/min-cut in the specificity graph | Algorithm |
| **Guide Interference** | When two guides in a library share off-target sites, creating compound risk | Library design |
| **Minimum Specificity Cut** | Min-cut that separates a guide from its most problematic off-target cluster | Per-guide metric |

### Experimental Terms

| Term | Definition | Context |
|------|------------|---------|
| **CRISPR Experiment** | Complete record of guides, conditions, and outcomes | Aggregate context |
| **Delivery Method** | How CRISPR components enter the cell (RNP, plasmid, viral, lipid) | Experimental variable |
| **Editing Efficiency** | Fraction of cells with intended edit | Outcome metric |
| **Guide Library** | Collection of guides designed for a screen or multiplexed editing | Scale context |

---

## Bounded Contexts

### Context Map

```
+-----------------------------------------------------------------------------+
|                       CRISPR ENGINEERING CONTEXT                             |
|                           (Core Domain)                                      |
|  +---------------+  +---------------+  +---------------+  +---------------+  |
|  | Guide Design  |  | Off-Target    |  | Editing       |  | Library       |  |
|  | Subcontext    |  | Analysis      |  | Outcome       |  | Optimization  |  |
|  +---------------+  +---------------+  +---------------+  +---------------+  |
+-----------------------------------------------------------------------------+
         |                  |                  |                  |
         | Upstream         | Upstream         | Downstream      | Upstream
         v                  v                  v                  v
+------------------+ +------------------+ +------------------+ +------------------+
|   REFERENCE      | |   EPIGENOMICS    | |   SEQUENCING     | |   VARIANT        |
|   GENOME         | |   CONTEXT        | |   PIPELINE       | |   CALLING        |
|  (External)      | | (DDD-003)        | | (Infrastructure) | | (External)       |
+------------------+ +------------------+ +------------------+ +------------------+
```

### CRISPR Engineering Context (Core)

**Responsibility**: Design optimal guide RNAs, predict and mitigate off-target effects, and track editing outcomes.

**Key Aggregates**:
- GuideRNA (Aggregate Root)
- CRISPRExperiment
- GuideLibrary
- SpecificityGraph

**Anti-Corruption Layers**:
- Genome ACL (translates reference genome to PAM-site catalogue)
- Epigenomics ACL (translates chromatin state to editing accessibility features)
- Sequencing ACL (translates GUIDE-seq/CIRCLE-seq reads to off-target observations)
- Variant ACL (translates genotype to personalized off-target prediction)

---

## Aggregates

### GuideRNA (Root Aggregate)

The central aggregate representing a designed guide RNA and its complete specificity profile.

```
+-----------------------------------------------------------------------+
|                        GUIDE RNA                                       |
|                    (Aggregate Root)                                    |
+-----------------------------------------------------------------------+
|  guide_id: GuideId                                                     |
|  sequence: NucleotideSequence (20nt canonical)                         |
|  pam: PamSequence (e.g., "NGG")                                       |
|  cas_variant: CasVariant { SpCas9 | SaCas9 | Cas12a | Cas13 | ... }  |
|  target_locus: GenomicRegion                                           |
|  target_gene: Option<String>                                           |
|  strand: Strand { Plus | Minus }                                       |
|  on_target_score: f64 (0.0 - 1.0)                                     |
|  structure_score: f64 (secondary structure penalty)                     |
|  gc_content: f64                                                       |
|  off_targets: Vec<OffTargetSite>                                       |
|  specificity_score: f64 (aggregate off-target safety metric)           |
|  design_status: DesignStatus { Draft | Scored | Validated | Retired } |
+-----------------------------------------------------------------------+
|  +------------------------------------------------------------------+ |
|  | OffTargetSite (Entity)                                           | |
|  |  ot_id: OffTargetId                                              | |
|  |  locus: GenomicRegion                                            | |
|  |  aligned_sequence: String (with mismatches marked)               | |
|  |  mismatches: Vec<MismatchPosition>                               | |
|  |  bulges: Vec<BulgePosition>                                      | |
|  |  total_mismatches: u8                                            | |
|  |  total_bulges: u8                                                | |
|  |  cfd_score: f64 (Cutting Frequency Determination)                | |
|  |  cutting_frequency: f64 (experimentally observed, if available)  | |
|  |  risk_score: f64 (composite safety metric)                       | |
|  |  in_gene: Option<String>                                         | |
|  |  in_exon: bool                                                   | |
|  |  chromatin_accessible: Option<bool>                              | |
|  +------------------------------------------------------------------+ |
|  +------------------------------------------------------------------+ |
|  | MismatchPosition (Value Object)                                  | |
|  |  position: u8 (0-19, PAM-distal to PAM-proximal)                | |
|  |  guide_base: Nucleotide                                          | |
|  |  target_base: Nucleotide                                         | |
|  |  positional_weight: f64                                          | |
|  +------------------------------------------------------------------+ |
|  +------------------------------------------------------------------+ |
|  | BulgePosition (Value Object)                                     | |
|  |  position: u8                                                    | |
|  |  bulge_type: BulgeType { RNA | DNA }                             | |
|  |  size: u8                                                        | |
|  +------------------------------------------------------------------+ |
+-----------------------------------------------------------------------+
|  Invariants:                                                           |
|  - sequence length matches cas_variant expected spacer length          |
|  - on_target_score in [0.0, 1.0]                                      |
|  - specificity_score in [0.0, 1.0]                                     |
|  - gc_content in [0.0, 1.0]                                           |
|  - Off-targets sorted by risk_score descending                         |
|  - total_mismatches + total_bulges <= cas_variant.max_tolerated        |
|  - PAM matches cas_variant.pam_pattern                                 |
|  - target_locus is on the correct strand relative to PAM              |
+-----------------------------------------------------------------------+
```

### CRISPRExperiment (Aggregate)

Records the full lifecycle of a CRISPR editing experiment.

```
+-----------------------------------------------------------------------+
|                    CRISPR EXPERIMENT                                    |
|                    (Aggregate Root)                                    |
+-----------------------------------------------------------------------+
|  experiment_id: ExperimentId                                           |
|  name: String                                                          |
|  guides: Vec<GuideId>                                                  |
|  cell_type: CellType                                                   |
|  delivery_method: DeliveryMethod { RNP | Plasmid | Viral | Lipid }    |
|  donor_template: Option<DonorTemplate>                                 |
|  outcomes: Vec<EditingOutcome>                                         |
|  overall_efficiency: Option<f64>                                       |
|  status: ExperimentStatus { Designed | InProgress | Completed }       |
|  created_at: Timestamp                                                 |
|  completed_at: Option<Timestamp>                                       |
+-----------------------------------------------------------------------+
|  +------------------------------------------------------------------+ |
|  | EditingOutcome (Entity)                                          | |
|  |  outcome_id: OutcomeId                                           | |
|  |  guide_id: GuideId                                               | |
|  |  target_site: GenomicRegion                                      | |
|  |  outcome_type: OutcomeType { HDR | NHEJ | Precise | NoEdit }    | |
|  |  frequency: f64 (fraction of reads)                              | |
|  |  indel_spectrum: Vec<IndelEvent>                                 | |
|  |  read_count: u32                                                 | |
|  +------------------------------------------------------------------+ |
|  +------------------------------------------------------------------+ |
|  | IndelEvent (Value Object)                                        | |
|  |  indel_type: IndelType { Insertion | Deletion }                  | |
|  |  size: i32 (positive for insertion, negative for deletion)       | |
|  |  position: u64                                                   | |
|  |  sequence: Option<String>                                        | |
|  |  frequency: f64                                                  | |
|  +------------------------------------------------------------------+ |
|  +------------------------------------------------------------------+ |
|  | DonorTemplate (Value Object)                                     | |
|  |  template_type: TemplateType { ssODN | dsDNA | Plasmid }        | |
|  |  sequence: String                                                | |
|  |  left_homology_arm: u32 (length in bp)                          | |
|  |  right_homology_arm: u32 (length in bp)                         | |
|  |  edit_payload: String                                            | |
|  +------------------------------------------------------------------+ |
+-----------------------------------------------------------------------+
|  Invariants:                                                           |
|  - At least one guide must be associated                               |
|  - outcome frequencies for a guide sum to <= 1.0                       |
|  - completed_at is set iff status == Completed                         |
|  - Outcomes only recorded for guides in the experiment's guide list    |
|  - overall_efficiency = sum of non-NoEdit outcomes / total reads       |
+-----------------------------------------------------------------------+
```

### GuideLibrary (Aggregate)

A collection of guides designed for a screen, with pairwise specificity analysis.

```
+-----------------------------------------------------------------------+
|                     GUIDE LIBRARY                                      |
|                    (Aggregate Root)                                    |
+-----------------------------------------------------------------------+
|  library_id: LibraryId                                                 |
|  name: String                                                          |
|  guides: Vec<GuideId>                                                  |
|  target_genes: Vec<String>                                             |
|  library_type: LibraryType { KnockOut | CRISPRi | CRISPRa | Base }   |
|  specificity_graph: Option<SpecificityGraph>                           |
|  pairwise_interference: Vec<GuideInterference>                         |
|  optimization_status: OptimizationStatus                               |
+-----------------------------------------------------------------------+
|  +------------------------------------------------------------------+ |
|  | GuideInterference (Value Object)                                 | |
|  |  guide_a: GuideId                                                | |
|  |  guide_b: GuideId                                                | |
|  |  shared_off_targets: u32                                         | |
|  |  max_cross_reactivity: f64                                       | |
|  |  gomory_hu_cut_value: f64                                        | |
|  +------------------------------------------------------------------+ |
+-----------------------------------------------------------------------+
|  Invariants:                                                           |
|  - No duplicate guides in the library                                  |
|  - specificity_graph covers all guides if computed                     |
|  - pairwise_interference is symmetric (a,b) == (b,a)                  |
|  - Library contains >= 1 guide per target gene                         |
+-----------------------------------------------------------------------+
```

### SpecificityGraph (Aggregate)

The graph model encoding guide-to-guide off-target interference, powered by `ruvector-mincut`.

```
+-----------------------------------------------------------------------+
|                   SPECIFICITY GRAPH                                     |
|                    (Aggregate Root)                                    |
+-----------------------------------------------------------------------+
|  graph_id: GraphId                                                     |
|  library_id: LibraryId                                                 |
|  dynamic_graph: ruvector_mincut::DynamicGraph                          |
|  gomory_hu_tree: Option<GomoryHuTree>                                  |
|  guide_vertex_map: BiMap<GuideId, VertexId>                            |
|  version: u64                                                          |
+-----------------------------------------------------------------------+
|  +------------------------------------------------------------------+ |
|  | GomoryHuTree (Value Object)                                      | |
|  |  tree: ruvector_mincut::HierarchicalDecomposition                | |
|  |  computed_at: Timestamp                                          | |
|  |  num_guides: usize                                               | |
|  |                                                                  | |
|  |  fn all_pairs_min_cut(&self, a: GuideId, b: GuideId) -> f64     | |
|  |  fn most_interfering_pair(&self) -> (GuideId, GuideId, f64)     | |
|  |  fn isolation_score(&self, guide: GuideId) -> f64               | |
|  +------------------------------------------------------------------+ |
+-----------------------------------------------------------------------+
|  Invariants:                                                           |
|  - Vertex count equals number of guides in library                     |
|  - Edge weights are non-negative (cross-reactivity scores)             |
|  - gomory_hu_tree is invalidated when graph changes                    |
|  - version monotonically increases                                     |
+-----------------------------------------------------------------------+
```

---

## Value Objects

### NucleotideSequence

```rust
struct NucleotideSequence {
    bases: String,  // ACGT only, uppercase
    length: usize,
}

impl NucleotideSequence {
    fn gc_content(&self) -> f64;
    fn reverse_complement(&self) -> NucleotideSequence;
    fn hamming_distance(&self, other: &NucleotideSequence) -> u8;
    fn contains_homopolymer(&self, length: usize) -> bool;
}
```

### PamSequence

```rust
struct PamSequence {
    pattern: String,  // IUPAC codes: N=any, R=A/G, Y=C/T, etc.
    orientation: PamOrientation,  // ThreePrime (Cas9) | FivePrime (Cas12a)
}

impl PamSequence {
    fn matches(&self, dna: &str) -> bool;
    fn enumerate_sites(&self, reference: &str) -> Vec<usize>;
}
```

### CfdScore

```rust
struct CfdScore {
    score: f64,  // 0.0 (no cutting) to 1.0 (perfect match)
    mismatch_penalties: Vec<(u8, f64)>,
    pam_penalty: f64,
}

impl CfdScore {
    fn from_alignment(guide: &NucleotideSequence, target: &str, pam: &str) -> Self;
    fn is_concerning(&self) -> bool { self.score > 0.1 }
}
```

### EditingEfficiency

```rust
struct EditingEfficiency {
    total_reads: u32,
    edited_reads: u32,
    hdr_fraction: f64,
    nhej_fraction: f64,
    unedited_fraction: f64,
}

impl EditingEfficiency {
    fn overall(&self) -> f64 { 1.0 - self.unedited_fraction }
    fn precision(&self) -> f64 { self.hdr_fraction / self.overall().max(0.001) }
}
```

---

## Domain Events

### Guide Design Events

| Event | Trigger | Payload |
|-------|---------|---------|
| `GuideDesigned` | Guide RNA design complete | guide_id, sequence, target_locus, on_target_score |
| `GuideScored` | On-target scoring complete | guide_id, on_target_score, structure_score, gc_content |
| `GuideRetired` | Guide removed from consideration | guide_id, reason |

### Off-Target Events

| Event | Trigger | Payload |
|-------|---------|---------|
| `OffTargetsScored` | Off-target analysis complete | guide_id, off_target_count, specificity_score |
| `HighRiskOffTargetFound` | Off-target in exon with high CFD | guide_id, ot_id, locus, gene, cfd_score |
| `SpecificityGraphUpdated` | New off-target data incorporated | graph_id, version, edge_delta |
| `GomoryHuTreeComputed` | All-pairs specificity computed | graph_id, num_guides, computation_time_ms |

### Editing Outcome Events

| Event | Trigger | Payload |
|-------|---------|---------|
| `EditingOutcomeRecorded` | Sequencing analysis complete | experiment_id, guide_id, outcome_type, frequency |
| `ExperimentCompleted` | All outcomes analyzed | experiment_id, overall_efficiency, guide_rankings |
| `UnexpectedOutcomeDetected` | Off-target editing confirmed | experiment_id, guide_id, ot_locus, frequency |

### Optimization Events

| Event | Trigger | Payload |
|-------|---------|---------|
| `GuideOptimized` | Guide re-scored with new data | guide_id, old_score, new_score, data_source |
| `LibraryOptimized` | Library-wide optimization complete | library_id, guides_replaced, specificity_improvement |
| `InterferenceResolved` | Problematic guide pair replaced | library_id, guide_a, guide_b, replacement |

---

## Domain Services

### GuideDesignService

Designs candidate guide RNAs for a target gene or region.

```rust
trait GuideDesignService {
    /// Enumerate all valid guide RNA candidates for a target region
    ///
    /// Scans the reference genome for PAM sites, extracts spacer sequences,
    /// and filters by basic quality criteria (GC content, homopolymers, etc.)
    async fn enumerate_candidates(
        &self,
        target: &GenomicRegion,
        cas_variant: CasVariant,
        config: &GuideDesignConfig,
    ) -> Result<Vec<GuideRNA>, CrisprError>;

    /// Score on-target activity using attention-based model
    ///
    /// Uses ruvector-attention to predict cutting efficiency from sequence
    /// features, local chromatin context, and learned positional weights.
    fn score_on_target(
        &self,
        guide: &GuideRNA,
        chromatin_state: Option<&ChromatinStateVector>,
    ) -> Result<f64, CrisprError>;

    /// Rank candidates by combined on-target and specificity scores
    fn rank_candidates(
        &self,
        candidates: &[GuideRNA],
        weights: &RankingWeights,
    ) -> Vec<(GuideId, f64)>;
}

struct GuideDesignConfig {
    min_gc: f64,
    max_gc: f64,
    max_homopolymer: usize,
    avoid_tttt: bool,  // Pol III terminator
    min_on_target_score: f64,
    max_off_targets: usize,
    cas_variant: CasVariant,
}

struct RankingWeights {
    on_target: f64,      // Weight for cutting efficiency
    specificity: f64,    // Weight for off-target safety
    structure: f64,      // Weight for RNA secondary structure
    gc_penalty: f64,     // Penalty for extreme GC content
}
```

### OffTargetAnalyzer

Predicts off-target sites and computes specificity scores.

```rust
trait OffTargetAnalyzer {
    /// Find all potential off-target sites in the genome
    ///
    /// Searches the reference genome for sequences similar to the guide,
    /// allowing up to max_mismatches mismatches and max_bulges bulges.
    async fn find_off_targets(
        &self,
        guide: &GuideRNA,
        reference: &ReferenceGenome,
        config: &OffTargetConfig,
    ) -> Result<Vec<OffTargetSite>, CrisprError>;

    /// Score off-target sites using CFD and chromatin context
    fn score_off_targets(
        &self,
        guide: &GuideRNA,
        off_targets: &mut [OffTargetSite],
        chromatin: Option<&ChromatinLandscape>,
    ) -> Result<(), CrisprError>;

    /// Compute aggregate specificity score for a guide
    fn compute_specificity(
        &self,
        guide: &GuideRNA,
    ) -> f64;
}

struct OffTargetConfig {
    max_mismatches: u8,
    max_bulges: u8,
    max_off_targets: usize,
    include_pam_variants: bool,
    genome_index: GenomeIndexType,  // Bowtie2 | Cas-OFFinder | BwaAln
}
```

### SpecificityGraphService

Manages the guide specificity graph using `ruvector-mincut`.

```rust
trait SpecificityGraphService {
    /// Build specificity graph for a guide library
    ///
    /// Models guides as vertices. Two guides share an edge if they have
    /// overlapping off-target sites. Edge weight = max cross-reactivity score.
    ///
    /// Uses ruvector_mincut::MinCutBuilder to create the underlying DynamicGraph.
    fn build_graph(
        &self,
        library: &GuideLibrary,
        guides: &[GuideRNA],
    ) -> Result<SpecificityGraph, CrisprError>;

    /// Compute Gomory-Hu tree for all-pairs guide specificity
    ///
    /// Leverages Abboud et al. (2024) algorithm via ruvector-mincut:
    /// Given m edges, compute the Gomory-Hu tree in m^{1+o(1)} time.
    /// This gives the minimum cut between ANY pair of guides, enabling:
    /// - Identification of the most interfering guide pair
    /// - Per-guide isolation scores (how separable from its worst neighbor)
    /// - Optimal guide replacement decisions
    fn compute_gomory_hu_tree(
        &self,
        graph: &SpecificityGraph,
    ) -> Result<GomoryHuTree, CrisprError>;

    /// Dynamic update: incorporate new off-target data
    ///
    /// Uses ruvector_mincut::DynamicMinCut with insert_edge/delete_edge
    /// to incrementally update the specificity graph as new experimental
    /// off-target data arrives (El-Hayek et al. dynamic min-cut).
    fn update_with_observation(
        &self,
        graph: &mut SpecificityGraph,
        guide_id: GuideId,
        new_off_targets: &[OffTargetSite],
    ) -> Result<(), CrisprError>;

    /// Find the minimum specificity cut for a single guide
    ///
    /// Returns the min-cut value separating this guide from its closest
    /// off-target cluster in the specificity graph.
    fn guide_min_cut(
        &self,
        graph: &SpecificityGraph,
        guide_id: GuideId,
    ) -> Result<f64, CrisprError>;

    /// Identify the most problematic guide pair in the library
    fn worst_interference(
        &self,
        gomory_hu: &GomoryHuTree,
    ) -> Result<(GuideId, GuideId, f64), CrisprError>;
}
```

### GuideOptimizer

Optimizes guide selection using graph-theoretic methods.

```rust
trait GuideOptimizer {
    /// Optimize a guide library to minimize off-target interference
    ///
    /// Uses the Gomory-Hu tree to identify and replace interfering guides:
    /// 1. Compute all-pairs min-cut via Gomory-Hu
    /// 2. Find the pair with lowest min-cut (highest interference)
    /// 3. Replace the weaker guide with a better alternative
    /// 4. Repeat until all pairwise min-cuts exceed threshold
    async fn optimize_library(
        &self,
        library: &mut GuideLibrary,
        all_candidates: &[GuideRNA],
        config: &OptimizationConfig,
    ) -> Result<OptimizationReport, CrisprError>;

    /// Re-optimize a single guide given new off-target data
    ///
    /// Uses dynamic min-cut (El-Hayek et al.) to efficiently re-evaluate
    /// guide specificity without recomputing the entire graph.
    fn reoptimize_guide(
        &self,
        library: &mut GuideLibrary,
        guide_id: GuideId,
        new_data: &[OffTargetSite],
        candidates: &[GuideRNA],
    ) -> Result<Option<GuideId>, CrisprError>;
}

struct OptimizationConfig {
    min_pairwise_min_cut: f64,   // Minimum acceptable min-cut between any pair
    max_iterations: usize,
    min_on_target_score: f64,     // Don't sacrifice too much on-target activity
    preserve_gene_coverage: bool, // Ensure every target gene has >= 1 guide
    use_dynamic_mincut: bool,     // Use subpolynomial dynamic updates
}

struct OptimizationReport {
    guides_replaced: Vec<(GuideId, GuideId)>,  // (old, new) pairs
    iterations: usize,
    min_pairwise_cut_before: f64,
    min_pairwise_cut_after: f64,
    specificity_improvement: f64,
}
```

### EditingOutcomePredictor

Predicts editing outcomes using GNN-based models.

```rust
trait EditingOutcomePredictor {
    /// Predict indel spectrum at a target site
    ///
    /// Uses ruvector-gnn to model the local sequence context as a graph:
    /// - Nodes: nucleotide positions around the cut site
    /// - Edges: biochemical interactions (base pairing, stacking)
    /// - GNN message passing predicts repair outcome distribution
    fn predict_indel_spectrum(
        &self,
        guide: &GuideRNA,
        local_sequence: &str,  // +/- 50bp around cut site
    ) -> Result<Vec<IndelEvent>, CrisprError>;

    /// Predict HDR vs NHEJ ratio given donor template
    fn predict_repair_outcome(
        &self,
        guide: &GuideRNA,
        donor: &DonorTemplate,
        cell_type: &CellType,
    ) -> Result<EditingEfficiency, CrisprError>;
}
```

### GuideSearchService

Searches for similar guides using `ruvector-hyperbolic-hnsw`.

```rust
trait GuideSearchService {
    /// Index a guide library in hyperbolic specificity space
    ///
    /// Guide specificity has natural hierarchy:
    /// - Root: all guides
    /// - Level 1: guides grouped by target gene
    /// - Level 2: guides grouped by off-target profile similarity
    /// - Leaves: individual guides
    ///
    /// Hyperbolic HNSW preserves this hierarchy for efficient search.
    fn index_library(
        &self,
        library: &GuideLibrary,
        guides: &[GuideRNA],
    ) -> Result<GuideIndex, CrisprError>;

    /// Find guides with similar specificity profiles
    fn search_similar(
        &self,
        query_guide: &GuideRNA,
        k: usize,
    ) -> Result<Vec<(GuideId, f64)>, CrisprError>;

    /// Find replacement candidates for a problematic guide
    fn find_replacements(
        &self,
        target_region: &GenomicRegion,
        avoid_off_targets: &[OffTargetSite],
        k: usize,
    ) -> Result<Vec<GuideRNA>, CrisprError>;
}
```

---

## Repositories

### GuideRNARepository

```rust
trait GuideRNARepository {
    async fn store(&self, guide: GuideRNA) -> Result<(), StoreError>;
    async fn find_by_id(&self, id: GuideId) -> Option<GuideRNA>;
    async fn find_by_target_gene(&self, gene: &str) -> Vec<GuideRNA>;
    async fn find_by_target_region(&self, region: &GenomicRegion) -> Vec<GuideRNA>;
    async fn find_by_sequence(&self, sequence: &NucleotideSequence) -> Option<GuideRNA>;
    async fn update_scores(&self, id: GuideId, on_target: f64, specificity: f64) -> Result<(), StoreError>;
}
```

### CRISPRExperimentRepository

```rust
trait CRISPRExperimentRepository {
    async fn store(&self, experiment: CRISPRExperiment) -> Result<(), StoreError>;
    async fn find_by_id(&self, id: ExperimentId) -> Option<CRISPRExperiment>;
    async fn find_by_guide(&self, guide_id: GuideId) -> Vec<CRISPRExperiment>;
    async fn record_outcome(
        &self, experiment_id: ExperimentId, outcome: EditingOutcome
    ) -> Result<(), StoreError>;
}
```

### GuideLibraryRepository

```rust
trait GuideLibraryRepository {
    async fn store(&self, library: GuideLibrary) -> Result<(), StoreError>;
    async fn find_by_id(&self, id: LibraryId) -> Option<GuideLibrary>;
    async fn store_specificity_graph(
        &self, library_id: LibraryId, graph: SpecificityGraph
    ) -> Result<(), StoreError>;
}
```

---

## Factories

### GuideRNAFactory

```rust
impl GuideRNAFactory {
    fn create_from_target(
        target_sequence: &str,
        pam_site_offset: usize,
        cas_variant: CasVariant,
        target_locus: GenomicRegion,
        strand: Strand,
    ) -> Result<GuideRNA, CrisprError> {
        let sequence = NucleotideSequence::from_str(
            &target_sequence[pam_site_offset..pam_site_offset + cas_variant.spacer_length()]
        )?;
        let pam = cas_variant.extract_pam(target_sequence, pam_site_offset)?;

        Ok(GuideRNA {
            guide_id: GuideId::new(),
            sequence,
            pam,
            cas_variant,
            target_locus,
            target_gene: None,
            strand,
            on_target_score: 0.0,    // Unscored
            structure_score: 0.0,
            gc_content: sequence.gc_content(),
            off_targets: vec![],
            specificity_score: 0.0,
            design_status: DesignStatus::Draft,
        })
    }
}
```

### SpecificityGraphFactory

```rust
impl SpecificityGraphFactory {
    /// Build from a set of guides and their off-target profiles
    fn build(
        library_id: LibraryId,
        guides: &[GuideRNA],
    ) -> Result<SpecificityGraph, CrisprError> {
        let mut graph = DynamicGraph::new();
        let mut guide_vertex_map = BiMap::new();

        // Assign each guide a vertex
        for (i, guide) in guides.iter().enumerate() {
            let vertex_id = i as u64;
            guide_vertex_map.insert(guide.guide_id.clone(), vertex_id);
        }

        // For each pair of guides, compute shared off-target weight
        for i in 0..guides.len() {
            for j in (i+1)..guides.len() {
                let weight = compute_cross_reactivity(&guides[i], &guides[j]);
                if weight > 0.0 {
                    graph.insert_edge(i as u64, j as u64, weight).ok();
                }
            }
        }

        Ok(SpecificityGraph {
            graph_id: GraphId::new(),
            library_id,
            dynamic_graph: graph,
            gomory_hu_tree: None,
            guide_vertex_map,
            version: 1,
        })
    }
}
```

---

## How Min-Cut Powers CRISPR Analysis

### 1. Genome as a Graph: Guide Binding Creates "Cuts"

The genome is modeled as a graph where potential binding sites are vertices. A guide RNA binding at a site effectively creates a "cut" -- it severs the local chromatin and DNA continuity. Off-target binding creates unintended cuts. The min-cut between intended and unintended cut clusters quantifies guide specificity.

### 2. Off-Target Prediction as Min-Cut Problem

For a single guide, its off-target sites and the on-target site form a weighted graph. Edge weights encode sequence similarity. The minimum cut separating the on-target vertex from the off-target cluster gives a formal measure of how "separable" the intended target is from potential off-targets. A low min-cut means the guide is poorly specific.

### 3. Gomory-Hu Tree for All-Pairs Guide Specificity

Given a library of 10,000 guides for a genome-wide screen, computing pairwise off-target interference naively requires O(n^2) = 100 million comparisons. The Gomory-Hu tree (leveraging Abboud et al. via `ruvector-mincut`) computes this in m^{1+o(1)} time, where m is the number of edges in the specificity graph. The tree compactly encodes the min-cut between every pair of guides, enabling:

- **Library deconfliction**: Identify the pair of guides with lowest specificity separation and replace one.
- **Coverage guarantees**: Ensure every target gene has guides that are well-separated from all other library members.
- **Batch optimization**: Process the entire Gomory-Hu tree to find an optimal subset of guides.

### 4. Dynamic Min-Cut for Real-Time Guide Optimization

When new experimental data arrives (GUIDE-seq, CIRCLE-seq results), off-target profiles change. Rather than recomputing the entire specificity graph from scratch, `ruvector-mincut`'s `DynamicMinCut` with `insert_edge` and `delete_edge` (El-Hayek et al.) supports subpolynomial-time incremental updates. This enables:

- **Live experiment tracking**: As off-target sequencing results stream in, the specificity graph updates in real time.
- **Adaptive guide selection**: Guides that develop problematic off-target profiles are flagged immediately.
- **Iterative refinement**: Each round of experimental validation improves the model without full recomputation.

---

## Anti-Corruption Layers

### Genome ACL

```rust
impl GenomeAntiCorruptionLayer {
    fn enumerate_pam_sites(
        &self,
        reference: &ReferenceGenome,
        region: &GenomicRegion,
        cas_variant: CasVariant,
    ) -> Result<Vec<PamSite>, AclError> {
        // Scan both strands for PAM matches
        // Map to genomic coordinates
        // Filter by mappability and repeat content
    }
}
```

### Epigenomics ACL

```rust
impl EpigenomicsAntiCorruptionLayer {
    /// Translate chromatin state to editing accessibility prediction
    fn translate_chromatin_context(
        &self,
        site: &GenomicRegion,
        epigenomics: &EpigenomicsContext,
    ) -> Result<ChromatinAccessibilityFeature, AclError> {
        // Query MethylationProfile for local methylation
        // Query ChromatinLandscape for accessibility at the cut site
        // Combine into feature vector for on-target scoring
    }
}
```

### Sequencing ACL

```rust
impl SequencingAntiCorruptionLayer {
    /// Translate GUIDE-seq/CIRCLE-seq reads to off-target observations
    fn translate_off_target_reads(
        &self,
        reads: &AlignmentSource,
        guide: &GuideRNA,
    ) -> Result<Vec<ExperimentalOffTarget>, AclError> {
        // Parse UMI-tagged reads
        // Deduplicate by unique molecular identifier
        // Map to genomic coordinates
        // Filter by mapping quality
    }
}
```

---

## Event Flow: End-to-End Guide Design Pipeline

```
1. User specifies target gene
   |
2. GuideDesignService.enumerate_candidates()
   |  -> Emits: GuideDesigned (for each candidate)
   |
3. GuideDesignService.score_on_target()  [uses ruvector-attention]
   |  -> Emits: GuideScored
   |
4. OffTargetAnalyzer.find_off_targets()
   |
5. OffTargetAnalyzer.score_off_targets()
   |  -> Emits: OffTargetsScored
   |  -> Emits: HighRiskOffTargetFound (if exonic off-target with CFD > 0.1)
   |
6. SpecificityGraphService.build_graph()      [uses ruvector-mincut]
   |
7. SpecificityGraphService.compute_gomory_hu_tree()
   |  -> Emits: GomoryHuTreeComputed
   |
8. GuideOptimizer.optimize_library()
   |  -> Emits: InterferenceResolved (for each replacement)
   |  -> Emits: LibraryOptimized
   |
9. [Wet lab performs experiment]
   |
10. EditingOutcomeRecorded
    |
11. SpecificityGraphService.update_with_observation()  [dynamic min-cut]
    |  -> Emits: SpecificityGraphUpdated
    |
12. GuideOptimizer.reoptimize_guide()
    |  -> Emits: GuideOptimized
```

---

## Context Boundaries Summary

| Boundary | Upstream | Downstream | Integration Pattern |
|----------|----------|------------|---------------------|
| Genome -> CRISPR | Reference Genome | CRISPR Context | ACL (PamSite catalogue) |
| Epigenomics -> CRISPR | Epigenomics Context (DDD-003) | CRISPR Context | Published Language (ChromatinState) |
| Sequencing -> CRISPR | Sequencing Pipeline | CRISPR Context | ACL (ExperimentalOffTarget) |
| Variant -> CRISPR | Variant Calling | CRISPR Context | ACL (PersonalizedGenome) |
| CRISPR -> Experiment | CRISPR Context | Lab Systems | Domain Events (GuideDesigned, LibraryOptimized) |

---

## References

- DDD-001: Coherence Gate Domain Model
- DDD-003: Epigenomics Domain Model
- Evans, Eric. "Domain-Driven Design." Addison-Wesley, 2003.
- Vernon, Vaughn. "Implementing Domain-Driven Design." Addison-Wesley, 2013.
- Doench, J. et al. "Optimized sgRNA design to maximize activity and minimize off-target effects of CRISPR-Cas9." Nature Biotechnology, 2016.
- Hsu, P. et al. "DNA targeting specificity of RNA-guided Cas9 nucleases." Nature Biotechnology, 2013.
- Abboud, A. et al. "Gomory-Hu Tree in Almost-Linear Time." FOCS, 2023.
- El-Hayek, S. et al. "Dynamic Minimum Cut." SODA, 2024.
