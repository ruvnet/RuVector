# DDD-003: Epigenomics Domain Model

**Status**: Proposed
**Date**: 2026-02-11
**Authors**: ruv.io, RuVector Team
**Related ADR**: ADR-014-coherence-engine, ADR-015-coherence-gated-transformer
**Related DDD**: DDD-001-coherence-gate-domain, DDD-004-crispr-engineering-domain

---

## Overview

This document defines the Domain-Driven Design model for the Epigenomics bounded context within the RuVector DNA Analyzer. The domain covers methylation pattern analysis, chromatin accessibility profiling, histone modification mapping, and 3D genome architecture reconstruction at single-base resolution. It integrates deeply with four RuVector crates: `ruvector-mincut` for TAD boundary detection, `ruvector-attention` for long-range interaction prediction, `ruvector-gnn` for 3D genome functional predictions, and `ruvector-hyperbolic-hnsw` for hierarchical chromatin state search.

**SOTA extensions** (2024-2026): This model now additionally covers single-cell multi-omics integration via coupled autoencoders, persistent-homology-based TAD detection from Hi-C data, DNA methylation clocks for epigenetic age prediction, Enformer-style chromatin accessibility prediction from sequence, biomolecular condensate modeling via intrinsically disordered region (IDR) analysis, and 4D nucleome time-resolved chromatin modeling using polymer loop-extrusion simulations.

---

## Strategic Design

### Domain Vision Statement

> The Epigenomics domain provides multi-layered, single-base-resolution analysis of the regulatory genome -- methylation landscapes, open chromatin, histone marks, and three-dimensional folding -- enabling researchers to move from raw sequencing data to actionable regulatory insights through graph-native algorithms that were previously infeasible at genome scale.

### Core Domain

**Regulatory Landscape Reconstruction** is the core domain. The differentiating capability is:

- Not alignment (that is an upstream infrastructure concern)
- Not variant calling (that belongs to the Genomics bounded context)
- **The novel capability**: Reconstructing the full regulatory state of a genomic locus by integrating methylation, accessibility, histone marks, and 3D contact structure into a single coherent model, powered by graph-cut and attention-based algorithms.

### Supporting Domains

| Domain | Role | Boundary |
|--------|------|----------|
| **Alignment Pipeline** | Produce aligned reads (BAM/CRAM) | Generic, infrastructure |
| **Reference Genome** | Chromosome coordinates, gene annotations | Generic, external |
| **Variant Calling** | SNP/indel context for allele-specific analysis | Separate bounded context |
| **Sequencing QC** | Read quality, bisulfite conversion rates | Generic, infrastructure |

### Generic Subdomains

- Logging and observability
- File I/O (BAM, BED, BigWig, .hic)
- Configuration and parameter management
- Coordinate system transformations (0-based, 1-based, BED intervals)

---

## Ubiquitous Language

### Core Terms

| Term | Definition | Context |
|------|------------|---------|
| **Methylation Level** | Fraction of reads showing methylation at a cytosine (0.0-1.0) | Core metric |
| **CpG Context** | Cytosine followed by guanine; the primary methylation target in mammals | Sequence context |
| **CHG/CHH Context** | Non-CpG methylation contexts (H = A, C, or T); prevalent in plants | Sequence context |
| **Chromatin Accessibility** | Degree to which genomic DNA is physically accessible to transcription factors | Core metric |
| **Peak** | A statistically significant region of enriched signal (accessibility or histone mark) | Statistical entity |
| **Histone Mark** | A covalent post-translational modification on a histone tail (e.g., H3K4me3, H3K27ac) | Modification type |
| **TAD** | Topologically Associating Domain; a self-interacting genomic region in 3D space | 3D structure |
| **Compartment** | A/B classification of chromatin: A = active/open, B = inactive/closed | 3D structure |
| **Contact Map** | Matrix of interaction frequencies between genomic loci from Hi-C data | 3D data |
| **Enhancer** | A distal regulatory element that activates gene expression | Functional annotation |
| **Promoter** | Region immediately upstream of a gene's transcription start site | Functional annotation |

### Algorithmic Terms

| Term | Definition | Context |
|------|------------|---------|
| **Insulation Score** | Local measure of boundary strength between adjacent TADs | TAD detection |
| **Graph Cut** | Partition of the contact map graph that identifies TAD boundaries via min-cut | RuVector integration |
| **Attention Score** | Learned weight for long-range locus-to-locus interactions | Enhancer prediction |
| **Hyperbolic Embedding** | Representation of hierarchical chromatin states in Poincare ball space | State search |

### SOTA Terms (2024-2026)

| Term | Definition | Context |
|------|------------|---------|
| **Coupled Autoencoder** | Joint encoder that maps scRNA-seq + scATAC-seq + CUT&Tag to a shared latent space per cell | Single-cell multi-omics |
| **Persistent Homology** | Algebraic topology method that tracks birth/death of topological features (connected components, loops) across filtration scales | TAD boundary detection |
| **Betti Curve** | Summary statistic of persistent homology: count of k-dimensional holes as a function of filtration parameter | TAD boundary scoring |
| **Epigenetic Clock** | Regression model predicting biological age from DNA methylation at specific CpG sites | Methylation clock |
| **Enformer** | Deep learning architecture combining convolutional layers with transformer attention over 200 kb sequence context to predict epigenomic tracks | Accessibility prediction |
| **Biomolecular Condensate** | Membrane-less organelle formed by liquid-liquid phase separation of proteins with intrinsically disordered regions | Condensate modeling |
| **IDR** | Intrinsically Disordered Region; protein segment lacking stable 3D structure, drives phase separation via multivalent interactions | Condensate modeling |
| **Loop Extrusion** | Model in which cohesin/condensin actively extrude chromatin loops until blocked by CTCF, producing TADs | 4D nucleome |
| **4D Nucleome** | Time-resolved 3D genome organization capturing dynamic chromatin rearrangements across cell-cycle or differentiation | Temporal modeling |

---

## Bounded Contexts

### Context Map

```
+-----------------------------------------------------------------------------+
|                         EPIGENOMICS CONTEXT                                  |
|                           (Core Domain)                                      |
|  +---------------+  +---------------+  +---------------+  +---------------+  |
|  | Methylation   |  | Chromatin     |  | Histone       |  | 3D Genome     |  |
|  | Subcontext    |  | Subcontext    |  | Subcontext    |  | Subcontext    |  |
|  +---------------+  +---------------+  +---------------+  +---------------+  |
|  +---------------+  +---------------+  +---------------+  +---------------+  |
|  | SingleCell    |  | Methylation   |  | Condensate    |  | 4D Nucleome   |  |
|  | MultiOmics    |  | Clock         |  | Modeling      |  | Dynamics      |  |
|  +---------------+  +---------------+  +---------------+  +---------------+  |
+-----------------------------------------------------------------------------+
         |                  |                  |                  |
         | Upstream         | Upstream         | Upstream         | Upstream
         v                  v                  v                  v
+------------------+ +------------------+ +------------------+ +------------------+
|   ALIGNMENT      | |   PEAK CALLING   | |  CHIP-SEQ        | |   HI-C           |
|   PIPELINE       | |   PIPELINE       | |  PIPELINE        | |   PIPELINE       |
|  (Infrastructure)| | (Infrastructure) | | (Infrastructure) | | (Infrastructure) |
+------------------+ +------------------+ +------------------+ +------------------+
         |                                                            |
         | Downstream                                                 | Downstream
         v                                                            v
+------------------+                                         +------------------+
|   CRISPR         |                                         |   VARIANT        |
|   ENGINEERING    |                                         |   CALLING        |
|   CONTEXT        |                                         |   CONTEXT        |
+------------------+                                         +------------------+
```

### Epigenomics Context (Core)

**Responsibility**: Reconstruct the regulatory landscape from multi-omic epigenomic data.

**Key Aggregates**:
- MethylationProfile (Aggregate Root)
- ChromatinLandscape
- HistoneModificationMap
- GenomeTopology
- SingleCellMultiOmeProfile (SOTA)
- EpigeneticAgePrediction (SOTA)
- CondensateModel (SOTA)
- NucleomeDynamics (SOTA)

**Anti-Corruption Layers**:
- Alignment ACL (translates BAM records to methylation calls)
- HiC ACL (translates contact matrices to graph structures)
- Annotation ACL (translates gene annotations to promoter/enhancer loci)
- SingleCell ACL (translates 10x Multiome fragments to cell-by-feature matrices)

---

## Aggregates

### MethylationProfile (Root Aggregate)

The central aggregate for a sample's methylation state across a genomic region.

```
+-----------------------------------------------------------------------+
|                     METHYLATION PROFILE                                |
|                    (Aggregate Root)                                    |
+-----------------------------------------------------------------------+
|  profile_id: ProfileId                                                 |
|  sample_id: SampleId                                                   |
|  region: GenomicRegion { chromosome, start, end }                      |
|  assembly: Assembly (e.g., hg38, mm39)                                 |
|  sites: Vec<MethylationSite>                                           |
|  global_methylation: f64                                               |
|  dmr_calls: Vec<DifferentiallyMethylatedRegion>                        |
+-----------------------------------------------------------------------+
|  +------------------------------------------------------------------+ |
|  | MethylationSite (Entity)                                         | |
|  |  site_id: SiteId                                                 | |
|  |  position: GenomicPosition { chromosome: String, offset: u64 }   | |
|  |  methylation_level: f64 (0.0 - 1.0)                             | |
|  |  coverage: u32                                                   | |
|  |  context: MethylationContext { CpG | CHG | CHH }                | |
|  |  strand: Strand { Plus | Minus }                                 | |
|  |  confidence: f64                                                 | |
|  +------------------------------------------------------------------+ |
|  +------------------------------------------------------------------+ |
|  | DifferentiallyMethylatedRegion (Entity)                          | |
|  |  dmr_id: DmrId                                                  | |
|  |  region: GenomicRegion                                           | |
|  |  mean_delta: f64                                                 | |
|  |  p_value: f64                                                    | |
|  |  q_value: f64                                                    | |
|  |  direction: Direction { Hyper | Hypo }                           | |
|  +------------------------------------------------------------------+ |
+-----------------------------------------------------------------------+
|  Invariants:                                                           |
|  - methylation_level in [0.0, 1.0] for every site                     |
|  - coverage >= 1 for every reported site                               |
|  - sites sorted by genomic position                                    |
|  - All sites fall within the profile's region                          |
|  - q_value uses Benjamini-Hochberg correction                          |
+-----------------------------------------------------------------------+
```

### ChromatinLandscape (Aggregate)

Represents accessible chromatin regions for a sample.

```
+-----------------------------------------------------------------------+
|                    CHROMATIN LANDSCAPE                                  |
|                    (Aggregate Root)                                    |
+-----------------------------------------------------------------------+
|  landscape_id: LandscapeId                                             |
|  sample_id: SampleId                                                   |
|  assay_type: AssayType { ATACseq | DNaseseq | FAIREseq }              |
|  regions: Vec<ChromatinRegion>                                         |
|  footprints: Vec<TranscriptionFactorFootprint>                         |
+-----------------------------------------------------------------------+
|  +------------------------------------------------------------------+ |
|  | ChromatinRegion (Entity)                                         | |
|  |  region_id: RegionId                                             | |
|  |  start: u64                                                      | |
|  |  end: u64                                                        | |
|  |  chromosome: String                                              | |
|  |  accessibility_score: f64                                        | |
|  |  peak_summit: u64                                                | |
|  |  p_value: f64                                                    | |
|  |  fold_enrichment: f64                                            | |
|  |  motifs: Vec<MotifHit>                                           | |
|  +------------------------------------------------------------------+ |
|  +------------------------------------------------------------------+ |
|  | MotifHit (Value Object)                                          | |
|  |  motif_id: String                                                | |
|  |  transcription_factor: String                                    | |
|  |  position: u64                                                   | |
|  |  score: f64                                                      | |
|  |  strand: Strand                                                  | |
|  |  p_value: f64                                                    | |
|  +------------------------------------------------------------------+ |
+-----------------------------------------------------------------------+
|  Invariants:                                                           |
|  - accessibility_score >= 0.0                                          |
|  - end > start for every region                                        |
|  - Regions do not overlap after merging                                 |
|  - peak_summit falls within [start, end]                               |
+-----------------------------------------------------------------------+
```

### HistoneModificationMap (Aggregate)

Tracks histone marks across the genome for a single mark type and sample.

```
+-----------------------------------------------------------------------+
|                  HISTONE MODIFICATION MAP                              |
|                    (Aggregate Root)                                    |
+-----------------------------------------------------------------------+
|  map_id: MapId                                                         |
|  sample_id: SampleId                                                   |
|  mark_type: HistoneMarkType                                            |
|       { H3K4me1 | H3K4me3 | H3K27ac | H3K27me3 | H3K36me3 |         |
|         H3K9me3 | H4K20me1 | Custom(String) }                         |
|  modifications: Vec<HistoneModification>                               |
|  broad_domains: Vec<BroadDomain>                                       |
+-----------------------------------------------------------------------+
|  +------------------------------------------------------------------+ |
|  | HistoneModification (Entity)                                     | |
|  |  mod_id: ModId                                                   | |
|  |  position: GenomicPosition                                       | |
|  |  signal_intensity: f64                                           | |
|  |  fold_change: f64                                                | |
|  |  peak_type: PeakType { Narrow | Broad }                         | |
|  +------------------------------------------------------------------+ |
|  +------------------------------------------------------------------+ |
|  | BroadDomain (Entity)                                             | |
|  |  domain_id: DomainId                                             | |
|  |  region: GenomicRegion                                           | |
|  |  mean_signal: f64                                                | |
|  |  chromatin_state: ChromatinState { Active | Poised | Repressed } | |
|  +------------------------------------------------------------------+ |
+-----------------------------------------------------------------------+
|  Invariants:                                                           |
|  - signal_intensity >= 0.0                                             |
|  - fold_change >= 0.0                                                  |
|  - One map per (sample, mark_type) pair                                |
|  - BroadDomain regions do not overlap                                  |
+-----------------------------------------------------------------------+
```

### GenomeTopology (Aggregate)

Represents the 3D structure of the genome from Hi-C or similar data.

```
+-----------------------------------------------------------------------+
|                     GENOME TOPOLOGY                                    |
|                    (Aggregate Root)                                    |
+-----------------------------------------------------------------------+
|  topology_id: TopologyId                                               |
|  sample_id: SampleId                                                   |
|  resolution: u32 (bin size in bp: 1000, 5000, 10000, ...)             |
|  chromosome: String                                                    |
|  contact_graph: ContactGraph                                           |
|  tads: Vec<TAD>                                                        |
|  compartments: Vec<CompartmentSegment>                                 |
|  interactions: Vec<EnhancerPromoterInteraction>                        |
+-----------------------------------------------------------------------+
|  +------------------------------------------------------------------+ |
|  | ContactGraph (Value Object)                                      | |
|  |  bins: Vec<GenomicBin>                                           | |
|  |  contacts: SparseMatrix<f64> (ICE-normalized)                    | |
|  |  total_contacts: u64                                             | |
|  |                                                                  | |
|  |  fn as_dynamic_graph(&self) -> ruvector_mincut::DynamicGraph     | |
|  |  fn insulation_scores(&self, window: u32) -> Vec<f64>            | |
|  +------------------------------------------------------------------+ |
|  +------------------------------------------------------------------+ |
|  | TAD (Entity)                                                     | |
|  |  tad_id: TadId                                                   | |
|  |  boundary_left: GenomicPosition                                  | |
|  |  boundary_right: GenomicPosition                                 | |
|  |  insulation_score_left: f64                                      | |
|  |  insulation_score_right: f64                                     | |
|  |  intra_tad_contacts: f64                                         | |
|  |  inter_tad_contacts: f64                                         | |
|  |  compartment: Compartment { A | B | Intermediate }              | |
|  |  sub_tads: Vec<TadId>  (hierarchical nesting)                    | |
|  +------------------------------------------------------------------+ |
|  +------------------------------------------------------------------+ |
|  | EnhancerPromoterInteraction (Entity)                             | |
|  |  interaction_id: InteractionId                                   | |
|  |  enhancer_locus: GenomicRegion                                   | |
|  |  promoter_locus: GenomicRegion                                   | |
|  |  target_gene: String                                             | |
|  |  interaction_strength: f64                                       | |
|  |  linear_distance: u64 (bp)                                      | |
|  |  attention_score: f64 (from ruvector-attention)                  | |
|  |  confidence: f64                                                 | |
|  +------------------------------------------------------------------+ |
|  +------------------------------------------------------------------+ |
|  | CompartmentSegment (Value Object)                                | |
|  |  region: GenomicRegion                                           | |
|  |  compartment: Compartment                                        | |
|  |  eigenvector_value: f64 (PC1 of contact matrix)                  | |
|  +------------------------------------------------------------------+ |
+-----------------------------------------------------------------------+
|  Invariants:                                                           |
|  - TAD boundaries are non-overlapping at the same hierarchy level      |
|  - intra_tad_contacts > inter_tad_contacts (TAD definition)           |
|  - Contact matrix is symmetric                                         |
|  - Resolution must be a positive integer divisor of chromosome length  |
|  - interaction_strength in [0.0, 1.0]                                  |
|  - sub_tads are strictly contained within parent TAD boundaries        |
+-----------------------------------------------------------------------+
```

### SingleCellMultiOmeProfile (SOTA Aggregate)

Joint representation of single-cell multi-omics measurements (scRNA-seq + scATAC-seq + CUT&Tag) using coupled autoencoders for latent space alignment.

**References**:
- Hao, Y. et al. "Dictionary learning for integrative, multimodal and scalable single-cell analysis." Nature Biotechnology, 2024. (WNN / bridge integration)
- Gong, B. et al. "cobolt: integrative analysis of multimodal single-cell sequencing data." Genome Biology, 2021. (multi-omics VAE)
- Ashuach, T. et al. "MultiVI: deep generative model for the integration of multimodal data." Nature Methods, 2023.

```
+-----------------------------------------------------------------------+
|               SINGLE-CELL MULTI-OME PROFILE                           |
|                    (Aggregate Root)                                    |
+-----------------------------------------------------------------------+
|  profile_id: ScMultiOmeProfileId                                       |
|  sample_id: SampleId                                                   |
|  cell_count: usize                                                     |
|  modalities: Vec<OmicsModality>                                        |
|  latent_space: CoupledLatentSpace                                      |
|  cell_clusters: Vec<CellCluster>                                       |
|  gene_regulatory_network: Option<GeneRegulatoryNetwork>                |
+-----------------------------------------------------------------------+
|  +------------------------------------------------------------------+ |
|  | OmicsModality (Entity)                                           | |
|  |  modality_type: ModalityType                                     | |
|  |       { ScRNAseq | ScATACseq | CUTandTag(HistoneMarkType)       | |
|  |       | ScBisulfite | ScCUTandRUN(HistoneMarkType) }             | |
|  |  cell_ids: Vec<CellBarcode>                                      | |
|  |  feature_count: usize                                            | |
|  |  feature_matrix: SparseMatrix<f32>  (cells x features)           | |
|  |  quality_metrics: ModalityQC                                     | |
|  +------------------------------------------------------------------+ |
|  +------------------------------------------------------------------+ |
|  | CoupledLatentSpace (Value Object)                                | |
|  |  -- Joint embedding from coupled VAE / MultiVI --                | |
|  |  latent_dim: usize (typically 20-50)                             | |
|  |  cell_embeddings: Matrix<f32>  (cells x latent_dim)              | |
|  |  modality_weights: Vec<f64>  (learned per-modality contribution) | |
|  |  reconstruction_loss: f64                                        | |
|  |  kl_divergence: f64                                              | |
|  |                                                                  | |
|  |  fn get_cell_embedding(&self, cell: CellBarcode) -> Vec<f32>     | |
|  |  fn cross_modality_impute(                                       | |
|  |      &self, source: ModalityType, target: ModalityType,          | |
|  |      cell: CellBarcode                                           | |
|  |  ) -> Vec<f32>                                                   | |
|  +------------------------------------------------------------------+ |
|  +------------------------------------------------------------------+ |
|  | CellCluster (Entity)                                             | |
|  |  cluster_id: ClusterId                                           | |
|  |  cell_ids: Vec<CellBarcode>                                      | |
|  |  cell_type_annotation: Option<String>                            | |
|  |  marker_genes: Vec<(String, f64)>                                | |
|  |  marker_peaks: Vec<(GenomicRegion, f64)>                         | |
|  |  mean_embedding: Vec<f32>                                        | |
|  +------------------------------------------------------------------+ |
|  +------------------------------------------------------------------+ |
|  | GeneRegulatoryNetwork (Entity)                                   | |
|  |  -- Inferred from joint RNA + ATAC via SCENIC+ / FigR --        | |
|  |  grn_id: GrnId                                                   | |
|  |  tf_target_edges: Vec<TfTargetEdge>                              | |
|  |  enhancer_gene_links: Vec<EnhancerGeneLink>                      | |
|  |  num_tfs: usize                                                  | |
|  |  num_target_genes: usize                                         | |
|  +------------------------------------------------------------------+ |
+-----------------------------------------------------------------------+
|  Invariants:                                                           |
|  - All modalities share a common set of cell barcodes (or a subset)    |
|  - latent_dim is consistent across the CoupledLatentSpace              |
|  - cell_embeddings row count == cell_count                             |
|  - modality_weights sum to 1.0                                         |
|  - CellCluster assignments partition the cell set (no overlap)         |
+-----------------------------------------------------------------------+
```

### EpigeneticAgePrediction (SOTA Aggregate)

DNA methylation clock predictions using Horvath/Hannum elastic net models on CpG sites.

**References**:
- Horvath, S. "DNA methylation age of human tissues and cell types." Genome Biology, 2013. (353 CpG multi-tissue clock)
- Hannum, G. et al. "Genome-wide methylation profiles reveal quantitative views of human aging rates." Molecular Cell, 2013. (71 CpG blood clock)
- Lu, A.T. et al. "DNA methylation GrimAge version 2." Aging, 2022. (GrimAge2 mortality predictor)
- Belsky, D.W. et al. "DunedinPACE, a DNA methylation biomarker of the pace of aging." eLife, 2022. (rate-of-aging clock)

```
+-----------------------------------------------------------------------+
|               EPIGENETIC AGE PREDICTION                                |
|                    (Aggregate Root)                                    |
+-----------------------------------------------------------------------+
|  prediction_id: AgePredictionId                                        |
|  sample_id: SampleId                                                   |
|  methylation_profile_id: ProfileId                                     |
|  clock_results: Vec<ClockResult>                                       |
|  age_acceleration: Option<AgeAcceleration>                             |
+-----------------------------------------------------------------------+
|  +------------------------------------------------------------------+ |
|  | ClockResult (Entity)                                             | |
|  |  clock_type: EpigeneticClockType                                 | |
|  |       { Horvath353 | HannumBlood | PhenoAge | GrimAge2 |        | |
|  |         DunedinPACE | SkinBlood | PediatricBonerol |             | |
|  |         Custom(String) }                                         | |
|  |  predicted_age: f64 (years)                                      | |
|  |  chronological_age: Option<f64>                                  | |
|  |  cpg_sites_used: usize                                           | |
|  |  cpg_sites_available: usize                                      | |
|  |  confidence_interval: (f64, f64)  (95% CI)                       | |
|  |  model_coefficients: Vec<ClockCpGWeight>                         | |
|  +------------------------------------------------------------------+ |
|  +------------------------------------------------------------------+ |
|  | ClockCpGWeight (Value Object)                                    | |
|  |  cpg_id: String  (e.g., "cg00075967")                           | |
|  |  chromosome: String                                              | |
|  |  position: u64                                                   | |
|  |  methylation_level: f64                                          | |
|  |  elastic_net_coefficient: f64                                    | |
|  |  contribution_to_age: f64  (coefficient * methylation)           | |
|  +------------------------------------------------------------------+ |
|  +------------------------------------------------------------------+ |
|  | AgeAcceleration (Value Object)                                   | |
|  |  -- Residual from regressing DNAm age on chronological age --    | |
|  |  acceleration_years: f64 (positive = older than expected)        | |
|  |  acceleration_type: AccelerationType                             | |
|  |       { Intrinsic | Extrinsic | GrimAgeAccel | PACEAccel }     | |
|  |  percentile: f64  (population percentile)                        | |
|  +------------------------------------------------------------------+ |
+-----------------------------------------------------------------------+
|  Invariants:                                                           |
|  - predicted_age >= 0.0                                                |
|  - cpg_sites_used <= cpg_sites_available                               |
|  - confidence_interval.0 <= predicted_age <= confidence_interval.1     |
|  - acceleration computed only when chronological_age is provided       |
+-----------------------------------------------------------------------+
```

### CondensateModel (SOTA Aggregate)

Models biomolecular condensates formed by phase separation of transcription factors with intrinsically disordered regions (IDRs).

**References**:
- Sabari, B. et al. "Coactivator condensation at super-enhancers links phase separation and gene control." Science, 2018.
- Boija, A. et al. "Transcription Factors Activate Genes through the Phase-Separation Capacity of Their Activation Domains." Cell, 2018.
- Erdel, F. & Rippe, K. "Formation of Chromatin Subcompartments by Phase Separation." Biophysical Journal, 2018.
- Lancaster, A.K. et al. "PLAAC: a web and command-line application to identify proteins with prion-like amino acid composition." Bioinformatics, 2014.

```
+-----------------------------------------------------------------------+
|                   CONDENSATE MODEL                                     |
|                    (Aggregate Root)                                    |
+-----------------------------------------------------------------------+
|  condensate_id: CondensateId                                           |
|  locus: GenomicRegion  (super-enhancer or promoter cluster)            |
|  constituent_tfs: Vec<TranscriptionFactorIDR>                          |
|  phase_separation_score: f64 (0.0-1.0)                                |
|  partition_coefficient: f64                                            |
|  condensate_type: CondensateType                                       |
|       { SuperEnhancer | Heterochromatin | Nucleolar |                 |
|         SplicingSpeckle | Custom(String) }                            |
+-----------------------------------------------------------------------+
|  +------------------------------------------------------------------+ |
|  | TranscriptionFactorIDR (Entity)                                  | |
|  |  tf_name: String                                                 | |
|  |  idr_regions: Vec<ProteinRegion>                                 | |
|  |  idr_fraction: f64  (fraction of protein that is disordered)     | |
|  |  prion_like_score: f64  (PLAAC/PLD score)                        | |
|  |  charge_pattern: ChargeBlockiness                                | |
|  |  aromatic_content: f64  (Tyr+Phe+Trp fraction in IDR)           | |
|  |  valence: u32  (number of interaction stickers)                  | |
|  |  saturation_concentration: f64  (uM, predicted)                  | |
|  +------------------------------------------------------------------+ |
|  +------------------------------------------------------------------+ |
|  | ChargeBlockiness (Value Object)                                  | |
|  |  -- Das-Pappu kappa parameter for charge patterning --           | |
|  |  kappa: f64 (0=well-mixed, 1=fully segregated)                   | |
|  |  net_charge_per_residue: f64                                     | |
|  |  fraction_positive: f64                                          | |
|  |  fraction_negative: f64                                          | |
|  +------------------------------------------------------------------+ |
+-----------------------------------------------------------------------+
|  Invariants:                                                           |
|  - phase_separation_score in [0.0, 1.0]                                |
|  - idr_fraction in [0.0, 1.0]                                          |
|  - At least one TF constituent per condensate                          |
|  - partition_coefficient > 0.0                                          |
+-----------------------------------------------------------------------+
```

### NucleomeDynamics (SOTA Aggregate)

Time-resolved 4D nucleome model capturing chromatin organization dynamics using polymer loop-extrusion simulations.

**References**:
- Fudenberg, G. et al. "Formation of Chromosomal Domains by Loop Extrusion." Cell Reports, 2016.
- Sanborn, A.L. et al. "Chromatin extrusion explains key features of loop and domain formation." PNAS, 2015.
- Abramo, K. et al. "A chromosome folding intermediate at the condensin-to-cohesin transition during telophase." Nature Cell Biology, 2019.
- Dekker, J. et al. "The 4D nucleome project." Nature, 2017.

```
+-----------------------------------------------------------------------+
|                   NUCLEOME DYNAMICS                                     |
|                    (Aggregate Root)                                    |
+-----------------------------------------------------------------------+
|  dynamics_id: NucleomeDynamicsId                                       |
|  sample_id: SampleId                                                   |
|  chromosome: String                                                    |
|  time_points: Vec<TimePointSnapshot>                                   |
|  loop_extrusion_params: LoopExtrusionParams                            |
|  trajectory: PolymerTrajectory                                         |
+-----------------------------------------------------------------------+
|  +------------------------------------------------------------------+ |
|  | TimePointSnapshot (Entity)                                       | |
|  |  time_point: f64  (hours or cell-cycle fraction)                 | |
|  |  condition: String  (e.g., "G1", "S-phase", "mitosis")          | |
|  |  contact_graph: ContactGraph                                     | |
|  |  tads: Vec<TAD>                                                  | |
|  |  compartments: Vec<CompartmentSegment>                           | |
|  |  loop_anchors: Vec<LoopAnchor>                                   | |
|  +------------------------------------------------------------------+ |
|  +------------------------------------------------------------------+ |
|  | LoopExtrusionParams (Value Object)                               | |
|  |  cohesin_loading_rate: f64  (per kb per minute)                  | |
|  |  cohesin_unloading_rate: f64  (per minute)                       | |
|  |  extrusion_speed: f64  (kb per second)                           | |
|  |  ctcf_binding_sites: Vec<CtcfSite>                               | |
|  |  ctcf_blocking_probability: f64  (0.0-1.0)                      | |
|  |  condensin_present: bool  (mitotic condensin)                    | |
|  +------------------------------------------------------------------+ |
|  +------------------------------------------------------------------+ |
|  | CtcfSite (Value Object)                                          | |
|  |  position: GenomicPosition                                       | |
|  |  orientation: CtcfOrientation { Forward | Reverse }              | |
|  |  binding_strength: f64                                           | |
|  |  motif_score: f64                                                | |
|  +------------------------------------------------------------------+ |
|  +------------------------------------------------------------------+ |
|  | LoopAnchor (Value Object)                                        | |
|  |  anchor_left: GenomicPosition                                    | |
|  |  anchor_right: GenomicPosition                                   | |
|  |  loop_strength: f64                                              | |
|  |  ctcf_convergent: bool  (convergent CTCF orientation)            | |
|  +------------------------------------------------------------------+ |
|  +------------------------------------------------------------------+ |
|  | PolymerTrajectory (Value Object)                                 | |
|  |  -- Polymer simulation output --                                 | |
|  |  monomer_count: usize  (number of polymer beads)                 | |
|  |  bp_per_monomer: u32                                             | |
|  |  frames: Vec<PolymerFrame>  (3D coords per time step)            | |
|  |  mean_squared_displacement: Vec<f64>                             | |
|  +------------------------------------------------------------------+ |
+-----------------------------------------------------------------------+
|  Invariants:                                                           |
|  - time_points sorted by time_point ascending                          |
|  - At least 2 time points for temporal analysis                        |
|  - ctcf_blocking_probability in [0.0, 1.0]                            |
|  - Loop anchors have convergent CTCF if ctcf_convergent == true        |
|  - monomer_count * bp_per_monomer covers the chromosome region         |
+-----------------------------------------------------------------------+
```

---

## Value Objects

### GenomicPosition

```rust
struct GenomicPosition {
    chromosome: String,
    offset: u64,
}

impl GenomicPosition {
    fn distance_to(&self, other: &GenomicPosition) -> Option<u64>;
    fn is_same_chromosome(&self, other: &GenomicPosition) -> bool;
}
```

### GenomicRegion

```rust
struct GenomicRegion {
    chromosome: String,
    start: u64,
    end: u64,
}

impl GenomicRegion {
    fn length(&self) -> u64 { self.end - self.start }
    fn overlaps(&self, other: &GenomicRegion) -> bool;
    fn contains(&self, pos: &GenomicPosition) -> bool;
    fn midpoint(&self) -> u64 { (self.start + self.end) / 2 }
}
```

### InsulationScore

```rust
struct InsulationScore {
    position: GenomicPosition,
    score: f64,
    window_size: u32,
    is_boundary: bool,
    boundary_strength: f64,
}

impl InsulationScore {
    fn is_tad_boundary(&self, threshold: f64) -> bool {
        self.is_boundary && self.boundary_strength >= threshold
    }
}
```

### ChromatinStateVector

Embedding of chromatin state for hyperbolic search.

```rust
struct ChromatinStateVector {
    methylation: f64,
    accessibility: f64,
    h3k4me3: f64,
    h3k27ac: f64,
    h3k27me3: f64,
    compartment_score: f64,
}

impl ChromatinStateVector {
    fn to_embedding(&self) -> Vec<f32>;
    fn bivalent_score(&self) -> f64 {
        (self.h3k4me3 * self.h3k27me3).sqrt()
    }
}
```

### SOTA Value Objects

#### PersistentHomologyResult

Persistent homology output for TAD boundary detection from Hi-C contact maps.

**References**:
- Edelsbrunner, H. & Harer, J. "Computational Topology." AMS, 2010.
- Cang, Z. & Wei, G.-W. "TopologyNet: Topology based deep convolutional and multi-task neural networks for biomolecular property predictions." PLoS Computational Biology, 2017.
- Carriere, M. et al. "Persistent homology for Hi-C data analysis." bioRxiv, 2020.

```rust
/// Represents a single topological feature tracked through a filtration
/// of the Hi-C contact matrix.
///
/// Maps to ruvector-mincut for complementary boundary detection:
/// min-cut finds boundaries by severing weak inter-TAD contacts,
/// while persistent homology finds them as topological transitions.
struct PersistenceInterval {
    dimension: u8,           // 0 = connected component, 1 = loop/cycle
    birth: f64,              // Filtration value where feature appears
    death: f64,              // Filtration value where feature disappears
    persistence: f64,        // death - birth; long-lived = significant
    representative_cycle: Option<Vec<(usize, usize)>>,  // Simplices in cycle
}

impl PersistenceInterval {
    fn is_significant(&self, threshold: f64) -> bool {
        self.persistence >= threshold
    }
}

/// Full persistent homology result for a Hi-C contact map region.
/// Uses ruvector-hyperbolic-hnsw for efficient nearest-neighbor
/// queries during Vietoris-Rips complex construction.
struct PersistentHomologyResult {
    intervals: Vec<PersistenceInterval>,
    betti_curve_0: Vec<(f64, usize)>,  // (filtration, B0 count)
    betti_curve_1: Vec<(f64, usize)>,  // (filtration, B1 count)
    tad_boundaries_detected: Vec<GenomicPosition>,  // Inferred from H0 deaths
    wasserstein_distance: Option<f64>,  // Distance to reference topology
}
```

#### EpigeneticClockCoefficients

```rust
/// Pre-trained elastic net coefficients for a specific epigenetic clock.
/// These are loaded from published models (Horvath 2013, Hannum 2013, etc.)
/// and applied to MethylationProfile data.
struct EpigeneticClockCoefficients {
    clock_type: EpigeneticClockType,
    intercept: f64,
    cpg_weights: Vec<(String, f64)>,  // (CpG probe ID, elastic net coefficient)
    age_transformation: AgeTransformation,  // Linear | LogLinearHorvath
    training_samples: usize,
    median_absolute_error: f64,  // reported MAE from original publication
}

/// Horvath's clock uses a custom age transformation:
/// F(age) = log(age+1) - log(adult_age+1) for age < adult_age
///        = (age - adult_age) / (adult_age + 1) for age >= adult_age
enum AgeTransformation {
    Linear,
    LogLinearHorvath { adult_age: f64 },
}
```

#### EnformerPrediction

Chromatin accessibility prediction from DNA sequence using Enformer-style deep models.

**References**:
- Avsec, Z. et al. "Effective gene expression prediction from sequence by integrating long-range interactions." Nature Methods, 2021. (Enformer)
- Linder, J. et al. "Predicting RNA-seq coverage from DNA sequence as a unifying model of gene regulation." bioRxiv, 2023.

```rust
/// Predicted chromatin accessibility signal from DNA sequence,
/// generated by an Enformer-style convolutional + transformer model.
///
/// Integrates with ruvector-attention for the transformer layers:
/// the model's self-attention heads learn long-range regulatory
/// grammar (enhancer-promoter interactions) from 200 kb context.
struct EnformerPrediction {
    sequence_region: GenomicRegion,
    context_window: u64,         // Total input window (e.g., 196_608 bp)
    target_window: u64,          // Central prediction window (e.g., 114_688 bp)
    bin_size: u32,               // Output resolution (128 bp bins)
    predicted_tracks: Vec<PredictedTrack>,
    attribution_scores: Option<Vec<f64>>,  // ISM or DeepLIFT per-base importance
}

struct PredictedTrack {
    track_name: String,          // e.g., "ATAC-seq", "H3K27ac", "DNase"
    cell_type: String,
    values: Vec<f64>,            // Predicted signal per output bin
    pearson_r: Option<f64>,      // Correlation with observed (if available)
}
```

#### CondensatePartitionCoefficient

```rust
/// Quantifies preferential partitioning of a molecule into a condensate
/// versus the dilute phase. Used by CondensateModel to predict which
/// transcription factors concentrate at super-enhancers.
struct CondensatePartitionCoefficient {
    protein_name: String,
    condensate_concentration: f64,  // uM, inside condensate
    dilute_concentration: f64,      // uM, outside condensate
    partition_coefficient: f64,     // condensate / dilute
    temperature_kelvin: f64,
}

impl CondensatePartitionCoefficient {
    fn is_enriched(&self) -> bool {
        self.partition_coefficient > 2.0
    }
}
```

#### LoopExtrusionSimulationConfig

```rust
/// Configuration for polymer loop-extrusion simulations of the 4D nucleome.
/// Uses OpenMM-style energy functions translated to Rust for integration
/// with ruvector-gnn (GNN predicts CTCF occupancy feeding into simulation).
struct LoopExtrusionSimulationConfig {
    chromosome_length_bp: u64,
    bp_per_monomer: u32,           // Resolution (typically 1-5 kb)
    num_cohesins: usize,           // Number of cohesin complexes
    cohesin_processivity: f64,     // Mean extrusion distance in kb
    ctcf_sites: Vec<CtcfSite>,
    ctcf_block_prob: f64,          // P(cohesin stops at CTCF) per encounter
    simulation_steps: usize,
    time_step_seconds: f64,
    confinement_radius: f64,       // Nuclear confinement in um
    self_avoidance: bool,          // Excluded volume interactions
}
```

---

## Domain Events

### Methylation Events

| Event | Trigger | Payload |
|-------|---------|---------|
| `MethylationCalled` | Bisulfite/nanopore processing complete | profile_id, site_count, mean_level |
| `DMRIdentified` | Differential analysis complete | dmr_id, region, delta, direction |
| `MethylationDriftDetected` | Temporal comparison | region, old_level, new_level, magnitude |

### Chromatin Events

| Event | Trigger | Payload |
|-------|---------|---------|
| `PeaksCalled` | Peak calling pipeline complete | landscape_id, peak_count, frip_score |
| `FootprintDetected` | Motif analysis complete | tf_name, position, score |
| `AccessibilityChanged` | Differential accessibility | region, fold_change, direction |

### Histone Events

| Event | Trigger | Payload |
|-------|---------|---------|
| `HistoneMarksProfiled` | ChIP-seq processing complete | map_id, mark_type, peak_count |
| `ChromatinStateAssigned` | Multi-mark integration | region, state, confidence |
| `BivalentDomainIdentified` | H3K4me3+H3K27me3 co-occurrence | region, bivalent_score |

### 3D Genome Events

| Event | Trigger | Payload |
|-------|---------|---------|
| `ContactMapBuilt` | Hi-C pipeline complete | topology_id, resolution, bin_count |
| `TADBoundaryDetected` | Min-cut analysis complete | tad_id, boundaries, cut_value |
| `CompartmentSwitched` | A/B compartment change detected | region, old_compartment, new_compartment |
| `EnhancerPromoterLinked` | Interaction prediction complete | enhancer, promoter, strength, gene |
| `TopologyReconstructed` | Full 3D model built | topology_id, tad_count, interaction_count |

### SOTA Domain Events

| Event | Trigger | Payload |
|-------|---------|---------|
| `SingleCellProfileIntegrated` | Coupled autoencoder training converged | profile_id, cell_count, modalities, reconstruction_loss |
| `CellClustersIdentified` | Leiden/Louvain clustering on latent space | profile_id, cluster_count, modularity |
| `GeneRegulatoryNetworkInferred` | SCENIC+/FigR GRN inference complete | grn_id, tf_count, edge_count |
| `EpigeneticAgePredicted` | Clock regression applied to methylation | prediction_id, clock_type, predicted_age, acceleration |
| `AgeAccelerationDetected` | Significant deviation from chronological age | prediction_id, acceleration_years, percentile |
| `TADBoundaryDetectedByHomology` | Persistent homology identifies boundary | topology_id, boundary_position, persistence_value |
| `AccessibilityPredictedFromSequence` | Enformer model inference complete | region, track_count, mean_pearson_r |
| `CondensateFormed` | Phase separation score exceeds threshold | condensate_id, locus, constituent_tfs, phase_score |
| `LoopExtrusionSimulated` | Polymer simulation complete | dynamics_id, frames, mean_loop_size |
| `ChromatinDynamicsChanged` | Temporal comparison of 4D nucleome | dynamics_id, time_a, time_b, tad_changes, loop_changes |

---

## Domain Services

### MethylationCaller

Converts aligned bisulfite-seq or nanopore reads to per-site methylation calls.

```rust
trait MethylationCaller {
    /// Call methylation from aligned reads
    async fn call(
        &self,
        alignments: &AlignmentSource,
        region: &GenomicRegion,
        config: &MethylationCallerConfig,
    ) -> Result<MethylationProfile, EpigenomicsError>;

    /// Identify differentially methylated regions between two profiles
    fn call_dmrs(
        &self,
        control: &MethylationProfile,
        treatment: &MethylationProfile,
        config: &DmrConfig,
    ) -> Result<Vec<DifferentiallyMethylatedRegion>, EpigenomicsError>;

    /// Supported calling modes
    fn supported_modes(&self) -> Vec<CallingMode>;
}

struct MethylationCallerConfig {
    min_coverage: u32,
    min_base_quality: u8,
    context_filter: Vec<MethylationContext>,
    calling_mode: CallingMode,  // Bisulfite | Nanopore | EMseq
}
```

### ChromatinPeakCaller

Identifies accessible chromatin regions from ATAC-seq or DNase-seq data.

```rust
trait ChromatinPeakCaller {
    /// Call peaks from accessibility data
    async fn call_peaks(
        &self,
        signal: &SignalTrack,
        control: Option<&SignalTrack>,
        config: &PeakCallerConfig,
    ) -> Result<ChromatinLandscape, EpigenomicsError>;

    /// Identify TF footprints within accessible regions
    fn call_footprints(
        &self,
        landscape: &ChromatinLandscape,
        motif_database: &MotifDatabase,
    ) -> Result<Vec<TranscriptionFactorFootprint>, EpigenomicsError>;
}

struct PeakCallerConfig {
    q_value_threshold: f64,
    min_peak_length: u32,
    max_gap: u32,
    peak_model: PeakModel,  // Narrow | Broad | Mixed
}
```

### TADDetector

Detects TAD boundaries from Hi-C contact maps using `ruvector-mincut`.

```rust
trait TADDetector {
    /// Detect TADs from contact map using graph min-cut
    ///
    /// Algorithm:
    /// 1. Convert contact map to DynamicGraph (bins = vertices, contacts = weighted edges)
    /// 2. Apply ruvector-mincut to find minimum cuts that partition the graph
    /// 3. Boundaries correspond to min-cut edges in the contact graph
    /// 4. Hierarchical TADs via recursive application at multiple resolutions
    async fn detect_tads(
        &self,
        contact_graph: &ContactGraph,
        config: &TadDetectorConfig,
    ) -> Result<Vec<TAD>, EpigenomicsError>;

    /// Detect hierarchical TAD structure using j-tree decomposition
    fn detect_hierarchical(
        &self,
        contact_graph: &ContactGraph,
        resolutions: &[u32],
    ) -> Result<Vec<TAD>, EpigenomicsError>;

    /// Compute insulation scores using graph connectivity
    fn compute_insulation(
        &self,
        contact_graph: &ContactGraph,
        window_sizes: &[u32],
    ) -> Result<Vec<InsulationScore>, EpigenomicsError>;
}

struct TadDetectorConfig {
    min_tad_size: u32,         // Minimum TAD size in bins
    max_tad_size: u32,         // Maximum TAD size in bins
    boundary_strength: f64,     // Min-cut threshold for boundary calls
    hierarchical: bool,         // Enable nested TAD detection
    algorithm: TadAlgorithm,    // MinCut | Insulation | Armatus | Dixon
}

/// TAD detection via ruvector-mincut integration
enum TadAlgorithm {
    /// Use ruvector_mincut::MinCutBuilder for exact boundary detection
    MinCut,
    /// Use ruvector_mincut::SubpolynomialMinCut for dynamic updates
    DynamicMinCut,
    /// Use ruvector_mincut::jtree::JTreeHierarchy for hierarchical detection
    JTreeHierarchical,
    /// Classical insulation score method
    Insulation,
    /// Armatus algorithm
    Armatus,
    /// Dixon et al. directionality index
    Dixon,
    /// Persistent homology via Vietoris-Rips filtration (SOTA)
    PersistentHomology,
}
```

### EnhancerPredictor

Multi-omic integration for enhancer-promoter interaction prediction using `ruvector-attention`.

```rust
trait EnhancerPredictor {
    /// Predict enhancer-promoter interactions from multi-omic features
    ///
    /// Uses ruvector-attention's GraphAttention trait to learn long-range
    /// interaction weights between distal regulatory elements and promoters.
    async fn predict_interactions(
        &self,
        topology: &GenomeTopology,
        chromatin: &ChromatinLandscape,
        histones: &[HistoneModificationMap],
        methylation: &MethylationProfile,
        config: &EnhancerPredictorConfig,
    ) -> Result<Vec<EnhancerPromoterInteraction>, EpigenomicsError>;

    /// Score a single candidate interaction
    fn score_interaction(
        &self,
        enhancer: &ChromatinRegion,
        promoter: &GenomicRegion,
        features: &InteractionFeatures,
    ) -> Result<f64, EpigenomicsError>;
}

struct EnhancerPredictorConfig {
    max_distance: u64,          // Maximum linear distance to consider (e.g., 2 Mb)
    min_contact_score: f64,     // Minimum Hi-C interaction to seed candidates
    attention_heads: usize,     // Number of attention heads for interaction scoring
    attention_dim: usize,       // Attention dimension
    use_geometric: bool,        // Use GeometricAttention in hyperbolic space
    curvature: f32,             // Curvature for geometric attention (negative)
}
```

### ChromatinStateSearchService

Hierarchical chromatin state search using `ruvector-hyperbolic-hnsw`.

```rust
trait ChromatinStateSearchService {
    /// Index chromatin state vectors in hyperbolic space
    ///
    /// Chromatin states have natural hierarchy (e.g., active > promoter > TSS-proximal).
    /// Hyperbolic HNSW preserves this hierarchy during nearest-neighbor search.
    fn index_states(
        &self,
        states: &[(GenomicRegion, ChromatinStateVector)],
        config: &HyperbolicIndexConfig,
    ) -> Result<ChromatinStateIndex, EpigenomicsError>;

    /// Find genomic regions with similar chromatin state
    fn search_similar(
        &self,
        query: &ChromatinStateVector,
        k: usize,
    ) -> Result<Vec<(GenomicRegion, f64)>, EpigenomicsError>;
}

struct HyperbolicIndexConfig {
    curvature: f64,
    use_tangent_pruning: bool,
    prune_factor: usize,
    shard_by_chromosome: bool,
}
```

### Genome3DGraphService

3D genome graph analysis using `ruvector-gnn`.

```rust
trait Genome3DGraphService {
    /// Build a GNN over the 3D genome contact graph
    ///
    /// Nodes = genomic bins with epigenomic features
    /// Edges = Hi-C contacts weighted by interaction frequency
    /// The GNN propagates regulatory signals through 3D proximity.
    fn build_graph(
        &self,
        topology: &GenomeTopology,
        features: &BinFeatureMatrix,
    ) -> Result<GenomeGraph, EpigenomicsError>;

    /// Predict functional annotations using GNN message passing
    fn predict_function(
        &self,
        graph: &GenomeGraph,
        query_bin: usize,
        depth: usize,
    ) -> Result<FunctionalPrediction, EpigenomicsError>;
}
```

### SOTA Domain Services

#### SingleCellMultiOmicsIntegrator

Joint modeling of scRNA-seq + scATAC-seq + CUT&Tag using coupled variational autoencoders.

**References**:
- Ashuach, T. et al. "MultiVI: deep generative model for the integration of multimodal data." Nature Methods, 2023.
- Hao, Y. et al. "Dictionary learning for integrative, multimodal and scalable single-cell analysis." Nature Biotechnology, 2024.
- Gonzalez-Blas, C.B. et al. "SCENIC+: single-cell multiomic inference of enhancers and gene regulatory networks." Nature Methods, 2023.

```rust
/// Integrates multiple single-cell modalities into a shared latent space.
///
/// The coupled autoencoder architecture:
/// 1. Per-modality encoder: scRNA -> z_rna, scATAC -> z_atac, CUT&Tag -> z_ct
/// 2. Cross-modal alignment loss: minimize MMD(z_rna, z_atac) for shared cells
/// 3. Joint decoder: z_shared -> reconstructed counts for all modalities
///
/// Leverages ruvector-attention for the cross-attention layers that align
/// modalities, and ruvector-gnn for the gene regulatory network inference
/// (TF -> target gene edges learned from joint RNA+ATAC patterns).
trait SingleCellMultiOmicsIntegrator {
    /// Train coupled autoencoder on multi-modal single-cell data
    async fn integrate(
        &self,
        modalities: &[OmicsModality],
        config: &MultiOmicsIntegrationConfig,
    ) -> Result<SingleCellMultiOmeProfile, EpigenomicsError>;

    /// Impute missing modality for cells measured in only one assay
    fn impute_cross_modality(
        &self,
        profile: &SingleCellMultiOmeProfile,
        source_modality: ModalityType,
        target_modality: ModalityType,
        cells: &[CellBarcode],
    ) -> Result<SparseMatrix<f32>, EpigenomicsError>;

    /// Infer gene regulatory network from integrated data
    fn infer_grn(
        &self,
        profile: &SingleCellMultiOmeProfile,
        config: &GrnInferenceConfig,
    ) -> Result<GeneRegulatoryNetwork, EpigenomicsError>;
}

struct MultiOmicsIntegrationConfig {
    latent_dim: usize,                // Shared latent dimensions (20-50)
    n_epochs: usize,                  // Training epochs
    learning_rate: f64,               // Adam LR (typically 1e-3)
    batch_size: usize,
    alignment_weight: f64,            // Weight for cross-modal alignment loss
    kl_weight: f64,                   // Beta-VAE KL divergence weight
    use_adversarial_alignment: bool,  // Use discriminator for domain adaptation
    min_cells_per_feature: usize,     // Feature filtering threshold
    highly_variable_genes: usize,     // Number of HVGs to retain
}

struct GrnInferenceConfig {
    method: GrnMethod,  // SCENIC_Plus | FigR | CellOracle
    min_tf_expression: f64,
    min_peak_accessibility: f64,
    correlation_threshold: f64,
    use_motif_prior: bool,
}

enum GrnMethod {
    /// SCENIC+ (Gonzalez-Blas et al., 2023): motif-based TF-target inference
    ScenicPlus,
    /// FigR (Kartha et al., 2022): functional inference of gene regulation
    FigR,
    /// CellOracle (Kamimoto et al., 2023): GRN + in-silico perturbation
    CellOracle,
}
```

#### PersistentHomologyTADDetector

TAD boundary detection using persistent homology on Hi-C contact maps.

**References**:
- Carriere, M. et al. "Persistent homology based characterization of the contact map in Hi-C data." bioRxiv, 2020.
- Otter, N. et al. "A roadmap for the computation of persistent homology." EPJ Data Science, 2017.
- Cang, Z. & Wei, G.-W. "TopologyNet." PLoS Computational Biology, 2017.

```rust
/// Detects TAD boundaries by applying persistent homology to the Hi-C contact
/// matrix treated as a weighted simplicial complex.
///
/// Algorithm:
/// 1. Threshold the Hi-C contact matrix at varying scales (sublevel filtration)
/// 2. At each threshold, connected components in the contact graph = proto-TADs
/// 3. Track births/deaths of components via persistent homology (H0)
/// 4. Long-lived components correspond to genuine TADs
/// 5. Deaths correspond to TAD boundary merging events
///
/// Integration with ruvector-mincut:
/// - Min-cut boundaries and persistent homology boundaries are compared
/// - High concordance increases boundary confidence
/// - Persistent homology additionally detects higher-dimensional features
///   (H1 = chromatin loops) that min-cut alone cannot capture
trait PersistentHomologyTADDetector {
    /// Compute persistent homology of the Hi-C contact map
    fn compute_persistence(
        &self,
        contact_graph: &ContactGraph,
        config: &PersistenceConfig,
    ) -> Result<PersistentHomologyResult, EpigenomicsError>;

    /// Detect TAD boundaries from persistence diagram
    fn boundaries_from_persistence(
        &self,
        persistence: &PersistentHomologyResult,
        min_persistence: f64,
    ) -> Result<Vec<GenomicPosition>, EpigenomicsError>;

    /// Compare with min-cut boundaries for consensus calling
    fn consensus_boundaries(
        &self,
        mincut_boundaries: &[GenomicPosition],
        homology_boundaries: &[GenomicPosition],
        max_distance: u32,
    ) -> Vec<(GenomicPosition, f64)>;  // (position, confidence)
}

struct PersistenceConfig {
    filtration_type: FiltrationType,
    max_dimension: u8,           // 0 = components only, 1 = include loops
    persistence_threshold: f64,  // Minimum lifetime to call significant
    use_cubical: bool,           // Use cubical complex (faster for matrices)
}

enum FiltrationType {
    /// Sublevel set: threshold contact frequencies from low to high
    Sublevel,
    /// Rips complex: treat bins as points, contacts as distances
    VietorisRips,
    /// Alpha complex: Delaunay-based, exact but slower
    Alpha,
}
```

#### EpigeneticAgePredictor

Predicts biological age from DNA methylation using elastic net regression on CpG sites.

**References**:
- Horvath, S. "DNA methylation age of human tissues and cell types." Genome Biology, 2013.
- Hannum, G. et al. "Genome-wide methylation profiles reveal quantitative views of human aging rates." Molecular Cell, 2013.
- Lu, A.T. et al. "DNA methylation GrimAge strongly associates with dietary variables and predicts all-cause mortality." Aging, 2022.
- Belsky, D.W. et al. "DunedinPACE, a DNA methylation biomarker of the pace of aging." eLife, 2022.

```rust
/// Predicts epigenetic (biological) age from a MethylationProfile.
///
/// Implementation:
/// 1. Extract methylation levels at clock-specific CpG sites
/// 2. Apply pre-trained elastic net coefficients: age = intercept + sum(coeff_i * meth_i)
/// 3. Apply age transformation (Horvath uses log-linear for young ages)
/// 4. Compute age acceleration as residual from chronological age regression
///
/// Maps to ruvector-hyperbolic-hnsw: Clock CpG sites are indexed in
/// hyperbolic space by their tissue-specificity hierarchy, enabling
/// rapid lookup of clock-relevant sites across different tissue clocks.
trait EpigeneticAgePredictor {
    /// Predict epigenetic age using one or more clocks
    fn predict_age(
        &self,
        methylation: &MethylationProfile,
        clocks: &[EpigeneticClockCoefficients],
    ) -> Result<Vec<ClockResult>, EpigenomicsError>;

    /// Compute age acceleration (requires chronological age)
    fn compute_acceleration(
        &self,
        clock_result: &ClockResult,
        chronological_age: f64,
        reference_population: &AgeReferencePopulation,
    ) -> Result<AgeAcceleration, EpigenomicsError>;

    /// Train a custom clock from paired methylation + age data
    fn train_custom_clock(
        &self,
        training_data: &[(MethylationProfile, f64)],  // (profile, age)
        config: &ClockTrainingConfig,
    ) -> Result<EpigeneticClockCoefficients, EpigenomicsError>;
}

struct ClockTrainingConfig {
    regularization: ElasticNetParams,
    cross_validation_folds: usize,
    max_cpg_sites: usize,          // Maximum features to retain
    feature_selection: FeatureSelection,  // Variance | Correlation | Lasso
}

struct ElasticNetParams {
    alpha: f64,         // Mixing parameter: 0 = ridge, 1 = lasso
    lambda_path: Vec<f64>,  // Regularization strengths to test
    max_iterations: usize,
    tolerance: f64,
}
```

#### ChromatinAccessibilityPredictor

Sequence-based chromatin accessibility prediction using Enformer-style models.

**References**:
- Avsec, Z. et al. "Effective gene expression prediction from sequence by integrating long-range interactions." Nature Methods, 2021.
- Linder, J. et al. "Predicting RNA-seq coverage from DNA sequence." bioRxiv, 2023.
- Kelley, D.R. "Cross-species regulatory sequence activity prediction." PLoS Computational Biology, 2020. (Basenji2)

```rust
/// Predicts chromatin accessibility (ATAC-seq, DNase-seq) and histone
/// modification signals directly from DNA sequence context.
///
/// Architecture overview (Enformer-style):
/// 1. Convolutional stem: DNA one-hot -> local features (7 dilated conv layers)
/// 2. Transformer tower: self-attention over 1536 bins spanning ~200 kb
///    (implemented via ruvector-attention's MultiHeadAttention)
/// 3. Prediction head: per-bin regression for each epigenomic track
///
/// RuVector integration:
/// - ruvector-attention: Transformer self-attention layers with relative
///   positional encoding (Enformer uses 1536 bins with relative PE)
/// - ruvector-gnn: Optional GNN head for 3D-aware prediction refinement
trait ChromatinAccessibilityPredictor {
    /// Predict epigenomic tracks from DNA sequence
    fn predict_from_sequence(
        &self,
        sequence: &str,            // DNA sequence (196,608 bp for Enformer)
        region: &GenomicRegion,
        config: &EnformerConfig,
    ) -> Result<EnformerPrediction, EpigenomicsError>;

    /// Compute in-silico mutagenesis (ISM) scores
    ///
    /// For each position, compute the effect of every possible mutation
    /// on all predicted tracks. Identifies regulatory variants.
    fn in_silico_mutagenesis(
        &self,
        prediction: &EnformerPrediction,
        region_of_interest: &GenomicRegion,
    ) -> Result<Vec<VariantEffect>, EpigenomicsError>;

    /// Predict effect of a specific variant on chromatin accessibility
    fn predict_variant_effect(
        &self,
        reference_seq: &str,
        variant_seq: &str,
        region: &GenomicRegion,
    ) -> Result<Vec<PredictedTrack>, EpigenomicsError>;
}

struct EnformerConfig {
    model_path: String,           // Path to pre-trained ONNX/SafeTensors weights
    context_length: u64,          // Input sequence length (default: 196_608)
    target_length: u64,           // Output prediction length (default: 114_688)
    bin_size: u32,                // Output bin resolution (default: 128)
    num_heads: usize,             // Transformer attention heads (default: 8)
    num_transformer_layers: usize,// Transformer depth (default: 11)
    tracks_to_predict: Vec<String>,// Subset of output tracks
    batch_size: usize,
    use_gpu: bool,
}
```

#### CondensatePredictor

Predicts biomolecular condensate formation at regulatory loci from IDR properties.

**References**:
- Sabari, B. et al. "Coactivator condensation at super-enhancers links phase separation and gene control." Science, 2018.
- Boija, A. et al. "Transcription Factors Activate Genes through the Phase-Separation Capacity of Their Activation Domains." Cell, 2018.
- Ruff, K.M. et al. "Sequence grammar underlying the unfolding and phase separation of globular proteins." Molecular Cell, 2022.
- Lin, Y. et al. "Formation and Maturation of Phase-Separated Liquid Droplets by RNA-Binding Proteins." Molecular Cell, 2015.

```rust
/// Predicts whether a genomic locus (e.g., super-enhancer) will nucleate
/// a biomolecular condensate based on the IDR properties of transcription
/// factors bound there.
///
/// Integration with ruvector-gnn: TF binding sites and their IDR features
/// are nodes in a regulatory graph. GNN message passing propagates phase
/// separation propensity through the TF interaction network to predict
/// which loci form condensates.
///
/// Integration with ruvector-hyperbolic-hnsw: Condensate types form a
/// hierarchy (transcriptional condensate > super-enhancer condensate >
/// mediator condensate). Hyperbolic HNSW searches this hierarchy.
trait CondensatePredictor {
    /// Predict phase separation propensity for a locus
    fn predict_condensate(
        &self,
        locus: &GenomicRegion,
        bound_tfs: &[TranscriptionFactorIDR],
        chromatin: &ChromatinLandscape,
        config: &CondensateConfig,
    ) -> Result<CondensateModel, EpigenomicsError>;

    /// Predict IDR properties from protein sequence
    fn predict_idr(
        &self,
        protein_sequence: &str,
        protein_name: &str,
    ) -> Result<TranscriptionFactorIDR, EpigenomicsError>;

    /// Estimate saturation concentration for condensate formation
    fn estimate_saturation_concentration(
        &self,
        constituent_tfs: &[TranscriptionFactorIDR],
        temperature_kelvin: f64,
    ) -> Result<f64, EpigenomicsError>;
}

struct CondensateConfig {
    phase_separation_threshold: f64,  // Min score to call condensate formation
    include_rna_component: bool,      // Model RNA as condensate scaffold
    temperature_kelvin: f64,
    salt_concentration_mm: f64,       // Ionic strength affects phase behavior
    use_flory_huggins: bool,          // Use Flory-Huggins theory for phase diagram
}
```

#### NucleomeDynamicsSimulator

Time-resolved 4D nucleome modeling using polymer loop-extrusion simulations.

**References**:
- Fudenberg, G. et al. "Formation of Chromosomal Domains by Loop Extrusion." Cell Reports, 2016.
- Sanborn, A.L. et al. "Chromatin extrusion explains key features of loop and domain formation." PNAS, 2015.
- Nuebler, J. et al. "Chromatin organization by an interplay of loop extrusion and compartmental segregation." PNAS, 2018.
- Dekker, J. et al. "The 4D nucleome project." Nature, 2017.

```rust
/// Simulates time-resolved 3D genome organization using polymer models
/// with loop extrusion by cohesin/condensin motors.
///
/// Simulation algorithm:
/// 1. Initialize chromatin as a confined self-avoiding polymer chain
/// 2. Place cohesin complexes at random positions
/// 3. At each time step:
///    a. Cohesins extrude loops bidirectionally at extrusion_speed
///    b. If cohesin encounters CTCF in convergent orientation, block with P(block)
///    c. Cohesins unbind stochastically at unloading_rate
///    d. New cohesins load at loading_rate
///    e. Polymer chain relaxes under energy function (excluded volume + confinement)
/// 4. Sample contact maps at specified time points
///
/// RuVector integration:
/// - ruvector-gnn: Predicts CTCF binding strength and cohesin loading rates
///   from sequence and epigenomic features (GNN on regulatory element graph)
/// - ruvector-mincut: Validates simulated TAD boundaries against min-cut
///   boundaries from observed Hi-C data
/// - ruvector-attention: Attention-based comparison of simulated vs observed
///   contact maps for parameter fitting
trait NucleomeDynamicsSimulator {
    /// Run loop extrusion simulation
    fn simulate(
        &self,
        config: &LoopExtrusionSimulationConfig,
        initial_conditions: Option<&PolymerTrajectory>,
    ) -> Result<NucleomeDynamics, EpigenomicsError>;

    /// Fit simulation parameters to observed Hi-C data
    fn fit_to_hic(
        &self,
        observed_topology: &GenomeTopology,
        ctcf_sites: &[CtcfSite],
        config: &SimulationFitConfig,
    ) -> Result<LoopExtrusionParams, EpigenomicsError>;

    /// Compare simulated and observed contact maps
    fn compare_contact_maps(
        &self,
        simulated: &ContactGraph,
        observed: &ContactGraph,
    ) -> Result<ContactMapComparison, EpigenomicsError>;

    /// Predict dynamic TAD changes across conditions
    fn predict_dynamics(
        &self,
        condition_a_params: &LoopExtrusionParams,
        condition_b_params: &LoopExtrusionParams,
    ) -> Result<Vec<DynamicTADChange>, EpigenomicsError>;
}

struct SimulationFitConfig {
    optimization_method: OptMethod,  // GradientDescent | BayesianOpt | GridSearch
    max_iterations: usize,
    loss_function: ContactMapLoss,   // Pearson | SCC | HiCRep
    parameters_to_fit: Vec<FitParameter>,
}

enum ContactMapLoss {
    /// Pearson correlation of contact frequencies
    Pearson,
    /// Stratum-adjusted correlation coefficient (Yang et al., 2017)
    SCC,
    /// HiCRep reproducibility score (Yang et al., 2017)
    HiCRep,
}
```

---

## Repositories

### MethylationProfileRepository

```rust
trait MethylationProfileRepository {
    async fn store(&self, profile: MethylationProfile) -> Result<(), StoreError>;
    async fn find_by_id(&self, id: ProfileId) -> Option<MethylationProfile>;
    async fn find_by_sample_and_region(
        &self, sample: SampleId, region: &GenomicRegion
    ) -> Option<MethylationProfile>;
    async fn query_sites_in_region(
        &self, profile_id: ProfileId, region: &GenomicRegion
    ) -> Vec<MethylationSite>;
}
```

### GenomeTopologyRepository

```rust
trait GenomeTopologyRepository {
    async fn store(&self, topology: GenomeTopology) -> Result<(), StoreError>;
    async fn find_by_id(&self, id: TopologyId) -> Option<GenomeTopology>;
    async fn find_tads_overlapping(
        &self, topology_id: TopologyId, region: &GenomicRegion
    ) -> Vec<TAD>;
    async fn find_interactions_for_gene(
        &self, topology_id: TopologyId, gene: &str
    ) -> Vec<EnhancerPromoterInteraction>;
}
```

### ChromatinLandscapeRepository

```rust
trait ChromatinLandscapeRepository {
    async fn store(&self, landscape: ChromatinLandscape) -> Result<(), StoreError>;
    async fn find_by_id(&self, id: LandscapeId) -> Option<ChromatinLandscape>;
    async fn find_peaks_in_region(
        &self, landscape_id: LandscapeId, region: &GenomicRegion
    ) -> Vec<ChromatinRegion>;
}
```

### SOTA Repositories

```rust
trait SingleCellMultiOmeRepository {
    async fn store(&self, profile: SingleCellMultiOmeProfile) -> Result<(), StoreError>;
    async fn find_by_id(&self, id: ScMultiOmeProfileId) -> Option<SingleCellMultiOmeProfile>;
    async fn find_by_sample(&self, sample: SampleId) -> Vec<SingleCellMultiOmeProfile>;
    async fn query_cells_by_cluster(
        &self, profile_id: ScMultiOmeProfileId, cluster_id: ClusterId
    ) -> Vec<CellBarcode>;
}

trait EpigeneticAgePredictionRepository {
    async fn store(&self, prediction: EpigeneticAgePrediction) -> Result<(), StoreError>;
    async fn find_by_id(&self, id: AgePredictionId) -> Option<EpigeneticAgePrediction>;
    async fn find_by_sample(&self, sample: SampleId) -> Vec<EpigeneticAgePrediction>;
    async fn find_accelerated(
        &self, min_acceleration_years: f64
    ) -> Vec<EpigeneticAgePrediction>;
}

trait NucleomeDynamicsRepository {
    async fn store(&self, dynamics: NucleomeDynamics) -> Result<(), StoreError>;
    async fn find_by_id(&self, id: NucleomeDynamicsId) -> Option<NucleomeDynamics>;
    async fn find_by_sample_and_chromosome(
        &self, sample: SampleId, chromosome: &str
    ) -> Vec<NucleomeDynamics>;
}

trait CondensateModelRepository {
    async fn store(&self, model: CondensateModel) -> Result<(), StoreError>;
    async fn find_by_id(&self, id: CondensateId) -> Option<CondensateModel>;
    async fn find_by_locus(&self, region: &GenomicRegion) -> Vec<CondensateModel>;
    async fn find_by_tf(&self, tf_name: &str) -> Vec<CondensateModel>;
}
```

---

## Factories

### GenomeTopologyFactory

```rust
impl GenomeTopologyFactory {
    /// Build topology from Hi-C contact matrix
    fn from_hic(
        contact_matrix: &SparseMatrix<f64>,
        chromosome: &str,
        resolution: u32,
        tad_detector: &dyn TADDetector,
        enhancer_predictor: &dyn EnhancerPredictor,
    ) -> Result<GenomeTopology, EpigenomicsError> {
        // 1. Build ContactGraph from sparse matrix
        // 2. Run TAD detection via ruvector-mincut
        // 3. Assign compartments via eigenvector decomposition
        // 4. Predict enhancer-promoter interactions via ruvector-attention
        // 5. Assemble GenomeTopology aggregate
    }
}
```

---

## Integration with RuVector Crates

### ruvector-mincut: TAD Boundary Detection

The contact map is converted to a `DynamicGraph` where genomic bins are vertices and ICE-normalized contact frequencies are edge weights. TAD boundaries correspond to minimum cuts in this graph.

```rust
fn contact_graph_to_dynamic_graph(contact: &ContactGraph) -> DynamicGraph {
    let graph = DynamicGraph::new();
    for (bin_i, bin_j, weight) in contact.iter_contacts() {
        graph.insert_edge(bin_i as u64, bin_j as u64, weight).ok();
    }
    graph
}

fn detect_tad_boundaries(
    contact: &ContactGraph,
    config: &TadDetectorConfig,
) -> Vec<TAD> {
    let graph = contact_graph_to_dynamic_graph(contact);
    let mut mincut = MinCutBuilder::new()
        .exact()
        .with_graph(graph)
        .build()
        .unwrap();

    // Sliding window: find local min-cuts that define TAD boundaries
    // Uses JTreeHierarchy for hierarchical TAD nesting
}
```

### ruvector-mincut: Persistent Homology Consensus (SOTA)

Combines min-cut and persistent homology for high-confidence TAD boundary calls.

```rust
/// Consensus boundary detection: min-cut + persistent homology.
///
/// Min-cut captures the "weakest link" severing adjacent TADs.
/// Persistent homology captures the "topological lifetime" of TAD structure.
/// Boundaries called by both methods receive highest confidence.
fn consensus_tad_boundaries(
    contact: &ContactGraph,
    mincut_config: &TadDetectorConfig,
    persistence_config: &PersistenceConfig,
    max_distance_bins: u32,
) -> Vec<(GenomicPosition, f64)> {
    // 1. Detect boundaries via ruvector-mincut
    let mincut_bounds = detect_tad_boundaries(contact, mincut_config);

    // 2. Detect boundaries via persistent homology
    let persistence = compute_persistence(contact, persistence_config);
    let homology_bounds = boundaries_from_persistence(&persistence, 0.1);

    // 3. Merge: boundaries within max_distance_bins are considered concordant
    // 4. Assign confidence: concordant = 1.0, min-cut-only = 0.7, homology-only = 0.6
}
```

### ruvector-attention: Enhancer-Promoter Prediction

Enhancer and promoter loci become nodes in a graph. Edge features encode linear distance, Hi-C contact strength, and shared chromatin marks. `GraphAttention::compute_with_edges` learns which enhancers regulate which promoters.

### ruvector-attention: Enformer Transformer Layers (SOTA)

The Enformer-style chromatin accessibility predictor uses `ruvector-attention`'s `MultiHeadAttention` with relative positional encoding for the transformer tower that processes 200 kb sequence context.

```rust
/// Enformer transformer tower using ruvector-attention.
///
/// Each of the 11 transformer layers applies multi-head attention
/// with relative positional encoding over 1536 sequence bins.
fn enformer_transformer_layer(
    input: &Tensor,            // (batch, 1536, d_model)
    config: &EnformerConfig,
) -> Tensor {
    // Uses ruvector_attention::MultiHeadAttention with:
    // - relative positional encoding (learned, symmetric)
    // - 8 attention heads
    // - d_model = 1536, d_key = d_value = 192
    // - Pre-layer normalization
}
```

### ruvector-gnn: 3D Genome Functional Prediction

Each genomic bin is a node with a feature vector (methylation, accessibility, histone signals). Hi-C contacts form the edges. GNN message passing (`RuvectorLayer`) propagates regulatory signals through 3D proximity to predict gene activity, replication timing, and mutation impact.

### ruvector-gnn: Condensate and Loop Extrusion Predictions (SOTA)

`ruvector-gnn` is used for two SOTA applications:

1. **Condensate prediction**: TF binding sites are nodes with IDR features. GNN propagates phase separation propensity to predict which loci nucleate condensates.
2. **Loop extrusion parameterization**: CTCF sites and cohesin loading sites are nodes. GNN predicts binding strengths and loading rates from sequence + epigenomic features, feeding into loop extrusion simulations.

### ruvector-hyperbolic-hnsw: Chromatin State Search

Chromatin states (combinations of marks, accessibility, methylation) form a natural hierarchy: active/inactive at the top, with finer states (bivalent, poised enhancer, strong enhancer, etc.) below. `HyperbolicHnsw` indexes these states in Poincare ball space for efficient hierarchical nearest-neighbor search across the genome.

### ruvector-hyperbolic-hnsw: Single-Cell Embedding Search (SOTA)

Single-cell multi-omic embeddings from the coupled autoencoder are indexed in hyperbolic space, where the cell-type hierarchy (stem cell > progenitor > differentiated) maps naturally to the Poincare ball. This enables:

- Efficient nearest-neighbor search across millions of cells
- Hierarchical cell type discovery respecting differentiation trees
- Cross-dataset integration using hyperbolic alignment

---

## Anti-Corruption Layers

### Alignment ACL

```rust
impl AlignmentAcl {
    fn translate_bisulfite_read(&self, read: &BamRecord) -> Result<Vec<MethylationCall>, AclError> {
        // Extract M/C base modifications from BAM MM/ML tags (SAMv1.6)
        // Convert phred-scaled quality to confidence
        // Map to genomic coordinates using CIGAR alignment
    }
}
```

### HiC ACL

```rust
impl HiCAntiCorruptionLayer {
    fn translate_contact_matrix(
        &self, hic_file: &HicFile, chromosome: &str, resolution: u32
    ) -> Result<ContactGraph, AclError> {
        // Read .hic or .cool format
        // Apply ICE normalization
        // Build sparse symmetric contact matrix
        // Convert to ContactGraph value object
    }
}
```

### SingleCell ACL (SOTA)

```rust
impl SingleCellAntiCorruptionLayer {
    /// Translate 10x Multiome output to internal OmicsModality representation
    fn translate_multiome(
        &self,
        fragments_file: &str,       // ATAC fragments.tsv.gz
        matrix_dir: &str,           // RNA filtered_feature_bc_matrix/
        config: &SingleCellAclConfig,
    ) -> Result<Vec<OmicsModality>, AclError> {
        // 1. Parse 10x fragments file -> cell x peak matrix (scATAC)
        // 2. Parse 10x matrix -> cell x gene matrix (scRNA)
        // 3. Filter cells by QC metrics (min_genes, min_peaks, max_mito)
        // 4. Intersect cell barcodes across modalities
        // 5. Return standardized OmicsModality objects
    }
}
```

---

## Context Boundaries Summary

| Boundary | Upstream | Downstream | Integration Pattern |
|----------|----------|------------|---------------------|
| Alignment -> Epigenomics | Alignment Pipeline | Epigenomics Context | ACL (MethylationCall) |
| HiC -> Epigenomics | Hi-C Pipeline | Epigenomics Context | ACL (ContactGraph) |
| Epigenomics -> CRISPR | Epigenomics Context | CRISPR Context | Published Language (ChromatinState, MethylationProfile) |
| Epigenomics -> Variant | Epigenomics Context | Variant Calling Context | Domain Events (DMRIdentified) |
| SingleCell -> Epigenomics | 10x Multiome Pipeline | Epigenomics Context | ACL (OmicsModality) |
| Epigenomics -> Aging | Epigenomics Context | Clinical Context | Published Language (EpigeneticAgePrediction) |

---

## References

- DDD-001: Coherence Gate Domain Model
- DDD-002: Syndrome Processing Domain Model
- Evans, Eric. "Domain-Driven Design." Addison-Wesley, 2003.
- Vernon, Vaughn. "Implementing Domain-Driven Design." Addison-Wesley, 2013.
- Rao, S. et al. "A 3D Map of the Human Genome at Kilobase Resolution." Cell, 2014.
- Dixon, J. et al. "Topological Domains in Mammalian Genomes." Nature, 2012.
- Buenrostro, J. et al. "ATAC-seq: A Method for Assaying Chromatin Accessibility Genome-Wide." Current Protocols, 2015.
- Horvath, S. "DNA methylation age of human tissues and cell types." Genome Biology, 2013.
- Hannum, G. et al. "Genome-wide methylation profiles reveal quantitative views of human aging rates." Molecular Cell, 2013.
- Lu, A.T. et al. "DNA methylation GrimAge version 2." Aging, 2022.
- Belsky, D.W. et al. "DunedinPACE, a DNA methylation biomarker of the pace of aging." eLife, 2022.
- Avsec, Z. et al. "Effective gene expression prediction from sequence by integrating long-range interactions." Nature Methods, 2021.
- Ashuach, T. et al. "MultiVI: deep generative model for the integration of multimodal data." Nature Methods, 2023.
- Hao, Y. et al. "Dictionary learning for integrative, multimodal and scalable single-cell analysis." Nature Biotechnology, 2024.
- Gonzalez-Blas, C.B. et al. "SCENIC+: single-cell multiomic inference of enhancers and gene regulatory networks." Nature Methods, 2023.
- Sabari, B. et al. "Coactivator condensation at super-enhancers links phase separation and gene control." Science, 2018.
- Boija, A. et al. "Transcription Factors Activate Genes through the Phase-Separation Capacity of Their Activation Domains." Cell, 2018.
- Fudenberg, G. et al. "Formation of Chromosomal Domains by Loop Extrusion." Cell Reports, 2016.
- Sanborn, A.L. et al. "Chromatin extrusion explains key features of loop and domain formation." PNAS, 2015.
- Dekker, J. et al. "The 4D nucleome project." Nature, 2017.
- Edelsbrunner, H. & Harer, J. "Computational Topology." AMS, 2010.
- Carriere, M. et al. "Persistent homology based characterization of the contact map in Hi-C data." bioRxiv, 2020.
- Ruff, K.M. et al. "Sequence grammar underlying the unfolding and phase separation of globular proteins." Molecular Cell, 2022.
- Nuebler, J. et al. "Chromatin organization by an interplay of loop extrusion and compartmental segregation." PNAS, 2018.
- Yang, T. et al. "HiCRep: assessing the reproducibility of Hi-C data using a stratum-adjusted correlation coefficient." Genome Research, 2017.
