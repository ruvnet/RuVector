# DDD-004: Pharmacogenomics Domain Model

**Status**: Proposed
**Date**: 2026-02-11
**Authors**: ruv.io, RuVector Team
**Related ADR**: ADR-001-ruqu-architecture
**Related DDD**: DDD-001-coherence-gate-domain, DDD-003-population-genomics-domain

---

## Overview

This document defines the Domain-Driven Design model for the Pharmacogenomics bounded context -- the clinical decision-support subsystem that predicts individual drug responses from genomic data, determines metabolizer phenotypes from star allele diplotypes, generates evidence-graded dosing recommendations, and assesses adverse drug reaction (ADR) risk. The design integrates RuVector's GNN for drug-gene interaction prediction, attention mechanisms for sequence-to-phenotype inference, and vector similarity search for finding patients with analogous pharmacogenomic profiles.

---

## Strategic Design

### Domain Vision Statement

> The Pharmacogenomics domain translates an individual's genetic variation at pharmacogenes into actionable clinical guidance: metabolizer status classification, dose adjustments, drug interaction warnings, and adverse reaction risk scores -- powered by RuVector's graph neural networks for interaction prediction and HNSW-based similar-patient retrieval for treatment outcome inference.

### Core Domain

**Pharmacogenetic Interpretation** is the core domain. This is the novel analytical capability:

- Not genotyping (that is upstream infrastructure shared with Population Genomics)
- Not drug databases (those are external reference data)
- **The novel capability**: Predicting individualized drug response by combining star-allele diplotyping, metabolizer phenotype inference, evidence-weighted dosing recommendation generation, and GNN-powered interaction discovery -- with RuVector enabling fast similar-patient lookup to ground predictions in real-world treatment outcomes.

### Supporting Domains

| Domain | Role | Boundary |
|--------|------|----------|
| **Variant Ingestion** | Provide called genotypes at pharmacogene loci | Shared with Population Genomics |
| **Drug Knowledge Base** | PharmGKB, CPIC, DPWG guideline data | External, read-only |
| **Clinical Data** | Patient medication lists, lab values, outcomes | External, EHR integration |
| **Population Genomics** | Ancestry-adjusted allele frequencies for star allele calling | Upstream bounded context |

### Generic Subdomains

- Audit logging (FDA 21 CFR Part 11 compliance)
- Version control for guideline updates
- Patient consent and data governance

---

## Ubiquitous Language

### Core Terms

| Term | Definition | Context |
|------|------------|---------|
| **Star Allele** | A named haplotype at a pharmacogene (e.g., CYP2D6*4, CYP2C19*17) | Value object |
| **Diplotype** | The pair of star alleles an individual carries at a gene (e.g., *1/*4) | Value object |
| **Metabolizer Status** | The predicted enzyme activity phenotype: PM, IM, NM, RM, UM | Value object |
| **Activity Score** | Numeric score assigned to a diplotype (sum of allele activities) | Value object |
| **Drug-Gene Interaction** | A relationship between a drug and a gene where genetic variation affects response | Entity |
| **Dosing Recommendation** | An evidence-graded adjustment to standard drug dosing based on genotype | Entity |
| **Adverse Drug Reaction** | A harmful, unintended response to a drug linked to a genetic variant | Entity |

### Classification Terms

| Term | Definition | Context |
|------|------------|---------|
| **PM** | Poor Metabolizer -- little to no enzyme activity, drug may accumulate | Phenotype |
| **IM** | Intermediate Metabolizer -- reduced activity | Phenotype |
| **NM** | Normal Metabolizer -- typical activity (formerly "EM") | Phenotype |
| **RM** | Rapid Metabolizer -- increased activity | Phenotype |
| **UM** | Ultrarapid Metabolizer -- greatly increased activity, prodrug over-activation | Phenotype |

### Evidence Terms

| Term | Definition | Context |
|------|------------|---------|
| **Evidence Level** | Strength of pharmacogenomic evidence (1A, 1B, 2A, 2B, 3, 4 per PharmGKB) | Value object |
| **CPIC Level** | Clinical Pharmacogenetics Implementation Consortium guideline strength (A, B, C) | Value object |
| **Actionability** | Whether a drug-gene pair has sufficient evidence for clinical action | Classification |

---

## Bounded Contexts

### Context Map

```
+-----------------------------------------------------------------------------+
|                     PHARMACOGENOMICS CONTEXT                                 |
|                           (Core Domain)                                      |
|  +-------------+  +-------------+  +-------------+  +-------------+         |
|  | Star Allele |  | Metabolizer |  |   Dosing    |  |     ADR     |         |
|  |   Caller    |  |  Classifier |  |  Recommender|  |   Assessor  |         |
|  +-------------+  +-------------+  +-------------+  +-------------+         |
+-----------+--------------+--------------+--------------+--------------------+
            |              |              |              |
            | Upstream     | Upstream     | External     | Downstream
            v              v              v              v
+-----------------+ +-----------------+ +-----------------+ +-----------------+
|    VARIANT      | |   POPULATION    | |     DRUG        | |   CLINICAL      |
|    INGESTION    | |   GENOMICS      | |  KNOWLEDGE BASE | |  DECISION       |
|  (Supporting)   | |  (Upstream BC)  | |   (External)    | |  SUPPORT        |
+-----------------+ +-----------------+ +-----------------+ +-----------------+
```

### Pharmacogenomics Context (Core)

**Responsibility**: Determine star allele diplotypes, classify metabolizer status, generate dosing recommendations, assess ADR risk, and predict drug-gene interactions using GNN.

**Key Aggregates**:
- PharmacogeneticProfile
- DrugGeneInteraction
- DosingRecommendation
- AdverseDrugReaction

**Anti-Corruption Layers**:
- Variant Adapter (translates raw genotypes at pharmacogene loci to star allele calls)
- Drug Knowledge Base Adapter (translates PharmGKB/CPIC/DPWG data to domain model)
- Clinical Data Adapter (translates EHR medication and lab data to domain format)
- Population Adapter (translates ancestry-adjusted frequencies from Population Genomics)

---

## Aggregates

### PharmacogeneticProfile (Root Aggregate)

The central aggregate containing all pharmacogenomic information for an individual.

```
+---------------------------------------------------------------------+
|                  PHARMACOGENETIC PROFILE                              |
|                       (Aggregate Root)                               |
+---------------------------------------------------------------------+
|  profile_id: ProfileId (UUID)                                         |
|  individual_id: IndividualId (link to Population Genomics)            |
|  star_alleles: Vec<StarAlleleDiplotype>                               |
|  metabolizer_statuses: Vec<MetabolizerStatus>                         |
|  drug_interactions: Vec<DrugInteractionAssessment>                    |
|  profile_vector: ProfileVector (384-dim, for similar-patient search)  |
|  evidence_summary: EvidenceSummary                                    |
|  created_at: Timestamp                                                |
|  guideline_version: GuidelineVersion                                  |
+---------------------------------------------------------------------+
|  +---------------------------------------------------------------+  |
|  | StarAlleleDiplotype (Value Object)                            |  |
|  |  gene: PharmaGene (CYP2D6, CYP2C19, CYP2C9, CYP3A5, ...)    |  |
|  |  allele_1: StarAllele (e.g., *1)                              |  |
|  |  allele_2: StarAllele (e.g., *4)                              |  |
|  |  activity_score: f64 (sum of allele activity values)          |  |
|  |  call_confidence: f64 (0.0 - 1.0)                            |  |
|  |  phasing_method: PhasingMethod { Statistical | Read | Family } |  |
|  +---------------------------------------------------------------+  |
|  +---------------------------------------------------------------+  |
|  | MetabolizerStatus (Value Object)                              |  |
|  |  gene: PharmaGene                                             |  |
|  |  phenotype: MetabolizerPhenotype { PM | IM | NM | RM | UM }   |  |
|  |  activity_score: f64                                          |  |
|  |  classification_source: ClassificationSource { CPIC | DPWG }  |  |
|  +---------------------------------------------------------------+  |
|  +---------------------------------------------------------------+  |
|  | DrugInteractionAssessment (Entity)                            |  |
|  |  assessment_id: AssessmentId                                  |  |
|  |  drug: DrugId                                                 |  |
|  |  gene: PharmaGene                                             |  |
|  |  effect: DrugEffect { Efficacy | Toxicity | Dosing }          |  |
|  |  severity: Severity { High | Moderate | Low | Informational }  |  |
|  |  recommendation: String (clinical action)                     |  |
|  |  evidence_level: EvidenceLevel                                |  |
|  +---------------------------------------------------------------+  |
+---------------------------------------------------------------------+
|  Invariants:                                                          |
|  - Each gene appears at most once in star_alleles                    |
|  - Each gene in metabolizer_statuses must have a corresponding       |
|    star_allele diplotype                                              |
|  - activity_score >= 0.0                                              |
|  - call_confidence in [0.0, 1.0]                                      |
|  - guideline_version must be a valid published version                |
+---------------------------------------------------------------------+
```

### DrugGeneInteraction (Aggregate Root)

A curated or predicted relationship between a drug and a gene.

```
+---------------------------------------------------------------------+
|                   DRUG-GENE INTERACTION                               |
|                       (Aggregate Root)                               |
+---------------------------------------------------------------------+
|  interaction_id: InteractionId                                        |
|  drug: DrugId                                                         |
|  drug_name: String                                                    |
|  gene: PharmaGene                                                     |
|  alleles_affected: Vec<StarAllele>                                    |
|  effect: DrugEffect { Efficacy | Toxicity | Dosing }                  |
|  evidence_level: EvidenceLevel { Level1A | 1B | 2A | 2B | 3 | 4 }    |
|  cpic_level: Option<CpicLevel { A | B | C }>                         |
|  phenotype_mapping: HashMap<MetabolizerPhenotype, ClinicalImpact>     |
|  source: InteractionSource { CPIC | DPWG | PharmGKB | Predicted }     |
|  prediction_confidence: Option<f64> (for GNN-predicted interactions)  |
+---------------------------------------------------------------------+
|  +---------------------------------------------------------------+  |
|  | ClinicalImpact (Value Object)                                 |  |
|  |  impact_description: String                                   |  |
|  |  dose_adjustment: Option<DoseAdjustment>                      |  |
|  |  alternative_drugs: Vec<DrugId>                               |  |
|  |  monitoring_required: bool                                    |  |
|  +---------------------------------------------------------------+  |
+---------------------------------------------------------------------+
|  Invariants:                                                          |
|  - evidence_level must be from valid PharmGKB hierarchy               |
|  - prediction_confidence required iff source == Predicted             |
|  - prediction_confidence (when present) in [0.0, 1.0]                |
|  - phenotype_mapping must cover at least PM and NM phenotypes        |
+---------------------------------------------------------------------+
```

### DosingRecommendation (Aggregate Root)

An evidence-graded dosing adjustment for a specific drug-genotype combination.

```
+---------------------------------------------------------------------+
|                  DOSING RECOMMENDATION                                |
|                       (Aggregate Root)                               |
+---------------------------------------------------------------------+
|  recommendation_id: RecommendationId                                  |
|  drug: DrugId                                                         |
|  genotype_context: GenotypeContext                                     |
|  standard_dose: Dose                                                  |
|  recommended_dose: Dose                                               |
|  adjustment_factor: f64 (multiplier, e.g., 0.5 for half-dose)        |
|  confidence: ConfidenceGrade { High | Moderate | Low }                |
|  source_guideline: GuidelineReference                                 |
|  clinical_rationale: String                                           |
|  contraindicated: bool                                                |
+---------------------------------------------------------------------+
|  +---------------------------------------------------------------+  |
|  | GenotypeContext (Value Object)                                 |  |
|  |  gene: PharmaGene                                             |  |
|  |  diplotype: StarAlleleDiplotype                               |  |
|  |  metabolizer_status: MetabolizerPhenotype                     |  |
|  |  activity_score: f64                                          |  |
|  +---------------------------------------------------------------+  |
|  +---------------------------------------------------------------+  |
|  | Dose (Value Object)                                           |  |
|  |  amount: f64                                                  |  |
|  |  unit: DoseUnit { Mg | MgPerKg | MgPerM2 }                   |  |
|  |  frequency: Frequency { Daily | BID | TID | QID | Weekly }    |  |
|  |  route: Route { Oral | IV | IM | Topical | Inhaled }          |  |
|  +---------------------------------------------------------------+  |
|  +---------------------------------------------------------------+  |
|  | GuidelineReference (Value Object)                             |  |
|  |  source: GuidelineSource { CPIC | DPWG | FDA | InHouse }      |  |
|  |  guideline_id: String                                         |  |
|  |  version: String                                              |  |
|  |  publication_date: Date                                       |  |
|  |  url: Option<String>                                          |  |
|  +---------------------------------------------------------------+  |
+---------------------------------------------------------------------+
|  Invariants:                                                          |
|  - adjustment_factor > 0.0 (unless contraindicated, then 0.0)        |
|  - If contraindicated, recommended_dose must be zero                  |
|  - source_guideline must reference a published, version-tracked       |
|    guideline                                                          |
|  - confidence must reflect the weakest evidence in the chain          |
+---------------------------------------------------------------------+
```

### AdverseDrugReaction (Aggregate Root)

A genetically-mediated adverse drug reaction risk assessment.

```
+---------------------------------------------------------------------+
|                   ADVERSE DRUG REACTION                               |
|                       (Aggregate Root)                               |
+---------------------------------------------------------------------+
|  adr_id: AdrId                                                        |
|  drug: DrugId                                                         |
|  variant: VariantId (typically HLA allele or specific SNP)            |
|  gene: PharmaGene                                                     |
|  reaction_type: ReactionType { SJS | DRESS | Hepatotoxicity |        |
|                                 QTProlongation | Myopathy |           |
|                                 Anaphylaxis | Other(String) }         |
|  severity: AdrSeverity { LifeThreatening | Severe | Moderate | Mild } |
|  frequency: AdrFrequency { value: f64, population: PopulationId }     |
|  risk_allele: Allele                                                  |
|  risk_score: f64 (0.0 - 1.0, individual's risk given genotype)       |
|  evidence_level: EvidenceLevel                                        |
|  clinical_action: ClinicalAction { Avoid | Monitor | TestFirst }      |
+---------------------------------------------------------------------+
|  +---------------------------------------------------------------+  |
|  | AdrFrequency (Value Object)                                   |  |
|  |  value: f64 (proportion of risk-allele carriers affected)     |  |
|  |  population: PopulationId (ancestry-specific frequency)       |  |
|  |  total_cases: usize                                           |  |
|  |  total_exposed: usize                                         |  |
|  +---------------------------------------------------------------+  |
+---------------------------------------------------------------------+
|  Invariants:                                                          |
|  - risk_score in [0.0, 1.0]                                          |
|  - frequency.value in [0.0, 1.0]                                     |
|  - LifeThreatening ADRs must have clinical_action == Avoid            |
|  - frequency.total_exposed > 0                                        |
+---------------------------------------------------------------------+
```

---

## Value Objects

### StarAllele

A named haplotype at a pharmacogene.

```rust
struct StarAllele {
    gene: PharmaGene,
    name: String,               // e.g., "*4", "*17"
    defining_variants: Vec<VariantId>,
    function: AlleleFunction,   // NoFunction, Decreased, Normal, Increased
    activity_value: f64,        // 0.0, 0.5, 1.0, 1.5, 2.0 per CPIC
}

impl StarAllele {
    fn is_no_function(&self) -> bool {
        matches!(self.function, AlleleFunction::NoFunction)
    }
    fn is_increased_function(&self) -> bool {
        matches!(self.function, AlleleFunction::Increased)
    }
}
```

### ProfileVector

384-dimensional embedding of a pharmacogenomic profile for similar-patient search.

```rust
struct ProfileVector {
    embedding: [f32; 384],
    genes_encoded: Vec<PharmaGene>,
    encoding_version: u32,
}

impl ProfileVector {
    /// Construct from star allele diplotypes
    /// Encodes metabolizer status, activity scores, and interaction pattern
    /// into a dense vector suitable for ruvector-core HNSW indexing
    fn from_profile(profile: &PharmacogeneticProfile) -> Self;

    /// Find similar pharmacogenomic profiles via HNSW nearest-neighbor
    fn similarity(&self, other: &ProfileVector) -> f32;
}
```

### EvidenceLevel

PharmGKB evidence classification.

```rust
enum EvidenceLevel {
    Level1A,  // Variant-drug in CPIC/DPWG guideline, clinical annotation
    Level1B,  // Variant-drug in PharmGKB clinical annotation
    Level2A,  // Variant-drug in VIP with known functional significance
    Level2B,  // Variant in VIP with evidence of PGx association
    Level3,   // Annotations with limited evidence
    Level4,   // Case reports, non-significant studies
}

impl EvidenceLevel {
    fn is_actionable(&self) -> bool {
        matches!(self, Self::Level1A | Self::Level1B | Self::Level2A)
    }
    fn confidence_weight(&self) -> f64 {
        match self {
            Self::Level1A => 1.0,
            Self::Level1B => 0.9,
            Self::Level2A => 0.75,
            Self::Level2B => 0.5,
            Self::Level3 => 0.25,
            Self::Level4 => 0.1,
        }
    }
}
```

### DoseAdjustment

Quantified dose modification.

```rust
struct DoseAdjustment {
    direction: AdjustmentDirection { Increase | Decrease | Contraindicate | NoChange },
    factor: f64,               // e.g., 0.5 for 50% reduction
    absolute_change: Option<Dose>,  // alternative: specify exact dose
    titration_required: bool,
    monitoring_parameters: Vec<MonitoringParameter>,
}

impl DoseAdjustment {
    fn apply_to(&self, standard: &Dose) -> Dose {
        match self.direction {
            AdjustmentDirection::Contraindicate => Dose::zero(),
            _ => standard.scale(self.factor),
        }
    }
}
```

---

## Domain Events

### Profile Events

| Event | Trigger | Payload |
|-------|---------|---------|
| `ProfileGenotyped` | Star alleles called for all pharmacogenes | profile_id, individual_id, genes_typed |
| `MetabolizerStatusDetermined` | Phenotype classified from diplotype | profile_id, gene, phenotype, activity_score |
| `ProfileVectorComputed` | 384-dim embedding generated | profile_id, vector_hash |
| `SimilarPatientsFound` | HNSW search returned matches | profile_id, match_count, top_similarity |

### Recommendation Events

| Event | Trigger | Payload |
|-------|---------|---------|
| `DosingRecommendationGenerated` | Dose adjustment computed | recommendation_id, drug, gene, adjustment |
| `ContraindicationDetected` | Drug contraindicated for genotype | drug, gene, diplotype, reason |
| `GuidelineUpdated` | New CPIC/DPWG version loaded | source, version, affected_drugs |
| `InteractionPredicted` | GNN predicts new drug-gene pair | interaction_id, drug, gene, confidence |

### ADR Events

| Event | Trigger | Payload |
|-------|---------|---------|
| `AdverseReactionRiskAssessed` | ADR risk computed for patient | adr_id, drug, risk_score, severity |
| `HighRiskAlertGenerated` | Risk score exceeds critical threshold | individual_id, drug, reaction_type, score |
| `PopulationRiskUpdated` | Frequency recalculated with new data | adr_id, population, new_frequency |

---

## Domain Services

### StarAlleleCallerService

Determine star allele diplotypes from raw genotype data.

```rust
trait StarAlleleCallerService {
    /// Call star alleles at a pharmacogene from genotype data
    /// Uses combinatorial matching against known haplotype definitions.
    /// Ancestry from Population Genomics context adjusts prior probabilities
    /// for ambiguous calls.
    async fn call_diplotype(
        &self,
        individual: &IndividualId,
        gene: PharmaGene,
        genotypes: &PharmaGeneGenotypes,
        ancestry: &[AncestryComponent],
    ) -> Result<StarAlleleDiplotype, CallerError>;

    /// Batch call for all pharmacogenes
    async fn call_all_genes(
        &self,
        individual: &IndividualId,
    ) -> Result<Vec<StarAlleleDiplotype>, CallerError>;
}
```

### MetabolizerClassificationService

Classify metabolizer phenotype from diplotype and activity score.

```rust
trait MetabolizerClassificationService {
    /// Classify metabolizer phenotype using CPIC activity score system
    fn classify(
        &self,
        diplotype: &StarAlleleDiplotype,
        classification_system: ClassificationSource,
    ) -> MetabolizerStatus;

    /// Batch classify for all genes in a profile
    fn classify_profile(
        &self,
        profile: &PharmacogeneticProfile,
    ) -> Vec<MetabolizerStatus>;
}
```

### DosingRecommendationService

Generate evidence-graded dosing recommendations.

```rust
trait DosingRecommendationService {
    /// Generate dosing recommendation for a drug-genotype pair
    /// Retrieves applicable CPIC/DPWG guidelines, applies the recommendation
    /// for the patient's metabolizer status, and grades confidence.
    async fn recommend(
        &self,
        drug: DrugId,
        profile: &PharmacogeneticProfile,
    ) -> Result<DosingRecommendation, RecommendationError>;

    /// Check all current medications against profile
    async fn screen_medication_list(
        &self,
        medications: &[DrugId],
        profile: &PharmacogeneticProfile,
    ) -> Result<Vec<DosingRecommendation>, RecommendationError>;

    /// Find similar patients and their outcomes for prediction grounding
    /// Uses ruvector-core HNSW search on ProfileVector
    async fn find_similar_outcomes(
        &self,
        profile: &PharmacogeneticProfile,
        drug: DrugId,
        k: usize,
    ) -> Result<Vec<SimilarPatientOutcome>, RecommendationError>;
}
```

### DrugGeneInteractionPredictionService

Predict novel drug-gene interactions using GNN.

```rust
trait DrugGeneInteractionPredictionService {
    /// Predict interaction using ruvector-gnn
    /// Models the drug-gene-variant network as a heterogeneous graph.
    /// Node features: drug structure embeddings, gene expression profiles,
    /// variant functional annotations.
    /// Edge prediction: GNN link prediction for (drug, gene) pairs.
    ///
    /// Uses ruvector-gnn::RuvectorLayer for message passing and
    /// ruvector-gnn::differentiable_search for candidate retrieval.
    async fn predict_interaction(
        &self,
        drug: DrugId,
        gene: PharmaGene,
    ) -> Result<PredictedInteraction, PredictionError>;

    /// Batch predict all potential interactions for a new drug
    async fn predict_all_for_drug(
        &self,
        drug: DrugId,
    ) -> Result<Vec<PredictedInteraction>, PredictionError>;

    /// Update GNN model with newly confirmed interactions
    /// Uses ruvector-gnn::ElasticWeightConsolidation to prevent
    /// catastrophic forgetting of previously learned patterns.
    async fn update_model(
        &self,
        confirmed: &[DrugGeneInteraction],
    ) -> Result<(), PredictionError>;
}
```

### AdverseDrugReactionService

Assess ADR risk for a patient-drug combination.

```rust
trait AdverseDrugReactionService {
    /// Assess ADR risk for a specific drug
    async fn assess_risk(
        &self,
        individual: &IndividualId,
        drug: DrugId,
        profile: &PharmacogeneticProfile,
    ) -> Result<Vec<AdverseDrugReaction>, AdrError>;

    /// Screen all medications for ADR risks
    async fn screen_all(
        &self,
        individual: &IndividualId,
        medications: &[DrugId],
        profile: &PharmacogeneticProfile,
    ) -> Result<AdrScreeningReport, AdrError>;

    /// Use ruvector-attention for sequence -> ADR risk prediction
    /// Applies multi-head attention over the patient's variant sequence
    /// at HLA and pharmacogene loci to predict ADR susceptibility.
    async fn predict_from_sequence(
        &self,
        variant_sequence: &[VariantId],
        drug: DrugId,
    ) -> Result<AdrRiskPrediction, AdrError>;
}
```

### SimilarPatientSearchService

Find patients with analogous pharmacogenomic profiles.

```rust
trait SimilarPatientSearchService {
    /// Find k most similar pharmacogenomic profiles
    /// Uses ruvector-core HNSW nearest-neighbor search on ProfileVector.
    /// Complexity: O(log n) per query where n = indexed patients.
    async fn find_similar(
        &self,
        profile: &PharmacogeneticProfile,
        k: usize,
    ) -> Result<Vec<SimilarProfile>, SearchError>;

    /// Find similar patients who took a specific drug, with outcomes
    /// Combines HNSW search with metadata filtering for drug and outcome.
    async fn find_similar_with_outcome(
        &self,
        profile: &PharmacogeneticProfile,
        drug: DrugId,
        k: usize,
    ) -> Result<Vec<SimilarPatientOutcome>, SearchError>;
}
```

---

## Repositories

### PharmacogeneticProfileRepository

```rust
trait PharmacogeneticProfileRepository {
    /// Store profile with vector indexed in HNSW
    async fn store(&self, profile: PharmacogeneticProfile) -> Result<(), StoreError>;

    /// Find by profile ID
    async fn find_by_id(&self, id: &ProfileId) -> Option<PharmacogeneticProfile>;

    /// Find by individual ID
    async fn find_by_individual(
        &self,
        individual: &IndividualId,
    ) -> Option<PharmacogeneticProfile>;

    /// Search by vector similarity (delegates to ruvector-core HNSW)
    async fn search_similar(
        &self,
        vector: &ProfileVector,
        k: usize,
    ) -> Result<Vec<(ProfileId, f32)>, SearchError>;

    /// Find all profiles with a specific metabolizer status
    async fn find_by_metabolizer_status(
        &self,
        gene: PharmaGene,
        phenotype: MetabolizerPhenotype,
    ) -> Vec<ProfileId>;
}
```

### DrugGeneInteractionRepository

```rust
trait DrugGeneInteractionRepository {
    async fn store(&self, interaction: DrugGeneInteraction) -> Result<(), StoreError>;
    async fn find_by_id(&self, id: &InteractionId) -> Option<DrugGeneInteraction>;
    async fn find_by_drug(&self, drug: &DrugId) -> Vec<DrugGeneInteraction>;
    async fn find_by_gene(&self, gene: &PharmaGene) -> Vec<DrugGeneInteraction>;
    async fn find_by_drug_and_gene(
        &self,
        drug: &DrugId,
        gene: &PharmaGene,
    ) -> Option<DrugGeneInteraction>;
    async fn find_actionable(&self) -> Vec<DrugGeneInteraction>;
    async fn find_predicted(&self, min_confidence: f64) -> Vec<DrugGeneInteraction>;
}
```

### DosingRecommendationRepository

```rust
trait DosingRecommendationRepository {
    async fn store(&self, rec: DosingRecommendation) -> Result<(), StoreError>;
    async fn find_by_id(&self, id: &RecommendationId) -> Option<DosingRecommendation>;
    async fn find_by_drug(&self, drug: &DrugId) -> Vec<DosingRecommendation>;
    async fn find_contraindications(&self) -> Vec<DosingRecommendation>;
    async fn find_by_guideline(
        &self,
        source: GuidelineSource,
        version: &str,
    ) -> Vec<DosingRecommendation>;
}
```

### AdverseDrugReactionRepository

```rust
trait AdverseDrugReactionRepository {
    async fn store(&self, adr: AdverseDrugReaction) -> Result<(), StoreError>;
    async fn find_by_id(&self, id: &AdrId) -> Option<AdverseDrugReaction>;
    async fn find_by_drug(&self, drug: &DrugId) -> Vec<AdverseDrugReaction>;
    async fn find_by_variant(&self, variant: &VariantId) -> Vec<AdverseDrugReaction>;
    async fn find_life_threatening(&self) -> Vec<AdverseDrugReaction>;
    async fn find_by_severity(
        &self,
        min_severity: AdrSeverity,
    ) -> Vec<AdverseDrugReaction>;
}
```

---

## RuVector Integration Architecture

### Crate Mapping

| Domain Operation | RuVector Crate | Mechanism |
|-----------------|----------------|-----------|
| Profile vector storage + similar-patient search | `ruvector-core` | 384-dim HNSW index, O(log n) ANN |
| Drug-gene interaction prediction (GNN) | `ruvector-gnn` | Heterogeneous graph link prediction |
| Catastrophic forgetting prevention | `ruvector-gnn::ewc` | Elastic Weight Consolidation |
| Sequence -> metabolizer status prediction | `ruvector-attention` | Multi-head attention over variant sequence |
| HLA-mediated ADR risk attention | `ruvector-attention` | Graph attention for HLA-variant interactions |
| Similar patient outcome retrieval | `ruvector-core` | HNSW + metadata filtered search |
| Drug interaction graph queries | `ruvector-graph` | Cypher-style traversal of drug-gene network |

### GNN Architecture for Drug-Gene Prediction

The drug-gene interaction graph is modeled as a heterogeneous graph with three node types and two edge types:

```
Node Types:
  - Drug nodes: feature = molecular fingerprint embedding (384-dim)
  - Gene nodes: feature = gene expression + pathway embedding (384-dim)
  - Variant nodes: feature = functional annotation embedding (384-dim)

Edge Types:
  - (Drug) --[INTERACTS_WITH]--> (Gene): known or predicted interaction
  - (Variant) --[AFFECTS]--> (Gene): variant's functional effect on gene

GNN Pipeline (ruvector-gnn):
  1. RuvectorLayer message passing (2-3 hops)
  2. Edge score prediction for (Drug, Gene) pairs
  3. EWC regularization during incremental learning
  4. ReplayBuffer for experience replay on confirmed interactions
```

**Performance**: Link prediction inference in O(|V| + |E|) per query via ruvector-gnn forward pass. Model update with EWC in O(|params|) per new interaction batch.

### Attention for Sequence-to-Phenotype

```
Input: Ordered variant sequence at pharmacogene locus [v1, v2, ..., vk]
       Each variant embedded as 384-dim vector via ruvector-core

Attention Pipeline (ruvector-attention):
  1. ScaledDotProductAttention over variant embeddings
  2. Multi-head attention (8 heads) captures different functional aspects
  3. Classification head: attention output -> MetabolizerPhenotype

Use case: Complex genes like CYP2D6 with copy number variation,
          hybrid alleles, and ambiguous star allele calls where
          rule-based calling fails.
```

### Performance Targets

| Operation | Target | Complexity | Mechanism |
|-----------|--------|------------|-----------|
| Star allele calling (per gene) | <10ms | O(h * v) where h=haplotypes, v=variants | Combinatorial matching |
| Metabolizer classification | <1ms | O(1) lookup from diplotype | Activity score table |
| Dosing recommendation | <50ms | O(g) where g = genes in profile | Guideline lookup + scoring |
| Similar patient search | <100ms | O(log n) HNSW | ruvector-core |
| GNN interaction prediction | <200ms | O(|V| + |E|) forward pass | ruvector-gnn |
| ADR risk assessment (per drug) | <50ms | O(v) where v = risk variants | Variant lookup + scoring |
| Full medication screening | <500ms | O(m * g) where m=medications | Parallel drug-gene lookup |
| Attention-based phenotype prediction | <100ms | O(k^2 * d) where k=variants, d=384 | ruvector-attention |

---

## Invariants and Business Rules

### Profile Invariants

1. **Gene Uniqueness**: Each pharmacogene appears at most once in star_alleles
2. **Status Consistency**: Every MetabolizerStatus must have a corresponding StarAlleleDiplotype
3. **Activity Score Non-Negative**: All activity scores >= 0.0
4. **Confidence Range**: Call confidence in [0.0, 1.0]
5. **Guideline Currency**: Profile must be annotated with the guideline version used

### Recommendation Invariants

1. **Contraindication Override**: If contraindicated == true, recommended_dose must be zero
2. **Evidence Chain**: Confidence grade must reflect the weakest evidence link
3. **Guideline Traceability**: Every recommendation must link to a specific published guideline with version
4. **Adjustment Positivity**: adjustment_factor > 0.0 unless contraindicated

### ADR Invariants

1. **Life-Threatening Action**: LifeThreatening severity must result in clinical_action == Avoid
2. **Frequency Validity**: AdrFrequency.value in [0.0, 1.0], total_exposed > 0
3. **Risk Score Range**: risk_score in [0.0, 1.0]
4. **Population Specificity**: ADR frequencies must be tagged with ancestry population

### GNN Prediction Invariants

1. **Confidence Threshold**: Only interactions with prediction_confidence >= 0.7 are surfaced
2. **No Self-Interaction**: Drug cannot have an interaction with itself
3. **Forgetting Prevention**: EWC penalty weight must be applied during model updates
4. **Replay Coverage**: ReplayBuffer must maintain minimum 1000 entries before allowing new training

---

## Anti-Corruption Layers

### Drug Knowledge Base ACL

Translates PharmGKB/CPIC/DPWG data to domain model.

```rust
impl DrugKnowledgeBaseAcl {
    /// Translate CPIC guideline to domain DrugGeneInteraction
    fn translate_cpic(&self, guideline: CpicGuideline) -> Vec<DrugGeneInteraction> {
        guideline.recommendations.iter().map(|rec| {
            DrugGeneInteraction {
                interaction_id: InteractionId::new(),
                drug: self.map_drug_id(&rec.drug),
                gene: self.map_gene(&rec.gene),
                alleles_affected: self.map_alleles(&rec.alleles),
                effect: self.classify_effect(&rec),
                evidence_level: EvidenceLevel::Level1A, // CPIC = 1A
                cpic_level: Some(self.map_cpic_strength(&rec.strength)),
                phenotype_mapping: self.build_phenotype_map(&rec),
                source: InteractionSource::CPIC,
                prediction_confidence: None,
            }
        }).collect()
    }

    /// Translate PharmGKB clinical annotation
    fn translate_pharmgkb(&self, annotation: PharmgkbAnnotation) -> DrugGeneInteraction;
}
```

### Clinical Data ACL

Translates EHR medication and lab data to domain format.

```rust
impl ClinicalDataAcl {
    /// Translate medication list from EHR format to domain DrugIds
    fn translate_medications(&self, ehr_meds: &[EhrMedication]) -> Vec<DrugId> {
        ehr_meds.iter()
            .filter_map(|med| self.rxnorm_to_drug_id(&med.rxnorm_code))
            .collect()
    }

    /// Translate lab values relevant to pharmacogenomics
    fn translate_labs(&self, labs: &[EhrLab]) -> Vec<MonitoringParameter> {
        labs.iter()
            .filter(|lab| self.is_pgx_relevant(&lab.loinc_code))
            .map(|lab| self.to_monitoring_parameter(lab))
            .collect()
    }
}
```

### Population Genomics ACL

Translates ancestry data for ancestry-adjusted star allele calling.

```rust
impl PopulationGenomicsAcl {
    /// Get ancestry-adjusted allele frequencies for star allele priors
    fn get_adjusted_frequencies(
        &self,
        gene: PharmaGene,
        ancestry: &[AncestryComponent],
    ) -> HashMap<StarAllele, f64> {
        // Weight population-specific star allele frequencies
        // by the individual's ancestry proportions
        ancestry.iter()
            .flat_map(|comp| {
                let pop_freqs = self.reference_frequencies(gene, &comp.population_label);
                pop_freqs.iter().map(move |(allele, freq)| {
                    (allele.clone(), freq * comp.proportion)
                })
            })
            .fold(HashMap::new(), |mut acc, (allele, weighted_freq)| {
                *acc.entry(allele).or_insert(0.0) += weighted_freq;
                acc
            })
    }
}
```

---

## Event Flow: End-to-End Pharmacogenomic Report

```
1. [Variant Ingestion] GenotypeDataReceived
       |
       v
2. [Population Genomics] IndividualGenotyped -> AncestryInferred
       |
       v
3. [Pharmacogenomics] StarAlleleCallerService.call_all_genes()
       |  -> ProfileGenotyped event
       v
4. [Pharmacogenomics] MetabolizerClassificationService.classify_profile()
       |  -> MetabolizerStatusDetermined event (per gene)
       v
5. [Pharmacogenomics] ProfileVector computed, indexed in HNSW
       |  -> ProfileVectorComputed event
       v
6. [Pharmacogenomics] DosingRecommendationService.screen_medication_list()
       |  -> DosingRecommendationGenerated event (per drug)
       |  -> ContraindicationDetected event (if applicable)
       v
7. [Pharmacogenomics] AdverseDrugReactionService.screen_all()
       |  -> AdverseReactionRiskAssessed event (per drug-variant pair)
       |  -> HighRiskAlertGenerated event (if risk > threshold)
       v
8. [Pharmacogenomics] SimilarPatientSearchService.find_similar_with_outcome()
       |  -> SimilarPatientsFound event
       v
9. [Clinical Decision Support] Report assembled and delivered
```

---

## Context Boundaries Summary

| Boundary | Upstream | Downstream | Integration Pattern |
|----------|----------|------------|---------------------|
| Variant -> PGx | Variant Ingestion | Pharmacogenomics | Published Language (PharmaGeneGenotypes) |
| PopGen -> PGx | Population Genomics | Pharmacogenomics | ACL (AncestryComponent) |
| DrugKB -> PGx | Drug Knowledge Base | Pharmacogenomics | ACL (DrugGeneInteraction) |
| PGx -> CDS | Pharmacogenomics | Clinical Decision Support | Domain Events (DosingRecommendationGenerated) |
| EHR -> PGx | Clinical Data | Pharmacogenomics | ACL (medication list, labs) |

---

## References

- DDD-001: Coherence Gate Domain Model
- DDD-003: Population Genomics Domain Model
- arXiv:2512.13105 -- El-Hayek, Henzinger, Li. "Deterministic and Exact Fully-dynamic Minimum Cut" (SODA 2025)
- CPIC (Clinical Pharmacogenetics Implementation Consortium) Guidelines
- PharmGKB Clinical Annotations and VIP Summaries
- DPWG (Dutch Pharmacogenetics Working Group) Guidelines
- Evans, Eric. "Domain-Driven Design." Addison-Wesley, 2003.
- Vernon, Vaughn. "Implementing Domain-Driven Design." Addison-Wesley, 2013.
