//! # rvDNA — AI-Native Genomic Analysis
//!
//! Fast, accurate genomic analysis in pure Rust with WASM support.
//! Includes the `.rvdna` binary file format for storing pre-computed
//! AI features alongside raw DNA sequences.
//!
//! - **K-mer HNSW Indexing**: Sequence similarity search via vector embeddings
//! - **Smith-Waterman Alignment**: Local alignment with CIGAR and mapping quality
//! - **Bayesian Variant Calling**: SNP/indel detection with Phred quality scores
//! - **Protein Translation**: DNA-to-protein with GNN contact graph prediction
//! - **Epigenomics**: Methylation profiling and Horvath biological age clock
//! - **Pharmacogenomics**: CYP enzyme star allele calling and drug recommendations
//! - **Pipeline Orchestration**: DAG-based multi-stage execution
//! - **RVDNA Format**: AI-native binary file format with pre-computed tensors

#![warn(missing_docs)]
#![allow(clippy::all)]

pub mod error;
pub mod types;
pub mod kmer;
pub mod alignment;
pub mod variant;
pub mod protein;
pub mod epigenomics;
pub mod pharma;
pub mod pipeline;
pub mod rvdna;
pub mod real_data;

pub use error::{DnaError, Result};
pub use types::{
    AlignmentResult, AnalysisConfig, CigarOp, ContactGraph, DnaSequence, GenomicPosition,
    KmerIndex, Nucleotide, ProteinResidue, ProteinSequence, QualityScore, Variant,
};
pub use variant::{
    FilterStatus, Genotype, PileupColumn, VariantCall, VariantCaller, VariantCallerConfig,
};
pub use protein::{AminoAcid, translate_dna, molecular_weight, isoelectric_point};
pub use epigenomics::{CpGSite, HorvathClock, MethylationProfile, CancerSignalDetector, CancerSignalResult};
pub use alignment::{AlignmentConfig, SmithWaterman};
pub use pharma::{
    call_star_allele, get_recommendations, predict_phenotype, DrugRecommendation,
    MetabolizerPhenotype, PharmaVariant, StarAllele,
    Cyp2c19Allele, call_cyp2c19_allele, predict_cyp2c19_phenotype,
};
pub use rvdna::{
    Codec, RvdnaHeader, RvdnaReader, RvdnaWriter, RvdnaStats,
    SparseAttention, VariantTensor, KmerVectorBlock,
    encode_2bit, decode_2bit, fasta_to_rvdna,
};

pub use ruvector_core::{
    types::{DbOptions, DistanceMetric, HnswConfig, SearchQuery, SearchResult, VectorEntry},
    VectorDB,
};

/// Prelude module for common imports
pub mod prelude {
    pub use crate::alignment::*;
    pub use crate::epigenomics::*;
    pub use crate::error::{DnaError, Result};
    pub use crate::kmer::*;
    pub use crate::pharma::*;
    pub use crate::protein::*;
    pub use crate::types::*;
    pub use crate::variant::*;
}
