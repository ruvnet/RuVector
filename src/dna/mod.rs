//! DNA genomic analysis domain for RuVector.
//!
//! Provides k-mer embeddings, FASTA/FASTQ parsing, sequence search,
//! variant calling, and analysis pipeline orchestration.

pub mod sequence;
pub mod kmer;
pub mod fasta;
pub mod embedding;
pub mod search;
pub mod variant;
pub mod pipeline;
