//! DNA Analysis Pipeline orchestrator.
//!
//! Provides a staged pipeline for processing genomic data:
//! 1. Parse: Read FASTA/FASTQ input
//! 2. QC: Quality control filtering
//! 3. Embed: Convert sequences to k-mer embeddings
//! 4. Search: Find similar references in the index
//! 5. Call: Detect variants against matched references
//!
//! Designed for streaming/incremental processing.

use super::embedding::{DnaEmbedder, EmbeddingConfig};
use super::fasta::{parse_fasta, parse_fastq};
use super::search::{GenomicReference, GenomicSearchIndex, GenomicSearchResult};
use super::sequence::{PackedSequence, QualityScores, SequenceRead, SequenceError};
use super::variant::{Variant, VariantCallerConfig};

use std::fmt;

/// Pipeline stage identifier
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PipelineStage {
    Parse,
    QualityControl,
    Embed,
    Search,
    VariantCall,
    Complete,
}

impl fmt::Display for PipelineStage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PipelineStage::Parse => write!(f, "Parse"),
            PipelineStage::QualityControl => write!(f, "QC"),
            PipelineStage::Embed => write!(f, "Embed"),
            PipelineStage::Search => write!(f, "Search"),
            PipelineStage::VariantCall => write!(f, "VariantCall"),
            PipelineStage::Complete => write!(f, "Complete"),
        }
    }
}

/// Pipeline configuration
#[derive(Debug, Clone)]
pub struct PipelineConfig {
    /// Minimum mean quality score to pass QC
    pub min_mean_quality: f64,
    /// Minimum sequence length to pass QC
    pub min_length: usize,
    /// K-mer size for embedding
    pub kmer_k: usize,
    /// Embedding dimension
    pub embedding_dim: usize,
    /// Number of top search results
    pub search_top_k: usize,
    /// Variant caller config
    pub variant_config: VariantCallerConfig,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        PipelineConfig {
            min_mean_quality: 20.0,
            min_length: 50,
            kmer_k: 6,
            embedding_dim: 384,
            search_top_k: 5,
            variant_config: VariantCallerConfig::default(),
        }
    }
}

/// Statistics collected during pipeline execution
#[derive(Debug, Clone, Default)]
pub struct PipelineStats {
    pub total_reads: usize,
    pub passed_qc: usize,
    pub failed_qc: usize,
    pub embedded: usize,
    pub search_hits: usize,
    pub variants_called: usize,
    pub variants_passed_filter: usize,
}

/// Result of running the pipeline on a single read
#[derive(Debug)]
pub struct ReadResult {
    pub read_id: String,
    pub search_results: Vec<GenomicSearchResult>,
    pub variants: Vec<Variant>,
}

/// The DNA analysis pipeline
pub struct DnaPipeline {
    config: PipelineConfig,
    embedder: DnaEmbedder,
    index: GenomicSearchIndex,
    stats: PipelineStats,
    current_stage: PipelineStage,
}

impl DnaPipeline {
    /// Create a new pipeline with default config
    pub fn new(config: PipelineConfig) -> Self {
        let emb_config = EmbeddingConfig {
            k: config.kmer_k,
            output_dim: config.embedding_dim,
            canonical: true,
        };
        let embedder = DnaEmbedder::new(emb_config);
        let index = GenomicSearchIndex::new(config.embedding_dim);

        DnaPipeline {
            config,
            embedder,
            index,
            stats: PipelineStats::default(),
            current_stage: PipelineStage::Parse,
        }
    }

    /// Add a reference genome to the search index
    pub fn add_reference(&mut self, id: &str, organism: &str, taxonomy: &str, sequence: &PackedSequence) {
        let embedding = self.embedder.embed(sequence);
        let _ = self.index.add_reference(embedding, GenomicReference {
            id: id.to_string(),
            organism: organism.to_string(),
            taxonomy: taxonomy.to_string(),
            sequence_length: sequence.len(),
        });
    }

    /// Quality control: filter reads by quality and length
    pub fn quality_control(&self, read: &SequenceRead) -> bool {
        if read.sequence.len() < self.config.min_length {
            return false;
        }
        read.quality.mean_quality() >= self.config.min_mean_quality
    }

    /// Process a single read through the pipeline
    pub fn process_read(&mut self, read: &SequenceRead) -> Option<ReadResult> {
        self.stats.total_reads += 1;

        // Stage 1: QC
        self.current_stage = PipelineStage::QualityControl;
        if !self.quality_control(read) {
            self.stats.failed_qc += 1;
            return None;
        }
        self.stats.passed_qc += 1;

        // Stage 2: Embed
        self.current_stage = PipelineStage::Embed;
        let embedding = self.embedder.embed(&read.sequence);
        self.stats.embedded += 1;

        // Stage 3: Search
        self.current_stage = PipelineStage::Search;
        let search_results = self.index.search(&embedding, self.config.search_top_k)
            .unwrap_or_default();
        self.stats.search_hits += search_results.len();

        // Stage 4: Variant calling (against top hit if available)
        self.current_stage = PipelineStage::VariantCall;
        let variants = Vec::new(); // Would need reference sequence access for real variant calling

        self.current_stage = PipelineStage::Complete;

        Some(ReadResult {
            read_id: read.id.clone(),
            search_results,
            variants,
        })
    }

    /// Process a batch of FASTQ reads
    pub fn process_fastq(&mut self, fastq_content: &str) -> Result<Vec<ReadResult>, SequenceError> {
        self.current_stage = PipelineStage::Parse;
        let reads = parse_fastq(fastq_content)?;

        let results: Vec<ReadResult> = reads.iter()
            .filter_map(|read| self.process_read(read))
            .collect();

        Ok(results)
    }

    /// Process FASTA records (no quality, all pass QC)
    pub fn process_fasta(&mut self, fasta_content: &str) -> Result<Vec<ReadResult>, SequenceError> {
        self.current_stage = PipelineStage::Parse;
        let records = parse_fasta(fasta_content)?;

        let results: Vec<ReadResult> = records.iter().map(|rec| {
            self.stats.total_reads += 1;
            self.stats.passed_qc += 1;

            let embedding = self.embedder.embed(&rec.sequence);
            self.stats.embedded += 1;

            let search_results = self.index.search(&embedding, self.config.search_top_k)
                .unwrap_or_default();
            self.stats.search_hits += search_results.len();

            ReadResult {
                read_id: rec.id.clone(),
                search_results,
                variants: Vec::new(),
            }
        }).collect();

        Ok(results)
    }

    /// Get pipeline statistics
    pub fn stats(&self) -> &PipelineStats {
        &self.stats
    }

    /// Get current pipeline stage
    pub fn current_stage(&self) -> PipelineStage {
        self.current_stage
    }

    /// Number of references in the index
    pub fn reference_count(&self) -> usize {
        self.index.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pipeline_with_references() {
        let mut pipeline = DnaPipeline::new(PipelineConfig::default());

        // Add reference genomes
        let phix = PackedSequence::from_ascii(
            b"GAGTTTTATCGCTTCCATGACGCAGAAGTTAACACTTTCGGATATTTCTGATGAGTCGAAAAATTATCTTGATAAAGCAGGAATTACTACTGCTTGTTTACGA"
        ).unwrap();
        pipeline.add_reference("NC_001422", "PhiX174", "Viruses;Microviridae", &phix);

        let ecoli_seg = PackedSequence::from_ascii(
            b"AGCTTTTCATTCTGACTGCAACGGGCAATATGTCTCTGTGTGGATTAAAAAAAGAGTGTCTGATAGCAGCTTCTGAACTGGTTACCTGCCGTGAGTAAATTAAA"
        ).unwrap();
        pipeline.add_reference("U00096", "E.coli K-12", "Bacteria;Proteobacteria", &ecoli_seg);

        assert_eq!(pipeline.reference_count(), 2);
    }

    #[test]
    fn test_qc_filtering() {
        let pipeline = DnaPipeline::new(PipelineConfig {
            min_mean_quality: 20.0,
            min_length: 10,
            ..Default::default()
        });

        // High quality read
        let good_read = SequenceRead {
            id: "good".into(),
            sequence: PackedSequence::from_ascii(b"ACGTACGTACGTACGT").unwrap(),
            quality: QualityScores::from_phred33(b"IIIIIIIIIIIIIIII"), // Q=40
        };
        assert!(pipeline.quality_control(&good_read));

        // Low quality read
        let bad_read = SequenceRead {
            id: "bad".into(),
            sequence: PackedSequence::from_ascii(b"ACGTACGTACGTACGT").unwrap(),
            quality: QualityScores::from_phred33(b"!!!!!!!!!!!!!!!!"), // Q=0
        };
        assert!(!pipeline.quality_control(&bad_read));
    }

    #[test]
    fn test_full_pipeline_fastq() {
        let mut pipeline = DnaPipeline::new(PipelineConfig {
            min_length: 8,
            min_mean_quality: 10.0,
            ..Default::default()
        });

        // Add a reference
        let reference = PackedSequence::from_ascii(b"ACGTACGTACGTACGTACGTACGTACGTACGT").unwrap();
        pipeline.add_reference("ref1", "TestOrg", "Test;Taxonomy", &reference);

        let fastq = "@read1\nACGTACGTACGTACGT\n+\nIIIIIIIIIIIIIIII\n@read2\nGGCCTTAAGGCCTTAA\n+\nIIIIIIIIIIIIIIII";
        let results = pipeline.process_fastq(fastq).unwrap();

        assert_eq!(results.len(), 2);
        assert_eq!(pipeline.stats().total_reads, 2);
        assert_eq!(pipeline.stats().passed_qc, 2);
        assert_eq!(pipeline.stats().embedded, 2);
    }

    #[test]
    fn test_pipeline_stats() {
        let mut pipeline = DnaPipeline::new(PipelineConfig {
            min_length: 20,
            min_mean_quality: 30.0,
            ..Default::default()
        });

        // Short read should fail QC
        let short_read = SequenceRead {
            id: "short".into(),
            sequence: PackedSequence::from_ascii(b"ACGT").unwrap(),
            quality: QualityScores::from_phred33(b"IIII"),
        };

        let result = pipeline.process_read(&short_read);
        assert!(result.is_none());
        assert_eq!(pipeline.stats().failed_qc, 1);
    }
}
