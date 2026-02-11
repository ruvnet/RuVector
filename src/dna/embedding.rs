//! DNA sequence to vector embedding for HNSW search.
//!
//! Uses k-mer frequency vectors projected to a fixed dimension via
//! a deterministic random projection matrix (Johnson-Lindenstrauss).

use super::kmer::kmer_frequency_vector;
use super::sequence::PackedSequence;

/// Configuration for DNA embedding
#[derive(Clone, Debug)]
pub struct EmbeddingConfig {
    /// k-mer size for embedding
    pub k: usize,
    /// Output embedding dimension (after projection)
    pub output_dim: usize,
    /// Whether to use canonical k-mers (strand-agnostic)
    pub canonical: bool,
}

impl Default for EmbeddingConfig {
    fn default() -> Self {
        EmbeddingConfig {
            k: 6,
            output_dim: 384,
            canonical: true,
        }
    }
}

/// DNA sequence embedder using k-mer frequency + random projection
pub struct DnaEmbedder {
    config: EmbeddingConfig,
    projection: Vec<Vec<f32>>,
}

impl DnaEmbedder {
    /// Create a new embedder with deterministic random projection
    pub fn new(config: EmbeddingConfig) -> Self {
        let vocab_size = 4usize.pow(config.k as u32);
        let projection = Self::generate_projection(vocab_size, config.output_dim, 42);
        DnaEmbedder { config, projection }
    }

    fn generate_projection(input_dim: usize, output_dim: usize, seed: u64) -> Vec<Vec<f32>> {
        let mut state = seed;
        let scale = 1.0 / (output_dim as f32).sqrt();

        (0..input_dim)
            .map(|_| {
                (0..output_dim)
                    .map(|_| {
                        state = state
                            .wrapping_mul(6364136223846793005)
                            .wrapping_add(1442695040888963407);
                        let val = ((state >> 33) as f32) / (u32::MAX as f32) * 2.0 - 1.0;
                        val * scale
                    })
                    .collect()
            })
            .collect()
    }

    /// Embed a DNA sequence into a fixed-dimensional vector
    pub fn embed(&self, sequence: &PackedSequence) -> Vec<f32> {
        let freqs = kmer_frequency_vector(sequence, self.config.k);
        let mut embedding = vec![0.0f32; self.config.output_dim];

        for (i, &freq) in freqs.iter().enumerate() {
            if freq > 0.0 && i < self.projection.len() {
                for (j, &proj_val) in self.projection[i].iter().enumerate() {
                    embedding[j] += freq * proj_val;
                }
            }
        }

        // L2 normalize
        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 1e-10 {
            for x in &mut embedding {
                *x /= norm;
            }
        }

        embedding
    }

    /// Embed multiple sequences
    pub fn embed_batch(&self, sequences: &[PackedSequence]) -> Vec<Vec<f32>> {
        sequences.iter().map(|seq| self.embed(seq)).collect()
    }

    /// Cosine similarity between two embeddings
    pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len());
        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm_a < 1e-10 || norm_b < 1e-10 {
            return 0.0;
        }
        dot / (norm_a * norm_b)
    }

    /// Euclidean distance between two embeddings
    pub fn euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len());
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).powi(2))
            .sum::<f32>()
            .sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_embedding_dimension() {
        let embedder = DnaEmbedder::new(EmbeddingConfig::default());
        let seq = PackedSequence::from_ascii(b"ACGTACGTACGTACGTACGTACGT").unwrap();
        let emb = embedder.embed(&seq);
        assert_eq!(emb.len(), 384);
    }

    #[test]
    fn test_embedding_normalized() {
        let embedder = DnaEmbedder::new(EmbeddingConfig::default());
        let seq = PackedSequence::from_ascii(b"ACGTACGTACGTACGTACGTACGT").unwrap();
        let emb = embedder.embed(&seq);
        let norm: f32 = emb.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 0.01, "Should be unit normalized: {}", norm);
    }

    #[test]
    fn test_similar_sequences_closer() {
        let embedder = DnaEmbedder::new(EmbeddingConfig::default());

        let seq_a = PackedSequence::from_ascii(b"ACGTACGTACGTACGTACGTACGTACGTACGT").unwrap();
        let seq_b = PackedSequence::from_ascii(b"ACGTACGTACGTACGTACGTACGTACGTACGA").unwrap();
        let seq_c = PackedSequence::from_ascii(b"TTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT").unwrap();

        let emb_a = embedder.embed(&seq_a);
        let emb_b = embedder.embed(&seq_b);
        let emb_c = embedder.embed(&seq_c);

        let sim_ab = DnaEmbedder::cosine_similarity(&emb_a, &emb_b);
        let sim_ac = DnaEmbedder::cosine_similarity(&emb_a, &emb_c);

        assert!(sim_ab > sim_ac, "Similar seqs should have higher cosine: ab={}, ac={}", sim_ab, sim_ac);
    }

    #[test]
    fn test_deterministic() {
        let embedder = DnaEmbedder::new(EmbeddingConfig::default());
        let seq = PackedSequence::from_ascii(b"ACGTACGTACGTACGT").unwrap();
        let emb1 = embedder.embed(&seq);
        let emb2 = embedder.embed(&seq);
        assert_eq!(emb1, emb2, "Embeddings should be deterministic");
    }
}
