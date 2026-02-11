//! Genomic vector search for species identification and sequence similarity.
//!
//! Provides a lightweight in-memory index that stores DNA embeddings with
//! metadata, supporting cosine similarity search for species identification.

/// Metadata for an indexed genomic reference
#[derive(Clone, Debug)]
pub struct GenomicReference {
    pub id: String,
    pub organism: String,
    pub taxonomy: String,
    pub sequence_length: usize,
}

/// Search result from the genomic index
#[derive(Clone, Debug)]
pub struct GenomicSearchResult {
    pub reference: GenomicReference,
    pub similarity: f32,
    pub rank: usize,
}

/// In-memory genomic search index (brute-force for <10K vectors, sufficient for reference DBs)
pub struct GenomicSearchIndex {
    embeddings: Vec<Vec<f32>>,
    references: Vec<GenomicReference>,
    dimension: usize,
}

impl GenomicSearchIndex {
    pub fn new(dimension: usize) -> Self {
        GenomicSearchIndex {
            embeddings: Vec::new(),
            references: Vec::new(),
            dimension,
        }
    }

    /// Add a reference genome embedding
    pub fn add_reference(&mut self, embedding: Vec<f32>, reference: GenomicReference) -> Result<(), String> {
        if embedding.len() != self.dimension {
            return Err(format!("Expected dimension {}, got {}", self.dimension, embedding.len()));
        }
        self.embeddings.push(embedding);
        self.references.push(reference);
        Ok(())
    }

    /// Search for the most similar references to a query embedding
    pub fn search(&self, query: &[f32], top_k: usize) -> Result<Vec<GenomicSearchResult>, String> {
        if query.len() != self.dimension {
            return Err(format!("Expected dimension {}, got {}", self.dimension, query.len()));
        }

        let mut scored: Vec<(usize, f32)> = self.embeddings.iter().enumerate()
            .map(|(i, emb)| (i, cosine_similarity(query, emb)))
            .collect();

        // Sort by similarity descending
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let results = scored.iter().take(top_k).enumerate().map(|(rank, &(idx, sim))| {
            GenomicSearchResult {
                reference: self.references[idx].clone(),
                similarity: sim,
                rank: rank + 1,
            }
        }).collect();

        Ok(results)
    }

    /// Search with taxonomy filter
    pub fn search_filtered(&self, query: &[f32], top_k: usize, taxonomy_prefix: &str) -> Result<Vec<GenomicSearchResult>, String> {
        if query.len() != self.dimension {
            return Err(format!("Expected dimension {}, got {}", self.dimension, query.len()));
        }

        let mut scored: Vec<(usize, f32)> = self.embeddings.iter().enumerate()
            .filter(|(i, _)| self.references[*i].taxonomy.starts_with(taxonomy_prefix))
            .map(|(i, emb)| (i, cosine_similarity(query, emb)))
            .collect();

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let results = scored.iter().take(top_k).enumerate().map(|(rank, &(idx, sim))| {
            GenomicSearchResult {
                reference: self.references[idx].clone(),
                similarity: sim,
                rank: rank + 1,
            }
        }).collect();

        Ok(results)
    }

    pub fn len(&self) -> usize { self.embeddings.len() }
    pub fn is_empty(&self) -> bool { self.embeddings.is_empty() }
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a < 1e-10 || norm_b < 1e-10 { return 0.0; }
    dot / (norm_a * norm_b)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mock_embedding(dim: usize, seed: u64) -> Vec<f32> {
        let mut state = seed;
        let v: Vec<f32> = (0..dim).map(|_| {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((state >> 33) as f32) / (u32::MAX as f32) * 2.0 - 1.0
        }).collect();
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        v.iter().map(|x| x / norm).collect()
    }

    #[test]
    fn test_add_and_search() {
        let dim = 128;
        let mut index = GenomicSearchIndex::new(dim);

        let refs = vec![
            ("phix174", "PhiX174", "Viruses;ssDNA"),
            ("ecoli", "E. coli K-12", "Bacteria;Proteobacteria"),
            ("human_mito", "Human mitochondrion", "Eukaryota;Chordata"),
        ];

        for (i, (id, org, tax)) in refs.iter().enumerate() {
            index.add_reference(
                mock_embedding(dim, (i + 1) as u64),
                GenomicReference {
                    id: id.to_string(),
                    organism: org.to_string(),
                    taxonomy: tax.to_string(),
                    sequence_length: 5000 * (i + 1),
                },
            ).unwrap();
        }

        // Query with same embedding as phix174 should return phix174 first
        let query = mock_embedding(dim, 1);
        let results = index.search(&query, 3).unwrap();
        assert_eq!(results[0].reference.id, "phix174");
        assert!(results[0].similarity > 0.99);
    }

    #[test]
    fn test_taxonomy_filter() {
        let dim = 64;
        let mut index = GenomicSearchIndex::new(dim);

        index.add_reference(mock_embedding(dim, 1), GenomicReference {
            id: "virus1".into(), organism: "PhiX".into(), taxonomy: "Viruses;ssDNA".into(), sequence_length: 5000,
        }).unwrap();
        index.add_reference(mock_embedding(dim, 2), GenomicReference {
            id: "bact1".into(), organism: "E.coli".into(), taxonomy: "Bacteria;Proteo".into(), sequence_length: 4600000,
        }).unwrap();

        let results = index.search_filtered(&mock_embedding(dim, 1), 5, "Viruses").unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].reference.id, "virus1");
    }
}
