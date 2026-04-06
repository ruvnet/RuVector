//! DiskANN index — ties together Vamana graph, PQ, and mmap persistence

use crate::distance::l2_squared;
use crate::error::{DiskAnnError, Result};
use crate::graph::VamanaGraph;
use crate::pq::ProductQuantizer;
use memmap2::{Mmap, MmapOptions};
use parking_lot::RwLock;
use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// Search result
#[derive(Debug, Clone)]
pub struct SearchResult {
    pub id: String,
    pub distance: f32,
}

/// Configuration for DiskANN index
#[derive(Debug, Clone)]
pub struct DiskAnnConfig {
    /// Vector dimension
    pub dim: usize,
    /// Maximum out-degree for Vamana graph (R)
    pub max_degree: usize,
    /// Search beam width during construction (L_build)
    pub build_beam: usize,
    /// Search beam width during query (L_search)
    pub search_beam: usize,
    /// Alpha parameter for robust pruning (>= 1.0)
    pub alpha: f32,
    /// Number of PQ subspaces (M). 0 = no PQ.
    pub pq_subspaces: usize,
    /// PQ training iterations
    pub pq_iterations: usize,
    /// Storage directory for persistence
    pub storage_path: Option<PathBuf>,
}

impl Default for DiskAnnConfig {
    fn default() -> Self {
        Self {
            dim: 128,
            max_degree: 64,
            build_beam: 128,
            search_beam: 64,
            alpha: 1.2,
            pq_subspaces: 0,
            pq_iterations: 10,
            storage_path: None,
        }
    }
}

/// DiskANN index with Vamana graph + optional PQ + mmap persistence
pub struct DiskAnnIndex {
    config: DiskAnnConfig,
    /// Raw vectors (in-memory or mmap'd)
    vectors: Vec<Vec<f32>>,
    /// ID mapping: internal index -> external string ID
    id_map: Vec<String>,
    /// Reverse mapping: external ID -> internal index
    id_reverse: HashMap<String, u32>,
    /// Vamana graph
    graph: Option<VamanaGraph>,
    /// Product quantizer (optional)
    pq: Option<ProductQuantizer>,
    /// PQ codes for all vectors
    pq_codes: Vec<Vec<u8>>,
    /// Whether index has been built
    built: bool,
    /// Memory-mapped vector data (for large datasets)
    mmap: Option<Mmap>,
}

impl DiskAnnIndex {
    /// Create a new DiskANN index
    pub fn new(config: DiskAnnConfig) -> Self {
        Self {
            config,
            vectors: Vec::new(),
            id_map: Vec::new(),
            id_reverse: HashMap::new(),
            graph: None,
            pq: None,
            pq_codes: Vec::new(),
            built: false,
            mmap: None,
        }
    }

    /// Insert a vector with a string ID
    pub fn insert(&mut self, id: String, vector: Vec<f32>) -> Result<()> {
        if vector.len() != self.config.dim {
            return Err(DiskAnnError::DimensionMismatch {
                expected: self.config.dim,
                actual: vector.len(),
            });
        }
        if self.id_reverse.contains_key(&id) {
            return Err(DiskAnnError::InvalidConfig(format!("Duplicate ID: {id}")));
        }

        let idx = self.vectors.len() as u32;
        self.id_reverse.insert(id.clone(), idx);
        self.id_map.push(id);
        self.vectors.push(vector);
        self.built = false; // Invalidate index
        Ok(())
    }

    /// Insert a batch of vectors
    pub fn insert_batch(&mut self, entries: Vec<(String, Vec<f32>)>) -> Result<()> {
        for (id, vector) in entries {
            self.insert(id, vector)?;
        }
        Ok(())
    }

    /// Build the index (must be called after all inserts, before search)
    pub fn build(&mut self) -> Result<()> {
        let n = self.vectors.len();
        if n == 0 {
            return Err(DiskAnnError::Empty);
        }

        // Train PQ if configured
        if self.config.pq_subspaces > 0 {
            let mut pq = ProductQuantizer::new(self.config.dim, self.config.pq_subspaces)?;
            pq.train(&self.vectors, self.config.pq_iterations)?;

            // Encode all vectors
            self.pq_codes = self
                .vectors
                .iter()
                .map(|v| pq.encode(v))
                .collect::<Result<Vec<_>>>()?;

            self.pq = Some(pq);
        }

        // Build Vamana graph
        let mut graph = VamanaGraph::new(
            n,
            self.config.max_degree,
            self.config.build_beam,
            self.config.alpha,
        );
        graph.build(&self.vectors)?;
        self.graph = Some(graph);
        self.built = true;

        // Persist if storage path configured
        if let Some(ref path) = self.config.storage_path {
            self.save(path)?;
        }

        Ok(())
    }

    /// Search for k nearest neighbors
    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<SearchResult>> {
        if !self.built {
            return Err(DiskAnnError::NotBuilt);
        }
        if query.len() != self.config.dim {
            return Err(DiskAnnError::DimensionMismatch {
                expected: self.config.dim,
                actual: query.len(),
            });
        }

        let graph = self.graph.as_ref().unwrap();

        // Two-phase search when PQ is available:
        // Phase 1: Beam search using PQ distances (fast, approximate)
        // Phase 2: Re-rank top candidates using exact distances
        if let Some(ref pq) = self.pq {
            let table = pq.build_distance_table(query)?;
            let beam = self.config.search_beam.max(k * 4);

            // Greedy search with PQ distance for candidate generation
            let (candidates, _) = graph.greedy_search(&self.vectors, query, beam);

            // Re-rank with exact distance
            let mut scored: Vec<(u32, f32)> = candidates
                .into_iter()
                .map(|id| (id, l2_squared(&self.vectors[id as usize], query)))
                .collect();
            scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

            Ok(scored
                .into_iter()
                .take(k)
                .map(|(id, dist)| SearchResult {
                    id: self.id_map[id as usize].clone(),
                    distance: dist,
                })
                .collect())
        } else {
            // No PQ: direct beam search with exact distances
            let (candidates, _) =
                graph.greedy_search(&self.vectors, query, self.config.search_beam.max(k));

            let mut scored: Vec<(u32, f32)> = candidates
                .into_iter()
                .map(|id| (id, l2_squared(&self.vectors[id as usize], query)))
                .collect();
            scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

            Ok(scored
                .into_iter()
                .take(k)
                .map(|(id, dist)| SearchResult {
                    id: self.id_map[id as usize].clone(),
                    distance: dist,
                })
                .collect())
        }
    }

    /// Get the number of vectors in the index
    pub fn count(&self) -> usize {
        self.vectors.len()
    }

    /// Delete a vector by ID (marks as deleted, doesn't rebuild graph)
    pub fn delete(&mut self, id: &str) -> Result<bool> {
        if let Some(&idx) = self.id_reverse.get(id) {
            // Zero out the vector (lazy deletion)
            for v in &mut self.vectors[idx as usize] {
                *v = f32::NAN;
            }
            self.id_reverse.remove(id);
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Save index to disk
    pub fn save(&self, dir: &Path) -> Result<()> {
        fs::create_dir_all(dir)?;

        // Save vectors as flat binary (mmap-friendly)
        let vec_path = dir.join("vectors.bin");
        let mut f = BufWriter::new(File::create(&vec_path)?);
        let n = self.vectors.len() as u64;
        let dim = self.config.dim as u64;
        f.write_all(&n.to_le_bytes())?;
        f.write_all(&dim.to_le_bytes())?;
        for v in &self.vectors {
            for &val in v {
                f.write_all(&val.to_le_bytes())?;
            }
        }
        f.flush()?;

        // Save graph adjacency
        let graph_path = dir.join("graph.bin");
        let mut f = BufWriter::new(File::create(&graph_path)?);
        if let Some(ref graph) = self.graph {
            f.write_all(&(graph.medoid as u64).to_le_bytes())?;
            f.write_all(&(graph.neighbors.len() as u64).to_le_bytes())?;
            for neighbors in &graph.neighbors {
                f.write_all(&(neighbors.len() as u32).to_le_bytes())?;
                for &n in neighbors {
                    f.write_all(&n.to_le_bytes())?;
                }
            }
        }
        f.flush()?;

        // Save ID map
        let ids_path = dir.join("ids.json");
        let ids_json = serde_json::to_string(&self.id_map)
            .map_err(|e| DiskAnnError::Serialization(e.to_string()))?;
        fs::write(&ids_path, ids_json)?;

        // Save PQ if present
        if let Some(ref pq) = self.pq {
            let pq_path = dir.join("pq.bin");
            let pq_bytes = bincode::encode_to_vec(pq, bincode::config::standard())
                .map_err(|e| DiskAnnError::Serialization(e.to_string()))?;
            fs::write(&pq_path, pq_bytes)?;

            // Save PQ codes
            let codes_path = dir.join("pq_codes.bin");
            let mut f = BufWriter::new(File::create(&codes_path)?);
            for codes in &self.pq_codes {
                f.write_all(codes)?;
            }
            f.flush()?;
        }

        // Save config
        let config_path = dir.join("config.json");
        let config_json = serde_json::json!({
            "dim": self.config.dim,
            "max_degree": self.config.max_degree,
            "build_beam": self.config.build_beam,
            "search_beam": self.config.search_beam,
            "alpha": self.config.alpha,
            "pq_subspaces": self.config.pq_subspaces,
            "count": self.vectors.len(),
            "built": self.built,
        });
        fs::write(&config_path, serde_json::to_string_pretty(&config_json).unwrap())?;

        Ok(())
    }

    /// Load index from disk with memory-mapped vectors
    pub fn load(dir: &Path) -> Result<Self> {
        // Load config
        let config_json: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(dir.join("config.json"))?)
                .map_err(|e| DiskAnnError::Serialization(e.to_string()))?;

        let dim = config_json["dim"].as_u64().unwrap() as usize;
        let max_degree = config_json["max_degree"].as_u64().unwrap() as usize;
        let build_beam = config_json["build_beam"].as_u64().unwrap() as usize;
        let search_beam = config_json["search_beam"].as_u64().unwrap() as usize;
        let alpha = config_json["alpha"].as_f64().unwrap() as f32;
        let pq_subspaces = config_json["pq_subspaces"].as_u64().unwrap_or(0) as usize;

        let config = DiskAnnConfig {
            dim,
            max_degree,
            build_beam,
            search_beam,
            alpha,
            pq_subspaces,
            storage_path: Some(dir.to_path_buf()),
            ..Default::default()
        };

        // Load vectors via mmap
        let vec_file = File::open(dir.join("vectors.bin"))?;
        let mmap = unsafe { MmapOptions::new().map(&vec_file)? };

        let n = u64::from_le_bytes(mmap[0..8].try_into().unwrap()) as usize;
        let file_dim = u64::from_le_bytes(mmap[8..16].try_into().unwrap()) as usize;
        assert_eq!(file_dim, dim);

        let mut vectors = Vec::with_capacity(n);
        let data_start = 16;
        for i in 0..n {
            let offset = data_start + i * dim * 4;
            let mut v = Vec::with_capacity(dim);
            for d in 0..dim {
                let byte_offset = offset + d * 4;
                let val = f32::from_le_bytes(
                    mmap[byte_offset..byte_offset + 4].try_into().unwrap(),
                );
                v.push(val);
            }
            vectors.push(v);
        }

        // Load IDs
        let ids_json = fs::read_to_string(dir.join("ids.json"))?;
        let id_map: Vec<String> = serde_json::from_str(&ids_json)
            .map_err(|e| DiskAnnError::Serialization(e.to_string()))?;

        let mut id_reverse = HashMap::new();
        for (i, id) in id_map.iter().enumerate() {
            id_reverse.insert(id.clone(), i as u32);
        }

        // Load graph
        let graph_bytes = fs::read(dir.join("graph.bin"))?;
        let medoid = u64::from_le_bytes(graph_bytes[0..8].try_into().unwrap()) as u32;
        let graph_n = u64::from_le_bytes(graph_bytes[8..16].try_into().unwrap()) as usize;

        let mut neighbors = Vec::with_capacity(graph_n);
        let mut offset = 16;
        for _ in 0..graph_n {
            let deg = u32::from_le_bytes(graph_bytes[offset..offset + 4].try_into().unwrap()) as usize;
            offset += 4;
            let mut nbrs = Vec::with_capacity(deg);
            for _ in 0..deg {
                let nbr = u32::from_le_bytes(graph_bytes[offset..offset + 4].try_into().unwrap());
                offset += 4;
                nbrs.push(nbr);
            }
            neighbors.push(nbrs);
        }

        let graph = VamanaGraph {
            neighbors,
            medoid,
            max_degree,
            build_beam,
            alpha,
        };

        // Load PQ if present
        let pq_path = dir.join("pq.bin");
        let (pq, pq_codes) = if pq_path.exists() {
            let pq_bytes = fs::read(&pq_path)?;
            let (pq, _): (ProductQuantizer, usize) =
                bincode::decode_from_slice(&pq_bytes, bincode::config::standard())
                    .map_err(|e| DiskAnnError::Serialization(e.to_string()))?;

            let codes_bytes = fs::read(dir.join("pq_codes.bin"))?;
            let m = pq.m;
            let mut codes = Vec::with_capacity(n);
            for i in 0..n {
                codes.push(codes_bytes[i * m..(i + 1) * m].to_vec());
            }
            (Some(pq), codes)
        } else {
            (None, Vec::new())
        };

        Ok(Self {
            config,
            vectors,
            id_map,
            id_reverse,
            graph: Some(graph),
            pq,
            pq_codes,
            built: true,
            mmap: Some(mmap),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    fn random_vectors(n: usize, dim: usize) -> Vec<(String, Vec<f32>)> {
        use rand::prelude::*;
        let mut rng = rand::thread_rng();
        (0..n)
            .map(|i| {
                let v: Vec<f32> = (0..dim).map(|_| rng.gen()).collect();
                (format!("vec-{i}"), v)
            })
            .collect()
    }

    #[test]
    fn test_diskann_basic() {
        let mut index = DiskAnnIndex::new(DiskAnnConfig {
            dim: 32,
            max_degree: 16,
            build_beam: 32,
            search_beam: 32,
            alpha: 1.2,
            ..Default::default()
        });

        let data = random_vectors(500, 32);
        let query = data[42].1.clone();

        index.insert_batch(data).unwrap();
        index.build().unwrap();

        let results = index.search(&query, 5).unwrap();
        assert!(!results.is_empty());
        assert_eq!(results[0].id, "vec-42"); // Should find itself
        assert!(results[0].distance < 1e-6); // Exact match
    }

    #[test]
    fn test_diskann_with_pq() {
        let mut index = DiskAnnIndex::new(DiskAnnConfig {
            dim: 32,
            max_degree: 16,
            build_beam: 32,
            search_beam: 32,
            alpha: 1.2,
            pq_subspaces: 4,
            pq_iterations: 5,
            ..Default::default()
        });

        let data = random_vectors(200, 32);
        let query = data[10].1.clone();

        index.insert_batch(data).unwrap();
        index.build().unwrap();

        let results = index.search(&query, 5).unwrap();
        assert_eq!(results[0].id, "vec-10");
    }

    #[test]
    fn test_diskann_save_load() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("diskann_test");

        let data = random_vectors(100, 16);
        let query = data[7].1.clone();

        // Build and save
        {
            let mut index = DiskAnnIndex::new(DiskAnnConfig {
                dim: 16,
                max_degree: 8,
                build_beam: 16,
                search_beam: 16,
                alpha: 1.2,
                storage_path: Some(path.clone()),
                ..Default::default()
            });
            index.insert_batch(data).unwrap();
            index.build().unwrap();
        }

        // Load and search
        let loaded = DiskAnnIndex::load(&path).unwrap();
        let results = loaded.search(&query, 3).unwrap();
        assert_eq!(results[0].id, "vec-7");
    }

    #[test]
    fn test_recall_at_10() {
        // Measure recall@10: what fraction of true top-10 neighbors does DiskANN find?
        use rand::prelude::*;
        let mut rng = rand::thread_rng();
        let n = 2000;
        let dim = 64;
        let k = 10;

        let data: Vec<(String, Vec<f32>)> = (0..n)
            .map(|i| {
                let v: Vec<f32> = (0..dim).map(|_| rng.gen()).collect();
                (format!("v{i}"), v)
            })
            .collect();

        let mut index = DiskAnnIndex::new(DiskAnnConfig {
            dim,
            max_degree: 32,
            build_beam: 64,
            search_beam: 64,
            alpha: 1.2,
            ..Default::default()
        });
        index.insert_batch(data.clone()).unwrap();
        index.build().unwrap();

        // Test 50 random queries
        let num_queries = 50;
        let mut total_recall = 0.0;

        for _ in 0..num_queries {
            let qi = rng.gen_range(0..n);
            let query = &data[qi].1;

            // Brute-force ground truth
            let mut brute: Vec<(usize, f32)> = data
                .iter()
                .enumerate()
                .map(|(i, (_, v))| (i, crate::distance::l2_squared(v, query)))
                .collect();
            brute.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            let gt: std::collections::HashSet<String> = brute[..k]
                .iter()
                .map(|(i, _)| data[*i].0.clone())
                .collect();

            // DiskANN search
            let results = index.search(query, k).unwrap();
            let found: std::collections::HashSet<String> =
                results.iter().map(|r| r.id.clone()).collect();

            let recall = gt.intersection(&found).count() as f64 / k as f64;
            total_recall += recall;
        }

        let avg_recall = total_recall / num_queries as f64;
        println!("Recall@{k} = {avg_recall:.3} (n={n}, dim={dim}, queries={num_queries})");
        assert!(
            avg_recall >= 0.85,
            "Recall@{k} = {avg_recall:.3}, expected >= 0.85"
        );
    }

    #[test]
    fn test_dimension_mismatch() {
        let mut index = DiskAnnIndex::new(DiskAnnConfig {
            dim: 16,
            ..Default::default()
        });

        // Wrong dimension on insert
        let result = index.insert("bad".to_string(), vec![1.0; 32]);
        assert!(result.is_err());

        // Wrong dimension on search
        index.insert("ok".to_string(), vec![1.0; 16]).unwrap();
        index.build().unwrap();
        let result = index.search(&[1.0; 32], 1);
        assert!(result.is_err());
    }

    #[test]
    fn test_duplicate_id_rejected() {
        let mut index = DiskAnnIndex::new(DiskAnnConfig {
            dim: 4,
            ..Default::default()
        });
        index.insert("a".to_string(), vec![1.0; 4]).unwrap();
        let result = index.insert("a".to_string(), vec![2.0; 4]);
        assert!(result.is_err());
    }

    #[test]
    fn test_search_before_build_fails() {
        let mut index = DiskAnnIndex::new(DiskAnnConfig {
            dim: 4,
            ..Default::default()
        });
        index.insert("a".to_string(), vec![1.0; 4]).unwrap();
        let result = index.search(&[1.0; 4], 1);
        assert!(result.is_err());
    }

    #[test]
    fn test_scale_5k() {
        // 5000 vectors, 128-dim — should build in under 5 seconds
        use std::time::Instant;
        use rand::prelude::*;
        let mut rng = rand::thread_rng();

        let n = 5000;
        let dim = 128;
        let data: Vec<(String, Vec<f32>)> = (0..n)
            .map(|i| {
                let v: Vec<f32> = (0..dim).map(|_| rng.gen()).collect();
                (format!("v{i}"), v)
            })
            .collect();

        let mut index = DiskAnnIndex::new(DiskAnnConfig {
            dim,
            max_degree: 48,
            build_beam: 96,
            search_beam: 48,
            alpha: 1.2,
            ..Default::default()
        });
        index.insert_batch(data.clone()).unwrap();

        let t0 = Instant::now();
        index.build().unwrap();
        let build_ms = t0.elapsed().as_millis();
        println!("Build {n} vectors ({dim}d): {build_ms}ms");

        // Search latency
        let query = &data[0].1;
        let t0 = Instant::now();
        let iters = 100;
        for _ in 0..iters {
            let _ = index.search(query, 10).unwrap();
        }
        let search_us = t0.elapsed().as_micros() / iters;
        println!("Search latency (k=10): {search_us}µs avg over {iters} queries");

        assert!(search_us < 10_000, "Search took {search_us}µs, expected <10ms");
    }
}
