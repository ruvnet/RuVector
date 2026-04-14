//! Learned entry point routing — skip top-layer traversal.
//!
//! Queries in the same region follow similar paths through the HNSW
//! graph. The first 2-3 nodes in the search path are predictable once
//! we learn which graph region a query vector belongs to.
//!
//! # Training process
//!
//! 1. Accumulate 1000+ `(query, search_path)` records via [`record_search`].
//! 2. Call [`train`] which runs k-means on the query vectors.
//! 3. For each cluster, the most common first 2-3 path nodes become
//!    the "highway on-ramps" for that region.
//! 4. An EML router learns `query_features -> region_id`.
//!
//! At search time: [`predict_entries`] predicts the region, then
//! returns cached entry points so the caller can start the search
//! 2-3 hops closer to the answer.

use eml_core::EmlModel;
use serde::{Deserialize, Serialize};

/// Maximum number of entry-point candidates returned per prediction.
const MAX_ENTRY_CANDIDATES: usize = 3;

/// Minimum number of search records before training is attempted.
const MIN_TRAINING_RECORDS: usize = 200;

/// A recorded search observation: query vector + the path taken.
#[derive(Debug, Clone)]
struct SearchRecord {
    query: Vec<f32>,
    /// First few node IDs in the search path (typically 2-3).
    path_prefix: Vec<usize>,
}

/// Learned entry point routing — skip top-layer traversal.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchPathPredictor {
    /// Region centroids from k-means clustering of query vectors.
    centroids: Vec<Vec<f32>>,
    /// For each region: the best entry node IDs (learned from search
    /// history). Outer vec is indexed by region_id.
    region_entry_points: Vec<Vec<usize>>,
    /// EML model: query_features -> region_id.
    router: EmlModel,
    /// Number of regions (k for k-means).
    k: usize,
    /// Dimensionality of query vectors.
    dim: usize,
    /// Whether the predictor has been trained.
    trained: bool,
    /// Accumulated search records (skipped in serde).
    #[serde(skip)]
    records: Vec<SearchRecord>,
}

impl SearchPathPredictor {
    /// Create a new untrained predictor.
    ///
    /// # Arguments
    /// - `k_regions`: Number of regions for k-means clustering.
    /// - `dim`: Dimensionality of query vectors.
    pub fn new(k_regions: usize, dim: usize) -> Self {
        let k = k_regions.max(2);
        // Router input: dim features (we'll sample/compress to fit).
        // Router output: 1 head (region_id as continuous value).
        let input_count = dim.min(8);
        let router = EmlModel::new(4, input_count, 1);

        Self {
            centroids: Vec::new(),
            region_entry_points: Vec::new(),
            router,
            k,
            dim,
            trained: false,
            records: Vec::new(),
        }
    }

    /// Whether the predictor has been trained and is ready for queries.
    pub fn is_trained(&self) -> bool {
        self.trained
    }

    /// Number of accumulated search records.
    pub fn record_count(&self) -> usize {
        self.records.len()
    }

    /// Number of regions.
    pub fn num_regions(&self) -> usize {
        self.k
    }

    /// Predict entry points for this query (skip top-layer traversal).
    ///
    /// Returns a list of node IDs that are good starting points for
    /// the bottom-layer search. If the predictor is not trained, returns
    /// an empty vec (caller should fall back to normal entry point).
    pub fn predict_entries(&self, query: &[f32]) -> Vec<usize> {
        if !self.trained || self.centroids.is_empty() {
            return Vec::new();
        }

        let region = self.predict_region(query);
        if region < self.region_entry_points.len() {
            self.region_entry_points[region].clone()
        } else {
            Vec::new()
        }
    }

    /// Record a completed search for training.
    ///
    /// # Arguments
    /// - `query`: The query vector that was searched.
    /// - `path`: The node IDs visited during the search, in order.
    ///   Only the first few (up to 3) are used.
    pub fn record_search(&mut self, query: &[f32], path: &[usize]) {
        if path.is_empty() {
            return;
        }
        let prefix_len = path.len().min(MAX_ENTRY_CANDIDATES);
        self.records.push(SearchRecord {
            query: query.to_vec(),
            path_prefix: path[..prefix_len].to_vec(),
        });
    }

    /// Build regions from accumulated search data and train the router.
    ///
    /// Returns `true` if training succeeded.
    pub fn train(&mut self) -> bool {
        if self.records.len() < MIN_TRAINING_RECORDS {
            return false;
        }

        // Step 1: k-means on query vectors.
        // Clone queries to avoid borrowing self.records while mutating self.centroids.
        let queries_owned: Vec<Vec<f32>> = self.records.iter().map(|r| r.query.clone()).collect();
        let queries: Vec<&[f32]> = queries_owned.iter().map(|q| q.as_slice()).collect();
        let assignments = self.run_kmeans(&queries);

        // Step 2: For each cluster, find the most common path-prefix
        // nodes. These become the entry points for that region.
        let mut region_paths: Vec<Vec<Vec<usize>>> = vec![Vec::new(); self.k];
        for (i, record) in self.records.iter().enumerate() {
            let cluster = assignments[i];
            region_paths[cluster].push(record.path_prefix.clone());
        }

        self.region_entry_points = Vec::with_capacity(self.k);
        for cluster_paths in &region_paths {
            let entries = Self::find_common_entries(cluster_paths);
            self.region_entry_points.push(entries);
        }

        // Step 3: Train EML router: query_features -> region_id.
        let input_count = self.router.input_count();
        let mut router = EmlModel::new(4, input_count, 1);

        for (i, record) in self.records.iter().enumerate() {
            let features = self.extract_features(&record.query);
            let region_id = assignments[i] as f64 / self.k as f64;
            router.record(&features, &[Some(region_id)]);
        }
        router.train();
        self.router = router;
        self.trained = true;
        true
    }

    // ---------------------------------------------------------------
    // Internal helpers
    // ---------------------------------------------------------------

    /// Predict region ID for a query vector.
    fn predict_region(&self, query: &[f32]) -> usize {
        // Primary: use centroid distance (fast and reliable).
        let mut best_region = 0;
        let mut best_dist = f32::MAX;
        for (i, centroid) in self.centroids.iter().enumerate() {
            let d = l2_distance(query, centroid);
            if d < best_dist {
                best_dist = d;
                best_region = i;
            }
        }
        best_region
    }

    /// Extract compressed feature vector for the EML router.
    fn extract_features(&self, query: &[f32]) -> Vec<f64> {
        let input_count = self.router.input_count();
        let mut features = vec![0.0f64; input_count];
        for (i, f) in features.iter_mut().enumerate() {
            // Sample evenly spaced dimensions from the query.
            let idx = if self.dim > 1 {
                (i * self.dim) / input_count
            } else {
                0
            };
            if idx < query.len() {
                *f = query[idx] as f64;
            }
        }
        features
    }

    /// Run k-means clustering on the query vectors.
    /// Returns the cluster assignment for each query.
    fn run_kmeans(&mut self, queries: &[&[f32]]) -> Vec<usize> {
        let n = queries.len();
        let dim = self.dim;

        // Initialize centroids: pick evenly spaced queries.
        self.centroids = Vec::with_capacity(self.k);
        for i in 0..self.k {
            let idx = (i * n) / self.k;
            let mut centroid = vec![0.0f32; dim];
            let src = queries[idx];
            for (j, c) in centroid.iter_mut().enumerate() {
                if j < src.len() {
                    *c = src[j];
                }
            }
            self.centroids.push(centroid);
        }

        let mut assignments = vec![0usize; n];
        let max_iters = 20;

        for _ in 0..max_iters {
            // Assignment step.
            let mut changed = false;
            for (i, q) in queries.iter().enumerate() {
                let mut best = 0;
                let mut best_d = f32::MAX;
                for (c, centroid) in self.centroids.iter().enumerate() {
                    let d = l2_distance(q, centroid);
                    if d < best_d {
                        best_d = d;
                        best = c;
                    }
                }
                if assignments[i] != best {
                    changed = true;
                    assignments[i] = best;
                }
            }
            if !changed {
                break;
            }

            // Update step: recompute centroids.
            let mut sums = vec![vec![0.0f64; dim]; self.k];
            let mut counts = vec![0usize; self.k];
            for (i, q) in queries.iter().enumerate() {
                let c = assignments[i];
                counts[c] += 1;
                for (j, val) in q.iter().enumerate() {
                    if j < dim {
                        sums[c][j] += *val as f64;
                    }
                }
            }
            for c in 0..self.k {
                if counts[c] > 0 {
                    for j in 0..dim {
                        self.centroids[c][j] = (sums[c][j] / counts[c] as f64) as f32;
                    }
                }
            }
        }

        assignments
    }

    /// Find the most common entry-point nodes across a set of path prefixes.
    fn find_common_entries(paths: &[Vec<usize>]) -> Vec<usize> {
        if paths.is_empty() {
            return Vec::new();
        }

        // Count node frequency across all path prefixes.
        let mut freq: std::collections::HashMap<usize, usize> = std::collections::HashMap::new();
        for path in paths {
            for &node in path {
                *freq.entry(node).or_insert(0) += 1;
            }
        }

        // Sort by frequency descending, take top MAX_ENTRY_CANDIDATES.
        let mut entries: Vec<(usize, usize)> = freq.into_iter().collect();
        entries.sort_by(|a, b| b.1.cmp(&a.1));
        entries
            .into_iter()
            .take(MAX_ENTRY_CANDIDATES)
            .map(|(node, _)| node)
            .collect()
    }
}

/// Squared L2 distance between two vectors.
fn l2_distance(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len().min(b.len());
    let mut sum = 0.0f32;
    for i in 0..len {
        let d = a[i] - b[i];
        sum += d * d;
    }
    sum
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_predictor_not_trained() {
        let p = SearchPathPredictor::new(4, 128);
        assert!(!p.is_trained());
        assert_eq!(p.record_count(), 0);
        assert_eq!(p.num_regions(), 4);
    }

    #[test]
    fn predict_entries_untrained_returns_empty() {
        let p = SearchPathPredictor::new(4, 3);
        let entries = p.predict_entries(&[0.1, 0.2, 0.3]);
        assert!(entries.is_empty());
    }

    #[test]
    fn record_search_increments_count() {
        let mut p = SearchPathPredictor::new(4, 3);
        p.record_search(&[1.0, 2.0, 3.0], &[10, 20, 30]);
        assert_eq!(p.record_count(), 1);
    }

    #[test]
    fn record_search_ignores_empty_path() {
        let mut p = SearchPathPredictor::new(4, 3);
        p.record_search(&[1.0, 2.0, 3.0], &[]);
        assert_eq!(p.record_count(), 0);
    }

    #[test]
    fn train_insufficient_data_returns_false() {
        let mut p = SearchPathPredictor::new(4, 3);
        for i in 0..50 {
            let v = i as f32 / 50.0;
            p.record_search(&[v, v * 2.0, v * 3.0], &[i, i + 1]);
        }
        assert!(!p.train());
    }

    #[test]
    fn train_with_sufficient_data_succeeds() {
        let mut p = SearchPathPredictor::new(3, 3);

        // Region A: queries near (0, 0, 0), paths through nodes 100-102.
        for i in 0..100 {
            let v = i as f32 * 0.001;
            p.record_search(&[v, v, v], &[100, 101, 102]);
        }
        // Region B: queries near (1, 1, 1), paths through nodes 200-202.
        for i in 0..100 {
            let v = 1.0 + i as f32 * 0.001;
            p.record_search(&[v, v, v], &[200, 201, 202]);
        }
        // Region C: queries near (5, 5, 5), paths through nodes 300-302.
        for i in 0..100 {
            let v = 5.0 + i as f32 * 0.001;
            p.record_search(&[v, v, v], &[300, 301, 302]);
        }

        assert!(p.train());
        assert!(p.is_trained());

        // Verify region A query returns region A entries.
        let entries_a = p.predict_entries(&[0.05, 0.05, 0.05]);
        assert!(!entries_a.is_empty());
        assert!(entries_a.len() <= 3);
        // The entries for region A should include node 100 (most common
        // first node in that cluster).
        assert!(
            entries_a.contains(&100),
            "Expected region A entries to contain 100, got {:?}",
            entries_a
        );

        // Verify region B query returns region B entries.
        let entries_b = p.predict_entries(&[1.05, 1.05, 1.05]);
        assert!(!entries_b.is_empty());
        assert!(
            entries_b.contains(&200),
            "Expected region B entries to contain 200, got {:?}",
            entries_b
        );
    }

    #[test]
    fn l2_distance_correctness() {
        let a = [1.0f32, 2.0, 3.0];
        let b = [4.0f32, 5.0, 6.0];
        let d = l2_distance(&a, &b);
        // (3^2 + 3^2 + 3^2) = 27
        assert!((d - 27.0).abs() < 1e-6);
    }

    #[test]
    fn l2_distance_identical_is_zero() {
        let a = [1.0f32, 2.0, 3.0];
        let d = l2_distance(&a, &a);
        assert!((d - 0.0).abs() < 1e-6);
    }

    #[test]
    fn find_common_entries_empty() {
        let entries = SearchPathPredictor::find_common_entries(&[]);
        assert!(entries.is_empty());
    }

    #[test]
    fn find_common_entries_selects_most_frequent() {
        let paths = vec![
            vec![10, 20, 30],
            vec![10, 20, 40],
            vec![10, 25, 30],
            vec![15, 20, 30],
        ];
        let entries = SearchPathPredictor::find_common_entries(&paths);
        // node 10 appears 3 times, 20 appears 3 times, 30 appears 3 times
        assert_eq!(entries.len(), 3);
        // All of 10, 20, 30 have frequency 3
        assert!(entries.contains(&10));
        assert!(entries.contains(&20));
        assert!(entries.contains(&30));
    }

    #[test]
    fn serialization_roundtrip() {
        let mut p = SearchPathPredictor::new(2, 3);
        // Train with minimal data
        for i in 0..150 {
            let v = i as f32 / 150.0;
            p.record_search(&[v, v * 0.5, v * 2.0], &[i % 10, i % 5]);
        }
        for i in 0..150 {
            let v = 10.0 + i as f32 / 150.0;
            p.record_search(&[v, v * 0.5, v * 2.0], &[100 + i % 10, 100 + i % 5]);
        }
        p.train();

        let json = serde_json::to_string(&p).unwrap();
        let p2: SearchPathPredictor = serde_json::from_str(&json).unwrap();
        assert_eq!(p.k, p2.k);
        assert_eq!(p.dim, p2.dim);
        assert_eq!(p.trained, p2.trained);
        assert_eq!(p.centroids.len(), p2.centroids.len());
        assert_eq!(p.region_entry_points.len(), p2.region_entry_points.len());
    }
}
