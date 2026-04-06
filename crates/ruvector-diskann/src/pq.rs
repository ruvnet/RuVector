//! Product Quantization for compressed distance computation
//!
//! Splits D-dimensional vectors into M subspaces of D/M dimensions each,
//! then quantizes each subspace independently using k-means (k=256 centroids).

use crate::distance::l2_squared;
use crate::error::{DiskAnnError, Result};
use rand::prelude::*;
use bincode::{Decode, Encode};
use serde::{Deserialize, Serialize};

/// Product Quantizer with M subspaces, 256 centroids each (1 byte per subspace)
#[derive(Clone, Serialize, Deserialize, Encode, Decode)]
pub struct ProductQuantizer {
    /// Number of subspaces
    pub m: usize,
    /// Dimensions per subspace
    pub dsub: usize,
    /// Total dimensions
    pub dim: usize,
    /// Centroids: [m][256][dsub]
    pub centroids: Vec<Vec<Vec<f32>>>,
    /// Whether the PQ has been trained
    pub trained: bool,
}

impl ProductQuantizer {
    /// Create a new PQ with M subspaces for D-dimensional vectors
    pub fn new(dim: usize, m: usize) -> Result<Self> {
        if dim % m != 0 {
            return Err(DiskAnnError::InvalidConfig(format!(
                "dim ({dim}) must be divisible by m ({m})"
            )));
        }
        let dsub = dim / m;
        Ok(Self {
            m,
            dsub,
            dim,
            centroids: Vec::new(),
            trained: false,
        })
    }

    /// Train PQ centroids using k-means on training vectors
    pub fn train(&mut self, vectors: &[Vec<f32>], iterations: usize) -> Result<()> {
        if vectors.is_empty() {
            return Err(DiskAnnError::Empty);
        }
        if vectors[0].len() != self.dim {
            return Err(DiskAnnError::DimensionMismatch {
                expected: self.dim,
                actual: vectors[0].len(),
            });
        }

        let k = 256usize; // 1 byte per code
        let n = vectors.len();
        let mut rng = rand::thread_rng();

        self.centroids = Vec::with_capacity(self.m);

        for sub in 0..self.m {
            let offset = sub * self.dsub;

            // Extract subvectors for this subspace
            let subvectors: Vec<&[f32]> = vectors
                .iter()
                .map(|v| &v[offset..offset + self.dsub])
                .collect();

            // Initialize centroids with k-means++ seeding
            let mut centers = Vec::with_capacity(k);
            centers.push(subvectors[rng.gen_range(0..n)].to_vec());

            for _ in 1..k.min(n) {
                // Compute min distance from each point to nearest center
                let dists: Vec<f32> = subvectors
                    .iter()
                    .map(|sv| {
                        centers
                            .iter()
                            .map(|c| l2_squared(sv, c))
                            .fold(f32::MAX, f32::min)
                    })
                    .collect();

                let total: f32 = dists.iter().sum();
                if total < 1e-10 {
                    // All points are identical, fill remaining with the same
                    while centers.len() < k {
                        centers.push(centers[0].clone());
                    }
                    break;
                }

                // Weighted random selection
                let mut r = rng.gen::<f32>() * total;
                for (i, &d) in dists.iter().enumerate() {
                    r -= d;
                    if r <= 0.0 {
                        centers.push(subvectors[i].to_vec());
                        break;
                    }
                }
                if centers.len() < k && r > 0.0 {
                    centers.push(subvectors[rng.gen_range(0..n)].to_vec());
                }
            }

            // Pad if fewer training points than k
            while centers.len() < k {
                centers.push(centers[rng.gen_range(0..centers.len())].clone());
            }

            // Lloyd's iterations
            let mut assignments = vec![0u8; n];
            for _ in 0..iterations {
                // Assign
                for (i, sv) in subvectors.iter().enumerate() {
                    let mut best_dist = f32::MAX;
                    let mut best_c = 0u8;
                    for (c, center) in centers.iter().enumerate() {
                        let d = l2_squared(sv, center);
                        if d < best_dist {
                            best_dist = d;
                            best_c = c as u8;
                        }
                    }
                    assignments[i] = best_c;
                }

                // Update centroids
                let mut counts = vec![0usize; k];
                let mut sums = vec![vec![0.0f32; self.dsub]; k];

                for (i, &a) in assignments.iter().enumerate() {
                    let ci = a as usize;
                    counts[ci] += 1;
                    for d in 0..self.dsub {
                        sums[ci][d] += subvectors[i][d];
                    }
                }

                for c in 0..k {
                    if counts[c] > 0 {
                        for d in 0..self.dsub {
                            centers[c][d] = sums[c][d] / counts[c] as f32;
                        }
                    }
                }
            }

            self.centroids.push(centers);
        }

        self.trained = true;
        Ok(())
    }

    /// Encode a vector into PQ codes (M bytes)
    pub fn encode(&self, vector: &[f32]) -> Result<Vec<u8>> {
        if !self.trained {
            return Err(DiskAnnError::PqNotTrained);
        }
        if vector.len() != self.dim {
            return Err(DiskAnnError::DimensionMismatch {
                expected: self.dim,
                actual: vector.len(),
            });
        }

        let mut codes = Vec::with_capacity(self.m);
        for sub in 0..self.m {
            let offset = sub * self.dsub;
            let subvec = &vector[offset..offset + self.dsub];

            let mut best_dist = f32::MAX;
            let mut best_c = 0u8;
            for (c, center) in self.centroids[sub].iter().enumerate() {
                let d = l2_squared(subvec, center);
                if d < best_dist {
                    best_dist = d;
                    best_c = c as u8;
                }
            }
            codes.push(best_c);
        }
        Ok(codes)
    }

    /// Build flat asymmetric distance table for a query vector
    /// Returns flat table[subspace * 256 + centroid_id] = distance
    pub fn build_distance_table(&self, query: &[f32]) -> Result<Vec<f32>> {
        if !self.trained {
            return Err(DiskAnnError::PqNotTrained);
        }
        if query.len() != self.dim {
            return Err(DiskAnnError::DimensionMismatch {
                expected: self.dim,
                actual: query.len(),
            });
        }

        let k = 256;
        let mut table = vec![0.0f32; self.m * k];
        for sub in 0..self.m {
            let offset = sub * self.dsub;
            let subquery = &query[offset..offset + self.dsub];

            for (c, center) in self.centroids[sub].iter().enumerate() {
                table[sub * k + c] = l2_squared(subquery, center);
            }
        }
        Ok(table)
    }

    /// Compute approximate distance using flat precomputed table
    #[inline]
    pub fn distance_with_table(&self, codes: &[u8], table: &[f32]) -> f32 {
        crate::distance::pq_asymmetric_distance(codes, table, 256)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pq_train_encode() {
        let mut pq = ProductQuantizer::new(8, 2).unwrap();
        let vectors: Vec<Vec<f32>> = (0..100)
            .map(|i| (0..8).map(|d| (i * 7 + d) as f32 / 100.0).collect())
            .collect();
        pq.train(&vectors, 5).unwrap();

        let codes = pq.encode(&vectors[0]).unwrap();
        assert_eq!(codes.len(), 2); // M=2 subspaces

        let table = pq.build_distance_table(&vectors[0]).unwrap();
        let dist = pq.distance_with_table(&codes, &table);
        // Self-distance through PQ should be very small
        assert!(dist < 0.1, "self-distance was {dist}");
    }
}
