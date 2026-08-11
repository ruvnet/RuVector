use crate::quantize::{Nibble4BitQuantizer, Scalar8BitQuantizer};

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Precision {
    High, // 8-bit
    Low,  // 4-bit
}

/// Policy for assigning precision to each vector.
#[derive(Clone, Debug)]
pub enum PrecisionPolicy {
    /// All vectors at 8-bit (baseline).
    UniformHigh,
    /// Graph-degree-guided: high-degree nodes → High precision.
    GraphGuided { high_fraction: f32 },
    /// Access-frequency-guided: frequently accessed → High precision.
    AccessFreq { high_fraction: f32 },
}

#[derive(Clone, Debug)]
pub struct StoreConfig {
    pub policy: PrecisionPolicy,
    pub n: usize,
    pub dim: usize,
}

/// Stores quantized vectors with per-vector precision assignment.
pub struct AdaptiveQuantStore {
    pub precision: Vec<Precision>,
    pub high_q: Scalar8BitQuantizer,
    pub low_q: Nibble4BitQuantizer,
    /// For High precision: one entry per vector (empty if Low).
    encoded: Vec<Option<Vec<u8>>>,
    /// For Low precision: one entry per vector (empty if High).
    encoded_low: Vec<Option<Vec<u8>>>,
    pub dim: usize,
    pub n: usize,
}

impl AdaptiveQuantStore {
    pub fn build(
        data: &[Vec<f32>],
        precision: Vec<Precision>,
        high_q: Scalar8BitQuantizer,
        low_q: Nibble4BitQuantizer,
    ) -> Self {
        let n = data.len();
        let dim = data[0].len();
        let mut encoded = vec![None; n];
        let mut encoded_low = vec![None; n];
        for (i, vec) in data.iter().enumerate() {
            match precision[i] {
                Precision::High => encoded[i] = Some(high_q.encode(vec)),
                Precision::Low => encoded_low[i] = Some(low_q.encode(vec)),
            }
        }
        Self {
            precision,
            high_q,
            low_q,
            encoded,
            encoded_low,
            dim,
            n,
        }
    }

    /// Reconstruct a vector to f32 for distance computation.
    pub fn reconstruct(&self, id: usize) -> Vec<f32> {
        match self.precision[id] {
            Precision::High => self.high_q.decode(self.encoded[id].as_ref().unwrap()),
            Precision::Low => self
                .low_q
                .decode(self.encoded_low[id].as_ref().unwrap(), self.dim),
        }
    }

    /// L2-squared distance from query to vector id (via reconstruction).
    pub fn distance(&self, query: &[f32], id: usize) -> f32 {
        let rec = self.reconstruct(id);
        query
            .iter()
            .zip(rec.iter())
            .map(|(a, b)| (a - b) * (a - b))
            .sum()
    }

    /// Brute-force search: returns top-k indices by approximate L2.
    pub fn search(&self, query: &[f32], k: usize) -> Vec<usize> {
        let mut dists: Vec<(f32, usize)> =
            (0..self.n).map(|i| (self.distance(query, i), i)).collect();
        dists.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        dists[..k.min(self.n)].iter().map(|(_, i)| *i).collect()
    }

    /// Total memory for encoded vectors in bytes (excluding struct overhead).
    pub fn encoded_bytes(&self) -> usize {
        let high_bytes: usize = self
            .encoded
            .iter()
            .filter_map(|e| e.as_ref())
            .map(|v| v.len())
            .sum();
        let low_bytes: usize = self
            .encoded_low
            .iter()
            .filter_map(|e| e.as_ref())
            .map(|v| v.len())
            .sum();
        high_bytes + low_bytes
    }

    /// Baseline memory if all were 8-bit.
    pub fn baseline_bytes(&self) -> usize {
        self.n * Scalar8BitQuantizer::bytes_per_vec(self.dim)
    }

    /// Memory ratio vs baseline (< 1.0 means savings).
    pub fn memory_ratio(&self) -> f32 {
        self.encoded_bytes() as f32 / self.baseline_bytes() as f32
    }

    pub fn high_precision_count(&self) -> usize {
        self.precision
            .iter()
            .filter(|&&p| p == Precision::High)
            .count()
    }

    pub fn low_precision_count(&self) -> usize {
        self.precision
            .iter()
            .filter(|&&p| p == Precision::Low)
            .count()
    }
}

/// Builder: create an AdaptiveQuantStore from data and a precision mask.
pub fn build_uniform_high(data: &[Vec<f32>]) -> AdaptiveQuantStore {
    let high_q = Scalar8BitQuantizer::fit(data);
    let low_q = Nibble4BitQuantizer::fit(data);
    let precision = vec![Precision::High; data.len()];
    AdaptiveQuantStore::build(data, precision, high_q, low_q)
}

pub fn build_graph_guided(data: &[Vec<f32>], high_mask: &[bool]) -> AdaptiveQuantStore {
    let high_q = Scalar8BitQuantizer::fit(data);
    let low_q = Nibble4BitQuantizer::fit(data);
    let precision = high_mask
        .iter()
        .map(|&h| if h { Precision::High } else { Precision::Low })
        .collect();
    AdaptiveQuantStore::build(data, precision, high_q, low_q)
}

pub fn build_access_freq(
    data: &[Vec<f32>],
    access_freq: &[usize],
    high_fraction: f32,
) -> AdaptiveQuantStore {
    let high_q = Scalar8BitQuantizer::fit(data);
    let low_q = Nibble4BitQuantizer::fit(data);
    let n = data.len();
    let mut indexed: Vec<(usize, usize)> = access_freq.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| b.1.cmp(&a.1));
    let n_high = (n as f32 * high_fraction).round() as usize;
    let mut precision = vec![Precision::Low; n];
    for (i, _) in &indexed[..n_high] {
        precision[*i] = Precision::High;
    }
    AdaptiveQuantStore::build(data, precision, high_q, low_q)
}

/// Random selection baseline: randomly pick `high_fraction` of vectors for 8-bit.
pub fn build_random_mixed(data: &[Vec<f32>], high_fraction: f32, seed: u64) -> AdaptiveQuantStore {
    use rand::rngs::StdRng;
    use rand::Rng;
    use rand::SeedableRng;
    let high_q = Scalar8BitQuantizer::fit(data);
    let low_q = Nibble4BitQuantizer::fit(data);
    let n = data.len();
    let n_high = (n as f32 * high_fraction).round() as usize;
    let mut rng = StdRng::seed_from_u64(seed);
    let mut indices: Vec<usize> = (0..n).collect();
    // Fisher-Yates partial shuffle for first n_high elements
    for i in 0..n_high {
        let j = rng.gen_range(i..n);
        indices.swap(i, j);
    }
    let mut precision = vec![Precision::Low; n];
    for &idx in &indices[..n_high] {
        precision[idx] = Precision::High;
    }
    AdaptiveQuantStore::build(data, precision, high_q, low_q)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::generate_clustered;

    #[test]
    fn test_uniform_high_search() {
        let data = generate_clustered(200, 16, 5, 42);
        let store = build_uniform_high(&data);
        let query = &data[0];
        let results = store.search(query, 5);
        assert_eq!(results.len(), 5);
        // The query vector itself should be among top results (distance ≈ 0 after quant)
        assert!(results.contains(&0), "Query vector should be in top-5");
    }

    #[test]
    fn test_memory_is_less_for_graph_guided() {
        let data = generate_clustered(200, 32, 5, 42);
        let high_mask: Vec<bool> = (0..200).map(|i| i < 40).collect(); // 20% high
        let graph_store = build_graph_guided(&data, &high_mask);
        let base_store = build_uniform_high(&data);
        assert!(
            graph_store.encoded_bytes() < base_store.encoded_bytes(),
            "Graph-guided should use less memory than uniform 8-bit"
        );
    }

    #[test]
    fn test_reconstruction_fidelity_high() {
        let data = generate_clustered(50, 8, 3, 7);
        let store = build_uniform_high(&data);
        for (i, orig) in data.iter().enumerate() {
            let rec = store.reconstruct(i);
            let max_err: f32 = orig
                .iter()
                .zip(rec.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            // 8-bit quantization: max error ≈ scale/2
            assert!(
                max_err < 0.5,
                "8-bit reconstruction error {} too large",
                max_err
            );
        }
    }

    #[test]
    fn test_precision_counts() {
        let data = generate_clustered(100, 16, 5, 1);
        let high_mask: Vec<bool> = (0..100).map(|i| i < 20).collect();
        let store = build_graph_guided(&data, &high_mask);
        assert_eq!(store.high_precision_count(), 20);
        assert_eq!(store.low_precision_count(), 80);
    }
}
