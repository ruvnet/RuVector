//! StreamPQ: online reservoir-sampled PQ with periodic full codebook retrain.
//!
//! Design:
//!  - Stores raw vectors alongside PQ codes so re-encoding is always correct.
//!  - Reservoir sampling (Vitter Algorithm R) maintains a bounded window of
//!    representative recent vectors (expected uniform sample of all seen data).
//!  - Every `update_freq` inserts, run full k-means on the reservoir to
//!    produce a fresh codebook, then re-encode ALL stored vectors.
//!
//! Why full retrain instead of EMA one-pass:
//!  - With a shift ≈ (cluster_spacing / 2), the EMA averages two different
//!    Phase-B clusters into the same centroid bin (they both fall closest to
//!    the same Phase-A centroid and get blended together). Full k-means on the
//!    reservoir restarts from data and correctly separates all clusters once the
//!    reservoir is dominated by the new distribution.
//!
//! Tradeoff:
//!  - Memory: O(n × dims) for raw vectors + O(n × M) for codes.
//!  - Update cost: O(reservoir × M × K × TRAIN_ITERS) per refresh.
//!  - Benefit: codebook tracks distribution drift; recall stays high as the
//!    agent's embedding distribution shifts across tasks or topics.

use crate::{AnnVariant, Hit};
use crate::pq::{Codebook, M, K};
use rand::{Rng, SeedableRng};

pub struct StreamPq {
    codebook: Option<Codebook>,
    codes: Vec<Vec<u8>>,
    raw_vecs: Vec<Vec<f32>>, // stored for correct re-encoding after codebook updates
    reservoir: Vec<Vec<f32>>,
    reservoir_cap: usize,
    seen_count: usize,
    inserts_since_update: usize,
    update_freq: usize,
    rng: rand::rngs::StdRng,
}

impl StreamPq {
    pub fn new(reservoir_cap: usize, update_freq: usize) -> Self {
        Self {
            codebook: None,
            codes: Vec::new(),
            raw_vecs: Vec::new(),
            reservoir: Vec::with_capacity(reservoir_cap),
            reservoir_cap,
            seen_count: 0,
            inserts_since_update: 0,
            update_freq,
            rng: rand::rngs::StdRng::seed_from_u64(99),
        }
    }

    /// Add `v` to the reservoir using Vitter's Algorithm R.
    fn reservoir_add(&mut self, v: Vec<f32>) {
        if self.reservoir.len() < self.reservoir_cap {
            self.reservoir.push(v);
        } else {
            let j = self.rng.gen_range(0..self.seen_count + 1);
            if j < self.reservoir_cap {
                self.reservoir[j] = v;
            }
        }
        self.seen_count += 1;
    }

    /// Trigger a full codebook retrain + re-encoding if interval elapsed.
    fn maybe_refresh(&mut self) {
        self.inserts_since_update += 1;
        if self.inserts_since_update < self.update_freq {
            return;
        }
        if self.reservoir.len() < M * K {
            return; // too few reservoir samples to train meaningfully
        }
        self.inserts_since_update = 0;
        // Full k-means retrain on the reservoir sample.
        // Using seen_count as seed diversifies initialization across refreshes.
        let new_cb = Codebook::train(&self.reservoir, M, K, self.seen_count as u64);
        self.codebook = Some(new_cb);
        // Re-encode ALL stored raw vectors with the fresh codebook.
        // O(n × M × ds) — necessary for consistent ADC distance computation.
        if let Some(cb) = &self.codebook {
            self.codes = self.raw_vecs.iter().map(|v| cb.encode(v)).collect();
        }
    }
}

impl AnnVariant for StreamPq {
    fn build(&mut self, vectors: &[Vec<f32>]) {
        if vectors.is_empty() {
            return;
        }
        for v in vectors {
            self.reservoir_add(v.clone());
        }
        self.raw_vecs = vectors.to_vec();
        let cb = Codebook::train(vectors, M, K, 2);
        self.codes = vectors.iter().map(|v| cb.encode(v)).collect();
        self.codebook = Some(cb);
    }

    fn insert(&mut self, vector: Vec<f32>) {
        // Encode and store before updating the reservoir (so ordering is stable).
        if let Some(cb) = &self.codebook {
            let code = cb.encode(&vector);
            self.codes.push(code);
        }
        self.raw_vecs.push(vector.clone());
        self.reservoir_add(vector);
        self.maybe_refresh();
    }

    fn search(&self, query: &[f32], k: usize) -> Vec<Hit> {
        let cb = match &self.codebook {
            Some(c) => c,
            None => return vec![],
        };
        let table = cb.adc_table(query);
        let mut hits: Vec<Hit> = self
            .codes
            .iter()
            .enumerate()
            .map(|(id, code)| Hit { id, dist: Codebook::adc_dist(&table, code) })
            .collect();
        hits.sort_unstable_by(|a, b| a.dist.partial_cmp(&b.dist).unwrap());
        hits.truncate(k);
        hits
    }

    fn name(&self) -> &str { "StreamPQ" }
    fn len(&self) -> usize { self.codes.len() }
    fn memory_bytes(&self) -> usize {
        let code_bytes: usize = self.codes.iter().map(|c| c.len()).sum();
        let raw_bytes: usize = self.raw_vecs.iter().map(|v| v.len() * 4).sum();
        let reservoir_bytes: usize = self.reservoir.iter().map(|v| v.len() * 4).sum();
        let centroid_bytes = self.codebook.as_ref().map_or(0, |cb| {
            cb.centroids.iter().flat_map(|sub| sub.iter()).map(|c| c.len() * 4).sum()
        });
        code_bytes + raw_bytes + reservoir_bytes + centroid_bytes
    }
}
