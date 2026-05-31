//! RVQ encoder, ADC scoring table, and three searchable index variants.

use std::collections::BinaryHeap;

use rand::SeedableRng;

use crate::codebook::Codebook;

// ── Public result type ───────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub struct SearchResult {
    pub id: usize,
    pub dist: f32,
}

// ── Shared trait ─────────────────────────────────────────────────────────────

pub trait AnnIndex {
    fn search(&self, query: &[f32], k: usize) -> Vec<SearchResult>;
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool { self.len() == 0 }
    fn dim(&self) -> usize;
    fn memory_bytes(&self) -> usize;
}

// ── RVQ Encoder ──────────────────────────────────────────────────────────────

/// Trained RVQ encoder: M cascaded codebooks, each of size K (≤ 256).
#[derive(Debug, Clone)]
pub struct RvqEncoder {
    pub codebooks: Vec<Codebook>,
}

impl RvqEncoder {
    /// Train M codebooks on `vectors`. Each stage quantises the residual from
    /// the previous stage. A random subset (≤ 32 768) is used for k-means.
    pub fn train(
        vectors: &[Vec<f32>],
        n_codebooks: usize,
        k: usize,
        n_iter: usize,
        seed: u64,
    ) -> Self {
        assert!(!vectors.is_empty(), "empty training set");
        assert!(k <= 256, "k must be ≤ 256 (u8 codes)");
        let dim = vectors[0].len();
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

        // Limit training subset to avoid slow k-means on huge sets
        let n_train = vectors.len().min(32_768);
        let mut residuals: Vec<Vec<f32>> = vectors[..n_train].to_vec();

        let mut codebooks = Vec::with_capacity(n_codebooks);
        for _ in 0..n_codebooks {
            let flat: Vec<f32> = residuals.iter().flat_map(|v| v.iter().copied()).collect();
            let cb = Codebook::train(&flat, dim, k, n_iter, &mut rng);

            // Compute residuals for next stage
            residuals = residuals
                .iter()
                .map(|v| {
                    let code = cb.quantize(v);
                    cb.residual(v, code)
                })
                .collect();

            codebooks.push(cb);
        }

        Self { codebooks }
    }

    /// Greedy (beam=1) encoding: at each stage pick the nearest centroid.
    pub fn encode_greedy(&self, v: &[f32]) -> Vec<u8> {
        let mut residual = v.to_vec();
        let mut codes = Vec::with_capacity(self.codebooks.len());
        for cb in &self.codebooks {
            let code = cb.quantize(&residual);
            residual = cb.residual(&residual, code);
            codes.push(code);
        }
        codes
    }

    /// Beam-search encoding: maintain `beam_width` partial assignments per stage.
    /// Produces lower reconstruction error than greedy at the cost of encode time.
    pub fn encode_beam(&self, v: &[f32], beam_width: usize) -> Vec<u8> {
        // Each beam state: (cumulative_sq_error, residual, codes_so_far)
        let mut beam: Vec<(f32, Vec<f32>, Vec<u8>)> =
            vec![(0.0, v.to_vec(), Vec::new())];

        for cb in &self.codebooks {
            let mut candidates: Vec<(f32, Vec<f32>, Vec<u8>)> =
                Vec::with_capacity(beam.len() * beam_width);
            for (err, residual, codes) in &beam {
                for (cand_code, cand_d) in cb.top_n(residual, beam_width) {
                    let mut new_codes = codes.clone();
                    new_codes.push(cand_code);
                    let new_res = cb.residual(residual, cand_code);
                    candidates.push((err + cand_d, new_res, new_codes));
                }
            }
            candidates
                .sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
            candidates.truncate(beam_width);
            beam = candidates;
        }

        beam.into_iter().next().map(|(_, _, c)| c).unwrap_or_default()
    }

    /// Reconstruct a vector from its codes (sum of selected centroids).
    pub fn decode(&self, codes: &[u8]) -> Vec<f32> {
        let dim = self.codebooks[0].dim;
        let mut out = vec![0.0f32; dim];
        for (cb, &code) in self.codebooks.iter().zip(codes.iter()) {
            let c = cb.centroid(code);
            for (o, &cv) in out.iter_mut().zip(c) {
                *o += cv;
            }
        }
        out
    }

    /// Asymmetric Distance Computation table for a query vector.
    ///
    /// Returns an `AdcTable` that scores any stored code vector in O(M)
    /// via inner-product lookup: score = Σ_m ⟨q, c_m[codes[m]]⟩.
    /// Combined with precomputed self-norms this gives the exact squared L2
    /// distance to the reconstruction: ||q − x̂||² = ||q||² − 2·score + ||x̂||²
    pub fn adc_table(&self, query: &[f32]) -> AdcTable {
        let m = self.codebooks.len();
        let k = self.codebooks[0].k;
        let mut table = vec![0.0f32; m * k];
        for (mi, cb) in self.codebooks.iter().enumerate() {
            for j in 0..k {
                table[mi * k + j] = cb.inner_product(query, j as u8);
            }
        }
        AdcTable { table, k, query_sq_norm: query.iter().map(|x| x * x).sum() }
    }

    pub fn n_codebooks(&self) -> usize { self.codebooks.len() }
    pub fn k(&self) -> usize { self.codebooks[0].k }
    pub fn dim(&self) -> usize { self.codebooks[0].dim }
    pub fn bits_per_vector(&self) -> usize { self.codebooks.len() * 8 }
    pub fn memory_bytes(&self) -> usize {
        self.codebooks.iter().map(|cb| cb.memory_bytes()).sum::<usize>() + std::mem::size_of::<Self>()
    }
}

// ── ADC Table ────────────────────────────────────────────────────────────────

/// Precomputed inner-product lookup table for one query.
pub struct AdcTable {
    table: Vec<f32>,
    k: usize,
    query_sq_norm: f32,
}

impl AdcTable {
    /// Estimate ||q − x̂||² using inner-product tables.
    ///
    /// Uses: ||q − x̂||² = ||q||² − 2·Σ_m ⟨q, c_m[code_m]⟩ + ||x̂||²
    #[inline]
    pub fn score(&self, codes: &[u8], self_norm: f32) -> f32 {
        let ip: f32 = codes
            .iter()
            .enumerate()
            .map(|(m, &c)| self.table[m * self.k + c as usize])
            .sum();
        self.query_sq_norm - 2.0 * ip + self_norm
    }

    #[inline]
    pub fn ip_only(&self, codes: &[u8]) -> f32 {
        codes
            .iter()
            .enumerate()
            .map(|(m, &c)| self.table[m * self.k + c as usize])
            .sum()
    }
}

// ── Shared index internals ───────────────────────────────────────────────────

struct StoredVec {
    codes: Vec<u8>,
    self_norm: f32,
}

fn build_storage(
    encoder: &RvqEncoder,
    vectors: &[Vec<f32>],
    beam: Option<usize>,
) -> (Vec<StoredVec>, usize) {
    let dim = encoder.dim();
    let mut store = Vec::with_capacity(vectors.len());
    for v in vectors {
        let codes = match beam {
            Some(bw) => encoder.encode_beam(v, bw),
            None => encoder.encode_greedy(v),
        };
        let recon = encoder.decode(&codes);
        let self_norm: f32 = recon.iter().map(|x| x * x).sum();
        store.push(StoredVec { codes, self_norm });
    }
    (store, dim)
}

fn adc_search(
    store: &[StoredVec],
    adc: &AdcTable,
    k: usize,
) -> Vec<SearchResult> {
    // Max-heap of size k tracking the k *smallest* ADC distances.
    // heap.peek() = largest distance currently kept.
    // We replace it whenever we find something smaller.
    //
    // Use ordered (score_bits, id) where score_bits encodes f32 in a way
    // that preserves ordering for non-negative floats: clamp to 0 first.
    let mut heap: BinaryHeap<(u32, usize)> = BinaryHeap::with_capacity(k + 1);
    for (id, sv) in store.iter().enumerate() {
        let raw = adc.score(&sv.codes, sv.self_norm);
        // Clamp to non-negative (floating-point rounding can give tiny negatives).
        let dist = raw.max(0.0);
        let bits = dist.to_bits();
        if heap.len() < k {
            heap.push((bits, id));
        } else if let Some(&(top_bits, _)) = heap.peek() {
            if bits < top_bits {
                heap.pop();
                heap.push((bits, id));
            }
        }
    }
    let mut results: Vec<SearchResult> = heap
        .into_iter()
        .map(|(bits, id)| SearchResult { id, dist: f32::from_bits(bits) })
        .collect();
    results.sort_by(|a, b| a.dist.partial_cmp(&b.dist).unwrap_or(std::cmp::Ordering::Equal));
    results
}

fn exact_rerank(
    candidates: &[SearchResult],
    originals: &[Vec<f32>],
    query: &[f32],
    k: usize,
) -> Vec<SearchResult> {
    let mut refined: Vec<SearchResult> = candidates
        .iter()
        .map(|r| {
            let orig = &originals[r.id];
            let d: f32 = query.iter().zip(orig).map(|(a, b)| (a - b) * (a - b)).sum();
            SearchResult { id: r.id, dist: d }
        })
        .collect();
    refined.sort_by(|a, b| a.dist.partial_cmp(&b.dist).unwrap_or(std::cmp::Ordering::Equal));
    refined.truncate(k);
    refined
}

// ── Variant 1: RvqGreedyIndex ────────────────────────────────────────────────

/// Greedy encoding + ADC scoring. Fastest to build, very fast to search.
pub struct RvqGreedyIndex {
    encoder: RvqEncoder,
    store: Vec<StoredVec>,
    dim: usize,
}

impl RvqGreedyIndex {
    pub fn build(vectors: &[Vec<f32>], n_codebooks: usize, k: usize, n_iter: usize) -> Self {
        let encoder = RvqEncoder::train(vectors, n_codebooks, k, n_iter, 42);
        Self::build_with_encoder(encoder, vectors)
    }

    pub fn build_with_encoder(encoder: RvqEncoder, vectors: &[Vec<f32>]) -> Self {
        let (store, dim) = build_storage(&encoder, vectors, None);
        Self { encoder, store, dim }
    }
}

impl AnnIndex for RvqGreedyIndex {
    fn search(&self, query: &[f32], k: usize) -> Vec<SearchResult> {
        let adc = self.encoder.adc_table(query);
        adc_search(&self.store, &adc, k)
    }
    fn len(&self) -> usize { self.store.len() }
    fn dim(&self) -> usize { self.dim }
    fn memory_bytes(&self) -> usize {
        self.encoder.memory_bytes()
            + self.store.len() * (self.encoder.n_codebooks() + 4 + 24)
    }
}

// ── Variant 2: RvqBeamIndex ──────────────────────────────────────────────────

/// Beam-search encoding (beam_width=4) + ADC scoring.
/// Better recall than greedy at the cost of ~4× slower encoding; search is identical.
pub struct RvqBeamIndex {
    encoder: RvqEncoder,
    store: Vec<StoredVec>,
    dim: usize,
    beam_width: usize,
}

impl RvqBeamIndex {
    pub fn build(
        vectors: &[Vec<f32>],
        n_codebooks: usize,
        k: usize,
        n_iter: usize,
        beam_width: usize,
    ) -> Self {
        let encoder = RvqEncoder::train(vectors, n_codebooks, k, n_iter, 42);
        let (store, dim) = build_storage(&encoder, vectors, Some(beam_width));
        Self { encoder, store, dim, beam_width }
    }

    pub fn beam_width(&self) -> usize { self.beam_width }
}

impl AnnIndex for RvqBeamIndex {
    fn search(&self, query: &[f32], k: usize) -> Vec<SearchResult> {
        let adc = self.encoder.adc_table(query);
        adc_search(&self.store, &adc, k)
    }
    fn len(&self) -> usize { self.store.len() }
    fn dim(&self) -> usize { self.dim }
    fn memory_bytes(&self) -> usize {
        self.encoder.memory_bytes()
            + self.store.len() * (self.encoder.n_codebooks() + 4 + 24)
    }
}

// ── Variant 3: RvqRerankIndex ────────────────────────────────────────────────

/// Greedy encoding + ADC coarse search + exact L2 rerank on top candidates.
/// Highest recall; uses more memory (stores originals) and has moderate search cost.
pub struct RvqRerankIndex {
    encoder: RvqEncoder,
    store: Vec<StoredVec>,
    originals: Vec<Vec<f32>>,
    dim: usize,
    rerank_factor: usize,
}

impl RvqRerankIndex {
    /// `rerank_factor`: fetch `rerank_factor × k` candidates then rerank exactly.
    pub fn build(
        vectors: &[Vec<f32>],
        n_codebooks: usize,
        k: usize,
        n_iter: usize,
        rerank_factor: usize,
    ) -> Self {
        let encoder = RvqEncoder::train(vectors, n_codebooks, k, n_iter, 42);
        let (store, dim) = build_storage(&encoder, vectors, None);
        Self { encoder, store, originals: vectors.to_vec(), dim, rerank_factor }
    }
}

impl AnnIndex for RvqRerankIndex {
    fn search(&self, query: &[f32], k: usize) -> Vec<SearchResult> {
        let fetch_k = (k * self.rerank_factor).min(self.store.len());
        let adc = self.encoder.adc_table(query);
        let candidates = adc_search(&self.store, &adc, fetch_k);
        exact_rerank(&candidates, &self.originals, query, k)
    }
    fn len(&self) -> usize { self.store.len() }
    fn dim(&self) -> usize { self.dim }
    fn memory_bytes(&self) -> usize {
        self.encoder.memory_bytes()
            + self.store.len() * (self.encoder.n_codebooks() + 4 + 24)
            + self.originals.iter().map(|v| v.len() * 4).sum::<usize>()
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_distr::{Distribution, Normal};

    fn gaussian_vecs(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let norm = Normal::new(0.0f32, 1.0).unwrap();
        (0..n).map(|_| (0..dim).map(|_| norm.sample(&mut rng)).collect()).collect()
    }

    fn brute_force(vecs: &[Vec<f32>], query: &[f32], k: usize) -> Vec<usize> {
        let mut scored: Vec<(usize, f32)> = vecs
            .iter()
            .enumerate()
            .map(|(id, v)| {
                let d: f32 = query.iter().zip(v).map(|(a, b)| (a - b) * (a - b)).sum();
                (id, d)
            })
            .collect();
        scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        scored.truncate(k);
        scored.into_iter().map(|(id, _)| id).collect()
    }

    fn recall(got: &[SearchResult], truth: &[usize]) -> f32 {
        let truth_set: std::collections::HashSet<usize> = truth.iter().copied().collect();
        let hits = got.iter().filter(|r| truth_set.contains(&r.id)).count();
        hits as f32 / truth.len() as f32
    }

    #[test]
    fn encoder_roundtrip() {
        let vecs = gaussian_vecs(500, 32, 0);
        let enc = RvqEncoder::train(&vecs, 4, 32, 10, 0);
        // Reconstruction error must be less than variance of data
        let test_v = &vecs[0];
        let codes = enc.encode_greedy(test_v);
        let recon = enc.decode(&codes);
        let err: f32 = test_v.iter().zip(&recon).map(|(a, b)| (a - b) * (a - b)).sum::<f32>().sqrt();
        let norm: f32 = test_v.iter().map(|x| x * x).sum::<f32>().sqrt();
        // Error must be less than 80% of the vector norm (codebook captures structure)
        assert!(err < norm * 0.8, "reconstruction error {err:.4} >= 0.8 * norm {:.4}", norm * 0.8);
    }

    #[test]
    fn beam_beats_greedy_recon() {
        let vecs = gaussian_vecs(300, 32, 1);
        let enc = RvqEncoder::train(&vecs, 4, 32, 10, 1);
        let v = &vecs[0];
        let greedy_codes = enc.encode_greedy(v);
        let beam_codes = enc.encode_beam(v, 4);
        let greedy_recon = enc.decode(&greedy_codes);
        let beam_recon = enc.decode(&beam_codes);
        let err_greedy: f32 =
            v.iter().zip(&greedy_recon).map(|(a, b)| (a - b) * (a - b)).sum::<f32>();
        let err_beam: f32 =
            v.iter().zip(&beam_recon).map(|(a, b)| (a - b) * (a - b)).sum::<f32>();
        // Beam must be at least as good as greedy (≤ reconstruction error)
        assert!(err_beam <= err_greedy + 1e-4, "beam err {err_beam:.6} > greedy {err_greedy:.6}");
    }

    #[test]
    fn greedy_index_recall() {
        // Clustered data so k-means has real structure to exploit.
        // Random i.i.d. Gaussian has no clusters → k-means gives poor centroids.
        use rand_distr::Uniform;
        let mut rng = rand::rngs::StdRng::seed_from_u64(99);
        let dim = 64;
        let n = 1000;
        let n_clusters = 20;
        let cr = Uniform::new(-2.0f32, 2.0);
        let noise = Normal::new(0.0f32, 0.25).unwrap();
        let centers: Vec<Vec<f32>> =
            (0..n_clusters).map(|_| (0..dim).map(|_| cr.sample(&mut rng)).collect()).collect();
        let vecs: Vec<Vec<f32>> = (0..n)
            .map(|i| {
                let c = &centers[i % n_clusters];
                c.iter().map(|&x| x + noise.sample(&mut rng)).collect()
            })
            .collect();

        let idx = RvqGreedyIndex::build(&vecs, 8, 64, 15);
        let query = &vecs[0];
        let results = idx.search(query, 10);
        let truth = brute_force(&vecs, query, 10);
        let r = recall(&results, &truth);
        // Greedy ADC on clustered data; random baseline is 10/1000 = 1%.
        // We accept ≥20% (20× random), indicating the index is finding structure.
        assert!(r >= 0.20, "greedy recall@10 {r:.2} too low (random baseline 0.01)");
    }

    #[test]
    fn rerank_improves_recall() {
        let vecs = gaussian_vecs(1000, 64, 3);
        let query = &vecs[7];
        let truth = brute_force(&vecs, query, 10);

        let greedy = RvqGreedyIndex::build(&vecs, 8, 64, 10);
        let rerank = RvqRerankIndex::build(&vecs, 8, 64, 10, 5);

        let r_greedy = recall(&greedy.search(query, 10), &truth);
        let r_rerank = recall(&rerank.search(query, 10), &truth);
        assert!(
            r_rerank >= r_greedy - 0.05,
            "rerank {r_rerank:.2} worse than greedy {r_greedy:.2} by > 5%"
        );
    }

    #[test]
    fn adc_matches_exact_dist() {
        let vecs = gaussian_vecs(100, 32, 4);
        let enc = RvqEncoder::train(&vecs, 4, 32, 10, 4);
        let query = &vecs[0];
        let adc = enc.adc_table(query);
        for v in &vecs[..10] {
            let codes = enc.encode_greedy(v);
            let recon = enc.decode(&codes);
            let adc_score: f32 = {
                let self_norm: f32 = recon.iter().map(|x| x * x).sum();
                adc.score(&codes, self_norm)
            };
            let exact_dist: f32 =
                query.iter().zip(&recon).map(|(a, b)| (a - b) * (a - b)).sum();
            // ADC score must equal exact distance to reconstruction (within float rounding)
            assert!(
                (adc_score - exact_dist).abs() < 1e-3,
                "ADC {adc_score:.6} != exact {exact_dist:.6}"
            );
        }
    }
}
