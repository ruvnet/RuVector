//! Flat (brute-force) index variants.
//!
//! `FlatExact` — full-precision exhaustive scan; ground-truth baseline.
//! `FlatSq8`   — 8-bit quantized scan over the whole corpus, then exact re-rank.
//!
//! Both implement [`NnSearch`] with identical public API.

use crate::{l2_sq, NnResult, NnSearch, ScalarQuantizer};

/// Exact brute-force nearest-neighbour index.  Defines ground truth for recall.
pub struct FlatExact {
    vecs: Vec<Vec<f32>>,
}

impl FlatExact {
    pub fn new() -> Self {
        Self { vecs: Vec::new() }
    }
}

impl Default for FlatExact {
    fn default() -> Self {
        Self::new()
    }
}

impl NnSearch for FlatExact {
    fn insert(&mut self, vector: Vec<f32>) {
        self.vecs.push(vector);
    }

    fn search(&self, query: &[f32], k: usize) -> Vec<NnResult> {
        let mut dists: Vec<(f32, usize)> = self
            .vecs
            .iter()
            .enumerate()
            .map(|(i, v)| (l2_sq(query, v), i))
            .collect();
        dists.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        dists.truncate(k);
        dists
            .into_iter()
            .map(|(d, id)| NnResult { id, distance: d })
            .collect()
    }

    fn len(&self) -> usize {
        self.vecs.len()
    }

    fn memory_bytes(&self) -> usize {
        self.vecs.iter().map(|v| v.len() * 4).sum()
    }
}

/// Flat index with 8-bit scalar quantization + full-precision re-ranking.
///
/// Search flow:
/// 1. Scan all SQ8 codes (fast integer math, 4× less data bandwidth).
/// 2. Collect top `ef` candidates by quantized distance.
/// 3. Re-rank with exact f32 L2 on stored originals.
pub struct FlatSq8 {
    originals: Vec<Vec<f32>>,
    codes: Vec<Vec<u8>>,
    quantizer: Option<ScalarQuantizer>,
    ef: usize,
}

impl FlatSq8 {
    /// `ef` is the number of candidates retained before full-precision re-rank.
    /// Typical values: ef = k * 4 to k * 10.
    pub fn new(ef: usize) -> Self {
        Self {
            originals: Vec::new(),
            codes: Vec::new(),
            quantizer: None,
            ef,
        }
    }

    /// Initialise quantizer from first-batch training data.
    pub fn init_quantizer(&mut self, q: ScalarQuantizer) {
        self.quantizer = Some(q);
    }
}

impl NnSearch for FlatSq8 {
    fn insert(&mut self, vector: Vec<f32>) {
        let code = match &self.quantizer {
            Some(q) => q.encode8(&vector),
            None => panic!("call init_quantizer before insert"),
        };
        self.codes.push(code);
        self.originals.push(vector);
    }

    fn search(&self, query: &[f32], k: usize) -> Vec<NnResult> {
        let q_code = match &self.quantizer {
            Some(q) => q.encode8(query),
            None => return vec![],
        };
        let qr = self.quantizer.as_ref().unwrap();

        // Phase 1: quantized scan
        let mut approx: Vec<(f32, usize)> = self
            .codes
            .iter()
            .enumerate()
            .map(|(i, c)| (qr.l2_sq8(&q_code, c), i))
            .collect();
        approx.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        approx.truncate(self.ef);

        // Phase 2: exact re-rank
        let mut exact: Vec<NnResult> = approx
            .iter()
            .map(|(_, i)| NnResult {
                id: *i,
                distance: l2_sq(query, &self.originals[*i]),
            })
            .collect();
        exact.sort_by(|a, b| {
            a.distance
                .partial_cmp(&b.distance)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        exact.truncate(k);
        exact
    }

    fn len(&self) -> usize {
        self.originals.len()
    }

    fn memory_bytes(&self) -> usize {
        let orig: usize = self.originals.iter().map(|v| v.len() * 4).sum();
        let codes: usize = self.codes.iter().map(|c| c.len()).sum();
        orig + codes
    }
}
