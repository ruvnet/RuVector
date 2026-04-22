//! Public value types for the analysis layer.

use serde::Serialize;

/// Parameters for the analysis layer.
#[derive(Clone, Debug)]
pub struct AnalysisConfig {
    /// Motif window width (ms). Default 100 ms (see research §05 §6).
    pub motif_window_ms: f32,
    /// Number of bins inside a motif window. Default 10 → 10 ms bin.
    pub motif_bins: usize,
    /// Embedding dimension for SDPA encoder. Default 64.
    pub embed_dim: usize,
    /// Max motifs to retain in the index. Default 256.
    pub index_capacity: usize,
    /// Mincut edge budget: keep at most `mincut_top_k` connectome
    /// edges (ranked by recent spike pair count) in the weighted cut
    /// graph.
    pub mincut_top_k: usize,
    /// Clamp weights to `[min_w, max_w]` before handing to mincut.
    pub min_w: f64,
    /// See `min_w`.
    pub max_w: f64,
    /// Deterministic projection seed.
    pub proj_seed: u64,
}

impl Default for AnalysisConfig {
    fn default() -> Self {
        Self {
            motif_window_ms: 100.0,
            motif_bins: 10,
            embed_dim: 64,
            index_capacity: 256,
            mincut_top_k: 4096,
            min_w: 0.01,
            max_w: 1_000.0,
            proj_seed: 0xB16F_ACE_C0DE_BABE,
        }
    }
}

/// Functional partition emitted by mincut.
#[derive(Clone, Debug, Serialize)]
pub struct FunctionalPartition {
    /// Global mincut value.
    pub cut_value: f64,
    /// Partition side A (neuron ids).
    pub side_a: Vec<u32>,
    /// Partition side B (neuron ids).
    pub side_b: Vec<u32>,
    /// Number of connectome edges admitted into the mincut graph.
    pub edges_considered: u64,
    /// Composition of side A by class (counts).
    pub side_a_class_histogram: Vec<(String, u32)>,
    /// Composition of side B by class (counts).
    pub side_b_class_histogram: Vec<(String, u32)>,
}

/// One repeated motif surfaced by the encoder + kNN.
#[derive(Clone, Debug, Serialize)]
pub struct MotifHit {
    /// Representative window mid-time (ms).
    pub t_ms: f32,
    /// Number of windows clustered under this motif.
    pub frequency: u32,
    /// Representative spike count in the window.
    pub spike_count: u32,
    /// Dominant participating class.
    pub dominant_class: String,
    /// L2 distance of the closest other motif (tighter = more repeated).
    pub nearest_distance: f32,
}

/// Summary of a motif-window raster for pretty-printing / JSON output.
#[derive(Clone, Debug, Serialize)]
pub struct MotifSignature {
    /// Per-class per-bin activation rates.
    pub per_class_rates: Vec<Vec<f32>>,
    /// Participating neuron count.
    pub participants: u32,
}

/// In-process motif index. Brute-force cosine + L2 distance with
/// capacity eviction.
pub struct MotifIndex {
    capacity: usize,
    pub(crate) vectors: Vec<Vec<f32>>,
    pub(crate) windows: Vec<MotifWindow>,
}

#[derive(Clone, Debug)]
pub(crate) struct MotifWindow {
    pub(crate) t_center_ms: f32,
    pub(crate) spike_count: u32,
    pub(crate) dominant_class_idx: u8,
}

impl MotifIndex {
    pub(crate) fn new(capacity: usize) -> Self {
        Self {
            capacity,
            vectors: Vec::with_capacity(capacity),
            windows: Vec::with_capacity(capacity),
        }
    }

    /// Number of indexed motifs.
    pub fn len(&self) -> usize {
        self.vectors.len()
    }

    /// Whether the index is empty.
    pub fn is_empty(&self) -> bool {
        self.vectors.is_empty()
    }

    /// Raw SDPA-embedded vectors for every indexed window, in insert
    /// order. Exposed for tests that need a ground-truth-labeled
    /// precision@k against a multi-protocol corpus (see
    /// `tests/acceptance_core.rs::ac_2_motif_emergence_labeled_corpus`).
    pub fn vectors(&self) -> &[Vec<f32>] {
        &self.vectors
    }

    /// Raster-regime signature for each indexed window, in insert
    /// order: `(dominant_class_idx, spike_count, t_center_ms)`. The
    /// metadata the SDPA encoder's embedding is actually sensitive
    /// to — unlike the stimulus-protocol labels that discovery #10
    /// and #12 showed the encoder does *not* track on this substrate.
    /// Exposed for `tests/ac_2_raster_regime_labels.rs` (ADR §17
    /// item 10 "labels" axis lever).
    pub fn window_signatures(&self) -> Vec<(u8, u32, f32)> {
        self.windows
            .iter()
            .map(|w| (w.dominant_class_idx, w.spike_count, w.t_center_ms))
            .collect()
    }

    pub(crate) fn insert(&mut self, v: Vec<f32>, w: MotifWindow) {
        if self.vectors.len() == self.capacity {
            self.vectors.remove(0);
            self.windows.remove(0);
        }
        self.vectors.push(v);
        self.windows.push(w);
    }

    pub(crate) fn top_k(&self, k: usize) -> Vec<MotifHit> {
        if self.vectors.len() < 2 {
            return Vec::new();
        }
        let mut nearest: Vec<(usize, f32)> = Vec::with_capacity(self.vectors.len());
        for i in 0..self.vectors.len() {
            let mut best = f32::INFINITY;
            let mut best_j = i;
            for j in 0..self.vectors.len() {
                if i == j {
                    continue;
                }
                let d = l2(&self.vectors[i], &self.vectors[j]);
                if d < best {
                    best = d;
                    best_j = j;
                }
            }
            nearest.push((best_j, best));
        }
        let mut idx: Vec<usize> = (0..self.vectors.len()).collect();
        idx.sort_by(|a, b| {
            nearest[*a]
                .1
                .partial_cmp(&nearest[*b].1)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let mut taken: Vec<bool> = vec![false; self.vectors.len()];
        let mut out: Vec<MotifHit> = Vec::with_capacity(k);
        for i in idx {
            if taken[i] {
                continue;
            }
            let radius = (nearest[i].1 * 0.6).max(1e-6);
            let mut freq: u32 = 1;
            for j in 0..self.vectors.len() {
                if j == i || taken[j] {
                    continue;
                }
                if l2(&self.vectors[i], &self.vectors[j]) <= radius {
                    taken[j] = true;
                    freq += 1;
                }
            }
            taken[i] = true;
            out.push(MotifHit {
                t_ms: self.windows[i].t_center_ms,
                frequency: freq,
                spike_count: self.windows[i].spike_count,
                dominant_class: class_name(self.windows[i].dominant_class_idx),
                nearest_distance: nearest[i].1,
            });
            if out.len() == k {
                break;
            }
        }
        out
    }
}

#[inline]
pub(crate) fn l2(a: &[f32], b: &[f32]) -> f32 {
    let mut s = 0.0_f32;
    for i in 0..a.len().min(b.len()) {
        let d = a[i] - b[i];
        s += d * d;
    }
    s.sqrt()
}

pub(crate) fn class_name(i: u8) -> String {
    match i {
        0 => "PhotoReceptor",
        1 => "Chemosensory",
        2 => "Mechanosensory",
        3 => "OpticLocal",
        4 => "KenyonCell",
        5 => "MbOutput",
        6 => "CentralComplex",
        7 => "LateralAccessory",
        8 => "Descending",
        9 => "Ascending",
        10 => "Motor",
        11 => "LocalInter",
        12 => "Projection",
        13 => "Modulatory",
        14 => "Other",
        _ => "Unknown",
    }
    .to_string()
}
