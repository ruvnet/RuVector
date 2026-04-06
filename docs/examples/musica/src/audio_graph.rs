//! Audio graph construction: STFT bins -> weighted graph for mincut partitioning.
//!
//! Each time-frequency bin becomes a graph node. Edges encode similarity:
//! - Spectral proximity (nearby frequency bins in the same frame)
//! - Temporal continuity (same frequency bin across adjacent frames)
//! - Harmonic alignment (integer frequency ratios within a frame)
//! - Phase coherence (phase difference stability across frames)

use crate::stft::{StftResult, TfBin};
use ruvector_mincut::graph::DynamicGraph;
use std::f64::consts::PI;

/// Parameters controlling graph construction from STFT.
#[derive(Debug, Clone)]
pub struct GraphParams {
    /// Minimum magnitude threshold — bins below this are pruned.
    pub magnitude_floor: f64,
    /// Maximum spectral distance (in bins) for spectral edges.
    pub spectral_radius: usize,
    /// Weight multiplier for spectral proximity edges.
    pub spectral_weight: f64,
    /// Weight multiplier for temporal continuity edges.
    pub temporal_weight: f64,
    /// Weight multiplier for harmonic alignment edges.
    pub harmonic_weight: f64,
    /// Phase coherence threshold (radians) — edges below this get boosted.
    pub phase_threshold: f64,
    /// Maximum number of harmonic ratios to check.
    pub max_harmonics: usize,
    /// Whether to enable phase coherence edges.
    pub use_phase: bool,
}

impl Default for GraphParams {
    fn default() -> Self {
        Self {
            magnitude_floor: 0.01,
            spectral_radius: 3,
            spectral_weight: 1.0,
            temporal_weight: 2.0,
            harmonic_weight: 1.5,
            phase_threshold: PI / 4.0,
            max_harmonics: 4,
            use_phase: true,
        }
    }
}

/// Result of graph construction.
pub struct AudioGraph {
    /// The dynamic graph for mincut.
    pub graph: DynamicGraph,
    /// Map from node ID to TF bin info.
    pub node_bins: Vec<TfBin>,
    /// Number of frames in the STFT.
    pub num_frames: usize,
    /// Number of frequency bins per frame.
    pub num_freq_bins: usize,
    /// Total nodes (after pruning).
    pub num_nodes: usize,
    /// Total edges inserted.
    pub num_edges: usize,
    /// Node IDs indexed by (frame, freq_bin), None if pruned.
    node_map: Vec<Option<u64>>,
}

impl AudioGraph {
    /// Look up the node ID for a given (frame, freq_bin).
    pub fn node_id(&self, frame: usize, freq_bin: usize) -> Option<u64> {
        if frame < self.num_frames && freq_bin < self.num_freq_bins {
            self.node_map[frame * self.num_freq_bins + freq_bin]
        } else {
            None
        }
    }
}

/// Build a weighted graph from STFT analysis for mincut partitioning.
pub fn build_audio_graph(stft: &StftResult, params: &GraphParams) -> AudioGraph {
    let graph = DynamicGraph::new();
    let mut node_bins = Vec::new();
    let mut node_map = vec![None; stft.num_frames * stft.num_freq_bins];
    let mut node_count = 0u64;
    let mut edge_count = 0usize;

    // Phase 1: Create nodes for bins above magnitude floor
    for bin in &stft.bins {
        if bin.magnitude >= params.magnitude_floor {
            let nid = node_count;
            graph.add_vertex(nid);
            node_map[bin.frame * stft.num_freq_bins + bin.freq_bin] = Some(nid);
            node_bins.push(*bin);
            node_count += 1;
        }
    }

    // Phase 2: Add edges

    // 2a. Spectral proximity — connect nearby frequency bins in the same frame
    for frame in 0..stft.num_frames {
        for f1 in 0..stft.num_freq_bins {
            let n1 = match node_map[frame * stft.num_freq_bins + f1] {
                Some(id) => id,
                None => continue,
            };
            let mag1 = stft.bins[frame * stft.num_freq_bins + f1].magnitude;

            for df in 1..=params.spectral_radius {
                let f2 = f1 + df;
                if f2 >= stft.num_freq_bins {
                    break;
                }
                let n2 = match node_map[frame * stft.num_freq_bins + f2] {
                    Some(id) => id,
                    None => continue,
                };
                let mag2 = stft.bins[frame * stft.num_freq_bins + f2].magnitude;

                // Weight: geometric mean of magnitudes, decaying with distance
                let w = params.spectral_weight
                    * (mag1 * mag2).sqrt()
                    / (1.0 + df as f64);

                if w > 1e-6 {
                    let _ = graph.insert_edge(n1, n2, w);
                    edge_count += 1;
                }
            }
        }
    }

    // 2b. Temporal continuity — connect same freq bin across adjacent frames
    for frame in 0..stft.num_frames.saturating_sub(1) {
        for f in 0..stft.num_freq_bins {
            let n1 = match node_map[frame * stft.num_freq_bins + f] {
                Some(id) => id,
                None => continue,
            };
            let n2 = match node_map[(frame + 1) * stft.num_freq_bins + f] {
                Some(id) => id,
                None => continue,
            };

            let bin1 = &stft.bins[frame * stft.num_freq_bins + f];
            let bin2 = &stft.bins[(frame + 1) * stft.num_freq_bins + f];

            let mag_sim = (bin1.magnitude * bin2.magnitude).sqrt();
            let mut w = params.temporal_weight * mag_sim;

            // Phase coherence bonus
            if params.use_phase {
                let phase_diff = (bin2.phase - bin1.phase).abs();
                let wrapped = if phase_diff > PI {
                    2.0 * PI - phase_diff
                } else {
                    phase_diff
                };
                if wrapped < params.phase_threshold {
                    w *= 1.5; // Coherent phases get 50% boost
                }
            }

            if w > 1e-6 {
                let _ = graph.insert_edge(n1, n2, w);
                edge_count += 1;
            }
        }
    }

    // 2c. Harmonic alignment — connect bins at integer frequency ratios
    for frame in 0..stft.num_frames {
        for f1 in 1..stft.num_freq_bins {
            let n1 = match node_map[frame * stft.num_freq_bins + f1] {
                Some(id) => id,
                None => continue,
            };
            let mag1 = stft.bins[frame * stft.num_freq_bins + f1].magnitude;

            for h in 2..=params.max_harmonics {
                let f2 = f1 * h;
                if f2 >= stft.num_freq_bins {
                    break;
                }
                let n2 = match node_map[frame * stft.num_freq_bins + f2] {
                    Some(id) => id,
                    None => continue,
                };
                let mag2 = stft.bins[frame * stft.num_freq_bins + f2].magnitude;

                let w = params.harmonic_weight
                    * (mag1 * mag2).sqrt()
                    / h as f64; // Decay with harmonic number

                if w > 1e-6 {
                    let _ = graph.insert_edge(n1, n2, w);
                    edge_count += 1;
                }
            }
        }
    }

    AudioGraph {
        graph,
        node_bins,
        num_frames: stft.num_frames,
        num_freq_bins: stft.num_freq_bins,
        num_nodes: node_count as usize,
        num_edges: edge_count,
        node_map,
    }
}

/// Partition quality metrics.
#[derive(Debug, Clone)]
pub struct PartitionMetrics {
    /// Intra-partition coherence (sum of internal edge weights / total).
    pub internal_coherence: f64,
    /// Inter-partition cut weight (boundary cost).
    pub cut_weight: f64,
    /// Normalized cut (cut / min(partition_size)).
    pub normalized_cut: f64,
    /// Number of nodes per partition.
    pub partition_sizes: Vec<usize>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stft;
    use std::f64::consts::PI;

    #[test]
    fn test_build_audio_graph_basic() {
        let sr = 8000.0;
        let dur = 0.1;
        let n = (sr * dur) as usize;
        let signal: Vec<f64> = (0..n)
            .map(|i| (2.0 * PI * 440.0 * i as f64 / sr).sin())
            .collect();

        let result = stft::stft(&signal, 256, 128, sr);
        let ag = build_audio_graph(&result, &GraphParams::default());

        assert!(ag.num_nodes > 0, "Should have nodes");
        assert!(ag.num_edges > 0, "Should have edges");
        println!(
            "Audio graph: {} nodes, {} edges",
            ag.num_nodes, ag.num_edges
        );
    }

    #[test]
    fn test_graph_has_temporal_edges() {
        let sr = 8000.0;
        let n = 1024;
        let signal: Vec<f64> = (0..n)
            .map(|i| (2.0 * PI * 440.0 * i as f64 / sr).sin())
            .collect();

        let result = stft::stft(&signal, 256, 128, sr);
        let ag = build_audio_graph(&result, &GraphParams::default());

        // With a 440 Hz tone, there should be strong temporal edges
        // at the corresponding frequency bin across frames
        assert!(ag.num_frames >= 2, "Need multiple frames");
        assert!(ag.num_edges > ag.num_frames, "Should have cross-frame edges");
    }
}
