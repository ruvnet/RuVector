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
    /// Weight multiplier for onset/transient edges.
    pub onset_weight: f64,
    /// Onset detection threshold (spectral flux ratio).
    pub onset_threshold: f64,
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
            onset_weight: 1.5,
            onset_threshold: 2.0,
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
        let base = frame * stft.num_freq_bins;
        for f1 in 0..stft.num_freq_bins {
            let n1 = match node_map[base + f1] {
                Some(id) => id,
                None => continue,
            };
            let mag1 = stft.bins[base + f1].magnitude;

            let f_end = (f1 + params.spectral_radius + 1).min(stft.num_freq_bins);
            for f2 in (f1 + 1)..f_end {
                let n2 = match node_map[base + f2] {
                    Some(id) => id,
                    None => continue,
                };
                let mag2 = stft.bins[base + f2].magnitude;
                let df = f2 - f1;

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
    //     Enhanced with instantaneous frequency (IF) consistency.
    //     IF = (phase[t+1] - phase[t]) / (2π * hop_time)
    //     Bins from the same source have consistent IF across frames.
    let hop_time = stft.hop_size as f64 / stft.sample_rate;
    for frame in 0..stft.num_frames.saturating_sub(1) {
        let base1 = frame * stft.num_freq_bins;
        let base2 = (frame + 1) * stft.num_freq_bins;
        for f in 0..stft.num_freq_bins {
            let n1 = match node_map[base1 + f] {
                Some(id) => id,
                None => continue,
            };
            let n2 = match node_map[base2 + f] {
                Some(id) => id,
                None => continue,
            };

            let bin1 = &stft.bins[base1 + f];
            let bin2 = &stft.bins[base2 + f];

            let mag_sim = (bin1.magnitude * bin2.magnitude).sqrt();
            let mut w = params.temporal_weight * mag_sim;

            if params.use_phase {
                // Phase coherence: wrapped phase difference
                let phase_diff = (bin2.phase - bin1.phase).abs();
                let wrapped = if phase_diff > PI {
                    2.0 * PI - phase_diff
                } else {
                    phase_diff
                };
                if wrapped < params.phase_threshold {
                    w *= 1.5; // Coherent phases get 50% boost
                }

                // Instantaneous frequency consistency bonus:
                // Expected phase advance for bin f = 2π * f * hop_time * sr / window_size
                // IF deviation from expected = how far the true frequency is from bin center
                let expected_phase_advance = 2.0 * PI * f as f64 * hop_time * stft.sample_rate
                    / (stft.num_freq_bins as f64 * 2.0); // num_freq_bins = window_size/2+1
                let if_deviation = {
                    let mut d = (bin2.phase - bin1.phase) - expected_phase_advance;
                    // Wrap to [-π, π]
                    d = d % (2.0 * PI);
                    if d > PI { d -= 2.0 * PI; }
                    if d < -PI { d += 2.0 * PI; }
                    d.abs()
                };
                // Small IF deviation = stable sinusoidal component → stronger edge
                if if_deviation < PI / 6.0 {
                    w *= 1.3; // Stable IF bonus
                }
            }

            if w > 1e-6 {
                let _ = graph.insert_edge(n1, n2, w);
                edge_count += 1;
            }
        }
    }

    // 2b2. Cross-frequency IF edges — connect nearby freq bins across adjacent
    //      frames when they share similar instantaneous frequency.
    //      This helps separate close tones that smear across bins.
    if params.use_phase && stft.num_frames >= 2 {
        for frame in 0..stft.num_frames.saturating_sub(1) {
            let base1 = frame * stft.num_freq_bins;
            let base2 = (frame + 1) * stft.num_freq_bins;
            for f1 in 0..stft.num_freq_bins {
                let n1 = match node_map[base1 + f1] {
                    Some(id) => id,
                    None => continue,
                };
                let mag1 = stft.bins[base1 + f1].magnitude;
                let phase1 = stft.bins[base1 + f1].phase;

                // Check nearby bins in the next frame
                let f_start = f1.saturating_sub(2);
                let f_end = (f1 + 3).min(stft.num_freq_bins);
                for f2 in f_start..f_end {
                    if f2 == f1 { continue; } // Already handled above
                    let n2 = match node_map[base2 + f2] {
                        Some(id) => id,
                        None => continue,
                    };
                    let mag2 = stft.bins[base2 + f2].magnitude;
                    let phase2 = stft.bins[base2 + f2].phase;

                    // Both bins should have similar IF (phase advance rate)
                    let if1 = (stft.bins[base2 + f1].phase - phase1) / (2.0 * PI * hop_time);
                    let if2 = (phase2 - stft.bins[base1 + f2].phase) / (2.0 * PI * hop_time);

                    // Only check if both f2 bins exist in both frames
                    if node_map[base2 + f1].is_none() || node_map[base1 + f2].is_none() {
                        continue;
                    }

                    let if_diff = (if1 - if2).abs();
                    let freq_resolution = stft.sample_rate / (stft.num_freq_bins as f64 * 2.0);

                    // If IFs are within one bin width, these bins track the same component
                    if if_diff < freq_resolution * 2.0 {
                        let w = params.temporal_weight * 0.5
                            * (mag1 * mag2).sqrt()
                            / (1.0 + (f2 as f64 - f1 as f64).abs());
                        if w > 1e-6 {
                            let _ = graph.insert_edge(n1, n2, w);
                            edge_count += 1;
                        }
                    }
                }
            }
        }
    }

    // 2c. Harmonic alignment — connect bins at integer frequency ratios
    for frame in 0..stft.num_frames {
        let base = frame * stft.num_freq_bins;
        // Precompute max f1 for each harmonic to avoid inner bound checks
        let max_f1 = if params.max_harmonics >= 2 {
            stft.num_freq_bins / 2
        } else {
            stft.num_freq_bins
        };
        for f1 in 1..max_f1 {
            let n1 = match node_map[base + f1] {
                Some(id) => id,
                None => continue,
            };
            let mag1 = stft.bins[base + f1].magnitude;

            let h_max = ((stft.num_freq_bins - 1) / f1).min(params.max_harmonics);
            for h in 2..=h_max {
                let f2 = f1 * h;
                let n2 = match node_map[base + f2] {
                    Some(id) => id,
                    None => continue,
                };
                let mag2 = stft.bins[base + f2].magnitude;

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

    // 2d. Onset/transient detection edges
    // Bins that share an onset (sudden energy increase) belong together.
    // Spectral flux = sum of positive magnitude changes across frames.
    // Bins with simultaneous onset get strong connecting edges.
    if params.onset_weight > 0.0 && stft.num_frames >= 2 {
        for frame in 1..stft.num_frames {
            let base_prev = (frame - 1) * stft.num_freq_bins;
            let base_curr = frame * stft.num_freq_bins;

            // Detect which bins have onset in this frame
            let mut onset_bins: Vec<usize> = Vec::new();
            for f in 0..stft.num_freq_bins {
                let mag_prev = stft.bins[base_prev + f].magnitude;
                let mag_curr = stft.bins[base_curr + f].magnitude;
                // Onset = significant positive magnitude change
                if mag_prev > 1e-6 && mag_curr / mag_prev > params.onset_threshold {
                    if node_map[base_curr + f].is_some() {
                        onset_bins.push(f);
                    }
                } else if mag_prev < 1e-6 && mag_curr > params.magnitude_floor * 2.0 {
                    // New energy appearing from silence
                    if node_map[base_curr + f].is_some() {
                        onset_bins.push(f);
                    }
                }
            }

            // Connect onset bins within the same frame (they likely belong to same transient)
            let max_onset_pairs = onset_bins.len().min(20); // Cap to avoid O(n^2)
            for i in 0..max_onset_pairs {
                for j in (i + 1)..max_onset_pairs {
                    let f1 = onset_bins[i];
                    let f2 = onset_bins[j];
                    let n1 = match node_map[base_curr + f1] {
                        Some(id) => id,
                        None => continue,
                    };
                    let n2 = match node_map[base_curr + f2] {
                        Some(id) => id,
                        None => continue,
                    };
                    let mag1 = stft.bins[base_curr + f1].magnitude;
                    let mag2 = stft.bins[base_curr + f2].magnitude;
                    let w = params.onset_weight * (mag1 * mag2).sqrt()
                        / (1.0 + (f2 as f64 - f1 as f64).abs() * 0.1);
                    if w > 1e-6 {
                        let _ = graph.insert_edge(n1, n2, w);
                        edge_count += 1;
                    }
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
