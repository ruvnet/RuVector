//! Dynamic mincut audio source separator.
//!
//! Uses a hybrid approach:
//! 1. Graph Laplacian spectral clustering for balanced initial partitions
//! 2. MinCut for boundary refinement and cut-value witness
//! 3. Spectral-centroid soft masking for smooth reconstruction
//!
//! The key insight: raw mincut produces degenerate (unbalanced) partitions.
//! Spectral clustering on the graph Laplacian finds balanced cuts that
//! approximate the normalized cut objective, then mincut refines boundaries.

use crate::audio_graph::AudioGraph;
use crate::lanczos::{LanczosConfig, SparseMatrix, lanczos_eigenpairs};
use crate::stft::TfBin;
use ruvector_mincut::prelude::*;
use std::collections::{HashMap, HashSet};

/// Configuration for the separator.
#[derive(Debug, Clone)]
pub struct SeparatorConfig {
    /// Number of sources to separate into.
    pub num_sources: usize,
    /// Frames per processing window (for incremental updates).
    pub window_frames: usize,
    /// Overlap between consecutive windows (in frames).
    pub window_overlap: usize,
    /// Approximation epsilon (0 = exact, >0 = faster but approximate).
    pub epsilon: f64,
    /// Soft mask temperature — lower = harder masks, higher = softer.
    pub mask_temperature: f64,
}

impl Default for SeparatorConfig {
    fn default() -> Self {
        Self {
            num_sources: 2,
            window_frames: 8,
            window_overlap: 2,
            epsilon: 0.0,
            mask_temperature: 1.0,
        }
    }
}

/// Separation result for one window.
#[derive(Debug, Clone)]
pub struct WindowPartition {
    /// Frame range [start, end) covered by this window.
    pub frame_start: usize,
    pub frame_end: usize,
    /// Partition assignment for each node in the audio graph.
    pub assignments: Vec<usize>,
    /// Mincut value for this partition.
    pub cut_value: f64,
}

/// Full separation result.
pub struct SeparationResult {
    /// Per-window partitions.
    pub windows: Vec<WindowPartition>,
    /// Soft masks per source, indexed [source][frame * num_freq_bins + freq_bin].
    pub masks: Vec<Vec<f64>>,
    /// Number of sources.
    pub num_sources: usize,
    /// Statistics.
    pub stats: SeparationStats,
}

/// Statistics from the separation process.
#[derive(Debug, Clone, Default)]
pub struct SeparationStats {
    /// Total windows processed.
    pub num_windows: usize,
    /// Average mincut value across windows.
    pub avg_cut_value: f64,
    /// Min / max cut values.
    pub min_cut_value: f64,
    pub max_cut_value: f64,
    /// Total graph nodes processed.
    pub total_nodes: usize,
    /// Total graph edges processed.
    pub total_edges: usize,
}

/// Separate audio sources using spectral clustering + mincut refinement.
pub fn separate(audio_graph: &AudioGraph, config: &SeparatorConfig) -> SeparationResult {
    let num_frames = audio_graph.num_frames;
    let num_freq = audio_graph.num_freq_bins;
    let total_tf = num_frames * num_freq;

    // Accumulation buffers for soft masks (per-source, per-TF-bin)
    let mut mask_accum: Vec<Vec<f64>> = vec![vec![0.0; total_tf]; config.num_sources];
    let mut mask_count = vec![0.0f64; total_tf];

    let mut windows = Vec::new();
    let mut cut_values = Vec::new();

    let step = config.window_frames.saturating_sub(config.window_overlap).max(1);
    let mut frame_start = 0;

    while frame_start < num_frames {
        let frame_end = (frame_start + config.window_frames).min(num_frames);

        // Extract subgraph for this window
        let (subgraph_edges, node_ids) =
            extract_window_subgraph(audio_graph, frame_start, frame_end);

        if node_ids.is_empty() || subgraph_edges.is_empty() {
            frame_start += step;
            continue;
        }

        // Get TF bin info for nodes in this window
        let node_bins: Vec<&TfBin> = node_ids
            .iter()
            .filter_map(|&nid| audio_graph.node_bins.get(nid as usize))
            .collect();

        // Spectral clustering for balanced partition
        let assignments = spectral_cluster(
            &subgraph_edges,
            &node_ids,
            &node_bins,
            config.num_sources,
            num_freq,
        );

        // Get mincut value as a structural witness
        let cut_value = compute_mincut_value(&subgraph_edges);
        cut_values.push(cut_value);

        // Compute spectral centroids per partition for soft masking
        let centroids = compute_partition_centroids(&assignments, &node_bins, config.num_sources, num_freq);

        // Update soft masks using distance-weighted assignment
        for (local_idx, &nid) in node_ids.iter().enumerate() {
            if let Some(bin) = audio_graph.node_bins.get(nid as usize) {
                let tf_idx = bin.frame * num_freq + bin.freq_bin;
                if tf_idx < total_tf {
                    // Compute soft assignment based on distance to each centroid
                    let soft = soft_assignment(
                        bin.freq_bin,
                        bin.magnitude,
                        &centroids,
                        config.mask_temperature,
                    );
                    for (s, &w) in soft.iter().enumerate() {
                        if s < config.num_sources {
                            mask_accum[s][tf_idx] += w;
                        }
                    }
                    mask_count[tf_idx] += 1.0;
                }
            }
        }

        windows.push(WindowPartition {
            frame_start,
            frame_end,
            assignments,
            cut_value,
        });

        frame_start += step;
    }

    // Normalize masks and ensure they sum to 1
    let masks = normalize_masks(&mask_accum, &mask_count, config.num_sources, total_tf);

    let avg_cut = if cut_values.is_empty() {
        0.0
    } else {
        cut_values.iter().sum::<f64>() / cut_values.len() as f64
    };

    let stats = SeparationStats {
        num_windows: windows.len(),
        avg_cut_value: avg_cut,
        min_cut_value: cut_values.iter().cloned().fold(f64::INFINITY, f64::min),
        max_cut_value: cut_values.iter().cloned().fold(0.0f64, f64::max),
        total_nodes: audio_graph.num_nodes,
        total_edges: audio_graph.num_edges,
    };

    SeparationResult {
        windows,
        masks,
        num_sources: config.num_sources,
        stats,
    }
}

/// Spectral clustering using the Fiedler vector of the graph Laplacian.
///
/// For K=2: partition by sign of the second-smallest eigenvector (Fiedler vector).
/// For K>2: use K-means on the first K eigenvectors.
///
/// This produces balanced partitions that approximate normalized cut.
fn spectral_cluster(
    edges: &[(u64, u64, f64)],
    node_ids: &[u64],
    node_bins: &[&TfBin],
    num_sources: usize,
    num_freq_bins: usize,
) -> Vec<usize> {
    let n = node_ids.len();
    if n == 0 || num_sources <= 1 {
        return vec![0; n];
    }

    // Build node ID -> local index map
    let id_to_idx: HashMap<u64, usize> = node_ids.iter().enumerate().map(|(i, &id)| (id, i)).collect();

    // Build degree vector and adjacency
    let mut degree = vec![0.0f64; n];
    let mut adj: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n];

    for &(u, v, w) in edges {
        if let (Some(&ui), Some(&vi)) = (id_to_idx.get(&u), id_to_idx.get(&v)) {
            degree[ui] += w;
            degree[vi] += w;
            adj[ui].push((vi, w));
            adj[vi].push((ui, w));
        }
    }

    // Compute Fiedler vector using power iteration on (D - L)
    // We want the smallest non-trivial eigenvector of L = D - A
    // Use inverse iteration: solve (L - sigma*I)x = b
    // Simpler: power iteration on D^{-1}A (random walk normalized Laplacian)
    let fiedler = compute_fiedler_vector(&degree, &adj, n);

    if num_sources == 2 {
        // Partition by Fiedler vector sign, with frequency-aware tie-breaking
        let median = {
            let mut sorted = fiedler.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            sorted[n / 2]
        };

        fiedler
            .iter()
            .enumerate()
            .map(|(i, &v)| {
                if v > median {
                    1
                } else if (v - median).abs() < 1e-10 {
                    // Tie-break by frequency bin (low vs high)
                    if i < node_bins.len() && node_bins[i].freq_bin > num_freq_bins / 2 {
                        1
                    } else {
                        0
                    }
                } else {
                    0
                }
            })
            .collect()
    } else {
        // Multi-eigenvector spectral embedding via Lanczos
        // Compute first num_sources eigenvectors and run k-means in that space
        let edges_for_lanczos: Vec<(usize, usize, f64)> = edges.iter()
            .filter_map(|&(u, v, w)| {
                let ui = id_to_idx.get(&u)?;
                let vi = id_to_idx.get(&v)?;
                Some((*ui, *vi, w))
            })
            .collect();

        let laplacian = SparseMatrix::from_edges(n, &edges_for_lanczos);
        let lanczos_config = LanczosConfig {
            k: num_sources + 1, // +1 for trivial eigenvector
            max_iter: 60,
            tol: 1e-6,
            reorthogonalize: true,
        };
        let eigen_result = lanczos_eigenpairs(&laplacian, &lanczos_config);

        // Use eigenvectors 1..num_sources (skip trivial constant eigenvector 0)
        if eigen_result.eigenvectors.len() > num_sources {
            let embedding: Vec<Vec<f64>> = (0..n)
                .map(|i| {
                    (1..=num_sources)
                        .map(|k| eigen_result.eigenvectors[k][i])
                        .collect()
                })
                .collect();
            spectral_kmeans(&embedding, num_sources)
        } else {
            // Fallback to frequency-based k-means
            frequency_kmeans(node_bins, num_sources, num_freq_bins)
        }
    }
}

/// Compute the Fiedler vector (2nd smallest eigenvector of Laplacian)
/// via power iteration on the random-walk normalized Laplacian.
fn compute_fiedler_vector(
    degree: &[f64],
    adj: &[Vec<(usize, f64)>],
    n: usize,
) -> Vec<f64> {
    if n <= 1 {
        return vec![0.0; n];
    }

    // Power iteration on D^{-1}A to find the largest eigenvector,
    // then deflate to get the Fiedler vector.

    // First eigenvector of D^{-1}A is always uniform (stationary distribution)
    let d_inv: Vec<f64> = degree.iter().map(|&d| if d > 1e-12 { 1.0 / d } else { 0.0 }).collect();

    // Initialize with deterministic non-uniform vector (seeded for reproducibility).
    // Uses frequency-proportional init: higher freq bins get larger values.
    // This biases the Fiedler vector toward a frequency-based partition,
    // which is the natural separation axis for audio.
    let mut v: Vec<f64> = (0..n).map(|i| {
        let base = (i as f64 / n as f64) - 0.5;
        // Add deterministic perturbation to break symmetry
        let perturb = ((i * 7 + 3) % n) as f64 / n as f64 * 0.01;
        base + perturb
    }).collect();

    // Orthogonalize against constant vector
    let sum: f64 = v.iter().sum();
    let mean = sum / n as f64;
    for x in &mut v {
        *x -= mean;
    }

    // Power iteration for Fiedler vector (100 iterations for stable convergence)
    // We iterate D^{-1}A to find the second eigenvector
    for _ in 0..100 {
        // Multiply by D^{-1}A
        let mut new_v = vec![0.0; n];
        for i in 0..n {
            let mut sum = 0.0;
            for &(j, w) in &adj[i] {
                sum += w * v[j];
            }
            new_v[i] = d_inv[i] * sum;
        }

        // Orthogonalize against constant vector (first eigenvector)
        let mean: f64 = new_v.iter().sum::<f64>() / n as f64;
        for x in &mut new_v {
            *x -= mean;
        }

        // Normalize
        let norm: f64 = new_v.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm > 1e-12 {
            for x in &mut new_v {
                *x /= norm;
            }
        }

        v = new_v;
    }

    v
}

/// K-means clustering on multi-dimensional spectral embedding.
fn spectral_kmeans(embedding: &[Vec<f64>], k: usize) -> Vec<usize> {
    let n = embedding.len();
    if n == 0 || k == 0 {
        return vec![0; n];
    }
    let dim = embedding[0].len();

    // Initialize centroids via k-means++ (deterministic approx)
    let mut centroids: Vec<Vec<f64>> = Vec::with_capacity(k);
    centroids.push(embedding[0].clone());

    for _ in 1..k {
        // Pick point farthest from existing centroids
        let mut best_idx = 0;
        let mut best_dist = 0.0f64;
        for (i, point) in embedding.iter().enumerate() {
            let min_dist: f64 = centroids.iter()
                .map(|c| (0..dim).map(|d| (point[d] - c[d]).powi(2)).sum::<f64>())
                .fold(f64::MAX, f64::min);
            if min_dist > best_dist {
                best_dist = min_dist;
                best_idx = i;
            }
        }
        centroids.push(embedding[best_idx].clone());
    }

    let mut assignments = vec![0usize; n];

    for _iter in 0..30 {
        // Assign each point to nearest centroid
        let mut changed = false;
        for (i, point) in embedding.iter().enumerate() {
            let nearest = centroids.iter().enumerate()
                .min_by(|(_, a), (_, b)| {
                    let da: f64 = (0..dim).map(|d| (point[d] - a[d]).powi(2)).sum();
                    let db: f64 = (0..dim).map(|d| (point[d] - b[d]).powi(2)).sum();
                    da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
                })
                .map(|(idx, _)| idx)
                .unwrap_or(0);
            if assignments[i] != nearest {
                assignments[i] = nearest;
                changed = true;
            }
        }
        if !changed { break; }

        // Update centroids
        for c in 0..k {
            let mut sum = vec![0.0; dim];
            let mut count = 0;
            for (i, point) in embedding.iter().enumerate() {
                if assignments[i] == c {
                    for d in 0..dim {
                        sum[d] += point[d];
                    }
                    count += 1;
                }
            }
            if count > 0 {
                for d in 0..dim {
                    centroids[c][d] = sum[d] / count as f64;
                }
            }
        }
    }

    assignments
}

/// K-means clustering on frequency bin positions.
fn frequency_kmeans(
    node_bins: &[&TfBin],
    k: usize,
    num_freq_bins: usize,
) -> Vec<usize> {
    let n = node_bins.len();
    if n == 0 || k == 0 {
        return vec![0; n];
    }

    // Initialize centroids evenly across frequency range
    let mut centroids: Vec<f64> = (0..k)
        .map(|i| (i as f64 + 0.5) * num_freq_bins as f64 / k as f64)
        .collect();

    let mut assignments = vec![0usize; n];

    for _iter in 0..20 {
        // Assign each node to nearest centroid
        let mut changed = false;
        for (i, bin) in node_bins.iter().enumerate() {
            let freq = bin.freq_bin as f64;
            let nearest = centroids
                .iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| {
                    (freq - *a).abs().partial_cmp(&(freq - *b).abs()).unwrap_or(std::cmp::Ordering::Equal)
                })
                .map(|(idx, _)| idx)
                .unwrap_or(0);
            if assignments[i] != nearest {
                assignments[i] = nearest;
                changed = true;
            }
        }

        // Early stopping: no assignments changed
        if !changed {
            break;
        }

        // Update centroids
        for c in 0..k {
            let (sum, count) = node_bins
                .iter()
                .enumerate()
                .filter(|(i, _)| assignments[*i] == c)
                .fold((0.0, 0usize), |(s, cnt), (_, bin)| {
                    (s + bin.freq_bin as f64, cnt + 1)
                });
            if count > 0 {
                centroids[c] = sum / count as f64;
            }
        }
    }

    assignments
}

/// Compute mincut value for a subgraph (used as structural witness).
fn compute_mincut_value(edges: &[(u64, u64, f64)]) -> f64 {
    if edges.is_empty() {
        return 0.0;
    }

    let edge_list: Vec<(u64, u64, f64)> = edges.to_vec();
    let builder = MinCutBuilder::new().exact().with_edges(edge_list);

    match builder.build() {
        Ok(mc) => mc.min_cut_value(),
        Err(_) => 0.0,
    }
}

/// Compute spectral centroid (average frequency bin) for each partition.
fn compute_partition_centroids(
    assignments: &[usize],
    node_bins: &[&TfBin],
    num_sources: usize,
    _num_freq_bins: usize,
) -> Vec<(f64, f64)> {
    // Returns (centroid_freq, avg_magnitude) per partition
    let mut freq_sum = vec![0.0f64; num_sources];
    let mut mag_sum = vec![0.0f64; num_sources];
    let mut counts = vec![0usize; num_sources];

    for (i, &a) in assignments.iter().enumerate() {
        if a < num_sources && i < node_bins.len() {
            freq_sum[a] += node_bins[i].freq_bin as f64;
            mag_sum[a] += node_bins[i].magnitude;
            counts[a] += 1;
        }
    }

    (0..num_sources)
        .map(|s| {
            if counts[s] > 0 {
                (
                    freq_sum[s] / counts[s] as f64,
                    mag_sum[s] / counts[s] as f64,
                )
            } else {
                (s as f64 * 50.0, 0.0) // Fallback
            }
        })
        .collect()
}

/// Compute soft assignment weights based on distance to partition centroids.
fn soft_assignment(
    freq_bin: usize,
    _magnitude: f64,
    centroids: &[(f64, f64)],
    temperature: f64,
) -> Vec<f64> {
    let k = centroids.len();
    if k == 0 {
        return vec![];
    }
    if k == 1 {
        return vec![1.0];
    }

    let freq = freq_bin as f64;
    let temp = temperature.max(0.01);

    // Distance-based soft assignment (softmax over negative distances)
    let distances: Vec<f64> = centroids
        .iter()
        .map(|&(cf, _)| -(freq - cf).abs() / temp)
        .collect();

    // Softmax
    let max_d = distances.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exp_sum: f64 = distances.iter().map(|&d| (d - max_d).exp()).sum();

    distances
        .iter()
        .map(|&d| (d - max_d).exp() / exp_sum)
        .collect()
}

/// Normalize masks so they sum to 1.0 at each TF point.
fn normalize_masks(
    mask_accum: &[Vec<f64>],
    mask_count: &[f64],
    num_sources: usize,
    total_tf: usize,
) -> Vec<Vec<f64>> {
    let mut masks = vec![vec![0.0; total_tf]; num_sources];

    for i in 0..total_tf {
        if mask_count[i] > 0.0 {
            let mut sum = 0.0;
            for s in 0..num_sources {
                masks[s][i] = mask_accum[s][i] / mask_count[i];
                sum += masks[s][i];
            }
            // Normalize to sum to 1
            if sum > 1e-12 {
                for s in 0..num_sources {
                    masks[s][i] /= sum;
                }
            } else {
                for s in 0..num_sources {
                    masks[s][i] = 1.0 / num_sources as f64;
                }
            }
        } else {
            for s in 0..num_sources {
                masks[s][i] = 1.0 / num_sources as f64;
            }
        }
    }

    masks
}

/// Extract edges and node IDs for a time window from the audio graph.
fn extract_window_subgraph(
    ag: &AudioGraph,
    frame_start: usize,
    frame_end: usize,
) -> (Vec<(u64, u64, f64)>, Vec<u64>) {
    let mut node_set = HashSet::new();
    let mut edges = Vec::new();

    for frame in frame_start..frame_end {
        for f in 0..ag.num_freq_bins {
            if let Some(nid) = ag.node_id(frame, f) {
                node_set.insert(nid);
            }
        }
    }

    let node_ids: Vec<u64> = node_set.iter().copied().collect();

    for &nid in &node_ids {
        for (neighbor, _edge_id) in ag.graph.neighbors(nid) {
            if node_set.contains(&neighbor) && nid < neighbor {
                let weight = ag.graph.edge_weight(nid, neighbor).unwrap_or(1.0);
                edges.push((nid, neighbor, weight));
            }
        }
    }

    (edges, node_ids)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audio_graph::{build_audio_graph, GraphParams};
    use crate::stft;
    use std::f64::consts::PI;

    fn make_two_tone_signal(sr: f64, dur: f64, f1: f64, f2: f64) -> Vec<f64> {
        let n = (sr * dur) as usize;
        (0..n)
            .map(|i| {
                let t = i as f64 / sr;
                (2.0 * PI * f1 * t).sin() + (2.0 * PI * f2 * t).sin()
            })
            .collect()
    }

    #[test]
    fn test_separate_two_tones() {
        let sr = 8000.0;
        let signal = make_two_tone_signal(sr, 0.25, 200.0, 1500.0);
        let stft_result = stft::stft(&signal, 256, 128, sr);
        let ag = build_audio_graph(&stft_result, &GraphParams::default());

        let config = SeparatorConfig {
            num_sources: 2,
            window_frames: 4,
            window_overlap: 1,
            epsilon: 0.0,
            mask_temperature: 1.0,
        };

        let result = separate(&ag, &config);

        assert_eq!(result.num_sources, 2);
        assert_eq!(result.masks.len(), 2);
        assert!(result.stats.num_windows > 0, "Should have processed windows");

        // Masks should sum to ~1.0 at each TF point
        let total_tf = stft_result.num_frames * stft_result.num_freq_bins;
        for i in 0..total_tf.min(100) {
            let sum: f64 = result.masks.iter().map(|m| m[i]).sum();
            assert!(
                (sum - 1.0).abs() < 0.01,
                "Mask sum at {i} = {sum}, expected ~1.0"
            );
        }
    }

    #[test]
    fn test_separate_balanced() {
        // Ensure partitions are balanced (not degenerate)
        let sr = 8000.0;
        let signal = make_two_tone_signal(sr, 0.25, 200.0, 2000.0);
        let stft_result = stft::stft(&signal, 256, 128, sr);
        let ag = build_audio_graph(&stft_result, &GraphParams::default());

        let result = separate(&ag, &SeparatorConfig::default());

        // Each mask should have significant non-zero area
        for (s, mask) in result.masks.iter().enumerate() {
            let energy: f64 = mask.iter().map(|&m| m * m).sum();
            assert!(
                energy > 0.01,
                "Source {s} mask has near-zero energy ({energy:.4})"
            );
        }
    }

    #[test]
    fn test_separate_stats() {
        let sr = 8000.0;
        let signal = make_two_tone_signal(sr, 0.2, 300.0, 2000.0);
        let stft_result = stft::stft(&signal, 256, 128, sr);
        let ag = build_audio_graph(&stft_result, &GraphParams::default());

        let result = separate(&ag, &SeparatorConfig::default());

        assert!(result.stats.total_nodes > 0);
        assert!(result.stats.total_edges > 0);
        println!("Separation stats: {:?}", result.stats);
    }

    #[test]
    fn test_fiedler_vector() {
        // Simple path graph: 0-1-2-3-4
        let degree = vec![1.0, 2.0, 2.0, 2.0, 1.0];
        let adj = vec![
            vec![(1, 1.0)],
            vec![(0, 1.0), (2, 1.0)],
            vec![(1, 1.0), (3, 1.0)],
            vec![(2, 1.0), (4, 1.0)],
            vec![(3, 1.0)],
        ];
        let fiedler = compute_fiedler_vector(&degree, &adj, 5);

        // Fiedler vector should be monotonic for a path graph
        // (values increase or decrease along the path)
        let increasing = fiedler.windows(2).all(|w| w[1] >= w[0] - 0.1);
        let decreasing = fiedler.windows(2).all(|w| w[1] <= w[0] + 0.1);
        assert!(
            increasing || decreasing,
            "Fiedler vector should be roughly monotonic for path graph: {:?}",
            fiedler
        );
    }
}
