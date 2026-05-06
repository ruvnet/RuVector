//! `ruvector-mincut` orchestration on the coactivation-weighted
//! connectome.

use ruvector_mincut::MinCutBuilder;

use crate::connectome::{Connectome, NeuronId};
use crate::lif::Spike;

use super::types::{class_name, AnalysisConfig, FunctionalPartition};

/// Build a functional partition using mincut weighted by recent
/// spike coactivation. Only the last `motif_window_ms` of spikes are
/// considered so the partition tracks recent dynamics.
pub fn functional_partition(
    cfg: &AnalysisConfig,
    conn: &Connectome,
    spikes: &[Spike],
) -> FunctionalPartition {
    let n = conn.num_neurons();
    let mut coact: Vec<f64> = vec![0.0; conn.num_synapses()];
    let cutoff = spikes
        .last()
        .map(|s| s.t_ms - cfg.motif_window_ms)
        .unwrap_or(0.0);
    // Build last-spike-time index.
    let mut last_spike = vec![f32::NEG_INFINITY; n];
    for s in spikes {
        if s.t_ms < cutoff {
            continue;
        }
        last_spike[s.neuron.idx()] = s.t_ms;
    }
    for pre_idx in 0..n {
        let ls_pre = last_spike[pre_idx];
        if ls_pre.is_finite() {
            let row_start = conn.row_ptr()[pre_idx] as usize;
            let row_end = conn.row_ptr()[pre_idx + 1] as usize;
            for (k, syn) in conn.synapses()[row_start..row_end].iter().enumerate() {
                let ls_post = last_spike[syn.post.idx()];
                if ls_post.is_finite() {
                    let gap = (ls_post - ls_pre).abs();
                    let w = (-gap / 20.0).exp() as f64;
                    coact[row_start + k] += w * syn.weight as f64;
                }
            }
        }
    }

    let mut ranked: Vec<(usize, f64)> = coact
        .iter()
        .copied()
        .enumerate()
        .filter(|(_, w)| *w > 0.0)
        .collect();
    ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    ranked.truncate(cfg.mincut_top_k);
    if ranked.len() < 2 {
        return FunctionalPartition {
            cut_value: 0.0,
            side_a: Vec::new(),
            side_b: Vec::new(),
            edges_considered: 0,
            side_a_class_histogram: Vec::new(),
            side_b_class_histogram: Vec::new(),
        };
    }
    let row_ptr = conn.row_ptr();
    let syn = conn.synapses();
    let mut agg: std::collections::HashMap<(u64, u64), f64> =
        std::collections::HashMap::with_capacity(ranked.len());
    for (flat, w) in ranked {
        let pre = match row_ptr.binary_search(&(flat as u32)) {
            Ok(x) => x,
            Err(x) => x.saturating_sub(1),
        };
        let post = syn[flat].post.idx();
        if pre == post {
            continue;
        }
        let u = (pre.min(post)) as u64 + 1;
        let v = (pre.max(post)) as u64 + 1;
        *agg.entry((u, v)).or_insert(0.0) += w;
    }
    let mut mc_edges: Vec<(u64, u64, f64)> = agg
        .into_iter()
        .map(|((u, v), w)| (u, v, w.clamp(cfg.min_w, cfg.max_w)))
        .collect();
    mc_edges.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));
    let edges_considered = mc_edges.len() as u64;
    if mc_edges.is_empty() {
        return FunctionalPartition {
            cut_value: 0.0,
            side_a: Vec::new(),
            side_b: Vec::new(),
            edges_considered: 0,
            side_a_class_histogram: Vec::new(),
            side_b_class_histogram: Vec::new(),
        };
    }
    let mc = MinCutBuilder::new()
        .exact()
        .with_edges(mc_edges)
        .build()
        .expect("mincut build");
    let cut_value = mc.min_cut_value();
    let result = mc.min_cut();
    let (side_a, side_b) = result
        .partition
        .map(|(a, b)| {
            (
                a.iter()
                    .map(|x| (*x as u32).saturating_sub(1))
                    .collect::<Vec<_>>(),
                b.iter()
                    .map(|x| (*x as u32).saturating_sub(1))
                    .collect::<Vec<_>>(),
            )
        })
        .unwrap_or_default();
    let side_a_hist = class_histogram(&side_a, conn);
    let side_b_hist = class_histogram(&side_b, conn);
    FunctionalPartition {
        cut_value,
        side_a,
        side_b,
        edges_considered,
        side_a_class_histogram: side_a_hist,
        side_b_class_histogram: side_b_hist,
    }
}

fn class_histogram(side: &[u32], conn: &Connectome) -> Vec<(String, u32)> {
    let mut counts = [0_u32; 15];
    for id in side {
        let m = conn.meta(NeuronId(*id));
        counts[m.class as usize] += 1;
    }
    let mut out = Vec::new();
    for (i, c) in counts.iter().enumerate() {
        if *c > 0 {
            out.push((class_name(i as u8), *c));
        }
    }
    out.sort_by_key(|entry| std::cmp::Reverse(entry.1));
    out
}
