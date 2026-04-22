//! Spike-window motif retrieval: raster build → SDPA-backed
//! embedding → bounded in-memory kNN.

use ruvector_attention::attention::ScaledDotProductAttention;
use ruvector_attention::traits::Attention;

use crate::connectome::Connectome;
use crate::lif::Spike;

use super::types::{AnalysisConfig, MotifHit, MotifIndex, MotifWindow};

pub(crate) fn retrieve_motifs(
    cfg: &AnalysisConfig,
    sdpa: &ScaledDotProductAttention,
    w_q: &[f32],
    w_k: &[f32],
    w_v: &[f32],
    conn: &Connectome,
    spikes: &[Spike],
    k: usize,
) -> (MotifIndex, Vec<MotifHit>) {
    let mut index = MotifIndex::new(cfg.index_capacity);
    if spikes.is_empty() {
        return (index, Vec::new());
    }
    let t_end = spikes.last().map(|s| s.t_ms).unwrap_or(0.0);
    let win = cfg.motif_window_ms;
    let bins = cfg.motif_bins;
    let step = win / 2.0;
    let mut t = 0.0;
    while t + win <= t_end + step {
        let (raster, meta) = build_raster(conn, spikes, t, win, bins);
        if meta.spike_count == 0 {
            t += step;
            continue;
        }
        let q = project(&raster, w_q, cfg.embed_dim);
        let k_mat = row_major_project(&raster, w_k, cfg.embed_dim);
        let v_mat = row_major_project(&raster, w_v, cfg.embed_dim);
        let k_refs: Vec<&[f32]> = k_mat.iter().map(|r| r.as_slice()).collect();
        let v_refs: Vec<&[f32]> = v_mat.iter().map(|r| r.as_slice()).collect();
        let vec = sdpa
            .compute(&q, &k_refs, &v_refs)
            .unwrap_or_else(|_| q.clone());
        index.insert(
            vec,
            MotifWindow {
                t_center_ms: t + win * 0.5,
                spike_count: meta.spike_count,
                dominant_class_idx: meta.dominant_class_idx,
            },
        );
        t += step;
    }
    let hits = index.top_k(k);
    (index, hits)
}

struct WindowMeta {
    spike_count: u32,
    dominant_class_idx: u8,
}

fn build_raster(
    conn: &Connectome,
    spikes: &[Spike],
    t_start: f32,
    win: f32,
    bins: usize,
) -> (Vec<Vec<f32>>, WindowMeta) {
    let mut raster = vec![vec![0.0_f32; bins]; 15];
    let mut class_counts = [0_u32; 15];
    let mut total: u32 = 0;
    let bin_ms = win / bins as f32;
    for s in spikes {
        if s.t_ms < t_start {
            continue;
        }
        if s.t_ms >= t_start + win {
            break;
        }
        let bi = ((s.t_ms - t_start) / bin_ms) as usize;
        if bi >= bins {
            continue;
        }
        let cls = conn.meta(s.neuron).class as usize;
        raster[cls][bi] += 1.0;
        class_counts[cls] += 1;
        total += 1;
    }
    for row in raster.iter_mut() {
        let norm: f32 = row.iter().map(|v| v * v).sum::<f32>().sqrt();
        if norm > 0.0 {
            for v in row.iter_mut() {
                *v /= norm;
            }
        }
    }
    let mut dom_idx = 0_u8;
    let mut dom_cnt = 0_u32;
    for (i, c) in class_counts.iter().enumerate() {
        if *c > dom_cnt {
            dom_cnt = *c;
            dom_idx = i as u8;
        }
    }
    (
        raster,
        WindowMeta {
            spike_count: total,
            dominant_class_idx: dom_idx,
        },
    )
}

fn project(raster: &[Vec<f32>], w: &[f32], d: usize) -> Vec<f32> {
    let bins = raster[0].len();
    let mut flat = Vec::with_capacity(15 * bins);
    for r in raster {
        flat.extend_from_slice(r);
    }
    matvec(&flat, w, d)
}

fn row_major_project(raster: &[Vec<f32>], w: &[f32], d: usize) -> Vec<Vec<f32>> {
    let bins = raster[0].len();
    let mut out = Vec::with_capacity(bins);
    for b in 0..bins {
        let mut flat = Vec::with_capacity(15 * bins);
        for r in raster {
            let mut row = vec![0.0_f32; bins];
            row[b] = r[b];
            flat.extend_from_slice(&row);
        }
        out.push(matvec(&flat, w, d));
    }
    out
}

fn matvec(x: &[f32], w: &[f32], d: usize) -> Vec<f32> {
    let rows = x.len();
    let mut out = vec![0.0_f32; d];
    for r in 0..rows {
        let xr = x[r];
        if xr == 0.0 {
            continue;
        }
        let base = r * d;
        for c in 0..d {
            out[c] += xr * w[base + c];
        }
    }
    out
}

pub(crate) fn det_projection(rows: usize, cols: usize, seed: u64) -> Vec<f32> {
    let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    let mut out = Vec::with_capacity(rows * cols);
    for _ in 0..(rows * cols) {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let u = (state >> 32) as u32;
        let x = (u as f32 / u32::MAX as f32) * 2.0 - 1.0;
        out.push(x);
    }
    let scale = 1.0 / (rows as f32).sqrt();
    for v in out.iter_mut() {
        *v *= scale;
    }
    out
}
