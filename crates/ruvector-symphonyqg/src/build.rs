use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};

use crate::graph;
use crate::graph::{batch_pad, dist_cosine, dist_l2_sq, encode, SymphonyGraph};
use crate::{Config, Metric};

/// Sampled-greedy k-NN graph construction.
///
/// For each vertex, exact distances are computed to `ef_construction` randomly
/// sampled candidates, the top `m_base` are kept as neighbours, and the list is
/// then padded to the nearest multiple of BATCH_SIZE so every SIMD scan lane is
/// fully occupied (no wasted lanes).
///
/// Time:  O(n · ef_c · D)  — linear in corpus size.
/// Space: O(n · m · (4 + code_bytes))  — adjacency + inline codes.
pub fn build(vecs: &[Vec<f32>], config: &Config) -> SymphonyGraph {
    let n = vecs.len();
    assert!(n > 1, "need at least 2 vectors");
    assert!(
        vecs.iter().all(|v| v.len() == config.dim),
        "all vectors must have dim={}",
        config.dim
    );

    let dim = config.dim;
    let m_base = config.m_base;
    let m = batch_pad(m_base);
    let ef_c = config.ef_construction.min(n - 1);
    let code_bytes = dim / 8;

    let mut rng = rand::rngs::StdRng::seed_from_u64(config.seed);

    // ── Random rotation: sign flip + permutation ───────────────────────────
    // This is a diagonal-signed permutation matrix — orthogonal (O(D) cost),
    // sufficient to break axis-aligned correlations without needing SVD.
    let signs: Vec<f32> = (0..dim)
        .map(|_| if rng.gen::<bool>() { 1.0 } else { -1.0 })
        .collect();
    let mut perm: Vec<usize> = (0..dim).collect();
    perm.shuffle(&mut rng);

    // ── Flatten + pre-encode ───────────────────────────────────────────────
    let mut flat = vec![0.0f32; n * dim];
    for (i, v) in vecs.iter().enumerate() {
        flat[i * dim..(i + 1) * dim].copy_from_slice(v);
    }
    let all_codes: Vec<Vec<u8>> = vecs.iter().map(|v| encode(v, &signs, &perm)).collect();

    let dist_fn: fn(&[f32], &[f32]) -> f32 = match config.metric {
        Metric::Euclidean => dist_l2_sq,
        Metric::Cosine => dist_cosine,
    };

    // ── Build adjacency ────────────────────────────────────────────────────
    let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];

    // Candidate pool indices for reuse.
    let mut pool: Vec<usize> = Vec::with_capacity(n);

    for v in 0..n {
        let v_vec = &flat[v * dim..(v + 1) * dim];

        // Sample ef_c candidates: full shuffle then truncate for an unbiased sample.
        pool.clear();
        pool.extend((0..n).filter(|&u| u != v));
        if pool.len() > ef_c {
            pool.shuffle(&mut rng);
            pool.truncate(ef_c);
        }

        let mut dists: Vec<(f32, usize)> = pool
            .iter()
            .map(|&u| (dist_fn(v_vec, &flat[u * dim..(u + 1) * dim]), u))
            .collect();
        dists.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        // Take only real neighbours — at most `m` but typically fewer when
        // the candidate pool was small. Padding slots are filled below with
        // PADDING_SENTINEL so search.rs can skip them in O(1).
        adj[v] = dists.iter().map(|&(_, u)| u).take(m).collect();
    }

    // ── Pack into a single per-vertex contiguous block ─────────────────────
    // Layout per vertex:  [ ids: m × u32 ][ codes: m × code_bytes (as u32) ]
    // - IDs default to PADDING_SENTINEL so unused slots are skipped at search.
    // - Codes default to zero (constant Hamming distance from any query → the
    //   batch popcount produces a uniform score, then the sentinel ID skip
    //   discards it before any heap insert). This matches the SymphonyQG
    //   paper's "no spurious-edge contribution" invariant.
    // Single allocation = perfect cache co-location of a vertex's IDs + codes.
    let stride_u32 = SymphonyGraph::block_stride_u32_for(m, code_bytes);
    let mut blocks = vec![0u32; n * stride_u32];
    // Initialise ID half of every block to the sentinel.
    for v in 0..n {
        let off = v * stride_u32;
        for slot in &mut blocks[off..off + m] {
            *slot = graph::PADDING_SENTINEL;
        }
    }

    let mut self_codes = vec![0u8; n * code_bytes];

    for v in 0..n {
        self_codes[v * code_bytes..(v + 1) * code_bytes].copy_from_slice(&all_codes[v]);

        let off = v * stride_u32;
        let codes_u32_len = (m * code_bytes) / 4;

        // 1. Write IDs (first m u32s of the block).
        for (j, &nb) in adj[v].iter().enumerate() {
            blocks[off + j] = nb as u32;
        }

        // 2. Write codes (next codes_u32_len u32s, viewed as code_bytes-strided u8s).
        // Borrow only the codes section mutably so it doesn't overlap the IDs.
        let codes_u32 = &mut blocks[off + m..off + m + codes_u32_len];
        // SAFETY: we allocated `blocks` as `Vec<u32>` (4-byte aligned by the
        // global allocator), and `m * code_bytes` is a multiple of 4 because
        // `m` is a multiple of BATCH_SIZE=32. Casting `&mut [u32] → &mut [u8]`
        // is alignment-safe (u8 < u32 alignment) and length-safe (4× len).
        let codes_bytes: &mut [u8] = unsafe {
            core::slice::from_raw_parts_mut(codes_u32.as_mut_ptr() as *mut u8, codes_u32.len() * 4)
        };
        for (j, &nb) in adj[v].iter().enumerate() {
            let dst = j * code_bytes;
            codes_bytes[dst..dst + code_bytes].copy_from_slice(&all_codes[nb]);
        }
    }

    SymphonyGraph {
        n,
        dim,
        code_bytes,
        m,
        block_stride_u32: stride_u32,
        vectors: flat,
        blocks,
        self_codes,
        signs,
        perm,
        entry: 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Config;

    fn make_vecs(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
        use rand::Rng;
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        (0..n)
            .map(|_| (0..dim).map(|_| rng.gen::<f32>()).collect())
            .collect()
    }

    #[test]
    fn build_basic() {
        let cfg = Config {
            dim: 64,
            m_base: 8,
            ef_construction: 50,
            ..Config::default()
        };
        let vecs = make_vecs(200, 64, 1);
        let g = build(&vecs, &cfg);
        assert_eq!(g.n, 200);
        assert_eq!(g.dim, 64);
        assert!(g.m >= 8 && g.m % 32 == 0);
    }

    #[test]
    fn neighbors_all_valid() {
        let cfg = Config {
            dim: 64,
            m_base: 8,
            ef_construction: 50,
            ..Config::default()
        };
        let vecs = make_vecs(200, 64, 2);
        let g = build(&vecs, &cfg);
        for v in 0..g.n {
            for &nb in g.neighbors_of(v) {
                assert!((nb as usize) < g.n, "out-of-bounds neighbor {nb} for v={v}");
            }
        }
    }

    #[test]
    fn encode_deterministic() {
        let cfg = Config {
            dim: 128,
            ..Config::default()
        };
        let vecs = make_vecs(10, 128, 3);
        let g = build(&vecs, &cfg);
        let c1 = g.encode_query(&vecs[0]);
        let c2 = g.encode_query(&vecs[0]);
        assert_eq!(c1, c2);
    }
}
