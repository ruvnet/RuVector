//! Partitioning equivalence for the three 1-N GEMMs (`gemm.rs`).
//!
//! The reference is the pre-tiling kernel's exact dispatch: fixed 512-entity
//! chunks for logits / grad_E and 32-row, full-width chunks for grad_Q. The
//! shape-adaptive tiling (and a set of deliberately odd tilings) must match it
//! **bitwise**, in every rayon pool size, because tiles only ever cut output
//! axes — never a reduction axis.

use super::gemm::{
    entity_tile, grad_entities_tn, grad_entities_tn_tiled, grad_queries_nn, grad_queries_nn_tiled,
    logits_nt, logits_nt_tiled, query_tiles,
};
use super::tests::rand;

/// Pre-tiling constants (the kernel as of 167c830b0).
const REF_ENTITY_CHUNK: usize = 512;
const REF_QUERY_CHUNK: usize = 32;

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

/// `[logits, grad_E (beta=0), grad_E (beta=1 onto a base), grad_Q]`.
type Outs = [Vec<u32>; 4];

struct Case {
    b: usize,
    n: usize,
    dim: usize,
    q: Vec<f32>,
    e: Vec<f32>,
    dl: Vec<f32>,
    base: Vec<f32>,
}

impl Case {
    fn new(b: usize, n: usize, dim: usize) -> Self {
        let s = (b * 131 + n * 7 + dim) as u64;
        Self {
            b,
            n,
            dim,
            q: rand(b * dim, s + 1, 0.5),
            e: rand(n * dim, s + 2, 0.5),
            dl: rand(b * n, s + 3, 0.1),
            base: rand(n * dim, s + 4, 0.2),
        }
    }

    /// Run with explicit tiles: `(entity_tile, query_row_tile, query_col_tile)`.
    fn tiled(&self, et: usize, qr: usize, qc: usize) -> Outs {
        let (b, n, dim) = (self.b, self.n, self.dim);
        let mut lg = vec![f32::NAN; b * n];
        logits_nt_tiled(&self.q, &self.e, b, n, dim, &mut lg, et);
        let mut ge0 = vec![f32::NAN; n * dim];
        grad_entities_tn_tiled(&self.dl, &self.q, b, n, dim, 0.0, &mut ge0, et);
        let mut ge1 = self.base.clone();
        grad_entities_tn_tiled(&self.dl, &self.q, b, n, dim, 1.0, &mut ge1, et);
        let mut gq = vec![f32::NAN; b * dim];
        grad_queries_nn_tiled(&self.dl, &self.e, b, n, dim, &mut gq, qr, qc);
        [bits(&lg), bits(&ge0), bits(&ge1), bits(&gq)]
    }

    fn reference(&self) -> Outs {
        self.tiled(REF_ENTITY_CHUNK, REF_QUERY_CHUNK, self.dim)
    }

    /// The production entry points (adaptive tiles for the current pool).
    fn adaptive(&self) -> Outs {
        let (b, n, dim) = (self.b, self.n, self.dim);
        let mut lg = vec![f32::NAN; b * n];
        logits_nt(&self.q, &self.e, b, n, dim, &mut lg);
        let mut ge0 = vec![f32::NAN; n * dim];
        grad_entities_tn(&self.dl, &self.q, b, n, dim, 0.0, &mut ge0);
        let mut ge1 = self.base.clone();
        grad_entities_tn(&self.dl, &self.q, b, n, dim, 1.0, &mut ge1);
        let mut gq = vec![f32::NAN; b * dim];
        grad_queries_nn(&self.dl, &self.e, b, n, dim, &mut gq);
        [bits(&lg), bits(&ge0), bits(&ge1), bits(&gq)]
    }
}

/// Shapes straddle the sgemm micro-tile (8), `mc` (64), `kc` (256), the old
/// 512-entity chunk and the 128-row query tile; B=100 is the WN18RR C1 batch.
const SHAPES: &[(usize, usize, usize)] = &[
    (1, 1, 1),
    (3, 7, 5),
    (100, 1030, 66),
    (130, 777, 300),
    (33, 2049, 19),
    (1, 513, 64),
];

fn check_all(what: &str) {
    for &(b, n, dim) in SHAPES {
        let c = Case::new(b, n, dim);
        let want = c.reference();
        let names = ["logits", "grad_E beta=0", "grad_E beta=1", "grad_Q"];
        let check = |got: Outs, how: &str| {
            for (i, name) in names.iter().enumerate() {
                assert!(
                    got[i] == want[i],
                    "{what} {how}: {name} differs from reference at B={b} N={n} D={dim}"
                );
            }
        };
        check(c.adaptive(), "adaptive");
        let (et, (qr, qc)) = (entity_tile(n), query_tiles(b, dim));
        assert!(et >= 1 && qr >= 1 && qc >= 1 && qc <= dim.max(1));
        for (et, qr, qc) in [(1, 1, 1), (7, 5, 3), (64, 128, 8), (4096, 1000, 4096)] {
            check(c.tiled(et, qr, qc), &format!("tiles ({et},{qr},{qc})"));
        }
    }
}

#[test]
fn tiled_gemms_match_fixed_chunk_reference_bitwise() {
    check_all("current pool");
}

/// The adaptive tile sizes depend on the pool's thread count; the results
/// must not.
#[cfg(feature = "parallel")]
#[test]
fn tiled_gemms_are_bitwise_identical_across_pool_sizes() {
    for t in [1usize, 3, 16] {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(t)
            .build()
            .unwrap();
        pool.install(|| check_all(&format!("threads={t}")));
        // The tiling really adapts: 16 workers split a B=100 grad_Q by
        // columns, one worker does not.
        let (_, qc) = pool.install(|| query_tiles(100, 2000));
        if t == 1 {
            assert_eq!(qc, 2000);
        } else {
            assert!(qc < 2000, "threads={t}: grad_Q not column-tiled");
        }
    }
}
