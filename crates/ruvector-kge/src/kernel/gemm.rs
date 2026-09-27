//! The three SGEMMs of a 1-N step, on `matrixmultiply::sgemm` (plan M2).
//!
//! ```text
//! logits  [B×N]   = Q  [B×D] · Eᵀ          (forward;   reduces over D)
//! grad_E  [N×D]   = dLᵀ[N×B] · Q  [B×D]    (backward;  reduces over B)
//! grad_Q  [B×D]   = dL [B×N] · E  [N×D]    (backward;  reduces over N)
//! ```
//!
//! Transposes are pure stride swaps — nothing is copied.
//!
//! **Partitioning.** Work is split only along the two *output* axes of each
//! product, never along its reduction axis:
//!
//! - `logits` and `grad_E` are tiled along the entity axis: each task owns a
//!   block of entities (logits columns / grad_E rows) and writes it alone —
//!   no reduction across tasks, no atomics.
//! - `grad_Q` reduces over the entity axis, so splitting entities would need a
//!   cross-task sum. It is instead tiled in 2-D over (query rows × embedding
//!   columns). With small batches (WN18RR C1: B = 100) the old query-row-only
//!   split gave 4 tasks for a third of the step's flops; column tiles give
//!   every worker a share.
//!
//! **Determinism.** `sgemm` produces every output element with the same
//! fixed-order reduction (fixed `kc` blocking along the reduction axis, one
//! runtime-selected micro-kernel; masked edge tiles run the same kernel into a
//! scratch tile) no matter how many rows/columns the call covers. Since tiles
//! only ever cut the output axes, tile sizes may adapt to the shape and the
//! current thread count and every output element is still computed by exactly
//! one call in exactly one order: results are bitwise identical for any tile
//! shape and any thread count, with or without the `parallel` feature
//! (asserted by `tests_gemm.rs` against the fixed-chunk reference and by the
//! kernel tests across pools of 1/3/16 threads). `matrixmultiply`'s own
//! `threading` feature is not enabled.

use matrixmultiply::sgemm;

/// Upper bound of an entity tile (logits columns, grad_E rows).
const ENTITY_TILE: usize = 512;
/// Smallest entity tile (below this the per-call packing dominates).
const ENTITY_TILE_MIN: usize = 64;
/// Query rows per grad_Q tile.
const QUERY_TILE: usize = 128;
/// grad_Q column tiles are multiples of this (sgemm micro-tile width).
const COL_ALIGN: usize = 8;
/// Target tasks per worker thread (load balance vs per-call packing).
const TASKS_PER_THREAD: usize = 2;

/// A raw output pointer shared across tile workers. Each worker writes a
/// disjoint set of elements, so there is no data race.
#[derive(Clone, Copy)]
struct OutPtr(*mut f32);
// SAFETY: workers write disjoint element sets (disjoint tiles) and the owning
// `&mut [f32]` outlives every worker (the dispatch below is scoped).
unsafe impl Send for OutPtr {}
unsafe impl Sync for OutPtr {}

/// Worker threads available to the current dispatch (1 without `parallel`).
fn threads() -> usize {
    #[cfg(feature = "parallel")]
    {
        rayon::current_num_threads().max(1)
    }
    #[cfg(not(feature = "parallel"))]
    {
        1
    }
}

/// Run `f(start, end)` over `[0, total)` in chunks of `chunk` — in parallel
/// under the `parallel` feature (inside whatever rayon pool is current),
/// sequentially otherwise.
pub(crate) fn for_each_chunk<F>(total: usize, chunk: usize, f: F)
where
    F: Fn(usize, usize) + Send + Sync,
{
    for_each_tile(1, 1, total, chunk, move |_, _, c0, c1| f(c0, c1));
}

/// Run `f(r0, r1, c0, c1)` over the `row_tile × col_tile` tiles covering
/// `[0, rows) × [0, cols)`, in parallel under the `parallel` feature.
fn for_each_tile<F>(rows: usize, row_tile: usize, cols: usize, col_tile: usize, f: F)
where
    F: Fn(usize, usize, usize, usize) + Send + Sync,
{
    let (nr, nc) = (rows.div_ceil(row_tile), cols.div_ceil(col_tile));
    let tile = move |t: usize| {
        let (r, c) = (t / nc, t % nc);
        f(
            r * row_tile,
            ((r + 1) * row_tile).min(rows),
            c * col_tile,
            ((c + 1) * col_tile).min(cols),
        )
    };
    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        (0..nr * nc).into_par_iter().for_each(tile);
    }
    #[cfg(not(feature = "parallel"))]
    (0..nr * nc).for_each(tile);
}

/// Entity tile for `n` entities: at most [`ENTITY_TILE`], smaller when that
/// would leave workers idle. (Output-axis only — numerically irrelevant.)
pub(crate) fn entity_tile(n: usize) -> usize {
    let want = threads() * TASKS_PER_THREAD;
    n.div_ceil(want)
        .next_multiple_of(ENTITY_TILE_MIN)
        .clamp(ENTITY_TILE_MIN, ENTITY_TILE)
}

/// `(row_tile, col_tile)` for the `b × dim` grad_Q output: [`QUERY_TILE`]
/// rows, and enough column tiles for ~[`TASKS_PER_THREAD`] tasks per worker.
pub(crate) fn query_tiles(b: usize, dim: usize) -> (usize, usize) {
    let t = threads();
    if t == 1 || dim == 0 {
        return (QUERY_TILE, dim.max(1));
    }
    let col_tiles = (t * TASKS_PER_THREAD).div_ceil(b.div_ceil(QUERY_TILE).max(1));
    let cols = dim.div_ceil(col_tiles).next_multiple_of(COL_ALIGN);
    (QUERY_TILE, cols.min(dim))
}

/// `logits[b×n] = q[b×dim] · e[n×dim]ᵀ`, all row-major.
pub(crate) fn logits_nt(q: &[f32], e: &[f32], b: usize, n: usize, dim: usize, out: &mut [f32]) {
    logits_nt_tiled(q, e, b, n, dim, out, entity_tile(n));
}

pub(crate) fn logits_nt_tiled(
    q: &[f32],
    e: &[f32],
    b: usize,
    n: usize,
    dim: usize,
    out: &mut [f32],
    tile: usize,
) {
    assert!(q.len() >= b * dim && e.len() >= n * dim && out.len() >= b * n);
    if b == 0 || n == 0 {
        return;
    }
    let (qp, ep, cp) = (
        q.as_ptr() as usize,
        e.as_ptr() as usize,
        OutPtr(out.as_mut_ptr()),
    );
    for_each_chunk(n, tile, move |j0, j1| {
        let cp = cp;
        // SAFETY: bounds asserted above. A = q (b×dim, rs=dim, cs=1);
        // B = e[j0..j1]ᵀ (dim×(j1-j0), rs=1, cs=dim); C = out[:, j0..j1]
        // (rs=n, cs=1). Tiles write disjoint column ranges of `out`.
        unsafe {
            sgemm(
                b,
                dim,
                j1 - j0,
                1.0,
                qp as *const f32,
                dim as isize,
                1,
                (ep as *const f32).add(j0 * dim),
                1,
                dim as isize,
                0.0,
                cp.0.add(j0),
                n as isize,
                1,
            );
        }
    });
}

/// `out[n×dim] = dl[b×n]ᵀ · q[b×dim]` — the dense entity-table gradient.
/// `beta = 0` overwrites `out`; `beta = 1` accumulates into it.
#[allow(clippy::too_many_arguments)]
pub(crate) fn grad_entities_tn(
    dl: &[f32],
    q: &[f32],
    b: usize,
    n: usize,
    dim: usize,
    beta: f32,
    out: &mut [f32],
) {
    grad_entities_tn_tiled(dl, q, b, n, dim, beta, out, entity_tile(n));
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn grad_entities_tn_tiled(
    dl: &[f32],
    q: &[f32],
    b: usize,
    n: usize,
    dim: usize,
    beta: f32,
    out: &mut [f32],
    tile: usize,
) {
    assert!(dl.len() >= b * n && q.len() >= b * dim && out.len() >= n * dim);
    if n == 0 {
        return;
    }
    if b == 0 {
        if beta == 0.0 {
            out[..n * dim].fill(0.0);
        }
        return;
    }
    let (dp, qp, cp) = (
        dl.as_ptr() as usize,
        q.as_ptr() as usize,
        OutPtr(out.as_mut_ptr()),
    );
    for_each_chunk(n, tile, move |i0, i1| {
        let cp = cp;
        // SAFETY: A = dl[:, i0..i1]ᵀ ((i1-i0)×b, rs=1, cs=n); B = q (b×dim,
        // rs=dim, cs=1); C = out[i0..i1] ((i1-i0)×dim, rs=dim, cs=1).
        // Tiles write disjoint row ranges of `out`.
        unsafe {
            sgemm(
                i1 - i0,
                b,
                dim,
                1.0,
                (dp as *const f32).add(i0),
                1,
                n as isize,
                qp as *const f32,
                dim as isize,
                1,
                beta,
                cp.0.add(i0 * dim),
                dim as isize,
                1,
            );
        }
    });
}

/// `out[b×dim] = dl[b×n] · e[n×dim]` — the query gradient.
pub(crate) fn grad_queries_nn(
    dl: &[f32],
    e: &[f32],
    b: usize,
    n: usize,
    dim: usize,
    out: &mut [f32],
) {
    let (rt, ct) = query_tiles(b, dim);
    grad_queries_nn_tiled(dl, e, b, n, dim, out, rt, ct);
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn grad_queries_nn_tiled(
    dl: &[f32],
    e: &[f32],
    b: usize,
    n: usize,
    dim: usize,
    out: &mut [f32],
    row_tile: usize,
    col_tile: usize,
) {
    assert!(dl.len() >= b * n && e.len() >= n * dim && out.len() >= b * dim);
    if b == 0 || dim == 0 {
        return;
    }
    if n == 0 {
        out[..b * dim].fill(0.0);
        return;
    }
    let (dp, ep, cp) = (
        dl.as_ptr() as usize,
        e.as_ptr() as usize,
        OutPtr(out.as_mut_ptr()),
    );
    for_each_tile(b, row_tile, dim, col_tile, move |i0, i1, c0, c1| {
        let cp = cp;
        // SAFETY: A = dl[i0..i1] ((i1-i0)×n, rs=n, cs=1); B = e[:, c0..c1]
        // (n×(c1-c0), rs=dim, cs=1); C = out[i0..i1, c0..c1] (rs=dim, cs=1).
        // Tiles write disjoint (row, column) blocks of `out`; the reduction
        // over n runs whole inside one call.
        unsafe {
            sgemm(
                i1 - i0,
                n,
                c1 - c0,
                1.0,
                (dp as *const f32).add(i0 * n),
                n as isize,
                1,
                (ep as *const f32).add(c0),
                dim as isize,
                1,
                0.0,
                cp.0.add(i0 * dim + c0),
                dim as isize,
                1,
            );
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rand(len: usize, seed: u64) -> Vec<f32> {
        let mut s = seed | 1;
        (0..len)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                ((s >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0) as f32
            })
            .collect()
    }

    fn close(a: &[f32], b: &[f64]) {
        for (i, (&x, &y)) in a.iter().zip(b).enumerate() {
            assert!(
                (x as f64 - y).abs() <= 1e-5 * y.abs().max(1.0),
                "elem {i}: {x} vs {y}"
            );
        }
    }

    /// Shapes straddle the chunk sizes so edge chunks and masked tiles run.
    #[test]
    fn three_gemms_match_naive() {
        for &(b, n, dim) in &[(3usize, 7usize, 5usize), (33, 1030, 19), (1, 513, 64)] {
            let q = rand(b * dim, 1 + b as u64);
            let e = rand(n * dim, 2 + n as u64);
            let dl = rand(b * n, 3 + dim as u64);

            let mut lg = vec![0.0; b * n];
            logits_nt(&q, &e, b, n, dim, &mut lg);
            let mut want = vec![0.0f64; b * n];
            for i in 0..b {
                for j in 0..n {
                    want[i * n + j] = (0..dim)
                        .map(|k| q[i * dim + k] as f64 * e[j * dim + k] as f64)
                        .sum();
                }
            }
            close(&lg, &want);

            let mut ge = vec![f32::NAN; n * dim];
            grad_entities_tn(&dl, &q, b, n, dim, 0.0, &mut ge);
            let mut want = vec![0.0f64; n * dim];
            for j in 0..n {
                for k in 0..dim {
                    want[j * dim + k] = (0..b)
                        .map(|i| dl[i * n + j] as f64 * q[i * dim + k] as f64)
                        .sum();
                }
            }
            close(&ge, &want);

            let mut gq = vec![f32::NAN; b * dim];
            grad_queries_nn(&dl, &e, b, n, dim, &mut gq);
            let mut want = vec![0.0f64; b * dim];
            for i in 0..b {
                for k in 0..dim {
                    want[i * dim + k] = (0..n)
                        .map(|j| dl[i * n + j] as f64 * e[j * dim + k] as f64)
                        .sum();
                }
            }
            close(&gq, &want);
        }
    }
}
