//! The three SGEMMs of a 1-N step, on `matrixmultiply::sgemm` (plan M2).
//!
//! ```text
//! logits  [B×N]   = Q  [B×D] · Eᵀ          (forward)
//! grad_E  [N×D]   = dLᵀ[N×B] · Q  [B×D]    (backward to the entity table)
//! grad_Q  [B×D]   = dL [B×N] · E  [N×D]    (backward to the queries)
//! ```
//!
//! Transposes are pure stride swaps — nothing is copied.
//!
//! **Determinism.** Every call is split into chunks of *output* rows or
//! columns whose size is a compile-time constant, never a function of the
//! thread count. Each output element is therefore produced by exactly one
//! `sgemm` call, whose reduction over the inner dimension runs in a fixed
//! order (fixed `kc` blocking, fixed micro-kernel). There is no cross-thread
//! reduction anywhere, so results are bitwise identical for any thread count,
//! with or without the `parallel` feature (asserted by the kernel tests).
//! `matrixmultiply`'s own `threading` feature is not enabled.

use matrixmultiply::sgemm;

/// Entity-axis chunk (logits columns, grad_E rows). Fixed: see module docs.
const ENTITY_CHUNK: usize = 512;
/// Query-axis chunk (grad_Q rows). Fixed: see module docs.
const QUERY_CHUNK: usize = 32;

/// A raw output pointer shared across chunk workers. Each worker writes a
/// disjoint set of elements, so there is no data race.
#[derive(Clone, Copy)]
struct OutPtr(*mut f32);
// SAFETY: workers write disjoint element sets (disjoint chunk ranges) and the
// owning `&mut [f32]` outlives every worker (the dispatch below is scoped).
unsafe impl Send for OutPtr {}
unsafe impl Sync for OutPtr {}

/// Run `f(start, end)` over `[0, total)` in fixed-size chunks — in parallel
/// under the `parallel` feature (inside whatever rayon pool is current),
/// sequentially otherwise. Chunk boundaries never depend on the thread count.
pub(crate) fn for_each_chunk<F>(total: usize, chunk: usize, f: F)
where
    F: Fn(usize, usize) + Send + Sync,
{
    let n_chunks = total.div_ceil(chunk);
    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        (0..n_chunks)
            .into_par_iter()
            .for_each(|c| f(c * chunk, ((c + 1) * chunk).min(total)));
    }
    #[cfg(not(feature = "parallel"))]
    for c in 0..n_chunks {
        f(c * chunk, ((c + 1) * chunk).min(total));
    }
}

/// `logits[b×n] = q[b×dim] · e[n×dim]ᵀ`, all row-major.
pub(crate) fn logits_nt(q: &[f32], e: &[f32], b: usize, n: usize, dim: usize, out: &mut [f32]) {
    assert!(q.len() >= b * dim && e.len() >= n * dim && out.len() >= b * n);
    if b == 0 || n == 0 {
        return;
    }
    let (qp, ep, cp) = (
        q.as_ptr() as usize,
        e.as_ptr() as usize,
        OutPtr(out.as_mut_ptr()),
    );
    for_each_chunk(n, ENTITY_CHUNK, move |j0, j1| {
        let cp = cp;
        // SAFETY: bounds asserted above. A = q (b×dim, rs=dim, cs=1);
        // B = e[j0..j1]ᵀ (dim×(j1-j0), rs=1, cs=dim); C = out[:, j0..j1]
        // (rs=n, cs=1). Chunks write disjoint column ranges of `out`.
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
    for_each_chunk(n, ENTITY_CHUNK, move |i0, i1| {
        let cp = cp;
        // SAFETY: A = dl[:, i0..i1]ᵀ ((i1-i0)×b, rs=1, cs=n); B = q (b×dim,
        // rs=dim, cs=1); C = out[i0..i1] ((i1-i0)×dim, rs=dim, cs=1).
        // Chunks write disjoint row ranges of `out`.
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
    assert!(dl.len() >= b * n && e.len() >= n * dim && out.len() >= b * dim);
    if b == 0 {
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
    for_each_chunk(b, QUERY_CHUNK, move |i0, i1| {
        let cp = cp;
        // SAFETY: A = dl[i0..i1] ((i1-i0)×n, rs=n, cs=1); B = e (n×dim,
        // rs=dim, cs=1); C = out[i0..i1] (rs=dim, cs=1). Disjoint rows.
        unsafe {
            sgemm(
                i1 - i0,
                n,
                dim,
                1.0,
                (dp as *const f32).add(i0 * n),
                n as isize,
                1,
                ep as *const f32,
                dim as isize,
                1,
                0.0,
                cp.0.add(i0 * dim),
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
