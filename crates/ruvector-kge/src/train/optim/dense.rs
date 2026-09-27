//! Row-parallel updates for the dense layout (1-N training touches every
//! entity row every batch, so the sequential per-row walk over a 40,943 ×
//! 2,000 table was a single-core, memory-bound phase of every step).
//!
//! Each row's update reads and writes only that row of the parameters, the
//! gradient and the optimizer state, so splitting the table into fixed row
//! chunks gives **bitwise-identical** results to the sequential ascending-id
//! walk, for any thread count and with or without the `parallel` feature.
//! These are fast paths: they return `false` (having done nothing) unless both
//! the gradient and every state table are dense and exactly table-shaped, and
//! the caller then falls back to the generic per-row loop.

use super::{adagrad_step, adam_step, AdamHyper, Rows};

/// Rows per parallel work item.
const ROW_CHUNK: usize = 64;

/// Zip `params`, `grads` and `touched` with any state buffers (row-major,
/// `d` wide) and their `touched` flags, cut into [`ROW_CHUNK`]-row chunks,
/// and run `$f` on each chunk — in parallel under the `parallel` feature.
macro_rules! for_row_chunks {
    ($d:expr, $p:expr, $g:expr, $gt:expr, [$($s:expr),*], [$($st:expr),*], $f:expr) => {{
        let c = ROW_CHUNK * $d;
        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;
            $p.par_chunks_mut(c)
                .zip($g.par_chunks(c))
                .zip($gt.par_chunks(ROW_CHUNK))
                $(.zip($s.par_chunks_mut(c)))*
                $(.zip($st.par_chunks_mut(ROW_CHUNK)))*
                .for_each($f);
        }
        #[cfg(not(feature = "parallel"))]
        {
            $p.chunks_mut(c)
                .zip($g.chunks(c))
                .zip($gt.chunks(ROW_CHUNK))
                $(.zip($s.chunks_mut(c)))*
                $(.zip($st.chunks_mut(ROW_CHUNK)))*
                .for_each($f);
        }
    }};
}

/// Dense Adagrad over every touched row of `params` (`rows × d`).
pub(super) fn adagrad(
    params: &mut [f32],
    grads: &Rows,
    acc: &mut Rows,
    d: usize,
    lr: f32,
    eps: f32,
) -> bool {
    let (
        Rows::Dense {
            buf: g,
            touched: gt,
        },
        Rows::Dense {
            buf: a,
            touched: at,
        },
    ) = (grads, acc)
    else {
        return false;
    };
    let rows = gt.len();
    if d == 0 || params.len() != rows * d || g.len() != rows * d || a.len() != rows * d {
        return false;
    }
    if at.len() != rows {
        return false;
    }
    for_row_chunks!(d, params, g, gt, [a], [at], |((((p, g), gt), a), at)| {
        for (i, &t) in gt.iter().enumerate() {
            if t {
                at[i] = true;
                let r = i * d..(i + 1) * d;
                adagrad_step(&mut p[r.clone()], &g[r.clone()], &mut a[r], lr, eps);
            }
        }
    });
    true
}

/// Dense Adam over every touched row of `params` (`rows × d`).
pub(super) fn adam(
    params: &mut [f32],
    grads: &Rows,
    m: &mut Rows,
    v: &mut Rows,
    d: usize,
    h: &AdamHyper,
) -> bool {
    let (
        Rows::Dense {
            buf: g,
            touched: gt,
        },
        Rows::Dense {
            buf: mb,
            touched: mt,
        },
        Rows::Dense {
            buf: vb,
            touched: vt,
        },
    ) = (grads, m, v)
    else {
        return false;
    };
    let rows = gt.len();
    let n = rows * d;
    if d == 0 || params.len() != n || g.len() != n || mb.len() != n || vb.len() != n {
        return false;
    }
    if mt.len() != rows || vt.len() != rows {
        return false;
    }
    for_row_chunks!(d, params, g, gt, [mb, vb], [mt, vt], |(
        (((((p, g), gt), m), v), mt),
        vt,
    )| {
        for (i, &t) in gt.iter().enumerate() {
            if t {
                mt[i] = true;
                vt[i] = true;
                let r = i * d..(i + 1) * d;
                adam_step(
                    &mut p[r.clone()],
                    &g[r.clone()],
                    &mut m[r.clone()],
                    &mut v[r],
                    h,
                );
            }
        }
    });
    true
}

/// Zero every touched row of a dense `buf` and reset its `touched` flags.
pub(super) fn clear(buf: &mut [f32], touched: &mut [bool], d: usize) {
    if d == 0 || buf.len() != touched.len() * d {
        for (i, t) in touched.iter_mut().enumerate() {
            if std::mem::take(t) {
                buf[i * d..(i + 1) * d].fill(0.0);
            }
        }
        return;
    }
    let work = |(b, ts): (&mut [f32], &mut [bool])| {
        for (i, t) in ts.iter_mut().enumerate() {
            if std::mem::take(t) {
                b[i * d..(i + 1) * d].fill(0.0);
            }
        }
    };
    let c = ROW_CHUNK * d;
    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        buf.par_chunks_mut(c)
            .zip(touched.par_chunks_mut(ROW_CHUNK))
            .for_each(work);
    }
    #[cfg(not(feature = "parallel"))]
    buf.chunks_mut(c)
        .zip(touched.chunks_mut(ROW_CHUNK))
        .for_each(work);
}

#[cfg(test)]
mod tests {
    use super::super::{Grads, OptimKind, Optimizer, StateLayout};
    use crate::Tables;

    /// Row-parallel dense updates == the sparse per-row walk, bitwise, for
    /// tables spanning several row chunks with a sparse touched set, over
    /// three batches (so state carries) — and in every pool size.
    #[test]
    fn row_parallel_dense_matches_sparse_bitwise() {
        let (ne, nr, d) = (3 * super::ROW_CHUNK + 5, 3usize, 6usize);
        let kinds = [
            OptimKind::Adagrad { epsilon: 1e-6 },
            OptimKind::Adam {
                beta1: 0.9,
                beta2: 0.99,
                epsilon: 1e-6,
            },
        ];
        let run = |kind: OptimKind, layout: StateLayout| {
            let mut t = Tables::new(ne, nr, d, 9);
            let mut opt = Optimizer::new(kind, 0.05, layout, d, ne, nr);
            let mut g = Grads::with_layout(layout, d, ne, nr);
            for step in 0..3u32 {
                g.clear();
                for id in (step..ne as u32).step_by(7 + step as usize) {
                    let v: Vec<f32> = (0..d)
                        .map(|j| ((id * 31 + j as u32 * 17 + step) % 13) as f32 * 0.1 - 0.6)
                        .collect();
                    g.add_entity(id, &v);
                    g.add_relation(id % nr as u32, &v);
                }
                opt.apply(&mut t, &g).unwrap();
            }
            (t, opt.export_state())
        };
        for kind in kinds {
            let want = run(kind, StateLayout::Sparse);
            assert_eq!(run(kind, StateLayout::Dense), want, "{kind:?}");
            #[cfg(feature = "parallel")]
            for n in [1usize, 3, 16] {
                let pool = rayon::ThreadPoolBuilder::new()
                    .num_threads(n)
                    .build()
                    .unwrap();
                let got = pool.install(|| run(kind, StateLayout::Dense));
                assert_eq!(got, want, "{kind:?} threads={n}");
            }
        }
    }
}
