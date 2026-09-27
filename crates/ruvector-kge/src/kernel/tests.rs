//! Kernel correctness: batched loss/grads vs a naive per-triple f64 oracle,
//! finite differences of that oracle's loss, HolE vs the crate's own
//! per-triple `one_vs_all_step`, and bitwise determinism.

use super::complex::{complex_one_n_step, ComplexWorkspace};
use super::hole::HolEKernel;
use super::{one_n_softmax_ce, OneNQuery, Reduction, Workspace};
use crate::{HolE, Side, Tables, Triple};

pub(super) fn rand(len: usize, seed: u64, scale: f32) -> Vec<f32> {
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    (0..len)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            scale * ((s >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0) as f32
        })
        .collect()
}

/// Naive per-triple scorer in f64 with analytic grads.
trait RefModel {
    fn score(&self, s: &[f64], r: &[f64], o: &[f64]) -> f64;
    fn grad(&self, s: &[f64], r: &[f64], o: &[f64]) -> [Vec<f64>; 3];
}

/// ComplEx `Re Σ r s conj(o)`, rows `[re; im]`.
struct RefComplex;
impl RefModel for RefComplex {
    fn score(&self, s: &[f64], r: &[f64], o: &[f64]) -> f64 {
        let k = s.len() / 2;
        (0..k)
            .map(|i| {
                let (a, b) = (
                    r[i] * s[i] - r[k + i] * s[k + i],
                    r[i] * s[k + i] + r[k + i] * s[i],
                );
                a * o[i] + b * o[k + i]
            })
            .sum()
    }
    fn grad(&self, s: &[f64], r: &[f64], o: &[f64]) -> [Vec<f64>; 3] {
        let k = s.len() / 2;
        let (mut gs, mut gr, mut go) = (vec![0.0; 2 * k], vec![0.0; 2 * k], vec![0.0; 2 * k]);
        for i in 0..k {
            let (sr, si, rr, ri, or, oi) = (s[i], s[k + i], r[i], r[k + i], o[i], o[k + i]);
            // score_i = rr sr or − ri si or + rr si oi + ri sr oi
            gs[i] = rr * or + ri * oi;
            gs[k + i] = -ri * or + rr * oi;
            gr[i] = sr * or + si * oi;
            gr[k + i] = -si * or + sr * oi;
            go[i] = rr * sr - ri * si;
            go[k + i] = rr * si + ri * sr;
        }
        [gs, gr, go]
    }
}

/// HolE `r · (s ⋆ o)` by the direct O(d²) definition.
struct RefHolE;
impl RefModel for RefHolE {
    fn score(&self, s: &[f64], r: &[f64], o: &[f64]) -> f64 {
        let d = s.len();
        (0..d)
            .map(|k| r[k] * (0..d).map(|i| s[i] * o[(i + k) % d]).sum::<f64>())
            .sum()
    }
    fn grad(&self, s: &[f64], r: &[f64], o: &[f64]) -> [Vec<f64>; 3] {
        let d = s.len();
        let (mut gs, mut gr, mut go) = (vec![0.0; d], vec![0.0; d], vec![0.0; d]);
        for k in 0..d {
            for i in 0..d {
                let j = (i + k) % d;
                gs[i] += r[k] * o[j];
                gr[k] += s[i] * o[j];
                go[j] += r[k] * s[i];
            }
        }
        [gs, gr, go]
    }
}

struct RefOut {
    loss: f64,
    ge: Vec<f64>,
    gr: Vec<f64>,
}

fn row(t: &[f64], i: usize, dim: usize) -> &[f64] {
    &t[i * dim..(i + 1) * dim]
}

/// Per-triple 1-N softmax CE, f64, one `score`/`grad` call per candidate.
fn reference(
    m: &dyn RefModel,
    ents: &[f64],
    rels: &[f64],
    dim: usize,
    qs: &[OneNQuery],
    red: Reduction,
) -> RefOut {
    let n = ents.len() / dim;
    let scale = match red {
        Reduction::Sum => 1.0,
        Reduction::Mean => 1.0 / qs.len() as f64,
    };
    let (mut ge, mut gr) = (vec![0.0; ents.len()], vec![0.0; rels.len()]);
    let mut loss = 0.0;
    for q in qs {
        let (a, r) = (q.anchor as usize, q.relation as usize);
        let triple = |e: usize| match q.side {
            Side::Tail => (a, e),
            Side::Head => (e, a),
        };
        let sc: Vec<f64> = (0..n)
            .map(|e| {
                let (s, o) = triple(e);
                m.score(row(ents, s, dim), row(rels, r, dim), row(ents, o, dim))
            })
            .collect();
        let mx = sc.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let z: f64 = sc.iter().map(|x| (x - mx).exp()).sum();
        loss += mx + z.ln() - sc[q.target as usize];
        for (e, &se) in sc.iter().enumerate() {
            let c = scale * ((se - mx).exp() / z - if e == q.target as usize { 1.0 } else { 0.0 });
            let (s, o) = triple(e);
            let [gs, grr, go] = m.grad(row(ents, s, dim), row(rels, r, dim), row(ents, o, dim));
            for k in 0..dim {
                ge[s * dim + k] += c * gs[k];
                gr[r * dim + k] += c * grr[k];
                ge[o * dim + k] += c * go[k];
            }
        }
    }
    RefOut {
        loss: loss * scale,
        ge,
        gr,
    }
}

/// `|a − b| ≤ tol · max(1, ‖b‖∞)` elementwise (norm-relative, so entries that
/// cancel to ~0 are not held to an unreachable pointwise relative bound).
pub(super) fn assert_close(what: &str, got: &[f32], want: &[f64], tol: f64) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    let scale = want.iter().fold(1.0f64, |m, x| m.max(x.abs()));
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        assert!(
            (g as f64 - w).abs() <= tol * scale,
            "{what}[{i}]: kernel {g} vs reference {w} (scale {scale})"
        );
    }
}

pub(super) fn to64(v: &[f32]) -> Vec<f64> {
    v.iter().map(|&x| x as f64).collect()
}

fn queries(n: usize, nr: usize, b: usize, seed: u64) -> Vec<OneNQuery> {
    let r = rand(4 * b, seed, 1.0);
    (0..b)
        .map(|i| {
            let u = |j: usize, m: usize| {
                (((r[4 * i + j] + 1.0) * 0.5 * m as f32) as usize).min(m - 1) as u32
            };
            OneNQuery {
                anchor: u(0, n),
                relation: u(1, nr),
                side: if r[4 * i + 2] > 0.0 {
                    Side::Tail
                } else {
                    Side::Head
                },
                target: u(3, n),
            }
        })
        .collect()
}

fn run_complex(
    ents: &[f32],
    rels: &[f32],
    k: usize,
    qs: &[OneNQuery],
    red: Reduction,
) -> (f32, Vec<f32>, Vec<f32>) {
    let (mut ge, mut gr) = (vec![0.0; ents.len()], vec![0.0; rels.len()]);
    let mut ws = ComplexWorkspace::new();
    let l = complex_one_n_step(ents, rels, k, qs, red, &mut ws, &mut ge, &mut gr).unwrap();
    (l, ge, gr)
}

#[test]
fn complex_kernel_matches_per_triple_oracle() {
    // Shapes straddle the GEMM chunk sizes (512 entities, 32 queries).
    for &(n, nr, k, b) in &[(40usize, 3usize, 4usize, 5usize), (600, 7, 24, 70)] {
        let ents = rand(n * 2 * k, 11 + n as u64, 0.5);
        let rels = rand(nr * 2 * k, 13 + k as u64, 0.5);
        let qs = queries(n, nr, b, 17 + b as u64);
        for red in [Reduction::Sum, Reduction::Mean] {
            let (l, ge, gr) = run_complex(&ents, &rels, k, &qs, red);
            let want = reference(&RefComplex, &to64(&ents), &to64(&rels), 2 * k, &qs, red);
            assert!(
                (l as f64 - want.loss).abs() <= 1e-5 * want.loss.abs().max(1.0),
                "loss {l} vs {}",
                want.loss
            );
            assert_close("grad_entities", &ge, &want.ge, 1e-5);
            assert_close("grad_relations", &gr, &want.gr, 1e-5);
        }
    }
}

/// Central finite differences of the f64 per-triple loss vs the kernel's
/// analytic gradient, on every entity and relation coordinate.
#[test]
fn complex_kernel_gradient_fd_check() {
    let (n, nr, k, b) = (12usize, 2usize, 3usize, 6usize);
    let ents = rand(n * 2 * k, 101, 0.8);
    let rels = rand(nr * 2 * k, 103, 0.8);
    let qs = queries(n, nr, b, 107);
    let (_, ge, gr) = run_complex(&ents, &rels, k, &qs, Reduction::Mean);
    fd_check(&RefComplex, &ents, &rels, 2 * k, &qs, &ge, &gr);
}

fn fd_check(
    m: &dyn RefModel,
    ents: &[f32],
    rels: &[f32],
    dim: usize,
    qs: &[OneNQuery],
    ge: &[f32],
    gr: &[f32],
) {
    let (e64, r64) = (to64(ents), to64(rels));
    let h = 1e-5;
    let loss = |e: &[f64], r: &[f64]| reference(m, e, r, dim, qs, Reduction::Mean).loss;
    for i in 0..e64.len() {
        let (mut p, mut q) = (e64.clone(), e64.clone());
        p[i] += h;
        q[i] -= h;
        let fd = (loss(&p, &r64) - loss(&q, &r64)) / (2.0 * h);
        assert!(
            (fd - ge[i] as f64).abs() <= 1e-4 * fd.abs().max(1.0),
            "ent[{i}]: fd {fd} vs {}",
            ge[i]
        );
    }
    for i in 0..r64.len() {
        let (mut p, mut q) = (r64.clone(), r64.clone());
        p[i] += h;
        q[i] -= h;
        let fd = (loss(&e64, &p) - loss(&e64, &q)) / (2.0 * h);
        assert!(
            (fd - gr[i] as f64).abs() <= 1e-4 * fd.abs().max(1.0),
            "rel[{i}]: fd {fd} vs {}",
            gr[i]
        );
    }
}

fn run_hole(tables: &Tables, qs: &[OneNQuery], red: Reduction) -> (f32, Vec<f32>, Vec<f32>) {
    let d = tables.dims();
    let hole = HolE::new(d).unwrap();
    let mut k = HolEKernel::new(d).unwrap();
    let mut ge = vec![0.0; tables.num_entities() * d];
    let mut gr = vec![0.0; tables.num_relations() * d];
    let l = k.step(&hole, tables, qs, red, &mut ge, &mut gr).unwrap();
    (l, ge, gr)
}

#[test]
fn hole_kernel_matches_per_triple_oracle_and_fd() {
    for &(n, nr, d, b) in &[(30usize, 3usize, 8usize, 7usize), (540, 5, 32, 40)] {
        let tables = Tables::new(n, nr, d, 23 + d as u64);
        let qs = queries(n, nr, b, 29 + n as u64);
        let (l, ge, gr) = run_hole(&tables, &qs, Reduction::Sum);
        let want = reference(
            &RefHolE,
            &to64(tables.entities_raw()),
            &to64(tables.relations_raw()),
            d,
            &qs,
            Reduction::Sum,
        );
        assert!((l as f64 - want.loss).abs() <= 1e-5 * want.loss.abs().max(1.0));
        assert_close("hole grad_entities", &ge, &want.ge, 1e-5);
        assert_close("hole grad_relations", &gr, &want.gr, 1e-5);
    }
    let tables = Tables::new(10, 2, 6, 31);
    let qs = queries(10, 2, 5, 37);
    let (_, ge, gr) = run_hole(&tables, &qs, Reduction::Mean);
    fd_check(
        &RefHolE,
        tables.entities_raw(),
        tables.relations_raw(),
        6,
        &qs,
        &ge,
        &gr,
    );
}

/// The plan's named oracle: the crate's per-triple `one_vs_all_step` runs
/// tail + head for a positive; the kernel's two rows must give the same loss.
#[test]
fn hole_kernel_loss_equals_one_vs_all_step() {
    let (n, nr, d) = (64usize, 4usize, 16usize);
    let tables = Tables::new(n, nr, d, 41);
    let hole = HolE::new(d).unwrap();
    for t in [
        Triple::new(3, 1, 9),
        Triple::new(60, 3, 0),
        Triple::new(7, 0, 7),
    ] {
        let mut grads = crate::train::optim::Grads::new(d);
        let oracle = crate::train::loss::one_vs_all_step(&tables, &hole, t, &mut grads).unwrap();
        let qs = [
            OneNQuery {
                anchor: t.s,
                relation: t.r,
                side: Side::Tail,
                target: t.o,
            },
            OneNQuery {
                anchor: t.o,
                relation: t.r,
                side: Side::Head,
                target: t.s,
            },
        ];
        let (l, _, _) = run_hole(&tables, &qs, Reduction::Sum);
        assert!(
            (l - oracle).abs() <= 1e-5 * oracle.abs().max(1.0),
            "kernel {l} vs one_vs_all_step {oracle}"
        );
    }
}

#[test]
fn rejects_bad_shapes_and_targets() {
    let mut ws = Workspace::new();
    let (q, e) = (vec![0.0f32; 4], vec![0.0f32; 6]);
    let (mut gq, mut ge) = (vec![0.0; 4], vec![0.0; 6]);
    // dim 2: 2 queries, 3 entities; target 3 is out of range.
    assert!(one_n_softmax_ce(
        &q,
        &e,
        2,
        &[0, 3],
        Reduction::Sum,
        &mut ws,
        &mut gq,
        &mut ge
    )
    .is_err());
    assert!(one_n_softmax_ce(
        &q,
        &e,
        3,
        &[0, 1],
        Reduction::Sum,
        &mut ws,
        &mut gq,
        &mut ge
    )
    .is_err());
    assert!(one_n_softmax_ce(
        &q,
        &e,
        2,
        &[0, 2],
        Reduction::Sum,
        &mut ws,
        &mut gq,
        &mut ge
    )
    .is_ok());
}

/// FNV-1a over the bit patterns — a stable fingerprint of a whole step.
pub(super) fn fingerprint(l: f32, a: &[f32], b: &[f32]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for x in std::iter::once(l)
        .chain(a.iter().copied())
        .chain(b.iter().copied())
    {
        h = (h ^ x.to_bits() as u64).wrapping_mul(0x100_0000_01b3);
    }
    h
}

/// Same inputs → bitwise-identical loss and grads, run to run, and (under
/// `parallel`) across thread counts 1, 3, 16. Prints the fingerprint so the
/// default and `parallel` builds can be compared too (they must match).
#[test]
fn deterministic_bitwise() {
    let (n, nr, k, b) = (1500usize, 6usize, 40usize, 100usize);
    let ents = rand(n * 2 * k, 201, 0.3);
    let rels = rand(nr * 2 * k, 203, 0.3);
    let qs = queries(n, nr, b, 207);
    let run = || {
        let (l, ge, gr) = run_complex(&ents, &rels, k, &qs, Reduction::Mean);
        fingerprint(l, &ge, &gr)
    };
    let first = run();
    assert_eq!(first, run(), "run-to-run");
    #[cfg(feature = "parallel")]
    for t in [1usize, 3, 16] {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(t)
            .build()
            .unwrap();
        assert_eq!(first, pool.install(run), "threads={t}");
    }
    let tables = Tables::new(700, 4, 64, 211);
    let hq = queries(700, 4, 50, 213);
    let hrun = || {
        let (l, ge, gr) = run_hole(&tables, &hq, Reduction::Sum);
        fingerprint(l, &ge, &gr)
    };
    let hfirst = hrun();
    assert_eq!(hfirst, hrun(), "hole run-to-run");
    #[cfg(feature = "parallel")]
    for t in [1usize, 3, 16] {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(t)
            .build()
            .unwrap();
        assert_eq!(hfirst, pool.install(hrun), "hole threads={t}");
    }
    println!("kernel fingerprint complex={first:016x} hole={hfirst:016x}");
}
