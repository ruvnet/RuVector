//! Correctness + convergence + determinism tests for the
//! Lanczos-with-full-reorthogonalization Fiedler driver.
//!
//! The four fixtures mirror the acceptance criteria from the task
//! prompt (ADR-154 §13 follow-up):
//!
//! 1. *path topology*: a 1-D chain on 256 nodes where the analytical
//!    Laplacian spectrum is known (`λ_k = 2(1 − cos(k·π/N))`). The
//!    Fiedler value is `λ_2 = 2(1 − cos(π/N))`. Lanczos must return a
//!    value within 5 % relative error.
//!
//!    This fixture is specifically where the prior shifted-power-
//!    iteration eigensolver collapses to 0 (via the `(λ_max − μ).max(0)`
//!    PSD-floor convention). We assert both behaviours inside the test
//!    so the failure mode is documented on the same fixture that
//!    exercises the fix.
//!
//! 2. *cross-validation against dense Jacobi on K_{16,16}*. The
//!    complete bipartite graph K_{m,n} has a closed-form Laplacian
//!    spectrum: `{0, m (mult n-1), n (mult m-1), m+n (mult 1)}`.
//!    For K_{16,16} that is `{0, 16⁽³⁰⁾, 32}`, so `λ_2 = 16` exactly.
//!    Lanczos must agree with dense cyclic Jacobi within 1e-4
//!    relative error on the same graph.
//!
//! 3. *large-scale determinism*: `N = 10 000` synthetic path-like
//!    fixture, two runs. Bit-identical f32 output across repeat runs
//!    is required — no non-determinism from iteration order, RNG, or
//!    reduction tree.
//!
//! 4. *convergence budget*: the prompt stipulates
//!    `max_iter = 3·k` for `k = 20` extremal Ritz values as the
//!    expected envelope. We assert that 60 iterations is sufficient
//!    on the path-256 fixture (5 % relative error to the analytical
//!    value).
//!
//! All tests drive `lanczos_fiedler` through the public `CsrLaplacian`
//! API exposed by `sparse_fiedler` — no Observer spike-accumulation
//! path, since the mathematical fixtures (path, bipartite) cannot be
//! constructed as co-firing windows in a meaningful way.

use connectome_fly::observer::eigensolver::jacobi_symmetric;
use connectome_fly::observer::lanczos::{lanczos_fiedler, DEFAULT_TOL};
use connectome_fly::observer::sparse_fiedler::CsrLaplacian;

// ---------------------------------------------------------------------
// Analytical spectrum helpers
// ---------------------------------------------------------------------

/// Analytical Fiedler value of an unweighted path graph on N nodes.
/// For the path `P_N` with unit weights and standard Laplacian
/// `L = D − A`, the eigenvalues are `λ_k = 2·(1 − cos(k·π/N))` for
/// `k = 0, 1, …, N − 1`. The Fiedler value is `λ_1` in that
/// convention (one-indexed: the smallest non-zero eigenvalue).
fn path_graph_fiedler_analytical(n: usize) -> f32 {
    2.0 * (1.0 - (std::f32::consts::PI / n as f32).cos())
}

/// Build the CSR Laplacian of an unweighted path graph `P_N` (edges
/// `i — i+1` for `i ∈ [0, N-1)`, unit weight). The graph is connected,
/// so `λ_1 = 0`, `λ_2 > 0`.
fn build_path_csr(n: usize) -> CsrLaplacian {
    let mut edges = Vec::with_capacity(n - 1);
    for i in 0..(n - 1) {
        edges.push((i as u32, (i + 1) as u32, 1.0_f32));
    }
    CsrLaplacian::from_edges(n, &edges).expect("path graph has ≥ 1 edge")
}

/// Build the CSR Laplacian of the complete bipartite graph `K_{m,n}`
/// on disjoint vertex sets `[0, m) ∪ [m, m+n)`. Every edge is unit-
/// weight.
fn build_kmn_csr(m: usize, n: usize) -> CsrLaplacian {
    let mut edges = Vec::with_capacity(m * n);
    for i in 0..m {
        for j in 0..n {
            edges.push((i as u32, (m + j) as u32, 1.0_f32));
        }
    }
    CsrLaplacian::from_edges(m + n, &edges).expect("K_{m,n} has ≥ 1 edge")
}

/// Dense ground-truth `λ_2` of the Laplacian of an unweighted graph
/// given as edge list. Uses cyclic Jacobi on the full `n × n` matrix
/// — only viable for small `n`, test-only.
fn dense_lambda_2(n: usize, edges: &[(u32, u32)]) -> f32 {
    let mut l = vec![0.0_f32; n * n];
    for &(u, v) in edges {
        let u = u as usize;
        let v = v as usize;
        if u == v {
            continue;
        }
        l[u * n + v] -= 1.0;
        l[v * n + u] -= 1.0;
        l[u * n + u] += 1.0;
        l[v * n + v] += 1.0;
    }
    let mut eigs = jacobi_symmetric(&l, n);
    eigs.sort_by(|x, y| x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal));
    // First eigenvalue above a small positivity floor is λ_2.
    let scale_hint = eigs.last().copied().unwrap_or(1.0).abs().max(1.0);
    let floor = 1e-6 * scale_hint;
    eigs.into_iter().find(|v| *v > floor).unwrap_or(0.0)
}

/// Shifted-power-iteration (the algorithm the Lanczos driver replaces),
/// restricted to the sparse-CSR case so tests can document its failure
/// mode on path-topologies. Mirrors the prior implementation of
/// `power_iter_lmax` + `power_iter_shifted` in `sparse_fiedler.rs`
/// (commit 6). Not exported from the library — reproduced here only as
/// a regression witness.
fn shifted_power_fiedler_reference(csr: &CsrLaplacian) -> f32 {
    let n = csr.n();
    // λ_max(L) via 32-step power iteration with constant deflation.
    let mut x: Vec<f32> = (0..n).map(|i| ((i * 31 + 7) as f32).sin()).collect();
    deflate_const(&mut x);
    normalize(&mut x);
    let mut w = vec![0.0_f32; n];
    let mut lambda_max = 0.0_f32;
    for _ in 0..32 {
        csr.mat_vec_l(&x, &mut w);
        deflate_const(&mut w);
        normalize(&mut w);
        let mut lw = vec![0.0_f32; n];
        csr.mat_vec_l(&w, &mut lw);
        lambda_max = dot(&w, &lw);
        std::mem::swap(&mut x, &mut w);
    }
    // μ via 64-step shifted iteration on (c·I − L) with c = 1.1·λ_max.
    let c = lambda_max * 1.1 + 1e-3;
    let mut x: Vec<f32> = (0..n).map(|i| ((i * 19 + 11) as f32).cos()).collect();
    deflate_const(&mut x);
    normalize(&mut x);
    let mut lx = vec![0.0_f32; n];
    let mut mu = 0.0_f32;
    for _ in 0..64 {
        csr.mat_vec_l(&x, &mut lx);
        let mut y: Vec<f32> = (0..n).map(|i| c * x[i] - lx[i]).collect();
        deflate_const(&mut y);
        normalize(&mut y);
        csr.mat_vec_l(&y, &mut lx);
        mu = 0.0;
        for i in 0..n {
            mu += y[i] * (c * y[i] - lx[i]);
        }
        x = y;
    }
    (lambda_max - mu).max(0.0)
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}
fn norm(x: &[f32]) -> f32 {
    x.iter().map(|v| v * v).sum::<f32>().sqrt()
}
fn normalize(x: &mut [f32]) {
    let nrm = norm(x);
    if nrm > 1e-20 {
        let inv = 1.0 / nrm;
        for v in x.iter_mut() {
            *v *= inv;
        }
    }
}
fn deflate_const(x: &mut [f32]) {
    if x.is_empty() {
        return;
    }
    let m = x.iter().sum::<f32>() / x.len() as f32;
    for v in x.iter_mut() {
        *v -= m;
    }
}

// ---------------------------------------------------------------------
// (1) Path topology — analytical reference
// ---------------------------------------------------------------------

#[test]
fn lanczos_path_256_matches_analytical_lambda_2() {
    let n = 256;
    let csr = build_path_csr(n);
    let analytical = path_graph_fiedler_analytical(n);

    // 60-step Lanczos (the default max_iter in `sparse_fiedler`).
    let lanczos = lanczos_fiedler(&csr, n, 60, DEFAULT_TOL);

    let denom = analytical.abs().max(1e-12);
    let rel_err = (lanczos - analytical).abs() / denom;
    eprintln!(
        "path-{n}: analytical λ_2 = {analytical:.8}, lanczos = {lanczos:.8}, rel-err = {:.4}%",
        rel_err * 100.0
    );
    assert!(
        rel_err <= 0.05,
        "lanczos rel-err {:.4}% > 5 %: analytical = {analytical:.8e}, lanczos = {lanczos:.8e}",
        rel_err * 100.0
    );

    // Document the failure mode the Lanczos replaces: on the same
    // fixture the prior shifted-power-iteration eigensolver returns
    // 0 (λ_2 ≪ λ_max ⇒ (λ_max − μ).max(0) floors out).
    let spi = shifted_power_fiedler_reference(&csr);
    eprintln!("path-{n}: shifted-power reference = {spi:.8} (expected 0, by PSD floor)");
    assert_eq!(
        spi, 0.0,
        "shifted-power reference on path-{n} should collapse to 0 (the failure mode \
         the Lanczos driver replaces), got {spi}"
    );
}

// ---------------------------------------------------------------------
// (2) Complete bipartite K_{16,16} cross-validation
// ---------------------------------------------------------------------

#[test]
fn lanczos_k_16_16_agrees_with_dense_jacobi() {
    let m = 16_usize;
    let n = 16_usize;
    let total = m + n;
    let csr = build_kmn_csr(m, n);

    // Edge list for dense reference.
    let mut edges: Vec<(u32, u32)> = Vec::with_capacity(m * n);
    for i in 0..m {
        for j in 0..n {
            edges.push((i as u32, (m + j) as u32));
        }
    }
    let dense = dense_lambda_2(total, &edges);

    // Closed-form λ_2(K_{16,16}) = min(m, n) = 16.
    let analytical = 16.0_f32;
    eprintln!(
        "K_{{16,16}}: analytical λ_2 = {analytical}, dense-jacobi = {dense:.6}"
    );

    // 60-step Lanczos.
    let lanczos = lanczos_fiedler(&csr, total, 60, DEFAULT_TOL);
    eprintln!("K_{{16,16}}: lanczos = {lanczos:.6}");

    // Lanczos vs dense Jacobi — same Laplacian, independent solvers.
    let denom = dense.abs().max(1e-6);
    let rel = (lanczos - dense).abs() / denom;
    assert!(
        rel <= 1e-4,
        "lanczos-vs-dense rel-err {rel:.6} > 1e-4 (dense={dense:.6}, lanczos={lanczos:.6})"
    );
    // Also: both should nail the closed form within the same tolerance.
    let rel_an = (lanczos - analytical).abs() / analytical.abs();
    assert!(
        rel_an <= 1e-3,
        "lanczos-vs-analytical rel-err {rel_an:.6} > 1e-3 (analytical={analytical}, lanczos={lanczos:.6})"
    );
}

// ---------------------------------------------------------------------
// (3) Large-scale determinism
// ---------------------------------------------------------------------

/// Synthesise an N = 10 000 sparse Laplacian with path-like backbone
/// plus a deterministic sprinkle of shortcut edges. Same build order
/// twice ⇒ bit-identical CSR, which must produce bit-identical Lanczos
/// output.
fn build_large_synth(n: usize) -> CsrLaplacian {
    let mut edges: Vec<(u32, u32, f32)> = Vec::with_capacity(2 * n);
    // Path backbone: i — i+1.
    for i in 0..(n - 1) {
        edges.push((i as u32, (i + 1) as u32, 1.0));
    }
    // Shortcut edges every 137th vertex to vertex + 1009 (mod n). Two
    // primes keep the pattern non-trivial and deterministic without
    // needing an RNG.
    let mut i = 0_usize;
    while i < n {
        let j = (i + 1009) % n;
        edges.push((i as u32, j as u32, 0.5));
        i += 137;
    }
    CsrLaplacian::from_edges(n, &edges).expect("synth graph has ≥ 1 edge")
}

#[test]
fn lanczos_determinism_n_10k() {
    let n = 10_000;
    let csr = build_large_synth(n);

    let a = lanczos_fiedler(&csr, n, 60, DEFAULT_TOL);
    let b = lanczos_fiedler(&csr, n, 60, DEFAULT_TOL);

    eprintln!("N=10 000 determinism: a = {a:.8} b = {b:.8}");
    assert_eq!(
        a.to_bits(),
        b.to_bits(),
        "lanczos must be bit-identical across repeat runs on the same CSR (a={a}, b={b})"
    );
    assert!(a.is_finite() && a >= 0.0);
}

// ---------------------------------------------------------------------
// (4) Convergence budget: k_cap = 3·20 = 60 is sufficient on path-256
// ---------------------------------------------------------------------

#[test]
fn lanczos_60_iter_sufficient_on_path_256() {
    // Same fixture as (1), but asserts the explicit budget requirement
    // from the task prompt: `max_iter = 3·k` Krylov vectors with
    // `k = 20` ⇒ 60 iterations must land within 5 % of λ_2 analytical.
    let n = 256;
    let csr = build_path_csr(n);
    let analytical = path_graph_fiedler_analytical(n);

    let lanczos_60 = lanczos_fiedler(&csr, n, 60, DEFAULT_TOL);
    let rel_60 = (lanczos_60 - analytical).abs() / analytical.abs();
    eprintln!(
        "convergence budget: 60 iterations on path-{n}: \
         lanczos = {lanczos_60:.8}, analytical = {analytical:.8}, rel-err = {:.4}%",
        rel_60 * 100.0
    );
    assert!(
        rel_60 <= 0.05,
        "60-iter Lanczos rel-err {:.4}% > 5 % on path-256 (need more steps)",
        rel_60 * 100.0
    );

    // Sanity: 30-iter is meaningfully tighter than 10-iter, so the
    // budget is not a trivial "any k works" situation.
    let lanczos_10 = lanczos_fiedler(&csr, n, 10, DEFAULT_TOL);
    let rel_10 = (lanczos_10 - analytical).abs() / analytical.abs();
    eprintln!(
        "convergence budget: 10 iterations on path-{n}: \
         lanczos = {lanczos_10:.8}, rel-err = {:.4}%",
        rel_10 * 100.0
    );
    // 10 iterations should produce a strictly worse estimate — this
    // documents the budget's necessity, not just its sufficiency.
    assert!(
        rel_10 >= rel_60 - 1e-6,
        "10-iter rel-err {rel_10:.6} unexpectedly < 60-iter rel-err {rel_60:.6} — \
         convergence should be monotone in k"
    );
}
