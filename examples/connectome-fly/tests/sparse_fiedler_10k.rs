//! Scale + correctness tests for the sparse-Fiedler observer path.
//!
//! The dense-Laplacian Fiedler path used at `n ≤ 1024` allocates
//! `2 · n² · 4 B`, which is 8 MB at N=1024 but 800 MB at N=10 000 and
//! 153 GB at N=139 000 (FlyWire v783). The sparse path this file
//! exercises allocates `O(n + nnz)` instead.
//!
//! Tests:
//!
//! 1. `sparse_fiedler_scales_to_10k` — synthesises a 30 000-spike
//!    co-firing window over ~2 000 active neurons (N=10 000) and
//!    asserts a finite non-NaN Fiedler value returned in < 200 ms.
//! 2. `sparse_vs_dense_within_five_percent` — at N=256, runs full
//!    dense cyclic Jacobi on the same Laplacian as ground truth and
//!    asserts the sparse Lanczos path agrees within 5 % relative
//!    error. The prior baseline in this test used
//!    `approx_fiedler_power` — measured to be itself a loose estimate
//!    (returning 14.0 vs true λ_2 ≈ 107.0 on this fixture), so the
//!    ground-truth now comes from the small-n Jacobi path that
//!    `Observer::compute_fiedler` uses at `n ≤ 96`. At N=256, full
//!    Jacobi is still only `O(n³) ≈ 17M` ops — fine as a test
//!    reference.

use std::collections::VecDeque;
use std::time::Instant;

use connectome_fly::observer::eigensolver::jacobi_symmetric;
use connectome_fly::observer::sparse_fiedler::sparse_fiedler;
use connectome_fly::{NeuronId, Spike};

#[test]
fn sparse_fiedler_scales_to_10k() {
    // Construct a co-firing window at N=10 000 with 2 000 active
    // neurons organised as two "chains": neurons fire in sequence so
    // τ-coincidence links only consecutive neurons (bounded degree),
    // not all-to-all (unbounded degree). This keeps λ_max modest so
    // the shifted power iteration has room to resolve λ_2 above the
    // f32 noise floor.
    //
    // Community A: neurons 0..1000, sequential spacing 0.5 ms.
    // Community B: neurons 1000..2000, sequential spacing 0.5 ms.
    // Bridge: neurons 500 ↔ 1500 co-fire on a few bursts.
    let cluster_size = 1000_u32;
    let step_ms = 0.5_f32; // intra-cluster spike spacing (~10 neighbours within τ)
    let n_bursts = 30_u32;
    let bridge_count = 5_u32;
    let n_total: u32 = 10_000;

    let mut window: VecDeque<Spike> = VecDeque::new();
    for b in 0..n_bursts {
        let t_a = b as f32 * 2000.0;
        let t_b = t_a + 1000.0;
        for i in 0..cluster_size {
            window.push_back(Spike {
                t_ms: t_a + i as f32 * step_ms,
                neuron: NeuronId(i),
            });
        }
        for i in 0..cluster_size {
            window.push_back(Spike {
                t_ms: t_b + i as f32 * step_ms,
                neuron: NeuronId(cluster_size + i),
            });
        }
        for k in 0..bridge_count {
            let src = (k * 37) % cluster_size;
            let dst = cluster_size + (k * 41) % cluster_size;
            let t_bridge = t_a + 500.0 + k as f32 * 2.0;
            window.push_back(Spike {
                t_ms: t_bridge,
                neuron: NeuronId(src),
            });
            window.push_back(Spike {
                t_ms: t_bridge + 0.1,
                neuron: NeuronId(dst),
            });
        }
    }

    let mut active: Vec<NeuronId> = (0..2 * cluster_size).map(NeuronId).collect();
    active.sort();
    active.dedup();

    assert!(
        active.len() >= 1500,
        "synth: expected ≥ 1500 active neurons, got {}",
        active.len()
    );
    assert!(
        window.len() >= 25_000,
        "synth: expected ≥ 25k spikes, got {}",
        window.len()
    );
    // Guard: we built this fixture to live beyond the dense 1024
    // threshold so the sparse dispatch is the one actually exercised.
    assert!(
        active.len() > 1024,
        "synth: n_active {} ≤ 1024 — dense path would be used",
        active.len()
    );
    let _ = n_total; // documentation anchor for future flywire scale

    // Drive the sparse path directly — we bypass the Observer so we
    // control the scale test independently of the detect cadence.
    let t0 = Instant::now();
    let fiedler = sparse_fiedler(&active, &window, 1024);
    let dt = t0.elapsed();

    assert!(
        fiedler.is_finite() && !fiedler.is_nan(),
        "sparse fiedler returned non-finite: {fiedler}"
    );
    // Fiedler is the second-smallest eigenvalue of a PSD matrix — ≥ 0.
    assert!(
        fiedler >= 0.0,
        "negative fiedler on valid window: {fiedler}"
    );
    eprintln!(
        "sparse_fiedler_scales_to_10k: n_active={} spikes={} fiedler={:.5} \
         elapsed={:?}",
        active.len(),
        window.len(),
        fiedler,
        dt
    );
    assert!(
        dt.as_millis() < 200,
        "sparse fiedler took {:?} — target < 200 ms on reference host",
        dt
    );
}

#[test]
fn sparse_vs_dense_within_five_percent() {
    // Build a connected ring-with-chords window at N=256:
    //  - every neuron in the active set fires once per tick
    //  - τ-window captures neighbours in the firing order
    //  - each tick is offset to avoid cross-tick τ-coupling
    // The resulting co-firing graph is a path (dominant) plus a
    // handful of chord edges where tick boundaries overlap. This is
    // well-connected, so λ_2(L) > 0, and small enough (n=256) for the
    // dense shifted-power path to be stable.
    let n_active = 256_u32;
    let mut window: VecDeque<Spike> = VecDeque::new();
    // 8 ticks, each tick fires all neurons in order with a 0.1 ms
    // inter-spike interval. Ticks are 200 ms apart so τ=5 ms does not
    // couple across ticks.
    for tick in 0..8 {
        let t0 = tick as f32 * 200.0;
        for i in 0..n_active {
            window.push_back(Spike {
                t_ms: t0 + i as f32 * 0.1,
                neuron: NeuronId(i),
            });
        }
    }
    // Plus a handful of chord edges: (i, i+17 mod n) for every 4th i.
    for i in (0..n_active).step_by(4) {
        let j = (i + 17) % n_active;
        let t = 2000.0 + i as f32 * 0.01;
        window.push_back(Spike {
            t_ms: t,
            neuron: NeuronId(i),
        });
        window.push_back(Spike {
            t_ms: t + 0.05,
            neuron: NeuronId(j),
        });
    }

    let mut active: Vec<NeuronId> = (0..n_active).map(NeuronId).collect();
    active.sort();
    active.dedup();

    // --- Dense reference (same construction as Observer::compute_fiedler).
    // Build adjacency, then Laplacian, then full Jacobi — ground truth.
    let n = active.len();
    let index_of = |id: NeuronId| -> Option<usize> { active.binary_search(&id).ok() };
    let tau = 5.0_f32;
    let mut a = vec![0.0_f32; n * n];
    let spikes: Vec<_> = window.iter().copied().collect();
    for (i, sa) in spikes.iter().enumerate() {
        let Some(ai) = index_of(sa.neuron) else {
            continue;
        };
        for sb in &spikes[i + 1..] {
            if (sb.t_ms - sa.t_ms).abs() > tau {
                break;
            }
            if let Some(bi) = index_of(sb.neuron) {
                if ai != bi {
                    a[ai * n + bi] += 1.0;
                    a[bi * n + ai] += 1.0;
                }
            }
        }
    }
    let mut l = vec![0.0_f32; n * n];
    for i in 0..n {
        let mut d = 0.0_f32;
        for j in 0..n {
            d += a[i * n + j];
            if i != j {
                l[i * n + j] = -a[i * n + j];
            }
        }
        l[i * n + i] = d;
    }
    let mut eigs = jacobi_symmetric(&l, n);
    eigs.sort_by(|x, y| x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal));
    // λ_2 is the first eigenvalue that clears a small positivity floor.
    let scale_hint = eigs.last().copied().unwrap_or(1.0).abs().max(1.0);
    let floor = 1e-6 * scale_hint;
    let dense = eigs
        .iter()
        .copied()
        .find(|v| *v > floor)
        .unwrap_or(0.0);

    // --- Sparse under test. ---
    let sparse = sparse_fiedler(&active, &window, 1024);

    eprintln!("sparse_vs_dense: n={n} dense(jacobi λ_2)={dense:.6} sparse(lanczos λ_2)={sparse:.6}");
    assert!(
        dense.is_finite() && sparse.is_finite(),
        "one path returned non-finite (dense={dense}, sparse={sparse})"
    );

    // Relative error vs ground-truth λ_2.
    let denom = dense.abs().max(1e-6);
    let rel = (sparse - dense).abs() / denom;
    assert!(
        rel <= 0.05,
        "sparse-vs-dense relative error {rel:.4} > 0.05 (dense={dense:.6}, sparse={sparse:.6})"
    );
}
