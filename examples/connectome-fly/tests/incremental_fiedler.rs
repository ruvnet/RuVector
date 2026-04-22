//! Correctness + microbench for the incremental Fiedler accumulator.
//!
//! ADR-154 §16 lever 3. The accumulator replaces the O(S²) per-detect
//! pair sweep in `Observer::compute_fiedler` with an amortised
//! `BTreeMap<(NeuronId, NeuronId), u32>` of τ-coincident pair counts,
//! updated in `Observer::on_spike`.
//!
//! Tests:
//!
//! 1. `incremental_adjacency_matches_pair_sweep` — on a ~200-spike
//!    fixture, the dense `n × n` adjacency produced by
//!    `snapshot_adjacency` matches the exact O(S²) pair-sweep
//!    adjacency byte-for-byte (counts are integers, so "within 1e-4
//!    relative" collapses to exact equality at this scale).
//! 2. `incremental_fiedler_matches_pair_sweep_within_1e_4` — full
//!    `approx_fiedler_power` run on both adjacencies yields the same
//!    Fiedler value within 1e-4 relative error (per the task spec).
//! 3. `incremental_accumulator_is_bit_deterministic` — repeated
//!    construction from identical spike streams produces identical
//!    adjacency vectors (AC-1-style).
//! 4. `push_expire_round_trip_is_empty` — after pushing every spike
//!    in a long stream and expiring them all, the accumulator is
//!    empty (no leaked edges).
//! 5. `per_detect_microbench` — saturated-window fixture measures
//!    per-detect wallclock of the incremental path vs the pair-sweep
//!    path. Prints the speedup; fails only if the incremental path is
//!    *slower* than the pair-sweep path (the regression guard).

use std::collections::VecDeque;
use std::time::Instant;

use connectome_fly::observer::eigensolver::approx_fiedler_power;
use connectome_fly::observer::incremental_fiedler::{IncrementalCofireAccumulator, COFIRE_TAU_MS};
use connectome_fly::{NeuronId, Spike};

// ---------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------

/// Reproduce the pre-commit pair-sweep adjacency: for every pair of
/// window spikes within τ, increment the `(ai, bi)` entry. Used as
/// the reference the incremental accumulator must match.
fn pair_sweep_adjacency(active: &[NeuronId], window: &VecDeque<Spike>) -> Vec<f32> {
    let n = active.len();
    let mut a = vec![0.0_f32; n * n];
    let spikes: Vec<_> = window.iter().copied().collect();
    for (i, sa) in spikes.iter().enumerate() {
        let Ok(ai) = active.binary_search(&sa.neuron) else {
            continue;
        };
        for sb in &spikes[i + 1..] {
            if (sb.t_ms - sa.t_ms).abs() > COFIRE_TAU_MS {
                break;
            }
            let Ok(bi) = active.binary_search(&sb.neuron) else {
                continue;
            };
            if ai == bi {
                continue;
            }
            a[ai * n + bi] += 1.0;
            a[bi * n + ai] += 1.0;
        }
    }
    a
}

/// Build an accumulator by replaying a spike stream through the
/// `push` / `expire` cycle that `Observer::on_spike` performs,
/// honouring a sliding `window_ms`. Returns the accumulator and the
/// final window.
fn replay(stream: &[Spike], window_ms: f32) -> (IncrementalCofireAccumulator, VecDeque<Spike>) {
    let mut acc = IncrementalCofireAccumulator::new();
    let mut window: VecDeque<Spike> = VecDeque::new();
    for &s in stream {
        window.push_back(s);
        let prior_len = window.len() - 1;
        acc.push(s, window.iter().take(prior_len));
        let cutoff = s.t_ms - window_ms;
        while let Some(front) = window.front() {
            if front.t_ms < cutoff {
                let q = window.pop_front().unwrap();
                acc.expire(q, window.iter());
            } else {
                break;
            }
        }
    }
    (acc, window)
}

/// Extract sorted-deduped active-neuron list from a window.
fn active_from_window(window: &VecDeque<Spike>) -> Vec<NeuronId> {
    let mut v: Vec<NeuronId> = window.iter().map(|s| s.neuron).collect();
    v.sort();
    v.dedup();
    v
}

/// Seeded LCG (matches `Observer`-style determinism needs without
/// pulling rand into dev-deps). Returns a generator producing
/// reproducible u32 values.
fn lcg(seed: u64) -> impl FnMut() -> u32 {
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0xDEAD_BEEF_CAFE_F00D;
    move || {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (s >> 32) as u32
    }
}

/// Build a ~200-spike fixture: two weakly-coupled clusters, with
/// intra-cluster bursts and a handful of cross-cluster co-fires.
/// Deterministic.
fn fixture_200_spikes() -> Vec<Spike> {
    let mut out: Vec<Spike> = Vec::with_capacity(220);
    let mut rng = lcg(42);
    // Cluster A: neurons 0..8, bursts every 8 ms for 100 ms.
    for k in 0..12 {
        let t0 = 5.0 + k as f32 * 8.0;
        for i in 0..8_u32 {
            out.push(Spike {
                t_ms: t0 + (rng() % 1000) as f32 * 0.002,
                neuron: NeuronId(i),
            });
        }
    }
    // Cluster B: neurons 8..16, offset 4 ms from A.
    for k in 0..12 {
        let t0 = 9.0 + k as f32 * 8.0;
        for i in 8..16_u32 {
            out.push(Spike {
                t_ms: t0 + (rng() % 1000) as f32 * 0.002,
                neuron: NeuronId(i),
            });
        }
    }
    // Cross-cluster bridge spikes: neurons 3 ↔ 11 co-fire.
    for k in 0..10 {
        let t = 6.0 + k as f32 * 10.0;
        out.push(Spike {
            t_ms: t,
            neuron: NeuronId(3),
        });
        out.push(Spike {
            t_ms: t + 0.3,
            neuron: NeuronId(11),
        });
    }
    // Non-decreasing time order (the engine emits spikes monotonically
    // by t_ms; our incremental push assumes it).
    out.sort_by(|a, b| a.t_ms.partial_cmp(&b.t_ms).unwrap());
    out
}

/// Saturated-window fixture used by the microbench. ~21 000 spikes
/// inside a 50 ms sliding window over N=1024 neurons — the same
/// regime as the AC-4 saturated bench after commit 10.
fn fixture_saturated_window(n_neurons: u32, target_len: usize) -> Vec<Spike> {
    // Uniformly distribute spikes in [0, 50] ms across all neurons,
    // with enough density to hit `target_len` inside the 50 ms window.
    let mut out: Vec<Spike> = Vec::with_capacity(target_len);
    let mut rng = lcg(0xABCD_1234);
    for i in 0..target_len {
        let t = (i as f32 / target_len as f32) * 50.0;
        let neuron = NeuronId(rng() % n_neurons);
        out.push(Spike { t_ms: t, neuron });
    }
    // Time order is already monotone by construction.
    out
}

// ---------------------------------------------------------------------
// Equivalence tests
// ---------------------------------------------------------------------

#[test]
fn incremental_adjacency_matches_pair_sweep() {
    let stream = fixture_200_spikes();
    // Use a window large enough to retain all spikes — isolates the
    // push/expire round-trip from the comparison. (Spec window is
    // 50 ms; our stream spans ~100 ms, so a 200 ms window keeps
    // everything and still exercises the full push path.)
    let (acc, window) = replay(&stream, 200.0);
    let active = active_from_window(&window);
    let a_incr = acc.snapshot_adjacency(&active);
    let a_ref = pair_sweep_adjacency(&active, &window);
    assert_eq!(a_incr.len(), a_ref.len());
    // Counts are integers; expected byte-exact.
    for (i, (x, y)) in a_incr.iter().zip(a_ref.iter()).enumerate() {
        assert_eq!(
            x.to_bits(),
            y.to_bits(),
            "adjacency mismatch at [{i}]: incr={x} ref={y}"
        );
    }
}

#[test]
fn incremental_adjacency_matches_after_window_expiry() {
    // Sliding-window case: spikes older than `window_ms` must have
    // been correctly decremented out of the accumulator.
    let stream = fixture_200_spikes();
    let (acc, window) = replay(&stream, 50.0); // real observer window
    let active = active_from_window(&window);
    let a_incr = acc.snapshot_adjacency(&active);
    let a_ref = pair_sweep_adjacency(&active, &window);
    for (i, (x, y)) in a_incr.iter().zip(a_ref.iter()).enumerate() {
        assert_eq!(
            x.to_bits(),
            y.to_bits(),
            "sliding-window adjacency mismatch at [{i}]: incr={x} ref={y}"
        );
    }
}

#[test]
fn incremental_fiedler_matches_pair_sweep_within_1e_4() {
    // End-to-end: run `approx_fiedler_power` on both adjacencies and
    // assert the Fiedler value agrees within 1e-4 relative error (the
    // task's acceptance threshold).
    let stream = fixture_200_spikes();
    let (acc, window) = replay(&stream, 200.0);
    let active = active_from_window(&window);
    let n = active.len();
    assert!(n >= 2, "fixture must produce at least 2 active neurons");
    let a_incr = acc.snapshot_adjacency(&active);
    let a_ref = pair_sweep_adjacency(&active, &window);
    let fiedler_incr = approx_fiedler_power(&a_incr, n);
    let fiedler_ref = approx_fiedler_power(&a_ref, n);
    // Since a_incr == a_ref bit-for-bit, the eigensolver must produce
    // identical outputs. We still assert within 1e-4 relative so the
    // test documents the spec.
    let denom = fiedler_ref.abs().max(1e-6);
    let rel = (fiedler_incr - fiedler_ref).abs() / denom;
    assert!(
        rel < 1e-4,
        "fiedler disagreement: incr={fiedler_incr} ref={fiedler_ref} rel={rel}"
    );
}

#[test]
fn incremental_accumulator_is_bit_deterministic() {
    // AC-1 analogue: repeated construction from the same spike stream
    // produces the same adjacency byte-for-byte. BTreeMap iteration
    // order must not depend on allocator state or hash randomisation.
    let stream = fixture_200_spikes();
    let (acc_a, win_a) = replay(&stream, 50.0);
    let (acc_b, win_b) = replay(&stream, 50.0);
    let active_a = active_from_window(&win_a);
    let active_b = active_from_window(&win_b);
    assert_eq!(active_a, active_b);
    let aa = acc_a.snapshot_adjacency(&active_a);
    let ab = acc_b.snapshot_adjacency(&active_b);
    assert_eq!(aa.len(), ab.len());
    for (i, (x, y)) in aa.iter().zip(ab.iter()).enumerate() {
        assert_eq!(x.to_bits(), y.to_bits(), "determinism drift at [{i}]");
    }
    // Sparse-snapshot triples also stable.
    let sa: Vec<_> = acc_a.snapshot_sparse().collect();
    let sb: Vec<_> = acc_b.snapshot_sparse().collect();
    assert_eq!(sa, sb);
}

#[test]
fn push_expire_round_trip_is_empty() {
    // After every spike in the stream has entered and then exited the
    // window, the accumulator must be empty (no leaked edges).
    let stream = fixture_200_spikes();
    // 0 ms window → every spike expires immediately after the next
    // push moves the cutoff past it. But we need the window to retain
    // at least the newly-pushed spike; use a vanishingly-small window.
    // Instead: replay normally at 50 ms, then drain all remaining
    // spikes with synthetic "ages" well past the window.
    let (mut acc, mut window) = replay(&stream, 50.0);
    // Drain: pop every remaining spike from the front, calling expire.
    while let Some(q) = window.pop_front() {
        acc.expire(q, window.iter());
    }
    assert_eq!(
        acc.edge_count(),
        0,
        "push/expire left {} leaked edges",
        acc.edge_count()
    );
}

// ---------------------------------------------------------------------
// Microbench — not a correctness test, but fails if the incremental
// path is slower than the pair-sweep path.
// ---------------------------------------------------------------------

/// Run the pair-sweep and incremental paths on the same saturated
/// window and print per-detect wallclock. Included in the test suite
/// (not a Criterion bench) because `benches/` would need Criterion
/// wiring in `Cargo.toml` and the spec excludes Cargo.toml edits.
#[test]
fn per_detect_microbench() {
    let n_neurons: u32 = 1024;
    // Target ~21 000 spikes inside a 50 ms window — the saturated
    // regime cited in the task. A straight 50 ms fixture runs the
    // accumulator through ~21k pushes; for the pair-sweep baseline
    // we just time one `compute_fiedler`-equivalent sweep over the
    // populated window.
    let stream = fixture_saturated_window(n_neurons, 21_000);
    let (acc, window) = replay(&stream, 50.0);
    let active = active_from_window(&window);
    let n = active.len();
    println!(
        "microbench: saturated window = {} spikes, active = {} neurons, \
         accumulator edges = {}",
        window.len(),
        n,
        acc.edge_count()
    );

    // --- Pair-sweep baseline: allocate adjacency + walk every pair. ---
    // Warmup + median-of-5. The inner `compute_fiedler` in
    // `Observer::compute_fiedler` also runs the eigensolver; we time
    // only the adjacency build (the work the accumulator replaces),
    // so the measured ratio is the adjacency-build speedup alone.
    // The eigensolver cost is identical on both paths.
    let runs = 5;
    let mut sweep_ns = Vec::with_capacity(runs);
    for _ in 0..runs {
        let t0 = Instant::now();
        let _ = pair_sweep_adjacency(&active, &window);
        sweep_ns.push(t0.elapsed().as_nanos());
    }
    let mut incr_ns = Vec::with_capacity(runs);
    for _ in 0..runs {
        let t0 = Instant::now();
        let _ = acc.snapshot_adjacency(&active);
        incr_ns.push(t0.elapsed().as_nanos());
    }
    sweep_ns.sort();
    incr_ns.sort();
    let sweep_med = sweep_ns[runs / 2] as f64;
    let incr_med = incr_ns[runs / 2] as f64;
    let speedup = sweep_med / incr_med.max(1.0);
    println!(
        "per-detect adjacency build (median of {runs}):\n\
         \tpair-sweep  = {:.3} ms\n\
         \tincremental = {:.3} ms\n\
         \tspeedup     = {:.2}x",
        sweep_med / 1.0e6,
        incr_med / 1.0e6,
        speedup,
    );
    // Regression guard: the incremental path must not be slower. We
    // do not assert >= 5× here — that is a target, not a hard bound.
    // Environment noise (CI vs local) can shift the ratio; the printed
    // number is the source of truth for the commit message.
    assert!(
        speedup > 1.0,
        "incremental path regressed vs pair-sweep: speedup={speedup:.2}x"
    );
}
