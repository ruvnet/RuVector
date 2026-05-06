#![allow(clippy::needless_range_loop)]
//! ADR-154 §3.4 — AC-5: causal perturbation.
//!
//! Core differentiating claim. Removing the top-K edges surfaced by
//! `ruvector-mincut` changes the downstream population-firing pattern
//! by more than σ of a random-cut baseline. Pass (demo-scale floor):
//! `mean_cut > mean_rand` and `z_cut ≥ 1.5σ` and `z_cut > z_rand`.
//! SOTA target (ADR-154 §3.4 AC-5): z_cut ≥ 5σ, z_rand ≤ 1σ.

use connectome_fly::{
    Analysis, AnalysisConfig, Connectome, ConnectomeConfig, Engine, EngineConfig, Observer, Spike,
    Stimulus,
};

fn run_one(conn: &Connectome, stim: &Stimulus, t_end_ms: f32) -> Vec<Spike> {
    let mut eng = Engine::new(conn, EngineConfig::default());
    let mut obs = Observer::new(conn.num_neurons());
    eng.run_with(stim, &mut obs, t_end_ms);
    obs.spikes().to_vec()
}

fn late_window_rate(spikes: &[Spike], t_start: f32, t_end: f32, n: usize) -> f32 {
    let mut count = 0_u32;
    for s in spikes {
        if s.t_ms >= t_start && s.t_ms < t_end {
            count += 1;
        }
    }
    let dur_s = ((t_end - t_start) / 1000.0).max(1e-3);
    count as f32 / (n as f32 * dur_s)
}

fn stddev(xs: &[f32]) -> f32 {
    let m: f32 = xs.iter().copied().sum::<f32>() / xs.len() as f32;
    let v: f32 = xs.iter().map(|x| (x - m) * (x - m)).sum::<f32>() / xs.len() as f32;
    v.sqrt()
}

#[test]
fn ac_5_causal_perturbation() {
    let conn = Connectome::generate(&ConnectomeConfig::default());
    let stim = Stimulus::pulse_train(conn.sensory_neurons(), 80.0, 250.0, 85.0, 120.0);
    let control_spikes = run_one(&conn, &stim, 400.0);
    let an = Analysis::new(AnalysisConfig::default());
    let part = an.functional_partition(&conn, &control_spikes);
    if part.side_a.is_empty() || part.side_b.is_empty() {
        panic!("ac-5: degenerate partition; cannot derive boundary edges");
    }
    let side_a_set: std::collections::HashSet<u32> = part.side_a.iter().copied().collect();
    let row_ptr = conn.row_ptr();
    let syn = conn.synapses();
    let mut boundary: Vec<usize> = Vec::new();
    let mut interior: Vec<usize> = Vec::new();
    for pre_idx in 0..conn.num_neurons() {
        let s = row_ptr[pre_idx] as usize;
        let e = row_ptr[pre_idx + 1] as usize;
        for flat in s..e {
            let post_idx = syn[flat].post.idx();
            let a = side_a_set.contains(&(pre_idx as u32));
            let b = side_a_set.contains(&(post_idx as u32));
            if a != b {
                boundary.push(flat);
            } else {
                interior.push(flat);
            }
        }
    }
    let k = 100_usize.min(boundary.len()).min(interior.len());
    assert!(
        k > 0,
        "ac-5: not enough boundary/interior edges ({} boundary, {} interior)",
        boundary.len(),
        interior.len()
    );
    let perturbed_boundary = conn.with_synapse_weights_zeroed(&boundary[..k]);
    let perturbed_interior = conn.with_synapse_weights_zeroed(&interior[..k]);

    let mut deltas_cut: Vec<f32> = Vec::new();
    let mut deltas_rand: Vec<f32> = Vec::new();
    for trial in 0..5_u32 {
        let phase = trial as f32 * 0.4;
        let stim_t =
            Stimulus::pulse_train(conn.sensory_neurons(), 80.0 + phase, 250.0, 85.0, 120.0);
        let ctrl_spikes = run_one(&conn, &stim_t, 400.0);
        let ctrl = late_window_rate(&ctrl_spikes, 300.0, 400.0, conn.num_neurons());
        let cut_spikes = run_one(&perturbed_boundary, &stim_t, 400.0);
        let cut = late_window_rate(&cut_spikes, 300.0, 400.0, conn.num_neurons());
        let rnd_spikes = run_one(&perturbed_interior, &stim_t, 400.0);
        let rnd = late_window_rate(&rnd_spikes, 300.0, 400.0, conn.num_neurons());
        deltas_cut.push((cut - ctrl).abs());
        deltas_rand.push((rnd - ctrl).abs());
    }
    let sigma = stddev(&deltas_rand).max(1e-3);
    let mean_cut = deltas_cut.iter().copied().sum::<f32>() / deltas_cut.len() as f32;
    let mean_rand = deltas_rand.iter().copied().sum::<f32>() / deltas_rand.len() as f32;
    let z_cut = mean_cut / sigma;
    let z_rand = mean_rand / sigma;
    eprintln!(
        "ac-5: mean_cut={mean_cut:.3} Hz  mean_rand={mean_rand:.3} Hz  \
         sigma={sigma:.3} Hz  z_cut={z_cut:.2}  z_rand={z_rand:.2}"
    );
    assert!(
        mean_cut > mean_rand,
        "ac-5: mincut-edge perturbation did not exceed random perturbation \
         (cut_mean={mean_cut:.3} rand_mean={mean_rand:.3})"
    );
    assert!(
        z_cut >= 1.5,
        "ac-5: cut perturbation z-score below 1.5σ bound (z_cut={z_cut:.3})"
    );
    assert!(
        z_cut > z_rand,
        "ac-5: cut perturbation did not exceed random-perturbation baseline \
         (z_cut={z_cut:.3} z_rand={z_rand:.3})"
    );
}
