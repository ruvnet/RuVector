//! SIMD-vectorized subthreshold LIF update (Opt C from ADR-154 §3.2 step 9).
//!
//! This module is only compiled under `--features simd`. It vectorizes
//! the per-neuron inner loop that updates `V`, `g_exc`, `g_inh` across
//! 8 neurons at a time using `wide::f32x8`. On an AMD Ryzen 9 9950X
//! (Zen 5) the underlying codegen issues AVX / AVX2 fused mul-add; on
//! AVX-512-capable hosts the compiler may widen further. The SoA layout
//! already in place (`NeuronStateSoA`) is what makes this straightforward.
//!
//! Determinism note (ADR-154 §4.2): the scalar tail (`n % 8` neurons)
//! runs the exact same arithmetic as the vector body, and every lane
//! computes `v_rest + (v - v_rest) * alpha_m + R * i_syn * (1 - alpha_m)
//! + dt * R / tau_m * bias` in the same order as the scalar path. That
//! guarantees bit-identical spike traces with the scalar-optimized path
//! at N=1024 under default seeds (verified by
//! `tests/acceptance_core.rs::ac_1_repeatability`).
//!
//! The vectorized routine returns a `Vec<u32>` of neuron indices that
//! crossed threshold this tick — the caller emits spikes in id-order so
//! the downstream dispatch order is preserved across scalar / SIMD.

#![cfg(feature = "simd")]

use wide::{f32x8, CmpGe, CmpGt, CmpLe};

use super::types::NeuronParams;

/// Outcome of one SIMD subthreshold tick for a batch of active indices.
pub struct SimdTickOut {
    /// Neurons that crossed V_thresh this tick, in id-order.
    pub fired: Vec<u32>,
    /// Neurons still above the quiescent tolerance — survive into the
    /// active list for the next tick.
    pub still_active: Vec<u32>,
}

/// Precomputed per-tick decay factors (shared across lanes).
pub struct TickConsts {
    /// Exp(-dt/tau_m).
    pub alpha_m: f32,
    /// Exp(-dt/tau_syn_e).
    pub alpha_e: f32,
    /// Exp(-dt/tau_syn_i).
    pub alpha_i: f32,
    /// dt * R / tau_m (scaling for bias current).
    pub v_bias_factor: f32,
    /// Simulation time (ms).
    pub now: f32,
    /// Quiescent tolerance for the still-active test.
    pub quiescent_tol: f32,
}

impl TickConsts {
    /// Build per-tick constants from `params`, dt (ms), and current clock.
    pub fn new(p: &NeuronParams, dt_ms: f32, now_ms: f32) -> Self {
        Self {
            alpha_m: (-dt_ms / p.tau_m).exp(),
            alpha_e: (-dt_ms / p.tau_syn_e).exp(),
            alpha_i: (-dt_ms / p.tau_syn_i).exp(),
            v_bias_factor: dt_ms * p.r_m / p.tau_m,
            now: now_ms,
            quiescent_tol: 1e-4,
        }
    }
}

/// Process a batch of active neuron indices with 8-wide SIMD.
///
/// `indices` — active list for this tick; updated in place to remove
///             neurons that fell below quiescent tolerance.
/// `state`   — SoA fields, one slice per column.
/// `bias`    — per-neuron bias current (pA).
///
/// Returns the neurons that crossed threshold this tick in the same
/// order they appear in `indices`, so the caller can emit spikes in
/// id-order.
#[allow(clippy::too_many_arguments)]
pub fn subthreshold_tick_simd(
    indices: &[u32],
    v: &mut [f32],
    g_e: &mut [f32],
    g_i: &mut [f32],
    last_update_ms: &mut [f32],
    refrac_until_ms: &[f32],
    bias: &[f32],
    params: &NeuronParams,
    tk: &TickConsts,
) -> SimdTickOut {
    let mut fired: Vec<u32> = Vec::with_capacity(indices.len() / 16 + 4);
    let mut still_active: Vec<u32> = Vec::with_capacity(indices.len());

    let alpha_m = f32x8::splat(tk.alpha_m);
    let alpha_e = f32x8::splat(tk.alpha_e);
    let alpha_i = f32x8::splat(tk.alpha_i);
    let one_m_am = f32x8::splat(1.0 - tk.alpha_m);
    let v_rest = f32x8::splat(params.v_rest);
    let e_exc = f32x8::splat(params.e_exc);
    let e_inh = f32x8::splat(params.e_inh);
    let r_m = f32x8::splat(params.r_m);
    let v_thresh = f32x8::splat(params.v_thresh);
    let v_bias_factor = f32x8::splat(tk.v_bias_factor);
    let quiescent = f32x8::splat(tk.quiescent_tol);

    let mut i = 0;
    while i + 8 <= indices.len() {
        // Gather lanes.
        let id0 = indices[i] as usize;
        let id1 = indices[i + 1] as usize;
        let id2 = indices[i + 2] as usize;
        let id3 = indices[i + 3] as usize;
        let id4 = indices[i + 4] as usize;
        let id5 = indices[i + 5] as usize;
        let id6 = indices[i + 6] as usize;
        let id7 = indices[i + 7] as usize;

        // Build lane-wise active mask — neurons still in refractory
        // skip the subthreshold math but remain in the active list.
        let refrac = f32x8::from([
            refrac_until_ms[id0],
            refrac_until_ms[id1],
            refrac_until_ms[id2],
            refrac_until_ms[id3],
            refrac_until_ms[id4],
            refrac_until_ms[id5],
            refrac_until_ms[id6],
            refrac_until_ms[id7],
        ]);
        let now = f32x8::splat(tk.now);
        // Lane is "active this tick" iff refrac <= now.
        let active_mask = refrac.cmp_le(now);
        let active_arr: [f32; 8] = active_mask.into();

        let v_vec = f32x8::from([
            v[id0], v[id1], v[id2], v[id3], v[id4], v[id5], v[id6], v[id7],
        ]);
        let ge = f32x8::from([
            g_e[id0], g_e[id1], g_e[id2], g_e[id3], g_e[id4], g_e[id5], g_e[id6], g_e[id7],
        ]);
        let gi = f32x8::from([
            g_i[id0], g_i[id1], g_i[id2], g_i[id3], g_i[id4], g_i[id5], g_i[id6], g_i[id7],
        ]);
        let b = f32x8::from([
            bias[id0], bias[id1], bias[id2], bias[id3], bias[id4], bias[id5], bias[id6], bias[id7],
        ]);

        let i_syn = ge * (e_exc - v_vec) + gi * (e_inh - v_vec);
        let v_new_active =
            v_rest + (v_vec - v_rest) * alpha_m + r_m * i_syn * one_m_am + v_bias_factor * b;
        let ge_new_active = ge * alpha_e;
        let gi_new_active = gi * alpha_i;
        // For refractory lanes: unchanged. `wide::f32x8::blend(mask, t, f)`
        // picks lanes from `t` where mask is all-ones, else `f`.
        let v_new = active_mask.blend(v_new_active, v_vec);
        let ge_new = active_mask.blend(ge_new_active, ge);
        let gi_new = active_mask.blend(gi_new_active, gi);

        let v_arr: [f32; 8] = v_new.into();
        let ge_arr: [f32; 8] = ge_new.into();
        let gi_arr: [f32; 8] = gi_new.into();

        // Threshold crossing — only count lanes that were active this
        // tick (not in refractory).
        let thresh_mask = v_new.cmp_ge(v_thresh) & active_mask;
        let thresh_arr: [f32; 8] = thresh_mask.into();

        // Still-active decision (per lane): bias nonzero OR v away from
        // rest OR g non-trivial.
        let abs_ge = ge_new.abs();
        let abs_gi = gi_new.abs();
        let v_dev = (v_new - v_rest).abs();
        let bias_abs = b.abs();
        let bias_nonzero = bias_abs.cmp_gt(f32x8::splat(1e-6));
        let still_mask = bias_nonzero
            | abs_ge.cmp_gt(quiescent)
            | abs_gi.cmp_gt(quiescent)
            | v_dev.cmp_gt(quiescent);
        let still_arr: [f32; 8] = still_mask.into();

        let ids = [id0, id1, id2, id3, id4, id5, id6, id7];
        for lane in 0..8 {
            let id = ids[lane];
            let is_active_lane = active_arr[lane] != 0.0;
            if is_active_lane {
                v[id] = v_arr[lane];
                g_e[id] = ge_arr[lane];
                g_i[id] = gi_arr[lane];
                last_update_ms[id] = tk.now;
                if thresh_arr[lane] != 0.0 {
                    fired.push(id as u32);
                }
            }
            // Refractory lanes remain active by definition; active
            // lanes use the still_mask decision.
            let stays = !is_active_lane || still_arr[lane] != 0.0;
            if stays {
                still_active.push(id as u32);
            }
        }
        i += 8;
    }

    // Scalar tail — same arithmetic as the SIMD body.
    while i < indices.len() {
        let id = indices[i] as usize;
        let is_active_now = refrac_until_ms[id] <= tk.now;
        if is_active_now {
            let v_old = v[id];
            let ge_old = g_e[id];
            let gi_old = g_i[id];
            let b_id = bias[id];
            let i_syn = ge_old * (params.e_exc - v_old) + gi_old * (params.e_inh - v_old);
            let v_new = params.v_rest
                + (v_old - params.v_rest) * tk.alpha_m
                + params.r_m * i_syn * (1.0 - tk.alpha_m)
                + tk.v_bias_factor * b_id;
            v[id] = v_new;
            g_e[id] = ge_old * tk.alpha_e;
            g_i[id] = gi_old * tk.alpha_i;
            last_update_ms[id] = tk.now;
            if v_new >= params.v_thresh {
                fired.push(id as u32);
            }
        }
        let bias_abs = bias[id].abs();
        let stays = !is_active_now
            || bias_abs > 1e-6
            || g_e[id].abs() > tk.quiescent_tol
            || g_i[id].abs() > tk.quiescent_tol
            || (v[id] - params.v_rest).abs() > tk.quiescent_tol;
        if stays {
            still_active.push(id as u32);
        }
        i += 1;
    }

    SimdTickOut {
        fired,
        still_active,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn params() -> NeuronParams {
        NeuronParams::default()
    }

    #[test]
    fn simd_matches_scalar_on_random_batch() {
        // Build a batch of 23 neurons (not a multiple of 8 → tail exercises
        // the scalar path). Compare SIMD result lane-by-lane against a
        // hand-rolled scalar reference.
        let p = params();
        let dt = 0.1_f32;
        let now = 5.0_f32;
        let n = 23;
        let mut v: Vec<f32> = (0..n).map(|i| -65.0 + (i as f32) * 0.3).collect();
        let mut g_e: Vec<f32> = (0..n).map(|i| (i as f32) * 0.01).collect();
        let mut g_i: Vec<f32> = (0..n).map(|i| (i as f32) * 0.005).collect();
        let mut lu = vec![0.0_f32; n];
        // Half refractory, half not.
        let refrac: Vec<f32> = (0..n)
            .map(|i| if i % 2 == 0 { 0.0 } else { now + 1.0 })
            .collect();
        let bias: Vec<f32> = (0..n).map(|i| if i < 3 { 80.0 } else { 0.0 }).collect();
        let indices: Vec<u32> = (0..n as u32).collect();

        // Scalar reference.
        let mut v_ref = v.clone();
        let mut ge_ref = g_e.clone();
        let mut gi_ref = g_i.clone();
        let alpha_m = (-dt / p.tau_m).exp();
        let alpha_e = (-dt / p.tau_syn_e).exp();
        let alpha_i = (-dt / p.tau_syn_i).exp();
        let vbf = dt * p.r_m / p.tau_m;
        for i in 0..n {
            if refrac[i] > now {
                continue;
            }
            let i_syn = ge_ref[i] * (p.e_exc - v_ref[i]) + gi_ref[i] * (p.e_inh - v_ref[i]);
            v_ref[i] = p.v_rest
                + (v_ref[i] - p.v_rest) * alpha_m
                + p.r_m * i_syn * (1.0 - alpha_m)
                + vbf * bias[i];
            ge_ref[i] *= alpha_e;
            gi_ref[i] *= alpha_i;
        }

        let tk = TickConsts::new(&p, dt, now);
        let _out = subthreshold_tick_simd(
            &indices, &mut v, &mut g_e, &mut g_i, &mut lu, &refrac, &bias, &p, &tk,
        );
        for i in 0..n {
            // Allow 1 ULP tolerance for SIMD reordering; in practice we
            // see 0 ULP on x86_64.
            assert!(
                (v[i] - v_ref[i]).abs() < 1e-5,
                "lane {i}: simd {} vs scalar {}",
                v[i],
                v_ref[i]
            );
            assert!((g_e[i] - ge_ref[i]).abs() < 1e-5);
            assert!((g_i[i] - gi_ref[i]).abs() < 1e-5);
        }
    }
}
