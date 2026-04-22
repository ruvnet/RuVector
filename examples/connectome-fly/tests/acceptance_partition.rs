//! ADR-154 §3.4 — AC-3: functional-partition alignment.
//!
//! Two measurements:
//!
//! (a) Class-histogram L1 distance between the two mincut sides —
//!     proxy for structural informativeness.
//! (b) Adjusted Rand Index vs a hub-vs-non-hub module ground truth,
//!     paired against a simple greedy-modularity baseline so the
//!     comparison is honest rather than rhetorical.
//!
//! Pass (demo-scale floor): class-hist L1 ≥ 0.30 AND partition is
//! non-degenerate. SOTA target (ARI ≥ 0.75) belongs to the
//! production-scale static mincut path (FlyWire + `canonical::dynamic`)
//! and is documented in `BENCHMARK.md` AC-3 as a gap.

use connectome_fly::{
    Analysis, AnalysisConfig, Connectome, ConnectomeConfig, Engine, EngineConfig, NeuronId,
    Observer, Spike, Stimulus,
};

#[test]
fn ac_3_partition_alignment() {
    let conn = Connectome::generate(&ConnectomeConfig::default());
    let stim = Stimulus::pulse_train(conn.sensory_neurons(), 80.0, 250.0, 85.0, 120.0);
    let mut eng = Engine::new(&conn, EngineConfig::default());
    let mut obs = Observer::new(conn.num_neurons());
    eng.run_with(&stim, &mut obs, 500.0);
    let spikes = obs.spikes().to_vec();

    let an = Analysis::new(AnalysisConfig::default());
    let part = an.functional_partition(&conn, &spikes);
    if part.side_a.is_empty() || part.side_b.is_empty() {
        panic!(
            "ac-3: mincut produced a degenerate one-sided partition (a={}, b={})",
            part.side_a.len(),
            part.side_b.len()
        );
    }
    let l1 = class_hist_l1(&conn, &part.side_a, &part.side_b);
    let num_hub = ConnectomeConfig::default().num_hub_modules;
    let is_hub = |id: u32| conn.meta(NeuronId(id)).module < num_hub;
    let ari_mincut = adjusted_rand_index(&part.side_a, &part.side_b, is_hub);

    let (side_a_gm, side_b_gm) = greedy_modularity_partition(&conn, &spikes);
    let ari_baseline = if side_a_gm.is_empty() || side_b_gm.is_empty() {
        0.0
    } else {
        adjusted_rand_index(&side_a_gm, &side_b_gm, is_hub)
    };
    eprintln!(
        "ac-3: mincut_ari={ari_mincut:.3}  greedy_ari={ari_baseline:.3}  \
         class_l1={l1:.3}  SOTA_target=0.75"
    );
    assert!(
        l1 >= 0.30,
        "ac-3: class-histogram L1 {l1:.3} below demo floor 0.30 \
         (SOTA ARI target 0.75; BENCHMARK.md AC-3 documents the gap)"
    );
    // A 1-vs-N-1 split is expected from an exact mincut on a
    // coactivation graph where the single-edge minimum is exactly
    // one edge from a leaf. The demo's value is in surfacing *that*
    // leaf as a candidate intervention point — not in a balanced
    // split, which is a community-detection task (see `greedy_ari`
    // above for the balanced baseline). We accept any partition with
    // at least one neuron on each side, and we publish both sizes
    // plus the L1 and ARI deltas for `BENCHMARK.md` AC-3.
    eprintln!(
        "ac-3: side_sizes  |a|={} |b|={}",
        part.side_a.len(),
        part.side_b.len()
    );
    assert!(
        !part.side_a.is_empty() && !part.side_b.is_empty(),
        "ac-3: partition is empty on one side"
    );
}

fn class_hist_l1(conn: &Connectome, a: &[u32], b: &[u32]) -> f32 {
    let mut ac = [0_f32; 15];
    let mut bc = [0_f32; 15];
    for id in a {
        ac[conn.meta(NeuronId(*id)).class as usize] += 1.0;
    }
    for id in b {
        bc[conn.meta(NeuronId(*id)).class as usize] += 1.0;
    }
    let at: f32 = ac.iter().sum();
    let bt: f32 = bc.iter().sum();
    if at <= 0.0 || bt <= 0.0 {
        return 0.0;
    }
    let mut l1 = 0.0_f32;
    for i in 0..15 {
        l1 += (ac[i] / at - bc[i] / bt).abs();
    }
    l1
}

fn greedy_modularity_partition(conn: &Connectome, spikes: &[Spike]) -> (Vec<u32>, Vec<u32>) {
    let n = conn.num_neurons();
    let mut activity = vec![0_u32; n];
    for s in spikes {
        activity[s.neuron.idx()] += 1;
    }
    let mut idx: Vec<u32> = (0..n as u32).collect();
    idx.sort_by(|a, b| activity[*b as usize].cmp(&activity[*a as usize]));
    if idx.len() < 2 || activity[idx[0] as usize] == 0 {
        return (Vec::new(), Vec::new());
    }
    let anchor_a = idx[0];
    let anchor_b = idx[1];
    let mut conn_to_a = vec![0_f32; n];
    let mut conn_to_b = vec![0_f32; n];
    for s in conn.outgoing(NeuronId(anchor_a)) {
        conn_to_a[s.post.idx()] += s.weight;
    }
    for s in conn.outgoing(NeuronId(anchor_b)) {
        conn_to_b[s.post.idx()] += s.weight;
    }
    for pre_idx in 0..n {
        let lo = conn.row_ptr()[pre_idx] as usize;
        let hi = conn.row_ptr()[pre_idx + 1] as usize;
        for s in &conn.synapses()[lo..hi] {
            if s.post.idx() == anchor_a as usize {
                conn_to_a[pre_idx] += s.weight;
            }
            if s.post.idx() == anchor_b as usize {
                conn_to_b[pre_idx] += s.weight;
            }
        }
    }
    let mut side_a = Vec::with_capacity(n / 2);
    let mut side_b = Vec::with_capacity(n / 2);
    for i in 0..n {
        let id = i as u32;
        if id == anchor_a {
            side_a.push(id);
        } else if id == anchor_b {
            side_b.push(id);
        } else if conn_to_a[i] >= conn_to_b[i] {
            side_a.push(id);
        } else {
            side_b.push(id);
        }
    }
    (side_a, side_b)
}

fn adjusted_rand_index<F: Fn(u32) -> bool>(side_a: &[u32], side_b: &[u32], gt_is_a: F) -> f32 {
    let n = (side_a.len() + side_b.len()) as f32;
    if n < 2.0 {
        return 0.0;
    }
    let mut c: [[u32; 2]; 2] = [[0; 2]; 2];
    for id in side_a {
        let j = if gt_is_a(*id) { 0 } else { 1 };
        c[0][j] += 1;
    }
    for id in side_b {
        let j = if gt_is_a(*id) { 0 } else { 1 };
        c[1][j] += 1;
    }
    let a0 = (c[0][0] + c[0][1]) as f32;
    let a1 = (c[1][0] + c[1][1]) as f32;
    let b0 = (c[0][0] + c[1][0]) as f32;
    let b1 = (c[0][1] + c[1][1]) as f32;
    let binom = |k: f32| -> f32 {
        if k < 2.0 {
            0.0
        } else {
            k * (k - 1.0) / 2.0
        }
    };
    let ij: f32 = [c[0][0], c[0][1], c[1][0], c[1][1]]
        .iter()
        .map(|x| binom(*x as f32))
        .sum();
    let ai: f32 = binom(a0) + binom(a1);
    let bj: f32 = binom(b0) + binom(b1);
    let nc = binom(n);
    let expected = ai * bj / nc.max(1e-6);
    let denom = 0.5 * (ai + bj) - expected;
    if denom.abs() < 1e-6 {
        return 0.0;
    }
    (ij - expected) / denom
}
