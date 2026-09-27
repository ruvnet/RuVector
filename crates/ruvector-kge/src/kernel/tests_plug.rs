//! `GemmOneToN` against the recipe lane's `OneToN` seam: its contract, the
//! `NaiveOneToN` reference, and end-to-end `Trainer::fit_with_kernel`.

use super::tests::{assert_close, rand, to64};
use super::{one_n_softmax_ce, Reduction, Workspace};
use crate::train::{NaiveOneToN, OneToN};
use crate::{Tables, Triple};

/// `GemmOneToN` honours the recipe lane's `OneToN` contract: grad_q
/// overwritten with `scale·dL/dQ`, grad_ent accumulated (`+=`), unscaled loss.
#[test]
fn gemm_one_to_n_contract() {
    let (n, dim, b) = (530usize, 12usize, 9usize);
    let ents = rand(n * dim, 301, 0.5);
    let q = rand(b * dim, 303, 0.5);
    let targets: Vec<u32> = (0..b as u32).map(|i| (i * 57) % n as u32).collect();
    let scale = 0.25f32;
    let mut ws = Workspace::new();
    let (mut gq0, mut ge0) = (vec![0.0; b * dim], vec![0.0; n * dim]);
    let summed = one_n_softmax_ce(
        &q,
        &ents,
        dim,
        &targets,
        Reduction::Sum,
        &mut ws,
        &mut gq0,
        &mut ge0,
    )
    .unwrap();

    let k = super::GemmOneToN::new();
    let (mut gq, mut ge) = (vec![7.0; b * dim], vec![1.0; n * dim]);
    let l = k
        .softmax_ce(&q, &ents, dim, &targets, scale, &mut gq, &mut ge)
        .unwrap();
    assert!(
        (l - summed).abs() <= 1e-6 * summed.abs(),
        "unscaled loss {l} vs {summed}"
    );
    for (g, g0) in gq.iter().zip(&gq0) {
        assert!((g - scale * g0).abs() <= 1e-6, "grad_q overwritten+scaled");
    }
    for (g, g0) in ge.iter().zip(&ge0) {
        assert!(
            (g - (1.0 + scale * g0)).abs() <= 1e-6,
            "grad_ent accumulated+scaled"
        );
    }
}

/// Against the recipe lane's reference implementation of the same seam.
#[test]
fn gemm_one_to_n_matches_naive_one_to_n() {
    for &(n, dim, b) in &[(17usize, 6usize, 3usize), (700, 20, 45)] {
        let ents = rand(n * dim, 401 + n as u64, 0.7);
        let q = rand(b * dim, 403 + b as u64, 0.7);
        let targets: Vec<u32> = (0..b as u32).map(|i| (i * 31 + 5) % n as u32).collect();
        let prior = rand(n * dim, 409, 0.1);
        let run = |k: &dyn OneToN| {
            let (mut gq, mut ge) = (vec![0.0; b * dim], prior.clone());
            let l = k
                .softmax_ce(&q, &ents, dim, &targets, 0.5, &mut gq, &mut ge)
                .unwrap();
            (l, gq, ge)
        };
        let (l0, gq0, ge0) = run(&NaiveOneToN);
        let (l1, gq1, ge1) = run(&super::GemmOneToN::new());
        assert!(
            (l1 - l0).abs() <= 1e-5 * l0.abs().max(1.0),
            "loss {l1} vs {l0}"
        );
        assert_close("grad_q vs naive", &gq1, &to64(&gq0), 1e-5);
        assert_close("grad_ent vs naive", &ge1, &to64(&ge0), 1e-5);
    }
}

/// End-to-end through the recipe trainer: ComplEx-N3-R (reciprocal, dense
/// Adagrad, mean) trained with the GEMM kernel ends within 1e-4 of the same
/// run on `NaiveOneToN` (different, but each fixed, summation orders).
#[test]
fn fit_with_gemm_kernel_matches_naive_kernel() {
    use crate::data::TripleStore;
    use crate::scorer::ComplEx;
    use crate::train::{LossKind, StateLayout, TrainConfig, Trainer};
    let (ne, nr, d) = (40usize, 3usize, 16usize);
    let triples: Vec<Triple> = (0..60u32)
        .map(|i| Triple::new(i % ne as u32, i % nr as u32, (i * 7 + 3) % ne as u32))
        .collect();
    let store = TripleStore::with_counts(triples, Some(ne), Some(nr)).unwrap();
    let cfg: TrainConfig = serde_json::from_str(
        r#"{"loss":{"kind":"one_vs_all"},"reciprocal":true,"n3_form":"moduli",
            "loss_reduction":"mean","init":{"kind":"normal","scale":0.1},
            "optim_state":"dense","n3_lambda":0.01,"dims":16,"epochs":4,
            "batch_size":16,"lr":0.1,"seed":3}"#,
    )
    .unwrap();
    assert_eq!(cfg.loss, LossKind::OneVsAll);
    assert_eq!(cfg.optim_state, StateLayout::Dense);
    let sc = ComplEx::new(d).unwrap();
    let run = |k: &dyn OneToN| {
        let mut t = Tables::new(ne, 2 * nr, d, 5);
        let mut losses = Vec::new();
        Trainer::fit_with_kernel(&mut t, &sc, &store, &cfg, k, |p| losses.push(p.loss)).unwrap();
        (t, losses)
    };
    let (t0, l0) = run(&NaiveOneToN);
    let (t1, l1) = run(&super::GemmOneToN::new());
    for (a, b) in l0.iter().zip(&l1) {
        assert!(
            (a - b).abs() <= 1e-4 * a.abs().max(1.0),
            "epoch loss {b} vs {a}"
        );
    }
    assert_close(
        "entities after fit",
        t1.entities_raw(),
        &to64(t0.entities_raw()),
        1e-4,
    );
    assert_close(
        "relations after fit",
        t1.relations_raw(),
        &to64(t0.relations_raw()),
        1e-4,
    );
}
