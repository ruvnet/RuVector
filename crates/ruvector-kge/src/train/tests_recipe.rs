//! M1 recipe tests: finite-difference checks for every new loss/regulariser
//! term, the batched 1-N path against the per-triple oracle, layout
//! equivalence of the optimizer state, reciprocal conventions, and an
//! end-to-end ComplEx-N3-R sanity run.

use super::grad::testing::DistMult;
use super::optim::Grads;
use super::*;
use crate::data::TripleStore;
use crate::scorer::complex::{conj, ComplEx};
use crate::{eval, Tables, Triple};

const D: usize = 8;
const NE: usize = 12;
const NR: usize = 3; // base relations; reciprocal tables hold 2·NR rows

fn cx() -> ComplEx {
    ComplEx::new(D).unwrap()
}

/// O(1)-scale random tables (Xavier bound ≈ 0.61 at d=8), so the cubic N3
/// term is not vanishingly small as it is at the 1e-3 recipe init.
fn fd_tables() -> Tables {
    Tables::new(NE, 2 * NR, D, 0xF00D)
}

fn examples() -> Vec<Triple> {
    vec![
        Triple::new(0, 0, 1),
        Triple::new(2, 1, 3),
        Triple::new(4, 2, 0),
        Triple::new(1, 3, 5), // an inverse row (3 = 0 + NR)
        Triple::new(7, 5, 7), // self-loop, inverse row
    ]
}

fn row_grad(g: Option<&[f32]>) -> Vec<f32> {
    g.map(|v| v.to_vec()).unwrap_or_else(|| vec![0.0; D])
}

/// Central-difference check of every coordinate of every row against the
/// gradients `step` accumulates. `step` returns the (unscaled) loss.
fn fd_check(name: &str, tables: &Tables, step: &dyn Fn(&Tables, &mut Grads) -> f32) {
    let mut g = Grads::new(D);
    step(tables, &mut g);
    let h = 1e-2f32;
    let eval = |t: &Tables| {
        let mut scratch = Grads::new(D);
        step(t, &mut scratch)
    };
    for is_rel in [false, true] {
        let rows = if is_rel {
            tables.num_relations()
        } else {
            tables.num_entities()
        };
        for id in 0..rows as u32 {
            let analytic = row_grad(if is_rel { g.relation(id) } else { g.entity(id) });
            for j in 0..D {
                let bump = |delta: f32| {
                    let mut t = tables.clone();
                    let raw = if is_rel {
                        t.relations_raw_mut()
                    } else {
                        t.entities_raw_mut()
                    };
                    raw[id as usize * D + j] += delta;
                    eval(&t)
                };
                let fd = (bump(h) - bump(-h)) / (2.0 * h);
                assert!(
                    (fd - analytic[j]).abs() <= 2e-3 + 1e-2 * fd.abs(),
                    "{name}: {} {id}[{j}] fd {fd} vs analytic {}",
                    if is_rel { "rel" } else { "ent" },
                    analytic[j]
                );
            }
        }
    }
}

#[test]
fn fd_one_vs_all_tail_complex() {
    let (t, sc) = (fd_tables(), cx());
    fd_check("1vsAll-tail", &t, &|tb, g| {
        examples()
            .iter()
            .map(|&e| loss::one_vs_all_tail_step(tb, &sc, e, g).unwrap())
            .sum()
    });
}

#[test]
fn fd_one_vs_all_both_sides_complex() {
    let (t, sc) = (fd_tables(), cx());
    fd_check("1vsAll-both", &t, &|tb, g| {
        examples()
            .iter()
            .map(|&e| loss::one_vs_all_step(tb, &sc, e, g).unwrap())
            .sum()
    });
}

#[test]
fn fd_relation_prediction() {
    let (t, sc) = (fd_tables(), cx());
    fd_check("rp", &t, &|tb, g| {
        examples()
            .iter()
            .map(|&e| rp::relation_prediction_step(tb, &sc, e, 0.7, g).unwrap())
            .sum()
    });
}

#[test]
fn fd_n3_moduli_and_elementwise() {
    let t = fd_tables();
    for form in [N3Form::Moduli, N3Form::Elementwise] {
        fd_check(&format!("n3 {form:?}"), &t, &|tb, g| {
            let mut buf = Vec::new();
            examples()
                .iter()
                .map(|&e| apply_n3(tb, e, 0.3, form, g, &mut buf).unwrap())
                .sum()
        });
    }
    // Moduli differs from elementwise, and a zero pair has zero gradient.
    let mut buf = Vec::new();
    let pm = loss::n3_moduli_grad(&[3.0, 0.0, 4.0, 0.0], 1.0, &mut buf);
    assert!((pm - 125.0).abs() < 1e-4, "|3+4i|^3 = 125, got {pm}");
    assert_eq!(buf[1], 0.0);
    assert_eq!(buf[3], 0.0);
    assert!((buf[0] - 3.0 * 5.0 * 3.0).abs() < 1e-4);
}

/// Mean reduction scales every accumulated gradient by `1/B`.
#[test]
fn mean_scale_multiplies_all_grads() {
    let (t, sc) = (fd_tables(), cx());
    let run = |scale: f32| {
        let mut g = Grads::new(D);
        g.set_scale(scale);
        let mut buf = Vec::new();
        for &e in &examples() {
            loss::one_vs_all_tail_step(&t, &sc, e, &mut g).unwrap();
            rp::relation_prediction_step(&t, &sc, e, 0.5, &mut g).unwrap();
            apply_n3(&t, e, 0.1, N3Form::Moduli, &mut g, &mut buf).unwrap();
        }
        g
    };
    let (full, quarter) = (run(1.0), run(0.25));
    for id in 0..NE as u32 {
        let a = row_grad(full.entity(id));
        let b = row_grad(quarter.entity(id));
        for (x, y) in a.iter().zip(&b) {
            assert_eq!(x * 0.25, *y, "power-of-two scale is exact");
        }
    }
}

fn assert_grads_close(a: &Grads, b: &Grads, what: &str) {
    for id in 0..NE as u32 {
        for (x, y) in row_grad(a.entity(id)).iter().zip(&row_grad(b.entity(id))) {
            assert!(
                (x - y).abs() <= 1e-4 + 1e-4 * x.abs(),
                "{what} ent {id}: {x} vs {y}"
            );
        }
    }
    for id in 0..(2 * NR) as u32 {
        for (x, y) in row_grad(a.relation(id))
            .iter()
            .zip(&row_grad(b.relation(id)))
        {
            assert!(
                (x - y).abs() <= 1e-4 + 1e-4 * x.abs(),
                "{what} rel {id}: {x} vs {y}"
            );
        }
    }
}

/// The batched 1-N path equals the per-triple oracle (M2's gate, checked
/// here against the naive kernel), for tail-only and both-sides, with and
/// without a mean scale.
#[test]
fn batched_one_to_n_matches_per_triple_oracle() {
    let (t, sc) = (fd_tables(), cx());
    for both in [false, true] {
        for scale in [1.0f32, 0.2] {
            let mut oracle = Grads::new(D);
            oracle.set_scale(scale);
            let mut want = 0.0f32;
            for &e in &examples() {
                want += if both {
                    loss::one_vs_all_step(&t, &sc, e, &mut oracle).unwrap()
                } else {
                    loss::one_vs_all_tail_step(&t, &sc, e, &mut oracle).unwrap()
                };
            }
            let mut dense = Grads::with_layout(StateLayout::Dense, D, NE, 2 * NR);
            dense.set_scale(scale);
            let got = one_to_n::batched_step(&t, &sc, &NaiveOneToN, &examples(), both, &mut dense)
                .unwrap();
            assert!(
                (got - want).abs() <= 1e-4 * want.abs(),
                "loss {got} vs {want}"
            );
            assert_grads_close(&oracle, &dense, &format!("both={both} scale={scale}"));
        }
    }
    // A sparse accumulator is refused, not silently mis-summed.
    let mut sparse = Grads::new(D);
    assert!(
        one_to_n::batched_step(&t, &sc, &NaiveOneToN, &examples(), false, &mut sparse).is_err()
    );
}

#[test]
fn one_to_n_shape_contract() {
    let q = vec![0.0; 2 * D];
    let ent = vec![0.0; NE * D];
    let (mut gq, mut ge) = (vec![0.0; 2 * D], vec![0.0; NE * D]);
    let k = NaiveOneToN;
    assert!(k
        .softmax_ce(&q, &ent, D, &[0, 1], 1.0, &mut gq, &mut ge)
        .is_ok());
    assert!(k
        .softmax_ce(&q, &ent, D, &[0, NE as u32], 1.0, &mut gq, &mut ge)
        .is_err());
    assert!(k
        .softmax_ce(&q, &ent, D, &[0], 1.0, &mut gq, &mut ge)
        .is_err());
    assert!(k
        .softmax_ce(&q, &ent, 0, &[0, 1], 1.0, &mut gq, &mut ge)
        .is_err());
}

/// Dense and sparse optimizer state are a storage choice only: bitwise
/// identical tables for Adagrad and Adam, sampled and 1-N losses.
#[test]
fn dense_and_sparse_state_are_bitwise_identical() {
    let store = TripleStore::with_counts(examples(), Some(NE), Some(2 * NR)).unwrap();
    let optimizers = [
        OptimKind::Adagrad { epsilon: 1e-8 },
        OptimKind::Adam {
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
        },
    ];
    let losses = [
        LossKind::OneVsAll,
        LossKind::SelfAdversarial {
            neg_count: 4,
            temperature: 1.0,
            margin: 3.0,
        },
    ];
    for optimizer in optimizers {
        for loss in losses {
            let run = |layout: StateLayout, bilinear: bool| {
                let mut t = fd_tables();
                let cfg = TrainConfig {
                    dims: D,
                    epochs: 3,
                    batch_size: 2,
                    lr: 0.05,
                    optimizer,
                    loss,
                    n3_lambda: 0.01,
                    seed: 4,
                    optim_state: layout,
                    ..Default::default()
                };
                if bilinear {
                    Trainer::fit(&mut t, &cx(), &store, &cfg, |_| {}).unwrap();
                } else {
                    Trainer::fit(&mut t, &DistMult::new(D), &store, &cfg, |_| {}).unwrap();
                }
                t
            };
            for bilinear in [false, true] {
                assert_eq!(
                    run(StateLayout::Sparse, bilinear),
                    run(StateLayout::Dense, bilinear),
                    "{optimizer:?} {loss:?} bilinear={bilinear}"
                );
            }
        }
    }
}

#[test]
fn reciprocal_conventions_and_validation() {
    let t = Tables::new(NE, 2 * NR, D, 1);
    assert_eq!(reciprocal::base_relations(&t), NR);
    assert_eq!(reciprocal::inverse_relation(&t, 2).unwrap(), 5);
    assert!(reciprocal::inverse_relation(&t, NR as u32).is_err());
    let aug = reciprocal::augment(&t, &[Triple::new(1, 2, 3)]).unwrap();
    assert_eq!(aug, vec![Triple::new(1, 2, 3), Triple::new(3, 5, 1)]);

    // |R| rows only (what the bindings build today): typed error, no panic.
    let store = TripleStore::with_counts(vec![Triple::new(0, 2, 1)], Some(NE), Some(NR)).unwrap();
    let mut narrow = Tables::new(NE, NR, D, 1);
    let cfg = TrainConfig {
        dims: D,
        reciprocal: true,
        ..Default::default()
    };
    assert!(matches!(
        Trainer::fit(&mut narrow, &cx(), &store, &cfg, |_| {}),
        Err(KgeError::Invalid(_))
    ));
    let bad_rp = TrainConfig {
        dims: D,
        rp_weight: -1.0,
        ..Default::default()
    };
    assert!(Trainer::fit(&mut narrow, &cx(), &store, &bad_rp, |_| {}).is_err());
}

#[test]
fn recipe_json_round_trips_and_defaults_are_legacy() {
    let c: TrainConfig = serde_json::from_str(
        r#"{"loss":{"kind":"one_vs_all"},"reciprocal":true,"n3_form":"moduli",
            "loss_reduction":"mean","init":{"kind":"normal"},"optim_state":"dense",
            "rp_weight":0.05,"n3_lambda":0.1,"dims":2000}"#,
    )
    .unwrap();
    assert!(c.reciprocal);
    assert_eq!(c.n3_form, N3Form::Moduli);
    assert_eq!(c.loss_reduction, Reduction::Mean);
    assert_eq!(c.init, Init::Normal { scale: 1e-3 });
    assert_eq!(c.optim_state, StateLayout::Dense);
    assert_eq!(c.complex_rank(), 1000);
    let back: TrainConfig = serde_json::from_str(&serde_json::to_string(&c).unwrap()).unwrap();
    assert_eq!(back, c);
    let d = TrainConfig::default();
    assert!(!d.reciprocal && d.rp_weight == 0.0 && d.init == Init::Keep);
    assert_eq!(
        (d.n3_form, d.loss_reduction, d.optim_state),
        (N3Form::Elementwise, Reduction::Sum, StateLayout::Sparse)
    );
}

/// With tied reciprocal rows (`r⁻¹ = conj(r)`), reciprocal eval — tail ranks
/// of `test ∪ reciprocals` against the reciprocal-augmented filter — equals
/// ordinary tail+head eval of `test` (M1 acceptance).
#[test]
fn tied_conj_reciprocal_eval_equals_plain_eval() {
    let (store, ne, nr) = synthetic_kg();
    let sc = ComplEx::new(16).unwrap();
    let mut t = Tables::new(ne, 2 * nr, 16, 77);
    for r in 0..nr as u32 {
        let c = conj(t.relation(r).unwrap());
        t.relation_mut(r + nr as u32).unwrap().copy_from_slice(&c);
    }
    let test: Vec<Triple> = store.triples().iter().step_by(7).copied().collect();
    let ecfg = eval::EvalConfig::random(5);
    let plain = eval::evaluate_ranks(&t, &sc, &store, &test, &ecfg).unwrap();
    let aug_store = augmented_store(&t, &store, ne, nr);
    let aug_test = reciprocal::augment(&t, &test).unwrap();
    let rec = eval::evaluate_ranks(&t, &sc, &aug_store, &aug_test, &ecfg).unwrap();
    let n = test.len();
    for i in 0..n {
        assert_eq!(rec[2 * i], plain[2 * i], "tail rank of test {i}");
        assert_eq!(rec[2 * (n + i)], plain[2 * i + 1], "head rank of test {i}");
    }
}

/// 100 entities in 5 clusters; asymmetric successor (r0), symmetric two-step
/// (r1), asymmetric cluster tag (r2) — ComplEx can fit all three.
fn synthetic_kg() -> (TripleStore, usize, usize) {
    let (clusters, per) = (5usize, 20u32);
    let mut triples = Vec::new();
    for c in 0..clusters as u32 {
        let base = c * per;
        for a in 0..per {
            triples.push(Triple::new(base + a, 0, base + (a + 1) % per));
            let d = base + (a + 2) % per;
            triples.push(Triple::new(base + a, 1, d));
            triples.push(Triple::new(d, 1, base + a));
            triples.push(Triple::new(base + a, 2, base));
        }
    }
    let ne = clusters * per as usize;
    let store = TripleStore::with_counts(triples, Some(ne), Some(3)).unwrap();
    (store, ne, 3)
}

fn augmented_store(t: &Tables, store: &TripleStore, ne: usize, nr: usize) -> TripleStore {
    let aug = reciprocal::augment(t, store.triples()).unwrap();
    TripleStore::with_counts(aug, Some(ne), Some(2 * nr)).unwrap()
}

/// End-to-end ComplEx-N3-R (+RP) on synthetic data: filtered reciprocal MRR
/// rises well above the random-init floor.
#[test]
fn complex_n3_reciprocal_recipe_lifts_mrr() {
    let (store, ne, nr) = synthetic_kg();
    let split = store.split(1, [0.85, 0.0, 0.15]).unwrap();
    let train = TripleStore::with_counts(split.train.clone(), Some(ne), Some(nr)).unwrap();
    let dims = 32;
    let sc = ComplEx::new(dims).unwrap();
    let mut t = Tables::new(ne, 2 * nr, dims, 3);
    let cfg = TrainConfig {
        dims,
        epochs: 40,
        batch_size: 64,
        lr: 0.1,
        optimizer: OptimKind::Adagrad { epsilon: 1e-10 },
        loss: LossKind::OneVsAll,
        n3_lambda: 0.01,
        seed: 21,
        reciprocal: true,
        init: Init::Normal { scale: 1e-3 },
        n3_form: N3Form::Moduli,
        loss_reduction: Reduction::Mean,
        rp_weight: 0.05,
        optim_state: StateLayout::Dense,
        one_n_kernel: OneNKernel::Naive,
    };
    let aug_store = augmented_store(&t, &store, ne, nr);
    let aug_test = reciprocal::augment(&t, &split.test).unwrap();
    let ecfg = eval::EvalConfig::random(123);
    let mrr = |t: &Tables| {
        eval::evaluate(t, &sc, &aug_store, &aug_test, &ecfg)
            .unwrap()
            .tail
            .mrr
    };
    // Baseline at the recipe's own init (what `fit` starts from).
    let mut base_t = t.clone();
    init::apply_init(&mut base_t, cfg.init, cfg.seed);
    let baseline = mrr(&base_t);

    let (mut first, mut last, mut rp_seen) = (None, 0.0f32, 0.0f32);
    Trainer::fit(&mut t, &sc, &train, &cfg, |p| {
        first.get_or_insert(p.loss);
        last = p.loss;
        rp_seen = p.rp_loss;
    })
    .unwrap();
    let trained = mrr(&t);
    // Per-half MRR: `aug_test` is `test ++ reciprocals`, so the second half
    // are the inverse-relation (head-as-tail) queries, answerable only if
    // the `r + R` rows were trained.
    let ranks = eval::evaluate_ranks(&t, &sc, &aug_store, &aug_test, &ecfg).unwrap();
    let n = split.test.len();
    let half_mrr = |range: std::ops::Range<usize>| {
        range
            .clone()
            .map(|i| 1.0 / ranks[2 * i] as f32)
            .sum::<f32>()
            / range.len() as f32
    };
    let (fwd, inv) = (half_mrr(0..n), half_mrr(n..2 * n));
    let first = first.unwrap();
    println!(
        "[recipe] loss {first:.4} -> {last:.4}, rp {rp_seen:.4}; reciprocal filtered MRR {baseline:.4} -> {trained:.4} (forward {fwd:.4}, inverse {inv:.4})"
    );
    assert!(last < first, "loss should fall: {first} -> {last}");
    assert!(rp_seen > 0.0, "RP term was active");
    assert!(
        trained > baseline + 0.3 && trained > 0.4,
        "trained MRR {trained:.4} vs baseline {baseline:.4}"
    );
    assert!(fwd > 0.4, "forward-half MRR {fwd:.4}");
    assert!(inv > 0.4, "inverse-relation-half MRR {inv:.4}");
}
