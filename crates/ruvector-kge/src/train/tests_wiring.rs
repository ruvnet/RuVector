//! Fit-level wiring tests for the M1 recipe: `Trainer::fit` must equal a
//! hand-written training loop built from the tested helpers (reciprocal
//! augmentation, tail-only 1-vs-all, batch-mean scale, N3 form, RP, init),
//! bitwise, on both the per-triple path (HolE) and the batched 1-N path
//! (ComplEx). The helper tests in `tests_recipe` check each term in
//! isolation; these check that `fit` actually uses them as documented.

use super::optim::{Grads, Optimizer};
use super::*;
use crate::data::TripleStore;
use crate::scorer::complex::ComplEx;
use crate::{HolE, Tables, Triple};

const D: usize = 8;
const NE: usize = 12;
const NR: usize = 3;

/// Base-relation triples; 5 → 10 reciprocal examples, so `batch_size = 3`
/// leaves a short last batch of 1.
fn base_store() -> TripleStore {
    let triples = vec![
        Triple::new(0, 0, 1),
        Triple::new(2, 1, 3),
        Triple::new(4, 2, 0),
        Triple::new(1, 0, 5),
        Triple::new(7, 2, 7),
    ];
    TripleStore::with_counts(triples, Some(NE), Some(NR)).unwrap()
}

fn tables() -> Tables {
    Tables::new(NE, 2 * NR, D, 0xBEEF)
}

/// A recipe config every wiring decision is observable under: Normal init
/// (not a no-op), Mean reduction with a large Adagrad epsilon (so the
/// optimizer is *not* scale-invariant and the per-batch `1/len` shows),
/// complex-moduli N3 and RP both on.
fn recipe(optimizer: OptimKind) -> TrainConfig {
    TrainConfig {
        dims: D,
        epochs: 2,
        batch_size: 3,
        lr: 0.05,
        optimizer,
        loss: LossKind::OneVsAll,
        n3_lambda: 0.05,
        seed: 17,
        reciprocal: true,
        init: Init::Normal { scale: 0.3 },
        n3_form: N3Form::Moduli,
        loss_reduction: Reduction::Mean,
        rp_weight: 0.2,
        optim_state: StateLayout::Dense,
        one_n_kernel: OneNKernel::Naive,
    }
}

fn optimizers() -> [OptimKind; 2] {
    [
        OptimKind::Adagrad { epsilon: 0.5 },
        OptimKind::Adam {
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 0.5,
        },
    ]
}

/// The documented reciprocal recipe written out by hand: init, then
/// `augment(S)` in shuffle order, each example a **tail** query only
/// (batched tail-only 1-N when `batched`, else `one_vs_all_tail_step`), RP
/// and N3 per example, every term scaled by `scale(batch_len)`.
fn oracle(
    tables: &mut Tables,
    scorer: &dyn Differentiable,
    batched: Option<&ComplEx>,
    store: &TripleStore,
    cfg: &TrainConfig,
    scale: &dyn Fn(usize) -> f32,
) {
    assert!(cfg.reciprocal && cfg.loss == LossKind::OneVsAll);
    init::apply_init(tables, cfg.init, cfg.seed);
    let ex = reciprocal::augment(tables, store.triples()).unwrap();
    let (ne, nr) = (tables.num_entities(), tables.num_relations());
    let mut opt = Optimizer::new(cfg.optimizer, cfg.lr, cfg.optim_state, D, ne, nr);
    let mut order: Vec<usize> = (0..ex.len()).collect();
    let mut buf = Vec::new();
    for epoch in 0..cfg.epochs {
        shuffle(&mut order, cfg.seed, epoch);
        for batch in order.chunks(cfg.batch_size) {
            let mut g = Grads::with_layout(StateLayout::Dense, D, ne, nr);
            g.set_scale(scale(batch.len()));
            let bex: Vec<Triple> = batch.iter().map(|&i| ex[i]).collect();
            if let Some(b) = batched {
                one_to_n::batched_step(tables, b, &NaiveOneToN, &bex, false, &mut g).unwrap();
            }
            for &t in &bex {
                if batched.is_none() {
                    loss::one_vs_all_tail_step(tables, scorer, t, &mut g).unwrap();
                }
                rp::relation_prediction_step(tables, scorer, t, cfg.rp_weight, &mut g).unwrap();
                apply_n3(tables, t, cfg.n3_lambda, cfg.n3_form, &mut g, &mut buf).unwrap();
            }
            opt.apply(tables, &g).unwrap();
        }
    }
}

fn fit(scorer: &dyn Differentiable, store: &TripleStore, cfg: &TrainConfig) -> Tables {
    let mut t = tables();
    Trainer::fit(&mut t, scorer, store, cfg, |_| {}).unwrap();
    t
}

fn run_oracle(
    scorer: &dyn Differentiable,
    batched: Option<&ComplEx>,
    cfg: &TrainConfig,
    scale: &dyn Fn(usize) -> f32,
) -> Tables {
    let mut t = tables();
    oracle(&mut t, scorer, batched, &base_store(), cfg, scale);
    t
}

/// (a)+(b)+(c)+(d): `fit` equals the hand-written recipe loop bitwise, on the
/// per-triple (HolE) and batched (ComplEx) paths, for Adagrad and Adam. The
/// negative controls show the comparison is sensitive to each wiring choice.
#[test]
fn reciprocal_recipe_fit_equals_manual_loop() {
    let (hole, cx) = (HolE::new(D).unwrap(), ComplEx::new(D).unwrap());
    assert!(
        hole.as_bilinear().is_none(),
        "HolE takes the per-triple path"
    );
    assert!(cx.as_bilinear().is_some(), "ComplEx takes the batched path");
    let paths: [(&dyn Differentiable, Option<&ComplEx>, &str); 2] =
        [(&hole, None, "hole"), (&cx, Some(&cx), "complex")];
    let mean = |len: usize| 1.0 / len as f32;
    for optimizer in optimizers() {
        let cfg = recipe(optimizer);
        for (sc, batched, name) in paths {
            let what = format!("{name} {optimizer:?}");
            let got = fit(sc, &base_store(), &cfg);
            assert_eq!(
                got,
                run_oracle(sc, batched, &cfg, &mean),
                "{what}: fit != oracle"
            );

            // Negative controls: each would-be regression changes the tables.
            let full_batch = |_: usize| 1.0 / cfg.batch_size as f32;
            assert_ne!(
                got,
                run_oracle(sc, batched, &cfg, &full_batch),
                "{what}: short last batch must divide by its own length"
            );
            assert_ne!(got, run_oracle(sc, batched, &cfg, &|_| 1.0), "{what}: mean");
            let sum_cfg = TrainConfig {
                loss_reduction: Reduction::Sum,
                ..cfg
            };
            assert_eq!(
                fit(sc, &base_store(), &sum_cfg),
                run_oracle(sc, batched, &sum_cfg, &|_| 1.0),
                "{what}: sum reduction = unit scale"
            );
            let elem = TrainConfig {
                n3_form: N3Form::Elementwise,
                ..cfg
            };
            assert_ne!(got, fit(sc, &base_store(), &elem), "{what}: n3_form");
            assert_eq!(
                fit(sc, &base_store(), &elem),
                run_oracle(sc, batched, &elem, &mean),
                "{what}: elementwise N3 wiring"
            );
        }
    }
}

/// Reciprocal training is not "plain 1-vs-all on the augmented store": the
/// same examples in the same order, but trained both-sides, give different
/// tables — on both paths. Catches a reciprocal run routed back to
/// `one_vs_all_step` / `both_sides = true`.
#[test]
fn reciprocal_differs_from_both_sides_on_augmented_store() {
    let (hole, cx) = (HolE::new(D).unwrap(), ComplEx::new(D).unwrap());
    let t = tables();
    let aug = TripleStore::with_counts(
        reciprocal::augment(&t, base_store().triples()).unwrap(),
        Some(NE),
        Some(2 * NR),
    )
    .unwrap();
    for sc in [&hole as &dyn Differentiable, &cx] {
        let cfg = TrainConfig {
            rp_weight: 0.0,
            ..recipe(OptimKind::Adagrad { epsilon: 1e-8 })
        };
        let rec = fit(sc, &base_store(), &cfg);
        // Identical example list and order (augment of a store whose
        // relations are all < NR is `triples ++ reciprocals`, as fit builds).
        let plain = fit(
            sc,
            &aug,
            &TrainConfig {
                reciprocal: false,
                ..cfg
            },
        );
        assert_ne!(rec, plain);
        // Without augmentation (base triples only, tail-only) also differs:
        // the inverse rows must be trained.
        let untouched_inverse = {
            let mut t = tables();
            init::apply_init(&mut t, cfg.init, cfg.seed);
            (NR..2 * NR).all(|r| rec.relation(r as u32).unwrap() == t.relation(r as u32).unwrap())
        };
        assert!(!untouched_inverse, "every inverse row r+R is trained");
    }
}

/// (c) `init` is applied at the start of `fit`: with `lr = 0` the tables end
/// exactly at `apply_init(seed)`, whatever they held before; and a trained
/// run does not depend on the prior contents.
#[test]
fn fit_applies_init_before_training() {
    let cx = ComplEx::new(D).unwrap();
    let cfg = TrainConfig {
        lr: 0.0,
        ..recipe(OptimKind::Adagrad { epsilon: 1e-8 })
    };
    let mut want = tables();
    init::apply_init(&mut want, cfg.init, cfg.seed);
    for prior_seed in [1u64, 2] {
        let mut t = Tables::new(NE, 2 * NR, D, prior_seed);
        Trainer::fit(&mut t, &cx, &base_store(), &cfg, |_| {}).unwrap();
        assert_eq!(t, want, "lr = 0 leaves exactly the init draw");
    }
    let trained = TrainConfig { lr: 0.05, ..cfg };
    let run = |prior_seed| {
        let mut t = Tables::new(NE, 2 * NR, D, prior_seed);
        Trainer::fit(&mut t, &cx, &base_store(), &trained, |_| {}).unwrap();
        t
    };
    assert_eq!(run(1), run(2), "init overwrites prior contents");
    // `Keep` (the default) trains from the caller's rows instead.
    let keep = TrainConfig {
        init: Init::Keep,
        ..trained
    };
    let mut a = Tables::new(NE, 2 * NR, D, 1);
    let mut b = Tables::new(NE, 2 * NR, D, 2);
    Trainer::fit(&mut a, &cx, &base_store(), &keep, |_| {}).unwrap();
    Trainer::fit(&mut b, &cx, &base_store(), &keep, |_| {}).unwrap();
    assert_ne!(a, b);
}

/// (d) `n3_form` reaches the regulariser. With the data loss pinned to a
/// constant (a single entity, so softmax CE is 0 with zero gradient) and RP
/// off, N3 is the only active term: Moduli and Elementwise must move the
/// rows differently, and each must match its hand-applied gradient.
#[test]
fn n3_form_is_wired_when_n3_is_the_only_term() {
    let cx = ComplEx::new(D).unwrap();
    let store = TripleStore::with_counts(vec![Triple::new(0, 0, 0)], Some(1), Some(1)).unwrap();
    let start = Tables::new(1, 2, D, 5);
    let run = |form: N3Form| {
        let cfg = TrainConfig {
            dims: D,
            epochs: 1,
            batch_size: 4,
            lr: 0.1,
            optimizer: OptimKind::Adagrad { epsilon: 1e-8 },
            n3_lambda: 0.5,
            reciprocal: true,
            n3_form: form,
            ..Default::default()
        };
        let mut t = start.clone();
        Trainer::fit(&mut t, &cx, &store, &cfg, |_| {}).unwrap();
        t
    };
    let (moduli, elem) = (run(N3Form::Moduli), run(N3Form::Elementwise));
    assert_ne!(moduli, elem);
    assert_ne!(moduli, start);
    // Hand-apply: examples (0,0,0) and (0,1,0) in one batch, N3 only.
    for (form, got) in [(N3Form::Moduli, &moduli), (N3Form::Elementwise, &elem)] {
        let mut want = start.clone();
        let mut g = Grads::with_layout(StateLayout::Dense, D, 1, 2);
        let mut buf = Vec::new();
        for t in [Triple::new(0, 0, 0), Triple::new(0, 1, 0)] {
            apply_n3(&want, t, 0.5, form, &mut g, &mut buf).unwrap();
        }
        let mut opt = Optimizer::new(
            OptimKind::Adagrad { epsilon: 1e-8 },
            0.1,
            StateLayout::Sparse,
            D,
            1,
            2,
        );
        opt.apply(&mut want, &g).unwrap();
        for (a, b) in got.entities_raw().iter().zip(want.entities_raw()) {
            assert!((a - b).abs() <= 1e-6, "{form:?} ent {a} vs {b}");
        }
        for (a, b) in got.relations_raw().iter().zip(want.relations_raw()) {
            assert!((a - b).abs() <= 1e-6, "{form:?} rel {a} vs {b}");
        }
    }
}

/// `fit` routes reciprocal 1-vs-all for a non-bilinear scorer (HolE) through
/// the per-triple tail-only step: it trains (loss falls), and differs from
/// both-sides training on the same augmented examples.
#[test]
fn hole_reciprocal_fit_routes_tail_only() {
    let sc = HolE::new(D).unwrap();
    let aug = TripleStore::with_counts(
        reciprocal::augment(&tables(), base_store().triples()).unwrap(),
        Some(NE),
        Some(2 * NR),
    )
    .unwrap();
    let cfg = |reciprocal| TrainConfig {
        dims: D,
        epochs: 8,
        batch_size: 4,
        lr: 0.1,
        n3_lambda: 0.0,
        seed: 2,
        reciprocal,
        ..Default::default()
    };
    let (mut first, mut last) = (None, 0.0f32);
    let mut rec = tables();
    Trainer::fit(&mut rec, &sc, &base_store(), &cfg(true), |p| {
        first.get_or_insert(p.loss);
        last = p.loss;
    })
    .unwrap();
    assert!(last < first.unwrap(), "{first:?} -> {last}");
    let mut plain = tables();
    Trainer::fit(&mut plain, &sc, &aug, &cfg(false), |_| {}).unwrap();
    assert_ne!(rec, plain);
}
