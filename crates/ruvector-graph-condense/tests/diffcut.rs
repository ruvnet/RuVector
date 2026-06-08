//! Public-API integration tests for the differentiable min-cut condenser.
//! (Internal gradient-check / maths tests live in the `diffcut` module itself.)

use ruvector_graph_condense::{
    CondenseError, DiffCutCondenser, DiffCutConfig, InitStrategy, Optimizer, PlantedPartition,
};
use ruvector_mincut::DynamicGraph;

fn barbell() -> DynamicGraph {
    let g = DynamicGraph::new();
    for &(u, v, w) in &[
        (0, 1, 1.0),
        (1, 2, 1.0),
        (2, 0, 1.0),
        (3, 4, 1.0),
        (4, 5, 1.0),
        (5, 3, 1.0),
        (2, 3, 0.05),
    ] {
        g.insert_edge(u, v, w).unwrap();
    }
    g
}

#[test]
fn loss_decreases_during_training() {
    // From a *random* start with SGD, training must reduce the loss (a clean
    // descent test, independent of the warm-start prior).
    let g = barbell();
    let base = DiffCutConfig {
        num_clusters: 2,
        learning_rate: 0.3,
        init: InitStrategy::Random,
        optimizer: Optimizer::Sgd { momentum: 0.0 },
        iterations: 1,
        seed: 7,
        ..Default::default()
    };
    let early = DiffCutCondenser::new(base.clone())
        .train(&g)
        .unwrap()
        .loss();
    let late = DiffCutCondenser::new(DiffCutConfig {
        iterations: 300,
        ..base
    })
    .train(&g)
    .unwrap()
    .loss();
    assert!(
        late.total < early.total,
        "training did not reduce loss: {} -> {}",
        early.total,
        late.total
    );
    assert!(late.cut < -0.7, "cut term {} not minimised", late.cut);
}

#[test]
fn recovers_barbell_partition() {
    let g = barbell();
    let res = DiffCutCondenser::new(DiffCutConfig {
        num_clusters: 2,
        ..Default::default()
    })
    .train(&g)
    .unwrap();
    let mut regions = res.hard_regions();
    for r in &mut regions {
        r.sort_unstable();
    }
    regions.sort_by_key(|r| r[0]);
    assert_eq!(regions, vec![vec![0, 1, 2], vec![3, 4, 5]]);
}

/// Weighted dominant-class purity of a hard assignment vs. ground-truth
/// communities (vertex `v` belongs to community `v / community_size`).
fn purity(regions: &[Vec<u64>], community_size: u64) -> f64 {
    let mut correct = 0u64;
    let mut total = 0u64;
    for r in regions {
        let mut counts: std::collections::HashMap<u64, u64> = std::collections::HashMap::new();
        for &v in r {
            *counts.entry(v / community_size).or_default() += 1;
        }
        correct += counts.values().copied().max().unwrap_or(0);
        total += r.len() as u64;
    }
    correct as f64 / total.max(1) as f64
}

#[test]
fn warm_start_recovers_many_clusters() {
    // The headline "works on big problems" test: K = 8 on 8 planted communities.
    let pp = PlantedPartition {
        num_communities: 8,
        community_size: 24,
        dim: 8,
        p_intra: 0.5,
        p_inter: 0.002,
        seed: 3,
        ..Default::default()
    };
    let (g, _f) = pp.generate();
    let res = DiffCutCondenser::new(DiffCutConfig {
        num_clusters: 8,
        ..Default::default() // Adam + warm-start
    })
    .train(&g)
    .unwrap();
    let pur = purity(&res.hard_regions(), pp.community_size as u64);
    assert!(pur > 0.85, "warm-start purity at K=8 too low: {pur}");
}

#[test]
fn warm_start_beats_random_at_large_k() {
    // Same graph, same budget: warm-start should reach a lower (better) loss
    // than random init at large K — the whole point of the optimisation work.
    let pp = PlantedPartition {
        num_communities: 8,
        community_size: 20,
        dim: 8,
        p_intra: 0.5,
        p_inter: 0.002,
        seed: 11,
        ..Default::default()
    };
    let (g, _f) = pp.generate();
    let common = DiffCutConfig {
        num_clusters: 8,
        iterations: 200,
        seed: 1,
        ..Default::default()
    };
    let warm = DiffCutCondenser::new(common.clone()).train(&g).unwrap();
    let rand = DiffCutCondenser::new(DiffCutConfig {
        init: InitStrategy::Random,
        ..common
    })
    .train(&g)
    .unwrap();
    assert!(
        warm.loss().total <= rand.loss().total,
        "warm-start ({}) not better than random ({})",
        warm.loss().total,
        rand.loss().total
    );
    let pur_warm = purity(&warm.hard_regions(), pp.community_size as u64);
    let pur_rand = purity(&rand.hard_regions(), pp.community_size as u64);
    assert!(
        pur_warm >= pur_rand,
        "warm purity {pur_warm} < random purity {pur_rand}"
    );
}

#[test]
fn determinism_same_seed_same_result() {
    let g = barbell();
    let cfg = DiffCutConfig {
        num_clusters: 2,
        iterations: 200,
        seed: 5,
        ..Default::default()
    };
    let a = DiffCutCondenser::new(cfg.clone()).train(&g).unwrap();
    let b = DiffCutCondenser::new(cfg).train(&g).unwrap();
    assert_eq!(a.soft_assignment(), b.soft_assignment());
    assert_eq!(a.loss(), b.loss());
}

#[test]
fn parallel_matches_sequential_exactly() {
    // Row-parallel A·S is deterministic, so parallel must equal sequential
    // bit-for-bit (same seed, same config otherwise).
    let pp = PlantedPartition {
        num_communities: 6,
        community_size: 24,
        dim: 8,
        seed: 4,
        ..Default::default()
    };
    let (g, _f) = pp.generate();
    let base = DiffCutConfig {
        num_clusters: 6,
        iterations: 120,
        seed: 2,
        tolerance: 0.0, // disable early-stop so both run identical iterations
        ..Default::default()
    };
    let seq = DiffCutCondenser::new(base.clone()).train(&g).unwrap();
    let par = DiffCutCondenser::new(DiffCutConfig {
        parallel: true,
        ..base
    })
    .train(&g)
    .unwrap();
    assert_eq!(seq.soft_assignment(), par.soft_assignment());
    assert_eq!(seq.loss(), par.loss());
}

#[test]
fn minibatch_recovers_structure() {
    // Stochastic edge-minibatch should still recover the planted communities
    // (warm-start prior + refinement), at a fraction of the per-step edge cost.
    let pp = PlantedPartition {
        num_communities: 6,
        community_size: 24,
        dim: 8,
        p_intra: 0.5,
        p_inter: 0.002,
        seed: 9,
        ..Default::default()
    };
    let (g, _f) = pp.generate();
    let res = DiffCutCondenser::new(DiffCutConfig {
        num_clusters: 6,
        minibatch_edges: Some(256),
        iterations: 150,
        seed: 1,
        ..Default::default()
    })
    .train(&g)
    .unwrap();
    let pur = purity(&res.hard_regions(), pp.community_size as u64);
    assert!(pur > 0.8, "minibatch purity too low: {pur}");
}

#[test]
fn early_stopping_cuts_iterations() {
    // Warm-start lands near the optimum, so early-stop should finish well under
    // the iteration cap.
    let pp = PlantedPartition {
        num_communities: 6,
        community_size: 20,
        dim: 8,
        seed: 6,
        ..Default::default()
    };
    let (g, _f) = pp.generate();
    let res = DiffCutCondenser::new(DiffCutConfig {
        num_clusters: 6,
        iterations: 1000,
        tolerance: 1e-4,
        seed: 1,
        ..Default::default()
    })
    .train(&g)
    .unwrap();
    assert!(
        res.iterations_run() < 1000,
        "early-stop did not trigger: {}",
        res.iterations_run()
    );
}

#[test]
fn empty_graph_errors() {
    let g = DynamicGraph::new();
    assert!(matches!(
        DiffCutCondenser::new(DiffCutConfig::default())
            .train(&g)
            .unwrap_err(),
        CondenseError::EmptyGraph
    ));
}

#[test]
fn zero_clusters_errors() {
    let g = barbell();
    let err = DiffCutCondenser::new(DiffCutConfig {
        num_clusters: 0,
        ..Default::default()
    })
    .train(&g)
    .unwrap_err();
    assert!(matches!(err, CondenseError::InvalidConfig(_)));
}

#[test]
fn public_min_cut_loss_dimension_check() {
    use ruvector_graph_condense::min_cut_loss;
    let g = barbell();
    let err = min_cut_loss(&g, &[0.5; 3], 2, 1.0).unwrap_err();
    assert!(matches!(err, CondenseError::DimensionMismatch { .. }));
}
