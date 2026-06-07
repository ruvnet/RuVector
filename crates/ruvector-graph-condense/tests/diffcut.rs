//! Public-API integration tests for the differentiable min-cut condenser.
//! (Internal gradient-check / maths tests live in the `diffcut` module itself.)

use ruvector_graph_condense::{CondenseError, DiffCutCondenser, DiffCutConfig};
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
    let g = barbell();
    let cfg = DiffCutConfig {
        num_clusters: 2,
        ortho_weight: 1.0,
        learning_rate: 0.3,
        momentum: 0.0,
        iterations: 1,
        seed: 7,
    };
    let early = DiffCutCondenser::new(cfg.clone()).train(&g).unwrap().loss();
    let late = DiffCutCondenser::new(DiffCutConfig {
        iterations: 300,
        ..cfg
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
    // A clean two-cluster solution drives the cut term toward -1.
    assert!(late.cut < -0.7, "cut term {} not minimised", late.cut);
}

#[test]
fn recovers_barbell_partition() {
    let g = barbell();
    let res = DiffCutCondenser::new(DiffCutConfig {
        num_clusters: 2,
        ortho_weight: 1.0,
        learning_rate: 0.3,
        momentum: 0.0,
        iterations: 400,
        seed: 1,
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
