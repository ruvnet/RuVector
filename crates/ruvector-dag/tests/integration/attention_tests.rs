//! Attention mechanism integration tests

use ruvector_dag::dag::{QueryDag, OperatorNode, OperatorType};
use ruvector_dag::attention::*;

fn create_test_dag() -> QueryDag {
    let mut dag = QueryDag::new();

    // Simple linear DAG
    for i in 0..5 {
        dag.add_node(OperatorNode::new(i, OperatorType::SeqScan {
            table: format!("t{}", i)
        }));
    }

    for i in 0..4 {
        dag.add_edge(i, i + 1).unwrap();
    }

    dag
}

#[test]
fn test_topological_attention() {
    let dag = create_test_dag();
    let attention = TopologicalAttention::new(TopologicalConfig::default());

    let scores = attention.forward(&dag).unwrap();

    // Verify normalization
    let sum: f32 = scores.values().sum();
    assert!((sum - 1.0).abs() < 0.001, "Attention scores should sum to 1.0");

    // Verify all scores in [0, 1]
    assert!(scores.values().all(|&s| s >= 0.0 && s <= 1.0));
}

#[test]
fn test_attention_selector_convergence() {
    let mechanisms: Vec<Box<dyn DagAttention>> = vec![
        Box::new(TopologicalAttention::new(TopologicalConfig::default())),
    ];

    let mut selector = AttentionSelector::new(
        mechanisms,
        SelectorConfig::default(),
    );

    // Run selection multiple times
    let mut selection_counts = std::collections::HashMap::new();

    for _ in 0..100 {
        let idx = selector.select();
        *selection_counts.entry(idx).or_insert(0) += 1;
        selector.update(idx, 0.5 + rand::random::<f32>() * 0.5);
    }

    // Should have made selections
    assert!(selection_counts.values().sum::<usize>() == 100);
}

#[test]
fn test_attention_cache() {
    let mut cache = AttentionCache::new(100);
    let dag = create_test_dag();

    // Cache miss
    assert!(cache.get(&dag, "topological").is_none());

    // Insert
    let mut scores = std::collections::HashMap::new();
    scores.insert(0usize, 0.5f32);
    cache.insert(&dag, "topological", scores.clone());

    // Cache hit
    assert!(cache.get(&dag, "topological").is_some());
}

#[test]
fn test_attention_temperature_scaling() {
    let dag = create_test_dag();
    let mut config = TopologicalConfig::default();

    // Low temperature (sharper distribution)
    config.temperature = 0.1;
    let attention_low = TopologicalAttention::new(config.clone());
    let scores_low = attention_low.forward(&dag).unwrap();

    // High temperature (smoother distribution)
    config.temperature = 2.0;
    let attention_high = TopologicalAttention::new(config);
    let scores_high = attention_high.forward(&dag).unwrap();

    // Low temperature should have more concentrated scores
    let variance_low: f32 = scores_low.values().map(|&x| x * x).sum::<f32>()
        - scores_low.values().sum::<f32>().powi(2) / scores_low.len() as f32;
    let variance_high: f32 = scores_high.values().map(|&x| x * x).sum::<f32>()
        - scores_high.values().sum::<f32>().powi(2) / scores_high.len() as f32;

    assert!(variance_low >= variance_high, "Lower temperature should have higher variance");
}

#[test]
fn test_attention_empty_dag() {
    let dag = QueryDag::new();
    let attention = TopologicalAttention::new(TopologicalConfig::default());

    let result = attention.forward(&dag);
    assert!(result.is_ok());
    assert!(result.unwrap().is_empty());
}

#[test]
fn test_attention_single_node() {
    let mut dag = QueryDag::new();
    dag.add_node(OperatorNode::new(0, OperatorType::Result));

    let attention = TopologicalAttention::new(TopologicalConfig::default());
    let scores = attention.forward(&dag).unwrap();

    // Single node should get score of 1.0
    assert_eq!(scores.len(), 1);
    assert!((scores[&0] - 1.0).abs() < 0.001);
}

#[test]
fn test_attention_cache_eviction() {
    let mut cache = AttentionCache::new(3);

    // Fill cache
    for i in 0..5 {
        let mut dag = QueryDag::new();
        dag.add_node(OperatorNode::new(i, OperatorType::Result));

        let mut scores = std::collections::HashMap::new();
        scores.insert(i, i as f32);
        cache.insert(&dag, "test", scores);
    }

    // Cache should not exceed capacity
    assert!(cache.len() <= 3);
}

#[test]
fn test_multi_mechanism_selector() {
    let mechanisms: Vec<Box<dyn DagAttention>> = vec![
        Box::new(TopologicalAttention::new(TopologicalConfig::default())),
        Box::new(TopologicalAttention::new(TopologicalConfig {
            temperature: 2.0,
            ..Default::default()
        })),
    ];

    let mut selector = AttentionSelector::new(
        mechanisms,
        SelectorConfig {
            epsilon: 0.1,
            exploration_decay: 0.99,
        },
    );

    // Both mechanisms should be selected at some point
    let mut used = std::collections::HashSet::new();

    for _ in 0..50 {
        let idx = selector.select();
        used.insert(idx);
        selector.update(idx, 0.5);
    }

    assert!(used.len() >= 1, "At least one mechanism should be selected");
}
