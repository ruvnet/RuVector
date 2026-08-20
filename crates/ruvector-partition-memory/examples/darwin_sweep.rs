//! Bounded Darwin-style parameter sweep over `floor_min`, run once, after
//! the pre-declared main.rs acceptance benchmark had already produced its
//! REJECT verdict. This does NOT retroactively change that verdict — it is
//! a separate, explicitly bounded exploration (generations=1,
//! candidates_per_generation=4, matching the nightly harness's default
//! budget) asking whether retention-budget tuning alone can rescue the
//! rejected hypothesis, or whether the failure is structural (the
//! partition step, not the retention step). The partition (coherence_ratio
//! fixed at 0.35, the value used in the accepted run) is computed once and
//! reused, since `floor_min` only affects retention, not partitioning.
//!
//! Fitness (declared before running, per ADR guidance — not fit to the
//! result): 0.5*worst_cluster_recall + 0.3*overall_recall + 0.2*correctness,
//! where correctness = 1.0 if the witness chain verifies, else 0.0.

use ruvector_partition_memory::corpus::{generate, CorpusConfig};
use ruvector_partition_memory::graph::build_knn_edges;
use ruvector_partition_memory::metrics::evaluate;
use ruvector_partition_memory::partition::{adaptive_partition, AdaptiveConfig};
use ruvector_partition_memory::retention::{retain_partitioned, RetentionPolicy};

fn recency_context(
    records: &[ruvector_partition_memory::corpus::MemoryRecord],
    n_ctx: usize,
) -> Vec<Vec<f32>> {
    let mut sorted: Vec<_> = records.iter().collect();
    sorted.sort_by(|a, b| b.last_accessed_tick.cmp(&a.last_accessed_tick));
    sorted
        .into_iter()
        .take(n_ctx)
        .map(|r| r.embedding.clone())
        .collect()
}

fn fitness(worst: f64, overall: f64, correctness: f64) -> f64 {
    0.5 * worst + 0.3 * overall + 0.2 * correctness
}

fn main() {
    let cfg = CorpusConfig {
        n: 4000,
        ..CorpusConfig::default()
    };
    let corpus = generate(&cfg);
    let target_size = (corpus.records.len() as f64 * 0.5).round() as usize;
    let context = recency_context(&corpus.records, 20);
    let vertices: Vec<u64> = corpus.records.iter().map(|r| r.id).collect();
    let edges = build_knn_edges(&corpus.records, 10);

    let adaptive_cfg = AdaptiveConfig {
        coherence_ratio: 0.35,
        ..AdaptiveConfig::default()
    };
    let result = adaptive_partition(&edges, &vertices, &adaptive_cfg);
    let correctness = if result.witness.verify() { 1.0 } else { 0.0 };
    println!(
        "parent partition (fixed for the whole sweep): {} partitions, sizes={:?}, correctness={correctness}",
        result.clusters.len(),
        result.clusters.iter().map(|c| c.len()).collect::<Vec<_>>()
    );

    let candidates = [1usize, 3, 8, 15];
    let mut best: Option<(usize, f64)> = None;
    println!("gen=1 candidates_per_generation={}", candidates.len());
    for &floor_min in &candidates {
        let policy = RetentionPolicy { floor_min };
        let ids = retain_partitioned(
            &corpus.records,
            &result.clusters,
            &context,
            target_size,
            &policy,
        );
        let report = evaluate(&corpus, &ids, &format!("floor_min={floor_min}"), 0, 0);
        let f = fitness(
            report.worst_cluster_recall,
            report.overall_recall,
            correctness,
        );
        println!(
            "floor_min={floor_min:<3} retained={:<5} overall_recall={:.4} worst_cluster_recall={:.4} fitness={:.4}",
            report.retained_count, report.overall_recall, report.worst_cluster_recall, f
        );
        if best.map(|(_, bf)| f > bf).unwrap_or(true) {
            best = Some((floor_min, f));
        }
    }
    let (winner, winner_fitness) = best.unwrap();
    let parent_floor_min = 3usize; // the value used in the accepted (rejected) main.rs run
    let beats_parent = winner != parent_floor_min
        && candidates.contains(&winner)
        && winner_fitness
            > fitness(
                {
                    let policy = RetentionPolicy {
                        floor_min: parent_floor_min,
                    };
                    let ids = retain_partitioned(
                        &corpus.records,
                        &result.clusters,
                        &context,
                        target_size,
                        &policy,
                    );
                    evaluate(&corpus, &ids, "parent", 0, 0).worst_cluster_recall
                },
                {
                    let policy = RetentionPolicy {
                        floor_min: parent_floor_min,
                    };
                    let ids = retain_partitioned(
                        &corpus.records,
                        &result.clusters,
                        &context,
                        target_size,
                        &policy,
                    );
                    evaluate(&corpus, &ids, "parent", 0, 0).overall_recall
                },
                correctness,
            );
    println!("winner: floor_min={winner} fitness={winner_fitness:.4} beats_parent(floor_min=3)={beats_parent}");
    println!(
        "DARWIN_RESULT: {}",
        if beats_parent {
            "PROMOTE"
        } else {
            "KEEP_PARENT"
        }
    );
}
