use ruvector_partition_memory::corpus::{generate, CorpusConfig};
use ruvector_partition_memory::graph::build_knn_edges;
use ruvector_partition_memory::partition::{adaptive_partition, AdaptiveConfig};
use std::time::Instant;

fn main() {
    for n in [500usize, 4000] {
        let cfg = CorpusConfig {
            n,
            ..CorpusConfig::default()
        };
        let corpus = generate(&cfg);
        let edges = build_knn_edges(&corpus.records, 10);
        let verts: Vec<u64> = corpus.records.iter().map(|r| r.id).collect();
        let mut cluster_of = std::collections::HashMap::new();
        for r in &corpus.records {
            cluster_of.insert(r.id, r.cluster);
        }

        for ratio in [0.5f64, 0.35, 0.2] {
            let cfg2 = AdaptiveConfig {
                coherence_ratio: ratio,
                ..AdaptiveConfig::default()
            };
            let t0 = Instant::now();
            let result = adaptive_partition(&edges, &verts, &cfg2);
            let us = t0.elapsed().as_micros();
            let sizes: Vec<usize> = result.clusters.iter().map(|c| c.len()).collect();
            // purity: for each partition, fraction belonging to its majority cluster
            let purities: Vec<f64> = result
                .clusters
                .iter()
                .map(|part| {
                    let mut hist = std::collections::HashMap::new();
                    for v in part {
                        *hist.entry(cluster_of[v]).or_insert(0usize) += 1;
                    }
                    let max = *hist.values().max().unwrap_or(&0);
                    max as f64 / part.len() as f64
                })
                .collect();
            println!(
                "n={n} ratio={ratio} partitions={} sizes={:?} purities={:?} us={us}",
                result.clusters.len(),
                sizes,
                purities
                    .iter()
                    .map(|p| format!("{p:.2}"))
                    .collect::<Vec<_>>()
            );
        }
    }
}
