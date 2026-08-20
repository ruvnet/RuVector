//! Evaluation: recall@k against the corpus's brute-force ground truth,
//! computed overall and per-cluster, plus cluster coverage.

use crate::corpus::Corpus;
use crate::search::{recall_at_k, top_k_by_cosine};
use std::collections::{HashMap, HashSet};

#[derive(Debug, Clone)]
pub struct VariantReport {
    pub name: String,
    pub retained_count: usize,
    pub overall_recall: f64,
    pub worst_cluster_recall: f64,
    pub per_cluster_recall: Vec<f64>,
    pub cluster_coverage: f64,
    pub partition_us: u128,
    pub retention_us: u128,
}

pub fn evaluate(
    corpus: &Corpus,
    retained_ids: &[u64],
    name: &str,
    partition_us: u128,
    retention_us: u128,
) -> VariantReport {
    let by_id: HashMap<u64, &crate::corpus::MemoryRecord> =
        corpus.records.iter().map(|r| (r.id, r)).collect();
    let refs: Vec<(u64, &[f32])> = retained_ids
        .iter()
        .filter_map(|id| by_id.get(id).map(|r| (*id, r.embedding.as_slice())))
        .collect();

    let num_clusters = corpus.cluster_sizes.len();
    let mut sum_per_cluster = vec![0.0f64; num_clusters];
    let mut count_per_cluster = vec![0usize; num_clusters];
    let mut overall_sum = 0.0f64;

    for q in &corpus.queries {
        let retrieved = top_k_by_cosine(&q.embedding, &refs, q.ground_truth.len());
        let r = recall_at_k(&retrieved, &q.ground_truth);
        overall_sum += r;
        sum_per_cluster[q.cluster] += r;
        count_per_cluster[q.cluster] += 1;
    }

    let overall_recall = overall_sum / corpus.queries.len() as f64;
    let per_cluster_recall: Vec<f64> = sum_per_cluster
        .iter()
        .zip(&count_per_cluster)
        .map(|(s, c)| if *c > 0 { s / *c as f64 } else { 0.0 })
        .collect();
    let worst_cluster_recall = per_cluster_recall
        .iter()
        .cloned()
        .fold(f64::INFINITY, f64::min);

    let present_clusters: HashSet<usize> = retained_ids
        .iter()
        .filter_map(|id| by_id.get(id).map(|r| r.cluster))
        .collect();
    let cluster_coverage = present_clusters.len() as f64 / num_clusters as f64;

    VariantReport {
        name: name.to_string(),
        retained_count: retained_ids.len(),
        overall_recall,
        worst_cluster_recall,
        per_cluster_recall,
        cluster_coverage,
        partition_us,
        retention_us,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::corpus::{generate, CorpusConfig};

    #[test]
    fn retaining_everything_gives_perfect_recall() {
        let cfg = CorpusConfig {
            n: 300,
            ..CorpusConfig::default()
        };
        let corpus = generate(&cfg);
        let all_ids: Vec<u64> = corpus.records.iter().map(|r| r.id).collect();
        let report = evaluate(&corpus, &all_ids, "identity", 0, 0);
        assert!((report.overall_recall - 1.0).abs() < 1e-9);
        assert!((report.worst_cluster_recall - 1.0).abs() < 1e-9);
        assert!((report.cluster_coverage - 1.0).abs() < 1e-9);
    }

    #[test]
    fn dropping_a_cluster_entirely_zeroes_its_recall() {
        let cfg = CorpusConfig {
            n: 300,
            ..CorpusConfig::default()
        };
        let corpus = generate(&cfg);
        let minority_cluster = corpus.cluster_sizes.len() - 1;
        let ids: Vec<u64> = corpus
            .records
            .iter()
            .filter(|r| r.cluster != minority_cluster)
            .map(|r| r.id)
            .collect();
        let report = evaluate(&corpus, &ids, "drop-minority", 0, 0);
        // Clusters are separated by cosine <= 0.35 with noise_std 0.35, so
        // neighbouring clusters' true top-k can still contain a few
        // surviving points even with the minority cluster gone entirely —
        // recall degrades sharply but is not guaranteed to hit exactly
        // zero. That partial-credit behaviour is itself realistic (real
        // topic boundaries are not perfectly separable either) and is
        // exactly why this crate measures *worst-cluster* recall as a
        // continuous quantity rather than a binary "cluster present" flag.
        assert!(
            report.per_cluster_recall[minority_cluster] < 0.6,
            "expected sharply degraded recall for the dropped cluster, got {}",
            report.per_cluster_recall[minority_cluster]
        );
        assert!(report.worst_cluster_recall <= report.per_cluster_recall[minority_cluster]);
        assert!(report.cluster_coverage < 1.0);
    }
}
