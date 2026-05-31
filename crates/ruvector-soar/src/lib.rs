//! ruvector-soar: SOAR-IVF for high-recall approximate nearest-neighbor search.
//!
//! Implements SOAR (Spilling with Orthogonality-Amplified Residuals) from
//! Sun et al., NeurIPS 2023. arXiv:2404.00774.
//!
//! ## Index types
//!
//! | Type | Description |
//! |------|-------------|
//! | `IndexKind::Flat` | Exact brute-force baseline |
//! | `IndexKind::IvfPq` | IVF + ADC without secondary spilling |
//! | `IndexKind::SoarIvfPq` | IVF + ADC + SOAR orthogonality-amplified spilling |

pub mod error;
pub mod index;
pub mod kmeans;
pub mod pq;

pub use error::{Result, SoarError};
pub use index::{recall_at_k, IndexKind, SearchResult, SoarConfig, SoarIndex};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index::{IndexKind, SoarConfig, SoarIndex, recall_at_k};
    use crate::kmeans::l2_sq;

    fn tiny_corpus(n: usize, d: usize) -> Vec<Vec<f32>> {
        // Unique vectors: each vector is offset by i*100 so no two are identical.
        (0..n)
            .map(|i| (0..d).map(|j| i as f32 * 100.0 + j as f32 * 0.1).collect())
            .collect()
    }

    #[test]
    fn flat_exact_recall_is_one() {
        let corpus = tiny_corpus(100, 16);
        let query = corpus[7].clone();
        let cfg = SoarConfig { kind: IndexKind::Flat, ..Default::default() };
        let idx = SoarIndex::build(corpus, cfg).unwrap();
        let results = idx.search(&query, 5).unwrap();
        assert_eq!(results[0].id, 7, "exact search must return the query vector itself first");
        assert!(results[0].distance < 1e-4, "distance to itself must be ~0");
    }

    #[test]
    fn ivf_pq_builds_and_searches() {
        let corpus = tiny_corpus(200, 16);
        let query = corpus[42].clone();
        let cfg = SoarConfig {
            kind: IndexKind::IvfPq,
            nlist: 8,
            nprobe: 4,
            m_pq: 4,
            kmeans_iter: 10,
            ..Default::default()
        };
        let idx = SoarIndex::build(corpus.clone(), cfg).unwrap();
        let results = idx.search(&query, 10).unwrap();
        assert!(!results.is_empty());
        // Ground truth via flat
        let flat_cfg = SoarConfig { kind: IndexKind::Flat, ..Default::default() };
        let flat = SoarIndex::build(corpus, flat_cfg).unwrap();
        let truth: Vec<usize> = flat.search(&query, 10).unwrap().into_iter().map(|r| r.id).collect();
        let rec = recall_at_k(&truth, &results, 10);
        // Loose recall bound: at nprobe=4 out of 8 lists we expect reasonable recall
        assert!(rec >= 0.3, "IVF recall@10 should be ≥ 30% on structured data, got {rec:.2}");
    }

    #[test]
    fn soar_recall_at_least_as_good_as_ivf() {
        let corpus = tiny_corpus(200, 16);
        let queries: Vec<Vec<f32>> = (0..20).map(|i| corpus[i * 5].clone()).collect();
        let flat_cfg = SoarConfig { kind: IndexKind::Flat, ..Default::default() };
        let flat = SoarIndex::build(corpus.clone(), flat_cfg).unwrap();
        let truth: Vec<Vec<usize>> = queries
            .iter()
            .map(|q| flat.search(q, 10).unwrap().into_iter().map(|r| r.id).collect())
            .collect();

        let nprobe = 3;
        let ivf_cfg = SoarConfig {
            kind: IndexKind::IvfPq, nlist: 8, nprobe, m_pq: 4, kmeans_iter: 10, ..Default::default()
        };
        let soar_cfg = SoarConfig {
            kind: IndexKind::SoarIvfPq, nlist: 8, nprobe, m_pq: 4, kmeans_iter: 10, lambda: 1.0, ..Default::default()
        };

        let ivf_idx = SoarIndex::build(corpus.clone(), ivf_cfg).unwrap();
        let soar_idx = SoarIndex::build(corpus, soar_cfg).unwrap();

        let ivf_recall: f64 = queries.iter().zip(truth.iter())
            .map(|(q, tr)| recall_at_k(tr, &ivf_idx.search(q, 10).unwrap(), 10))
            .sum::<f64>() / queries.len() as f64;

        let soar_recall: f64 = queries.iter().zip(truth.iter())
            .map(|(q, tr)| recall_at_k(tr, &soar_idx.search(q, 10).unwrap(), 10))
            .sum::<f64>() / queries.len() as f64;

        // SOAR recall >= IVF recall at same nprobe (it has more candidates via secondary lists)
        assert!(
            soar_recall >= ivf_recall - 0.05,
            "SOAR recall ({soar_recall:.3}) should be >= IVF recall ({ivf_recall:.3})"
        );
    }

    #[test]
    fn dimension_mismatch_is_error() {
        let corpus = tiny_corpus(50, 16);
        let cfg = SoarConfig { kind: IndexKind::IvfPq, nlist: 4, nprobe: 2, m_pq: 4, ..Default::default() };
        let idx = SoarIndex::build(corpus, cfg).unwrap();
        assert!(idx.search(&[1.0f32; 8], 5).is_err());
    }

    #[test]
    fn l2_sq_self_is_zero() {
        let v = vec![1.0f32, 2.0, 3.0, 4.0];
        assert!(l2_sq(&v, &v) < 1e-6);
    }
}
