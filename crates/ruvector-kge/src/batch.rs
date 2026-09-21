//! Batch scoring — one query against the whole entity table as a matmul
//! (ADR-002 §4). The trick production KGE servers use: build the Fourier-domain
//! (index) copy of the entity table once, then scoring a fixed relation against
//! every entity is a single mat-vec of the query vector against that matrix.
//!
//! For HolE the index dot product is the exact score, so `top_k` here is exact
//! and doubles as the ground truth the ANN recall test measures against.

use crate::scorer::Scorer;
use crate::{Candidate, EntityId, Tables};

/// The entity table transformed into its index (Fourier, for HolE)
/// representation, stored row-major for cache-friendly mat-vec scoring.
pub struct BatchScorer {
    n: usize,
    idx_dims: usize,
    rows: Vec<f32>,
}

impl BatchScorer {
    /// Build the index-representation matrix from `tables` under `scorer`.
    /// One `index_vector` per entity (for HolE, one FFT each) — done once,
    /// reused across every query.
    pub fn build<S: Scorer + ?Sized>(tables: &Tables, scorer: &S) -> Self {
        let n = tables.num_entities();
        let idx_dims = scorer.index_dims();
        let mut rows = Vec::with_capacity(n * idx_dims);
        for e in 0..n {
            let entity = tables
                .entity(e as EntityId)
                .expect("entity id in range by construction");
            let row = scorer.index_vector(entity);
            debug_assert_eq!(row.len(), idx_dims);
            rows.extend_from_slice(&row);
        }
        Self { n, idx_dims, rows }
    }

    /// Number of entities in the matrix.
    pub fn len(&self) -> usize {
        self.n
    }

    /// Whether the matrix is empty.
    pub fn is_empty(&self) -> bool {
        self.n == 0
    }

    /// Score `query` (a [`Scorer::query_vector`], length `index_dims`) against
    /// every entity: `scores[e] = row_e · query`.
    pub fn scores(&self, query: &[f32]) -> Vec<f32> {
        assert_eq!(
            query.len(),
            self.idx_dims,
            "scores: query length must equal index_dims"
        );
        (0..self.n)
            .map(|i| {
                dot8(
                    &self.rows[i * self.idx_dims..(i + 1) * self.idx_dims],
                    query,
                )
            })
            .collect()
    }

    /// Exact top-`k` entities by score, descending. For HolE these scores are
    /// exact; a `k` larger than the table returns the whole table sorted.
    pub fn top_k(&self, query: &[f32], k: usize) -> Vec<Candidate> {
        let scores = self.scores(query);
        let mut cands: Vec<Candidate> = scores
            .into_iter()
            .enumerate()
            .map(|(e, score)| Candidate {
                entity: e as EntityId,
                score,
            })
            .collect();
        cands.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        cands.truncate(k);
        cands
    }
}

/// SIMD-friendly dot product: eight independent accumulators over chunks of 8
/// so the autovectoriser can widen the inner loop; scalar tail. No `unsafe`.
#[inline]
fn dot8(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut acc = [0.0f32; 8];
    let chunks = a.len() / 8;
    for c in 0..chunks {
        let base = c * 8;
        for j in 0..8 {
            acc[j] += a[base + j] * b[base + j];
        }
    }
    let mut s = acc.iter().sum::<f32>();
    for i in (chunks * 8)..a.len() {
        s += a[i] * b[i];
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scorer::HolE;
    use crate::Side;

    #[test]
    fn dot8_matches_naive() {
        let a: Vec<f32> = (0..37).map(|i| i as f32 * 0.1).collect();
        let b: Vec<f32> = (0..37).map(|i| (37 - i) as f32 * 0.2).collect();
        let naive: f32 = a.iter().zip(&b).map(|(x, y)| x * y).sum();
        assert!((dot8(&a, &b) - naive).abs() < 1e-3);
    }

    #[test]
    fn batch_top_k_matches_per_triple_score() {
        let d = 64;
        let scorer = HolE::new(d).unwrap();
        let tables = Tables::new(200, 5, d, 7);
        let batch = BatchScorer::build(&tables, &scorer);

        let s: EntityId = 3;
        let r: crate::RelationId = 2;
        let q = scorer.query_vector(
            tables.relation(r).unwrap(),
            tables.entity(s).unwrap(),
            Side::Tail,
        );

        // Batch scores must equal per-triple score() for every entity.
        let batch_scores = batch.scores(&q);
        for (o, &bs) in batch_scores.iter().enumerate() {
            let direct = scorer.score(
                tables.entity(s).unwrap(),
                tables.relation(r).unwrap(),
                tables.entity(o as EntityId).unwrap(),
            );
            assert!(
                (bs - direct).abs() <= 1e-3 + 1e-3 * direct.abs(),
                "entity {o}: batch {bs} vs direct {direct}"
            );
        }

        // top_k is sorted and consistent with the exact scores.
        let top = batch.top_k(&q, 10);
        assert_eq!(top.len(), 10);
        for w in top.windows(2) {
            assert!(w[0].score >= w[1].score);
        }
    }
}
