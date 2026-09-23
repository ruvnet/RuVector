//! ANN candidate retrieval over the Fourier-domain entity table (ADR-001 §3,
//! ADR-002 §4). Wraps `ruvector-router-core`'s HNSW: build an index over every
//! entity's [`Scorer::index_vector`], retrieve candidates for a query, then
//! exact-rerank with the true score.
//!
//! ## Why the MIPS→L2 route rather than `DistanceMetric::DotProduct`
//!
//! ADR-001 names a `DotProduct` HNSW. router-core's `DotProduct` is already a
//! proper distance for that: `distance::dot_product` returns `-(a·b)`, so the
//! index's smallest-distance-first search is a maximum-inner-product search.
//! Do **not** negate the query on top of that — doing so searches for the
//! *minimum* inner product (recall@10 = 0.000, as reported in #1009). With the
//! query passed through unchanged the same table measures recall@10 = 1.000; see
//! `dotproduct_recall_matches_mips`.
//!
//! This module keeps the L2 reduction because Euclidean is a metric (non-negative,
//! zero on identity), which the neighbour-selection heuristic's pruning
//! comparisons are written against. The standard MIPS→L2 reduction (Bachrach
//! et al. 2014) appends one coordinate so Euclidean nearest-neighbour equals
//! maximum inner product. With `φ² = max_e ‖x_e‖²`,
//! `x_e' = [x_e ; √(φ² − ‖x_e‖²)]` and `q' = [q ; 0]`, then
//! `‖q' − x_e'‖² = ‖q‖² + φ² − 2 q·x_e`, so minimising L2 maximises `q·x_e`.

use crate::scorer::Scorer;
use crate::{Candidate, EntityId, KgeError, Result, Tables};
use ruvector_router_core::index::{HnswConfig, HnswIndex};
use ruvector_router_core::types::{DistanceMetric, SearchQuery};

/// HNSW index over the entity table's index representation, with the MIPS→L2
/// augmentation applied so retrieval maximises the score's inner product.
pub struct AnnIndex {
    hnsw: HnswIndex,
    aug_dims: usize,
}

impl AnnIndex {
    /// Build the index over every entity's [`Scorer::index_vector`].
    /// Two passes: gather index vectors and `φ² = max‖x‖²`, then insert each as
    /// `[x ; √(φ² − ‖x‖²)]`.
    pub fn build<S: Scorer + ?Sized>(tables: &Tables, scorer: &S) -> Result<Self> {
        let n = tables.num_entities();
        let idx_dims = scorer.index_dims();
        let aug_dims = idx_dims + 1;

        let mut vecs: Vec<Vec<f32>> = Vec::with_capacity(n);
        let mut max_sq = 0.0f32;
        for e in 0..n {
            let v = scorer.index_vector(tables.entity(e as EntityId)?);
            let sq: f32 = v.iter().map(|x| x * x).sum();
            if sq > max_sq {
                max_sq = sq;
            }
            vecs.push(v);
        }

        let cfg = HnswConfig {
            m: 16,
            ef_construction: 200,
            ef_search: 200,
            metric: DistanceMetric::Euclidean,
            dimensions: aug_dims,
        };
        let hnsw = HnswIndex::new(cfg);

        for (e, v) in vecs.into_iter().enumerate() {
            let sq: f32 = v.iter().map(|x| x * x).sum();
            let extra = (max_sq - sq).max(0.0).sqrt();
            let mut aug = v;
            aug.push(extra);
            hnsw.insert(e.to_string(), aug)
                .map_err(|err| KgeError::Scorer(format!("ann insert: {err}")))?;
        }

        Ok(Self { hnsw, aug_dims })
    }

    /// Retrieve up to `k` candidate entity ids for `query` (a
    /// [`Scorer::query_vector`], length `index_dims`), exploring with `ef`.
    pub fn candidates(&self, query: &[f32], k: usize, ef: usize) -> Result<Vec<EntityId>> {
        if query.len() + 1 != self.aug_dims {
            return Err(KgeError::Dims {
                expected: self.aug_dims - 1,
                got: query.len(),
            });
        }
        // Query side of the MIPS→L2 reduction appends a zero coordinate.
        let mut q = Vec::with_capacity(self.aug_dims);
        q.extend_from_slice(query);
        q.push(0.0);

        let results = self
            .hnsw
            .search(&SearchQuery {
                vector: q,
                k,
                filters: None,
                threshold: None,
                ef_search: Some(ef),
            })
            .map_err(|err| KgeError::Scorer(format!("ann search: {err}")))?;

        Ok(results
            .into_iter()
            .filter_map(|r| r.id.parse::<EntityId>().ok())
            .collect())
    }

    /// Exact-rerank candidate ids by the true score and return the top `k`.
    /// `exact(entity)` computes the real [`Scorer::score`] for the query's
    /// fixed relation and anchor.
    pub fn rerank<F>(cands: &[EntityId], exact: F, k: usize) -> Vec<Candidate>
    where
        F: Fn(EntityId) -> f32,
    {
        let mut scored: Vec<Candidate> = cands
            .iter()
            .map(|&entity| Candidate {
                entity,
                score: exact(entity),
            })
            .collect();
        scored.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        scored.truncate(k);
        scored
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::batch::BatchScorer;
    use crate::scorer::HolE;
    use crate::{RelationId, Side};

    /// recall@10 of ANN + exact rerank versus exhaustive scoring on a synthetic
    /// 2,000-entity table (ADR-006 gate: ≥ 0.9).
    #[test]
    fn recall_at_10_vs_exhaustive() {
        let d = 64;
        let n_entities = 2000usize;
        let n_relations = 20usize;
        let ef = 400usize;
        let k = 10usize;

        let scorer = HolE::new(d).unwrap();
        let tables = Tables::new(n_entities, n_relations, d, 0xC0FFEE);
        let ann = AnnIndex::build(&tables, &scorer).unwrap();
        let batch = BatchScorer::build(&tables, &scorer);

        // 25 tail queries (s, r, ?), pseudo-random s/r.
        let n_queries = 25usize;
        let mut hits = 0usize;
        for t in 0..n_queries as u64 {
            let s = ((t.wrapping_mul(7919)) % n_entities as u64) as EntityId;
            let r = ((t.wrapping_mul(104729)) % n_relations as u64) as RelationId;
            let q = scorer.query_vector(
                tables.relation(r).unwrap(),
                tables.entity(s).unwrap(),
                Side::Tail,
            );

            // Exhaustive ground-truth top-k (exact).
            let truth: Vec<EntityId> = batch.top_k(&q, k).into_iter().map(|c| c.entity).collect();

            // ANN candidates + exact rerank.
            let cands = ann.candidates(&q, ef, ef).unwrap();
            let sre = tables.entity(s).unwrap().to_vec();
            let rre = tables.relation(r).unwrap().to_vec();
            let reranked = AnnIndex::rerank(
                &cands,
                |e| scorer.score(&sre, &rre, tables.entity(e).unwrap()),
                k,
            );
            let got: std::collections::HashSet<EntityId> =
                reranked.into_iter().map(|c| c.entity).collect();
            hits += truth.iter().filter(|e| got.contains(e)).count();
        }

        let recall = hits as f32 / (n_queries * k) as f32;
        println!("ANN recall@{k} = {recall:.4} over {n_queries} queries (ef={ef}, n={n_entities}, d={d})");
        assert!(recall >= 0.9, "recall@{k} = {recall:.4} < 0.9");
    }

    /// Gate for #1009: router-core's `DotProduct` HNSW, queried with the
    /// *unmodified* query vector, is a maximum-inner-product search.
    #[test]
    fn dotproduct_recall_matches_mips() {
        use ruvector_router_core::index::{HnswConfig, HnswIndex};
        use ruvector_router_core::types::{DistanceMetric, SearchQuery};

        let d = 64;
        let n_entities = 2000usize;
        let n_relations = 20usize;
        let ef = 400usize;
        let k = 10usize;

        let scorer = HolE::new(d).unwrap();
        let tables = Tables::new(n_entities, n_relations, d, 0xC0FFEE);
        let batch = BatchScorer::build(&tables, &scorer);

        let cfg = HnswConfig {
            m: 16,
            ef_construction: 200,
            ef_search: 200,
            metric: DistanceMetric::DotProduct,
            dimensions: scorer.index_dims(),
        };
        let hnsw = HnswIndex::new(cfg);
        for e in 0..n_entities {
            hnsw.insert(
                e.to_string(),
                scorer.index_vector(tables.entity(e as EntityId).unwrap()),
            )
            .unwrap();
        }

        let mut hits = 0usize;
        let n_queries = 25usize;
        for t in 0..n_queries as u64 {
            let s = ((t.wrapping_mul(7919)) % n_entities as u64) as EntityId;
            let r = ((t.wrapping_mul(104729)) % n_relations as u64) as RelationId;
            let q = scorer.query_vector(
                tables.relation(r).unwrap(),
                tables.entity(s).unwrap(),
                Side::Tail,
            );
            let truth: Vec<EntityId> = batch.top_k(&q, k).into_iter().map(|c| c.entity).collect();
            let results = hnsw
                .search(&SearchQuery {
                    vector: q.clone(),
                    k: ef,
                    filters: None,
                    threshold: None,
                    ef_search: Some(ef),
                })
                .unwrap();
            let cands: Vec<EntityId> = results
                .into_iter()
                .filter_map(|r| r.id.parse().ok())
                .collect();
            let sre = tables.entity(s).unwrap().to_vec();
            let rre = tables.relation(r).unwrap().to_vec();
            let got: std::collections::HashSet<EntityId> = AnnIndex::rerank(
                &cands,
                |e| scorer.score(&sre, &rre, tables.entity(e).unwrap()),
                k,
            )
            .into_iter()
            .map(|c| c.entity)
            .collect();
            hits += truth.iter().filter(|e| got.contains(e)).count();
        }
        let recall = hits as f32 / (n_queries * k) as f32;
        println!("DotProduct recall@{k} = {recall:.4}");
        assert!(recall >= 0.9, "DotProduct recall@{k} = {recall:.4} < 0.9");
    }
}
