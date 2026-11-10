//! Filtered search: a cardinality-based plan per query, as Qdrant and Weaviate do.
//!
//! The metadata view gives the exact match count `m` (for a multi-clause filter, the smallest clause's count — an upper
//! bound), so the selectivity `s = m / n` is known before searching. On an HNSW index:
//!
//! 1. if `s` is high enough that a plain search widened to `~2k / s` candidates stays cheap (at most
//!    `FilteredSearch::MAX_OVERFETCH_PER_EF · ef`), the index does that and post-filters;
//! 2. otherwise, if `m` is small, the matching vectors are scanned exactly (`m` distance computations);
//! 3. otherwise the index runs the ACORN-style predicate-aware walk (`index::flat_graph::filtered`) under a work
//!    budget.
//!
//! A graph search that comes back short of `min(k, m)`, or stopped on its budget (typically a filter correlated with
//! the vectors, so that no passing vector is near the query), is redone as an exact scan. Results are therefore never
//! fewer than `min(k, #matching)`. A flat index always scans the matches exactly.

use super::VectorDB;
use crate::error::Result;
use crate::index::{FilteredSearch, VectorIndex};
use crate::types::{SearchQuery, SearchResult, VectorId};
use serde_json::Value;
use std::collections::{HashMap, HashSet};

/// How a filtered query is executed. [`VectorDB::search`] always uses `Auto`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FilterPlan {
    /// Pick per query from the match count, falling back to `Exact` if the graph search returns short or incomplete.
    #[default]
    Auto,
    /// The predicate-aware graph walk alone: no selectivity hint (so no over-fetch shortcut) and no fallback. For
    /// measurement.
    Graph,
    /// Exact scan of the matching vectors only.
    Exact,
}

/// Scan exactly when `m² ≤ EXACT_RATIO · ef · n`: a graph search touches on the order of `ef · n / m` nodes per
/// query at selectivity `m / n`, each dearer than a scan step. Calibrated with the `filtered_search_bench` example.
const EXACT_RATIO: usize = 4;
/// The walk may spend this multiple of its expected `ef · n / m` visits before giving up to the exact fallback.
const BUDGET_SLACK: usize = 4;

fn prefers_exact(m: usize, ef: usize, n: usize) -> bool {
    m.saturating_mul(m) <= EXACT_RATIO.saturating_mul(ef).saturating_mul(n)
}

fn walk_budget(m: usize, ef: usize, n: usize) -> usize {
    let expected = ef.saturating_mul(n) / m.max(1);
    expected
        .saturating_mul(BUDGET_SLACK)
        .max(ef.saturating_mul(8))
        .min(n.max(1))
}

impl VectorDB {
    pub(super) fn search_filtered(
        &self,
        query: &SearchQuery,
        filter: &HashMap<String, Value>,
        plan: FilterPlan,
    ) -> Result<Vec<SearchResult>> {
        let index = self.index.read();
        let meta = self.metadata.read();
        let Some(clauses) = meta.clauses(filter) else {
            return Ok(vec![]);
        };
        let k = query.k;
        if k == 0 {
            return Ok(vec![]);
        }
        let n = index.len();
        let m = clauses[0].len();
        let passes = |id: &str| clauses.iter().all(|s| s.contains(id));
        let ef = query
            .ef_search
            .or(self.options.hnsw_config.as_ref().map(|c| c.ef_search))
            .unwrap_or(k)
            .max(k);

        // Turbo4 over-fetches 2k for its f32 re-rank (ADR-297 §2).
        let verify = self.turbo4_verify();
        let fetch_k = if verify { k.saturating_mul(2) } else { k };
        let spec = FilteredSearch {
            predicate: &passes,
            selectivity: (plan == FilterPlan::Auto).then(|| m as f64 / n.max(1) as f64),
            budget: Some(walk_budget(m, ef, n)),
        };
        // A flat index scans everything per query anyway, so scanning just the matches is never worse.
        let graph_index = cfg!(feature = "hnsw") && self.options.hnsw_config.is_some();
        let exact = match plan {
            FilterPlan::Exact => true,
            FilterPlan::Graph => false,
            FilterPlan::Auto => {
                !graph_index
                    || (spec.graph_overfetch(fetch_k, ef).is_none() && prefers_exact(m, ef, n))
            }
        };
        let mut results = if exact {
            self.exact_filtered(&**index, query, &clauses)?
        } else {
            let found =
                index.search_ef_filtered(&query.vector, fetch_k, Some(ef.max(fetch_k)), &spec)?;
            if plan == FilterPlan::Auto && (!found.complete || found.results.len() < k.min(m)) {
                self.exact_filtered(&**index, query, &clauses)?
            } else {
                let mut results = found.results;
                self.enrich(&mut results);
                if verify {
                    self.rerank(&query.vector, &mut results);
                    results.truncate(k);
                }
                results
            }
        };
        // The view mirrors storage, so this drops nothing; it keeps the result contract independent of the view.
        results.retain(|r| {
            r.metadata.as_ref().is_some_and(|md| {
                filter
                    .iter()
                    .all(|(key, value)| md.get(key).is_some_and(|v| v == value))
            })
        });
        Ok(results)
    }

    /// Exact top-k over the vectors that pass every clause.
    fn exact_filtered(
        &self,
        index: &dyn VectorIndex,
        query: &SearchQuery,
        clauses: &[&HashSet<VectorId>],
    ) -> Result<Vec<SearchResult>> {
        let ids: Vec<&str> = clauses[0]
            .iter()
            .filter(|id| clauses[1..].iter().all(|s| s.contains(*id)))
            .map(String::as_str)
            .collect();
        let mut scored = match index.exact_distances(&query.vector, &ids) {
            Some(scored) => scored?,
            None => self.storage_distances(&query.vector, &ids)?,
        };
        let by_dist = |a: &(usize, f32), b: &(usize, f32)| a.1.total_cmp(&b.1);
        if scored.len() > query.k {
            scored.select_nth_unstable_by(query.k, by_dist);
            scored.truncate(query.k);
        }
        scored.sort_unstable_by(by_dist);
        let mut results: Vec<SearchResult> = scored
            .into_iter()
            .map(|(pos, score)| SearchResult {
                id: ids[pos].to_string(),
                score,
                vector: None,
                metadata: None,
            })
            .collect();
        self.enrich(&mut results);
        Ok(results)
    }

    /// Exact distances from stored vectors, for an index that keeps none. With Turbo4 these use the same metric as
    /// the verification re-rank, so both plans score alike.
    fn storage_distances(&self, query: &[f32], ids: &[&str]) -> Result<Vec<(usize, f32)>> {
        let mut out = Vec::with_capacity(ids.len());
        for (pos, id) in ids.iter().enumerate() {
            if let Some(entry) = self.storage.get(id)? {
                let d = if self.turbo4_verify() {
                    crate::encoding::metric_distance(
                        self.options.distance_metric,
                        query,
                        &entry.vector,
                    )
                } else {
                    crate::distance::distance(query, &entry.vector, self.options.distance_metric)?
                };
                out.push((pos, d));
            }
        }
        Ok(out)
    }
}
