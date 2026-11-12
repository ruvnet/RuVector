//! Index structures for efficient vector search

pub mod flat;
#[cfg(feature = "hnsw")]
pub(crate) mod flat_graph;
#[cfg(feature = "hnsw")]
pub mod hnsw;
#[cfg(feature = "hnsw")]
pub mod turbo4;

use crate::error::Result;
use crate::types::{SearchResult, VectorId};

/// Trait for vector index implementations
pub trait VectorIndex: Send + Sync {
    /// Add a vector to the index
    fn add(&mut self, id: VectorId, vector: Vec<f32>) -> Result<()>;

    /// Add multiple vectors in batch
    fn add_batch(&mut self, entries: Vec<(VectorId, Vec<f32>)>) -> Result<()> {
        for (id, vector) in entries {
            self.add(id, vector)?;
        }
        Ok(())
    }

    /// Search for k nearest neighbors
    fn search(&self, query: &[f32], k: usize) -> Result<Vec<SearchResult>>;

    /// Search with a per-query candidate-list size (HNSW `efSearch`). `None` means the index's configured value.
    /// Indexes without such a knob ignore it.
    fn search_ef(
        &self,
        query: &[f32],
        k: usize,
        _ef_search: Option<usize>,
    ) -> Result<Vec<SearchResult>> {
        self.search(query, k)
    }

    /// The `k` nearest vectors whose id passes `filter.predicate`.
    ///
    /// The default over-fetches through [`search_ef`](Self::search_ef) and post-filters, starting from
    /// [`FilteredSearch::overfetch`] and widening ×4 until `k` vectors pass, the index is exhausted or the budget is
    /// reached. Graph indexes override it with a traversal that applies the predicate while walking.
    fn search_ef_filtered(
        &self,
        query: &[f32],
        k: usize,
        ef_search: Option<usize>,
        filter: &FilteredSearch<'_>,
    ) -> Result<FilteredResults> {
        let n = self.len();
        if k == 0 || n == 0 {
            return Ok(FilteredResults::complete(vec![]));
        }
        let limit = filter.budget.unwrap_or(usize::MAX).clamp(1, n);
        let mut fetch = filter
            .overfetch(k)
            .unwrap_or(k.saturating_mul(4))
            .min(limit);
        loop {
            let raw = self.search_ef(query, fetch, ef_search.map(|ef| ef.max(fetch)))?;
            let exhausted = raw.len() < fetch || fetch >= n;
            let mut hits: Vec<SearchResult> = raw
                .into_iter()
                .filter(|r| (filter.predicate)(&r.id))
                .collect();
            if hits.len() >= k || exhausted || fetch >= limit {
                hits.truncate(k);
                return Ok(FilteredResults {
                    complete: hits.len() >= k || exhausted,
                    results: hits,
                });
            }
            fetch = fetch.saturating_mul(4).min(limit);
        }
    }

    /// Exact distances from `query` to each listed id the index holds, as `(position in ids, distance)`; ids it
    /// does not hold are skipped. `None`
    /// when the index keeps no full-precision vectors to score against (Turbo4 stores only codes); the caller
    /// then scores from storage.
    fn exact_distances(&self, _query: &[f32], _ids: &[&str]) -> Option<Result<Vec<(usize, f32)>>> {
        None
    }

    /// Remove a vector from the index
    fn remove(&mut self, id: &VectorId) -> Result<bool>;

    /// Get the number of vectors in the index
    fn len(&self) -> usize;

    /// Check if the index is empty
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// A metadata filter as an index sees it: a membership test on ids plus what the caller knows about it.
pub struct FilteredSearch<'a> {
    /// True when the vector with this id may be returned.
    pub predicate: &'a dyn Fn(&str) -> bool,
    /// Estimated fraction of the indexed vectors that pass, when the caller knows it.
    pub selectivity: Option<f64>,
    /// Upper bound on the distance computations spent looking for passing vectors; `None` is unbounded.
    pub budget: Option<usize>,
}

impl FilteredSearch<'_> {
    /// Over-fetch a plain top-k search by this factor of the expected `k / selectivity` before post-filtering.
    pub const OVERFETCH_SLACK: f64 = 2.0;
    /// Graph indexes try the over-fetch only while it needs at most this multiple of `ef` candidates; beyond that
    /// a plain search is no longer cheap.
    pub const MAX_OVERFETCH_PER_EF: usize = 4;

    /// How many unfiltered neighbours to fetch so that about `OVERFETCH_SLACK · k` of them pass, if the
    /// selectivity is known. Assumes the filter is independent of the query's neighbourhood; when it is not, the
    /// post-filtered result comes back short and the caller must search harder.
    pub fn overfetch(&self, k: usize) -> Option<usize> {
        let s = self.selectivity.filter(|s| *s > 0.0)?;
        let want = (k as f64 * Self::OVERFETCH_SLACK / s.min(1.0)).ceil();
        Some(if want >= usize::MAX as f64 {
            usize::MAX
        } else {
            want as usize
        })
    }

    /// [`overfetch`](Self::overfetch), if a graph index with beam width `ef` should try it.
    pub fn graph_overfetch(&self, k: usize, ef: usize) -> Option<usize> {
        self.overfetch(k)
            .filter(|f| *f <= ef.max(k).saturating_mul(Self::MAX_OVERFETCH_PER_EF))
    }
}

/// What a filtered index search returns.
#[derive(Debug, Default)]
pub struct FilteredResults {
    /// Passing vectors, ascending by distance; at most `k`.
    pub results: Vec<SearchResult>,
    /// False when the search stopped on its work budget, so nearer passing vectors may have been missed (and
    /// `results` may hold fewer than `k` although more exist). A caller that needs exact completeness re-runs
    /// the query as an exact scan.
    pub complete: bool,
}

impl FilteredResults {
    /// Results the index vouches for.
    pub fn complete(results: Vec<SearchResult>) -> Self {
        Self {
            results,
            complete: true,
        }
    }
}
