//! Filtered HNSW query path: a selectivity planner in front of the
//! predicate-aware graph traversal in `rvf_index::HnswGraph::search_filtered`.
//!
//! Before this path existed every query carrying a metadata filter took the
//! exact scan. The planner now:
//!
//! 1. estimates the filter's selectivity on a deterministic stride sample of
//!    live ids (charged to the query's metadata-filter budget like any other
//!    evaluation);
//! 2. sends low-selectivity filters (`< FILTERED_HNSW_MIN_SELECTIVITY`) to
//!    the exact scan, which evaluates the predicate first and computes
//!    distances only for the matches -- cheaper than walking a graph whose
//!    nodes mostly fail;
//! 3. under `QualityPreference::Auto` / `PreferQuality`, sends the query to
//!    the exact scan when the graph's measured recall@10 at this `ef` is
//!    below `FILTERED_HNSW_MIN_GRAPH_RECALL` (see `graph_recall`; measured
//!    once per index and cached), since the filtered walk cannot beat the
//!    graph it walks;
//! 4. otherwise runs the filtered traversal under a visit budget (a third of
//!    the store), and falls back to the exact scan whenever the
//!    traversal exhausts that budget or returns fewer than `k` hits (which
//!    also covers "fewer than `k` matches exist", so the result count is
//!    always `min(k, matches)`).
//!
//! COW children run the same traversal over the child's own graph (or an
//! exact scan of a small child slab) and over the parent's graph, with the
//! child's membership, overrides and COW-resolved metadata applied as the
//! parent-side predicate.

use super::*;
use rvf_types::quality::QualityPreference;

/// Metadata-filter accounting for one query, shared by the index path and
/// any exact-scan fallback so the two cannot each spend a full budget.
pub(super) struct FilterScan {
    pub(super) started: std::time::Instant,
    pub(super) records: u64,
    pub(super) bytes: u64,
}

impl FilterScan {
    pub(super) fn new() -> Self {
        Self {
            started: std::time::Instant::now(),
            records: 0,
            bytes: 0,
        }
    }
}

/// Ids sampled to estimate a filter's selectivity.
const SELECTIVITY_SAMPLE: usize = 512;

/// Below this estimated selectivity the exact scan (predicate first, then
/// distances for the matches only) beats the filtered traversal. Measured
/// with `examples/filtered_query_bench.rs` (100k x 128 low-rank, L2,
/// ef = 256, 200 queries) as p50 speedup over the scan: 3.2x at 50%, 2.1x at
/// 25%, 1.8x at 20%, 1.0x at 10%, 1.0x at 5% -- the threshold sits between 10% and
/// 20%, where the traversal's random-access predicate evaluations stop
/// paying for themselves against the sequential scan.
pub(crate) const FILTERED_HNSW_MIN_SELECTIVITY: f64 = 0.15;

/// Minimum measured graph recall@10 for the filtered traversal to serve a
/// query under `QualityPreference::Auto` / `PreferQuality`. Below it the
/// graph cannot deliver a near-exact filtered answer at the planner's `ef`
/// (the filtered walk lands a few points under the unfiltered graph), so the
/// exact scan answers instead. Measured with `examples/filtered_query_bench.rs`
/// (100k x 128, L2): low-rank, embedding-like data calibrates at 1.00 and
/// keeps the HNSW speedup; uniform i.i.d. data calibrates at about 0.7, where
/// the filtered walk returned 0.60-0.63 recall@10 (#1370) and is now exact.
pub(crate) const FILTERED_HNSW_MIN_GRAPH_RECALL: f64 = 0.95;

/// Stored vectors used as calibration queries by `graph_recall`.
const GRAPH_RECALL_SAMPLES: usize = 16;

/// Neighbours compared per calibration query.
const GRAPH_RECALL_K: usize = 10;

/// Predicate evaluations the traversal may spend before giving up and
/// letting the exact scan answer: a third of the store (so on large stores a
/// bad estimate costs at most ~1.33x the scan it replaced), but never less
/// than `64 * ef` (about `ef` pops at full layer-0 degree), which small
/// stores need to converge at all.
fn visit_budget(ef: usize, total: usize) -> usize {
    (total / 3).max(ef.saturating_mul(64)).min(total)
}

fn to_results(hits: Vec<(u64, f32)>) -> Vec<SearchResult> {
    hits.into_iter()
        .map(|(id, distance)| SearchResult {
            id,
            distance,
            retrieval_quality: rvf_types::quality::RetrievalQuality::Full,
            metadata: None,
        })
        .collect()
}

impl RvfStore {
    /// Whether a filtered query may try the filtered index path.
    pub(super) fn filtered_index_eligible(&self, options: &QueryOptions) -> bool {
        if options.force_exact || options.filter.is_none() {
            return false;
        }
        if self.cow_engine.is_some() {
            return self.parent_path.is_some();
        }
        if self.membership_filter.is_some() {
            return false;
        }
        self.index_population_ok()
    }

    /// Enough vectors for a graph, and few enough soft-deleted ones.
    fn index_population_ok(&self) -> bool {
        let total = self.vectors.len();
        if total < INDEX_MIN_VECTORS {
            return false;
        }
        let deleted = self.deletion_bitmap.count();
        (deleted as f64) <= (total as f64) * INDEX_MAX_DELETED_FRACTION
    }

    /// Evaluate `filter` (against this store's metadata) for one id,
    /// charging the query's filter budget.
    pub(super) fn filter_passes(
        &self,
        filter: &FilterExpr,
        id: u64,
        options: &QueryOptions,
        scan: &mut FilterScan,
    ) -> Result<bool, RvfError> {
        let record = self.metadata.fields(id);
        self.check_metadata_filter_budget(
            filter::record_decoded_size(record) as u64,
            options,
            scan,
        )?;
        Ok(filter::evaluate_record(filter, record))
    }

    /// Estimated fraction of `ids` (restricted to `visible`) passing
    /// `filter`, from a deterministic stride sample. `None` when no sampled
    /// id is visible.
    fn estimate_selectivity<'a>(
        &self,
        filter: &FilterExpr,
        options: &QueryOptions,
        scan: &mut FilterScan,
        ids: impl Iterator<Item = &'a u64>,
        population: usize,
        visible: impl Fn(u64) -> bool,
    ) -> Result<Option<f64>, RvfError> {
        let step = (population / SELECTIVITY_SAMPLE).max(1);
        let (mut sampled, mut passed) = (0usize, 0usize);
        for &id in ids.step_by(step).take(SELECTIVITY_SAMPLE) {
            if !visible(id) {
                continue;
            }
            sampled += 1;
            if self.filter_passes(filter, id, options, scan)? {
                passed += 1;
            }
        }
        Ok((sampled > 0).then(|| passed as f64 / sampled as f64))
    }

    /// `ef` used by the filtered traversal: the same floor as the unfiltered
    /// index path, so on a given graph a filtered query is about as accurate
    /// as an unfiltered one (measured on uniform-random 100k x 128, where the
    /// graph itself is weak: 0.68 vs 0.71 recall@10 at 50% selectivity; a
    /// floor of 128 dropped that to 0.51). Larger `ef_search` is honored.
    fn filtered_ef(k: usize, options: &QueryOptions) -> usize {
        (options.ef_search as usize).max(k).max(INDEX_MIN_EF_SEARCH)
    }

    /// Recall@10 of the unfiltered graph walk at `ef`, measured against the
    /// exact scan with [`GRAPH_RECALL_SAMPLES`] stored vectors (spread over
    /// the store) as queries, the query vector itself excluded from both
    /// lists. Cached on the index until the graph doubles in size, so the
    /// O(samples x N) scan is paid once per index, not per query.
    pub(super) fn graph_recall(&self, idx: &VectorIndex, ef: usize) -> f64 {
        if let Some(recall) = idx.cached_graph_recall(ef) {
            return recall;
        }
        let k = GRAPH_RECALL_K;
        let metric = self.options.metric;
        let live = |id: u64| !self.deletion_bitmap.is_deleted(id);
        let step = (self.vectors.len() / GRAPH_RECALL_SAMPLES).max(1);
        // Headroom for soft-deleted hits, as on the unfiltered index path.
        let fetch = k + 1 + self.deletion_bitmap.count().min(2 * k + 16);
        let samples: Vec<u64> = self
            .vectors
            .ids()
            .step_by(step)
            .copied()
            .filter(|&id| live(id))
            .take(GRAPH_RECALL_SAMPLES)
            .collect();

        let (mut hits, mut slots) = (0usize, 0usize);
        for &q in &samples {
            let Some(query) = self.vectors.get(q) else {
                continue;
            };
            let norm_sq = if metric == DistanceMetric::Cosine {
                query.iter().map(|x| x * x).sum()
            } else {
                0.0f32
            };
            let mut exact: Vec<(f32, u64)> = self
                .vectors
                .iter()
                .filter(|&(id, _)| id != q && live(id))
                .map(|(id, v)| (compute_distance(query, v, &metric, norm_sq), id))
                .collect();
            let take = k.min(exact.len());
            if take == 0 {
                continue;
            }
            exact.select_nth_unstable_by(take - 1, |a, b| a.0.total_cmp(&b.0));
            let truth: HashSet<u64> = exact[..take].iter().map(|&(_, id)| id).collect();
            let found = idx
                .search(query, fetch, ef.max(fetch), &self.vectors, metric)
                .into_iter()
                .filter(|&(id, _)| id != q && live(id))
                .take(take)
                .filter(|(id, _)| truth.contains(id))
                .count();
            hits += found;
            slots += take;
        }
        let recall = if slots == 0 {
            1.0
        } else {
            hits as f64 / slots as f64
        };
        idx.store_graph_recall(ef, recall);
        recall
    }

    /// Serve a filtered query through the HNSW index.
    ///
    /// `Ok(None)` means "use the exact scan" (low selectivity, index still
    /// building, budget exhausted, or fewer than `k` hits); `Err` is a
    /// metadata-filter budget, deadline or cancellation abort.
    pub(super) fn query_via_filtered_index(
        &self,
        vector: &[f32],
        k: usize,
        options: &QueryOptions,
        filter: &FilterExpr,
        scan: &mut FilterScan,
    ) -> Result<Option<Vec<SearchResult>>, RvfError> {
        if k == 0 {
            return Ok(None);
        }
        if self.cow_engine.is_some() {
            return self.query_via_filtered_index_cow(vector, k, options, filter, scan);
        }
        let selectivity = self.estimate_selectivity(
            filter,
            options,
            scan,
            self.vectors.ids(),
            self.vectors.len(),
            |id| !self.deletion_bitmap.is_deleted(id),
        )?;
        if !selectivity.is_some_and(|s| s >= FILTERED_HNSW_MIN_SELECTIVITY) {
            return Ok(None);
        }
        let ef = Self::filtered_ef(k, options);
        if !matches!(
            options.quality_preference,
            QualityPreference::PreferLatency | QualityPreference::AcceptDegraded
        ) {
            match self.with_index(|idx| self.graph_recall(idx, ef)) {
                Some(recall) if recall >= FILTERED_HNSW_MIN_GRAPH_RECALL => {}
                _ => return Ok(None),
            }
        }
        let budget = visit_budget(ef, self.vectors.len());

        let mut abort: Option<RvfError> = None;
        let mut predicate = |id: u64| {
            if abort.is_some() || self.deletion_bitmap.is_deleted(id) {
                return false;
            }
            match self.filter_passes(filter, id, options, scan) {
                Ok(pass) => pass,
                Err(e) => {
                    abort = Some(e);
                    false
                }
            }
        };
        let outcome = self.with_index(|idx| {
            idx.search_filtered(
                vector,
                k,
                ef,
                budget,
                &self.vectors,
                self.options.metric,
                &mut predicate,
            )
        });
        if let Some(e) = abort {
            return Err(e);
        }
        match outcome {
            Some(o) if !o.budget_exhausted && o.results.len() >= k => {
                Ok(Some(to_results(o.results)))
            }
            _ => Ok(None),
        }
    }

    /// COW variant: filtered traversal over the child and parent graphs,
    /// merged with child-wins semantics (see `query_via_index_cow`).
    fn query_via_filtered_index_cow(
        &self,
        vector: &[f32],
        k: usize,
        options: &QueryOptions,
        filter: &FilterExpr,
        scan: &mut FilterScan,
    ) -> Result<Option<Vec<SearchResult>>, RvfError> {
        let mut guard = self.parent_store.lock().unwrap_or_else(|e| e.into_inner());
        if guard.is_none() {
            if let Some(ref parent_path) = self.parent_path {
                if let Ok(p) = RvfStore::open_readonly(parent_path) {
                    *guard = Some(Box::new(p));
                }
            }
        }
        let parent = match guard.as_ref() {
            Some(p) if p.index_population_ok() && p.options.metric == self.options.metric => p,
            _ => return Ok(None),
        };

        let child_ids: HashSet<u64> = self.vectors.ids().copied().collect();
        let parent_visible = |id: u64| {
            self.membership_filter
                .as_ref()
                .is_none_or(|mf| mf.contains(id))
                && !child_ids.contains(&id)
                && !parent.deletion_bitmap.is_deleted(id)
        };

        let selectivity = self.estimate_selectivity(
            filter,
            options,
            scan,
            parent.vectors.ids(),
            parent.vectors.len(),
            parent_visible,
        )?;
        if !selectivity.is_some_and(|s| s >= FILTERED_HNSW_MIN_SELECTIVITY) {
            return Ok(None);
        }
        let ef = Self::filtered_ef(k, options);
        let mut merged: Vec<(u64, f32)> = Vec::with_capacity(2 * k);
        let mut abort: Option<RvfError> = None;

        // ── Parent arm ──────────────────────────────────────────────────
        let budget = visit_budget(ef, parent.vectors.len());
        let mut parent_pred = |id: u64| {
            if abort.is_some() || !parent_visible(id) {
                return false;
            }
            match self.filter_passes(filter, id, options, scan) {
                Ok(pass) => pass,
                Err(e) => {
                    abort = Some(e);
                    false
                }
            }
        };
        let outcome = parent.with_index(|idx| {
            idx.search_filtered(
                vector,
                k,
                ef,
                budget,
                &parent.vectors,
                parent.options.metric,
                &mut parent_pred,
            )
        });
        if let Some(e) = abort.take() {
            return Err(e);
        }
        match outcome {
            Some(o) if !o.budget_exhausted => merged.extend(o.results),
            _ => return Ok(None),
        }

        // ── Child arm ───────────────────────────────────────────────────
        if self.index_population_ok() {
            let budget = visit_budget(ef, self.vectors.len());
            let mut child_pred = |id: u64| {
                if abort.is_some() || self.deletion_bitmap.is_deleted(id) {
                    return false;
                }
                match self.filter_passes(filter, id, options, scan) {
                    Ok(pass) => pass,
                    Err(e) => {
                        abort = Some(e);
                        false
                    }
                }
            };
            let outcome = self.with_index(|idx| {
                idx.search_filtered(
                    vector,
                    k,
                    ef,
                    budget,
                    &self.vectors,
                    self.options.metric,
                    &mut child_pred,
                )
            });
            if let Some(e) = abort {
                return Err(e);
            }
            match outcome {
                Some(o) if !o.budget_exhausted => merged.extend(o.results),
                _ => return Ok(None),
            }
        } else {
            let query_norm_sq = if self.options.metric == DistanceMetric::Cosine {
                vector.iter().map(|x| x * x).sum()
            } else {
                0.0f32
            };
            for (id, v) in self.vectors.iter() {
                if self.deletion_bitmap.is_deleted(id) {
                    continue;
                }
                if self.filter_passes(filter, id, options, scan)? {
                    let d = compute_distance(vector, v, &self.options.metric, query_norm_sq);
                    merged.push((id, d));
                }
            }
        }

        merged.sort_by(|a, b| a.1.total_cmp(&b.1).then_with(|| a.0.cmp(&b.0)));
        merged.truncate(k);
        if merged.len() < k {
            return Ok(None);
        }
        Ok(Some(to_results(merged)))
    }
}
