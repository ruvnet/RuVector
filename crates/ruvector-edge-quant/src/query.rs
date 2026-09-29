//! Query: 1-bit symmetric scan (ruvector-rabitq's popcount kernel) with an
//! optional exact f32 rerank of the best `top_k × rerank_factor` candidates.
//!
//! The f32 originals live in the store, not in the shard (they would blow the
//! 14 MB resident cap at 50k × 384), so rerank goes through a
//! [`RerankSource`] callback that fetches only the candidate rows.

use crate::budget;
use crate::error::{BudgetResource, QuantError, Result};
use crate::shard::QuantShard;
use ruvector_edge_store::distance::{self, Metric};
use std::cmp::Ordering;
use std::collections::BinaryHeap;

/// Fetches f32 originals for rerank. The store implements this over its
/// rows (DO SQLite / resident `VectorShard`).
pub trait RerankSource {
    /// Return one entry per key, aligned with `keys`. `None` for a row that
    /// no longer exists (deleted between scan and fetch): it is dropped from
    /// the results rather than failing the query.
    fn fetch(&mut self, keys: &[u64]) -> std::result::Result<Vec<Option<Vec<f32>>>, String>;
}

/// Per-query knobs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QueryOptions {
    /// Results wanted (`1..=budget.max_top_k`).
    pub top_k: u32,
    /// Candidates reranked = `top_k × rerank_factor`, clamped to
    /// `budget.max_rerank_candidates` (ignored without a rerank source;
    /// `0`/`1` rerank exactly `top_k`). At 50k × 384 clustered data,
    /// 50 gives recall@10 ≈ 0.999 and 20 gives ≈ 0.77 (see `tests/scale.rs`).
    /// Above `top_k = 20` the default factor hits the 1,000-candidate cap;
    /// the clamp is reported in [`QueryStats::clamped`], and `tests/scale.rs`
    /// gates recall@100 at the capped 1,000 candidates.
    pub rerank_factor: u32,
}

impl Default for QueryOptions {
    fn default() -> Self {
        QueryOptions {
            top_k: 10,
            rerank_factor: 50,
        }
    }
}

/// One result.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct QueryHit {
    /// Store row key.
    pub key: u64,
    /// Distance in the collection metric (exact if `exact`, else the
    /// RaBitQ estimate).
    pub distance: f32,
    /// `true` when reranked against the f32 original.
    pub exact: bool,
}

/// Work actually done (for the response and admission telemetry).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct QueryStats {
    /// Codes scanned.
    pub scanned: u64,
    /// Candidates kept by the scan.
    pub candidates: u64,
    /// Candidates reranked against f32.
    pub reranked: u64,
    /// Candidates the store no longer had.
    pub missing: u64,
    /// Work units charged (checked before the scan).
    pub units: u64,
    /// Candidates the options asked for (`top_k × rerank_factor` with a
    /// rerank source, else `top_k`), before the budget cap.
    pub requested_candidates: u64,
    /// `true` when `requested_candidates` exceeded
    /// `budget.max_rerank_candidates` and the scan kept only the cap
    /// (effective factor = cap / top_k). Surface it to the caller.
    pub clamped: bool,
}

/// Query result.
#[derive(Debug, Clone, PartialEq)]
pub struct QueryOutcome {
    /// Best first; ties broken by key.
    pub hits: Vec<QueryHit>,
    /// Work accounting.
    pub stats: QueryStats,
}

#[derive(Clone, Copy)]
struct Cand {
    score: f64,
    key: u64,
    pos: u32,
}
impl Cand {
    fn cmp_key(&self, o: &Self) -> Ordering {
        self.score
            .total_cmp(&o.score)
            .then_with(|| self.key.cmp(&o.key))
    }
}
impl PartialEq for Cand {
    fn eq(&self, o: &Self) -> bool {
        self.cmp_key(o) == Ordering::Equal
    }
}
impl Eq for Cand {}
impl PartialOrd for Cand {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
        Some(self.cmp(o))
    }
}
impl Ord for Cand {
    fn cmp(&self, o: &Self) -> Ordering {
        self.cmp_key(o)
    }
}

/// Estimated distance (lower is better) from the estimated cosine, in f64:
/// stored norms are finite f32 (≤ 3.4e38), so their squares and products
/// cannot overflow here, where f32 would turn `|x| ≳ 1.8e19` into NaN/inf.
#[inline]
fn estimate(metric: Metric, q_norm: f64, x_norm: f32, est_cos: f32) -> f64 {
    let (x, c) = (f64::from(x_norm), f64::from(est_cos));
    match metric {
        Metric::L2 => (q_norm * q_norm + x * x - 2.0 * q_norm * x * c)
            .max(0.0)
            .sqrt(),
        Metric::Cosine => 1.0 - c,
        Metric::Dot => -(q_norm * x * c),
    }
}

impl QuantShard {
    /// Top-k search. Budgets (top_k, rerank candidates, work units) are
    /// checked before any scanning; an empty shard returns no hits.
    pub fn query(
        &self,
        q: &[f32],
        opts: &QueryOptions,
        rerank: Option<&mut dyn RerankSource>,
    ) -> Result<QueryOutcome> {
        let b = self.cfg.budget;
        if q.len() != self.cfg.dim {
            return Err(QuantError::DimensionMismatch {
                expected: self.cfg.dim,
                actual: q.len(),
            });
        }
        if q.iter().any(|x| !x.is_finite()) {
            return Err(QuantError::NonFinite);
        }
        if opts.top_k == 0 {
            return Err(QuantError::InvalidConfig("top_k must be >= 1"));
        }
        budget::check(
            BudgetResource::TopK,
            u64::from(b.max_top_k),
            u64::from(opts.top_k),
        )?;
        let n = self.keys.len() as u64;
        let top_k = u64::from(opts.top_k);
        let (want, requested) = if rerank.is_some() {
            // Reranking fewer than `top_k` rows cannot answer the query: 413.
            let cap = u64::from(b.max_rerank_candidates);
            budget::check(BudgetResource::RerankCandidates, cap, top_k)?;
            let requested = top_k * u64::from(opts.rerank_factor.max(1));
            (requested.min(cap), requested)
        } else {
            (top_k, top_k)
        };
        let keep = want.min(n);
        let rerank_rows = if rerank.is_some() { keep } else { 0 };
        let units = budget::query_units(n, self.cfg.dim, self.cfg.rotation, rerank_rows);
        budget::check(BudgetResource::QueryUnits, b.max_query_units, units)?;
        let mut stats = QueryStats {
            units,
            requested_candidates: requested,
            clamped: requested > want,
            ..QueryStats::default()
        };
        if n == 0 {
            return Ok(QueryOutcome {
                hits: Vec::new(),
                stats,
            });
        }

        let cands = self.scan_topk(q, keep as usize);
        stats.scanned = n;
        stats.candidates = cands.len() as u64;
        let hits = match rerank {
            None => cands
                .iter()
                .take(opts.top_k as usize)
                .map(|c| QueryHit {
                    key: c.key,
                    distance: distance::to_wire(c.score),
                    exact: false,
                })
                .collect(),
            Some(src) => self.rerank(q, &cands, opts.top_k as usize, src, &mut stats)?,
        };
        Ok(QueryOutcome { hits, stats })
    }

    /// Symmetric scan: best `keep` rows by estimated distance, sorted.
    fn scan_topk(&self, q: &[f32], keep: usize) -> Vec<Cand> {
        let dim = self.cfg.dim;
        let mut qcode = vec![0u64; self.n_words];
        let mut unit = Vec::with_capacity(dim);
        let mut rotated = vec![0.0f32; dim];
        crate::shard::encode_into(&self.rotation, q, &mut unit, &mut rotated, &mut qcode);
        // f64: a query need not have an f32-representable norm (it is not stored).
        let q_norm = distance::norm(q);
        let n = self.keys.len();
        let mut agree = vec![0u32; n];
        ruvector_rabitq::scan::scan(
            &self.packed,
            self.n_words,
            n,
            &qcode,
            self.last_word_mask,
            &mut agree,
        );
        let metric = self.cfg.metric;
        let mut heap: BinaryHeap<Cand> = BinaryHeap::with_capacity(keep + 1);
        for (i, &a) in agree.iter().enumerate() {
            // `a ∈ 0..=dim` (masked popcount), so the LUT index is in range.
            let est_cos = self.cos_lut[a as usize];
            let c = Cand {
                score: estimate(metric, q_norm, self.norms[i], est_cos),
                key: self.keys[i],
                pos: i as u32,
            };
            if heap.len() < keep {
                heap.push(c);
            } else if let Some(top) = heap.peek() {
                if c < *top {
                    heap.pop();
                    heap.push(c);
                }
            }
        }
        heap.into_sorted_vec()
    }

    fn rerank(
        &self,
        q: &[f32],
        cands: &[Cand],
        top_k: usize,
        src: &mut dyn RerankSource,
        stats: &mut QueryStats,
    ) -> Result<Vec<QueryHit>> {
        let keys: Vec<u64> = cands.iter().map(|c| c.key).collect();
        let rows = src.fetch(&keys).map_err(QuantError::Rerank)?;
        if rows.len() != keys.len() {
            return Err(QuantError::Rerank("row count mismatch".into()));
        }
        let metric = self.cfg.metric;
        let q_norm = distance::norm(q);
        let mut scored: Vec<(f64, u64)> = Vec::with_capacity(rows.len());
        for (c, row) in cands.iter().zip(rows) {
            debug_assert_eq!(self.keys[c.pos as usize], c.key);
            let Some(row) = row else {
                stats.missing += 1;
                continue;
            };
            if row.len() != self.cfg.dim {
                return Err(QuantError::Rerank("row dimension mismatch".into()));
            }
            let d = distance::distance(metric, q, q_norm, &row, distance::norm(&row));
            scored.push((d, c.key));
            stats.reranked += 1;
        }
        scored.sort_by(|a, b| a.0.total_cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
        Ok(scored
            .into_iter()
            .take(top_k)
            .map(|(d, key)| QueryHit {
                key,
                distance: distance::to_wire(d),
                exact: true,
            })
            .collect())
    }
}
