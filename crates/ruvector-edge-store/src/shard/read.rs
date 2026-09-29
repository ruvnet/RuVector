//! Shard reads: two-stage top-k (code-space candidates from the resident
//! index, then an exact f32 rerank from SQLite) with a metadata filter, and
//! fetch.
//!
//! * `flat`: every live row passing the filter is scored against its int8
//!   code; the best `rerank` candidates are reranked exactly.
//! * `hnsw`: a beam of width `ef` over codes gives the candidates. With a
//!   filter the beam is post-filtered; if fewer than `top_k` survive and
//!   at most [`EXACT_FILTER_MAX`] rows match, every matching row is
//!   reranked (exact).
//!
//! Ranking is by exact `f64` distance then id, as at M1, so results merge
//! across shards bit-exactly. Each query reports its step cost so the
//! executor can enforce the per-request budget ([`MAX_QUERY_STEPS`]).

use super::ann::{iid32, Ann};
use super::codec::{IndexConfig, ShardConfig};
use super::rerank::{exact_top_k, fetch_ids};
use super::VectorShard;
use crate::distance::{norm, Metric};
use crate::error::{ErrorCode, OpError};
use crate::filter::Filter;
use crate::ports::SqlStore;
use ruvector_edge_tenancy::quota::limits::MAX_TOP_K;
use ruvector_edge_tenancy::{DoMeta, IdentityCheck, VectorId};
use serde::{Deserialize, Serialize};
use serde_json::Value as Json;
use std::cmp::Ordering;
use std::collections::BinaryHeap;

/// Maximum ids per fetch (§7).
pub const MAX_FETCH_IDS: usize = 100;
/// Per-request scan budget in steps (one step = one code distance, one
/// filter value comparison or one reranked row), `413 budget_exceeded`
/// beyond it.
pub const MAX_QUERY_STEPS: u64 = 8_000_000;
/// Largest client `ef` (HNSW beam width).
pub const MAX_EF: u32 = 2048;
/// Default `ef` for cosine HNSW (recall@10 ≥ 0.95 on isotropic 384-d).
pub const EF_DEFAULT_COSINE: u32 = 1024;
/// Default `ef` for l2/dot HNSW.
pub const EF_DEFAULT_L2_DOT: u32 = 512;
/// Rerank at least this many candidates (`max(40, 4·top_k)` by default).
pub const DEFAULT_RERANK_MIN: u32 = 40;
/// Largest client `rerank` (rows fetched from SQLite per shard).
pub const MAX_RERANK: u32 = 1000;
/// HNSW + filter: rerank every matching row when there are this few.
pub const EXACT_FILTER_MAX: usize = 2000;

/// A query (the §7 body minus `text`, which is M3).
#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct QueryRequest {
    /// Query vector, length = collection dimension.
    pub vector: Vec<f32>,
    /// `1..=100`.
    pub top_k: u32,
    /// Optional filter on declared keys (see [`crate::filter`]).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub filter: Option<Json>,
    /// Any of `"metadata"`, `"values"`.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub include: Vec<String>,
    /// HNSW beam width, `1..=MAX_EF` (ignored by `flat`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ef: Option<u32>,
    /// Candidates reranked exactly, `top_k..=MAX_RERANK`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rerank: Option<u32>,
}

/// One result row.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Match {
    /// Vector id.
    pub id: String,
    /// Distance, lower is closer: cosine `1 - cos`, l2 Euclidean, dot `-a·b`
    /// (clamped to a finite `f32`).
    pub distance: f32,
    /// Metadata, when requested and present.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<Json>,
    /// Values, when requested.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub values: Option<Vec<f32>>,
    /// Exact `f64` rank key (cross-shard merge).
    #[serde(skip)]
    pub(crate) score: f64,
}

impl Match {
    /// The exact `f64` rank key. DO adapter hook: a gateway that fans a
    /// query out to `VectorShard` Durable Objects carries this key over the
    /// wire and merges by `(rank_score, id)` ascending, exactly like
    /// [`Match::rank_cmp`], instead of by the clamped `f32` distance.
    pub fn rank_score(&self) -> f64 {
        self.score
    }

    /// Merge order: `(score, id)` ascending.
    pub(crate) fn rank_cmp(&self, other: &Match) -> Ordering {
        self.score
            .total_cmp(&other.score)
            .then_with(|| self.id.cmp(&other.id))
    }
}

/// Query result.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct QueryOutcome {
    /// Top-k by `(distance, id)` ascending.
    pub matches: Vec<Match>,
    /// Rows scored (work-unit input).
    pub scanned: u64,
}

/// A query validated against a collection config.
#[derive(Debug, Clone)]
pub struct ValidQuery {
    /// Parsed filter.
    pub filter: Filter,
    /// Query norm (`f64`).
    pub q_norm: f64,
    /// `include` has `"metadata"`.
    pub want_meta: bool,
    /// `include` has `"values"`.
    pub want_values: bool,
    /// Resolved beam width (HNSW).
    pub ef: usize,
    /// Resolved rerank candidate count.
    pub rerank: usize,
    /// Index kind the query runs against.
    pub index: IndexConfig,
}

/// Validate a query against a config before any shard is touched.
pub fn validate_query(req: &QueryRequest, cfg: &ShardConfig) -> Result<ValidQuery, OpError> {
    let (mut want_meta, mut want_values) = (false, false);
    for i in &req.include {
        match i.as_str() {
            "metadata" => want_meta = true,
            "values" => want_values = true,
            _ => return Err(OpError::invalid("unknown include")),
        }
    }
    if req.vector.len() != cfg.dim as usize {
        return Err(OpError::new(
            ErrorCode::DimensionMismatch,
            "dimension mismatch",
        ));
    }
    if !super::write::f32_norm_ok(&req.vector) {
        return Err(OpError::new(ErrorCode::NonFiniteValue, "non-finite value"));
    }
    if req.top_k == 0 || req.top_k > MAX_TOP_K {
        return Err(OpError::invalid("top_k out of range"));
    }
    if req.ef.is_some_and(|e| e == 0 || e > MAX_EF) {
        return Err(OpError::invalid("ef out of range"));
    }
    if req.rerank.is_some_and(|r| r < req.top_k || r > MAX_RERANK) {
        return Err(OpError::invalid("rerank out of range"));
    }
    let q_norm = norm(&req.vector);
    if cfg.metric == Metric::Cosine && ruvector_edge_index::norm(&req.vector) <= 0.0 {
        return Err(OpError::invalid("zero query norm"));
    }
    let filter = match &req.filter {
        Some(f) => Filter::parse(f, &cfg.filterable_keys)?,
        None => Filter::default(),
    };
    let default_ef = match cfg.metric {
        Metric::Cosine => EF_DEFAULT_COSINE,
        Metric::L2 | Metric::Dot => EF_DEFAULT_L2_DOT,
    };
    let rerank = req
        .rerank
        .unwrap_or_else(|| DEFAULT_RERANK_MIN.max(4 * req.top_k).min(MAX_RERANK));
    Ok(ValidQuery {
        filter,
        q_norm,
        want_meta,
        want_values,
        ef: req.ef.unwrap_or(default_ef).max(rerank) as usize,
        rerank: rerank as usize,
        index: cfg.index,
    })
}

/// Code-space candidate: max-heap on `(distance, iid)`.
#[derive(Clone, Copy)]
struct Cand(f32, u32);
impl PartialEq for Cand {
    fn eq(&self, o: &Self) -> bool {
        self.cmp(o) == Ordering::Equal
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
        self.0.total_cmp(&o.0).then_with(|| self.1.cmp(&o.1))
    }
}

impl VectorShard {
    fn ensure_readable(&self) -> Result<(), OpError> {
        if self.poisoned {
            return Err(OpError::new(
                ErrorCode::ShardUnavailable,
                "shard must be reopened",
            ));
        }
        Ok(())
    }

    /// Steps a query costs on this shard (checked before it runs).
    pub fn query_steps(&self, v: &ValidQuery) -> u64 {
        // Flat: one step per row per filter comparison (as at M1). HNSW:
        // the beam's code distances, plus a filter pass when filtered.
        // The rerank reads f32 rows from SQLite: `rerank` of them, or for a
        // filtered HNSW query up to `EXACT_FILTER_MAX` (the exact fallback
        // when the post-filtered beam is short).
        let len = self.slab.len() as u64;
        let rows = len.saturating_mul(v.filter.cost());
        let (scan, rerank) = match v.index {
            IndexConfig::Flat => (rows, v.rerank as u64),
            IndexConfig::Hnsw { m, .. } => {
                let beam = (v.ef as u64).saturating_mul(2 * u64::from(m));
                if v.filter.is_empty() {
                    (beam, v.rerank as u64)
                } else {
                    let exact = len.min(EXACT_FILTER_MAX as u64);
                    (beam.saturating_add(rows), (v.rerank as u64).max(exact))
                }
            }
        };
        scan.saturating_add(rerank)
    }

    /// Top-k. `catalog` is the collection config from the ledger; it
    /// validates requests against an uninitialised shard, which answers
    /// empty (§4.3 `EmptyRead`). Loads the index on first use.
    pub fn query(
        &mut self,
        store: &dyn SqlStore,
        expected: &DoMeta,
        catalog: &ShardConfig,
        req: &QueryRequest,
    ) -> Result<QueryOutcome, OpError> {
        self.ensure_readable()?;
        let state = self.check(expected, false)?;
        let cfg = self.config.clone().unwrap_or_else(|| catalog.clone());
        let v = validate_query(req, &cfg)?;
        let empty = QueryOutcome {
            matches: Vec::new(),
            scanned: 0,
        };
        if state == IdentityCheck::EmptyRead || self.slab.len() == 0 {
            return Ok(empty);
        }
        self.index_for_request(store)?;
        let Some(ann) = self.ann.as_ref() else {
            return Ok(empty);
        };
        let k = req.top_k as usize;
        let (cands, scanned) = match ann {
            Ann::Flat(_) => self.flat_candidates(ann, &req.vector, &v),
            Ann::Hnsw(h) => {
                let pool = if v.filter.is_empty() { v.rerank } else { v.ef };
                let hits = h
                    .search(&req.vector, pool, v.ef)
                    .map_err(|_| OpError::invalid("query rejected by index"))?;
                let mut ids: Vec<i64> = hits
                    .iter()
                    .map(|h| i64::from(h.iid))
                    .filter(|&iid| self.passes(&v.filter, iid))
                    .take(v.rerank)
                    .collect();
                let mut scanned = hits.len() as u64;
                if ids.len() < k && !v.filter.is_empty() {
                    let all: Vec<i64> = (0..self.slab.len())
                        .filter(|&s| v.filter.matches(&self.slab.filt[s]))
                        .map(|s| self.slab.iids[s])
                        .collect();
                    if all.len() <= EXACT_FILTER_MAX {
                        scanned += all.len() as u64;
                        ids = all;
                    }
                }
                (ids, scanned)
            }
        };
        let matches = exact_top_k(store, &self.slab, &cfg, &req.vector, &v, &cands, k)?;
        Ok(QueryOutcome { matches, scanned })
    }

    fn passes(&self, filter: &Filter, iid: i64) -> bool {
        self.slab
            .slot_of(iid)
            .is_some_and(|s| filter.is_empty() || filter.matches(&self.slab.filt[s]))
    }

    /// Flat: score every live row passing the filter against its code;
    /// keep the best `rerank`.
    fn flat_candidates(&self, ann: &Ann, q: &[f32], v: &ValidQuery) -> (Vec<i64>, u64) {
        let Ann::Flat(f) = ann else {
            return (Vec::new(), 0);
        };
        let pq = f.quant().prepare(q);
        let mut heap: BinaryHeap<Cand> = BinaryHeap::with_capacity(v.rerank + 1);
        for s in 0..self.slab.len() {
            if !v.filter.is_empty() && !v.filter.matches(&self.slab.filt[s]) {
                continue;
            }
            let Ok(iid) = iid32(self.slab.iids[s]) else {
                continue;
            };
            let Some(code) = f.code(iid) else { continue };
            let c = Cand(pq.distance(code), iid);
            if heap.len() < v.rerank {
                heap.push(c);
            } else if heap.peek().is_some_and(|top| c < *top) {
                heap.pop();
                heap.push(c);
            }
        }
        let ids = heap
            .into_sorted_vec()
            .into_iter()
            .map(|c| i64::from(c.1))
            .collect();
        (ids, self.slab.len() as u64)
    }

    /// Fetch up to 100 distinct ids from SQLite; absent ids are omitted,
    /// duplicates are answered once. Results are in first-request order,
    /// metadata always included, values when `include_values`.
    pub fn fetch(
        &self,
        store: &dyn SqlStore,
        expected: &DoMeta,
        ids: &[String],
        include_values: bool,
    ) -> Result<Vec<Match>, OpError> {
        self.ensure_readable()?;
        if ids.is_empty() {
            return Err(OpError::invalid("no ids"));
        }
        if ids.len() > MAX_FETCH_IDS {
            return Err(OpError::new(ErrorCode::PayloadTooLarge, "too many ids"));
        }
        for id in ids {
            VectorId::parse(id)?;
        }
        if self.check(expected, false)? == IdentityCheck::EmptyRead {
            return Ok(Vec::new());
        }
        let dim = self.config.as_ref().map_or(0, |c| c.dim as usize);
        let mut seen = std::collections::BTreeSet::new();
        let wanted: Vec<&String> = ids
            .iter()
            .filter(|id| seen.insert(id.as_str()) && self.slab.index.contains_key(*id))
            .collect();
        Ok(fetch_ids(store, &self.slab, dim, &wanted, include_values)?)
    }
}
