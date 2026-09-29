//! Shard reads: exact flat-scan top-k with metadata post-filter, and fetch.
//!
//! A query keeps a bounded max-heap of `top_k` candidates (O(k) memory,
//! never one entry per row), ranks by `f64` distance then id, and reports
//! its step cost (`rows × filter cost`) so the executor can enforce the
//! per-request step budget ([`MAX_QUERY_STEPS`], §10 layer 3).

use super::codec::ShardConfig;
use super::VectorShard;
use crate::distance::{distance, norm, to_wire, Metric};
use crate::error::{ErrorCode, OpError};
use crate::filter::Filter;
use ruvector_edge_tenancy::quota::limits::MAX_TOP_K;
use ruvector_edge_tenancy::{DoMeta, IdentityCheck, VectorId};
use serde::{Deserialize, Serialize};
use serde_json::Value as Json;
use std::cmp::Ordering;
use std::collections::BinaryHeap;

/// Maximum ids per fetch (§7).
pub const MAX_FETCH_IDS: usize = 100;

/// Per-request scan budget in steps (one step = one row distance or one
/// filter value comparison), `413 budget_exceeded` beyond it.
pub const MAX_QUERY_STEPS: u64 = 8_000_000;

/// A query (the §7 body minus `text`/`ef`, which are M2b/M3).
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
    /// Rows scanned (work-unit input).
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
    if req.vector.iter().any(|v| !v.is_finite()) {
        return Err(OpError::new(ErrorCode::NonFiniteValue, "non-finite value"));
    }
    if req.top_k == 0 || req.top_k > MAX_TOP_K {
        return Err(OpError::invalid("top_k out of range"));
    }
    let q_norm = norm(&req.vector);
    if cfg.metric == Metric::Cosine && q_norm == 0.0 {
        return Err(OpError::invalid("zero query norm"));
    }
    let filter = match &req.filter {
        Some(f) => Filter::parse(f, &cfg.filterable_keys)?,
        None => Filter::default(),
    };
    Ok(ValidQuery {
        filter,
        q_norm,
        want_meta,
        want_values,
    })
}

/// Heap entry: max-heap on `(distance, id)`, so the worst kept candidate
/// is on top.
struct Cand<'a> {
    d: f64,
    id: &'a str,
    slot: usize,
}
impl PartialEq for Cand<'_> {
    fn eq(&self, o: &Self) -> bool {
        self.cmp(o) == Ordering::Equal
    }
}
impl Eq for Cand<'_> {}
impl PartialOrd for Cand<'_> {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
        Some(self.cmp(o))
    }
}
impl Ord for Cand<'_> {
    fn cmp(&self, o: &Self) -> Ordering {
        self.d.total_cmp(&o.d).then_with(|| self.id.cmp(o.id))
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

    /// Scan steps a query with `filter` costs on this shard.
    pub fn query_steps(&self, filter: &Filter) -> u64 {
        (self.slab.len() as u64).saturating_mul(filter.cost())
    }

    /// Exact top-k. `catalog` is the collection config from the ledger; it
    /// validates requests against an uninitialised shard, which answers
    /// empty (§4.3 `EmptyRead`).
    pub fn query(
        &self,
        expected: &DoMeta,
        catalog: &ShardConfig,
        req: &QueryRequest,
    ) -> Result<QueryOutcome, OpError> {
        self.ensure_readable()?;
        let state = self.check(expected, false)?;
        let cfg = self.config.as_ref().unwrap_or(catalog);
        let v = validate_query(req, cfg)?;
        if state == IdentityCheck::EmptyRead {
            return Ok(QueryOutcome {
                matches: Vec::new(),
                scanned: 0,
            });
        }
        let dim = cfg.dim as usize;
        let slab = &self.slab;
        let k = req.top_k as usize;
        let mut heap: BinaryHeap<Cand<'_>> = BinaryHeap::with_capacity(k + 1);
        for s in 0..slab.len() {
            if !v.filter.is_empty() && !v.filter.matches(&slab.filt[s]) {
                continue;
            }
            let d = distance(
                cfg.metric,
                &req.vector,
                v.q_norm,
                slab.row(s, dim),
                slab.norms[s],
            );
            let c = Cand {
                d,
                id: &slab.ids[s],
                slot: s,
            };
            if heap.len() < k {
                heap.push(c);
            } else if heap.peek().is_some_and(|top| c < *top) {
                heap.pop();
                heap.push(c);
            }
        }
        let matches = heap
            .into_sorted_vec()
            .into_iter()
            .map(|c| Match {
                id: c.id.to_string(),
                distance: to_wire(c.d),
                metadata: if v.want_meta {
                    slab.metadata(c.slot)
                } else {
                    None
                },
                values: v.want_values.then(|| slab.row(c.slot, dim).to_vec()),
                score: c.d,
            })
            .collect();
        Ok(QueryOutcome {
            matches,
            scanned: slab.len() as u64,
        })
    }

    /// Fetch up to 100 distinct ids; absent ids are omitted, duplicates are
    /// answered once. Results are in first-request order, metadata always
    /// included, values when `include_values`.
    pub fn fetch(
        &self,
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
        Ok(ids
            .iter()
            .filter(|id| seen.insert(id.as_str()))
            .filter_map(|id| self.slab.index.get(id).map(|&s| (id, s)))
            .map(|(id, s)| Match {
                id: id.clone(),
                distance: 0.0,
                metadata: self.slab.metadata(s),
                values: include_values.then(|| self.slab.row(s, dim).to_vec()),
                score: 0.0,
            })
            .collect())
    }
}
