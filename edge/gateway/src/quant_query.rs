//! `QuantShard` query: 1-bit RaBitQ scan over the resident codes, exact
//! f32 rerank of the best `top_k × factor` candidates (≤ 1,000) read from
//! SQLite, then the result rows' metadata / values.
//!
//! Budgets, all checked before the scan: the fan-out step budget
//! (`MAX_QUERY_STEPS`, rows scanned + candidates reranked, shared across
//! shards like the `VectorShard` scan steps) and the quant crate's own
//! per-query work budget (`max_query_units`); either refusal is
//! `413 budget_exceeded`.

use crate::quant_shard::QuantHost;
use crate::quant_store::{self as qs, FullRow, QMeta};
use crate::wire::{MatchWire, ShardOut};
use ruvector_edge_quant::{QueryOptions, RerankSource};
use ruvector_edge_store::shard::MAX_QUERY_STEPS;
use ruvector_edge_store::{ErrorCode, OpError, QueryRequest, SqlStore};
use std::collections::BTreeMap;

/// Wire match for a stored row at `distance`.
pub fn wire(r: FullRow, distance: f32, metadata: bool, values: bool) -> MatchWire {
    MatchWire {
        id: r.id,
        distance,
        metadata: if metadata {
            r.md.and_then(|m| serde_json::from_str(&m).ok())
        } else {
            None
        },
        values: values.then_some(r.values),
        score_bits: f64::from(distance).to_bits(),
    }
}

/// Top-k of one shard.
pub fn query(
    host: &mut QuantHost,
    key: &str,
    store: &dyn SqlStore,
    meta: &QMeta,
    q: &QueryRequest,
    steps_before: u64,
) -> Result<ShardOut, OpError> {
    if q.filter.is_some() {
        return Err(OpError::invalid("filter is not supported by index rabitq"));
    }
    let (mut want_meta, mut want_values) = (false, false);
    for i in &q.include {
        match i.as_str() {
            "metadata" => want_meta = true,
            "values" => want_values = true,
            _ => return Err(OpError::invalid("unknown include")),
        }
    }
    if meta.ident.is_none() || meta.count == 0 {
        let steps = check_steps(steps_before, 0, 0)?;
        return Ok(ShardOut::Matches {
            matches: Vec::new(),
            scanned: 0,
            steps,
        });
    }
    if q.vector.len() != meta.dim as usize {
        return Err(OpError::new(
            ErrorCode::DimensionMismatch,
            "dimension mismatch",
        ));
    }
    let opts = options(q.top_k, q.rerank);
    let cap = u64::from(host.budget.max_rerank_candidates);
    let wanted = u64::from(opts.top_k) * u64::from(opts.rerank_factor);
    let steps = check_steps(steps_before, meta.count, wanted.min(cap).min(meta.count))?;
    let r = host.ready(key, store, meta)?;
    let out =
        r.q.query(&q.vector, &opts, Some(&mut SqlRerank(store)))
            .map_err(crate::quant_load::qerr)?;
    let keys: Vec<u64> = out.hits.iter().map(|h| h.key).collect();
    let mut rows: BTreeMap<u64, FullRow> = qs::rows_by_rks(store, &keys)?
        .into_iter()
        .map(|r| (r.rk, r))
        .collect();
    let matches = out
        .hits
        .iter()
        .filter_map(|h| {
            rows.remove(&h.key)
                .map(|r| wire(r, h.distance, want_meta, want_values))
        })
        .collect();
    Ok(ShardOut::Matches {
        matches,
        scanned: out.stats.scanned,
        steps,
    })
}

/// Steps a query over `n` rows reranking `candidates` is charged (the
/// `VectorShard` scan-step unit: rows touched).
pub fn query_steps(n: u64, candidates: u64) -> u64 {
    n.saturating_add(candidates)
}

/// Guard against the fan-out step budget before scanning.
pub fn check_steps(steps_before: u64, n: u64, candidates: u64) -> Result<u64, OpError> {
    let steps = steps_before.saturating_add(query_steps(n, candidates));
    if steps > MAX_QUERY_STEPS {
        return Err(OpError::new(ErrorCode::BudgetExceeded, "query step budget"));
    }
    Ok(steps)
}

/// Default rerank factor (recall@10 ≈ 0.999 at 50k × 384, crate docs).
pub fn options(top_k: u32, rerank: Option<u32>) -> QueryOptions {
    let factor = rerank.map_or(QueryOptions::default().rerank_factor, |r| {
        r.div_ceil(top_k.max(1)).max(1)
    });
    QueryOptions {
        top_k,
        rerank_factor: factor,
    }
}

/// f32 originals from SQLite for rerank candidates.
pub struct SqlRerank<'a>(pub &'a dyn SqlStore);

impl RerankSource for SqlRerank<'_> {
    fn fetch(&mut self, keys: &[u64]) -> Result<Vec<Option<Vec<f32>>>, String> {
        let rows = qs::rows_by_rks(self.0, keys).map_err(|e| e.detail.to_string())?;
        let mut by: BTreeMap<u64, Vec<f32>> = rows.into_iter().map(|r| (r.rk, r.values)).collect();
        Ok(keys.iter().map(|k| by.remove(k)).collect())
    }
}
