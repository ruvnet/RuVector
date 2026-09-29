//! Min-cut of a stored graph (ADR-351 §3 rv-mincut on rv-graph): the
//! `GraphStore` converts its property graph into the analytics edge list
//! and answers inline or hands the gateway a job input.
//!
//! Deterministic: edges are merged in edge-id order, so parallel / reverse
//! edge weights are summed in the same order whatever the resident graph's
//! history (cold-loaded or mutated in place) — floating-point addition is
//! not associative, and the report digest hashes `value.to_bits()`.
//!
//! Workers Free: the export is O(edges) on the request path, so a stored
//! graph above [`MAX_MINCUT_EDGES`] is refused (`413`) from its counters,
//! before it is loaded.

use crate::graph_store::{num, resident, GraphHost};
use crate::mincut_core::{self as mc, Plan};
use ruvector_edge_analytics::QueryMode;
use ruvector_edge_store::{ErrorCode, OpError, SqlStore};
use rvlite::cypher::graph_store::Value as CValue;
use rvlite::cypher::PropertyGraph;
use std::collections::BTreeMap;

/// Edges of a stored graph one request may export for a min-cut (the
/// inline routing threshold; the state cap keeps graphs near it anyway).
pub const MAX_MINCUT_EDGES: u64 = mc::INLINE_MAX_EDGES;

/// `(u, v, weight)` over dense vertex indices.
pub type WeightedEdges = Vec<(u64, u64, f64)>;

/// The graph's edges as an undirected weighted list over dense vertex
/// indices (sorted node ids), duplicate / reverse edges summed in edge-id
/// order, self-loops dropped; weight = the `weight` property (default 1).
pub fn edge_list(g: &PropertyGraph) -> Result<(WeightedEdges, Vec<String>), OpError> {
    let mut all = g.all_edges();
    all.sort_unstable_by(|a, b| a.id.cmp(&b.id));
    let mut ids: Vec<&str> = Vec::with_capacity(all.len() * 2);
    for e in &all {
        ids.push(&e.from);
        ids.push(&e.to);
    }
    ids.sort_unstable();
    ids.dedup();
    let idx = |s: &str| ids.binary_search(&s).map(|i| i as u64).unwrap_or(0);
    let mut merged: BTreeMap<(u64, u64), f64> = BTreeMap::new();
    for e in &all {
        let w = match e.get_property("weight") {
            None => 1.0,
            Some(CValue::Float(f)) => *f,
            Some(CValue::Integer(i)) => *i as f64,
            Some(_) => return Err(OpError::invalid("edge weight must be a number")),
        };
        if !w.is_finite() || w < 0.0 {
            return Err(OpError::invalid("edge weight must be finite and >= 0"));
        }
        let (a, b) = (idx(&e.from), idx(&e.to));
        if a != b {
            *merged.entry((a.min(b), a.max(b))).or_insert(0.0) += w;
        }
    }
    let labels = ids.iter().map(|s| s.to_string()).collect();
    Ok((
        merged.into_iter().map(|((u, v), w)| (u, v, w)).collect(),
        labels,
    ))
}

/// `GraphCall::Mincut` on the stored graph.
pub fn mincut(
    host: &mut GraphHost,
    key: &str,
    store: &dyn SqlStore,
    kv: &[(String, String)],
    mode: &QueryMode,
) -> Result<crate::graph_wire::GraphOut, OpError> {
    use crate::graph_wire::GraphOut;
    let edges_n = num(kv, "edges");
    mc::job_admissible(edges_n)?;
    if edges_n > MAX_MINCUT_EDGES {
        return Err(OpError::new(
            ErrorCode::BudgetExceeded,
            "graph too large to export for a min-cut in one request",
        ));
    }
    let revision = num(kv, "revision");
    let r = resident(host, key, store, kv)?;
    let (edges, labels) = edge_list(&r.g)?;
    let uid = crate::graph_wire::job_graph_uid(key);
    match mc::route(uid, revision, &edges, mode)? {
        Plan::Inline(report) => Ok(GraphOut::Cut {
            report: mc::report_json(&report, mode, Some(&labels)),
        }),
        Plan::Job => Ok(GraphOut::JobInput {
            edges: mc::edges_json(&edges),
            labels,
            revision,
        }),
    }
}
