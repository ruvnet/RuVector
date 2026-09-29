//! `GraphStore` persistence (ADR-351 §3 rv-graph): the property graph is
//! stored as rvlite's serde [`GraphState`], JSON-encoded and split into
//! ≤ 1 MiB `gchunks` rows (DO SQLite rows are capped at 2 MB), plus a
//! `gmeta` key/value table (identity, revision, counters, sizes).
//!
//! Id counters: `CypherEngine::export_state` records the node / edge
//! *counts* as the next ids and `load_state` ignores them, so a `CREATE`
//! after a reload would reuse `n0` and overwrite a live node. The store
//! therefore owns the counters: after every mutation the next ids are
//! `max(saved, 1 + the largest generated id)`, they are written into
//! `GraphState.next_{node,edge}_id`, and a load restores them into the
//! `PropertyGraph` before re-adding nodes and edges.

use ruvector_edge_store::{ErrorCode, OpError, SqlStore, Value};
use rvlite::cypher::graph_store::{Edge, Node, Value as CValue};
use rvlite::cypher::PropertyGraph;
use rvlite::storage::state::{EdgeState, GraphState, NodeState, PropertyValue};

/// Chunk size of the persisted JSON.
pub const CHUNK_BYTES: usize = 1 << 20;

const SCHEMA: [&str; 2] = [
    "CREATE TABLE IF NOT EXISTS gmeta (k TEXT PRIMARY KEY, v TEXT)",
    "CREATE TABLE IF NOT EXISTS gchunks (idx INTEGER PRIMARY KEY, bytes BLOB)",
];
const META_ALL: &str = "SELECT k, v FROM gmeta";
const META_PUT: &str = "INSERT OR REPLACE INTO gmeta (k, v) VALUES (?, ?)";
const CHUNKS_ALL: &str = "SELECT idx, bytes FROM gchunks ORDER BY idx";
const CHUNKS_CLEAR: &str = "DELETE FROM gchunks";
const CHUNK_PUT: &str = "INSERT INTO gchunks (idx, bytes) VALUES (?, ?)";

/// Storage failure → `503`.
pub fn io<E>(_: E) -> OpError {
    OpError::new(ErrorCode::ShardUnavailable, "graph storage")
}

/// Create the tables.
pub fn ensure_schema(store: &dyn SqlStore) -> Result<(), OpError> {
    for sql in SCHEMA {
        store.exec(sql, &[]).map_err(io)?;
    }
    Ok(())
}

/// Erase the graph (rows only: the tables stay, empty, so a same-name
/// re-create claims a fresh identity; every non-create call on a graph
/// without `ident` is `404`).
pub fn wipe(store: &dyn SqlStore) -> Result<(), OpError> {
    for sql in [CHUNKS_CLEAR, "DELETE FROM gmeta"] {
        store.exec(sql, &[]).map_err(io)?;
    }
    Ok(())
}

/// `gmeta` as key/value pairs.
pub fn meta(store: &dyn SqlStore) -> Result<Vec<(String, String)>, OpError> {
    let rows = store.query(META_ALL, &[]).map_err(io)?;
    Ok(rows
        .into_iter()
        .filter_map(|r| match (r.first(), r.get(1)) {
            (Some(Value::Text(k)), Some(Value::Text(v))) => Some((k.clone(), v.clone())),
            _ => None,
        })
        .collect())
}

/// One `gmeta` value.
pub fn get<'a>(kv: &'a [(String, String)], k: &str) -> Option<&'a str> {
    kv.iter().find(|(key, _)| key == k).map(|(_, v)| v.as_str())
}

/// Set one `gmeta` value.
pub fn put(store: &dyn SqlStore, k: &str, v: &str) -> Result<(), OpError> {
    store
        .exec(META_PUT, &[k.into(), v.into()])
        .map(|_| ())
        .map_err(io)
}

fn to_prop(v: &CValue) -> PropertyValue {
    match v {
        CValue::Null => PropertyValue::Null,
        CValue::Boolean(b) => PropertyValue::Boolean(*b),
        CValue::Integer(i) => PropertyValue::Integer(*i),
        CValue::Float(f) => PropertyValue::Float(*f),
        CValue::String(s) => PropertyValue::String(s.clone()),
        CValue::List(l) => PropertyValue::List(l.iter().map(to_prop).collect()),
        CValue::Map(m) => {
            PropertyValue::Map(m.iter().map(|(k, v)| (k.clone(), to_prop(v))).collect())
        }
    }
}

fn from_prop(p: &PropertyValue) -> CValue {
    match p {
        PropertyValue::Null => CValue::Null,
        PropertyValue::Boolean(b) => CValue::Boolean(*b),
        PropertyValue::Integer(i) => CValue::Integer(*i),
        PropertyValue::Float(f) => CValue::Float(*f),
        PropertyValue::String(s) => CValue::String(s.clone()),
        PropertyValue::List(l) => CValue::List(l.iter().map(from_prop).collect()),
        PropertyValue::Map(m) => {
            CValue::Map(m.iter().map(|(k, v)| (k.clone(), from_prop(v))).collect())
        }
    }
}

/// `1 + k` for an id `"{prefix}{k}"`, else 0.
fn next_after(id: &str, prefix: char) -> usize {
    id.strip_prefix(prefix)
        .filter(|d| !d.is_empty() && d.bytes().all(|b| b.is_ascii_digit()))
        .and_then(|d| d.parse::<usize>().ok())
        .map_or(0, |k| k.saturating_add(1))
}

/// Next node / edge ids: never below `saved`, always above every id of
/// the generated form present in `g`.
pub fn counters(g: &PropertyGraph, saved: (usize, usize)) -> (usize, usize) {
    let n = g.all_nodes().iter().map(|n| next_after(&n.id, 'n')).max();
    let e = g.all_edges().iter().map(|e| next_after(&e.id, 'e')).max();
    (saved.0.max(n.unwrap_or(0)), saved.1.max(e.unwrap_or(0)))
}

/// The persisted form of `g` with the store's counters.
pub fn export(g: &PropertyGraph, next: (usize, usize)) -> GraphState {
    let mut nodes: Vec<NodeState> = g
        .all_nodes()
        .into_iter()
        .map(|n| NodeState {
            id: n.id.clone(),
            labels: n.labels.clone(),
            properties: n
                .properties
                .iter()
                .map(|(k, v)| (k.clone(), to_prop(v)))
                .collect(),
        })
        .collect();
    nodes.sort_by(|a, b| a.id.cmp(&b.id));
    let mut edges: Vec<EdgeState> = g
        .all_edges()
        .into_iter()
        .map(|e| EdgeState {
            id: e.id.clone(),
            from: e.from.clone(),
            to: e.to.clone(),
            edge_type: e.edge_type.clone(),
            properties: e
                .properties
                .iter()
                .map(|(k, v)| (k.clone(), to_prop(v)))
                .collect(),
        })
        .collect();
    edges.sort_by(|a, b| a.id.cmp(&b.id));
    GraphState {
        nodes,
        edges,
        next_node_id: next.0,
        next_edge_id: next.1,
    }
}

/// An empty graph whose id generators continue at `next`.
fn graph_at(next: (usize, usize)) -> Result<PropertyGraph, OpError> {
    let empty = serde_json::json!({
        "nodes": {}, "edges": {}, "label_index": {}, "edge_type_index": {},
        "outgoing_edges": {}, "incoming_edges": {},
        "next_node_id": next.0, "next_edge_id": next.1,
    });
    serde_json::from_value(empty).map_err(|_| OpError::new(ErrorCode::ServerError, "graph init"))
}

/// Rebuild a graph from its persisted form (fresh indexes).
pub fn import(s: &GraphState) -> Result<PropertyGraph, OpError> {
    let mut g = graph_at((s.next_node_id, s.next_edge_id))?;
    for n in &s.nodes {
        let mut node = Node::new(n.id.clone()).with_labels(n.labels.clone());
        for (k, v) in &n.properties {
            node.set_property(k.clone(), from_prop(v));
        }
        g.add_node(node);
    }
    for e in &s.edges {
        let mut edge = Edge::new(
            e.id.clone(),
            e.from.clone(),
            e.to.clone(),
            e.edge_type.clone(),
        );
        for (k, v) in &e.properties {
            edge.set_property(k.clone(), from_prop(v));
        }
        g.add_edge(edge)
            .map_err(|_| OpError::new(ErrorCode::ShardUnavailable, "graph edge"))?;
    }
    Ok(g)
}

/// An empty graph (counters at 0).
pub fn empty() -> Result<PropertyGraph, OpError> {
    graph_at((0, 0))
}

/// Serialise `state`, refusing (`413`) above `max_bytes` before anything
/// is written, then replace the stored chunks; returns `(bytes, chunks)`.
pub fn save(
    store: &dyn SqlStore,
    state: &GraphState,
    max_bytes: u64,
) -> Result<(u64, u64), OpError> {
    let bytes = serde_json::to_vec(state).map_err(io)?;
    if bytes.len() as u64 > max_bytes {
        return Err(OpError::new(ErrorCode::BudgetExceeded, "graph state limit"));
    }
    store.exec(CHUNKS_CLEAR, &[]).map_err(io)?;
    let mut n = 0u64;
    for (i, c) in bytes.chunks(CHUNK_BYTES).enumerate() {
        store
            .exec(CHUNK_PUT, &[Value::Int(i as i64), Value::Blob(c.to_vec())])
            .map_err(io)?;
        n += 1;
    }
    Ok((bytes.len() as u64, n))
}

/// Read the stored state (`chunks` from `gmeta`; a mismatch is corrupt).
pub fn load(store: &dyn SqlStore, chunks: u64) -> Result<GraphState, OpError> {
    let rows = store.query(CHUNKS_ALL, &[]).map_err(io)?;
    if rows.len() as u64 != chunks {
        return Err(OpError::new(ErrorCode::ShardUnavailable, "graph chunks"));
    }
    let mut bytes = Vec::new();
    for r in rows {
        match r.get(1) {
            Some(Value::Blob(b)) => bytes.extend_from_slice(b),
            _ => return Err(OpError::new(ErrorCode::ShardUnavailable, "graph chunk")),
        }
    }
    serde_json::from_slice(&bytes)
        .map_err(|_| OpError::new(ErrorCode::ShardUnavailable, "graph state"))
}
