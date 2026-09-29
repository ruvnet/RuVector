//! `GraphStore` Durable Object core (ADR-351 §3 rv-graph, M4): one tenant
//! graph per DO (`idFromName(do_name(tenant, graph, uid(name), 0))`), plus
//! the per-tenant catalog instance ([`CATALOG`]). Pure and synchronous: one
//! call = one coalesced SQLite commit.
//!
//! The resident state is rvlite's `PropertyGraph`, cold-loaded from the
//! persisted `GraphState` (`graph_persist`) and kept in an isolate-wide
//! [`GraphHost`] with LRU eviction. Read-only Cypher runs on it directly
//! and never writes; a mutation (Cypher write, bulk edges) runs on it,
//! then is checked against the size limits and persisted — on any failure
//! the resident graph is dropped, so storage (the last committed revision)
//! stays authoritative.

use crate::graph_cypher as gc;
use crate::graph_persist as gp;
use crate::graph_wire::{
    graph_do_name, name_ok, BulkEdge, GraphCall, GraphOut, GraphRequest, CATALOG,
};
use ruvector_edge_store::{ErrorCode, OpError, ResidentRegistry, SqlStore, Value};
use ruvector_edge_tenancy::TenantKey;
use rvlite::cypher::graph_store::{Edge, Node, Value as CValue};
use rvlite::cypher::PropertyGraph;
use serde_json::{json, Value as Json};
use std::collections::BTreeMap;

/// Nodes per graph (§10: ≤ 50k nodes).
pub const MAX_NODES: usize = 50_000;
/// Edges per graph (§10: ≤ 200k edges).
pub const MAX_EDGES: usize = 200_000;
/// Persisted state per graph, checked on every commit and before every
/// cold load (`413`). Sized to Workers Free (10 ms CPU per invocation):
/// the state is one JSON document, so a cold load and a commit are both
/// O(whole graph). Measured (release profile, native; wasm ≈ 1.1–1.5×;
/// `graph_tests::graph_store_costs`): per ~500 KB of state a cold load ≈
/// 5 ms and a 1k-edge bulk insert + persist ≈ 6–7.7 ms (12 ms at ~950 KB),
/// so the cap is 256 KiB: ≤ ~2.6 ms cold load, ≤ ~5 ms bulk insert +
/// persist, ≈ 2 ms stored-graph min-cut. With the ~4x resident estimate a
/// graph stays ≤ 1 MiB resident, well inside
/// [`GRAPH_RESIDENT_CAP_BYTES`]. (On Workers Paid this could be raised
/// towards the §10 ceiling; persisting deltas would lift it on Free.)
pub const MAX_STATE_BYTES: u64 = 256 << 10;
/// A cold load of more persisted state than this is the whole turn: the
/// request answers `503 shard_unavailable` (retry) and the retry is served
/// from the now-resident graph, so no turn pays load + mutation (≈ 7 ms at
/// the cap).
pub const LOAD_TURN_BYTES: u64 = 128 << 10;
/// Edges per bulk request (Free: one request mutates, exports and
/// persists the graph).
pub const MAX_BULK_EDGES: usize = 1_000;
/// This isolate's resident graph budget: the graph share of the 56 MB
/// isolate resident cap (ADR-351 §6.1), split explicitly with `VectorShard`
/// (`shard_core::VECTOR_RESIDENT_CAP_BYTES`) and `QuantShard`
/// (`quant_shard::QUANT_RESIDENT_CAP_BYTES`) because DOs of one script can
/// share an isolate.
pub const GRAPH_RESIDENT_CAP_BYTES: u64 = 12_000_000;
/// Graphs per tenant.
pub const MAX_GRAPHS: usize = 20;
/// Longest node id / edge type / label in a bulk request.
pub const MAX_KEY_BYTES: usize = 256;

const CAT_SCHEMA: &str =
    "CREATE TABLE IF NOT EXISTS gcat (name TEXT PRIMARY KEY, created_at INTEGER, sub TEXT)";
const CAT_ALL: &str = "SELECT name, created_at, sub FROM gcat ORDER BY name";
const CAT_PUT: &str = "INSERT INTO gcat (name, created_at, sub) VALUES (?, ?, ?)";

/// A loaded graph.
pub struct Resident {
    /// The property graph.
    pub g: PropertyGraph,
    next: (usize, usize),
}

/// Resident graphs of one isolate.
pub struct GraphHost {
    graphs: BTreeMap<String, Resident>,
    registry: ResidentRegistry,
}

impl Default for GraphHost {
    fn default() -> Self {
        GraphHost {
            graphs: BTreeMap::new(),
            registry: ResidentRegistry::new(GRAPH_RESIDENT_CAP_BYTES),
        }
    }
}

impl GraphHost {
    /// Forget `key`'s resident graph.
    pub fn evict(&mut self, key: &str) {
        self.graphs.remove(key);
        self.registry.remove(key);
    }
}

fn not_found() -> OpError {
    OpError::not_found()
}

/// A numeric `gmeta` value (0 when absent).
pub fn num(kv: &[(String, String)], k: &str) -> u64 {
    gp::get(kv, k).and_then(|v| v.parse().ok()).unwrap_or(0)
}

/// Serve one encoded [`GraphRequest`] for the DO whose id is `key`.
pub fn serve(
    host: &mut GraphHost,
    key: &str,
    own_name: Option<&str>,
    store: &dyn SqlStore,
    body: &[u8],
) -> String {
    let reply: Result<GraphOut, crate::wire::WireErr> =
        match serde_json::from_slice::<GraphRequest>(body) {
            Ok(req) => handle(host, key, own_name, store, req)
                .map_err(|e| crate::wire::WireErr::from_op(&e)),
            Err(_) => Err(crate::wire::WireErr::from_op(&OpError::invalid(
                "malformed graph call",
            ))),
        };
    serde_json::to_string(&reply)
        .unwrap_or_else(|_| String::from(r#"{"Err":{"code":"server_error"}}"#))
}

/// Run one call.
pub fn handle(
    host: &mut GraphHost,
    key: &str,
    own_name: Option<&str>,
    store: &dyn SqlStore,
    req: GraphRequest,
) -> Result<GraphOut, OpError> {
    let tenant = TenantKey::parse(&req.tenant_key).map_err(|_| OpError::invalid("tenant"))?;
    let catalog = req.graph == CATALOG;
    if !catalog && !name_ok(&req.graph) {
        return Err(OpError::invalid("graph name"));
    }
    let expected = graph_do_name(&tenant, &req.graph);
    if own_name.is_some_and(|n| n != expected.as_str()) {
        return Err(not_found());
    }
    // Only a create initialises storage: a probe of an unknown graph name
    // (or a tenant's first catalog read) must not persist an empty DO.
    let create = matches!(
        req.call,
        GraphCall::Init { .. } | GraphCall::CatalogAdd { .. }
    );
    if create {
        gp::ensure_schema(store)?;
    }
    let kv = match gp::meta(store) {
        Ok(kv) => kv,
        Err(_) if matches!(req.call, GraphCall::CatalogList) && catalog => {
            return Ok(GraphOut::Graphs { graphs: Vec::new() })
        }
        Err(_) if !create => return Err(not_found()),
        Err(e) => return Err(e),
    };
    let ident = gp::get(&kv, "ident");
    if ident.is_some_and(|i| i != expected.as_str()) {
        return Err(not_found());
    }
    let claim = |store: &dyn SqlStore| gp::put(store, "ident", expected.as_str());
    match (req.call, catalog) {
        (GraphCall::CatalogList, true) => catalog_list(store),
        (GraphCall::CatalogAdd { name, sub, now }, true) => {
            if ident.is_none() {
                claim(store)?;
            }
            catalog_add(store, &name, &sub, now)
        }
        (GraphCall::Init { now }, false) => {
            if ident.is_none() {
                claim(store)?;
                gp::put(store, "name", &req.graph)?;
                gp::put(store, "created_at", &now.to_string())?;
            }
            Ok(GraphOut::Info {
                view: view(&req.graph, &gp::meta(store)?),
            })
        }
        (_, true) | (GraphCall::CatalogList | GraphCall::CatalogAdd { .. }, false) => {
            Err(OpError::invalid("call not valid for this instance"))
        }
        _ if ident.is_none() => Err(not_found()),
        (GraphCall::Stats, _) => Ok(GraphOut::Info {
            view: view(&req.graph, &kv),
        }),
        (GraphCall::Cypher { query, write }, _) => cypher(host, key, store, &kv, &query, write),
        (
            GraphCall::AddEdges {
                edges,
                edge_type,
                label,
            },
            _,
        ) => add_edges(host, key, store, &kv, &edges, &edge_type, &label).map(|_| GraphOut::Info {
            view: view(&req.graph, &gp::meta(store).unwrap_or_default()),
        }),
        (GraphCall::Mincut { mode }, _) => {
            crate::graph_mincut::mincut(host, key, store, &kv, &mode)
        }
    }
}

fn view(name: &str, kv: &[(String, String)]) -> Json {
    json!({
        "name": name,
        "created_at": num(kv, "created_at"),
        "revision": num(kv, "revision"),
        "nodes": num(kv, "nodes"),
        "edges": num(kv, "edges"),
        "state_bytes": num(kv, "bytes"),
    })
}

fn catalog_list(store: &dyn SqlStore) -> Result<GraphOut, OpError> {
    store.exec(CAT_SCHEMA, &[]).map_err(gp::io)?;
    let rows = store.query(CAT_ALL, &[]).map_err(gp::io)?;
    let graphs = rows
        .into_iter()
        .filter_map(|r| match (r.first(), r.get(1)) {
            (Some(Value::Text(n)), Some(Value::Int(t))) => {
                Some(json!({ "name": n, "created_at": t }))
            }
            _ => None,
        })
        .collect();
    Ok(GraphOut::Graphs { graphs })
}

fn catalog_add(store: &dyn SqlStore, name: &str, sub: &str, now: u64) -> Result<GraphOut, OpError> {
    if !name_ok(name) {
        return Err(OpError::invalid("graph name"));
    }
    store.exec(CAT_SCHEMA, &[]).map_err(gp::io)?;
    let rows = store.query(CAT_ALL, &[]).map_err(gp::io)?;
    if rows
        .iter()
        .any(|r| matches!(r.first(), Some(Value::Text(n)) if n == name))
    {
        return Ok(GraphOut::Added { created: false });
    }
    if rows.len() >= MAX_GRAPHS {
        return Err(OpError::new(ErrorCode::QuotaExceeded, "graph limit"));
    }
    let params = [name.into(), Value::Int(now as i64), sub.into()];
    store.exec(CAT_PUT, &params).map_err(gp::io)?;
    Ok(GraphOut::Added { created: true })
}

fn state_limit() -> OpError {
    OpError::new(ErrorCode::BudgetExceeded, "graph state limit")
}

/// The resident graph, cold-loaded from storage on first use; a stored
/// state over [`MAX_STATE_BYTES`] is `413` before anything is read.
pub fn resident<'a>(
    host: &'a mut GraphHost,
    key: &str,
    store: &dyn SqlStore,
    kv: &[(String, String)],
) -> Result<&'a mut Resident, OpError> {
    if !host.graphs.contains_key(key) {
        if num(kv, "bytes") > MAX_STATE_BYTES {
            return Err(state_limit());
        }
        let chunks = num(kv, "chunks");
        let (g, next) = if chunks == 0 {
            (gp::empty()?, (0, 0))
        } else {
            let s = gp::load(store, chunks)?;
            let next = (s.next_node_id, s.next_edge_id);
            (gp::import(&s)?, next)
        };
        host.graphs.insert(key.to_string(), Resident { g, next });
        // Resident estimate: ~4x the persisted JSON (maps + indexes).
        for v in host.registry.touch(key, num(kv, "bytes").saturating_mul(4)) {
            host.graphs.remove(&v);
        }
        if num(kv, "bytes") > LOAD_TURN_BYTES {
            return Err(OpError::new(
                ErrorCode::ShardUnavailable,
                "graph loading, retry",
            ));
        }
    }
    host.graphs
        .get_mut(key)
        .ok_or(OpError::new(ErrorCode::ServerError, "graph vanished"))
}

fn cypher(
    host: &mut GraphHost,
    key: &str,
    store: &dyn SqlStore,
    kv: &[(String, String)],
    query: &str,
    write: bool,
) -> Result<GraphOut, OpError> {
    let parsed = gc::parse(query)?;
    if !parsed.read_only && !write {
        return Err(OpError::new(
            ErrorCode::RoleRequired,
            "mutating cypher needs write",
        ));
    }
    let r = resident(host, key, store, kv)?;
    if parsed.read_only {
        let (result, _) = gc::run(&mut r.g, &parsed)?;
        return Ok(GraphOut::Rows { result });
    }
    let out = gc::run(&mut r.g, &parsed).and_then(|(result, _)| {
        commit(r, store, kv)?;
        Ok(result)
    });
    out.map(|result| GraphOut::Rows { result })
        .inspect_err(|_| host.evict(key))
}

/// Check the limits and persist the mutated resident graph.
fn commit(r: &mut Resident, store: &dyn SqlStore, kv: &[(String, String)]) -> Result<(), OpError> {
    let st = r.g.stats();
    if st.node_count > MAX_NODES || st.edge_count > MAX_EDGES {
        return Err(OpError::new(ErrorCode::BudgetExceeded, "graph size limit"));
    }
    r.next = gp::counters(&r.g, r.next);
    let state = gp::export(&r.g, r.next);
    let (bytes, chunks) = gp::save(store, &state, MAX_STATE_BYTES)?;
    for (k, v) in [
        ("revision", num(kv, "revision") + 1),
        ("nodes", st.node_count as u64),
        ("edges", st.edge_count as u64),
        ("bytes", bytes),
        ("chunks", chunks),
    ] {
        gp::put(store, k, &v.to_string())?;
    }
    Ok(())
}

fn key_ok(s: &str) -> bool {
    !s.is_empty() && s.len() <= MAX_KEY_BYTES && !s.chars().any(char::is_control)
}

fn add_edges(
    host: &mut GraphHost,
    key: &str,
    store: &dyn SqlStore,
    kv: &[(String, String)],
    edges: &[BulkEdge],
    edge_type: &str,
    label: &str,
) -> Result<(), OpError> {
    if edges.is_empty() {
        return Err(OpError::invalid("no edges"));
    }
    if edges.len() > MAX_BULK_EDGES {
        return Err(OpError::new(ErrorCode::PayloadTooLarge, "too many edges"));
    }
    if !key_ok(edge_type) || !key_ok(label) {
        return Err(OpError::invalid("edge type / label"));
    }
    let parts: Vec<(String, String, f64)> = edges.iter().map(BulkEdge::parts).collect();
    for (a, b, w) in &parts {
        if !key_ok(a) || !key_ok(b) || !w.is_finite() || *w < 0.0 {
            return Err(OpError::invalid("edge endpoints / weight"));
        }
    }
    // Checked again (with nodes) on commit; this refuses before loading.
    if num(kv, "edges") as usize + parts.len() > MAX_EDGES {
        return Err(OpError::new(ErrorCode::BudgetExceeded, "graph size limit"));
    }
    let r = resident(host, key, store, kv)?;
    let res = (|| {
        for (a, b, w) in &parts {
            for id in [a, b] {
                if r.g.get_node(id).is_none() {
                    r.g.add_node(Node::new(id.clone()).with_label(label.to_string()));
                }
            }
            let id = r.g.generate_edge_id();
            let e = Edge::new(id, a.clone(), b.clone(), edge_type.to_string())
                .with_property("weight".into(), CValue::Float(*w));
            r.g.add_edge(e).map_err(|_| OpError::invalid("edge"))?;
        }
        commit(r, store, kv)
    })();
    res.inspect_err(|_| host.evict(key))
}
