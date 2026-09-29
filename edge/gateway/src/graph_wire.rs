//! Gateway ↔ `GraphStore` / `AnalyticsJob` Durable Object wire (ADR-351
//! §3 rv-graph / rv-mincut, M4). As with the vector wire, the tenant comes
//! only from the verified token, DO names are derived from it (never from
//! request input), and each DO asserts the request against its stored
//! identity (`404` on mismatch).

use ruvector_edge_analytics::QueryMode;
use ruvector_edge_tenancy::{do_name, CollectionUid, DoName, Service, ShardIndex, TenantKey};
use serde::{Deserialize, Serialize};
use serde_json::Value as Json;
use sha2::{Digest, Sha256};

/// Reserved graph name of the per-tenant catalog instance (user graph
/// names start with `[a-z0-9]`, so it can never collide).
pub const CATALOG: &str = "$catalog";

/// `^[a-z0-9][a-z0-9_-]{0,62}$` (the collection-name rule).
pub fn name_ok(n: &str) -> bool {
    let b = n.as_bytes();
    (1..=63).contains(&b.len())
        && (b[0].is_ascii_lowercase() || b[0].is_ascii_digit())
        && b.iter()
            .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || *c == b'_' || *c == b'-')
}

fn uid_of(tag: &str, name: &str) -> CollectionUid {
    let d = Sha256::digest(format!("v1|{tag}|{name}").as_bytes());
    let mut b = [0u8; 16];
    b.copy_from_slice(&d[..16]);
    CollectionUid::from_bytes(b)
}

fn shard0() -> ShardIndex {
    ShardIndex::parse("0").expect("shard 0")
}

/// DO name of `tenant`'s graph `name` (or the [`CATALOG`]).
pub fn graph_do_name(tenant: &TenantKey, name: &str) -> DoName {
    do_name(tenant, Service::Graph, &uid_of("graph", name), shard0())
}

/// DO name of `tenant`'s min-cut job `job_id`.
pub fn job_do_name(tenant: &TenantKey, job_id: &str) -> DoName {
    do_name(
        tenant,
        Service::Mincut,
        &uid_of("mincut-job", job_id),
        shard0(),
    )
}

/// Graph uid bound into a job's persisted edge list.
pub fn job_graph_uid(job_id: &str) -> [u8; 16] {
    let d = Sha256::digest(format!("v1|mincut-graph|{job_id}").as_bytes());
    let mut b = [0u8; 16];
    b.copy_from_slice(&d[..16]);
    b
}

/// One bulk edge: `[from, to]` or `[from, to, weight]`; node keys are
/// strings or non-negative integers (used as the node id text).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum BulkEdge {
    /// Unit weight.
    Pair(NodeKey, NodeKey),
    /// Explicit weight.
    Weighted(NodeKey, NodeKey, f64),
}

/// A node key in a bulk edge.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum NodeKey {
    /// Integer key.
    Int(u64),
    /// Text key.
    Text(String),
}

impl NodeKey {
    /// The node id this key names.
    pub fn id(&self) -> String {
        match self {
            NodeKey::Int(i) => i.to_string(),
            NodeKey::Text(s) => s.clone(),
        }
    }
}

impl BulkEdge {
    /// `(from, to, weight)`.
    pub fn parts(&self) -> (String, String, f64) {
        match self {
            BulkEdge::Pair(a, b) => (a.id(), b.id(), 1.0),
            BulkEdge::Weighted(a, b, w) => (a.id(), b.id(), *w),
        }
    }
}

/// A `GraphStore` request.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GraphRequest {
    /// Tenant key from the verified token.
    pub tenant_key: String,
    /// Graph name ([`CATALOG`] for the catalog instance).
    pub graph: String,
    /// The call.
    pub call: GraphCall,
}

/// `GraphStore` calls.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "call", rename_all = "snake_case")]
pub enum GraphCall {
    /// Catalog: every graph.
    CatalogList,
    /// Catalog: record a graph (`created: false` if it exists).
    CatalogAdd { name: String, sub: String, now: u64 },
    /// Graph: initialise (idempotent).
    Init { now: u64 },
    /// Graph: counters.
    Stats,
    /// Graph: one Cypher query; `write` is the gateway's authorization
    /// (a mutating query without it is refused).
    Cypher { query: String, write: bool },
    /// Graph: bulk edge insert.
    AddEdges {
        edges: Vec<BulkEdge>,
        edge_type: String,
        label: String,
    },
    /// Graph: min-cut over its edges (inline, or the job input).
    Mincut { mode: QueryMode },
}

/// `GraphStore` results.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "out", rename_all = "snake_case")]
pub enum GraphOut {
    /// For `CatalogList`.
    Graphs { graphs: Vec<Json> },
    /// For `CatalogAdd`.
    Added { created: bool },
    /// For `Init` / `Stats` / `AddEdges`.
    Info { view: Json },
    /// For `Cypher`.
    Rows { result: Json },
    /// For `Mincut`, answered inline.
    Cut { report: Json },
    /// For `Mincut` over the inline budget: the job input (edge list JSON
    /// `[[u, v, w], …]`, vertex labels by index, graph revision).
    JobInput {
        edges: String,
        labels: Vec<String>,
        revision: u64,
    },
}

/// An `AnalyticsJob` request.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct JobRequest {
    /// Tenant key from the verified token.
    pub tenant_key: String,
    /// Job id.
    pub job_id: String,
    /// The call.
    pub call: JobCall,
}

/// `AnalyticsJob` calls.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "call", rename_all = "snake_case")]
pub enum JobCall {
    /// Queue a job (`409` if the id exists).
    Submit {
        mode: QueryMode,
        /// Edge list JSON `[[u, v, w], …]` (≤ 1 MiB).
        edges: String,
        /// Vertex labels by index (graph-sourced jobs).
        labels: Option<Vec<String>>,
        /// Source graph name, when graph-sourced.
        graph: Option<String>,
        revision: u64,
        now_ms: u64,
    },
    /// Status / result at `now_ms` (a job without progress for
    /// `mincut_job::STALL_MS` is reported, and recorded, as failed).
    Get {
        #[serde(default)]
        now_ms: u64,
    },
}

/// `AnalyticsJob` result: the public job view.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct JobView {
    /// Public JSON.
    pub view: Json,
}
