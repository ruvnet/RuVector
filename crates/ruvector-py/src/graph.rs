//! `GraphDB` Python class — wraps `ruvector_graph::GraphDB`.
//!
//! M1 of the ADR-352 graph slice: raw CRUD (create/get node, create/get
//! edge, outgoing-edge traversal) over the in-memory graph. M2 adds
//! `query_cypher`, a thin pyo3 wrapper around the `MATCH` executor in the
//! `cypher_eval`/`cypher_exec` child modules (ported from
//! `ruvector-graph-node`; see that module's doc comment) — this file
//! stays the only pyo3-aware one in the slice.
//!
//! Not `unsendable`: `ruvector_graph::GraphDB`'s own test suite
//! (`test_concurrent_updates_do_not_lose_writes` in
//! `crates/ruvector-graph/src/graph.rs`) moves an `Arc<GraphDB>` into
//! `std::thread::spawn` from multiple threads — that only compiles if
//! `GraphDB: Send + Sync`. Every field is a `DashMap`/`Arc` over
//! `Send + Sync` value types (`String`, `Node`, `Edge`, `HashSet<String>`),
//! and the one field that would NOT be `Send + Sync` unconditionally
//! (`storage: Option<GraphStorage>`) is behind `#[cfg(feature =
//! "storage")]`, which this crate's `ruvector-graph` dependency disables
//! (`default-features = false, features = ["simd"]` in `Cargo.toml`) — so
//! it doesn't even exist in this build. pyo3 0.29 enforces `Send + Sync`
//! for a non-`unsendable` `#[pyclass]` at compile time, so getting this
//! wrong would be a build failure here, not a runtime surprise later (the
//! same reasoning `rabitq.rs` and `hnsw.rs` already document for their own
//! wrapped types).

mod convert;
mod cypher_bridge;
mod cypher_eval;
mod cypher_exec;

use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict};

use ruvector_graph::{EdgeBuilder, GraphDB as RGraphDB, NodeBuilder};

use crate::error::RuVectorError;
use convert::{edge_to_py, node_to_py, py_dict_to_properties, py_to_labels, to_pyerr_graph};

/// In-memory property graph — `ruvector_graph::GraphDB`. Raw CRUD surface:
/// create/get nodes, create/get edges, outgoing-edge traversal. Cypher
/// `MATCH` execution via `query_cypher` (see that method's doc comment
/// for the exact scope and the module docstring for how it's wired up).
#[pyclass(name = "GraphDB", module = "ruvector._native")]
pub struct GraphDB {
    inner: RGraphDB,
}

#[pymethods]
impl GraphDB {
    /// Create a new, empty in-memory graph. Takes no arguments: the
    /// underlying `GraphDB::new()` is a bare in-memory constructor (the
    /// persistent-storage constructor, `GraphDB::with_storage`, is gated
    /// behind the `storage` feature, which this crate deliberately does
    /// not enable — see the module docstring).
    #[new]
    fn new() -> Self {
        Self {
            inner: RGraphDB::new(),
        }
    }

    /// Create a node with `labels` (a list of strings, default none) and
    /// `properties` (an arbitrary JSON-compatible dict, default none).
    ///
    /// `id`, if given, must not already exist. The real
    /// `GraphDB::create_node` has no such guard: it silently `insert`s
    /// over an existing id while adding (not replacing) the new node's
    /// labels/properties into the label/property indexes, leaving the old
    /// node's index entries stale. Pre-checking here and raising instead
    /// turns that footgun into a clear error at the Python boundary. When
    /// `id` is omitted, the real `NodeBuilder` behaviour applies: a fresh
    /// UUID is generated.
    ///
    /// Returns the node's id.
    #[pyo3(signature = (labels = None, properties = None, *, id = None))]
    fn create_node(
        &self,
        labels: Option<&Bound<'_, PyAny>>,
        properties: Option<&Bound<'_, PyDict>>,
        id: Option<&str>,
    ) -> PyResult<String> {
        if let Some(id) = id {
            if self.inner.get_node(id).is_some() {
                return Err(RuVectorError::new_err(format!(
                    "node {id:?} already exists; create_node does not overwrite"
                )));
            }
        }
        let mut builder = NodeBuilder::new();
        if let Some(id) = id {
            builder = builder.id(id);
        }
        if let Some(labels) = labels {
            builder = builder.labels(py_to_labels(labels)?);
        }
        if let Some(properties) = properties {
            builder = builder.properties(py_dict_to_properties(properties)?);
        }
        self.inner
            .create_node(builder.build())
            .map_err(to_pyerr_graph)
    }

    /// Look up a node by id. Returns `None` if it does not exist (the real
    /// `GraphDB::get_node` returns `Option`, never an error, for a missing
    /// id) — otherwise `{"id": ..., "labels": [...], "properties": {...}}`.
    fn get_node<'py>(
        &self,
        py: Python<'py>,
        node_id: &str,
    ) -> PyResult<Option<Bound<'py, PyDict>>> {
        self.inner
            .get_node(node_id)
            .map(|node| node_to_py(py, &node))
            .transpose()
    }

    /// Create an edge `from_id` -> `to_id` of type `relation_type`, with an
    /// optional properties dict. The real `GraphDB::create_edge` already
    /// rejects an edge whose endpoint doesn't exist
    /// (`GraphError::NodeNotFound`, surfaced here as `RuVectorError`) —
    /// that validation needs no extra guard on this side.
    ///
    /// `id`, if given, must not already exist — same silent-overwrite
    /// footgun and same pre-check as `create_node`.
    ///
    /// Returns the edge's id.
    #[pyo3(signature = (from_id, to_id, relation_type, properties = None, *, id = None))]
    fn create_edge(
        &self,
        from_id: &str,
        to_id: &str,
        relation_type: &str,
        properties: Option<&Bound<'_, PyDict>>,
        id: Option<&str>,
    ) -> PyResult<String> {
        if let Some(id) = id {
            if self.inner.get_edge(id).is_some() {
                return Err(RuVectorError::new_err(format!(
                    "edge {id:?} already exists; create_edge does not overwrite"
                )));
            }
        }
        let mut builder = EdgeBuilder::new(from_id.to_string(), to_id.to_string(), relation_type);
        if let Some(id) = id {
            builder = builder.id(id);
        }
        if let Some(properties) = properties {
            builder = builder.properties(py_dict_to_properties(properties)?);
        }
        self.inner
            .create_edge(builder.build())
            .map_err(to_pyerr_graph)
    }

    /// Look up an edge by id. Returns `None` if it does not exist —
    /// otherwise `{"id": ..., "from": ..., "to": ..., "type": ...,
    /// "properties": {...}}`.
    fn get_edge<'py>(
        &self,
        py: Python<'py>,
        edge_id: &str,
    ) -> PyResult<Option<Bound<'py, PyDict>>> {
        self.inner
            .get_edge(edge_id)
            .map(|edge| edge_to_py(py, &edge))
            .transpose()
    }

    /// Edges whose `from` is `node_id`. Empty list (not an error) for an
    /// unknown node id, matching `GraphDB::get_outgoing_edges`'s own
    /// contract (it treats a missing node the same as a node with no
    /// outgoing edges).
    fn get_outgoing_edges<'py>(
        &self,
        py: Python<'py>,
        node_id: &str,
    ) -> PyResult<Vec<Bound<'py, PyDict>>> {
        self.inner
            .get_outgoing_edges(&node_id.to_string())
            .iter()
            .map(|edge| edge_to_py(py, edge))
            .collect()
    }

    /// Execute a Cypher string, limited to `MATCH` execution (see the
    /// `cypher_exec` child module's doc comment for the exact scope: no
    /// cross-pattern joins, variable-length paths, or aggregations).
    ///
    /// Returns `{"nodes": [...], "edges": [...]}` using the same per-row
    /// shape as `get_node`/`get_edge` — **not** a `RETURN`-projected row
    /// set. The `RETURN` clause is parsed (so `MATCH (n) RETURN n` is
    /// valid syntax) but never applied as a projection: the result is
    /// always every node/edge the `MATCH` touched, regardless of what
    /// `RETURN` names. `CREATE` raises explicitly — mirrors
    /// `ruvector-graph-node`'s own `query()` contract, which refuses
    /// writes through a query string rather than silently discarding
    /// them. Other write statement types (`MERGE`/`SET`/`DELETE`/
    /// `REMOVE`) are currently accepted but not executed, matching that
    /// same upstream contract (this port did not newly invent that
    /// choice; it carries it over as-is).
    ///
    /// Raises `RuVectorError` on a parse error or on anything the
    /// executor could not honour (e.g. a variable-length relationship).
    fn query_cypher<'py>(&self, py: Python<'py>, cypher: &str) -> PyResult<Bound<'py, PyDict>> {
        cypher_bridge::run_query(py, &self.inner, cypher)
    }

    /// Node count. `len(graph)` follows the networkx convention of
    /// counting nodes, not nodes + edges.
    fn __len__(&self) -> usize {
        self.inner.node_count()
    }

    fn __repr__(&self) -> String {
        format!(
            "GraphDB(nodes={}, edges={})",
            self.inner.node_count(),
            self.inner.edge_count()
        )
    }
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<GraphDB>()?;
    Ok(())
}
