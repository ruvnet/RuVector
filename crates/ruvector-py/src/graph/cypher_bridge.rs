//! Glue between `GraphDB.query_cypher` (in the parent `graph` module) and
//! the ported `cypher_eval`/`cypher_exec` executor: parse, apply the
//! statement policy, run `MATCH` execution, and pack the result into a
//! Python dict. Split out of `graph.rs` purely to keep that file under
//! this codebase's 500-line-per-file convention — there is no other
//! reason for the split, so this is the one child module under
//! `src/graph/` that *is* pyo3-aware (`cypher_eval.rs`/`cypher_exec.rs`
//! stay pyo3-free on purpose; see their own doc comments).

use pyo3::prelude::*;
use pyo3::types::PyDict;

use ruvector_graph::cypher::{parse_cypher, Statement};
use ruvector_graph::{Edge, GraphDB as RGraphDB, Node};

use super::cypher_exec;
use crate::error::RuVectorError;

/// Execute `cypher` against `gdb` and pack the result as
/// `{"nodes": [...], "edges": [...]}`, using the same per-row dict shape
/// `graph.rs`'s `node_to_py`/`edge_to_py` produce for `get_node`/
/// `get_edge`. See `GraphDB.query_cypher`'s doc comment (in `graph.rs`)
/// for the exact contract this implements — statement policy, what
/// `RETURN` does and does not do, and why `CREATE` raises.
pub(super) fn run_query<'py>(
    py: Python<'py>,
    gdb: &RGraphDB,
    cypher: &str,
) -> PyResult<Bound<'py, PyDict>> {
    let parsed = parse_cypher(cypher)
        .map_err(|e| RuVectorError::new_err(format!("Cypher parse error: {e}")))?;

    let mut result_nodes: Vec<Node> = Vec::new();
    let mut result_edges: Vec<Edge> = Vec::new();
    let mut unsupported: Vec<String> = Vec::new();

    // Release the GIL for the actual graph walk — `execute_match` touches
    // no Python objects, only `DashMap` lookups, so this is the same
    // calculus `hnsw.rs` and `rabitq.rs` already apply around their own
    // pure-Rust work: a threaded host (e.g. an MCP server dispatching
    // tool calls across worker threads) can overlap two `query_cypher`
    // calls, or overlap this with other Python work, instead of one
    // large `MATCH (n)` full scan serializing everything behind the GIL.
    py.detach(|| {
        for statement in &parsed.statements {
            match statement {
                Statement::Match(match_clause) => {
                    let outcome = cypher_exec::execute_match(gdb, match_clause);
                    unsupported.extend(outcome.unsupported);
                    result_nodes.extend(outcome.nodes);
                    result_edges.extend(outcome.edges);
                }
                // Writes through a query string were previously accepted and
                // silently discarded by the upstream NAPI binding this executor
                // was ported from. Refuse them instead — the typed
                // `create_node`/`create_edge` methods are the supported write
                // path and a dropped write is worse than a loud error.
                Statement::Create(_) => unsupported.push(
                    "CREATE via query_cypher() is not supported; use create_node()/create_edge()"
                        .to_string(),
                ),
                Statement::Return(_) => {}
                _ => {}
            }
        }
    });

    if !unsupported.is_empty() {
        unsupported.sort();
        unsupported.dedup();
        return Err(RuVectorError::new_err(format!(
            "Unsupported Cypher: {}",
            unsupported.join("; ")
        )));
    }

    let nodes: PyResult<Vec<_>> = result_nodes
        .iter()
        .map(|n| super::node_to_py(py, n))
        .collect();
    let edges: PyResult<Vec<_>> = result_edges
        .iter()
        .map(|e| super::edge_to_py(py, e))
        .collect();
    let dict = PyDict::new(py);
    dict.set_item("nodes", nodes?)?;
    dict.set_item("edges", edges?)?;
    Ok(dict)
}
