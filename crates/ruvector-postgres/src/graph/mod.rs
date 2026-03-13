// Graph operations module for ruvector-postgres
//
// Provides graph storage, traversal, Cypher query support, and SPARQL (W3C standard)
// Graph and RDF data is persisted to PostgreSQL tables for durability across connections.

pub mod cypher;
pub mod operators;
pub mod sparql;
pub mod storage;
pub mod traversal;

pub use cypher::{execute_cypher, CypherQuery};
pub use storage::{Edge, EdgeStore, GraphStore, Node, NodeStore};
pub use traversal::{bfs, dfs, shortest_path_dijkstra, PathResult};

use dashmap::DashMap;
use pgrx::JsonB;
use std::collections::HashMap;
use std::sync::Arc;

/// Global graph storage registry (in-memory cache, backed by PG tables)
static GRAPH_REGISTRY: once_cell::sync::Lazy<DashMap<String, Arc<GraphStore>>> =
    once_cell::sync::Lazy::new(|| DashMap::new());

/// Ensure persistence tables exist (idempotent)
fn ensure_graph_tables() {
    use pgrx::prelude::*;
    Spi::run(
        "CREATE TABLE IF NOT EXISTS _ruvector_graphs (
            name TEXT PRIMARY KEY
        )",
    )
    .ok();
    Spi::run(
        "CREATE TABLE IF NOT EXISTS _ruvector_nodes (
            graph_name TEXT NOT NULL REFERENCES _ruvector_graphs(name) ON DELETE CASCADE,
            id BIGINT NOT NULL,
            labels TEXT[] NOT NULL DEFAULT '{}',
            properties JSONB NOT NULL DEFAULT '{}',
            PRIMARY KEY (graph_name, id)
        )",
    )
    .ok();
    Spi::run(
        "CREATE TABLE IF NOT EXISTS _ruvector_edges (
            graph_name TEXT NOT NULL REFERENCES _ruvector_graphs(name) ON DELETE CASCADE,
            id BIGINT NOT NULL,
            source BIGINT NOT NULL,
            target BIGINT NOT NULL,
            edge_type TEXT NOT NULL,
            properties JSONB NOT NULL DEFAULT '{}',
            PRIMARY KEY (graph_name, id)
        )",
    )
    .ok();
}

/// Load a graph from PostgreSQL tables into the in-memory cache
fn load_graph_from_tables(name: &str) -> Option<Arc<GraphStore>> {
    use pgrx::prelude::*;
    use serde_json::Value as JsonValue;

    // Check if graph exists in tables
    let exists = Spi::get_one_with_args::<bool>(
        "SELECT EXISTS(SELECT 1 FROM _ruvector_graphs WHERE name = $1)",
        vec![(PgBuiltInOids::TEXTOID.oid(), name.into_datum())],
    )
    .ok()
    .flatten()
    .unwrap_or(false);

    if !exists {
        return None;
    }

    let graph = Arc::new(GraphStore::new());

    // Load nodes
    let _ = Spi::connect(|client| {
        let tup_table = client.select(
            "SELECT id, labels, properties FROM _ruvector_nodes WHERE graph_name = $1 ORDER BY id",
            None,
            Some(vec![(PgBuiltInOids::TEXTOID.oid(), name.into_datum())]),
        )?;

        for row in tup_table {
            let id: i64 = row.get_by_name::<i64, _>("id")?.unwrap_or(0);
            let labels: Vec<String> = row
                .get_by_name::<Vec<String>, _>("labels")?
                .unwrap_or_default();
            let props_json: JsonB = row
                .get_by_name::<JsonB, _>("properties")?
                .unwrap_or(JsonB(serde_json::json!({})));

            let props: HashMap<String, JsonValue> = if let JsonValue::Object(map) = props_json.0 {
                map.into_iter().collect()
            } else {
                HashMap::new()
            };

            let mut node = Node::new(id as u64);
            node.labels = labels;
            node.properties = props;
            graph.nodes.insert(node);

            // Advance the ID counter past loaded IDs
            while graph.nodes.next_id() <= id as u64 {
                // next_id auto-increments
            }
        }
        Ok::<_, spi::Error>(())
    });

    // Load edges
    let _ = Spi::connect(|client| {
        let tup_table = client.select(
            "SELECT id, source, target, edge_type, properties FROM _ruvector_edges WHERE graph_name = $1 ORDER BY id",
            None,
            Some(vec![(PgBuiltInOids::TEXTOID.oid(), name.into_datum())]),
        )?;

        for row in tup_table {
            let id: i64 = row.get_by_name::<i64, _>("id")?.unwrap_or(0);
            let source: i64 = row.get_by_name::<i64, _>("source")?.unwrap_or(0);
            let target: i64 = row.get_by_name::<i64, _>("target")?.unwrap_or(0);
            let edge_type: String = row
                .get_by_name::<String, _>("edge_type")?
                .unwrap_or_default();
            let props_json: JsonB = row
                .get_by_name::<JsonB, _>("properties")?
                .unwrap_or(JsonB(serde_json::json!({})));

            let props: HashMap<String, JsonValue> = if let JsonValue::Object(map) = props_json.0 {
                map.into_iter().collect()
            } else {
                HashMap::new()
            };

            let mut edge = Edge::new(id as u64, source as u64, target as u64, edge_type);
            edge.properties = props;
            graph.edges.insert(edge);

            while graph.edges.next_id() <= id as u64 {}
        }
        Ok::<_, spi::Error>(())
    });

    GRAPH_REGISTRY.insert(name.to_string(), graph.clone());
    Some(graph)
}

/// Persist a graph entry to the backing table
fn persist_graph_name(name: &str) {
    use pgrx::prelude::*;
    Spi::run_with_args(
        "INSERT INTO _ruvector_graphs (name) VALUES ($1) ON CONFLICT DO NOTHING",
        Some(vec![(PgBuiltInOids::TEXTOID.oid(), name.into_datum())]),
    )
    .ok();
}

/// Persist a node to the backing table
pub fn persist_node(graph_name: &str, node: &Node) {
    use pgrx::prelude::*;
    let props = JsonB(serde_json::to_value(&node.properties).unwrap_or_default());
    Spi::run_with_args(
        "INSERT INTO _ruvector_nodes (graph_name, id, labels, properties)
         VALUES ($1, $2, $3, $4)
         ON CONFLICT (graph_name, id) DO UPDATE SET labels = $3, properties = $4",
        Some(vec![
            (PgBuiltInOids::TEXTOID.oid(), graph_name.into_datum()),
            (PgBuiltInOids::INT8OID.oid(), (node.id as i64).into_datum()),
            (
                PgBuiltInOids::TEXTARRAYOID.oid(),
                node.labels.clone().into_datum(),
            ),
            (PgBuiltInOids::JSONBOID.oid(), props.into_datum()),
        ]),
    )
    .ok();
}

/// Persist an edge to the backing table
pub fn persist_edge(graph_name: &str, edge: &Edge) {
    use pgrx::prelude::*;
    let props = JsonB(serde_json::to_value(&edge.properties).unwrap_or_default());
    Spi::run_with_args(
        "INSERT INTO _ruvector_edges (graph_name, id, source, target, edge_type, properties)
         VALUES ($1, $2, $3, $4, $5, $6)
         ON CONFLICT (graph_name, id) DO UPDATE SET source = $3, target = $4, edge_type = $5, properties = $6",
        Some(vec![
            (PgBuiltInOids::TEXTOID.oid(), graph_name.into_datum()),
            (PgBuiltInOids::INT8OID.oid(), (edge.id as i64).into_datum()),
            (
                PgBuiltInOids::INT8OID.oid(),
                (edge.source as i64).into_datum(),
            ),
            (
                PgBuiltInOids::INT8OID.oid(),
                (edge.target as i64).into_datum(),
            ),
            (
                PgBuiltInOids::TEXTOID.oid(),
                edge.edge_type.clone().into_datum(),
            ),
            (PgBuiltInOids::JSONBOID.oid(), props.into_datum()),
        ]),
    )
    .ok();
}

/// Get or create a graph by name (with persistence)
pub fn get_or_create_graph(name: &str) -> Arc<GraphStore> {
    if let Some(g) = GRAPH_REGISTRY.get(name) {
        return g.clone();
    }

    // Try loading from tables
    ensure_graph_tables();
    if let Some(g) = load_graph_from_tables(name) {
        return g;
    }

    // Create new
    persist_graph_name(name);
    GRAPH_REGISTRY
        .entry(name.to_string())
        .or_insert_with(|| Arc::new(GraphStore::new()))
        .clone()
}

/// Get an existing graph by name (checks tables if not in cache)
pub fn get_graph(name: &str) -> Option<Arc<GraphStore>> {
    if let Some(g) = GRAPH_REGISTRY.get(name) {
        return Some(g.clone());
    }

    // Try loading from persistent storage
    ensure_graph_tables();
    load_graph_from_tables(name)
}

/// Delete a graph by name (from cache and tables)
pub fn delete_graph(name: &str) -> bool {
    use pgrx::prelude::*;
    GRAPH_REGISTRY.remove(name);
    // CASCADE deletes nodes and edges
    Spi::run_with_args(
        "DELETE FROM _ruvector_graphs WHERE name = $1",
        Some(vec![(PgBuiltInOids::TEXTOID.oid(), name.into_datum())]),
    )
    .ok();
    true
}

/// List all graph names (from persistent storage)
pub fn list_graphs() -> Vec<String> {
    use pgrx::prelude::*;
    ensure_graph_tables();

    let mut names: Vec<String> = Vec::new();
    let _ = Spi::connect(|client| {
        let tup_table = client.select(
            "SELECT name FROM _ruvector_graphs ORDER BY name",
            None,
            None,
        )?;
        for row in tup_table {
            if let Some(name) = row.get_by_name::<String, _>("name")? {
                names.push(name);
            }
        }
        Ok::<_, spi::Error>(())
    });

    // Also include any in-memory-only graphs
    for entry in GRAPH_REGISTRY.iter() {
        if !names.contains(entry.key()) {
            names.push(entry.key().clone());
        }
    }

    names
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_graph_registry_in_memory() {
        // Pure in-memory test (no PG context)
        let graph = Arc::new(GraphStore::new());
        GRAPH_REGISTRY.insert("unit_test_graph".to_string(), graph.clone());

        let g2 = GRAPH_REGISTRY.get("unit_test_graph").map(|g| g.clone());
        assert!(g2.is_some());
        assert!(Arc::ptr_eq(&graph, &g2.unwrap()));

        GRAPH_REGISTRY.remove("unit_test_graph");
        assert!(GRAPH_REGISTRY.get("unit_test_graph").is_none());
    }
}
