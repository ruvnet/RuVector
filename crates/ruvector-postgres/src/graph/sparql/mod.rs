// SPARQL (SPARQL Protocol and RDF Query Language) module for ruvector-postgres
//
// Provides W3C-compliant SPARQL 1.1 query support for RDF data with
// PostgreSQL storage backend and vector similarity extensions.
//
// Features:
// - SPARQL 1.1 Query Language (SELECT, CONSTRUCT, ASK, DESCRIBE)
// - SPARQL 1.1 Update Language (INSERT, DELETE, LOAD, CLEAR)
// - RDF triple store with efficient indexing (SPO, POS, OSP)
// - Property paths (sequence, alternative, inverse, transitive)
// - Aggregates and GROUP BY
// - FILTER expressions and built-in functions
// - Vector similarity extensions for hybrid semantic search
// - Standard result formats (JSON, XML, CSV, TSV)

// Allow warnings for incomplete SPARQL features
#![allow(dead_code)]
#![allow(unused_variables)]
#![allow(unused_mut)]

pub mod ast;
pub mod executor;
pub mod functions;
pub mod parser;
pub mod results;
pub mod triple_store;

pub use ast::{
    Aggregate, AskQuery, ConstructQuery, DeleteData, DescribeQuery, Expression, Filter,
    GraphPattern, GroupCondition, InsertData, Iri, Literal, Modify, OrderCondition, QueryForm,
    RdfTerm, SelectQuery, SolutionModifier, SparqlQuery, TriplePattern, UpdateOperation,
};
pub use executor::{execute_sparql, SparqlContext};
pub use parser::parse_sparql;
pub use results::{format_results, ResultFormat, SparqlResults};
pub use triple_store::{Triple, TripleIndex, TripleStore};

use dashmap::DashMap;
use once_cell::sync::Lazy;
use std::sync::Arc;

/// Global RDF triple store registry (in-memory cache, backed by PG tables)
static TRIPLE_STORE_REGISTRY: Lazy<DashMap<String, Arc<TripleStore>>> =
    Lazy::new(|| DashMap::new());

/// Ensure RDF persistence tables exist (idempotent)
fn ensure_rdf_tables() {
    use pgrx::prelude::*;
    Spi::run(
        "CREATE TABLE IF NOT EXISTS _ruvector_rdf_stores (
            name TEXT PRIMARY KEY
        )",
    )
    .ok();
    Spi::run(
        "CREATE TABLE IF NOT EXISTS _ruvector_triples (
            store_name TEXT NOT NULL REFERENCES _ruvector_rdf_stores(name) ON DELETE CASCADE,
            id BIGINT NOT NULL,
            subject TEXT NOT NULL,
            predicate TEXT NOT NULL,
            object TEXT NOT NULL,
            graph_name TEXT,
            PRIMARY KEY (store_name, id)
        )",
    )
    .ok();
}

/// Load a triple store from PostgreSQL tables
fn load_store_from_tables(name: &str) -> Option<Arc<TripleStore>> {
    use pgrx::prelude::*;

    let exists = Spi::get_one_with_args::<bool>(
        "SELECT EXISTS(SELECT 1 FROM _ruvector_rdf_stores WHERE name = $1)",
        vec![(PgBuiltInOids::TEXTOID.oid(), name.into_datum())],
    )
    .ok()
    .flatten()
    .unwrap_or(false);

    if !exists {
        return None;
    }

    let store = Arc::new(TripleStore::new());

    let _ = pgrx::prelude::Spi::connect(|client| {
        let tup_table = client.select(
            "SELECT id, subject, predicate, object, graph_name FROM _ruvector_triples WHERE store_name = $1 ORDER BY id",
            None,
            Some(vec![(PgBuiltInOids::TEXTOID.oid(), name.into_datum())]),
        )?;

        for row in tup_table {
            let subject: String = row.get_by_name::<String, _>("subject")?.unwrap_or_default();
            let predicate: String = row
                .get_by_name::<String, _>("predicate")?
                .unwrap_or_default();
            let object: String = row.get_by_name::<String, _>("object")?.unwrap_or_default();
            let graph_name: Option<String> = row.get_by_name::<String, _>("graph_name")?;

            let triple = Triple::from_strings(&subject, &predicate, &object);
            store.insert_into_graph(triple, graph_name.as_deref());
        }
        Ok::<_, pgrx::spi::Error>(())
    });

    TRIPLE_STORE_REGISTRY.insert(name.to_string(), store.clone());
    Some(store)
}

/// Persist a triple to the backing table
pub fn persist_triple(store_name: &str, id: u64, triple: &Triple, graph: Option<&str>) {
    use pgrx::prelude::*;
    let subj = triple_store::term_to_key(&triple.subject);
    let pred = triple.predicate.as_str().to_string();
    let obj = triple_store::term_to_key(&triple.object);

    Spi::run_with_args(
        "INSERT INTO _ruvector_triples (store_name, id, subject, predicate, object, graph_name)
         VALUES ($1, $2, $3, $4, $5, $6)
         ON CONFLICT (store_name, id) DO NOTHING",
        Some(vec![
            (PgBuiltInOids::TEXTOID.oid(), store_name.into_datum()),
            (PgBuiltInOids::INT8OID.oid(), (id as i64).into_datum()),
            (PgBuiltInOids::TEXTOID.oid(), subj.into_datum()),
            (PgBuiltInOids::TEXTOID.oid(), pred.into_datum()),
            (PgBuiltInOids::TEXTOID.oid(), obj.into_datum()),
            (
                PgBuiltInOids::TEXTOID.oid(),
                graph.map(|g| g.to_string()).into_datum(),
            ),
        ]),
    )
    .ok();
}

/// Get or create a triple store by name (with persistence)
pub fn get_or_create_store(name: &str) -> Arc<TripleStore> {
    if let Some(s) = TRIPLE_STORE_REGISTRY.get(name) {
        return s.clone();
    }

    ensure_rdf_tables();
    if let Some(s) = load_store_from_tables(name) {
        return s;
    }

    // Create new
    use pgrx::prelude::*;
    Spi::run_with_args(
        "INSERT INTO _ruvector_rdf_stores (name) VALUES ($1) ON CONFLICT DO NOTHING",
        Some(vec![(PgBuiltInOids::TEXTOID.oid(), name.into_datum())]),
    )
    .ok();

    TRIPLE_STORE_REGISTRY
        .entry(name.to_string())
        .or_insert_with(|| Arc::new(TripleStore::new()))
        .clone()
}

/// Get an existing triple store by name (checks tables if not in cache)
pub fn get_store(name: &str) -> Option<Arc<TripleStore>> {
    if let Some(s) = TRIPLE_STORE_REGISTRY.get(name) {
        return Some(s.clone());
    }

    ensure_rdf_tables();
    load_store_from_tables(name)
}

/// Delete a triple store by name (from cache and tables)
pub fn delete_store(name: &str) -> bool {
    use pgrx::prelude::*;
    TRIPLE_STORE_REGISTRY.remove(name);
    // CASCADE deletes triples
    Spi::run_with_args(
        "DELETE FROM _ruvector_rdf_stores WHERE name = $1",
        Some(vec![(PgBuiltInOids::TEXTOID.oid(), name.into_datum())]),
    )
    .ok();
    true
}

/// List all triple store names (from persistent storage)
pub fn list_stores() -> Vec<String> {
    use pgrx::prelude::*;
    ensure_rdf_tables();

    let mut names: Vec<String> = Vec::new();
    let _ = Spi::connect(|client| {
        let tup_table = client.select(
            "SELECT name FROM _ruvector_rdf_stores ORDER BY name",
            None,
            None,
        )?;
        for row in tup_table {
            if let Some(name) = row.get_by_name::<String, _>("name")? {
                names.push(name);
            }
        }
        Ok::<_, pgrx::spi::Error>(())
    });

    for entry in TRIPLE_STORE_REGISTRY.iter() {
        if !names.contains(entry.key()) {
            names.push(entry.key().clone());
        }
    }

    names
}

/// SPARQL error type
#[derive(Debug, Clone, thiserror::Error)]
pub enum SparqlError {
    #[error("Parse error: {0}")]
    ParseError(String),

    #[error("Variable not bound: {0}")]
    UnboundVariable(String),

    #[error("Type mismatch: expected {expected}, got {actual}")]
    TypeMismatch { expected: String, actual: String },

    #[error("Store not found: {0}")]
    StoreNotFound(String),

    #[error("Invalid IRI: {0}")]
    InvalidIri(String),

    #[error("Invalid literal: {0}")]
    InvalidLiteral(String),

    #[error("Unsupported operation: {0}")]
    UnsupportedOperation(String),

    #[error("Execution error: {0}")]
    ExecutionError(String),

    #[error("Aggregate error: {0}")]
    AggregateError(String),

    #[error("Property path error: {0}")]
    PropertyPathError(String),
}

/// Result type for SPARQL operations
pub type SparqlResult<T> = Result<T, SparqlError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_store_registry() {
        let store1 = get_or_create_store("test_sparql_store");
        let store2 = get_store("test_sparql_store");

        assert!(store2.is_some());
        assert!(Arc::ptr_eq(&store1, &store2.unwrap()));

        let stores = list_stores();
        assert!(stores.contains(&"test_sparql_store".to_string()));

        assert!(delete_store("test_sparql_store"));
        assert!(get_store("test_sparql_store").is_none());
    }
}
