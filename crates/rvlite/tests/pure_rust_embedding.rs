//! Pure-Rust embedding API (server-side use, e.g. workers-rs).
//!
//! Nothing here touches wasm-bindgen, so it must pass with
//! `cargo test -p rvlite --no-default-features`, which is the configuration
//! that drops the `browser` feature. It also runs under the default features.

use rvlite::cypher::{ContextValue, CypherEngine, CypherError, Value};
use rvlite::sparql::{execute_sparql, parse_sparql, Iri, RdfTerm, Triple, TripleStore};
use rvlite::sql::{SqlEngine, SqlParser};
use rvlite::GraphState;

fn string_cell(value: &ContextValue) -> Option<&str> {
    match value.as_value() {
        Some(Value::String(s)) => Some(s.as_str()),
        _ => None,
    }
}

fn knows_names(engine: &mut CypherEngine) -> Vec<(String, String)> {
    let result = engine
        .run("MATCH (a:Person)-[r:KNOWS]->(b:Person) RETURN a.name AS src, b.name AS dst")
        .expect("MATCH should succeed");
    assert_eq!(result.columns, vec!["src".to_string(), "dst".to_string()]);

    let mut pairs: Vec<(String, String)> = result
        .rows
        .iter()
        .map(|row| {
            let from = string_cell(&row["src"]).expect("a.name is a string");
            let to = string_cell(&row["dst"]).expect("b.name is a string");
            (from.to_string(), to.to_string())
        })
        .collect();
    pairs.sort();
    pairs
}

#[test]
fn cypher_create_match_round_trip() {
    let mut engine = CypherEngine::new();

    engine
        .run("CREATE (a:Person {name: 'Alice', age: 30})-[r:KNOWS {since: 2020}]->(b:Person {name: 'Bob', age: 25})")
        .expect("CREATE should succeed");
    engine
        .run("CREATE (c:Person {name: 'Carol'})")
        .expect("CREATE should succeed");

    let stats = engine.graph_stats();
    assert_eq!(stats.node_count, 3);
    assert_eq!(stats.edge_count, 1);

    let people = engine
        .run("MATCH (n:Person) RETURN n")
        .expect("MATCH should succeed");
    assert_eq!(people.rows.len(), 3);

    let adults = engine
        .run("MATCH (n:Person) WHERE n.age > 26 RETURN n")
        .expect("filtered MATCH should succeed");
    assert_eq!(adults.rows.len(), 1);
    let alice = adults.rows[0]
        .values()
        .find_map(ContextValue::as_node)
        .expect("row holds a node");
    assert_eq!(
        alice.get_property("name"),
        Some(&Value::String("Alice".to_string()))
    );

    assert_eq!(
        knows_names(&mut engine),
        vec![("Alice".to_string(), "Bob".to_string())]
    );
}

#[test]
fn cypher_state_survives_serde_round_trip() {
    let mut engine = CypherEngine::new();
    engine
        .run("CREATE (a:Person {name: 'Alice'})-[r:KNOWS]->(b:Person {name: 'Bob'})")
        .unwrap();

    // Persist the way a server would (e.g. into KV / Durable Object storage).
    let json = serde_json::to_string(&engine.export_state()).unwrap();
    let state: GraphState = serde_json::from_str(&json).unwrap();

    let mut restored = CypherEngine::new();
    restored
        .load_state(&state)
        .expect("load_state should succeed");

    assert_eq!(restored.graph_stats().node_count, 2);
    assert_eq!(restored.graph_stats().edge_count, 1);
    assert_eq!(
        knows_names(&mut restored),
        vec![("Alice".to_string(), "Bob".to_string())]
    );

    restored.clear();
    assert_eq!(restored.graph_stats().node_count, 0);
}

#[test]
fn cypher_errors_keep_js_compatible_text() {
    let mut engine = CypherEngine::new();
    let err = engine.run("THIS IS NOT CYPHER").unwrap_err();
    assert!(matches!(err, CypherError::Parse(_)), "{:?}", err);
    assert!(err.to_string().starts_with("Parse error: "), "{}", err);
}

#[test]
fn sql_engine_is_pure_rust() {
    let engine = SqlEngine::new();
    for sql in [
        "CREATE TABLE docs (id TEXT, embedding VECTOR(3))",
        "INSERT INTO docs (id, embedding) VALUES ('a', [1.0, 0.0, 0.0])",
        "INSERT INTO docs (id, embedding) VALUES ('b', [0.0, 1.0, 0.0])",
    ] {
        let stmt = SqlParser::new(sql).unwrap().parse().unwrap();
        engine.execute(stmt).unwrap();
    }

    let stmt = SqlParser::new("SELECT * FROM docs ORDER BY embedding <-> [1.0, 0.1, 0.0] LIMIT 1")
        .unwrap()
        .parse()
        .unwrap();
    let result = engine.execute(stmt).unwrap();
    assert_eq!(result.rows.len(), 1);
    assert!(result.rows[0].get("id").is_some());
}

#[test]
fn sparql_engine_is_pure_rust() {
    let store = TripleStore::new();
    store.insert(Triple::new(
        RdfTerm::iri("http://example.org/alice"),
        Iri::new("http://example.org/knows"),
        RdfTerm::iri("http://example.org/bob"),
    ));

    let query = parse_sparql("SELECT ?who WHERE { ?who <http://example.org/knows> ?o }").unwrap();
    match execute_sparql(&store, &query).unwrap() {
        rvlite::sparql::executor::QueryResult::Select(select) => {
            assert_eq!(select.bindings.len(), 1)
        }
        other => panic!("expected SELECT result, got {:?}", other),
    }
}
