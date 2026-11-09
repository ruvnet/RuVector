//! Hardening tests: malformed or hostile queries must return errors, never
//! panic, loop forever, or overflow the stack. The release wasm is built with
//! `panic = "abort"`, so any panic here is a trap that kills the instance.

use rvlite::cypher::{parse_cypher, tokenize};
use rvlite::sparql::parse_sparql;
use rvlite::sql::{SqlEngine, SqlParser};

/// Run on a thread with Rust's default 2 MiB stack (debug frames are much
/// larger than release wasm ones) so an unguarded recursion crashes here
/// instead of passing on a large main-thread stack.
fn small_stack<F: FnOnce() + Send + 'static>(f: F) {
    std::thread::Builder::new()
        .stack_size(2 << 20)
        .spawn(f)
        .unwrap()
        .join()
        .expect("parser crashed");
}

fn sql_parse(q: &str) -> Result<rvlite::sql::SqlStatement, rvlite::sql::ParseError> {
    SqlParser::new(q)?.parse()
}

fn sql_run(engine: &SqlEngine, q: &str) -> Result<rvlite::sql::ExecutionResult, String> {
    let stmt = sql_parse(q).map_err(|e| e.to_string())?;
    engine.execute(stmt).map_err(|e| e.to_string())
}

#[test]
fn cypher_dollar_does_not_loop_forever() {
    assert!(tokenize("$x").is_err());
    assert!(parse_cypher("$CREATE (a:A)").is_err());
    assert!(parse_cypher("MATCH (n) WHERE n.id = $id RETURN n").is_err());
}

#[test]
fn cypher_deep_nesting_is_rejected() {
    small_stack(|| {
        let n = 10_000;
        let parens = format!("RETURN {}1{}", "(".repeat(n), ")".repeat(n));
        assert!(parse_cypher(&parens).is_err());
        assert!(parse_cypher(&format!("RETURN {}true", "NOT ".repeat(n))).is_err());
        assert!(parse_cypher(&format!("RETURN {}1", "-".repeat(n))).is_err());
        assert!(parse_cypher(&format!("RETURN {}{}", "[".repeat(n), "]".repeat(n))).is_err());
        let maps = format!("RETURN {}1{}", "{a: ".repeat(n), "}".repeat(n));
        assert!(parse_cypher(&maps).is_err());
        let chain = format!("MATCH (a){} RETURN a", "-[:R]->(b)".repeat(n));
        assert!(parse_cypher(&chain).is_err());
    });
}

#[test]
fn cypher_reasonable_nesting_still_parses() {
    let n = 32;
    let parens = format!("RETURN {}1{}", "(".repeat(n), ")".repeat(n));
    assert!(parse_cypher(&parens).is_ok());
    let chain = format!("MATCH (a){} RETURN a", "-[:R]->(b)".repeat(n));
    assert!(parse_cypher(&chain).is_ok());
}

#[test]
fn sparql_multibyte_input_does_not_panic() {
    for q in [
        "SELECT ?s WHERE { ?s ?p ?o . é }",
        "PREFIX ex: <http://ex.org/> SELECT ?x WHERE { ?x ex:knows ?y . FILTER(?y != ex:b) ß} LIMIT 1",
        "٣",
        "SELECT ²",
        "ASK é",
    ] {
        let _ = parse_sparql(q);
    }
}

#[test]
fn sparql_deep_nesting_is_rejected() {
    small_stack(|| {
        let n = 10_000;
        let groups = format!(
            "SELECT ?s WHERE {}?s ?p ?o{}",
            "{ ".repeat(n),
            " }".repeat(n)
        );
        assert!(parse_sparql(&groups).is_err());
        let filter = format!(
            "SELECT ?s WHERE {{ ?s ?p ?o FILTER({}?o{}) }}",
            "(".repeat(n),
            ")".repeat(n)
        );
        assert!(parse_sparql(&filter).is_err());
        let path = format!(
            "SELECT ?s WHERE {{ ?s {}<http://p>{} ?o }}",
            "(".repeat(n),
            ")".repeat(n)
        );
        assert!(parse_sparql(&path).is_err());
    });
}

#[test]
fn sparql_reasonable_nesting_still_parses() {
    let n = 32;
    let filter = format!(
        "SELECT ?s WHERE {{ ?s ?p ?o FILTER({}?o{}) }}",
        "(".repeat(n),
        ")".repeat(n)
    );
    assert!(parse_sparql(&filter).is_ok());
    let groups = format!(
        "SELECT ?s WHERE {}?s ?p ?o{}",
        "{ ".repeat(n),
        " }".repeat(n)
    );
    assert!(parse_sparql(&groups).is_ok());
}

#[test]
fn sql_bad_numbers_are_errors() {
    assert!(sql_parse("SELECT * FROM t WHERE id = 99999999999999999999").is_err());
    assert!(sql_parse("SELECT * FROM t WHERE id = 1²").is_err());
    assert!(sql_parse("INSERT INTO t (id, n) VALUES ('a', 99999999999999999999)").is_err());
    assert!(sql_parse("INSERT INTO t (id, n) VALUES ('a', ٣)").is_err());
}

#[test]
fn sql_deep_not_is_rejected() {
    small_stack(|| {
        let q = format!("SELECT * FROM t WHERE {}a = 1", "NOT ".repeat(100_000));
        assert!(sql_parse(&q).is_err());
    });
}

#[test]
fn sql_negative_vector_components() {
    let engine = SqlEngine::new();
    sql_run(&engine, "CREATE TABLE docs (id TEXT, embedding VECTOR(3))").unwrap();
    sql_run(
        &engine,
        "INSERT INTO docs (id, embedding) VALUES ('a', [-1.0, 0.5, -0.25])",
    )
    .unwrap();
    let res = sql_run(
        &engine,
        "SELECT * FROM docs ORDER BY embedding <-> [-1.0, 0.5, -0.25] LIMIT 1",
    )
    .unwrap();
    assert_eq!(res.rows.len(), 1);
}

#[test]
fn sql_vector_dimensions_are_bounded() {
    let engine = SqlEngine::new();
    assert!(sql_run(&engine, "CREATE TABLE big (id TEXT, v VECTOR(4000000000))").is_err());
    assert!(sql_run(&engine, "CREATE TABLE zero (id TEXT, v VECTOR(0))").is_err());
}

#[test]
fn sql_huge_limit_with_filter_does_not_overflow() {
    let engine = SqlEngine::new();
    sql_run(&engine, "CREATE TABLE docs (id TEXT, embedding VECTOR(3))").unwrap();
    sql_run(
        &engine,
        "INSERT INTO docs (id, embedding) VALUES ('a', [1.0, 2.0, 3.0])",
    )
    .unwrap();
    let res = sql_run(
        &engine,
        "SELECT * FROM docs WHERE id = 'a' ORDER BY embedding <-> [1.0, 2.0, 3.0] LIMIT 99999999999999999999",
    )
    .unwrap();
    assert_eq!(res.rows.len(), 1);
}

#[test]
fn sql_unsupported_where_shape_is_an_error_not_a_full_scan() {
    let engine = SqlEngine::new();
    sql_run(
        &engine,
        "CREATE TABLE docs (id TEXT, name TEXT, embedding VECTOR(3))",
    )
    .unwrap();
    sql_run(
        &engine,
        "INSERT INTO docs (id, name, embedding) VALUES ('a', 'x', [1.0, 2.0, 3.0])",
    )
    .unwrap();
    assert!(sql_run(&engine, "SELECT * FROM docs WHERE 'b' = id").is_err());
    assert!(sql_run(&engine, "SELECT * FROM docs WHERE id = name").is_err());
}
