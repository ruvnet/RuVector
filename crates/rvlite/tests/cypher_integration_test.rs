//! Integration tests for Cypher query engine in rvlite

use rvlite::cypher::*;

#[test]
fn test_create_single_node() {
    let mut graph = PropertyGraph::new();
    let query = "CREATE (n:Person {name: 'Alice', age: 30})";

    let ast = parse_cypher(query).expect("Failed to parse query");
    let mut executor = Executor::new(&mut graph);
    let result = executor.execute(&ast);

    assert!(result.is_ok(), "Execution failed: {:?}", result.err());
    assert_eq!(graph.stats().node_count, 1);

    // Verify node was created with correct properties
    let nodes = graph.find_nodes_by_label("Person");
    assert_eq!(nodes.len(), 1);

    let node = nodes[0];
    assert_eq!(
        node.get_property("name"),
        Some(&Value::String("Alice".to_string()))
    );
    assert_eq!(node.get_property("age"), Some(&Value::Integer(30)));
}

#[test]
fn test_create_relationship() {
    let mut graph = PropertyGraph::new();
    let query =
        "CREATE (a:Person {name: 'Alice'})-[r:KNOWS {since: 2020}]->(b:Person {name: 'Bob'})";

    let ast = parse_cypher(query).expect("Failed to parse query");
    let mut executor = Executor::new(&mut graph);
    let result = executor.execute(&ast);

    assert!(result.is_ok(), "Execution failed: {:?}", result.err());

    let stats = graph.stats();
    assert_eq!(stats.node_count, 2, "Should have 2 nodes");
    assert_eq!(stats.edge_count, 1, "Should have 1 edge");

    // Verify nodes
    let persons = graph.find_nodes_by_label("Person");
    assert_eq!(persons.len(), 2);

    // Verify edge
    let knows_edges = graph.find_edges_by_type("KNOWS");
    assert_eq!(knows_edges.len(), 1);
    assert_eq!(
        knows_edges[0].get_property("since"),
        Some(&Value::Integer(2020))
    );
}

#[test]
fn test_match_nodes() {
    let mut graph = PropertyGraph::new();

    // Create test data
    let create = "CREATE (a:Person {name: 'Alice', age: 30}), (b:Person {name: 'Bob', age: 25})";
    let ast = parse_cypher(create).expect("Failed to parse CREATE");
    let mut executor = Executor::new(&mut graph);
    executor.execute(&ast).expect("Failed to execute CREATE");

    // Match all persons
    let match_query = "MATCH (n:Person) RETURN n";
    let ast = parse_cypher(match_query).expect("Failed to parse MATCH");
    let mut executor = Executor::new(&mut graph);
    let result = executor.execute(&ast);

    assert!(result.is_ok(), "Match execution failed: {:?}", result.err());
}

#[test]
fn test_match_relationship() {
    let mut graph = PropertyGraph::new();

    // Create test data
    let create = "CREATE (a:Person {name: 'Alice'})-[r:KNOWS]->(b:Person {name: 'Bob'})";
    let ast = parse_cypher(create).expect("Failed to parse CREATE");
    let mut executor = Executor::new(&mut graph);
    executor.execute(&ast).expect("Failed to execute CREATE");

    // Match the relationship
    let match_query = "MATCH (a:Person)-[r:KNOWS]->(b:Person) RETURN a, r, b";
    let ast = parse_cypher(match_query).expect("Failed to parse MATCH");
    let mut executor = Executor::new(&mut graph);
    let result = executor.execute(&ast);

    assert!(
        result.is_ok(),
        "Relationship match failed: {:?}",
        result.err()
    );
}

#[test]
fn test_parser_coverage() {
    let test_queries = vec![
        // Simple node creation
        "CREATE (n:Label)",
        "CREATE (n:Label {prop: 'value'})",
        "CREATE (n:Label {x: 1, y: 2})",
        // Simple relationship
        "CREATE (a)-[r:TYPE]->(b)",
        "CREATE (a:A)-[r:TYPE]->(b:B)",
        // Match patterns
        "MATCH (n) RETURN n",
        "MATCH (n:Label) RETURN n",
        "MATCH (n {prop: 'value'}) RETURN n",
        "MATCH (a)-[r]->(b) RETURN a, r, b",
        "MATCH (a)-[r:TYPE]->(b) RETURN a, b",
        // WHERE clauses
        "MATCH (n:Person) WHERE n.age > 18 RETURN n",
        "MATCH (n:Person) WHERE n.name = 'Alice' RETURN n",
        // Multiple statements
        "CREATE (n:Person) RETURN n",
        "MATCH (n:Person) DELETE n",
        // Complex patterns
        "CREATE (a:A)-[r1:R1]->(b:B)-[r2:R2]->(c:C)",
    ];

    for query in test_queries {
        let result = parse_cypher(query);
        assert!(
            result.is_ok(),
            "Failed to parse query: {}\nError: {:?}",
            query,
            result.err()
        );
    }
}

#[test]
fn test_tokenizer() {
    let query = "MATCH (n:Person {name: 'Alice', age: 30}) WHERE n.age > 18 RETURN n.name";
    let tokens = tokenize(query).expect("Failed to tokenize");

    assert!(!tokens.is_empty());

    // Should have MATCH keyword
    assert!(tokens.iter().any(|t| matches!(t.kind, TokenKind::Match)));

    // Should have WHERE keyword
    assert!(tokens.iter().any(|t| matches!(t.kind, TokenKind::Where)));

    // Should have RETURN keyword
    assert!(tokens.iter().any(|t| matches!(t.kind, TokenKind::Return)));

    // Should have string literal 'Alice'
    assert!(tokens
        .iter()
        .any(|t| matches!(&t.kind, TokenKind::String(s) if s == "Alice")));

    // Should have integer 30
    assert!(tokens
        .iter()
        .any(|t| matches!(t.kind, TokenKind::Integer(30))));
}

#[test]
#[cfg(target_family = "wasm")]
fn test_cypher_engine() {
    let mut engine = CypherEngine::new();

    // Test CREATE
    let create_query = "CREATE (n:Person {name: 'Alice', age: 30})";
    let result = parse_cypher(create_query);
    assert!(result.is_ok());

    let stats_result = engine.stats();
    assert!(stats_result.is_ok());

    // Test clear
    engine.clear();
}

#[test]
fn test_property_graph_operations() {
    let mut graph = PropertyGraph::new();

    // Create nodes
    let node1 = Node::new(graph.generate_node_id())
        .with_label("Person".to_string())
        .with_property("name".to_string(), Value::String("Alice".to_string()))
        .with_property("age".to_string(), Value::Integer(30));

    let node2 = Node::new(graph.generate_node_id())
        .with_label("Person".to_string())
        .with_property("name".to_string(), Value::String("Bob".to_string()));

    let id1 = graph.add_node(node1.clone());
    let id2 = graph.add_node(node2.clone());

    // Create edge
    let edge = Edge::new(
        graph.generate_edge_id(),
        id1.clone(),
        id2.clone(),
        "KNOWS".to_string(),
    )
    .with_property("since".to_string(), Value::Integer(2020));

    graph.add_edge(edge).expect("Failed to add edge");

    // Verify
    assert_eq!(graph.stats().node_count, 2);
    assert_eq!(graph.stats().edge_count, 1);

    // Test node lookup
    let found = graph.get_node(&id1);
    assert!(found.is_some());
    assert_eq!(found.unwrap().id, id1);

    // Test label index
    let persons = graph.find_nodes_by_label("Person");
    assert_eq!(persons.len(), 2);

    // Test edge type index
    let knows = graph.find_edges_by_type("KNOWS");
    assert_eq!(knows.len(), 1);
}

#[test]
fn test_expression_evaluation() {
    let mut graph = PropertyGraph::new();

    // Create a node
    let query = "CREATE (n:Person {name: 'Alice', age: 30, active: true})";
    let ast = parse_cypher(query).unwrap();
    let mut executor = Executor::new(&mut graph);
    executor.execute(&ast).unwrap();

    // Test value types
    let nodes = graph.find_nodes_by_label("Person");
    assert_eq!(nodes.len(), 1);

    let node = nodes[0];
    assert_eq!(node.get_property("name").unwrap().as_str(), Some("Alice"));
    assert_eq!(node.get_property("age").unwrap().as_i64(), Some(30));
    assert_eq!(node.get_property("active").unwrap().as_bool(), Some(true));
}

#[test]
fn test_match_return_multiple_rows() {
    let mut graph = PropertyGraph::new();

    // Create three Person nodes
    let create =
        "CREATE (a:Person {name: 'Alice', age: 30}), (b:Person {name: 'Bob', age: 25}), (c:Person {name: 'Charlie', age: 35})";
    let ast = parse_cypher(create).expect("Failed to parse CREATE");
    let mut executor = Executor::new(&mut graph);
    executor.execute(&ast).expect("Failed to execute CREATE");

    assert_eq!(graph.stats().node_count, 3);

    // MATCH all persons and RETURN their properties
    let match_query = "MATCH (n:Person) RETURN n.name, n.age";
    let ast = parse_cypher(match_query).expect("Failed to parse MATCH RETURN");
    let mut executor = Executor::new(&mut graph);
    let result = executor.execute(&ast).expect("MATCH RETURN failed");

    // Should have 3 rows (one per Person node)
    assert_eq!(
        result.rows.len(),
        3,
        "Expected 3 rows but got {}. Rows: {:?}",
        result.rows.len(),
        result.rows
    );

    // Column names should be "n.name" and "n.age" (not "?column?")
    assert_eq!(result.columns.len(), 2);
    assert!(
        result.columns.contains(&"n.name".to_string()),
        "Expected column 'n.name', got {:?}",
        result.columns
    );
    assert!(
        result.columns.contains(&"n.age".to_string()),
        "Expected column 'n.age', got {:?}",
        result.columns
    );

    // Collect all names from the result rows
    let mut names: Vec<String> = result
        .rows
        .iter()
        .filter_map(|row| {
            row.get("n.name").and_then(|v| {
                if let ContextValue::Value(Value::String(s)) = v {
                    Some(s.clone())
                } else {
                    None
                }
            })
        })
        .collect();
    names.sort();

    assert_eq!(names, vec!["Alice", "Bob", "Charlie"]);
}

#[test]
fn test_match_where_return_multiple_rows() {
    let mut graph = PropertyGraph::new();

    // Create nodes with varying ages
    let create =
        "CREATE (a:Person {name: 'Alice', age: 30}), (b:Person {name: 'Bob', age: 17}), (c:Person {name: 'Charlie', age: 25})";
    let ast = parse_cypher(create).expect("Failed to parse CREATE");
    let mut executor = Executor::new(&mut graph);
    executor.execute(&ast).expect("Failed to execute CREATE");

    // MATCH with WHERE filter
    let match_query = "MATCH (n:Person) WHERE n.age > 18 RETURN n.name";
    let ast = parse_cypher(match_query).expect("Failed to parse MATCH WHERE RETURN");
    let mut executor = Executor::new(&mut graph);
    let result = executor.execute(&ast).expect("MATCH WHERE RETURN failed");

    // Bob (age 17) should be filtered out
    assert_eq!(
        result.rows.len(),
        2,
        "Expected 2 rows (adults only) but got {}",
        result.rows.len()
    );

    let mut names: Vec<String> = result
        .rows
        .iter()
        .filter_map(|row| {
            row.get("n.name").and_then(|v| {
                if let ContextValue::Value(Value::String(s)) = v {
                    Some(s.clone())
                } else {
                    None
                }
            })
        })
        .collect();
    names.sort();

    assert_eq!(names, vec!["Alice", "Charlie"]);
}

#[test]
fn test_match_return_variable_node() {
    let mut graph = PropertyGraph::new();

    // Create two nodes
    let create = "CREATE (a:Person {name: 'Alice'}), (b:Person {name: 'Bob'})";
    let ast = parse_cypher(create).expect("Failed to parse CREATE");
    let mut executor = Executor::new(&mut graph);
    executor.execute(&ast).expect("Failed to execute CREATE");

    // RETURN the whole node variable
    let match_query = "MATCH (n:Person) RETURN n";
    let ast = parse_cypher(match_query).expect("Failed to parse MATCH RETURN n");
    let mut executor = Executor::new(&mut graph);
    let result = executor.execute(&ast).expect("MATCH RETURN n failed");

    assert_eq!(result.rows.len(), 2, "Expected 2 rows for RETURN n");
    assert_eq!(result.columns, vec!["n".to_string()]);

    // Each row should contain a Node
    for row in &result.rows {
        let val = row.get("n").expect("Missing 'n' column");
        assert!(
            matches!(val, ContextValue::Node(_)),
            "Expected Node but got {:?}",
            val
        );
    }
}
