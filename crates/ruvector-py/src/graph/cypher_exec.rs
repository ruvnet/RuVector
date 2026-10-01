//! Cypher `MATCH` execution over an in-memory [`GraphDB`].
//!
//! Ported from `crates/ruvector-graph-node/src/cypher_exec.rs` @
//! `cebb38bc4`; logic unchanged. Before that module existed,
//! `GraphDatabase::query()` (the NAPI binding) handled exactly one shape —
//! `MATCH (n:Label)` — by consulting the label index. Every other shape
//! fell through an empty `if` branch and returned zero rows: the
//! label-less `MATCH (n)`, any `WHERE` filter, and every relationship
//! pattern. See ruvnet/ruvector#879. This port brings that fix to
//! `ruvector-py` without depending on the NAPI crate: everything here was
//! already `ruvector_graph`-only (plus `std`), so it copies over as-is.
//!
//! Scope. This is a single-pattern matcher, not a planner: each pattern in the
//! `MATCH` is resolved independently against the graph and its rows unioned.
//! Cross-pattern joins, variable-length paths and aggregations remain
//! unimplemented — but they now report themselves as unsupported through
//! [`ExecOutcome::unsupported`] instead of silently returning an empty set,
//! which is the failure mode that made #879 hard to spot from the outside.
//!
//! Note on operator coverage: the evaluator (`cypher_eval.rs`) implements
//! `CONTAINS`, `STARTS WITH`, `ENDS WITH`, `IN`, `IS NULL` and `=~` because
//! `BinaryOperator` declares them, but the lexer does not yet tokenise any
//! of those, so no parse can reach those arms today. They are kept so that
//! extending the lexer is a one-sided change; `=~` is the exception and is
//! reported unsupported because no regex engine is linked into this crate.

use std::collections::HashSet;

use ruvector_graph::cypher::ast::{BinaryOperator, Expression, MatchClause, NodePattern, Pattern};
use ruvector_graph::{Edge, GraphDB, Node};

use super::cypher_eval::{eval, lookup_property, values_equal, Binding, Bindings};

/// Does a node satisfy the inline property map of its pattern —
/// the `{id: 'n1'}` in `MATCH (n {id: 'n1'})`?
fn matches_inline_props(
    node: &Node,
    props: &Option<ruvector_graph::cypher::ast::PropertyMap>,
    bindings: &Bindings,
) -> bool {
    let Some(props) = props else {
        return true;
    };
    props.iter().all(|(key, expected)| {
        let actual = lookup_property(&Binding::Node(node.clone()), key);
        values_equal(&actual, &eval(expected, bindings))
    })
}

fn edge_matches_inline_props(
    edge: &Edge,
    props: &Option<ruvector_graph::cypher::ast::PropertyMap>,
    bindings: &Bindings,
) -> bool {
    let Some(props) = props else {
        return true;
    };
    props.iter().all(|(key, expected)| {
        let actual = lookup_property(&Binding::Edge(edge.clone()), key);
        values_equal(&actual, &eval(expected, bindings))
    })
}

fn node_matches_labels(node: &Node, labels: &[String]) -> bool {
    labels.iter().all(|want| node.has_label(want))
}

/// Candidate nodes for a node pattern, narrowed by the label index when it can be.
fn candidates_for(gdb: &GraphDB, pattern: &NodePattern) -> Vec<Node> {
    match pattern.labels.first() {
        // Multiple labels are conjunctive: seed from the most selective index
        // we have (the first label) and filter the rest.
        Some(label) => gdb
            .get_nodes_by_label(label)
            .into_iter()
            .filter(|n| node_matches_labels(n, &pattern.labels))
            .collect(),
        // No label to index on — this is the `MATCH (n)` full scan that #879
        // reported as returning nothing.
        None => gdb.all_nodes(),
    }
}

/// The rows a `MATCH` produced, plus anything in it this executor could not honour.
#[derive(Debug, Default)]
pub(super) struct ExecOutcome {
    pub(super) nodes: Vec<Node>,
    pub(super) edges: Vec<Edge>,
    /// Human-readable descriptions of clauses that were parsed but not executed.
    /// The caller surfaces these as an error rather than returning a partial
    /// result that looks complete.
    pub(super) unsupported: Vec<String>,
}

impl ExecOutcome {
    fn push_node(&mut self, seen: &mut HashSet<String>, node: Node) {
        if seen.insert(node.id.clone()) {
            self.nodes.push(node);
        }
    }
}

/// Execute one `MATCH` clause against the graph.
pub(super) fn execute_match(gdb: &GraphDB, clause: &MatchClause) -> ExecOutcome {
    let mut outcome = ExecOutcome::default();
    let mut seen_nodes: HashSet<String> = HashSet::new();
    let mut seen_edges: HashSet<String> = HashSet::new();
    let predicate = clause.where_clause.as_ref().map(|w| &w.condition);

    if let Some(condition) = predicate {
        if uses_regex(condition) {
            outcome
                .unsupported
                .push("WHERE ... =~ (regex matching) is not supported".to_string());
        }
    }

    for pattern in &clause.patterns {
        exec_pattern(
            gdb,
            pattern,
            predicate,
            &mut outcome,
            &mut seen_nodes,
            &mut seen_edges,
        );
    }
    outcome
}

fn uses_regex(expr: &Expression) -> bool {
    match expr {
        Expression::BinaryOp { left, op, right } => {
            *op == BinaryOperator::Matches || uses_regex(left) || uses_regex(right)
        }
        Expression::UnaryOp { operand, .. } => uses_regex(operand),
        _ => false,
    }
}

fn exec_pattern(
    gdb: &GraphDB,
    pattern: &Pattern,
    predicate: Option<&Expression>,
    outcome: &mut ExecOutcome,
    seen_nodes: &mut HashSet<String>,
    seen_edges: &mut HashSet<String>,
) {
    match pattern {
        Pattern::Node(np) => {
            for node in candidates_for(gdb, np) {
                let mut bindings = Bindings::new();
                if let Some(var) = &np.variable {
                    bindings.insert(var.clone(), Binding::Node(node.clone()));
                }
                if !matches_inline_props(&node, &np.properties, &bindings) {
                    continue;
                }
                if let Some(condition) = predicate {
                    if !eval(condition, &bindings).truthy() {
                        continue;
                    }
                }
                outcome.push_node(seen_nodes, node);
            }
        }
        Pattern::Relationship(rp) => {
            // A typed relationship uses the edge-type index; an untyped one
            // has to scan, same reasoning as the label-less node pattern.
            let edges = match &rp.rel_type {
                Some(t) => gdb.get_edges_by_type(t),
                None => gdb.all_edges(),
            };
            // Only a direct `(a)-[r]->(b)` target is resolved; a chained
            // pattern nests another Pattern here and needs a real join.
            let target: Option<&NodePattern> = match rp.to.as_ref() {
                Pattern::Node(np) => Some(np),
                _ => {
                    outcome.unsupported.push(
                        "chained relationship patterns ((a)-[]->(b)<-[]-(c)) are not supported"
                            .to_string(),
                    );
                    None
                }
            };
            if rp.range.is_some() {
                outcome
                    .unsupported
                    .push("variable-length relationships ([*1..n]) are not supported".to_string());
            }

            for edge in edges {
                let Some(from_node) = gdb.get_node(&edge.from) else {
                    continue;
                };
                let Some(to_node) = gdb.get_node(&edge.to) else {
                    continue;
                };
                // Undirected patterns accept the edge in either orientation;
                // Incoming flips which endpoint the `from` pattern must match.
                let orientations: &[(&Node, &Node)] = match rp.direction {
                    ruvector_graph::cypher::ast::Direction::Outgoing => &[(&from_node, &to_node)],
                    ruvector_graph::cypher::ast::Direction::Incoming => &[(&to_node, &from_node)],
                    ruvector_graph::cypher::ast::Direction::Undirected => {
                        &[(&from_node, &to_node), (&to_node, &from_node)]
                    }
                };

                for (src, dst) in orientations {
                    if !node_matches_labels(src, &rp.from.labels) {
                        continue;
                    }
                    if let Some(tp) = target {
                        if !node_matches_labels(dst, &tp.labels) {
                            continue;
                        }
                    }

                    let mut bindings = Bindings::new();
                    if let Some(var) = &rp.variable {
                        bindings.insert(var.clone(), Binding::Edge(edge.clone()));
                    }
                    if let Some(var) = &rp.from.variable {
                        bindings.insert(var.clone(), Binding::Node((*src).clone()));
                    }
                    if let Some(tp) = target {
                        if let Some(var) = &tp.variable {
                            bindings.insert(var.clone(), Binding::Node((*dst).clone()));
                        }
                    }

                    if !matches_inline_props(src, &rp.from.properties, &bindings) {
                        continue;
                    }
                    if let Some(tp) = target {
                        if !matches_inline_props(dst, &tp.properties, &bindings) {
                            continue;
                        }
                    }
                    if !edge_matches_inline_props(&edge, &rp.properties, &bindings) {
                        continue;
                    }
                    if let Some(condition) = predicate {
                        if !eval(condition, &bindings).truthy() {
                            continue;
                        }
                    }

                    if seen_edges.insert(edge.id.clone()) {
                        outcome.edges.push(edge.clone());
                    }
                    outcome.push_node(seen_nodes, (*src).clone());
                    outcome.push_node(seen_nodes, (*dst).clone());
                    break;
                }
            }
        }
        Pattern::Path(path) => exec_pattern(
            gdb,
            &path.pattern,
            predicate,
            outcome,
            seen_nodes,
            seen_edges,
        ),
        Pattern::Hyperedge(_) => outcome
            .unsupported
            .push("hyperedge patterns in MATCH are not supported".to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ruvector_graph::cypher::parse_cypher;
    use ruvector_graph::cypher::Statement;
    use ruvector_graph::{EdgeBuilder, NodeBuilder, PropertyValue};

    /// Two people and one `knows` edge between them.
    fn fixture() -> GraphDB {
        let gdb = GraphDB::new();
        for (id, name, age) in [("n1", "alice", 30i64), ("n2", "bob", 41)] {
            gdb.create_node(
                NodeBuilder::new()
                    .id(id)
                    .label("Person")
                    .property("name", PropertyValue::String(name.to_string()))
                    .property("age", PropertyValue::Integer(age))
                    .build(),
            )
            .expect("create node");
        }
        gdb.create_node(NodeBuilder::new().id("c1").label("Company").build())
            .expect("create company");
        gdb.create_edge(
            EdgeBuilder::new("n1".to_string(), "n2".to_string(), "knows")
                .id("e1")
                .property("since", PropertyValue::Integer(2020))
                .build(),
        )
        .expect("create edge");
        gdb
    }

    fn run(gdb: &GraphDB, cypher: &str) -> ExecOutcome {
        let parsed = parse_cypher(cypher).expect("parse");
        let clause = parsed
            .statements
            .iter()
            .find_map(|s| match s {
                Statement::Match(m) => Some(m),
                _ => None,
            })
            .expect("a MATCH statement");
        execute_match(gdb, clause)
    }

    /// The headline defect in #879: the label-less pattern returned nothing.
    #[test]
    fn label_less_match_returns_every_node() {
        let gdb = fixture();
        let out = run(&gdb, "MATCH (n) RETURN n");
        assert_eq!(out.nodes.len(), 3, "MATCH (n) must see all 3 nodes");
        assert!(out.unsupported.is_empty());
    }

    /// The point-lookup shape #879 calls "the standard query shape".
    #[test]
    fn where_filters_on_identity() {
        let gdb = fixture();
        let out = run(&gdb, "MATCH (n) WHERE n.id = 'n1' RETURN n");
        assert_eq!(out.nodes.len(), 1);
        assert_eq!(out.nodes[0].id, "n1");
    }

    #[test]
    fn where_filters_on_stored_property() {
        let gdb = fixture();
        let out = run(&gdb, "MATCH (n) WHERE n.name = 'bob' RETURN n");
        assert_eq!(out.nodes.len(), 1);
        assert_eq!(out.nodes[0].id, "n2");
    }

    #[test]
    fn where_numeric_comparison_and_conjunction() {
        let gdb = fixture();
        let out = run(&gdb, "MATCH (n:Person) WHERE n.age > 35 RETURN n");
        assert_eq!(out.nodes.len(), 1);
        assert_eq!(out.nodes[0].id, "n2");

        let out = run(
            &gdb,
            "MATCH (n:Person) WHERE n.age > 20 AND n.age < 35 RETURN n",
        );
        assert_eq!(out.nodes.len(), 1);
        assert_eq!(out.nodes[0].id, "n1");
    }

    /// A label filter still works — the one shape that worked before must not regress.
    #[test]
    fn label_scan_still_works() {
        let gdb = fixture();
        let out = run(&gdb, "MATCH (n:Person) RETURN n");
        assert_eq!(out.nodes.len(), 2);
        let out = run(&gdb, "MATCH (n:Company) RETURN n");
        assert_eq!(out.nodes.len(), 1);
        assert_eq!(out.nodes[0].id, "c1");
    }

    /// `result_edges` was declared and never written to, for any query shape.
    #[test]
    fn relationship_pattern_returns_edges() {
        let gdb = fixture();
        let out = run(&gdb, "MATCH (a)-[r:knows]->(b) RETURN a, r, b");
        assert_eq!(out.edges.len(), 1, "the knows edge must be returned");
        assert_eq!(out.edges[0].id, "e1");
        assert_eq!(out.edges[0].from, "n1");
        assert_eq!(out.edges[0].to, "n2");
        // Both endpoints come back with it.
        let mut ids: Vec<_> = out.nodes.iter().map(|n| n.id.as_str()).collect();
        ids.sort();
        assert_eq!(ids, vec!["n1", "n2"]);
    }

    #[test]
    fn untyped_relationship_pattern_scans_all_edges() {
        let gdb = fixture();
        let out = run(&gdb, "MATCH (a)-[r]->(b) RETURN r");
        assert_eq!(out.edges.len(), 1);
    }

    #[test]
    fn relationship_direction_is_honoured() {
        let gdb = fixture();
        // n2 has no outgoing edge, so anchoring the source on a Company label
        // must yield nothing rather than matching either endpoint.
        let out = run(&gdb, "MATCH (a:Company)-[r:knows]->(b) RETURN r");
        assert!(out.edges.is_empty());
    }

    #[test]
    fn inline_property_pattern_filters() {
        let gdb = fixture();
        let out = run(&gdb, "MATCH (n {name: 'alice'}) RETURN n");
        assert_eq!(out.nodes.len(), 1);
        assert_eq!(out.nodes[0].id, "n1");
    }

    #[test]
    fn inequality_operator() {
        let gdb = fixture();
        let out = run(&gdb, "MATCH (n:Person) WHERE n.age <> 30 RETURN n");
        assert_eq!(out.nodes.len(), 1);
        assert_eq!(out.nodes[0].id, "n2");
    }

    /// A missing property compares false rather than panicking or matching.
    #[test]
    fn absent_property_never_matches() {
        let gdb = fixture();
        let out = run(&gdb, "MATCH (n) WHERE n.nonexistent = 'x' RETURN n");
        assert!(out.nodes.is_empty());
    }

    /// NaN must not panic the comparison path.
    #[test]
    fn nan_comparison_is_false_not_a_panic() {
        let gdb = GraphDB::new();
        gdb.create_node(
            NodeBuilder::new()
                .id("nan")
                .property("score", PropertyValue::Float(f64::NAN))
                .build(),
        )
        .expect("create node");
        let out = run(&gdb, "MATCH (n) WHERE n.score > 0 RETURN n");
        assert!(out.nodes.is_empty());
        let out = run(&gdb, "MATCH (n) WHERE n.score = 0 RETURN n");
        assert!(out.nodes.is_empty());
    }

    /// Unsupported constructs must announce themselves. Returning an empty set
    /// is what let #879 hide for three releases.
    #[test]
    fn variable_length_relationship_is_reported_unsupported() {
        let gdb = fixture();
        let out = run(&gdb, "MATCH (a)-[r*1..2]->(b) RETURN r");
        assert!(
            out.unsupported
                .iter()
                .any(|u| u.contains("variable-length")),
            "expected a variable-length notice, got {:?}",
            out.unsupported
        );
    }

    #[test]
    fn chained_relationship_is_reported_unsupported() {
        let gdb = fixture();
        let out = run(&gdb, "MATCH (a)-[r:knows]->(b)<-[s:knows]-(c) RETURN r");
        assert!(
            out.unsupported.iter().any(|u| u.contains("chained")),
            "expected a chained-pattern notice, got {:?}",
            out.unsupported
        );
    }
}
