// Cypher query executor

use super::ast::*;
use crate::graph::storage::GraphStore;
use serde_json::{json, Value as JsonValue};
use std::collections::HashMap;

// Direction is re-exported from ast::*

/// Execute a parsed Cypher query
pub fn execute_cypher(
    graph: &GraphStore,
    query: &CypherQuery,
    params: Option<&JsonValue>,
) -> Result<JsonValue, String> {
    let mut context = ExecutionContext::new(params);

    for clause in &query.clauses {
        match clause {
            Clause::Match(m) => execute_match(graph, m, &mut context)?,
            Clause::Create(c) => execute_create(graph, c, &mut context)?,
            Clause::Return(r) => return execute_return(graph, r, &context),
            Clause::Where(w) => execute_where(graph, w, &mut context)?,
            Clause::Set(s) => execute_set(graph, s, &mut context)?,
            Clause::Delete(d) => execute_delete(graph, d, &mut context)?,
            Clause::With(w) => execute_with(graph, w, &mut context)?,
        }
    }

    // If no RETURN clause, return empty result
    Ok(json!([]))
}

/// Execution context holding variable bindings
struct ExecutionContext<'a> {
    bindings: Vec<HashMap<String, Binding>>,
    params: Option<&'a JsonValue>,
}

impl<'a> ExecutionContext<'a> {
    fn new(params: Option<&'a JsonValue>) -> Self {
        Self {
            bindings: vec![HashMap::new()],
            params,
        }
    }

    fn bind(&mut self, var: &str, binding: Binding) {
        if let Some(last) = self.bindings.last_mut() {
            last.insert(var.to_string(), binding);
        }
    }

    fn get(&self, var: &str) -> Option<&Binding> {
        for bindings in self.bindings.iter().rev() {
            if let Some(binding) = bindings.get(var) {
                return Some(binding);
            }
        }
        None
    }

    fn get_param(&self, name: &str) -> Option<&JsonValue> {
        self.params.and_then(|p| p.get(name))
    }

    fn push_scope(&mut self) {
        self.bindings.push(HashMap::new());
    }

    fn pop_scope(&mut self) {
        self.bindings.pop();
    }
}

#[derive(Debug, Clone)]
enum Binding {
    Node(u64),
    Edge(u64),
    Value(JsonValue),
}

fn execute_match(
    graph: &GraphStore,
    match_clause: &MatchClause,
    context: &mut ExecutionContext,
) -> Result<(), String> {
    for pattern in &match_clause.patterns {
        match_pattern(graph, pattern, context)?;
    }
    Ok(())
}

fn match_pattern(
    graph: &GraphStore,
    pattern: &Pattern,
    context: &mut ExecutionContext,
) -> Result<(), String> {
    // Collect the pattern as alternating nodes and relationships:
    // (a:Person)-[:KNOWS]->(b:Person) = [Node(a), Rel(KNOWS), Node(b)]
    let mut node_patterns: Vec<&NodePattern> = Vec::new();
    let mut rel_patterns: Vec<&RelationshipPattern> = Vec::new();

    for element in &pattern.elements {
        match element {
            PatternElement::Node(np) => node_patterns.push(np),
            PatternElement::Relationship(rp) => rel_patterns.push(rp),
        }
    }

    // Case 1: Single node pattern — find all matching nodes
    if rel_patterns.is_empty() {
        if let Some(np) = node_patterns.first() {
            let candidates = find_matching_nodes(graph, np);
            if candidates.is_empty() {
                return Ok(());
            }
            // Create a binding row per matching node
            let mut rows: Vec<HashMap<String, Binding>> = Vec::new();
            for node in &candidates {
                let mut row = HashMap::new();
                if let Some(var) = &np.variable {
                    row.insert(var.clone(), Binding::Node(node.id));
                }
                rows.push(row);
            }
            context.bindings = rows;
            return Ok(());
        }
        return Ok(());
    }

    // Case 2: Pattern with relationships — traverse edges
    // For each relationship pattern, we need pairs of (source_node, target_node)
    // Pattern: (a)-[r:TYPE]->(b) means: find all edges of TYPE,
    // check source matches a's pattern and target matches b's pattern
    let mut result_rows: Vec<HashMap<String, Binding>> = Vec::new();

    // We process node-rel-node triples
    for i in 0..rel_patterns.len() {
        let src_pattern = node_patterns.get(i);
        let dst_pattern = node_patterns.get(i + 1);
        let rel_pattern = rel_patterns[i];

        // Get candidate edges by type
        let edges = if let Some(ref rel_type) = rel_pattern.rel_type {
            graph.edges.find_by_type(rel_type)
        } else {
            graph.edges.all_edges()
        };

        for edge in &edges {
            let (src_id, dst_id) = match rel_pattern.direction {
                Direction::Outgoing | Direction::Both => (edge.source, edge.target),
                Direction::Incoming => (edge.target, edge.source),
            };

            // Check source node matches pattern
            if let Some(sp) = src_pattern {
                if !node_matches_pattern(graph, src_id, sp) {
                    continue;
                }
            }

            // Check target node matches pattern
            if let Some(dp) = dst_pattern {
                if !node_matches_pattern(graph, dst_id, dp) {
                    continue;
                }
            }

            // Reject self-references when variables are different
            if let (Some(sp), Some(dp)) = (src_pattern, dst_pattern) {
                if let (Some(sv), Some(dv)) = (&sp.variable, &dp.variable) {
                    if sv != dv && src_id == dst_id {
                        continue;
                    }
                }
            }

            // Build binding row
            let mut row = HashMap::new();
            if let Some(sp) = src_pattern {
                if let Some(var) = &sp.variable {
                    row.insert(var.clone(), Binding::Node(src_id));
                }
            }
            if let Some(dp) = dst_pattern {
                if let Some(var) = &dp.variable {
                    row.insert(var.clone(), Binding::Node(dst_id));
                }
            }
            if let Some(var) = &rel_pattern.variable {
                row.insert(var.clone(), Binding::Edge(edge.id));
            }

            result_rows.push(row);

            // Also match the reverse direction for Both
            if rel_pattern.direction == Direction::Both && edge.source != edge.target {
                let mut rev_row = HashMap::new();
                if let Some(sp) = src_pattern {
                    if let Some(var) = &sp.variable {
                        rev_row.insert(var.clone(), Binding::Node(edge.target));
                    }
                }
                if let Some(dp) = dst_pattern {
                    if let Some(var) = &dp.variable {
                        rev_row.insert(var.clone(), Binding::Node(edge.source));
                    }
                }
                if let Some(var) = &rel_pattern.variable {
                    rev_row.insert(var.clone(), Binding::Edge(edge.id));
                }
                if let (Some(sp), Some(dp)) = (src_pattern, dst_pattern) {
                    if node_matches_pattern(graph, edge.target, sp)
                        && node_matches_pattern(graph, edge.source, dp)
                    {
                        result_rows.push(rev_row);
                    }
                }
            }
        }
    }

    if !result_rows.is_empty() {
        context.bindings = result_rows;
    }

    Ok(())
}

/// Find all nodes matching a node pattern (labels + properties)
fn find_matching_nodes(
    graph: &GraphStore,
    pattern: &NodePattern,
) -> Vec<crate::graph::storage::Node> {
    let candidates = if pattern.labels.is_empty() {
        graph.nodes.all_nodes()
    } else {
        graph.nodes.find_by_label(&pattern.labels[0])
    };

    candidates
        .into_iter()
        .filter(|node| {
            // Check all labels
            if !pattern.labels.iter().all(|l| node.has_label(l)) {
                return false;
            }
            // Check properties
            pattern.properties.iter().all(|(key, expr)| {
                if let Some(node_value) = node.get_property(key) {
                    if let Expression::Literal(expected) = expr {
                        node_value == expected
                    } else {
                        false
                    }
                } else {
                    false
                }
            })
        })
        .collect()
}

/// Check if a specific node ID matches a node pattern
fn node_matches_pattern(graph: &GraphStore, node_id: u64, pattern: &NodePattern) -> bool {
    if let Some(node) = graph.nodes.get(node_id) {
        // Check labels
        if !pattern.labels.iter().all(|l| node.has_label(l)) {
            return false;
        }
        // Check properties
        pattern.properties.iter().all(|(key, expr)| {
            if let Some(node_value) = node.get_property(key) {
                if let Expression::Literal(expected) = expr {
                    node_value == expected
                } else {
                    false
                }
            } else {
                false
            }
        })
    } else {
        false
    }
}

fn execute_create(
    graph: &GraphStore,
    create_clause: &CreateClause,
    context: &mut ExecutionContext,
) -> Result<(), String> {
    for pattern in &create_clause.patterns {
        create_pattern(graph, pattern, context)?;
    }
    Ok(())
}

fn create_pattern(
    graph: &GraphStore,
    pattern: &Pattern,
    context: &mut ExecutionContext,
) -> Result<(), String> {
    let mut last_node_id: Option<u64> = None;

    for element in &pattern.elements {
        match element {
            PatternElement::Node(node_pattern) => {
                let node_id = create_node(graph, node_pattern, context)?;
                last_node_id = Some(node_id);

                if let Some(var) = &node_pattern.variable {
                    context.bind(var, Binding::Node(node_id));
                }
            }
            PatternElement::Relationship(rel_pattern) => {
                if let Some(source_id) = last_node_id {
                    // For CREATE, we need to get the target node from context or create it
                    // This is simplified - production code would handle more complex patterns
                    let edge_id = create_relationship(graph, rel_pattern, source_id, context)?;

                    if let Some(var) = &rel_pattern.variable {
                        context.bind(var, Binding::Edge(edge_id));
                    }
                }
            }
        }
    }

    Ok(())
}

fn create_node(
    graph: &GraphStore,
    pattern: &NodePattern,
    context: &ExecutionContext,
) -> Result<u64, String> {
    let mut properties = HashMap::new();

    for (key, expr) in &pattern.properties {
        let value = evaluate_expression(expr, context)?;
        properties.insert(key.clone(), value);
    }

    let node_id = graph.add_node(pattern.labels.clone(), properties);
    Ok(node_id)
}

fn create_relationship(
    graph: &GraphStore,
    pattern: &RelationshipPattern,
    source_id: u64,
    context: &ExecutionContext,
) -> Result<u64, String> {
    let mut properties = HashMap::new();

    for (key, expr) in &pattern.properties {
        let value = evaluate_expression(expr, context)?;
        properties.insert(key.clone(), value);
    }

    let edge_type = pattern
        .rel_type
        .clone()
        .unwrap_or_else(|| "RELATED".to_string());

    // Resolve target from the next node in the pattern (bound in context)
    // Look through bindings for any node binding that isn't the source
    let target_id = context
        .bindings
        .iter()
        .rev()
        .flat_map(|b| b.values())
        .find_map(|binding| match binding {
            Binding::Node(id) if *id != source_id => Some(*id),
            _ => None,
        })
        .unwrap_or(source_id);

    graph.add_edge(source_id, target_id, edge_type, properties)
}

fn execute_return(
    graph: &GraphStore,
    return_clause: &ReturnClause,
    context: &ExecutionContext,
) -> Result<JsonValue, String> {
    let mut results = Vec::new();

    // If no bindings, return empty
    if context.bindings.is_empty() || context.bindings[0].is_empty() {
        return Ok(json!([]));
    }

    // For each binding combination
    for bindings in &context.bindings {
        if bindings.is_empty() {
            continue;
        }

        let mut row = serde_json::Map::new();

        for item in &return_clause.items {
            let value = evaluate_return_item(graph, item, bindings)?;
            let key = item.alias.clone().unwrap_or_else(|| {
                // Generate key from expression
                match &item.expression {
                    Expression::Variable(v) => v.clone(),
                    Expression::Property(v, p) => format!("{}.{}", v, p),
                    _ => "result".to_string(),
                }
            });

            row.insert(key, value);
        }

        results.push(JsonValue::Object(row));
    }

    // Apply DISTINCT
    if return_clause.distinct {
        results.sort_by(|a, b| a.to_string().cmp(&b.to_string()));
        results.dedup();
    }

    // Apply SKIP
    if let Some(skip) = return_clause.skip {
        results = results.into_iter().skip(skip).collect();
    }

    // Apply LIMIT
    if let Some(limit) = return_clause.limit {
        results.truncate(limit);
    }

    Ok(JsonValue::Array(results))
}

fn evaluate_return_item(
    graph: &GraphStore,
    item: &ReturnItem,
    bindings: &HashMap<String, Binding>,
) -> Result<JsonValue, String> {
    match &item.expression {
        Expression::Variable(var) => {
            if let Some(binding) = bindings.get(var) {
                match binding {
                    Binding::Node(id) => {
                        if let Some(node) = graph.nodes.get(*id) {
                            Ok(serde_json::to_value(&node).unwrap())
                        } else {
                            Ok(JsonValue::Null)
                        }
                    }
                    Binding::Edge(id) => {
                        if let Some(edge) = graph.edges.get(*id) {
                            Ok(serde_json::to_value(&edge).unwrap())
                        } else {
                            Ok(JsonValue::Null)
                        }
                    }
                    Binding::Value(v) => Ok(v.clone()),
                }
            } else {
                Ok(JsonValue::Null)
            }
        }
        Expression::Property(var, prop) => {
            if let Some(Binding::Node(id)) = bindings.get(var) {
                if let Some(node) = graph.nodes.get(*id) {
                    Ok(node.get_property(prop).cloned().unwrap_or(JsonValue::Null))
                } else {
                    Ok(JsonValue::Null)
                }
            } else {
                Ok(JsonValue::Null)
            }
        }
        Expression::Literal(value) => Ok(value.clone()),
        _ => Err("Unsupported return expression".to_string()),
    }
}

fn execute_where(
    _graph: &GraphStore,
    where_clause: &WhereClause,
    context: &mut ExecutionContext,
) -> Result<(), String> {
    // Evaluate WHERE condition and filter bindings
    // Simplified implementation
    let result = evaluate_expression(&where_clause.condition, context)?;

    if !result.as_bool().unwrap_or(false) {
        // Clear bindings if condition is false
        if let Some(last) = context.bindings.last_mut() {
            last.clear();
        }
    }

    Ok(())
}

fn execute_set(
    _graph: &GraphStore,
    _set_clause: &SetClause,
    _context: &mut ExecutionContext,
) -> Result<(), String> {
    // Simplified SET implementation
    Ok(())
}

fn execute_delete(
    _graph: &GraphStore,
    _delete_clause: &DeleteClause,
    _context: &mut ExecutionContext,
) -> Result<(), String> {
    // Simplified DELETE implementation
    Ok(())
}

fn execute_with(
    _graph: &GraphStore,
    _with_clause: &WithClause,
    _context: &mut ExecutionContext,
) -> Result<(), String> {
    // Simplified WITH implementation
    Ok(())
}

fn evaluate_expression(expr: &Expression, context: &ExecutionContext) -> Result<JsonValue, String> {
    match expr {
        Expression::Literal(value) => Ok(value.clone()),
        Expression::Variable(var) => {
            if let Some(binding) = context.get(var) {
                match binding {
                    Binding::Value(v) => Ok(v.clone()),
                    Binding::Node(id) => Ok(json!({ "id": id })),
                    Binding::Edge(id) => Ok(json!({ "id": id })),
                }
            } else {
                Ok(JsonValue::Null)
            }
        }
        Expression::Parameter(name) => {
            Ok(context.get_param(name).cloned().unwrap_or(JsonValue::Null))
        }
        Expression::BinaryOp(left, op, right) => {
            let left_val = evaluate_expression(left, context)?;
            let right_val = evaluate_expression(right, context)?;

            match op {
                BinaryOperator::Eq => Ok(json!(left_val == right_val)),
                BinaryOperator::Neq => Ok(json!(left_val != right_val)),
                BinaryOperator::Lt => {
                    if let (Some(l), Some(r)) = (left_val.as_f64(), right_val.as_f64()) {
                        Ok(json!(l < r))
                    } else {
                        Ok(json!(false))
                    }
                }
                BinaryOperator::Gt => {
                    if let (Some(l), Some(r)) = (left_val.as_f64(), right_val.as_f64()) {
                        Ok(json!(l > r))
                    } else {
                        Ok(json!(false))
                    }
                }
                _ => Err(format!("Unsupported binary operator: {:?}", op)),
            }
        }
        _ => Err("Unsupported expression type".to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_execute_create() {
        let graph = GraphStore::new();

        let pattern = Pattern::new().with_element(PatternElement::Node(
            NodePattern::new()
                .with_variable("n")
                .with_label("Person")
                .with_property("name", Expression::literal("Alice")),
        ));

        let create = CreateClause::new(vec![pattern]);
        let query = CypherQuery::new()
            .with_clause(Clause::Create(create))
            .with_clause(Clause::Return(ReturnClause::new(vec![ReturnItem::new(
                Expression::variable("n"),
            )])));

        let result = execute_cypher(&graph, &query, None);
        assert!(result.is_ok());

        let json = result.unwrap();
        assert!(json.is_array());
    }

    #[test]
    fn test_execute_match() {
        let graph = GraphStore::new();

        // Create a node first
        graph.add_node(
            vec!["Person".to_string()],
            HashMap::from([("name".to_string(), "Alice".into())]),
        );

        let pattern = Pattern::new().with_element(PatternElement::Node(
            NodePattern::new().with_variable("n").with_label("Person"),
        ));

        let match_clause = MatchClause::new(vec![pattern]);
        let query = CypherQuery::new()
            .with_clause(Clause::Match(match_clause))
            .with_clause(Clause::Return(ReturnClause::new(vec![ReturnItem::new(
                Expression::property("n", "name"),
            )])));

        let result = execute_cypher(&graph, &query, None);
        assert!(result.is_ok());
    }
}
