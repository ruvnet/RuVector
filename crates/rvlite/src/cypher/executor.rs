//! Cypher query executor for in-memory property graph
//!
//! Fixed: MATCH now correctly returns multiple rows (Issue #269).
//! The executor uses a ResultSet (Vec<ExecutionContext>) pipeline where each
//! clause transforms the set of row contexts, preserving all matched results.

use super::ast::*;
use super::graph_store::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ExecutionError {
    #[error("Graph error: {0}")]
    GraphError(#[from] GraphError),
    #[error("Variable not found: {0}")]
    VariableNotFound(String),
    #[error("Type error: {0}")]
    TypeError(String),
    #[error("Unsupported operation: {0}")]
    UnsupportedOperation(String),
    #[error("Execution error: {0}")]
    ExecutionError(String),
}

/// Execution context holding variable bindings for a single row
#[derive(Debug, Clone)]
pub struct ExecutionContext {
    pub variables: HashMap<String, ContextValue>,
}

impl ExecutionContext {
    pub fn new() -> Self {
        Self {
            variables: HashMap::new(),
        }
    }

    pub fn bind(&mut self, name: String, value: ContextValue) {
        self.variables.insert(name, value);
    }

    pub fn get(&self, name: &str) -> Option<&ContextValue> {
        self.variables.get(name)
    }
}

impl Default for ExecutionContext {
    fn default() -> Self {
        Self::new()
    }
}

/// Value in execution context (node, edge, or property value)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ContextValue {
    Node(Node),
    Edge(Edge),
    Value(Value),
    List(Vec<ContextValue>),
    Map(HashMap<String, ContextValue>),
}

impl ContextValue {
    pub fn as_node(&self) -> Option<&Node> {
        match self {
            ContextValue::Node(n) => Some(n),
            _ => None,
        }
    }

    pub fn as_value(&self) -> Option<&Value> {
        match self {
            ContextValue::Value(v) => Some(v),
            _ => None,
        }
    }
}

/// Query execution result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionResult {
    pub columns: Vec<String>,
    pub rows: Vec<HashMap<String, ContextValue>>,
}

impl ExecutionResult {
    pub fn new(columns: Vec<String>) -> Self {
        Self {
            columns,
            rows: Vec::new(),
        }
    }

    pub fn add_row(&mut self, row: HashMap<String, ContextValue>) {
        self.rows.push(row);
    }
}

/// A set of row contexts flowing through the execution pipeline.
/// Each clause transforms Vec<ExecutionContext> → Vec<ExecutionContext>.
type ResultSet = Vec<ExecutionContext>;

/// Cypher query executor
pub struct Executor<'a> {
    graph: &'a mut PropertyGraph,
}

impl<'a> Executor<'a> {
    pub fn new(graph: &'a mut PropertyGraph) -> Self {
        Self { graph }
    }

    /// Execute a parsed Cypher query
    pub fn execute(&mut self, query: &Query) -> Result<ExecutionResult, ExecutionError> {
        // Start with a single empty context (one row with no bindings).
        // Each statement transforms this ResultSet.
        let mut result_set: ResultSet = vec![ExecutionContext::new()];
        let mut final_result = None;

        for statement in &query.statements {
            final_result = Some(self.execute_statement(statement, &mut result_set)?);
        }

        final_result
            .ok_or_else(|| ExecutionError::ExecutionError("No statements to execute".to_string()))
    }

    fn execute_statement(
        &mut self,
        statement: &Statement,
        result_set: &mut ResultSet,
    ) -> Result<ExecutionResult, ExecutionError> {
        match statement {
            Statement::Create(clause) => self.execute_create(clause, result_set),
            Statement::Match(clause) => self.execute_match(clause, result_set),
            Statement::Return(clause) => self.execute_return(clause, result_set),
            Statement::Set(clause) => self.execute_set(clause, result_set),
            Statement::Delete(clause) => self.execute_delete(clause, result_set),
            _ => Err(ExecutionError::UnsupportedOperation(format!(
                "Statement {:?} not yet implemented",
                statement
            ))),
        }
    }

    fn execute_create(
        &mut self,
        clause: &CreateClause,
        result_set: &mut ResultSet,
    ) -> Result<ExecutionResult, ExecutionError> {
        // CREATE applies to the first context (or a new one if empty)
        if result_set.is_empty() {
            result_set.push(ExecutionContext::new());
        }
        let context = &mut result_set[0];

        for pattern in &clause.patterns {
            self.create_pattern(pattern, context)?;
        }

        Ok(ExecutionResult::new(vec![]))
    }

    fn create_pattern(
        &mut self,
        pattern: &Pattern,
        context: &mut ExecutionContext,
    ) -> Result<(), ExecutionError> {
        match pattern {
            Pattern::Node(node_pattern) => {
                let node = self.create_node(node_pattern)?;
                if let Some(var) = &node_pattern.variable {
                    context.bind(var.clone(), ContextValue::Node(node));
                }
                Ok(())
            }
            Pattern::Relationship(rel_pattern) => {
                self.create_relationship(rel_pattern, context)?;
                Ok(())
            }
            _ => Err(ExecutionError::UnsupportedOperation(
                "Only simple node and relationship patterns supported in CREATE".to_string(),
            )),
        }
    }

    fn create_node(&mut self, pattern: &NodePattern) -> Result<Node, ExecutionError> {
        let id = self.graph.generate_node_id();
        let mut node = Node::new(id).with_labels(pattern.labels.clone());

        // Set properties
        if let Some(props) = &pattern.properties {
            for (key, expr) in props {
                let value = self.evaluate_expression(expr, &ExecutionContext::new())?;
                node.set_property(key.clone(), value);
            }
        }

        let node_id = self.graph.add_node(node.clone());
        node.id = node_id;
        Ok(node)
    }

    fn create_relationship(
        &mut self,
        pattern: &RelationshipPattern,
        context: &mut ExecutionContext,
    ) -> Result<(), ExecutionError> {
        // Get or create source node
        let from_node = if let Some(var) = &pattern.from.variable {
            if let Some(ContextValue::Node(n)) = context.get(var) {
                n.clone()
            } else {
                self.create_node(&pattern.from)?
            }
        } else {
            self.create_node(&pattern.from)?
        };

        // Get or create target node (only handle simple node targets for now)
        let to_node = match &*pattern.to {
            Pattern::Node(node_pattern) => {
                if let Some(var) = &node_pattern.variable {
                    if let Some(ContextValue::Node(n)) = context.get(var) {
                        n.clone()
                    } else {
                        self.create_node(node_pattern)?
                    }
                } else {
                    self.create_node(node_pattern)?
                }
            }
            _ => {
                return Err(ExecutionError::UnsupportedOperation(
                    "Complex relationship targets not yet supported".to_string(),
                ))
            }
        };

        // Create the edge
        let edge_type = pattern
            .rel_type
            .clone()
            .unwrap_or_else(|| "RELATED_TO".to_string());
        let edge_id = self.graph.generate_edge_id();
        let mut edge = Edge::new(edge_id, from_node.id.clone(), to_node.id.clone(), edge_type);

        // Set properties
        if let Some(props) = &pattern.properties {
            for (key, expr) in props {
                let value = self.evaluate_expression(expr, context)?;
                edge.set_property(key.clone(), value);
            }
        }

        let edge_id = self.graph.add_edge(edge.clone())?;
        if let Some(var) = &pattern.variable {
            edge.id = edge_id;
            context.bind(var.clone(), ContextValue::Edge(edge));
        }

        Ok(())
    }

    /// Execute MATCH: find all matching patterns and expand the result set.
    ///
    /// For each existing row context, MATCH finds all matching nodes/relationships
    /// and produces a new row context for each match. This correctly handles
    /// multiple results (fixes Issue #269).
    fn execute_match(
        &mut self,
        clause: &MatchClause,
        result_set: &mut ResultSet,
    ) -> Result<ExecutionResult, ExecutionError> {
        let mut new_result_set = Vec::new();

        // For each existing context row, expand with matches
        for existing_ctx in result_set.iter() {
            let mut matches = Vec::new();

            for pattern in &clause.patterns {
                let pattern_matches = self.match_pattern(pattern)?;
                if matches.is_empty() {
                    // First pattern: each match becomes a new context
                    for m in pattern_matches {
                        let mut ctx = existing_ctx.clone();
                        for (var, val) in m.variables {
                            ctx.bind(var, val);
                        }
                        matches.push(ctx);
                    }
                } else {
                    // Subsequent patterns: cross-product with existing matches
                    let mut cross = Vec::new();
                    for prev in &matches {
                        for m in &pattern_matches {
                            let mut ctx = prev.clone();
                            for (var, val) in &m.variables {
                                ctx.bind(var.clone(), val.clone());
                            }
                            cross.push(ctx);
                        }
                    }
                    matches = cross;
                }
            }

            // Apply WHERE filter if present
            if let Some(where_clause) = &clause.where_clause {
                matches.retain(|ctx| {
                    self.evaluate_condition(&where_clause.condition, ctx)
                        .unwrap_or(false)
                });
            }

            new_result_set.extend(matches);
        }

        *result_set = new_result_set;
        Ok(ExecutionResult::new(vec![]))
    }

    fn match_pattern(&self, pattern: &Pattern) -> Result<Vec<ExecutionContext>, ExecutionError> {
        match pattern {
            Pattern::Node(node_pattern) => self.match_node_pattern(node_pattern),
            Pattern::Relationship(rel_pattern) => self.match_relationship_pattern(rel_pattern),
            _ => Err(ExecutionError::UnsupportedOperation(
                "Pattern type not yet supported in MATCH".to_string(),
            )),
        }
    }

    fn match_node_pattern(
        &self,
        pattern: &NodePattern,
    ) -> Result<Vec<ExecutionContext>, ExecutionError> {
        let mut contexts = Vec::new();

        // Find nodes matching labels
        let candidates: Vec<&Node> = if pattern.labels.is_empty() {
            self.graph.find_nodes(|_| true)
        } else {
            let mut nodes = Vec::new();
            for label in &pattern.labels {
                nodes.extend(self.graph.find_nodes_by_label(label));
            }
            nodes
        };

        // Filter by properties
        for node in candidates {
            if let Some(props) = &pattern.properties {
                let mut matches = true;
                for (key, expr) in props {
                    let expected_value =
                        self.evaluate_expression(expr, &ExecutionContext::new())?;
                    if node.get_property(key) != Some(&expected_value) {
                        matches = false;
                        break;
                    }
                }
                if !matches {
                    continue;
                }
            }

            let mut ctx = ExecutionContext::new();
            if let Some(var) = &pattern.variable {
                ctx.bind(var.clone(), ContextValue::Node(node.clone()));
            }
            contexts.push(ctx);
        }

        Ok(contexts)
    }

    fn match_relationship_pattern(
        &self,
        pattern: &RelationshipPattern,
    ) -> Result<Vec<ExecutionContext>, ExecutionError> {
        let mut contexts = Vec::new();

        // Match source nodes
        let from_contexts = self.match_node_pattern(&pattern.from)?;

        for from_ctx in from_contexts {
            // Get the source node
            let from_node = if let Some(var) = &pattern.from.variable {
                from_ctx
                    .get(var)
                    .and_then(|v| v.as_node())
                    .ok_or_else(|| ExecutionError::VariableNotFound(var.clone()))?
            } else {
                continue;
            };

            // Find matching edges
            let edges = match pattern.direction {
                Direction::Outgoing => self.graph.get_outgoing_edges(&from_node.id),
                Direction::Incoming => self.graph.get_incoming_edges(&from_node.id),
                Direction::Undirected => {
                    let mut all = self.graph.get_outgoing_edges(&from_node.id);
                    all.extend(self.graph.get_incoming_edges(&from_node.id));
                    all
                }
            };

            for edge in edges {
                // Filter by type
                if let Some(rel_type) = &pattern.rel_type {
                    if &edge.edge_type != rel_type {
                        continue;
                    }
                }

                // Filter by properties
                if let Some(props) = &pattern.properties {
                    let mut matches = true;
                    for (key, expr) in props {
                        let expected_value =
                            self.evaluate_expression(expr, &ExecutionContext::new())?;
                        if edge.get_property(key) != Some(&expected_value) {
                            matches = false;
                            break;
                        }
                    }
                    if !matches {
                        continue;
                    }
                }

                // Get target node
                let to_node_id = if pattern.direction == Direction::Incoming {
                    &edge.from
                } else {
                    &edge.to
                };

                if let Some(to_node) = self.graph.get_node(to_node_id) {
                    let mut ctx = from_ctx.clone();
                    if let Some(var) = &pattern.variable {
                        ctx.bind(var.clone(), ContextValue::Edge(edge.clone()));
                    }

                    // Bind target node if it's a simple node pattern
                    if let Pattern::Node(to_pattern) = &*pattern.to {
                        if let Some(var) = &to_pattern.variable {
                            ctx.bind(var.clone(), ContextValue::Node(to_node.clone()));
                        }
                    }

                    contexts.push(ctx);
                }
            }
        }

        Ok(contexts)
    }

    /// Execute RETURN: project columns from each row context.
    ///
    /// Produces one output row per context in the result set (fixes Issue #269).
    fn execute_return(
        &self,
        clause: &ReturnClause,
        result_set: &ResultSet,
    ) -> Result<ExecutionResult, ExecutionError> {
        let mut columns = Vec::new();

        // Determine column names from the first item
        for item in &clause.items {
            let col_name = item
                .alias
                .clone()
                .unwrap_or_else(|| match &item.expression {
                    Expression::Variable(var) => var.clone(),
                    Expression::Property { object, property } => {
                        if let Expression::Variable(var) = &**object {
                            format!("{}.{}", var, property)
                        } else {
                            "?column?".to_string()
                        }
                    }
                    _ => "?column?".to_string(),
                });
            columns.push(col_name);
        }

        let mut result = ExecutionResult::new(columns.clone());

        // Produce one row per context
        for context in result_set {
            let mut row = HashMap::new();

            for (i, item) in clause.items.iter().enumerate() {
                let col_name = &columns[i];
                let value = self.evaluate_expression_ctx(&item.expression, context)?;
                row.insert(col_name.clone(), value);
            }

            result.add_row(row);
        }

        Ok(result)
    }

    /// Execute SET: apply property updates to all rows in the result set.
    fn execute_set(
        &mut self,
        clause: &SetClause,
        result_set: &ResultSet,
    ) -> Result<ExecutionResult, ExecutionError> {
        for context in result_set {
            for item in &clause.items {
                match item {
                    SetItem::Property {
                        variable,
                        property,
                        value,
                    } => {
                        let val = self.evaluate_expression(value, context)?;
                        if let Some(ContextValue::Node(node)) = context.get(variable) {
                            if let Some(node_mut) = self.graph.get_node_mut(&node.id) {
                                node_mut.set_property(property.clone(), val);
                            }
                        }
                    }
                    _ => {
                        return Err(ExecutionError::UnsupportedOperation(
                            "Only property SET supported".to_string(),
                        ))
                    }
                }
            }
        }

        Ok(ExecutionResult::new(vec![]))
    }

    /// Execute DELETE: remove nodes/edges for all rows in the result set.
    fn execute_delete(
        &mut self,
        clause: &DeleteClause,
        result_set: &ResultSet,
    ) -> Result<ExecutionResult, ExecutionError> {
        for context in result_set {
            for expr in &clause.expressions {
                if let Expression::Variable(var) = expr {
                    if let Some(ctx_val) = context.get(var) {
                        match ctx_val {
                            ContextValue::Node(node) => {
                                if clause.detach {
                                    // Ignore errors for already-deleted nodes
                                    let _ = self.graph.delete_node(&node.id);
                                } else {
                                    return Err(ExecutionError::ExecutionError(
                                        "Cannot delete node with relationships without DETACH"
                                            .to_string(),
                                    ));
                                }
                            }
                            ContextValue::Edge(edge) => {
                                let _ = self.graph.delete_edge(&edge.id);
                            }
                            _ => {}
                        }
                    }
                }
            }
        }

        Ok(ExecutionResult::new(vec![]))
    }

    fn evaluate_expression(
        &self,
        expr: &Expression,
        context: &ExecutionContext,
    ) -> Result<Value, ExecutionError> {
        match expr {
            Expression::Integer(n) => Ok(Value::Integer(*n)),
            Expression::Float(f) => Ok(Value::Float(*f)),
            Expression::String(s) => Ok(Value::String(s.clone())),
            Expression::Boolean(b) => Ok(Value::Boolean(*b)),
            Expression::Null => Ok(Value::Null),
            Expression::Variable(var) => {
                if let Some(ContextValue::Value(v)) = context.get(var) {
                    Ok(v.clone())
                } else {
                    Err(ExecutionError::VariableNotFound(var.clone()))
                }
            }
            Expression::Property { object, property } => {
                if let Expression::Variable(var) = &**object {
                    if let Some(ContextValue::Node(node)) = context.get(var) {
                        Ok(node.get_property(property).cloned().unwrap_or(Value::Null))
                    } else {
                        Err(ExecutionError::VariableNotFound(var.clone()))
                    }
                } else {
                    Err(ExecutionError::UnsupportedOperation(
                        "Nested property access not supported".to_string(),
                    ))
                }
            }
            _ => Err(ExecutionError::UnsupportedOperation(format!(
                "Expression {:?} not yet implemented",
                expr
            ))),
        }
    }

    fn evaluate_expression_ctx(
        &self,
        expr: &Expression,
        context: &ExecutionContext,
    ) -> Result<ContextValue, ExecutionError> {
        match expr {
            Expression::Variable(var) => context
                .get(var)
                .cloned()
                .ok_or_else(|| ExecutionError::VariableNotFound(var.clone())),
            Expression::Property { object, property } => {
                if let Expression::Variable(var) = &**object {
                    if let Some(ContextValue::Node(node)) = context.get(var) {
                        Ok(ContextValue::Value(
                            node.get_property(property).cloned().unwrap_or(Value::Null),
                        ))
                    } else {
                        Err(ExecutionError::VariableNotFound(var.clone()))
                    }
                } else {
                    Err(ExecutionError::UnsupportedOperation(
                        "Nested property access not supported".to_string(),
                    ))
                }
            }
            _ => {
                let val = self.evaluate_expression(expr, context)?;
                Ok(ContextValue::Value(val))
            }
        }
    }

    fn evaluate_condition(
        &self,
        expr: &Expression,
        context: &ExecutionContext,
    ) -> Result<bool, ExecutionError> {
        match expr {
            Expression::Boolean(b) => Ok(*b),
            Expression::BinaryOp { left, op, right } => {
                let left_val = self.evaluate_expression(left, context)?;
                let right_val = self.evaluate_expression(right, context)?;

                match op {
                    BinaryOperator::Equal => Ok(left_val == right_val),
                    BinaryOperator::NotEqual => Ok(left_val != right_val),
                    BinaryOperator::GreaterThan => {
                        if let (Some(l), Some(r)) = (left_val.as_i64(), right_val.as_i64()) {
                            Ok(l > r)
                        } else {
                            Ok(false)
                        }
                    }
                    BinaryOperator::LessThan => {
                        if let (Some(l), Some(r)) = (left_val.as_i64(), right_val.as_i64()) {
                            Ok(l < r)
                        } else {
                            Ok(false)
                        }
                    }
                    _ => Err(ExecutionError::UnsupportedOperation(format!(
                        "Operator {:?} not implemented",
                        op
                    ))),
                }
            }
            _ => Err(ExecutionError::UnsupportedOperation(
                "Complex conditions not yet supported".to_string(),
            )),
        }
    }
}
