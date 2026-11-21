//! Cypher query executor for in-memory property graph

use super::ast::*;
use super::graph_store::*;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
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

/// Execution context holding variable bindings
#[derive(Debug, Clone)]
pub struct ExecutionContext {
    pub variables: HashMap<String, ContextValue>,
    /// All matched binding sets from MATCH clauses; populated by execute_match, consumed by execute_return.
    /// Each entry is one complete set of bindings for one match row.
    pub matched_rows: Vec<HashMap<String, ContextValue>>,
}

impl ExecutionContext {
    pub fn new() -> Self {
        Self {
            variables: HashMap::new(),
            matched_rows: Vec::new(),
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

/// Cypher query executor
pub struct Executor<'a> {
    graph: &'a mut PropertyGraph,
    /// Set once a MATCH has run in the current query, so an empty `matched_rows`
    /// means "no rows" rather than "no MATCH".
    matched: bool,
}

impl<'a> Executor<'a> {
    pub fn new(graph: &'a mut PropertyGraph) -> Self {
        Self {
            graph,
            matched: false,
        }
    }

    /// Execute a parsed Cypher query
    pub fn execute(&mut self, query: &Query) -> Result<ExecutionResult, ExecutionError> {
        let mut context = ExecutionContext::new();
        let mut result = None;
        self.matched = false;

        for statement in &query.statements {
            result = Some(self.execute_statement(statement, &mut context)?);
        }

        result.ok_or_else(|| ExecutionError::ExecutionError("No statements to execute".to_string()))
    }

    fn execute_statement(
        &mut self,
        statement: &Statement,
        context: &mut ExecutionContext,
    ) -> Result<ExecutionResult, ExecutionError> {
        match statement {
            Statement::Create(clause) => self.execute_create(clause, context),
            Statement::Match(clause) => self.execute_match(clause, context),
            Statement::Return(clause) => self.execute_return(clause, context),
            Statement::Set(clause) => self.execute_set(clause, context),
            Statement::Delete(clause) => self.execute_delete(clause, context),
            _ => Err(ExecutionError::UnsupportedOperation(format!(
                "Statement {:?} not yet implemented",
                statement
            ))),
        }
    }

    fn execute_create(
        &mut self,
        clause: &CreateClause,
        context: &mut ExecutionContext,
    ) -> Result<ExecutionResult, ExecutionError> {
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

    fn execute_match(
        &mut self,
        clause: &MatchClause,
        context: &mut ExecutionContext,
    ) -> Result<ExecutionResult, ExecutionError> {
        // Comma-separated patterns combine as a cross product; a variable that
        // appears in more than one pattern must bind to the same element.
        let mut matches = vec![ExecutionContext::new()];

        for pattern in &clause.patterns {
            let pattern_matches = self.match_pattern(pattern)?;
            let mut combined = Vec::new();
            for left in &matches {
                for right in &pattern_matches {
                    if let Some(ctx) = merge_bindings(left, right) {
                        combined.push(ctx);
                    }
                }
            }
            matches = combined;
        }

        // Apply WHERE filter if present
        if let Some(where_clause) = &clause.where_clause {
            matches.retain(|ctx| {
                self.evaluate_condition(&where_clause.condition, ctx)
                    .unwrap_or(false)
            });
        }

        // Store each matched binding set as a separate row so RETURN can iterate all of them
        context.matched_rows = matches.into_iter().map(|ctx| ctx.variables).collect();
        self.matched = true;

        Ok(ExecutionResult::new(vec![]))
    }

    /// One context per matched row, each holding the outer bindings plus that
    /// row's bindings. Without a preceding MATCH this is the outer context alone.
    fn row_contexts(&self, context: &ExecutionContext) -> Vec<ExecutionContext> {
        if !self.matched {
            return vec![context.clone()];
        }
        context
            .matched_rows
            .iter()
            .map(|bindings| {
                let mut row_ctx = ExecutionContext::new();
                row_ctx.variables = context.variables.clone();
                for (k, v) in bindings {
                    row_ctx.bind(k.clone(), v.clone());
                }
                row_ctx
            })
            .collect()
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
        Ok(self
            .matching_nodes(pattern)?
            .into_iter()
            .map(|node| {
                let mut ctx = ExecutionContext::new();
                if let Some(var) = &pattern.variable {
                    ctx.bind(var.clone(), ContextValue::Node(node.clone()));
                }
                ctx
            })
            .collect())
    }

    fn matching_nodes(&self, pattern: &NodePattern) -> Result<Vec<&Node>, ExecutionError> {
        let candidates = if pattern.labels.is_empty() {
            self.graph.find_nodes(|_| true)
        } else {
            pattern
                .labels
                .iter()
                .flat_map(|label| self.graph.find_nodes_by_label(label))
                .collect()
        };

        let mut nodes = Vec::new();
        for node in candidates {
            if self.matches_properties(pattern.properties.as_ref(), &node.properties)? {
                nodes.push(node);
            }
        }
        Ok(nodes)
    }

    fn matches_properties(
        &self,
        pattern: Option<&PropertyMap>,
        properties: &HashMap<String, Value>,
    ) -> Result<bool, ExecutionError> {
        if let Some(pattern) = pattern {
            let ctx = ExecutionContext::new();
            for (key, expr) in pattern {
                let expected = self.evaluate_expression(expr, &ctx)?;
                if properties.get(key) != Some(&expected) {
                    return Ok(false);
                }
            }
        }
        Ok(true)
    }

    fn match_relationship_pattern(
        &self,
        pattern: &RelationshipPattern,
    ) -> Result<Vec<ExecutionContext>, ExecutionError> {
        let Pattern::Node(target) = &*pattern.to else {
            return Err(ExecutionError::UnsupportedOperation(
                "Chained relationship patterns not yet supported in MATCH".to_string(),
            ));
        };

        let (min, max) = match &pattern.range {
            Some(range) => (range.min.unwrap_or(1), range.max.unwrap_or(usize::MAX)),
            None => (1, 1),
        };
        if min > max {
            return Err(ExecutionError::ExecutionError(
                "Minimum relationship length exceeds maximum".to_string(),
            ));
        }

        let mut contexts = Vec::new();
        for from in self.matching_nodes(&pattern.from)? {
            let mut path = Vec::new();
            if min == 0 {
                contexts.extend(self.relationship_match(pattern, target, from, from, &path)?);
            }
            if max == 0 {
                continue;
            }

            // Each stack frame retains the current node and the edges still to try.
            // An edge can appear only once per path, so even an open range terminates on cycles.
            let mut stack = vec![(from, self.relationship_edges(from, pattern.direction))];
            let mut used_edges = HashSet::new();
            while let Some((node, edges)) = stack.last_mut() {
                let Some(edge) = edges.next() else {
                    stack.pop();
                    if let Some(edge) = path.pop() {
                        used_edges.remove(&edge.id);
                    }
                    continue;
                };

                if pattern
                    .rel_type
                    .as_ref()
                    .is_some_and(|t| t != &edge.edge_type)
                    || used_edges.contains(&edge.id)
                    || !self.matches_properties(pattern.properties.as_ref(), &edge.properties)?
                {
                    continue;
                }

                let to_id = if edge.from == node.id {
                    &edge.to
                } else {
                    &edge.from
                };
                let Some(to) = self.graph.get_node(to_id) else {
                    continue;
                };

                path.push(edge);
                if path.len() >= min {
                    contexts.extend(self.relationship_match(pattern, target, from, to, &path)?);
                }
                if path.len() < max {
                    used_edges.insert(&edge.id);
                    stack.push((to, self.relationship_edges(to, pattern.direction)));
                } else {
                    path.pop();
                }
            }
        }

        Ok(contexts)
    }

    fn relationship_edges(&self, node: &Node, direction: Direction) -> std::vec::IntoIter<&Edge> {
        match direction {
            Direction::Outgoing => self.graph.get_outgoing_edges(&node.id),
            Direction::Incoming => self.graph.get_incoming_edges(&node.id),
            Direction::Undirected => {
                let mut edges = self.graph.get_outgoing_edges(&node.id);
                // Self-loops are already included in the outgoing edges.
                edges.extend(
                    self.graph
                        .get_incoming_edges(&node.id)
                        .into_iter()
                        .filter(|edge| edge.from != node.id),
                );
                edges
            }
        }
        .into_iter()
    }

    fn relationship_match(
        &self,
        pattern: &RelationshipPattern,
        target: &NodePattern,
        from: &Node,
        to: &Node,
        path: &[&Edge],
    ) -> Result<Option<ExecutionContext>, ExecutionError> {
        if (pattern.from.variable.is_some()
            && pattern.from.variable == target.variable
            && from.id != to.id)
            || !target.labels.iter().all(|label| to.has_label(label))
            || !self.matches_properties(target.properties.as_ref(), &to.properties)?
        {
            return Ok(None);
        }

        let mut ctx = ExecutionContext::new();
        if let Some(var) = &pattern.from.variable {
            ctx.bind(var.clone(), ContextValue::Node(from.clone()));
        }
        if let Some(var) = &pattern.variable {
            let value = if pattern.range.is_some() {
                ContextValue::List(
                    path.iter()
                        .map(|&edge| ContextValue::Edge(edge.clone()))
                        .collect(),
                )
            } else {
                ContextValue::Edge(path[0].clone())
            };
            ctx.bind(var.clone(), value);
        }
        if let Some(var) = &target.variable {
            ctx.bind(var.clone(), ContextValue::Node(to.clone()));
        }

        Ok(Some(ctx))
    }

    fn execute_return(
        &self,
        clause: &ReturnClause,
        context: &ExecutionContext,
    ) -> Result<ExecutionResult, ExecutionError> {
        let mut columns = Vec::new();
        for item in &clause.items {
            let col_name = column_name(item);
            if !columns.contains(&col_name) {
                columns.push(col_name);
            }
        }

        let mut result = ExecutionResult::new(columns.clone());

        for row_ctx in self.row_contexts(context) {
            let mut row = HashMap::new();
            for col_name in &columns {
                // Find the expression for this column
                let expr = clause
                    .items
                    .iter()
                    .find(|item| &column_name(item) == col_name)
                    .map(|item| &item.expression);

                if let Some(expr) = expr {
                    let value = self.evaluate_expression_ctx(expr, &row_ctx)?;
                    row.insert(col_name.clone(), value);
                }
            }
            result.add_row(row);
        }

        Ok(result)
    }

    fn execute_set(
        &mut self,
        clause: &SetClause,
        context: &ExecutionContext,
    ) -> Result<ExecutionResult, ExecutionError> {
        for row_ctx in self.row_contexts(context) {
            for item in &clause.items {
                match item {
                    SetItem::Property {
                        variable,
                        property,
                        value,
                    } => {
                        let val = self.evaluate_expression(value, &row_ctx)?;
                        if let Some(ContextValue::Node(node)) = row_ctx.get(variable) {
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

    fn execute_delete(
        &mut self,
        clause: &DeleteClause,
        context: &ExecutionContext,
    ) -> Result<ExecutionResult, ExecutionError> {
        // Cross products can bind the same element in several rows; delete it once.
        let mut deleted_nodes = HashSet::new();
        let mut deleted_edges = HashSet::new();

        for row_ctx in self.row_contexts(context) {
            for expr in &clause.expressions {
                if let Expression::Variable(var) = expr {
                    if let Some(ctx_val) = row_ctx.get(var) {
                        match ctx_val {
                            ContextValue::Node(node) => {
                                if clause.detach {
                                    if deleted_nodes.insert(node.id.clone()) {
                                        self.graph.delete_node(&node.id)?;
                                    }
                                } else {
                                    return Err(ExecutionError::ExecutionError(
                                        "Cannot delete node with relationships without DETACH"
                                            .to_string(),
                                    ));
                                }
                            }
                            ContextValue::Edge(edge) => {
                                if deleted_edges.insert(edge.id.clone()) {
                                    self.graph.delete_edge(&edge.id)?;
                                }
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

/// Column name for a RETURN item: its alias, else `n` or `n.prop`.
fn column_name(item: &ReturnItem) -> String {
    if let Some(alias) = &item.alias {
        return alias.clone();
    }
    match &item.expression {
        Expression::Variable(var) => var.clone(),
        Expression::Property { object, property } => match &**object {
            Expression::Variable(var) => format!("{}.{}", var, property),
            _ => "?column?".to_string(),
        },
        _ => "?column?".to_string(),
    }
}

/// Combine two binding sets, or `None` if they bind one variable to different elements.
fn merge_bindings(left: &ExecutionContext, right: &ExecutionContext) -> Option<ExecutionContext> {
    let mut merged = left.clone();
    for (var, value) in &right.variables {
        if let Some(existing) = merged.get(var) {
            let same = match (existing, value) {
                (ContextValue::Node(a), ContextValue::Node(b)) => a.id == b.id,
                (ContextValue::Edge(a), ContextValue::Edge(b)) => a.id == b.id,
                (ContextValue::Value(a), ContextValue::Value(b)) => a == b,
                _ => false,
            };
            if !same {
                return None;
            }
        } else {
            merged.bind(var.clone(), value.clone());
        }
    }
    Some(merged)
}
