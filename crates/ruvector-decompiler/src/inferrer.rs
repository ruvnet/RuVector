//! Name inference with confidence scoring.
//!
//! Infers human-readable names for minified declarations based on string
//! context, property correlation, and structural heuristics.

use crate::types::{Declaration, InferredName, Module};

/// Known string-to-purpose mappings for HIGH confidence inference.
static KNOWN_PATTERNS: &[(&str, &str)] = &[
    ("tools/call", "mcp_tool_call"),
    ("tools/list", "mcp_tool_list"),
    ("permission", "permission_handler"),
    ("authenticate", "auth_handler"),
    ("authorization", "auth_handler"),
    ("Bearer", "auth_token"),
    ("localStorage", "local_storage"),
    ("sessionStorage", "session_storage"),
    ("fetch", "http_client"),
    ("XMLHttpRequest", "xhr_client"),
    ("addEventListener", "event_listener"),
    ("createElement", "dom_builder"),
    ("querySelector", "dom_query"),
    ("console.log", "logger"),
    ("console.error", "error_logger"),
    ("JSON.parse", "json_parser"),
    ("JSON.stringify", "json_serializer"),
    ("Promise", "async_handler"),
    ("WebSocket", "websocket_client"),
    ("Error", "error_handler"),
    ("jsonrpc", "rpc_handler"),
    ("protocolVersion", "protocol_handler"),
    ("serverInfo", "server_info"),
    ("mcp-server", "mcp_server"),
    ("capabilities", "capabilities_handler"),
    ("router", "router"),
    ("middleware", "middleware"),
    ("database", "db_client"),
    ("schema", "schema_def"),
    ("validate", "validator"),
    ("render", "renderer"),
    ("component", "component"),
    ("state", "state_manager"),
    ("dispatch", "dispatcher"),
    ("reducer", "reducer"),
    ("action", "action_creator"),
    ("selector", "selector"),
    ("effect", "side_effect"),
    ("subscribe", "subscriber"),
    ("unsubscribe", "unsubscriber"),
    ("emit", "event_emitter"),
    ("plugin", "plugin_handler"),
    ("config", "config"),
    ("env", "env_config"),
];

/// Known property-to-purpose mappings for MEDIUM confidence inference.
static PROPERTY_PATTERNS: &[(&str, &str)] = &[
    ("name", "named_entity"),
    ("type", "typed_entity"),
    ("value", "value_holder"),
    ("handler", "handler"),
    ("callback", "callback"),
    ("listener", "listener"),
    ("options", "options"),
    ("params", "params"),
    ("query", "query"),
    ("body", "request_body"),
    ("headers", "headers"),
    ("status", "status"),
    ("response", "response"),
    ("request", "request"),
    ("path", "path_handler"),
    ("url", "url_handler"),
    ("method", "method_handler"),
    ("children", "container_node"),
    ("parent", "nested_node"),
    ("props", "props_handler"),
];

/// Infer names for all declarations across all modules.
pub fn infer_names(modules: &[Module]) -> Vec<InferredName> {
    let mut inferred = Vec::new();

    for module in modules {
        for decl in &module.declarations {
            if let Some(inf) = infer_declaration_name(decl) {
                inferred.push(inf);
            }
        }
    }

    inferred
}

/// Attempt to infer a name for a single declaration.
fn infer_declaration_name(decl: &Declaration) -> Option<InferredName> {
    // Strategy 1: HIGH confidence -- direct string literal match.
    for lit in &decl.string_literals {
        for &(pattern, name) in KNOWN_PATTERNS {
            if lit.contains(pattern) {
                return Some(InferredName {
                    original: decl.name.clone(),
                    inferred: name.to_string(),
                    confidence: 0.95,
                    evidence: vec![format!(
                        "string literal \"{}\" matches known pattern \"{}\"",
                        lit, pattern
                    )],
                });
            }
        }
    }

    // Strategy 2: MEDIUM confidence -- property access correlation.
    for prop in &decl.property_accesses {
        for &(pattern, name) in PROPERTY_PATTERNS {
            if prop == pattern {
                return Some(InferredName {
                    original: decl.name.clone(),
                    inferred: name.to_string(),
                    confidence: 0.7,
                    evidence: vec![format!(
                        "property access .{} suggests purpose \"{}\"",
                        prop, name
                    )],
                });
            }
        }
    }

    // Strategy 3: MEDIUM confidence -- multiple string literals suggest purpose.
    if decl.string_literals.len() >= 2 {
        let joined = decl.string_literals.join("_");
        let inferred = sanitize_name(&joined, 30);
        if !inferred.is_empty() && inferred != decl.name {
            return Some(InferredName {
                original: decl.name.clone(),
                inferred,
                confidence: 0.65,
                evidence: vec![format!(
                    "multiple string literals: {:?}",
                    &decl.string_literals[..decl.string_literals.len().min(3)]
                )],
            });
        }
    }

    // Strategy 4: LOW confidence -- structural heuristics.
    let structural_name = match decl.kind {
        crate::types::DeclKind::Function => {
            if decl.references.is_empty() {
                Some(("utility_fn", 0.4))
            } else {
                Some(("helper_fn", 0.35))
            }
        }
        crate::types::DeclKind::Class => Some(("entity_class", 0.45)),
        _ => {
            if !decl.references.is_empty() {
                Some(("composed_value", 0.3))
            } else {
                None
            }
        }
    };

    structural_name.map(|(name, confidence)| InferredName {
        original: decl.name.clone(),
        inferred: name.to_string(),
        confidence,
        evidence: vec![format!(
            "structural: {} declaration with {} references",
            decl.kind,
            decl.references.len()
        )],
    })
}

/// Sanitize a string into a valid identifier name, truncating to `max_len`.
fn sanitize_name(raw: &str, max_len: usize) -> String {
    let cleaned: String = raw
        .chars()
        .filter(|c| c.is_alphanumeric() || *c == '_')
        .take(max_len)
        .collect();
    cleaned
}

/// Feedback from a ground-truth comparison for self-learning.
#[derive(Debug, Clone)]
pub struct InferenceFeedback {
    /// The minified name.
    pub original: String,
    /// The name our inferrer produced.
    pub inferred: String,
    /// The known correct name (ground truth).
    pub correct: String,
    /// Whether our inference was correct (fuzzy match).
    pub was_correct: bool,
    /// The evidence that led to the inference.
    pub evidence: Vec<String>,
}

/// Learn from ground-truth comparison results.
///
/// Takes a list of feedback entries and returns a summary of learned
/// patterns. In a production system this would persist to SONA; here
/// we return the analysis for callers to store or log.
///
/// Returns `(successes, failures)` -- lists of patterns that worked
/// and patterns that did not, suitable for feeding back into the
/// inference engine.
pub fn learn_from_ground_truth(
    feedback: &[InferenceFeedback],
) -> (Vec<LearnedPattern>, Vec<LearnedPattern>) {
    let mut successes = Vec::new();
    let mut failures = Vec::new();

    for fb in feedback {
        let pattern = LearnedPattern {
            minified_name: fb.original.clone(),
            inferred_name: fb.inferred.clone(),
            correct_name: fb.correct.clone(),
            evidence: fb.evidence.clone(),
        };

        if fb.was_correct {
            successes.push(pattern);
        } else {
            failures.push(pattern);
        }
    }

    (successes, failures)
}

/// A pattern learned from ground-truth feedback.
#[derive(Debug, Clone)]
pub struct LearnedPattern {
    /// The minified name.
    pub minified_name: String,
    /// What we inferred.
    pub inferred_name: String,
    /// The actual correct name.
    pub correct_name: String,
    /// Evidence that led to the inference.
    pub evidence: Vec<String>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{DeclKind, Declaration, Module};

    fn make_module(decls: Vec<Declaration>) -> Module {
        Module {
            name: "test".to_string(),
            index: 0,
            declarations: decls,
            source: String::new(),
            byte_range: (0, 0),
        }
    }

    fn make_decl(
        name: &str,
        kind: DeclKind,
        strings: &[&str],
        props: &[&str],
    ) -> Declaration {
        Declaration {
            name: name.to_string(),
            kind,
            byte_range: (0, 10),
            string_literals: strings.iter().map(|s| s.to_string()).collect(),
            property_accesses: props.iter().map(|s| s.to_string()).collect(),
            references: vec![],
        }
    }

    #[test]
    fn test_high_confidence_string_match() {
        let decl = make_decl("a", DeclKind::Var, &["tools/call"], &[]);
        let modules = vec![make_module(vec![decl])];
        let inferred = infer_names(&modules);
        assert_eq!(inferred.len(), 1);
        assert_eq!(inferred[0].inferred, "mcp_tool_call");
        assert!(inferred[0].confidence > 0.9);
    }

    #[test]
    fn test_medium_confidence_property() {
        let decl = make_decl("b", DeclKind::Var, &[], &["handler"]);
        let modules = vec![make_module(vec![decl])];
        let inferred = infer_names(&modules);
        assert_eq!(inferred.len(), 1);
        assert_eq!(inferred[0].inferred, "handler");
        assert!(inferred[0].confidence >= 0.6);
        assert!(inferred[0].confidence <= 0.9);
    }

    #[test]
    fn test_low_confidence_structural() {
        let decl = make_decl("c", DeclKind::Class, &[], &[]);
        let modules = vec![make_module(vec![decl])];
        let inferred = infer_names(&modules);
        assert_eq!(inferred.len(), 1);
        assert!(inferred[0].confidence < 0.6);
    }
}
