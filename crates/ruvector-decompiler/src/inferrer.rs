//! Name inference with confidence scoring, training data, and folder naming.
//!
//! Strategies: neural model, training corpus, string patterns, property
//! correlation, structural heuristics. Also provides graph-based folder
//! name inference for hierarchical module trees.

use crate::training::TrainingCorpus;
use crate::types::{Declaration, InferredName, Module};

// ---- Hardcoded Patterns (fallback) ----

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
///
/// Uses the built-in training corpus for domain-specific inference,
/// falling back to hardcoded pattern tables.
pub fn infer_names(modules: &[Module]) -> Vec<InferredName> {
    let corpus = TrainingCorpus::builtin();
    infer_names_with_corpus(modules, &corpus)
}

/// Infer names using a specific training corpus.
pub fn infer_names_with_corpus(
    modules: &[Module],
    corpus: &TrainingCorpus,
) -> Vec<InferredName> {
    let mut inferred = Vec::new();

    for module in modules {
        for decl in &module.declarations {
            if let Some(inf) = infer_declaration_name(decl, corpus) {
                inferred.push(inf);
            }
        }
    }

    inferred
}

/// Infer a name for a single declaration using all strategies.
pub(crate) fn infer_declaration_name(
    decl: &Declaration,
    corpus: &TrainingCorpus,
) -> Option<InferredName> {
    let mut best: Option<InferredName> = None;

    // Strategy 0: Training corpus match (domain-specific).
    if let Some((pattern, score)) = corpus.match_declaration(decl) {
        best = keep_best(best, InferredName {
            original: decl.name.clone(),
            inferred: pattern.inferred_name.clone(),
            confidence: score.min(0.98),
            evidence: vec![format!(
                "training corpus match: {} (score: {:.2}, module_hint: {:?})",
                pattern.inferred_name,
                score,
                pattern.module_hint
            )],
        });
    }

    // Strategy 1: HIGH confidence -- direct string literal match.
    'outer: for lit in &decl.string_literals {
        for &(pattern, name) in KNOWN_PATTERNS {
            if lit.contains(pattern) {
                best = keep_best(best, InferredName {
                    original: decl.name.clone(),
                    inferred: name.to_string(),
                    confidence: 0.95,
                    evidence: vec![format!(
                        "string literal \"{}\" matches known pattern \"{}\"",
                        lit, pattern
                    )],
                });
                break 'outer;
            }
        }
    }

    // Early return if we have a very strong match.
    if best.as_ref().map_or(false, |b| b.confidence > 0.9) {
        return best;
    }

    // Strategy 2: MEDIUM confidence -- property access correlation.
    for prop in &decl.property_accesses {
        for &(pattern, name) in PROPERTY_PATTERNS {
            if prop == pattern {
                best = keep_best(best, InferredName {
                    original: decl.name.clone(),
                    inferred: name.to_string(),
                    confidence: 0.7,
                    evidence: vec![format!(
                        "property access .{} suggests purpose \"{}\"",
                        prop, name
                    )],
                });
                break;
            }
        }
    }

    // Strategy 3: MEDIUM confidence -- multiple string literals.
    if decl.string_literals.len() >= 2 {
        let joined = decl.string_literals.join("_");
        let inferred = sanitize_name(&joined, 30);
        if !inferred.is_empty() && inferred != decl.name {
            best = keep_best(best, InferredName {
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

    if best.is_some() {
        return best;
    }

    // Strategy 4: LOW confidence -- structural heuristics.
    let structural = match decl.kind {
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

    structural.map(|(name, confidence)| InferredName {
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

/// Keep the candidate with the higher confidence score.
fn keep_best(
    current: Option<InferredName>,
    candidate: InferredName,
) -> Option<InferredName> {
    match current {
        Some(c) if c.confidence >= candidate.confidence => Some(c),
        _ => Some(candidate),
    }
}

/// Sanitize a string into a valid identifier name, truncating to `max_len`.
fn sanitize_name(raw: &str, max_len: usize) -> String {
    raw.chars()
        .filter(|c| c.is_alphanumeric() || *c == '_')
        .take(max_len)
        .collect()
}

/// Infer a folder name for a group of modules from their graph context.
/// Uses TF-IDF-like scoring: strings frequent in this group but rare
/// globally get the highest score. Names emerge from graph structure.
pub fn infer_folder_name(modules: &[Module], all_modules: &[Module]) -> String {
    if modules.is_empty() {
        return "root".to_string();
    }
    let group_freq = collect_string_freq(modules);
    if group_freq.is_empty() {
        return infer_from_module_names(modules);
    }
    let global_freq = collect_string_freq(all_modules);

    // Score by discriminativeness: high in group, rare globally.
    let mut scored: Vec<(&str, f64)> = group_freq
        .iter()
        .map(|(&s, &gf)| {
            let global = *global_freq.get(s).unwrap_or(&0) as f64;
            let local = gf as f64;
            (s, (local / (global + 1.0)) * (local + 1.0).ln())
        })
        .collect();
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    scored.first().map_or_else(
        || infer_from_module_names(modules),
        |(best, _)| sanitize_folder_name(best),
    )
}

/// Collect meaningful string frequencies across modules.
fn collect_string_freq<'a>(modules: &'a [Module]) -> std::collections::HashMap<&'a str, usize> {
    let mut freq = std::collections::HashMap::new();
    for module in modules {
        for decl in &module.declarations {
            for s in decl.string_literals.iter().chain(decl.property_accesses.iter()) {
                let s = s.as_str();
                if is_meaningful_string(s) {
                    *freq.entry(s).or_insert(0) += 1;
                }
            }
        }
    }
    freq
}

/// Check if a string is meaningful enough to use for naming.
fn is_meaningful_string(s: &str) -> bool {
    let len = s.len();
    if len < 2 || len > 50 { return false; }
    if s.chars().all(|c| c.is_ascii_digit()) { return false; }
    if s.contains("://") || s.starts_with('/') || s.starts_with('.') { return false; }
    if s.contains('\n') || s.contains('\t') { return false; }
    s.chars().any(|c| c.is_alphabetic())
}

/// Infer a folder name from module name common prefix.
fn infer_from_module_names(modules: &[Module]) -> String {
    if modules.len() == 1 {
        return sanitize_folder_name(&modules[0].name);
    }
    let names: Vec<&str> = modules.iter().map(|m| m.name.as_str()).collect();
    if let Some(first) = names.first() {
        let prefix_len = names.iter().skip(1).fold(first.len(), |acc, name| {
            first.chars().zip(name.chars()).take(acc)
                .take_while(|(a, b)| a == b).count()
        });
        if prefix_len >= 2 { return sanitize_folder_name(&first[..prefix_len]); }
    }
    format!("group_{}", modules.len())
}

/// Sanitize a string into a valid folder name.
fn sanitize_folder_name(raw: &str) -> String {
    let cleaned: String = raw.chars()
        .map(|c| if c.is_alphanumeric() || c == '_' || c == '-' { c.to_ascii_lowercase() } else { '_' })
        .collect();
    let trimmed = cleaned.trim_matches('_');
    if trimmed.is_empty() { "module".to_string() } else { trimmed.to_string() }
}

/// Feedback from a ground-truth comparison for self-learning.
#[derive(Debug, Clone)]
pub struct InferenceFeedback {
    pub original: String,
    pub inferred: String,
    pub correct: String,
    pub was_correct: bool,
    pub evidence: Vec<String>,
}

/// Learn from ground-truth comparison results.
///
/// Returns `(successes, failures)`.
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
    pub minified_name: String,
    pub inferred_name: String,
    pub correct_name: String,
    pub evidence: Vec<String>,
}

// ---------------------------------------------------------------------------
// Neural name inference context (shared with `neural` module)
// ---------------------------------------------------------------------------

/// Context signals passed to the neural inferrer for a single declaration.
#[derive(Debug, Clone)]
pub struct InferenceContext {
    /// String literals found near the declaration.
    pub string_literals: Vec<String>,
    /// Property names accessed on the declaration.
    pub property_accesses: Vec<String>,
    /// Declaration kind as a string (e.g., "function", "var", "class").
    pub kind: String,
}

impl InferenceContext {
    /// Build an `InferenceContext` from a declaration.
    pub fn from_declaration(decl: &Declaration) -> Self {
        Self {
            string_literals: decl.string_literals.clone(),
            property_accesses: decl.property_accesses.clone(),
            kind: decl.kind.to_string(),
        }
    }
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

    #[test]
    fn test_training_corpus_mcp() {
        let decl = make_decl(
            "x",
            DeclKind::Var,
            &["protocolVersion", "serverInfo", "capabilities"],
            &["protocolVersion", "serverInfo"],
        );
        let modules = vec![make_module(vec![decl])];
        let inferred = infer_names(&modules);
        assert_eq!(inferred.len(), 1);
        assert!(
            inferred[0].inferred.contains("Mcp")
                || inferred[0].inferred.contains("protocol")
                || inferred[0].inferred.contains("capabilities"),
            "Expected MCP-related name, got: {}",
            inferred[0].inferred
        );
        assert!(inferred[0].confidence > 0.85);
    }

    #[test]
    fn test_training_corpus_bash_tool() {
        let decl = make_decl(
            "y",
            DeclKind::Var,
            &["Bash", "Read", "Edit", "Write"],
            &["description", "inputSchema"],
        );
        let modules = vec![make_module(vec![decl])];
        let inferred = infer_names(&modules);
        assert_eq!(inferred.len(), 1);
        assert!(
            inferred[0].inferred.contains("Tool"),
            "Expected Tool-related name, got: {}",
            inferred[0].inferred
        );
        assert!(inferred[0].confidence > 0.85);
    }
}
