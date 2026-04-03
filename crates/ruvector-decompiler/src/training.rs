//! Training corpus for domain-specific name inference.
//!
//! Loads patterns from JSON data files (e.g., Claude Code patterns)
//! and matches declarations against them for high-quality name inference.

use std::collections::HashSet;

use crate::types::Declaration;

/// A training pattern mapping context signals to a known name.
#[derive(Debug, Clone)]
pub struct TrainingPattern {
    /// String literals that appear near the declaration.
    pub context_strings: Vec<String>,
    /// Property names accessed on the declaration.
    pub property_names: Vec<String>,
    /// The inferred human-readable name.
    pub inferred_name: String,
    /// Optional module classification hint.
    pub module_hint: Option<String>,
    /// Confidence score (0.0 to 1.0).
    pub confidence: f64,
}

/// A corpus of training patterns for domain-specific inference.
#[derive(Debug, Clone)]
pub struct TrainingCorpus {
    pub patterns: Vec<TrainingPattern>,
}

impl TrainingCorpus {
    /// Create an empty corpus.
    pub fn new() -> Self {
        Self {
            patterns: Vec::new(),
        }
    }

    /// Load training data from a JSON string.
    ///
    /// Expected format: array of objects with fields:
    /// - `context_strings`: `[String]`
    /// - `property_names`: `[String]`
    /// - `inferred_name`: `String`
    /// - `module_hint`: `String` (optional)
    /// - `confidence`: `f64`
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        let raw: Vec<RawPattern> = serde_json::from_str(json)?;
        let patterns = raw
            .into_iter()
            .map(|r| TrainingPattern {
                context_strings: r.context_strings,
                property_names: r.property_names,
                inferred_name: r.inferred_name,
                module_hint: r.module_hint,
                confidence: r.confidence,
            })
            .collect();
        Ok(Self { patterns })
    }

    /// Load the built-in Claude Code patterns.
    pub fn builtin() -> Self {
        let json = include_str!("../data/claude-code-patterns.json");
        Self::from_json(json).unwrap_or_else(|_| Self::new())
    }

    /// Match a declaration against the training corpus.
    ///
    /// Returns the best-matching pattern with a computed match score.
    /// Requires at least one context string or property name match.
    pub fn match_declaration(
        &self,
        decl: &Declaration,
    ) -> Option<(&TrainingPattern, f64)> {
        let decl_strings: HashSet<&str> = decl
            .string_literals
            .iter()
            .map(|s| s.as_str())
            .collect();
        let decl_props: HashSet<&str> = decl
            .property_accesses
            .iter()
            .map(|s| s.as_str())
            .collect();

        let mut best: Option<(&TrainingPattern, f64)> = None;

        for pattern in &self.patterns {
            // Count context string matches (substring matching).
            let string_matches: usize = pattern
                .context_strings
                .iter()
                .filter(|cs| {
                    decl_strings.iter().any(|ds| ds.contains(cs.as_str()))
                        || decl
                            .string_literals
                            .iter()
                            .any(|lit| lit.contains(cs.as_str()))
                })
                .count();

            // Count property name matches (exact).
            let prop_matches: usize = pattern
                .property_names
                .iter()
                .filter(|pn| decl_props.contains(pn.as_str()))
                .count();

            let total_signals =
                pattern.context_strings.len() + pattern.property_names.len();
            if total_signals == 0 {
                continue;
            }

            let match_ratio =
                (string_matches + prop_matches) as f64 / total_signals as f64;

            // Require at least one match to consider this pattern.
            if string_matches + prop_matches == 0 {
                continue;
            }

            // Weighted score: match_ratio * pattern confidence.
            let score = match_ratio * pattern.confidence;

            if let Some((_, best_score)) = best {
                if score > best_score {
                    best = Some((pattern, score));
                }
            } else {
                best = Some((pattern, score));
            }
        }

        // Only return if the score is meaningful (>= 0.3).
        best.filter(|(_, score)| *score >= 0.3)
    }
}

#[derive(serde::Deserialize)]
struct RawPattern {
    context_strings: Vec<String>,
    property_names: Vec<String>,
    inferred_name: String,
    module_hint: Option<String>,
    confidence: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::DeclKind;

    fn make_decl(
        name: &str,
        strings: &[&str],
        props: &[&str],
    ) -> Declaration {
        Declaration {
            name: name.to_string(),
            kind: DeclKind::Var,
            byte_range: (0, 10),
            string_literals: strings.iter().map(|s| s.to_string()).collect(),
            property_accesses: props.iter().map(|s| s.to_string()).collect(),
            references: vec![],
        }
    }

    #[test]
    fn test_training_corpus_from_json() {
        let json = r#"[
            {
                "context_strings": ["test_pattern"],
                "property_names": [],
                "inferred_name": "TestHandler",
                "module_hint": null,
                "confidence": 0.95
            }
        ]"#;
        let corpus = TrainingCorpus::from_json(json).unwrap();
        assert_eq!(corpus.patterns.len(), 1);
        assert_eq!(corpus.patterns[0].inferred_name, "TestHandler");
    }

    #[test]
    fn test_builtin_corpus_loads() {
        let corpus = TrainingCorpus::builtin();
        assert!(
            corpus.patterns.len() >= 200,
            "Expected at least 200 builtin patterns, got {}",
            corpus.patterns.len()
        );
    }

    #[test]
    fn test_corpus_match_mcp() {
        let decl = make_decl(
            "x",
            &["protocolVersion", "serverInfo", "capabilities"],
            &["protocolVersion", "serverInfo"],
        );
        let corpus = TrainingCorpus::builtin();
        let result = corpus.match_declaration(&decl);
        assert!(result.is_some());
        let (pattern, score) = result.unwrap();
        assert!(
            pattern.inferred_name.contains("Mcp")
                || pattern.inferred_name.contains("Protocol"),
            "Expected MCP-related name, got: {}",
            pattern.inferred_name
        );
        assert!(score > 0.3);
    }

    #[test]
    fn test_corpus_match_tool_definitions() {
        let decl = make_decl(
            "y",
            &["Bash", "Read", "Edit", "Write"],
            &["description", "inputSchema"],
        );
        let corpus = TrainingCorpus::builtin();
        let result = corpus.match_declaration(&decl);
        assert!(result.is_some());
        let (pattern, _) = result.unwrap();
        assert!(
            pattern.inferred_name.contains("Tool"),
            "Expected Tool-related name, got: {}",
            pattern.inferred_name
        );
    }
}
