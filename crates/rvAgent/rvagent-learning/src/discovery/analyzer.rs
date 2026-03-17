//! Pattern analyzer for discovered code

use super::{DiscoveryCategory, DiscoveryLog, QualityAssessment, ToolUsage};
use anyhow::Result;
use std::collections::HashMap;

/// Pattern analyzer for code discovery
pub struct PatternAnalyzer {
    /// Minimum lines for a pattern to be significant
    min_pattern_lines: usize,

    /// Minimum occurrences for a pattern
    min_occurrences: usize,

    /// Known pattern signatures
    known_patterns: HashMap<String, PatternSignature>,
}

/// A pattern signature for matching
#[derive(Debug, Clone)]
pub struct PatternSignature {
    /// Pattern name
    pub name: String,

    /// Category
    pub category: DiscoveryCategory,

    /// Keywords to look for
    pub keywords: Vec<String>,

    /// Anti-patterns (must not contain)
    pub anti_patterns: Vec<String>,

    /// Base quality score
    pub base_quality: f64,
}

/// A discovered pattern candidate
#[derive(Debug, Clone)]
pub struct PatternCandidate {
    /// Pattern name/type
    pub pattern_type: String,

    /// Category
    pub category: DiscoveryCategory,

    /// Code snippet
    pub code_snippet: String,

    /// File path
    pub file_path: String,

    /// Line number
    pub line_number: usize,

    /// Confidence score
    pub confidence: f64,
}

impl PatternAnalyzer {
    /// Create a new pattern analyzer
    pub fn new() -> Self {
        let mut analyzer = Self {
            min_pattern_lines: 3,
            min_occurrences: 1,
            known_patterns: HashMap::new(),
        };
        analyzer.register_default_patterns();
        analyzer
    }

    /// Register default pattern signatures
    fn register_default_patterns(&mut self) {
        // Error handling patterns
        self.known_patterns.insert(
            "result_chain".to_string(),
            PatternSignature {
                name: "Result Chaining".to_string(),
                category: DiscoveryCategory::ErrorHandling,
                keywords: vec!["?".to_string(), "Result".to_string(), "Ok(".to_string()],
                anti_patterns: vec!["unwrap()".to_string(), "expect(".to_string()],
                base_quality: 0.75,
            },
        );

        // Performance patterns
        self.known_patterns.insert(
            "lazy_static".to_string(),
            PatternSignature {
                name: "Lazy Static Initialization".to_string(),
                category: DiscoveryCategory::Performance,
                keywords: vec!["lazy_static".to_string(), "once_cell".to_string(), "LazyLock".to_string()],
                anti_patterns: vec![],
                base_quality: 0.7,
            },
        );

        // Security patterns
        self.known_patterns.insert(
            "input_validation".to_string(),
            PatternSignature {
                name: "Input Validation".to_string(),
                category: DiscoveryCategory::Security,
                keywords: vec!["validate".to_string(), "sanitize".to_string(), "check_".to_string()],
                anti_patterns: vec!["unsafe".to_string()],
                base_quality: 0.8,
            },
        );

        // Architecture patterns
        self.known_patterns.insert(
            "builder_pattern".to_string(),
            PatternSignature {
                name: "Builder Pattern".to_string(),
                category: DiscoveryCategory::Architecture,
                keywords: vec!["Builder".to_string(), ".build()".to_string(), "with_".to_string()],
                anti_patterns: vec![],
                base_quality: 0.7,
            },
        );

        // Testing patterns
        self.known_patterns.insert(
            "test_helpers".to_string(),
            PatternSignature {
                name: "Test Helpers".to_string(),
                category: DiscoveryCategory::Testing,
                keywords: vec!["#[test]".to_string(), "assert".to_string(), "mock".to_string()],
                anti_patterns: vec![],
                base_quality: 0.65,
            },
        );

        // Async patterns
        self.known_patterns.insert(
            "async_stream".to_string(),
            PatternSignature {
                name: "Async Stream Processing".to_string(),
                category: DiscoveryCategory::Performance,
                keywords: vec!["async".to_string(), "Stream".to_string(), ".await".to_string()],
                anti_patterns: vec!["block_on".to_string()],
                base_quality: 0.75,
            },
        );
    }

    /// Analyze code content for patterns
    pub fn analyze(&self, content: &str, file_path: &str) -> Vec<PatternCandidate> {
        let mut candidates = Vec::new();

        for (pattern_id, signature) in &self.known_patterns {
            if self.matches_signature(content, signature) {
                // Find the relevant code snippet
                let (snippet, line_num) = self.extract_snippet(content, &signature.keywords);

                candidates.push(PatternCandidate {
                    pattern_type: pattern_id.clone(),
                    category: signature.category.clone(),
                    code_snippet: snippet,
                    file_path: file_path.to_string(),
                    line_number: line_num,
                    confidence: signature.base_quality,
                });
            }
        }

        candidates
    }

    /// Check if content matches a pattern signature
    fn matches_signature(&self, content: &str, signature: &PatternSignature) -> bool {
        // Must contain at least some keywords
        let keyword_matches = signature
            .keywords
            .iter()
            .filter(|kw| content.contains(kw.as_str()))
            .count();

        if keyword_matches < 2 {
            return false;
        }

        // Must not contain anti-patterns
        for anti in &signature.anti_patterns {
            if content.contains(anti.as_str()) {
                return false;
            }
        }

        true
    }

    /// Extract a code snippet around the pattern
    fn extract_snippet(&self, content: &str, keywords: &[String]) -> (String, usize) {
        let lines: Vec<&str> = content.lines().collect();

        // Find line with most keyword matches
        let mut best_line = 0;
        let mut best_matches = 0;

        for (i, line) in lines.iter().enumerate() {
            let matches = keywords.iter().filter(|kw| line.contains(kw.as_str())).count();
            if matches > best_matches {
                best_matches = matches;
                best_line = i;
            }
        }

        // Extract context around the line
        let start = best_line.saturating_sub(3);
        let end = (best_line + 4).min(lines.len());

        let snippet = lines[start..end].join("\n");

        (snippet, best_line + 1)
    }

    /// Convert pattern candidate to discovery log
    pub fn candidate_to_discovery(&self, candidate: &PatternCandidate) -> DiscoveryLog {
        let signature = self.known_patterns.get(&candidate.pattern_type);
        let title = signature
            .map(|s| s.name.clone())
            .unwrap_or_else(|| candidate.pattern_type.clone());

        let description = format!(
            "Discovered {} pattern in {} at line {}",
            title, candidate.file_path, candidate.line_number
        );

        DiscoveryLog::new(candidate.category.clone(), title, description)
            .with_source_file(&candidate.file_path)
            .with_code_snippet(&candidate.code_snippet)
            .with_tool(ToolUsage::sona("Pattern signature matching"))
            .with_quality(QualityAssessment {
                novelty: 0.5, // Will be computed via HNSW later
                usefulness: candidate.confidence,
                clarity: 0.7,
                correctness: candidate.confidence,
                generalizability: 0.6,
                composite: candidate.confidence,
                confidence: candidate.confidence,
                method: super::AssessmentMethod::Heuristic,
            })
    }

    /// Analyze multiple files and return discoveries
    pub fn analyze_files(&self, files: &[(String, String)]) -> Vec<DiscoveryLog> {
        let mut discoveries = Vec::new();

        for (path, content) in files {
            let candidates = self.analyze(content, path);
            for candidate in candidates {
                discoveries.push(self.candidate_to_discovery(&candidate));
            }
        }

        discoveries
    }
}

impl Default for PatternAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

/// Analyze code complexity (simplified)
pub fn analyze_complexity(content: &str) -> f64 {
    let lines: Vec<&str> = content.lines().collect();
    let total_lines = lines.len();

    if total_lines == 0 {
        return 0.0;
    }

    // Count complexity indicators
    let mut complexity_score = 0;

    for line in &lines {
        let trimmed = line.trim();

        // Nesting depth
        let indent = line.len() - trimmed.len();
        complexity_score += indent / 4;

        // Branching
        if trimmed.starts_with("if ")
            || trimmed.starts_with("match ")
            || trimmed.starts_with("while ")
            || trimmed.starts_with("for ")
        {
            complexity_score += 1;
        }

        // Error handling
        if trimmed.contains("?") || trimmed.contains("unwrap()") {
            complexity_score += 1;
        }
    }

    // Normalize to 0-1 range
    let normalized = (complexity_score as f64) / (total_lines as f64 * 2.0);
    normalized.min(1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pattern_analyzer() {
        let analyzer = PatternAnalyzer::new();

        let code = r#"
pub fn build_config() -> Result<Config> {
    let builder = ConfigBuilder::new()
        .with_name("test")
        .with_timeout(30)
        .build()?;
    Ok(builder)
}
"#;

        let candidates = analyzer.analyze(code, "config.rs");
        assert!(!candidates.is_empty());
    }

    #[test]
    fn test_complexity_analysis() {
        let simple = "fn hello() { println!(\"hi\"); }";
        let complex = r#"
fn complex(x: i32) -> Result<i32> {
    if x > 0 {
        match x {
            1 => Ok(1),
            2 => {
                if x > 1 {
                    Ok(2)
                } else {
                    Err("bad")
                }
            }
            _ => Ok(x),
        }
    } else {
        Err("negative")
    }
}
"#;

        let simple_score = analyze_complexity(simple);
        let complex_score = analyze_complexity(complex);

        assert!(complex_score > simple_score);
    }
}
