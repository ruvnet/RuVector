//! Discovery logging with full method provenance
//!
//! This module provides:
//! - DiscoveryLog struct with tool and method attribution
//! - Scanner for codebase pattern discovery
//! - Quality assessment integration

mod scanner;
mod analyzer;

pub use scanner::CodebaseScanner;
pub use analyzer::PatternAnalyzer;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use uuid::Uuid;

/// A logged discovery with full provenance
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiscoveryLog {
    /// Unique identifier for this discovery
    pub id: Uuid,

    /// When the discovery was made
    pub discovered_at: DateTime<Utc>,

    /// Category of discovery
    pub category: DiscoveryCategory,

    /// Human-readable title
    pub title: String,

    /// Detailed description
    pub description: String,

    /// Source file(s) where discovered
    pub source_files: Vec<PathBuf>,

    /// Code snippet demonstrating the discovery
    pub code_snippet: Option<String>,

    /// Tools used to make the discovery
    pub tools_used: Vec<ToolUsage>,

    /// Quality assessment scores
    pub quality: QualityAssessment,

    /// Whether submitted to π.ruv.io
    pub submitted_to_cloud: bool,

    /// Cloud submission ID if submitted
    pub cloud_memory_id: Option<String>,

    /// Tags for categorization
    pub tags: Vec<String>,
}

/// Categories of discoveries
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum DiscoveryCategory {
    /// Code optimization pattern
    Optimization,

    /// Security improvement
    Security,

    /// Performance enhancement
    Performance,

    /// Architectural pattern
    Architecture,

    /// Testing pattern
    Testing,

    /// Error handling pattern
    ErrorHandling,

    /// API design pattern
    ApiDesign,

    /// Documentation pattern
    Documentation,

    /// Other/uncategorized
    Other,
}

/// Record of a tool used in discovery
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolUsage {
    /// Type of tool
    pub tool_type: ToolType,

    /// Specific tool name
    pub tool_name: String,

    /// How it was used
    pub usage_description: String,

    /// Duration in milliseconds
    pub duration_ms: Option<u64>,

    /// Input provided to the tool
    pub input_summary: Option<String>,

    /// Output received from the tool
    pub output_summary: Option<String>,
}

/// Types of tools in RuVector ecosystem
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum ToolType {
    /// SONA pattern recognition
    Sona,

    /// ReasoningBank pattern storage
    ReasoningBank,

    /// HNSW similarity search
    Hnsw,

    /// Gemini reasoning
    Gemini,

    /// MCP tool call
    Mcp,

    /// rvAgent coordination
    RvAgent,

    /// RuVector solver
    Solver,

    /// Code analysis
    Analysis,

    /// π.ruv.io cloud brain
    PiRuvIo,

    /// EWC++ consolidation
    Ewc,

    /// Custom tool
    Custom,
}

/// Quality assessment for a discovery
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct QualityAssessment {
    /// Novelty score (0.0 - 1.0)
    pub novelty: f64,

    /// Usefulness score (0.0 - 1.0)
    pub usefulness: f64,

    /// Clarity score (0.0 - 1.0)
    pub clarity: f64,

    /// Correctness score (0.0 - 1.0)
    pub correctness: f64,

    /// Generalizability score (0.0 - 1.0)
    pub generalizability: f64,

    /// Composite score (weighted average)
    pub composite: f64,

    /// Confidence in the assessment
    pub confidence: f64,

    /// Assessment method used
    pub method: AssessmentMethod,
}

/// Method used for quality assessment
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum AssessmentMethod {
    /// No assessment performed
    #[default]
    None,

    /// Heuristic-based assessment
    Heuristic,

    /// HNSW similarity comparison
    HnswSimilarity,

    /// Gemini reasoning
    Gemini,

    /// Combined methods
    Hybrid,
}

impl DiscoveryLog {
    /// Create a new discovery log
    pub fn new(category: DiscoveryCategory, title: impl Into<String>, description: impl Into<String>) -> Self {
        Self {
            id: Uuid::new_v4(),
            discovered_at: Utc::now(),
            category,
            title: title.into(),
            description: description.into(),
            source_files: vec![],
            code_snippet: None,
            tools_used: vec![],
            quality: QualityAssessment::default(),
            submitted_to_cloud: false,
            cloud_memory_id: None,
            tags: vec![],
        }
    }

    /// Add a source file
    pub fn with_source_file(mut self, path: impl Into<PathBuf>) -> Self {
        self.source_files.push(path.into());
        self
    }

    /// Add a code snippet
    pub fn with_code_snippet(mut self, snippet: impl Into<String>) -> Self {
        self.code_snippet = Some(snippet.into());
        self
    }

    /// Add tool usage
    pub fn with_tool(mut self, tool: ToolUsage) -> Self {
        self.tools_used.push(tool);
        self
    }

    /// Add tags
    pub fn with_tags(mut self, tags: Vec<String>) -> Self {
        self.tags = tags;
        self
    }

    /// Set quality assessment
    pub fn with_quality(mut self, quality: QualityAssessment) -> Self {
        self.quality = quality;
        self
    }

    /// Check if discovery passes quality threshold
    pub fn passes_quality_threshold(&self, threshold: f64) -> bool {
        self.quality.composite >= threshold
    }

    /// Get method attribution summary
    pub fn method_attribution(&self) -> String {
        let tools: Vec<String> = self
            .tools_used
            .iter()
            .map(|t| format!("{} ({})", t.tool_name, t.usage_description))
            .collect();

        if tools.is_empty() {
            "No tools recorded".to_string()
        } else {
            tools.join(" → ")
        }
    }
}

impl ToolUsage {
    /// Create SONA tool usage
    pub fn sona(description: impl Into<String>) -> Self {
        Self {
            tool_type: ToolType::Sona,
            tool_name: "SONA".to_string(),
            usage_description: description.into(),
            duration_ms: None,
            input_summary: None,
            output_summary: None,
        }
    }

    /// Create Gemini tool usage
    pub fn gemini(description: impl Into<String>) -> Self {
        Self {
            tool_type: ToolType::Gemini,
            tool_name: "Gemini 2.5 Flash".to_string(),
            usage_description: description.into(),
            duration_ms: None,
            input_summary: None,
            output_summary: None,
        }
    }

    /// Create HNSW tool usage
    pub fn hnsw(description: impl Into<String>) -> Self {
        Self {
            tool_type: ToolType::Hnsw,
            tool_name: "HNSW".to_string(),
            usage_description: description.into(),
            duration_ms: None,
            input_summary: None,
            output_summary: None,
        }
    }

    /// Create π.ruv.io tool usage
    pub fn pi_ruvio(description: impl Into<String>) -> Self {
        Self {
            tool_type: ToolType::PiRuvIo,
            tool_name: "π.ruv.io".to_string(),
            usage_description: description.into(),
            duration_ms: None,
            input_summary: None,
            output_summary: None,
        }
    }

    /// Set duration
    pub fn with_duration(mut self, ms: u64) -> Self {
        self.duration_ms = Some(ms);
        self
    }

    /// Set input summary
    pub fn with_input(mut self, summary: impl Into<String>) -> Self {
        self.input_summary = Some(summary.into());
        self
    }

    /// Set output summary
    pub fn with_output(mut self, summary: impl Into<String>) -> Self {
        self.output_summary = Some(summary.into());
        self
    }
}

impl QualityAssessment {
    /// Compute composite score from individual scores
    pub fn compute_composite(&mut self) {
        self.composite = self.novelty * 0.3
            + self.usefulness * 0.25
            + self.clarity * 0.15
            + self.correctness * 0.2
            + self.generalizability * 0.1;
    }

    /// Create from Gemini response
    pub fn from_gemini_scores(
        novelty: f64,
        usefulness: f64,
        clarity: f64,
        correctness: f64,
        generalizability: f64,
    ) -> Self {
        let mut assessment = Self {
            novelty,
            usefulness,
            clarity,
            correctness,
            generalizability,
            composite: 0.0,
            confidence: 0.8,
            method: AssessmentMethod::Gemini,
        };
        assessment.compute_composite();
        assessment
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_discovery_log_creation() {
        let log = DiscoveryLog::new(
            DiscoveryCategory::Optimization,
            "Test Discovery",
            "A test discovery description",
        )
        .with_source_file("src/lib.rs")
        .with_tool(ToolUsage::sona("Pattern search"))
        .with_tags(vec!["test".to_string(), "optimization".to_string()]);

        assert_eq!(log.title, "Test Discovery");
        assert_eq!(log.source_files.len(), 1);
        assert_eq!(log.tools_used.len(), 1);
        assert_eq!(log.tags.len(), 2);
    }

    #[test]
    fn test_quality_assessment() {
        let quality = QualityAssessment::from_gemini_scores(0.9, 0.8, 0.7, 0.9, 0.6);
        assert!(quality.composite > 0.7);
        assert_eq!(quality.method, AssessmentMethod::Gemini);
    }

    #[test]
    fn test_method_attribution() {
        let log = DiscoveryLog::new(DiscoveryCategory::Security, "Security Pattern", "desc")
            .with_tool(ToolUsage::sona("Initial scan"))
            .with_tool(ToolUsage::gemini("Deep analysis"));

        let attribution = log.method_attribution();
        assert!(attribution.contains("SONA"));
        assert!(attribution.contains("Gemini"));
    }
}
