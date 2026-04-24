//! Gemini Flash Optimizer (ADR-110 Extension)
//!
//! Provides periodic optimization using Google Gemini Flash 2.5 for:
//! - Neural-symbolic rule refinement
//! - Pattern quality assessment
//! - Knowledge consolidation recommendations
//! - Working memory optimization hints
//!
//! This module is designed to run as a background task that periodically
//! analyzes the cognitive state and provides optimization suggestions.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::time::Duration;

// ─────────────────────────────────────────────────────────────────────────────
// Types
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for the Gemini optimizer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizerConfig {
    /// Gemini API endpoint
    pub api_base: String,
    /// Model ID (e.g., "gemini-2.5-flash-preview-05-20")
    pub model_id: String,
    /// Maximum tokens for response
    pub max_tokens: u32,
    /// Temperature for generation (0.0 = deterministic)
    pub temperature: f32,
    /// Optimization interval (seconds)
    pub interval_secs: u64,
    /// Minimum patterns to trigger optimization
    pub min_patterns: usize,
    /// Enable automatic rule refinement
    pub enable_rule_refinement: bool,
    /// Enable quality assessment
    pub enable_quality_assessment: bool,
}

impl Default for OptimizerConfig {
    fn default() -> Self {
        Self {
            api_base: "https://generativelanguage.googleapis.com/v1beta/models".to_string(),
            model_id: std::env::var("GEMINI_MODEL")
                .unwrap_or_else(|_| "gemini-2.5-flash".to_string()),
            max_tokens: 2048,
            temperature: 0.3,
            interval_secs: 3600, // 1 hour
            min_patterns: 10,
            enable_rule_refinement: true,
            enable_quality_assessment: true,
        }
    }
}

/// Optimization request sent to Gemini
#[derive(Debug, Serialize)]
pub struct OptimizationRequest {
    pub task: OptimizationTask,
    pub context: OptimizationContext,
    pub timestamp: DateTime<Utc>,
}

/// Types of optimization tasks
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OptimizationTask {
    /// Refine neural-symbolic rules based on patterns
    RuleRefinement,
    /// Assess quality of extracted propositions
    QualityAssessment,
    /// Suggest knowledge consolidation strategies
    KnowledgeConsolidation,
    /// Optimize working memory contents
    WorkingMemoryOptimization,
    /// Analyze trajectory patterns for learning improvements
    TrajectoryAnalysis,
}

/// Context provided to the optimizer
#[derive(Debug, Serialize)]
pub struct OptimizationContext {
    /// Current proposition count
    pub propositions: usize,
    /// Current rule count
    pub rules: usize,
    /// SONA patterns stored
    pub sona_patterns: usize,
    /// Working memory utilization
    pub working_memory_load: f64,
    /// Recent thought types distribution
    pub thought_distribution: std::collections::HashMap<String, usize>,
    /// Sample propositions for analysis
    pub sample_propositions: Vec<PropositionSample>,
    /// Memory count
    pub memory_count: usize,
}

/// A sample proposition for optimization analysis
#[derive(Debug, Serialize)]
pub struct PropositionSample {
    pub predicate: String,
    pub arguments: Vec<String>,
    pub confidence: f64,
    pub evidence_count: usize,
}

/// Result from an optimization run
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationResult {
    pub task: OptimizationTask,
    pub timestamp: DateTime<Utc>,
    pub suggestions: Vec<OptimizationSuggestion>,
    pub metrics: OptimizationMetrics,
    pub raw_response: Option<String>,
}

/// A single optimization suggestion
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationSuggestion {
    pub category: String,
    pub priority: f64,
    pub description: String,
    pub action: Option<String>,
}

/// Metrics from optimization run
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationMetrics {
    pub latency_ms: u64,
    pub tokens_used: Option<u64>,
    pub suggestions_generated: usize,
}

// ─────────────────────────────────────────────────────────────────────────────
// Optimizer
// ─────────────────────────────────────────────────────────────────────────────

/// Gemini Flash optimizer for periodic cognitive enhancement
pub struct GeminiOptimizer {
    config: OptimizerConfig,
    api_key: Option<String>,
    http: reqwest::Client,
    last_run: Option<DateTime<Utc>>,
    run_count: u64,
}

impl GeminiOptimizer {
    /// Create a new optimizer with the given config
    pub fn new(config: OptimizerConfig) -> Self {
        let api_key = std::env::var("GEMINI_API_KEY")
            .ok()
            .or_else(|| std::env::var("GOOGLE_API_KEY").ok());

        let http = reqwest::Client::builder()
            .timeout(Duration::from_secs(60))
            .build()
            .unwrap_or_default();

        Self {
            config,
            api_key,
            http,
            last_run: None,
            run_count: 0,
        }
    }

    /// Check if the optimizer is configured (has API key)
    pub fn is_configured(&self) -> bool {
        self.api_key.is_some()
    }

    /// Check if optimization is due (based on interval)
    pub fn is_due(&self) -> bool {
        match self.last_run {
            None => true,
            Some(last) => {
                let elapsed = (Utc::now() - last).num_seconds() as u64;
                elapsed >= self.config.interval_secs
            }
        }
    }

    /// Run optimization for a specific task
    pub async fn optimize(
        &mut self,
        task: OptimizationTask,
        context: OptimizationContext,
    ) -> Result<OptimizationResult, String> {
        let api_key = self
            .api_key
            .as_ref()
            .ok_or("Gemini API key not configured")?;

        let start = std::time::Instant::now();

        // Build the prompt based on task
        let prompt = self.build_prompt(&task, &context);

        // Call Gemini API
        let response = self.call_gemini(api_key, &prompt).await?;

        // Parse suggestions from response
        let suggestions = self.parse_suggestions(&response);

        let latency_ms = start.elapsed().as_millis() as u64;
        self.last_run = Some(Utc::now());
        self.run_count += 1;

        Ok(OptimizationResult {
            task,
            timestamp: Utc::now(),
            suggestions: suggestions.clone(),
            metrics: OptimizationMetrics {
                latency_ms,
                tokens_used: None, // Could parse from response if available
                suggestions_generated: suggestions.len(),
            },
            raw_response: Some(response),
        })
    }

    /// Build optimization prompt for Gemini
    fn build_prompt(&self, task: &OptimizationTask, context: &OptimizationContext) -> String {
        let task_instruction = match task {
            OptimizationTask::RuleRefinement => {
                "Analyze the neural-symbolic rules and suggest refinements. Focus on:\n\
                 - Redundant rules that could be merged\n\
                 - Missing rules that could improve inference\n\
                 - Rules with low confidence that need more evidence\n\
                 - Transitivity chains that could be optimized"
            }
            OptimizationTask::QualityAssessment => {
                "Assess the quality of extracted propositions. Focus on:\n\
                 - Propositions with low evidence counts\n\
                 - Potentially conflicting propositions\n\
                 - Propositions that need reinforcement\n\
                 - Quality score distributions"
            }
            OptimizationTask::KnowledgeConsolidation => {
                "Suggest knowledge consolidation strategies. Focus on:\n\
                 - Clusters that could be merged\n\
                 - Redundant knowledge that could be pruned\n\
                 - Knowledge gaps that need addressing\n\
                 - Cross-domain connections"
            }
            OptimizationTask::WorkingMemoryOptimization => {
                "Optimize working memory contents. Focus on:\n\
                 - Items with low activation that could be evicted\n\
                 - Important items that need boosting\n\
                 - Memory organization improvements\n\
                 - Attention allocation"
            }
            OptimizationTask::TrajectoryAnalysis => {
                "Analyze learning trajectories for improvements. Focus on:\n\
                 - Successful learning patterns to reinforce\n\
                 - Failed patterns to avoid\n\
                 - Trajectory clustering opportunities\n\
                 - Learning rate adjustments"
            }
        };

        format!(
            "You are a cognitive optimizer for a neural-symbolic AI system.\n\n\
             TASK: {:?}\n\n\
             {}\n\n\
             CURRENT STATE:\n\
             - Propositions: {}\n\
             - Rules: {}\n\
             - SONA patterns: {}\n\
             - Working memory load: {:.1}%\n\
             - Memory count: {}\n\n\
             SAMPLE PROPOSITIONS:\n{}\n\n\
             Provide 3-5 specific, actionable suggestions in JSON format:\n\
             [{{\n\
               \"category\": \"<category>\",\n\
               \"priority\": <0.0-1.0>,\n\
               \"description\": \"<what to do>\",\n\
               \"action\": \"<specific action>\"\n\
             }}]",
            task,
            task_instruction,
            context.propositions,
            context.rules,
            context.sona_patterns,
            context.working_memory_load * 100.0,
            context.memory_count,
            context
                .sample_propositions
                .iter()
                .take(5)
                .map(|p| format!(
                    "  - {}({}) [conf={:.2}, evidence={}]",
                    p.predicate,
                    p.arguments.join(", "),
                    p.confidence,
                    p.evidence_count
                ))
                .collect::<Vec<_>>()
                .join("\n")
        )
    }

    /// Call Gemini API with Google Search grounding.
    ///
    /// When `GEMINI_GROUNDING=true` (default), enables Google Search grounding
    /// so Gemini verifies its outputs against live web sources. Grounding metadata
    /// (source URLs, confidence) is logged for auditability.
    async fn call_gemini(&self, api_key: &str, prompt: &str) -> Result<String, String> {
        let url = format!(
            "{}/{}:generateContent?key={}",
            self.config.api_base, self.config.model_id, api_key
        );

        let grounding_enabled =
            std::env::var("GEMINI_GROUNDING").unwrap_or_else(|_| "true".to_string()) == "true";

        let mut body = serde_json::json!({
            "contents": [{
                "role": "user",
                "parts": [{"text": prompt}]
            }],
            "generationConfig": {
                "maxOutputTokens": self.config.max_tokens,
                "temperature": self.config.temperature
            }
        });

        // Add Google Search grounding tool
        if grounding_enabled {
            body["tools"] = serde_json::json!([{
                "google_search": {}
            }]);
        }

        let response = self
            .http
            .post(&url)
            .header("content-type", "application/json")
            .json(&body)
            .send()
            .await
            .map_err(|e| format!("HTTP error: {}", e))?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();
            return Err(format!("Gemini API error {}: {}", status, error_text));
        }

        let json: serde_json::Value = response
            .json()
            .await
            .map_err(|e| format!("JSON parse error: {}", e))?;

        // Log grounding metadata if present (source URLs, support scores)
        if let Some(candidate) = json.get("candidates").and_then(|c| c.get(0)) {
            if let Some(grounding) = candidate.get("groundingMetadata") {
                let sources = grounding
                    .get("groundingChunks")
                    .and_then(|c| c.as_array())
                    .map(|a| a.len())
                    .unwrap_or(0);
                let support = grounding
                    .get("groundingSupports")
                    .and_then(|s| s.as_array())
                    .map(|a| a.len())
                    .unwrap_or(0);
                let query = grounding
                    .get("webSearchQueries")
                    .and_then(|q| q.as_array())
                    .and_then(|a| a.first())
                    .and_then(|q| q.as_str())
                    .unwrap_or("none");
                tracing::info!(
                    sources = sources,
                    supports = support,
                    query = query,
                    "[optimizer] Grounding: {} sources, {} supports, query='{}'",
                    sources,
                    support,
                    query
                );
            }
        }

        // Extract text from response
        json.get("candidates")
            .and_then(|c| c.get(0))
            .and_then(|c| c.get("content"))
            .and_then(|c| c.get("parts"))
            .and_then(|p| p.get(0))
            .and_then(|p| p.get("text"))
            .and_then(|t| t.as_str())
            .map(|s| s.to_string())
            .ok_or_else(|| "Failed to extract response text".to_string())
    }

    /// Parse suggestions from Gemini response
    fn parse_suggestions(&self, response: &str) -> Vec<OptimizationSuggestion> {
        // Try to find JSON array in response
        let json_start = response.find('[');
        let json_end = response.rfind(']');

        if let (Some(start), Some(end)) = (json_start, json_end) {
            let json_str = &response[start..=end];
            if let Ok(suggestions) = serde_json::from_str::<Vec<OptimizationSuggestion>>(json_str) {
                return suggestions;
            }
        }

        // Fallback: create a single suggestion from the response
        vec![OptimizationSuggestion {
            category: "general".to_string(),
            priority: 0.5,
            description: response.chars().take(500).collect(),
            action: None,
        }]
    }

    /// Get run statistics
    pub fn stats(&self) -> OptimizerStats {
        OptimizerStats {
            configured: self.is_configured(),
            run_count: self.run_count,
            last_run: self.last_run,
            next_due: self
                .last_run
                .map(|lr| lr + chrono::Duration::seconds(self.config.interval_secs as i64)),
        }
    }
}

impl Default for GeminiOptimizer {
    fn default() -> Self {
        Self::new(OptimizerConfig::default())
    }
}

/// Optimizer statistics
#[derive(Debug, Serialize)]
pub struct OptimizerStats {
    pub configured: bool,
    pub run_count: u64,
    pub last_run: Option<DateTime<Utc>>,
    pub next_due: Option<DateTime<Utc>>,
}

// ─────────────────────────────────────────────────────────────────────────────
// API Types
// ─────────────────────────────────────────────────────────────────────────────

/// Request for POST /v1/optimize
#[derive(Debug, Deserialize)]
pub struct OptimizeRequest {
    pub task: Option<OptimizationTask>,
}

/// Response for POST /v1/optimize
#[derive(Debug, Serialize)]
pub struct OptimizeResponse {
    pub result: Option<OptimizationResult>,
    pub error: Option<String>,
    pub stats: OptimizerStats,
}

/// Response for GET /v1/optimizer/status
#[derive(Debug, Serialize)]
pub struct OptimizerStatusResponse {
    pub stats: OptimizerStats,
    pub config: OptimizerConfig,
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_optimizer_creation() {
        let optimizer = GeminiOptimizer::default();
        assert!(!optimizer.is_configured() || std::env::var("GEMINI_API_KEY").is_ok());
    }

    #[test]
    fn test_is_due_initially() {
        let optimizer = GeminiOptimizer::default();
        assert!(optimizer.is_due()); // Should be due when never run
    }

    #[test]
    fn test_parse_suggestions() {
        let optimizer = GeminiOptimizer::default();

        let response = r#"Here are my suggestions:
        [
            {
                "category": "rules",
                "priority": 0.8,
                "description": "Merge redundant rules",
                "action": "Combine rule_1 and rule_2"
            }
        ]
        "#;

        let suggestions = optimizer.parse_suggestions(response);
        assert_eq!(suggestions.len(), 1);
        assert_eq!(suggestions[0].category, "rules");
    }

    #[test]
    fn test_build_prompt() {
        let optimizer = GeminiOptimizer::default();
        let context = OptimizationContext {
            propositions: 10,
            rules: 5,
            sona_patterns: 50,
            working_memory_load: 0.7,
            thought_distribution: std::collections::HashMap::new(),
            sample_propositions: vec![],
            memory_count: 100,
        };

        let prompt = optimizer.build_prompt(&OptimizationTask::RuleRefinement, &context);
        assert!(prompt.contains("RuleRefinement"));
        assert!(prompt.contains("Propositions: 10"));
    }
}
