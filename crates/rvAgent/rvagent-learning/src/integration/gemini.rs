//! Gemini 2.5 Flash integration for GOAP reasoning

use crate::goap::{GoapPlan, LearningGoal, LearningWorldState, PlannedAction};
use anyhow::{anyhow, Result};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Gemini GOAP reasoner using Gemini 2.5 Flash
pub struct GeminiGoapReasoner {
    /// HTTP client
    client: Client,

    /// API key
    api_key: String,

    /// Model to use
    model: String,

    /// Maximum tokens
    max_tokens: usize,

    /// Temperature
    temperature: f32,
}

/// Gemini API request
#[derive(Debug, Serialize)]
struct GeminiRequest {
    contents: Vec<Content>,
    generation_config: GenerationConfig,
}

#[derive(Debug, Serialize)]
struct Content {
    parts: Vec<Part>,
}

#[derive(Debug, Serialize)]
struct Part {
    text: String,
}

#[derive(Debug, Serialize)]
struct GenerationConfig {
    temperature: f32,
    max_output_tokens: usize,
    response_mime_type: String,
}

/// Gemini API response
#[derive(Debug, Deserialize)]
struct GeminiResponse {
    candidates: Vec<Candidate>,
}

#[derive(Debug, Deserialize)]
struct Candidate {
    content: ContentResponse,
}

#[derive(Debug, Deserialize)]
struct ContentResponse {
    parts: Vec<PartResponse>,
}

#[derive(Debug, Deserialize)]
struct PartResponse {
    text: String,
}

/// Parsed GOAP reasoning response
#[derive(Debug, Clone, Deserialize)]
pub struct GoapReasoningResponse {
    /// Suggested actions in order
    pub actions: Vec<SuggestedAction>,

    /// Reasoning explanation
    pub reasoning: String,

    /// Expected outcome
    pub expected_outcome: String,

    /// Confidence score
    pub confidence: f64,
}

/// A suggested action from Gemini
#[derive(Debug, Clone, Deserialize)]
pub struct SuggestedAction {
    /// Action name
    pub action: String,

    /// Why this action
    pub rationale: String,

    /// Expected effect
    pub expected_effect: String,

    /// Estimated cost
    pub cost: f64,
}

/// Quality assessment response from Gemini
#[derive(Debug, Clone, Deserialize)]
pub struct QualityResponse {
    /// Novelty score
    pub novelty: f64,

    /// Usefulness score
    pub usefulness: f64,

    /// Clarity score
    pub clarity: f64,

    /// Correctness score
    pub correctness: f64,

    /// Generalizability score
    pub generalizability: f64,

    /// Reasoning for scores
    pub reasoning: String,
}

impl GeminiGoapReasoner {
    /// Create a new Gemini reasoner
    pub fn new(api_key: impl Into<String>) -> Self {
        let client = Client::builder()
            .timeout(Duration::from_secs(60))
            .build()
            .expect("Failed to create HTTP client");

        Self {
            client,
            api_key: api_key.into(),
            model: "gemini-2.5-flash-preview-05-20".to_string(),
            max_tokens: 4096,
            temperature: 0.3,
        }
    }

    /// Set the model
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }

    /// Set max tokens
    pub fn with_max_tokens(mut self, tokens: usize) -> Self {
        self.max_tokens = tokens;
        self
    }

    /// Set temperature
    pub fn with_temperature(mut self, temp: f32) -> Self {
        self.temperature = temp;
        self
    }

    /// Generate a GOAP plan using Gemini reasoning
    pub async fn reason_plan(
        &self,
        state: &LearningWorldState,
        goal: &LearningGoal,
    ) -> Result<GoapPlan> {
        let prompt = self.build_planning_prompt(state, goal);

        let response = self.call_gemini(&prompt).await?;

        // Parse the JSON response
        let reasoning: GoapReasoningResponse = serde_json::from_str(&response)
            .map_err(|e| anyhow!("Failed to parse Gemini response: {}", e))?;

        // Convert to GoapPlan
        let actions: Vec<PlannedAction> = reasoning
            .actions
            .iter()
            .map(|a| PlannedAction {
                action: a.action.clone(),
                params: serde_json::json!({}),
                parallel_with: vec![],
                cost: a.cost,
            })
            .collect();

        let total_cost: f64 = actions.iter().map(|a| a.cost).sum();

        Ok(GoapPlan {
            actions,
            estimated_cost: total_cost,
            reasoning: reasoning.reasoning,
            expected_state: state.clone(), // Would be updated based on effects
        })
    }

    /// Assess quality of a discovery using Gemini
    pub async fn assess_quality(
        &self,
        title: &str,
        description: &str,
        code_snippet: Option<&str>,
    ) -> Result<QualityResponse> {
        let prompt = self.build_quality_prompt(title, description, code_snippet);

        let response = self.call_gemini(&prompt).await?;

        serde_json::from_str(&response)
            .map_err(|e| anyhow!("Failed to parse quality response: {}", e))
    }

    /// Build the GOAP planning prompt
    fn build_planning_prompt(&self, state: &LearningWorldState, goal: &LearningGoal) -> String {
        let state_json = serde_json::to_string_pretty(state).unwrap_or_default();
        let goal_json = serde_json::to_string_pretty(goal).unwrap_or_default();

        format!(
            r#"You are a GOAP (Goal-Oriented Action Planning) reasoner for a learning system.

## Current World State
```json
{}
```

## Goal
```json
{}
```

## Available Actions
1. scan_codebase - Scan codebase for patterns (cost: 2.0)
   - Preconditions: not scanning, quota >= 1
   - Effects: patterns_discovered++, files_analyzed++

2. analyze_patterns_gemini - Use Gemini to analyze patterns (cost: 3.0)
   - Preconditions: patterns >= 1
   - Effects: quality computed, novelty computed

3. log_discovery - Log a discovery with provenance (cost: 1.0)
   - Preconditions: patterns >= 1
   - Effects: logged++, pending_logs--

4. submit_to_pi - Submit to π.ruv.io (cost: 2.0)
   - Preconditions: pending >= 1, cloud connected
   - Effects: submitted++, pending--

5. consolidate_sona - Run SONA consolidation (cost: 3.0)
   - Preconditions: consolidation_due
   - Effects: consolidation_due = false

6. refresh_connection - Reconnect to π.ruv.io (cost: 1.0)
   - Preconditions: not connected
   - Effects: connected = true

## Instructions
Generate an optimal action sequence to achieve the goal.
Respond with JSON in this exact format:
{{
  "actions": [
    {{
      "action": "action_name",
      "rationale": "why this action",
      "expected_effect": "what it will change",
      "cost": 1.0
    }}
  ],
  "reasoning": "overall reasoning for this plan",
  "expected_outcome": "what state we expect at the end",
  "confidence": 0.85
}}

Respond ONLY with valid JSON, no other text."#,
            state_json, goal_json
        )
    }

    /// Build the quality assessment prompt
    fn build_quality_prompt(
        &self,
        title: &str,
        description: &str,
        code_snippet: Option<&str>,
    ) -> String {
        let code_section = code_snippet
            .map(|c| format!("\n\n## Code Snippet\n```\n{}\n```", c))
            .unwrap_or_default();

        format!(
            r#"You are a code quality assessor. Evaluate this discovery:

## Title
{}

## Description
{}
{}

## Instructions
Assess this discovery on the following criteria (0.0 to 1.0):
- novelty: How new/unique is this pattern?
- usefulness: How useful is it for developers?
- clarity: How clear and understandable?
- correctness: How technically correct?
- generalizability: How applicable across projects?

Respond with JSON in this exact format:
{{
  "novelty": 0.8,
  "usefulness": 0.7,
  "clarity": 0.9,
  "correctness": 0.85,
  "generalizability": 0.6,
  "reasoning": "Brief explanation of scores"
}}

Respond ONLY with valid JSON, no other text."#,
            title, description, code_section
        )
    }

    /// Call the Gemini API
    async fn call_gemini(&self, prompt: &str) -> Result<String> {
        let url = format!(
            "https://generativelanguage.googleapis.com/v1beta/models/{}:generateContent?key={}",
            self.model, self.api_key
        );

        let request = GeminiRequest {
            contents: vec![Content {
                parts: vec![Part {
                    text: prompt.to_string(),
                }],
            }],
            generation_config: GenerationConfig {
                temperature: self.temperature,
                max_output_tokens: self.max_tokens,
                response_mime_type: "application/json".to_string(),
            },
        };

        let response = self.client
            .post(&url)
            .json(&request)
            .send()
            .await?;

        if !response.status().is_success() {
            let status = response.status();
            let text = response.text().await.unwrap_or_default();
            return Err(anyhow!("Gemini API error {}: {}", status, text));
        }

        let gemini_response: GeminiResponse = response.json().await?;

        gemini_response
            .candidates
            .first()
            .and_then(|c| c.content.parts.first())
            .map(|p| p.text.clone())
            .ok_or_else(|| anyhow!("Empty response from Gemini"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reasoner_creation() {
        let reasoner = GeminiGoapReasoner::new("test-api-key")
            .with_model("gemini-2.5-flash")
            .with_temperature(0.5);

        assert_eq!(reasoner.model, "gemini-2.5-flash");
        assert_eq!(reasoner.temperature, 0.5);
    }

    #[test]
    fn test_planning_prompt() {
        let reasoner = GeminiGoapReasoner::new("test-key");
        let state = LearningWorldState::default();
        let goal = LearningGoal::DiscoverPatterns { target_count: 5 };

        let prompt = reasoner.build_planning_prompt(&state, &goal);

        assert!(prompt.contains("GOAP"));
        assert!(prompt.contains("scan_codebase"));
        assert!(prompt.contains("patterns_discovered"));
    }

    #[test]
    fn test_quality_prompt() {
        let reasoner = GeminiGoapReasoner::new("test-key");

        let prompt = reasoner.build_quality_prompt(
            "Test Pattern",
            "A test description",
            Some("fn test() {}"),
        );

        assert!(prompt.contains("novelty"));
        assert!(prompt.contains("usefulness"));
        assert!(prompt.contains("fn test()"));
    }
}
