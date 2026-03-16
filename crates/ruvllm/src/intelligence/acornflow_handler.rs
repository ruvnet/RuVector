//! Acornflow extension handler for ruvLLM signal processing.
//!
//! Demonstrates the ADR-045 `SignalExtensionHandler` trait by extracting
//! features from the `"acornflow"` extension namespace that the
//! execution-analysis workflow attaches to signals.
//!
//! ## Extension Data Shape
//!
//! ```json
//! {
//!   "acornflow": {
//!     "source_execution_id": "exec-123",
//!     "workflow_name": "deploy-service",
//!     "reasoning_chain": [
//!       { "nodeName": "plan", "status": "completed", "durationMs": 2300 },
//!       { "nodeName": "implement", "status": "completed", "durationMs": 15000 }
//!     ],
//!     "chain_length": 2,
//!     "semantic_refs": { "files": [...] },
//!     "quality_aggregate": { "mean": 0.85, "min": 0.7, "max": 1.0 }
//!   }
//! }
//! ```

use crate::error::Result;
use crate::intelligence::{QualitySignal, SignalExtensionHandler};
use std::collections::HashMap;

/// Extension handler for signals produced by Acornflow's execution-analysis workflow.
///
/// Extracts trajectory features (reasoning chain stats, semantic refs),
/// embedding context (workflow name, chain summary), and router features
/// (chain length, quality aggregate) from the `"acornflow"` namespace.
pub struct AcornflowExtensionHandler;

impl SignalExtensionHandler for AcornflowExtensionHandler {
    fn namespace(&self) -> &str {
        "acornflow"
    }

    fn extract_trajectory_features(
        &self,
        data: &serde_json::Value,
        _signal: &QualitySignal,
    ) -> Result<HashMap<String, serde_json::Value>> {
        let mut features = HashMap::new();

        if let Some(exec_id) = data.get("source_execution_id") {
            features.insert("source_execution_id".into(), exec_id.clone());
        }

        if let Some(wf_name) = data.get("workflow_name") {
            features.insert("workflow_name".into(), wf_name.clone());
        }

        if let Some(chain) = data.get("reasoning_chain").and_then(|v| v.as_array()) {
            features.insert("chain_length".into(), serde_json::json!(chain.len()));

            // Extract step names for pattern matching
            let step_names: Vec<&str> = chain
                .iter()
                .filter_map(|step| step.get("nodeName").and_then(|n| n.as_str()))
                .collect();
            features.insert("chain_steps".into(), serde_json::json!(step_names));

            // Total duration across all steps
            let total_ms: f64 = chain
                .iter()
                .filter_map(|step| step.get("durationMs").and_then(|d| d.as_f64()))
                .sum();
            features.insert("total_duration_ms".into(), serde_json::json!(total_ms));
        }

        if let Some(refs) = data.get("semantic_refs") {
            if let Some(files) = refs.get("files").and_then(|f| f.as_array()) {
                features.insert("semantic_file_count".into(), serde_json::json!(files.len()));
            }
        }

        Ok(features)
    }

    fn extract_embedding_context(
        &self,
        data: &serde_json::Value,
        _signal: &QualitySignal,
    ) -> Result<Vec<String>> {
        let mut context = vec![];

        // Add workflow name for clustering signals by workflow type
        if let Some(name) = data.get("workflow_name").and_then(|v| v.as_str()) {
            context.push(format!("workflow:{}", name));
        }

        // Add chain step summary for semantic similarity
        if let Some(chain) = data.get("reasoning_chain").and_then(|v| v.as_array()) {
            let steps: Vec<&str> = chain
                .iter()
                .filter_map(|step| step.get("nodeName").and_then(|n| n.as_str()))
                .collect();
            if !steps.is_empty() {
                context.push(format!("steps:{}", steps.join("->")));
            }
        }

        Ok(context)
    }

    fn extract_router_features(
        &self,
        data: &serde_json::Value,
        _signal: &QualitySignal,
    ) -> Result<HashMap<String, f32>> {
        let mut features = HashMap::new();

        if let Some(len) = data.get("chain_length").and_then(|v| v.as_f64()) {
            features.insert("chain_length".into(), len as f32);
        }

        if let Some(agg) = data.get("quality_aggregate") {
            if let Some(mean) = agg.get("mean").and_then(|v| v.as_f64()) {
                features.insert("quality_mean".into(), mean as f32);
            }
            if let Some(min) = agg.get("min").and_then(|v| v.as_f64()) {
                features.insert("quality_min".into(), min as f32);
            }
        }

        // Semantic complexity indicator
        if let Some(refs) = data.get("semantic_refs") {
            if let Some(files) = refs.get("files").and_then(|f| f.as_array()) {
                features.insert("semantic_file_count".into(), files.len() as f32);
            }
        }

        Ok(features)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::intelligence::{IntelligenceLoader, Outcome, QualitySignal};

    fn make_acornflow_signal() -> QualitySignal {
        let mut extensions = HashMap::new();
        extensions.insert(
            "acornflow".to_string(),
            serde_json::json!({
                "source_execution_id": "exec-456",
                "workflow_name": "deploy-service",
                "reasoning_chain": [
                    {"nodeName": "plan", "status": "completed", "durationMs": 2300},
                    {"nodeName": "implement", "status": "completed", "durationMs": 15000},
                    {"nodeName": "review", "status": "completed", "durationMs": 4200}
                ],
                "chain_length": 3,
                "semantic_refs": {
                    "files": [
                        {"filePath": "src/auth.ts", "symbols": []},
                        {"filePath": "src/api.ts", "symbols": []}
                    ]
                },
                "quality_aggregate": {"mean": 0.85, "min": 0.7, "max": 1.0, "factorCount": 5}
            }),
        );

        QualitySignal {
            id: "analysis-exec-456-1234".to_string(),
            task_description: "Execution analysis for exec-456".to_string(),
            outcome: Outcome::Success,
            quality_score: 0.85,
            human_verdict: None,
            quality_factors: None,
            completed_at: "2025-02-21T12:00:00Z".to_string(),
            extensions: Some(extensions),
        }
    }

    #[test]
    fn trajectory_features_extraction() {
        let handler = AcornflowExtensionHandler;
        let signal = make_acornflow_signal();
        let ext_data = &signal.extensions.as_ref().unwrap()["acornflow"];

        let features = handler
            .extract_trajectory_features(ext_data, &signal)
            .unwrap();

        assert_eq!(features["source_execution_id"], "exec-456");
        assert_eq!(features["workflow_name"], "deploy-service");
        assert_eq!(features["chain_length"], 3);
        assert_eq!(features["total_duration_ms"], 21500.0);
        assert_eq!(features["semantic_file_count"], 2);

        let steps = features["chain_steps"].as_array().unwrap();
        assert_eq!(steps.len(), 3);
        assert_eq!(steps[0], "plan");
    }

    #[test]
    fn embedding_context_extraction() {
        let handler = AcornflowExtensionHandler;
        let signal = make_acornflow_signal();
        let ext_data = &signal.extensions.as_ref().unwrap()["acornflow"];

        let context = handler
            .extract_embedding_context(ext_data, &signal)
            .unwrap();

        assert_eq!(context.len(), 2);
        assert_eq!(context[0], "workflow:deploy-service");
        assert_eq!(context[1], "steps:plan->implement->review");
    }

    #[test]
    fn router_features_extraction() {
        let handler = AcornflowExtensionHandler;
        let signal = make_acornflow_signal();
        let ext_data = &signal.extensions.as_ref().unwrap()["acornflow"];

        let features = handler.extract_router_features(ext_data, &signal).unwrap();

        assert!((features["chain_length"] - 3.0).abs() < f32::EPSILON);
        assert!((features["quality_mean"] - 0.85).abs() < f32::EPSILON);
        assert!((features["quality_min"] - 0.7).abs() < f32::EPSILON);
        assert!((features["semantic_file_count"] - 2.0).abs() < f32::EPSILON);
    }

    #[test]
    fn end_to_end_through_loader() {
        let mut loader = IntelligenceLoader::new();
        loader.register_extension_handler(Box::new(AcornflowExtensionHandler));

        let signal = make_acornflow_signal();
        let results = loader.process_extensions(&signal);

        // Trajectory
        assert_eq!(results.trajectory_features["chain_length"], 3);
        assert_eq!(
            results.trajectory_features["workflow_name"],
            "deploy-service"
        );

        // Embedding
        assert!(results
            .embedding_context
            .contains(&"workflow:deploy-service".to_string()));

        // Router
        assert!((results.router_features["chain_length"] - 3.0).abs() < f32::EPSILON);
    }
}
