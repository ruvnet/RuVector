//! Main routing engine combining all components

use crate::circuit_breaker::CircuitBreaker;
use crate::error::{Result, TinyDancerError};
use crate::feature_engineering::FeatureEngineer;
use crate::model::FastGRNN;
use crate::types::{RouterConfig, RoutingDecision, RoutingRequest, RoutingResponse};
use crate::uncertainty::UncertaintyEstimator;
use crate::voi::{self, Belief, VoiConfig, VoiDecision};
use parking_lot::RwLock;
use std::sync::Arc;
use std::time::Instant;

/// Main router for AI agent routing
pub struct Router {
    config: RouterConfig,
    model: Arc<RwLock<FastGRNN>>,
    feature_engineer: FeatureEngineer,
    uncertainty_estimator: UncertaintyEstimator,
    circuit_breaker: Option<CircuitBreaker>,
}

impl Router {
    /// Create a new router with the given configuration
    pub fn new(config: RouterConfig) -> Result<Self> {
        // Load or create model
        let model = if std::path::Path::new(&config.model_path).exists() {
            FastGRNN::load(&config.model_path)?
        } else {
            FastGRNN::new(Default::default())?
        };

        let circuit_breaker = if config.enable_circuit_breaker {
            Some(CircuitBreaker::new(config.circuit_breaker_threshold))
        } else {
            None
        };

        // Validate the optional VoI gate at construction so route() can
        // trust it (PIR WP28, ADR-331).
        if let Some(gate) = &config.voi {
            gate.escalation.validate()?;
            VoiConfig {
                value_of_success: gate.value_of_success,
                latency_price: gate.latency_price,
            }
            .validate()?;
        }

        Ok(Self {
            config,
            model: Arc::new(RwLock::new(model)),
            feature_engineer: FeatureEngineer::new(),
            uncertainty_estimator: UncertaintyEstimator::new(),
            circuit_breaker,
        })
    }

    /// Create a router with default configuration
    pub fn default() -> Result<Self> {
        Self::new(RouterConfig::default())
    }

    /// Route a request through the system
    pub fn route(&self, request: RoutingRequest) -> Result<RoutingResponse> {
        let start = Instant::now();

        // Check circuit breaker
        if let Some(ref cb) = self.circuit_breaker {
            if !cb.is_closed() {
                return Err(TinyDancerError::CircuitBreakerError(
                    "Circuit breaker is open".to_string(),
                ));
            }
        }

        // Feature engineering
        let feature_start = Instant::now();
        let feature_vectors = self.feature_engineer.extract_batch_features(
            &request.query_embedding,
            &request.candidates,
            request.metadata.as_ref(),
        )?;
        let feature_time_us = feature_start.elapsed().as_micros() as u64;

        // Model inference
        let model = self.model.read();
        let mut decisions = Vec::new();

        for (candidate, features) in request.candidates.iter().zip(feature_vectors.iter()) {
            match model.forward(&features.features, None) {
                Ok(score) => {
                    // Estimate uncertainty
                    let uncertainty = self
                        .uncertainty_estimator
                        .estimate(&features.features, score);

                    // A non-finite score or uncertainty is a model failure on
                    // ANY path, not just the VoI-gated one: a NaN confidence
                    // reaching the decision sort would panic partial_cmp, and
                    // a panic in an inference-hot-path router takes down the
                    // caller instead of letting it fall back. Reject here —
                    // the single choke point both paths share — and trip the
                    // circuit breaker, matching the gated path's contract.
                    if !score.is_finite() || !uncertainty.is_finite() {
                        if let Some(ref cb) = self.circuit_breaker {
                            cb.record_failure();
                        }
                        return Err(TinyDancerError::InvalidInput(format!(
                            "non-finite model output for candidate {}: score={score}, uncertainty={uncertainty}",
                            candidate.id
                        )));
                    }

                    // Determine routing decision. With a VoI gate configured,
                    // escalation to the powerful model is an estimator
                    // purchase: buy it only when the expected quality gain
                    // outweighs its cost (PIR WP28, ADR-331). Otherwise the
                    // legacy threshold rule applies unchanged.
                    let use_lightweight = match &self.config.voi {
                        Some(gate) => match self.voi_use_lightweight(gate, score, uncertainty) {
                            Ok(lightweight) => lightweight,
                            // A non-finite model output is a model failure, so
                            // it must reach the circuit breaker before the
                            // error propagates — otherwise a degenerate model
                            // can be hammered indefinitely through the gated
                            // path without ever tripping the breaker.
                            Err(e) => {
                                if let Some(ref cb) = self.circuit_breaker {
                                    cb.record_failure();
                                }
                                return Err(e);
                            }
                        },
                        None => {
                            score >= self.config.confidence_threshold
                                && uncertainty <= self.config.max_uncertainty
                        }
                    };

                    decisions.push(RoutingDecision {
                        candidate_id: candidate.id.clone(),
                        confidence: score,
                        use_lightweight,
                        uncertainty,
                    });

                    // Record success with circuit breaker
                    if let Some(ref cb) = self.circuit_breaker {
                        cb.record_success();
                    }
                }
                Err(e) => {
                    // Record failure with circuit breaker
                    if let Some(ref cb) = self.circuit_breaker {
                        cb.record_failure();
                    }
                    return Err(e);
                }
            }
        }

        // Sort by confidence (descending). total_cmp keeps the sort total as
        // defense-in-depth — non-finite confidences are already rejected
        // above, but a panic must never depend on that invariant holding.
        decisions.sort_by(|a, b| b.confidence.total_cmp(&a.confidence));

        let inference_time_us = start.elapsed().as_micros() as u64;

        Ok(RoutingResponse {
            decisions,
            inference_time_us,
            candidates_processed: request.candidates.len(),
            feature_time_us,
        })
    }

    /// VoI escalation decision for one scored candidate: belief mean is the
    /// model score, belief std is the conformal uncertainty (floored so the
    /// prior stays proper), and the outside option is the configured
    /// confidence threshold. Non-finite scores are rejected here, BEFORE any
    /// comparison — a NaN score must fail loudly rather than silently pick a
    /// route.
    ///
    /// The gate is **escalate-only** — the downgrade-only analog used
    /// elsewhere in the program for metric-integrity gates. A positive-VoI
    /// purchase can flip a would-be-lightweight route into an escalation, but
    /// [`VoiDecision::Route`] (no purchase worth making) falls back to the
    /// legacy threshold rule rather than asserting "lightweight". Reading
    /// `Route` as "use the cheap model" would make the gate fail open:
    /// `confidence_threshold` and `max_uncertainty` would never be consulted
    /// on the gated path, so the least-confident queries — exactly the ones
    /// escalation exists for — would silently take the cheap model whenever
    /// escalation is priced above the VoI.
    fn voi_use_lightweight(
        &self,
        gate: &crate::types::VoiGateConfig,
        score: f32,
        uncertainty: f32,
    ) -> Result<bool> {
        if !score.is_finite() || !uncertainty.is_finite() {
            return Err(TinyDancerError::InvalidInput(format!(
                "non-finite model output reached the VoI gate: score={score}, uncertainty={uncertainty}"
            )));
        }
        let belief = Belief::new(score as f64, f64::from(uncertainty).max(1e-6))?;
        let decision = voi::decide(
            belief,
            f64::from(self.config.confidence_threshold),
            std::slice::from_ref(&gate.escalation),
            &VoiConfig {
                value_of_success: gate.value_of_success,
                latency_price: gate.latency_price,
            },
        )?;
        Ok(match decision {
            // Escalation is worth buying.
            VoiDecision::Buy(_) => false,
            // Nothing worth buying — the legacy rule still governs.
            VoiDecision::Route => {
                score >= self.config.confidence_threshold
                    && uncertainty <= self.config.max_uncertainty
            }
        })
    }

    /// Reload the model from disk
    pub fn reload_model(&self) -> Result<()> {
        let new_model = FastGRNN::load(&self.config.model_path)?;
        let mut model = self.model.write();
        *model = new_model;
        Ok(())
    }

    /// Get router configuration
    pub fn config(&self) -> &RouterConfig {
        &self.config
    }

    /// Get circuit breaker status
    pub fn circuit_breaker_status(&self) -> Option<bool> {
        self.circuit_breaker.as_ref().map(|cb| cb.is_closed())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Candidate;
    use chrono::Utc;
    use std::collections::HashMap;

    #[test]
    fn test_router_creation() {
        let router = Router::default().unwrap();
        assert!(router.circuit_breaker_status().is_some());
    }

    #[test]
    fn test_routing() {
        let router = Router::default().unwrap();

        // The default FastGRNN model expects input dimension to match feature count (5)
        // Features: semantic_similarity, recency, frequency, success_rate, metadata_overlap
        let candidates = vec![
            Candidate {
                id: "1".to_string(),
                embedding: vec![0.5; 384], // Embeddings can be any size
                metadata: HashMap::new(),
                created_at: Utc::now().timestamp(),
                access_count: 10,
                success_rate: 0.95,
            },
            Candidate {
                id: "2".to_string(),
                embedding: vec![0.3; 384],
                metadata: HashMap::new(),
                created_at: Utc::now().timestamp(),
                access_count: 5,
                success_rate: 0.85,
            },
        ];

        let request = RoutingRequest {
            query_embedding: vec![0.5; 384],
            candidates,
            metadata: None,
        };

        let response = router.route(request).unwrap();
        assert_eq!(response.decisions.len(), 2);
        assert!(response.inference_time_us > 0);
    }

    fn gated_router(cost: f64, noise_std: f64) -> Router {
        use crate::types::VoiGateConfig;
        use crate::voi::EstimatorSpec;

        let mut config = RouterConfig::default();
        config.voi = Some(VoiGateConfig {
            value_of_success: 1.0,
            latency_price: 0.0,
            escalation: EstimatorSpec {
                cost,
                latency_us: 0.0,
                noise_std,
            },
        });
        Router::new(config).unwrap()
    }

    fn legacy_lightweight(config: &RouterConfig, score: f32, uncertainty: f32) -> bool {
        score >= config.confidence_threshold && uncertainty <= config.max_uncertainty
    }

    #[test]
    fn test_routing_with_voi_gate() {
        let request = |candidates| RoutingRequest {
            query_embedding: vec![0.5; 384],
            candidates,
            metadata: None,
        };
        let candidate = Candidate {
            id: "1".to_string(),
            embedding: vec![0.5; 384],
            metadata: HashMap::new(),
            created_at: Utc::now().timestamp(),
            access_count: 10,
            success_rate: 0.95,
        };

        // A free perfect estimator is always bought: escalate.
        let router = gated_router(0.0, 0.0);
        let response = router.route(request(vec![candidate])).unwrap();
        assert!(!response.decisions[0].use_lightweight);
    }

    #[test]
    fn nan_confidence_errors_instead_of_panicking_the_sort() {
        // Regression test for #901: a NaN candidate confidence used to reach
        // `decisions.sort_by(partial_cmp().unwrap())` and panic whenever two
        // or more candidates were sorted. Poisoning the model weights with
        // NaN reproduces the issue's failure scenario (a corrupted or badly
        // quantized model emitting NaN scores) on the default, ungated path —
        // route() must refuse with an error, never panic.
        let router = Router::default().unwrap();
        for w in router.model.write().weights_mut() {
            w.fill(f32::NAN);
        }

        let candidate = |id: &str, fill: f32| Candidate {
            id: id.to_string(),
            embedding: vec![fill; 384],
            metadata: HashMap::new(),
            created_at: Utc::now().timestamp(),
            access_count: 10,
            success_rate: 0.9,
        };

        let request = RoutingRequest {
            query_embedding: vec![0.5; 384],
            candidates: vec![candidate("a", 0.5), candidate("b", 0.3)],
            metadata: None,
        };

        let result =
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| router.route(request)));
        let outcome = result.expect("route() must not panic on NaN confidence");
        let err = outcome.expect_err("NaN confidence must be rejected with an error");
        assert!(
            matches!(err, TinyDancerError::InvalidInput(_)),
            "expected InvalidInput, got: {err:?}"
        );
    }

    #[test]
    fn voi_gate_is_escalate_only_not_fail_open() {
        // Regression: reading VoiDecision::Route as "use lightweight" made
        // the gate fail open, voiding confidence_threshold and
        // max_uncertainty on the gated path. An unpurchasable escalation
        // must leave the legacy rule exactly as it was.
        let router = gated_router(1e9, 0.0);
        let config = router.config();
        for &(score, uncertainty) in &[
            (0.10f32, 0.05f32), // far below threshold — must escalate
            (0.90, 0.90),       // confident but 6x over max_uncertainty
            (0.90, 0.05),       // genuinely safe for the cheap model
            (0.85, 0.15),       // exactly on both boundaries
            (0.84, 0.15),       // a hair under the confidence threshold
        ] {
            let gate = config.voi.as_ref().unwrap();
            let gated = router
                .voi_use_lightweight(gate, score, uncertainty)
                .unwrap();
            assert_eq!(
                gated,
                legacy_lightweight(config, score, uncertainty),
                "gated path diverged from legacy at score={score} uncertainty={uncertainty}"
            );
        }
    }

    #[test]
    fn voi_gate_escalates_the_low_confidence_tail() {
        // The three audited scenarios: each must escalate (use_lightweight
        // == false), whatever the escalation is priced at.
        let cases = [
            // (cost, noise_std, score, uncertainty)
            (0.01, 0.1, 0.10f32, 0.05f32), // cheap escalation, unconfident
            (0.0, 0.0, 0.10, 0.05),        // free perfect oracle, unconfident
            (0.50, 0.1, 0.90, 0.90),       // pricey escalation, over-uncertain
        ];
        for &(cost, noise_std, score, uncertainty) in &cases {
            let router = gated_router(cost, noise_std);
            let gate = router.config().voi.as_ref().unwrap();
            assert!(
                !router.voi_use_lightweight(gate, score, uncertainty).unwrap(),
                "failed to escalate: cost={cost} noise={noise_std} score={score} uncertainty={uncertainty}"
            );
        }
    }

    #[test]
    fn voi_gate_failures_trip_the_circuit_breaker() {
        // A model emitting non-finite scores must reach the breaker through
        // the gated path: without record_failure on that branch, a degenerate
        // model can be hammered indefinitely because the errors return before
        // the breaker is ever told.
        let router = gated_router(0.01, 0.1);
        let threshold = router.config().circuit_breaker_threshold;

        // Poison a bias so every forward() yields NaN.
        {
            let mut model = router.model.write();
            for bias in model.biases_mut() {
                bias.fill(f32::NAN);
            }
        }

        let request = RoutingRequest {
            query_embedding: vec![0.5; 384],
            candidates: vec![Candidate {
                id: "1".to_string(),
                embedding: vec![0.5; 384],
                metadata: HashMap::new(),
                created_at: Utc::now().timestamp(),
                access_count: 1,
                success_rate: 0.9,
            }],
            metadata: None,
        };

        assert!(router.circuit_breaker_status().unwrap(), "starts closed");
        for i in 0..threshold {
            assert!(
                router.route(request.clone()).is_err(),
                "non-finite score must error on attempt {i}"
            );
        }
        assert!(
            !router.circuit_breaker_status().unwrap(),
            "breaker must open after {threshold} consecutive non-finite outputs"
        );
    }

    #[test]
    fn voi_gate_rejects_non_finite_model_output() {
        // Fail loud rather than resolving NaN to a route in either direction.
        let router = gated_router(0.01, 0.1);
        let gate = router.config().voi.as_ref().unwrap();
        assert!(router.voi_use_lightweight(gate, f32::NAN, 0.05).is_err());
        assert!(router.voi_use_lightweight(gate, 0.5, f32::NAN).is_err());
        assert!(router
            .voi_use_lightweight(gate, f32::INFINITY, 0.05)
            .is_err());
    }

    #[test]
    fn test_invalid_voi_gate_rejected_at_construction() {
        use crate::types::VoiGateConfig;
        use crate::voi::EstimatorSpec;

        let mut config = RouterConfig::default();
        config.voi = Some(VoiGateConfig {
            value_of_success: f64::NAN,
            latency_price: 0.0,
            escalation: EstimatorSpec {
                cost: 0.0,
                latency_us: 0.0,
                noise_std: 0.1,
            },
        });
        assert!(Router::new(config).is_err());
    }
}
