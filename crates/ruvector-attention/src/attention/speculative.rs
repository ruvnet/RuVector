//! Speculative decoding with draft-verify paradigm.
//!
//! Speculative decoding (Leviathan et al., 2023) achieves 2-3x inference speedup
//! with **zero quality loss** by exploiting the asymmetry between generating and
//! verifying tokens. A small "draft" model proposes gamma candidate tokens cheaply,
//! then the large "target" model verifies all candidates in a single forward pass.
//!
//! The key insight: autoregressive generation is memory-bandwidth-bound, not
//! compute-bound. The target model's forward pass for gamma+1 positions costs
//! nearly the same as a single-token forward pass because the GPU is underutilized
//! during single-token generation. By batching gamma+1 positions, we amortize the
//! cost of the target model across multiple accepted tokens.
//!
//! The rejection sampling scheme guarantees that the output distribution is
//! **identical** to sampling from the target model alone -- no approximation.

use crate::error::{AttentionError, AttentionResult};

/// Token identifier.
pub type TokenId = u32;

/// Configuration for speculative decoding.
#[derive(Clone, Debug)]
pub struct SpeculativeConfig {
    /// Number of draft tokens to generate per step (typically 4-8).
    pub gamma: usize,
    /// Sampling temperature. Values > 1.0 increase randomness.
    pub temperature: f32,
    /// Nucleus sampling threshold. Tokens with cumulative probability above
    /// this are excluded.
    pub top_p: f32,
    /// Maximum sequence length for the generation.
    pub max_seq_len: usize,
}

impl SpeculativeConfig {
    /// Creates a new configuration with the given draft length.
    pub fn new(gamma: usize) -> Self {
        Self {
            gamma,
            temperature: 1.0,
            top_p: 1.0,
            max_seq_len: 2048,
        }
    }

    /// Validates the configuration parameters.
    pub fn validate(&self) -> AttentionResult<()> {
        let err = |msg: &str| Err(AttentionError::InvalidConfig(msg.into()));
        if self.gamma == 0 {
            return err("gamma must be > 0");
        }
        if self.gamma > 32 {
            return err("gamma must be <= 32");
        }
        if self.temperature <= 0.0 {
            return err("temperature must be > 0");
        }
        if self.top_p <= 0.0 || self.top_p > 1.0 {
            return err("top_p must be in (0, 1]");
        }
        if self.max_seq_len == 0 {
            return err("max_seq_len must be > 0");
        }
        Ok(())
    }
}

/// Draft model trait: a small, fast model that proposes candidate tokens.
pub trait DraftModel: Send + Sync {
    /// Generates `gamma` draft tokens given a prefix.
    ///
    /// Returns a vector of (token_id, probability) pairs representing the
    /// draft model's greedy/sampled choices and their probabilities under
    /// the draft distribution.
    fn draft_tokens(
        &self,
        prefix: &[TokenId],
        gamma: usize,
    ) -> Vec<(TokenId, f32)>;
}

/// Target model trait: the large, accurate model that verifies drafts.
pub trait TargetModel: Send + Sync {
    /// Evaluates the target model on all draft positions in one forward pass.
    ///
    /// Given the prefix and the draft tokens, returns the target model's full
    /// probability distribution at each of the `gamma + 1` positions (gamma
    /// verification positions plus one bonus position).
    ///
    /// Each inner `Vec<(TokenId, f32)>` is a sparse probability distribution
    /// over the vocabulary (only tokens with nonzero probability need appear).
    fn verify_batch(
        &self,
        prefix: &[TokenId],
        draft_tokens: &[TokenId],
    ) -> Vec<Vec<(TokenId, f32)>>;
}

/// Result of a single speculative decoding step.
#[derive(Clone, Debug)]
pub struct AcceptedTokens {
    /// The tokens accepted in this step (1 to gamma+1).
    pub tokens: Vec<TokenId>,
    /// Fraction of draft tokens that were accepted.
    pub acceptance_rate: f32,
    /// Number of draft model calls made.
    pub draft_calls: usize,
    /// Number of target model calls made (always 1 per step).
    pub target_calls: usize,
}

/// Aggregate statistics for a speculative decoding session.
#[derive(Clone, Debug, Default)]
pub struct DecodingStats {
    /// Total tokens generated across all steps.
    pub tokens_generated: usize,
    /// Running acceptance rate.
    pub acceptance_rate: f32,
    /// Observed speedup ratio vs autoregressive decoding.
    pub speedup_ratio: f32,
    /// Average draft model latency in milliseconds.
    pub draft_latency_ms: f64,
    /// Average target model latency in milliseconds.
    pub target_latency_ms: f64,
}

/// Computes the theoretical speedup from speculative decoding.
///
/// Formula: `(gamma * alpha) / (1 + gamma * (1 - alpha))`
///
/// where `gamma` is the draft length and `alpha` is the acceptance rate.
/// At alpha=1.0 (all accepted) speedup approaches gamma.
/// At alpha=0.0 (all rejected) speedup is 0 (worse than baseline).
pub fn theoretical_speedup(gamma: usize, acceptance_rate: f32) -> f32 {
    let g = gamma as f32;
    let a = acceptance_rate.clamp(0.0, 1.0);
    let denominator = 1.0 + g * (1.0 - a);
    if denominator <= 0.0 {
        return 0.0;
    }
    (g * a) / denominator
}

/// The core speculative decoder implementing the Leviathan et al. algorithm.
pub struct SpeculativeDecoder;

impl SpeculativeDecoder {
    /// Performs one speculative decoding step.
    ///
    /// # Algorithm
    ///
    /// 1. Draft model generates gamma candidate tokens with probabilities q_i.
    /// 2. Target model verifies all gamma+1 positions in one forward pass,
    ///    producing distributions p_i.
    /// 3. For each draft token i (left to right):
    ///    - If p_i(t_i) >= q_i(t_i): accept unconditionally.
    ///    - Otherwise: accept with probability p_i(t_i) / q_i(t_i).
    ///    - On rejection: sample from adjusted distribution max(0, p_i - q_i)
    ///      (normalized), then stop.
    /// 4. If all gamma tokens accepted: bonus sample from p_{gamma+1}.
    pub fn decode_step(
        prefix: &[TokenId],
        draft: &dyn DraftModel,
        target: &dyn TargetModel,
        config: &SpeculativeConfig,
        rng_values: Option<&[f32]>,
    ) -> AttentionResult<AcceptedTokens> {
        config.validate()?;

        let draft_results = draft.draft_tokens(prefix, config.gamma);
        if draft_results.is_empty() {
            return Err(AttentionError::EmptyInput(
                "draft model returned no tokens".into(),
            ));
        }

        let draft_tokens: Vec<TokenId> =
            draft_results.iter().map(|(t, _)| *t).collect();
        let draft_probs: Vec<f32> =
            draft_results.iter().map(|(_, p)| *p).collect();

        let target_dists = target.verify_batch(prefix, &draft_tokens);
        if target_dists.len() < draft_tokens.len() + 1 {
            return Err(AttentionError::ComputationError(
                "target model must return gamma+1 distributions".into(),
            ));
        }

        let mut accepted = Vec::new();
        let mut rejected = false;

        for i in 0..draft_tokens.len() {
            let token = draft_tokens[i];
            let q_i = draft_probs[i];
            let p_i = prob_of_token(&target_dists[i], token);

            let rng_val = rng_values
                .and_then(|v| v.get(i).copied())
                .unwrap_or(0.0);

            if p_i >= q_i {
                // Accept unconditionally: target agrees at least as much.
                accepted.push(token);
            } else if rng_val < p_i / q_i {
                // Accept with probability p_i / q_i.
                accepted.push(token);
            } else {
                // Reject: sample from adjusted distribution max(0, p - q).
                let adjusted = sample_adjusted(
                    &target_dists[i],
                    &draft_tokens,
                    &draft_probs,
                    i,
                );
                accepted.push(adjusted);
                rejected = true;
                break;
            }
        }

        // If all gamma tokens accepted, bonus sample from p_{gamma+1}.
        if !rejected {
            let bonus_dist = &target_dists[draft_tokens.len()];
            if let Some(&(token, _)) = bonus_dist.first() {
                accepted.push(token);
            }
        }

        let num_draft = draft_tokens.len();
        let num_accepted_from_draft = if rejected {
            accepted.len().saturating_sub(1)
        } else {
            num_draft
        };
        let acceptance_rate = if num_draft > 0 {
            num_accepted_from_draft as f32 / num_draft as f32
        } else {
            0.0
        };

        Ok(AcceptedTokens {
            tokens: accepted,
            acceptance_rate,
            draft_calls: 1,
            target_calls: 1,
        })
    }
}

/// Look up the probability of a specific token in a sparse distribution.
fn prob_of_token(dist: &[(TokenId, f32)], token: TokenId) -> f32 {
    dist.iter()
        .find(|(t, _)| *t == token)
        .map(|(_, p)| *p)
        .unwrap_or(0.0)
}

/// Sample from the adjusted distribution max(0, p_i - q_i), normalized.
///
/// For simplicity, we take the token with the highest adjusted probability.
/// In production, this would use proper categorical sampling.
fn sample_adjusted(
    target_dist: &[(TokenId, f32)],
    draft_tokens: &[TokenId],
    draft_probs: &[f32],
    position: usize,
) -> TokenId {
    let mut best_token = target_dist
        .first()
        .map(|(t, _)| *t)
        .unwrap_or(0);
    let mut best_score = f32::NEG_INFINITY;

    for &(token, p_target) in target_dist {
        let p_draft = if token == draft_tokens[position] {
            draft_probs[position]
        } else {
            0.0
        };
        let adjusted = (p_target - p_draft).max(0.0);
        if adjusted > best_score {
            best_score = adjusted;
            best_token = token;
        }
    }
    best_token
}

// ---------------------------------------------------------------------------
// Medusa-style parallel decoding
// ---------------------------------------------------------------------------

/// A single Medusa prediction head that produces candidate tokens
/// from a shared hidden state.
pub trait MedusaHead: Send + Sync {
    /// Predicts candidate tokens for one future position.
    ///
    /// Returns a sparse distribution over the vocabulary.
    fn predict(&self, prefix: &[TokenId]) -> Vec<(TokenId, f32)>;
}

/// Result of Medusa-style tree verification.
#[derive(Clone, Debug)]
pub struct MedusaResult {
    /// Accepted tokens from the best verified path.
    pub tokens: Vec<TokenId>,
    /// Number of candidate paths evaluated.
    pub paths_evaluated: usize,
}

/// Performs simplified Medusa-style parallel decoding.
///
/// Instead of a single draft sequence, multiple independent heads each
/// predict one future token, forming a tree of candidates. The target
/// model verifies the most promising path in one forward pass.
pub fn medusa_decode(
    prefix: &[TokenId],
    heads: &[&dyn MedusaHead],
    target: &dyn TargetModel,
    config: &SpeculativeConfig,
) -> AttentionResult<MedusaResult> {
    config.validate()?;

    if heads.is_empty() {
        return Err(AttentionError::EmptyInput(
            "at least one Medusa head required".into(),
        ));
    }

    // Each head predicts one position ahead.
    let head_predictions: Vec<Vec<(TokenId, f32)>> = heads
        .iter()
        .map(|h| h.predict(prefix))
        .collect();

    // Build the greedy candidate path (top-1 from each head).
    let candidate_path: Vec<TokenId> = head_predictions
        .iter()
        .filter_map(|dist| dist.first().map(|(t, _)| *t))
        .collect();

    if candidate_path.is_empty() {
        return Err(AttentionError::EmptyInput(
            "heads produced no predictions".into(),
        ));
    }

    // Verify the candidate path with the target model.
    let target_dists = target.verify_batch(prefix, &candidate_path);

    // Accept tokens while the target model agrees.
    let mut accepted = Vec::new();
    for (i, &token) in candidate_path.iter().enumerate() {
        if i >= target_dists.len() {
            break;
        }
        let p = prob_of_token(&target_dists[i], token);
        if p > 0.0 {
            accepted.push(token);
        } else {
            break;
        }
    }

    // If nothing was accepted, take the target model's top choice at pos 0.
    if accepted.is_empty() {
        if let Some(dist) = target_dists.first() {
            if let Some(&(token, _)) = dist.first() {
                accepted.push(token);
            }
        }
    }

    Ok(MedusaResult {
        tokens: accepted,
        paths_evaluated: 1, // greedy path only in this simplified version
    })
}

// ---------------------------------------------------------------------------
// Mock implementations for testing
// ---------------------------------------------------------------------------

/// A mock draft model with a configurable token sequence and probability.
pub struct SimpleDraftModel {
    /// Tokens the draft model will propose, cycling if gamma > len.
    pub tokens: Vec<TokenId>,
    /// Probability assigned to each drafted token.
    pub probability: f32,
}

impl DraftModel for SimpleDraftModel {
    fn draft_tokens(
        &self,
        _prefix: &[TokenId],
        gamma: usize,
    ) -> Vec<(TokenId, f32)> {
        (0..gamma)
            .map(|i| {
                let token = self.tokens[i % self.tokens.len()];
                (token, self.probability)
            })
            .collect()
    }
}

/// A mock target model that returns configurable distributions.
pub struct SimpleTargetModel {
    /// Distributions to return for each position.
    /// If `verify_batch` requests more positions than available,
    /// the last distribution is repeated.
    pub distributions: Vec<Vec<(TokenId, f32)>>,
}

impl TargetModel for SimpleTargetModel {
    fn verify_batch(
        &self,
        _prefix: &[TokenId],
        draft_tokens: &[TokenId],
    ) -> Vec<Vec<(TokenId, f32)>> {
        let needed = draft_tokens.len() + 1;
        (0..needed)
            .map(|i| {
                if i < self.distributions.len() {
                    self.distributions[i].clone()
                } else {
                    self.distributions
                        .last()
                        .cloned()
                        .unwrap_or_else(|| vec![(0, 1.0)])
                }
            })
            .collect()
    }
}

/// A mock Medusa head that always predicts a fixed token.
pub struct SimpleMedusaHead {
    /// The token this head predicts.
    pub token: TokenId,
    /// Probability assigned to the prediction.
    pub probability: f32,
}

impl MedusaHead for SimpleMedusaHead {
    fn predict(&self, _prefix: &[TokenId]) -> Vec<(TokenId, f32)> {
        vec![(self.token, self.probability)]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_config() -> SpeculativeConfig {
        SpeculativeConfig::new(4)
    }

    // -- Config validation tests --

    #[test]
    fn test_config_valid() {
        assert!(default_config().validate().is_ok());
    }

    #[test]
    fn test_config_gamma_zero() {
        let mut cfg = default_config();
        cfg.gamma = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_gamma_too_large() {
        let mut cfg = default_config();
        cfg.gamma = 33;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_bad_temperature() {
        let mut cfg = default_config();
        cfg.temperature = 0.0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_bad_top_p() {
        let mut cfg = default_config();
        cfg.top_p = 0.0;
        assert!(cfg.validate().is_err());

        cfg.top_p = 1.1;
        assert!(cfg.validate().is_err());
    }

    // -- Full acceptance test --

    #[test]
    fn test_full_acceptance() {
        // Target probability >= draft probability at every position -> all accept.
        let draft = SimpleDraftModel {
            tokens: vec![10, 20, 30, 40],
            probability: 0.5,
        };
        let target = SimpleTargetModel {
            distributions: vec![
                vec![(10, 0.8)],
                vec![(20, 0.7)],
                vec![(30, 0.6)],
                vec![(40, 0.9)],
                vec![(50, 1.0)], // bonus position
            ],
        };

        let result = SpeculativeDecoder::decode_step(
            &[1, 2, 3],
            &draft,
            &target,
            &default_config(),
            None,
        )
        .unwrap();

        // All 4 draft tokens accepted + 1 bonus = 5 tokens.
        assert_eq!(result.tokens.len(), 5);
        assert_eq!(result.tokens, vec![10, 20, 30, 40, 50]);
        assert!((result.acceptance_rate - 1.0).abs() < f32::EPSILON);
    }

    // -- Full rejection test --

    #[test]
    fn test_full_rejection() {
        // Target probability 0 for the draft token -> immediate rejection.
        let draft = SimpleDraftModel {
            tokens: vec![10, 20, 30, 40],
            probability: 0.9,
        };
        // The target gives 0 prob to token 10, but high prob to token 99.
        let target = SimpleTargetModel {
            distributions: vec![
                vec![(99, 0.9)],
                vec![(99, 0.9)],
                vec![(99, 0.9)],
                vec![(99, 0.9)],
                vec![(99, 1.0)],
            ],
        };

        let result = SpeculativeDecoder::decode_step(
            &[1],
            &draft,
            &target,
            &default_config(),
            Some(&[1.0, 1.0, 1.0, 1.0]), // rng=1.0 forces rejection
        )
        .unwrap();

        // First token rejected, replaced by adjusted sample (token 99).
        assert_eq!(result.tokens.len(), 1);
        assert_eq!(result.tokens[0], 99);
        assert!((result.acceptance_rate - 0.0).abs() < f32::EPSILON);
    }

    // -- Partial acceptance test --

    #[test]
    fn test_partial_acceptance() {
        let draft = SimpleDraftModel {
            tokens: vec![10, 20, 30, 40],
            probability: 0.5,
        };
        // Accept first two (p >= q), reject third (p=0).
        let target = SimpleTargetModel {
            distributions: vec![
                vec![(10, 0.8)],
                vec![(20, 0.6)],
                vec![(77, 0.9)], // no prob for 30 -> reject
                vec![(40, 0.9)],
                vec![(50, 1.0)],
            ],
        };

        let result = SpeculativeDecoder::decode_step(
            &[1],
            &draft,
            &target,
            &default_config(),
            Some(&[0.0, 0.0, 1.0, 0.0]), // rng=1.0 at pos 2 forces reject
        )
        .unwrap();

        // Accepted: 10, 20, then rejected at 30 -> adjusted sample = 77.
        assert_eq!(result.tokens.len(), 3);
        assert_eq!(result.tokens[0], 10);
        assert_eq!(result.tokens[1], 20);
        assert_eq!(result.tokens[2], 77);
        assert!((result.acceptance_rate - 0.5).abs() < f32::EPSILON);
    }

    // -- Rejection sampling produces adjusted distribution token --

    #[test]
    fn test_rejection_sampling_distribution() {
        let draft = SimpleDraftModel {
            tokens: vec![10],
            probability: 0.8,
        };
        // Target gives 0.3 to token 10 and 0.7 to token 42.
        // Adjusted: max(0, 0.3 - 0.8) = 0 for 10, max(0, 0.7 - 0) = 0.7 for 42.
        // So adjusted sample should be 42.
        let target = SimpleTargetModel {
            distributions: vec![
                vec![(10, 0.3), (42, 0.7)],
                vec![(99, 1.0)],
            ],
        };

        let cfg = SpeculativeConfig::new(1);
        let result = SpeculativeDecoder::decode_step(
            &[1],
            &draft,
            &target,
            &cfg,
            Some(&[1.0]), // force rejection
        )
        .unwrap();

        assert_eq!(result.tokens.len(), 1);
        assert_eq!(result.tokens[0], 42);
    }

    // -- Speedup calculation --

    #[test]
    fn test_theoretical_speedup() {
        // gamma=4, alpha=1.0 -> speedup = 4*1 / (1+4*0) = 4.0
        let s = theoretical_speedup(4, 1.0);
        assert!((s - 4.0).abs() < 1e-5);

        // gamma=4, alpha=0.0 -> speedup = 0 / (1+4) = 0.0
        let s = theoretical_speedup(4, 0.0);
        assert!(s.abs() < 1e-5);

        // gamma=4, alpha=0.8 -> 4*0.8 / (1+4*0.2) = 3.2 / 1.8 ~= 1.778
        let s = theoretical_speedup(4, 0.8);
        assert!((s - 3.2 / 1.8).abs() < 1e-4);

        // gamma=8, alpha=0.9 -> 7.2 / 1.8 = 4.0
        let s = theoretical_speedup(8, 0.9);
        assert!((s - 7.2 / 1.8).abs() < 1e-4);
    }

    // -- Medusa tree verification --

    #[test]
    fn test_medusa_decode() {
        let h1 = SimpleMedusaHead {
            token: 10,
            probability: 0.9,
        };
        let h2 = SimpleMedusaHead {
            token: 20,
            probability: 0.8,
        };
        let target = SimpleTargetModel {
            distributions: vec![
                vec![(10, 0.7)],
                vec![(20, 0.6)],
                vec![(99, 1.0)],
            ],
        };

        let heads: Vec<&dyn MedusaHead> = vec![&h1, &h2];
        let result =
            medusa_decode(&[1, 2], &heads, &target, &default_config()).unwrap();

        assert_eq!(result.tokens, vec![10, 20]);
        assert_eq!(result.paths_evaluated, 1);
    }

    #[test]
    fn test_medusa_no_heads() {
        let target = SimpleTargetModel {
            distributions: vec![vec![(1, 1.0)]],
        };
        let heads: Vec<&dyn MedusaHead> = vec![];
        let result =
            medusa_decode(&[1], &heads, &target, &default_config());
        assert!(result.is_err());
    }

    // -- Edge case: probabilistic acceptance --

    #[test]
    fn test_probabilistic_acceptance() {
        // p_i(t_i) < q_i(t_i) but rng is low enough to accept.
        let draft = SimpleDraftModel {
            tokens: vec![10],
            probability: 0.8,
        };
        let target = SimpleTargetModel {
            distributions: vec![
                vec![(10, 0.4)], // p/q = 0.5
                vec![(99, 1.0)],
            ],
        };

        let cfg = SpeculativeConfig::new(1);
        // rng = 0.3 < 0.5 (p/q) -> accept
        let result = SpeculativeDecoder::decode_step(
            &[1],
            &draft,
            &target,
            &cfg,
            Some(&[0.3]),
        )
        .unwrap();

        // Accepted draft token + bonus
        assert_eq!(result.tokens, vec![10, 99]);
        assert!((result.acceptance_rate - 1.0).abs() < f32::EPSILON);
    }

    // -- Edge case: empty prefix --

    #[test]
    fn test_empty_prefix() {
        let draft = SimpleDraftModel {
            tokens: vec![5],
            probability: 0.5,
        };
        let target = SimpleTargetModel {
            distributions: vec![
                vec![(5, 0.9)],
                vec![(6, 1.0)],
            ],
        };

        let cfg = SpeculativeConfig::new(1);
        let result = SpeculativeDecoder::decode_step(
            &[],
            &draft,
            &target,
            &cfg,
            None,
        )
        .unwrap();

        assert_eq!(result.tokens, vec![5, 6]);
    }
}
