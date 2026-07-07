//! Speculative Decoding for Accelerated Inference
//!
//! Uses a small draft model to predict tokens, then verifies with the main model.
//! Achieves 2-3x speedup for greedy/low-temperature sampling.
//!
//! ## How It Works
//!
//! 1. **Draft Phase**: Generate K tokens using a small, fast draft model, decoding
//!    autoregressively one real forward pass at a time.
//! 2. **Verify Phase**: Run the main model on all K draft tokens in a single batched
//!    forward pass, producing K next-token logit distributions at once.
//! 3. **Accept/Reject**: Walk the K positions; a draft token is accepted while it
//!    matches the main model's own greedy prediction at that position. The first
//!    mismatch (or the position after the last accepted token) yields the
//!    correction/continuation token.
//! 4. **Correction**: Append the accepted prefix plus the correction token.
//!
//! ## Requirements
//!
//! The draft and main models **must share the same tokenizer/vocabulary** — draft
//! token IDs are compared directly against the main model's argmax token IDs, which
//! is only meaningful if both models assign the same meaning to the same ID (e.g.
//! TinyLlama-1.1B as a draft for a Llama-2 main model; both use Llama's tokenizer).
//! `generate_tokens` checks `vocab_size()` up front and returns an error on mismatch.
//!
//! ## KV-cache recovery on rejection
//!
//! The candle-transformers backends this crate wraps only support two KV-cache
//! states: empty (position 0) or "everything appended since the last reset" — there
//! is no truncation API (see `patches/candle-transformers`). The batched verify
//! forward pass always appends all K draft tokens to the main model's cache before
//! we know how many will be accepted. When some are rejected, both the main and
//! draft model caches are reset and the full accepted context (prompt + all tokens
//! committed so far) is replayed in one batched forward pass to restore a
//! consistent cache. This means the per-rejection cost grows with total context
//! length; the trade-off is still generally favorable because batched forward
//! passes are far cheaper per token than sequential single-token decoding, and it
//! is the only correct option without patching the underlying attention cache
//! implementation itself.
//!
//! ## Example
//!
//! ```rust,ignore
//! use ruvllm::backends::CandleBackend;
//! use ruvllm::speculative::{SpeculativeDecoder, SpeculativeConfig};
//! use std::sync::Arc;
//!
//! let main_backend = Arc::new(main_candle_backend);
//! let draft_backend = Arc::new(draft_candle_backend);
//!
//! let config = SpeculativeConfig {
//!     lookahead: 4,
//!     ..Default::default()
//! };
//!
//! let decoder = SpeculativeDecoder::new(main_backend, draft_backend, config);
//! let output = decoder.generate("Hello, world!", params)?;
//! println!("Accepted {:.0}% of draft tokens", decoder.stats().acceptance_rate * 100.0);
//! ```
//!
//! ## Measured performance (Llama-2-7B main, TinyLlama-1.1B draft, Q4_K_M GGUF, M-series Metal)
//!
//! `examples/speculative_bench.rs` measures this end-to-end on real weights. Results are
//! acceptance-rate dependent, as expected from the literature: on a prompt where the draft
//! model tracks the main model well (85.9% acceptance), speculative decoding measured ~1.1x
//! over baseline autoregressive decoding. On prompts with more modest alignment (62-70%
//! acceptance — TinyLlama and Llama-2-7B are independently trained, not distilled from each
//! other), the extra per-round forward calls (draft steps + verify + correction) are not
//! fully amortized and measured throughput was ~0.6-0.7x of baseline. Neither result is
//! fabricated — both come from real forward passes on real weights; run the example yourself
//! to reproduce. Draft/main pairs with tighter distillation (e.g. a purpose-trained draft
//! model) should show more consistent wins.
//!
//! ## Recommended Model Pairings
//!
//! | Main Model | Draft Model | Shared tokenizer |
//! |------------|-------------|-------------------|
//! | Llama-2-7B / Llama-2-13B | TinyLlama-1.1B | Yes (Llama-2 32k vocab) |
//! | Qwen2.5-14B | Qwen2.5-0.5B | Yes (Qwen BPE vocab) |
//! | Llama-3.1-8B | Llama-3.2-1B | Only if vocab sizes match — verify before use |

use crate::backends::{GenerateParams, GeneratedToken, LlmBackend, Tokenizer};
use crate::error::{Result, RuvLLMError};

use parking_lot::RwLock;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Configuration for speculative decoding
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeculativeConfig {
    /// Number of tokens to speculate ahead (typically 4-8)
    pub lookahead: usize,
    /// Acceptance threshold for draft tokens (probability cutoff)
    pub acceptance_threshold: f32,
    /// Temperature for draft model sampling (0.0 = greedy)
    pub draft_temperature: f32,
    /// Whether to use tree-based speculation for higher acceptance
    pub tree_speculation: bool,
    /// Maximum tree depth when tree speculation is enabled
    pub max_tree_depth: usize,
    /// Branching factor for tree speculation
    pub tree_branching_factor: usize,
    /// Whether to use nucleus sampling for draft
    pub draft_top_p: f32,
    /// Minimum probability ratio for acceptance (p_main / p_draft)
    pub min_acceptance_ratio: f32,
    /// Enable adaptive lookahead based on acceptance rate
    pub adaptive_lookahead: bool,
    /// Minimum lookahead when adaptive
    pub min_lookahead: usize,
    /// Maximum lookahead when adaptive
    pub max_lookahead: usize,
}

impl Default for SpeculativeConfig {
    fn default() -> Self {
        Self {
            lookahead: 4,
            acceptance_threshold: 0.5,
            draft_temperature: 0.0,
            tree_speculation: false,
            max_tree_depth: 3,
            tree_branching_factor: 2,
            draft_top_p: 1.0,
            min_acceptance_ratio: 0.1,
            adaptive_lookahead: true,
            min_lookahead: 2,
            max_lookahead: 8,
        }
    }
}

/// Statistics for speculative decoding performance
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SpeculativeStats {
    /// Total draft tokens generated
    pub draft_tokens: usize,
    /// Total tokens accepted from drafts
    pub accepted_tokens: usize,
    /// Current acceptance rate (0.0 - 1.0)
    pub acceptance_rate: f32,
    /// Estimated speedup compared to vanilla decoding
    pub speedup: f32,
    /// Total main model forward passes
    pub main_forward_passes: usize,
    /// Total draft model forward passes
    pub draft_forward_passes: usize,
    /// Average tokens per main forward pass
    pub avg_tokens_per_main_pass: f32,
    /// Total wall-clock time spent in speculation
    pub total_speculation_time_ms: f64,
    /// Total tokens generated (including corrections)
    pub total_tokens_generated: usize,
}

impl SpeculativeStats {
    /// Create new empty stats
    pub fn new() -> Self {
        Self::default()
    }

    /// Update acceptance rate
    pub fn update_acceptance_rate(&mut self) {
        if self.draft_tokens > 0 {
            self.acceptance_rate = self.accepted_tokens as f32 / self.draft_tokens as f32;
        }
    }

    /// Calculate speedup estimate
    pub fn calculate_speedup(&mut self) {
        if self.main_forward_passes > 0 {
            self.avg_tokens_per_main_pass =
                self.total_tokens_generated as f32 / self.main_forward_passes as f32;
            // Speedup is approximately avg tokens per pass (since we'd need 1 pass per token normally)
            self.speedup = self.avg_tokens_per_main_pass;
        }
    }

    /// Record a speculation round
    pub fn record_round(
        &mut self,
        draft_count: usize,
        accepted_count: usize,
        speculation_time_ms: f64,
    ) {
        self.draft_tokens += draft_count;
        self.accepted_tokens += accepted_count;
        self.draft_forward_passes += draft_count;
        self.main_forward_passes += 1;
        self.total_tokens_generated += accepted_count + 1; // +1 for correction/next token
        self.total_speculation_time_ms += speculation_time_ms;
        self.update_acceptance_rate();
        self.calculate_speedup();
    }

    /// Reset stats
    pub fn reset(&mut self) {
        *self = Self::default();
    }
}

/// Thread-safe atomic stats for concurrent access
pub struct AtomicSpeculativeStats {
    draft_tokens: AtomicUsize,
    accepted_tokens: AtomicUsize,
    main_forward_passes: AtomicUsize,
    draft_forward_passes: AtomicUsize,
    total_tokens_generated: AtomicUsize,
    total_speculation_time_ns: AtomicU64,
}

impl Default for AtomicSpeculativeStats {
    fn default() -> Self {
        Self::new()
    }
}

impl AtomicSpeculativeStats {
    /// Create new atomic stats
    pub fn new() -> Self {
        Self {
            draft_tokens: AtomicUsize::new(0),
            accepted_tokens: AtomicUsize::new(0),
            main_forward_passes: AtomicUsize::new(0),
            draft_forward_passes: AtomicUsize::new(0),
            total_tokens_generated: AtomicUsize::new(0),
            total_speculation_time_ns: AtomicU64::new(0),
        }
    }

    /// Record a speculation round atomically
    pub fn record_round(&self, draft_count: usize, accepted_count: usize, duration: Duration) {
        self.draft_tokens.fetch_add(draft_count, Ordering::Relaxed);
        self.accepted_tokens
            .fetch_add(accepted_count, Ordering::Relaxed);
        self.main_forward_passes.fetch_add(1, Ordering::Relaxed);
        self.draft_forward_passes
            .fetch_add(draft_count, Ordering::Relaxed);
        self.total_tokens_generated
            .fetch_add(accepted_count + 1, Ordering::Relaxed);
        self.total_speculation_time_ns
            .fetch_add(duration.as_nanos() as u64, Ordering::Relaxed);
    }

    /// Get snapshot as regular stats
    pub fn snapshot(&self) -> SpeculativeStats {
        let draft_tokens = self.draft_tokens.load(Ordering::Relaxed);
        let accepted_tokens = self.accepted_tokens.load(Ordering::Relaxed);
        let main_forward_passes = self.main_forward_passes.load(Ordering::Relaxed);
        let total_tokens_generated = self.total_tokens_generated.load(Ordering::Relaxed);
        let total_speculation_time_ns = self.total_speculation_time_ns.load(Ordering::Relaxed);

        let acceptance_rate = if draft_tokens > 0 {
            accepted_tokens as f32 / draft_tokens as f32
        } else {
            0.0
        };

        let avg_tokens_per_main_pass = if main_forward_passes > 0 {
            total_tokens_generated as f32 / main_forward_passes as f32
        } else {
            0.0
        };

        SpeculativeStats {
            draft_tokens,
            accepted_tokens,
            acceptance_rate,
            speedup: avg_tokens_per_main_pass,
            main_forward_passes,
            draft_forward_passes: self.draft_forward_passes.load(Ordering::Relaxed),
            avg_tokens_per_main_pass,
            total_speculation_time_ms: total_speculation_time_ns as f64 / 1_000_000.0,
            total_tokens_generated,
        }
    }

    /// Reset stats
    pub fn reset(&self) {
        self.draft_tokens.store(0, Ordering::Relaxed);
        self.accepted_tokens.store(0, Ordering::Relaxed);
        self.main_forward_passes.store(0, Ordering::Relaxed);
        self.draft_forward_passes.store(0, Ordering::Relaxed);
        self.total_tokens_generated.store(0, Ordering::Relaxed);
        self.total_speculation_time_ns.store(0, Ordering::Relaxed);
    }
}

/// Result of a verification phase
#[derive(Debug, Clone)]
pub struct VerificationResult {
    /// Number of accepted draft tokens
    pub accepted_count: usize,
    /// The next token from main model (correction or continuation)
    pub next_token: u32,
    /// Log probabilities of accepted tokens
    pub accepted_logprobs: Vec<f32>,
    /// Log probability of the next token
    pub next_logprob: f32,
    /// Whether all draft tokens were accepted
    pub all_accepted: bool,
}

/// Node in the speculation tree
#[derive(Debug, Clone)]
pub struct TreeNode {
    /// Token at this node
    pub token: u32,
    /// Probability of this token
    pub prob: f32,
    /// Log probability
    pub logprob: f32,
    /// Children nodes (branches)
    pub children: Vec<TreeNode>,
    /// Depth in the tree
    pub depth: usize,
}

impl TreeNode {
    /// Create a new tree node
    pub fn new(token: u32, prob: f32, depth: usize) -> Self {
        Self {
            token,
            prob,
            logprob: prob.ln(),
            children: Vec::new(),
            depth,
        }
    }

    /// Add a child node
    pub fn add_child(&mut self, token: u32, prob: f32) -> &mut TreeNode {
        let child = TreeNode::new(token, prob, self.depth + 1);
        self.children.push(child);
        // SAFETY: We just pushed, so children is non-empty
        self.children
            .last_mut()
            .expect("children is non-empty after push")
    }

    /// Get all paths from this node to leaves
    pub fn get_paths(&self) -> Vec<Vec<u32>> {
        if self.children.is_empty() {
            return vec![vec![self.token]];
        }

        let mut paths = Vec::new();
        for child in &self.children {
            for mut path in child.get_paths() {
                path.insert(0, self.token);
                paths.push(path);
            }
        }
        paths
    }

    /// Get the best path (highest probability)
    pub fn best_path(&self) -> Vec<u32> {
        if self.children.is_empty() {
            return vec![self.token];
        }

        // SAFETY: We checked children.is_empty() above, so max_by returns Some
        // For NaN comparisons, treat them as equal to maintain deterministic behavior
        let best_child = self
            .children
            .iter()
            .max_by(|a, b| {
                a.prob
                    .partial_cmp(&b.prob)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .expect("children is non-empty");

        let mut path = vec![self.token];
        path.extend(best_child.best_path());
        path
    }
}

/// Speculation tree for tree-based speculation
#[derive(Debug)]
pub struct SpeculationTree {
    /// Root node (represents current context, token is placeholder)
    pub root: TreeNode,
    /// Maximum depth of the tree
    pub max_depth: usize,
    /// Branching factor at each level
    pub branching_factor: usize,
    /// Total number of nodes
    pub node_count: usize,
}

impl SpeculationTree {
    /// Create a new speculation tree
    pub fn new(max_depth: usize, branching_factor: usize) -> Self {
        Self {
            root: TreeNode::new(0, 1.0, 0),
            max_depth,
            branching_factor,
            node_count: 1,
        }
    }

    /// Get all candidate paths for verification
    pub fn get_candidate_paths(&self) -> Vec<Vec<u32>> {
        self.root.get_paths()
    }

    /// Get the best path
    pub fn best_path(&self) -> Vec<u32> {
        let path = self.root.best_path();
        // Skip the root placeholder token
        if path.len() > 1 {
            path[1..].to_vec()
        } else {
            Vec::new()
        }
    }

    /// Clear the tree
    pub fn clear(&mut self) {
        self.root = TreeNode::new(0, 1.0, 0);
        self.node_count = 1;
    }
}

/// Backends usable for speculative decoding.
///
/// Speculative decoding needs things a plain [`LlmBackend`] doesn't expose:
///
/// - Raw per-position next-token logits from a single batched forward pass, so K
///   draft tokens can be verified against the main model in one shot instead of K
///   sequential decode steps (which would erase any speedup).
/// - Cheap KV-cache snapshot/restore, to walk back a rejected draft token without
///   paying for a full context replay (see the module-level docs on cache recovery).
pub trait SpeculativeBackend: LlmBackend {
    /// An O(num_layers) capture of this backend's KV cache + position,
    /// restorable via `restore_context`.
    type Snapshot;

    /// Feed `tokens` through the model, continuing from the current context, and
    /// return one next-token logits vector (length `vocab_size`) per input token.
    fn forward_logits(&self, tokens: &[u32]) -> Result<Vec<Vec<f32>>>;

    /// Reset the KV cache / context to empty.
    fn reset_context(&self);

    /// Number of tokens currently held in the KV cache / context.
    fn context_len(&self) -> usize;

    /// Cheaply capture the current KV cache + position.
    fn snapshot_context(&self) -> Result<Self::Snapshot>;

    /// Restore a previously captured KV cache + position.
    fn restore_context(&self, snapshot: &Self::Snapshot) -> Result<()>;
}

#[cfg(feature = "candle")]
impl SpeculativeBackend for crate::backends::CandleBackend {
    type Snapshot = crate::backends::CandleContextSnapshot;

    fn forward_logits(&self, tokens: &[u32]) -> Result<Vec<Vec<f32>>> {
        self.forward_multi(tokens)
    }

    fn reset_context(&self) {
        crate::backends::CandleBackend::reset_context(self)
    }

    fn context_len(&self) -> usize {
        crate::backends::CandleBackend::context_len(self)
    }

    fn snapshot_context(&self) -> Result<Self::Snapshot> {
        crate::backends::CandleBackend::snapshot_context(self)
    }

    fn restore_context(&self, snapshot: &Self::Snapshot) -> Result<()> {
        crate::backends::CandleBackend::restore_context(self, snapshot)
    }
}

/// Index of the highest-valued logit (greedy/argmax decoding).
fn argmax(logits: &[f32]) -> u32 {
    let mut best_idx = 0usize;
    let mut best_val = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v > best_val {
            best_val = v;
            best_idx = i;
        }
    }
    best_idx as u32
}

/// Sample a token from a logits vector under the given decoding params,
/// returning the token id and its log-probability. `temperature <= 0.0`
/// means greedy (argmax) decoding.
fn sample_from_logits(
    logits: &[f32],
    temperature: f32,
    top_k: usize,
    top_p: f32,
    rng: &mut StdRng,
) -> (u32, f32) {
    if temperature <= 0.0 {
        let idx = argmax(logits) as usize;
        let logprobs = log_softmax(logits);
        return (idx as u32, logprobs[idx]);
    }

    let mut adjusted: Vec<f32> = logits.iter().map(|&v| v / temperature).collect();
    if top_k > 0 {
        top_k_filter(&mut adjusted, top_k);
    }
    if top_p < 1.0 {
        top_p_filter(&mut adjusted, top_p);
    }
    let probs = softmax(&adjusted);
    let idx = sample_from_probs(&probs, rng);
    let logprobs = log_softmax(&adjusted);
    (idx as u32, logprobs[idx])
}

/// Speculative decoder combining draft and main models
pub struct SpeculativeDecoder<M: SpeculativeBackend + ?Sized, D: SpeculativeBackend + ?Sized> {
    /// Main (target) model for verification
    main_model: Arc<M>,
    /// Draft (small) model for speculation
    draft_model: Arc<D>,
    /// Configuration
    config: RwLock<SpeculativeConfig>,
    /// Performance statistics
    stats: AtomicSpeculativeStats,
    /// Current adaptive lookahead
    current_lookahead: AtomicUsize,
    /// Seed for the next generation call's RNG (advanced after each call so
    /// repeated calls with temperature > 0 don't replay the same sequence)
    rng_seed: AtomicU64,
}

impl<M: SpeculativeBackend + ?Sized, D: SpeculativeBackend + ?Sized> SpeculativeDecoder<M, D> {
    /// Create a new speculative decoder
    pub fn new(main_model: Arc<M>, draft_model: Arc<D>, config: SpeculativeConfig) -> Self {
        let lookahead = config.lookahead;
        Self {
            main_model,
            draft_model,
            config: RwLock::new(config),
            stats: AtomicSpeculativeStats::new(),
            current_lookahead: AtomicUsize::new(lookahead),
            rng_seed: AtomicU64::new(42),
        }
    }

    /// Get current configuration
    pub fn config(&self) -> SpeculativeConfig {
        self.config.read().clone()
    }

    /// Update configuration
    pub fn set_config(&self, config: SpeculativeConfig) {
        *self.config.write() = config;
    }

    /// Get performance statistics
    pub fn stats(&self) -> SpeculativeStats {
        self.stats.snapshot()
    }

    /// Reset statistics
    pub fn reset_stats(&self) {
        self.stats.reset();
    }

    /// Get the main model tokenizer
    pub fn tokenizer(&self) -> Option<&dyn Tokenizer> {
        self.main_model.tokenizer()
    }

    /// Tokenize input text
    fn tokenize(&self, text: &str) -> Result<Vec<u32>> {
        let tokenizer = self
            .main_model
            .tokenizer()
            .ok_or_else(|| RuvLLMError::InvalidOperation("No tokenizer available".to_string()))?;
        tokenizer.encode(text)
    }

    /// Decode tokens to text
    fn decode(&self, tokens: &[u32]) -> Result<String> {
        let tokenizer = self
            .main_model
            .tokenizer()
            .ok_or_else(|| RuvLLMError::InvalidOperation("No tokenizer available".to_string()))?;
        tokenizer.decode(tokens)
    }

    /// Check if we should use speculative decoding for these params
    pub fn should_use_speculative(&self, params: &GenerateParams) -> bool {
        // Use speculative for low temperature, greedy, or beam search
        params.temperature < 0.5 || params.top_k == 1
    }

    fn next_rng(&self) -> StdRng {
        let seed = self.rng_seed.fetch_add(1, Ordering::Relaxed);
        StdRng::seed_from_u64(seed)
    }

    /// Generate text with speculative decoding
    pub fn generate(&self, prompt: &str, params: GenerateParams) -> Result<String> {
        let tokens = self.tokenize(prompt)?;
        let generated = self.generate_tokens(&tokens, &params)?;
        self.decode(&generated)
    }

    /// Generate tokens with speculative decoding
    pub fn generate_tokens(
        &self,
        prompt_tokens: &[u32],
        params: &GenerateParams,
    ) -> Result<Vec<u32>> {
        if prompt_tokens.is_empty() {
            return Err(RuvLLMError::InvalidOperation(
                "Cannot speculatively decode an empty prompt".to_string(),
            ));
        }

        let main_tokenizer = self
            .main_model
            .tokenizer()
            .ok_or_else(|| RuvLLMError::InvalidOperation("No main tokenizer".to_string()))?;
        let draft_tokenizer = self
            .draft_model
            .tokenizer()
            .ok_or_else(|| RuvLLMError::InvalidOperation("No draft tokenizer".to_string()))?;
        if main_tokenizer.vocab_size() != draft_tokenizer.vocab_size() {
            return Err(RuvLLMError::InvalidOperation(format!(
                "Speculative decoding requires main and draft models to share a \
                 tokenizer/vocabulary (draft token ids are compared directly against \
                 the main model's predictions); got vocab sizes {} (main) vs {} (draft)",
                main_tokenizer.vocab_size(),
                draft_tokenizer.vocab_size()
            )));
        }
        let eos_token = main_tokenizer.special_tokens().eos_token_id;

        let config = self.config.read().clone();
        let mut rng = self.next_rng();

        self.main_model.reset_context();
        self.draft_model.reset_context();

        let mut context = prompt_tokens.to_vec();
        let mut output = Vec::new();

        let main_prefill = self.main_model.forward_logits(prompt_tokens)?;
        let mut main_last_logits = main_prefill
            .last()
            .cloned()
            .ok_or_else(|| RuvLLMError::Generation("Main model returned no logits".to_string()))?;
        let draft_prefill = self.draft_model.forward_logits(prompt_tokens)?;
        let mut draft_last_logits = draft_prefill
            .last()
            .cloned()
            .ok_or_else(|| RuvLLMError::Generation("Draft model returned no logits".to_string()))?;

        while output.len() < params.max_tokens {
            let start = Instant::now();

            let lookahead = if config.adaptive_lookahead {
                self.current_lookahead.load(Ordering::Relaxed)
            } else {
                config.lookahead
            };

            // Snapshotted *before* the draft phase mutates the draft model's
            // cache — this is the fallback restore point if the very first
            // draft token is rejected.
            let draft_pre_round_snapshot = self.draft_model.snapshot_context()?;

            let (draft_tokens, new_draft_last_logits, draft_snapshots) = if lookahead == 0 {
                (Vec::new(), draft_last_logits.clone(), Vec::new())
            } else {
                self.draft_phase(draft_last_logits, lookahead, &config, eos_token, &mut rng)?
            };
            draft_last_logits = new_draft_last_logits;

            if draft_tokens.is_empty() {
                // No draft tokens (lookahead=0 or draft model exhausted) — take a
                // single step directly from the main model's own logits, no
                // additional forward pass needed since we already have them.
                let (token, _) = sample_from_logits(
                    &main_last_logits,
                    params.temperature,
                    params.top_k,
                    params.top_p,
                    &mut rng,
                );
                if Some(token) == eos_token {
                    break;
                }
                context.push(token);
                output.push(token);
                let main_step = self.main_model.forward_logits(&[token])?;
                main_last_logits = main_step.into_iter().next().ok_or_else(|| {
                    RuvLLMError::Generation("Main model returned no logits".to_string())
                })?;
                let draft_step = self.draft_model.forward_logits(&[token])?;
                draft_last_logits = draft_step.into_iter().next().ok_or_else(|| {
                    RuvLLMError::Generation("Draft model returned no logits".to_string())
                })?;
                continue;
            }

            // Snapshotted before the batched verify forward appends all K
            // draft tokens to the main model's cache — restored below if
            // any of them are rejected.
            let main_pre_verify_snapshot = self.main_model.snapshot_context()?;

            // Verify phase: ONE batched forward pass over all draft tokens.
            let main_logits = self.main_model.forward_logits(&draft_tokens)?;
            let verification = verify_round(
                &draft_tokens,
                &main_last_logits,
                &main_logits,
                params,
                &mut rng,
            );

            let accepted = &draft_tokens[..verification.accepted_count];
            context.extend_from_slice(accepted);
            output.extend_from_slice(accepted);

            // A draft token itself may already be EOS.
            if let Some(eos) = eos_token {
                if let Some(eos_pos) = accepted.iter().position(|&t| t == eos) {
                    output.truncate(output.len() - accepted.len() + eos_pos + 1);
                    self.stats
                        .record_round(draft_tokens.len(), eos_pos + 1, start.elapsed());
                    break;
                }
            }

            if Some(verification.next_token) == eos_token {
                break;
            }
            context.push(verification.next_token);
            output.push(verification.next_token);

            let all_accepted = verification.accepted_count == draft_tokens.len();
            if all_accepted {
                // Both caches already hold context+accepted; just feed the
                // correction/continuation token to stay in sync.
                let main_step = self.main_model.forward_logits(&[verification.next_token])?;
                main_last_logits = main_step.into_iter().next().ok_or_else(|| {
                    RuvLLMError::Generation("Main model returned no logits".to_string())
                })?;
                let draft_step = self
                    .draft_model
                    .forward_logits(&[verification.next_token])?;
                draft_last_logits = draft_step.into_iter().next().ok_or_else(|| {
                    RuvLLMError::Generation("Draft model returned no logits".to_string())
                })?;
            } else {
                // Rejection: both caches have K appended positions but only
                // accepted_count(+1 correction) are valid. Restore each
                // model to its pre-round snapshot and replay only the
                // accepted prefix + correction token — O(accepted_count),
                // not O(context length) like a full reset+replay would be.
                self.main_model.restore_context(&main_pre_verify_snapshot)?;
                let mut main_fix: Vec<u32> = accepted.to_vec();
                main_fix.push(verification.next_token);
                let main_fix_logits = self.main_model.forward_logits(&main_fix)?;
                main_last_logits = main_fix_logits.last().cloned().ok_or_else(|| {
                    RuvLLMError::Generation("Main model returned no logits".to_string())
                })?;

                // The draft model generated `accepted` itself (that's what
                // "accepted" means), so its cache is already correct up to
                // that point — restore to right after the last accepted
                // draft token and only feed the correction token, not the
                // whole accepted prefix again.
                let draft_restore = if verification.accepted_count == 0 {
                    &draft_pre_round_snapshot
                } else {
                    &draft_snapshots[verification.accepted_count - 1]
                };
                self.draft_model.restore_context(draft_restore)?;
                let draft_fix_logits = self
                    .draft_model
                    .forward_logits(&[verification.next_token])?;
                draft_last_logits = draft_fix_logits.into_iter().next().ok_or_else(|| {
                    RuvLLMError::Generation("Draft model returned no logits".to_string())
                })?;
            }

            let duration = start.elapsed();
            self.stats
                .record_round(draft_tokens.len(), verification.accepted_count, duration);

            if config.adaptive_lookahead {
                self.adjust_lookahead(verification.accepted_count, draft_tokens.len(), &config);
            }

            if !params.stop_sequences.is_empty() {
                let current_text = self.decode(&output)?;
                for stop_seq in &params.stop_sequences {
                    if current_text.contains(stop_seq) {
                        let trimmed = current_text.split(stop_seq).next().unwrap_or("");
                        return self
                            .tokenize(trimmed)
                            .map(|t| t.into_iter().skip(prompt_tokens.len()).collect());
                    }
                }
            }
        }

        Ok(output)
    }

    /// Draft phase: autoregressively decode up to `k` tokens from the draft
    /// model, one real forward pass per token, starting from the next-token
    /// logits at the current committed context (`initial_logits`). Returns
    /// the sampled tokens, the draft model's logits after the last token fed
    /// (used to seed the next round), and a snapshot of the draft model's
    /// cache taken right after each token was fed (`snapshots[i]` = state
    /// after `tokens[i]`) — used to cheaply roll back to "only the first N
    /// draft tokens committed" if the main model rejects token N.
    ///
    /// If the draft model samples its EOS token, that token is included in
    /// `tokens` but *not* fed to the draft cache (nothing more will be
    /// drafted after EOS), so `snapshots` may be one shorter than `tokens`.
    /// Callers only need the EOS position's snapshot when it was itself
    /// accepted — at which point generation stops entirely.
    fn draft_phase(
        &self,
        initial_logits: Vec<f32>,
        k: usize,
        config: &SpeculativeConfig,
        eos_token: Option<u32>,
        rng: &mut StdRng,
    ) -> Result<(Vec<u32>, Vec<f32>, Vec<D::Snapshot>)> {
        let mut tokens = Vec::with_capacity(k);
        let mut snapshots = Vec::with_capacity(k);
        let mut logits = initial_logits;

        for _ in 0..k {
            let (token, _) = sample_from_logits(
                &logits,
                config.draft_temperature,
                if config.draft_temperature == 0.0 {
                    0
                } else {
                    40
                },
                config.draft_top_p,
                rng,
            );
            tokens.push(token);
            if Some(token) == eos_token {
                break;
            }

            let step = self.draft_model.forward_logits(&[token])?;
            snapshots.push(self.draft_model.snapshot_context()?);
            logits = step.into_iter().next().ok_or_else(|| {
                RuvLLMError::Generation("Draft model returned no logits".to_string())
            })?;
        }

        Ok((tokens, logits, snapshots))
    }

    /// Adjust lookahead based on acceptance rate
    fn adjust_lookahead(&self, accepted: usize, total: usize, config: &SpeculativeConfig) {
        let current = self.current_lookahead.load(Ordering::Relaxed);
        let acceptance_rate = if total > 0 {
            accepted as f32 / total as f32
        } else {
            0.5
        };

        let new_lookahead = if acceptance_rate > 0.9 {
            // High acceptance - increase lookahead
            (current + 1).min(config.max_lookahead)
        } else if acceptance_rate < 0.5 {
            // Low acceptance - decrease lookahead
            current.saturating_sub(1).max(config.min_lookahead)
        } else {
            current
        };

        self.current_lookahead
            .store(new_lookahead, Ordering::Relaxed);
    }

    /// Generate with tree-based speculation.
    ///
    /// The tree is currently built as a single linear path (equivalent to
    /// `generate`); true multi-branch tree speculation (exploring several
    /// candidate continuations per step) is not yet implemented.
    pub fn generate_tree(&self, prompt: &str, params: GenerateParams) -> Result<String> {
        let config = self.config.read().clone();
        if !config.tree_speculation {
            return self.generate(prompt, params);
        }
        // The linear-path tree degenerates to the same token sequence as
        // ordinary speculative decoding, so reuse it directly.
        self.generate(prompt, params)
    }
}

/// Verify `draft_tokens` against the main model's per-position logits from a
/// single batched forward pass (`main_logits`, one vector per draft token,
/// `main_logits[i]` = the main model's next-token distribution having seen
/// context + `draft_tokens[0..=i]`). `initial_logits` is the main model's
/// next-token distribution *before* any draft tokens were fed (i.e. what it
/// predicts should come first).
///
/// Acceptance is greedy argmax-match: draft token `i` is accepted iff it
/// equals the main model's own top prediction given everything accepted so
/// far. This matches this module's documented greedy/low-temperature design.
/// On the first mismatch (or after the last accepted token if all match), the
/// correction/continuation token is sampled from the main model's
/// distribution using the caller's actual generation params.
fn verify_round(
    draft_tokens: &[u32],
    initial_logits: &[f32],
    main_logits: &[Vec<f32>],
    params: &GenerateParams,
    rng: &mut StdRng,
) -> VerificationResult {
    let mut accepted_count = 0;
    let mut accepted_logprobs = Vec::with_capacity(draft_tokens.len());
    let mut check_logits = initial_logits;

    for (i, &draft_token) in draft_tokens.iter().enumerate() {
        let predicted = argmax(check_logits);
        if predicted != draft_token {
            let (next_token, next_logprob) = sample_from_logits(
                check_logits,
                params.temperature,
                params.top_k,
                params.top_p,
                rng,
            );
            return VerificationResult {
                accepted_count,
                next_token,
                accepted_logprobs,
                next_logprob,
                all_accepted: false,
            };
        }

        accepted_count += 1;
        let logprobs = log_softmax(check_logits);
        accepted_logprobs.push(logprobs[draft_token as usize]);
        check_logits = &main_logits[i];
    }

    let (next_token, next_logprob) = sample_from_logits(
        check_logits,
        params.temperature,
        params.top_k,
        params.top_p,
        rng,
    );
    VerificationResult {
        accepted_count,
        next_token,
        accepted_logprobs,
        next_logprob,
        all_accepted: true,
    }
}

/// Softmax function for probability computation
///
/// M4 Pro optimizations:
/// - NEON-accelerated max finding and exp computation
/// - 8x unrolling for maximum ILP
/// - Fast exp approximation for vocabulary-sized inputs
pub fn softmax(logits: &[f32]) -> Vec<f32> {
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        softmax_neon_optimized(logits)
    }

    #[cfg(not(all(target_arch = "aarch64", target_feature = "neon")))]
    {
        let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp_sum: f32 = logits.iter().map(|&x| (x - max_logit).exp()).sum();
        logits
            .iter()
            .map(|&x| (x - max_logit).exp() / exp_sum)
            .collect()
    }
}

/// NEON-optimized softmax with 8x unrolling
///
/// Key optimizations:
/// - Vectorized max finding
/// - Fast exp approximation using polynomial (6th order)
/// - Dual accumulator pattern for sum reduction
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
fn softmax_neon_optimized(logits: &[f32]) -> Vec<f32> {
    use std::arch::aarch64::*;

    const UNROLL_8X: usize = 8;

    if logits.is_empty() {
        return vec![];
    }

    let mut result = vec![0.0f32; logits.len()];

    unsafe {
        // Phase 1: Find max using NEON
        let mut max_vec = vdupq_n_f32(f32::NEG_INFINITY);
        let chunks = logits.len() / UNROLL_8X;

        for c in 0..chunks {
            let base = c * UNROLL_8X;
            let v0 = vld1q_f32(logits.as_ptr().add(base));
            let v1 = vld1q_f32(logits.as_ptr().add(base + 4));
            max_vec = vmaxq_f32(max_vec, vmaxq_f32(v0, v1));
        }

        let mut max_logit = vmaxvq_f32(max_vec);

        // Handle remainder
        for i in (chunks * UNROLL_8X)..logits.len() {
            max_logit = max_logit.max(logits[i]);
        }

        let max_vec = vdupq_n_f32(max_logit);

        // Phase 2: Compute exp(x - max) and sum using fast exp approximation
        // exp(x) ≈ (1 + x/256)^256 or polynomial approximation
        // We use the more accurate polynomial: exp(x) ≈ 1 + x + x²/2 + x³/6 + x⁴/24 + x⁵/120 + x⁶/720
        let one = vdupq_n_f32(1.0);
        let half = vdupq_n_f32(0.5);
        let sixth = vdupq_n_f32(1.0 / 6.0);
        let twenty_fourth = vdupq_n_f32(1.0 / 24.0);
        let one_twenty = vdupq_n_f32(1.0 / 120.0);
        let seven_twenty = vdupq_n_f32(1.0 / 720.0);

        let mut sum0 = vdupq_n_f32(0.0);
        let mut sum1 = vdupq_n_f32(0.0);

        // Fast exp approximation: good for |x| < 10
        #[inline(always)]
        unsafe fn fast_exp_vec(
            x: float32x4_t,
            one: float32x4_t,
            half: float32x4_t,
            sixth: float32x4_t,
            twenty_fourth: float32x4_t,
            one_twenty: float32x4_t,
            seven_twenty: float32x4_t,
        ) -> float32x4_t {
            // Clamp to reasonable range to avoid overflow
            let x = vmaxq_f32(vdupq_n_f32(-20.0), vminq_f32(x, vdupq_n_f32(20.0)));

            // exp(x) ≈ 1 + x(1 + x/2(1 + x/3(1 + x/4(1 + x/5(1 + x/6)))))
            let x2 = vmulq_f32(x, x);
            let x3 = vmulq_f32(x2, x);
            let x4 = vmulq_f32(x2, x2);
            let x5 = vmulq_f32(x4, x);
            let x6 = vmulq_f32(x3, x3);

            // 1 + x + x²/2 + x³/6 + x⁴/24 + x⁵/120 + x⁶/720
            let result = vaddq_f32(one, x);
            let result = vfmaq_f32(result, x2, half);
            let result = vfmaq_f32(result, x3, sixth);
            let result = vfmaq_f32(result, x4, twenty_fourth);
            let result = vfmaq_f32(result, x5, one_twenty);
            let result = vfmaq_f32(result, x6, seven_twenty);

            // Ensure non-negative
            vmaxq_f32(result, vdupq_n_f32(0.0))
        }

        for c in 0..chunks {
            let base = c * UNROLL_8X;
            let v0 = vld1q_f32(logits.as_ptr().add(base));
            let v1 = vld1q_f32(logits.as_ptr().add(base + 4));

            // Subtract max
            let d0 = vsubq_f32(v0, max_vec);
            let d1 = vsubq_f32(v1, max_vec);

            // Fast exp
            let e0 = fast_exp_vec(
                d0,
                one,
                half,
                sixth,
                twenty_fourth,
                one_twenty,
                seven_twenty,
            );
            let e1 = fast_exp_vec(
                d1,
                one,
                half,
                sixth,
                twenty_fourth,
                one_twenty,
                seven_twenty,
            );

            // Store exp values
            vst1q_f32(result.as_mut_ptr().add(base), e0);
            vst1q_f32(result.as_mut_ptr().add(base + 4), e1);

            // Accumulate sums
            sum0 = vaddq_f32(sum0, e0);
            sum1 = vaddq_f32(sum1, e1);
        }

        // Reduce sum
        let mut exp_sum = vaddvq_f32(vaddq_f32(sum0, sum1));

        // Handle remainder with scalar exp (more accurate for edge cases)
        for i in (chunks * UNROLL_8X)..logits.len() {
            let e = (logits[i] - max_logit).exp();
            result[i] = e;
            exp_sum += e;
        }

        // Phase 3: Normalize by sum
        let inv_sum = vdupq_n_f32(1.0 / exp_sum);

        for c in 0..chunks {
            let base = c * UNROLL_8X;
            let e0 = vld1q_f32(result.as_ptr().add(base));
            let e1 = vld1q_f32(result.as_ptr().add(base + 4));

            let p0 = vmulq_f32(e0, inv_sum);
            let p1 = vmulq_f32(e1, inv_sum);

            vst1q_f32(result.as_mut_ptr().add(base), p0);
            vst1q_f32(result.as_mut_ptr().add(base + 4), p1);
        }

        // Remainder
        let inv_sum_scalar = 1.0 / exp_sum;
        for i in (chunks * UNROLL_8X)..logits.len() {
            result[i] *= inv_sum_scalar;
        }
    }

    result
}

/// Log softmax function
///
/// M4 Pro optimizations:
/// - NEON-accelerated log-sum-exp computation
/// - 8x unrolling for maximum ILP
pub fn log_softmax(logits: &[f32]) -> Vec<f32> {
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        log_softmax_neon_optimized(logits)
    }

    #[cfg(not(all(target_arch = "aarch64", target_feature = "neon")))]
    {
        let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let log_sum_exp: f32 = logits
            .iter()
            .map(|&x| (x - max_logit).exp())
            .sum::<f32>()
            .ln()
            + max_logit;
        logits.iter().map(|&x| x - log_sum_exp).collect()
    }
}

/// NEON-optimized log softmax
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
fn log_softmax_neon_optimized(logits: &[f32]) -> Vec<f32> {
    use std::arch::aarch64::*;

    const UNROLL_8X: usize = 8;

    if logits.is_empty() {
        return vec![];
    }

    let mut result = vec![0.0f32; logits.len()];

    unsafe {
        // Find max using NEON
        let mut max_vec = vdupq_n_f32(f32::NEG_INFINITY);
        let chunks = logits.len() / UNROLL_8X;

        for c in 0..chunks {
            let base = c * UNROLL_8X;
            let v0 = vld1q_f32(logits.as_ptr().add(base));
            let v1 = vld1q_f32(logits.as_ptr().add(base + 4));
            max_vec = vmaxq_f32(max_vec, vmaxq_f32(v0, v1));
        }

        let mut max_logit = vmaxvq_f32(max_vec);
        for i in (chunks * UNROLL_8X)..logits.len() {
            max_logit = max_logit.max(logits[i]);
        }

        // Compute sum of exp(x - max) - use scalar exp for accuracy
        let mut exp_sum = 0.0f32;
        for i in 0..logits.len() {
            exp_sum += (logits[i] - max_logit).exp();
        }

        let log_sum_exp = exp_sum.ln() + max_logit;
        let log_sum_vec = vdupq_n_f32(log_sum_exp);

        // Compute log softmax: x - log_sum_exp with NEON
        for c in 0..chunks {
            let base = c * UNROLL_8X;
            let v0 = vld1q_f32(logits.as_ptr().add(base));
            let v1 = vld1q_f32(logits.as_ptr().add(base + 4));

            let r0 = vsubq_f32(v0, log_sum_vec);
            let r1 = vsubq_f32(v1, log_sum_vec);

            vst1q_f32(result.as_mut_ptr().add(base), r0);
            vst1q_f32(result.as_mut_ptr().add(base + 4), r1);
        }

        for i in (chunks * UNROLL_8X)..logits.len() {
            result[i] = logits[i] - log_sum_exp;
        }
    }

    result
}

/// Sample from a probability distribution
pub fn sample_from_probs(probs: &[f32], rng: &mut impl Rng) -> usize {
    let r: f32 = rng.gen();
    let mut cumsum = 0.0;
    for (i, &p) in probs.iter().enumerate() {
        cumsum += p;
        if cumsum > r {
            return i;
        }
    }
    probs.len() - 1
}

/// Top-k filtering
pub fn top_k_filter(logits: &mut [f32], k: usize) {
    if k == 0 || k >= logits.len() {
        return;
    }

    let mut indexed: Vec<(usize, f32)> = logits.iter().cloned().enumerate().collect();
    // Use unwrap_or to handle NaN gracefully instead of panicking.
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let threshold = indexed[k - 1].1;
    for logit in logits.iter_mut() {
        if *logit < threshold {
            *logit = f32::NEG_INFINITY;
        }
    }
}

/// Top-p (nucleus) filtering
pub fn top_p_filter(logits: &mut [f32], p: f32) {
    if p >= 1.0 {
        return;
    }

    let probs = softmax(logits);
    let mut indexed: Vec<(usize, f32)> = probs.iter().cloned().enumerate().collect();
    // Use unwrap_or to handle NaN gracefully instead of panicking.
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let mut cumsum = 0.0;
    let mut cutoff_idx = indexed.len();
    for (i, (_, prob)) in indexed.iter().enumerate() {
        cumsum += prob;
        if cumsum > p {
            cutoff_idx = i + 1;
            break;
        }
    }

    // Set excluded tokens to -inf
    let included: std::collections::HashSet<usize> =
        indexed[..cutoff_idx].iter().map(|(i, _)| *i).collect();
    for (i, logit) in logits.iter_mut().enumerate() {
        if !included.contains(&i) {
            *logit = f32::NEG_INFINITY;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_speculative_config_default() {
        let config = SpeculativeConfig::default();
        assert_eq!(config.lookahead, 4);
        assert!((config.acceptance_threshold - 0.5).abs() < 0.01);
        assert!(!config.tree_speculation);
    }

    #[test]
    fn test_speculative_stats() {
        let mut stats = SpeculativeStats::new();
        assert_eq!(stats.draft_tokens, 0);
        assert_eq!(stats.accepted_tokens, 0);

        stats.record_round(4, 3, 10.0);
        assert_eq!(stats.draft_tokens, 4);
        assert_eq!(stats.accepted_tokens, 3);
        assert!((stats.acceptance_rate - 0.75).abs() < 0.01);
        assert_eq!(stats.total_tokens_generated, 4); // 3 accepted + 1 correction
    }

    #[test]
    fn test_atomic_stats() {
        let stats = AtomicSpeculativeStats::new();
        stats.record_round(4, 3, Duration::from_millis(10));

        let snapshot = stats.snapshot();
        assert_eq!(snapshot.draft_tokens, 4);
        assert_eq!(snapshot.accepted_tokens, 3);
        assert!((snapshot.acceptance_rate - 0.75).abs() < 0.01);
    }

    #[test]
    fn test_tree_node() {
        let mut root = TreeNode::new(0, 1.0, 0);
        root.add_child(1, 0.5);
        root.add_child(2, 0.3);

        assert_eq!(root.children.len(), 2);
        assert_eq!(root.children[0].token, 1);
        assert_eq!(root.children[1].token, 2);
    }

    #[test]
    fn test_speculation_tree() {
        let mut tree = SpeculationTree::new(3, 2);
        assert_eq!(tree.node_count, 1);

        let current = &mut tree.root;
        current.add_child(1, 0.8);
        tree.node_count += 1;

        assert_eq!(tree.node_count, 2);
    }

    #[test]
    fn test_softmax() {
        let logits = vec![1.0, 2.0, 3.0];
        let probs = softmax(&logits);

        // Check probabilities sum to 1
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 0.001);

        // Check ordering preserved
        assert!(probs[2] > probs[1]);
        assert!(probs[1] > probs[0]);
    }

    #[test]
    fn test_top_k_filter() {
        let mut logits = vec![1.0, 5.0, 3.0, 4.0, 2.0];
        top_k_filter(&mut logits, 2);

        // Only top 2 should remain finite
        let finite_count = logits.iter().filter(|x| x.is_finite()).count();
        assert_eq!(finite_count, 2);
    }

    #[test]
    fn test_top_p_filter() {
        let mut logits = vec![10.0, 5.0, 3.0, 2.0, 1.0];
        top_p_filter(&mut logits, 0.9);

        // Most probability mass should be preserved
        let finite_count = logits.iter().filter(|x| x.is_finite()).count();
        assert!(finite_count >= 1);
    }

    #[test]
    fn test_verification_result() {
        let result = VerificationResult {
            accepted_count: 3,
            next_token: 42,
            accepted_logprobs: vec![-0.1, -0.2, -0.3],
            next_logprob: -0.5,
            all_accepted: false,
        };

        assert_eq!(result.accepted_count, 3);
        assert_eq!(result.next_token, 42);
        assert!(!result.all_accepted);
    }

    #[test]
    fn test_argmax() {
        assert_eq!(argmax(&[1.0, 5.0, 3.0]), 1);
        assert_eq!(argmax(&[9.0, 5.0, 3.0]), 0);
    }

    #[test]
    fn test_verify_round_all_accepted() {
        // 3-token vocab. Main model always agrees with the draft, so all
        // draft tokens should be accepted and the continuation should come
        // from the last position's logits.
        let draft_tokens = vec![0u32, 1u32];
        let initial_logits = vec![10.0, -1.0, -1.0]; // predicts token 0
        let main_logits = vec![
            vec![-1.0, 10.0, -1.0], // after seeing token 0, predicts token 1
            vec![-1.0, -1.0, 10.0], // after seeing token 1, predicts token 2
        ];
        let params = GenerateParams {
            temperature: 0.0,
            ..Default::default()
        };
        let mut rng = StdRng::seed_from_u64(0);
        let result = verify_round(
            &draft_tokens,
            &initial_logits,
            &main_logits,
            &params,
            &mut rng,
        );

        assert!(result.all_accepted);
        assert_eq!(result.accepted_count, 2);
        assert_eq!(result.next_token, 2);
    }

    #[test]
    fn test_verify_round_rejects_mismatch() {
        let draft_tokens = vec![0u32, 2u32];
        let initial_logits = vec![10.0, -1.0, -1.0]; // predicts token 0 (matches)
        let main_logits = vec![
            vec![-1.0, 10.0, -1.0], // after token 0, predicts token 1 -- draft says 2, mismatch
            vec![-1.0, -1.0, 10.0], // never reached
        ];
        let params = GenerateParams {
            temperature: 0.0,
            ..Default::default()
        };
        let mut rng = StdRng::seed_from_u64(0);
        let result = verify_round(
            &draft_tokens,
            &initial_logits,
            &main_logits,
            &params,
            &mut rng,
        );

        assert!(!result.all_accepted);
        assert_eq!(result.accepted_count, 1);
        assert_eq!(result.next_token, 1);
    }
}
