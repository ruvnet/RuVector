//! Proof engine for generating proof tokens.
//!
//! The proof engine generates cryptographically-bound proof tokens
//! for mutations based on the appropriate tier.

use crate::cache::ProofCache;
use crate::error::ProofResult;
use crate::routing::{RoutingContext, TierRouter};
use crate::{DEFAULT_CACHE_TTL_NS, DEFAULT_VALIDITY_WINDOW_NS};
use ruvix_types::{ProofPayload, ProofTier, ProofToken};

/// Configuration for the proof engine.
#[derive(Debug, Clone, Copy)]
pub struct ProofEngineConfig {
    /// Default validity window for proof tokens in nanoseconds.
    pub validity_window_ns: u64,
    /// Cache TTL in nanoseconds.
    pub cache_ttl_ns: u64,
    /// Whether to enable proof caching.
    pub enable_cache: bool,
    /// Tier router configuration.
    pub router: TierRouter,
}

impl Default for ProofEngineConfig {
    fn default() -> Self {
        Self {
            validity_window_ns: DEFAULT_VALIDITY_WINDOW_NS,
            cache_ttl_ns: DEFAULT_CACHE_TTL_NS,
            enable_cache: true,
            router: TierRouter::default(),
        }
    }
}

/// Statistics for the proof engine.
#[derive(Debug, Clone, Copy, Default)]
pub struct ProofEngineStats {
    /// Total proofs generated.
    pub proofs_generated: u64,
    /// Reflex-tier proofs generated.
    pub reflex_proofs: u64,
    /// Standard-tier proofs generated.
    pub standard_proofs: u64,
    /// Deep-tier proofs generated.
    pub deep_proofs: u64,
    /// Proof generation failures.
    pub failures: u64,
}

/// The proof engine generates proof tokens for mutations.
#[derive(Debug)]
pub struct ProofEngine {
    /// Configuration.
    config: ProofEngineConfig,
    /// Proof cache.
    cache: ProofCache,
    /// Nonce counter (monotonically increasing).
    nonce_counter: u64,
    /// Statistics.
    stats: ProofEngineStats,
}

impl ProofEngine {
    /// Creates a new proof engine with default configuration.
    #[must_use]
    pub fn new(config: ProofEngineConfig) -> Self {
        Self {
            config,
            cache: ProofCache::new(),
            nonce_counter: 0,
            stats: ProofEngineStats::default(),
        }
    }

    /// Returns the engine configuration.
    #[inline]
    #[must_use]
    pub const fn config(&self) -> &ProofEngineConfig {
        &self.config
    }

    /// Returns the engine statistics.
    #[inline]
    #[must_use]
    pub const fn stats(&self) -> &ProofEngineStats {
        &self.stats
    }

    /// Returns a mutable reference to the proof cache.
    #[inline]
    pub fn cache_mut(&mut self) -> &mut ProofCache {
        &mut self.cache
    }

    /// Allocates a new nonce value.
    #[inline]
    fn alloc_nonce(&mut self) -> u64 {
        let nonce = self.nonce_counter;
        self.nonce_counter = self.nonce_counter.wrapping_add(1);
        nonce
    }

    /// Generates a proof token with automatic tier routing.
    ///
    /// # Arguments
    ///
    /// * `mutation_hash` - Hash of the mutation data
    /// * `routing_ctx` - Routing context for tier selection
    /// * `current_time_ns` - Current time in nanoseconds since epoch
    ///
    /// # Returns
    ///
    /// A proof token authorized for the mutation.
    pub fn generate(
        &mut self,
        mutation_hash: &[u8; 32],
        routing_ctx: &RoutingContext,
        current_time_ns: u64,
    ) -> ProofResult<ProofToken> {
        let tier = self.config.router.route(routing_ctx);
        self.generate_for_tier(mutation_hash, tier, current_time_ns)
    }

    /// Generates a proof token for a specific tier.
    pub fn generate_for_tier(
        &mut self,
        mutation_hash: &[u8; 32],
        tier: ProofTier,
        current_time_ns: u64,
    ) -> ProofResult<ProofToken> {
        let nonce = self.alloc_nonce();
        let valid_until_ns = current_time_ns.saturating_add(self.config.validity_window_ns);

        let payload = match tier {
            ProofTier::Reflex => self.generate_reflex_payload(mutation_hash),
            ProofTier::Standard => self.generate_standard_payload(mutation_hash),
            ProofTier::Deep => self.generate_deep_payload(mutation_hash),
        };

        let token = ProofToken::new(*mutation_hash, tier, payload, valid_until_ns, nonce);

        // Cache the proof if enabled
        if self.config.enable_cache {
            self.cache
                .insert(mutation_hash, nonce, tier, current_time_ns)?;
        }

        // Update statistics
        self.stats.proofs_generated += 1;
        match tier {
            ProofTier::Reflex => self.stats.reflex_proofs += 1,
            ProofTier::Standard => self.stats.standard_proofs += 1,
            ProofTier::Deep => self.stats.deep_proofs += 1,
        }

        Ok(token)
    }

    /// Generates a Reflex-tier proof for high-frequency vector updates.
    ///
    /// Target latency: <100ns
    pub fn generate_reflex_proof(
        &mut self,
        mutation_hash: &[u8; 32],
        current_time_ns: u64,
    ) -> ProofResult<ProofToken> {
        self.generate_for_tier(mutation_hash, ProofTier::Reflex, current_time_ns)
    }

    /// Generates a Standard-tier proof with Merkle witness.
    ///
    /// Target latency: <100us
    pub fn generate_standard_proof(
        &mut self,
        mutation_hash: &[u8; 32],
        merkle_root: &[u8; 32],
        leaf_index: u32,
        merkle_path: &[[u8; 32]],
        current_time_ns: u64,
    ) -> ProofResult<ProofToken> {
        let nonce = self.alloc_nonce();
        let valid_until_ns = current_time_ns.saturating_add(self.config.validity_window_ns);

        // Construct Merkle witness payload
        let mut path = [[0u8; 32]; 32];
        let path_len = merkle_path.len().min(32);
        for (i, hash) in merkle_path.iter().take(path_len).enumerate() {
            path[i] = *hash;
        }

        let payload = ProofPayload::MerkleWitness {
            root: *merkle_root,
            leaf_index,
            path_len: path_len as u8,
            path,
        };

        let token = ProofToken::new(*mutation_hash, ProofTier::Standard, payload, valid_until_ns, nonce);

        if self.config.enable_cache {
            self.cache
                .insert(mutation_hash, nonce, ProofTier::Standard, current_time_ns)?;
        }

        self.stats.proofs_generated += 1;
        self.stats.standard_proofs += 1;

        Ok(token)
    }

    /// Generates a Deep-tier proof with coherence certificate.
    ///
    /// Target latency: <10ms
    pub fn generate_deep_proof(
        &mut self,
        mutation_hash: &[u8; 32],
        score_before: u16,
        score_after: u16,
        partition_id: u32,
        signature: &[u8; 64],
        current_time_ns: u64,
    ) -> ProofResult<ProofToken> {
        let nonce = self.alloc_nonce();
        let valid_until_ns = current_time_ns.saturating_add(self.config.validity_window_ns);

        let payload = ProofPayload::CoherenceCert {
            score_before,
            score_after,
            partition_id,
            signature: *signature,
        };

        let token = ProofToken::new(*mutation_hash, ProofTier::Deep, payload, valid_until_ns, nonce);

        if self.config.enable_cache {
            self.cache
                .insert(mutation_hash, nonce, ProofTier::Deep, current_time_ns)?;
        }

        self.stats.proofs_generated += 1;
        self.stats.deep_proofs += 1;

        Ok(token)
    }

    /// Generates a Reflex payload (hash-based).
    fn generate_reflex_payload(&self, mutation_hash: &[u8; 32]) -> ProofPayload {
        ProofPayload::Hash {
            hash: *mutation_hash,
        }
    }

    /// Generates a Standard payload (placeholder Merkle witness).
    fn generate_standard_payload(&self, mutation_hash: &[u8; 32]) -> ProofPayload {
        // For auto-generated proofs, create a minimal Merkle witness
        // with the mutation hash as both root and first path element
        let mut path = [[0u8; 32]; 32];
        path[0] = *mutation_hash;

        ProofPayload::MerkleWitness {
            root: *mutation_hash,
            leaf_index: 0,
            path_len: 1,
            path,
        }
    }

    /// Generates a Deep payload (placeholder coherence cert).
    fn generate_deep_payload(&self, _mutation_hash: &[u8; 32]) -> ProofPayload {
        // For auto-generated proofs, create a placeholder coherence cert
        ProofPayload::CoherenceCert {
            score_before: 10000, // 100% coherence
            score_after: 10000,
            partition_id: 0,
            signature: [0u8; 64],
        }
    }

    /// Resets the engine state (clears cache, resets stats).
    pub fn reset(&mut self) {
        self.cache.clear();
        self.stats = ProofEngineStats::default();
        // Note: nonce counter is NOT reset to prevent reuse
    }
}

impl Default for ProofEngine {
    fn default() -> Self {
        Self::new(ProofEngineConfig::default())
    }
}

/// Builder for constructing proof engines.
#[derive(Debug, Default)]
pub struct ProofEngineBuilder {
    config: ProofEngineConfig,
}

impl ProofEngineBuilder {
    /// Creates a new proof engine builder.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            config: ProofEngineConfig {
                validity_window_ns: DEFAULT_VALIDITY_WINDOW_NS,
                cache_ttl_ns: DEFAULT_CACHE_TTL_NS,
                enable_cache: true,
                router: TierRouter {
                    reflex_to_standard_threshold: 10,
                    standard_to_deep_threshold: 100,
                    allow_downgrade: false,
                },
            },
        }
    }

    /// Sets the validity window in nanoseconds.
    #[must_use]
    pub const fn validity_window_ns(mut self, window: u64) -> Self {
        self.config.validity_window_ns = window;
        self
    }

    /// Sets the cache TTL in nanoseconds.
    #[must_use]
    pub const fn cache_ttl_ns(mut self, ttl: u64) -> Self {
        self.config.cache_ttl_ns = ttl;
        self
    }

    /// Enables or disables proof caching.
    #[must_use]
    pub const fn enable_cache(mut self, enable: bool) -> Self {
        self.config.enable_cache = enable;
        self
    }

    /// Sets the tier router.
    #[must_use]
    pub const fn router(mut self, router: TierRouter) -> Self {
        self.config.router = router;
        self
    }

    /// Builds the proof engine.
    #[must_use]
    pub fn build(self) -> ProofEngine {
        ProofEngine::new(self.config)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::routing::{MutationType, RoutingContextBuilder};

    #[test]
    fn test_proof_engine_new() {
        let engine = ProofEngine::default();
        assert_eq!(engine.stats().proofs_generated, 0);
    }

    #[test]
    fn test_generate_reflex_proof() {
        let mut engine = ProofEngine::default();
        let hash = [0u8; 32];

        let token = engine.generate_reflex_proof(&hash, 1000).unwrap();
        assert_eq!(token.tier, ProofTier::Reflex);
        assert_eq!(token.mutation_hash, hash);
        assert!(!token.is_expired(1000));

        assert_eq!(engine.stats().reflex_proofs, 1);
        assert_eq!(engine.stats().proofs_generated, 1);
    }

    #[test]
    fn test_generate_standard_proof() {
        let mut engine = ProofEngine::default();
        let hash = [0u8; 32];
        let root = [1u8; 32];
        let path = vec![[2u8; 32], [3u8; 32]];

        let token = engine
            .generate_standard_proof(&hash, &root, 5, &path, 1000)
            .unwrap();

        assert_eq!(token.tier, ProofTier::Standard);

        if let ProofPayload::MerkleWitness {
            root: r,
            leaf_index,
            path_len,
            ..
        } = token.payload
        {
            assert_eq!(r, root);
            assert_eq!(leaf_index, 5);
            assert_eq!(path_len, 2);
        } else {
            panic!("Expected MerkleWitness payload");
        }

        assert_eq!(engine.stats().standard_proofs, 1);
    }

    #[test]
    fn test_generate_deep_proof() {
        let mut engine = ProofEngine::default();
        let hash = [0u8; 32];
        let sig = [0xABu8; 64];

        let token = engine
            .generate_deep_proof(&hash, 9500, 9600, 42, &sig, 1000)
            .unwrap();

        assert_eq!(token.tier, ProofTier::Deep);

        if let ProofPayload::CoherenceCert {
            score_before,
            score_after,
            partition_id,
            signature,
        } = token.payload
        {
            assert_eq!(score_before, 9500);
            assert_eq!(score_after, 9600);
            assert_eq!(partition_id, 42);
            assert_eq!(signature, sig);
        } else {
            panic!("Expected CoherenceCert payload");
        }

        assert_eq!(engine.stats().deep_proofs, 1);
    }

    #[test]
    fn test_generate_with_routing() {
        let mut engine = ProofEngine::default();
        let hash = [0u8; 32];

        let ctx = RoutingContextBuilder::new()
            .mutation_type(MutationType::VectorUpdate)
            .affected_nodes(1)
            .build();

        let token = engine.generate(&hash, &ctx, 1000).unwrap();
        assert_eq!(token.tier, ProofTier::Reflex);

        let ctx = RoutingContextBuilder::new()
            .mutation_type(MutationType::GraphEdge)
            .affected_nodes(50)
            .build();

        let token = engine.generate(&hash, &ctx, 1000).unwrap();
        assert_eq!(token.tier, ProofTier::Standard);
    }

    #[test]
    fn test_nonce_uniqueness() {
        let mut engine = ProofEngine::default();
        let hash = [0u8; 32];

        let token1 = engine.generate_reflex_proof(&hash, 1000).unwrap();
        let token2 = engine.generate_reflex_proof(&hash, 1000).unwrap();

        assert_ne!(token1.nonce, token2.nonce);
    }

    #[test]
    fn test_proof_validity_window() {
        let engine = ProofEngineBuilder::new()
            .validity_window_ns(1000)
            .build();

        assert_eq!(engine.config().validity_window_ns, 1000);
    }

    #[test]
    fn test_proof_caching() {
        let mut engine = ProofEngine::default();
        let hash = [0u8; 32];

        let token = engine.generate_reflex_proof(&hash, 1000).unwrap();

        // Token should be in cache
        let entry = engine
            .cache_mut()
            .lookup_and_consume(&hash, token.nonce, 1000)
            .unwrap();

        assert_eq!(entry.nonce, token.nonce);
    }

    #[test]
    fn test_disable_caching() {
        let mut engine = ProofEngineBuilder::new().enable_cache(false).build();

        let hash = [0u8; 32];
        let token = engine.generate_reflex_proof(&hash, 1000).unwrap();

        // Token should NOT be in cache
        let result = engine
            .cache_mut()
            .lookup_and_consume(&hash, token.nonce, 1000);

        assert!(result.is_err());
    }

    #[test]
    fn test_reset_preserves_nonces() {
        let mut engine = ProofEngine::default();
        let hash = [0u8; 32];

        let token1 = engine.generate_reflex_proof(&hash, 1000).unwrap();
        let nonce_before = token1.nonce;

        engine.reset();

        let token2 = engine.generate_reflex_proof(&hash, 1000).unwrap();
        assert!(token2.nonce > nonce_before);
    }

    #[test]
    fn test_builder_pattern() {
        let engine = ProofEngineBuilder::new()
            .validity_window_ns(50_000_000)
            .cache_ttl_ns(25_000_000)
            .enable_cache(true)
            .router(TierRouter {
                reflex_to_standard_threshold: 5,
                standard_to_deep_threshold: 50,
                allow_downgrade: true,
            })
            .build();

        assert_eq!(engine.config().validity_window_ns, 50_000_000);
        assert_eq!(engine.config().cache_ttl_ns, 25_000_000);
        assert!(engine.config().enable_cache);
        assert_eq!(engine.config().router.reflex_to_standard_threshold, 5);
    }
}
