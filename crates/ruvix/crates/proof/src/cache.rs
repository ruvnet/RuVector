//! Proof cache with security requirements from ADR-087 Section 20.4.
//!
//! The proof cache provides:
//! - 100ms TTL (time-to-live) for cached entries
//! - Single-use nonce consumption (removed on first verification)
//! - Maximum 64 entries to bound memory usage
//! - Scoped to (mutation_hash, nonce) pairs

use crate::error::{ProofError, ProofResult};
use crate::{DEFAULT_CACHE_TTL_NS, MAX_CACHE_ENTRIES};
use ruvix_types::ProofTier;

/// Configuration for the proof cache.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProofCacheConfig {
    /// Maximum number of entries in the cache.
    pub max_entries: usize,
    /// TTL for cache entries in nanoseconds.
    pub ttl_ns: u64,
    /// Whether to enable cache eviction on TTL expiry.
    pub evict_expired: bool,
}

impl Default for ProofCacheConfig {
    fn default() -> Self {
        Self {
            max_entries: MAX_CACHE_ENTRIES,
            ttl_ns: DEFAULT_CACHE_TTL_NS,
            evict_expired: true,
        }
    }
}

/// A cache entry storing proof verification state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CacheEntry {
    /// Hash of the mutation (first 8 bytes for compact storage).
    pub mutation_hash_prefix: u64,
    /// Nonce for this proof.
    pub nonce: u64,
    /// Proof tier.
    pub tier: ProofTier,
    /// Timestamp when the entry was created (ns since epoch).
    pub created_at_ns: u64,
    /// Timestamp when the entry expires (ns since epoch).
    pub expires_at_ns: u64,
    /// Whether this entry has been consumed.
    pub consumed: bool,
}

impl CacheEntry {
    /// Creates a new cache entry.
    #[inline]
    #[must_use]
    pub const fn new(
        mutation_hash_prefix: u64,
        nonce: u64,
        tier: ProofTier,
        created_at_ns: u64,
        ttl_ns: u64,
    ) -> Self {
        Self {
            mutation_hash_prefix,
            nonce,
            tier,
            created_at_ns,
            expires_at_ns: created_at_ns.saturating_add(ttl_ns),
            consumed: false,
        }
    }

    /// Checks if this entry has expired.
    #[inline]
    #[must_use]
    pub const fn is_expired(&self, current_time_ns: u64) -> bool {
        current_time_ns > self.expires_at_ns
    }

    /// Checks if this entry matches the given key.
    #[inline]
    #[must_use]
    pub const fn matches(&self, mutation_hash_prefix: u64, nonce: u64) -> bool {
        self.mutation_hash_prefix == mutation_hash_prefix && self.nonce == nonce
    }
}

/// Proof cache with single-use nonce semantics.
///
/// The cache stores recently generated proofs for fast verification.
/// Each (mutation_hash, nonce) pair can only be used once.
#[derive(Debug)]
pub struct ProofCache {
    /// Fixed-size array of cache entries.
    entries: [Option<CacheEntry>; MAX_CACHE_ENTRIES],
    /// Number of active entries.
    count: usize,
    /// Next slot to use for insertion (circular buffer).
    next_slot: usize,
    /// Configuration.
    config: ProofCacheConfig,
    /// Statistics.
    stats: CacheStats,
}

/// Cache statistics.
#[derive(Debug, Clone, Copy, Default)]
pub struct CacheStats {
    /// Total cache hits.
    pub hits: u64,
    /// Total cache misses.
    pub misses: u64,
    /// Total entries inserted.
    pub insertions: u64,
    /// Total entries evicted (TTL or capacity).
    pub evictions: u64,
    /// Total nonces consumed.
    pub nonces_consumed: u64,
}

impl ProofCache {
    /// Creates a new proof cache with default configuration.
    #[must_use]
    pub fn new() -> Self {
        Self::with_config(ProofCacheConfig::default())
    }

    /// Creates a new proof cache with custom configuration.
    #[must_use]
    pub fn with_config(config: ProofCacheConfig) -> Self {
        Self {
            entries: [None; MAX_CACHE_ENTRIES],
            count: 0,
            next_slot: 0,
            config,
            stats: CacheStats::default(),
        }
    }

    /// Returns the current number of entries.
    #[inline]
    #[must_use]
    pub const fn len(&self) -> usize {
        self.count
    }

    /// Returns true if the cache is empty.
    #[inline]
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Returns the cache statistics.
    #[inline]
    #[must_use]
    pub const fn stats(&self) -> &CacheStats {
        &self.stats
    }

    /// Extracts a 64-bit prefix from a 32-byte hash.
    #[inline]
    #[must_use]
    pub fn hash_prefix(mutation_hash: &[u8; 32]) -> u64 {
        u64::from_le_bytes([
            mutation_hash[0],
            mutation_hash[1],
            mutation_hash[2],
            mutation_hash[3],
            mutation_hash[4],
            mutation_hash[5],
            mutation_hash[6],
            mutation_hash[7],
        ])
    }

    /// Inserts a new proof into the cache.
    ///
    /// Returns an error if the cache is full and no expired entries can be evicted.
    pub fn insert(
        &mut self,
        mutation_hash: &[u8; 32],
        nonce: u64,
        tier: ProofTier,
        current_time_ns: u64,
    ) -> ProofResult<()> {
        let prefix = Self::hash_prefix(mutation_hash);

        // Try to find an empty slot or expired entry
        if self.config.evict_expired {
            self.evict_expired(current_time_ns);
        }

        // Find insertion slot
        let slot = if self.count < self.config.max_entries {
            // Find first empty slot
            let mut found = None;
            for (i, entry) in self.entries.iter().enumerate() {
                if entry.is_none() {
                    found = Some(i);
                    break;
                }
            }
            found.unwrap_or(self.next_slot)
        } else {
            // Cache is full, use circular buffer
            let slot = self.next_slot;
            self.next_slot = (self.next_slot + 1) % self.config.max_entries;
            if self.entries[slot].is_some() {
                self.stats.evictions += 1;
                self.count = self.count.saturating_sub(1);
            }
            slot
        };

        let entry = CacheEntry::new(prefix, nonce, tier, current_time_ns, self.config.ttl_ns);
        self.entries[slot] = Some(entry);
        self.count += 1;
        self.stats.insertions += 1;

        Ok(())
    }

    /// Looks up and consumes a proof from the cache.
    ///
    /// This is a single-use operation: the nonce is removed on first use.
    /// Returns the cache entry if found and valid, or an error.
    pub fn lookup_and_consume(
        &mut self,
        mutation_hash: &[u8; 32],
        nonce: u64,
        current_time_ns: u64,
    ) -> ProofResult<CacheEntry> {
        let prefix = Self::hash_prefix(mutation_hash);

        for entry in &mut self.entries {
            if let Some(ref mut e) = entry {
                if e.matches(prefix, nonce) {
                    // Check if already consumed
                    if e.consumed {
                        self.stats.misses += 1;
                        return Err(ProofError::NonceReused { nonce });
                    }

                    // Check expiry
                    if e.is_expired(current_time_ns) {
                        self.stats.misses += 1;
                        return Err(ProofError::Expired {
                            valid_until: e.expires_at_ns,
                            current_time: current_time_ns,
                        });
                    }

                    // Mark as consumed and return
                    e.consumed = true;
                    self.stats.hits += 1;
                    self.stats.nonces_consumed += 1;
                    return Ok(*e);
                }
            }
        }

        self.stats.misses += 1;
        Err(ProofError::InternalError { code: 0x01 }) // Not found
    }

    /// Checks if a nonce has been consumed without consuming it.
    #[must_use]
    pub fn is_nonce_consumed(&self, mutation_hash: &[u8; 32], nonce: u64) -> bool {
        let prefix = Self::hash_prefix(mutation_hash);

        for entry in &self.entries {
            if let Some(ref e) = entry {
                if e.matches(prefix, nonce) {
                    return e.consumed;
                }
            }
        }

        false
    }

    /// Evicts expired entries from the cache.
    pub fn evict_expired(&mut self, current_time_ns: u64) {
        for entry in &mut self.entries {
            if let Some(ref e) = entry {
                if e.is_expired(current_time_ns) {
                    *entry = None;
                    self.count = self.count.saturating_sub(1);
                    self.stats.evictions += 1;
                }
            }
        }
    }

    /// Clears all entries from the cache.
    pub fn clear(&mut self) {
        for entry in &mut self.entries {
            *entry = None;
        }
        self.count = 0;
        self.next_slot = 0;
    }
}

impl Default for ProofCache {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_entry_new() {
        let entry = CacheEntry::new(0x1234, 1, ProofTier::Reflex, 1000, 100_000_000);
        assert_eq!(entry.mutation_hash_prefix, 0x1234);
        assert_eq!(entry.nonce, 1);
        assert_eq!(entry.tier, ProofTier::Reflex);
        assert_eq!(entry.created_at_ns, 1000);
        assert_eq!(entry.expires_at_ns, 100_001_000);
        assert!(!entry.consumed);
    }

    #[test]
    fn test_cache_entry_expiry() {
        let entry = CacheEntry::new(0x1234, 1, ProofTier::Reflex, 1000, 100);
        assert!(!entry.is_expired(1000));
        assert!(!entry.is_expired(1100));
        assert!(entry.is_expired(1101));
    }

    #[test]
    fn test_cache_entry_matches() {
        let entry = CacheEntry::new(0x1234, 42, ProofTier::Reflex, 0, 100);
        assert!(entry.matches(0x1234, 42));
        assert!(!entry.matches(0x1234, 43));
        assert!(!entry.matches(0x5678, 42));
    }

    #[test]
    fn test_proof_cache_insert() {
        let mut cache = ProofCache::new();
        let hash = [0u8; 32];

        cache.insert(&hash, 1, ProofTier::Reflex, 1000).unwrap();
        assert_eq!(cache.len(), 1);
        assert_eq!(cache.stats().insertions, 1);
    }

    #[test]
    fn test_proof_cache_lookup_and_consume() {
        let mut cache = ProofCache::new();
        let hash = [0u8; 32];

        cache.insert(&hash, 1, ProofTier::Reflex, 1000).unwrap();

        // First lookup succeeds
        let entry = cache.lookup_and_consume(&hash, 1, 1000).unwrap();
        assert_eq!(entry.nonce, 1);
        assert_eq!(cache.stats().hits, 1);
        assert_eq!(cache.stats().nonces_consumed, 1);

        // Second lookup fails (nonce reused)
        let err = cache.lookup_and_consume(&hash, 1, 1000).unwrap_err();
        assert!(matches!(err, ProofError::NonceReused { nonce: 1 }));
    }

    #[test]
    fn test_proof_cache_expired_lookup() {
        let mut cache = ProofCache::with_config(ProofCacheConfig {
            ttl_ns: 100,
            ..Default::default()
        });
        let hash = [0u8; 32];

        cache.insert(&hash, 1, ProofTier::Reflex, 1000).unwrap();

        // Lookup after TTL fails
        let err = cache.lookup_and_consume(&hash, 1, 1200).unwrap_err();
        assert!(matches!(err, ProofError::Expired { .. }));
    }

    #[test]
    fn test_proof_cache_evict_expired() {
        let mut cache = ProofCache::with_config(ProofCacheConfig {
            ttl_ns: 100,
            ..Default::default()
        });
        let hash = [0u8; 32];

        cache.insert(&hash, 1, ProofTier::Reflex, 1000).unwrap();
        assert_eq!(cache.len(), 1);

        cache.evict_expired(1200);
        assert_eq!(cache.len(), 0);
        assert_eq!(cache.stats().evictions, 1);
    }

    #[test]
    fn test_proof_cache_max_entries() {
        let mut cache = ProofCache::with_config(ProofCacheConfig {
            max_entries: 4,
            ttl_ns: 1_000_000,
            evict_expired: false,
        });

        for i in 0..6u64 {
            let mut hash = [0u8; 32];
            hash[0..8].copy_from_slice(&i.to_le_bytes());
            cache.insert(&hash, i, ProofTier::Reflex, 1000).unwrap();
        }

        // Should have evicted oldest entries
        assert!(cache.len() <= 4);
        assert!(cache.stats().evictions >= 2);
    }

    #[test]
    fn test_proof_cache_clear() {
        let mut cache = ProofCache::new();
        let hash = [0u8; 32];

        cache.insert(&hash, 1, ProofTier::Reflex, 1000).unwrap();
        cache.insert(&hash, 2, ProofTier::Standard, 1000).unwrap();

        cache.clear();
        assert!(cache.is_empty());
    }

    #[test]
    fn test_is_nonce_consumed() {
        let mut cache = ProofCache::new();
        let hash = [0u8; 32];

        cache.insert(&hash, 1, ProofTier::Reflex, 1000).unwrap();

        assert!(!cache.is_nonce_consumed(&hash, 1));
        let _ = cache.lookup_and_consume(&hash, 1, 1000);
        assert!(cache.is_nonce_consumed(&hash, 1));
    }

    #[test]
    fn test_hash_prefix() {
        let mut hash = [0u8; 32];
        hash[0..8].copy_from_slice(&0x0102030405060708u64.to_le_bytes());

        let prefix = ProofCache::hash_prefix(&hash);
        assert_eq!(prefix, 0x0102030405060708);
    }
}
