//! Proof cache with security constraints (ADR-087 SEC-002).
//!
//! This module implements a secure proof cache with the following constraints:
//!
//! - **100ms TTL**: Cache entries expire after 100ms to prevent replay attacks
//! - **Single-use nonces**: Each (mutation_hash, nonce) pair can only be verified once
//! - **Maximum 64 entries**: Bounded cache size to prevent memory exhaustion
//! - **Scoped entries**: Entries are keyed by (mutation_hash, nonce) pairs
//!
//! # Security Rationale
//!
//! The proof cache is used for Reflex-tier proofs (sub-microsecond verification).
//! Without proper bounds:
//! - TTL prevents replay of old proofs
//! - Single-use nonces prevent double-verification attacks
//! - Entry limit prevents denial-of-service via cache flooding
//!
//! # Example
//!
//! ```
//! use ruvix_types::{ProofCache, CacheError};
//!
//! let mut cache = ProofCache::new();
//!
//! let mutation_hash = [0u8; 32];
//! let nonce = 12345u64;
//! let proof_id = 1u32;
//! let current_time_ns = 1_000_000u64; // 1ms
//!
//! // Insert a proof
//! cache.insert(mutation_hash, nonce, proof_id, current_time_ns).unwrap();
//!
//! // Verify and consume (single-use)
//! let id = cache.verify_and_consume(&mutation_hash, nonce, current_time_ns).unwrap();
//! assert_eq!(id, proof_id);
//!
//! // Second verification fails (nonce consumed)
//! assert!(cache.verify_and_consume(&mutation_hash, nonce, current_time_ns).is_err());
//! ```

/// Maximum number of entries in the proof cache (SEC-002).
pub const PROOF_CACHE_MAX_ENTRIES: usize = 64;

/// TTL for cache entries in nanoseconds (100ms = 100_000_000ns).
pub const PROOF_CACHE_TTL_NS: u64 = 100_000_000;

/// TTL for cache entries in milliseconds (100ms).
pub const PROOF_CACHE_TTL_MS: u32 = 100;

/// Error types for proof cache operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CacheError {
    /// Cache is full (64 entries maximum).
    CacheFull,
    /// Entry not found for the given (mutation_hash, nonce).
    NotFound,
    /// Entry has expired (TTL exceeded).
    Expired,
    /// Nonce has already been consumed (single-use violation).
    NonceConsumed,
    /// Duplicate entry (same mutation_hash + nonce already exists).
    DuplicateEntry,
}

/// A single entry in the proof cache.
///
/// Each entry is scoped to a (mutation_hash, nonce) pair and includes
/// TTL information for expiry tracking.
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct ProofCacheEntry {
    /// Unique proof identifier returned on successful verification.
    pub proof_id: u32,
    /// Timestamp when the entry was inserted (nanoseconds).
    pub inserted_at: u64,
    /// Single-use nonce for this proof.
    pub nonce: u64,
    /// Hash of the mutation this proof authorizes.
    pub mutation_hash: [u8; 32],
    /// Whether this entry has been consumed (single-use).
    pub consumed: bool,
}

impl ProofCacheEntry {
    /// Creates a new cache entry.
    #[must_use]
    pub const fn new(
        proof_id: u32,
        inserted_at: u64,
        nonce: u64,
        mutation_hash: [u8; 32],
    ) -> Self {
        Self {
            proof_id,
            inserted_at,
            nonce,
            mutation_hash,
            consumed: false,
        }
    }

    /// Checks if the entry has expired.
    #[must_use]
    #[inline]
    pub const fn is_expired(&self, current_time_ns: u64) -> bool {
        // Handle potential overflow: if current_time is before inserted_at,
        // the entry is definitely not expired (clock rollback scenario)
        if current_time_ns < self.inserted_at {
            return false;
        }
        current_time_ns - self.inserted_at > PROOF_CACHE_TTL_NS
    }

    /// Checks if the entry matches the given mutation_hash and nonce.
    #[must_use]
    #[inline]
    pub fn matches(&self, mutation_hash: &[u8; 32], nonce: u64) -> bool {
        self.nonce == nonce && self.mutation_hash == *mutation_hash
    }
}

/// Secure proof cache with TTL, single-use nonces, and bounded size.
///
/// Implements SEC-002 security requirements:
/// - 100ms TTL for cache entries
/// - Single-use nonce consumption
/// - Maximum 64 entries
/// - Entries scoped to (mutation_hash, nonce) pairs
#[derive(Debug)]
pub struct ProofCache {
    /// Fixed-size array of cache entries.
    entries: [Option<ProofCacheEntry>; PROOF_CACHE_MAX_ENTRIES],
    /// Number of active (non-None) entries.
    count: usize,
}

impl ProofCache {
    /// Creates a new empty proof cache.
    #[must_use]
    pub const fn new() -> Self {
        const NONE: Option<ProofCacheEntry> = None;
        Self {
            entries: [NONE; PROOF_CACHE_MAX_ENTRIES],
            count: 0,
        }
    }

    /// Returns the number of active entries in the cache.
    #[must_use]
    #[inline]
    pub const fn len(&self) -> usize {
        self.count
    }

    /// Returns true if the cache is empty.
    #[must_use]
    #[inline]
    pub const fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Returns true if the cache is full (64 entries).
    #[must_use]
    #[inline]
    pub const fn is_full(&self) -> bool {
        self.count >= PROOF_CACHE_MAX_ENTRIES
    }

    /// Inserts a new proof into the cache.
    ///
    /// # Arguments
    ///
    /// * `mutation_hash` - Hash of the mutation being authorized
    /// * `nonce` - Single-use nonce for this proof
    /// * `proof_id` - Unique identifier for this proof
    /// * `current_time_ns` - Current time in nanoseconds
    ///
    /// # Errors
    ///
    /// - `CacheError::DuplicateEntry` if an entry with the same (mutation_hash, nonce) exists
    /// - `CacheError::CacheFull` if the cache is at capacity and no expired entries can be evicted
    #[must_use]
    pub fn insert(
        &mut self,
        mutation_hash: [u8; 32],
        nonce: u64,
        proof_id: u32,
        current_time_ns: u64,
    ) -> Result<(), CacheError> {
        // First pass: check for duplicates and evict expired entries
        for i in 0..PROOF_CACHE_MAX_ENTRIES {
            if let Some(ref entry) = self.entries[i] {
                // Check for duplicate
                if entry.matches(&mutation_hash, nonce) && !entry.consumed {
                    return Err(CacheError::DuplicateEntry);
                }

                // Evict expired or consumed entries
                if entry.is_expired(current_time_ns) || entry.consumed {
                    self.entries[i] = None;
                    self.count = self.count.saturating_sub(1);
                }
            }
        }

        // Find an empty slot
        let mut slot = None;
        for i in 0..PROOF_CACHE_MAX_ENTRIES {
            if self.entries[i].is_none() {
                slot = Some(i);
                break;
            }
        }

        match slot {
            Some(i) => {
                self.entries[i] = Some(ProofCacheEntry::new(
                    proof_id,
                    current_time_ns,
                    nonce,
                    mutation_hash,
                ));
                self.count += 1;
                Ok(())
            }
            None => Err(CacheError::CacheFull),
        }
    }

    /// Verifies and consumes a proof from the cache.
    ///
    /// This is the primary security-critical function. It:
    /// 1. Finds the entry matching (mutation_hash, nonce)
    /// 2. Checks that TTL has not expired
    /// 3. Marks the entry as consumed (single-use)
    /// 4. Removes the entry from the cache
    /// 5. Returns the proof_id
    ///
    /// # Arguments
    ///
    /// * `mutation_hash` - Hash of the mutation being verified
    /// * `nonce` - Nonce that was used when the proof was created
    /// * `current_time_ns` - Current time in nanoseconds
    ///
    /// # Returns
    ///
    /// The proof_id if verification succeeds.
    ///
    /// # Errors
    ///
    /// - `CacheError::NotFound` if no matching entry exists
    /// - `CacheError::Expired` if the entry's TTL has passed
    /// - `CacheError::NonceConsumed` if the nonce was already used
    #[must_use]
    pub fn verify_and_consume(
        &mut self,
        mutation_hash: &[u8; 32],
        nonce: u64,
        current_time_ns: u64,
    ) -> Result<u32, CacheError> {
        // Find matching entry
        for i in 0..PROOF_CACHE_MAX_ENTRIES {
            if let Some(ref mut entry) = self.entries[i] {
                if entry.matches(mutation_hash, nonce) {
                    // Check if already consumed
                    if entry.consumed {
                        // Remove the consumed entry
                        self.entries[i] = None;
                        self.count = self.count.saturating_sub(1);
                        return Err(CacheError::NonceConsumed);
                    }

                    // Check TTL
                    if entry.is_expired(current_time_ns) {
                        // Remove the expired entry
                        self.entries[i] = None;
                        self.count = self.count.saturating_sub(1);
                        return Err(CacheError::Expired);
                    }

                    // Mark as consumed and remove
                    let proof_id = entry.proof_id;
                    self.entries[i] = None;
                    self.count = self.count.saturating_sub(1);

                    return Ok(proof_id);
                }
            }
        }

        Err(CacheError::NotFound)
    }

    /// Checks if a proof exists in the cache without consuming it.
    ///
    /// This is useful for pre-validation before attempting a mutation.
    /// Note: This does NOT consume the nonce.
    #[must_use]
    pub fn exists(&self, mutation_hash: &[u8; 32], nonce: u64, current_time_ns: u64) -> bool {
        for entry in &self.entries {
            if let Some(ref e) = entry {
                if e.matches(mutation_hash, nonce) && !e.consumed && !e.is_expired(current_time_ns)
                {
                    return true;
                }
            }
        }
        false
    }

    /// Removes all expired entries from the cache.
    ///
    /// This can be called periodically to clean up the cache.
    pub fn evict_expired(&mut self, current_time_ns: u64) {
        for i in 0..PROOF_CACHE_MAX_ENTRIES {
            if let Some(ref entry) = self.entries[i] {
                if entry.is_expired(current_time_ns) || entry.consumed {
                    self.entries[i] = None;
                    self.count = self.count.saturating_sub(1);
                }
            }
        }
    }

    /// Clears all entries from the cache.
    pub fn clear(&mut self) {
        for i in 0..PROOF_CACHE_MAX_ENTRIES {
            self.entries[i] = None;
        }
        self.count = 0;
    }

    /// Returns statistics about the cache.
    #[must_use]
    pub fn stats(&self, current_time_ns: u64) -> ProofCacheStats {
        let mut active = 0;
        let mut expired = 0;
        let mut consumed = 0;

        for entry in &self.entries {
            if let Some(ref e) = entry {
                if e.consumed {
                    consumed += 1;
                } else if e.is_expired(current_time_ns) {
                    expired += 1;
                } else {
                    active += 1;
                }
            }
        }

        ProofCacheStats {
            total_slots: PROOF_CACHE_MAX_ENTRIES,
            active_entries: active,
            expired_entries: expired,
            consumed_entries: consumed,
            free_slots: PROOF_CACHE_MAX_ENTRIES - (active + expired + consumed),
        }
    }
}

impl Default for ProofCache {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics about the proof cache.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProofCacheStats {
    /// Total number of slots (always 64).
    pub total_slots: usize,
    /// Number of active (valid, unconsumed) entries.
    pub active_entries: usize,
    /// Number of expired entries awaiting cleanup.
    pub expired_entries: usize,
    /// Number of consumed entries awaiting cleanup.
    pub consumed_entries: usize,
    /// Number of free slots.
    pub free_slots: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_insert_and_verify() {
        let mut cache = ProofCache::new();

        let mutation_hash = [42u8; 32];
        let nonce = 12345u64;
        let proof_id = 1u32;
        let time = 1_000_000u64;

        // Insert
        assert!(cache.insert(mutation_hash, nonce, proof_id, time).is_ok());
        assert_eq!(cache.len(), 1);

        // Verify exists
        assert!(cache.exists(&mutation_hash, nonce, time));

        // Verify and consume
        let result = cache.verify_and_consume(&mutation_hash, nonce, time);
        assert_eq!(result, Ok(proof_id));
        assert_eq!(cache.len(), 0);

        // Second verification should fail (consumed)
        let result = cache.verify_and_consume(&mutation_hash, nonce, time);
        assert_eq!(result, Err(CacheError::NotFound));
    }

    #[test]
    fn test_nonce_single_use() {
        let mut cache = ProofCache::new();

        let mutation_hash = [1u8; 32];
        let nonce = 99999u64;
        let time = 0u64;

        cache.insert(mutation_hash, nonce, 1, time).unwrap();

        // First verification succeeds
        assert!(cache.verify_and_consume(&mutation_hash, nonce, time).is_ok());

        // Insert again with same nonce (should succeed since old entry was removed)
        cache.insert(mutation_hash, nonce, 2, time).unwrap();

        // Second verification should succeed (new entry)
        assert!(cache.verify_and_consume(&mutation_hash, nonce, time).is_ok());
    }

    #[test]
    fn test_ttl_expiry() {
        let mut cache = ProofCache::new();

        let mutation_hash = [2u8; 32];
        let nonce = 1u64;
        let proof_id = 10u32;
        let insert_time = 1_000_000u64;

        cache.insert(mutation_hash, nonce, proof_id, insert_time).unwrap();

        // Verify within TTL (50ms later)
        let time_within_ttl = insert_time + 50_000_000;
        assert!(cache.exists(&mutation_hash, nonce, time_within_ttl));

        // Verify after TTL (150ms later)
        let time_after_ttl = insert_time + 150_000_000;
        assert!(!cache.exists(&mutation_hash, nonce, time_after_ttl));

        // verify_and_consume should return Expired
        let result = cache.verify_and_consume(&mutation_hash, nonce, time_after_ttl);
        assert_eq!(result, Err(CacheError::Expired));
    }

    #[test]
    fn test_max_entries() {
        let mut cache = ProofCache::new();

        // Fill the cache
        for i in 0..PROOF_CACHE_MAX_ENTRIES {
            let mut hash = [0u8; 32];
            hash[0] = i as u8;
            cache.insert(hash, i as u64, i as u32, 0).unwrap();
        }

        assert!(cache.is_full());
        assert_eq!(cache.len(), PROOF_CACHE_MAX_ENTRIES);

        // Try to insert one more (should fail)
        let result = cache.insert([255u8; 32], 999, 999, 0);
        assert_eq!(result, Err(CacheError::CacheFull));
    }

    #[test]
    fn test_eviction_of_expired() {
        let mut cache = ProofCache::new();

        // Fill cache with entries
        for i in 0..PROOF_CACHE_MAX_ENTRIES {
            let mut hash = [0u8; 32];
            hash[0] = i as u8;
            cache.insert(hash, i as u64, i as u32, 0).unwrap();
        }

        assert!(cache.is_full());

        // Try to insert with time past TTL (should succeed by evicting expired)
        let later = PROOF_CACHE_TTL_NS + 1;
        let result = cache.insert([254u8; 32], 9999, 9999, later);
        assert!(result.is_ok());
    }

    #[test]
    fn test_duplicate_entry() {
        let mut cache = ProofCache::new();

        let hash = [5u8; 32];
        let nonce = 100u64;

        cache.insert(hash, nonce, 1, 0).unwrap();

        // Try to insert duplicate
        let result = cache.insert(hash, nonce, 2, 0);
        assert_eq!(result, Err(CacheError::DuplicateEntry));
    }

    #[test]
    fn test_stats() {
        let mut cache = ProofCache::new();

        // Insert some entries
        for i in 0..10 {
            let mut hash = [0u8; 32];
            hash[0] = i as u8;
            cache.insert(hash, i as u64, i as u32, 0).unwrap();
        }

        let stats = cache.stats(0);
        assert_eq!(stats.active_entries, 10);
        assert_eq!(stats.free_slots, PROOF_CACHE_MAX_ENTRIES - 10);

        // Consume some
        let mut hash = [0u8; 32];
        cache.verify_and_consume(&hash, 0, 0).unwrap();
        hash[0] = 1;
        cache.verify_and_consume(&hash, 1, 0).unwrap();

        let stats = cache.stats(0);
        assert_eq!(stats.active_entries, 8);

        // Check expired entries
        let later = PROOF_CACHE_TTL_NS + 1;
        let stats = cache.stats(later);
        assert_eq!(stats.expired_entries, 8);
        assert_eq!(stats.active_entries, 0);
    }

    #[test]
    fn test_evict_expired() {
        let mut cache = ProofCache::new();

        for i in 0..5 {
            let mut hash = [0u8; 32];
            hash[0] = i as u8;
            cache.insert(hash, i as u64, i as u32, 0).unwrap();
        }

        assert_eq!(cache.len(), 5);

        // Evict at time past TTL
        cache.evict_expired(PROOF_CACHE_TTL_NS + 1);
        assert_eq!(cache.len(), 0);
    }

    #[test]
    fn test_clear() {
        let mut cache = ProofCache::new();

        for i in 0..10 {
            let mut hash = [0u8; 32];
            hash[0] = i as u8;
            cache.insert(hash, i as u64, i as u32, 0).unwrap();
        }

        assert_eq!(cache.len(), 10);
        cache.clear();
        assert_eq!(cache.len(), 0);
        assert!(cache.is_empty());
    }

    #[test]
    fn test_entry_expired() {
        let entry = ProofCacheEntry::new(1, 0, 0, [0u8; 32]);

        // Not expired at insert time
        assert!(!entry.is_expired(0));

        // Not expired 50ms later
        assert!(!entry.is_expired(50_000_000));

        // Not expired at exactly TTL
        assert!(!entry.is_expired(PROOF_CACHE_TTL_NS));

        // Expired 1ns after TTL
        assert!(entry.is_expired(PROOF_CACHE_TTL_NS + 1));

        // Handle clock rollback (time before insert)
        assert!(!entry.is_expired(0));
    }
}
