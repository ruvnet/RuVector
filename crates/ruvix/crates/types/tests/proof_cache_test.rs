//! Security tests for ADR-087 SEC-002: Proof cache with TTL/nonce/bounds.
//!
//! These tests verify the proof cache security requirements:
//! - 100ms TTL for cache entries
//! - Single-use nonce consumption
//! - Maximum 64 entries
//! - Entries scoped to (mutation_hash, nonce) pairs

use ruvix_types::proof_cache_mod::{
    CacheError, ProofCache, ProofCacheEntry, PROOF_CACHE_MAX_ENTRIES, PROOF_CACHE_TTL_NS,
};

// =============================================================================
// SEC-002: Proof Cache TTL Tests
// =============================================================================

#[test]
fn test_sec002_ttl_constant_is_100ms() {
    // 100ms = 100_000_000 nanoseconds
    assert_eq!(PROOF_CACHE_TTL_NS, 100_000_000);
}

#[test]
fn test_sec002_entry_not_expired_within_ttl() {
    let entry = ProofCacheEntry::new(1, 0, 0, [0u8; 32]);

    // At insert time
    assert!(!entry.is_expired(0));

    // 50ms later
    assert!(!entry.is_expired(50_000_000));

    // 99ms later
    assert!(!entry.is_expired(99_000_000));

    // Exactly at TTL
    assert!(!entry.is_expired(PROOF_CACHE_TTL_NS));
}

#[test]
fn test_sec002_entry_expired_after_ttl() {
    let entry = ProofCacheEntry::new(1, 0, 0, [0u8; 32]);

    // 1ns after TTL
    assert!(entry.is_expired(PROOF_CACHE_TTL_NS + 1));

    // 1ms after TTL
    assert!(entry.is_expired(PROOF_CACHE_TTL_NS + 1_000_000));

    // Way after TTL
    assert!(entry.is_expired(PROOF_CACHE_TTL_NS * 10));
}

#[test]
fn test_sec002_verify_fails_after_ttl() {
    let mut cache = ProofCache::new();

    let hash = [1u8; 32];
    let nonce = 42u64;
    let insert_time = 1_000_000u64;

    cache.insert(hash, nonce, 1, insert_time).unwrap();

    // Verify within TTL works
    let time_within_ttl = insert_time + 50_000_000;
    assert!(cache.exists(&hash, nonce, time_within_ttl));

    // Verify after TTL fails with Expired error
    let time_after_ttl = insert_time + PROOF_CACHE_TTL_NS + 1;
    assert_eq!(
        cache.verify_and_consume(&hash, nonce, time_after_ttl),
        Err(CacheError::Expired)
    );
}

// =============================================================================
// SEC-002: Single-Use Nonce Tests
// =============================================================================

#[test]
fn test_sec002_nonce_single_use() {
    let mut cache = ProofCache::new();

    let hash = [2u8; 32];
    let nonce = 12345u64;

    cache.insert(hash, nonce, 1, 0).unwrap();

    // First verification succeeds
    let result = cache.verify_and_consume(&hash, nonce, 0);
    assert_eq!(result, Ok(1));

    // Second verification fails (nonce consumed)
    let result = cache.verify_and_consume(&hash, nonce, 0);
    assert_eq!(result, Err(CacheError::NotFound));
}

#[test]
fn test_sec002_consumed_nonce_cannot_be_reused_with_same_hash() {
    let mut cache = ProofCache::new();

    let hash = [3u8; 32];
    let nonce = 99999u64;

    cache.insert(hash, nonce, 1, 0).unwrap();
    cache.verify_and_consume(&hash, nonce, 0).unwrap();

    // Entry was removed, so we can insert again
    cache.insert(hash, nonce, 2, 0).unwrap();

    // New entry should work
    let result = cache.verify_and_consume(&hash, nonce, 0);
    assert_eq!(result, Ok(2));
}

#[test]
fn test_sec002_different_nonces_are_independent() {
    let mut cache = ProofCache::new();

    let hash = [4u8; 32];
    let nonce1 = 1u64;
    let nonce2 = 2u64;

    cache.insert(hash, nonce1, 10, 0).unwrap();
    cache.insert(hash, nonce2, 20, 0).unwrap();

    // Both should be verifiable
    assert_eq!(cache.verify_and_consume(&hash, nonce1, 0), Ok(10));
    assert_eq!(cache.verify_and_consume(&hash, nonce2, 0), Ok(20));
}

#[test]
fn test_sec002_different_hashes_are_independent() {
    let mut cache = ProofCache::new();

    let hash1 = [5u8; 32];
    let hash2 = [6u8; 32];
    let nonce = 100u64;

    cache.insert(hash1, nonce, 1, 0).unwrap();
    cache.insert(hash2, nonce, 2, 0).unwrap();

    // Both should be verifiable independently
    assert_eq!(cache.verify_and_consume(&hash1, nonce, 0), Ok(1));
    assert_eq!(cache.verify_and_consume(&hash2, nonce, 0), Ok(2));
}

// =============================================================================
// SEC-002: Max Entries Bound Tests
// =============================================================================

#[test]
fn test_sec002_max_entries_is_64() {
    assert_eq!(PROOF_CACHE_MAX_ENTRIES, 64);
}

#[test]
fn test_sec002_cache_enforces_max_entries() {
    let mut cache = ProofCache::new();

    // Fill the cache with 64 entries
    for i in 0..PROOF_CACHE_MAX_ENTRIES {
        let mut hash = [0u8; 32];
        hash[0] = i as u8;
        cache.insert(hash, i as u64, i as u32, 0).unwrap();
    }

    assert!(cache.is_full());
    assert_eq!(cache.len(), PROOF_CACHE_MAX_ENTRIES);

    // 65th entry should fail
    let result = cache.insert([255u8; 32], 999, 999, 0);
    assert_eq!(result, Err(CacheError::CacheFull));
}

#[test]
fn test_sec002_expired_entries_are_evicted_on_insert() {
    let mut cache = ProofCache::new();

    // Fill cache
    for i in 0..PROOF_CACHE_MAX_ENTRIES {
        let mut hash = [0u8; 32];
        hash[0] = i as u8;
        cache.insert(hash, i as u64, i as u32, 0).unwrap();
    }

    assert!(cache.is_full());

    // Try to insert with time past TTL - should evict expired and succeed
    let time_after_ttl = PROOF_CACHE_TTL_NS + 1;
    let result = cache.insert([254u8; 32], 9999, 9999, time_after_ttl);
    assert!(result.is_ok());
}

// =============================================================================
// SEC-002: Scoped Entries Tests
// =============================================================================

#[test]
fn test_sec002_entries_scoped_to_mutation_hash_nonce_pair() {
    let mut cache = ProofCache::new();

    let hash = [7u8; 32];
    let nonce = 555u64;

    cache.insert(hash, nonce, 1, 0).unwrap();

    // Correct hash, wrong nonce - not found
    let wrong_nonce = 556u64;
    assert_eq!(
        cache.verify_and_consume(&hash, wrong_nonce, 0),
        Err(CacheError::NotFound)
    );

    // Wrong hash, correct nonce - not found
    let wrong_hash = [8u8; 32];
    assert_eq!(
        cache.verify_and_consume(&wrong_hash, nonce, 0),
        Err(CacheError::NotFound)
    );

    // Correct pair - found
    assert_eq!(cache.verify_and_consume(&hash, nonce, 0), Ok(1));
}

#[test]
fn test_sec002_duplicate_entry_rejected() {
    let mut cache = ProofCache::new();

    let hash = [9u8; 32];
    let nonce = 777u64;

    cache.insert(hash, nonce, 1, 0).unwrap();

    // Same (hash, nonce) should be rejected
    let result = cache.insert(hash, nonce, 2, 0);
    assert_eq!(result, Err(CacheError::DuplicateEntry));
}

// =============================================================================
// Additional Proof Cache Tests
// =============================================================================

#[test]
fn test_cache_stats() {
    let mut cache = ProofCache::new();

    // Insert 10 entries
    for i in 0..10 {
        let mut hash = [0u8; 32];
        hash[0] = i as u8;
        cache.insert(hash, i as u64, i as u32, 0).unwrap();
    }

    let stats = cache.stats(0);
    assert_eq!(stats.total_slots, PROOF_CACHE_MAX_ENTRIES);
    assert_eq!(stats.active_entries, 10);
    assert_eq!(stats.free_slots, PROOF_CACHE_MAX_ENTRIES - 10);

    // After TTL, entries should show as expired
    let stats = cache.stats(PROOF_CACHE_TTL_NS + 1);
    assert_eq!(stats.expired_entries, 10);
    assert_eq!(stats.active_entries, 0);
}

#[test]
fn test_cache_evict_expired() {
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
fn test_cache_clear() {
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
fn test_entry_handles_clock_rollback() {
    // If current time is before insert time (clock rollback), entry is not expired
    let entry = ProofCacheEntry::new(1, 1_000_000, 0, [0u8; 32]);

    // Current time before insert time
    assert!(!entry.is_expired(0));
    assert!(!entry.is_expired(500_000));
}
