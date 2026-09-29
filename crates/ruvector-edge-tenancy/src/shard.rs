//! Shard routing (ADR-351 §4.4).
//!
//! A vector id is routed with `hash(id) mod N`, where the hash is fixed as the
//! first 8 bytes (big-endian) of `sha256("v1|shard|" + id)`. It must agree
//! between the gateway and every Durable Object forever, so it is not
//! `DefaultHasher`/SipHash (unstable across Rust releases) and it is pinned
//! by golden-value tests.

use crate::error::TenancyError;
use crate::validate::VectorId;
use sha2::{Digest, Sha256};

/// Maximum shards per collection through M3 (the DO 6-outgoing-connection
/// limit; a larger count needs the M4+ two-level merge).
pub const MAX_SHARDS: u32 = 6;

const SHARD_DOMAIN: &[u8] = b"v1|shard|";

/// Number of shards in a collection, `1..=MAX_SHARDS`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ShardCount(u32);

impl ShardCount {
    /// A single shard (every collection starts here).
    pub const ONE: ShardCount = ShardCount(1);

    /// Validate a shard count.
    pub fn new(n: u32) -> Result<Self, TenancyError> {
        if (1..=MAX_SHARDS).contains(&n) {
            Ok(ShardCount(n))
        } else {
            Err(TenancyError::InvalidShard)
        }
    }

    /// The count. Also the number of rate-limit tokens a fan-out query costs
    /// (`shards_queried`, ADR §10).
    pub fn get(self) -> u32 {
        self.0
    }

    /// Every shard index, for query fan-out.
    pub fn indices(self) -> impl Iterator<Item = ShardIndex> {
        (0..self.0).map(ShardIndex)
    }
}

/// A shard index, `0..MAX_SHARDS`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ShardIndex(u32);

impl ShardIndex {
    /// Shard 0.
    pub const ZERO: ShardIndex = ShardIndex(0);

    /// Validate an index against a collection's shard count.
    pub fn new(index: u32, count: ShardCount) -> Result<Self, TenancyError> {
        if index < count.get() {
            Ok(ShardIndex(index))
        } else {
            Err(TenancyError::InvalidShard)
        }
    }

    /// Strictly parse a stored decimal index (canonical form, no sign, no
    /// leading zeros, `< MAX_SHARDS`).
    pub fn parse(stored: &str) -> Result<Self, TenancyError> {
        let canonical = !stored.is_empty()
            && stored.bytes().all(|b| b.is_ascii_digit())
            && (stored == "0" || !stored.starts_with('0'))
            && stored.len() <= 10;
        if !canonical {
            return Err(TenancyError::MalformedIdentifier("shard"));
        }
        let n: u32 = stored
            .parse()
            .map_err(|_| TenancyError::MalformedIdentifier("shard"))?;
        if n < MAX_SHARDS {
            Ok(ShardIndex(n))
        } else {
            Err(TenancyError::MalformedIdentifier("shard"))
        }
    }

    /// The index.
    pub fn get(self) -> u32 {
        self.0
    }
}

/// The stable 64-bit routing hash of a vector id.
pub fn shard_hash(id: &VectorId) -> u64 {
    let mut h = Sha256::new();
    h.update(SHARD_DOMAIN);
    h.update(id.as_str().as_bytes());
    let digest = h.finalize();
    let mut first = [0u8; 8];
    first.copy_from_slice(&digest[..8]);
    u64::from_be_bytes(first)
}

/// Route a vector id to its shard: `shard_hash(id) mod count`.
pub fn shard_for(id: &VectorId, count: ShardCount) -> ShardIndex {
    // `count >= 1` by construction, so the remainder is always `< count`.
    let idx = shard_hash(id) % u64::from(count.get());
    ShardIndex(idx as u32)
}
