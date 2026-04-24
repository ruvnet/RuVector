//! Rate limiting using BudgetTokenBucket pattern
//!
//! Includes periodic cleanup of stale buckets to prevent unbounded memory growth.

use dashmap::DashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

/// Rate limiter with per-contributor AND per-IP token buckets.
///
/// Per-contributor limits prevent a single key from flooding.
/// Per-IP limits prevent Sybil attacks (rotating API keys from the same source).
pub struct RateLimiter {
    write_buckets: DashMap<String, TokenBucket>,
    read_buckets: DashMap<String, TokenBucket>,
    /// Per-IP write buckets — prevents Sybil key rotation (ADR-082)
    ip_write_buckets: DashMap<String, TokenBucket>,
    /// Per-IP read buckets
    ip_read_buckets: DashMap<String, TokenBucket>,
    write_limit: u32,
    read_limit: u32,
    /// IP-level write limit (higher than per-key, catches rotation)
    ip_write_limit: u32,
    /// IP-level read limit
    ip_read_limit: u32,
    window: Duration,
    /// Counter for triggering periodic cleanup
    ops_counter: AtomicU64,
    /// Cleanup every N operations
    cleanup_interval: u64,
    /// Per-IP vote tracking: maps "ip:memory_id" -> (vote_count, first_vote_time)
    /// Entries older than 24h are evicted during periodic cleanup.
    ip_votes: DashMap<String, (u32, Instant)>,
}

struct TokenBucket {
    tokens: u32,
    max_tokens: u32,
    last_refill: Instant,
    window: Duration,
}

impl TokenBucket {
    fn new(max_tokens: u32, window: Duration) -> Self {
        Self {
            tokens: max_tokens,
            max_tokens,
            last_refill: Instant::now(),
            window,
        }
    }

    fn try_consume(&mut self) -> bool {
        self.refill();
        if self.tokens > 0 {
            self.tokens -= 1;
            true
        } else {
            false
        }
    }

    fn refill(&mut self) {
        let elapsed = self.last_refill.elapsed();
        if elapsed >= self.window {
            self.tokens = self.max_tokens;
            self.last_refill = Instant::now();
        }
    }

    /// Whether this bucket is stale (unused for more than 2 windows)
    fn is_stale(&self) -> bool {
        self.last_refill.elapsed() > self.window * 2
    }
}

impl RateLimiter {
    pub fn new(write_limit: u32, read_limit: u32) -> Self {
        Self {
            write_buckets: DashMap::new(),
            read_buckets: DashMap::new(),
            ip_write_buckets: DashMap::new(),
            ip_read_buckets: DashMap::new(),
            write_limit,
            read_limit,
            ip_write_limit: write_limit * 3, // 1500/hr per IP (allows some key rotation)
            ip_read_limit: read_limit * 3,   // 15000/hr per IP
            window: Duration::from_secs(3600),
            ops_counter: AtomicU64::new(0),
            cleanup_interval: 1000,
            ip_votes: DashMap::new(),
        }
    }

    pub fn default_limits() -> Self {
        Self::new(500, 5000)
    }

    pub fn check_write(&self, contributor: &str) -> bool {
        self.maybe_cleanup();
        let mut entry = self
            .write_buckets
            .entry(contributor.to_string())
            .or_insert_with(|| TokenBucket::new(self.write_limit, self.window));
        entry.try_consume()
    }

    /// Check per-IP write limit (anti-Sybil). Call in addition to check_write.
    pub fn check_ip_write(&self, ip: &str) -> bool {
        self.maybe_cleanup();
        let mut entry = self
            .ip_write_buckets
            .entry(ip.to_string())
            .or_insert_with(|| TokenBucket::new(self.ip_write_limit, self.window));
        entry.try_consume()
    }

    pub fn check_read(&self, contributor: &str) -> bool {
        self.maybe_cleanup();
        let mut entry = self
            .read_buckets
            .entry(contributor.to_string())
            .or_insert_with(|| TokenBucket::new(self.read_limit, self.window));
        entry.try_consume()
    }

    /// Check per-IP read limit. Call in addition to check_read.
    pub fn check_ip_read(&self, ip: &str) -> bool {
        self.maybe_cleanup();
        let mut entry = self
            .ip_read_buckets
            .entry(ip.to_string())
            .or_insert_with(|| TokenBucket::new(self.ip_read_limit, self.window));
        entry.try_consume()
    }

    /// Check if an IP has already voted on a memory (anti-Sybil vote dedup).
    /// Returns false if the IP already voted on this memory within the last 24h.
    pub fn check_ip_vote(&self, ip: &str, memory_id: &str) -> bool {
        let key = format!("{ip}:{memory_id}");
        let now = Instant::now();
        let mut entry = self.ip_votes.entry(key).or_insert((0, now));
        // Allow re-vote if previous vote is older than 24h
        if entry.0 >= 1 && now.duration_since(entry.1) < Duration::from_secs(86400) {
            return false;
        }
        if entry.0 >= 1 {
            // Reset after 24h window
            entry.0 = 0;
        }
        entry.0 += 1;
        entry.1 = now;
        true
    }

    /// Periodically clean up stale buckets to prevent unbounded memory growth
    fn maybe_cleanup(&self) {
        let count = self.ops_counter.fetch_add(1, Ordering::Relaxed);
        if count % self.cleanup_interval != 0 {
            return;
        }

        let write_before = self.write_buckets.len();
        let read_before = self.read_buckets.len();

        self.write_buckets.retain(|_, bucket| !bucket.is_stale());
        self.read_buckets.retain(|_, bucket| !bucket.is_stale());
        self.ip_write_buckets.retain(|_, bucket| !bucket.is_stale());
        self.ip_read_buckets.retain(|_, bucket| !bucket.is_stale());
        // Evict vote entries older than 24h
        let vote_before = self.ip_votes.len();
        self.ip_votes
            .retain(|_, (_, timestamp)| timestamp.elapsed() < Duration::from_secs(86400));
        let vote_evicted = vote_before - self.ip_votes.len();

        let write_evicted = write_before - self.write_buckets.len();
        let read_evicted = read_before - self.read_buckets.len();

        if write_evicted > 0 || read_evicted > 0 || vote_evicted > 0 {
            tracing::debug!(
                "Rate limiter cleanup: evicted {write_evicted} write + {read_evicted} read stale buckets + {vote_evicted} stale votes"
            );
        }
    }
}
