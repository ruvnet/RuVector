//! Proof statistics command implementation.

use crate::ShellBackend;
use alloc::format;
use alloc::string::String;

/// Execute the proofs command.
#[must_use]
pub fn execute<B: ShellBackend>(backend: &B) -> String {
    let stats = backend.proof_stats();

    let verify_rate = if stats.generated > 0 {
        (stats.verified as f64 / stats.generated as f64 * 100.0) as u32
    } else {
        0
    };

    let cache_hit_rate = if stats.cache_hits + stats.cache_misses > 0 {
        (stats.cache_hits as f64 / (stats.cache_hits + stats.cache_misses) as f64 * 100.0) as u32
    } else {
        0
    };

    let total_proofs = stats.tier0_count + stats.tier1_count + stats.tier2_count;
    let tier0_pct = if total_proofs > 0 {
        (stats.tier0_count as f64 / total_proofs as f64 * 100.0) as u32
    } else {
        0
    };
    let tier1_pct = if total_proofs > 0 {
        (stats.tier1_count as f64 / total_proofs as f64 * 100.0) as u32
    } else {
        0
    };
    let tier2_pct = if total_proofs > 0 {
        (stats.tier2_count as f64 / total_proofs as f64 * 100.0) as u32
    } else {
        0
    };

    format!(
        r"Proof Statistics
================
Generated:        {}
Verified:         {} ({}%)
Rejected:         {}

Cache:
  Entries:        {} / 64
  Hits:           {}
  Misses:         {}
  Hit Rate:       {}%

By Tier:
  Tier 0 (Reflex):   {} ({}%)
  Tier 1 (Standard): {} ({}%)
  Tier 2 (Deep):     {} ({}%)",
        stats.generated,
        stats.verified,
        verify_rate,
        stats.rejected,
        stats.cache_entries,
        stats.cache_hits,
        stats.cache_misses,
        cache_hit_rate,
        stats.tier0_count,
        tier0_pct,
        stats.tier1_count,
        tier1_pct,
        stats.tier2_count,
        tier2_pct
    )
}

#[cfg(test)]
mod tests {
    use crate::ProofStats;

    #[test]
    fn test_proof_stats_defaults() {
        let stats = ProofStats::default();
        assert_eq!(stats.generated, 0);
        assert_eq!(stats.verified, 0);
    }
}
