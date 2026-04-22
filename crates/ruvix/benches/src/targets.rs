//! ADR-087 target latency definitions and verification.
//!
//! This module defines the target latencies for all 12 RuVix syscalls
//! as specified in ADR-087 Section 3.2.

use std::time::Duration;

/// Target latency specification for a syscall.
#[derive(Debug, Clone, Copy)]
pub struct TargetSpec {
    /// Syscall name.
    pub name: &'static str,
    /// Target p95 latency.
    pub target: Duration,
    /// Proof tier (if applicable).
    pub proof_tier: Option<ProofTierSpec>,
    /// Notes about the target.
    pub notes: &'static str,
}

/// Proof tier specification.
#[derive(Debug, Clone, Copy)]
pub enum ProofTierSpec {
    /// Reflex tier: <100ns proof overhead
    Reflex,
    /// Standard tier: <1us proof overhead
    Standard,
    /// Deep tier: <100us proof overhead
    Deep,
}

impl ProofTierSpec {
    /// Returns the target overhead for this tier.
    pub const fn target_overhead(&self) -> Duration {
        match self {
            Self::Reflex => Duration::from_nanos(100),
            Self::Standard => Duration::from_micros(1),
            Self::Deep => Duration::from_micros(100),
        }
    }

    /// Returns the tier name.
    pub const fn name(&self) -> &'static str {
        match self {
            Self::Reflex => "Reflex",
            Self::Standard => "Standard",
            Self::Deep => "Deep",
        }
    }
}

/// All target specifications from ADR-087.
pub const TARGET_SPECS: &[TargetSpec] = &[
    TargetSpec {
        name: "task_spawn",
        target: Duration::from_micros(10),
        proof_tier: None,
        notes: "RVF component spawning with capability distribution",
    },
    TargetSpec {
        name: "cap_grant",
        target: Duration::from_nanos(500),
        proof_tier: None,
        notes: "O(1) capability table lookup and derivation",
    },
    TargetSpec {
        name: "region_map",
        target: Duration::from_micros(5),
        proof_tier: None,
        notes: "Region-based memory with slab allocation",
    },
    TargetSpec {
        name: "queue_send",
        target: Duration::from_nanos(200),
        proof_tier: None,
        notes: "Zero-copy lock-free SPSC/MPSC queue",
    },
    TargetSpec {
        name: "queue_recv",
        target: Duration::from_nanos(200),
        proof_tier: None,
        notes: "Zero-copy lock-free SPSC/MPSC queue",
    },
    TargetSpec {
        name: "timer_wait",
        target: Duration::from_nanos(100),
        proof_tier: None,
        notes: "Scheduler timer integration",
    },
    TargetSpec {
        name: "rvf_mount",
        target: Duration::from_millis(1),
        proof_tier: Some(ProofTierSpec::Deep),
        notes: "RVF package signature verification + proof",
    },
    TargetSpec {
        name: "attest_emit",
        target: Duration::from_nanos(500),
        proof_tier: Some(ProofTierSpec::Reflex),
        notes: "82-byte attestation record to witness log",
    },
    TargetSpec {
        name: "vector_get",
        target: Duration::from_nanos(100),
        proof_tier: None,
        notes: "Vector retrieval (no proof required for read)",
    },
    TargetSpec {
        name: "vector_put_proved",
        target: Duration::from_nanos(500),
        proof_tier: Some(ProofTierSpec::Reflex),
        notes: "Proof-gated vector mutation with Reflex tier",
    },
    TargetSpec {
        name: "graph_apply_proved",
        target: Duration::from_micros(1),
        proof_tier: Some(ProofTierSpec::Standard),
        notes: "Proof-gated graph mutation with Standard tier",
    },
    TargetSpec {
        name: "sensor_subscribe",
        target: Duration::from_micros(5),
        proof_tier: None,
        notes: "Sensor to queue binding registration",
    },
];

/// Returns the target spec for a given syscall name.
pub fn spec_for(name: &str) -> Option<&'static TargetSpec> {
    TARGET_SPECS.iter().find(|s| s.name == name)
}

/// Verifies that a measurement meets its target.
#[derive(Debug, Clone, Copy)]
pub struct TargetVerification {
    /// Whether the target was met.
    pub met: bool,
    /// Actual p95 latency.
    pub actual_p95: Duration,
    /// Target latency.
    pub target: Duration,
    /// Margin (actual / target).
    pub margin: f64,
}

impl TargetVerification {
    /// Creates a new verification result.
    pub fn new(actual_p95: Duration, target: Duration) -> Self {
        let margin = actual_p95.as_nanos() as f64 / target.as_nanos() as f64;
        Self {
            met: actual_p95 <= target,
            actual_p95,
            target,
            margin,
        }
    }

    /// Returns a human-readable status.
    pub fn status(&self) -> &'static str {
        if self.met {
            if self.margin <= 0.5 {
                "PASS (2x margin)"
            } else if self.margin <= 0.8 {
                "PASS"
            } else {
                "PASS (tight)"
            }
        } else if self.margin <= 1.5 {
            "FAIL (close)"
        } else {
            "FAIL"
        }
    }
}

/// Summary of all target verifications.
#[derive(Debug, Clone)]
pub struct TargetSummary {
    /// Total syscalls verified.
    pub total: usize,
    /// Syscalls meeting target.
    pub passing: usize,
    /// Syscalls failing target.
    pub failing: usize,
    /// Individual verifications.
    pub verifications: Vec<(String, TargetVerification)>,
}

impl TargetSummary {
    /// Creates a new empty summary.
    pub fn new() -> Self {
        Self {
            total: 0,
            passing: 0,
            failing: 0,
            verifications: Vec::new(),
        }
    }

    /// Adds a verification to the summary.
    pub fn add(&mut self, name: &str, verification: TargetVerification) {
        self.total += 1;
        if verification.met {
            self.passing += 1;
        } else {
            self.failing += 1;
        }
        self.verifications.push((name.to_string(), verification));
    }

    /// Returns the overall pass rate.
    pub fn pass_rate(&self) -> f64 {
        if self.total > 0 {
            self.passing as f64 / self.total as f64
        } else {
            0.0
        }
    }

    /// Returns whether all targets were met.
    pub fn all_passing(&self) -> bool {
        self.failing == 0 && self.total > 0
    }
}

impl Default for TargetSummary {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spec_lookup() {
        let spec = spec_for("cap_grant").unwrap();
        assert_eq!(spec.target, Duration::from_nanos(500));
        assert!(spec.proof_tier.is_none());
    }

    #[test]
    fn test_proof_tier_specs() {
        let spec = spec_for("vector_put_proved").unwrap();
        assert!(matches!(spec.proof_tier, Some(ProofTierSpec::Reflex)));

        let spec = spec_for("graph_apply_proved").unwrap();
        assert!(matches!(spec.proof_tier, Some(ProofTierSpec::Standard)));
    }

    #[test]
    fn test_target_verification() {
        let v = TargetVerification::new(Duration::from_nanos(400), Duration::from_nanos(500));
        assert!(v.met);
        assert_eq!(v.status(), "PASS");

        let v = TargetVerification::new(Duration::from_nanos(600), Duration::from_nanos(500));
        assert!(!v.met);
    }

    #[test]
    fn test_target_summary() {
        let mut summary = TargetSummary::new();
        summary.add(
            "op1",
            TargetVerification::new(Duration::from_nanos(400), Duration::from_nanos(500)),
        );
        summary.add(
            "op2",
            TargetVerification::new(Duration::from_nanos(600), Duration::from_nanos(500)),
        );

        assert_eq!(summary.total, 2);
        assert_eq!(summary.passing, 1);
        assert_eq!(summary.failing, 1);
        assert!((summary.pass_rate() - 0.5).abs() < 0.001);
    }
}
