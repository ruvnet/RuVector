//! Conflict resolution for concurrent multi-agent shared memory.
//!
//! When several autonomous agents write to the same semantic memory slot
//! without a synchronized wall clock or a single sequencer, something has to
//! decide which write survives. The standard answer — last-write-wins (LWW)
//! by wall-clock timestamp — is vulnerable to clock skew between agents and
//! is semantically blind: it cannot tell a high-value memory update from a
//! low-value one. A vector-clock LWW fixes the skew problem for *causally
//! related* writes but still has to arbitrarily tie-break writes that are
//! genuinely concurrent (neither happened-before the other).
//!
//! This crate adds a third policy, [`StructuralCoherenceMerge`], that never
//! overrides real causal order (a happens-before write is always superseded
//! by what it happened before) but, for genuinely concurrent conflicts,
//! breaks the tie using `emergent-time`'s
//! [`StructuralProperTime`](emergent_time::structural_clock::StructuralProperTime)
//! — the magnitude of structural state-change the write represents — combined
//! with a coherence score against the current shared context window.

pub mod scenario;
pub mod vclock;

pub use emergent_time::structural_clock::{
    Clock, StateSnapshot, StructuralMetric, StructuralProperTime,
};
pub use vclock::{AgentId, CausalOrder, VectorClock};

pub type MemoryKey = u64;

/// One agent's proposed write to a shared memory key.
#[derive(Clone, Debug)]
pub struct MemoryWrite {
    pub agent_id: AgentId,
    pub key: MemoryKey,
    /// Local wall-clock timestamp in milliseconds (may be skewed vs. other agents).
    pub wall_ts_ms: f64,
    pub vclock: VectorClock,
    /// The writing agent's local state immediately before this write.
    pub prev_snapshot: StateSnapshot,
    /// The new memory's structural state.
    pub snapshot: StateSnapshot,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Winner {
    A,
    B,
}

/// A merge policy's decision on one conflicting pair, with enough detail to
/// audit *why* — analogous to a witness log entry
/// (see `ruvector-proof-gate` / `ruvector-retrieval-receipt` for the
/// cryptographic hardening of this idea in the write/read paths).
#[derive(Clone, Debug)]
pub struct Decision {
    pub winner: Winner,
    pub reason: &'static str,
    pub tau_a: f64,
    pub tau_b: f64,
    pub causal_order: CausalOrder,
}

pub trait MergePolicy {
    fn name(&self) -> &'static str;
    fn resolve(&self, a: &MemoryWrite, b: &MemoryWrite, context: &[Vec<f64>]) -> Decision;
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn norm(a: &[f64]) -> f64 {
    dot(a, a).sqrt()
}

/// f64 cosine coherence against a context window — the same "max similarity
/// to any context vector" signal as `ruvector-agent-memory::scoring`, lifted
/// to the f64 embeddings `StructuralProperTime` operates on.
fn coherence_score(v: &[f64], context: &[Vec<f64>]) -> f64 {
    let nv = norm(v);
    if nv < 1e-9 || context.is_empty() {
        return 0.0;
    }
    context
        .iter()
        .map(|q| {
            let nq = norm(q);
            if nq < 1e-9 {
                0.0
            } else {
                (dot(v, q) / (nv * nq)).clamp(-1.0, 1.0)
            }
        })
        .fold(f64::NEG_INFINITY, f64::max)
        .max(0.0)
}

/// Baseline: last-write-wins by raw wall-clock timestamp. Ignores causal
/// order and content entirely — the ubiquitous default in eventually
/// consistent systems.
pub struct LwwWallClock;
impl MergePolicy for LwwWallClock {
    fn name(&self) -> &'static str {
        "LwwWallClock"
    }
    fn resolve(&self, a: &MemoryWrite, b: &MemoryWrite, _context: &[Vec<f64>]) -> Decision {
        let causal_order = a.vclock.compare(&b.vclock);
        let winner = if a.wall_ts_ms >= b.wall_ts_ms {
            Winner::A
        } else {
            Winner::B
        };
        Decision {
            winner,
            reason: "later wall_ts_ms",
            tau_a: 0.0,
            tau_b: 0.0,
            causal_order,
        }
    }
}

/// Variant A: last-write-wins ordered by vector clock. Never violates causal
/// order, but ties among genuinely concurrent writes are broken by agent id
/// — a fixed, content-blind rule.
pub struct LwwVectorClock;
impl MergePolicy for LwwVectorClock {
    fn name(&self) -> &'static str {
        "LwwVectorClock"
    }
    fn resolve(&self, a: &MemoryWrite, b: &MemoryWrite, _context: &[Vec<f64>]) -> Decision {
        let causal_order = a.vclock.compare(&b.vclock);
        let (winner, reason) = match causal_order {
            CausalOrder::Before => (Winner::B, "b causally follows a"),
            CausalOrder::After => (Winner::A, "a causally follows b"),
            CausalOrder::Equal => (Winner::B, "equal clocks, arbitrary"),
            CausalOrder::Concurrent => {
                if a.agent_id >= b.agent_id {
                    (Winner::A, "concurrent, higher agent_id")
                } else {
                    (Winner::B, "concurrent, higher agent_id")
                }
            }
        };
        Decision {
            winner,
            reason,
            tau_a: 0.0,
            tau_b: 0.0,
            causal_order,
        }
    }
}

/// Variant B (the candidate): respects causal order exactly like
/// [`LwwVectorClock`], but for genuinely concurrent conflicts, breaks the tie
/// using each write's structural proper-time magnitude (how much the write
/// actually moves the agent's state) weighted by its coherence with the
/// current shared context.
#[derive(Default)]
pub struct StructuralCoherenceMerge {
    pub metric: StructuralMetric,
}

impl MergePolicy for StructuralCoherenceMerge {
    fn name(&self) -> &'static str {
        "StructuralCoherenceMerge"
    }
    fn resolve(&self, a: &MemoryWrite, b: &MemoryWrite, context: &[Vec<f64>]) -> Decision {
        let causal_order = a.vclock.compare(&b.vclock);
        let clock = StructuralProperTime::new(self.metric);
        let tau_a = clock.tick(&a.prev_snapshot, &a.snapshot);
        let tau_b = clock.tick(&b.prev_snapshot, &b.snapshot);

        if let Some((winner, reason)) = match causal_order {
            CausalOrder::Before => Some((Winner::B, "b causally follows a")),
            CausalOrder::After => Some((Winner::A, "a causally follows b")),
            _ => None,
        } {
            return Decision {
                winner,
                reason,
                tau_a,
                tau_b,
                causal_order,
            };
        }

        let coh_a = coherence_score(&a.snapshot.embedding, context);
        let coh_b = coherence_score(&b.snapshot.embedding, context);
        // Structural magnitude scaled by contextual relevance: a large
        // structural shift only counts fully if it also lands somewhere the
        // current context cares about.
        let score_a = tau_a * (0.5 + 0.5 * coh_a);
        let score_b = tau_b * (0.5 + 0.5 * coh_b);
        let winner = if score_a >= score_b {
            Winner::A
        } else {
            Winner::B
        };
        Decision {
            winner,
            reason: "concurrent, higher tau*coherence",
            tau_a,
            tau_b,
            causal_order,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scenario::{generate, ScenarioConfig};

    #[test]
    fn vclock_and_structural_never_violate_causal_order() {
        let cfg = ScenarioConfig {
            num_conflicts: 20,
            num_causal_controls: 300,
            wall_skew_ms: 5000.0, // deliberately large, to try to trip a violation
            ..Default::default()
        };
        let cases = generate(&cfg);
        let vc = LwwVectorClock;
        let sc = StructuralCoherenceMerge::default();
        for c in cases.iter().filter(|c| !c.is_concurrent) {
            let d_vc = vc.resolve(&c.a, &c.b, &c.context);
            let d_sc = sc.resolve(&c.a, &c.b, &c.context);
            assert_eq!(
                d_vc.winner,
                Winner::B,
                "vclock policy must keep b (causally later)"
            );
            assert_eq!(
                d_sc.winner,
                Winner::B,
                "structural policy must keep b (causally later)"
            );
        }
    }

    #[test]
    fn wall_clock_lww_can_violate_causal_order_under_skew() {
        let cfg = ScenarioConfig {
            num_conflicts: 0,
            num_causal_controls: 500,
            wall_skew_ms: 5000.0,
            jitter_ms: 10.0,
            ..Default::default()
        };
        let cases = generate(&cfg);
        let wc = LwwWallClock;
        let violations = cases
            .iter()
            .filter(|c| !c.is_concurrent)
            .filter(|c| wc.resolve(&c.a, &c.b, &c.context).winner == Winner::A)
            .count();
        assert!(
            violations > 0,
            "expected wall-clock LWW to violate causal order at least once under 5000ms skew"
        );
    }

    #[test]
    fn structural_merge_prefers_larger_coherent_shift() {
        let base = StateSnapshot::full(vec![0.0, 0.0], 1.0, 0.5, 0.0, 0.0);
        let small = StateSnapshot::full(vec![0.1, 0.0], 0.9, 0.55, 0.02, 0.9);
        let large = StateSnapshot::full(vec![1.0, 0.0], 0.2, 0.95, 0.4, 0.05);
        let mut vc_a = VectorClock::new();
        vc_a.tick(1);
        let mut vc_b = VectorClock::new();
        vc_b.tick(2);
        let a = MemoryWrite {
            agent_id: 1,
            key: 0,
            wall_ts_ms: 0.0,
            vclock: vc_a,
            prev_snapshot: base.clone(),
            snapshot: small,
        };
        let b = MemoryWrite {
            agent_id: 2,
            key: 0,
            wall_ts_ms: 0.0,
            vclock: vc_b,
            prev_snapshot: base,
            snapshot: large,
        };
        let context = vec![vec![1.0, 0.0]];
        let sc = StructuralCoherenceMerge::default();
        let d = sc.resolve(&a, &b, &context);
        assert_eq!(d.winner, Winner::B);
        assert!(d.tau_b > d.tau_a);
    }
}
