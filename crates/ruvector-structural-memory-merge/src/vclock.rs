//! Minimal vector clock for detecting happens-before vs. concurrent writes
//! across agents, without a synchronized wall clock.

use std::collections::BTreeMap;

pub type AgentId = u32;

/// Causal relationship between two events' vector clocks.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CausalOrder {
    /// `self` happened before `other`.
    Before,
    /// `self` happened after `other`.
    After,
    /// Neither dominates: the writes are concurrent (a genuine conflict).
    Concurrent,
    /// Identical clocks.
    Equal,
}

/// A Lamport-style vector clock: one logical counter per known agent.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct VectorClock(BTreeMap<AgentId, u64>);

impl VectorClock {
    pub fn new() -> Self {
        VectorClock(BTreeMap::new())
    }

    pub fn get(&self, agent: AgentId) -> u64 {
        *self.0.get(&agent).unwrap_or(&0)
    }

    /// Advance this clock's own counter for `agent` by one local event.
    pub fn tick(&mut self, agent: AgentId) {
        let e = self.0.entry(agent).or_insert(0);
        *e += 1;
    }

    /// Merge in another clock's knowledge (component-wise max), as happens
    /// when an agent observes a remote write.
    pub fn merge(&mut self, other: &VectorClock) {
        for (&agent, &v) in other.0.iter() {
            let e = self.0.entry(agent).or_insert(0);
            if v > *e {
                *e = v;
            }
        }
    }

    /// Compare against another clock to determine causal order.
    pub fn compare(&self, other: &VectorClock) -> CausalOrder {
        let agents: std::collections::BTreeSet<AgentId> =
            self.0.keys().chain(other.0.keys()).copied().collect();
        // self_leq_other: self <= other componentwise (self dominated by other).
        // other_leq_self: other <= self componentwise (other dominated by self).
        let mut self_leq_other = true;
        let mut other_leq_self = true;
        for a in agents {
            let sv = self.get(a);
            let ov = other.get(a);
            if sv > ov {
                self_leq_other = false;
            }
            if ov > sv {
                other_leq_self = false;
            }
        }
        match (self_leq_other, other_leq_self) {
            (true, true) => CausalOrder::Equal,
            (true, false) => CausalOrder::Before,
            (false, true) => CausalOrder::After,
            (false, false) => CausalOrder::Concurrent,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ticking_own_agent_is_before() {
        let mut a = VectorClock::new();
        a.tick(1);
        let mut b = a.clone();
        b.tick(1);
        assert_eq!(a.compare(&b), CausalOrder::Before);
        assert_eq!(b.compare(&a), CausalOrder::After);
    }

    #[test]
    fn independent_agents_are_concurrent() {
        let mut a = VectorClock::new();
        a.tick(1);
        let mut b = VectorClock::new();
        b.tick(2);
        assert_eq!(a.compare(&b), CausalOrder::Concurrent);
    }

    #[test]
    fn merge_then_tick_dominates() {
        let mut a = VectorClock::new();
        a.tick(1);
        let mut b = VectorClock::new();
        b.tick(2);
        // b observes a's write, then ticks locally: b now happens-after a.
        b.merge(&a);
        b.tick(2);
        assert_eq!(a.compare(&b), CausalOrder::Before);
    }

    #[test]
    fn equal_clocks() {
        let a = VectorClock::new();
        let b = VectorClock::new();
        assert_eq!(a.compare(&b), CausalOrder::Equal);
    }
}
