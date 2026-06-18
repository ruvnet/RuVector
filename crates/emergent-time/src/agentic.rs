//! Causal event time for agents.
//!
//! Wall-clock time is often noise for an agent: an hour with no state change
//! carries no information, while one contradiction can reorganize everything.
//! Here time is read off the *causal structure* of events, not timestamps.
//!
//! Each event records its causal parents and an information measure (entropy of
//! the agent's belief/state). Two derived quantities give an internal clock:
//!
//! * **causal depth** — the longest chain of causes leading to the event, a
//!   robust integer ordering that survives shuffled arrival order;
//! * **internal time** — accumulated *irreversible* change (entropy reduction)
//!   along the causal history, a continuous monotone.

/// A single event in the agent's causal history.
#[derive(Clone, Debug)]
pub struct Event {
    /// Causal parents (indices of earlier events).
    pub parents: Vec<usize>,
    /// Information content of the agent's state at this event (nats).
    pub entropy: f64,
}

/// An append-only causal DAG of events.
#[derive(Default)]
pub struct CausalTimeline {
    events: Vec<Event>,
}

impl CausalTimeline {
    pub fn new() -> Self {
        CausalTimeline { events: Vec::new() }
    }

    pub fn len(&self) -> usize {
        self.events.len()
    }

    pub fn is_empty(&self) -> bool {
        self.events.is_empty()
    }

    pub fn event(&self, id: usize) -> &Event {
        &self.events[id]
    }

    /// Append an event with the given parents and state entropy. Parents must
    /// reference already-added events (smaller ids), keeping the graph acyclic.
    /// Returns the new event id.
    pub fn add(&mut self, parents: Vec<usize>, entropy: f64) -> usize {
        let id = self.events.len();
        for &p in &parents {
            assert!(p < id, "parent {p} must precede event {id}");
        }
        self.events.push(Event { parents, entropy });
        id
    }

    /// Causal depth: longest path from any root to `id`. Roots have depth 0.
    /// This is the agent's discrete event-time, independent of arrival order.
    pub fn causal_depth(&self, id: usize) -> usize {
        let mut memo = vec![usize::MAX; self.events.len()];
        self.depth_rec(id, &mut memo)
    }

    fn depth_rec(&self, id: usize, memo: &mut [usize]) -> usize {
        if memo[id] != usize::MAX {
            return memo[id];
        }
        let d = self.events[id]
            .parents
            .iter()
            .map(|&p| 1 + self.depth_rec(p, memo))
            .max()
            .unwrap_or(0);
        memo[id] = d;
        d
    }

    /// Internal time: accumulated entropy *reduction* along the maximal causal
    /// path to `id`. Each causal step contributes `max(0, S_parent - S_event)`,
    /// so internal time is non-decreasing along every causal edge — a quiet
    /// stretch adds nothing, a sharp drop in uncertainty adds a lot.
    pub fn internal_time(&self, id: usize) -> f64 {
        let mut memo = vec![f64::NAN; self.events.len()];
        self.itime_rec(id, &mut memo)
    }

    fn itime_rec(&self, id: usize, memo: &mut [f64]) -> f64 {
        if !memo[id].is_nan() {
            return memo[id];
        }
        let s = self.events[id].entropy;
        let t = self.events[id]
            .parents
            .iter()
            .map(|&p| {
                let reduction = (self.events[p].entropy - s).max(0.0);
                self.itime_rec(p, memo) + reduction
            })
            .fold(0.0f64, f64::max);
        memo[id] = t;
        t
    }

    /// Topological ordering (events sorted by causal depth, ties by id). This is
    /// the sequence an internal observer experiences, recovered without any
    /// clock.
    pub fn causal_order(&self) -> Vec<usize> {
        let mut ids: Vec<usize> = (0..self.events.len()).collect();
        ids.sort_by(|&a, &b| {
            self.causal_depth(a)
                .cmp(&self.causal_depth(b))
                .then(a.cmp(&b))
        });
        ids
    }

    /// Verify internal time is non-decreasing across every causal edge.
    pub fn preserves_causal_order(&self) -> bool {
        for id in 0..self.events.len() {
            let t = self.internal_time(id);
            for &p in &self.events[id].parents {
                if self.internal_time(p) > t + 1e-9 {
                    return false;
                }
            }
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn depth_follows_longest_chain() {
        let mut tl = CausalTimeline::new();
        let a = tl.add(vec![], 2.0);
        let b = tl.add(vec![a], 1.8);
        let c = tl.add(vec![a], 1.9);
        let d = tl.add(vec![b, c], 1.0);
        assert_eq!(tl.causal_depth(a), 0);
        assert_eq!(tl.causal_depth(d), 2);
        assert_eq!(tl.causal_order()[0], a);
        assert_eq!(*tl.causal_order().last().unwrap(), d);
    }

    #[test]
    fn internal_time_jumps_on_uncertainty_drop() {
        let mut tl = CausalTimeline::new();
        let a = tl.add(vec![], 3.0);
        // idle step: no entropy change -> no internal time advance
        let b = tl.add(vec![a], 3.0);
        // contradiction resolved: big entropy drop -> big internal-time jump
        let c = tl.add(vec![b], 0.5);
        assert!((tl.internal_time(b) - 0.0).abs() < 1e-12);
        assert!((tl.internal_time(c) - 2.5).abs() < 1e-12);
        assert!(tl.preserves_causal_order());
    }
}
