//! Learning the router from the ledger.
//!
//! Every routed decision already emits a [`crate::routing::Witness`] — which
//! lenses were consulted, the confidence at each step, the action, the outcome.
//! Stack those up and the ledger *is* a trajectory dataset. This module learns
//! a stop/continue routing policy from that log by tabular first-visit
//! Monte-Carlo control, so the router improves from its own recorded provenance
//! instead of hand-set thresholds — the retrieval analogue of MetaHarness
//! learning a harness from its trace.
//!
//! State = `(lenses_consulted, confidence_bin)`. Action = stop or continue.
//! An exploratory behaviour policy (stop/continue at random) is logged, then
//! greedy improvement reads the best action per state off the learned Q-table.

/// The two routing actions available at each consultation step.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Action {
    Stop,
    Continue,
}

/// One logged episode: the `(lens_count, conf_bin, action)` states it visited
/// and the realized return (terminal utility minus accumulated cost).
#[derive(Debug, Clone)]
pub struct Episode {
    pub visited: Vec<(usize, usize, Action)>,
    pub ret: f32,
}

/// A learned tabular stop/continue policy indexed by `(lens_count, conf_bin)`.
#[derive(Debug, Clone)]
pub struct LearnedPolicy {
    max_lens: usize,
    n_bins: usize,
    q_stop: Vec<f32>,
    q_cont: Vec<f32>,
    n_stop: Vec<u32>,
    n_cont: Vec<u32>,
}

impl LearnedPolicy {
    fn idx(&self, lens_count: usize, bin: usize) -> usize {
        let lc = lens_count.min(self.max_lens.saturating_sub(1));
        let b = bin.min(self.n_bins - 1);
        lc * self.n_bins + b
    }

    /// Greedy action for a state; defaults to `Continue` for never-seen states
    /// with room to escalate (optimistic), else `Stop`.
    pub fn act(&self, lens_count: usize, bin: usize) -> Action {
        let i = self.idx(lens_count, bin);
        if self.n_stop[i] == 0 && self.n_cont[i] == 0 {
            return if lens_count + 1 < self.max_lens {
                Action::Continue
            } else {
                Action::Stop
            };
        }
        if self.q_stop[i] >= self.q_cont[i] {
            Action::Stop
        } else {
            Action::Continue
        }
    }

    /// Bin a confidence in `[0,1]` the same way learning did.
    pub fn bin_of(&self, confidence: f32) -> usize {
        ((confidence.clamp(0.0, 0.999_999)) * self.n_bins as f32) as usize
    }

    pub fn q_values(&self, lens_count: usize, bin: usize) -> (f32, f32) {
        let i = self.idx(lens_count, bin);
        (self.q_stop[i], self.q_cont[i])
    }
}

/// Learn a policy from logged episodes by first-visit Monte-Carlo averaging of
/// returns per `(state, action)`.
pub fn learn(episodes: &[Episode], max_lens: usize, n_bins: usize) -> LearnedPolicy {
    let max_lens = max_lens.max(1);
    let n_bins = n_bins.max(1);
    let size = max_lens * n_bins;
    let mut p = LearnedPolicy {
        max_lens,
        n_bins,
        q_stop: vec![0.0; size],
        q_cont: vec![0.0; size],
        n_stop: vec![0; size],
        n_cont: vec![0; size],
    };

    for ep in episodes {
        let mut seen = std::collections::BTreeSet::new();
        for &(lc, bin, action) in &ep.visited {
            let key = (lc.min(max_lens - 1), bin.min(n_bins - 1), action);
            if !seen.insert(key) {
                continue; // first-visit
            }
            let i = p.idx(lc, bin);
            match action {
                Action::Stop => {
                    p.q_stop[i] = running_mean(p.q_stop[i], p.n_stop[i], ep.ret);
                    p.n_stop[i] += 1;
                }
                Action::Continue => {
                    p.q_cont[i] = running_mean(p.q_cont[i], p.n_cont[i], ep.ret);
                    p.n_cont[i] += 1;
                }
            }
        }
    }
    p
}

#[inline]
fn running_mean(mean: f32, n: u32, x: f32) -> f32 {
    mean + (x - mean) / (n as f32 + 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn learns_to_continue_when_low_conf_and_stop_when_high() {
        // Synthetic ledger: stopping in a low-confidence state (bin 0) returns
        // little; continuing from it returns more. High-confidence (bin 9)
        // stopping returns a lot.
        let mut eps = Vec::new();
        for _ in 0..200 {
            // low-conf stop → return 0.1
            eps.push(Episode { visited: vec![(1, 0, Action::Stop)], ret: 0.1 });
            // low-conf continue → return 0.8
            eps.push(Episode { visited: vec![(1, 0, Action::Continue)], ret: 0.8 });
            // high-conf stop → return 0.9
            eps.push(Episode { visited: vec![(1, 9, Action::Stop)], ret: 0.9 });
            // high-conf continue → return 0.5 (wasted cost)
            eps.push(Episode { visited: vec![(1, 9, Action::Continue)], ret: 0.5 });
        }
        let p = learn(&eps, 3, 10);
        assert_eq!(p.act(1, 0), Action::Continue);
        assert_eq!(p.act(1, 9), Action::Stop);
    }
}
