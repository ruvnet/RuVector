//! Contrastive absence sensing: detect a *missing* expected continuation as a
//! structured signal, not a threshold alert. The expected temporal pattern is a
//! sequence of zone events (e.g. `bed_exit → bathroom_path → return_path`); when
//! a continuation edge never arrives within its deadline, the sequence graph is
//! left incomplete — that incompleteness is the signal.

/// A structured absence: an expected next step that did not occur in time.
#[derive(Debug, Clone, PartialEq)]
pub struct Absence {
    /// The step that was expected but never arrived.
    pub missing_step: String,
    /// The last step that *did* occur.
    pub after: String,
    /// How long we have waited past the last observed step.
    pub elapsed: u64,
}

/// Monitors progress through an expected sequence and flags missing
/// continuations.
#[derive(Debug, Clone)]
pub struct SequenceMonitor {
    steps: Vec<String>,
    deadline: u64,
    pos: usize,
    last_t: Option<u64>,
    started: bool,
}

impl SequenceMonitor {
    /// New monitor for an ordered list of expected zone events, with a
    /// per-step deadline (in the same time units as observations).
    pub fn new(steps: Vec<String>, deadline: u64) -> Self {
        Self {
            steps,
            deadline,
            pos: 0,
            last_t: None,
            started: false,
        }
    }

    /// Whether the full sequence has completed.
    pub fn complete(&self) -> bool {
        self.pos >= self.steps.len()
    }

    /// Record that an event happened in `zone` at time `t`. Advances the
    /// sequence if it matches the next expected step.
    pub fn observe_zone(&mut self, zone: &str, t: u64) {
        if self.complete() {
            return;
        }
        if self.steps[self.pos] == zone {
            self.pos += 1;
            self.last_t = Some(t);
            self.started = true;
        }
    }

    /// Check for a missing continuation as of `now`. Returns an [`Absence`] if
    /// the sequence has started, is not complete, and the next step is overdue.
    pub fn check(&self, now: u64) -> Option<Absence> {
        if !self.started || self.complete() {
            return None;
        }
        let last = self.last_t?;
        let elapsed = now.saturating_sub(last);
        if elapsed > self.deadline {
            Some(Absence {
                missing_step: self.steps[self.pos].clone(),
                after: self.steps[self.pos - 1].clone(),
                elapsed,
            })
        } else {
            None
        }
    }

    /// Reset to the start (e.g. for a new day/cycle).
    pub fn reset(&mut self) {
        self.pos = 0;
        self.last_t = None;
        self.started = false;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn routine() -> SequenceMonitor {
        SequenceMonitor::new(
            vec![
                "bed_exit".to_string(),
                "bathroom_path".to_string(),
                "return_path".to_string(),
            ],
            100,
        )
    }

    #[test]
    fn missing_return_is_flagged() {
        let mut m = routine();
        m.observe_zone("bed_exit", 0);
        m.observe_zone("bathroom_path", 10);
        assert!(m.check(50).is_none()); // still within deadline
        let a = m.check(200).expect("overdue return");
        assert_eq!(a.missing_step, "return_path");
        assert_eq!(a.after, "bathroom_path");
        assert!(a.elapsed > 100);
    }

    #[test]
    fn completed_routine_is_silent() {
        let mut m = routine();
        m.observe_zone("bed_exit", 0);
        m.observe_zone("bathroom_path", 10);
        m.observe_zone("return_path", 20);
        assert!(m.complete());
        assert!(m.check(10_000).is_none());
    }

    #[test]
    fn unstarted_routine_is_silent() {
        let m = routine();
        assert!(m.check(10_000).is_none());
    }
}
