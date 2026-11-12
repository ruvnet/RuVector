//! Invariants that survive compaction verbatim (ADR-274 §2.2).
//!
//! Safety constraints and task statements **erode through successive compaction
//! cycles with no failure signal** — a documented mechanism ("governance
//! decay"), not a jailbreak. Each summarization pass paraphrases a little more
//! away until a rule that was explicit at turn 1 is gone by turn 200, and
//! nothing in the transcript marks the moment it disappeared.
//!
//! The fix is cheap and absolute: a small set of invariants is re-emitted
//! **byte-identical** after every compaction. They are never inputs to a
//! summarizer, never masked (ADR-274 §3.1), and never paraphrased.
//!
//! Keep this set small. Everything here is paid for on every turn after a
//! compaction, and a bloated invariant set recreates the context pressure
//! compaction exists to relieve.

use crate::messages::Message;

/// A rule that must never be summarized away.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Invariant {
    /// Short stable label, for debugging and dedup.
    pub id: String,
    /// The exact text to re-emit. Reproduced byte-for-byte.
    pub text: String,
}

impl Invariant {
    pub fn new(id: impl Into<String>, text: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            text: text.into(),
        }
    }
}

/// The set of invariants carried across compaction boundaries.
#[derive(Debug, Clone, Default)]
pub struct InvariantSet {
    invariants: Vec<Invariant>,
}

impl InvariantSet {
    pub fn new() -> Self {
        Self::default()
    }

    /// Add an invariant. Re-adding an existing `id` replaces it, so a caller
    /// cannot accidentally accumulate near-duplicate copies of a rule.
    pub fn insert(&mut self, invariant: Invariant) {
        match self.invariants.iter_mut().find(|i| i.id == invariant.id) {
            Some(existing) => *existing = invariant,
            None => self.invariants.push(invariant),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.invariants.is_empty()
    }

    pub fn len(&self) -> usize {
        self.invariants.len()
    }

    pub fn iter(&self) -> impl Iterator<Item = &Invariant> {
        self.invariants.iter()
    }

    /// Render the block re-emitted after a compaction.
    ///
    /// Returns `None` when empty so callers never inject an empty stub.
    pub fn render(&self) -> Option<String> {
        if self.invariants.is_empty() {
            return None;
        }
        let mut out = String::from(
            "<invariants>\nThese were established earlier and remain in force. \
             They are reproduced exactly and are not a summary.\n",
        );
        for inv in &self.invariants {
            out.push('\n');
            out.push_str(&inv.text);
            out.push('\n');
        }
        out.push_str("</invariants>");
        Some(out)
    }

    /// Append the invariant block to a compacted history.
    ///
    /// Call this immediately after any operation that drops or rewrites
    /// history. Appending at the end — rather than restoring the original
    /// position — is deliberate: recency is what survives a long context, and
    /// the whole point is that these rules must not be the first thing lost.
    pub fn reinject(&self, mut messages: Vec<Message>) -> Vec<Message> {
        if let Some(block) = self.render() {
            messages.push(Message::system(block));
        }
        messages
    }
}

impl FromIterator<Invariant> for InvariantSet {
    fn from_iter<T: IntoIterator<Item = Invariant>>(iter: T) -> Self {
        let mut set = Self::new();
        for inv in iter {
            set.insert(inv);
        }
        set
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn set() -> InvariantSet {
        [
            Invariant::new("task", "TASK: Fix the failing parser test."),
            Invariant::new("safety", "Never force-push to the default branch."),
        ]
        .into_iter()
        .collect()
    }

    #[test]
    fn renders_text_byte_for_byte() {
        let block = set().render().unwrap();
        // The exact strings must appear untransformed — no wrapping, no
        // reflowing, no paraphrase.
        assert!(block.contains("TASK: Fix the failing parser test."));
        assert!(block.contains("Never force-push to the default branch."));
    }

    #[test]
    fn survives_a_round_trip_unchanged() {
        let original = set();
        let block = original.render().unwrap();
        // Simulate several compaction cycles: each one re-renders from the
        // same source, so the text can never drift.
        for _ in 0..10 {
            assert_eq!(original.render().unwrap(), block);
        }
    }

    #[test]
    fn reinject_appends_a_system_message() {
        let compacted = vec![Message::system("summary of earlier work")];
        let out = set().reinject(compacted);
        assert_eq!(out.len(), 2);
        match out.last().unwrap() {
            Message::System(s) => {
                assert!(s.content.contains("TASK: Fix the failing parser test."));
                assert!(s.content.contains("not a summary"));
            }
            other => panic!("expected a system message, got {other:?}"),
        }
    }

    #[test]
    fn empty_set_injects_nothing() {
        let empty = InvariantSet::new();
        assert!(empty.render().is_none());
        let msgs = vec![Message::human("hi")];
        assert_eq!(empty.reinject(msgs.clone()), msgs);
    }

    #[test]
    fn reinserting_an_id_replaces_rather_than_duplicates() {
        let mut s = set();
        s.insert(Invariant::new("task", "TASK: Updated objective."));
        assert_eq!(s.len(), 2, "an updated rule must not accumulate copies");
        let block = s.render().unwrap();
        assert!(block.contains("TASK: Updated objective."));
        assert!(!block.contains("Fix the failing parser test"));
    }

    #[test]
    fn insertion_order_is_stable() {
        let s = set();
        let ids: Vec<&str> = s.iter().map(|i| i.id.as_str()).collect();
        assert_eq!(ids, vec!["task", "safety"]);
    }
}
