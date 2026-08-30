//! Compaction: score every memory once against the final context, keep the
//! top `budget`, and compare against an oracle nearest-neighbour set.

use std::collections::HashSet;
use std::time::{Duration, Instant};

use emergent_time::{Clock, StateSnapshot};

use crate::scenario::{cosine, MemoryItem};

#[derive(Clone, Copy, Debug)]
pub struct CompactionWeights {
    pub w_coherence: f64,
    pub w_recency: f64,
    /// Decay half-scale, as a fraction of the clock's *own* total elapsed
    /// internal time over the session. Fixed identically across all clocks so
    /// no clock gets a hand-tuned decay scale — see crate-level docs.
    pub tau_fraction: f64,
}

impl Default for CompactionWeights {
    fn default() -> Self {
        CompactionWeights {
            w_coherence: 0.5,
            w_recency: 0.5,
            tau_fraction: 0.2,
        }
    }
}

pub struct CompactionResult {
    pub kept: HashSet<usize>,
    /// Wall-clock time to build the clock's cumulative-time array and score
    /// every memory. Excludes session/snapshot generation (shared setup cost,
    /// identical for all clocks).
    pub elapsed: Duration,
}

/// Score every memory against `final_context` using `clock`'s notion of age,
/// keep the top `budget` by score.
pub fn compact<C: Clock>(
    clock: &C,
    snapshots: &[StateSnapshot],
    memories: &[MemoryItem],
    final_context: &[f64],
    budget: usize,
    weights: CompactionWeights,
) -> CompactionResult {
    let start = Instant::now();
    let cumulative = clock.cumulative(snapshots);
    let total_time = *cumulative.last().unwrap_or(&0.0);
    let tau = (weights.tau_fraction * total_time).max(1e-9);
    let final_time = *cumulative.last().unwrap_or(&0.0);

    let mut scored: Vec<(usize, f64)> = memories
        .iter()
        .map(|m| {
            let age = (final_time - cumulative[m.write_step]).max(0.0);
            let coh = cosine(&m.embedding, final_context);
            let rec = (-age / tau).exp();
            let score = weights.w_coherence * coh + weights.w_recency * rec;
            (m.id, score)
        })
        .collect();
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    let kept: HashSet<usize> = scored.into_iter().take(budget).map(|(id, _)| id).collect();
    let elapsed = start.elapsed();
    CompactionResult { kept, elapsed }
}

/// True top-`k` memories by raw cosine similarity to `final_context` — what
/// an unlimited-memory oracle would return for the final query.
pub fn oracle_top_k(memories: &[MemoryItem], final_context: &[f64], k: usize) -> HashSet<usize> {
    let mut scored: Vec<(usize, f64)> = memories
        .iter()
        .map(|m| (m.id, cosine(&m.embedding, final_context)))
        .collect();
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    scored.into_iter().take(k).map(|(id, _)| id).collect()
}

/// Fraction of the oracle top-k that survived compaction.
pub fn recall_at_k(kept: &HashSet<usize>, oracle: &HashSet<usize>) -> f64 {
    if oracle.is_empty() {
        return 1.0;
    }
    let hit = oracle.iter().filter(|id| kept.contains(*id)).count();
    hit as f64 / oracle.len() as f64
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clocks::structural_embedding_clock;
    use crate::scenario::{generate_session, ScenarioConfig};
    use emergent_time::WallClock;

    #[test]
    fn kept_set_never_exceeds_budget() {
        let cfg = ScenarioConfig::default();
        let session = generate_session(&cfg);
        let final_context = session.contexts.last().unwrap().clone();
        let budget = 10;
        let res = compact(
            &WallClock,
            &session.snapshots,
            &session.memories,
            &final_context,
            budget,
            CompactionWeights::default(),
        );
        assert!(res.kept.len() <= budget);
    }

    #[test]
    fn oracle_recall_of_itself_is_one() {
        let cfg = ScenarioConfig::default();
        let session = generate_session(&cfg);
        let final_context = session.contexts.last().unwrap().clone();
        let oracle = oracle_top_k(&session.memories, &final_context, 15);
        assert!((recall_at_k(&oracle, &oracle) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn structural_clock_runs_end_to_end() {
        let cfg = ScenarioConfig::default();
        let session = generate_session(&cfg);
        let final_context = session.contexts.last().unwrap().clone();
        let clock = structural_embedding_clock();
        let res = compact(
            &clock,
            &session.snapshots,
            &session.memories,
            &final_context,
            20,
            CompactionWeights::default(),
        );
        assert!(!res.kept.is_empty());
    }
}
