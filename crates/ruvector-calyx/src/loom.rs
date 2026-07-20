//! Cross-lens relationships (the paper's *Loom*).
//!
//! "Count" — the second of Calyx's four verbs — derives relationships *between*
//! lens outputs rather than averaging them away. The highest-value signal is
//! often the *disagreement*: two records look alike under the semantic lens but
//! diverge under the structural lens. This module measures that.

use std::collections::BTreeMap;

use crate::constellation::Constellation;
use crate::scoring::cosine_sim;

/// Per-lens cosine similarity between a query's slots and a record's slots.
///
/// Only lenses present on *both* sides are scored. Returns a map keyed by lens
/// name. This is the raw material for both the guard (min agreement across
/// lenses) and disagreement detection.
pub fn agreement(query_slots: &BTreeMap<String, Vec<f32>>, record: &Constellation) -> Vec<(String, f32)> {
    let mut out = Vec::new();
    for (lens, qv) in query_slots {
        if let Some(rv) = record.slot(lens) {
            if rv.len() == qv.len() {
                out.push((lens.clone(), cosine_sim(qv, rv)));
            }
        }
    }
    out
}

/// The minimum agreement across all shared lenses — the conservative,
/// fail-closed summary the guard uses. A high min means *every* lens agrees;
/// one dissenting lens drags it down.
pub fn min_agreement(query_slots: &BTreeMap<String, Vec<f32>>, record: &Constellation) -> f32 {
    agreement(query_slots, record)
        .into_iter()
        .map(|(_, s)| s)
        .fold(f32::INFINITY, f32::min)
        .clamp(0.0, 1.0)
}

/// The mean agreement across shared lenses.
pub fn mean_agreement(query_slots: &BTreeMap<String, Vec<f32>>, record: &Constellation) -> f32 {
    let scores = agreement(query_slots, record);
    if scores.is_empty() {
        return 0.0;
    }
    scores.iter().map(|(_, s)| s).sum::<f32>() / scores.len() as f32
}

/// The lens with the largest *gap* below the mean agreement — i.e. the most
/// informative dissenter. Returns `None` when fewer than two lenses are shared.
pub fn top_dissenter(
    query_slots: &BTreeMap<String, Vec<f32>>,
    record: &Constellation,
) -> Option<(String, f32)> {
    let scores = agreement(query_slots, record);
    if scores.len() < 2 {
        return None;
    }
    let mean = scores.iter().map(|(_, s)| s).sum::<f32>() / scores.len() as f32;
    scores
        .into_iter()
        .map(|(l, s)| (l, mean - s))
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
        .filter(|(_, gap)| *gap > 0.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn qslots() -> BTreeMap<String, Vec<f32>> {
        let mut m = BTreeMap::new();
        m.insert("sem".to_string(), vec![1.0, 0.0]);
        m.insert("str".to_string(), vec![1.0, 0.0]);
        m
    }

    #[test]
    fn disagreement_is_caught_by_min() {
        // Record agrees semantically but disagrees structurally.
        let rec = Constellation::new(1)
            .with_slot("sem", vec![1.0, 0.0])
            .with_slot("str", vec![0.0, 1.0]);
        let q = qslots();
        assert!(mean_agreement(&q, &rec) > 0.4); // mean hides it
        assert!(min_agreement(&q, &rec) < 0.1); // min catches it
        let (lens, _gap) = top_dissenter(&q, &rec).unwrap();
        assert_eq!(lens, "str");
    }
}
