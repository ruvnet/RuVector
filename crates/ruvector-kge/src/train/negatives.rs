//! Negative sampling for self-adversarial training. Corruptions are drawn
//! uniformly over the entity id space (the RotatE default, arXiv:1902.10197);
//! the self-adversarial *weighting* of those negatives lives in `loss.rs`.

use crate::data::Rng;
use crate::EntityId;

/// Draw `count` corrupted entity ids uniformly from `[0, num_entities)`.
/// Uniform (not filtered) corruption: a corruption that happens to be a true
/// fact is a tolerated, low-probability event at these entity counts, matching
/// standard practice.
pub(crate) fn sample_corruptions(
    rng: &mut Rng,
    num_entities: usize,
    count: usize,
    out: &mut Vec<EntityId>,
) {
    out.clear();
    if num_entities == 0 {
        return;
    }
    out.reserve(count);
    for _ in 0..count {
        out.push(rng.below(num_entities as u64) as EntityId);
    }
}
