//! Pure helpers for the engine: the calibration-slice split, the provisional
//! `TrainReport` head/calibration rules, `noul` label parsing, and the stable
//! content hash used for the compiled/artifact caches (no `RandomState`).

use super::*;

/// A (training, calibration) split of class examples: `(embedding, class_idx)`.
pub(super) type ClassSplit = (Vec<(Vec<f32>, usize)>, Vec<(Vec<f32>, usize)>);
/// A (training, calibration) split of noul examples: `(embedding, label 0/1)`.
pub(super) type NoulSplit = (Vec<(Vec<f32>, f32)>, Vec<(Vec<f32>, f32)>);

/// Split relevant class examples into (training, calibration) — every Nth
/// example (0-based `i % CALIB_EVERY == CALIB_EVERY - 1`) goes to calibration.
pub(super) fn split_class(relevant: &[(&Vec<f32>, usize)]) -> ClassSplit {
    let mut train = Vec::new();
    let mut calib = Vec::new();
    for (i, (emb, ci)) in relevant.iter().enumerate() {
        let pair = ((*emb).clone(), *ci);
        if i % CALIB_EVERY == CALIB_EVERY - 1 {
            calib.push(pair);
        } else {
            train.push(pair);
        }
    }
    (train, calib)
}

pub(super) fn split_noul(relevant: &[(Vec<f32>, f32)]) -> NoulSplit {
    let mut train = Vec::new();
    let mut calib = Vec::new();
    for (i, pair) in relevant.iter().enumerate() {
        if i % CALIB_EVERY == CALIB_EVERY - 1 {
            calib.push(pair.clone());
        } else {
            train.push(pair.clone());
        }
    }
    (train, calib)
}

/// The head a `TrainReport` announces, from example counts alone (criteria
/// arrive at decide time). A yes/no label set implies the logistic head; ≥ 2
/// classes each with ≥ 4 examples implies the probe; otherwise the prototype.
/// Counts only the *training* slice (the calibration slice is excluded), so the
/// report matches the head `decide` will actually pick.
pub(super) fn provisional_head(examples: &[(Vec<f32>, String)]) -> Head {
    let mut counts: BTreeMap<&str, usize> = BTreeMap::new();
    for (_, l) in examples
        .iter()
        .enumerate()
        .filter(|(i, _)| i % CALIB_EVERY != CALIB_EVERY - 1)
        .map(|(_, e)| e)
    {
        *counts.entry(l.as_str()).or_insert(0) += 1;
    }
    if !counts.is_empty() && counts.keys().all(|k| parse_noul_label(k).is_some()) {
        return Head::Logistic;
    }
    if counts.len() >= 2 && counts.values().all(|&c| c >= MIN_EXAMPLES_PER_CLASS) {
        Head::LinearProbe
    } else {
        Head::NearestPrototype
    }
}

pub(super) fn provisional_calibrated(examples: &[(Vec<f32>, String)], id: &str) -> bool {
    let calib = (0..examples.len())
        .filter(|i| i % CALIB_EVERY == CALIB_EVERY - 1)
        .count();
    calib >= MIN_CALIBRATION && !is_test_double(id)
}

pub(super) fn is_test_double(id: &str) -> bool {
    id.ends_with("@test-double")
}

/// Normalise a `noul` label to `1.0` / `0.0`, or `None` if it is neither.
pub(super) fn parse_noul_label(label: &str) -> Option<f32> {
    match label.trim().to_lowercase().as_str() {
        "yes" | "1" | "true" | "y" | "positive" | "pos" => Some(1.0),
        "no" | "0" | "false" | "n" | "negative" | "neg" => Some(0.0),
        _ => None,
    }
}

/// Stable FNV-1a over `id` and the canonical serialisation of the question, so
/// the compiled cache key is deterministic across runs and targets.
pub(super) fn stable_hash(id: &str, q: &Question) -> Result<u64> {
    let ser = serde_json::to_string(q)
        .map_err(|e| TypesafeError::Invalid(format!("question not serialisable: {e}")))?;
    let mut h = fnv1a(id.as_bytes());
    h = fnv1a_continue(h, &[0]);
    Ok(fnv1a_continue(h, ser.as_bytes()))
}

/// Combine two hashes into the artifact-cache key `(content hash, generation)`.
pub(super) fn mix(a: u64, b: u64) -> u64 {
    let mut h = fnv1a_continue(fnv1a(&a.to_le_bytes()), &b.to_le_bytes());
    h ^= h >> 33;
    h
}

fn fnv1a(bytes: &[u8]) -> u64 {
    fnv1a_continue(0xcbf2_9ce4_8422_2325, bytes)
}

fn fnv1a_continue(mut h: u64, bytes: &[u8]) -> u64 {
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}
