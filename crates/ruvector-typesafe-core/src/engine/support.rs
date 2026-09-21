//! Pure helpers for the engine: `noul` label parsing, the test-double check,
//! and the stable content hash used for the compiled/artifact caches (no
//! `RandomState`, deterministic across runs and targets).

use crate::{Question, Result, TypesafeError};

pub(super) fn is_test_double(id: &str) -> bool {
    id.ends_with("@test-double")
}

/// Whether the `i`-th example (0-based, insertion order) is a calibration
/// position: every `stride`-th one, i.e. `i % stride == stride - 1`. A
/// `usize::MAX` stride (calibration disabled) selects nothing.
pub(super) fn is_calib_pos(i: usize, stride: usize) -> bool {
    stride != usize::MAX && stride >= 2 && i % stride == stride - 1
}

/// Split `items` (insertion order) into `(train, calibration)` by carving every
/// `stride`-th element out for calibration — the class-stratified, reproducible
/// slice the plain `train` path uses. Generic over the example payload.
pub(super) fn carve_calibration<T: Clone>(items: Vec<T>, stride: usize) -> (Vec<T>, Vec<T>) {
    let mut train = Vec::new();
    let mut calib = Vec::new();
    for (i, item) in items.into_iter().enumerate() {
        if is_calib_pos(i, stride) {
            calib.push(item);
        } else {
            train.push(item);
        }
    }
    (train, calib)
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
