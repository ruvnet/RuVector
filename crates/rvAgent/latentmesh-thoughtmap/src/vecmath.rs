//! Tiny, dependency-free vector helpers shared by capability grounding and
//! causal attribution. Kept private to the crate.

/// Dot product of two equal-length slices. Extra elements on the longer slice
/// are ignored (both are truncated to the shorter length).
pub(crate) fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// L2 norm.
pub(crate) fn norm(a: &[f32]) -> f32 {
    dot(a, a).sqrt()
}

/// Whether every element of a latent vector is finite (no NaN/±inf).
pub(crate) fn is_finite(v: &[f32]) -> bool {
    v.iter().all(|x| x.is_finite())
}

/// Cosine similarity in `[-1, 1]`. Returns `0.0` when either vector is empty,
/// has zero magnitude, or is **non-finite** (an undefined direction contributes
/// no evidence).
///
/// NaN-safety is load-bearing for the causal gate: a NaN slipping through here
/// would make every `<`-comparison in [`crate::causal`] false and silently fall
/// through to Accept. A non-finite operand ⇒ `0.0` (no match), never NaN.
pub(crate) fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let na = norm(a);
    let nb = norm(b);
    if na == 0.0 || nb == 0.0 || !na.is_finite() || !nb.is_finite() {
        return 0.0;
    }
    let sim = dot(a, b) / (na * nb);
    if !sim.is_finite() {
        return 0.0;
    }
    sim.clamp(-1.0, 1.0)
}

/// Element-wise mean of a set of equal-length vectors. Returns an empty vector
/// when the input is empty.
///
/// **Non-finite vectors are skipped** so a single NaN/±inf signature cannot
/// poison the map-wide control population the causal specificity check depends
/// on (defense-in-depth behind the `incorporate` choke point).
pub(crate) fn mean(vectors: &[Vec<f32>]) -> Vec<f32> {
    let mut iter = vectors.iter().filter(|v| is_finite(v));
    let Some(first) = iter.next() else {
        return Vec::new();
    };
    let mut acc = first.clone();
    let mut count = 1.0f32;
    for v in iter {
        for (a, x) in acc.iter_mut().zip(v.iter()) {
            *a += x;
        }
        count += 1.0;
    }
    for a in acc.iter_mut() {
        *a /= count;
    }
    acc
}
