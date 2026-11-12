//! Arena growth policy.
//!
//! `Vec::resize` grows by amortised doubling, which after an exact-sized
//! lazy load would double the resident set on the first replayed insert —
//! and wasm linear memory never shrinks (ADR §1.2). Arenas here instead
//! grow by `reserve_exact` in proportional steps of `max(min, len/16)`
//! elements (≤ ~6% slack, logarithmically many reallocations), and callers
//! can pre-size exactly with `reserve`.

/// Minimum growth step in slots.
pub(crate) const MIN_SLOTS: usize = 256;
/// Minimum growth step in HNSW upper-layer rows.
pub(crate) const MIN_ROWS: usize = 16;

/// Growth step for an arena currently holding `len` units.
pub(crate) fn step(len: usize, min: usize) -> usize {
    (len / 16).max(min)
}

/// Make room for `need` elements; if `v` must reallocate, reserve exactly
/// `max(need, target)` so the next few pushes do not reallocate again.
pub(crate) fn fit<T>(v: &mut Vec<T>, need: usize, target: usize) {
    if need > v.capacity() {
        v.reserve_exact(target.max(need) - v.len());
    }
}

/// Bytes that [`fit`] would newly allocate for `v`.
pub(crate) fn fit_bytes<T>(v: &Vec<T>, need: usize, target: usize) -> usize {
    if need > v.capacity() {
        (target.max(need) - v.capacity()) * std::mem::size_of::<T>()
    } else {
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grows_by_bounded_steps() {
        let mut v = vec![0u8; 10_000];
        assert_eq!(fit_bytes(&v, 10_001, 10_000 + step(10_000, MIN_SLOTS)), 625);
        fit(&mut v, 10_001, 10_000 + step(10_000, MIN_SLOTS));
        assert_eq!(v.capacity(), 10_625);
        fit(&mut v, 10_600, 99_999); // fits: no reallocation
        assert_eq!(v.capacity(), 10_625);
        assert_eq!(step(0, MIN_SLOTS), MIN_SLOTS);
    }
}
