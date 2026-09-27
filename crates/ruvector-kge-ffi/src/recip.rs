//! Reciprocal-relation plumbing for the binding (ADR-007 §3, plan M3).
//! Duplicated VERBATIM in `ruvector-kge-ffi/src/recip.rs` and
//! `ruvector-kge-wasm/src/recip.rs`.
//!
//! A model built with `{"reciprocal":true}` keeps `2·R` relation rows laid out
//! `[base 0..R | inverse R..2R]` — the core's `train::reciprocal` convention,
//! `R = rows / 2`. The layout is *positional*, so every path that resizes the
//! tables (`addTriples` growth, the snapshot-training swap that replays growth)
//! must move the inverse block to its new offset instead of copying a prefix:
//! [`copy_rows`] is the one place that does it.

use ruvector_kge::{Side, Tables};

/// `serde` helper: skip a `false` flag so older payloads keep their hash.
pub(crate) fn is_false(b: &bool) -> bool {
    !*b
}

/// Relation rows for `labels` relations: `max(1, labels)`, doubled when
/// reciprocal.
pub(crate) fn rows_for(labels: usize, reciprocal: bool) -> usize {
    let base = labels.max(1);
    if reciprocal {
        base.saturating_mul(2)
    } else {
        base
    }
}

/// Copy the rows `src` and `dst` share by id into `dst` (same `dims`): entity
/// rows `0..min`, and relation rows by prefix — or, under the reciprocal
/// layout, base rows `0..m` in place and inverse rows `R_src + i → R_dst + i`
/// (`m = min(R_src, R_dst)`). Rows of `dst` beyond the shared ids are kept.
pub(crate) fn copy_rows(src: &Tables, dst: &mut Tables, reciprocal: bool) {
    let d = src.dims();
    debug_assert_eq!(d, dst.dims(), "copy_rows needs equal dims");
    if d != dst.dims() {
        return;
    }
    let ne = src.num_entities().min(dst.num_entities()) * d;
    dst.entities_raw_mut()[..ne].copy_from_slice(&src.entities_raw()[..ne]);
    let (sr, dr) = (src.num_relations(), dst.num_relations());
    let (s, t) = (src.relations_raw(), dst.relations_raw_mut());
    if !reciprocal {
        let n = sr.min(dr) * d;
        t[..n].copy_from_slice(&s[..n]);
        return;
    }
    let (bs, bd) = (sr / 2, dr / 2);
    let m = bs.min(bd) * d;
    t[..m].copy_from_slice(&s[..m]);
    t[bd * d..bd * d + m].copy_from_slice(&s[bs * d..bs * d + m]);
}

/// The `(relation row, side)` a query on base relation `r` is scored with:
/// unchanged for a tail query or a non-reciprocal model; the inverse row
/// `r + R` as a **tail** query for a reciprocal head query `(?, r, o)`, which
/// is then answered as `(o, r⁻¹, ?)`.
pub(crate) fn query_route(tables: &Tables, r: u32, side: Side, reciprocal: bool) -> (u32, Side) {
    match side {
        Side::Head if reciprocal => (r + (tables.num_relations() / 2) as u32, Side::Tail),
        _ => (r, side),
    }
}

#[cfg(test)]
#[path = "recip_tests.rs"]
mod tests;
