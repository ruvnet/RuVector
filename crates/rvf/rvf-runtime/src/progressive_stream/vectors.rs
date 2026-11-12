//! Verified-only vector storage and the incremental VEC segment parser.

use rvf_index::VectorStore;
use std::collections::HashMap;

/// Vectors parsed from verified chunks only. Nothing is inserted unless every
/// byte of the record came from a released `VerifiedChunk`.
#[derive(Default)]
pub struct VerifiedVectors {
    range_steps: std::sync::atomic::AtomicU64,
    dim: usize,
    slot: HashMap<u64, u32>,
    /// Id of each slot, in slot order (contiguous scans).
    ids: Vec<u64>,
    data: Vec<f32>,
}

impl VerifiedVectors {
    pub(crate) fn set_dim(&mut self, dim: usize) {
        if self.dim == 0 {
            self.dim = dim;
        }
    }

    /// Reserve room for `n` more vectors (bounded by the caller's verified
    /// segment size).
    pub(crate) fn reserve(&mut self, n: usize) {
        if self.dim == 0 {
            return;
        }
        self.slot.reserve(n);
        self.ids.reserve(n);
        self.data.reserve(n.saturating_mul(self.dim));
    }

    /// Insert (or overwrite) a vector. Ignored when the dimension differs.
    pub(crate) fn insert(&mut self, id: u64, v: &[f32]) {
        if self.dim == 0 || v.len() != self.dim {
            return;
        }
        match self.slot.get(&id) {
            Some(&s) => {
                let at = s as usize * self.dim;
                self.data[at..at + self.dim].copy_from_slice(v);
            }
            None => {
                self.slot.insert(id, (self.data.len() / self.dim) as u32);
                self.ids.push(id);
                self.data.extend_from_slice(v);
            }
        }
    }

    /// Exact k nearest (squared L2) among the visible vectors, skipping
    /// `skip`. Contiguous scan over verified rows only. Returns
    /// `(id, distance)` ascending, ties by id.
    pub fn exact_topk(&self, q: &[f32], k: usize, skip: &dyn Fn(u64) -> bool) -> Vec<(u64, f32)> {
        self.exact_topk_rows(q, k, usize::MAX, skip)
    }

    /// [`exact_topk`](Self::exact_topk) over the first `rows` rows.
    pub(crate) fn exact_topk_rows(
        &self,
        q: &[f32],
        k: usize,
        rows: usize,
        skip: &dyn Fn(u64) -> bool,
    ) -> Vec<(u64, f32)> {
        if self.dim == 0 || q.len() != self.dim || k == 0 {
            return Vec::new();
        }
        let mut heap: std::collections::BinaryHeap<(OrdF32, u64)> =
            std::collections::BinaryHeap::with_capacity(k + 1);
        for (row, &id) in self.data.chunks_exact(self.dim).zip(&self.ids).take(rows) {
            let mut d = 0f32;
            for (a, b) in row.iter().zip(q) {
                let t = a - b;
                d += t * t;
            }
            if heap.len() < k || OrdF32(d) < heap.peek().map_or(OrdF32(f32::MAX), |x| x.0) {
                if skip(id) {
                    continue;
                }
                heap.push((OrdF32(d), id));
                if heap.len() > k {
                    heap.pop();
                }
            }
        }
        let mut out: Vec<(u64, f32)> = heap.into_iter().map(|(d, id)| (id, d.0)).collect();
        out.sort_by(|a, b| {
            a.1.partial_cmp(&b.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.0.cmp(&b.0))
        });
        out
    }

    /// The vector dimension (0 until known).
    pub fn dimension_hint(&self) -> usize {
        self.dim
    }

    /// Lookups performed by range visits so far (regression-test counter).
    pub fn range_steps(&self) -> u64 {
        self.range_steps.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// True if `id` is visible.
    pub fn contains(&self, id: u64) -> bool {
        self.slot.contains_key(&id)
    }

    /// Number of visible vectors.
    pub fn len(&self) -> usize {
        self.slot.len()
    }

    /// True when no vector is visible.
    pub fn is_empty(&self) -> bool {
        self.slot.is_empty()
    }

    /// All visible `(id, vector)` pairs (unordered).
    pub fn iter(&self) -> impl Iterator<Item = (u64, &[f32])> + '_ {
        self.slot.iter().map(|(&id, &s)| {
            let at = s as usize * self.dim;
            (id, &self.data[at..at + self.dim])
        })
    }

    pub(crate) fn clear(&mut self) {
        self.slot = HashMap::new();
        self.ids = Vec::new();
        self.data = Vec::new();
    }
}

impl VectorStore for VerifiedVectors {
    fn get_vector(&self, id: u64) -> Option<&[f32]> {
        let s = *self.slot.get(&id)? as usize;
        let at = s * self.dim;
        self.data.get(at..at + self.dim)
    }

    fn dimension(&self) -> usize {
        self.dim
    }

    /// Range visit bounded by the vectors present: a wide (or hostile) range
    /// costs at most one step per visible vector.
    fn for_each_in_range(&self, start: u64, end: u64, f: &mut dyn FnMut(u64, &[f32])) {
        use std::sync::atomic::Ordering::Relaxed;
        if end <= start {
            return;
        }
        if ((end - start) as u128) <= self.slot.len() as u128 {
            for id in start..end {
                self.range_steps.fetch_add(1, Relaxed);
                if let Some(v) = self.get_vector(id) {
                    f(id, v);
                }
            }
        } else {
            for (&id, &s) in &self.slot {
                self.range_steps.fetch_add(1, Relaxed);
                if id >= start && id < end {
                    let at = s as usize * self.dim;
                    f(id, &self.data[at..at + self.dim]);
                }
            }
        }
    }
}

/// What the incremental parser reports.
pub(crate) enum VecEvent<'a> {
    /// Room for `count` more records of dimension `dim`, sized from the bytes
    /// just received (never from a declared size).
    Reserve { count: usize, dim: usize },
    /// One complete record.
    Record(u64, &'a [f32]),
}

/// Incremental parser for one VEC segment's bytes (header, then payload
/// `dim u16 | count u32 | count * (id u64, dim * f32)`, then padding).
pub(crate) struct VecParse {
    stored_len: u64,
    seen: u64,
    header: Vec<u8>,
    payload_len: u64,
    in_payload: u64,
    dim: usize,
    remaining: u32,
    carry: Vec<u8>,
    prefix: Vec<u8>,
    scratch: Vec<f32>,
    dead: bool,
}

impl VecParse {
    pub(crate) fn new(stored_len: u64) -> Self {
        Self {
            stored_len,
            seen: 0,
            header: Vec::with_capacity(64),
            payload_len: 0,
            in_payload: 0,
            dim: 0,
            remaining: 0,
            carry: Vec::new(),
            prefix: Vec::with_capacity(6),
            scratch: Vec::new(),
            dead: false,
        }
    }

    /// Consume verified bytes, calling `emit(id, vector)` per complete record.
    pub(crate) fn feed(
        &mut self,
        mut b: &[u8],
        want_dim: usize,
        emit: &mut dyn FnMut(VecEvent<'_>),
    ) {
        if self.dead {
            return;
        }
        self.seen += b.len() as u64;
        if self.header.len() < 64 {
            let take = (64 - self.header.len()).min(b.len());
            self.header.extend_from_slice(&b[..take]);
            b = &b[take..];
            if self.header.len() == 64 {
                self.payload_len = u64::from_le_bytes(self.header[0x10..0x18].try_into().unwrap());
                if self.payload_len.saturating_add(64) > self.stored_len {
                    self.dead = true;
                    return;
                }
            }
        }
        if self.header.len() < 64 || b.is_empty() {
            return;
        }
        let room = self.payload_len - self.in_payload;
        let b = &b[..(b.len() as u64).min(room) as usize];
        self.in_payload += b.len() as u64;
        let mut b = b;
        if self.prefix.len() < 6 {
            let take = (6 - self.prefix.len()).min(b.len());
            self.prefix.extend_from_slice(&b[..take]);
            b = &b[take..];
            if self.prefix.len() == 6 {
                self.dim = u16::from_le_bytes([self.prefix[0], self.prefix[1]]) as usize;
                self.remaining = u32::from_le_bytes(self.prefix[2..6].try_into().unwrap());
                if (want_dim != 0 && self.dim != want_dim) || self.dim == 0 {
                    self.dead = true;
                    return;
                }
                let rec = 8 + 4 * self.dim as u64;
                if self.remaining as u64 * rec + 6 > self.payload_len {
                    self.dead = true;
                    return;
                }
            }
        }
        let rec = 8 + 4 * self.dim;
        // F4-01: reserve only for the bytes in hand (never from the declared
        // count or segment size), so memory follows received bytes.
        if self.dim != 0 && !b.is_empty() && self.remaining > 0 {
            emit(VecEvent::Reserve {
                count: (b.len() / rec + 1).min(self.remaining as usize),
                dim: self.dim,
            });
        }
        while !b.is_empty() && self.remaining > 0 {
            if self.carry.is_empty() && b.len() >= rec {
                self.record(&b[..rec], emit);
                b = &b[rec..];
                continue;
            }
            let take = (rec - self.carry.len()).min(b.len());
            self.carry.extend_from_slice(&b[..take]);
            b = &b[take..];
            if self.carry.len() == rec {
                let c = std::mem::take(&mut self.carry);
                self.record(&c, emit);
            }
        }
    }

    fn record(&mut self, r: &[u8], emit: &mut dyn FnMut(VecEvent<'_>)) {
        let id = u64::from_le_bytes(r[..8].try_into().unwrap());
        self.scratch.clear();
        self.scratch.extend(
            r[8..]
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])),
        );
        self.remaining -= 1;
        emit(VecEvent::Record(id, &self.scratch));
    }

    /// Bytes of this segment seen so far.
    #[cfg(test)]
    pub(crate) fn seen(&self) -> u64 {
        self.seen
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn seg(dim: u16, recs: &[(u64, Vec<f32>)]) -> Vec<u8> {
        let mut p = Vec::new();
        p.extend_from_slice(&dim.to_le_bytes());
        p.extend_from_slice(&(recs.len() as u32).to_le_bytes());
        for (id, v) in recs {
            p.extend_from_slice(&id.to_le_bytes());
            for x in v {
                p.extend_from_slice(&x.to_le_bytes());
            }
        }
        let mut s = vec![0u8; 64];
        s[0x10..0x18].copy_from_slice(&(p.len() as u64).to_le_bytes());
        s.extend_from_slice(&p);
        s.extend_from_slice(&[0u8; 17]); // padding
        s
    }

    #[test]
    fn any_split_yields_the_same_records() {
        let recs: Vec<_> = (0..7u64)
            .map(|i| (i * 3, vec![i as f32, 1.0, 2.0]))
            .collect();
        let bytes = seg(3, &recs);
        for step in [1usize, 2, 5, 7, 13, 64, 1000] {
            let mut p = VecParse::new(bytes.len() as u64);
            let mut got = Vec::new();
            for c in bytes.chunks(step) {
                p.feed(c, 3, &mut |e| {
                    if let VecEvent::Record(id, v) = e {
                        got.push((id, v.to_vec()));
                    }
                });
            }
            assert_eq!(got, recs, "step {step}");
            assert_eq!(p.seen(), bytes.len() as u64);
        }
    }

    #[test]
    fn dimension_mismatch_and_lying_counts_emit_nothing() {
        let bytes = seg(3, &[(1, vec![0.0; 3])]);
        let mut p = VecParse::new(bytes.len() as u64);
        let mut n = 0;
        p.feed(&bytes, 4, &mut |e| {
            n += matches!(e, VecEvent::Record(..)) as i32
        });
        assert_eq!(n, 0);
        let mut lie = bytes.clone();
        lie[64 + 2..64 + 6].copy_from_slice(&u32::MAX.to_le_bytes());
        let mut p = VecParse::new(lie.len() as u64);
        p.feed(&lie, 3, &mut |e| {
            n += matches!(e, VecEvent::Record(..)) as i32
        });
        assert_eq!(n, 0);
    }
}

/// Total order on non-NaN distances for the top-k heap.
#[derive(Clone, Copy, PartialEq)]
struct OrdF32(f32);
impl Eq for OrdF32 {}
impl PartialOrd for OrdF32 {
    fn partial_cmp(&self, o: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(o))
    }
}
impl Ord for OrdF32 {
    fn cmp(&self, o: &Self) -> std::cmp::Ordering {
        self.0
            .partial_cmp(&o.0)
            .unwrap_or(std::cmp::Ordering::Equal)
    }
}
